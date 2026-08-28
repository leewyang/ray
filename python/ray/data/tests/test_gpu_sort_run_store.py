import errno
import os
from pathlib import Path

import pyarrow as pa
import pytest

from ray.data._internal.gpu_sort.run_store import (
    LocalDiskRunStore,
    RunStoreCapacityError,
    RunStoreCleanupError,
)


def _table(offset=0):
    return pa.table(
        {
            "origin": pa.array(["ATL", None, "SFO"], type=pa.string()),
            "row_id": pa.array([offset, offset + 1, offset + 2], type=pa.int64()),
            "when": pa.array(
                [1_000_000 + offset, None, 3_000_000 + offset],
                type=pa.timestamp("us"),
            ),
        }
    )


def test_local_run_store_round_trip_and_buffers_survive_unlink(tmp_path):
    rank_directory = tmp_path / "sort-id" / "rank-0"
    store = LocalDiskRunStore(str(rank_directory))
    expected = _table()

    chunk = store.write(expected)
    assert Path(chunk.ref).exists()
    assert chunk.physical_size_bytes == Path(chunk.ref).stat().st_size

    with store.read_many([chunk]) as tables:
        restored = tables[0]
        assert restored.equals(expected)

    # Final passthrough closes its NativeFile to bound descriptors, then may
    # retain/coalesce/serialize the Arrow table through Ray's creation ack.
    # PyArrow buffers must keep that mmap alive even after source deletion.
    store.release([chunk])
    assert not Path(chunk.ref).exists()
    assert restored.to_pydict() == expected.to_pydict()

    stats = store.diagnostics()
    assert stats["local_run_write_bytes"] == expected.nbytes
    assert stats["local_run_read_bytes"] == expected.nbytes
    assert stats["local_run_physical_write_bytes"] == chunk.physical_size_bytes
    assert stats["local_run_physical_read_bytes"] == chunk.physical_size_bytes
    assert stats["local_run_live_bytes"] == 0
    assert stats["local_run_live_files"] == 0
    assert stats["local_run_cleanup_calls"] == 1
    assert stats["local_run_cleanup_files"] == 1
    assert stats["local_run_cleanup_bytes"] == chunk.physical_size_bytes
    assert stats["local_run_cleanup_s"] >= 0
    assert stats["local_run_cleanup_pending_files"] == 0
    assert stats["local_run_cleanup_pending_bytes"] == 0

    cleanup = {name: amount for name, amount in stats.items() if "cleanup" in name}
    store.close()
    store.close()
    assert not rank_directory.exists()
    assert {
        name: amount
        for name, amount in store.diagnostics().items()
        if "cleanup" in name
    } == cleanup


def test_local_run_store_batched_reads_preserve_requested_order(tmp_path):
    store = LocalDiskRunStore(str(tmp_path / "sort-id" / "rank-0"))
    chunks = [store.write(_table(offset)) for offset in (0, 10, 20)]

    with store.read_many([chunks[2], chunks[0]]) as tables:
        assert [table["row_id"].to_pylist() for table in tables] == [
            [20, 21, 22],
            [0, 1, 2],
        ]

    assert store.diagnostics()["local_run_read_calls"] == 1
    store.close()


def test_local_run_store_enforces_capacity_and_cleans_partial_write(
    tmp_path, monkeypatch
):
    table = _table()
    rank_directory = tmp_path / "sort-id" / "rank-0"
    store = LocalDiskRunStore(
        str(rank_directory), max_live_bytes=max(1, table.nbytes - 1)
    )

    with pytest.raises(RunStoreCapacityError, match="live-byte"):
        store.write(table)
    assert list(rank_directory.iterdir()) == []
    assert store.diagnostics()["local_run_write_errors"] == 1
    store.close()

    reserved = LocalDiskRunStore(
        str(tmp_path / "sort-id-3" / "rank-0"), min_free_bytes=table.nbytes
    )
    monkeypatch.setattr(reserved, "_available_bytes", lambda: table.nbytes)
    with pytest.raises(RunStoreCapacityError, match="free-space reserve"):
        reserved.write(table)
    assert list(Path(reserved.directory).iterdir()) == []
    reserved.close()

    store = LocalDiskRunStore(str(tmp_path / "sort-id-2" / "rank-0"))

    def reject_committed_size(_physical_bytes):
        raise RunStoreCapacityError("synthetic ENOSPC after IPC write")

    monkeypatch.setattr(store, "_admit_committed_size", reject_committed_size)
    with pytest.raises(RunStoreCapacityError, match="synthetic ENOSPC"):
        store.write(table)
    assert list(Path(store.directory).iterdir()) == []
    assert store.diagnostics()["local_run_live_files"] == 0
    store.close()


def test_local_run_store_atomic_replace_failure_rolls_back_temporary_file(
    tmp_path, monkeypatch
):
    directory = tmp_path / "sort-id" / "rank-0"
    store = LocalDiskRunStore(str(directory))

    def fail_replace(_source, _destination):
        raise OSError(errno.EIO, "synthetic atomic rename failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic atomic rename failure"):
        store.write(_table())

    assert list(directory.iterdir()) == []
    stats = store.diagnostics()
    assert stats["local_run_write_errors"] == 1
    assert stats["local_run_cleanup_calls"] == 1
    assert stats["local_run_cleanup_files"] == 1
    assert stats["local_run_cleanup_bytes"] > 0
    assert stats["local_run_cleanup_errors"] == 0
    assert stats["local_run_cleanup_pending_files"] == 0
    assert stats["local_run_cleanup_pending_bytes"] == 0
    store.close()


def test_local_run_store_pending_temp_bytes_reject_next_write(tmp_path, monkeypatch):
    directory = tmp_path / "sort-id" / "rank-0"
    store = LocalDiskRunStore(str(directory))
    table = _table()
    original_unlink = Path.unlink
    fail_unlink = True

    def unlink(path, *args, **kwargs):
        if path.name.endswith(".tmp") and fail_unlink:
            raise OSError("synthetic busy temporary file")
        return original_unlink(path, *args, **kwargs)

    def fail_replace(_source, _destination):
        raise OSError(errno.EIO, "synthetic publish failure")

    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic publish failure"):
        store.write(table)

    stats = store.diagnostics()
    pending_bytes = stats["local_run_cleanup_pending_bytes"]
    assert stats["local_run_live_bytes"] == 0
    assert stats["local_run_cleanup_pending_files"] == 1
    assert pending_bytes > 0
    assert store._accounted_disk_bytes() == pending_bytes

    store._max_live_bytes = pending_bytes + table.nbytes - 1
    with pytest.raises(RunStoreCapacityError, match="live-byte limit"):
        store.write(table)
    assert store.diagnostics()["local_run_cleanup_pending_bytes"] == pending_bytes
    assert len(list(directory.iterdir())) == 1

    fail_unlink = False
    store.close()
    stats = store.diagnostics()
    assert not directory.exists()
    assert stats["local_run_cleanup_calls"] == 2
    assert stats["local_run_cleanup_files"] == 1
    assert stats["local_run_cleanup_bytes"] == pending_bytes
    assert stats["local_run_cleanup_pending_files"] == 0
    assert stats["local_run_cleanup_pending_bytes"] == 0


def test_local_run_store_translates_enospc_and_rolls_back_partial_file(
    tmp_path, monkeypatch
):
    directory = tmp_path / "sort-id" / "rank-0"
    store = LocalDiskRunStore(str(directory))
    partial = b"partial-arrow-ipc"

    def exhaust_filesystem(path, _mode):
        Path(path).write_bytes(partial)
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(pa, "OSFile", exhaust_filesystem)
    with pytest.raises(
        RunStoreCapacityError, match="exhausted filesystem capacity"
    ) as raised:
        store.write(_table())

    assert isinstance(raised.value.__cause__, OSError)
    assert raised.value.__cause__.errno == errno.ENOSPC
    assert list(directory.iterdir()) == []
    stats = store.diagnostics()
    assert stats["local_run_write_errors"] == 1
    assert stats["local_run_cleanup_calls"] == 1
    assert stats["local_run_cleanup_files"] == 1
    assert stats["local_run_cleanup_bytes"] == len(partial)
    assert stats["local_run_cleanup_errors"] == 0
    assert stats["local_run_cleanup_pending_files"] == 0
    assert stats["local_run_cleanup_pending_bytes"] == 0
    store.close()


def test_local_run_store_base_exception_rolls_back_and_is_not_translated(
    tmp_path, monkeypatch
):
    directory = tmp_path / "sort-id" / "rank-0"
    store = LocalDiskRunStore(str(directory))

    class FatalWrite(BaseException):
        pass

    def fail_replace(_source, _destination):
        raise FatalWrite("synthetic fatal publish failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(FatalWrite, match="synthetic fatal publish failure"):
        store.write(_table())

    assert list(directory.iterdir()) == []
    stats = store.diagnostics()
    assert stats["local_run_write_errors"] == 1
    assert stats["local_run_cleanup_calls"] == 1
    assert stats["local_run_cleanup_files"] == 1
    assert stats["local_run_cleanup_bytes"] > 0
    assert stats["local_run_cleanup_errors"] == 0
    assert stats["local_run_cleanup_pending_files"] == 0
    store.close()


def test_local_run_store_detects_truncation_and_consumer_errors_are_not_reads(
    tmp_path,
):
    store = LocalDiskRunStore(str(tmp_path / "sort-id" / "rank-0"))
    good = store.write(_table())

    with pytest.raises(ValueError, match="consumer"):
        with store.read_many([good]):
            raise ValueError("consumer conversion failed")
    assert store.diagnostics()["local_run_read_errors"] == 0

    damaged = store.write(_table(10))
    with open(damaged.ref, "r+b") as file:
        file.truncate(max(1, damaged.physical_size_bytes // 2))
    with pytest.raises(IOError, match="file size changed"):
        with store.read_many([damaged]):
            pass
    assert store.diagnostics()["local_run_read_errors"] == 1
    store.close()


def test_local_run_store_rejects_unsafe_or_colliding_directories(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        LocalDiskRunStore("relative/run-dir")
    with pytest.raises(ValueError, match="dev/shm"):
        LocalDiskRunStore("/dev/shm/gpu-sort-test")

    directory = tmp_path / "sort-id" / "rank-0"
    directory.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        LocalDiskRunStore(str(directory))


def test_local_run_store_failed_unlink_remains_owned_and_close_retries(
    tmp_path, monkeypatch
):
    directory = tmp_path / "sort-id" / "rank-0"
    store = LocalDiskRunStore(str(directory))
    chunk = store.write(_table())
    original_unlink = Path.unlink
    unlink_attempts = 0

    def unlink(path, *args, **kwargs):
        nonlocal unlink_attempts
        if str(path) == chunk.ref:
            unlink_attempts += 1
            if unlink_attempts <= 2:
                raise OSError("synthetic busy file")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    store.close()

    stats = store.diagnostics()
    assert Path(chunk.ref).exists()
    assert stats["local_run_live_files"] == 1
    assert stats["local_run_live_bytes"] == chunk.physical_size_bytes
    assert stats["local_run_cleanup_calls"] == 1
    assert stats["local_run_cleanup_files"] == 0
    assert stats["local_run_cleanup_bytes"] == 0
    assert stats["local_run_cleanup_errors"] == 1
    assert stats["local_run_cleanup_s"] > 0
    first_cleanup_s = stats["local_run_cleanup_s"]
    assert stats["local_run_cleanup_pending_files"] == 1
    assert stats["local_run_cleanup_pending_bytes"] == chunk.physical_size_bytes
    assert not store._closed

    # The same path is both owned and pending, but admission counts its bytes
    # only once and one close() makes only one unlink attempt for it.
    store._max_live_bytes = chunk.physical_size_bytes + _table().nbytes
    store._admit(_table().nbytes)
    store.close()
    stats = store.diagnostics()
    assert Path(chunk.ref).exists()
    assert unlink_attempts == 2
    assert stats["local_run_cleanup_calls"] == 2
    assert stats["local_run_cleanup_errors"] == 2
    assert stats["local_run_cleanup_s"] > first_cleanup_s
    second_cleanup_s = stats["local_run_cleanup_s"]
    assert stats["local_run_cleanup_pending_files"] == 1
    assert not store._closed

    store.close()
    assert not directory.exists()
    stats = store.diagnostics()
    assert stats["local_run_live_files"] == 0
    assert unlink_attempts == 3
    assert stats["local_run_cleanup_calls"] == 3
    assert stats["local_run_cleanup_files"] == 1
    assert stats["local_run_cleanup_bytes"] == chunk.physical_size_bytes
    assert stats["local_run_cleanup_s"] > second_cleanup_s
    assert stats["local_run_cleanup_pending_files"] == 0
    assert stats["local_run_cleanup_pending_bytes"] == 0
    cleanup = {name: amount for name, amount in stats.items() if "cleanup" in name}
    store.close()
    assert {
        name: amount
        for name, amount in store.diagnostics().items()
        if "cleanup" in name
    } == cleanup


def test_local_run_store_strict_close_fails_leaks_and_remains_retryable(
    tmp_path, monkeypatch
):
    directory = tmp_path / "sort-id" / "rank-0"
    store = LocalDiskRunStore(str(directory))
    chunk = store.write(_table())
    original_unlink = Path.unlink
    fail_unlink = True

    def unlink(path, *args, **kwargs):
        if str(path) == chunk.ref and fail_unlink:
            raise OSError("synthetic persistent busy file")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(RunStoreCleanupError, match="tracked files"):
        store.close(strict=True)

    stats = store.diagnostics()
    assert Path(chunk.ref).exists()
    assert stats["local_run_live_files"] == 1
    assert stats["local_run_cleanup_pending_files"] == 1
    assert not store._closed

    fail_unlink = False
    store.close(strict=True)
    assert not directory.exists()
    assert store.diagnostics()["local_run_cleanup_pending_files"] == 0
    assert store._closed
    store.close(strict=True)


def test_local_run_store_exception_after_ownership_rolls_back_accounting(tmp_path):
    directory = tmp_path / "sort-id" / "rank-0"
    store = LocalDiskRunStore(str(directory))

    class FailingStats(dict):
        def __setitem__(self, key, value):
            if key == "local_run_write_bytes":
                raise RuntimeError("synthetic post-ownership failure")
            return super().__setitem__(key, value)

    store._stats = FailingStats(store._stats)
    with pytest.raises(RuntimeError, match="post-ownership failure"):
        store.write(_table())

    assert store._owned == {}
    assert store.diagnostics()["local_run_live_bytes"] == 0
    assert store.diagnostics()["local_run_live_files"] == 0
    assert list(directory.iterdir()) == []
    store.close()
