"""Storage backends for immutable GPU external-sort runs.

Intermediate runs are actor-local implementation details.  Keeping their
storage behind this small interface lets the external merge use either Ray's
object store or node-local Arrow IPC files without changing Dataset output
ownership: final blocks are always sealed by Ray.
"""

from __future__ import annotations

import errno
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ContextManager, Dict, Iterator, List, Optional, Protocol


@dataclass
class _RunChunk:
    """One bounded immutable run fragment.

    ``ref`` is intentionally opaque to the merge engine.  It is a Ray
    ObjectRef for :class:`PlasmaRunStore` and an absolute path string for
    :class:`LocalDiskRunStore`.
    """

    ref: Any
    rows: int
    size_bytes: int
    physical_size_bytes: int = 0


class RunStoreCapacityError(RuntimeError):
    """Raised before a local run write would violate its disk budget."""


class RunStoreCleanupError(RuntimeError):
    """Raised when strict local-run cleanup cannot prove that storage is clean."""


class RunStore(Protocol):
    """Storage contract used by the external merge engine."""

    mode: str

    def write(self, table: Any) -> _RunChunk: ...

    def read_many(self, chunks: List[_RunChunk]) -> ContextManager[List[Any]]: ...

    def release(self, chunks: List[_RunChunk]) -> None: ...

    def close(self, *, strict: bool = False) -> None: ...

    def diagnostics(self) -> Dict[str, Any]: ...


class PlasmaRunStore:
    """Store intermediate Arrow tables as Ray objects (the default)."""

    mode = "plasma"

    def __init__(self) -> None:
        self._stats: Dict[str, Any] = {
            "plasma_intermediate_write_bytes": 0,
            "plasma_intermediate_read_bytes": 0,
            "plasma_intermediate_write_calls": 0,
            "plasma_intermediate_read_calls": 0,
            "plasma_intermediate_write_s": 0.0,
            "plasma_intermediate_read_s": 0.0,
        }

    def write(self, table: Any) -> _RunChunk:
        import ray

        started = time.perf_counter()
        ref = ray.put(table)
        elapsed = time.perf_counter() - started
        size_bytes = int(table.nbytes)
        self._stats["plasma_intermediate_write_bytes"] += size_bytes
        self._stats["plasma_intermediate_write_calls"] += 1
        self._stats["plasma_intermediate_write_s"] += elapsed
        return _RunChunk(
            ref=ref,
            rows=int(table.num_rows),
            size_bytes=size_bytes,
            physical_size_bytes=size_bytes,
        )

    @contextmanager
    def read_many(self, chunks: List[_RunChunk]) -> Iterator[List[Any]]:
        import ray

        started = time.perf_counter()
        tables = ray.get([chunk.ref for chunk in chunks])
        elapsed = time.perf_counter() - started
        self._stats["plasma_intermediate_read_bytes"] += sum(
            int(chunk.size_bytes) for chunk in chunks
        )
        self._stats["plasma_intermediate_read_calls"] += 1
        self._stats["plasma_intermediate_read_s"] += elapsed
        yield tables

    def release(self, chunks: List[_RunChunk]) -> None:
        # ObjectRef lifetime is controlled by the owning _ExternalRun lists.
        # Calling ray.internal.free here could invalidate a reused passthrough
        # chunk whose ownership has just moved to a replacement run.
        return None

    def close(self, *, strict: bool = False) -> None:
        return None

    def diagnostics(self) -> Dict[str, Any]:
        return dict(self._stats)


class LocalDiskRunStore:
    """Actor-owned, uncompressed Arrow IPC storage for intermediate runs."""

    mode = "local_disk"
    _STRICT_CLEANUP_ATTEMPTS = 3

    def __init__(
        self,
        directory: str,
        *,
        min_free_bytes: int = 0,
        max_live_bytes: Optional[int] = None,
    ) -> None:
        root = Path(directory)
        if not root.is_absolute():
            raise ValueError("GPU local run directory must be an absolute path.")
        resolved = root.resolve(strict=False)
        if resolved == Path("/dev/shm") or Path("/dev/shm") in resolved.parents:
            raise ValueError("GPU local run directory cannot be inside /dev/shm.")
        if min_free_bytes < 0:
            raise ValueError("GPU local run minimum free bytes cannot be negative.")
        if max_live_bytes is not None and max_live_bytes <= 0:
            raise ValueError("GPU local run maximum live bytes must be positive.")

        # The caller supplies a unique rank directory. exist_ok=False turns a
        # stale or colliding trial into a hard failure instead of mixing runs.
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.mkdir(exist_ok=False)
        self._directory = resolved
        self._min_free_bytes = int(min_free_bytes)
        self._max_live_bytes = None if max_live_bytes is None else int(max_live_bytes)
        self._next_id = 0
        self._owned: Dict[str, int] = {}
        # Failed unlinks remain retryable. Values are the best known physical
        # sizes so pending cleanup cannot disappear from disk accounting.
        self._cleanup_pending: Dict[str, int] = {}
        self._closed = False
        self._stats: Dict[str, Any] = {
            "local_run_write_bytes": 0,
            "local_run_read_bytes": 0,
            "local_run_physical_write_bytes": 0,
            "local_run_physical_read_bytes": 0,
            "local_run_write_calls": 0,
            "local_run_read_calls": 0,
            "local_run_write_s": 0.0,
            "local_run_read_s": 0.0,
            "local_run_sync_s": 0.0,
            "local_run_sync_calls": 0,
            "local_run_live_bytes": 0,
            "local_run_peak_bytes": 0,
            "local_run_live_files": 0,
            "local_run_peak_files": 0,
            "local_run_write_errors": 0,
            "local_run_read_errors": 0,
            # Cleanup cost counts every tracked unlink attempt. Files and bytes
            # count only paths actually removed; failures remain pending.
            "local_run_cleanup_s": 0.0,
            "local_run_cleanup_calls": 0,
            "local_run_cleanup_files": 0,
            "local_run_cleanup_bytes": 0,
            "local_run_cleanup_errors": 0,
            "local_run_cleanup_pending_files": 0,
            "local_run_cleanup_pending_bytes": 0,
        }

    @property
    def directory(self) -> str:
        return str(self._directory)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("GPU local run store is closed.")

    def _available_bytes(self) -> int:
        usage = os.statvfs(self._directory)
        return int(usage.f_bavail) * int(usage.f_frsize)

    def _sync_cleanup_accounting(self) -> None:
        self._stats["local_run_live_bytes"] = sum(self._owned.values())
        self._stats["local_run_live_files"] = len(self._owned)
        self._stats["local_run_cleanup_pending_files"] = len(self._cleanup_pending)
        self._stats["local_run_cleanup_pending_bytes"] = sum(
            self._cleanup_pending.values()
        )

    def _accounted_disk_bytes(self) -> int:
        """Return unique owned plus pending-only physical bytes."""

        return sum(self._owned.values()) + sum(
            physical_bytes
            for path, physical_bytes in self._cleanup_pending.items()
            if path not in self._owned
        )

    def _unlink_for_cleanup(self, path: Path, *, known_size_bytes: int = 0) -> bool:
        """Unlink one tracked file and update retry-safe cleanup telemetry."""

        key = str(path)
        known_size_bytes = max(
            int(known_size_bytes),
            int(self._owned.get(key, 0)),
            int(self._cleanup_pending.get(key, 0)),
        )
        started = time.perf_counter()
        existed = True
        physical_bytes = known_size_bytes
        try:
            physical_bytes = int(path.stat().st_size)
        except FileNotFoundError:
            existed = False
        except OSError:
            # Unlink may still succeed. Retain the best size already recorded if
            # it does not, rather than hiding pending disk consumption.
            pass

        if not existed and key not in self._owned and key not in self._cleanup_pending:
            return True

        try:
            path.unlink(missing_ok=True)
        except OSError:
            self._stats["local_run_cleanup_s"] += time.perf_counter() - started
            self._stats["local_run_cleanup_calls"] += 1
            self._stats["local_run_cleanup_errors"] += 1
            self._cleanup_pending[key] = physical_bytes
            self._sync_cleanup_accounting()
            return False

        self._stats["local_run_cleanup_s"] += time.perf_counter() - started
        self._stats["local_run_cleanup_calls"] += 1
        if existed:
            self._stats["local_run_cleanup_files"] += 1
            self._stats["local_run_cleanup_bytes"] += physical_bytes
        self._owned.pop(key, None)
        self._cleanup_pending.pop(key, None)
        self._sync_cleanup_accounting()
        return True

    def _admit(self, estimated_bytes: int) -> None:
        available = self._available_bytes()
        if available - int(estimated_bytes) < self._min_free_bytes:
            raise RunStoreCapacityError(
                "GPU local run write would violate the configured free-space "
                f"reserve: available={available}, estimated={estimated_bytes}, "
                f"reserve={self._min_free_bytes}."
            )
        live = self._accounted_disk_bytes()
        if (
            self._max_live_bytes is not None
            and live + int(estimated_bytes) > self._max_live_bytes
        ):
            raise RunStoreCapacityError(
                "GPU local run write would exceed the configured live-byte "
                f"limit: live={live}, estimated={estimated_bytes}, "
                f"limit={self._max_live_bytes}."
            )

    def _admit_committed_size(self, physical_bytes: int) -> None:
        # The temporary file already consumes ``physical_bytes`` on the
        # filesystem, so do not subtract it from statvfs a second time.
        available = self._available_bytes()
        if available < self._min_free_bytes:
            raise RunStoreCapacityError(
                "GPU local run write consumed the configured free-space "
                f"reserve: available={available}, reserve={self._min_free_bytes}."
            )
        live = self._accounted_disk_bytes()
        if (
            self._max_live_bytes is not None
            and live + int(physical_bytes) > self._max_live_bytes
        ):
            raise RunStoreCapacityError(
                "GPU local run IPC size would exceed the configured live-byte "
                f"limit: live={live}, physical={physical_bytes}, "
                f"limit={self._max_live_bytes}."
            )

    def write(self, table: Any) -> _RunChunk:
        import pyarrow as pa

        self._require_open()
        logical_bytes = int(table.nbytes)
        started = time.perf_counter()
        temporary: Optional[Path] = None
        final: Optional[Path] = None
        try:
            self._admit(logical_bytes)
            chunk_id = self._next_id
            self._next_id += 1
            final = self._directory / f"chunk-{chunk_id:012d}.arrow"
            temporary = self._directory / f".chunk-{chunk_id:012d}.tmp"
            # The actor is single-threaded, but O_EXCL also guards accidental
            # path reuse if execution is retried in a stale directory.
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(descriptor)
            with pa.OSFile(str(temporary), "wb") as sink:
                options = pa.ipc.IpcWriteOptions(compression=None)
                with pa.ipc.new_file(sink, table.schema, options=options) as writer:
                    writer.write_table(table)
            physical_bytes = int(temporary.stat().st_size)
            self._admit_committed_size(physical_bytes)
            sync_started = time.perf_counter()
            try:
                descriptor = os.open(temporary, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                os.replace(temporary, final)
                temporary = None
                directory_descriptor = os.open(
                    self._directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                )
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            finally:
                self._stats["local_run_sync_s"] += time.perf_counter() - sync_started
                self._stats["local_run_sync_calls"] += 1
            path = str(final)
            self._owned[path] = physical_bytes
            live_bytes = int(self._stats["local_run_live_bytes"]) + physical_bytes
            live_files = int(self._stats["local_run_live_files"]) + 1
            self._stats["local_run_write_bytes"] += logical_bytes
            self._stats["local_run_physical_write_bytes"] += physical_bytes
            self._stats["local_run_write_calls"] += 1
            self._stats["local_run_live_bytes"] = live_bytes
            self._stats["local_run_peak_bytes"] = max(
                int(self._stats["local_run_peak_bytes"]), live_bytes
            )
            self._stats["local_run_live_files"] = live_files
            self._stats["local_run_peak_files"] = max(
                int(self._stats["local_run_peak_files"]), live_files
            )
            return _RunChunk(
                ref=path,
                rows=int(table.num_rows),
                size_bytes=logical_bytes,
                physical_size_bytes=physical_bytes,
            )
        except BaseException as error:
            self._stats["local_run_write_errors"] += 1
            for path in (temporary, final):
                if path is not None:
                    self._unlink_for_cleanup(path)
            # An exception after ownership assignment must not leave a hidden
            # live-file count. Failed unlinks remain owned and retryable.
            self._sync_cleanup_accounting()
            if isinstance(error, OSError) and error.errno in {
                errno.ENOSPC,
                getattr(errno, "EDQUOT", -1),
            }:
                raise RunStoreCapacityError(
                    "GPU local run Arrow IPC write exhausted filesystem capacity."
                ) from error
            raise
        finally:
            self._stats["local_run_write_s"] += time.perf_counter() - started

    @contextmanager
    def read_many(self, chunks: List[_RunChunk]) -> Iterator[List[Any]]:
        import pyarrow as pa

        self._require_open()
        started = time.perf_counter()
        mappings = []
        tables = []

        def close_mappings() -> None:
            for mapping in reversed(mappings):
                try:
                    mapping.close()
                except OSError:
                    self._stats["local_run_cleanup_errors"] += 1

        try:
            for chunk in chunks:
                path = str(chunk.ref)
                if path not in self._owned:
                    raise RuntimeError(
                        f"GPU local run chunk is not owned by this actor: {path}."
                    )
                physical_bytes = int(Path(path).stat().st_size)
                if int(chunk.physical_size_bytes) > 0 and physical_bytes != int(
                    chunk.physical_size_bytes
                ):
                    raise IOError(
                        "GPU local run file size changed after commit: "
                        f"{path}, expected={chunk.physical_size_bytes}, "
                        f"actual={physical_bytes}."
                    )
                mapping = pa.memory_map(path, "r")
                mappings.append(mapping)
                table = pa.ipc.open_file(mapping).read_all()
                if int(table.num_rows) != int(chunk.rows):
                    raise IOError(
                        "GPU local run row count changed after commit: "
                        f"{path}, expected={chunk.rows}, actual={table.num_rows}."
                    )
                tables.append(table)
            self._stats["local_run_read_bytes"] += sum(
                int(chunk.size_bytes) for chunk in chunks
            )
            self._stats["local_run_physical_read_bytes"] += sum(
                int(chunk.physical_size_bytes) for chunk in chunks
            )
            self._stats["local_run_read_calls"] += 1
            # This intentionally measures IPC mmap/open/read_all metadata work.
            # Lazy page faults plus Arrow-to-GPU transfer are timed by the
            # backend's ``local_run_restore_s`` around this read context.
            self._stats["local_run_read_s"] += time.perf_counter() - started
        except BaseException:
            self._stats["local_run_read_errors"] += 1
            close_mappings()
            raise
        try:
            # Exceptions thrown by a consumer after ``yield`` are conversion
            # or merge failures, not local-read failures.
            yield tables
        finally:
            # Arrow buffers retain the mapped region while consumers still
            # own a table. Closing the file handle here bounds descriptors;
            # the backend additionally synchronizes H2D copies before exit.
            close_mappings()

    def release(self, chunks: List[_RunChunk]) -> None:
        for chunk in chunks:
            path = str(chunk.ref)
            physical_bytes = self._owned.get(path)
            if physical_bytes is None:
                continue
            # A failure retains ownership and pending accounting so close() can
            # retry and diagnostics cannot hide a leaked file.
            self._unlink_for_cleanup(Path(path), known_size_bytes=int(physical_bytes))

    def close(self, *, strict: bool = False) -> None:
        """Release all tracked files and remove this rank's directory.

        Best-effort close keeps failed unlinks owned and retryable, which is
        appropriate while unwinding an already-failed sort.  Successful sort
        completion uses ``strict=True`` so a persistent local-run leak becomes
        an operation failure instead of being hidden by actor teardown.
        """

        if self._closed:
            return
        retried_paths = set(self._cleanup_pending)
        for path, physical_bytes in list(self._cleanup_pending.items()):
            self._unlink_for_cleanup(Path(path), known_size_bytes=int(physical_bytes))
        # Snapshot handles because release mutates the ownership map.
        chunks = [
            _RunChunk(ref=path, rows=0, size_bytes=0, physical_size_bytes=size)
            for path, size in list(self._owned.items())
            if path not in retried_paths
        ]
        self.release(chunks)
        if self._owned or self._cleanup_pending:
            # A failed unlink remains retryable on the next idempotent close.
            if not strict:
                return
            for _ in range(self._STRICT_CLEANUP_ATTEMPTS - 1):
                self.close()
                if self._closed:
                    return
                if not self._owned and not self._cleanup_pending:
                    break
            if self._owned or self._cleanup_pending:
                remaining = sorted(set(self._owned) | set(self._cleanup_pending))
                preview = ", ".join(remaining[:4])
                raise RunStoreCleanupError(
                    "GPU local run cleanup left tracked files behind: "
                    f"directory={self._directory}, owned_files={len(self._owned)}, "
                    f"pending_files={len(self._cleanup_pending)}, "
                    f"accounted_bytes={self._accounted_disk_bytes()}, "
                    f"paths=[{preview}]."
                )
        try:
            self._directory.rmdir()
        except FileNotFoundError:
            pass
        except OSError as error:
            self._stats["local_run_cleanup_errors"] += 1
            if strict:
                for _ in range(self._STRICT_CLEANUP_ATTEMPTS - 1):
                    self.close()
                    if self._closed:
                        return
                raise RunStoreCleanupError(
                    "GPU local run cleanup could not remove the rank directory: "
                    f"{self._directory}."
                ) from error
            return
        self._closed = True
        try:
            self._directory.parent.rmdir()
        except OSError:
            # Other ranks normally still own sibling directories. This is not
            # a cleanup error; the last rank removes the empty sort directory.
            pass

    def diagnostics(self) -> Dict[str, Any]:
        return dict(self._stats)
