"""Opt-in two-GPU integration coverage for streaming external sort.

These tests exercise RAPIDS, UCXX, MPF, Ray's streaming executor, and a real
local-disk RunStore.  They are intentionally opt-in so ordinary CPU CI does
not pretend to validate the physical GPU path.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import math
import os
from pathlib import Path
import shutil
import time
from typing import Iterator, List, Optional, Sequence
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import ray
from ray.data import DataContext
from ray.data.block import BlockMetadata
from ray.data.datasource import Datasource, ReadTask
from ray.data._internal.gpu_sort.operator import get_last_run_stats


_RUN_GPU_INTEGRATION = os.environ.get("RAY_DATA_RUN_GPU_SORT_INTEGRATION", "0") == "1"
_GPU_ONLY = pytest.mark.skipif(
    not _RUN_GPU_INTEGRATION,
    reason="set RAY_DATA_RUN_GPU_SORT_INTEGRATION=1 on a two-GPU host",
)

_ORIGINS = ("ATL", "BOS", "DFW", "JFK", "LAX", "SEA", "SFO")
_ORIGIN_SCHEMA = pa.schema(
    [
        pa.field("Origin", pa.string(), nullable=False),
        pa.field("row_id", pa.int64(), nullable=False),
        pa.field("payload", pa.string(), nullable=True),
    ]
)
_ORIGIN_BLOCKS = 24
_ORIGIN_ROWS_PER_BLOCK = 4096
_INPUT_BUFFER_BUDGET_BYTES = 3 << 20


class _SlowOriginDatasource(Datasource):
    """One non-replayable-style reader that exposes block-level overlap."""

    def __init__(
        self,
        *,
        num_blocks: int = _ORIGIN_BLOCKS,
        rows_per_block: int = _ORIGIN_ROWS_PER_BLOCK,
        delay_s: float = 0.25,
    ) -> None:
        self._num_blocks = int(num_blocks)
        self._rows_per_block = int(rows_per_block)
        self._delay_s = float(delay_s)

    def estimate_inmemory_data_size(self) -> Optional[int]:
        # The payload is deliberately above Ray Data's configured target but
        # below its 1.5x slicing threshold. Each yield therefore flushes
        # immediately while remaining one physical input block.
        return self._num_blocks * self._rows_per_block * 416

    def get_read_tasks(
        self,
        parallelism: int,
        per_task_row_limit: Optional[int] = None,
        data_context: Optional[DataContext] = None,
    ) -> List[ReadTask]:
        del parallelism, data_context

        def read() -> Iterator[pa.Table]:
            for block_index in range(self._num_blocks):
                first = block_index * self._rows_per_block
                row_ids = np.arange(first, first + self._rows_per_block, dtype=np.int64)
                origins = [
                    _ORIGINS[(int(row_id) * 17 + block_index * 3) % len(_ORIGINS)]
                    for row_id in row_ids
                ]
                value = f"block-{block_index:02d}-" + "x" * 384
                payload = [
                    None if int(row_id) % 97 == 0 else value for row_id in row_ids
                ]
                yield pa.Table.from_arrays(
                    [
                        pa.array(origins, type=pa.string()),
                        pa.array(row_ids, type=pa.int64()),
                        pa.array(payload, type=pa.string()),
                    ],
                    schema=_ORIGIN_SCHEMA,
                )
                if block_index + 1 < self._num_blocks:
                    time.sleep(self._delay_s)

        return [
            ReadTask(
                read,
                BlockMetadata(
                    num_rows=self._num_blocks * self._rows_per_block,
                    size_bytes=self.estimate_inmemory_data_size(),
                    input_files=None,
                    exec_stats=None,
                ),
                per_task_row_limit=per_task_row_limit,
            )
        ]

    def num_rows(self) -> int:
        return self._num_blocks * self._rows_per_block


@pytest.fixture(scope="module")
def two_gpu_ray_runtime():
    if not _RUN_GPU_INTEGRATION:
        pytest.skip("two-GPU streaming-sort integration was not requested")

    ray.shutdown()
    configured = os.environ.get("RAY_DATA_GPU_SORT_TEST_RUN_DIR")
    if not configured:
        pytest.fail("RAY_DATA_GPU_SORT_TEST_RUN_DIR must be set under /raid")
    run_base = Path(configured).resolve()
    raid = Path("/raid").resolve()
    if run_base == raid or not run_base.is_relative_to(raid):
        pytest.fail("RAY_DATA_GPU_SORT_TEST_RUN_DIR must be a child of /raid")
    token = uuid.uuid4().hex[:10]
    runtime_root = run_base.parent / f"gpu-ray-{token}"
    socket_root = Path("/dev/shm/rgs") / token
    runtime_root.mkdir(parents=True, exist_ok=False)
    socket_root.parent.mkdir(parents=True, exist_ok=True)
    spill = runtime_root / "spill"
    spill.mkdir()
    worker_pythonpath = str(Path(__file__).resolve().parent)
    inherited_pythonpath = os.environ.get("PYTHONPATH")
    if inherited_pythonpath:
        worker_pythonpath += os.pathsep + inherited_pythonpath
    ray.init(
        num_cpus=4,
        num_gpus=2,
        object_store_memory=1 << 30,
        object_spilling_directory=str(spill),
        include_dashboard=False,
        log_to_driver=True,
        _temp_dir=str(socket_root),
        _system_config={"max_direct_call_object_size": 0},
        runtime_env={"env_vars": {"PYTHONPATH": worker_pythonpath}},
    )
    resources = ray.cluster_resources()
    assert int(resources.get("GPU", 0)) == 2
    assert int(resources.get("CPU", 0)) == 4
    try:
        yield
    finally:
        ray.shutdown()
        shutil.rmtree(runtime_root, ignore_errors=True)
        shutil.rmtree(socket_root, ignore_errors=True)


def _run_root(label: str) -> Path:
    configured = os.environ.get("RAY_DATA_GPU_SORT_TEST_RUN_DIR")
    if not configured:
        pytest.fail(
            "RAY_DATA_GPU_SORT_TEST_RUN_DIR must name an absolute local-disk "
            "directory (use /raid on the DGX)"
        )
    base = Path(configured)
    if not base.is_absolute():
        pytest.fail("RAY_DATA_GPU_SORT_TEST_RUN_DIR must be absolute")
    if base.resolve() == Path("/raid") or not base.resolve().is_relative_to(
        Path("/raid")
    ):
        pytest.fail("RAY_DATA_GPU_SORT_TEST_RUN_DIR must be a child of /raid")
    base.mkdir(parents=True, exist_ok=True)
    trial = base / f"{label}-{uuid.uuid4().hex}"
    trial.mkdir()
    return trial


@contextmanager
def _gpu_sort_context(run_root: Path):
    context = DataContext.get_current()
    names = {
        "gpu_sort_sample_seed": 0,
        "gpu_sort_sample_size": 65_536,
        "gpu_sort_streaming_sample_rows_per_block": 128,
        "gpu_sort_input_buffer_budget_bytes": _INPUT_BUFFER_BUDGET_BYTES,
        "gpu_sort_memory_budget_bytes": 512 << 20,
        "gpu_sort_auto_wave_fraction": 0.50,
        "gpu_sort_exchange_batch_bytes": 8 << 20,
        "gpu_sort_run_chunk_bytes": 4 << 20,
        "gpu_sort_merge_fan_in": 4,
        "gpu_sort_external_run_store": "local_disk",
        "gpu_sort_external_run_directory": str(run_root),
        "gpu_sort_external_run_min_free_bytes": 0,
        "gpu_sort_external_run_max_live_bytes": 1 << 30,
        "gpu_sort_setup_timeout_s": 180.0,
        "gpu_sort_ucx_tls": "cuda_copy,cuda_ipc,sm,tcp",
        "gpu_sort_ucx_sockaddr_tls_priority": "tcp",
        "gpu_sort_ucxx_progress_mode": "polling",
    }
    missing = object()
    old_configs = {name: context.get_config(name, missing) for name in names}
    old_num_actors = context.gpu_shuffle_num_actors
    old_preserve_order = context.execution_options.preserve_order
    old_min_block = context.target_min_block_size
    old_max_block = context.target_max_block_size
    context.gpu_shuffle_num_actors = 2
    context.execution_options.preserve_order = True
    context.target_min_block_size = 512 << 10
    context.target_max_block_size = (5 << 20) // 4
    for name, value in names.items():
        context.set_config(name, value)
    try:
        yield
    finally:
        context.gpu_shuffle_num_actors = old_num_actors
        context.execution_options.preserve_order = old_preserve_order
        context.target_min_block_size = old_min_block
        context.target_max_block_size = old_max_block
        for name, value in old_configs.items():
            if value is missing:
                context.remove_config(name)
            else:
                context.set_config(name, value)


def _execute_gpu_sort(
    dataset_factory,
    *,
    keys: Sequence[str],
    descending: Sequence[bool],
    run_root: Path,
) -> tuple[pa.Table, dict]:
    with _gpu_sort_context(run_root):
        # Dataset plans snapshot DataContext, so construct the source only
        # after all GPU-sort and block-sizing controls have been installed.
        dataset = dataset_factory()
        refs = dataset.sort(
            key=list(keys), descending=list(descending), backend="gpu"
        ).to_arrow_refs()
        tables = ray.get(refs)
        stats = get_last_run_stats()
    assert tables
    return pa.concat_tables(tables), stats


def _assert_external_local_run_proof(stats: dict, run_root: Path) -> None:
    assert stats["mode"] == "external"
    assert stats["run_store"] == "local_disk"
    assert stats["streaming_input"] is True
    assert len(stats["ranks"]) == 2
    assert stats["source_run_count"] > 0
    assert stats["local_run_write_bytes"] > 0
    assert stats["local_run_read_bytes"] > 0
    assert stats["local_run_peak_files"] > 0
    assert stats["plasma_intermediate_read_bytes"] == 0
    assert stats["plasma_intermediate_write_bytes"] == 0
    assert stats["plasma_intermediate_read_calls"] == 0
    assert stats["plasma_intermediate_write_calls"] == 0
    assert stats["cpu_sort_rows"] == 0
    assert stats["cpu_merge_rows"] == 0
    assert stats["fallback_count"] == 0
    assert stats["mpf_host_spill_bytes"] == 0
    assert stats["local_run_live_bytes"] == 0
    assert stats["local_run_live_files"] == 0
    assert stats["local_run_cleanup_pending_bytes"] == 0
    assert stats["local_run_cleanup_pending_files"] == 0
    assert stats["local_run_cleanup_errors"] == 0
    assert list(run_root.iterdir()) == []
    run_root.rmdir()


@_GPU_ONLY
def test_streaming_origin_sort_externalizes_before_eos_and_releases_inputs(
    two_gpu_ray_runtime,
):
    run_root = _run_root("origin")
    source = _SlowOriginDatasource()

    output, stats = _execute_gpu_sort(
        lambda: ray.data.read_datasource(source, override_num_blocks=1),
        keys=["Origin"],
        descending=[False],
        run_root=run_root,
    )

    total_rows = _ORIGIN_BLOCKS * _ORIGIN_ROWS_PER_BLOCK
    assert output.schema.equals(_ORIGIN_SCHEMA, check_metadata=True)
    assert output.num_rows == total_rows
    origins = output["Origin"].to_pylist()
    row_ids = output["row_id"].to_pylist()
    assert origins == sorted(origins)
    assert Counter(origins) == Counter(
        _ORIGINS[(row_id * 17 + (row_id // _ORIGIN_ROWS_PER_BLOCK) * 3) % len(_ORIGINS)]
        for row_id in range(total_rows)
    )
    assert sorted(row_ids) == list(range(total_rows))

    # A successful run plus these exact counts proves the reader's 24 yielded
    # blocks arrived as 24 singleton RefBundles: the operator rejects any
    # bundle containing more than one ObjectRef.
    assert stats["input_rows"] == total_rows
    assert stats["input_object_refs_received"] == _ORIGIN_BLOCKS
    assert stats["released_input_object_refs"] == _ORIGIN_BLOCKS
    assert stats["all_input_object_refs_released"] is True
    assert stats["sampled_block_count"] == _ORIGIN_BLOCKS
    assert stats["ingest_rpc_count"] == _ORIGIN_BLOCKS
    assert sum(rank["input_blocks"] for rank in stats["ranks"]) == _ORIGIN_BLOCKS
    assert stats["source_run_count"] == _ORIGIN_BLOCKS
    assert stats["source_run_count"] > len(stats["ranks"])
    assert all(rank["source_run_count"] > 1 for rank in stats["ranks"])
    assert stats["merge_pass_count"] + stats["direct_final_merge_count"] > 0
    assert stats["input_buffer_budget_bytes"] == _INPUT_BUFFER_BUDGET_BYTES
    assert stats["input_buffer_within_bound"] is True
    assert stats["peak_buffered_input_bytes"] <= (
        _INPUT_BUFFER_BUDGET_BYTES + stats["max_input_block_bytes"]
    )

    assert stats["ranks_started_at_ns"] <= stats["first_input_received_at_ns"]
    assert stats["first_gpu_run_committed_at_ns"] < stats["last_input_received_at_ns"]
    assert stats["first_gpu_run_committed_at_ns"] < stats["inputs_complete_at_ns"]
    assert stats["gpu_processing_began_before_eos"] is True
    _assert_external_local_run_proof(stats, run_root)


@_GPU_ONLY
def test_streaming_multikey_float_null_nan_matches_arrow(
    two_gpu_ray_runtime,
):
    run_root = _run_root("float-null-nan")
    rows = 2048
    score_values = [
        None,
        float("nan"),
        float("-inf"),
        -3.5,
        -0.0,
        0.0,
        2.25,
        float("inf"),
    ]
    scores = [
        score_values[(row_id * 5 + 3) % len(score_values)] for row_id in range(rows)
    ]
    secondary = [
        None if row_id % 11 == 0 else (row_id * 7) % 13 for row_id in range(rows)
    ]
    input_table = pa.Table.from_arrays(
        [
            pa.array(scores, type=pa.float64()),
            pa.array(secondary, type=pa.int64()),
            pa.array(range(rows), type=pa.int64()),
            pa.array([f"payload-{row_id % 29}" for row_id in range(rows)]),
        ],
        schema=pa.schema(
            [
                pa.field("score", pa.float64(), nullable=True),
                pa.field("secondary", pa.int64(), nullable=True),
                pa.field("row_id", pa.int64(), nullable=False),
                pa.field("payload", pa.string(), nullable=False),
            ]
        ),
    )
    input_blocks = [input_table.slice(offset, 256) for offset in range(0, rows, 256)]
    keys = ["score", "secondary", "row_id"]
    descending = [False, True, False]

    output, stats = _execute_gpu_sort(
        lambda: ray.data.from_arrow(input_blocks),
        keys=keys,
        descending=descending,
        run_root=run_root,
    )

    expected_indices = pc.sort_indices(
        input_table,
        sort_keys=[
            ("score", "ascending"),
            ("secondary", "descending"),
            ("row_id", "ascending"),
        ],
        null_placement="at_end",
    )
    expected = input_table.take(expected_indices)
    assert output.schema.equals(input_table.schema, check_metadata=True)
    assert output.num_rows == rows
    assert output["row_id"].to_pylist() == expected["row_id"].to_pylist()
    assert output["secondary"].to_pylist() == expected["secondary"].to_pylist()
    assert output["payload"].to_pylist() == expected["payload"].to_pylist()
    for actual, wanted in zip(
        output["score"].to_pylist(), expected["score"].to_pylist()
    ):
        if wanted is None:
            assert actual is None
        elif math.isnan(wanted):
            assert actual is not None and math.isnan(actual)
        else:
            assert actual == wanted

    assert stats["input_rows"] == rows
    assert stats["all_input_object_refs_released"] is True
    assert stats["input_buffer_within_bound"] is True
    _assert_external_local_run_proof(stats, run_root)
