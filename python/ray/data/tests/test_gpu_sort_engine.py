import math
import sys
import time
import types

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import ray
from ray.data._internal.gpu_sort import actor as gpu_sort_actor
from ray.data._internal.gpu_sort.backend import (
    _ExternalRun,
    _PreparedBatch,
    _RunChunk,
    _coalesce_arrow_inputs,
    _cpu_sample_boundaries,
    _input_batch_target_bytes,
    _sampled_arrow_row_weights,
    _scale_sample_weights,
    _scale_sample_weights_by_stratum,
    _stratified_sample_indices,
    _workspace_bounded_payload_bytes,
    lazy_load_backend,
)
from ray.data._internal.gpu_sort.config import GPUSortCapacityError, GPUSortConfig
from ray.data._internal.gpu_sort.operator import (
    _MPF_PROGRESS_RESERVE_BYTES,
    _InputBlock,
    _RankPool,
    _allocate_stratified_sample_quotas,
    _assign_blocks_by_locality,
    _communication_environment,
    _make_waves,
    _operator_config,
    _plan_exchange_round,
    _sampling_plan_digest,
    _validate_gpu_schema,
    _wave_target_bytes,
)
from ray.data._internal.logical.rules.combine_shuffles import CombineShuffles
from ray.data.tests.conftest import *  # noqa
from ray.tests.conftest import *  # noqa


MiB = 1 << 20


def _stub_module(monkeypatch, name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _stub_synchronous_cupy(monkeypatch):
    return _stub_module(
        monkeypatch,
        "cupy",
        cuda=types.SimpleNamespace(
            runtime=types.SimpleNamespace(deviceSynchronize=lambda: None)
        ),
    )


def _new_backend(backend_class, **attributes):
    backend = object.__new__(backend_class)
    backend.__dict__.update(attributes)
    return backend


class _PreparationTable:
    def __init__(self, rows, bytes_per_row=16 * MiB):
        self.num_rows = rows
        self.bytes_per_row = bytes_per_row
        self.nbytes = rows * bytes_per_row

    def slice(self, start, length=None):
        remaining = self.num_rows - start
        rows = remaining if length is None else min(remaining, length)
        return _PreparationTable(rows, self.bytes_per_row)


class _DeviceTable:
    def __init__(self, rows, size_bytes):
        self.rows = rows
        self.size_bytes = size_bytes

    def num_rows(self):
        return self.rows


@pytest.fixture
def gpu_backend_class(monkeypatch):
    """Load the backend methods without importing optional RAPIDS packages."""

    _stub_module(
        monkeypatch,
        "rapidsmpf.utils.ray_utils",
        BaseShufflingActor=type("BaseShufflingActor", (), {}),
    )
    return lazy_load_backend()


@pytest.fixture
def preparation_backend(gpu_backend_class, monkeypatch):
    """Build the minimal actor state needed by ``_prepare_more`` tests."""

    _stub_synchronous_cupy(monkeypatch)
    _stub_module(
        monkeypatch,
        "cudf",
        DataFrame=types.SimpleNamespace(from_arrow=lambda arrow: arrow),
    )

    def make(
        tables,
        *,
        memory=None,
        pool_max_bytes=400 * MiB,
        exchange_batch_bytes=512 * MiB,
        row_ordinal=0,
    ):
        memory = memory or types.SimpleNamespace(current_allocated=0)
        backend = _new_backend(
            gpu_backend_class,
            _config=GPUSortConfig(exchange_batch_bytes=exchange_batch_bytes),
            _pool_max_bytes=pool_max_bytes,
            _mr=memory,
            _prepared_wave_id=0,
            _prepared_batches={},
            _pending_arrow_batches=iter(tables),
            _pending_arrow_owners=[],
            _next_arrow_batch=None,
            _source_input_complete=False,
            _next_prepared_batch_id=0,
            _row_ordinal=row_ordinal,
            _stats={
                "input_batch_count": 0,
                "max_input_batch_bytes": 0,
                "min_input_batch_target_bytes": 0,
                "max_input_batch_target_bytes": 0,
                "h2d_bytes": 0,
                "prepared_batch_count": 0,
                "prepared_bytes": 0,
                "preparation_retry_count": 0,
            },
        )
        backend._estimated_device_bytes = lambda table: table.nbytes
        backend._update_peak = lambda: None
        return backend, memory

    return make


@pytest.fixture
def externalization_backend(gpu_backend_class, monkeypatch):
    """Build the shared state for bounded external-run tests."""

    _stub_synchronous_cupy(monkeypatch)
    _stub_module(monkeypatch, "pylibcudf")

    def make(
        tables,
        *,
        memory,
        pool_max_bytes=100,
        payload_limit_bytes=40,
        slice_table=None,
    ):
        backend = _new_backend(
            gpu_backend_class,
            _config=GPUSortConfig(final_sort_workspace_factor=3.0),
            _pool_max_bytes=pool_max_bytes,
            _mr=memory,
            _payload_limit_bytes=payload_limit_bytes,
            _device_tables={0: list(tables)},
            _device_bytes={0: sum(table.size_bytes for table in tables)},
            _runs={0: []},
            _started_at=0.0,
            _stats={
                "state": "DEVICE_ACCUMULATING",
                "mode": "resident",
                "first_externalize_s": None,
                "first_externalize_wave": None,
                "peak_device_bytes": 0,
                "run_sort_oom_retry_count": 0,
                "phases_s": {"run_sort": 0.0},
            },
        )
        backend._slice_table = slice_table or (
            lambda table, start, end: _DeviceTable(
                end - start,
                table.size_bytes * (end - start) // table.rows,
            )
        )
        backend._table_bytes = lambda table: table.size_bytes
        backend._store_table_as_run = lambda table, initial: _ExternalRun()
        return backend

    return make


def test_gpu_sort_api_and_logical_backend(ray_start_regular):
    ds = ray.data.from_arrow(pa.table({"k": [2, 1], "payload": ["b", "a"]}))

    assert ds.sort("k")._logical_plan.dag.backend == "cpu"
    gpu_ds = ds.sort("k", backend="gpu")
    gpu_sort = gpu_ds._logical_plan.dag
    assert gpu_sort.backend == "gpu"
    assert gpu_ds._plan.require_preserve_order()
    assert (
        CombineShuffles._combine(
            gpu_ds.sort("payload", backend="gpu")._logical_plan.dag
        ).backend
        == "gpu"
    )

    with pytest.raises(ValueError, match="backend"):
        ds.sort("k", backend="cuda")
    with pytest.raises(ValueError, match="boundaries"):
        ds.sort("k", boundaries=[1], backend="gpu")
    with pytest.raises(NotImplementedError, match="flat Arrow scalar"):
        ray.data.from_arrow(pa.table({"k": [[1], [2]]})).sort("k", backend="gpu")


def test_gpu_sort_schema_and_wave_planning():
    schema = pa.schema(
        [
            pa.field("s", pa.string()),
            pa.field("i", pa.int64()),
            pa.field("f", pa.float64()),
            pa.field("flag", pa.bool_()),
            pa.field("ts", pa.timestamp("us")),
        ]
    )
    _validate_gpu_schema(schema, ["s", "i", "f", "flag", "ts"])
    with pytest.raises(NotImplementedError, match="unsupported keys"):
        _validate_gpu_schema(pa.schema([pa.field("blob", pa.binary())]), ["blob"])

    assert GPUSortConfig(residency_budget_bytes="4 GiB").residency_budget_bytes == (
        4 << 30
    )
    config = GPUSortConfig()
    assert config.rmm_initial_fraction == config.rmm_max_fraction == 0.85
    local_config = GPUSortConfig(
        external_run_store="local_disk",
        external_run_directory="/raid/gpu-sort-runs",
        external_run_id="sort_123",
        external_run_min_free_bytes="4 TiB",
        external_run_max_live_bytes="2 GiB",
    )
    assert local_config.external_run_min_free_bytes == 4 << 40
    assert local_config.external_run_max_live_bytes == 2 << 30
    with pytest.raises(ValueError, match="external_run_store"):
        GPUSortConfig(external_run_store="disk")
    with pytest.raises(ValueError, match="external_run_directory"):
        GPUSortConfig(
            external_run_store="local_disk",
            external_run_directory="relative",
            external_run_id="sort",
        )
    with pytest.raises(ValueError, match="byte count"):
        GPUSortConfig(external_run_min_free_bytes=False)
    blocks = [
        [_InputBlock("a", 6, 1), _InputBlock("b", 6, 1)],
        [_InputBlock("c", 3, 1)],
    ]
    waves = _make_waves(blocks, target_bytes_per_rank=8)
    assert waves == [[["a"], ["c"]], [["b"], []]]
    assert _make_waves(blocks, target_bytes_per_rank=None) == [[["a", "b"], ["c"]]]


def test_gpu_sort_operator_config_wires_local_run_store_controls():
    values = {
        "gpu_sort_external_run_store": "local_disk",
        "gpu_sort_external_run_directory": "/raid/gpu-sort-runs",
        "gpu_sort_external_run_min_free_bytes": 4 << 40,
        "gpu_sort_external_run_max_live_bytes": 1 << 40,
        "gpu_sort_merge_fan_in": 7,
    }
    context = types.SimpleNamespace(
        get_config=lambda name, default: values.get(name, default)
    )

    config = _operator_config(context)

    assert config["external_run_store"] == "local_disk"
    assert config["external_run_directory"] == "/raid/gpu-sort-runs"
    assert len(config["external_run_id"]) == 32
    assert config["external_run_min_free_bytes"] == 4 << 40
    assert config["external_run_max_live_bytes"] == 1 << 40
    assert config["merge_fan_in"] == 7


def test_gpu_sort_communication_environment_has_actor_native_defaults(monkeypatch):
    names = (
        "RAY_DATA_GPU_SORT_UCX_TLS",
        "UCX_TLS",
        "RAY_DATA_GPU_SORT_UCX_SOCKADDR_TLS_PRIORITY",
        "UCX_SOCKADDR_TLS_PRIORITY",
        "RAY_DATA_GPU_SORT_UCXX_PROGRESS_MODE",
        "RAPIDSMPF_UCXX_PROGRESS_MODE",
        "UCX_MEMTYPE_CACHE",
        "UCX_LOG_LEVEL",
        "RAPIDSMPF_LOG",
        "CUDF_SPILL",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    context = types.SimpleNamespace(get_config=lambda _name, default: default)

    assert _communication_environment(context) == {
        "UCX_TLS": "cuda_copy,cuda_ipc,sm,tcp",
        "UCX_SOCKADDR_TLS_PRIORITY": "tcp",
        "UCX_MEMTYPE_CACHE": "n",
        "UCX_LOG_LEVEL": "warn",
        "RAPIDSMPF_LOG": "WARN",
        "RAPIDSMPF_UCXX_PROGRESS_MODE": "polling",
        "CUDF_SPILL": "0",
    }


def test_gpu_sort_communication_environment_honors_safe_overrides(monkeypatch):
    monkeypatch.setenv("UCX_TLS", "cuda_copy,sm,tcp")
    monkeypatch.setenv("UCX_SOCKADDR_TLS_PRIORITY", "sockcm")
    monkeypatch.setenv("RAPIDSMPF_UCXX_PROGRESS_MODE", "thread-blocking")
    # These settings are intentionally fixed for allocator and benchmark safety.
    monkeypatch.setenv("UCX_MEMTYPE_CACHE", "try")
    monkeypatch.setenv("UCX_LOG_LEVEL", "debug")
    monkeypatch.setenv("RAPIDSMPF_LOG", "TRACE")
    monkeypatch.setenv("CUDF_SPILL", "1")
    inherited_context = types.SimpleNamespace(get_config=lambda _name, default: default)

    inherited = _communication_environment(inherited_context)

    assert inherited["UCX_TLS"] == "cuda_copy,sm,tcp"
    assert inherited["UCX_SOCKADDR_TLS_PRIORITY"] == "sockcm"
    assert inherited["RAPIDSMPF_UCXX_PROGRESS_MODE"] == "thread-blocking"
    assert inherited["UCX_MEMTYPE_CACHE"] == "n"
    assert inherited["UCX_LOG_LEVEL"] == "warn"
    assert inherited["RAPIDSMPF_LOG"] == "WARN"
    assert inherited["CUDF_SPILL"] == "0"

    values = {
        "gpu_sort_ucx_tls": "cuda_copy,cuda_ipc,sm,tcp",
        "gpu_sort_ucx_sockaddr_tls_priority": "tcp",
        "gpu_sort_ucxx_progress_mode": "thread-polling",
    }
    context = types.SimpleNamespace(
        get_config=lambda name, default: values.get(name, default)
    )

    environment = _communication_environment(context)

    assert environment == {
        "UCX_TLS": "cuda_copy,cuda_ipc,sm,tcp",
        "UCX_SOCKADDR_TLS_PRIORITY": "tcp",
        "UCX_MEMTYPE_CACHE": "n",
        "UCX_LOG_LEVEL": "warn",
        "RAPIDSMPF_LOG": "WARN",
        "RAPIDSMPF_UCXX_PROGRESS_MODE": "thread-polling",
        "CUDF_SPILL": "0",
    }


def test_gpu_sort_rank_pool_installs_actor_scoped_communication_environment(
    monkeypatch,
):
    environment = {
        "UCX_TLS": "cuda_copy,cuda_ipc,sm,tcp",
        "UCX_SOCKADDR_TLS_PRIORITY": "tcp",
        "UCX_MEMTYPE_CACHE": "n",
        "UCX_LOG_LEVEL": "warn",
        "RAPIDSMPF_LOG": "WARN",
        "RAPIDSMPF_UCXX_PROGRESS_MODE": "thread-polling",
        "CUDF_SPILL": "0",
    }
    option_calls = []
    constructor_calls = []

    class RemoteMethod:
        def __init__(self, function):
            self._function = function

        def remote(self, *args):
            return self._function(*args)

    class Actor:
        def __init__(self, index):
            self.setup_root = RemoteMethod(lambda: (0, b"root"))
            self.setup_worker = RemoteMethod(
                lambda _address: {
                    "rank": index,
                    "memory_budget_bytes": 1,
                    "communication_environment": dict(environment),
                }
            )
            self.is_ready = RemoteMethod(lambda: True)

    class ActorFactory:
        @classmethod
        def options(cls, **options):
            option_calls.append(options)
            return cls()

        def remote(self, **arguments):
            constructor_calls.append(arguments)
            return Actor(arguments["index"])

    monkeypatch.setattr(gpu_sort_actor, "GPUSortActor", ActorFactory)
    monkeypatch.setattr(ray, "get", lambda value, timeout=None: value)
    pool = _RankPool(
        2,
        ["Origin"],
        [True],
        {"setup_timeout_s": 1.0},
        environment,
    )

    pool.start()

    assert len(pool.actors) == 2
    assert all(
        options["runtime_env"] == {"env_vars": environment} for options in option_calls
    )
    assert all(
        arguments["communication_environment"] == environment
        for arguments in constructor_calls
    )
    assert all(
        info["communication_environment"] == environment for info in pool.rank_infos
    )


def test_gpu_sort_actor_reports_effective_communication_environment(monkeypatch):
    environment = {
        "UCX_TLS": "cuda_copy,cuda_ipc,sm,tcp",
        "UCX_SOCKADDR_TLS_PRIORITY": "tcp",
        "UCX_MEMTYPE_CACHE": "n",
        "UCX_LOG_LEVEL": "warn",
        "RAPIDSMPF_LOG": "WARN",
        "RAPIDSMPF_UCXX_PROGRESS_MODE": "thread-polling",
        "CUDF_SPILL": "0",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    class Backend:
        def __init__(self, **_arguments):
            pass

        def setup_worker(self, _address):
            return {"rank": 0, "memory_budget_bytes": 123}

        def diagnostics(self):
            return {"rank": 0, "memory_budget_bytes": 123}

    from ray.data._internal.gpu_sort import backend as gpu_sort_backend

    monkeypatch.setattr(gpu_sort_backend, "get_backend_class", lambda: Backend)
    monkeypatch.setattr(
        ray,
        "get_runtime_context",
        lambda: types.SimpleNamespace(get_node_id=lambda: "node-1"),
    )
    actor_class = gpu_sort_actor.GPUSortActor.__ray_metadata__.modified_class
    actor = actor_class(
        nranks=1,
        index=0,
        key_columns=["Origin"],
        ascending=[True],
        num_partitions=1,
        config={},
        communication_environment=environment,
    )

    setup = actor.setup_worker(b"root")
    diagnostics = actor.diagnostics()

    assert setup["communication_environment"] == environment
    assert diagnostics["communication_environment"] == environment
    assert setup["node_id"] == diagnostics["node_id"] == "node-1"


def test_gpu_sort_bounded_input_batching_preserves_order_and_chunks():
    tables = [
        pa.table({"value": [0, 1]}),
        pa.table({"value": [2, 3, 4]}),
        pa.table({"value": pa.array([], type=pa.int64())}),
        pa.table({"value": [5, 6, 7, 8, 9]}),
        pa.table({"value": list(range(10, 20))}),
    ]
    batches = list(_coalesce_arrow_inputs(tables, target_bytes=64))

    assert [batch.nbytes for batch in batches] == [64, 64, 32]
    assert [batch["value"].to_pylist() for batch in batches] == [
        list(range(8)),
        list(range(8, 16)),
        list(range(16, 20)),
    ]
    assert all(batch.nbytes <= 64 for batch in batches)
    assert batches[0]["value"].num_chunks == 3
    with pytest.raises(GPUSortCapacityError, match="One Arrow input row"):
        list(_coalesce_arrow_inputs([pa.table({"value": ["x" * 64]})], 32))
    gib = 1 << 30
    assert _input_batch_target_bytes(1 * gib, 8 * gib, 2 * gib, 3.0) == gib
    assert _input_batch_target_bytes(1 * gib, 8 * gib, 6 * gib, 3.0) == (2 * gib // 3)
    assert _input_batch_target_bytes(1 * gib, 8 * gib, 8 * gib, 3.0) == 1


def test_gpu_sort_preparation_recalculates_headroom_and_makes_progress(
    preparation_backend,
):
    tables = [_PreparationTable(64, MiB) for _ in range(4)]
    backend, memory = preparation_backend(tables, pool_max_bytes=672 * MiB)

    def partition_and_pack(frame, _wave):
        memory.current_allocated += frame.nbytes
        return {0: object()}, (frame.nbytes,)

    backend._partition_and_pack = partition_and_pack
    backend._prepare_more()

    assert len(backend._prepared_batches) == 3
    assert memory.current_allocated == 192 * MiB
    assert not backend._source_input_complete

    backend._prepared_batches.clear()
    memory.current_allocated = 0
    backend._prepare_more()
    assert len(backend._prepared_batches) == 1
    assert backend._source_input_complete
    assert backend._stats["prepared_batch_count"] == 4


def test_gpu_sort_preparation_retries_smaller_prefix_before_exchange(
    preparation_backend,
):
    backend, memory = preparation_backend([_PreparationTable(4)], row_ordinal=7)
    attempts = []

    def partition_and_pack(frame, _wave):
        attempts.append(frame.num_rows)
        backend._row_ordinal += frame.num_rows
        if len(attempts) == 1:
            # The exact packed representation exceeds the Arrow-based bound.
            return {0: object()}, (120 * MiB,)
        if len(attempts) == 2:
            # Actual partition/pack scratch can still exceed the estimate.
            raise MemoryError("simulated RMM preparation OOM")
        memory.current_allocated += frame.nbytes
        return {0: object()}, (frame.nbytes,)

    backend._partition_and_pack = partition_and_pack
    backend._prepare_more()

    assert attempts == [4, 2, 1, 3]
    assert backend._stats["preparation_retry_count"] == 2
    assert backend._stats["prepared_batch_count"] == 2
    assert [
        batch.destination_bytes for batch in backend._prepared_batches.values()
    ] == [(16 * MiB,), (48 * MiB,)]
    assert backend._stats["prepared_bytes"] == 64 * MiB
    assert backend._row_ordinal == 11
    assert backend._source_input_complete


def test_gpu_sort_packed_metadata_uses_source_group_budget(preparation_backend):
    backend, memory = preparation_backend(
        [_PreparationTable(3)], exchange_batch_bytes=32 * MiB
    )
    attempts = []

    def partition_and_pack(frame, _wave):
        attempts.append(frame.num_rows)
        backend._row_ordinal += frame.num_rows
        physical = frame.nbytes + 8 * MiB
        memory.current_allocated += physical
        return {0: object()}, (physical,)

    backend._partition_and_pack = partition_and_pack
    backend._prepare_more()

    # Fixed pack metadata may exceed the decoded-payload batch cap without
    # exceeding the exact physical budget for the whole prepared source group.
    assert attempts == [2, 1]
    assert [
        batch.destination_bytes for batch in backend._prepared_batches.values()
    ] == [(40 * MiB,), (24 * MiB,)]
    assert backend._stats["prepared_bytes"] == 64 * MiB
    assert backend._stats["preparation_retry_count"] == 0
    assert backend._row_ordinal == 3
    assert backend._source_input_complete


def test_gpu_sort_preparation_defers_row_beyond_group_remainder(
    preparation_backend,
):
    backend, memory = preparation_backend(
        [_PreparationTable(4), _PreparationTable(1)], pool_max_bytes=340 * MiB
    )
    attempts = []

    def partition_and_pack(frame, _wave):
        attempts.append(frame.num_rows)
        backend._row_ordinal += frame.num_rows
        physical = frame.nbytes + 6 * MiB if frame.num_rows == 4 else 24 * MiB
        memory.current_allocated += physical
        return {0: object()}, (physical,)

    backend._partition_and_pack = partition_and_pack
    backend._prepare_more()

    # The first batch uses 70 of the 81 MiB source-group budget. The next row
    # is untouched when its 16 MiB estimate cannot fit the 11 MiB remainder.
    assert attempts == [4]
    assert backend._next_arrow_batch.num_rows == 1
    assert backend._row_ordinal == 4
    assert not backend._source_input_complete

    backend._prepared_batches.clear()
    memory.current_allocated = 0
    backend._prepare_more()

    assert attempts == [4, 1]
    assert backend._row_ordinal == 5
    assert backend._source_input_complete
    assert backend._stats["prepared_batch_count"] == 2


def test_gpu_sort_preparation_does_not_swallow_input_capacity_error(
    preparation_backend,
):
    def batches():
        yield _PreparationTable(4)
        raise GPUSortCapacityError("One Arrow input row exceeds the batch cap.")

    backend, memory = preparation_backend(batches())

    def partition_and_pack(frame, _wave):
        backend._row_ordinal += frame.num_rows
        memory.current_allocated += 70 * MiB
        return {0: object()}, (70 * MiB,)

    backend._partition_and_pack = partition_and_pack

    with pytest.raises(GPUSortCapacityError, match="exceeds the batch cap"):
        backend._prepare_more()
    assert backend._stats["prepared_batch_count"] == 1
    assert backend._row_ordinal == 4
    assert backend._next_arrow_batch is None


def test_gpu_sort_single_key_direct_sort_falls_back_for_payload(
    monkeypatch, gpu_backend_class
):
    calls = []
    _stub_module(
        monkeypatch,
        "pylibcudf",
        sorting=types.SimpleNamespace(
            sort=lambda table, order, nulls: calls.append("sort") or table,
            sort_by_key=lambda table, keys, order, nulls: (
                calls.append("sort_by_key") or table
            ),
        ),
    )

    backend = _new_backend(
        gpu_backend_class,
        _work_names=["key"],
        _merge_key_indices=[0],
        _order_and_nulls=lambda: (["ascending"], ["after"]),
        _comparison_table=lambda value, _names: ("keys", value),
    )
    table = object()

    assert backend._sort_table(table) is table
    assert calls == ["sort"]

    backend._work_names = ["key", "payload"]
    assert backend._sort_table(table) is table
    assert calls == ["sort", "sort_by_key"]


def test_gpu_sort_locality_assignment_balances_decoded_bytes():
    blocks = [
        _InputBlock("n0-large", 9, 1),
        _InputBlock("n0-small", 4, 1),
        _InputBlock("n1", 7, 1),
        _InputBlock("replicated", 7, 1),
        _InputBlock("unknown", 6, 1),
    ]
    locations = {
        "n0-large": {"node_ids": ["node-0"]},
        "n0-small": {"node_ids": ["node-0"]},
        "n1": {"node_ids": ["node-1"]},
        "replicated": {"node_ids": ["node-0", "node-1"]},
    }
    (
        assigned,
        assigned_bytes,
        assigned_blocks,
        local_bytes,
        local_blocks,
    ) = _assign_blocks_by_locality(blocks, ["node-0", "node-1"], locations)

    assert [[block.value for block in rank] for rank in assigned] == [
        ["n0-large", "n0-small", "unknown"],
        ["n1", "replicated"],
    ]
    assert assigned_bytes == [19, 14]
    assert assigned_blocks == [3, 2]
    assert local_bytes == [13, 14]
    assert local_blocks == [2, 2]


def test_gpu_sort_automatic_wave_target_uses_smallest_actor_budget():
    gib = 1 << 30
    small = [
        [_InputBlock("a", 2 * gib, 1)],
        [_InputBlock("b", 3 * gib, 1)],
    ]
    assert (
        _wave_target_bytes(
            small,
            explicit_residency_budget_bytes=None,
            actor_usable_budgets=[20 * gib, 16 * gib],
            auto_wave_fraction=0.50,
        )
        is None
    )

    large = [
        [_InputBlock("a", 5 * gib, 1), _InputBlock("b", 4 * gib, 1)],
        [_InputBlock("c", 2 * gib, 1)],
    ]
    assert (
        _wave_target_bytes(
            large,
            explicit_residency_budget_bytes=None,
            actor_usable_budgets=[20 * gib, 16 * gib],
            auto_wave_fraction=0.50,
        )
        == (16 * gib - _MPF_PROGRESS_RESERVE_BYTES) // 4
    )
    assert (
        _wave_target_bytes(
            large,
            explicit_residency_budget_bytes=None,
            actor_usable_budgets=[20 * gib, 16 * gib],
            auto_wave_fraction=0.375,
        )
        == (16 * gib - _MPF_PROGRESS_RESERVE_BYTES) // 4
    )
    assert _wave_target_bytes(
        large,
        explicit_residency_budget_bytes=None,
        actor_usable_budgets=[20 * gib, 16 * gib],
        auto_wave_fraction=0.10,
    ) == int(16 * gib * 0.10)
    assert (
        _wave_target_bytes(
            large,
            explicit_residency_budget_bytes=None,
            actor_usable_budgets=[512 << 20],
            auto_wave_fraction=0.50,
        )
        == (512 << 20) // 4 - _MPF_PROGRESS_RESERVE_BYTES // 4
    )
    assert (
        _wave_target_bytes(
            large,
            explicit_residency_budget_bytes=4 * gib,
            actor_usable_budgets=[],
            auto_wave_fraction=0.50,
        )
        == 2 * gib
    )

    with pytest.raises(RuntimeError, match="usable memory budgets"):
        _wave_target_bytes(
            large,
            explicit_residency_budget_bytes=None,
            actor_usable_budgets=[],
            auto_wave_fraction=0.50,
        )
    with pytest.raises(ValueError, match="wave fraction"):
        GPUSortConfig(auto_wave_fraction=0)
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        GPUSortConfig(sample_seed=-1)
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        GPUSortConfig(sample_seed=1 << 64)


def test_gpu_sort_skewed_shuffle_preserves_mpf_progress_headroom():
    ranks = 20
    pool_bytes = int(18.729156 * (1 << 30))
    failed_allocation = 17_576_192
    safety_margin = 64 << 20

    # The superseded B/4 policy exactly filled the RMM pool and could not make
    # the allocation observed in RAPIDS-MPF's progress thread.
    old_target = pool_bytes // 4
    assert pool_bytes - 4 * old_target < failed_allocation + safety_margin
    assert _MPF_PROGRESS_RESERVE_BYTES >= failed_allocation + safety_margin

    batch_bytes = 128 << 20
    prepared_bytes = 4 << 30
    current = [prepared_bytes] * ranks
    maximum = [pool_bytes] * ranks

    balanced = []
    for source in range(ranks):
        batches = []
        for batch in range(32):
            row = [batch_bytes // ranks] * ranks
            for destination in range(batch_bytes % ranks):
                row[(source + destination) % ranks] += 1
            batches.append({"batch_id": batch, "destination_bytes": row})
        balanced.append(batches)
    balanced_round = _plan_exchange_round(
        balanced,
        current_allocated_bytes=current,
        pool_max_bytes=maximum,
    )
    assert all(ids == tuple(range(32)) for ids in balanced_round.batch_ids_by_rank)
    assert min(balanced_round.modeled_headroom_bytes) >= (_MPF_PROGRESS_RESERVE_BYTES)

    all_to_one = [
        [
            {
                "batch_id": batch,
                "destination_bytes": [batch_bytes] + [0] * (ranks - 1),
            }
            for batch in range(8)
        ]
        for _ in range(ranks)
    ]
    first = _plan_exchange_round(
        all_to_one,
        current_allocated_bytes=current,
        pool_max_bytes=maximum,
    )
    assert 0 < sum(map(len, first.batch_ids_by_rank)) < ranks * 8
    assert min(first.modeled_headroom_bytes) >= _MPF_PROGRESS_RESERVE_BYTES

    # The old source-only B/4 rule could fill the pool exactly and leave no
    # room for the native allocation that aborted RAPIDS-MPF's progress thread.
    assert pool_bytes - 4 * old_target < failed_allocation + safety_margin


def test_gpu_sort_balanced_five_gb_wave_stays_in_one_exchange_round():
    ranks = 20
    pool_bytes = 20_110_278_656
    prepared_bytes = 5_000_079_558
    baseline_bytes = 16 << 20
    batch_count = 10
    batch_base, extra_batches = divmod(prepared_bytes, batch_count)

    balanced = []
    for source in range(ranks):
        batches = []
        for batch in range(batch_count):
            batch_bytes = batch_base + (batch < extra_batches)
            destination_base, extra_destinations = divmod(batch_bytes, ranks)
            destinations = [destination_base] * ranks
            for offset in range(extra_destinations):
                destinations[(source + batch + offset) % ranks] += 1
            batches.append({"batch_id": batch, "destination_bytes": destinations})
        balanced.append(batches)

    plan = _plan_exchange_round(
        balanced,
        current_allocated_bytes=[prepared_bytes + baseline_bytes] * ranks,
        pool_max_bytes=[pool_bytes] * ranks,
    )

    assert all(ids == tuple(range(batch_count)) for ids in plan.batch_ids_by_rank)
    assert min(plan.modeled_headroom_bytes) >= _MPF_PROGRESS_RESERVE_BYTES


def test_gpu_sort_skewed_exchange_rounds_are_deterministic_and_make_progress():
    ranks = 4
    mib = 1 << 20
    maximum = [1024 * mib, 960 * mib, 1024 * mib, 1024 * mib]
    current = [320 * mib] * ranks
    pending = [
        [
            {
                "batch_id": batch,
                "destination_bytes": [64 * mib] + [0] * (ranks - 1),
            }
            for batch in range(4)
        ]
        for _ in range(ranks)
    ]
    expected = {(rank, batch) for rank in range(ranks) for batch in range(4)}
    observed = set()
    rounds = 0
    while any(pending):
        plan = _plan_exchange_round(
            pending,
            current_allocated_bytes=current,
            pool_max_bytes=maximum,
        )
        repeated = _plan_exchange_round(
            pending,
            current_allocated_bytes=current,
            pool_max_bytes=maximum,
        )
        assert plan == repeated
        assert min(plan.modeled_headroom_bytes) >= _MPF_PROGRESS_RESERVE_BYTES
        selected = [set(values) for values in plan.batch_ids_by_rank]
        for rank, values in enumerate(selected):
            assert not (observed & {(rank, value) for value in values})
            observed.update((rank, value) for value in values)
        pending = [
            [batch for batch in items if batch["batch_id"] not in selected[rank]]
            for rank, items in enumerate(pending)
        ]
        # MPF has shut down, received tails have been externalized, and the
        # selected prepared source owners have been released.
        current = [current[rank] - plan.outgoing_bytes[rank] for rank in range(ranks)]
        rounds += 1

    assert observed == expected
    assert rounds > 1
    impossible = [[{"batch_id": 0, "destination_bytes": [mib]}]]
    with pytest.raises(GPUSortCapacityError, match="No prepared GPU shuffle batch"):
        _plan_exchange_round(
            impossible,
            current_allocated_bytes=[1024 * mib - _MPF_PROGRESS_RESERVE_BYTES],
            pool_max_bytes=[1024 * mib],
        )


def test_gpu_sort_inverse_inclusion_sample_weights():
    sampled = np.asarray([5, 5], dtype=np.uint64)
    small = _scale_sample_weights(sampled, population_rows=4, sample_rows=2)
    large = _scale_sample_weights(sampled, population_rows=12, sample_rows=2)
    assert small.tolist() == [10, 10]
    assert large.tolist() == [30, 30]
    assert small.dtype == large.dtype == np.dtype("uint64")

    uneven = _scale_sample_weights(
        np.asarray([5, 7], dtype=np.uint64), population_rows=5, sample_rows=2
    )
    assert uneven.tolist() == [12, 17]


def test_gpu_sort_global_stratified_sample_allocation():
    blocks = [
        _InputBlock("empty", 0, 0, ordinal=0),
        _InputBlock("tiny", 1, 1, ordinal=1),
        _InputBlock("medium", 9, 9, ordinal=2),
        _InputBlock("large", 20, 20, ordinal=3),
    ]
    quotas, target = _allocate_stratified_sample_quotas(blocks, 8)
    assert target == sum(quotas) == 8
    assert quotas == [0, 1, 2, 5]

    # One sample per nonempty block takes precedence over a smaller configured
    # target, including for one-row blocks.
    tiny = [_InputBlock(str(i), 1, 1, ordinal=i) for i in range(4)]
    assert _allocate_stratified_sample_quotas(tiny, 2) == ([1, 1, 1, 1], 4)

    # Equal largest remainders are resolved by logical ordinal, not list/rank.
    tied = [
        _InputBlock("later", 3, 3, ordinal=5),
        _InputBlock("earlier", 3, 3, ordinal=2),
    ]
    assert _allocate_stratified_sample_quotas(tied, 3) == ([1, 2], 3)


def test_gpu_sort_deterministic_stratified_indices_and_exact_weights():
    indices, widths = _stratified_sample_indices(10, 3, seed=7, block_ordinal=4)
    assert indices.tolist() == [1, 5, 8]
    assert widths.tolist() == [3, 3, 4]
    assert 0 <= indices[0] < 3
    assert 3 <= indices[1] < 6
    assert 6 <= indices[2] < 10
    assert sum(widths) == 10
    assert _scale_sample_weights_by_stratum(
        np.asarray([5, 7, 11], dtype=np.uint64), widths
    ).tolist() == [15, 21, 44]

    repeated, repeated_widths = _stratified_sample_indices(
        10, 3, seed=7, block_ordinal=4
    )
    assert np.array_equal(indices, repeated)
    assert np.array_equal(widths, repeated_widths)
    changed, _ = _stratified_sample_indices(10, 3, seed=8, block_ordinal=4)
    assert not np.array_equal(indices, changed)
    changed_ordinal, _ = _stratified_sample_indices(10, 3, seed=7, block_ordinal=5)
    assert not np.array_equal(indices, changed_ordinal)


def test_gpu_sort_sampling_resolves_and_aggregates_incrementally(
    gpu_backend_class, monkeypatch
):
    tables = [
        pa.table({"key": [10], "payload": ["a"]}),
        pa.table({"key": [20], "payload": ["bb"]}),
        pa.table({"key": [30], "payload": ["ccc"]}),
    ]
    events = []

    def resolve(entries):
        assert len(entries) == 1
        index = entries[0]
        events.append(("resolve", index))
        return [tables[index]]

    monkeypatch.setattr(gpu_sort_actor, "_resolve_blocks", resolve)
    backend = object.__new__(gpu_backend_class)
    backend._key_columns = ["key"]
    backend._arrow_schema = None
    backend._float_hidden = {}
    backend._stats = {
        "input_rows": 0,
        "input_bytes": 0,
        "peak_device_bytes": 0,
        "phases_s": {"sampling": 0.0},
    }
    backend._mr = None
    backend._update_peak = lambda: None
    backend.rank = lambda: 0
    to_arrow = backend._to_arrow_table

    def record_processing(block):
        events.append(("process", block["key"][0].as_py()))
        return to_arrow(block)

    backend._to_arrow_table = record_processing
    result = backend.sample_blocks(
        gpu_sort_actor._iter_resolved_blocks([0, 1, 2]),
        block_ordinals=[7, 8, 9],
        sample_quotas=[1, 1, 1],
        seed=17,
    )

    assert events == [
        ("resolve", 0),
        ("process", 10),
        ("resolve", 1),
        ("process", 20),
        ("resolve", 2),
        ("process", 30),
    ]
    assert result["rows"] == result["sample_rows"] == 3
    assert result["sampled_block_count"] == 3
    assert result["input_bytes"] == sum(table.nbytes for table in tables)
    assert result["sample"][backend._sample_block_name].to_pylist() == [7, 8, 9]
    assert result["sample"]["key"].to_pylist() == [10, 20, 30]


def test_gpu_sort_stratified_sampling_is_locality_invariant_and_breaks_periodicity():
    blocks = [
        _InputBlock("a", 40, 40, ordinal=0),
        _InputBlock("b", 80, 80, ordinal=1),
        _InputBlock("c", 120, 120, ordinal=2),
    ]
    quotas, target = _allocate_stratified_sample_quotas(blocks, 48)
    plan_digest = _sampling_plan_digest(blocks, quotas, seed=91, target_rows=target)
    remapped = [blocks[2], blocks[0], blocks[1]]
    remapped_quotas = [quotas[2], quotas[0], quotas[1]]
    assert (
        _sampling_plan_digest(remapped, remapped_quotas, seed=91, target_rows=target)
        == plan_digest
    )
    assert len(plan_digest) == 64

    def coordinates(block_order, quota_order):
        result = []
        for block, quota in zip(block_order, quota_order):
            selected, _ = _stratified_sample_indices(
                block.num_rows,
                quota,
                seed=91,
                block_ordinal=block.ordinal,
            )
            result.extend(
                (block.ordinal, i, int(row)) for i, row in enumerate(selected)
            )
        return sorted(result)

    assert coordinates(blocks, quotas) == coordinates(remapped, remapped_quotas)

    weight_name = "__weight"
    block_name = "__block"
    stratum_name = "__stratum"
    index_name = "__index"

    def sample_table(block, quota):
        selected, widths = _stratified_sample_indices(
            block.num_rows,
            quota,
            seed=91,
            block_ordinal=block.ordinal,
        )
        return pa.table(
            {
                "key": pa.array(
                    [block.ordinal * 1_000 + int(row) for row in selected],
                    type=pa.int64(),
                ),
                weight_name: pa.array(widths, type=pa.uint64()),
                block_name: pa.array([block.ordinal] * quota, type=pa.uint64()),
                stratum_name: pa.array(range(quota), type=pa.uint64()),
                index_name: pa.array(selected, type=pa.uint64()),
            }
        )

    by_ordinal = {
        block.ordinal: sample_table(block, quota)
        for block, quota in zip(blocks, quotas)
    }
    boundary_args = {
        "schema": next(iter(by_ordinal.values())).schema,
        "key_columns": ["key"],
        "ascending": [True],
        "num_partitions": 4,
        "null_position": "last",
        "weight_name": weight_name,
        "sample_block_name": block_name,
        "sample_stratum_name": stratum_name,
        "sample_index_name": index_name,
    }
    original_result = _cpu_sample_boundaries(
        [
            pa.concat_tables([by_ordinal[0], by_ordinal[2]]),
            by_ordinal[1],
        ],
        **boundary_args,
    )
    remapped_result = _cpu_sample_boundaries(
        [
            by_ordinal[2],
            pa.concat_tables([by_ordinal[1], by_ordinal[0]]),
        ],
        **boundary_args,
    )
    assert (
        original_result["sample_index_digest"] == remapped_result["sample_index_digest"]
    )
    assert original_result["boundary_digest"] == remapped_result["boundary_digest"]
    assert original_result["sample_bytes"] > original_result["planning_sample_bytes"]

    # A stride of four could lock onto one phase of this periodic input. One
    # randomized selection in every four-row stratum represents every phase.
    selected, widths = _stratified_sample_indices(400, 100, seed=17, block_ordinal=8)
    assert set((selected % 4).tolist()) == {0, 1, 2, 3}
    assert set(widths.tolist()) == {4}


def test_gpu_sort_cpu_sampled_weights_and_boundaries():
    payload_schema = pa.schema(
        [
            pa.field("fixed", pa.int32(), nullable=False),
            pa.field("category", pa.string()),
            pa.field("score", pa.float64()),
        ]
    )
    payload = pa.Table.from_arrays(
        [
            pa.array([1, 2, 3], type=pa.int32()),
            pa.array(["xy", "x" * 1_000, None], type=pa.string()),
            pa.array([float("nan"), 1.0, None], type=pa.float64()),
        ],
        schema=payload_schema,
    )

    # Row selection precedes byte accounting, so the large unsampled string is
    # never included in the variable-width calculation.
    sampled = payload.take(pa.array([0, 2], type=pa.int64()))
    weights = _sampled_arrow_row_weights(sampled)
    assert weights.dtype == np.dtype("uint64")
    assert weights.tolist() == [20, 18]

    weight_name = "__sample_weight"
    planning_schema = pa.schema(
        [
            pa.field("category", pa.string()),
            pa.field("fixed", pa.int64(), nullable=False),
            pa.field("score", pa.float64()),
            pa.field(weight_name, pa.uint64(), nullable=False),
        ]
    )
    planning_sample = pa.Table.from_arrays(
        [
            pa.array(["b", "a", "a", "a", "a", "b", None]),
            pa.array([1, 2, 3, 3, 3, 1, 9], type=pa.int64()),
            pa.array(
                [float("nan"), None, 2.0, -1.0, -1.0, 5.0, 0.0],
                type=pa.float64(),
            ),
            pa.array([10] * 7, type=pa.uint64()),
        ],
        schema=planning_schema,
    )
    result = _cpu_sample_boundaries(
        [planning_sample.slice(0, 3), planning_sample.slice(3)],
        schema=planning_schema,
        key_columns=["category", "fixed", "score"],
        ascending=[True, False, True],
        num_partitions=7,
        null_position="last",
        weight_name=weight_name,
    )

    boundaries = result["boundaries"].to_pylist()
    assert boundaries[:4] == [
        {"category": "a", "fixed": 3, "score": -1.0},
        {"category": "a", "fixed": 3, "score": -1.0},
        {"category": "a", "fixed": 3, "score": 2.0},
        {"category": "a", "fixed": 2, "score": None},
    ]
    assert boundaries[4] == {"category": "b", "fixed": 1, "score": 5.0}
    assert boundaries[5]["category"] == "b"
    assert boundaries[5]["fixed"] == 1
    assert math.isnan(boundaries[5]["score"])
    assert boundaries[0] == boundaries[1]
    assert result["sample_rows"] == 7
    assert result["sample_bytes"] > 0


def test_gpu_sort_comparator_matches_arrow_null_nan_order(
    gpu_backend_class, monkeypatch
):
    order = types.SimpleNamespace(ASCENDING="asc", DESCENDING="desc")
    null_order = types.SimpleNamespace(BEFORE="before", AFTER="after")
    _stub_module(
        monkeypatch,
        "pylibcudf",
        types=types.SimpleNamespace(Order=order, NullOrder=null_order),
    )

    values = pa.array([None, float("nan"), -1.0, 2.0])
    expected = {
        True: [-1.0, 2.0, "nan", None],
        False: [2.0, -1.0, "nan", None],
    }
    for ascending in (True, False):
        indices = pc.sort_indices(
            pa.table({"key": values}),
            sort_keys=[("key", "ascending" if ascending else "descending")],
            null_placement="at_end",
        )
        actual = pc.take(values, indices).to_pylist()
        normalized = [
            "nan" if isinstance(value, float) and math.isnan(value) else value
            for value in actual
        ]
        assert normalized == expected[ascending]

        backend = _new_backend(
            gpu_backend_class,
            _key_columns=["key"],
            _ascending=[ascending],
            _float_hidden={"key": ("is_null", "is_nan")},
            _config=GPUSortConfig(null_position="last"),
        )
        orders, nulls = backend._order_and_nulls()
        assert orders == [
            order.ASCENDING,
            order.ASCENDING,
            order.ASCENDING if ascending else order.DESCENDING,
        ]
        assert nulls == [
            null_order.AFTER,
            null_order.AFTER,
            null_order.AFTER if ascending else null_order.BEFORE,
        ]

        backend._float_hidden = {}
        for null_position in ("first", "last"):
            backend._config = GPUSortConfig(null_position=null_position)
            direct_orders, direct_nulls = backend._order_and_nulls()
            assert direct_orders == [order.ASCENDING if ascending else order.DESCENDING]
            expected_null_order = (
                null_order.BEFORE
                if (null_position == "first") == ascending
                else null_order.AFTER
            )
            assert direct_nulls == [expected_null_order]


def test_gpu_sort_typed_all_null_arrow_output(gpu_backend_class):
    class Column:
        null_count = 3

        def to_arrow(self):
            raise AssertionError("all-null columns must use the typed Arrow path")

    class Series:
        _column = Column()

    class Frame:
        def __len__(self):
            return 3

        def __getitem__(self, name):
            return Series()

    schema = pa.schema(
        [
            pa.field("name", pa.string()),
            pa.field("count", pa.int64()),
            pa.field("ratio", pa.float64()),
            pa.field("when", pa.timestamp("us")),
        ]
    )
    table = gpu_backend_class._generic_frame_to_arrow(Frame(), schema)
    assert table.schema.equals(schema, check_metadata=True)
    assert table.num_rows == 3
    for field, column in zip(schema, table.columns):
        assert column.type.equals(field.type)
        assert column.null_count == 3


def test_gpu_sort_equal_keys_are_deterministically_balanced(
    gpu_backend_class, monkeypatch
):
    captured = {}

    class Table:
        def __init__(self, rows):
            self.rows = rows

        def num_rows(self):
            return self.rows

    class Frame:
        columns = ["key"]

        def __init__(self, rows):
            self.table = Table(rows)

    class Series:
        def __init__(self, values):
            self.values = list(values)

        @classmethod
        def from_pylibcudf(cls, values):
            return cls(values)

        def to_pylibcudf(self):
            return self.values, None

    def elementwise_kernel(_inputs, _output, operation, _name):
        assert "(base + static_cast<unsigned long long>(i)) % width" in operation

        def apply(lower, upper, base):
            return [
                low + (int(base) + index) % (high - low + 1)
                for index, (low, high) in enumerate(zip(lower, upper))
            ]

        return apply

    _stub_module(monkeypatch, "cupy", ElementwiseKernel=elementwise_kernel, uint64=int)
    _stub_module(monkeypatch, "cudf", Series=Series)
    plc = _stub_module(
        monkeypatch,
        "pylibcudf",
        search=types.SimpleNamespace(
            lower_bound=lambda *_: [1] * captured["rows"],
            upper_bound=lambda *_: [3] * captured["rows"],
        ),
    )

    def partition(table, destinations, num_partitions):
        captured["destinations"] = list(destinations)
        counts = [
            captured["destinations"].count(index) for index in range(num_partitions)
        ]
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        return table, offsets

    plc.partitioning = types.SimpleNamespace(partition=partition)
    plc.contiguous_split = types.SimpleNamespace(
        ChunkedPack=types.SimpleNamespace(
            create=lambda table, _size, _stream: types.SimpleNamespace(
                get_total_contiguous_size=lambda: table.num_rows()
            )
        )
    )
    _stub_module(
        monkeypatch,
        "rapidsmpf.integrations.cudf.partition",
        split_and_pack=lambda *_: list(captured["destinations"]),
    )
    _stub_module(
        monkeypatch,
        "rapidsmpf.utils.cudf",
        cudf_to_pylibcudf_table=lambda frame: frame.table,
    )
    _stub_module(monkeypatch, "rmm.pylibrmm.stream", DEFAULT_STREAM=object())

    backend = _new_backend(
        gpu_backend_class,
        _duplicate_kernel=None,
        _row_ordinal=0,
        _boundary_keys=object(),
        _num_partitions=4,
        _buffer_resource=object(),
        _augment_table=lambda table, names: (table, names),
        _comparison_table=lambda table, names: table,
        _order_and_nulls=lambda: ([], []),
        _slice_table=lambda table, start, end: Table(end - start),
        rank=lambda: 0,
    )

    def distribute(rows):
        captured["rows"] = rows
        return backend._partition_and_pack(Frame(rows), wave_id=0)

    first, first_bytes = distribute(8)
    second, second_bytes = distribute(8)
    assert first == [1, 2, 3, 1, 2, 3, 1, 2]
    assert first_bytes == (0, 3, 3, 2)
    assert second_bytes == (0, 3, 2, 3)
    assert set(first + second) == {1, 2, 3}
    counts = [(first + second).count(partition) for partition in (1, 2, 3)]
    assert max(counts) - min(counts) <= 1


def test_gpu_sort_transitions_before_residency_overflow(gpu_backend_class):
    class Table:
        def __init__(self, rows, size_bytes):
            self.rows = rows
            self.size_bytes = size_bytes

        def num_rows(self):
            return self.rows

    backend = object.__new__(gpu_backend_class)
    backend._payload_limit_bytes = 10
    backend._device_tables = {0: []}
    backend._device_bytes = {0: 0}
    backend._stats = {"peak_live_bytes": 0}
    backend._table_bytes = lambda table: table.size_bytes
    backend._slice_table = lambda table, start, end: Table(
        end - start, max(1, table.size_bytes * (end - start) // table.rows)
    )
    transitions = []

    def externalize(partition, wave_id):
        transitions.append((partition, wave_id, backend._device_bytes[partition]))
        backend._device_tables[partition] = []
        backend._device_bytes[partition] = 0

    backend._externalize_device_tables = externalize
    backend._accept_received(0, Table(6, 6), wave_id=7)
    backend._accept_received(0, Table(6, 6), wave_id=7)
    assert transitions == [(0, 7, 6)]
    assert backend._device_bytes[0] == 6

    backend._accept_received(0, Table(4, 4), wave_id=8)
    assert transitions[-1] == (0, 8, 10)
    assert backend._device_bytes[0] == 0
    assert backend._stats["peak_live_bytes"] <= backend._payload_limit_bytes


def test_gpu_sort_quiesces_shuffle_before_accepting_received_data(
    gpu_backend_class, monkeypatch
):
    events = []
    _stub_synchronous_cupy(monkeypatch)
    _stub_module(monkeypatch, "cudf")

    class Table:
        def num_rows(self):
            return 1

    class Shuffler:
        def __init__(self):
            self.done = False

        def insert_chunks(self, chunks):
            assert chunks == {0: "owned"}
            events.append("insert")
            # RAPIDS-MPF moves each PackedData pointer into the C++ shuffler.
            chunks.clear()

        def insert_finished(self, partitions):
            assert partitions == [0]

        def finished(self):
            return self.done

        def wait_any(self):
            self.done = True
            return 0

        def extract(self, partition):
            assert partition == 0
            return [object()]

        def shutdown(self):
            assert backend._prepared_batches
            events.append("shutdown")

    backend = _new_backend(
        gpu_backend_class,
        _boundary_keys=object(),
        _num_partitions=1,
        _buffer_resource=object(),
        _statistics=object(),
        _pool_max_bytes=100,
        _mr=types.SimpleNamespace(current_allocated=0),
        _config=GPUSortConfig(
            exchange_batch_bytes=10,
            final_sort_workspace_factor=3.0,
        ),
        _stats={
            "initial_run_count": 0,
            "forced_subround_run_count": 0,
            "exchange_subround_count": 0,
            "phases_s": {"mpf_shuffle": 0.0},
        },
        _device_tables={0: [object()]},
        _prepared_wave_id=1,
        _source_input_complete=True,
        _prepared_batches={},
    )
    prepared = {
        batch_id: _PreparedBatch(chunks={0: "owned"}, destination_bytes=(1,))
        for batch_id in (3, 4)
    }
    backend._prepared_batches.update(prepared)

    def externalize(partition, wave):
        events.append("externalize")
        backend._device_tables[partition] = []
        backend._stats["initial_run_count"] += 1

    def create_shuffler(*args, **kwargs):
        events.append("create")
        return Shuffler()

    backend._externalize_device_tables = externalize
    backend.create_shuffler = create_shuffler
    backend.rank = lambda: 0
    backend.nranks = lambda: 1
    backend._unpack_one = lambda chunk: Table()

    def accept(partition, table, wave):
        events.append("accept")
        backend._device_tables[partition].append(table)

    backend._accept_received = accept
    backend._update_peak = lambda: None

    backend._externalize_before_next_wave(0)
    assert events == []
    backend._externalize_before_next_wave(1)
    first = backend.exchange_prepared_round(
        wave_id=1,
        exchange_id=7,
        batch_ids=[3],
        final_subround=False,
    )
    assert first["remaining_batches"] == 1
    assert prepared[3].chunks == {}
    assert prepared[4].chunks == {0: "owned"}
    assert list(backend._prepared_batches) == [4]
    result = backend.exchange_prepared_round(
        wave_id=1,
        exchange_id=8,
        batch_ids=[4],
        final_subround=True,
    )

    assert result["received_rows"] == 1
    assert prepared[4].chunks == {}
    assert backend._prepared_batches == {}
    assert backend._prepared_wave_id is None
    assert events == [
        "externalize",
        "create",
        "insert",
        "shutdown",
        "accept",
        "externalize",
        "create",
        "insert",
        "shutdown",
        "accept",
    ], events
    assert backend._stats["forced_subround_run_count"] == 1
    assert backend._stats["exchange_subround_count"] == 2


def test_gpu_sort_externalization_reserves_live_allocator_workspace(
    externalization_backend,
):
    class MemoryResource:
        def __init__(self):
            self.snapshots = iter([50, 50, 70])
            self.last = 70

        @property
        def current_allocated(self):
            self.last = next(self.snapshots, self.last)
            return self.last

    assert _workspace_bounded_payload_bytes(100, 50, 3.0, 6) == 22

    backend = externalization_backend([_DeviceTable(40, 40)], memory=MemoryResource())
    sorted_sizes = []

    def sort_table(table):
        sorted_sizes.append(table.size_bytes)
        return table

    backend._sort_table = sort_table
    backend._externalize_device_tables(0, wave_id=3)

    # The first 25-byte estimate is reduced to 15 after the simulated late
    # allocation. Every row still makes progress through bounded retries.
    assert sorted_sizes == [15, 15, 10]
    assert len(backend._runs[0]) == 3
    assert backend._stats["state"] == "EXTERNAL_RUNS"
    assert backend._stats["first_externalize_wave"] == 3


def test_gpu_sort_externalization_bounds_and_recovers_multitable_concat(
    externalization_backend, monkeypatch
):
    class MemoryResource:
        def __init__(self, current_allocated):
            self.current_allocated = current_allocated
            self.peak = current_allocated

        def allocate(self, size_bytes):
            self.current_allocated += size_bytes
            self.peak = max(self.peak, self.current_allocated)

        def free(self, size_bytes):
            self.current_allocated -= size_bytes

    memory = MemoryResource(current_allocated=10)

    class Table:
        def __init__(self, rows, size_bytes, *, owns_allocation=False):
            self.rows = rows
            self.size_bytes = size_bytes
            self.owns_allocation = owns_allocation
            if owns_allocation:
                memory.allocate(size_bytes)

        def num_rows(self):
            return self.rows

        def __del__(self):
            if self.owns_allocation:
                self.owns_allocation = False
                memory.free(self.size_bytes)

    concat_sizes = []
    concat_ooms = 0

    def bounded_concat(tables):
        nonlocal concat_ooms
        size_bytes = sum(table.size_bytes for table in tables)
        if memory.current_allocated + size_bytes > 100:
            raise AssertionError("concatenate exceeded the RMM pool")
        if concat_ooms == 0:
            concat_ooms += 1
            raise MemoryError("simulated concurrent MPF allocation")
        concat_sizes.append(size_bytes)
        return Table(
            sum(table.rows for table in tables),
            size_bytes,
            owns_allocation=True,
        )

    _stub_module(
        monkeypatch,
        "pylibcudf",
        concatenate=types.SimpleNamespace(concatenate=bounded_concat),
    )
    backend = externalization_backend(
        [Table(10, 10, owns_allocation=True) for _ in range(5)],
        memory=memory,
        payload_limit_bytes=50,
        slice_table=lambda table, start, end: (
            table
            if start == 0 and end == table.rows
            else Table(
                end - start,
                table.size_bytes * (end - start) // table.rows,
            )
        ),
    )
    sorted_sizes = []

    def sort_table(table):
        assert (
            memory.current_allocated + 2 * table.size_bytes <= backend._pool_max_bytes
        )
        sorted_sizes.append(table.size_bytes)
        return table

    backend._sort_table = sort_table
    backend._externalize_device_tables(0, wave_id=4)

    # The failed first concatenate loses no input. One table is sorted alone,
    # then the remaining sources continue through bounded groups.
    assert concat_sizes == [20, 20]
    assert sorted_sizes == [10, 20, 20]
    assert sum(sorted_sizes) == 50
    assert len(backend._runs[0]) == 3
    assert backend._stats["concat_oom_fallback_count"] == 1
    assert backend._device_tables[0] == []
    assert memory.current_allocated == 10
    assert memory.peak == 70


def test_gpu_sort_externalization_retries_smaller_run_after_sort_oom(
    externalization_backend,
):
    backend = externalization_backend(
        [_DeviceTable(40, 40)],
        memory=types.SimpleNamespace(current_allocated=50),
    )
    attempts = []

    def sort_table(table):
        attempts.append(table.size_bytes)
        if table.size_bytes > 12:
            raise MemoryError("simulated sort workspace race")
        return table

    backend._sort_table = sort_table
    backend._externalize_device_tables(0, wave_id=5)

    assert attempts == [25, 12, 25, 12, 16, 8, 8]
    assert sum(size for size in attempts if size <= 12) == 40
    assert len(backend._runs[0]) == 4
    assert backend._stats["run_sort_oom_retry_count"] == 3
    assert backend._device_tables[0] == []


def test_gpu_sort_resident_finalization_releases_sources_or_externalizes(
    gpu_backend_class, monkeypatch
):
    _stub_synchronous_cupy(monkeypatch)
    sources = [_DeviceTable(20, 20), _DeviceTable(20, 20)]
    _stub_module(
        monkeypatch,
        "pylibcudf",
        concatenate=types.SimpleNamespace(
            concatenate=lambda tables: _DeviceTable(
                sum(table.rows for table in tables),
                sum(table.size_bytes for table in tables),
            )
        ),
    )

    def make(tables, *, current_allocated, pool_max=200):
        backend = _new_backend(
            gpu_backend_class,
            _config=GPUSortConfig(final_sort_workspace_factor=3.0),
            _pool_max_bytes=pool_max,
            _mr=types.SimpleNamespace(current_allocated=current_allocated),
            _device_tables={0: tables},
            _device_bytes={0: sum(table.size_bytes for table in tables)},
            _runs={0: []},
            _stats={
                "resident_final_fallback_count": 0,
                "run_sort_oom_retry_count": 0,
                "phases_s": {"run_sort": 0.0},
            },
        )
        backend._table_bytes = lambda table: table.size_bytes
        externalized = []

        def externalize(partition, wave_id):
            externalized.append((list(backend._device_tables[partition]), wave_id))
            backend._device_tables[partition] = []
            backend._runs[partition].append(_ExternalRun())

        backend._externalize_device_tables = externalize
        return backend, externalized

    resident, resident_externalized = make(sources, current_allocated=40)

    def sort_after_release(table):
        assert sources == []
        return table

    resident._sort_table = sort_after_release
    assert resident._finish_resident_partition(0).size_bytes == 40
    assert resident_externalized == []

    unsafe_table = _DeviceTable(40, 40)
    unsafe, unsafe_externalized = make(
        [unsafe_table], current_allocated=80, pool_max=100
    )
    unsafe._sort_table = lambda table: pytest.fail("unsafe resident sort was entered")
    assert unsafe._finish_resident_partition(0) is None
    assert unsafe_externalized == [([unsafe_table], -1)]
    assert unsafe._stats["resident_final_fallback_count"] == 1

    oom_table = _DeviceTable(40, 40)
    oom, oom_externalized = make([oom_table], current_allocated=40)
    oom._sort_table = lambda table: (_ for _ in ()).throw(MemoryError())
    assert oom._finish_resident_partition(0) is None
    assert oom_externalized == [([oom_table], -1)]
    assert oom._stats["run_sort_oom_retry_count"] == 1


def test_gpu_sort_leaves_last_merge_group_for_direct_streaming(gpu_backend_class):
    backend = object.__new__(gpu_backend_class)
    backend._config = GPUSortConfig(merge_fan_in=4)
    backend._stats = {"merge_pass_count": 0}
    group_sizes = []

    def merge_group(group):
        group_sizes.append(len(group))
        return _ExternalRun()

    backend._merge_group = merge_group
    final_group = backend._merge_runs_until_final([_ExternalRun() for _ in range(17)])

    assert len(final_group) == 2
    assert backend._stats["merge_pass_count"] == 2
    assert group_sizes == [4, 4, 4, 4, 4]


def test_gpu_sort_restores_merge_heads_in_one_ray_get(gpu_backend_class, monkeypatch):
    refs = [object(), object(), object()]
    arrows = [object(), object(), object()]
    chunks = [
        _RunChunk(ref=ref, rows=index + 1, size_bytes=(index + 1) * 10)
        for index, ref in enumerate(refs)
    ]
    get_calls = []

    def get(requested):
        get_calls.append(requested)
        return arrows

    monkeypatch.setattr(ray, "get", get)
    ticks = iter([10.0, 11.25])
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))
    _stub_module(
        monkeypatch,
        "cudf",
        DataFrame=types.SimpleNamespace(
            from_arrow=lambda arrow: types.SimpleNamespace(table=("device", arrow))
        ),
    )
    _stub_module(
        monkeypatch,
        "rapidsmpf.utils.cudf",
        cudf_to_pylibcudf_table=lambda frame: frame.table,
    )

    backend = _new_backend(
        gpu_backend_class,
        _stats={
            "plasma_read_bytes": 0,
            "h2d_bytes": 0,
            "phases_s": {"orchestration": 0.0},
        },
    )

    tables = backend._load_run_chunks(chunks)

    assert get_calls == [refs]
    assert tables == [("device", arrow) for arrow in arrows]
    assert backend._stats["plasma_read_bytes"] == 60
    assert backend._stats["h2d_bytes"] == 60
    assert backend._stats["phases_s"]["orchestration"] == pytest.approx(1.25)


def test_gpu_sort_local_mmap_stays_open_until_h2d_is_synchronized(
    gpu_backend_class, monkeypatch
):
    events = []
    chunk = _RunChunk(ref="/raid/run.arrow", rows=1, size_bytes=8)
    arrow = object()

    class ReadContext:
        def __enter__(self):
            events.append("mapping-open")
            return [arrow]

        def __exit__(self, *_args):
            events.append("mapping-close")

    class Store:
        mode = "local_disk"

        def read_many(self, chunks):
            assert chunks == [chunk]
            return ReadContext()

        def diagnostics(self):
            return {}

    _stub_module(
        monkeypatch,
        "cudf",
        DataFrame=types.SimpleNamespace(
            from_arrow=lambda value: events.append("h2d")
            or types.SimpleNamespace(table=("device", value))
        ),
    )
    _stub_module(
        monkeypatch,
        "rapidsmpf.utils.cudf",
        cudf_to_pylibcudf_table=lambda frame: frame.table,
    )
    _stub_module(
        monkeypatch,
        "cupy",
        cuda=types.SimpleNamespace(
            runtime=types.SimpleNamespace(
                deviceSynchronize=lambda: events.append("synchronize")
            )
        ),
    )
    backend = _new_backend(
        gpu_backend_class,
        _run_store=Store(),
        _stats={
            "h2d_bytes": 0,
            "plasma_output_write_bytes": 0,
            "phases_s": {"orchestration": 0.0},
        },
    )
    ticks = iter([10.0, 11.5])
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))

    assert backend._load_run_chunks([chunk]) == [("device", arrow)]
    assert events == ["mapping-open", "h2d", "synchronize", "mapping-close"]
    assert backend._stats["local_run_restore_s"] == pytest.approx(1.5)


def test_gpu_sort_partial_run_write_releases_committed_chunks(gpu_backend_class):
    committed = _RunChunk(ref="new-0", rows=1, size_bytes=8)

    class Store:
        mode = "local_disk"

        def __init__(self):
            self.calls = 0
            self.released = []

        def write(self, _table):
            self.calls += 1
            if self.calls == 2:
                raise OSError("synthetic run write failure")
            return committed

        def release(self, chunks):
            self.released.extend(chunks)

        def diagnostics(self):
            return {}

    store = Store()
    table = _DeviceTable(2, 16)
    backend = _new_backend(
        gpu_backend_class,
        _run_store=store,
        _stats={
            "initial_run_count": 0,
            "externalized_rows": 0,
            "externalized_bytes": 0,
            "plasma_output_write_bytes": 0,
            "phases_s": {"plasma_seal": 0.0},
        },
    )
    backend._iter_table_as_work_arrow = lambda _table: iter(
        [pa.table({"value": [1]}), pa.table({"value": [2]})]
    )

    with pytest.raises(OSError, match="synthetic run write failure"):
        backend._store_table_as_run(table, initial=True)

    assert store.released == [committed]
    assert backend._stats["initial_run_count"] == 0


def test_gpu_sort_merge_commit_transfers_suffix_and_releases_consumed_chunks(
    gpu_backend_class,
):
    first = _RunChunk(ref="first", rows=1, size_bytes=1)
    second = _RunChunk(ref="second", rows=1, size_bytes=1)
    suffix = _RunChunk(ref="suffix", rows=1, size_bytes=1)
    replacement = _RunChunk(ref="replacement", rows=2, size_bytes=2)

    class Store:
        mode = "local_disk"

        def __init__(self):
            self.released = []

        def release(self, chunks):
            self.released.extend(chunks)

        def diagnostics(self):
            return {}

    store = Store()
    group = [_ExternalRun([first, suffix]), _ExternalRun([second])]
    backend = _new_backend(
        gpu_backend_class,
        _run_store=store,
        _stats={"replacement_run_count": 0, "plasma_output_write_bytes": 0},
    )
    device_output = object()
    backend._iter_merged_sources = lambda _group: iter([device_output, suffix])
    backend._store_table_as_run = lambda table, initial: _ExternalRun([replacement])

    output = backend._merge_group(group)

    assert output.chunks == [replacement, suffix]
    assert store.released == [first, second]
    assert all(run.chunks == [] for run in group)
    assert backend._stats["replacement_run_count"] == 1


def test_gpu_sort_merge_failure_rolls_back_outputs_and_preserves_sources(
    gpu_backend_class,
):
    sources = [
        _RunChunk(ref="source-0", rows=1, size_bytes=1),
        _RunChunk(ref="source-1", rows=1, size_bytes=1),
    ]
    replacement = _RunChunk(ref="replacement", rows=1, size_bytes=1)

    class Store:
        mode = "local_disk"

        def __init__(self):
            self.released = []

        def release(self, chunks):
            self.released.extend(chunks)

        def diagnostics(self):
            return {}

    store = Store()
    group = [_ExternalRun(list(sources))]
    backend = _new_backend(
        gpu_backend_class,
        _run_store=store,
        _stats={"replacement_run_count": 0, "plasma_output_write_bytes": 0},
    )

    def fail_after_one_output(_group):
        yield object()
        raise RuntimeError("synthetic merge failure")

    backend._iter_merged_sources = fail_after_one_output
    backend._store_table_as_run = lambda table, initial: _ExternalRun([replacement])

    with pytest.raises(RuntimeError, match="synthetic merge failure"):
        backend._merge_group(group)

    assert store.released == [replacement]
    assert group[0].chunks == sources
    assert backend._stats["replacement_run_count"] == 0


def test_gpu_sort_merge_batches_only_missing_run_heads(gpu_backend_class, monkeypatch):
    class Table:
        def __init__(self, value):
            self.value = value

        def num_rows(self):
            return 1

    _stub_synchronous_cupy(monkeypatch)
    _stub_module(
        monkeypatch,
        "pylibcudf",
        merge=types.SimpleNamespace(
            merge=lambda tables, *_args: Table(tuple(table.value for table in tables))
        ),
    )

    runs = [
        _ExternalRun(
            [
                _RunChunk(ref=f"ref-{rank}-{round_}", rows=1, size_bytes=10)
                for round_ in range(2)
            ]
        )
        for rank in range(4)
    ]
    backend = _new_backend(
        gpu_backend_class,
        _stats={"phases_s": {"gpu_merge": 0.0}},
        _merge_key_indices=[],
    )
    load_calls = []

    def load_run_chunks(requested):
        load_calls.append(requested)
        return [Table(chunk.ref) for chunk in requested]

    backend._load_run_chunks = load_run_chunks
    backend._watermark = lambda _tables: object()
    backend._upper_bound = lambda table, _watermark: table.num_rows()
    backend._slice_table = lambda table, _start, _end: table
    backend._order_and_nulls = lambda: ([], [])
    backend._update_peak = lambda: None

    merged = list(backend._iter_merged_sources(runs))

    expected_rounds = [[run.chunks[round_] for run in runs] for round_ in range(2)]
    assert load_calls == expected_rounds
    assert [table.value for table in merged] == [
        tuple(chunk.ref for chunk in round_chunks) for round_chunks in expected_rounds
    ]


def test_gpu_sort_external_merge_rejects_no_progress(gpu_backend_class, monkeypatch):
    class Table:
        def num_rows(self):
            return 1

    _stub_synchronous_cupy(monkeypatch)
    _stub_module(monkeypatch, "pylibcudf")

    runs = [
        _ExternalRun([_RunChunk(ref=rank, rows=1, size_bytes=1)]) for rank in range(2)
    ]
    backend = _new_backend(
        gpu_backend_class,
        _stats={"phases_s": {"gpu_merge": 0.0}},
        _load_run_chunks=lambda chunks: [Table() for _ in chunks],
        _watermark=lambda tables: object(),
        _upper_bound=lambda table, watermark: 0,
        _slice_table=lambda table, start, end: table,
    )

    with pytest.raises(RuntimeError, match="made no forward progress"):
        list(backend._iter_merged_sources(runs))


def test_gpu_sort_final_merge_streams_without_replacement_run(gpu_backend_class):
    backend = object.__new__(gpu_backend_class)
    backend._stats = {
        "merge_pass_count": 0,
        "direct_final_merge_count": 0,
    }
    sources = [object(), object()]
    backend._iter_merged_sources = lambda group: iter(sources)
    backend._iter_table_as_work_arrow = lambda table: iter([(table, "arrow")])
    backend._store_table_as_run = lambda *_args, **_kwargs: pytest.fail(
        "The final merge must not materialize a replacement run"
    )

    output = list(backend._iter_final_work_arrow([_ExternalRun(), _ExternalRun()]))

    assert output == [(sources[0], "arrow"), (sources[1], "arrow")]
    assert backend._stats["merge_pass_count"] == 1
    assert backend._stats["direct_final_merge_count"] == 1


def test_gpu_sort_streaming_output_folds_tiny_tail(gpu_backend_class):
    backend = object.__new__(gpu_backend_class)
    backend._output_arrow = lambda table: table
    tables = [
        pa.table({"value": np.arange(10, dtype=np.int64)}),
        pa.table({"value": np.arange(10, 20, dtype=np.int64)}),
        pa.table({"value": np.array([20], dtype=np.int64)}),
    ]

    blocks = list(backend._iter_coalesced_output_blocks(iter(tables), 80))

    assert [block.num_rows for block in blocks] == [10, 11]
    assert pa.concat_tables(blocks).equals(pa.concat_tables(tables))


def test_gpu_sort_final_sources_live_until_output_is_sealed(
    gpu_backend_class, monkeypatch
):
    _stub_module(monkeypatch, "cupy")
    _stub_module(monkeypatch, "pylibcudf")
    source_chunk = _RunChunk(ref="source", rows=10, size_bytes=80)
    source = _ExternalRun([source_chunk])
    table = pa.table({"value": np.arange(10, dtype=np.int64)})

    class Store:
        mode = "local_disk"

        def __init__(self):
            self.released = []
            self.closed = False

        def release(self, chunks):
            self.released.extend(chunks)

        def diagnostics(self):
            return {}

        def close(self, *, strict=False):
            self.closed = True
            self.strict = strict

    store = Store()
    backend = _new_backend(
        gpu_backend_class,
        _run_store=store,
        _device_tables={0: []},
        _runs={0: [source]},
        _run_chunk_bytes=1,
        _stats={
            "output_bytes": 0,
            "plasma_write_bytes": 0,
            "phases_s": {"plasma_seal": 0.0},
        },
    )
    backend._merge_runs_until_final = lambda runs: runs
    backend._output_arrow = lambda work_arrow: work_arrow
    backend._tagged_metadata = lambda block, partition, stats: (
        "metadata",
        partition,
        stats,
    )
    backend._update_peak = lambda: None

    def final_work_arrow(_group):
        assert source.chunks
        yield table
        assert source.chunks

    backend._iter_final_work_arrow = final_work_arrow
    stream = backend.finish_and_extract()

    assert next(stream).equals(table)
    assert source.chunks
    creation_stats = types.SimpleNamespace(object_creation_dur_s=1.25)
    assert stream.send(creation_stats) == ("metadata", 0, creation_stats)
    assert source.chunks
    assert store.released == []
    with pytest.raises(StopIteration):
        next(stream)
    assert source.chunks == []
    assert store.released == [source_chunk]
    assert store.closed
    assert store.strict
    assert backend._stats["phases_s"]["plasma_seal"] == 1.25
    assert backend._stats["plasma_output_write_calls"] == 1
    assert backend._stats["plasma_output_write_s"] == 1.25
