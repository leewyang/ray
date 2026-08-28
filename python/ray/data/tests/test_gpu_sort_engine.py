"""CPU-only tests for GPU-sort planning, spilling, and ownership invariants."""

import math
import pickle
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import ray.data._internal.gpu_sort.operator as gpu_sort_operator
from ray.data.dataset import Dataset
from ray.data._internal.execution.operators.base_physical_operator import (
    AllToAllOperator,
)
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.gpu_sort.backend import (
    _ExternalRun,
    _coalesce_arrow_inputs,
    _cpu_sample_boundaries,
    _input_batch_target_bytes,
    _sampled_arrow_row_weights,
    _scale_sample_weights_by_stratum,
    _stratified_sample_indices,
    _workspace_bounded_payload_bytes,
    lazy_load_backend,
)
from ray.data._internal.gpu_sort.config import GPUSortCapacityError, GPUSortConfig
from ray.data._internal.gpu_sort.operator import (
    GPUSortOperator,
    _InputBlock,
    _allocate_stratified_sample_quotas,
    _assign_blocks_by_locality,
    _make_waves,
    _validate_gpu_schema,
    _wave_target_bytes,
)
from ray.data._internal.logical.interfaces import LogicalOperator, LogicalPlan
from ray.data._internal.logical.operators import InputData, Sort
from ray.data._internal.logical.rules.combine_shuffles import CombineShuffles
from ray.data._internal.planner.exchange.sort_task_spec import SortKey
from ray.data._internal.planner.plan_all_to_all_op import plan_all_to_all_op
from ray.data.context import DataContext
from ray.data.tests.conftest import *  # noqa
from ray.tests.conftest import *  # noqa


@pytest.fixture
def gpu_backend_class(monkeypatch):
    """Load the backend methods without importing optional RAPIDS packages."""

    ray_utils = types.ModuleType("rapidsmpf.utils.ray_utils")
    ray_utils.BaseShufflingActor = type("BaseShufflingActor", (), {})
    monkeypatch.setitem(sys.modules, "rapidsmpf.utils.ray_utils", ray_utils)
    return lazy_load_backend()


def test_gpu_sort_api_and_logical_backend(monkeypatch):
    context = DataContext()
    ds = object.__new__(Dataset)
    ds._logical_plan = LogicalPlan(InputData(input_data=[]), context)
    ds._context = context

    def from_parent(_cls, _parent, logical_plan):
        result = object.__new__(Dataset)
        result._logical_plan = logical_plan
        result._context = context
        return result

    monkeypatch.setattr(Dataset, "_from_parent", classmethod(from_parent))

    assert ds.sort("k")._logical_plan.dag.backend == "cpu"
    gpu_ds = ds.sort("k", backend="gpu")
    gpu_sort = gpu_ds._logical_plan.dag
    assert gpu_sort.backend == "gpu"
    assert gpu_ds._logical_plan.require_preserve_order()
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


def test_gpu_sort_planner_selects_requested_backend(monkeypatch):
    context = DataContext()
    logical_input = MagicMock(LogicalOperator)
    physical_input = InputDataBuffer(context, input_data=[])

    def plan(backend):
        logical_sort = Sort(
            sort_key=SortKey("k"),
            backend=backend,
            input_dependencies=[logical_input],
        )
        return plan_all_to_all_op(logical_sort, [physical_input], context)

    assert isinstance(plan("cpu"), AllToAllOperator)

    monkeypatch.setattr(gpu_sort_operator, "_derive_num_ranks", lambda _: 1)
    assert isinstance(plan("gpu"), GPUSortOperator)


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
        _validate_gpu_schema(
            pa.schema([pa.field("decimal", pa.decimal128(12, 2))]), ["decimal"]
        )
    with pytest.raises(NotImplementedError, match="unsupported columns"):
        _validate_gpu_schema(
            pa.schema(
                [
                    pa.field("key", pa.int64()),
                    pa.field("unsupported", pa.null()),
                ]
            ),
            ["key"],
        )

    assert GPUSortConfig(residency_budget_bytes="4 GiB").residency_budget_bytes == (
        4 << 30
    )
    blocks = [
        [_InputBlock("a", 6, 1), _InputBlock("b", 6, 1)],
        [_InputBlock("c", 3, 1)],
    ]
    waves = _make_waves(blocks, target_bytes_per_rank=8)
    assert waves == [[["a"], ["c"]], [["b"], []]]
    assert _make_waves(blocks, target_bytes_per_rank=None) == [[["a", "b"], ["c"]]]


def test_gpu_sort_bounded_input_batching_preserves_order_and_chunks():
    tables = [
        pa.table({"value": [0, 1]}),
        pa.table({"value": [2, 3, 4]}),
        pa.table({"value": pa.array([], type=pa.int64())}),
        pa.table({"value": [5, 6, 7, 8, 9]}),
        pa.table({"value": list(range(10, 20))}),
    ]
    batches = list(_coalesce_arrow_inputs(tables, target_bytes=64))

    assert [batch.nbytes for batch in batches] == [40, 40, 64, 16]
    assert [batch["value"].to_pylist() for batch in batches] == [
        list(range(5)),
        list(range(5, 10)),
        list(range(10, 18)),
        list(range(18, 20)),
    ]
    assert batches[0]["value"].num_chunks == 2

    variable = pa.table({"value": ["a" * 5, "b" * 7, "c" * 9, "d" * 4]})
    variable_batches = list(_coalesce_arrow_inputs([variable], target_bytes=16))
    assert all(batch.nbytes <= 16 for batch in variable_batches)
    assert pa.concat_tables(variable_batches).equals(variable)

    with pytest.raises(GPUSortCapacityError, match="One Arrow row"):
        list(_coalesce_arrow_inputs([pa.table({"value": ["x" * 32]})], target_bytes=16))
    gib = 1 << 30
    assert _input_batch_target_bytes(1 * gib, 8 * gib, 2 * gib, 3.0) == gib
    assert _input_batch_target_bytes(1 * gib, 8 * gib, 6 * gib, 3.0) == (2 * gib // 3)
    assert _input_batch_target_bytes(1 * gib, 8 * gib, 8 * gib, 3.0) == 1


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
        == 8 * gib
    )
    assert (
        _wave_target_bytes(
            large,
            explicit_residency_budget_bytes=None,
            actor_usable_budgets=[20 * gib, 16 * gib],
            auto_wave_fraction=0.375,
        )
        == 6 * gib
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


def test_gpu_sort_actor_config_rejects_unknown_fields():
    config = GPUSortConfig(sample_size=32)
    assert GPUSortConfig.from_actor_dict(config.to_actor_dict()) == config

    with pytest.raises(ValueError, match="Unknown GPU sort configuration fields"):
        GPUSortConfig.from_actor_dict({"sample_size": 32, "sample_sze": 64})


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
    assert indices.tolist() == [0, 5, 9]
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


def test_gpu_sort_stratified_sampling_is_locality_invariant_and_breaks_periodicity():
    blocks = [
        _InputBlock("a", 40, 40, ordinal=0),
        _InputBlock("b", 80, 80, ordinal=1),
        _InputBlock("c", 120, 120, ordinal=2),
    ]
    quotas, _ = _allocate_stratified_sample_quotas(blocks, 48)
    remapped = [blocks[2], blocks[0], blocks[1]]
    remapped_quotas = [quotas[2], quotas[0], quotas[1]]

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
    assert original_result["boundaries"].equals(remapped_result["boundaries"])

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
        {"category": "a", "fixed": 3, "score": 2.0},
        {"category": "a", "fixed": 2, "score": None},
        {"category": "b", "fixed": 1, "score": 5.0},
    ]
    assert boundaries[4]["category"] == "b"
    assert boundaries[4]["fixed"] == 1
    assert math.isnan(boundaries[4]["score"])
    assert boundaries[5] == {"category": None, "fixed": 9, "score": 0.0}
    assert result["sample_rows"] == 7


def test_gpu_sort_comparator_matches_arrow_null_nan_order(
    gpu_backend_class, monkeypatch
):
    order = types.SimpleNamespace(ASCENDING="asc", DESCENDING="desc")
    null_order = types.SimpleNamespace(BEFORE="before", AFTER="after")
    plc = types.ModuleType("pylibcudf")
    plc.types = types.SimpleNamespace(Order=order, NullOrder=null_order)
    monkeypatch.setitem(sys.modules, "pylibcudf", plc)

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

        backend = object.__new__(gpu_backend_class)
        backend._key_columns = ["key"]
        backend._ascending = [ascending]
        backend._float_hidden = {"key": ("is_null", "is_nan")}
        backend._config = GPUSortConfig(null_position="last")
        orders, nulls = backend._order_and_nulls()
        assert orders == [
            order.ASCENDING,
            order.ASCENDING,
            order.ASCENDING if ascending else order.DESCENDING,
        ]
        assert nulls == [
            null_order.AFTER,
            null_order.AFTER,
            null_order.AFTER,
        ]

        backend._float_hidden = {}
        for null_position, expected_null_order in (
            ("first", null_order.BEFORE),
            ("last", null_order.AFTER),
        ):
            backend._config = GPUSortConfig(null_position=null_position)
            direct_orders, direct_nulls = backend._order_and_nulls()
            assert direct_orders == [order.ASCENDING if ascending else order.DESCENDING]
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

    cp = types.ModuleType("cupy")
    cp.ElementwiseKernel = elementwise_kernel
    cp.uint64 = int
    monkeypatch.setitem(sys.modules, "cupy", cp)

    cudf = types.ModuleType("cudf")
    cudf.Series = Series
    monkeypatch.setitem(sys.modules, "cudf", cudf)

    plc = types.ModuleType("pylibcudf")
    plc.search = types.SimpleNamespace(
        lower_bound=lambda *_: [1] * captured["rows"],
        upper_bound=lambda *_: [3] * captured["rows"],
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
    monkeypatch.setitem(sys.modules, "pylibcudf", plc)

    partition_module = types.ModuleType("rapidsmpf.integrations.cudf.partition")
    partition_module.split_and_pack = lambda *_: {
        destination: destination for destination in (1, 2, 3)
    }
    monkeypatch.setitem(
        sys.modules, "rapidsmpf.integrations.cudf.partition", partition_module
    )
    cudf_utils = types.ModuleType("rapidsmpf.utils.cudf")
    cudf_utils.cudf_to_pylibcudf_table = lambda frame: frame.table
    monkeypatch.setitem(sys.modules, "rapidsmpf.utils.cudf", cudf_utils)
    stream = types.ModuleType("rmm.pylibrmm.stream")
    stream.DEFAULT_STREAM = object()
    monkeypatch.setitem(sys.modules, "rmm.pylibrmm.stream", stream)

    backend = object.__new__(gpu_backend_class)
    backend._duplicate_kernel = None
    backend._row_ordinal = 0
    backend._boundary_keys = object()
    backend._num_partitions = 4
    backend._buffer_resource = object()
    backend._augment_table = lambda table, names: (table, names)
    backend._comparison_table = lambda table, names: table
    backend._order_and_nulls = lambda: ([], [])
    backend.rank = lambda: 2

    def distribute(rows):
        captured["rows"] = rows
        packed = backend._partition_and_pack(Frame(rows), wave_id=0)
        return list(captured["destinations"]), packed

    first, first_packed = distribute(8)
    second, second_packed = distribute(8)
    assert first == [3, 1, 2, 3, 1, 2, 3, 1]
    assert set(first + second) == {1, 2, 3}
    counts = [(first + second).count(partition) for partition in (1, 2, 3)]
    assert max(counts) - min(counts) <= 1
    assert first_packed == second_packed == {9: 1, 10: 2, 11: 3}


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
    backend._table_bytes = lambda table: table.size_bytes
    backend._slice_table = lambda table, start, end: Table(
        end - start, max(1, table.size_bytes * (end - start) // table.rows)
    )
    transitions = []

    def externalize(partition):
        transitions.append((partition, backend._device_bytes[partition]))
        backend._device_tables[partition] = []
        backend._device_bytes[partition] = 0

    backend._externalize_device_tables = externalize
    backend._accept_received(0, Table(6, 6))
    backend._accept_received(0, Table(6, 6))
    assert transitions == [(0, 6)]
    assert backend._device_bytes[0] == 6

    backend._accept_received(0, Table(4, 4))
    assert transitions[-1] == (0, 10)
    assert backend._device_bytes[0] == 0


def test_gpu_sort_externalization_reserves_live_allocator_workspace(
    gpu_backend_class, monkeypatch
):
    class Table:
        def __init__(self, rows, size_bytes):
            self.rows = rows
            self.size_bytes = size_bytes

        def num_rows(self):
            return self.rows

    cp = types.ModuleType("cupy")
    cp.cuda = types.SimpleNamespace(
        runtime=types.SimpleNamespace(deviceSynchronize=lambda: None)
    )
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setitem(sys.modules, "pylibcudf", types.ModuleType("pylibcudf"))

    assert _workspace_bounded_payload_bytes(100, 50, 3.0, 6) == 22

    backend = object.__new__(gpu_backend_class)
    backend._config = GPUSortConfig(
        exchange_batch_bytes=10,
        final_sort_workspace_factor=3.0,
    )
    backend._pool_max_bytes = 100
    backend._mr = types.SimpleNamespace(current_allocated=50)
    backend._payload_limit_bytes = 40
    backend._device_tables = {0: [Table(40, 40)]}
    backend._device_bytes = {0: 40}
    backend._runs = {0: []}
    backend._slice_table = lambda table, start, end: Table(
        end - start, table.size_bytes * (end - start) // table.rows
    )
    backend._table_bytes = lambda table: table.size_bytes
    sorted_sizes = []

    def sort_table(table):
        sorted_sizes.append(table.size_bytes)
        return table

    backend._sort_table = sort_table
    backend._store_table_as_run = lambda table: _ExternalRun()
    backend._externalize_device_tables(0)

    assert sorted_sizes == [25, 15]
    assert len(backend._runs[0]) == 2


def test_gpu_sort_externalization_bounds_multitable_concat(
    gpu_backend_class, monkeypatch
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

    cp = types.ModuleType("cupy")
    cp.cuda = types.SimpleNamespace(
        runtime=types.SimpleNamespace(deviceSynchronize=lambda: None)
    )
    monkeypatch.setitem(sys.modules, "cupy", cp)
    plc = types.ModuleType("pylibcudf")
    concat_sizes = []

    def bounded_concat(tables):
        size_bytes = sum(table.size_bytes for table in tables)
        if memory.current_allocated + size_bytes > 100:
            raise AssertionError("concatenate exceeded the RMM pool")
        concat_sizes.append(size_bytes)
        return Table(
            sum(table.rows for table in tables),
            size_bytes,
            owns_allocation=True,
        )

    plc.concatenate = types.SimpleNamespace(concatenate=bounded_concat)
    monkeypatch.setitem(sys.modules, "pylibcudf", plc)

    backend = object.__new__(gpu_backend_class)
    backend._config = GPUSortConfig(final_sort_workspace_factor=3.0)
    backend._pool_max_bytes = 100
    backend._mr = memory
    backend._payload_limit_bytes = 50
    backend._device_tables = {
        0: [Table(10, 10, owns_allocation=True) for _ in range(5)]
    }
    backend._device_bytes = {0: 50}
    backend._runs = {0: []}
    backend._slice_table = lambda table, start, end: (
        table
        if start == 0 and end == table.rows
        else Table(
            end - start,
            table.size_bytes * (end - start) // table.rows,
        )
    )
    backend._table_bytes = lambda table: table.size_bytes
    sorted_sizes = []

    def sort_table(table):
        assert (
            memory.current_allocated + 2 * table.size_bytes <= backend._pool_max_bytes
        )
        sorted_sizes.append(table.size_bytes)
        return table

    backend._sort_table = sort_table
    backend._store_table_as_run = lambda table: _ExternalRun()
    backend._externalize_device_tables(0)

    # Concatenating all five sources would require 60 + 50 > 100 bytes.
    # Bounded groups allocate and then release 20 and 30 source bytes.
    assert concat_sizes == [20, 30]
    assert sorted_sizes == [20, 30]
    assert len(backend._runs[0]) == 2
    assert backend._device_tables[0] == []
    assert memory.current_allocated == 10
    assert memory.peak == 80


def test_gpu_sort_leaves_last_merge_group_for_direct_streaming(gpu_backend_class):
    backend = object.__new__(gpu_backend_class)
    backend._config = GPUSortConfig(merge_fan_in=4)
    group_sizes = []

    def merge_group(group):
        group_sizes.append(len(group))
        return _ExternalRun()

    backend._merge_group = merge_group
    final_group = backend._merge_runs_until_final([_ExternalRun() for _ in range(17)])

    assert len(final_group) == 2
    assert group_sizes == [4, 4, 4, 4, 4]


def test_gpu_sort_final_merge_streams_without_replacement_run(gpu_backend_class):
    backend = object.__new__(gpu_backend_class)
    sources = [object(), object()]
    backend._iter_merged_sources = lambda group: iter(sources)
    backend._iter_table_as_work_arrow = lambda table: iter([(table, "arrow")])
    backend._store_table_as_run = lambda *_args, **_kwargs: pytest.fail(
        "The final merge must not materialize a replacement run"
    )

    output = list(backend._iter_final_work_arrow([_ExternalRun(), _ExternalRun()]))

    assert output == [(sources[0], "arrow"), (sources[1], "arrow")]


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
    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))
    monkeypatch.setitem(sys.modules, "pylibcudf", types.ModuleType("pylibcudf"))
    backend = object.__new__(gpu_backend_class)
    source = _ExternalRun([object()])
    table = pa.table({"value": np.arange(10, dtype=np.int64)})
    backend._device_tables = {0: []}
    backend._runs = {0: [source]}
    backend._run_chunk_bytes = 1
    backend._merge_runs_until_final = lambda runs: runs
    backend._output_arrow = lambda work_arrow: work_arrow
    backend._block_metadata = lambda block, stats: (
        "metadata",
        0,
        stats,
    )

    def final_work_arrow(_group):
        assert source.chunks
        yield table
        assert source.chunks

    backend._iter_final_work_arrow = final_work_arrow
    stream = backend.finish_and_extract()

    assert next(stream).equals(table)
    assert source.chunks
    creation_stats = types.SimpleNamespace(object_creation_dur_s=1.25)
    assert pickle.loads(stream.send(creation_stats)) == ("metadata", 0, 1.25)
    assert source.chunks
    with pytest.raises(StopIteration):
        next(stream)
    assert source.chunks == []
