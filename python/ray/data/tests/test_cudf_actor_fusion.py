"""CPU-only tests for opt-in actor-based cuDF ``map_batches`` fusion."""

from decimal import Decimal

import pytest

import ray
from ray.data._internal.compute import ActorPoolStrategy, TaskPoolStrategy
from ray.data._internal.execution.operators.actor_pool_map_operator import (
    ActorPoolMapOperator,
)
from ray.data._internal.execution.operators.map_transformer import BatchMapTransformFn
from ray.data._internal.execution.operators.task_pool_map_operator import (
    TaskPoolMapOperator,
)
from ray.data._internal.execution.operators.union_operator import UnionOperator
from ray.data._internal.logical.interfaces import LogicalPlan
from ray.data._internal.logical.operators import InputData, MapBatches, MapRows, Union
from ray.data._internal.logical.optimizers import LogicalOptimizer, get_execution_plan
from ray.data._internal.logical.rules import cudf_actor_fusion
from ray.data._internal.logical.rules.cudf_actor_fusion import (
    FuseCudfActorMapBatches,
    _CudfMapStage,
    _FusedCudfMapBatches,
)
from ray.data._internal.planner import create_planner
from ray.data._internal.stats import DatasetStats, _StatsManager
from ray.data.context import DEFAULT_ENABLE_CUDF_ACTOR_FUSION, DataContext
from ray.data.dataset import Dataset


class _FakeFrame:
    """Small stand-in for a cuDF DataFrame that never imports cuDF."""

    def __init__(self, rows):
        self.rows = tuple(rows)

    def __len__(self):
        return len(self.rows)


class _Udf1:
    def __call__(self, batch):
        return batch


class _Udf2:
    def __call__(self, batch):
        return batch


class _Udf3:
    def __call__(self, batch):
        return batch


class _Udf4:
    def __call__(self, batch):
        return batch


class _Udf5:
    def __call__(self, batch):
        return batch


class _AsyncUdf:
    async def __call__(self, batch):
        return batch


class _AsyncGeneratorUdf:
    async def __call__(self, batch):
        yield batch


class _GeneratorUdf:
    def __call__(self, batch):
        yield batch


class _StaticAsyncUdf:
    @staticmethod
    async def __call__(batch):
        return batch


class _ClassGeneratorUdf:
    @classmethod
    def __call__(cls, batch):
        yield batch


def _task_udf(batch):
    return batch


class _ReprBomb:
    def __repr__(self):
        raise AssertionError("planning must not repr UDF arguments")


class _TrackingIterable:
    def __init__(self, value):
        self.value = value
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        yield self.value


class _ConfiguredStage:
    def __init__(self, offset, *, scale):
        self._offset = offset
        self._scale = scale

    def __call__(self, batch, increment, *, bias):
        return _FakeFrame(
            (value + self._offset + increment) * self._scale + bias
            for value in batch.rows
        )


class _StatefulStage:
    def __init__(self, instances, delta):
        self.calls = 0
        self._delta = delta
        instances.append(self)

    def __call__(self, batch):
        self.calls += 1
        return _FakeFrame(value + self._delta for value in batch.rows)


class _KeepEvenRows:
    def __call__(self, batch):
        return _FakeFrame(value for value in batch.rows if value % 2 == 0)


class _ObserveAndDuplicateRows:
    def __init__(self, seen):
        self._seen = seen

    def __call__(self, batch):
        self._seen.append(batch.rows)
        return _FakeFrame(value for value in batch.rows for _ in range(2))


class _RecordIdentity:
    def __init__(self, calls, label):
        self._calls = calls
        self._label = label

    def __call__(self, batch):
        self._calls.append((self._label, batch))
        return batch


class _WrongOutput:
    def __call__(self, batch):
        return {"rows": batch.rows}


class _GeneratorOutput:
    def __call__(self, batch):
        return iter((batch,))


class _ListOutput:
    def __call__(self, batch):
        return [batch]


class _NoneOutput:
    def __call__(self, batch):
        return None


class _AwaitableValue:
    def __await__(self):
        yield


class _AwaitableOutput:
    def __call__(self, batch):
        return _AwaitableValue()


class _RaiseFromStage:
    def __call__(self, batch):
        raise ValueError("stage exploded")


class _RaiseDuringConstruction:
    def __init__(self):
        raise ValueError("constructor exploded")

    def __call__(self, batch):  # pragma: no cover - construction always fails.
        return batch


_DEFAULT = object()


@pytest.fixture
def fake_cudf(monkeypatch):
    monkeypatch.setattr(
        cudf_actor_fusion,
        "_is_cudf_dataframe",
        lambda value: isinstance(value, _FakeFrame),
    )


def _make_actor_map(
    input_op,
    fn=_Udf1,
    *,
    batch_size=8,
    batch_format="cudf",
    zero_copy_batch=True,
    compute=_DEFAULT,
    min_rows_per_bundled_input=_DEFAULT,
    can_modify_num_rows=False,
    fn_constructor_args=None,
    fn_constructor_kwargs=None,
    fn_args=None,
    fn_kwargs=None,
    ray_remote_args=_DEFAULT,
    ray_remote_args_fn=None,
    per_block_limit=None,
):
    if compute is _DEFAULT:
        compute = ActorPoolStrategy(size=2)
    if min_rows_per_bundled_input is _DEFAULT:
        min_rows_per_bundled_input = (
            batch_size if type(batch_size) is int and batch_size > 0 else None
        )
    if ray_remote_args is _DEFAULT:
        ray_remote_args = {"num_gpus": 1}
    return MapBatches(
        fn,
        input_dependencies=[input_op],
        batch_size=batch_size,
        batch_format=batch_format,
        zero_copy_batch=zero_copy_batch,
        min_rows_per_bundled_input=min_rows_per_bundled_input,
        can_modify_num_rows=can_modify_num_rows,
        fn_constructor_args=fn_constructor_args,
        fn_constructor_kwargs=fn_constructor_kwargs,
        fn_args=fn_args,
        fn_kwargs=fn_kwargs,
        compute=compute,
        ray_remote_args=ray_remote_args,
        ray_remote_args_fn=ray_remote_args_fn,
        per_block_limit=per_block_limit,
    )


def _make_actor_map_chain(*udf_classes, **kwargs):
    current = InputData([])
    operators = []
    for udf_class in udf_classes:
        current = _make_actor_map(current, udf_class, **kwargs)
        operators.append(current)
    return current, operators


def _fusion_context(*, enabled):
    context = DataContext.get_current().copy()
    context.enable_cudf_actor_fusion = enabled
    return context


def _create_unoptimized_physical_plan(root, *, enabled=True, context=None):
    if context is None:
        context = _fusion_context(enabled=enabled)
    plan, _ = create_planner().plan(LogicalPlan(root, context))
    return plan


def _apply_fusion_rule(root, *, enabled=True, context=None):
    return FuseCudfActorMapBatches().apply(
        _create_unoptimized_physical_plan(root, enabled=enabled, context=context)
    )


def _all_physical_operators(root):
    return list(root.post_order_iter())


def _actor_map_operators(plan):
    return [
        op
        for op in _all_physical_operators(plan.dag)
        if isinstance(op, ActorPoolMapOperator)
    ]


def _task_map_operators(plan):
    return [
        op
        for op in _all_physical_operators(plan.dag)
        if isinstance(op, TaskPoolMapOperator)
    ]


def _fused_logical_map(plan, physical_op=None):
    if physical_op is None:
        physical_op = plan.dag
    logical_op = plan.op_map[physical_op]
    assert isinstance(logical_op, MapBatches)
    assert logical_op.fn is _FusedCudfMapBatches
    return logical_op


def _fused_stages(plan, physical_op=None):
    logical_op = _fused_logical_map(plan, physical_op)
    assert logical_op.fn_constructor_args is not None
    return logical_op.fn_constructor_args[0]


def _empty_dataset_with_current_context(monkeypatch):
    monkeypatch.setattr(
        _StatsManager,
        "gen_dataset_id_from_stats_actor",
        staticmethod(lambda: "cudf-fusion-cpu-test"),
    )
    context = DataContext.get_current().copy()
    return Dataset(
        LogicalPlan(InputData([]), context),
        context,
        DatasetStats(metadata={}, parent=None),
    )


# Physical planning


def test_default_disabled_and_logical_plan_is_never_rewritten():
    assert DEFAULT_ENABLE_CUDF_ACTOR_FUSION is False
    assert DataContext().enable_cudf_actor_fusion is False
    root, operators = _make_actor_map_chain(_Udf1, _Udf2)

    optimized = LogicalOptimizer().optimize(
        LogicalPlan(root, _fusion_context(enabled=True))
    )
    assert optimized.dag is root
    assert root.fn is _Udf2
    assert root.input_dependencies[0] is operators[0]

    disabled = _create_unoptimized_physical_plan(root, enabled=False)
    assert FuseCudfActorMapBatches().apply(disabled) is disabled
    assert len(_actor_map_operators(disabled)) == 2

    logical_plan = LogicalPlan(root, _fusion_context(enabled=True))
    physical, _ = get_execution_plan(logical_plan)
    assert logical_plan.dag is root
    assert root.input_dependencies[0] is operators[0]
    assert len(_actor_map_operators(physical)) == 1


@pytest.mark.parametrize(
    "udf_classes",
    [(_Udf1, _Udf2), (_Udf1, _Udf2, _Udf3)],
    ids=["two-udfs", "three-udfs"],
)
def test_public_map_batches_api_fuses_at_physical_planning(monkeypatch, udf_classes):
    context = DataContext.get_current()
    previous = context.enable_cudf_actor_fusion
    context.enable_cudf_actor_fusion = True
    try:
        dataset = _empty_dataset_with_current_context(monkeypatch)
    finally:
        context.enable_cudf_actor_fusion = previous

    options = {
        "batch_format": "cudf",
        "batch_size": 250_000,
        "compute": ray.data.ActorPoolStrategy(size=4),
        "num_gpus": 1,
    }
    for udf_class in udf_classes:
        dataset = dataset.map_batches(udf_class, **options)

    logical_root = dataset._logical_plan.dag
    physical, _ = get_execution_plan(dataset._logical_plan)

    assert logical_root.fn is udf_classes[-1]
    assert dataset._logical_plan.dag is logical_root
    assert len(_actor_map_operators(physical)) == 1
    assert [op.fn for op in physical.dag._logical_operators] == list(udf_classes)


@pytest.mark.parametrize(
    "actor_pool_kwargs",
    [None, {"min_size": 1, "max_size": 4, "initial_size": 2}],
    ids=["default-unbounded", "bounded"],
)
def test_matching_autoscaling_actor_pools_fuse(monkeypatch, actor_pool_kwargs):
    context = DataContext.get_current()
    previous = context.enable_cudf_actor_fusion
    context.enable_cudf_actor_fusion = True
    try:
        dataset = _empty_dataset_with_current_context(monkeypatch)
    finally:
        context.enable_cudf_actor_fusion = previous

    for udf_class in (_Udf1, _Udf2):
        options = {"batch_format": "cudf", "batch_size": 8, "num_gpus": 1}
        if actor_pool_kwargs is not None:
            options["compute"] = ActorPoolStrategy(**actor_pool_kwargs)
        dataset = dataset.map_batches(udf_class, **options)

    physical, _ = get_execution_plan(dataset._logical_plan)

    assert len(_actor_map_operators(physical)) == 1
    assert [stage.udf_class for stage in _fused_stages(physical)] == [_Udf1, _Udf2]
    expected = ActorPoolStrategy(**(actor_pool_kwargs or {}))
    assert _fused_logical_map(physical).compute == expected


@pytest.mark.parametrize("can_modify_num_rows", [False, True])
def test_physical_contract_lineage_op_map_name_and_metadata(can_modify_num_rows):
    source = InputData([])
    first = _make_actor_map(source, _Udf1)
    second = _make_actor_map(
        first,
        _Udf2,
        can_modify_num_rows=can_modify_num_rows,
    )
    third = _make_actor_map(second, _Udf3)
    raw = _create_unoptimized_physical_plan(third)

    assert len(_actor_map_operators(raw)) == 3
    fused = FuseCudfActorMapBatches().apply(raw)
    actor = fused.dag
    fused_logical = _fused_logical_map(fused)

    assert isinstance(actor, ActorPoolMapOperator)
    assert len(_actor_map_operators(fused)) == 1
    assert [stage.udf_class for stage in _fused_stages(fused)] == [_Udf1, _Udf2, _Udf3]
    assert actor.name == "MapBatches(_Udf1->_Udf2->_Udf3)"
    assert fused_logical.name == actor.name
    assert fused_logical.can_modify_num_rows is can_modify_num_rows
    assert fused_logical.batch_size == first.batch_size
    assert fused_logical.compute == first.compute
    assert actor._logical_operators == [first, second, third]
    assert set(fused.op_map) == set(_all_physical_operators(actor))
    assert all(op not in fused.op_map for op in _actor_map_operators(raw))
    assert actor.input_dependencies[0].output_dependencies == [actor]
    transform_fns = actor.get_map_transformer().get_transform_fns()
    assert sum(isinstance(fn, BatchMapTransformFn) for fn in transform_fns) == 1


@pytest.mark.parametrize("override_stage", ["upstream", "downstream"])
def test_single_target_max_block_size_override_is_preserved(override_stage):
    root, _ = _make_actor_map_chain(_Udf1, _Udf2)
    raw = _create_unoptimized_physical_plan(root)
    downstream = raw.dag
    upstream = downstream.input_dependencies[0]
    overridden = upstream if override_stage == "upstream" else downstream
    overridden.override_target_max_block_size(1024)

    fused = FuseCudfActorMapBatches().apply(raw)

    assert len(_actor_map_operators(fused)) == 1
    assert fused.dag.target_max_block_size_override == 1024


def test_conflicting_target_max_block_size_overrides_prevent_fusion():
    root, _ = _make_actor_map_chain(_Udf1, _Udf2)
    raw = _create_unoptimized_physical_plan(root)
    downstream = raw.dag
    upstream = downstream.input_dependencies[0]
    upstream.override_target_max_block_size(1024)
    downstream.override_target_max_block_size(2048)

    result = FuseCudfActorMapBatches().apply(raw)

    assert result is raw
    assert len(_actor_map_operators(result)) == 2


@pytest.mark.parametrize("unsupported_stage", ["upstream", "downstream"])
def test_physical_operator_that_disallows_fusion_prevents_cudf_fusion(
    unsupported_stage,
):
    root, _ = _make_actor_map_chain(_Udf1, _Udf2)
    raw = _create_unoptimized_physical_plan(root)
    downstream = raw.dag
    upstream = downstream.input_dependencies[0]
    unsupported = upstream if unsupported_stage == "upstream" else downstream
    unsupported._supports_fusion = False

    result = FuseCudfActorMapBatches().apply(raw)

    assert result is raw
    assert len(_actor_map_operators(result)) == 2


def test_rule_is_idempotent_and_replanning_does_not_mutate_logical_chain():
    root, operators = _make_actor_map_chain(_Udf1, _Udf2, _Udf3)
    logical_plan = LogicalPlan(root, _fusion_context(enabled=True))
    raw, _ = create_planner().plan(logical_plan)

    fused = FuseCudfActorMapBatches().apply(raw)
    assert FuseCudfActorMapBatches().apply(fused) is fused

    first_physical, _ = get_execution_plan(logical_plan)
    second_physical, _ = get_execution_plan(logical_plan)

    assert logical_plan.dag is root
    assert root.input_dependencies[0] is operators[1]
    assert operators[1].input_dependencies[0] is operators[0]
    for physical in (first_physical, second_physical):
        assert len(_actor_map_operators(physical)) == 1
        assert physical.dag._logical_operators == operators
        transform_fns = physical.dag.get_map_transformer().get_transform_fns()
        assert sum(isinstance(fn, BatchMapTransformFn) for fn in transform_fns) == 1


def test_stock_task_to_actor_fusion_still_runs_after_cudf_fusion():
    source = InputData([])
    task = _make_actor_map(source, _task_udf, compute=TaskPoolStrategy())
    first = _make_actor_map(task, _Udf1)
    second = _make_actor_map(first, _Udf2)

    physical, _ = get_execution_plan(LogicalPlan(second, _fusion_context(enabled=True)))

    assert len(_actor_map_operators(physical)) == 1
    assert not _task_map_operators(physical)
    assert physical.dag._logical_operators == [task, first, second]
    assert set(physical.op_map) == set(_all_physical_operators(physical.dag))


def test_dataset_creation_captures_fusion_setting(monkeypatch):
    context = DataContext.get_current()
    previous = context.enable_cudf_actor_fusion
    try:
        context.enable_cudf_actor_fusion = True
        captured = _empty_dataset_with_current_context(monkeypatch)
        context.enable_cudf_actor_fusion = False

        options = {
            "batch_format": "cudf",
            "batch_size": 8,
            "compute": ray.data.ActorPoolStrategy(size=1),
            "num_gpus": 1,
        }
        dataset = captured.map_batches(_Udf1, **options).map_batches(_Udf2, **options)
        physical, _ = get_execution_plan(dataset._logical_plan)
    finally:
        context.enable_cudf_actor_fusion = previous

    assert captured.context.enable_cudf_actor_fusion is True
    assert len(_actor_map_operators(physical)) == 1


# Fused actor execution


def test_direct_fusion_passes_row_changing_frames_without_rebatching(fake_cudf):
    seen = []
    runner = _FusedCudfMapBatches(
        (
            _CudfMapStage(_KeepEvenRows, "keep-even"),
            _CudfMapStage(
                _ObserveAndDuplicateRows,
                "duplicate",
                constructor_args=(seen,),
            ),
        )
    )

    result = runner(_FakeFrame([1, 2, 3, 4]))

    assert seen == [(2, 4)]
    assert result.rows == (2, 2, 4, 4)


def test_empty_frame_is_passed_through_every_stage(fake_cudf):
    calls = []
    empty = _FakeFrame([])
    runner = _FusedCudfMapBatches(
        (
            _CudfMapStage(
                _RecordIdentity,
                "first",
                constructor_args=(calls, "first"),
            ),
            _CudfMapStage(
                _RecordIdentity,
                "second",
                constructor_args=(calls, "second"),
            ),
        )
    )

    result = runner(empty)

    assert result is empty
    assert calls == [("first", empty), ("second", empty)]


def test_constructor_and_call_arguments_survive_physical_fusion(fake_cudf):
    source = InputData([])
    first = _make_actor_map(
        source,
        _ConfiguredStage,
        fn_constructor_args=(2,),
        fn_constructor_kwargs={"scale": 3},
        fn_args=(4,),
        fn_kwargs={"bias": 5},
    )
    second = _make_actor_map(
        first,
        _ConfiguredStage,
        fn_constructor_args=(-1,),
        fn_constructor_kwargs={"scale": 1},
        fn_args=(0,),
        fn_kwargs={"bias": 0},
    )

    fused = _apply_fusion_rule(second)
    stages = _fused_stages(fused)

    assert stages[0].constructor_args == (2,)
    assert stages[0].constructor_kwargs == {"scale": 3}
    assert stages[0].call_args == (4,)
    assert stages[0].call_kwargs == {"bias": 5}
    assert stages[1].constructor_args == (-1,)
    runner = _FusedCudfMapBatches(stages)
    assert runner(_FakeFrame([1])).rows == (25,)


def test_argument_iterables_are_preserved_until_actor_execution(fake_cudf):
    constructor_args = _TrackingIterable(2)
    call_args = _TrackingIterable(4)
    constructor_kwargs = {"scale": 3}
    call_kwargs = {"bias": 5}
    source = InputData([])
    first = _make_actor_map(
        source,
        _ConfiguredStage,
        fn_constructor_args=constructor_args,
        fn_constructor_kwargs=constructor_kwargs,
        fn_args=call_args,
        fn_kwargs=call_kwargs,
    )
    second = _make_actor_map(first, _Udf2)

    stages = _fused_stages(_apply_fusion_rule(second))

    assert constructor_args.iterations == 0
    assert call_args.iterations == 0
    assert stages[0].constructor_args is constructor_args
    assert stages[0].constructor_kwargs is constructor_kwargs
    assert stages[0].call_args is call_args
    assert stages[0].call_kwargs is call_kwargs
    runner = _FusedCudfMapBatches(stages)
    assert constructor_args.iterations == 1
    assert call_args.iterations == 0
    assert runner(_FakeFrame([1])).rows == (26,)
    assert call_args.iterations == 1


def test_repeated_classes_construct_distinct_stage_instances(fake_cudf):
    instances = []
    runner = _FusedCudfMapBatches(
        (
            _CudfMapStage(
                _StatefulStage,
                "plus-one",
                constructor_args=(instances, 1),
            ),
            _CudfMapStage(
                _StatefulStage,
                "plus-ten",
                constructor_args=(instances, 10),
            ),
        )
    )

    result = runner(_FakeFrame([0]))

    assert len(instances) == 2
    assert instances[0] is not instances[1]
    assert [instance.calls for instance in instances] == [1, 1]
    assert result.rows == (11,)


def test_repeated_classes_have_distinct_stage_labels():
    root, _ = _make_actor_map_chain(_Udf1, _Udf1)

    labels = [stage.error_label for stage in _fused_stages(_apply_fusion_rule(root))]

    assert labels == [
        "1: MapBatches(_Udf1)",
        "2: MapBatches(_Udf1)",
    ]


def test_single_stage_run_remains_unfused():
    single = _make_actor_map(InputData([]), _Udf1)
    raw = _create_unoptimized_physical_plan(single)

    assert FuseCudfActorMapBatches().apply(raw) is raw
    assert len(_actor_map_operators(raw)) == 1
    assert raw.op_map[raw.dag] is single


@pytest.mark.parametrize(
    ("udf_class", "label"),
    [
        (_WrongOutput, "wrong-output"),
        (_GeneratorOutput, "generator-output"),
        (_ListOutput, "list-output"),
        (_NoneOutput, "none-output"),
        (_AwaitableOutput, "awaitable-output"),
    ],
)
def test_wrong_stage_types_and_generator_outputs_are_rejected(
    fake_cudf, udf_class, label
):
    runner = _FusedCudfMapBatches((_CudfMapStage(udf_class, label),))

    with pytest.raises(TypeError, match="expected cudf\\.DataFrame") as error:
        runner(_FakeFrame([1]))

    assert repr(label) in str(error.value)


def test_stage_exceptions_are_labeled_and_chained(fake_cudf):
    runner = _FusedCudfMapBatches((_CudfMapStage(_RaiseFromStage, "explode"),))

    with pytest.raises(RuntimeError, match="stage 'explode' failed") as error:
        runner(_FakeFrame([1]))

    assert isinstance(error.value.__cause__, ValueError)
    assert str(error.value.__cause__) == "stage exploded"


def test_constructor_exceptions_are_labeled_and_chained():
    with pytest.raises(
        RuntimeError, match="stage 'construct' failed during construction"
    ) as error:
        _FusedCudfMapBatches((_CudfMapStage(_RaiseDuringConstruction, "construct"),))

    assert isinstance(error.value.__cause__, ValueError)
    assert str(error.value.__cause__) == "constructor exploded"


# Fusion eligibility and compatibility


@pytest.mark.parametrize(
    "udf_class",
    [_GeneratorUdf, _StaticAsyncUdf, _ClassGeneratorUdf],
    ids=["generator", "staticmethod-async", "classmethod-generator"],
)
def test_descriptor_wrapped_async_and_generator_stages_remain_unfused(udf_class):
    source = InputData([])
    upstream = _make_actor_map(source, _Udf1)
    unsupported = _make_actor_map(upstream, udf_class)

    result = _apply_fusion_rule(unsupported)

    assert len(_actor_map_operators(result)) == 2
    assert result.op_map[result.dag] is unsupported


@pytest.mark.parametrize(
    "overrides",
    [
        {"batch_format": "pandas"},
        {"batch_size": 0},
        {"batch_size": True},
        {"zero_copy_batch": False},
        {"fn": lambda batch: batch, "compute": ActorPoolStrategy(size=2)},
        {"compute": TaskPoolStrategy()},
        {"compute": ActorPoolStrategy(size=2, max_tasks_in_flight_per_actor=2)},
        {"compute": ActorPoolStrategy(size=2, enable_true_multi_threading=True)},
        {"ray_remote_args": {}},
        {"ray_remote_args": {"num_gpus": 0}},
        {"ray_remote_args": {"num_gpus": True}},
        {"ray_remote_args": {"num_gpus": 1, "max_concurrency": 2}},
        {"ray_remote_args": {"num_gpus": 1, "max_concurrency": 1.0}},
        {"ray_remote_args": {"num_gpus": Decimal("1")}},
        {"ray_remote_args": {"num_gpus": 1, "name": "shared-name"}},
        {"ray_remote_args": {"num_gpus": 1, "namespace": "actors"}},
        {"ray_remote_args": {"num_gpus": 1, "lifetime": "detached"}},
        {"ray_remote_args": {"num_gpus": 1, "get_if_exists": False}},
        {"ray_remote_args": {"num_gpus": 1, "max_pending_calls": 1}},
        {"ray_remote_args_fn": lambda: {"num_gpus": 1}},
        {"per_block_limit": 1},
        {"fn": _AsyncUdf},
        {"fn": _AsyncGeneratorUdf},
    ],
    ids=[
        "non-cudf",
        "zero-batch-size",
        "bool-batch-size",
        "copying-batches",
        "function-udf",
        "task-pool",
        "tasks-in-flight",
        "multithreaded-actor",
        "missing-gpu",
        "no-gpu",
        "bool-gpu",
        "concurrent-actor-calls",
        "float-actor-concurrency",
        "decimal-gpu",
        "actor-name",
        "actor-namespace",
        "actor-lifetime",
        "actor-get-if-exists",
        "actor-pending-calls",
        "dynamic-remote-args",
        "per-block-limit",
        "async-call",
        "async-generator-call",
    ],
)
def test_ineligible_map_breaks_a_fusion_run(overrides):
    source = InputData([])
    upstream = _make_actor_map(source, _Udf1)
    downstream = _make_actor_map(upstream, **overrides)

    result = _apply_fusion_rule(downstream)

    assert len(_actor_map_operators(result)) == 2 or isinstance(
        result.dag, TaskPoolMapOperator
    )
    assert result.op_map[result.dag] is downstream


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("actor_task_retry_on_errors", [ValueError]),
        ("retried_map_errors", ["ValueError"]),
        ("raise_original_map_exception", True),
        ("max_tasks_in_flight_per_actor", 2),
    ],
    ids=[
        "selective-actor-retry",
        "selective-map-retry",
        "original-exception",
        "context-tasks-in-flight",
    ],
)
def test_context_settings_with_incompatible_runtime_semantics_prevent_fusion(
    setting, value
):
    root, _ = _make_actor_map_chain(_Udf1, _Udf2)
    context = _fusion_context(enabled=True)
    setattr(context, setting, value)
    raw = _create_unoptimized_physical_plan(root, context=context)

    result = FuseCudfActorMapBatches().apply(raw)

    assert result is raw
    assert len(_actor_map_operators(result)) == 2


@pytest.mark.parametrize(
    ("upstream_overrides", "downstream_overrides"),
    [
        ({"batch_size": 4}, {"batch_size": 8}),
        (
            {"compute": ActorPoolStrategy(size=1)},
            {"compute": ActorPoolStrategy(size=2)},
        ),
        (
            {"compute": ActorPoolStrategy(min_size=1, max_size=3, initial_size=1)},
            {"compute": ActorPoolStrategy(min_size=1, max_size=3, initial_size=2)},
        ),
        (
            {"ray_remote_args": {"num_gpus": 1, "num_cpus": 1}},
            {"ray_remote_args": {"num_gpus": 1, "num_cpus": 2}},
        ),
        (
            {"ray_remote_args": {"num_gpus": 1}},
            {"ray_remote_args": {"num_gpus": 1, "num_cpus": 1}},
        ),
        (
            {"ray_remote_args": {"num_gpus": 1, "memory": 1}},
            {"ray_remote_args": {"num_gpus": 1, "memory": 2}},
        ),
        (
            {"ray_remote_args": {"num_gpus": 1, "resources": {"worker": 1}}},
            {"ray_remote_args": {"num_gpus": 1, "resources": {"worker": 2}}},
        ),
        (
            {
                "ray_remote_args": {
                    "num_gpus": 1,
                    "resources": {"worker": Decimal("0")},
                }
            },
            {"ray_remote_args": {"num_gpus": 1, "resources": {"worker": 0}}},
        ),
        (
            {
                "ray_remote_args": {
                    "num_gpus": 1,
                    "runtime_env": {"env_vars": {"A": "1"}},
                }
            },
            {
                "ray_remote_args": {
                    "num_gpus": 1,
                    "runtime_env": {"env_vars": {"A": "2"}},
                }
            },
        ),
        (
            {
                "ray_remote_args": {
                    "num_gpus": 1,
                    "runtime_env": {"config": {"eager_install": 1}},
                }
            },
            {
                "ray_remote_args": {
                    "num_gpus": 1,
                    "runtime_env": {"config": {"eager_install": True}},
                }
            },
        ),
        (
            {"ray_remote_args": {"num_gpus": 1, "scheduling_strategy": "SPREAD"}},
            {"ray_remote_args": {"num_gpus": 1, "scheduling_strategy": "DEFAULT"}},
        ),
        (
            {"ray_remote_args": {"num_gpus": 1, "max_restarts": -1}},
            {"ray_remote_args": {"num_gpus": 1, "max_restarts": 0}},
        ),
        (
            {
                "ray_remote_args": {
                    "num_gpus": 1,
                    "placement_group_capture_child_tasks": True,
                }
            },
            {
                "ray_remote_args": {
                    "num_gpus": 1,
                    "placement_group_capture_child_tasks": False,
                }
            },
        ),
        (
            {"min_rows_per_bundled_input": 4},
            {"min_rows_per_bundled_input": 8},
        ),
    ],
    ids=[
        "batch-size",
        "actor-size",
        "actor-autoscaling-initial-size",
        "cpu",
        "implicit-versus-explicit-cpu",
        "memory",
        "custom-resource",
        "custom-resource-value-types",
        "runtime-environment",
        "runtime-environment-value-types",
        "scheduling",
        "retry",
        "placement",
        "input-bundling",
    ],
)
def test_individually_eligible_but_incompatible_maps_do_not_fuse(
    upstream_overrides, downstream_overrides
):
    source = InputData([])
    upstream = _make_actor_map(source, _Udf1, **upstream_overrides)
    downstream = _make_actor_map(upstream, _Udf2, **downstream_overrides)

    result = _apply_fusion_rule(downstream)

    assert len(_actor_map_operators(result)) == 2
    assert result.op_map[result.dag] is downstream


def test_identical_raw_remote_args_are_preserved():
    source = InputData([])
    remote_args = {
        "num_gpus": 1,
        "num_cpus": 2,
        "resources": {"worker": 1},
        "max_concurrency": 1,
        "runtime_env": {"env_vars": {"MODE": "test"}},
    }
    upstream = _make_actor_map(source, _Udf1, ray_remote_args=remote_args)
    downstream = _make_actor_map(
        upstream,
        _Udf2,
        ray_remote_args=dict(remote_args),
    )

    fused = _apply_fusion_rule(downstream)
    fused_logical = _fused_logical_map(fused)

    assert [stage.udf_class for stage in _fused_stages(fused)] == [_Udf1, _Udf2]
    assert fused_logical.ray_remote_args == remote_args


# Fusion boundaries


def test_cpu_barrier_keeps_two_composite_regions_separate():
    source = InputData([])
    first = _make_actor_map(source, _Udf1)
    second = _make_actor_map(first, _Udf2)
    barrier = MapRows(lambda row: row, input_dependencies=[second])
    third = _make_actor_map(barrier, _Udf3)
    fourth = _make_actor_map(third, _Udf4)

    result = _apply_fusion_rule(fourth)
    downstream = result.dag
    barrier_physical = downstream.input_dependencies[0]
    upstream = barrier_physical.input_dependencies[0]

    assert [stage.udf_class for stage in _fused_stages(result, downstream)] == [
        _Udf3,
        _Udf4,
    ]
    assert isinstance(barrier_physical, TaskPoolMapOperator)
    assert [stage.udf_class for stage in _fused_stages(result, upstream)] == [
        _Udf1,
        _Udf2,
    ]
    assert len(_actor_map_operators(result)) == 2


def test_non_cudf_actor_barrier_keeps_two_composite_regions_separate():
    source = InputData([])
    first = _make_actor_map(source, _Udf1)
    second = _make_actor_map(first, _Udf2)
    barrier = _make_actor_map(second, _Udf3, batch_format="pandas")
    third = _make_actor_map(barrier, _Udf4)
    fourth = _make_actor_map(third, _Udf5)

    result = _apply_fusion_rule(fourth)
    downstream = result.dag
    barrier_physical = downstream.input_dependencies[0]
    upstream = barrier_physical.input_dependencies[0]

    assert [stage.udf_class for stage in _fused_stages(result, downstream)] == [
        _Udf4,
        _Udf5,
    ]
    assert result.op_map[barrier_physical] is barrier
    assert [stage.udf_class for stage in _fused_stages(result, upstream)] == [
        _Udf1,
        _Udf2,
    ]
    assert len(_actor_map_operators(result)) == 3


def test_shared_logical_branch_is_not_absorbed_despite_separate_physical_copies():
    source = InputData([])
    shared = _make_actor_map(source, _Udf1)
    left_first = _make_actor_map(shared, _Udf2)
    left = _make_actor_map(left_first, _Udf3)
    right_first = _make_actor_map(shared, _Udf4)
    right = _make_actor_map(right_first, _Udf5)

    result = _apply_fusion_rule(Union([left, right]))
    left_fused, right_fused = result.dag.input_dependencies
    left_shared = left_fused.input_dependencies[0]
    right_shared = right_fused.input_dependencies[0]

    assert isinstance(result.dag, UnionOperator)
    assert [stage.udf_class for stage in _fused_stages(result, left_fused)] == [
        _Udf2,
        _Udf3,
    ]
    assert [stage.udf_class for stage in _fused_stages(result, right_fused)] == [
        _Udf4,
        _Udf5,
    ]
    assert left_shared is not right_shared
    assert result.op_map[left_shared] is shared
    assert result.op_map[right_shared] is shared
    assert left_shared._logical_operators == [shared]
    assert right_shared._logical_operators == [shared]
    assert len(_actor_map_operators(result)) == 4
    assert set(result.op_map) == set(_all_physical_operators(result.dag))


def test_udf_arguments_with_unusable_repr_can_be_physically_planned():
    source = InputData([])
    first = _make_actor_map(source, _Udf1, fn_args=(_ReprBomb(),))
    second = _make_actor_map(first, _Udf2)

    physical, _ = get_execution_plan(LogicalPlan(second, _fusion_context(enabled=True)))

    assert len(_actor_map_operators(physical)) == 1
