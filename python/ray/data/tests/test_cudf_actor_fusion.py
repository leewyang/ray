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


class _PlanA:
    def __call__(self, batch):
        return batch


class _PlanB:
    def __call__(self, batch):
        return batch


class _PlanC:
    def __call__(self, batch):
        return batch


class _PlanD:
    def __call__(self, batch):
        return batch


class _PlanE:
    def __call__(self, batch):
        return batch


class _AsyncPlanStage:
    async def __call__(self, batch):
        return batch


class _AsyncGeneratorPlanStage:
    async def __call__(self, batch):
        yield batch


class _GeneratorPlanStage:
    def __call__(self, batch):
        yield batch


class _StaticAsyncPlanStage:
    @staticmethod
    async def __call__(batch):
        return batch


class _ClassGeneratorPlanStage:
    @classmethod
    def __call__(cls, batch):
        yield batch


def _task_plan_stage(batch):
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


def _actor_map(
    input_op,
    fn=_PlanA,
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


def _chain(*stage_classes, **kwargs):
    current = InputData([])
    operators = []
    for stage_class in stage_classes:
        current = _actor_map(current, stage_class, **kwargs)
        operators.append(current)
    return current, operators


def _context(*, enabled):
    context = DataContext.get_current().copy()
    context.enable_cudf_actor_fusion = enabled
    return context


def _raw_physical_plan(root, *, enabled=True, context=None):
    if context is None:
        context = _context(enabled=enabled)
    plan, _ = create_planner().plan(LogicalPlan(root, context))
    return plan


def _apply(root, *, enabled=True, context=None):
    return FuseCudfActorMapBatches().apply(
        _raw_physical_plan(root, enabled=enabled, context=context)
    )


def _reachable(root):
    return list(root.post_order_iter())


def _actors(plan):
    return [op for op in _reachable(plan.dag) if isinstance(op, ActorPoolMapOperator)]


def _tasks(plan):
    return [op for op in _reachable(plan.dag) if isinstance(op, TaskPoolMapOperator)]


def _fused_logical(plan, physical_op=None):
    if physical_op is None:
        physical_op = plan.dag
    logical_op = plan.op_map[physical_op]
    assert isinstance(logical_op, MapBatches)
    assert logical_op.fn is _FusedCudfMapBatches
    return logical_op


def _stages(plan, physical_op=None):
    logical_op = _fused_logical(plan, physical_op)
    assert logical_op.fn_constructor_args is not None
    return logical_op.fn_constructor_args[0]


def _empty_dataset_from_current(monkeypatch):
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


def test_default_disabled_and_logical_plan_is_never_rewritten():
    assert DEFAULT_ENABLE_CUDF_ACTOR_FUSION is False
    assert DataContext().enable_cudf_actor_fusion is False
    root, operators = _chain(_PlanA, _PlanB)

    optimized = LogicalOptimizer().optimize(LogicalPlan(root, _context(enabled=True)))
    assert optimized.dag is root
    assert root.fn is _PlanB
    assert root.input_dependencies[0] is operators[0]

    disabled = _raw_physical_plan(root, enabled=False)
    assert FuseCudfActorMapBatches().apply(disabled) is disabled
    assert len(_actors(disabled)) == 2

    logical_plan = LogicalPlan(root, _context(enabled=True))
    physical, _ = get_execution_plan(logical_plan)
    assert logical_plan.dag is root
    assert root.input_dependencies[0] is operators[0]
    assert len(_actors(physical)) == 1


@pytest.mark.parametrize(
    "stage_classes",
    [(_PlanA, _PlanB), (_PlanA, _PlanB, _PlanC)],
    ids=["two-stages", "three-stages"],
)
def test_public_map_batches_api_fuses_at_physical_planning(monkeypatch, stage_classes):
    context = DataContext.get_current()
    previous = context.enable_cudf_actor_fusion
    context.enable_cudf_actor_fusion = True
    try:
        dataset = _empty_dataset_from_current(monkeypatch)
    finally:
        context.enable_cudf_actor_fusion = previous

    options = {
        "batch_format": "cudf",
        "batch_size": 250_000,
        "compute": ray.data.ActorPoolStrategy(size=4),
        "num_gpus": 1,
    }
    for stage_class in stage_classes:
        dataset = dataset.map_batches(stage_class, **options)

    logical_root = dataset._logical_plan.dag
    physical, _ = get_execution_plan(dataset._logical_plan)

    assert logical_root.fn is stage_classes[-1]
    assert dataset._logical_plan.dag is logical_root
    assert len(_actors(physical)) == 1
    assert [op.fn for op in physical.dag._logical_operators] == list(stage_classes)


@pytest.mark.parametrize("can_modify_num_rows", [False, True])
def test_physical_contract_lineage_op_map_name_and_metadata(can_modify_num_rows):
    source = InputData([])
    first = _actor_map(source, _PlanA)
    second = _actor_map(
        first,
        _PlanB,
        can_modify_num_rows=can_modify_num_rows,
    )
    third = _actor_map(second, _PlanC)
    raw = _raw_physical_plan(third)

    assert len(_actors(raw)) == 3
    fused = FuseCudfActorMapBatches().apply(raw)
    actor = fused.dag
    fused_logical = _fused_logical(fused)

    assert isinstance(actor, ActorPoolMapOperator)
    assert len(_actors(fused)) == 1
    assert [stage.fn for stage in _stages(fused)] == [_PlanA, _PlanB, _PlanC]
    assert actor.name == "MapBatches(_PlanA->_PlanB->_PlanC)"
    assert fused_logical.name == actor.name
    assert fused_logical.can_modify_num_rows is can_modify_num_rows
    assert fused_logical.batch_size == first.batch_size
    assert fused_logical.compute == first.compute
    assert actor._logical_operators == [first, second, third]
    assert set(fused.op_map) == set(_reachable(actor))
    assert all(op not in fused.op_map for op in _actors(raw))
    assert actor.input_dependencies[0].output_dependencies == [actor]
    transform_fns = actor.get_map_transformer().get_transform_fns()
    assert sum(isinstance(fn, BatchMapTransformFn) for fn in transform_fns) == 1


@pytest.mark.parametrize("override_stage", ["upstream", "downstream"])
def test_single_target_max_block_size_override_is_preserved(override_stage):
    root, _ = _chain(_PlanA, _PlanB)
    raw = _raw_physical_plan(root)
    downstream = raw.dag
    upstream = downstream.input_dependencies[0]
    overridden = upstream if override_stage == "upstream" else downstream
    overridden.override_target_max_block_size(1024)

    fused = FuseCudfActorMapBatches().apply(raw)

    assert len(_actors(fused)) == 1
    assert fused.dag.target_max_block_size_override == 1024


def test_conflicting_target_max_block_size_overrides_prevent_fusion():
    root, _ = _chain(_PlanA, _PlanB)
    raw = _raw_physical_plan(root)
    downstream = raw.dag
    upstream = downstream.input_dependencies[0]
    upstream.override_target_max_block_size(1024)
    downstream.override_target_max_block_size(2048)

    result = FuseCudfActorMapBatches().apply(raw)

    assert result is raw
    assert len(_actors(result)) == 2


@pytest.mark.parametrize("unsupported_stage", ["upstream", "downstream"])
def test_physical_operator_that_disallows_fusion_prevents_cudf_fusion(
    unsupported_stage,
):
    root, _ = _chain(_PlanA, _PlanB)
    raw = _raw_physical_plan(root)
    downstream = raw.dag
    upstream = downstream.input_dependencies[0]
    unsupported = upstream if unsupported_stage == "upstream" else downstream
    unsupported._supports_fusion = False

    result = FuseCudfActorMapBatches().apply(raw)

    assert result is raw
    assert len(_actors(result)) == 2


def test_rule_is_idempotent_and_replanning_does_not_mutate_logical_chain():
    root, operators = _chain(_PlanA, _PlanB, _PlanC)
    logical_plan = LogicalPlan(root, _context(enabled=True))
    raw, _ = create_planner().plan(logical_plan)

    fused = FuseCudfActorMapBatches().apply(raw)
    assert FuseCudfActorMapBatches().apply(fused) is fused

    first_physical, _ = get_execution_plan(logical_plan)
    second_physical, _ = get_execution_plan(logical_plan)

    assert logical_plan.dag is root
    assert root.input_dependencies[0] is operators[1]
    assert operators[1].input_dependencies[0] is operators[0]
    for physical in (first_physical, second_physical):
        assert len(_actors(physical)) == 1
        assert physical.dag._logical_operators == operators
        transform_fns = physical.dag.get_map_transformer().get_transform_fns()
        assert sum(isinstance(fn, BatchMapTransformFn) for fn in transform_fns) == 1


def test_stock_task_to_actor_fusion_still_runs_after_cudf_fusion():
    source = InputData([])
    task = _actor_map(source, _task_plan_stage, compute=TaskPoolStrategy())
    first = _actor_map(task, _PlanA)
    second = _actor_map(first, _PlanB)

    physical, _ = get_execution_plan(LogicalPlan(second, _context(enabled=True)))

    assert len(_actors(physical)) == 1
    assert not _tasks(physical)
    assert physical.dag._logical_operators == [task, first, second]
    assert set(physical.op_map) == set(_reachable(physical.dag))


def test_dataset_creation_captures_fusion_setting(monkeypatch):
    context = DataContext.get_current()
    previous = context.enable_cudf_actor_fusion
    try:
        context.enable_cudf_actor_fusion = True
        captured = _empty_dataset_from_current(monkeypatch)
        context.enable_cudf_actor_fusion = False

        options = {
            "batch_format": "cudf",
            "batch_size": 8,
            "compute": ray.data.ActorPoolStrategy(size=1),
            "num_gpus": 1,
        }
        dataset = captured.map_batches(_PlanA, **options).map_batches(_PlanB, **options)
        physical, _ = get_execution_plan(dataset._logical_plan)
    finally:
        context.enable_cudf_actor_fusion = previous

    assert captured.context.enable_cudf_actor_fusion is True
    assert len(_actors(physical)) == 1


def test_direct_fusion_passes_row_changing_frames_without_rebatching(fake_cudf):
    seen = []
    runner = _FusedCudfMapBatches(
        (
            _CudfMapStage(_KeepEvenRows, "keep-even"),
            _CudfMapStage(
                _ObserveAndDuplicateRows,
                "duplicate",
                fn_constructor_args=(seen,),
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
                fn_constructor_args=(calls, "first"),
            ),
            _CudfMapStage(
                _RecordIdentity,
                "second",
                fn_constructor_args=(calls, "second"),
            ),
        )
    )

    result = runner(empty)

    assert result is empty
    assert calls == [("first", empty), ("second", empty)]


def test_constructor_and_call_arguments_survive_physical_fusion(fake_cudf):
    source = InputData([])
    first = _actor_map(
        source,
        _ConfiguredStage,
        fn_constructor_args=(2,),
        fn_constructor_kwargs={"scale": 3},
        fn_args=(4,),
        fn_kwargs={"bias": 5},
    )
    second = _actor_map(
        first,
        _ConfiguredStage,
        fn_constructor_args=(-1,),
        fn_constructor_kwargs={"scale": 1},
        fn_args=(0,),
        fn_kwargs={"bias": 0},
    )

    fused = _apply(second)
    stages = _stages(fused)

    assert stages[0].fn_constructor_args == (2,)
    assert stages[0].fn_constructor_kwargs == {"scale": 3}
    assert stages[0].fn_args == (4,)
    assert stages[0].fn_kwargs == {"bias": 5}
    assert stages[1].fn_constructor_args == (-1,)
    runner = _FusedCudfMapBatches(stages)
    assert runner(_FakeFrame([1])).rows == (25,)


def test_argument_iterables_are_preserved_until_actor_execution(fake_cudf):
    constructor_args = _TrackingIterable(2)
    call_args = _TrackingIterable(4)
    constructor_kwargs = {"scale": 3}
    call_kwargs = {"bias": 5}
    source = InputData([])
    first = _actor_map(
        source,
        _ConfiguredStage,
        fn_constructor_args=constructor_args,
        fn_constructor_kwargs=constructor_kwargs,
        fn_args=call_args,
        fn_kwargs=call_kwargs,
    )
    second = _actor_map(first, _PlanB)

    stages = _stages(_apply(second))

    assert constructor_args.iterations == 0
    assert call_args.iterations == 0
    assert stages[0].fn_constructor_args is constructor_args
    assert stages[0].fn_constructor_kwargs is constructor_kwargs
    assert stages[0].fn_args is call_args
    assert stages[0].fn_kwargs is call_kwargs
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
                fn_constructor_args=(instances, 1),
            ),
            _CudfMapStage(
                _StatefulStage,
                "plus-ten",
                fn_constructor_args=(instances, 10),
            ),
        )
    )

    result = runner(_FakeFrame([0]))

    assert len(instances) == 2
    assert instances[0] is not instances[1]
    assert [instance.calls for instance in instances] == [1, 1]
    assert result.rows == (11,)


def test_repeated_classes_have_distinct_stage_labels():
    root, _ = _chain(_PlanA, _PlanA)

    labels = [stage.label for stage in _stages(_apply(root))]

    assert labels == [
        "1: MapBatches(_PlanA)",
        "2: MapBatches(_PlanA)",
    ]


def test_single_stage_run_remains_unfused():
    single = _actor_map(InputData([]), _PlanA)
    raw = _raw_physical_plan(single)

    assert FuseCudfActorMapBatches().apply(raw) is raw
    assert len(_actors(raw)) == 1
    assert raw.op_map[raw.dag] is single


@pytest.mark.parametrize(
    ("stage_class", "label"),
    [
        (_WrongOutput, "wrong-output"),
        (_GeneratorOutput, "generator-output"),
        (_ListOutput, "list-output"),
        (_NoneOutput, "none-output"),
        (_AwaitableOutput, "awaitable-output"),
    ],
)
def test_wrong_stage_types_and_generator_outputs_are_rejected(
    fake_cudf, stage_class, label
):
    runner = _FusedCudfMapBatches((_CudfMapStage(stage_class, label),))

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


@pytest.mark.parametrize(
    "stage_class",
    [_GeneratorPlanStage, _StaticAsyncPlanStage, _ClassGeneratorPlanStage],
    ids=["generator", "staticmethod-async", "classmethod-generator"],
)
def test_descriptor_wrapped_async_and_generator_stages_remain_unfused(stage_class):
    source = InputData([])
    upstream = _actor_map(source, _PlanA)
    unsupported = _actor_map(upstream, stage_class)

    result = _apply(unsupported)

    assert len(_actors(result)) == 2
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
        {"compute": ActorPoolStrategy(min_size=1, max_size=2)},
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
        {"fn": _AsyncPlanStage},
        {"fn": _AsyncGeneratorPlanStage},
    ],
    ids=[
        "non-cudf",
        "zero-batch-size",
        "bool-batch-size",
        "copying-batches",
        "function-udf",
        "task-pool",
        "autoscaling-actors",
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
    upstream = _actor_map(source, _PlanA)
    downstream = _actor_map(upstream, **overrides)

    result = _apply(downstream)

    assert len(_actors(result)) == 2 or isinstance(result.dag, TaskPoolMapOperator)
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
    root, _ = _chain(_PlanA, _PlanB)
    context = _context(enabled=True)
    setattr(context, setting, value)
    raw = _raw_physical_plan(root, context=context)

    result = FuseCudfActorMapBatches().apply(raw)

    assert result is raw
    assert len(_actors(result)) == 2


@pytest.mark.parametrize(
    ("upstream_overrides", "downstream_overrides"),
    [
        ({"batch_size": 4}, {"batch_size": 8}),
        (
            {"compute": ActorPoolStrategy(size=1)},
            {"compute": ActorPoolStrategy(size=2)},
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
    upstream = _actor_map(source, _PlanA, **upstream_overrides)
    downstream = _actor_map(upstream, _PlanB, **downstream_overrides)

    result = _apply(downstream)

    assert len(_actors(result)) == 2
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
    upstream = _actor_map(source, _PlanA, ray_remote_args=remote_args)
    downstream = _actor_map(
        upstream,
        _PlanB,
        ray_remote_args=dict(remote_args),
    )

    fused = _apply(downstream)
    fused_logical = _fused_logical(fused)

    assert [stage.fn for stage in _stages(fused)] == [_PlanA, _PlanB]
    assert fused_logical.ray_remote_args == remote_args


def test_cpu_barrier_keeps_two_composite_regions_separate():
    source = InputData([])
    first = _actor_map(source, _PlanA)
    second = _actor_map(first, _PlanB)
    barrier = MapRows(lambda row: row, input_dependencies=[second])
    third = _actor_map(barrier, _PlanC)
    fourth = _actor_map(third, _PlanD)

    result = _apply(fourth)
    downstream = result.dag
    barrier_physical = downstream.input_dependencies[0]
    upstream = barrier_physical.input_dependencies[0]

    assert [stage.fn for stage in _stages(result, downstream)] == [_PlanC, _PlanD]
    assert isinstance(barrier_physical, TaskPoolMapOperator)
    assert [stage.fn for stage in _stages(result, upstream)] == [_PlanA, _PlanB]
    assert len(_actors(result)) == 2


def test_non_cudf_actor_barrier_keeps_two_composite_regions_separate():
    source = InputData([])
    first = _actor_map(source, _PlanA)
    second = _actor_map(first, _PlanB)
    barrier = _actor_map(second, _PlanC, batch_format="pandas")
    third = _actor_map(barrier, _PlanD)
    fourth = _actor_map(third, _PlanE)

    result = _apply(fourth)
    downstream = result.dag
    barrier_physical = downstream.input_dependencies[0]
    upstream = barrier_physical.input_dependencies[0]

    assert [stage.fn for stage in _stages(result, downstream)] == [_PlanD, _PlanE]
    assert result.op_map[barrier_physical] is barrier
    assert [stage.fn for stage in _stages(result, upstream)] == [_PlanA, _PlanB]
    assert len(_actors(result)) == 3


def test_shared_logical_branch_is_not_absorbed_despite_separate_physical_copies():
    source = InputData([])
    shared = _actor_map(source, _PlanA)
    left_first = _actor_map(shared, _PlanB)
    left = _actor_map(left_first, _PlanC)
    right_first = _actor_map(shared, _PlanD)
    right = _actor_map(right_first, _PlanE)

    result = _apply(Union([left, right]))
    left_fused, right_fused = result.dag.input_dependencies
    left_shared = left_fused.input_dependencies[0]
    right_shared = right_fused.input_dependencies[0]

    assert isinstance(result.dag, UnionOperator)
    assert [stage.fn for stage in _stages(result, left_fused)] == [_PlanB, _PlanC]
    assert [stage.fn for stage in _stages(result, right_fused)] == [_PlanD, _PlanE]
    assert left_shared is not right_shared
    assert result.op_map[left_shared] is shared
    assert result.op_map[right_shared] is shared
    assert left_shared._logical_operators == [shared]
    assert right_shared._logical_operators == [shared]
    assert len(_actors(result)) == 4
    assert set(result.op_map) == set(_reachable(result.dag))


def test_udf_arguments_with_unusable_repr_can_be_physically_planned():
    source = InputData([])
    first = _actor_map(source, _PlanA, fn_args=(_ReprBomb(),))
    second = _actor_map(first, _PlanB)

    physical, _ = get_execution_plan(LogicalPlan(second, _context(enabled=True)))

    assert len(_actors(physical)) == 1
