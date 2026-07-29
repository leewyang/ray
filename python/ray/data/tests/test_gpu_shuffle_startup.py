import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ray.data._internal.execution import gpu_shuffle_startup
from ray.data._internal.execution.interfaces import (
    ExecutionOptions,
    ExecutionResources,
    PhysicalOperator,
)
from ray.data._internal.execution.operators.actor_pool_map_operator import (
    ActorPoolMapOperator,
)
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.task_pool_map_operator import (
    TaskPoolMapOperator,
)
from ray.data._internal.gpu_shuffle.hash_shuffle import (
    GPURankPool,
    GPUShuffleOperator,
)


class _PassthroughOperator(PhysicalOperator):
    pass


class _BlockingStartOperator(PhysicalOperator):
    def start(self, options, block_ref_counter):
        raise AssertionError("The planner must not start operators.")


class _OtherMapOperator(MapOperator):
    def _input_queues(self):
        return []

    def _output_queues(self):
        return []

    def _try_schedule_task(self, refs, strict):
        raise AssertionError("The planner must not schedule tasks.")

    def progress_str(self):
        return ""

    def current_logical_usage(self):
        return ExecutionResources.zero()

    def pending_logical_usage(self):
        return ExecutionResources.zero()

    def incremental_resource_usage(self):
        return ExecutionResources.zero()


class _CustomRankPool(GPURankPool):
    pass


class _CustomGPUShuffleOperator(GPUShuffleOperator):
    pass


class _CustomInputDataBuffer(InputDataBuffer):
    pass


def _bare_operator(cls, name):
    op = object.__new__(cls)
    op._name = name
    op._input_dependencies = []
    op._output_dependencies = []
    op._data_context = SimpleNamespace(
        wait_for_min_actors_s=0,
        scheduling_strategy="SPREAD",
        scheduling_strategy_large_args="DEFAULT",
    )
    op._additional_split_factor = None
    return op


def _input(name="Input"):
    return _bare_operator(InputDataBuffer, name)


def _task(remote_args=None, *, dynamic=False, name="Task"):
    op = _bare_operator(TaskPoolMapOperator, name)
    op._has_user_provided_ray_remote_args_fn = dynamic
    op._ray_remote_args_fn = object() if dynamic else None
    op._ray_remote_args = {} if remote_args is None else remote_args
    return op


def _actor(
    remote_args=None,
    *,
    sizes=(1, 1, 1),
    dynamic=False,
    wait_for_min_actors_s=0,
    name="Actor",
):
    op = _bare_operator(ActorPoolMapOperator, name)
    op._has_user_provided_ray_remote_args_fn = dynamic
    op._ray_remote_args_fn = object() if dynamic else None
    op._ray_remote_args = {} if remote_args is None else remote_args
    op._data_context.wait_for_min_actors_s = wait_for_min_actors_s
    op._actor_pool = MagicMock()
    op._actor_pool.min_size.return_value = sizes[0]
    op._actor_pool.initial_size.return_value = sizes[1]
    op._actor_pool.max_size.return_value = sizes[2]
    return op


def _other_map(remote_args=None, *, name="OtherMap"):
    op = _bare_operator(_OtherMapOperator, name)
    op._has_user_provided_ray_remote_args_fn = False
    op._ray_remote_args_fn = None
    op._ray_remote_args = {} if remote_args is None else remote_args
    return op


def _shuffle(*, nranks=2, custom_pool=False, custom_operator=False, name="GPUShuffle"):
    operator_cls = _CustomGPUShuffleOperator if custom_operator else GPUShuffleOperator
    op = _bare_operator(operator_cls, name)
    pool_cls = _CustomRankPool if custom_pool else GPURankPool
    op._rank_pool = object.__new__(pool_cls)
    op._rank_pool._nranks = nranks
    return op


def _topology(*ops):
    states = [object() for _ in ops]
    for index, op in enumerate(ops):
        op._input_dependencies = [] if index == 0 else [ops[index - 1]]
        op._output_dependencies = [] if index == len(ops) - 1 else [ops[index + 1]]
    return dict(zip(ops, states))


def _options(*, limits=None, excluded=None, label_selector=None):
    options = ExecutionOptions(
        resource_limits=limits,
        label_selector=label_selector,
    )
    if excluded is not None:
        options._set_exclude_resources(excluded)
    return options


def _plan(topology, capacity, *, options=None):
    if options is None:
        options = _options()
    with patch.object(gpu_shuffle_startup, "_known_capacity", return_value=capacity):
        return gpu_shuffle_startup.plan_gpu_shuffle_startup(topology, options)


def _supported_task_pipeline(*, task_resources=None, nranks=2):
    if task_resources is None:
        task_resources = {"num_cpus": 1, "num_gpus": 1}
    input_op = _input()
    producer = _task(task_resources)
    shuffle = _shuffle(nranks=nranks)
    return input_op, producer, shuffle, _topology(input_op, producer, shuffle)


def test_fallback_logs_reason():
    with patch.object(gpu_shuffle_startup.logger, "debug") as debug:
        gpu_shuffle_startup._fallback("proof failed")

    debug.assert_called_once_with("Using stock GPU shuffle startup: %s", "proof failed")


def test_is_linear_accepts_empty_and_well_formed_topologies():
    assert gpu_shuffle_startup._is_linear([])

    topology = _topology(
        _input(),
        _task({"num_gpus": 1}),
        _shuffle(),
    )
    assert gpu_shuffle_startup._is_linear(list(topology))


@pytest.mark.parametrize("broken_edge", ["input", "output"])
def test_is_linear_rejects_each_broken_edge_direction(broken_edge):
    input_op = _input()
    task = _task({"num_gpus": 1})
    shuffle = _shuffle()
    topology = _topology(input_op, task, shuffle)
    if broken_edge == "input":
        task._input_dependencies = []
    else:
        task._output_dependencies = []

    assert not gpu_shuffle_startup._is_linear(list(topology))


@pytest.mark.parametrize(
    "remote_args",
    [
        {"accelerator_type": "A100"},
        {"fallback_strategy": [{"label_selector": {"rack": "a"}}]},
        {"label_selector": {"rack": "a"}},
        {"memory": 1},
        {"object_store_memory": 1},
        {"placement_group": object()},
        {"placement_group_bundle_index": 0},
        {"placement_group_capture_child_tasks": False},
        {"resources": {"custom": 1}},
    ],
)
def test_has_unsupported_options_rejects_non_scalar_placement(remote_args):
    assert gpu_shuffle_startup._has_unsupported_options(_task(), remote_args)


@pytest.mark.parametrize(
    "remote_args",
    [
        {"get_if_exists": True},
        {"lifetime": "detached"},
        {"name": "worker"},
        {"namespace": "data"},
    ],
)
def test_has_unsupported_options_rejects_actor_identity_or_lifetime(remote_args):
    assert gpu_shuffle_startup._has_unsupported_options(_actor(), remote_args)


def test_has_unsupported_options_allows_task_name():
    assert not gpu_shuffle_startup._has_unsupported_options(
        _task(), {"name": "gpu-task"}
    )


@pytest.mark.parametrize(
    "scheduling_strategy",
    [None, "DEFAULT", "SPREAD"],
)
def test_has_unsupported_options_accepts_known_strategies(scheduling_strategy):
    assert not gpu_shuffle_startup._has_unsupported_options(
        _task(), {"scheduling_strategy": scheduling_strategy}
    )


def test_has_unsupported_options_rejects_unknown_strategy():
    assert gpu_shuffle_startup._has_unsupported_options(
        _task(), {"scheduling_strategy": "NODE_AFFINITY"}
    )


@pytest.mark.parametrize(
    "attribute",
    ["scheduling_strategy", "scheduling_strategy_large_args"],
)
def test_has_unsupported_options_rejects_context_placement(attribute):
    op = _task()
    setattr(op.data_context, attribute, object())

    assert gpu_shuffle_startup._has_unsupported_options(op, {})


@pytest.mark.parametrize(
    ("remote_args", "actor", "expected_cpu", "expected_gpu"),
    [
        ({"num_cpus": 2, "num_gpus": 3}, True, 2, 3),
        ({"num_gpus": 1}, True, 1, 1),
        ({"num_gpus": 0}, True, 1, 0),
        ({}, True, 0, 0),
        ({}, False, 1, 0),
        ({"num_cpus": 0, "num_gpus": 2}, False, 0, 2),
    ],
)
def test_processor_resources_applies_ray_cpu_defaults(
    remote_args, actor, expected_cpu, expected_gpu
):
    resources = gpu_shuffle_startup._processor_resources(remote_args, actor=actor)

    assert resources.cpu == expected_cpu
    assert resources.gpu == expected_gpu


def test_known_capacity_prefers_configured_cluster_maximum():
    options = _options(
        limits=ExecutionResources(cpu=6, gpu=10),
        excluded=ExecutionResources(cpu=2, gpu=1),
    )
    with (
        patch.object(
            gpu_shuffle_startup.ray._private.state.state,
            "get_max_resources_from_cluster_config",
            return_value={"CPU": 8, "GPU": 4},
        ),
        patch.object(gpu_shuffle_startup, "is_autoscaling_enabled") as autoscaling,
        patch.object(gpu_shuffle_startup.ray, "cluster_resources") as cluster,
    ):
        capacity = gpu_shuffle_startup._known_capacity(options)

    assert capacity == ExecutionResources(cpu=4, gpu=3)
    autoscaling.assert_not_called()
    cluster.assert_not_called()


def test_known_capacity_is_unknown_for_unbounded_autoscaling():
    with (
        patch.object(
            gpu_shuffle_startup.ray._private.state.state,
            "get_max_resources_from_cluster_config",
            return_value={},
        ),
        patch.object(gpu_shuffle_startup, "is_autoscaling_enabled", return_value=True),
        patch.object(gpu_shuffle_startup.ray, "cluster_resources") as cluster,
    ):
        capacity = gpu_shuffle_startup._known_capacity(_options())

    assert capacity is None
    cluster.assert_not_called()


def test_known_capacity_rejects_unbounded_configured_maximum():
    with (
        patch.object(
            gpu_shuffle_startup.ray._private.state.state,
            "get_max_resources_from_cluster_config",
            return_value={"CPU": sys.maxsize, "GPU": 4},
        ),
        patch.object(gpu_shuffle_startup, "is_autoscaling_enabled") as autoscaling,
        patch.object(gpu_shuffle_startup.ray, "cluster_resources") as cluster,
    ):
        capacity = gpu_shuffle_startup._known_capacity(_options())

    assert capacity is None
    autoscaling.assert_not_called()
    cluster.assert_not_called()


def test_known_capacity_uses_live_cluster_and_clamps_exclusions_at_zero():
    options = _options(excluded=ExecutionResources(cpu=3, gpu=0.25))
    with (
        patch.object(
            gpu_shuffle_startup.ray._private.state.state,
            "get_max_resources_from_cluster_config",
            return_value={},
        ),
        patch.object(gpu_shuffle_startup, "is_autoscaling_enabled", return_value=False),
        patch.object(
            gpu_shuffle_startup.ray,
            "cluster_resources",
            return_value={"CPU": 2, "GPU": 1},
        ),
    ):
        capacity = gpu_shuffle_startup._known_capacity(options)

    assert capacity == ExecutionResources(cpu=0, gpu=0.75)


def test_plan_supports_statically_safe_suffix():
    topology = _topology(
        _input(),
        _task({"num_gpus": 1}),
        _shuffle(),
        _actor(
            {"num_cpus": 1, "num_gpus": 1},
            wait_for_min_actors_s=0,
        ),
        _task({"scheduling_strategy": "SPREAD"}),
        _bare_operator(_PassthroughOperator, "Passthrough"),
    )

    assert _plan(topology, ExecutionResources(cpu=4, gpu=2)) is not None


@pytest.mark.parametrize(
    "suffix_op",
    [
        pytest.param(_task(dynamic=True), id="dynamic-remote-args"),
        pytest.param(
            _task({"resources": {"custom": 1}}),
            id="unsupported-remote-args",
        ),
        pytest.param(
            _actor({"num_gpus": 1}, wait_for_min_actors_s=1),
            id="blocking-actor",
        ),
        pytest.param(
            _bare_operator(_BlockingStartOperator, "Blocking"),
            id="custom-start",
        ),
    ],
)
def test_plan_rejects_unproven_suffix_startup(suffix_op):
    topology = _topology(
        _input(),
        _task({"num_gpus": 1}),
        _shuffle(),
        suffix_op,
    )

    assert _plan(topology, ExecutionResources(cpu=4, gpu=2)) is None


def test_plan_rejects_nonlinear_topology():
    input_op, producer, shuffle, topology = _supported_task_pipeline()
    producer._output_dependencies = []

    assert _plan(topology, ExecutionResources(cpu=2, gpu=2)) is None


@pytest.mark.parametrize("shuffle_count", [0, 2])
def test_plan_requires_exactly_one_shuffle(shuffle_count):
    ops = [_input(), _task({"num_gpus": 1})]
    ops.extend(_shuffle(name=f"Shuffle{index}") for index in range(shuffle_count))
    topology = _topology(*ops)

    assert _plan(topology, ExecutionResources(cpu=4, gpu=4)) is None


@pytest.mark.parametrize("customization", ["operator", "rank-pool"])
def test_plan_rejects_custom_gpu_shuffle(customization):
    topology = _topology(
        _input(),
        _task({"num_gpus": 1}),
        _shuffle(
            custom_pool=customization == "rank-pool",
            custom_operator=customization == "operator",
        ),
    )

    assert _plan(topology, ExecutionResources(cpu=2, gpu=2)) is None


def test_plan_rejects_execution_label_selector():
    _, _, _, topology = _supported_task_pipeline()
    options = _options(label_selector={"rack": "gpu"})

    assert (
        _plan(
            topology,
            ExecutionResources(cpu=2, gpu=2),
            options=options,
        )
        is None
    )


def test_plan_rejects_non_map_prefix_operator():
    topology = _topology(
        _input(),
        _bare_operator(_PassthroughOperator, "Passthrough"),
        _shuffle(),
    )

    assert _plan(topology, ExecutionResources(cpu=2, gpu=2)) is None


def test_plan_rejects_custom_input_operator():
    topology = _topology(
        _bare_operator(_CustomInputDataBuffer, "CustomInput"),
        _task({"num_gpus": 1}),
        _shuffle(),
    )

    assert _plan(topology, ExecutionResources(cpu=2, gpu=2)) is None


@pytest.mark.parametrize(
    "producer",
    [
        pytest.param(_task(dynamic=True), id="dynamic"),
        pytest.param(
            _task({"placement_group": object()}),
            id="unsupported-placement",
        ),
    ],
)
def test_plan_rejects_unproven_upstream_options(producer):
    topology = _topology(_input(), producer, _shuffle())

    assert _plan(topology, ExecutionResources(cpu=2, gpu=2)) is None


@pytest.mark.parametrize("sizes", [(1, 2, 2), (1, 1, 2)])
def test_plan_rejects_each_elastic_actor_pool_shape(sizes):
    topology = _topology(
        _input(),
        _actor({"num_gpus": 1}, sizes=sizes),
        _shuffle(),
    )

    assert _plan(topology, ExecutionResources(cpu=4, gpu=4)) is None


def test_plan_rejects_unknown_map_subclass():
    topology = _topology(
        _input(),
        _other_map({"num_gpus": 1}),
        _shuffle(),
    )

    assert _plan(topology, ExecutionResources(cpu=2, gpu=2)) is None


def test_plan_keeps_stock_startup_for_cpu_only_task():
    topology = _topology(
        _input(),
        _task({"num_cpus": 1}),
        _shuffle(),
    )

    assert _plan(topology, ExecutionResources(cpu=2, gpu=2)) is None


def test_plan_keeps_stock_startup_for_materialized_input():
    topology = _topology(_input(), _shuffle())

    assert _plan(topology, ExecutionResources(cpu=2, gpu=2)) is None


def test_plan_rejects_mixed_cpu_only_persistent_actor():
    topology = _topology(
        _input(),
        _task({"num_gpus": 1}),
        _actor({"num_cpus": 1}),
        _shuffle(),
    )

    assert _plan(topology, ExecutionResources(cpu=3, gpu=2)) is None


def test_plan_ignores_fixed_zero_resource_actor():
    input_op = _input()
    zero_resource_actor = _actor({})
    producer = _task({"num_cpus": 1, "num_gpus": 1})
    shuffle = _shuffle()
    topology = _topology(
        input_op,
        zero_resource_actor,
        producer,
        shuffle,
    )

    plan = _plan(topology, ExecutionResources(cpu=2, gpu=2))

    assert plan is not None
    assert next(reversed(plan[0])) is producer


def test_plan_builds_boundary_for_fixed_fractional_gpu_actor_pool():
    input_op = _input()
    producer = _actor(
        {"num_gpus": 0.5},
        sizes=(2, 2, 2),
    )
    shuffle = _shuffle(nranks=2)
    topology = _topology(input_op, producer, shuffle)

    plan = _plan(topology, ExecutionResources(cpu=4, gpu=2))

    assert plan is not None
    assert next(reversed(plan[0])) is producer


def test_plan_preserves_stock_overlap_for_four_plus_four_on_eight_gpus():
    topology = _topology(
        _input(),
        _actor({"num_gpus": 1}, sizes=(4, 4, 4)),
        _shuffle(nranks=4),
    )

    assert _plan(topology, ExecutionResources(cpu=8, gpu=8)) is None


def test_plan_rejects_unknown_capacity():
    _, _, _, topology = _supported_task_pipeline()

    assert _plan(topology, None) is None


def test_plan_rejects_upstream_floor_that_cannot_fit_independently():
    _, _, _, topology = _supported_task_pipeline(
        task_resources={"num_cpus": 3, "num_gpus": 1}
    )

    assert _plan(topology, ExecutionResources(cpu=2, gpu=2)) is None


def test_plan_rejects_shuffle_gang_that_cannot_fit_independently():
    _, _, _, topology = _supported_task_pipeline()

    assert _plan(topology, ExecutionResources(cpu=1, gpu=1)) is None


def test_plan_does_not_split_when_both_phases_fit_concurrently():
    _, _, _, topology = _supported_task_pipeline()

    assert _plan(topology, ExecutionResources(cpu=3, gpu=3)) is None


def test_plan_builds_boundary_for_task_producer():
    input_op, producer, shuffle, topology = _supported_task_pipeline()

    plan = _plan(topology, ExecutionResources(cpu=2, gpu=2))

    assert plan is not None
    assert plan[0] == (input_op, producer)
    assert plan[1] == (shuffle,)
    assert next(reversed(plan[0])) is producer


def test_plan_sums_fixed_gpu_actors_and_uses_maximum_transient_task():
    input_op = _input()
    actor = _actor(
        {"num_cpus": 1, "num_gpus": 1},
        sizes=(1, 1, 1),
    )
    small_task = _task(
        {"num_cpus": 1, "num_gpus": 1},
        name="SmallTask",
    )
    large_task = _task(
        {"num_cpus": 2, "num_gpus": 2},
        name="LargeTask",
    )
    shuffle = _shuffle(nranks=2)
    topology = _topology(
        input_op,
        actor,
        small_task,
        large_task,
        shuffle,
    )

    plan = _plan(topology, ExecutionResources(cpu=3, gpu=3))

    assert plan is not None
    assert next(reversed(plan[0])) is large_task


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
