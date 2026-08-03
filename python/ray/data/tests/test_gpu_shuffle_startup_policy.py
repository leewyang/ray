import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ray.data._internal.execution import gpu_shuffle_startup_policy
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
        raise AssertionError("The startup policy must not start operators.")


class _OtherMapOperator(MapOperator):
    def _input_queues(self):
        return []

    def _output_queues(self):
        return []

    def _try_schedule_task(self, refs, strict):
        raise AssertionError("The startup policy must not schedule tasks.")

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


def _shuffle(
    *,
    nranks=2,
    custom_pool=False,
    custom_operator=False,
    injected_pool=False,
    name="GPUShuffle",
):
    operator_cls = _CustomGPUShuffleOperator if custom_operator else GPUShuffleOperator
    op = _bare_operator(operator_cls, name)
    pool_cls = _CustomRankPool if custom_pool else GPURankPool
    op._rank_pool = object.__new__(pool_cls)
    op._rank_pool._nranks = nranks
    op._uses_default_rank_pool = not custom_pool and not injected_pool
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


def _build_segment_topologies(topology, capacity, *, options=None):
    if options is None:
        options = _options()
    with patch.object(
        gpu_shuffle_startup_policy,
        "_effective_capacity",
        return_value=capacity,
    ):
        return gpu_shuffle_startup_policy.build_gpu_shuffle_segment_topologies(
            topology, options
        )


def _uses_stock_startup(topology, capacity, *, options=None):
    segments = _build_segment_topologies(
        topology,
        capacity,
        options=options,
    )
    return len(segments) == 1 and segments[0] is topology


def _supported_task_pipeline(*, task_resources=None, nranks=2):
    if task_resources is None:
        task_resources = {"num_cpus": 1, "num_gpus": 1}
    input_op = _input()
    producer = _task(task_resources)
    shuffle = _shuffle(nranks=nranks)
    return input_op, producer, shuffle, _topology(input_op, producer, shuffle)


@pytest.mark.parametrize(
    ("operator_factory", "remote_args"),
    [
        pytest.param(
            _task,
            {"fallback_strategy": [{"label_selector": {"rack": "a"}}]},
            id="fallback-strategy",
        ),
        pytest.param(_task, {"label_selector": {"rack": "a"}}, id="label-selector"),
        pytest.param(_task, {"placement_group": object()}, id="placement-group"),
        pytest.param(
            _task, {"placement_group_bundle_index": 0}, id="placement-group-bundle"
        ),
        pytest.param(
            _task,
            {"placement_group_capture_child_tasks": False},
            id="capture-child-tasks",
        ),
        pytest.param(_actor, {"get_if_exists": True}, id="actor-reuse"),
        pytest.param(_actor, {"lifetime": "detached"}, id="actor-lifetime"),
        pytest.param(_actor, {"name": "worker"}, id="actor-name"),
        pytest.param(_actor, {"namespace": "data"}, id="actor-namespace"),
        pytest.param(_task, {"scheduling_strategy": None}, id="ambient-strategy"),
        pytest.param(
            _task,
            {"scheduling_strategy": "NODE_AFFINITY"},
            id="node-affinity",
        ),
        pytest.param(_task, {"accelerator_type": "A100"}, id="accelerator"),
        pytest.param(_task, {"memory": 1}, id="memory"),
        pytest.param(_task, {"object_store_memory": 1}, id="object-store-memory"),
        pytest.param(_task, {"resources": {"custom": 1}}, id="custom-resource"),
        pytest.param(_task, {"resources": {"CPU": 1}}, id="resource-cpu"),
    ],
)
def test_static_request_rejects_unmodeled_remote_options(operator_factory, remote_args):
    assert (
        gpu_shuffle_startup_policy._get_supported_cpu_gpu_request(
            operator_factory(), remote_args
        )
        is None
    )


@pytest.mark.parametrize(
    "attribute",
    ["scheduling_strategy", "scheduling_strategy_large_args"],
)
def test_remote_semantics_rejects_context_placement(attribute):
    op = _task()
    setattr(op.data_context, attribute, None)

    assert gpu_shuffle_startup_policy._get_supported_cpu_gpu_request(op, {}) is None


@pytest.mark.parametrize(
    ("remote_args", "is_actor", "expected_cpu", "expected_gpu"),
    [
        ({"num_cpus": 2, "num_gpus": 3}, True, 2, 3),
        ({"num_gpus": 1}, True, 1, 1),
        ({"num_gpus": 0}, True, 1, 0),
        ({}, True, 0, 0),
        ({}, False, 1, 0),
        ({"resources": {}}, False, 1, 0),
        ({"num_cpus": 0, "num_gpus": 2}, False, 0, 2),
        ({"name": "gpu-task"}, False, 1, 0),
        ({"scheduling_strategy": "DEFAULT"}, False, 1, 0),
        ({"scheduling_strategy": "SPREAD"}, False, 1, 0),
    ],
)
def test_static_request_accepts_supported_cpu_gpu_options(
    remote_args, is_actor, expected_cpu, expected_gpu
):
    resources = gpu_shuffle_startup_policy._get_supported_cpu_gpu_request(
        _actor() if is_actor else _task(), remote_args
    )

    assert resources is not None
    assert resources.cpu == expected_cpu
    assert resources.gpu == expected_gpu


def test_effective_capacity_prefers_configured_cluster_maximum():
    options = _options(
        limits=ExecutionResources(cpu=6, gpu=10),
        excluded=ExecutionResources(cpu=2, gpu=1),
    )
    with (
        patch.object(
            gpu_shuffle_startup_policy.ray._private.state.state,
            "get_max_resources_from_cluster_config",
            return_value={"CPU": 8, "GPU": 4},
        ),
        patch.object(
            gpu_shuffle_startup_policy, "is_autoscaling_enabled"
        ) as autoscaling,
        patch.object(gpu_shuffle_startup_policy.ray, "cluster_resources") as cluster,
    ):
        capacity = gpu_shuffle_startup_policy._effective_capacity(options)

    assert capacity == ExecutionResources(cpu=4, gpu=3)
    autoscaling.assert_not_called()
    cluster.assert_not_called()


def test_effective_capacity_is_unknown_for_unbounded_autoscaling():
    with (
        patch.object(
            gpu_shuffle_startup_policy.ray._private.state.state,
            "get_max_resources_from_cluster_config",
            return_value={},
        ),
        patch.object(
            gpu_shuffle_startup_policy, "is_autoscaling_enabled", return_value=True
        ),
        patch.object(gpu_shuffle_startup_policy.ray, "cluster_resources") as cluster,
    ):
        capacity = gpu_shuffle_startup_policy._effective_capacity(_options())

    assert capacity is None
    cluster.assert_not_called()


def test_effective_capacity_rejects_unbounded_configured_maximum():
    with (
        patch.object(
            gpu_shuffle_startup_policy.ray._private.state.state,
            "get_max_resources_from_cluster_config",
            return_value={"CPU": sys.maxsize, "GPU": 4},
        ),
        patch.object(
            gpu_shuffle_startup_policy, "is_autoscaling_enabled"
        ) as autoscaling,
        patch.object(gpu_shuffle_startup_policy.ray, "cluster_resources") as cluster,
    ):
        capacity = gpu_shuffle_startup_policy._effective_capacity(_options())

    assert capacity is None
    autoscaling.assert_not_called()
    cluster.assert_not_called()


def test_effective_capacity_uses_live_cluster_and_clamps_exclusions_at_zero():
    options = _options(excluded=ExecutionResources(cpu=3, gpu=0.25))
    with (
        patch.object(
            gpu_shuffle_startup_policy.ray._private.state.state,
            "get_max_resources_from_cluster_config",
            return_value={},
        ),
        patch.object(
            gpu_shuffle_startup_policy, "is_autoscaling_enabled", return_value=False
        ),
        patch.object(
            gpu_shuffle_startup_policy.ray,
            "cluster_resources",
            return_value={"CPU": 2, "GPU": 1},
        ),
    ):
        capacity = gpu_shuffle_startup_policy._effective_capacity(options)

    assert capacity == ExecutionResources(cpu=0, gpu=0.75)


def test_policy_supports_statically_safe_suffix():
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

    assert (
        len(_build_segment_topologies(topology, ExecutionResources(cpu=4, gpu=2))) == 2
    )


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
def test_policy_rejects_unproven_suffix_startup(suffix_op):
    topology = _topology(
        _input(),
        _task({"num_gpus": 1}),
        _shuffle(),
        suffix_op,
    )

    assert _uses_stock_startup(topology, ExecutionResources(cpu=4, gpu=2))


def test_policy_rejects_nonlinear_topology():
    input_op, producer, shuffle, topology = _supported_task_pipeline()
    producer._output_dependencies = []

    assert _uses_stock_startup(topology, ExecutionResources(cpu=2, gpu=2))


def test_policy_requires_a_shuffle():
    ops = [_input(), _task({"num_gpus": 1})]
    topology = _topology(*ops)

    assert _uses_stock_startup(topology, ExecutionResources(cpu=4, gpu=4))


def test_policy_builds_boundaries_for_cumulative_shuffle_pressure():
    input_op = _input()
    producer = _task({"num_gpus": 1}, name="Producer")
    first_shuffle = _shuffle(nranks=2, name="FirstShuffle")
    middle_task = _task({"num_gpus": 1}, name="MiddleTask")
    second_shuffle = _shuffle(nranks=2, name="SecondShuffle")
    topology = _topology(
        input_op,
        producer,
        first_shuffle,
        middle_task,
        second_shuffle,
    )

    segments = _build_segment_topologies(topology, ExecutionResources(cpu=3, gpu=3))

    assert [list(segment) for segment in segments] == [
        [input_op, producer],
        [first_shuffle, middle_task],
        [second_shuffle],
    ]
    assert _uses_stock_startup(topology, ExecutionResources(cpu=5, gpu=5))


def test_policy_supports_three_shuffles_with_cpu_and_empty_phases():
    input_op = _input()
    first_shuffle = _shuffle(nranks=1, name="FirstShuffle")
    cpu_task = _task({"num_cpus": 1}, name="CpuTask")
    second_shuffle = _shuffle(nranks=2, name="SecondShuffle")
    third_shuffle = _shuffle(nranks=3, name="ThirdShuffle")
    topology = _topology(
        input_op,
        first_shuffle,
        cpu_task,
        second_shuffle,
        third_shuffle,
    )

    segments = _build_segment_topologies(topology, ExecutionResources(cpu=3, gpu=3))

    assert [list(segment) for segment in segments] == [
        [input_op],
        [first_shuffle, cpu_task],
        [second_shuffle],
        [third_shuffle],
    ]


def test_policy_accounts_for_fixed_actors_across_shuffle_phases():
    input_op = _input()
    first_actor = _actor({"num_cpus": 0, "num_gpus": 1})
    first_shuffle = _shuffle(nranks=2, name="FirstShuffle")
    second_actor = _actor({"num_cpus": 0, "num_gpus": 1})
    second_shuffle = _shuffle(nranks=2, name="SecondShuffle")
    topology = _topology(
        input_op,
        first_actor,
        first_shuffle,
        second_actor,
        second_shuffle,
    )

    segments = _build_segment_topologies(topology, ExecutionResources(cpu=4, gpu=5))

    assert [list(segment) for segment in segments] == [
        [input_op, first_actor],
        [first_shuffle, second_actor],
        [second_shuffle],
    ]


def test_policy_rejects_unsupported_middle_operator():
    topology = _topology(
        _input(),
        _task({"num_gpus": 1}),
        _shuffle(name="FirstShuffle"),
        _bare_operator(_BlockingStartOperator, "Blocking"),
        _shuffle(name="SecondShuffle"),
    )

    assert _uses_stock_startup(topology, ExecutionResources(cpu=3, gpu=3))


@pytest.mark.parametrize(
    ("middle_task_args", "second_rank_count"),
    [({"num_gpus": 4}, 2), ({"num_gpus": 1}, 4)],
)
def test_policy_rejects_multi_shuffle_segment_that_cannot_fit(
    middle_task_args, second_rank_count
):
    topology = _topology(
        _input(),
        _task({"num_gpus": 1}),
        _shuffle(nranks=2, name="FirstShuffle"),
        _task(middle_task_args),
        _shuffle(nranks=second_rank_count, name="SecondShuffle"),
    )

    assert _uses_stock_startup(topology, ExecutionResources(cpu=3, gpu=3))


@pytest.mark.parametrize(
    "customization", ["operator", "rank-pool", "injected-rank-pool"]
)
def test_policy_rejects_custom_gpu_shuffle(customization):
    topology = _topology(
        _input(),
        _task({"num_gpus": 1}),
        _shuffle(
            custom_pool=customization == "rank-pool",
            custom_operator=customization == "operator",
            injected_pool=customization == "injected-rank-pool",
        ),
    )

    assert _uses_stock_startup(topology, ExecutionResources(cpu=2, gpu=2))


def test_policy_rejects_execution_label_selector():
    _, _, _, topology = _supported_task_pipeline()
    options = _options(label_selector={"rack": "gpu"})

    assert _uses_stock_startup(
        topology,
        ExecutionResources(cpu=2, gpu=2),
        options=options,
    )


def test_policy_rejects_non_map_prefix_operator():
    topology = _topology(
        _input(),
        _bare_operator(_PassthroughOperator, "Passthrough"),
        _shuffle(),
    )

    assert _uses_stock_startup(topology, ExecutionResources(cpu=2, gpu=2))


def test_policy_rejects_custom_input_operator():
    topology = _topology(
        _bare_operator(_CustomInputDataBuffer, "CustomInput"),
        _task({"num_gpus": 1}),
        _shuffle(),
    )

    assert _uses_stock_startup(topology, ExecutionResources(cpu=2, gpu=2))


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
def test_policy_rejects_unproven_upstream_options(producer):
    topology = _topology(_input(), producer, _shuffle())

    assert _uses_stock_startup(topology, ExecutionResources(cpu=2, gpu=2))


@pytest.mark.parametrize("sizes", [(1, 2, 2), (1, 1, 2)])
def test_policy_rejects_each_elastic_actor_pool_shape(sizes):
    topology = _topology(
        _input(),
        _actor({"num_gpus": 1}, sizes=sizes),
        _shuffle(),
    )

    assert _uses_stock_startup(topology, ExecutionResources(cpu=4, gpu=4))


def test_policy_rejects_unknown_map_subclass():
    topology = _topology(
        _input(),
        _other_map({"num_gpus": 1}),
        _shuffle(),
    )

    assert _uses_stock_startup(topology, ExecutionResources(cpu=2, gpu=2))


def test_policy_keeps_stock_startup_for_cpu_only_task():
    topology = _topology(
        _input(),
        _task({"num_cpus": 1}),
        _shuffle(),
    )

    assert _uses_stock_startup(topology, ExecutionResources(cpu=2, gpu=2))


def test_policy_keeps_stock_startup_for_materialized_input():
    topology = _topology(_input(), _shuffle())

    assert _uses_stock_startup(topology, ExecutionResources(cpu=2, gpu=2))


@pytest.mark.parametrize(
    "actor_args",
    [{"num_cpus": 1}, {"num_cpus": None, "num_gpus": None}],
)
def test_policy_rejects_mixed_non_gpu_actor(actor_args):
    topology = _topology(
        _input(),
        _task({"num_gpus": 1}),
        _actor(actor_args),
        _shuffle(),
    )

    assert _uses_stock_startup(topology, ExecutionResources(cpu=3, gpu=2))


def test_policy_builds_boundary_for_fixed_fractional_gpu_actor_pool():
    input_op = _input()
    producer = _actor(
        {"num_gpus": 0.5},
        sizes=(2, 2, 2),
    )
    shuffle = _shuffle(nranks=2)
    topology = _topology(input_op, producer, shuffle)

    segments = _build_segment_topologies(topology, ExecutionResources(cpu=4, gpu=2))

    assert list(segments[0]) == [input_op, producer]


def test_policy_preserves_stock_overlap_for_four_plus_four_on_eight_gpus():
    topology = _topology(
        _input(),
        _actor({"num_gpus": 1}, sizes=(4, 4, 4)),
        _shuffle(nranks=4),
    )

    assert _uses_stock_startup(topology, ExecutionResources(cpu=8, gpu=8))


def test_policy_rejects_unknown_capacity():
    _, _, _, topology = _supported_task_pipeline()

    assert _uses_stock_startup(topology, None)


def test_policy_rejects_upstream_progress_that_cannot_fit_independently():
    _, _, _, topology = _supported_task_pipeline(
        task_resources={"num_cpus": 3, "num_gpus": 1}
    )

    assert _uses_stock_startup(topology, ExecutionResources(cpu=2, gpu=2))


def test_policy_rejects_shuffle_gang_that_cannot_fit_independently():
    _, _, _, topology = _supported_task_pipeline()

    assert _uses_stock_startup(topology, ExecutionResources(cpu=1, gpu=1))


def test_policy_does_not_split_when_both_phases_fit_concurrently():
    _, _, _, topology = _supported_task_pipeline()

    assert _uses_stock_startup(topology, ExecutionResources(cpu=3, gpu=3))


def test_policy_builds_boundary_for_task_producer():
    input_op, producer, shuffle, topology = _supported_task_pipeline()

    segments = _build_segment_topologies(topology, ExecutionResources(cpu=2, gpu=2))

    assert [list(segment) for segment in segments] == [
        [input_op, producer],
        [shuffle],
    ]


def test_policy_sums_fixed_gpu_actors_and_uses_largest_task_request():
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

    segments = _build_segment_topologies(topology, ExecutionResources(cpu=3, gpu=3))

    assert list(segments[0]) == [input_op, actor, small_task, large_task]


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
