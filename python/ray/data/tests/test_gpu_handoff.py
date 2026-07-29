from unittest.mock import MagicMock, patch

import pytest

import ray
from ray.data._internal.compute import ActorPoolStrategy, TaskPoolStrategy
from ray.data._internal.execution.block_ref_counter import BlockRefCounter
from ray.data._internal.execution.gpu_handoff import (
    _OwnerState,
    create_gpu_handoff_policy,
)
from ray.data._internal.execution.interfaces import (
    ExecutionOptions,
    ExecutionResources,
    PhysicalOperator,
)
from ray.data._internal.execution.operators.hash_shuffle import (
    HashShufflingOperatorBase,
)
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.streaming_executor_state import (
    build_streaming_topology,
)
from ray.data._internal.gpu_shuffle.hash_shuffle import GPUShuffleOperator
from ray.data.context import DataContext


class _CPUHashShuffleStub(HashShufflingOperatorBase):
    def _get_operator_num_cpus_override(self):
        return 1


def _noop_counter():
    return BlockRefCounter(add_object_out_of_scope_callback=lambda *_: True)


def _actor(
    input_op,
    data_context,
    *,
    size=None,
    min_size=None,
    max_size=None,
    initial_size=None,
    ray_remote_args=None,
    ray_remote_args_fn=None,
):
    return MapOperator.create(
        map_transformer=MagicMock(),
        input_op=input_op,
        data_context=data_context,
        compute_strategy=ActorPoolStrategy(
            size=size,
            min_size=min_size,
            max_size=max_size,
            initial_size=initial_size,
        ),
        ray_remote_args=ray_remote_args or {"num_gpus": 1},
        ray_remote_args_fn=ray_remote_args_fn,
    )


def _task(
    input_op,
    data_context,
    *,
    ray_remote_args=None,
    ray_remote_args_fn=None,
):
    return MapOperator.create(
        map_transformer=MagicMock(),
        input_op=input_op,
        data_context=data_context,
        compute_strategy=TaskPoolStrategy(),
        ray_remote_args=ray_remote_args,
        ray_remote_args_fn=ray_remote_args_fn,
    )


def _rank(input_op, data_context, *, nranks=1):
    return GPUShuffleOperator(
        input_op,
        data_context,
        key_columns=("id",),
        num_partitions=nranks,
        nranks=nranks,
    )


def _topology(root, options):
    return build_streaming_topology(
        root,
        options,
        _noop_counter(),
        start_operators=False,
    )


def _options(cpu, gpu):
    return ExecutionOptions(
        resource_limits=ExecutionResources.for_limits(cpu=cpu, gpu=gpu)
    )


def test_adjacent_owners_coadmit_only_when_complete_claims_fit():
    ctx = DataContext()
    source = InputDataBuffer(ctx, input_data=[])
    first = _actor(source, ctx, size=1)
    second = _actor(first, ctx, size=1)
    third = _actor(second, ctx, size=1)

    policy = create_gpu_handoff_policy(_topology(third, _options(3, 3)), _options(3, 3))
    assert policy is not None
    assert list(policy._states.values()) == [_OwnerState.ADMITTED] * 3

    source = InputDataBuffer(ctx, input_data=[])
    first = _actor(source, ctx, size=1)
    second = _actor(first, ctx, size=1)
    third = _actor(second, ctx, size=1)
    policy = create_gpu_handoff_policy(_topology(third, _options(2, 2)), _options(2, 2))
    assert policy is not None
    assert list(policy._states.values()) == [
        _OwnerState.ADMITTED,
        _OwnerState.ADMITTED,
        _OwnerState.WAITING,
    ]


def test_fractional_claims_truncate_each_worker_at_ray_resource_precision():
    ctx = DataContext()
    options = _options(3, 0.3)
    source = InputDataBuffer(ctx, input_data=[])
    first = _actor(source, ctx, size=1, ray_remote_args={"num_gpus": 0.10009})
    second = _actor(first, ctx, size=1, ray_remote_args={"num_gpus": 0.10009})
    third = _actor(second, ctx, size=1, ray_remote_args={"num_gpus": 0.10009})

    policy = create_gpu_handoff_policy(_topology(third, options), options)
    assert policy is not None
    assert list(policy._states.values()) == [_OwnerState.ADMITTED] * 3

    options = _options(2, 0.99996)
    source = InputDataBuffer(ctx, input_data=[])
    first = _actor(source, ctx, size=1, ray_remote_args={"num_gpus": 0.5})
    second = _actor(first, ctx, size=1, ray_remote_args={"num_gpus": 0.5})
    policy = create_gpu_handoff_policy(_topology(second, options), options)
    assert policy is not None
    assert list(policy._states.values()) == [
        _OwnerState.ADMITTED,
        _OwnerState.WAITING,
    ]


def test_cpu_segment_is_an_initial_barrier_and_drains_after_release():
    ctx = DataContext()
    options = _options(4, 2)
    source = InputDataBuffer(ctx, input_data=[])
    first = _actor(source, ctx, size=1)
    bridge = _task(first, ctx)
    second = _actor(bridge, ctx, size=1)
    policy = create_gpu_handoff_policy(_topology(second, options), options)
    assert policy is not None

    assert policy._states[first] is _OwnerState.ADMITTED
    assert policy._states[second] is _OwnerState.WAITING

    first.has_execution_finished = MagicMock(return_value=True)
    bridge.has_execution_finished = MagicMock(return_value=False)
    bridge.has_completed = MagicMock(
        side_effect=AssertionError("handoff must not require has_completed()")
    )
    second._admit_gpu_handoff = MagicMock()

    policy.update()
    assert policy._states[first] is _OwnerState.RELEASED
    assert policy._states[second] is _OwnerState.WAITING
    assert policy.should_drain_output(first)
    assert policy.should_drain_output(bridge)

    bridge.has_execution_finished.return_value = True
    policy.update()
    assert policy._states[second] is _OwnerState.ADMITTED
    second._admit_gpu_handoff.assert_called_once_with()
    assert not policy.should_drain_output(first)
    assert not policy.should_drain_output(bridge)
    bridge.has_completed.assert_not_called()


def test_direct_nonfitting_owner_admits_after_predecessor_release():
    ctx = DataContext()
    options = _options(1, 1)
    source = InputDataBuffer(ctx, input_data=[])
    first = _actor(source, ctx, size=1)
    second = _actor(first, ctx, size=1)
    policy = create_gpu_handoff_policy(_topology(second, options), options)
    assert policy is not None
    assert policy._states[second] is _OwnerState.WAITING

    first.has_execution_finished = MagicMock(return_value=True)
    second._admit_gpu_handoff = MagicMock()
    policy.update()
    assert policy._states[first] is _OwnerState.RELEASED
    assert policy._states[second] is _OwnerState.ADMITTED
    second._admit_gpu_handoff.assert_called_once_with()


def test_releasing_claim_stays_held_until_teardown_succeeds():
    ctx = DataContext()
    options = _options(1, 1)
    source = InputDataBuffer(ctx, input_data=[])
    first = _actor(source, ctx, size=1)
    second = _actor(first, ctx, size=1)
    policy = create_gpu_handoff_policy(_topology(second, options), options)
    assert policy is not None
    first.has_execution_finished = MagicMock(return_value=True)
    first._release_gpu_handoff_resources = MagicMock(side_effect=[False, True])
    second._admit_gpu_handoff = MagicMock()

    policy.update()
    assert policy._states[first] is _OwnerState.RELEASING
    assert policy._states[second] is _OwnerState.WAITING
    second._admit_gpu_handoff.assert_not_called()

    policy.update()
    assert policy._states[first] is _OwnerState.RELEASED
    assert policy._states[second] is _OwnerState.ADMITTED
    second._admit_gpu_handoff.assert_called_once_with()


def test_elastic_max_is_normalized_without_clipping_initial_size():
    ctx = DataContext()
    options = _options(4, 2)
    source = InputDataBuffer(ctx, input_data=[])
    first = _actor(source, ctx, min_size=1, max_size=10, initial_size=2)
    second = _actor(first, ctx, size=1)

    policy = create_gpu_handoff_policy(_topology(second, options), options)
    assert policy is not None
    first_owner = policy._owners[0]
    assert first_owner.effective_max == 2
    assert first_owner.gpu_units == 2 * ray._raylet.RESOURCE_UNIT_SCALING

    source = InputDataBuffer(ctx, input_data=[])
    first = _actor(source, ctx, min_size=1)
    second = _actor(first, ctx, size=1)
    policy = create_gpu_handoff_policy(_topology(second, options), options)
    assert policy is not None
    assert policy._owners[0].effective_max == 2


def test_elastic_initial_size_is_a_mandatory_explicit_claim():
    ctx = DataContext()
    options = _options(4, 1)
    source = InputDataBuffer(ctx, input_data=[])
    first = _actor(source, ctx, min_size=1, max_size=10, initial_size=2)
    second = _actor(first, ctx, size=1)

    with pytest.raises(ValueError, match="explicit Ray Data execution capacity"):
        create_gpu_handoff_policy(_topology(second, options), options)


def test_explicit_mandatory_shortfall_errors_but_snapshot_shortfall_falls_back():
    ctx = DataContext()
    source = InputDataBuffer(ctx, input_data=[])
    first = _actor(source, ctx, size=2)
    second = _actor(first, ctx, size=1)

    explicit_options = _options(4, 1)
    with pytest.raises(ValueError, match="explicit Ray Data execution capacity"):
        create_gpu_handoff_policy(_topology(second, explicit_options), explicit_options)

    snapshot_options = ExecutionOptions()
    with patch(
        "ray.data._internal.execution.gpu_handoff.ray.cluster_resources",
        return_value={"CPU": 4, "GPU": 1},
    ) as snapshot:
        assert (
            create_gpu_handoff_policy(
                _topology(second, snapshot_options), snapshot_options
            )
            is None
        )
    snapshot.assert_called_once_with()


def test_explicit_shortfall_wins_over_an_earlier_snapshot_shortfall():
    ctx = DataContext()
    source = InputDataBuffer(ctx, input_data=[])
    first = _actor(
        source,
        ctx,
        size=1,
        ray_remote_args={"num_cpus": 1, "num_gpus": 2},
    )
    second = _actor(
        first,
        ctx,
        size=1,
        ray_remote_args={"num_cpus": 2, "num_gpus": 0.5},
    )
    options = ExecutionOptions(resource_limits=ExecutionResources.for_limits(cpu=1))

    with (
        patch(
            "ray.data._internal.execution.gpu_handoff.ray.cluster_resources",
            return_value={"CPU": 8, "GPU": 1},
        ),
        pytest.raises(ValueError, match="explicit Ray Data execution capacity"),
    ):
        create_gpu_handoff_policy(_topology(second, options), options)


def test_mixed_capacity_uses_one_snapshot_and_applies_exclusions():
    ctx = DataContext()
    source = InputDataBuffer(ctx, input_data=[])
    first = _actor(source, ctx, size=1)
    second = _actor(first, ctx, size=1)
    options = ExecutionOptions(
        resource_limits=ExecutionResources.for_limits(gpu=1.5),
        exclude_resources=ExecutionResources(cpu=1, gpu=0.5),
    )

    with patch(
        "ray.data._internal.execution.gpu_handoff.ray.cluster_resources",
        return_value={"CPU": 4, "GPU": 99},
    ) as snapshot:
        policy = create_gpu_handoff_policy(_topology(second, options), options)

    snapshot.assert_called_once_with()
    assert policy is not None
    assert policy._capacity.cpu_units == 3 * ray._raylet.RESOURCE_UNIT_SCALING
    assert policy._capacity.gpu_units == ray._raylet.RESOURCE_UNIT_SCALING
    assert list(policy._states.values()) == [
        _OwnerState.ADMITTED,
        _OwnerState.WAITING,
    ]


def test_explicitly_finished_waiting_owner_is_never_admitted():
    ctx = DataContext()
    options = _options(1, 1)
    source = InputDataBuffer(ctx, input_data=[])
    first = _actor(source, ctx, size=1)
    second = _actor(first, ctx, size=1)
    policy = create_gpu_handoff_policy(_topology(second, options), options)
    assert policy is not None
    second._admit_gpu_handoff = MagicMock()

    second._is_execution_marked_finished = True
    policy.update()

    assert policy._states[second] is _OwnerState.RELEASED
    second._admit_gpu_handoff.assert_not_called()


@pytest.mark.parametrize(
    "make_unsupported",
    [
        lambda first, ctx: _actor(first, ctx, size=1, ray_remote_args={"num_cpus": 1}),
        lambda first, ctx: _task(
            first, ctx, ray_remote_args_fn=lambda: {"num_cpus": 1}
        ),
        lambda first, ctx: _task(first, ctx, ray_remote_args={"num_gpus": 1}),
        lambda first, ctx: _task(
            first,
            ctx,
            ray_remote_args={"num_cpus": 1, "label_selector": {"zone": "a"}},
        ),
    ],
)
def test_unsupported_intermediate_falls_back_before_capacity_lookup(
    make_unsupported,
):
    ctx = DataContext()
    options = ExecutionOptions()
    source = InputDataBuffer(ctx, input_data=[])
    first = _actor(source, ctx, size=1)
    intermediate = make_unsupported(first, ctx)
    second = _actor(intermediate, ctx, size=1)

    with patch(
        "ray.data._internal.execution.gpu_handoff.ray.cluster_resources"
    ) as snapshot:
        assert create_gpu_handoff_policy(_topology(second, options), options) is None
    snapshot.assert_not_called()


def test_custom_actor_resource_and_label_selector_fall_back_before_snapshot():
    ctx = DataContext()
    source = InputDataBuffer(ctx, input_data=[])
    first = _actor(
        source,
        ctx,
        size=1,
        ray_remote_args={"num_gpus": 1, "resources": {"custom": 0.25}},
    )
    second = _actor(first, ctx, size=1)

    options = ExecutionOptions()
    with patch(
        "ray.data._internal.execution.gpu_handoff.ray.cluster_resources"
    ) as snapshot:
        assert create_gpu_handoff_policy(_topology(second, options), options) is None
    snapshot.assert_not_called()
    options.label_selector = {"accelerator": "L4"}
    source = InputDataBuffer(ctx, input_data=[])
    first = _actor(source, ctx, size=1)
    second = _actor(first, ctx, size=1)
    with patch(
        "ray.data._internal.execution.gpu_handoff.ray.cluster_resources"
    ) as snapshot:
        assert create_gpu_handoff_policy(_topology(second, options), options) is None
    snapshot.assert_not_called()


def test_cpu_hash_shuffle_actor_bridge_falls_back_before_snapshot():
    ctx = DataContext()
    options = ExecutionOptions()
    source = InputDataBuffer(ctx, input_data=[])
    first = _actor(source, ctx, size=1)
    bridge = object.__new__(_CPUHashShuffleStub)
    PhysicalOperator.__init__(bridge, "CPUHashShuffle", [first], ctx)
    second = _actor(bridge, ctx, size=1)

    with patch(
        "ray.data._internal.execution.gpu_handoff.ray.cluster_resources"
    ) as snapshot:
        assert create_gpu_handoff_policy(_topology(second, options), options) is None
    snapshot.assert_not_called()


def test_fanout_falls_back_before_capacity_lookup():
    ctx = DataContext()
    options = ExecutionOptions()
    source = InputDataBuffer(ctx, input_data=[])
    first = _actor(source, ctx, size=1)
    second = _actor(first, ctx, size=1)
    _task(first, ctx)

    with patch(
        "ray.data._internal.execution.gpu_handoff.ray.cluster_resources"
    ) as snapshot:
        assert create_gpu_handoff_policy(_topology(second, options), options) is None
    snapshot.assert_not_called()


def test_fanin_falls_back_before_capacity_lookup():
    ctx = DataContext()
    options = ExecutionOptions()
    first_source = InputDataBuffer(ctx, input_data=[])
    first = _actor(first_source, ctx, size=1)
    second_source = InputDataBuffer(ctx, input_data=[])
    fanin = PhysicalOperator("FanIn", [first, second_source], ctx)
    second = _actor(fanin, ctx, size=1)

    with patch(
        "ray.data._internal.execution.gpu_handoff.ray.cluster_resources"
    ) as snapshot:
        assert create_gpu_handoff_policy(_topology(second, options), options) is None
    snapshot.assert_not_called()


def test_rank_owners_are_claimed_and_serialized():
    ctx = DataContext()
    options = _options(1, 1)
    source = InputDataBuffer(ctx, input_data=[])
    first = _rank(source, ctx)
    second = _rank(first, ctx)

    policy = create_gpu_handoff_policy(_topology(second, options), options)
    assert policy is not None
    assert all(owner.effective_max is None for owner in policy._owners)
    assert list(policy._states.values()) == [
        _OwnerState.ADMITTED,
        _OwnerState.WAITING,
    ]
    assert first._gpu_handoff_managed
    assert second._gpu_handoff_managed
    assert not first._rank_pool.owns_resources
    assert not second._rank_pool.owns_resources


@pytest.mark.parametrize("unsupported", ["marker", "labels"])
def test_unsupported_rank_configuration_falls_back_before_snapshot(unsupported):
    ctx = DataContext()
    options = ExecutionOptions()
    source = InputDataBuffer(ctx, input_data=[])
    first = _rank(source, ctx)
    second = _rank(first, ctx)
    if unsupported == "marker":
        first._rank_pool._gpu_handoff_compatible = False
    else:
        first._rank_pool._label_selector = {"accelerator": "L4"}

    with patch(
        "ray.data._internal.execution.gpu_handoff.ray.cluster_resources"
    ) as snapshot:
        assert create_gpu_handoff_policy(_topology(second, options), options) is None
    snapshot.assert_not_called()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
