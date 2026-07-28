from typing import TYPE_CHECKING, Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from ray.data._internal.compute import ActorPoolStrategy, ComputeStrategy
from ray.data._internal.execution.interfaces import (
    PhysicalOperator,
    ResourceAdmissionGrant,
    ResourceAdmissionSpec,
)
from ray.data._internal.execution.interfaces.execution_options import (
    ExecutionOptions,
    ExecutionResources,
)
from ray.data._internal.execution.operators.actor_pool_map_operator import (
    ActorPoolMapOperator,
)
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.join import JoinOperator
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.resource_admission_controller import (
    _ResourceAdmissionController,
    _validate_resource_admission_spec,
)
from ray.data._internal.logical.operators import JoinType
from ray.data.context import DataContext

if TYPE_CHECKING:
    from ray.data._internal.execution.streaming_executor_state import Topology


def _mock_map_op(
    input_op: PhysicalOperator,
    ray_remote_args: Optional[Dict[str, Any]] = None,
    compute_strategy: Optional[ComputeStrategy] = None,
    name="Map",
):
    return MapOperator.create(
        MagicMock(),
        input_op,
        DataContext.get_current(),
        ray_remote_args=ray_remote_args or {},
        compute_strategy=compute_strategy,
        name=name,
    )


def _mock_join_op(left_input_op, right_input_op):
    left_input_op._logical_operators = [MagicMock()]
    right_input_op._logical_operators = [MagicMock()]

    with patch(
        "ray.data._internal.execution.operators.hash_shuffle."
        "_get_total_cluster_resources"
    ) as mock:
        mock.return_value = ExecutionResources(cpu=1)
        op = JoinOperator(
            DataContext.get_current(),
            left_input_op,
            right_input_op,
            ("id",),
            ("id",),
            JoinType.INNER,
            num_partitions=1,
            partition_size_hint=1,
        )

    op.start = MagicMock(side_effect=lambda *_: None)
    return op


def _mock_resource_admission_op(
    *,
    inputs=(),
    completed=False,
    internal_demand=False,
    inputs_complete=False,
    cpu=1,
    gpu=1,
    memory=0,
    custom_resources=(),
    label_selector=(),
    fixed_gang=False,
    minimum_units=1,
    max_units=4,
):
    op = MagicMock()
    op.input_dependencies = list(inputs)
    op._inputs_complete = inputs_complete
    op.has_execution_finished = MagicMock(return_value=completed)
    op.can_release_resource_admission = MagicMock(return_value=not internal_demand)
    op.num_active_tasks = MagicMock(return_value=0)
    op.current_logical_usage = MagicMock(return_value=ExecutionResources.zero())
    op.pending_logical_usage = MagicMock(return_value=ExecutionResources.zero())
    unit_resources = ExecutionResources(cpu=cpu, gpu=gpu, memory=memory)
    minimum_resources = unit_resources.scale(minimum_units)
    op.resource_admission_spec = MagicMock(
        return_value=ResourceAdmissionSpec(
            minimum_resources=minimum_resources,
            unit_resources=None if fixed_gang else unit_resources,
            min_units=1 if fixed_gang else minimum_units,
            max_units=1 if fixed_gang else max_units,
        )
    )
    op.resource_admission_incompatible = MagicMock(return_value=False)
    op.min_scheduling_resources = MagicMock(return_value=unit_resources)
    return op


def _mock_admission_resource_manager(limits: ExecutionResources) -> MagicMock:
    return MagicMock(
        _options=ExecutionOptions(),
        get_global_limits=MagicMock(return_value=limits),
        get_allocation_target=MagicMock(return_value=None),
        op_resource_allocator_enabled=MagicMock(return_value=True),
    )


def _update_controller(
    controller: _ResourceAdmissionController,
    limits: Optional[ExecutionResources] = None,
) -> None:
    if limits is None:
        limits = controller._resource_manager.get_global_limits()
    controller.update_admission(limits)
    controller.update_allocation_grants(limits)


def test_physical_operator_resource_admission_defaults():
    source = InputDataBuffer(DataContext(), input_data=[])

    assert source.resource_admission_spec() is None
    assert not source.resource_admission_incompatible()
    source.apply_resource_admission_grant(ResourceAdmissionGrant(1, True))
    assert source.can_release_resource_admission()


@pytest.mark.parametrize(
    ("max_units", "may_submit", "error"),
    [
        (-1, False, "max_units must be a non-negative integer"),
        (1.0, False, "max_units must be a non-negative integer"),
        (True, False, "max_units must be a non-negative integer"),
        (1, 1, "may_submit must be a boolean"),
        (0, True, "may_submit requires at least one granted unit"),
    ],
)
def test_resource_admission_grant_validation_errors(max_units, may_submit, error):
    with pytest.raises(ValueError, match=error):
        ResourceAdmissionGrant(max_units, may_submit)


@pytest.mark.parametrize(
    "spec,error",
    [
        (
            ResourceAdmissionSpec(
                minimum_resources=ExecutionResources.zero(),
                unit_resources=ExecutionResources(gpu=1),
                min_units=1,
                max_units=1,
            ),
            "minimum resources must be positive",
        ),
        (
            ResourceAdmissionSpec(
                minimum_resources=ExecutionResources(gpu=-1),
                unit_resources=ExecutionResources(gpu=-1),
                min_units=1,
                max_units=1,
            ),
            "minimum resources must be positive",
        ),
        (
            ResourceAdmissionSpec(
                minimum_resources=ExecutionResources(gpu=float("inf")),
                unit_resources=ExecutionResources(gpu=float("inf")),
                min_units=1,
                max_units=1,
            ),
            "minimum resources must be finite",
        ),
        (
            ResourceAdmissionSpec(
                minimum_resources=ExecutionResources(gpu=1),
                unit_resources=ExecutionResources(gpu=float("inf")),
                min_units=1,
                max_units=1,
            ),
            "unit resources must be finite",
        ),
        (
            ResourceAdmissionSpec(
                minimum_resources=ExecutionResources(gpu=1),
                unit_resources=ExecutionResources(gpu=1),
                min_units=0,
                max_units=1,
            ),
            "min_units must be a positive integer",
        ),
        (
            ResourceAdmissionSpec(
                minimum_resources=ExecutionResources(gpu=1),
                unit_resources=ExecutionResources(gpu=1),
                min_units=1.0,
                max_units=1,
            ),
            "min_units must be a positive integer",
        ),
        (
            ResourceAdmissionSpec(
                minimum_resources=ExecutionResources(gpu=1),
                unit_resources=ExecutionResources(gpu=1),
                min_units=True,
                max_units=1,
            ),
            "min_units must be a positive integer",
        ),
        (
            ResourceAdmissionSpec(
                minimum_resources=ExecutionResources(gpu=1),
                unit_resources=ExecutionResources(gpu=1),
                min_units=1,
                max_units=1.0,
            ),
            "max_units must be an integer or None",
        ),
        (
            ResourceAdmissionSpec(
                minimum_resources=ExecutionResources(gpu=1),
                unit_resources=ExecutionResources(gpu=1),
                min_units=1,
                max_units=True,
            ),
            "max_units must be an integer or None",
        ),
        (
            ResourceAdmissionSpec(
                minimum_resources=ExecutionResources(gpu=1),
                unit_resources=None,
                min_units=2,
                max_units=2,
            ),
            "Fixed resource-admission gangs require min_units == max_units == 1",
        ),
        (
            ResourceAdmissionSpec(
                minimum_resources=ExecutionResources(gpu=1),
                unit_resources=None,
                min_units=1,
                max_units=2,
            ),
            "Fixed resource-admission gangs require min_units == max_units == 1",
        ),
        (
            ResourceAdmissionSpec(
                minimum_resources=ExecutionResources(gpu=1),
                unit_resources=None,
                min_units=1,
                max_units=None,
            ),
            "Fixed resource-admission gangs require min_units == max_units == 1",
        ),
        (
            ResourceAdmissionSpec(
                minimum_resources=ExecutionResources(gpu=2),
                unit_resources=ExecutionResources(gpu=1),
                min_units=1,
                max_units=2,
            ),
            "Admission floor must equal unit resources times min_units",
        ),
        (
            ResourceAdmissionSpec(
                minimum_resources=ExecutionResources(gpu=1),
                unit_resources=ExecutionResources(gpu=-1),
                min_units=1,
                max_units=1,
            ),
            "Admission floor must equal unit resources times min_units",
        ),
        (
            ResourceAdmissionSpec(
                minimum_resources=ExecutionResources(gpu=2),
                unit_resources=ExecutionResources(gpu=1),
                min_units=2,
                max_units=1,
            ),
            "max_units must be at least min_units",
        ),
    ],
)
def test_resource_admission_spec_validation_errors(spec, error):
    with pytest.raises(ValueError, match=error):
        _validate_resource_admission_spec(spec)


def test_resource_admission_initializes_participant_with_zero_grant():
    owner = _mock_resource_admission_op()
    spec = owner.resource_admission_spec()
    states = {owner: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0))}

    controller = _ResourceAdmissionController(
        states,
        _mock_admission_resource_manager(ExecutionResources(cpu=1, gpu=1)),
    )

    assert controller.has_participants()
    assert controller.get_floor(owner) == spec.minimum_resources
    assert controller.get_grant(owner) == ResourceAdmissionGrant(0, False)
    owner.apply_resource_admission_grant.assert_called_once_with(
        ResourceAdmissionGrant(0, False)
    )


def test_resource_admission_initializes_unmanaged_topology_without_grants():
    unmanaged = MagicMock()
    unmanaged.resource_admission_spec.return_value = None
    states = {
        unmanaged: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0))
    }

    controller = _ResourceAdmissionController(
        states,
        _mock_admission_resource_manager(ExecutionResources(cpu=1, gpu=1)),
    )

    assert not controller.has_participants()
    assert controller.get_floor(unmanaged) is None
    assert controller.get_grant(unmanaged) is None
    unmanaged.resource_admission_incompatible.assert_not_called()
    unmanaged.apply_resource_admission_grant.assert_not_called()


def test_resource_admission_initialization_ignores_object_store_limit():
    owner = _mock_resource_admission_op()
    owner.resource_admission_spec.return_value = ResourceAdmissionSpec(
        minimum_resources=ExecutionResources(object_store_memory=100),
        unit_resources=ExecutionResources(object_store_memory=100),
        min_units=1,
        max_units=1,
    )
    states = {owner: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0))}
    resource_manager = MagicMock(
        _options=ExecutionOptions(
            resource_limits=ExecutionResources.for_limits(object_store_memory=50)
        )
    )

    controller = _ResourceAdmissionController(states, resource_manager)

    assert controller.get_floor(owner) == ExecutionResources(object_store_memory=100)
    assert controller.get_grant(owner) == ResourceAdmissionGrant(0, False)
    assert not controller._should_expose_demand(
        owner, ExecutionResources.zero(), ExecutionResources.zero()
    )


def test_resource_admission_topological_frontier():
    upstream = _mock_resource_admission_op()
    completed_cpu_middle = MagicMock(
        spec=PhysicalOperator, input_dependencies=[upstream]
    )
    completed_cpu_middle.resource_admission_spec.return_value = None
    completed_cpu_middle.post_order_iter.return_value = [
        upstream,
        completed_cpu_middle,
    ]
    completed_cpu_middle.min_scheduling_resources.return_value = (
        ExecutionResources.zero()
    )
    downstream = _mock_resource_admission_op(inputs=[completed_cpu_middle])
    last = _mock_resource_admission_op(inputs=[downstream])
    upstream.has_completed.side_effect = AssertionError("must not call has_completed")
    upstream.has_next.side_effect = AssertionError("must not call has_next")
    states = {
        upstream: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0)),
        downstream: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
        last: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
    }
    resource_manager = _mock_admission_resource_manager(
        ExecutionResources(cpu=8, gpu=1)
    )

    controller = _ResourceAdmissionController(states, resource_manager)
    _update_controller(controller)

    # Downstream demand protects its unfinished GPU ancestor. With one GPU, the
    # ancestor is admitted, the first non-fitting claimant is the frontier, and
    # later claimants cannot leapfrog it.
    assert controller.get_grant(upstream) == ResourceAdmissionGrant(1, True)
    assert controller.get_grant(downstream) == ResourceAdmissionGrant(0, False)
    assert controller.get_grant(last) == ResourceAdmissionGrant(0, False)

    # Once the ancestor is complete, the downstream stages advance without
    # waiting for every non-GPU intermediate to remain unfinished.
    upstream.has_execution_finished.return_value = True
    _update_controller(controller)
    assert controller.get_grant(upstream) == ResourceAdmissionGrant(0, False)
    assert controller.get_grant(downstream) == ResourceAdmissionGrant(1, True)
    assert controller.get_grant(last) == ResourceAdmissionGrant(0, False)
    upstream.has_completed.assert_not_called()
    upstream.has_next.assert_not_called()
    upstream.apply_resource_admission_grant.assert_called_with(
        ResourceAdmissionGrant(max_units=0, may_submit=False)
    )


def test_resource_admission_streams_when_all_claims_fit():
    upstream = _mock_resource_admission_op()
    downstream = _mock_resource_admission_op(inputs=[upstream])
    states = {
        upstream: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
        downstream: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
    }
    resource_manager = _mock_admission_resource_manager(
        ExecutionResources(cpu=8, gpu=2)
    )

    controller = _ResourceAdmissionController(states, resource_manager)
    controller.update_admission(resource_manager.get_global_limits())
    resource_manager.get_allocation_target.return_value = None
    controller.update_allocation_grants(resource_manager.get_global_limits())

    assert controller.get_grant(upstream) == ResourceAdmissionGrant(1, True)
    assert controller.get_grant(downstream) == ResourceAdmissionGrant(1, True)


def test_resource_admission_prewarms_draining_fixed_gang_successor():
    upstream = _mock_resource_admission_op(
        inputs_complete=True,
        fixed_gang=True,
        internal_demand=True,
    )
    downstream = _mock_resource_admission_op(inputs=[upstream])
    last = _mock_resource_admission_op(inputs=[downstream])
    states = {
        upstream: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0)),
        downstream: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0)),
        last: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0)),
    }
    resource_manager = _mock_admission_resource_manager(
        ExecutionResources(cpu=2, gpu=2)
    )
    controller = _ResourceAdmissionController(states, resource_manager)

    _update_controller(controller)

    assert controller.get_grant(upstream) == ResourceAdmissionGrant(1, True)
    assert controller.get_grant(downstream) == ResourceAdmissionGrant(1, True)
    assert controller.get_grant(last) == ResourceAdmissionGrant(0, False)


def test_incompatible_gpu_owner_falls_back_whole_topology():
    managed = _mock_resource_admission_op(internal_demand=True)
    dynamic = _mock_resource_admission_op(inputs=[managed], internal_demand=True)
    dynamic.resource_admission_spec.return_value = None
    dynamic.resource_admission_incompatible.return_value = True
    states = {
        managed: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
        dynamic: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
    }
    resource_manager = _mock_admission_resource_manager(
        ExecutionResources(cpu=2, gpu=2)
    )

    with patch(
        "ray.data._internal.execution.resource_admission_controller.logger.warning"
    ) as warning:
        controller = _ResourceAdmissionController(states, resource_manager)

    assert not controller.has_participants()
    assert controller.get_grant(managed) is None
    assert controller.get_floor(managed) is None
    managed.apply_resource_admission_grant.assert_not_called()
    warning.assert_called_once()
    assert (
        "disabling resource admission for the whole topology"
        in warning.call_args.args[0]
    )
    controller.update_admission(ExecutionResources(cpu=2, gpu=2))
    resource_manager.op_resource_allocator_enabled.side_effect = AssertionError(
        "empty controller should return before probing the allocator"
    )
    controller.update_allocation_grants(ExecutionResources(cpu=2, gpu=2))


def test_resource_admission_helper_edge_paths():
    fixed = _mock_resource_admission_op(
        fixed_gang=True,
        inputs_complete=True,
        internal_demand=True,
    )
    successor = _mock_resource_admission_op(inputs=[fixed])
    elastic = _mock_resource_admission_op(max_units=None, internal_demand=True)
    states = {
        op: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0))
        for op in (fixed, successor, elastic)
    }
    controller = _ResourceAdmissionController(
        states,
        _mock_admission_resource_manager(ExecutionResources(cpu=3, gpu=3)),
    )

    assert controller._should_prewarm_fixed_gang_successor(successor)
    assert not controller._should_expose_demand(
        elastic,
        ExecutionResources(cpu=3, gpu=3),
        ExecutionResources.zero(),
    )
    assert controller._should_expose_demand(
        fixed, ExecutionResources.zero(), ExecutionResources.zero()
    )

    fixed.current_logical_usage.return_value = (
        fixed.resource_admission_spec().minimum_resources
    )
    assert not controller._should_expose_demand(
        fixed, ExecutionResources.zero(), ExecutionResources.zero()
    )
    fixed.current_logical_usage.return_value = ExecutionResources.zero()

    fixed.can_release_resource_admission.return_value = True
    controller._set_grant(fixed, 1, may_submit=False)
    assert controller._should_expose_demand(
        fixed,
        ExecutionResources(cpu=3, gpu=3),
        ExecutionResources.zero(),
    )
    assert not controller._should_expose_demand(
        fixed,
        ExecutionResources(cpu=3, gpu=3),
        ExecutionResources(gpu=1),
    )
    assert controller._overlaps_transient_progress_floor(
        ExecutionResources(memory=1), ExecutionResources(memory=1)
    )
    assert not controller._overlaps_transient_progress_floor(
        ExecutionResources(gpu=1), ExecutionResources(cpu=1)
    )

    controller._set_grant(elastic, 10, may_submit=True)
    controller._set_grant(elastic, 0, may_submit=False)
    assert controller.get_grant(elastic) == ResourceAdmissionGrant(10, True)

    assert (
        controller._max_units_for_target(
            ExecutionResources(cpu=3, gpu=3),
            fixed.resource_admission_spec(),
        )
        == 1
    )
    assert (
        controller._max_units_for_target(
            ExecutionResources(cpu=3, gpu=3),
            elastic.resource_admission_spec(),
        )
        == 3
    )
    assert controller._unit_capacity(
        ExecutionResources.inf(),
        elastic.resource_admission_spec().unit_resources,
    ) == float("inf")


@pytest.mark.parametrize(
    (
        "execution_finished",
        "inputs_complete",
        "queued_inputs",
        "can_release",
        "expected",
    ),
    [
        (False, False, 0, True, False),
        (False, True, 1, True, False),
        (True, False, 0, False, False),
        (False, True, 0, False, False),
        (True, False, 0, True, True),
        (False, True, 0, True, True),
    ],
)
def test_admission_completion_requires_drained_releasable_owner(
    execution_finished,
    inputs_complete,
    queued_inputs,
    can_release,
    expected,
):
    owner = _mock_resource_admission_op(
        completed=execution_finished,
        inputs_complete=inputs_complete,
        internal_demand=not can_release,
    )
    state = MagicMock(total_enqueued_input_blocks=MagicMock(return_value=queued_inputs))

    assert _ResourceAdmissionController._is_admission_complete(owner, state) is expected


def test_owned_idle_open_stream_remains_admission_claimant():
    owner = _mock_resource_admission_op()
    owner.current_logical_usage.return_value = ExecutionResources(gpu=1)
    states = {owner: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0))}
    controller = _ResourceAdmissionController(
        states,
        _mock_admission_resource_manager(ExecutionResources(gpu=1)),
    )

    assert controller._is_claimant(owner)


def test_fixed_gang_successor_is_not_prewarmed_across_fan_in():
    fixed = _mock_resource_admission_op(fixed_gang=True, inputs_complete=True)
    other = _mock_resource_admission_op()
    successor = _mock_resource_admission_op(inputs=[fixed, other])
    states = {
        op: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0))
        for op in (fixed, other, successor)
    }
    controller = _ResourceAdmissionController(
        states,
        _mock_admission_resource_manager(ExecutionResources(cpu=3, gpu=3)),
    )

    assert not controller._should_prewarm_fixed_gang_successor(successor)


def test_transient_progress_floor_covers_queued_and_running_ancestors():
    queued = MagicMock()
    queued.input_dependencies = []
    queued.resource_admission_spec.return_value = None
    queued.resource_admission_incompatible.return_value = False
    queued.post_order_iter.return_value = [queued]
    queued.has_execution_finished.return_value = False
    queued._inputs_complete = True
    queued.current_logical_usage.return_value = ExecutionResources.zero()
    queued.min_scheduling_resources.return_value = ExecutionResources(cpu=2)

    running = MagicMock()
    running.input_dependencies = []
    running.resource_admission_spec.return_value = None
    running.resource_admission_incompatible.return_value = False
    running.post_order_iter.return_value = [running]
    running.has_execution_finished.return_value = False
    running._inputs_complete = True
    running.current_logical_usage.return_value = ExecutionResources(cpu=1, gpu=1)
    running.min_scheduling_resources.return_value = ExecutionResources(cpu=1, gpu=1)

    owner = _mock_resource_admission_op(inputs=[queued, running])
    states = {
        queued: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
        running: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0)),
        owner: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
    }
    controller = _ResourceAdmissionController(
        states,
        _mock_admission_resource_manager(ExecutionResources(cpu=3, gpu=2)),
    )

    # The ancestors can make progress sequentially, so overlapping CPU demand
    # uses the component-wise maximum (2 CPUs), not the sum (3 CPUs).
    assert controller._transient_progress_floor({owner}) == ExecutionResources(
        cpu=2, gpu=1
    )


def test_releasable_fixed_gang_yields_to_transient_ancestor_after_capacity_shrinks():
    task = MagicMock()
    task.input_dependencies = []
    task.resource_admission_spec.return_value = None
    task.resource_admission_incompatible.return_value = False
    task.post_order_iter.return_value = [task]
    task.has_execution_finished.return_value = False
    task._inputs_complete = False
    task.current_logical_usage.return_value = ExecutionResources.zero()
    task.min_scheduling_resources.return_value = ExecutionResources(gpu=1)

    gang = _mock_resource_admission_op(
        inputs=[task],
        cpu=0,
        gpu=1,
        fixed_gang=True,
    )
    states = {
        task: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
        gang: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
    }
    controller = _ResourceAdmissionController(
        states,
        _mock_admission_resource_manager(ExecutionResources(gpu=2)),
    )

    controller.update_admission(ExecutionResources(gpu=2))
    assert controller.get_grant(gang) == ResourceAdmissionGrant(1, True)
    assert controller._should_expose_demand(
        gang, ExecutionResources.zero(), ExecutionResources.zero()
    )

    controller.update_admission(ExecutionResources(gpu=1))

    assert controller.get_grant(gang) == ResourceAdmissionGrant(0, False)


@pytest.mark.parametrize("fixed_gang", [False, True])
def test_cold_owner_yields_autoscaling_demand_to_transient_ancestor(fixed_gang):
    task = MagicMock()
    task.input_dependencies = []
    task.resource_admission_spec.return_value = None
    task.resource_admission_incompatible.return_value = False
    task.post_order_iter.return_value = [task]
    task.has_execution_finished.return_value = False
    task._inputs_complete = False
    task.current_logical_usage.return_value = ExecutionResources.zero()
    task.min_scheduling_resources.return_value = ExecutionResources(gpu=1)

    owner = _mock_resource_admission_op(
        inputs=[task],
        cpu=0,
        gpu=1,
        fixed_gang=fixed_gang,
    )
    states = {
        task: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
        owner: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
    }
    controller = _ResourceAdmissionController(
        states,
        _mock_admission_resource_manager(ExecutionResources.zero()),
    )

    controller.update_admission(ExecutionResources.zero())
    assert controller.get_grant(owner) == ResourceAdmissionGrant(0, False)

    controller.update_admission(ExecutionResources(gpu=1))
    assert controller.get_grant(owner) == ResourceAdmissionGrant(0, False)

    task.has_execution_finished.return_value = True
    task._inputs_complete = True
    states[task].total_enqueued_input_blocks.return_value = 0
    controller.update_admission(ExecutionResources.zero())
    assert controller.get_grant(owner) == ResourceAdmissionGrant(1, False)


def test_gpu_task_ancestor_keeps_progress_floor_before_actor_admission():
    data_context = DataContext.get_current()
    source = InputDataBuffer(data_context, input_data=[])
    gpu_task = _mock_map_op(
        source,
        ray_remote_args={"num_cpus": 0, "num_gpus": 1},
        name="GPUTask",
    )
    managed = _mock_map_op(
        gpu_task,
        ray_remote_args={"num_cpus": 0, "num_gpus": 1},
        compute_strategy=ActorPoolStrategy(size=1),
        name="ManagedGPU",
    )
    states: "Topology" = {
        source: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0)),
        gpu_task: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
        managed: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
    }
    limits = ExecutionResources(gpu=1)
    controller = _ResourceAdmissionController(
        states,
        _mock_admission_resource_manager(limits),
    )

    controller.update_admission(limits)
    assert not gpu_task.resource_admission_incompatible()
    assert controller.has_participants()
    assert controller.get_grant(managed) == ResourceAdmissionGrant(0, False)

    gpu_task._inputs_complete = True
    states[gpu_task].total_enqueued_input_blocks.return_value = 0
    controller.update_admission(limits)

    assert controller.get_grant(managed) == ResourceAdmissionGrant(1, True)


@pytest.mark.parametrize(
    "task_remote_args,actor_remote_args,admitted_while_task_needs_progress",
    [
        ({"num_cpus": 1}, {"num_gpus": 1}, False),
        ({"num_cpus": 1}, {"num_cpus": 0, "num_gpus": 1}, True),
        ({"num_cpus": 0, "num_gpus": 1}, {"num_cpus": 0, "num_gpus": 1}, False),
    ],
)
def test_transient_ancestor_progress_floor_only_blocks_overlapping_actor(
    task_remote_args,
    actor_remote_args,
    admitted_while_task_needs_progress,
):
    data_context = DataContext.get_current()
    source = InputDataBuffer(data_context, input_data=[])
    task = _mock_map_op(
        source,
        ray_remote_args=task_remote_args,
        name="TransientTask",
    )
    managed = _mock_map_op(
        task,
        ray_remote_args=actor_remote_args,
        compute_strategy=ActorPoolStrategy(size=1),
        name="ManagedGPU",
    )
    states: "Topology" = {
        source: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0)),
        task: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
        managed: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
    }
    limits = ExecutionResources(cpu=1, gpu=1)
    controller = _ResourceAdmissionController(
        states,
        _mock_admission_resource_manager(limits),
    )

    controller.update_admission(limits)
    grant = controller.get_grant(managed)
    assert grant is not None
    assert grant.may_submit is admitted_while_task_needs_progress

    task._inputs_complete = True
    states[task].total_enqueued_input_blocks.return_value = 0
    controller.update_admission(limits)
    assert controller.get_grant(managed) == ResourceAdmissionGrant(1, True)


def test_cpu_actor_owner_falls_back_whole_topology():
    data_context = DataContext.get_current()
    source = InputDataBuffer(data_context, input_data=[])
    managed = _mock_map_op(
        source,
        ray_remote_args={"num_gpus": 1},
        compute_strategy=ActorPoolStrategy(size=1),
        name="ManagedGPU",
    )
    cpu_owner = _mock_map_op(
        managed,
        ray_remote_args={"num_cpus": 1},
        compute_strategy=ActorPoolStrategy(size=1),
        name="CPUOwner",
    )
    states: "Topology" = {
        source: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0)),
        managed: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
        cpu_owner: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0)),
    }

    with patch(
        "ray.data._internal.execution.resource_admission_controller.logger.warning"
    ) as warning:
        controller = _ResourceAdmissionController(
            states,
            _mock_admission_resource_manager(ExecutionResources(cpu=1, gpu=1)),
        )

    assert managed.resource_admission_spec() is not None
    assert cpu_owner.resource_admission_spec() is None
    assert cpu_owner.resource_admission_incompatible()
    assert not controller.has_participants()
    assert isinstance(managed, ActorPoolMapOperator)
    assert isinstance(cpu_owner, ActorPoolMapOperator)
    assert managed._admission_grant is None
    assert cpu_owner._admission_grant is None
    warning.assert_called_once()


def test_cpu_hash_shuffle_owner_falls_back_whole_topology():
    data_context = DataContext.get_current()
    source = InputDataBuffer(data_context, input_data=[])
    other_source = InputDataBuffer(data_context, input_data=[])
    managed = _mock_map_op(
        source,
        ray_remote_args={"num_gpus": 1},
        compute_strategy=ActorPoolStrategy(size=1),
        name="ManagedGPU",
    )
    cpu_hash_join = _mock_join_op(managed, other_source)
    states: "Topology" = {
        source: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0)),
        other_source: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0)),
        managed: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
        cpu_hash_join: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0)),
    }

    with patch(
        "ray.data._internal.execution.resource_admission_controller.logger.warning"
    ) as warning:
        controller = _ResourceAdmissionController(
            states,
            _mock_admission_resource_manager(ExecutionResources(cpu=2, gpu=1)),
        )

    assert cpu_hash_join.resource_admission_incompatible()
    assert not controller.has_participants()
    assert isinstance(managed, ActorPoolMapOperator)
    assert managed._admission_grant is None
    warning.assert_called_once()


def test_allocator_disabled_distributes_capacity_after_all_floors():
    first = _mock_resource_admission_op(internal_demand=True)
    second = _mock_resource_admission_op(internal_demand=True)
    gang = _mock_resource_admission_op(fixed_gang=True, internal_demand=True)
    states = {
        op: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1))
        for op in (first, second, gang)
    }
    limits = ExecutionResources(cpu=5, gpu=5)
    resource_manager = _mock_admission_resource_manager(limits)
    resource_manager.op_resource_allocator_enabled.return_value = False
    controller = _ResourceAdmissionController(states, resource_manager)

    controller.update_admission(limits)
    controller.update_allocation_grants(limits)

    assert controller.get_grant(first) == ResourceAdmissionGrant(3, True)
    assert controller.get_grant(second) == ResourceAdmissionGrant(1, True)
    assert controller.get_grant(gang) == ResourceAdmissionGrant(1, True)


def test_allocator_disabled_applies_grants_sequentially():
    first = _mock_resource_admission_op(internal_demand=True)
    second = _mock_resource_admission_op()
    states = {
        op: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1))
        for op in (first, second)
    }
    limits = ExecutionResources(cpu=4, gpu=4)
    resource_manager = _mock_admission_resource_manager(limits)
    resource_manager.op_resource_allocator_enabled.return_value = False
    controller = _ResourceAdmissionController(states, resource_manager)
    controller.update_admission(limits)

    denied = ResourceAdmissionGrant(0, False)
    first.apply_resource_admission_grant.side_effect = lambda _: controller._set_grant(
        second, 0, may_submit=False
    )
    second.apply_resource_admission_grant.reset_mock()

    controller.update_allocation_grants(limits)

    assert controller.get_grant(first) == ResourceAdmissionGrant(3, True)
    assert controller.get_grant(second) == denied
    second.apply_resource_admission_grant.assert_called_once_with(denied)


def test_allocator_disabled_skips_nonadmitted_owner():
    admitted = _mock_resource_admission_op(internal_demand=True)
    blocked = _mock_resource_admission_op(internal_demand=True)
    states = {
        op: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1))
        for op in (admitted, blocked)
    }
    limits = ExecutionResources(cpu=1, gpu=1)
    resource_manager = _mock_admission_resource_manager(limits)
    resource_manager.op_resource_allocator_enabled.return_value = False
    controller = _ResourceAdmissionController(states, resource_manager)

    controller.update_admission(limits)
    controller.update_allocation_grants(limits)

    assert controller.get_grant(admitted) == ResourceAdmissionGrant(1, True)
    assert controller.get_grant(blocked) == ResourceAdmissionGrant(0, False)


def test_allocator_disabled_preserves_complete_fractional_gpu_units():
    owner = _mock_resource_admission_op(
        cpu=0,
        gpu=0.1,
        max_units=10,
        internal_demand=True,
    )
    states = {owner: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1))}
    limits = ExecutionResources(gpu=0.3)
    resource_manager = _mock_admission_resource_manager(limits)
    resource_manager.op_resource_allocator_enabled.return_value = False
    controller = _ResourceAdmissionController(states, resource_manager)

    controller.update_admission(limits)
    controller.update_allocation_grants(limits)

    assert controller.get_grant(owner) == ResourceAdmissionGrant(3, True)


def test_allocator_disabled_preserves_transient_ancestor_progress_floor():
    gpu_task = MagicMock()
    gpu_task.input_dependencies = []
    gpu_task.resource_admission_spec.return_value = None
    gpu_task.resource_admission_incompatible.return_value = False
    gpu_task.post_order_iter.return_value = [gpu_task]
    gpu_task.has_execution_finished.return_value = False
    gpu_task._inputs_complete = False
    gpu_task.current_logical_usage.return_value = ExecutionResources.zero()
    gpu_task.min_scheduling_resources.return_value = ExecutionResources(gpu=1)

    owner = _mock_resource_admission_op(
        inputs=[gpu_task],
        cpu=0,
        gpu=1,
        max_units=2,
        internal_demand=True,
    )
    states = {
        gpu_task: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
        owner: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
    }
    limits = ExecutionResources(gpu=2)
    resource_manager = _mock_admission_resource_manager(limits)
    resource_manager.op_resource_allocator_enabled.return_value = False
    controller = _ResourceAdmissionController(states, resource_manager)

    _update_controller(controller)
    assert controller.get_grant(owner) == ResourceAdmissionGrant(1, True)

    gpu_task.has_execution_finished.return_value = True
    gpu_task._inputs_complete = True
    states[gpu_task].total_enqueued_input_blocks.return_value = 0
    _update_controller(controller)
    assert controller.get_grant(owner) == ResourceAdmissionGrant(2, True)


def test_allocator_disabled_passes_unused_capacity_to_later_owner():
    gang = _mock_resource_admission_op(fixed_gang=True, internal_demand=True)
    capped = _mock_resource_admission_op(max_units=1, internal_demand=True)
    elastic = _mock_resource_admission_op(internal_demand=True)
    states = {
        op: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1))
        for op in (gang, capped, elastic)
    }
    limits = ExecutionResources(cpu=5, gpu=5)
    resource_manager = _mock_admission_resource_manager(limits)
    resource_manager.op_resource_allocator_enabled.return_value = False
    controller = _ResourceAdmissionController(states, resource_manager)

    _update_controller(controller)

    assert controller.get_grant(gang) == ResourceAdmissionGrant(1, True)
    assert controller.get_grant(capped) == ResourceAdmissionGrant(1, True)
    assert controller.get_grant(elastic) == ResourceAdmissionGrant(3, True)


def test_allocator_queries_targets_only_for_admitted_owners():
    admitted = _mock_resource_admission_op(internal_demand=True)
    blocked = _mock_resource_admission_op(internal_demand=True)
    states = {
        op: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1))
        for op in (admitted, blocked)
    }
    limits = ExecutionResources(cpu=1, gpu=1)
    resource_manager = _mock_admission_resource_manager(limits)
    controller = _ResourceAdmissionController(states, resource_manager)
    controller.update_admission(limits)
    resource_manager.get_allocation_target.reset_mock()

    controller.update_allocation_grants(limits)

    resource_manager.get_allocation_target.assert_called_once_with(admitted)
    assert controller.get_grant(blocked) == ResourceAdmissionGrant(0, False)


@pytest.mark.parametrize(("target_gpu", "expected_units"), [(0.3, 3), (0.4, 4)])
def test_allocator_target_preserves_complete_fractional_gpu_units(
    target_gpu, expected_units
):
    owner = _mock_resource_admission_op(
        cpu=0,
        gpu=0.1,
        max_units=10,
        internal_demand=True,
    )
    states = {owner: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1))}
    limits = ExecutionResources(gpu=target_gpu)
    resource_manager = _mock_admission_resource_manager(limits)
    resource_manager.get_allocation_target.return_value = ExecutionResources(
        gpu=target_gpu
    )
    controller = _ResourceAdmissionController(states, resource_manager)

    controller.update_admission(limits)
    controller.update_allocation_grants(limits)

    assert controller.get_grant(owner) == ResourceAdmissionGrant(expected_units, True)


def test_elastic_grant_preserves_configured_minimum_units():
    actor_pool = _mock_resource_admission_op(
        cpu=1, gpu=1, minimum_units=2, max_units=4, internal_demand=True
    )
    states = {
        actor_pool: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0))
    }
    resource_manager = _mock_admission_resource_manager(
        ExecutionResources(cpu=2, gpu=2)
    )
    controller = _ResourceAdmissionController(states, resource_manager)

    _update_controller(controller)

    assert controller.get_grant(actor_pool) == ResourceAdmissionGrant(2, True)


def test_zero_aggregate_admission_floor_is_rejected():
    owner = _mock_resource_admission_op(
        cpu=0,
        gpu=0,
        custom_resources=(("custom_gpu", 1.0),),
        inputs_complete=True,
    )
    states = {owner: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0))}
    resource_manager = _mock_admission_resource_manager(ExecutionResources.zero())

    with pytest.raises(ValueError, match="minimum resources must be positive"):
        _ResourceAdmissionController(states, resource_manager)


def test_object_store_only_elastic_admission_target_is_bounded():
    spec = ResourceAdmissionSpec(
        minimum_resources=ExecutionResources(object_store_memory=100),
        unit_resources=ExecutionResources(object_store_memory=100),
        min_units=1,
        max_units=None,
    )

    assert (
        _ResourceAdmissionController._max_units_for_target(
            ExecutionResources(object_store_memory=450), spec
        )
        == 4
    )


def test_infinite_allocation_target_uses_finite_cap():
    capped = ResourceAdmissionSpec(
        minimum_resources=ExecutionResources(gpu=1),
        unit_resources=ExecutionResources(gpu=1),
        min_units=1,
        max_units=4,
    )
    uncapped = ResourceAdmissionSpec(
        minimum_resources=ExecutionResources(gpu=1),
        unit_resources=ExecutionResources(gpu=1),
        min_units=1,
        max_units=None,
    )

    assert (
        _ResourceAdmissionController._max_units_for_target(
            ExecutionResources.inf(), capped
        )
        == 4
    )
    with pytest.raises(ValueError, match="requires a finite allocation target"):
        _ResourceAdmissionController._max_units_for_target(
            ExecutionResources.inf(), uncapped
        )


def test_sticky_elastic_owner_keeps_floor_until_state_can_be_released():
    owner = _mock_resource_admission_op(internal_demand=True)
    states = {owner: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0))}
    resource_manager = _mock_admission_resource_manager(
        ExecutionResources(cpu=1, gpu=1)
    )
    controller = _ResourceAdmissionController(states, resource_manager)

    _update_controller(controller)
    assert controller.get_grant(owner) == ResourceAdmissionGrant(1, True)

    owner.can_release_resource_admission.return_value = False
    _update_controller(controller, ExecutionResources.zero())

    # Do not destroy a callable-class actor while future input or internal work
    # can still depend on its state, even if the refreshed capacity is short.
    assert controller.get_grant(owner) == ResourceAdmissionGrant(1, True)

    owner._inputs_complete = True
    owner.can_release_resource_admission.return_value = True
    _update_controller(controller, ExecutionResources.zero())

    assert controller.get_grant(owner) == ResourceAdmissionGrant(0, False)


def test_sticky_owner_rejects_direct_grant_revocation():
    owner = _mock_resource_admission_op(internal_demand=True)
    states = {owner: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0))}
    controller = _ResourceAdmissionController(
        states,
        _mock_admission_resource_manager(ExecutionResources(cpu=1, gpu=1)),
    )
    controller._grants[owner] = ResourceAdmissionGrant(1, True)
    owner.can_release_resource_admission.return_value = False
    owner.apply_resource_admission_grant.reset_mock()

    controller._set_grant(owner, max_units=0, may_submit=False)

    assert controller.get_grant(owner) == ResourceAdmissionGrant(1, True)
    owner.apply_resource_admission_grant.assert_not_called()


def test_pending_sticky_owner_is_promoted_behind_frontier():
    frontier = _mock_resource_admission_op()
    pending = _mock_resource_admission_op()
    states = {
        op: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1))
        for op in (frontier, pending)
    }
    controller = _ResourceAdmissionController(
        states,
        _mock_admission_resource_manager(ExecutionResources.zero()),
    )
    controller._set_grant(pending, max_units=1, may_submit=False)
    pending.can_release_resource_admission.return_value = False

    controller.update_admission(ExecutionResources.zero())

    assert controller.get_grant(frontier) == ResourceAdmissionGrant(1, False)
    assert controller.get_grant(pending) == ResourceAdmissionGrant(1, True)


def test_sticky_owner_bypasses_unrelated_earlier_frontier():
    frontier = _mock_resource_admission_op()
    sticky = _mock_resource_admission_op()
    states = {
        op: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1))
        for op in (frontier, sticky)
    }
    controller = _ResourceAdmissionController(
        states,
        _mock_admission_resource_manager(ExecutionResources(cpu=2, gpu=2)),
    )
    controller.update_admission(ExecutionResources(cpu=2, gpu=2))
    sticky.can_release_resource_admission.return_value = False

    controller.update_admission(ExecutionResources(cpu=1, gpu=1))

    assert controller.get_grant(frontier) == ResourceAdmissionGrant(0, False)
    assert controller.get_grant(sticky) == ResourceAdmissionGrant(1, True)


def test_set_grant_clamps_cap_and_suppresses_duplicate_callback():
    owner = _mock_resource_admission_op(max_units=4)
    states = {owner: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0))}
    controller = _ResourceAdmissionController(
        states,
        _mock_admission_resource_manager(ExecutionResources(cpu=1, gpu=1)),
    )
    owner.apply_resource_admission_grant.reset_mock()

    controller._set_grant(owner, max_units=10, may_submit=True)
    controller._set_grant(owner, max_units=10, may_submit=True)

    grant = ResourceAdmissionGrant(4, True)
    assert controller.get_grant(owner) == grant
    owner.apply_resource_admission_grant.assert_called_once_with(grant)


def test_uncapped_owner_accepts_unbounded_unit_grant():
    owner = _mock_resource_admission_op(max_units=None)
    states = {owner: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0))}
    controller = _ResourceAdmissionController(
        states,
        _mock_admission_resource_manager(ExecutionResources(cpu=1, gpu=1)),
    )
    owner.apply_resource_admission_grant.reset_mock()

    controller._set_grant(owner, max_units=10, may_submit=True)

    assert controller.get_grant(owner) == ResourceAdmissionGrant(10, True)
    owner.apply_resource_admission_grant.assert_called_once_with(
        ResourceAdmissionGrant(10, True)
    )


def test_resource_admission_releases_drained_ancestor_with_buffered_output():
    upstream = _mock_resource_admission_op(inputs_complete=True)
    upstream.current_logical_usage.return_value = ExecutionResources(gpu=1)
    downstream = _mock_resource_admission_op(inputs=[upstream])
    states = {
        upstream: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0)),
        downstream: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
    }
    resource_manager = _mock_admission_resource_manager(
        ExecutionResources(cpu=8, gpu=1)
    )
    controller = _ResourceAdmissionController(states, resource_manager)

    _update_controller(controller)

    assert not upstream.has_execution_finished()
    assert controller.get_grant(upstream) == ResourceAdmissionGrant(0, False)
    assert controller.get_grant(downstream) == ResourceAdmissionGrant(1, True)
    upstream.apply_resource_admission_grant.assert_called_with(
        ResourceAdmissionGrant(0, False)
    )


def test_resource_admission_fan_in_does_not_leapfrog():
    left = _mock_resource_admission_op(gpu=0.5)
    right = _mock_resource_admission_op(gpu=1)
    fan_in = MagicMock(spec=PhysicalOperator, input_dependencies=[left, right])
    fan_in.post_order_iter.return_value = [left, right, fan_in]
    downstream = _mock_resource_admission_op(inputs=[fan_in], gpu=0.5)
    states = {
        left: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0)),
        right: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0)),
        downstream: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
    }
    resource_manager = _mock_admission_resource_manager(
        ExecutionResources(cpu=8, gpu=1)
    )

    controller = _ResourceAdmissionController(states, resource_manager)
    controller.update_admission(resource_manager.get_global_limits())

    assert controller.get_grant(left) == ResourceAdmissionGrant(1, True)
    assert controller.get_grant(right) == ResourceAdmissionGrant(0, False)
    assert controller.get_grant(downstream) == ResourceAdmissionGrant(0, False)


def test_resource_admission_aggregates_fixed_gang_bundles():
    gang = _mock_resource_admission_op(
        fixed_gang=True,
        minimum_units=2,
        internal_demand=True,
    )
    states = {gang: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0))}
    resource_manager = _mock_admission_resource_manager(
        ExecutionResources(cpu=2, gpu=1)
    )
    controller = _ResourceAdmissionController(states, resource_manager)

    _update_controller(controller)
    gang.apply_resource_admission_grant.assert_called_with(
        ResourceAdmissionGrant(1, False)
    )

    _update_controller(controller, ExecutionResources(cpu=2, gpu=2))
    gang.apply_resource_admission_grant.assert_called_with(
        ResourceAdmissionGrant(1, True)
    )


def test_fixed_gang_keeps_cold_autoscaling_demand_until_upstream_handoff():
    upstream = _mock_resource_admission_op(internal_demand=True, max_units=1)
    gang = _mock_resource_admission_op(
        inputs=[upstream],
        fixed_gang=True,
        minimum_units=4,
        internal_demand=False,
    )
    states = {
        upstream: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
        gang: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
    }
    resource_manager = _mock_admission_resource_manager(
        ExecutionResources(cpu=1, gpu=1)
    )
    controller = _ResourceAdmissionController(states, resource_manager)

    controller.update_admission(resource_manager.get_global_limits())
    assert controller.get_grant(gang) == ResourceAdmissionGrant(1, False)

    controller.update_admission(ExecutionResources(cpu=4, gpu=4))
    assert controller.get_grant(gang) == ResourceAdmissionGrant(1, False)

    upstream.has_execution_finished.return_value = True
    upstream.can_release_resource_admission.return_value = True
    states[upstream].total_enqueued_input_blocks.return_value = 0
    controller.update_admission(ExecutionResources(cpu=4, gpu=4))
    assert controller.get_grant(gang) == ResourceAdmissionGrant(1, True)


def test_fixed_gang_waits_for_upstream_handoff_without_reserving_early():
    upstream = _mock_resource_admission_op(internal_demand=True)
    gang = _mock_resource_admission_op(
        inputs=[upstream],
        fixed_gang=True,
        internal_demand=True,
    )
    states = {
        upstream: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0)),
        gang: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0)),
    }
    resource_manager = _mock_admission_resource_manager(
        ExecutionResources(cpu=8, gpu=1)
    )
    controller = _ResourceAdmissionController(states, resource_manager)

    _update_controller(controller)

    assert controller.get_grant(upstream) == ResourceAdmissionGrant(1, True)
    gang.apply_resource_admission_grant.assert_called_with(
        ResourceAdmissionGrant(0, False)
    )


def test_admission_floor_exceeding_explicit_limits_fails_fast():
    gang = _mock_resource_admission_op(
        fixed_gang=True,
        minimum_units=2,
        internal_demand=True,
    )
    states = {gang: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0))}
    resource_manager = MagicMock(
        _options=ExecutionOptions(
            resource_limits=ExecutionResources.for_limits(cpu=8, gpu=1)
        )
    )

    with pytest.raises(
        ValueError, match="admission floor.*gpu=2.*exceeds the explicit"
    ):
        _ResourceAdmissionController(states, resource_manager)


def test_fractional_actor_floor_exceeding_explicit_limit_fails_fast():
    actor = _mock_resource_admission_op(gpu=0.5, internal_demand=True)
    states = {actor: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0))}
    resource_manager = MagicMock(
        _options=ExecutionOptions(
            resource_limits=ExecutionResources.for_limits(gpu=0.25)
        )
    )

    with pytest.raises(ValueError, match="admission floor.*gpu=0.5"):
        _ResourceAdmissionController(states, resource_manager)


def test_fixed_gang_uses_effective_limit_when_exclusion_is_permitted():
    gang = _mock_resource_admission_op(
        fixed_gang=True,
        minimum_units=2,
        internal_demand=True,
        memory=1,
    )
    states = {gang: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0))}
    options = ExecutionOptions(
        resource_limits=ExecutionResources.for_limits(cpu=8, memory=2),
        exclude_resources=ExecutionResources(memory=1),
    )
    # ExecutionOptions permits a logical-memory limit and exclusion together.
    options.validate()
    resource_manager = MagicMock()
    resource_manager._options = options

    with pytest.raises(
        ValueError, match="admission floor.*memory=2.0.*effective execution resource"
    ):
        _ResourceAdmissionController(states, resource_manager)


def test_started_sticky_gang_protects_unfinished_admission_ancestors():
    upstream = _mock_resource_admission_op(internal_demand=True)
    gang = _mock_resource_admission_op(
        inputs=[upstream],
        fixed_gang=True,
        internal_demand=True,
    )
    states = {
        upstream: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
        gang: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
    }
    resource_manager = _mock_admission_resource_manager(
        ExecutionResources(cpu=2, gpu=2)
    )
    controller = _ResourceAdmissionController(states, resource_manager)

    _update_controller(controller, ExecutionResources(cpu=2, gpu=2))
    assert controller.get_grant(upstream) == ResourceAdmissionGrant(1, True)
    assert controller.get_grant(gang) == ResourceAdmissionGrant(1, True)

    gang.can_release_resource_admission.return_value = False
    _update_controller(controller, ExecutionResources(cpu=1, gpu=1))

    # The gang cannot finish until its producer supplies all remaining input.
    # Preserve both progress floors even while refreshed capacity is short.
    assert controller.get_grant(upstream) == ResourceAdmissionGrant(1, True)
    assert controller.get_grant(gang) == ResourceAdmissionGrant(1, True)


@pytest.mark.parametrize("internal_demand", [False, True])
def test_stable_expanded_grant_does_not_churn_callbacks(internal_demand):
    owner = _mock_resource_admission_op(
        cpu=0,
        gpu=1,
        max_units=3,
        internal_demand=internal_demand,
    )
    states = {owner: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1))}
    limits = ExecutionResources(gpu=3)
    resource_manager = _mock_admission_resource_manager(limits)
    resource_manager.get_allocation_target.return_value = limits
    controller = _ResourceAdmissionController(states, resource_manager)

    _update_controller(controller)
    assert controller.get_grant(owner) == ResourceAdmissionGrant(3, True)
    owner.apply_resource_admission_grant.reset_mock()

    _update_controller(controller)

    assert controller.get_grant(owner) == ResourceAdmissionGrant(3, True)
    owner.apply_resource_admission_grant.assert_not_called()


def test_whole_unit_redistribution_skips_fixed_and_maxed_owners():
    fixed_gang = _mock_resource_admission_op(fixed_gang=True, internal_demand=True)
    capped_pool = _mock_resource_admission_op(max_units=1, internal_demand=True)
    elastic_pool = _mock_resource_admission_op(internal_demand=True)
    states = {
        op: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1))
        for op in (fixed_gang, capped_pool, elastic_pool)
    }
    limits = ExecutionResources(cpu=4, gpu=4)
    resource_manager = _mock_admission_resource_manager(limits)
    target = ExecutionResources(cpu=1.5, gpu=1.5)
    resource_manager.get_allocation_target.side_effect = lambda op: target
    controller = _ResourceAdmissionController(states, resource_manager)

    controller.update_admission(limits)
    controller.update_allocation_grants(limits)

    assert controller.get_grant(fixed_gang) == ResourceAdmissionGrant(1, True)
    assert controller.get_grant(capped_pool) == ResourceAdmissionGrant(1, True)
    assert controller.get_grant(elastic_pool) == ResourceAdmissionGrant(2, True)


def test_whole_unit_redistribution_supports_mixed_unit_shapes():
    cpu_heavy = _mock_resource_admission_op(cpu=2, gpu=1, internal_demand=True)
    balanced = _mock_resource_admission_op(cpu=1, gpu=1, internal_demand=True)
    states = {
        op: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1))
        for op in (cpu_heavy, balanced)
    }
    limits = ExecutionResources(cpu=5, gpu=3)
    resource_manager = _mock_admission_resource_manager(limits)
    targets = {
        cpu_heavy: ExecutionResources(cpu=3, gpu=1.5),
        balanced: ExecutionResources(cpu=1.5, gpu=1.5),
    }
    resource_manager.get_allocation_target.side_effect = targets.__getitem__
    controller = _ResourceAdmissionController(states, resource_manager)

    controller.update_admission(limits)
    controller.update_allocation_grants(limits)

    # The combined fractional remainder is 1.5 CPU / 1 GPU. It cannot form
    # another CPU-heavy unit, but it does form one complete balanced unit.
    assert controller.get_grant(cpu_heavy) == ResourceAdmissionGrant(1, True)
    assert controller.get_grant(balanced) == ResourceAdmissionGrant(2, True)


def test_whole_unit_redistribution_reuses_unplaced_complete_target():
    large_units = _mock_resource_admission_op(
        cpu=2,
        gpu=2,
        internal_demand=True,
    )
    small_units = _mock_resource_admission_op(
        cpu=1,
        gpu=1,
        internal_demand=True,
    )
    states = {
        op: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1))
        for op in (large_units, small_units)
    }
    limits = ExecutionResources(cpu=4, gpu=4)
    resource_manager = _mock_admission_resource_manager(limits)
    targets = {
        large_units: ExecutionResources(cpu=4, gpu=4),
        small_units: ExecutionResources(cpu=1, gpu=1),
    }
    resource_manager.get_allocation_target.side_effect = targets.__getitem__
    controller = _ResourceAdmissionController(states, resource_manager)

    _update_controller(controller)

    # The large owner's complete second unit cannot fit in the 1-CPU/1-GPU
    # remainder. Reuse that unplaced target budget for a small complete unit.
    assert controller.get_grant(large_units) == ResourceAdmissionGrant(1, True)
    assert controller.get_grant(small_units) == ResourceAdmissionGrant(2, True)


def test_whole_unit_redistribution_supports_multi_unit_floors():
    first = _mock_resource_admission_op(
        cpu=0,
        gpu=1,
        minimum_units=2,
        max_units=3,
        internal_demand=True,
    )
    second = _mock_resource_admission_op(
        cpu=0,
        gpu=1,
        minimum_units=2,
        max_units=3,
        internal_demand=True,
    )
    states = {
        op: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1))
        for op in (first, second)
    }
    limits = ExecutionResources(gpu=5)
    resource_manager = _mock_admission_resource_manager(limits)
    resource_manager.get_allocation_target.return_value = ExecutionResources(gpu=2.5)
    controller = _ResourceAdmissionController(states, resource_manager)

    _update_controller(controller)

    assert controller.get_grant(first) == ResourceAdmissionGrant(2, True)
    assert controller.get_grant(second) == ResourceAdmissionGrant(3, True)


def test_whole_unit_redistribution_excludes_targetless_owner():
    targetless = _mock_resource_admission_op(cpu=0, gpu=1, internal_demand=True)
    first = _mock_resource_admission_op(cpu=0, gpu=1, internal_demand=True)
    second = _mock_resource_admission_op(cpu=0, gpu=1, internal_demand=True)
    states = {
        op: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1))
        for op in (targetless, first, second)
    }
    limits = ExecutionResources(gpu=4)
    resource_manager = _mock_admission_resource_manager(limits)
    targets = {
        targetless: None,
        first: ExecutionResources(gpu=1.5),
        second: ExecutionResources(gpu=1.5),
    }
    resource_manager.get_allocation_target.side_effect = targets.__getitem__
    controller = _ResourceAdmissionController(states, resource_manager)

    _update_controller(controller)

    assert controller.get_grant(targetless) == ResourceAdmissionGrant(1, True)
    assert controller.get_grant(first) == ResourceAdmissionGrant(1, True)
    assert controller.get_grant(second) == ResourceAdmissionGrant(2, True)


def test_memory_limits_whole_unit_allocation():
    owner = _mock_resource_admission_op(
        cpu=1,
        gpu=1,
        memory=2,
        max_units=3,
        internal_demand=True,
    )
    states = {owner: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1))}
    limits = ExecutionResources(cpu=3, gpu=3, memory=5)
    resource_manager = _mock_admission_resource_manager(limits)
    resource_manager.get_allocation_target.return_value = limits
    controller = _ResourceAdmissionController(states, resource_manager)

    _update_controller(controller)

    assert controller.get_grant(owner) == ResourceAdmissionGrant(2, True)


def test_unit_capacity_does_not_round_genuine_shortfall():
    capacity = _ResourceAdmissionController._unit_capacity(
        ExecutionResources(gpu=0.29999),
        ExecutionResources(gpu=0.1),
    )

    assert 2 < capacity < 3


def test_whole_unit_redistribution_uses_smaller_zero_remainder_owner():
    small_units = _mock_resource_admission_op(
        cpu=0,
        gpu=1,
        max_units=3,
        internal_demand=True,
    )
    large_units = _mock_resource_admission_op(
        cpu=0,
        gpu=2,
        max_units=3,
        internal_demand=True,
    )
    states = {
        op: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1))
        for op in (small_units, large_units)
    }
    limits = ExecutionResources(gpu=5)
    resource_manager = _mock_admission_resource_manager(limits)
    targets = {
        small_units: ExecutionResources(gpu=2),
        large_units: ExecutionResources(gpu=3),
    }
    resource_manager.get_allocation_target.side_effect = targets.__getitem__
    controller = _ResourceAdmissionController(states, resource_manager)

    controller.update_admission(limits)
    controller.update_allocation_grants(limits)

    assert controller.get_grant(small_units) == ResourceAdmissionGrant(3, True)
    assert controller.get_grant(large_units) == ResourceAdmissionGrant(1, True)


def test_whole_unit_redistribution_can_pool_multiple_units_to_one_owner():
    small_units = _mock_resource_admission_op(
        cpu=0,
        gpu=1,
        max_units=4,
        internal_demand=True,
    )
    large_units = _mock_resource_admission_op(
        cpu=0,
        gpu=3,
        max_units=4,
        internal_demand=True,
    )
    states = {
        op: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1))
        for op in (small_units, large_units)
    }
    limits = ExecutionResources(gpu=7)
    resource_manager = _mock_admission_resource_manager(limits)
    targets = {
        small_units: ExecutionResources(gpu=2.5),
        large_units: ExecutionResources(gpu=4.5),
    }
    resource_manager.get_allocation_target.side_effect = targets.__getitem__
    controller = _ResourceAdmissionController(states, resource_manager)

    controller.update_admission(limits)
    controller.update_allocation_grants(limits)

    assert controller.get_grant(small_units) == ResourceAdmissionGrant(4, True)
    assert controller.get_grant(large_units) == ResourceAdmissionGrant(1, True)


def test_whole_unit_redistribution_skips_globally_constrained_owner():
    first = _mock_resource_admission_op(internal_demand=True)
    constrained = _mock_resource_admission_op(internal_demand=True)
    states = {
        op: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1))
        for op in (first, constrained)
    }
    limits = ExecutionResources(cpu=3, gpu=3)
    resource_manager = _mock_admission_resource_manager(limits)
    resource_manager.get_allocation_target.return_value = ExecutionResources(
        cpu=2, gpu=2
    )
    controller = _ResourceAdmissionController(states, resource_manager)

    controller.update_admission(limits)
    controller.update_allocation_grants(limits)

    assert controller.get_grant(first) == ResourceAdmissionGrant(2, True)
    assert controller.get_grant(constrained) == ResourceAdmissionGrant(1, True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
