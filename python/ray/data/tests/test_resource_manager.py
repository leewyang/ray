import math
import time
from dataclasses import replace
from datetime import timedelta
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, call, patch

import pytest
from freezegun import freeze_time

import ray
from ray.data._internal.compute import ComputeStrategy
from ray.data._internal.execution.block_ref_counter import BlockRefCounter
from ray.data._internal.execution.interfaces import (
    BlockEntry,
    PhysicalOperator,
    ResourceAdmissionGrant,
    ResourceAdmissionSpec,
)
from ray.data._internal.execution.interfaces.execution_options import (
    ExecutionOptions,
    ExecutionResources,
)
from ray.data._internal.execution.interfaces.physical_operator import (
    ObjectStoreUsage,
    TaskExecDriverStats,
)
from ray.data._internal.execution.operators.base_physical_operator import (
    AllToAllOperator,
)
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.join import JoinOperator
from ray.data._internal.execution.operators.limit_operator import LimitOperator
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.union_operator import UnionOperator
from ray.data._internal.execution.resource_admission_controller import (
    _ResourceAdmissionController,
)
from ray.data._internal.execution.resource_manager import (
    OpResourceAllocator,
    ReservationOpResourceAllocator,
    ResourceManager,
    create_resource_allocator,
)
from ray.data._internal.execution.streaming_executor_state import (
    IdleDetector,
    OutputBackpressureGuard,
    build_streaming_topology,
)
from ray.data._internal.execution.util import make_ref_bundles
from ray.data.block import TaskExecWorkerStats
from ray.data.context import DataContext
from ray.data.tests.conftest import *  # noqa
from ray.data.tests.conftest import noop_counter


def mock_map_op(
    input_op: PhysicalOperator,
    ray_remote_args: Optional[Dict[str, Any]] = None,
    compute_strategy: Optional[ComputeStrategy] = None,
    name="Map",
):
    op = MapOperator.create(
        MagicMock(),
        input_op,
        DataContext.get_current(),
        ray_remote_args=ray_remote_args or {},
        compute_strategy=compute_strategy,
        name=name,
    )
    return op


def mock_union_op(input_ops):
    op = UnionOperator(
        DataContext.get_current(),
        *input_ops,
    )
    op.start = MagicMock(side_effect=lambda *_: None)
    return op


def mock_join_op(left_input_op, right_input_op):
    left_input_op._logical_operators = [MagicMock()]
    right_input_op._logical_operators = [MagicMock()]

    with patch(
        "ray.data._internal.execution.operators.hash_shuffle._get_total_cluster_resources"
    ) as mock:
        mock.return_value = ExecutionResources(cpu=1)

        op = JoinOperator(
            DataContext.get_current(),
            left_input_op,
            right_input_op,
            ("id",),
            ("id",),
            "inner",
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


def _mock_allocator_op(minimum, maximum):
    op = MagicMock()
    op.throttling_disabled.return_value = False
    op.has_execution_finished.return_value = False
    op.min_max_resource_requirements.return_value = (minimum, maximum)
    op.min_scheduling_resources.return_value = minimum
    return op


def test_resource_manager_admission_and_allocation_target_accessors():
    op = MagicMock(spec=PhysicalOperator)
    manager = object.__new__(ResourceManager)
    manager._resource_admission_controller = MagicMock()
    grant = ResourceAdmissionGrant(1, True)
    floor = ExecutionResources(gpu=1)
    manager._resource_admission_controller.get_grant.return_value = grant
    manager._resource_admission_controller.get_floor.return_value = floor
    manager._op_resource_allocator = None
    op.can_add_input.return_value = True

    assert manager.get_resource_admission_grant(op) == grant
    assert manager.get_admission_floor(op) == floor
    assert manager.can_submit_new_task(op)
    assert manager.get_allocation(op) is None
    assert manager.get_allocation_target(op) is None

    allocator = MagicMock()
    target = ExecutionResources(gpu=2)
    allocator.can_submit_new_task.return_value = False
    allocator.get_allocation_target.return_value = target
    manager._op_resource_allocator = allocator
    assert not manager.can_submit_new_task(op)
    allocator.can_submit_new_task.assert_called_once_with(op)
    assert manager.get_allocation_target(op) == target
    assert OpResourceAllocator.get_allocation_target(MagicMock(), op) is None

    reservation = object.__new__(ReservationOpResourceAllocator)
    reservation._op_budgets = {}
    assert reservation.get_allocation(op) is None


def test_resource_manager_updates_admission_without_optional_allocator():
    manager = object.__new__(ResourceManager)
    manager._resource_admission_controller = MagicMock()
    manager._resource_admission_controller.has_participants.return_value = True
    manager._op_resource_allocator = None
    limits = ExecutionResources(cpu=1, gpu=1)
    manager.get_global_limits = MagicMock(return_value=limits)
    manager._get_completed_ops_usage = MagicMock(return_value=ExecutionResources.zero())

    manager._update_resource_allocations()

    manager._resource_admission_controller.update_admission.assert_called_once_with(
        limits
    )
    manager._resource_admission_controller.update_allocation_grants.assert_called_once_with(
        limits
    )


def test_resource_manager_updates_admission_before_allocator_targets():
    manager = object.__new__(ResourceManager)
    manager._resource_admission_controller = MagicMock()
    manager._resource_admission_controller.has_participants.return_value = True
    manager._op_resource_allocator = MagicMock()
    limits = ExecutionResources(cpu=2, gpu=2)
    completed = ExecutionResources(cpu=1)
    available = ExecutionResources(cpu=1, gpu=2)
    manager.get_global_limits = MagicMock(return_value=limits)
    manager._get_completed_ops_usage = MagicMock(return_value=completed)

    ordered = MagicMock()
    ordered.attach_mock(
        manager._resource_admission_controller.update_admission, "admission"
    )
    ordered.attach_mock(manager._op_resource_allocator.update_budgets, "budgets")
    ordered.attach_mock(
        manager._resource_admission_controller.update_allocation_grants, "allocation"
    )

    manager._update_resource_allocations()

    assert ordered.mock_calls == [
        call.admission(available),
        call.budgets(limits=available),
        call.allocation(available),
    ]


def test_completed_sticky_floor_is_not_double_counted():
    completed = _mock_resource_admission_op(
        completed=True,
        internal_demand=True,
        cpu=0,
        gpu=1,
        max_units=1,
    )
    downstream = _mock_resource_admission_op(
        inputs=[completed],
        internal_demand=True,
        cpu=0,
        gpu=1,
        max_units=1,
    )
    completed.output_dependencies = [downstream]
    downstream.output_dependencies = []
    for op in (completed, downstream):
        op.throttling_disabled.return_value = False

    topology = {
        completed: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=0)),
        downstream: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
    }
    manager = object.__new__(ResourceManager)
    manager._topology = topology
    manager._options = ExecutionOptions()
    manager._op_resource_allocator = None
    manager._op_usages = {
        completed: ExecutionResources(gpu=1),
        downstream: ExecutionResources.zero(),
    }
    limits = ExecutionResources(gpu=2)
    manager.get_global_limits = MagicMock(return_value=limits)
    controller = _ResourceAdmissionController(topology, manager)
    manager._resource_admission_controller = controller
    controller._set_grant(completed, 1, may_submit=True)

    manager._update_resource_allocations()

    assert controller.get_grant(completed) == ResourceAdmissionGrant(1, True)
    assert controller.get_grant(downstream) == ResourceAdmissionGrant(1, True)


def test_allocation_target_redistribution_exhausts_finite_capacities():
    op = MagicMock(spec=PhysicalOperator)
    op.min_max_resource_requirements.return_value = (
        ExecutionResources.zero(),
        ExecutionResources.zero(),
    )
    allocator = object.__new__(ReservationOpResourceAllocator)
    allocator._op_reserved = {op: ExecutionResources.zero()}
    allocator._reserved_for_op_outputs = {op: 0}
    allocator._total_shared = ExecutionResources(gpu=1)

    allocator._update_allocation_targets()

    assert allocator.get_allocation_target(op) == ExecutionResources.zero()


@pytest.mark.parametrize(
    "ray_remote_args,ray_remote_args_fn,expected",
    [
        ({"num_gpus": 1}, None, False),
        ({"num_cpus": 1}, lambda: {"num_gpus": 1}, True),
        ({"num_cpus": 1}, None, False),
    ],
)
def test_task_pool_resource_admission_compatibility(
    ray_remote_args, ray_remote_args_fn, expected
):
    data_context = DataContext.get_current()
    source = InputDataBuffer(data_context, input_data=[])
    task_pool = MapOperator.create(
        MagicMock(),
        source,
        data_context,
        ray_remote_args=ray_remote_args,
        ray_remote_args_fn=ray_remote_args_fn,
    )

    assert task_pool.resource_admission_incompatible() is expected


def test_task_pool_internal_remote_args_fn_remains_admission_compatible():
    data_context = DataContext.get_current()
    source = InputDataBuffer(data_context, input_data=[])
    task_pool = MapOperator.create(
        MagicMock(),
        source,
        data_context,
        ray_remote_args={"num_cpus": 1},
    )

    task_pool._ray_remote_args_fn = lambda: {"memory": 1024}

    assert not task_pool.resource_admission_incompatible()


@pytest.mark.parametrize(
    "grant", [ResourceAdmissionGrant(0, False), ResourceAdmissionGrant(1, False)]
)
def test_non_admitted_resource_grant_blocks_task_submission(grant):
    resource_manager = MagicMock(spec=ResourceManager)
    resource_manager.get_resource_admission_grant.return_value = grant
    resource_manager._op_resource_allocator = None

    assert not ResourceManager.can_submit_new_task(resource_manager, MagicMock())


def test_admitted_resource_floors_precede_default_sharing():
    gpu_minimum = ExecutionResources(cpu=1, gpu=1)
    gpu_maximum = ExecutionResources(cpu=4, gpu=4, object_store_memory=float("inf"))
    upstream_gpu = _mock_allocator_op(gpu_minimum, gpu_maximum)
    cpu_middle = _mock_allocator_op(ExecutionResources(cpu=1), ExecutionResources.inf())
    downstream_gpu = _mock_allocator_op(gpu_minimum, gpu_maximum)
    resource_manager = MagicMock()
    resource_manager._topology = {
        upstream_gpu: MagicMock(),
        cpu_middle: MagicMock(),
        downstream_gpu: MagicMock(),
    }
    resource_manager.get_resource_admission_grant.side_effect = lambda op: (
        ResourceAdmissionGrant(1, True)
        if op in (upstream_gpu, downstream_gpu)
        else None
    )
    resource_manager.get_admission_floor.side_effect = lambda op: (
        ExecutionResources(cpu=1, gpu=1)
        if op in (upstream_gpu, downstream_gpu)
        else None
    )
    resource_manager.get_mem_op_internal.return_value = 0
    resource_manager.get_mem_op_outputs.return_value = 0
    resource_manager.get_op_usage.return_value = ExecutionResources.zero()
    resource_manager._is_blocking_materializing_op.return_value = False
    allocator = ReservationOpResourceAllocator(resource_manager, 0.5)

    allocator.update_budgets(
        limits=ExecutionResources(cpu=8, gpu=2, object_store_memory=1_000)
    )

    assert allocator.get_allocation(upstream_gpu).gpu >= 1
    assert allocator.get_allocation(downstream_gpu).gpu >= 1
    assert allocator.get_allocation(cpu_middle).gpu == 0


def test_admitted_sticky_floor_tolerates_refreshed_capacity_shortfall():
    op = _mock_allocator_op(
        ExecutionResources(cpu=2, gpu=2),
        ExecutionResources(cpu=2, gpu=2, object_store_memory=float("inf")),
    )
    resource_manager = MagicMock()
    resource_manager._topology = {op: MagicMock()}
    resource_manager.get_resource_admission_grant.return_value = ResourceAdmissionGrant(
        1, True
    )
    resource_manager.get_admission_floor.return_value = ExecutionResources(cpu=2, gpu=2)
    resource_manager.get_mem_op_internal.return_value = 0
    resource_manager.get_mem_op_outputs.return_value = 0
    resource_manager.get_op_usage.return_value = ExecutionResources.zero()
    resource_manager._is_blocking_materializing_op.return_value = False
    allocator = ReservationOpResourceAllocator(resource_manager, 0.5)

    allocator.update_budgets(
        limits=ExecutionResources(cpu=1, gpu=1, object_store_memory=1_000)
    )

    assert allocator.get_allocation(op).gpu >= 2


def test_resource_admission_allocation_target_shrinks_for_new_claimant():
    maximum = ExecutionResources(
        cpu=float("inf"),
        gpu=4,
        memory=float("inf"),
        object_store_memory=float("inf"),
    )
    upstream = _mock_allocator_op(ExecutionResources(gpu=1), maximum)
    downstream = _mock_allocator_op(ExecutionResources(gpu=1), maximum)
    admission_grants = {
        upstream: ResourceAdmissionGrant(1, True),
        downstream: ResourceAdmissionGrant(0, False),
    }
    usages = {
        # The upstream pool has already grown to consume the whole GPU budget.
        upstream: ExecutionResources(gpu=2),
        downstream: ExecutionResources.zero(),
    }
    resource_manager = MagicMock()
    resource_manager._topology = {upstream: MagicMock(), downstream: MagicMock()}
    resource_manager.get_resource_admission_grant.side_effect = admission_grants.get
    resource_manager.get_admission_floor.return_value = ExecutionResources(gpu=1)
    resource_manager.get_mem_op_internal.return_value = 0
    resource_manager.get_mem_op_outputs.return_value = 0
    resource_manager.get_op_usage.side_effect = usages.get
    resource_manager._is_blocking_materializing_op.return_value = False
    allocator = ReservationOpResourceAllocator(resource_manager, 0.5)
    limits = ExecutionResources(gpu=2, object_store_memory=1_000)

    allocator.update_budgets(limits=limits)
    assert allocator.get_allocation_target(upstream).gpu == pytest.approx(2)

    # A newly demanded downstream pool receives its protected one-actor floor.
    # The upstream's current usage remains 2, but its independent target falls
    # to 1 so the autoscaler can reclaim an idle actor.
    admission_grants[downstream] = ResourceAdmissionGrant(1, True)
    allocator.update_budgets(limits=limits)
    assert allocator.get_allocation(upstream).gpu == pytest.approx(2)
    assert allocator.get_allocation_target(upstream).gpu == pytest.approx(1)
    assert allocator.get_allocation_target(downstream).gpu == pytest.approx(1)


def test_fixed_gang_does_not_dilute_elastic_allocation_target():
    def make_op(max_gpus):
        return _mock_allocator_op(
            ExecutionResources(gpu=1),
            ExecutionResources(
                cpu=float("inf"),
                gpu=max_gpus,
                memory=float("inf"),
                object_store_memory=float("inf"),
            ),
        )

    gang = make_op(max_gpus=1)
    elastic = make_op(max_gpus=4)
    resource_manager = MagicMock()
    resource_manager._topology = {gang: MagicMock(), elastic: MagicMock()}
    resource_manager.get_resource_admission_grant.return_value = ResourceAdmissionGrant(
        1, True
    )
    resource_manager.get_admission_floor.return_value = ExecutionResources(gpu=1)
    resource_manager.get_mem_op_internal.return_value = 0
    resource_manager.get_mem_op_outputs.return_value = 0
    resource_manager.get_op_usage.return_value = ExecutionResources.zero()
    resource_manager._is_blocking_materializing_op.return_value = False
    allocator = ReservationOpResourceAllocator(resource_manager, 0)

    allocator.update_budgets(
        limits=ExecutionResources(gpu=4, object_store_memory=1_000)
    )

    assert allocator.get_allocation_target(gang).gpu == pytest.approx(1)
    assert allocator.get_allocation_target(elastic).gpu == pytest.approx(3)


def test_partially_capped_pool_does_not_strand_allocation_target():
    def make_op(max_gpus):
        return _mock_allocator_op(
            ExecutionResources(gpu=1),
            ExecutionResources(
                cpu=float("inf"),
                gpu=max_gpus,
                memory=float("inf"),
                object_store_memory=float("inf"),
            ),
        )

    capped = make_op(max_gpus=2)
    elastic = make_op(max_gpus=6)
    resource_manager = MagicMock()
    resource_manager._topology = {capped: MagicMock(), elastic: MagicMock()}
    resource_manager.get_resource_admission_grant.return_value = ResourceAdmissionGrant(
        1, True
    )
    resource_manager.get_admission_floor.return_value = ExecutionResources(gpu=1)
    resource_manager.get_mem_op_internal.return_value = 0
    resource_manager.get_mem_op_outputs.return_value = 0
    resource_manager.get_op_usage.return_value = ExecutionResources.zero()
    resource_manager._is_blocking_materializing_op.return_value = False
    allocator = ReservationOpResourceAllocator(resource_manager, 0)

    allocator.update_budgets(
        limits=ExecutionResources(gpu=6, object_store_memory=1_000)
    )

    assert allocator.get_allocation_target(capped).gpu == pytest.approx(2)
    assert allocator.get_allocation_target(elastic).gpu == pytest.approx(4)


def mock_all_to_all_op(input_op, name="MockShuffle"):
    """Create a mock AllToAllOperator (shuffle) for testing."""
    op = AllToAllOperator(
        bulk_fn=MagicMock(),
        input_op=input_op,
        data_context=DataContext.get_current(),
        name=name,
    )
    return op


def _configure_admission_op_for_allocator(op):
    spec = op.resource_admission_spec()
    op.throttling_disabled.return_value = False
    maximum = (
        spec.unit_resources.scale(spec.max_units).copy(object_store_memory=float("inf"))
        if spec.max_units is not None
        else ExecutionResources.inf()
    )
    op.min_max_resource_requirements.return_value = (
        spec.minimum_resources,
        maximum,
    )
    op.min_scheduling_resources.return_value = spec.minimum_resources


def _mock_resource_manager_for_admission_allocator(topology, limits):
    resource_manager = MagicMock()
    resource_manager._topology = topology
    resource_manager._options = ExecutionOptions()
    resource_manager.get_global_limits.return_value = limits
    resource_manager.get_mem_op_internal.return_value = 0
    resource_manager.get_mem_op_outputs.return_value = 0
    resource_manager.get_op_usage.return_value = ExecutionResources.zero()
    resource_manager._is_blocking_materializing_op.return_value = False
    resource_manager.op_resource_allocator_enabled.return_value = True
    return resource_manager


def test_admitted_floor_after_pending_materializer_reduces_shared_target():
    upstream = _mock_resource_admission_op(internal_demand=True)
    downstream_branch = _mock_resource_admission_op(internal_demand=True)
    for op in (upstream, downstream_branch):
        _configure_admission_op_for_allocator(op)

    pending_materializer = mock_all_to_all_op(
        InputDataBuffer(DataContext.get_current(), [])
    )
    pending_materializer.has_completed = MagicMock(return_value=False)
    pending_materializer.throttling_disabled = MagicMock(return_value=True)
    topology = {
        upstream: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
        pending_materializer: MagicMock(),
        downstream_branch: MagicMock(
            total_enqueued_input_blocks=MagicMock(return_value=1)
        ),
    }
    limits = ExecutionResources(cpu=2, gpu=2, object_store_memory=1_000)
    resource_manager = _mock_resource_manager_for_admission_allocator(topology, limits)
    controller = _ResourceAdmissionController(topology, resource_manager)
    allocator = ReservationOpResourceAllocator(resource_manager, 0.5)
    resource_manager.get_resource_admission_grant.side_effect = controller.get_grant
    resource_manager.get_admission_floor.side_effect = controller.get_floor
    resource_manager.get_allocation_target.side_effect = allocator.get_allocation_target

    controller.update_admission(limits)
    allocator.update_budgets(limits=limits)
    controller.update_allocation_grants(limits)

    # Task budgets remain restricted to operators before the pending materializer,
    # but the independent branch's admitted floor is still globally protected.
    assert upstream in allocator._op_budgets
    assert downstream_branch not in allocator._op_budgets
    assert allocator.get_allocation_target(upstream).gpu == pytest.approx(1)
    assert controller.get_grant(upstream) == ResourceAdmissionGrant(1, True)
    assert controller.get_grant(downstream_branch) == ResourceAdmissionGrant(1, True)
    assert (
        sum(controller.get_grant(op).max_units for op in (upstream, downstream_branch))
        == 2
    )


def test_fractional_targets_redistribute_complete_elastic_unit():
    upstream = _mock_resource_admission_op(internal_demand=True)
    downstream = _mock_resource_admission_op(internal_demand=True)
    for op in (upstream, downstream):
        _configure_admission_op_for_allocator(op)

    topology = {
        upstream: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
        downstream: MagicMock(total_enqueued_input_blocks=MagicMock(return_value=1)),
    }
    limits = ExecutionResources(cpu=3, gpu=3, object_store_memory=1_000)
    resource_manager = _mock_resource_manager_for_admission_allocator(topology, limits)
    controller = _ResourceAdmissionController(topology, resource_manager)
    allocator = ReservationOpResourceAllocator(resource_manager, 0.5)
    resource_manager.get_resource_admission_grant.side_effect = controller.get_grant
    resource_manager.get_admission_floor.side_effect = controller.get_floor
    resource_manager.get_allocation_target.side_effect = allocator.get_allocation_target

    controller.update_admission(limits)
    allocator.update_budgets(limits=limits)
    assert allocator.get_allocation_target(upstream).gpu == pytest.approx(1.5)
    assert allocator.get_allocation_target(downstream).gpu == pytest.approx(1.5)

    controller.update_allocation_grants(limits)

    grants = [controller.get_grant(op).max_units for op in (upstream, downstream)]
    assert grants == [1, 2]

    # Capacity beyond the aggregate participant targets may be reserved for an
    # unmanaged operator. Do not consume it merely because the global limit grew.
    controller.update_allocation_grants(
        ExecutionResources(cpu=4, gpu=4, object_store_memory=1_000)
    )
    assert [controller.get_grant(op).max_units for op in (upstream, downstream)] == [
        1,
        2,
    ]


def _resource_manager_for_limits_only_test(
    options: ExecutionOptions,
    get_total_resources,
):
    """``ResourceManager`` requires a valid single-sink topology; these tests only
    call ``get_global_limits()`` and never iterate real operators."""
    sink = MagicMock(spec=PhysicalOperator)
    sink.output_dependencies = []
    sink.resource_admission_spec.return_value = None
    sink.min_scheduling_resources.return_value = ExecutionResources.zero()
    topology = {sink: MagicMock()}
    return ResourceManager(
        topology,
        options,
        get_total_resources,
        DataContext.get_current(),
        BlockRefCounter(add_object_out_of_scope_callback=lambda *_: True),
    )


class TestResourceManager:
    """Unit tests for ResourceManager."""

    def test_global_limits(self):
        cluster_resources = {"CPU": 10, "GPU": 5, "object_store_memory": 1000}
        default_object_store_memory_limit = math.ceil(
            cluster_resources["object_store_memory"]
            * ResourceManager.DEFAULT_OBJECT_STORE_MEMORY_LIMIT_FRACTION
        )

        def get_total_resources():
            return ExecutionResources.from_resource_dict(cluster_resources)

        # Test default resource limits.
        # When no resource limits are set, the resource limits should default to
        # the cluster resources for CPU/GPU, and
        # DEFAULT_OBJECT_STORE_MEMORY_LIMIT_FRACTION of cluster object store memory.
        options = ExecutionOptions()
        resource_manager = _resource_manager_for_limits_only_test(
            options, get_total_resources
        )
        expected = ExecutionResources(
            cpu=cluster_resources["CPU"],
            gpu=cluster_resources["GPU"],
            object_store_memory=default_object_store_memory_limit,
        )
        assert resource_manager.get_global_limits() == expected

        # Test setting resource_limits
        options = ExecutionOptions()
        options.resource_limits = ExecutionResources(
            cpu=1, gpu=2, object_store_memory=100
        )
        resource_manager = _resource_manager_for_limits_only_test(
            options, get_total_resources
        )
        expected = ExecutionResources(
            cpu=1,
            gpu=2,
            object_store_memory=100,
        )
        assert resource_manager.get_global_limits() == expected

        # Test setting exclude_resources
        # The actual limit should be the default limit minus the excluded resources.
        options = ExecutionOptions()
        options.exclude_resources = ExecutionResources(
            cpu=1, gpu=2, object_store_memory=100
        )
        resource_manager = _resource_manager_for_limits_only_test(
            options, get_total_resources
        )
        expected = ExecutionResources(
            cpu=cluster_resources["CPU"] - 1,
            gpu=cluster_resources["GPU"] - 2,
            object_store_memory=default_object_store_memory_limit - 100,
        )
        assert resource_manager.get_global_limits() == expected

        # Test that we don't support setting both resource_limits
        # and exclude_resources.
        with pytest.raises(ValueError):
            options = ExecutionOptions()
            options.resource_limits = ExecutionResources(cpu=2)
            options.exclude_resources = ExecutionResources(cpu=1)
            options.validate()

    def test_global_limits_cache(self):
        get_total_resources = MagicMock(return_value=ExecutionResources(4, 1, 0))

        cache_interval_s = 0.1
        with patch.object(
            ResourceManager,
            "GLOBAL_LIMITS_UPDATE_INTERVAL_S",
            cache_interval_s,
        ):
            resource_manager = _resource_manager_for_limits_only_test(
                ExecutionOptions(),
                get_total_resources,
            )
            expected_resource = ExecutionResources(4, 1, 0)
            # The first call should call ray.cluster_resources().
            assert resource_manager.get_global_limits() == expected_resource
            assert get_total_resources.call_count == 1
            # The second call should return the cached value.
            assert resource_manager.get_global_limits() == expected_resource
            assert get_total_resources.call_count == 1
            time.sleep(cache_interval_s)
            # After the cache interval, the third call should call
            # ray.cluster_resources() again.
            assert resource_manager.get_global_limits() == expected_resource
            assert get_total_resources.call_count == 2

    def test_update_usage(self):
        """Test calculating op_usage."""
        o1 = InputDataBuffer(DataContext.get_current(), [])
        o2 = mock_map_op(o1)
        o3 = mock_map_op(o2)
        topo = build_streaming_topology(o3, ExecutionOptions(), noop_counter())

        # Mock different metrics that contribute to the resource usage.
        mock_cpu = {
            o1: 0,
            o2: 5,
            o3: 8,
        }
        mock_pending_task_outputs = {
            o1: 0,
            o2: 100,
            o3: 200,
        }
        mock_internal_outqueue = {
            o1: 0,
            o2: 300,
            o3: 400,
        }
        mock_external_outqueue_sizes = {
            o1: 100,
            o2: 500,
            o3: 600,
        }
        mock_internal_inqueue = {
            o1: 0,
            o2: 700,
            o3: 800,
        }
        mock_pending_task_inputs = {
            o1: 0,
            o2: 900,
            o3: 1000,
        }

        for op in [o1, o2, o3]:
            op.current_logical_usage = MagicMock(
                return_value=ExecutionResources(cpu=mock_cpu[op], gpu=0, memory=0)
            )
            op.running_logical_usage = MagicMock(
                return_value=ExecutionResources(cpu=mock_cpu[op], gpu=0, memory=0)
            )
            op.pending_logical_usage = MagicMock(return_value=ExecutionResources.zero())
            op.extra_resource_usage = MagicMock(return_value=ExecutionResources.zero())
            op._metrics = MagicMock(
                obj_store_mem_pending_task_outputs=mock_pending_task_outputs[op],
                obj_store_mem_internal_outqueue=mock_internal_outqueue[op],
                obj_store_mem_internal_inqueue=mock_internal_inqueue[op],
                obj_store_mem_pending_task_inputs=mock_pending_task_inputs[op],
            )
            op._metrics.obj_store_mem_internal_inqueue_for_input = MagicMock(
                return_value=mock_internal_inqueue[op],
            )
            ref_bundle = MagicMock(
                size_bytes=MagicMock(return_value=mock_external_outqueue_sizes[op]),
                output_split_idx=None,
            )
            topo[op].add_output(ref_bundle)

        resource_manager = ResourceManager(
            topo,
            ExecutionOptions(),
            MagicMock(),
            DataContext.get_current(),
            BlockRefCounter(add_object_out_of_scope_callback=lambda *_: True),
        )
        resource_manager._op_resource_allocator = None
        resource_manager.update_usages()

        global_cpu = 0
        global_mem = 0
        for op in [o1, o2, o3]:
            if op == o1:
                # Resource usage of InputDataBuffer doesn't count.
                expected_mem = 0
            else:
                expected_mem = (
                    mock_pending_task_outputs[op]
                    + mock_internal_outqueue[op]
                    + mock_external_outqueue_sizes[op]
                )
                for next_op in op.output_dependencies:
                    expected_mem += (
                        +mock_internal_inqueue[next_op]
                        + mock_pending_task_inputs[next_op]
                    )
            op_usage = resource_manager.get_op_usage(op)
            assert op_usage.cpu == mock_cpu[op]
            assert op_usage.gpu == 0
            assert op_usage.object_store_memory == expected_mem
            if op != o1:
                # _mem_op_internal only includes pending_task_outputs
                assert (
                    resource_manager._mem_op_internal[op]
                    == mock_pending_task_outputs[op]
                )
                assert (
                    resource_manager._mem_op_outputs[op]
                    == expected_mem - resource_manager._mem_op_internal[op]
                )
            global_cpu += mock_cpu[op]
            global_mem += expected_mem

        assert resource_manager.get_global_usage() == ExecutionResources(
            global_cpu, 0, global_mem
        )

    def test_object_store_usage(self, restore_data_context):
        input = make_ref_bundles([[x] for x in range(1)])[0]
        # Set block metadata size_bytes to 1 (rather than mocking the method on the
        # instance, which doesn't survive dataclasses.replace in OpBufferQueue.pop).
        entry = input.blocks[0]
        input = replace(
            input,
            blocks=[BlockEntry(entry.ref, replace(entry.metadata, size_bytes=1))],
        )

        o1 = InputDataBuffer(DataContext.get_current(), [input])
        o2 = mock_map_op(o1)
        o3 = mock_map_op(o2)

        topo = build_streaming_topology(o3, ExecutionOptions(), noop_counter())
        resource_manager = ResourceManager(
            topo,
            ExecutionOptions(),
            MagicMock(return_value=ExecutionResources.zero()),
            DataContext.get_current(),
            BlockRefCounter(add_object_out_of_scope_callback=lambda *_: True),
        )
        ray.data.DataContext.get_current()._max_num_blocks_in_streaming_gen_buffer = 1
        ray.data.DataContext.get_current().target_max_block_size = 2

        resource_manager.update_usages()
        assert resource_manager.get_op_usage(o1).object_store_memory == 0
        assert resource_manager.get_op_usage(o2).object_store_memory == 0
        assert resource_manager.get_op_usage(o3).object_store_memory == 0

        # Objects in an operator's internal inqueue typically count toward the previous
        # operator's object store memory usage. However, data from an
        # `InputDataBuffer` aren't counted because they were created outside of this
        # execution.
        o2.metrics.on_input_queued(input, input_index=0)
        resource_manager.update_usages()
        assert resource_manager.get_op_usage(o1).object_store_memory == 0
        assert resource_manager.get_op_usage(o2).object_store_memory == 0
        assert resource_manager.get_op_usage(o3).object_store_memory == 0

        # During no-sample phase, obj_store_mem_pending_task_outputs uses fallback
        # estimate based on target_max_block_size.
        o2.metrics.on_input_dequeued(input, input_index=0)
        o2.metrics.on_task_submitted(0, input)
        resource_manager.update_usages()
        assert resource_manager.get_op_usage(o1).object_store_memory == 0
        # No sample available yet, returns None
        assert o2.metrics.obj_store_mem_pending_task_outputs is None
        op2_usage = resource_manager.get_op_usage(o2).object_store_memory
        # When pending task outputs is None, it's treated as 0
        assert op2_usage == 0
        assert resource_manager.get_op_usage(o3).object_store_memory == 0

        # When the task finishes, we move the data from the streaming generator to the
        # operator's internal outqueue.
        o2.metrics.on_output_queued(input)
        o2.metrics.on_task_finished(
            0,
            None,
            TaskExecWorkerStats(task_wall_time_s=0.0),
            TaskExecDriverStats(task_output_backpressure_s=0),
        )
        resource_manager.update_usages()
        assert resource_manager.get_op_usage(o1).object_store_memory == 0
        assert resource_manager.get_op_usage(o2).object_store_memory == 1
        assert resource_manager.get_op_usage(o3).object_store_memory == 0

        o2.metrics.on_output_dequeued(input)
        topo[o2].output_queue.append(input)
        resource_manager.update_usages()
        assert resource_manager.get_op_usage(o1).object_store_memory == 0
        assert resource_manager.get_op_usage(o2).object_store_memory == 1
        assert resource_manager.get_op_usage(o3).object_store_memory == 0

        # Objects in the current operator's internal inqueue count towards the previous
        # operator's object store memory usage.
        # NOTE: `pop()` returns a copy of the bundle (via `dataclasses.replace`), so we
        # must use the returned reference for subsequent o3 metric calls.
        o3_input = topo[o2].output_queue.pop()
        o3.metrics.on_input_queued(o3_input, input_index=0)
        resource_manager.update_usages()
        assert resource_manager.get_op_usage(o1).object_store_memory == 0
        assert resource_manager.get_op_usage(o2).object_store_memory == 1
        assert resource_manager.get_op_usage(o3).object_store_memory == 0

        # Task inputs count toward the previous operator's object store memory
        # usage. During no-sample phase, pending task outputs uses fallback estimate.
        o3.metrics.on_input_dequeued(o3_input, input_index=0)
        o3.metrics.on_task_submitted(0, o3_input)
        resource_manager.update_usages()
        assert resource_manager.get_op_usage(o1).object_store_memory == 0
        assert resource_manager.get_op_usage(o2).object_store_memory == 1
        # No sample available yet, returns None
        assert o3.metrics.obj_store_mem_pending_task_outputs is None
        op3_usage = resource_manager.get_op_usage(o3).object_store_memory
        # When pending task outputs is None, it's treated as 0
        assert op3_usage == 0

        # Task inputs no longer count once the task is finished.
        o3.metrics.on_output_queued(o3_input)
        o3.metrics.on_task_finished(
            0,
            None,
            TaskExecWorkerStats(task_wall_time_s=0.0),
            TaskExecDriverStats(task_output_backpressure_s=0),
        )
        resource_manager.update_usages()
        assert resource_manager.get_op_usage(o1).object_store_memory == 0
        assert resource_manager.get_op_usage(o2).object_store_memory == 0
        assert resource_manager.get_op_usage(o3).object_store_memory == 1

    def test_object_store_accounting_delegates_to_op(self, restore_data_context):
        """``ResourceManager`` must dispatch to ``op.estimate_object_store_usage`` so subclasses can override the accounting."""
        # Real upstream so the override op has a valid input dependency.
        input = make_ref_bundles([[x] for x in range(1)])[0]
        upstream = InputDataBuffer(DataContext.get_current(), [input])

        # Subclass that overrides the accounting to return hard-coded
        # values — bypasses the generic metrics+state computation.
        override = mock_map_op(upstream)
        override.estimate_object_store_usage = lambda state: ObjectStoreUsage(
            internal=42, outputs=100
        )

        topo = build_streaming_topology(override, ExecutionOptions(), noop_counter())
        resource_manager = ResourceManager(
            topo,
            ExecutionOptions(),
            MagicMock(return_value=ExecutionResources.zero()),
            DataContext.get_current(),
            BlockRefCounter(add_object_out_of_scope_callback=lambda *_: True),
        )

        resource_manager.update_usages()

        # The override's hard-coded values flow through unchanged into
        # both the per-component dicts and the aggregated op usage.
        assert resource_manager.get_mem_op_internal(override) == 42
        assert resource_manager.get_mem_op_outputs(override) == 100
        assert resource_manager.get_op_usage(override).object_store_memory == 42 + 100

    def test_get_completed_ops_usage(self, restore_data_context):
        """Test that _get_completed_ops_usage returns total usage of completed ops."""
        o1 = InputDataBuffer(DataContext.get_current(), [])
        o2 = mock_map_op(o1)
        o3 = LimitOperator(1, o2, DataContext.get_current())
        o4 = mock_map_op(o3)
        o5 = mock_map_op(o4)

        topo = build_streaming_topology(o5, ExecutionOptions(), noop_counter())

        o1.mark_execution_finished()
        o2.mark_execution_finished()

        op_usages = {
            o1: ExecutionResources.zero(),
            o2: ExecutionResources(cpu=2, object_store_memory=50),
            o3: ExecutionResources(cpu=1, object_store_memory=25),
            o4: ExecutionResources.zero(),
            o5: ExecutionResources.zero(),
        }

        resource_manager = ResourceManager(
            topo,
            ExecutionOptions(),
            MagicMock(),
            DataContext.get_current(),
            BlockRefCounter(add_object_out_of_scope_callback=lambda *_: True),
        )
        resource_manager.get_op_usage = MagicMock(side_effect=lambda op: op_usages[op])

        # o2 is completed and o3 is downstream ineligible (LimitOperator)
        # Total usage should be o2 + o3
        completed_ops_usage = resource_manager._get_completed_ops_usage()
        assert completed_ops_usage == ExecutionResources(cpu=3, object_store_memory=75)

    def test_get_completed_ops_usage_complex_graph(self, restore_data_context):
        """
        o1 (InputDataBuffer)
                |
                v
                o2 (MapOperator, completed)
                |
                v
                o3 (LimitOperator)
                |
                v                    o4 (InputDataBuffer)
                |                    |
                |                    v
                |                    o5 (MapOperator, completed)
                |                    |
                v                    v
                o6 (UnionOperator) <--
                |
                v
                o8 (JoinOperator) <-- o7 (InputDataBuffer, completed)
        """
        o1 = InputDataBuffer(DataContext.get_current(), [])
        o2 = mock_map_op(o1)
        o3 = LimitOperator(1, o2, DataContext.get_current())
        o4 = InputDataBuffer(DataContext.get_current(), [])
        o5 = mock_map_op(o4)
        o6 = mock_union_op([o3, o5])
        o7 = InputDataBuffer(DataContext.get_current(), [])
        o8 = mock_join_op(o7, o6)

        topo = build_streaming_topology(o8, ExecutionOptions(), noop_counter())

        o1.mark_execution_finished()
        o2.mark_execution_finished()
        o4.mark_execution_finished()
        o5.mark_execution_finished()
        o7.mark_execution_finished()

        op_usages = {
            o1: ExecutionResources.zero(),
            o2: ExecutionResources(cpu=2, object_store_memory=150),
            o3: ExecutionResources(cpu=2, object_store_memory=50),
            o4: ExecutionResources.zero(),
            o5: ExecutionResources(cpu=3, object_store_memory=100),
            o6: ExecutionResources.zero(),
            o7: ExecutionResources(cpu=1, object_store_memory=100),
            o8: ExecutionResources.zero(),
        }

        resource_manager = ResourceManager(
            topo,
            ExecutionOptions(),
            MagicMock(),
            DataContext.get_current(),
            BlockRefCounter(add_object_out_of_scope_callback=lambda *_: True),
        )
        resource_manager.get_op_usage = MagicMock(side_effect=lambda op: op_usages[op])

        # Completed ops: o2, o5, o7
        # Downstream ineligible: o3 (LimitOperator after o2)
        # Total usage should be o2 + o3 + o5 + o7
        completed_ops_usage = resource_manager._get_completed_ops_usage()

        assert completed_ops_usage == ExecutionResources(cpu=8, object_store_memory=400)

    def test_external_consumer_bytes_attributed_to_terminal_operator(
        self, restore_data_context
    ):
        """External consumer bytes (e.g., iterator prefetch buffers) are charged
        to the terminal operator's object store usage, not as a global deduction."""
        cluster_resources = ExecutionResources(cpu=10, gpu=0, object_store_memory=1000)

        o1 = InputDataBuffer(DataContext.get_current(), [])
        o2 = mock_map_op(o1)
        o3 = mock_map_op(o2)

        topo = build_streaming_topology(o3, ExecutionOptions(), noop_counter())

        o1.mark_execution_finished()
        o2.mark_execution_finished()

        resource_manager = ResourceManager(
            topo,
            ExecutionOptions(),
            lambda: cluster_resources,
            DataContext.get_current(),
            BlockRefCounter(add_object_out_of_scope_callback=lambda *_: True),
        )

        for op in [o1, o2, o3]:
            op.current_logical_usage = MagicMock(return_value=ExecutionResources.zero())
            op.running_logical_usage = MagicMock(return_value=ExecutionResources.zero())
            op.pending_logical_usage = MagicMock(return_value=ExecutionResources.zero())

        assert resource_manager._op_resource_allocator is not None

        resource_manager.update_usages()
        baseline_terminal = resource_manager.get_op_usage(o3).object_store_memory
        baseline_upstream = resource_manager.get_op_usage(o2).object_store_memory

        def _available_pool_object_store():
            return (
                resource_manager.get_global_limits()
                .subtract(resource_manager._get_completed_ops_usage())
                .max(ExecutionResources.zero())
                .object_store_memory
            )

        pool_before = _available_pool_object_store()

        resource_manager.set_external_consumer_bytes(200)
        resource_manager.update_usages()

        assert (
            resource_manager.get_op_usage(o3).object_store_memory
            == baseline_terminal + 200
        )
        assert (
            resource_manager.get_op_usage(o2).object_store_memory == baseline_upstream
        )
        assert _available_pool_object_store() == pool_before

        resource_manager.set_external_consumer_bytes(0)
        resource_manager.update_usages()
        assert (
            resource_manager.get_op_usage(o3).object_store_memory == baseline_terminal
        )

        # Very large external bytes: terminal usage reflects them; update still succeeds.
        resource_manager.set_external_consumer_bytes(999999)
        resource_manager.update_usages()
        assert (
            resource_manager.get_op_usage(o3).object_store_memory
            == baseline_terminal + 999999
        )

    def test_set_external_consumer_bytes_rejects_negative(self, restore_data_context):
        resource_manager = _resource_manager_for_limits_only_test(
            ExecutionOptions(),
            MagicMock(return_value=ExecutionResources.zero()),
        )
        with pytest.raises(AssertionError):
            resource_manager.set_external_consumer_bytes(-1)

    def test_external_consumer_bytes_input_data_buffer_sink(self, restore_data_context):
        """When the execute DAG is only an InputDataBuffer, prefetch bytes still
        attach to that terminal sink instead of being dropped by the
        InputDataBuffer early return."""
        buf = InputDataBuffer(DataContext.get_current(), [])
        topo = build_streaming_topology(buf, ExecutionOptions(), noop_counter())
        resource_manager = ResourceManager(
            topo,
            ExecutionOptions(),
            lambda: ExecutionResources(cpu=10, gpu=0, object_store_memory=1000),
            DataContext.get_current(),
            BlockRefCounter(add_object_out_of_scope_callback=lambda *_: True),
        )
        buf.current_logical_usage = MagicMock(return_value=ExecutionResources.zero())
        buf.running_logical_usage = MagicMock(return_value=ExecutionResources.zero())
        buf.pending_logical_usage = MagicMock(return_value=ExecutionResources.zero())

        resource_manager.update_usages()
        assert resource_manager.get_op_usage(buf).object_store_memory == 0

        resource_manager.set_external_consumer_bytes(150)
        resource_manager.update_usages()
        assert resource_manager.get_op_usage(buf).object_store_memory == 150

    def test_external_consumer_bytes_surfaced_in_op_usage_str(
        self, restore_data_context
    ):
        """The terminal operator's verbose usage string should include
        external_consumer=... when an external consumer is registered, so users
        can see how much of the operator's object-store memory is held by a
        downstream iterator vs. the operator's own queues."""
        cluster_resources = ExecutionResources(cpu=10, gpu=0, object_store_memory=1000)

        o1 = InputDataBuffer(DataContext.get_current(), [])
        o2 = mock_map_op(o1)
        o3 = mock_map_op(o2)

        topo = build_streaming_topology(o3, ExecutionOptions(), noop_counter())
        resource_manager = ResourceManager(
            topo,
            ExecutionOptions(),
            lambda: cluster_resources,
            DataContext.get_current(),
            BlockRefCounter(add_object_out_of_scope_callback=lambda *_: True),
        )

        for op in [o1, o2, o3]:
            op.current_logical_usage = MagicMock(return_value=ExecutionResources.zero())
            op.running_logical_usage = MagicMock(return_value=ExecutionResources.zero())
            op.pending_logical_usage = MagicMock(return_value=ExecutionResources.zero())

        resource_manager.update_usages()

        # No external consumer yet: nothing extra in the usage string.
        terminal_str = resource_manager.get_op_usage_str(o3, verbose=True)
        upstream_str = resource_manager.get_op_usage_str(o2, verbose=True)
        assert "external_consumer=" not in terminal_str
        assert "external_consumer=" not in upstream_str

        # Register an external consumer. Only the terminal operator's string
        # should pick up `external_consumer=...`.
        resource_manager.set_external_consumer_bytes(200)
        resource_manager.update_usages()
        terminal_str = resource_manager.get_op_usage_str(o3, verbose=True)
        upstream_str = resource_manager.get_op_usage_str(o2, verbose=True)
        assert "external_consumer=200.0B" in terminal_str
        assert "external_consumer=" not in upstream_str

        # The field is inside the existing `(in=...,out=...)` parenthetical.
        assert ",external_consumer=" in terminal_str

        # Non-verbose output omits the field (existing format unchanged).
        terminal_str_brief = resource_manager.get_op_usage_str(o3, verbose=False)
        assert "external_consumer=" not in terminal_str_brief

    def test_topology_rejects_multiple_terminal_operators(self, restore_data_context):
        ctx = DataContext.get_current()
        a = PhysicalOperator("a", [], ctx)
        b = PhysicalOperator("b", [], ctx)
        topology = {a: MagicMock(), b: MagicMock()}
        with pytest.raises(ValueError, match="Expected exactly one terminal operator"):
            ResourceManager(
                topology,
                ExecutionOptions(),
                MagicMock(return_value=ExecutionResources.zero()),
                DataContext.get_current(),
                BlockRefCounter(add_object_out_of_scope_callback=lambda *_: True),
            )

    def test_topology_rejects_empty_topology(self, restore_data_context):
        with pytest.raises(ValueError, match="topology must be non-empty"):
            ResourceManager(
                {},
                ExecutionOptions(),
                MagicMock(return_value=ExecutionResources.zero()),
                DataContext.get_current(),
                BlockRefCounter(add_object_out_of_scope_callback=lambda *_: True),
            )

    def test_topology_rejects_no_terminal_operator(self, restore_data_context):
        # Every op has a downstream in this dict, so there should be no operator with empty
        # output_dependencies (e.g. a 2-node cycle). Real streaming DAGs from
        # build_streaming_topology always have a unique sink.
        a = MagicMock(spec=PhysicalOperator)
        b = MagicMock(spec=PhysicalOperator)
        a.output_dependencies = [b]
        b.output_dependencies = [a]
        topology = {a: MagicMock(), b: MagicMock()}
        with pytest.raises(ValueError, match="No terminal operator found"):
            ResourceManager(
                topology,
                ExecutionOptions(),
                MagicMock(return_value=ExecutionResources.zero()),
                DataContext.get_current(),
                BlockRefCounter(add_object_out_of_scope_callback=lambda *_: True),
            )

    def test_is_blocking_materializing_op(self, restore_data_context):
        """Test _is_blocking_materializing_op correctly identifies blocking materializing ops.

        Cases tested:
        1. Operator itself is a blocking materializing op (AllToAllOperator) -> True
        2. Operator has downstream ineligible blocking materializing op -> True
        3. Operator with no downstream blocking materializing ops -> False

        Note: AllToAllOperator.throttling_disabled() returns True, making it
        ineligible for resource allocation. This means shuffle operators are
        always in the "downstream ineligible" chain from eligible operators.
        """
        # Build pipeline: o1 -> o2 -> o3 (limit) -> o4 (shuffle) -> o5
        o1 = InputDataBuffer(DataContext.get_current(), [])
        o2 = mock_map_op(o1, name="Map1")
        o3 = LimitOperator(1, o2, DataContext.get_current())
        o4 = mock_all_to_all_op(o3, name="Sort")
        o5 = mock_map_op(o4, name="Map2")

        topo = build_streaming_topology(o5, ExecutionOptions(), noop_counter())

        resource_manager = ResourceManager(
            topo,
            ExecutionOptions(),
            MagicMock(),
            DataContext.get_current(),
            BlockRefCounter(add_object_out_of_scope_callback=lambda *_: True),
        )

        # Case 1: Shuffle operator itself is blocking materializing
        assert resource_manager._is_blocking_materializing_op(o4) is True

        # Case 2: Map operator before shuffle (o2) should return True because
        # its downstream ineligible chain includes:
        # - o3 (LimitOperator - ineligible, not in eligible types)
        # - o4 (AllToAllOperator - ineligible because throttling_disabled=True)
        # Since o4 is a blocking materializing op, the check returns True
        assert resource_manager._is_blocking_materializing_op(o2) is True

        # o3 (LimitOperator) also returns True because its downstream ineligible
        # chain includes o4 (shuffle)
        assert resource_manager._is_blocking_materializing_op(o3) is True

        # Case 3: o5 (Map after shuffle) has no downstream ops -> False
        assert resource_manager._is_blocking_materializing_op(o5) is False

        # Case 4: Extend pipeline with ops that have no blocking materializing downstream
        # o5 -> o6 (limit) -> o7
        o6 = LimitOperator(1, o5, DataContext.get_current())
        o7 = mock_map_op(o6, name="Map3")

        topo2 = build_streaming_topology(o7, ExecutionOptions(), noop_counter())
        resource_manager2 = ResourceManager(
            topo2,
            ExecutionOptions(),
            MagicMock(),
            DataContext.get_current(),
            BlockRefCounter(add_object_out_of_scope_callback=lambda *_: True),
        )

        # o5's downstream (o6, o7) has no blocking materializing ops
        assert resource_manager2._is_blocking_materializing_op(o5) is False
        assert resource_manager2._is_blocking_materializing_op(o7) is False

    def test_capability_marks_optional_operator_as_blocking_materializing(
        self, restore_data_context
    ):
        upstream = mock_map_op(
            InputDataBuffer(DataContext.get_current(), []), name="Upstream"
        )
        materializer = mock_map_op(upstream, name="OptionalMaterializer")
        downstream = mock_map_op(materializer, name="Downstream")
        materializer.is_blocking_materializing = MagicMock(return_value=True)

        topology = build_streaming_topology(
            downstream, ExecutionOptions(), noop_counter()
        )
        resource_manager = ResourceManager(
            topology,
            ExecutionOptions(),
            MagicMock(),
            DataContext.get_current(),
            BlockRefCounter(add_object_out_of_scope_callback=lambda *_: True),
        )
        allocator = ReservationOpResourceAllocator(resource_manager, 0.5)

        assert resource_manager._is_blocking_materializing_op(materializer)
        assert downstream not in allocator._get_eligible_ops()

    def test_memory_limit_blocks_task_submission(self, restore_data_context):
        """Test that tasks are blocked when memory limit is exceeded."""
        # Cluster has 1000 bytes of memory
        cluster_resources = ExecutionResources(cpu=1, gpu=0, memory=1000)

        # Request 2000 bytes memory
        o1 = InputDataBuffer(DataContext.get_current(), [])
        o2 = mock_map_op(
            o1,
            ray_remote_args={"num_cpus": 1, "memory": 2000},
            name="HighMemoryTask",
        )

        topo = build_streaming_topology(o2, ExecutionOptions(), noop_counter())
        options = ExecutionOptions()

        resource_manager = ResourceManager(
            topology=topo,
            options=options,
            get_total_resources=lambda: cluster_resources,
            data_context=DataContext.get_current(),
            block_ref_counter=BlockRefCounter(
                add_object_out_of_scope_callback=lambda *_: True
            ),
        )
        resource_manager.update_usages()

        # Task cannot be submitted because it exceeds memory limit
        allocator = create_resource_allocator(
            resource_manager, DataContext.get_current()
        )
        assert allocator is not None
        allocator.update_budgets(limits=resource_manager.get_global_limits())
        can_submit = allocator.can_submit_new_task(o2)
        assert not can_submit, (
            "Task should be blocked: requires 2000 bytes but only 1000 bytes memory available"
        )


class TestOutputBackpressureGuard:
    """Tests for OutputBackpressureGuard.should_unblock."""

    def test_unblock_backpressure_terminal_operator(self, restore_data_context):
        """Terminal operator (no downstream eligible ops) with no external
        consumer should always unblock (e.g., write pipeline)."""
        o1 = InputDataBuffer(DataContext.get_current(), [])
        o2 = mock_map_op(o1)
        o3 = LimitOperator(1, o2, DataContext.get_current())

        topo = build_streaming_topology(o3, ExecutionOptions(), noop_counter())

        resource_manager = ResourceManager(
            topo,
            ExecutionOptions(),
            MagicMock(),
            DataContext.get_current(),
            BlockRefCounter(add_object_out_of_scope_callback=lambda *_: True),
        )
        guard = OutputBackpressureGuard(topo, resource_manager)

        # o2 is terminal (no downstream eligible ops beyond it) and no external
        # consumer — should unblock (e.g., write pipeline).
        assert guard.should_unblock(o2) is True

        # Add o4 operator - o2 is no longer terminal
        o4 = mock_map_op(o3)

        topo = build_streaming_topology(o4, ExecutionOptions(), noop_counter())

        resource_manager = ResourceManager(
            topo,
            ExecutionOptions(),
            MagicMock(),
            DataContext.get_current(),
            BlockRefCounter(add_object_out_of_scope_callback=lambda *_: True),
        )
        guard = OutputBackpressureGuard(topo, resource_manager)

        # Mock downstream (o4) having active tasks and input blocks (ie unblocking
        # conditions not met)
        o4.num_active_tasks = MagicMock(return_value=1)
        guard._idle_detector.detect_idle = MagicMock(return_value=False)

        # o2 is not terminal anymore, falls back to idle detector which returns False
        assert guard.should_unblock(o2) is False

    def test_no_unblock_backpressure_terminal_with_external_consumer(
        self, restore_data_context
    ):
        """Terminal operator with an external consumer should only unblock
        when consumers are starving (blocked waiting for output)."""
        o1 = InputDataBuffer(DataContext.get_current(), [])
        o2 = mock_map_op(o1)
        o3 = LimitOperator(1, o2, DataContext.get_current())

        topo = build_streaming_topology(o3, ExecutionOptions(), noop_counter())

        resource_manager = ResourceManager(
            topo,
            ExecutionOptions(),
            MagicMock(),
            DataContext.get_current(),
            BlockRefCounter(add_object_out_of_scope_callback=lambda *_: True),
        )
        guard = OutputBackpressureGuard(topo, resource_manager)

        # Register an external consumer (e.g., iter_batches or streaming_split).
        resource_manager.set_external_consumer_bytes(0)

        dag_output_state = topo[o3]

        # No consumers waiting — should NOT unblock (prevents pileup).
        dag_output_state._num_waiting_consumers = 0
        assert guard.should_unblock(o2) is False

        # Simulate a consumer blocked in get_output_blocking (starving).
        # The output node is o3 (LimitOperator), which tracks waiting consumers.
        dag_output_state._num_waiting_consumers = 1
        assert guard.should_unblock(o2) is True

        # Consumer gets data and stops waiting — should NOT unblock again.
        dag_output_state._num_waiting_consumers = 0
        assert guard.should_unblock(o2) is False

    def test_unblock_backpressure_downstream_idle(self, restore_data_context):
        """Unblock when downstream is idle (no active tasks) to maintain liveness."""
        o1 = InputDataBuffer(DataContext.get_current(), [])
        o2 = mock_map_op(o1)
        o3 = mock_map_op(o2)

        topo = build_streaming_topology(o3, ExecutionOptions(), noop_counter())

        resource_manager = ResourceManager(
            topo,
            ExecutionOptions(),
            MagicMock(),
            DataContext.get_current(),
            BlockRefCounter(add_object_out_of_scope_callback=lambda *_: True),
        )
        guard = OutputBackpressureGuard(topo, resource_manager)
        o3.num_active_tasks = MagicMock(return_value=0)

        # Case 1: Downstream cannot submit (resource constrained) - unblock to free resources
        resource_manager.op_resource_allocator.can_submit_new_task = MagicMock(
            return_value=False
        )
        assert guard.should_unblock(o2) is True

        # Case 2: Downstream can submit but has no externally dispatchable
        # bundle. Internal partial batches don't count because more upstream
        # outputs may be required to form a runnable task.
        resource_manager.op_resource_allocator.can_submit_new_task = MagicMock(
            return_value=True
        )
        topo[o3].total_enqueued_input_blocks = MagicMock(return_value=5)
        topo[o3].has_pending_bundles = MagicMock(return_value=False)
        assert guard.should_unblock(o2) is True

    def test_unblock_backpressure_fallback_to_idle_detector(self, restore_data_context):
        """When unblock conditions not met, falls back to idle detector result."""
        o1 = InputDataBuffer(DataContext.get_current(), [])
        o2 = mock_map_op(o1)
        o3 = mock_map_op(o2)

        topo = build_streaming_topology(o3, ExecutionOptions(), noop_counter())

        resource_manager = ResourceManager(
            topo,
            ExecutionOptions(),
            MagicMock(),
            DataContext.get_current(),
            BlockRefCounter(add_object_out_of_scope_callback=lambda *_: True),
        )
        guard = OutputBackpressureGuard(topo, resource_manager)

        # Case: Downstream has active tasks - falls back to idle detector
        o3.num_active_tasks = MagicMock(return_value=2)
        guard._idle_detector.detect_idle = MagicMock(return_value=False)
        assert guard.should_unblock(o2) is False

        # Case: Idle detector returns True - should unblock
        guard._idle_detector.detect_idle = MagicMock(return_value=True)
        assert guard.should_unblock(o2) is True

        # Case: Downstream has no active tasks but has an externally dispatchable
        # bundle - falls back to idle detector.
        resource_manager.op_resource_allocator.can_submit_new_task = MagicMock(
            return_value=True
        )
        o3.num_active_tasks = MagicMock(return_value=0)
        topo[o3].has_pending_bundles = MagicMock(return_value=True)
        guard._idle_detector.detect_idle = MagicMock(return_value=False)
        assert guard.should_unblock(o2) is False

    def test_unblock_for_admission_managed_operator_without_runnable_worker(
        self, restore_data_context
    ):
        o1 = InputDataBuffer(DataContext.get_current(), [])
        o2 = mock_map_op(o1)
        o3 = mock_map_op(o2)
        topo = build_streaming_topology(o3, ExecutionOptions(), noop_counter())
        resource_manager = ResourceManager(
            topo,
            ExecutionOptions(),
            MagicMock(),
            DataContext.get_current(),
            BlockRefCounter(add_object_out_of_scope_callback=lambda *_: True),
        )
        guard = OutputBackpressureGuard(topo, resource_manager)
        o3.num_active_tasks = MagicMock(return_value=0)
        o3.can_add_input = MagicMock(return_value=False)
        topo[o3].has_pending_bundles = MagicMock(return_value=True)
        resource_manager.op_resource_allocator.can_submit_new_task = MagicMock(
            return_value=True
        )
        guard._idle_detector.detect_idle = MagicMock(return_value=False)

        # Legacy operators keep the historical budget-only behavior.
        assert guard.should_unblock(o2) is False

        resource_manager.get_resource_admission_grant = MagicMock(
            return_value=ResourceAdmissionGrant(1, True)
        )
        assert guard.should_unblock(o2) is True

    def test_unblock_when_resource_allocator_disabled(self, restore_data_context):
        """When the op resource allocator is disabled, the guard treats
        downstream as schedulable (no budget to consult), so
        "downstream resource constrained" case never fires, but the other
        liveness conditions still do.
        """
        # Disable resource allocator
        DataContext.get_current().op_resource_reservation_enabled = False

        o1 = InputDataBuffer(DataContext.get_current(), [])
        o2 = mock_map_op(o1)
        o3 = mock_map_op(o2)

        topo = build_streaming_topology(o3, ExecutionOptions(), noop_counter())

        resource_manager = ResourceManager(
            topo,
            ExecutionOptions(),
            MagicMock(),
            DataContext.get_current(),
            BlockRefCounter(add_object_out_of_scope_callback=lambda *_: True),
        )
        assert not resource_manager.op_resource_allocator_enabled()

        guard = OutputBackpressureGuard(topo, resource_manager)
        o3.num_active_tasks = MagicMock(return_value=0)

        # "Downstream idle with empty input queue" case should fire and unblock.
        topo[o3].has_pending_bundles = MagicMock(return_value=False)
        assert guard.should_unblock(o2) is True


class TestIdleDetector:
    """Tests for IdleDetector."""

    def test_idle_detector(self, restore_data_context):
        """Test IdleDetector behavior through its public interface."""
        idle_detector = IdleDetector()
        op = MagicMock()
        op.metrics.num_task_outputs_generated = 0

        with freeze_time() as frozen:
            # First call initializes state, returns False
            assert idle_detector.detect_idle(op) is False

            # Call within interval returns False (rate limited)
            frozen.tick(timedelta(seconds=idle_detector.DETECTION_INTERVAL_S - 1))
            assert idle_detector.detect_idle(op) is False

            # Call after interval with no output returns True (idle)
            frozen.tick(timedelta(seconds=2))
            assert idle_detector.detect_idle(op) is True

            # Operator produces output - next detection returns False (active)
            op.metrics.num_task_outputs_generated = 5
            assert idle_detector.detect_idle(op) is False

            # After output, wait for interval with no new output - returns True (idle again)
            frozen.tick(timedelta(seconds=idle_detector.DETECTION_INTERVAL_S + 1))
            assert idle_detector.detect_idle(op) is True


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
