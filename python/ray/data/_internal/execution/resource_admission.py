from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ray.data._internal.execution.interfaces.execution_options import ExecutionResources

if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import PhysicalOperator
    from ray.data._internal.execution.resource_manager import ResourceManager
    from ray.data._internal.execution.streaming_executor_state import Topology


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResourceAdmissionSpec:
    """Static resource shape for a persistent physical operator.

    ``minimum_resources`` is the complete progress floor. For an elastic
    owner, ``unit_resources`` describes one worker and ``min_units`` describes
    how many workers form that floor. ``unit_resources=None`` denotes an
    indivisible fixed gang whose only valid grant is one complete gang.
    ``max_units=None`` leaves elastic growth unbounded.
    """

    minimum_resources: ExecutionResources
    unit_resources: Optional[ExecutionResources]
    min_units: int
    max_units: Optional[int]


@dataclass(frozen=True)
class ResourceAdmissionGrant:
    """Executor-owned capacity and submission grant for a persistent operator."""

    max_units: int
    may_submit: bool


def validate_resource_admission_spec(spec: ResourceAdmissionSpec) -> None:
    if spec.minimum_resources.is_zero() or not spec.minimum_resources.is_non_negative():
        raise ValueError("Resource admission minimum resources must be positive")
    if spec.min_units < 1:
        raise ValueError("Resource admission min_units must be positive")
    if spec.unit_resources is None and spec.min_units != 1:
        raise ValueError("Fixed resource-admission gangs require min_units == 1")
    if spec.unit_resources is not None and (
        not spec.unit_resources.is_non_negative()
        or spec.unit_resources.scale(spec.min_units) != spec.minimum_resources
    ):
        raise ValueError("Admission floor must equal unit resources times min_units")
    if spec.max_units is not None and spec.max_units < spec.min_units:
        raise ValueError("Resource admission max_units must be at least min_units")


class _ResourceAdmissionController:
    """Coordinate persistent resource owners without changing legacy operators.

    Admission happens in two phases. ``update_admission`` protects sticky
    owners that cannot safely release resources, admits complete progress
    floors in topology order, and exposes at most one blocked frontier to the
    cluster autoscaler. ``update_allocation_grants`` then expands admitted
    elastic owners within the allocator's usage-independent targets.
    """

    def __init__(self, topology: "Topology", resource_manager: "ResourceManager"):
        self._topology = topology
        self._resource_manager = resource_manager
        self._specs = {}
        for op in topology:
            if spec := op.resource_admission_spec():
                self._specs[op] = spec
        self._grants: dict[PhysicalOperator, ResourceAdmissionGrant] = {}

        if self._specs and any(op.resource_admission_incompatible() for op in topology):
            logger.warning(
                "GPU resource admission found an incompatible operator; disabling "
                "resource admission for the whole topology to avoid mixed ownership.",
            )
            self._specs.clear()

        options = resource_manager._options
        permitted = options.resource_limits.subtract(options.exclude_resources).max(
            ExecutionResources.zero()
        )
        for op, spec in self._specs.items():
            validate_resource_admission_spec(spec)
            if not spec.minimum_resources.satisfies_limit(
                permitted, ignore_object_store_memory=True
            ):
                raise ValueError(
                    f"{op} admission floor {spec.minimum_resources} exceeds the explicit "
                    f"effective execution resource limits {permitted}."
                )

        for op in self._specs:
            self._set_grant(op, 0, may_submit=False)

    def has_participants(self) -> bool:
        return bool(self._specs)

    def get_grant(self, op: "PhysicalOperator") -> Optional[ResourceAdmissionGrant]:
        return self._grants.get(op)

    def get_floor(self, op: "PhysicalOperator") -> Optional[ExecutionResources]:
        spec = self._specs.get(op)
        return spec.minimum_resources if spec is not None else None

    @staticmethod
    def _is_admission_complete(op: "PhysicalOperator", op_state) -> bool:
        return (
            (op.has_execution_finished() or op._inputs_complete)
            and op_state.total_enqueued_input_blocks() == 0
            and op.can_release_resource_admission()
        )

    def _should_prewarm_fixed_gang_successor(self, op: "PhysicalOperator") -> bool:
        if len(op.input_dependencies) != 1:
            return False
        upstream = op.input_dependencies[0]
        upstream_spec = self._specs.get(upstream)
        return (
            upstream_spec is not None
            and upstream_spec.unit_resources is None
            and upstream._inputs_complete
            and not upstream.has_execution_finished()
        )

    def _is_claimant(self, op: "PhysicalOperator") -> bool:
        state = self._topology[op]
        return (
            state.total_enqueued_input_blocks() > 0
            or self._should_prewarm_fixed_gang_successor(op)
            or not op.can_release_resource_admission()
            or (not op.current_logical_usage().is_zero() and not op._inputs_complete)
        )

    def _unfinished_ancestors(self, roots, unfinished):
        ancestors = set()
        for root in roots:
            for dependency in root.input_dependencies:
                ancestors.update(dependency.post_order_iter())
        return ancestors & self._specs.keys() & unfinished

    def _should_expose_demand(
        self, op: "PhysicalOperator", current_capacity: ExecutionResources
    ) -> bool:
        spec = self._specs[op]
        fixed_demand = spec.unit_resources is None
        if not fixed_demand and not op.can_release_resource_admission():
            return False
        if fixed_demand and self._grants[op].max_units > 0:
            return True
        return not (
            spec.minimum_resources.satisfies_limit(
                current_capacity, ignore_object_store_memory=True
            )
        )

    def _set_grant(self, op, max_units: int, *, may_submit: bool) -> None:
        spec = self._specs[op]
        if spec.max_units is not None:
            max_units = min(max_units, spec.max_units)
        grant = ResourceAdmissionGrant(
            max_units=max(0, max_units), may_submit=may_submit
        )
        current_grant = self._grants.get(op)
        if (
            not grant.may_submit
            and current_grant is not None
            and current_grant.may_submit
            and not op.can_release_resource_admission()
        ):
            return
        if current_grant == grant:
            return
        self._grants[op] = grant
        op.apply_resource_admission_grant(grant)

    def update_admission(self, limits: ExecutionResources) -> None:
        if not self._specs:
            return

        unfinished = {
            op
            for op in self._specs
            if not self._is_admission_complete(op, self._topology[op])
        }
        claimants = {op for op in unfinished if self._is_claimant(op)}
        claimants.update(self._unfinished_ancestors(claimants, unfinished))

        current_capacity = remaining = limits
        sticky_ops = {
            op
            for op in claimants
            if self._grants[op].may_submit and not op.can_release_resource_admission()
        }
        protected_ops = sticky_ops | self._unfinished_ancestors(sticky_ops, unfinished)
        for op in protected_ops:
            remaining = remaining.subtract(self._specs[op].minimum_resources).max(
                ExecutionResources.zero()
            )

        has_frontier = False
        for op, spec in self._specs.items():
            if op not in claimants:
                self._set_grant(op, 0, may_submit=False)
                continue
            if op in protected_ops:
                self._set_grant(op, spec.min_units, may_submit=True)
                continue
            if has_frontier:
                self._set_grant(op, 0, may_submit=False)
                continue

            minimum_resources = spec.minimum_resources
            if minimum_resources.satisfies_limit(
                remaining, ignore_object_store_memory=True
            ):
                self._set_grant(op, spec.min_units, may_submit=True)
                remaining = remaining.subtract(minimum_resources).max(
                    ExecutionResources.zero()
                )
            else:
                expose_demand = self._should_expose_demand(op, current_capacity)
                self._set_grant(
                    op,
                    spec.min_units if expose_demand else 0,
                    may_submit=False,
                )
                has_frontier = True

    @staticmethod
    def _max_units_for_target(
        target: ExecutionResources, spec: ResourceAdmissionSpec
    ) -> int:
        if spec.unit_resources is None:
            return 1
        units = target.floordiv(spec.unit_resources)
        capacities = (units.cpu, units.gpu, units.memory, units.object_store_memory)
        max_units = int(min(capacities))
        if spec.max_units is not None:
            max_units = min(max_units, spec.max_units)
        return max_units

    def update_allocation_grants(self, limits: ExecutionResources) -> None:
        remaining = None
        if not self._resource_manager.op_resource_allocator_enabled():
            floors = ExecutionResources.combine_sum(
                self._specs[op].minimum_resources
                for op, grant in self._grants.items()
                if grant.may_submit
            )
            remaining = limits.subtract(floors).max(ExecutionResources.zero())

        for op, spec in self._specs.items():
            if not self._grants[op].may_submit:
                continue
            floor_units = spec.min_units
            target = self._resource_manager.get_allocation_target(op)
            if target is None and remaining is not None:
                target = spec.minimum_resources.add(remaining)
            target_units = (
                floor_units
                if target is None
                else max(floor_units, self._max_units_for_target(target, spec))
            )
            if remaining is not None and spec.unit_resources is not None:
                claimed = spec.unit_resources.scale(target_units - floor_units)
                remaining = remaining.subtract(claimed).max(ExecutionResources.zero())
            self._set_grant(op, target_units, may_submit=True)
