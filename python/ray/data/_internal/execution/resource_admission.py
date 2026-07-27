from __future__ import annotations

import logging
import math
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

    def _transient_progress_floor(self, roots) -> ExecutionResources:
        """Return one schedulable progress unit for unmanaged ancestors.

        A persistent owner must not become sticky while an upstream task still
        needs the same processor or heap-memory resource to finish producing
        its stream. Ancestor task operators can run sequentially, so the
        component-wise maximum is a sufficient liveness floor and avoids
        over-reserving an entire transient pipeline.
        """
        ancestors = set()
        for root in roots:
            for dependency in root.input_dependencies:
                ancestors.update(dependency.post_order_iter())

        floor = ExecutionResources.zero()
        for op in ancestors.intersection(self._topology).difference(self._specs):
            state = self._topology[op]
            needs_progress = not op.has_execution_finished() and (
                not op._inputs_complete
                or state.total_enqueued_input_blocks() > 0
                or not op.current_logical_usage().is_zero()
            )
            if needs_progress:
                floor = floor.max(op.min_scheduling_resources())
        return floor

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

        current_capacity = limits
        remaining = limits.subtract(self._transient_progress_floor(claimants)).max(
            ExecutionResources.zero()
        )
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

    @classmethod
    def _max_units_for_target(
        cls, target: ExecutionResources, spec: ResourceAdmissionSpec
    ) -> int:
        if spec.unit_resources is None:
            return 1
        max_units = int(cls._unit_capacity(target, spec.unit_resources))
        if spec.max_units is not None:
            max_units = min(max_units, spec.max_units)
        return max_units

    def update_allocation_grants(self, limits: ExecutionResources) -> None:
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
                target = spec.minimum_resources.add(remaining)
                target_units = max(
                    floor_units, self._max_units_for_target(target, spec)
                )
                if spec.unit_resources is not None:
                    claimed = spec.unit_resources.scale(target_units - floor_units)
                    remaining = remaining.subtract(claimed).max(
                        ExecutionResources.zero()
                    )
                self._set_grant(op, target_units, may_submit=True)
            return

        admitted = [
            (op, spec)
            for op, spec in self._specs.items()
            if self._grants[op].may_submit
        ]
        remaining = limits.subtract(
            ExecutionResources.combine_sum(
                spec.minimum_resources for _, spec in admitted
            )
        ).max(ExecutionResources.zero())

        targets = {
            op: self._resource_manager.get_allocation_target(op) for op, _ in admitted
        }
        target_units = {
            op: (
                spec.min_units
                if targets[op] is None
                else max(
                    spec.min_units,
                    self._max_units_for_target(targets[op], spec),
                )
            )
            for op, spec in admitted
        }
        granted_units = {op: spec.min_units for op, spec in admitted}

        # First honor every complete unit represented by the allocator targets.
        # Floors were removed up front so one operator's expansion cannot consume
        # capacity required by another admitted owner.
        for op, spec in admitted:
            if spec.unit_resources is None:
                continue
            desired = target_units[op] - granted_units[op]
            additional = int(
                min(desired, self._unit_capacity(remaining, spec.unit_resources))
            )
            granted_units[op] += additional
            remaining = remaining.subtract(spec.unit_resources.scale(additional)).max(
                ExecutionResources.zero()
            )

        # Allocator targets are continuous, but elastic actor pools grow in whole
        # units. Pool-local flooring can strand an aggregate complete unit (for
        # example, two 1.5-GPU targets on three GPUs). Use largest remainders to
        # coordinate rounding without exceeding either the aggregate targets or
        # the global execution limits.
        rounding_budget = ExecutionResources.combine_sum(
            targets[op]
            .subtract(spec.unit_resources.scale(target_units[op]))
            .max(ExecutionResources.zero())
            for op, spec in admitted
            if targets[op] is not None and spec.unit_resources is not None
        )
        candidates = []
        for index, (op, spec) in enumerate(admitted):
            target = targets[op]
            if target is None or spec.unit_resources is None:
                continue
            if spec.max_units is not None and granted_units[op] >= spec.max_units:
                continue
            remainder = (
                self._unit_capacity(target, spec.unit_resources) - target_units[op]
            )
            candidates.append((remainder, index, op, spec))

        # Give each owner at most one unit per pass so topology peers get an
        # opportunity before an owner receives a second pooled unit. Owners with
        # no local remainder remain candidates: a smaller unit can consume
        # aggregate capacity stranded by another owner's larger unit shape.
        candidates.sort(key=lambda candidate: candidate[:2], reverse=True)
        while candidates:
            made_progress = False
            next_candidates = []
            for remainder, index, op, spec in candidates:
                unit = spec.unit_resources
                assert unit is not None
                if (
                    self._unit_capacity(rounding_budget, unit) >= 1
                    and self._unit_capacity(remaining, unit) >= 1
                ):
                    granted_units[op] += 1
                    rounding_budget = rounding_budget.subtract(unit).max(
                        ExecutionResources.zero()
                    )
                    remaining = remaining.subtract(unit).max(ExecutionResources.zero())
                    made_progress = True
                if spec.max_units is None or granted_units[op] < spec.max_units:
                    next_candidates.append((remainder, index, op, spec))
            if not made_progress:
                break
            candidates = next_candidates

        for op, _ in admitted:
            self._set_grant(op, granted_units[op], may_submit=True)

    @staticmethod
    def _unit_capacity(
        resources: ExecutionResources, unit: ExecutionResources
    ) -> float:
        capacities = [
            available / required
            for available, required in (
                (resources.cpu, unit.cpu),
                (resources.gpu, unit.gpu),
                (resources.memory, unit.memory),
                (resources.object_store_memory, unit.object_store_memory),
            )
            if required > 0
        ]
        assert capacities
        capacity = min(capacities)
        if not math.isfinite(capacity):
            return capacity

        # Decimal resource shapes commonly land one ULP below an integer
        # quotient (for example, 0.3 / 0.1). Snap only machine-roundoff-sized
        # errors so a complete fractional CPU/GPU unit is not under-granted.
        nearest_integer = round(capacity)
        tolerance = 8 * max(
            math.ulp(capacity),
            math.ulp(float(nearest_integer)),
        )
        if math.isclose(
            capacity,
            nearest_integer,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            return float(nearest_integer)
        return capacity
