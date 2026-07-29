from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import ray
from ray.actor import _actor_resources_from_options
from ray.data._internal.execution.interfaces import (
    ExecutionOptions,
    ExecutionResources,
    PhysicalOperator,
)
from ray.data._internal.execution.operators.actor_pool_map_operator import (
    ActorPoolMapOperator,
)
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.task_pool_map_operator import (
    TaskPoolMapOperator,
)

if TYPE_CHECKING:
    from ray.data._internal.execution.streaming_executor_state import Topology


class _OwnerState(Enum):
    WAITING = auto()
    ADMITTED = auto()
    RELEASING = auto()
    RELEASED = auto()


@dataclass(frozen=True)
class _Capacity:
    cpu_units: int
    gpu_units: int
    cpu_is_explicit: bool
    gpu_is_explicit: bool


@dataclass(frozen=True)
class _GPUOwner:
    op: PhysicalOperator
    index: int
    cpu_units: int
    gpu_units: int
    effective_max: Optional[int] = None


@dataclass(frozen=True)
class _HandoffBoundary:
    predecessor: _GPUOwner
    successor: _GPUOwner
    segment: Tuple[PhysicalOperator, ...]


def create_gpu_handoff_policy(
    topology: Topology,
    options: ExecutionOptions,
) -> Optional["GPUHandoffPolicy"]:
    """Return a narrow linear GPU handoff policy, or stock fallback.

    Structural fallback is resolved before cluster resources are queried. This
    keeps unsupported execution byte-for-byte on the stock lifecycle path.
    """

    structural_owners = _preflight_structure(topology, options)
    if structural_owners is None:
        return None

    capacity = _resolve_capacity(options)
    owners = _resolve_claims(structural_owners, capacity)
    if owners is None:
        # A current-cluster snapshot can be smaller than an autoscaler's future
        # capacity. Leave that case to stock Ray Data execution.
        return None

    operators = tuple(topology)
    boundaries = tuple(
        _HandoffBoundary(
            predecessor,
            successor,
            tuple(operators[predecessor.index + 1 : successor.index]),
        )
        for predecessor, successor in zip(owners, owners[1:])
    )
    return GPUHandoffPolicy(owners, boundaries, capacity)


class GPUHandoffPolicy:
    """Static CPU/GPU handoff policy for a linear sequence of GPU owners."""

    def __init__(
        self,
        owners: Tuple[_GPUOwner, ...],
        boundaries: Tuple[_HandoffBoundary, ...],
        capacity: _Capacity,
    ):
        self._owners = owners
        self._boundaries = boundaries
        self._capacity = capacity
        self._states: Dict[PhysicalOperator, _OwnerState] = {
            owner.op: _OwnerState.WAITING for owner in owners
        }
        self._drain_boundary_by_op = {
            op: boundary
            for boundary in boundaries
            for op in (boundary.predecessor.op, *boundary.segment)
        }

        admitted_cpu = admitted_gpu = 0
        for index, owner in enumerate(owners):
            if index and any(
                not op.has_execution_finished() for op in boundaries[index - 1].segment
            ):
                break
            if not _resources_fit(
                admitted_cpu + owner.cpu_units,
                admitted_gpu + owner.gpu_units,
                capacity,
            ):
                break
            self._states[owner.op] = _OwnerState.ADMITTED
            admitted_cpu += owner.cpu_units
            admitted_gpu += owner.gpu_units

        for owner in self._owners:
            initially_admitted = self._states[owner.op] is _OwnerState.ADMITTED
            args = (initially_admitted,)
            if owner.effective_max is not None:
                args = (owner.effective_max, initially_admitted)
            owner.op._configure_gpu_handoff(*args)  # type: ignore[attr-defined]

    def update(self) -> None:
        """Advance releases and admissions at the post-state-update boundary."""
        held_cpu = held_gpu = 0
        for owner in self._owners:
            state = self._states[owner.op]
            if state is _OwnerState.WAITING and owner.op._is_execution_marked_finished:
                # Downstream early termination can finish a dormant owner.
                # It must never be started merely so that it can be released.
                state = _OwnerState.RELEASED
            elif state is _OwnerState.ADMITTED and owner.op.has_execution_finished():
                state = _OwnerState.RELEASING
            if state is _OwnerState.RELEASING:
                if owner.op._release_gpu_handoff_resources():  # type: ignore[attr-defined]
                    state = _OwnerState.RELEASED
            self._states[owner.op] = state
            if state in (_OwnerState.ADMITTED, _OwnerState.RELEASING):
                held_cpu += owner.cpu_units
                held_gpu += owner.gpu_units

        # Admission is ordered. Once a frontier cannot advance, no later owner
        # can be admitted ahead of it.
        for index, owner in enumerate(self._owners):
            if self._states[owner.op] is not _OwnerState.WAITING:
                continue
            if index and any(
                not op.has_execution_finished()
                for op in self._boundaries[index - 1].segment
            ):
                break
            if not _resources_fit(
                held_cpu + owner.cpu_units,
                held_gpu + owner.gpu_units,
                self._capacity,
            ):
                break

            owner.op._admit_gpu_handoff()  # type: ignore[attr-defined]
            self._states[owner.op] = _OwnerState.ADMITTED
            held_cpu += owner.cpu_units
            held_gpu += owner.gpu_units

    def should_drain_output(self, op: PhysicalOperator) -> bool:
        """Whether output draining is needed to finish a static handoff range."""
        boundary = self._drain_boundary_by_op.get(op)
        if boundary is None:
            return False
        return (
            self._states[boundary.successor.op] is _OwnerState.WAITING
            and self._states[boundary.predecessor.op] is not _OwnerState.WAITING
        )


def _preflight_structure(
    topology: Topology,
    options: ExecutionOptions,
) -> Optional[Tuple[Tuple[PhysicalOperator, int], ...]]:
    from ray.data._internal.execution.operators.hash_shuffle import (
        HashShufflingOperatorBase,
    )

    operators = tuple(topology)
    if not operators or any(
        op.input_dependencies != ([] if index == 0 else [operators[index - 1]])
        or op.output_dependencies
        != ([] if index == len(operators) - 1 else [operators[index + 1]])
        for index, op in enumerate(operators)
    ):
        return None

    # Capacity is cluster-wide, so label-constrained placement cannot be proven
    # compatible in v1.
    if options.label_selector:
        return None

    owners = []
    for index, op in enumerate(operators):
        if _is_supported_owner(op):
            owners.append((op, index))
            continue

        # Hash-shuffle aggregators and eager actor pools can retain resources.
        if isinstance(op, HashShufflingOperatorBase):
            return None

        # Dynamic map placement/resource callbacks and GPU task operators are
        # outside the static v1 proof.
        if isinstance(op, MapOperator):
            if (
                op._ray_remote_args_fn is not None
                or op._ray_remote_args.get("label_selector")
            ):
                return None
            if (
                isinstance(op, TaskPoolMapOperator)
                and ExecutionResources.from_resource_dict(op._ray_remote_args).gpu > 0
            ):
                return None

        if op.get_autoscaling_actor_pools():
            return None
        base_usage = getattr(op, "base_resource_usage", ExecutionResources.zero())
        if base_usage.gpu > 0 or op.incremental_resource_usage().gpu > 0:
            return None

    return tuple(owners) if len(owners) >= 2 else None


def _is_supported_owner(op: PhysicalOperator) -> bool:
    if isinstance(op, ActorPoolMapOperator):
        if op._ray_remote_args_fn is not None:
            return False

        actor_resources, _ = _actor_resources_from_options(op._ray_remote_args)
        worker_gpu = float(actor_resources.get("GPU", 0) or 0)
        if worker_gpu <= 0 or _resource_units(worker_gpu) == 0:
            return False
        if any(
            float(value or 0) > 0
            for name, value in actor_resources.items()
            if name not in ("CPU", "GPU")
        ):
            return False
        if any(
            op._ray_remote_args.get(option) not in (None, False, 0)
            for option in (
                "label_selector",
                "placement_group",
                "placement_group_bundle_index",
                "placement_group_capture_child_tasks",
                "fallback_strategy",
            )
        ):
            return False
        if op._ray_remote_args.get("scheduling_strategy") not in (
            None,
            "DEFAULT",
            "SPREAD",
        ):
            return False
        if any(
            op._ray_remote_args.get(option) not in (None, False)
            for option in ("name", "lifetime", "get_if_exists")
        ):
            return False
        return True

    # Import GPU shuffle lazily so feature-off CPU execution retains stock
    # import and construction behavior.
    from ray.data._internal.gpu_shuffle.hash_shuffle import (
        GPURankPool,
        GPUShuffleOperator,
    )

    if isinstance(op, GPUShuffleOperator):
        pool = op._rank_pool
        return (
            type(pool) is GPURankPool
            and pool._gpu_handoff_compatible
            and not pool._label_selector
        )

    return False


def _resolve_capacity(options: ExecutionOptions) -> _Capacity:
    limits = options.resource_limits
    exclude = options.exclude_resources
    cpu_is_explicit = math.isfinite(limits.cpu)
    gpu_is_explicit = math.isfinite(limits.gpu)

    snapshot = {}
    if not cpu_is_explicit or not gpu_is_explicit:
        snapshot = ray.cluster_resources()

    cpu = limits.cpu if cpu_is_explicit else float(snapshot.get("CPU", 0))
    gpu = limits.gpu if gpu_is_explicit else float(snapshot.get("GPU", 0))
    return _Capacity(
        cpu_units=_resource_units(max(0.0, cpu - exclude.cpu)),
        gpu_units=_resource_units(max(0.0, gpu - exclude.gpu)),
        cpu_is_explicit=cpu_is_explicit,
        gpu_is_explicit=gpu_is_explicit,
    )


def _resolve_claims(
    structural_owners: Tuple[Tuple[PhysicalOperator, int], ...],
    capacity: _Capacity,
) -> Optional[Tuple[_GPUOwner, ...]]:
    from ray.data._internal.gpu_shuffle.hash_shuffle import GPUShuffleOperator

    owners = []
    snapshot_shortfall = False
    for op, index in structural_owners:
        effective_max = None
        if isinstance(op, GPUShuffleOperator):
            cpu_claim = gpu_claim = (
                op._rank_pool.nranks * ray._raylet.RESOURCE_UNIT_SCALING
            )
            required_cpu, required_gpu = cpu_claim, gpu_claim
        else:
            assert isinstance(op, ActorPoolMapOperator)
            actor_resources, _ = _actor_resources_from_options(op._ray_remote_args)
            worker_cpu = _resource_units(float(actor_resources.get("CPU", 0) or 0))
            worker_gpu = _resource_units(float(actor_resources.get("GPU", 0) or 0))
            pool = op._actor_pool
            mandatory_size = max(pool.min_size(), pool.initial_size())
            required_cpu = mandatory_size * worker_cpu
            required_gpu = mandatory_size * worker_gpu
            effective_max = int(min(pool.max_size(), capacity.gpu_units // worker_gpu))
            cpu_claim = effective_max * worker_cpu
            if pool.min_size() != pool.max_size():
                cpu_claim = min(cpu_claim, capacity.cpu_units)
            gpu_claim = effective_max * worker_gpu

        if not _mandatory_fits(required_cpu, required_gpu, capacity, op):
            snapshot_shortfall = True
        owners.append(_GPUOwner(op, index, cpu_claim, gpu_claim, effective_max))
    return None if snapshot_shortfall else tuple(owners)


def _mandatory_fits(
    required_cpu: int,
    required_gpu: int,
    capacity: _Capacity,
    op: PhysicalOperator,
) -> bool:
    cpu_excess = required_cpu > capacity.cpu_units
    gpu_excess = required_gpu > capacity.gpu_units
    if not cpu_excess and not gpu_excess:
        return True

    if (cpu_excess and capacity.cpu_is_explicit) or (
        gpu_excess and capacity.gpu_is_explicit
    ):
        raise ValueError(
            f"{op} requires at least CPU={_resource_value(required_cpu)}, "
            f"GPU={_resource_value(required_gpu)}, "
            f"but the explicit Ray Data execution capacity is "
            f"CPU={_resource_value(capacity.cpu_units)}, "
            f"GPU={_resource_value(capacity.gpu_units)}."
        )
    return False


def _resources_fit(cpu_units: int, gpu_units: int, capacity: _Capacity) -> bool:
    return cpu_units <= capacity.cpu_units and gpu_units <= capacity.gpu_units


def _resource_units(value: float) -> int:
    return int(value * ray._raylet.RESOURCE_UNIT_SCALING)


def _resource_value(units: int) -> float:
    return units / ray._raylet.RESOURCE_UNIT_SCALING
