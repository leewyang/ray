import logging
import sys
from typing import TYPE_CHECKING, Dict, Optional

import ray
from ray._common.utils import resources_from_ray_options
from ray.data._internal.cluster_autoscaler.util import is_autoscaling_enabled
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
from ray.data._internal.execution.streaming_executor_state import ExecutionSegments
from ray.data._internal.gpu_shuffle.hash_shuffle import GPUShuffleOperator

if TYPE_CHECKING:
    from ray.data._internal.execution.streaming_executor_state import Topology

logger = logging.getLogger(__name__)

_SCHEDULING_STRATEGIES = (None, "DEFAULT", "SPREAD")
# User heap requests are out of scope; framework estimates are not part of this
# CPU/GPU startup proof.
_UNSUPPORTED_REMOTE_OPTIONS = {
    "accelerator_type",
    "fallback_strategy",
    "label_selector",
    "memory",
    "object_store_memory",
    "placement_group",
    "placement_group_bundle_index",
    "placement_group_capture_child_tasks",
    "resources",
}
_UNSUPPORTED_ACTOR_OPTIONS = {"get_if_exists", "lifetime", "name", "namespace"}


def _fallback(reason: str) -> None:
    logger.debug("Using stock GPU shuffle startup: %s", reason)


def _is_linear(ops: list[PhysicalOperator]) -> bool:
    return all(
        op.input_dependencies == ([] if index == 0 else [ops[index - 1]])
        and op.output_dependencies
        == ([] if index == len(ops) - 1 else [ops[index + 1]])
        for index, op in enumerate(ops)
    )


def _has_unsupported_options(op: MapOperator, remote_args: Dict) -> bool:
    if any(key in remote_args for key in _UNSUPPORTED_REMOTE_OPTIONS):
        return True
    if type(op) is ActorPoolMapOperator and any(
        key in remote_args for key in _UNSUPPORTED_ACTOR_OPTIONS
    ):
        return True
    if "scheduling_strategy" in remote_args:
        strategies = (remote_args["scheduling_strategy"],)
    else:
        strategies = (
            op.data_context.scheduling_strategy,
            op.data_context.scheduling_strategy_large_args,
        )
    return not all(strategy in _SCHEDULING_STRATEGIES for strategy in strategies)


def _processor_resources(remote_args: Dict, *, actor: bool) -> ExecutionResources:
    resources = resources_from_ray_options(remote_args)
    gpu = resources.get("GPU", 0)
    cpu = resources.get("CPU", 1 if not actor or "GPU" in resources else 0)
    return ExecutionResources(cpu=cpu, gpu=gpu)


def _known_capacity(options: ExecutionOptions) -> Optional[ExecutionResources]:
    configured_max = ray._private.state.state.get_max_resources_from_cluster_config()
    if not configured_max:
        if is_autoscaling_enabled():
            return None
        configured_max = ray.cluster_resources()
    elif any(value == sys.maxsize for value in configured_max.values()):
        return None
    total = ExecutionResources.from_resource_dict(configured_max)
    return (
        options.resource_limits.min(total)
        .subtract(options.exclude_resources)
        .max(ExecutionResources.zero())
    )


def plan_gpu_shuffle_startup(
    topology: "Topology", options: ExecutionOptions
) -> Optional[ExecutionSegments]:
    """Return a boundary only for a statically proven CPU/GPU startup conflict."""
    ops = list(topology)
    if not _is_linear(ops):
        return _fallback("topology is not linear")

    shuffles = [op for op in ops if isinstance(op, GPUShuffleOperator)]
    if len(shuffles) != 1:
        return _fallback("topology does not contain exactly one GPU shuffle")
    shuffle = shuffles[0]
    shuffle_index = ops.index(shuffle)

    nranks = shuffle.get_rank_pool_size_if_default()
    if nranks is None:
        return _fallback("shuffle rank pool is custom")
    if options.label_selector:
        return _fallback("execution has a label selector")

    persistent = ExecutionResources.zero()
    transient = ExecutionResources.zero()

    for index, op in enumerate(ops):
        if op is shuffle or (index == 0 and type(op) is InputDataBuffer):
            continue
        in_prefix = index < shuffle_index
        if type(op) not in (TaskPoolMapOperator, ActorPoolMapOperator):
            if in_prefix or type(op).start is not PhysicalOperator.start:
                return _fallback(f"unsupported operator {type(op).__name__}")
            continue
        remote_args = op.get_static_ray_remote_args_for_planning()
        if remote_args is None or _has_unsupported_options(op, remote_args):
            return _fallback(f"{op.name} has dynamic or unsupported remote options")
        if not in_prefix:
            if (
                type(op) is ActorPoolMapOperator
                and op.data_context.wait_for_min_actors_s > 0
            ):
                return _fallback(f"{op.name} can block suffix startup")
            continue

        if type(op) is ActorPoolMapOperator:
            actor_pool = op.get_autoscaling_actor_pools()[0]
            min_size = actor_pool.min_size()
            initial_size = actor_pool.initial_size()
            if min_size != initial_size or initial_size != actor_pool.max_size():
                return _fallback(f"{op.name} has an elastic actor pool")
            actor_resources = _processor_resources(remote_args, actor=True)
            if actor_resources.gpu:
                persistent = persistent.add(actor_resources.scale(initial_size))
            elif actor_resources.cpu:
                return _fallback("CPU-only persistent actor is outside GPU-only scope")
        else:
            task_resources = _processor_resources(remote_args, actor=False)
            transient = transient.max(task_resources)

    upstream_floor = persistent.add(transient)
    if not upstream_floor.gpu:
        return None

    capacity = _known_capacity(options)
    if capacity is None:
        return _fallback("effective capacity is unknown while autoscaling")

    shuffle_requirement = ExecutionResources(cpu=nranks, gpu=nranks)
    if not upstream_floor.satisfies_limit(capacity):
        return _fallback("upstream progress floor cannot fit independently")
    if not shuffle_requirement.satisfies_limit(capacity):
        return _fallback("shuffle rank gang cannot fit independently")
    if upstream_floor.add(shuffle_requirement).satisfies_limit(capacity):
        return None

    return (
        tuple(ops[:shuffle_index]),
        tuple(ops[shuffle_index:]),
    )
