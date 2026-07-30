"""Conservative GPU-shuffle startup segmentation policy."""

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
from ray.data._internal.execution.streaming_executor_state import ExecutionSegmentSpec
from ray.data._internal.gpu_shuffle.hash_shuffle import GPUShuffleOperator

if TYPE_CHECKING:
    from ray.data._internal.execution.streaming_executor_state import Topology

logger = logging.getLogger(__name__)

_SUPPORTED_SCHEDULING_STRATEGIES = ("DEFAULT", "SPREAD")
_UNSUPPORTED_PLACEMENT_ARG_KEYS = {
    "fallback_strategy",
    "label_selector",
    "placement_group",
    "placement_group_bundle_index",
    "placement_group_capture_child_tasks",
}
_ACTOR_OWNERSHIP_ARG_KEYS = {"get_if_exists", "lifetime", "name", "namespace"}


def _fallback_to_stock(reason: str) -> None:
    logger.debug("Using stock GPU shuffle startup: %s", reason)


def _is_linear_topology(operators: list[PhysicalOperator]) -> bool:
    return all(
        op.input_dependencies == ([] if index == 0 else [operators[index - 1]])
        and op.output_dependencies
        == ([] if index == len(operators) - 1 else [operators[index + 1]])
        for index, op in enumerate(operators)
    )


def _get_supported_cpu_gpu_request(
    op: MapOperator, remote_args: Optional[Dict]
) -> Optional[ExecutionResources]:
    if remote_args is None:
        return None
    try:
        resources = resources_from_ray_options(remote_args)
    except ValueError:
        return None
    if resources.keys() - {"CPU", "GPU"}:
        return None
    if any(key in remote_args for key in _UNSUPPORTED_PLACEMENT_ARG_KEYS):
        return None
    is_actor = type(op) is ActorPoolMapOperator
    if is_actor and any(key in remote_args for key in _ACTOR_OWNERSHIP_ARG_KEYS):
        return None
    if "scheduling_strategy" in remote_args:
        strategies = (remote_args["scheduling_strategy"],)
    else:
        strategies = (
            op.data_context.scheduling_strategy,
            op.data_context.scheduling_strategy_large_args,
        )
    if not all(strategy in _SUPPORTED_SCHEDULING_STRATEGIES for strategy in strategies):
        return None
    gpu = resources.get("GPU", 0)
    cpu = resources.get("CPU", 1 if not is_actor or "GPU" in resources else 0)
    return ExecutionResources(cpu=cpu, gpu=gpu)


def _effective_capacity(options: ExecutionOptions) -> Optional[ExecutionResources]:
    cluster_max = ray._private.state.state.get_max_resources_from_cluster_config()
    if not cluster_max:
        if is_autoscaling_enabled():
            return None
        cluster_max = ray.cluster_resources()
    elif any(value == sys.maxsize for value in cluster_max.values()):
        return None
    total_capacity = ExecutionResources.from_resource_dict(cluster_max)
    return (
        options.resource_limits.min(total_capacity)
        .subtract(options.exclude_resources)
        .max(ExecutionResources.zero())
    )


def derive_gpu_shuffle_segments(
    topology: "Topology", options: ExecutionOptions
) -> Optional[ExecutionSegmentSpec]:
    """Return segments for a proven GPU producer/shuffle coexistence conflict."""
    operators = list(topology)
    if not _is_linear_topology(operators):
        return _fallback_to_stock("topology is not linear")

    gpu_shuffles = [op for op in operators if isinstance(op, GPUShuffleOperator)]
    if len(gpu_shuffles) != 1:
        return _fallback_to_stock("expected exactly one GPU shuffle")
    gpu_shuffle = gpu_shuffles[0]
    gpu_shuffle_index = operators.index(gpu_shuffle)

    rank_count = gpu_shuffle.get_default_rank_count()
    if rank_count is None:
        return _fallback_to_stock("shuffle rank pool is custom")
    if options.label_selector:
        return _fallback_to_stock("execution has a label selector")

    fixed_actor_resources = ExecutionResources.zero()
    max_task_resources = ExecutionResources.zero()

    for index, op in enumerate(operators):
        if op is gpu_shuffle or (index == 0 and type(op) is InputDataBuffer):
            continue
        is_upstream = index < gpu_shuffle_index
        if type(op) not in (TaskPoolMapOperator, ActorPoolMapOperator):
            if is_upstream or type(op).start is not PhysicalOperator.start:
                return _fallback_to_stock(f"unsupported operator {type(op).__name__}")
            continue
        remote_args = op.get_static_ray_remote_args()
        resource_request = _get_supported_cpu_gpu_request(op, remote_args)
        if resource_request is None:
            return _fallback_to_stock(
                f"{op.name} has dynamic or unsupported remote options"
            )
        if not is_upstream:
            if (
                type(op) is ActorPoolMapOperator
                and op.data_context.wait_for_min_actors_s > 0
            ):
                return _fallback_to_stock(f"{op.name} can block suffix startup")
            continue

        if type(op) is ActorPoolMapOperator:
            actor_pool = op.get_autoscaling_actor_pools()[0]
            min_size = actor_pool.min_size()
            initial_size = actor_pool.initial_size()
            if min_size != initial_size or initial_size != actor_pool.max_size():
                return _fallback_to_stock(f"{op.name} has an elastic actor pool")
            actor_request = resource_request
            if actor_request.gpu:
                fixed_actor_resources = fixed_actor_resources.add(
                    actor_request.scale(initial_size)
                )
            else:
                return _fallback_to_stock("non-GPU actor is outside GPU-only scope")
        else:
            task_resources = resource_request
            max_task_resources = max_task_resources.max(task_resources)

    upstream_progress = fixed_actor_resources.add(max_task_resources)
    if not upstream_progress.gpu:
        return None

    capacity = _effective_capacity(options)
    if capacity is None:
        return _fallback_to_stock("effective capacity is unknown while autoscaling")

    shuffle_resources = ExecutionResources(cpu=rank_count, gpu=rank_count)
    if not upstream_progress.satisfies_limit(capacity):
        return _fallback_to_stock("upstream progress cannot fit independently")
    if not shuffle_resources.satisfies_limit(capacity):
        return _fallback_to_stock("shuffle rank gang cannot fit independently")
    if upstream_progress.add(shuffle_resources).satisfies_limit(capacity):
        return None

    return (
        tuple(operators[:gpu_shuffle_index]),
        tuple(operators[gpu_shuffle_index:]),
    )
