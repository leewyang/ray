"""Conservative GPU-shuffle startup segmentation policy."""

import sys
from typing import Dict, Optional

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
from ray.data._internal.execution.streaming_executor_state import (
    Topology,
    build_execution_segment_topologies,
)
from ray.data._internal.gpu_shuffle.hash_shuffle import GPUShuffleOperator

# Segmentation relies on aggregate CPU/GPU capacity being a complete feasibility
# check; only these strategies preserve that assumption.
_SUPPORTED_SCHEDULING_STRATEGIES = ("DEFAULT", "SPREAD")
# These options add node or placement-group constraints absent from the aggregate
# feasibility check, so their presence forces stock startup.
_UNSUPPORTED_PLACEMENT_ARG_KEYS = {
    "fallback_strategy",
    "label_selector",
    "placement_group",
    "placement_group_bundle_index",
    "placement_group_capture_child_tasks",
}
# These actor options can alter identity, reuse, or lifetime, which breaks the
# proof's assumption that each operator creates and owns its actors.
_ACTOR_OWNERSHIP_ARG_KEYS = {"get_if_exists", "lifetime", "name", "namespace"}


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


def build_gpu_shuffle_segment_topologies(
    topology: Topology, options: ExecutionOptions
) -> list[Topology]:
    """Return ready-to-start segments, or the stock topology on fallback."""
    operators = list(topology)
    split_index = _derive_gpu_shuffle_split_index(operators, options)
    if split_index is None:
        return [topology]
    segment_spec = (tuple(operators[:split_index]), tuple(operators[split_index:]))
    return build_execution_segment_topologies(segment_spec, topology)


def _derive_gpu_shuffle_split_index(
    operators: list[PhysicalOperator], options: ExecutionOptions
) -> Optional[int]:
    """Return the GPU shuffle index when staged startup avoids a GPU deadlock.

    A split index is returned only when all of these are true:

    1. The pipeline is linear and has one standard GPU shuffle.
    2. Ray knows how every operator starts and how many CPUs and GPUs it needs.
    3. Any upstream actor pool has a fixed size and uses GPUs.
    4. The upstream work and shuffle each fit separately, but not at the same time.

    Return None otherwise, preserving the existing startup behavior.
    """
    if not _is_linear_topology(operators):
        return None

    gpu_shuffles = [op for op in operators if isinstance(op, GPUShuffleOperator)]
    if len(gpu_shuffles) != 1:
        return None
    gpu_shuffle = gpu_shuffles[0]
    gpu_shuffle_index = operators.index(gpu_shuffle)

    rank_count = gpu_shuffle.get_default_rank_count()
    if rank_count is None:
        return None
    if options.label_selector:
        return None

    fixed_actor_resources = ExecutionResources.zero()
    max_task_resources = ExecutionResources.zero()

    for index, op in enumerate(operators):
        if op is gpu_shuffle or (index == 0 and type(op) is InputDataBuffer):
            continue
        is_upstream = index < gpu_shuffle_index
        if type(op) not in (TaskPoolMapOperator, ActorPoolMapOperator):
            if is_upstream or type(op).start is not PhysicalOperator.start:
                return None
            continue
        remote_args = op.get_static_ray_remote_args()
        resource_request = _get_supported_cpu_gpu_request(op, remote_args)
        if resource_request is None:
            return None
        if not is_upstream:
            if (
                type(op) is ActorPoolMapOperator
                and op.data_context.wait_for_min_actors_s > 0
            ):
                return None
            continue

        if type(op) is ActorPoolMapOperator:
            actor_pool = op.get_autoscaling_actor_pools()[0]
            initial_size = actor_pool.initial_size()
            if (
                actor_pool.min_size() != initial_size
                or initial_size != actor_pool.max_size()
            ):
                return None
            if not resource_request.gpu:
                return None
            fixed_actor_resources = fixed_actor_resources.add(
                resource_request.scale(initial_size)
            )
        else:
            max_task_resources = max_task_resources.max(resource_request)

    upstream_progress = fixed_actor_resources.add(max_task_resources)
    if not upstream_progress.gpu:
        return None

    capacity = _effective_capacity(options)
    if capacity is None:
        return None

    shuffle_resources = ExecutionResources(cpu=rank_count, gpu=rank_count)
    if not upstream_progress.satisfies_limit(capacity):
        return None
    if not shuffle_resources.satisfies_limit(capacity):
        return None
    if upstream_progress.add(shuffle_resources).satisfies_limit(capacity):
        return None

    return gpu_shuffle_index
