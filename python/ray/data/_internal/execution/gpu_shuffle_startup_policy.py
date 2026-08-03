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
    split_indices = _derive_gpu_shuffle_split_indices(operators, options)
    if split_indices is None:
        return [topology]
    bounds = (0, *split_indices, len(operators))
    segment_spec = tuple(
        tuple(operators[start:end]) for start, end in zip(bounds, bounds[1:])
    )
    return build_execution_segment_topologies(segment_spec, topology)


def _derive_gpu_shuffle_split_indices(
    operators: list[PhysicalOperator], options: ExecutionOptions
) -> Optional[tuple[int, ...]]:
    """Return shuffle boundaries when staged startup avoids a GPU deadlock.

    Boundaries are returned only when all of these are true:

    1. The pipeline is linear and contains standard GPU shuffles.
    2. Ray knows how every operator starts and how many CPUs and GPUs it needs.
    3. Modeled actor pools have fixed sizes and use GPUs.
    4. Every phase and shuffle gang fits alone, but stock cannot safely progress.

    Return None otherwise, preserving the existing startup behavior.
    """
    if not _is_linear_topology(operators):
        return None

    shuffle_indices = tuple(
        index
        for index, op in enumerate(operators)
        if isinstance(op, GPUShuffleOperator)
    )
    if not shuffle_indices:
        return None
    if options.label_selector:
        return None

    shuffle_resources = []
    for index in shuffle_indices:
        rank_count = operators[index].get_default_rank_count()
        if rank_count is None:
            return None
        shuffle_resources.append(ExecutionResources(cpu=rank_count, gpu=rank_count))

    phase_count = len(shuffle_indices) + 1
    phase_actor_resources = [ExecutionResources.zero() for _ in range(phase_count)]
    phase_task_resources = [ExecutionResources.zero() for _ in range(phase_count)]
    total_actor_resources = ExecutionResources.zero()
    phase_index = 0

    for index, op in enumerate(operators):
        if isinstance(op, GPUShuffleOperator):
            phase_index += 1
            continue
        if index == 0 and type(op) is InputDataBuffer:
            continue
        if type(op) not in (TaskPoolMapOperator, ActorPoolMapOperator):
            if (
                index < shuffle_indices[-1]
                or type(op).start is not PhysicalOperator.start
            ):
                return None
            continue
        remote_args = op.get_static_ray_remote_args()
        resource_request = _get_supported_cpu_gpu_request(op, remote_args)
        if resource_request is None:
            return None
        is_actor = type(op) is ActorPoolMapOperator
        if is_actor and phase_index and op.data_context.wait_for_min_actors_s > 0:
            return None
        if len(shuffle_indices) == 1 and phase_index == 1:
            continue

        if is_actor:
            actor_pool = op.get_autoscaling_actor_pools()[0]
            initial_size = actor_pool.initial_size()
            if (
                actor_pool.min_size() != initial_size
                or initial_size != actor_pool.max_size()
            ):
                return None
            if not resource_request.gpu:
                return None
            actor_resources = resource_request.scale(initial_size)
            phase_actor_resources[phase_index] = phase_actor_resources[phase_index].add(
                actor_resources
            )
            total_actor_resources = total_actor_resources.add(actor_resources)
        else:
            phase_task_resources[phase_index] = phase_task_resources[phase_index].max(
                resource_request
            )

    capacity = _effective_capacity(options)
    if capacity is None:
        return None

    if len(shuffle_indices) == 1:
        upstream_progress = phase_actor_resources[0].add(phase_task_resources[0])
        shuffle_resource = shuffle_resources[0]
        if not upstream_progress.gpu:
            return None
        if not upstream_progress.satisfies_limit(capacity):
            return None
        if not shuffle_resource.satisfies_limit(capacity):
            return None
        if upstream_progress.add(shuffle_resource).satisfies_limit(capacity):
            return None
        return shuffle_indices

    remaining_shuffle_resources = ExecutionResources.combine_sum(shuffle_resources)
    for phase_index, task_resources in enumerate(phase_task_resources):
        stock_progress = total_actor_resources.add(task_resources).add(
            remaining_shuffle_resources
        )
        if not stock_progress.satisfies_limit(capacity):
            break
        if phase_index < len(shuffle_resources):
            remaining_shuffle_resources = remaining_shuffle_resources.subtract(
                shuffle_resources[phase_index]
            )
    else:
        return None

    if any(
        not actor_resources.add(task_resources).satisfies_limit(capacity)
        for actor_resources, task_resources in zip(
            phase_actor_resources, phase_task_resources
        )
    ) or any(
        not resources.satisfies_limit(capacity) for resources in shuffle_resources
    ):
        return None
    return shuffle_indices
