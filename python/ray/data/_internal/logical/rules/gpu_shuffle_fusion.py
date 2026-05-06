from typing import Dict

from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.task_pool_map_operator import (
    TaskPoolMapOperator,
)
from ray.data._internal.gpu_shuffle.hash_shuffle import GPUShuffleOperator
from ray.data._internal.logical.interfaces import PhysicalPlan, Rule

__all__ = ["FuseGPUShuffleMaps"]


class FuseGPUShuffleMaps(Rule):
    """Fuse compatible GPU map stages into GPUShuffleOperator rank actors.

    This keeps the public Dataset API unchanged. When enabled, the physical
    optimizer rewrites:

        TaskPoolMapOperator -> GPUShuffleOperator -> TaskPoolMapOperator

    into one GPUShuffleOperator whose rank actors run the map transformers
    before insert and after extract.
    """

    CONFIG_KEY = "gpu_shuffle_fuse_maps"

    def apply(self, plan: PhysicalPlan) -> PhysicalPlan:
        if not plan.context.get_config(self.CONFIG_KEY, False):
            return plan

        self._op_map: Dict[PhysicalOperator, object] = plan.op_map.copy()
        changed = True
        dag = plan.dag
        while changed:
            changed = False

            def transform(op: PhysicalOperator) -> PhysicalOperator:
                nonlocal changed
                fused = self._try_fuse_downstream_map(op)
                if fused is not op:
                    changed = True
                    return fused

                fused = self._try_fuse_upstream_map(op)
                if fused is not op:
                    changed = True
                    return fused

                return op

            dag = dag._apply_transform(transform)

        self._remove_output_deps(dag)
        self._update_output_deps(dag)
        return PhysicalPlan(dag, self._op_map, plan.context)

    def _try_fuse_upstream_map(self, op: PhysicalOperator) -> PhysicalOperator:
        if not isinstance(op, GPUShuffleOperator):
            return op
        if len(op.input_dependencies) != 1:
            return op

        up_op = op.input_dependencies[0]
        if not self._can_fuse_map(up_op):
            return op
        assert isinstance(up_op, MapOperator)
        if len(up_op.input_dependencies) != 1:
            return op

        fused = op.copy_with_fused_maps(
            input_op=up_op.input_dependencies[0],
            name=f"{up_op.name}->{op.name}",
            upstream_map_transformer=up_op.get_map_transformer(),
            upstream_map_task_kwargs=up_op.get_map_task_kwargs(),
        )
        logical_op = self._op_map.get(op) or self._op_map.get(up_op)
        self._op_map.pop(op, None)
        self._op_map.pop(up_op, None)
        self._op_map[fused] = logical_op
        return fused

    def _try_fuse_downstream_map(self, op: PhysicalOperator) -> PhysicalOperator:
        if not self._can_fuse_map(op):
            return op
        assert isinstance(op, MapOperator)
        if len(op.input_dependencies) != 1:
            return op

        up_op = op.input_dependencies[0]
        if not isinstance(up_op, GPUShuffleOperator):
            return op
        if len(up_op.input_dependencies) != 1:
            return op

        fused = up_op.copy_with_fused_maps(
            input_op=up_op.input_dependencies[0],
            name=f"{up_op.name}->{op.name}",
            downstream_map_transformer=op.get_map_transformer(),
            downstream_map_task_kwargs=op.get_map_task_kwargs(),
        )
        logical_op = self._op_map.get(op) or self._op_map.get(up_op)
        self._op_map.pop(op, None)
        self._op_map.pop(up_op, None)
        self._op_map[fused] = logical_op
        return fused

    @staticmethod
    def _can_fuse_map(op: PhysicalOperator) -> bool:
        if not isinstance(op, TaskPoolMapOperator):
            return False
        if not op.supports_fusion():
            return False
        if op.get_additional_split_factor() > 1:
            return False
        if len(op.input_dependencies) != 1:
            return False
        if op._ray_remote_args_fn is not None:
            return False
        if op._on_start is not None:
            return False

        remote_args = op._ray_remote_args
        if remote_args.get("num_gpus", 0) != 1:
            return False

        # These maps will run inside already-reserved GPU shuffle rank actors.
        # Avoid silently dropping extra scheduling constraints.
        allowed_remote_args = {"num_gpus", "name"}
        return set(remote_args).issubset(allowed_remote_args)

    def _remove_output_deps(self, op: PhysicalOperator) -> None:
        for input_op in op.input_dependencies:
            input_op._output_dependencies = []
            self._remove_output_deps(input_op)

    def _update_output_deps(self, op: PhysicalOperator) -> None:
        for input_op in op.input_dependencies:
            input_op._output_dependencies.append(op)
            self._update_output_deps(input_op)
