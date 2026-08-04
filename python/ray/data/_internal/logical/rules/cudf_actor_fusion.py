"""Physical fusion for consecutive actor-based cuDF ``map_batches`` UDFs."""

import inspect
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from ray.data._internal.compute import ActorPoolStrategy
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.operators.actor_pool_map_operator import (
    ActorPoolMapOperator,
)
from ray.data._internal.logical.interfaces import LogicalOperator, PhysicalPlan, Rule
from ray.data._internal.logical.operators.map_operator import MapBatches
from ray.data._internal.planner.plan_udf_map_op import create_udf_map_operator
from ray.data.block import _is_cudf_dataframe


# Actor identity, unsupported concurrency behavior, and unknown future options
# remain unfused. This allowlist makes new Ray actor settings conservative by default.
_SUPPORTED_REMOTE_ARGS = frozenset(
    "_labels accelerator_type allow_out_of_order_execution enable_task_events "
    "fallback_strategy label_selector max_concurrency max_restarts "
    "max_task_retries memory num_cpus num_gpus placement_group "
    "placement_group_bundle_index placement_group_capture_child_tasks resources "
    "runtime_env scheduling_strategy".split()
)


@dataclass(frozen=True)
class _CudfMapStage:
    """Serializable description of one callable-class UDF in a fused actor."""

    fn: type
    label: str
    # Keep arbitrary user arguments out of Ray's callable-class lookup hash and
    # equality. They remain serialized here and are passed through unchanged.
    fn_constructor_args: Iterable[Any] = field(default=(), compare=False, hash=False)
    fn_constructor_kwargs: Dict[str, Any] = field(
        default_factory=dict, compare=False, hash=False
    )
    fn_args: Iterable[Any] = field(default=(), compare=False, hash=False)
    fn_kwargs: Dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


class _FusedCudfMapBatches:
    """Callable-class UDF that directly passes cuDF outputs between stages."""

    def __init__(self, stages: Tuple[_CudfMapStage, ...]):
        self.stages = tuple(stages)
        instances = []
        # Each actor owns a distinct instance for every stage, including repeated
        # uses of the same callable class.
        for stage in self.stages:
            try:
                instance = stage.fn(
                    *stage.fn_constructor_args,
                    **stage.fn_constructor_kwargs,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Fused cuDF map_batches stage {stage.label!r} failed "
                    "during construction."
                ) from exc
            instances.append(instance)
        self._instances = tuple(instances)

    def __call__(self, batch: Any) -> Any:
        for stage, instance in zip(self.stages, self._instances):
            try:
                # Pass the exact returned cuDF object onward. In particular, do not
                # copy, normalize, convert, or rebatch between stages.
                batch = instance(batch, *stage.fn_args, **stage.fn_kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"Fused cuDF map_batches stage {stage.label!r} failed."
                ) from exc
            if not _is_cudf_dataframe(batch):
                output_type = f"{type(batch).__module__}.{type(batch).__qualname__}"
                raise TypeError(
                    f"Fused cuDF map_batches stage {stage.label!r} returned "
                    f"{output_type}; expected cudf.DataFrame."
                )
        return batch


@dataclass(frozen=True)
class _FusionSpec:
    actor_size: int
    batch_size: int
    min_rows_per_bundle: Optional[int]
    ray_remote_args: Mapping[str, Any] = field(compare=False)

    def compatible_with(self, other: "_FusionSpec") -> bool:
        if (
            self.actor_size != other.actor_size
            or self.batch_size != other.batch_size
            or not _same_config_value(
                self.min_rows_per_bundle,
                other.min_rows_per_bundle,
            )
        ):
            return False
        try:
            return _same_config_value(self.ray_remote_args, other.ray_remote_args)
        except Exception:
            return False


def _same_config_value(left: Any, right: Any) -> bool:
    """Compare nested settings without accepting array-like equality results."""

    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        if left.keys() != right.keys():
            return False
        return all(_same_config_value(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _same_config_value(left_value, right_value)
            for left_value, right_value in zip(left, right)
        )
    try:
        result = left == right
    except Exception:
        return False
    return type(result) is bool and result


class FuseCudfActorMapBatches(Rule):
    """Fuse maximal linear runs of compatible actor-based cuDF ``MapBatches``."""

    def apply(self, plan: PhysicalPlan) -> PhysicalPlan:
        if not self._context_allows_fusion(plan.context):
            return plan

        op_map = plan.op_map.copy()
        # Count consumers in the original graph. Shared logical or physical nodes
        # remain fusion barriers even as the rewrite replaces input edges.
        logical_consumer_counts = self._logical_consumer_counts(op_map)
        physical_consumer_counts = self._physical_consumer_counts(plan.dag)
        memo: Dict[PhysicalOperator, PhysicalOperator] = {}
        changed = False

        def rewrite(op: PhysicalOperator) -> PhysicalOperator:
            nonlocal changed
            if op in memo:
                return memo[op]

            run, target_max_block_size = self._collect_fusible_run(
                op,
                op_map,
                logical_consumer_counts,
                physical_consumer_counts,
            )

            if len(run) >= 2:
                logical_run = tuple(op_map[stage_op] for stage_op in run)
                fused_logical_op = self._create_fused_logical_op(logical_run)
                ingress_physical_op = run[-1]
                # Reuse the canonical planner construction path so actor-pool setup,
                # cuDF boundary conversion, and execution options remain stock Ray.
                fused_physical_op = create_udf_map_operator(
                    fused_logical_op,
                    rewrite(ingress_physical_op.input_dependencies[0]),
                    plan.context,
                    target_max_block_size_override=target_max_block_size,
                )
                # The composite is the planning node, but the original operators stay
                # attached as physical lineage metadata for plan introspection.
                fused_physical_op.set_logical_operators(*reversed(logical_run))
                for stage_op in run:
                    op_map.pop(stage_op)
                op_map[fused_physical_op] = fused_logical_op
                memo[op] = fused_physical_op
                changed = True
                return fused_physical_op

            rewritten_inputs = [rewrite(child) for child in op.input_dependencies]
            if any(
                rewritten is not original
                for rewritten, original in zip(
                    rewritten_inputs,
                    op.input_dependencies,
                )
            ):
                op._input_dependencies = rewritten_inputs
            memo[op] = op
            return op

        new_dag = rewrite(plan.dag)
        if not changed:
            return plan
        self._rebuild_output_dependencies(new_dag)
        return PhysicalPlan(new_dag, op_map, plan.context)

    @staticmethod
    def _context_allows_fusion(context: Any) -> bool:
        if getattr(context, "enable_cudf_actor_fusion", False) is not True:
            return False
        # Stage-labelled wrapping changes exception types, so selective retry
        # policies and original-exception mode cannot preserve stock behavior.
        # Boolean retry-all or retry-none settings remain safe.
        return (
            context.raise_original_map_exception is False
            and context.max_tasks_in_flight_per_actor is None
            and type(context.actor_task_retry_on_errors) is bool
            and type(context.retried_map_errors) is bool
        )

    @classmethod
    def _collect_fusible_run(
        cls,
        op: PhysicalOperator,
        op_map: Mapping[PhysicalOperator, LogicalOperator],
        logical_consumer_counts: Mapping[LogicalOperator, int],
        physical_consumer_counts: Mapping[PhysicalOperator, int],
    ) -> Tuple[Tuple[PhysicalOperator, ...], Optional[int]]:
        """Collect a maximal compatible run ordered downstream to upstream."""

        run_spec = cls._fusion_spec(op, op_map.get(op))
        run = [op]
        cursor = op
        target_max_block_size = op.target_max_block_size_override
        if run_spec is None:
            return tuple(run), target_max_block_size

        while len(cursor.input_dependencies) == 1:
            upstream = cursor.input_dependencies[0]
            upstream_logical_op = op_map.get(upstream)
            upstream_spec = cls._fusion_spec(upstream, upstream_logical_op)
            upstream_target_size = upstream.target_max_block_size_override
            # Require a single consumer in both representations so replacement
            # cannot absorb an operator that is still needed elsewhere.
            if (
                logical_consumer_counts.get(upstream_logical_op, 0) != 1
                or physical_consumer_counts.get(upstream, 0) != 1
                or upstream_spec is None
                or not run_spec.compatible_with(upstream_spec)
                or (
                    target_max_block_size is not None
                    and upstream_target_size is not None
                    and target_max_block_size != upstream_target_size
                )
            ):
                break
            run.append(upstream)
            cursor = upstream
            if target_max_block_size is None:
                target_max_block_size = upstream_target_size
        return tuple(run), target_max_block_size

    @classmethod
    def _create_fused_logical_op(
        cls,
        logical_run: Tuple[MapBatches, ...],
    ) -> MapBatches:
        """Create the private composite for a downstream-to-upstream run."""

        stages = tuple(
            cls._stage_from_op(stage_op, stage_index)
            for stage_index, stage_op in enumerate(reversed(logical_run), start=1)
        )
        ingress_logical_op = logical_run[-1]
        fused_logical_op = MapBatches(
            _FusedCudfMapBatches,
            input_dependencies=[ingress_logical_op.input_dependencies[0]],
            can_modify_num_rows=any(
                stage_op.can_modify_num_rows for stage_op in logical_run
            ),
            batch_size=ingress_logical_op.batch_size,
            batch_format="cudf",
            zero_copy_batch=True,
            fn_constructor_args=(stages,),
            min_rows_per_bundled_input=ingress_logical_op.min_rows_per_bundled_input,
            compute=ingress_logical_op.compute,
            ray_remote_args=dict(ingress_logical_op.ray_remote_args),
        )
        cls._set_fused_name(fused_logical_op, stages)
        return fused_logical_op

    @staticmethod
    def _logical_consumer_counts(
        op_map: Mapping[PhysicalOperator, LogicalOperator],
    ) -> Dict[LogicalOperator, int]:
        counts: Dict[LogicalOperator, int] = {}
        for op in set(op_map.values()):
            for child in op.input_dependencies:
                counts[child] = counts.get(child, 0) + 1
        return counts

    @staticmethod
    def _physical_consumer_counts(
        root: PhysicalOperator,
    ) -> Dict[PhysicalOperator, int]:
        counts: Dict[PhysicalOperator, int] = {}
        for op in set(root.post_order_iter()):
            for input_op in op.input_dependencies:
                counts[input_op] = counts.get(input_op, 0) + 1
        return counts

    @classmethod
    def _fusion_spec(
        cls,
        physical_op: PhysicalOperator,
        logical_op: Optional[LogicalOperator],
    ) -> Optional[_FusionSpec]:
        if (
            not isinstance(physical_op, ActorPoolMapOperator)
            or not physical_op.supports_fusion()
            or not isinstance(logical_op, MapBatches)
            or logical_op.fn is _FusedCudfMapBatches
        ):
            return None
        if (
            logical_op.batch_format != "cudf"
            or type(logical_op.batch_size) is not int
            or logical_op.batch_size <= 0
            or logical_op.zero_copy_batch is not True
            or logical_op.ray_remote_args_fn is not None
            or logical_op.per_block_limit is not None
            or not inspect.isclass(logical_op.fn)
        ):
            return None

        call = inspect.getattr_static(logical_op.fn, "__call__", None)
        if isinstance(call, (classmethod, staticmethod)):
            call = call.__func__
        if call is None or any(
            predicate(call)
            for predicate in (
                inspect.iscoroutinefunction,
                inspect.isasyncgenfunction,
                inspect.isgeneratorfunction,
            )
        ):
            return None

        actor_size = cls._fixed_actor_size(logical_op.compute)
        remote_args = logical_op.ray_remote_args
        if (
            actor_size is None
            or type(remote_args) is not dict
            or not remote_args.keys() <= _SUPPORTED_REMOTE_ARGS
        ):
            return None
        max_concurrency = remote_args.get("max_concurrency", 1)
        if type(max_concurrency) is not int or max_concurrency != 1:
            return None
        num_gpus = remote_args.get("num_gpus")
        if type(num_gpus) not in (int, float) or num_gpus != 1:
            return None

        return _FusionSpec(
            actor_size=actor_size,
            batch_size=logical_op.batch_size,
            min_rows_per_bundle=logical_op.min_rows_per_bundled_input,
            ray_remote_args=remote_args,
        )

    @staticmethod
    def _fixed_actor_size(compute: Any) -> Optional[int]:
        if type(compute) is not ActorPoolStrategy:
            return None
        size = compute.min_size
        if (
            type(size) is not int
            or size <= 0
            or type(compute.max_size) is not int
            or type(compute.initial_size) is not int
            or compute.max_size != size
            or compute.initial_size != size
            or compute.enable_true_multi_threading is not False
            or compute.max_tasks_in_flight_per_actor is not None
        ):
            return None
        return size

    @staticmethod
    def _set_fused_name(op: MapBatches, stages: Tuple[_CudfMapStage, ...]) -> None:
        names = "->".join(stage.fn.__name__ for stage in stages)
        object.__setattr__(op, "_name", f"MapBatches({names})")

    @staticmethod
    def _stage_from_op(op: MapBatches, stage_index: int) -> _CudfMapStage:
        # Preserve user containers without copying or iterating here. They are
        # unpacked only when the fused actor constructs or invokes each stage.
        return _CudfMapStage(
            fn=op.fn,
            label=f"{stage_index}: {op.name}",
            fn_constructor_args=(
                () if op.fn_constructor_args is None else op.fn_constructor_args
            ),
            fn_constructor_kwargs=(
                {} if op.fn_constructor_kwargs is None else op.fn_constructor_kwargs
            ),
            fn_args=() if op.fn_args is None else op.fn_args,
            fn_kwargs={} if op.fn_kwargs is None else op.fn_kwargs,
        )

    @staticmethod
    def _rebuild_output_dependencies(root: PhysicalOperator) -> None:
        # Input edges are rewritten in place; rebuild their reverse edges once after
        # the complete traversal so the physical DAG stays internally consistent.
        operators = set(root.post_order_iter())
        for op in operators:
            op._output_dependencies = []
        for op in operators:
            for input_op in op.input_dependencies:
                input_op._output_dependencies.append(op)
