"""Fuse compatible cuDF ``map_batches`` UDFs into one actor pool.

Normally, each call has its own actor pool::

    Input -> MapBatches(Udf1) -> MapBatches(Udf2) -> MapBatches(Udf3)

Fusion removes the intermediate boundaries::

    Input -> MapBatches(Udf1->Udf2->Udf3)

Each fused actor constructs the original UDFs and passes each cuDF result directly to
the next UDF. Ray's normal UDF map planner creates the replacement operator.
"""

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

# Driver-side plan rewrite


# Unknown actor options opt out because one fused operator may not preserve them.
_SUPPORTED_REMOTE_ARGS = frozenset(
    {
        "_labels",
        "accelerator_type",
        "allow_out_of_order_execution",
        "enable_task_events",
        "fallback_strategy",
        "label_selector",
        "max_concurrency",
        "max_restarts",
        "max_task_retries",
        "memory",
        "num_cpus",
        "num_gpus",
        "placement_group",
        "placement_group_bundle_index",
        "placement_group_capture_child_tasks",
        "resources",
        "runtime_env",
        "scheduling_strategy",
    }
)


class FuseCudfActorMapBatches(Rule):
    """Replace compatible consecutive cuDF actor maps with one actor map."""

    def apply(self, plan: PhysicalPlan) -> PhysicalPlan:
        """Replace eligible consecutive cuDF actor maps in a physical plan."""

        if not self._context_allows_fusion(plan.context):
            return plan

        # Absorbed operators are removed from this copy, not the input plan's map.
        rewritten_op_map = plan.op_map.copy()

        (
            logical_consumer_counts,
            physical_consumer_counts,
        ) = self._count_consumers_in_both_plan_views(rewritten_op_map, plan.dag)

        # Preserve shared DAG nodes by reusing the first replacement created for them.
        rewritten_operators: Dict[PhysicalOperator, PhysicalOperator] = {}
        fused_operators: set[PhysicalOperator] = set()

        def rewrite_subdag(physical_op: PhysicalOperator) -> PhysicalOperator:
            """Rewrite this operator and its inputs, returning the new downstream op."""
            if physical_op in rewritten_operators:
                return rewritten_operators[physical_op]

            # Find only the group ending here; recursion finds earlier groups.
            fusible_operators, target_max_block_size = self._collect_fusible_chain(
                physical_op,
                rewritten_op_map,
                logical_consumer_counts,
                physical_consumer_counts,
            )

            if len(fusible_operators) >= 2:
                # Continue before this group so fusion does not skip upstream work.
                rewritten_input = rewrite_subdag(
                    fusible_operators[0].input_dependencies[0]
                )

                fused_physical_op, fused_logical_op = self._plan_fused_map_operator(
                    fusible_operators,
                    rewritten_op_map,
                    rewritten_input,
                    plan.context,
                    target_max_block_size,
                )

                for stage_op in fusible_operators:
                    rewritten_op_map.pop(stage_op)
                rewritten_op_map[fused_physical_op] = fused_logical_op
                rewritten_operators[physical_op] = fused_physical_op
                fused_operators.add(fused_physical_op)
                return fused_physical_op

            # Nothing ends here, so continue through every input.
            rewritten_inputs = [
                rewrite_subdag(input_op) for input_op in physical_op.input_dependencies
            ]

            if any(
                rewritten is not original
                for rewritten, original in zip(
                    rewritten_inputs,
                    physical_op.input_dependencies,
                )
            ):
                # Output links are rebuilt after all input links are final.
                physical_op._input_dependencies = rewritten_inputs
            rewritten_operators[physical_op] = physical_op
            return physical_op

        # plan.dag is the final operator; its input links reach the whole plan.
        new_dag = rewrite_subdag(plan.dag)

        if not fused_operators:
            return plan
        self._rebuild_output_dependencies(new_dag)
        return PhysicalPlan(new_dag, rewritten_op_map, plan.context)

    @staticmethod
    def _context_allows_fusion(context: Any) -> bool:
        """Return whether the current Ray Data settings preserve fusion semantics."""

        if getattr(context, "enable_cudf_actor_fusion", False) is not True:
            return False

        # Fusion wraps errors, so it cannot preserve original types or type-based retries.
        if context.raise_original_map_exception is not False:
            return False

        if (
            type(context.actor_task_retry_on_errors) is not bool
            or type(context.retried_map_errors) is not bool
        ):
            return False

        # A custom admission limit may behave differently with one combined pool.
        return context.max_tasks_in_flight_per_actor is None

    @staticmethod
    def _count_consumers_in_both_plan_views(
        rewritten_op_map: Mapping[PhysicalOperator, LogicalOperator],
        physical_root: PhysicalOperator,
    ) -> Tuple[Dict[LogicalOperator, int], Dict[PhysicalOperator, int]]:
        """Count both plan views because planning can copy logical operators."""

        logical_counts: Dict[LogicalOperator, int] = {}
        for op in set(rewritten_op_map.values()):
            for input_op in op.input_dependencies:
                logical_counts[input_op] = logical_counts.get(input_op, 0) + 1

        physical_counts: Dict[PhysicalOperator, int] = {}
        for op in set(physical_root.post_order_iter()):
            for input_op in op.input_dependencies:
                physical_counts[input_op] = physical_counts.get(input_op, 0) + 1
        return logical_counts, physical_counts

    @classmethod
    def _collect_fusible_chain(
        cls,
        op: PhysicalOperator,
        rewritten_op_map: Mapping[PhysicalOperator, LogicalOperator],
        logical_consumer_counts: Mapping[LogicalOperator, int],
        physical_consumer_counts: Mapping[PhysicalOperator, int],
    ) -> Tuple[Tuple[PhysicalOperator, ...], Optional[int]]:
        """Return the maximal fusible group ending at ``op`` in execution order."""

        chain_config = cls._fusion_config_if_eligible(
            op,
            rewritten_op_map.get(op),
        )
        if chain_config is None:
            return (), None

        chain = [op]
        upstream_cursor = op

        # A fused operator can preserve only one block-size override.
        target_max_block_size = op.target_max_block_size_override

        while len(upstream_cursor.input_dependencies) == 1:
            upstream = upstream_cursor.input_dependencies[0]
            upstream_logical_op = rewritten_op_map.get(upstream)
            upstream_config = cls._fusion_config_if_eligible(
                upstream, upstream_logical_op
            )
            upstream_target_size = upstream.target_max_block_size_override
            upstream_is_shared = (
                logical_consumer_counts.get(upstream_logical_op, 0) != 1
                or physical_consumer_counts.get(upstream, 0) != 1
            )
            block_sizes_conflict = (
                target_max_block_size is not None
                and upstream_target_size is not None
                and target_max_block_size != upstream_target_size
            )

            if (
                upstream_is_shared
                or upstream_config is None
                or not chain_config.is_compatible_with(upstream_config)
                or block_sizes_conflict
            ):
                break
            chain.append(upstream)
            upstream_cursor = upstream

            if target_max_block_size is None:
                target_max_block_size = upstream_target_size

        return tuple(reversed(chain)), target_max_block_size

    @staticmethod
    def _fusion_config_if_eligible(
        physical_op: PhysicalOperator,
        logical_op: Optional[LogicalOperator],
    ) -> Optional["_CudfMapFusionConfig"]:
        """Return compatibility settings, or ``None`` when fusion is unsafe."""

        # A replacement produced by this rule must not be fused again.
        if (
            not isinstance(physical_op, ActorPoolMapOperator)
            or not physical_op.supports_fusion()
            or not isinstance(logical_op, MapBatches)
            or logical_op.fn is _FusedCudfMapBatches
        ):
            return None

        # Fusion removes intermediate batch boundaries, so their policies must collapse.
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

        # Inspect without running user or GPU setup on the driver.
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

        actor_pool = logical_op.compute
        remote_args = logical_op.ray_remote_args

        # Shared UDF instances require serial execution and known actor options.
        if (
            type(actor_pool) is not ActorPoolStrategy
            or actor_pool.enable_true_multi_threading is not False
            or actor_pool.max_tasks_in_flight_per_actor is not None
            or type(remote_args) is not dict
            or not remote_args.keys() <= _SUPPORTED_REMOTE_ARGS
        ):
            return None

        # Fusion currently requires one task and one full GPU per actor.
        max_concurrency = remote_args.get("max_concurrency", 1)
        if type(max_concurrency) is not int or max_concurrency != 1:
            return None
        num_gpus = remote_args.get("num_gpus")
        if type(num_gpus) not in (int, float) or num_gpus != 1:
            return None

        return _CudfMapFusionConfig(
            actor_pool=actor_pool,
            batch_size=logical_op.batch_size,
            min_rows_per_bundle=logical_op.min_rows_per_bundled_input,
            ray_remote_args=remote_args,
        )

    @staticmethod
    def _plan_fused_map_operator(
        physical_chain: Tuple[PhysicalOperator, ...],
        rewritten_op_map: Mapping[PhysicalOperator, LogicalOperator],
        input_physical_dag: PhysicalOperator,
        context: Any,
        target_max_block_size: Optional[int],
    ) -> Tuple[PhysicalOperator, MapBatches]:
        """Create the physical replacement and its synthetic logical operator."""

        logical_chain = tuple(
            rewritten_op_map[physical_op] for physical_op in physical_chain
        )

        # Pass descriptions so each actor constructs its own GPU-local UDF state.
        stages = tuple(
            _CudfMapStage(
                udf_class=logical_op.fn,
                error_label=f"{stage_index}: {logical_op.name}",
                constructor_args=(
                    ()
                    if logical_op.fn_constructor_args is None
                    else logical_op.fn_constructor_args
                ),
                constructor_kwargs=(
                    {}
                    if logical_op.fn_constructor_kwargs is None
                    else logical_op.fn_constructor_kwargs
                ),
                call_args=() if logical_op.fn_args is None else logical_op.fn_args,
                call_kwargs={}
                if logical_op.fn_kwargs is None
                else logical_op.fn_kwargs,
            )
            for stage_index, logical_op in enumerate(logical_chain, start=1)
        )
        first_logical_op = logical_chain[0]

        # Eligibility guarantees matching settings, so the first map defines ingress.
        fused_logical_op = MapBatches(
            _FusedCudfMapBatches,
            input_dependencies=[first_logical_op.input_dependencies[0]],
            can_modify_num_rows=any(
                logical_op.can_modify_num_rows for logical_op in logical_chain
            ),
            batch_size=first_logical_op.batch_size,
            batch_format="cudf",
            zero_copy_batch=True,
            fn_constructor_args=(stages,),
            min_rows_per_bundled_input=first_logical_op.min_rows_per_bundled_input,
            compute=first_logical_op.compute,
            ray_remote_args=dict(first_logical_op.ray_remote_args),
        )

        names = "->".join(stage.udf_class.__name__ for stage in stages)

        # _name is display-only, but MapBatches is frozen.
        object.__setattr__(fused_logical_op, "_name", f"MapBatches({names})")

        # Reuse Ray's planner to preserve normal actor and batch-conversion behavior.
        fused_physical_op = create_udf_map_operator(
            fused_logical_op,
            input_physical_dag,
            context,
            target_max_block_size_override=target_max_block_size,
        )

        # Preserve original lineage for stats and plan inspection.
        fused_physical_op.set_logical_operators(*logical_chain)
        return fused_physical_op, fused_logical_op

    @staticmethod
    def _rebuild_output_dependencies(root: PhysicalOperator) -> None:
        """Rebuild reverse links after rewriting input dependencies.

        Ray stores each edge in both directions, but the traversal updates only inputs.
        """

        operators = set(root.post_order_iter())

        for op in operators:
            op._output_dependencies = []

        for op in operators:
            for input_op in op.input_dependencies:
                input_op._output_dependencies.append(op)


@dataclass(frozen=True, eq=False)
class _CudfMapFusionConfig:
    """Settings that must match before actor maps can share one replacement."""

    actor_pool: ActorPoolStrategy
    batch_size: int
    min_rows_per_bundle: Optional[int]
    ray_remote_args: Mapping[str, Any]

    def is_compatible_with(self, other: "_CudfMapFusionConfig") -> bool:
        if (
            self.actor_pool != other.actor_pool
            or self.batch_size != other.batch_size
            or not _equal_config_values(
                self.min_rows_per_bundle,
                other.min_rows_per_bundle,
            )
        ):
            return False
        try:
            return _equal_config_values(self.ray_remote_args, other.ray_remote_args)
        except Exception:
            return False


def _equal_config_values(left: Any, right: Any) -> bool:
    """Compare nested settings only when equality produces one definite bool.

    This rejects type coercion and array-like equality results.
    """

    if type(left) is not type(right):
        return False

    if isinstance(left, Mapping):
        if left.keys() != right.keys():
            return False
        return all(_equal_config_values(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _equal_config_values(left_value, right_value)
            for left_value, right_value in zip(left, right)
        )

    try:
        result = left == right
    except Exception:
        return False

    return type(result) is bool and result


# Actor-side execution


@dataclass(frozen=True, eq=False)
class _CudfMapStage:
    """Arguments needed to construct and call one original UDF in the actor."""

    udf_class: type
    error_label: str
    constructor_args: Iterable[Any] = ()
    constructor_kwargs: Dict[str, Any] = field(default_factory=dict)
    call_args: Iterable[Any] = ()
    call_kwargs: Dict[str, Any] = field(default_factory=dict)


class _FusedCudfMapBatches:
    """Run several original UDFs as one callable actor transform."""

    def __init__(self, stages: Tuple[_CudfMapStage, ...]):
        """Construct each UDF in the actor that owns its GPU state."""

        self._stages = tuple(stages)
        instances = []
        for stage in self._stages:
            try:
                instance = stage.udf_class(
                    *stage.constructor_args,
                    **stage.constructor_kwargs,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Fused cuDF map_batches stage {stage.error_label!r} failed "
                    "during construction."
                ) from exc
            instances.append(instance)
        self._instances = tuple(instances)

    def __call__(self, batch: Any) -> Any:
        """Pass one cuDF DataFrame through every original UDF in order."""

        for stage, instance in zip(self._stages, self._instances):
            try:
                # Pass results directly to avoid conversion and rebatching between UDFs.
                batch = instance(batch, *stage.call_args, **stage.call_kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"Fused cuDF map_batches stage {stage.error_label!r} failed."
                ) from exc

            # No Ray boundary remains to enforce the next UDF's cuDF input contract.
            if not _is_cudf_dataframe(batch):
                output_type = f"{type(batch).__module__}.{type(batch).__qualname__}"
                raise TypeError(
                    f"Fused cuDF map_batches stage {stage.error_label!r} returned "
                    f"{output_type}; expected cudf.DataFrame."
                )
        return batch
