"""Fuse consecutive actor-based cuDF ``map_batches`` in the physical plan.

Ray normally plans each call as a separate actor pool::

    Input -> MapBatches(Udf1) -> MapBatches(Udf2) -> MapBatches(Udf3)

This rule replaces a compatible chain with one actor-pool operator::

    Input -> MapBatches(Udf1->Udf2->Udf3)

The replacement is lowered through Ray's normal UDF map planner. Each fused actor
constructs the original UDFs and passes every returned cuDF DataFrame directly to
the next UDF.
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


# Actor-side execution


@dataclass(frozen=True, eq=False)
class _CudfMapStage:
    """Data needed to reconstruct one original ``map_batches`` UDF in an actor."""

    udf_class: type
    error_label: str
    constructor_args: Iterable[Any] = ()
    constructor_kwargs: Dict[str, Any] = field(default_factory=dict)
    call_args: Iterable[Any] = ()
    call_kwargs: Dict[str, Any] = field(default_factory=dict)


class _FusedCudfMapBatches:
    """Run the original UDFs sequentially inside one Ray actor."""

    def __init__(self, stages: Tuple[_CudfMapStage, ...]):
        """Construct a separate UDF instance for every original map stage."""

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
        """Pass one cuDF batch through every UDF without an intermediate boundary."""

        for stage, instance in zip(self._stages, self._instances):
            try:
                # BatchMapTransformFn handles only the chain's ingress and egress.
                # Passing the returned object unchanged avoids conversion or rebatching.
                batch = instance(batch, *stage.call_args, **stage.call_kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"Fused cuDF map_batches stage {stage.error_label!r} failed."
                ) from exc
            if not _is_cudf_dataframe(batch):
                output_type = f"{type(batch).__module__}.{type(batch).__qualname__}"
                raise TypeError(
                    f"Fused cuDF map_batches stage {stage.error_label!r} returned "
                    f"{output_type}; expected cudf.DataFrame."
                )
        return batch


# Driver-side physical-plan rewrite


# New or unsupported actor options opt out until fusion can preserve their behavior.
_SUPPORTED_REMOTE_ARGS = frozenset(
    "_labels accelerator_type allow_out_of_order_execution enable_task_events "
    "fallback_strategy label_selector max_concurrency max_restarts "
    "max_task_retries memory num_cpus num_gpus placement_group "
    "placement_group_bundle_index placement_group_capture_child_tasks resources "
    "runtime_env scheduling_strategy".split()
)


@dataclass(frozen=True, eq=False)
class _CudfMapFusionConfig:
    """Execution settings that must match across a fused map chain."""

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
    """Compare settings without type coercion or array-like equality results."""

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


class FuseCudfActorMapBatches(Rule):
    """Replace compatible cuDF actor-map chains with one physical operator."""

    def apply(self, plan: PhysicalPlan) -> PhysicalPlan:
        """Fuse every eligible linear cuDF actor-map chain in ``plan``."""

        if not self._context_allows_fusion(plan.context):
            return plan

        op_map = plan.op_map.copy()
        (
            logical_consumer_counts,
            physical_consumer_counts,
        ) = self._count_logical_and_physical_consumers(op_map, plan.dag)

        # Planning can copy one shared logical node into multiple physical nodes.
        # The two consumer counts protect both forms of sharing, while this memo
        # preserves any physical nodes that are genuinely shared in the DAG.
        rewritten_operators: Dict[PhysicalOperator, PhysicalOperator] = {}
        did_fuse = False

        def rewrite_subdag(op: PhysicalOperator) -> PhysicalOperator:
            nonlocal did_fuse
            if op in rewritten_operators:
                return rewritten_operators[op]

            chain, target_max_block_size = self._collect_fusible_chain(
                op,
                op_map,
                logical_consumer_counts,
                physical_consumer_counts,
            )

            if len(chain) >= 2:
                rewritten_input = rewrite_subdag(chain[0].input_dependencies[0])
                fused_physical_op, fused_logical_op = self._plan_fused_map_operator(
                    chain,
                    op_map,
                    rewritten_input,
                    plan.context,
                    target_max_block_size,
                )
                # The replacement gets the temporary logical map in op_map; its
                # physical lineage still carries every original logical map.
                for stage_op in chain:
                    op_map.pop(stage_op)
                op_map[fused_physical_op] = fused_logical_op
                rewritten_operators[op] = fused_physical_op
                did_fuse = True
                return fused_physical_op

            rewritten_inputs = [
                rewrite_subdag(input_op) for input_op in op.input_dependencies
            ]
            if any(
                rewritten is not original
                for rewritten, original in zip(
                    rewritten_inputs,
                    op.input_dependencies,
                )
            ):
                # Input edges drive execution; producer-side output edges are repaired
                # once the full rewrite is complete.
                op._input_dependencies = rewritten_inputs
            rewritten_operators[op] = op
            return op

        new_dag = rewrite_subdag(plan.dag)
        if not did_fuse:
            return plan
        self._rebuild_output_dependencies(new_dag)
        return PhysicalPlan(new_dag, op_map, plan.context)

    @staticmethod
    def _context_allows_fusion(context: Any) -> bool:
        if getattr(context, "enable_cudf_actor_fusion", False) is not True:
            return False
        # Stage labels require wrapping exceptions. Original-exception propagation
        # and exception-class retry matching would therefore change after fusion.
        if context.raise_original_map_exception is not False:
            return False
        if (
            type(context.actor_task_retry_on_errors) is not bool
            or type(context.retried_map_errors) is not bool
        ):
            return False
        # The fused pool currently relies on Ray's default task-admission behavior.
        return context.max_tasks_in_flight_per_actor is None

    @staticmethod
    def _count_logical_and_physical_consumers(
        op_map: Mapping[PhysicalOperator, LogicalOperator],
        root: PhysicalOperator,
    ) -> Tuple[Dict[LogicalOperator, int], Dict[PhysicalOperator, int]]:
        """Count both DAG views because physical planning can copy logical nodes."""

        logical_counts: Dict[LogicalOperator, int] = {}
        for op in set(op_map.values()):
            for input_op in op.input_dependencies:
                logical_counts[input_op] = logical_counts.get(input_op, 0) + 1

        physical_counts: Dict[PhysicalOperator, int] = {}
        for op in set(root.post_order_iter()):
            for input_op in op.input_dependencies:
                physical_counts[input_op] = physical_counts.get(input_op, 0) + 1
        return logical_counts, physical_counts

    @classmethod
    def _collect_fusible_chain(
        cls,
        op: PhysicalOperator,
        op_map: Mapping[PhysicalOperator, LogicalOperator],
        logical_consumer_counts: Mapping[LogicalOperator, int],
        physical_consumer_counts: Mapping[PhysicalOperator, int],
    ) -> Tuple[Tuple[PhysicalOperator, ...], Optional[int]]:
        """Return the maximal eligible chain ending at ``op``, in execution order."""

        chain_config = cls._fusion_config_if_eligible(op, op_map.get(op))
        if chain_config is None:
            return (), None

        chain = [op]
        upstream_cursor = op
        target_max_block_size = op.target_max_block_size_override

        while len(upstream_cursor.input_dependencies) == 1:
            upstream = upstream_cursor.input_dependencies[0]
            upstream_logical_op = op_map.get(upstream)
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
            # One replacement cannot absorb work needed by another branch or retain
            # two different output block-size overrides.
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
    ) -> Optional[_CudfMapFusionConfig]:
        """Return an eligible map's compatibility settings, otherwise ``None``."""

        if (
            not isinstance(physical_op, ActorPoolMapOperator)
            or not physical_op.supports_fusion()
            or not isinstance(logical_op, MapBatches)
            # Reject the replacement callable on later optimizer passes.
            or logical_op.fn is _FusedCudfMapBatches
        ):
            return None

        # A single BatchMapTransformFn can preserve only one explicit ingress batch
        # policy. Requiring zero-copy input also avoids silently removing defensive
        # copies at the original boundaries. Dynamic actor options and per-block
        # limits cannot be collapsed either.
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

        # Static inspection rejects unsupported call styles without constructing a
        # user class or invoking one of its descriptors on the driver.
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
        # Every actor must serialize access to the UDF instances shared by the chain.
        if (
            type(actor_pool) is not ActorPoolStrategy
            or actor_pool.enable_true_multi_threading is not False
            or actor_pool.max_tasks_in_flight_per_actor is not None
            or type(remote_args) is not dict
            or not remote_args.keys() <= _SUPPORTED_REMOTE_ARGS
        ):
            return None

        # One task at a time preserves single-threaded access to all stage instances;
        # one full GPU per actor is the current fusion contract.
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
        op_map: Mapping[PhysicalOperator, LogicalOperator],
        input_physical_dag: PhysicalOperator,
        context: Any,
        target_max_block_size: Optional[int],
    ) -> Tuple[PhysicalOperator, MapBatches]:
        """Plan one actor-map replacement for an execution-ordered chain."""

        logical_chain = tuple(op_map[physical_op] for physical_op in physical_chain)
        # Preserve user argument containers until the actor constructs or calls a UDF.
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
        # Compatibility guarantees shared execution settings; the first map owns
        # the chain's physical input and ingress batching policy.
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
        # MapBatches is frozen, and _name affects display only.
        object.__setattr__(fused_logical_op, "_name", f"MapBatches({names})")

        # Reusing the normal planner preserves actor setup, cuDF conversion, and
        # BatchMapTransformFn behavior instead of rebuilding them in this rule.
        fused_physical_op = create_udf_map_operator(
            fused_logical_op,
            input_physical_dag,
            context,
            target_max_block_size_override=target_max_block_size,
        )
        fused_physical_op.set_logical_operators(*logical_chain)
        return fused_physical_op, fused_logical_op

    @staticmethod
    def _rebuild_output_dependencies(root: PhysicalOperator) -> None:
        # Physical operators store every edge on both its consumer and producer.
        # The rewrite changes consumer inputs first, so regenerate the reverse links.
        operators = set(root.post_order_iter())
        for op in operators:
            op._output_dependencies = []
        for op in operators:
            for input_op in op.input_dependencies:
                input_op._output_dependencies.append(op)
