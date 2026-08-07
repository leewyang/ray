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

# Actor-option allow-list: add an option only when fusion preserves its behavior.
_FUSION_SAFE_REMOTE_ARGS = {
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


def _is_synchronous_callable_class(udf_class: Any) -> bool:
    """Return whether the UDF can run in the fused actor's synchronous loop.

    The UDF must be a class whose ``__call__`` method is synchronous—not a
    coroutine, async generator, or generator. Inspecting the class without creating
    an instance keeps user setup and GPU initialization inside the Ray actor.

    Args:
        udf_class: UDF class to inspect.

    Returns:
        ``True`` when the UDF is a class with a synchronous ``__call__`` method.
    """

    if not inspect.isclass(udf_class):
        return False

    call_method = inspect.getattr_static(udf_class, "__call__", None)
    if isinstance(call_method, (classmethod, staticmethod)):
        call_method = call_method.__func__
    if call_method is None:
        return False
    if inspect.iscoroutinefunction(call_method):
        return False
    if inspect.isasyncgenfunction(call_method):
        return False
    if inspect.isgeneratorfunction(call_method):
        return False
    return True


class FuseCudfActorMapBatches(Rule):
    """Fuse compatible consecutive cuDF actor maps in a physical plan.

    Only linear groups whose batching, actor-pool, resource, and retry behavior can be
    preserved are replaced.
    """

    def apply(self, plan: PhysicalPlan) -> PhysicalPlan:
        """Rewrite all eligible cuDF actor-map groups in ``plan``.

        Traversal starts at the plan's final physical operator and follows input
        dependencies upstream. Unaffected physical operators are reused and may be
        relinked to fused replacements.

        Args:
            plan: Physical plan to inspect and rewrite.

        Returns:
            The original plan when fusion is disabled or no eligible group exists;
            otherwise, a new ``PhysicalPlan`` with an updated DAG and operator map.
        """

        if not self._context_allows_fusion(plan.context):
            return plan

        # Build a separate map so the input plan keeps its operator map.
        rewritten_op_map = plan.op_map.copy()

        # plan.dag is the final operator. Following its inputs reaches the whole plan.
        original_final_operator = plan.dag
        logical_consumer_counts = self._count_logical_consumers(rewritten_op_map)
        physical_consumer_counts = self._count_physical_consumers(
            original_final_operator
        )

        # Reuse the first rewrite when two paths reach the same operator.
        rewritten_operator_by_original: Dict[PhysicalOperator, PhysicalOperator] = {}
        new_fused_operators: list[PhysicalOperator] = []

        def rewrite_upstream_graph(physical_op: PhysicalOperator) -> PhysicalOperator:
            """Rewrite the subgraph ending at ``physical_op``.

            If a fusible group ends here, first rewrite the input before that group and
            then replace the group. Otherwise, rewrite every direct input. Cached results
            keep shared operators shared. Returns the downstream operator of the
            rewritten subgraph.
            """
            if physical_op in rewritten_operator_by_original:
                return rewritten_operator_by_original[physical_op]

            # Find only the group ending here; recursion finds earlier groups.
            fusible_operators, target_max_block_size = self._collect_fusible_operators(
                physical_op,
                rewritten_op_map,
                logical_consumer_counts,
                physical_consumer_counts,
            )

            if len(fusible_operators) >= 2:
                first_fusible_operator = fusible_operators[0]
                assert isinstance(first_fusible_operator, ActorPoolMapOperator)

                # Continue before this group so fusion does not skip upstream work.
                rewritten_group_input = rewrite_upstream_graph(
                    first_fusible_operator.input_dependency
                )

                fused_physical_op, fused_logical_op = self._create_fused_map_operator(
                    fusible_operators,
                    rewritten_op_map,
                    rewritten_group_input,
                    plan.context,
                    target_max_block_size,
                )

                for original_physical_op in fusible_operators:
                    del rewritten_op_map[original_physical_op]
                rewritten_op_map[fused_physical_op] = fused_logical_op
                rewritten_operator_by_original[physical_op] = fused_physical_op
                new_fused_operators.append(fused_physical_op)
                return fused_physical_op

            # Nothing ends here, so continue through every input.
            rewritten_inputs = []
            inputs_changed = False
            for original_input in physical_op.input_dependencies:
                rewritten_input = rewrite_upstream_graph(original_input)
                rewritten_inputs.append(rewritten_input)
                if rewritten_input is not original_input:
                    inputs_changed = True

            if inputs_changed:
                # Output links are rebuilt after all input links are final.
                physical_op._input_dependencies = rewritten_inputs
            rewritten_operator_by_original[physical_op] = physical_op
            return physical_op

        rewritten_final_operator = rewrite_upstream_graph(original_final_operator)

        if not new_fused_operators:
            return plan
        self._rebuild_output_dependencies(rewritten_final_operator)
        return PhysicalPlan(rewritten_final_operator, rewritten_op_map, plan.context)

    @staticmethod
    def _context_allows_fusion(context: Any) -> bool:
        """Return whether ``context`` enables fusion without changing semantics.

        Fusion is rejected when exception propagation, retry matching, or actor task
        admission would differ from the original sequence of actor pools.
        """

        if getattr(context, "enable_cudf_actor_fusion", False) is not True:
            return False

        if context.raise_original_map_exception is not False:
            return False

        if not isinstance(context.actor_task_retry_on_errors, bool):
            return False
        if not isinstance(context.retried_map_errors, bool):
            return False

        return context.max_tasks_in_flight_per_actor is None

    @staticmethod
    def _count_logical_consumers(
        rewritten_op_map: Mapping[PhysicalOperator, LogicalOperator],
    ) -> Dict[LogicalOperator, int]:
        """Count each logical operator's direct consumers.

        The set removes duplicate logical operators from the physical-to-logical map.
        Operators with no consumers are omitted.
        """

        logical_counts: Dict[LogicalOperator, int] = {}
        for logical_op in set(rewritten_op_map.values()):
            for input_logical_op in logical_op.input_dependencies:
                logical_counts[input_logical_op] = (
                    logical_counts.get(input_logical_op, 0) + 1
                )
        return logical_counts

    @staticmethod
    def _count_physical_consumers(
        final_physical_operator: PhysicalOperator,
    ) -> Dict[PhysicalOperator, int]:
        """Count each physical operator's direct consumers.

        The set ensures shared operators are counted once during traversal. Operators
        with no consumers are omitted.
        """

        physical_counts: Dict[PhysicalOperator, int] = {}
        for physical_op in set(final_physical_operator.post_order_iter()):
            for input_physical_op in physical_op.input_dependencies:
                physical_counts[input_physical_op] = (
                    physical_counts.get(input_physical_op, 0) + 1
                )
        return physical_counts

    @classmethod
    def _collect_fusible_operators(
        cls,
        physical_op: PhysicalOperator,
        rewritten_op_map: Mapping[PhysicalOperator, LogicalOperator],
        logical_consumer_counts: Mapping[LogicalOperator, int],
        physical_consumer_counts: Mapping[PhysicalOperator, int],
    ) -> Tuple[Tuple[PhysicalOperator, ...], Optional[int]]:
        """Collect the maximal fusible group ending at ``physical_op``.

        Walks upstream while the path has one input and stops before an operator that is
        shared, ineligible, incompatible, or has a conflicting block-size override.
        Both consumer-count maps are checked because physical planning can copy logical
        operators.

        Args:
            physical_op: Downstream operator where collection begins.
            rewritten_op_map: Physical-to-logical map for the rewritten plan.
            logical_consumer_counts: Consumer counts from the logical plan.
            physical_consumer_counts: Consumer counts from the physical plan.

        Returns:
            The operators in execution order and the block-size override to preserve.
            An empty tuple means ``physical_op`` is ineligible; one operator means there
            is no adjacent map to fuse.

        Examples:
            Given an eligible path::

                Input -> MapA -> MapB -> MapC

            starting at ``MapC`` returns ``((MapA, MapB, MapC), None)``. The
            second value is the block-size override to preserve. If ``CpuMap`` is
            not eligible::

                Input -> CpuMap -> MapB -> MapC

            starting at ``MapC`` returns ``((MapB, MapC), None)``. The main traversal
            continues from ``CpuMap`` so earlier parts of the plan are not skipped.
        """

        fusion_config = cls._fusion_config_if_eligible(
            physical_op,
            rewritten_op_map.get(physical_op),
        )
        if fusion_config is None:
            return (), None

        fusible_operators = [physical_op]
        current_physical_op = physical_op

        # A fused operator can preserve only one block-size override.
        target_max_block_size = physical_op.target_max_block_size_override

        while len(current_physical_op.input_dependencies) == 1:
            upstream_physical_op = current_physical_op.input_dependencies[0]
            upstream_logical_op = rewritten_op_map.get(upstream_physical_op)
            upstream_fusion_config = cls._fusion_config_if_eligible(
                upstream_physical_op, upstream_logical_op
            )
            upstream_target_size = upstream_physical_op.target_max_block_size_override
            has_one_logical_consumer = (
                logical_consumer_counts.get(upstream_logical_op, 0) == 1
            )
            has_one_physical_consumer = (
                physical_consumer_counts.get(upstream_physical_op, 0) == 1
            )
            block_sizes_conflict = (
                target_max_block_size is not None
                and upstream_target_size is not None
                and target_max_block_size != upstream_target_size
            )

            if (
                not has_one_logical_consumer
                or not has_one_physical_consumer
                or upstream_fusion_config is None
                or fusion_config != upstream_fusion_config
                or block_sizes_conflict
            ):
                break
            fusible_operators.append(upstream_physical_op)
            current_physical_op = upstream_physical_op

            if target_max_block_size is None:
                target_max_block_size = upstream_target_size

        return tuple(reversed(fusible_operators)), target_max_block_size

    @staticmethod
    def _fusion_config_if_eligible(
        physical_op: PhysicalOperator,
        logical_op: Optional[LogicalOperator],
    ) -> Optional["_CudfMapFusionConfig"]:
        """Return the settings used to compare an eligible cuDF actor map.

        Eligible maps use a synchronous callable class, fixed zero-copy cuDF batches,
        serial actor execution, one GPU, and supported static actor options. Returns
        ``None`` when one fused operator cannot preserve the map's behavior. UDF call
        style is inspected without constructing or invoking user code on the driver.
        Exact built-in types prevent values such as ``True`` from being treated as the
        integer ``1``.
        """

        if not isinstance(physical_op, ActorPoolMapOperator):
            return None
        if not physical_op.supports_fusion():
            return None
        if not isinstance(logical_op, MapBatches):
            return None

        # A replacement produced by this rule must not be fused again.
        if logical_op.fn is _FusedCudfMapBatches:
            return None

        # Fusion removes intermediate batch boundaries, so their policies must collapse.
        if logical_op.batch_format != "cudf":
            return None
        if type(logical_op.batch_size) is not int or logical_op.batch_size <= 0:
            return None
        if logical_op.zero_copy_batch is not True:
            return None
        if logical_op.ray_remote_args_fn is not None:
            return None
        if logical_op.per_block_limit is not None:
            return None
        if not _is_synchronous_callable_class(logical_op.fn):
            return None

        actor_pool = logical_op.compute
        remote_args = logical_op.ray_remote_args

        # Shared UDF instances require serial execution and known actor options.
        if type(actor_pool) is not ActorPoolStrategy:
            return None
        if actor_pool.enable_true_multi_threading is not False:
            return None
        if actor_pool.max_tasks_in_flight_per_actor is not None:
            return None
        if type(remote_args) is not dict:
            return None
        if any(key not in _FUSION_SAFE_REMOTE_ARGS for key in remote_args):
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
    def _create_fused_map_operator(
        physical_operators: Tuple[PhysicalOperator, ...],
        rewritten_op_map: Mapping[PhysicalOperator, LogicalOperator],
        input_physical_op: PhysicalOperator,
        context: Any,
        target_max_block_size: Optional[int],
    ) -> Tuple[PhysicalOperator, MapBatches]:
        """Create one replacement for an execution-ordered physical group.

        ``physical_operators`` must be ordered upstream to downstream, and
        ``input_physical_op`` must already be rewritten. This method builds stage
        descriptions, creates a synthetic ``MapBatches``, and asks Ray's standard UDF
        planner to create the executable operator. Original logical operators remain
        attached for stats and plan inspection.

        Args:
            physical_operators: Eligible operators ordered upstream to downstream.
            rewritten_op_map: Physical-to-logical map for the rewritten plan.
            input_physical_op: Rewritten input to the first operator in the group.
            context: Ray Data context captured by the plan.
            target_max_block_size: Output block-size override to preserve.

        Returns:
            The executable physical replacement and the synthetic logical operator
            required by ``PhysicalPlan.op_map``.
        """

        logical_operators = tuple(
            rewritten_op_map[physical_op] for physical_op in physical_operators
        )

        # Pass descriptions so each actor constructs its own GPU-local UDF state.
        stage_descriptions = []
        for stage_index, logical_op in enumerate(logical_operators, start=1):
            constructor_args = logical_op.fn_constructor_args
            if constructor_args is None:
                constructor_args = ()

            constructor_kwargs = logical_op.fn_constructor_kwargs
            if constructor_kwargs is None:
                constructor_kwargs = {}

            call_args = logical_op.fn_args
            if call_args is None:
                call_args = ()

            call_kwargs = logical_op.fn_kwargs
            if call_kwargs is None:
                call_kwargs = {}

            stage_descriptions.append(
                _CudfMapStage(
                    udf_class=logical_op.fn,
                    error_label=f"{stage_index}: {logical_op.name}",
                    constructor_args=constructor_args,
                    constructor_kwargs=constructor_kwargs,
                    call_args=call_args,
                    call_kwargs=call_kwargs,
                )
            )
        stages = tuple(stage_descriptions)
        first_logical_op = logical_operators[0]

        # Matching settings let the first map define the input policy.
        fused_logical_op = MapBatches(
            _FusedCudfMapBatches,
            input_dependencies=[first_logical_op.input_dependencies[0]],
            can_modify_num_rows=any(
                logical_op.can_modify_num_rows for logical_op in logical_operators
            ),
            batch_size=first_logical_op.batch_size,
            batch_format="cudf",
            zero_copy_batch=True,
            fn_constructor_args=(stages,),
            min_rows_per_bundled_input=first_logical_op.min_rows_per_bundled_input,
            compute=first_logical_op.compute,
            ray_remote_args=dict(first_logical_op.ray_remote_args),
        )

        udf_names = "->".join(stage.udf_class.__name__ for stage in stages)

        # _name is display-only, but MapBatches is frozen.
        object.__setattr__(fused_logical_op, "_name", f"MapBatches({udf_names})")

        # Reuse Ray's planner to preserve normal actor and batch-conversion behavior.
        fused_physical_op = create_udf_map_operator(
            fused_logical_op,
            input_physical_op,
            context,
            target_max_block_size_override=target_max_block_size,
        )

        # Preserve original lineage for stats and plan inspection.
        fused_physical_op.set_logical_operators(*logical_operators)
        return fused_physical_op, fused_logical_op

    @staticmethod
    def _rebuild_output_dependencies(final_physical_operator: PhysicalOperator) -> None:
        """Rebuild reverse links after rewriting input dependencies.

        Ray stores every edge in both directions, but traversal updates only consumer
        input links. This method treats those input links as authoritative and recreates
        each producer's output links.
        """

        physical_operators = set(final_physical_operator.post_order_iter())

        for physical_op in physical_operators:
            physical_op._output_dependencies = []

        for physical_op in physical_operators:
            for input_physical_op in physical_op.input_dependencies:
                input_physical_op._output_dependencies.append(physical_op)


@dataclass(frozen=True, eq=False)
class _CudfMapFusionConfig:
    """Execution settings that one fused actor-map group must share.

    Ambiguous or exception-raising comparisons make two stages incompatible.
    """

    actor_pool: ActorPoolStrategy
    batch_size: int
    min_rows_per_bundle: Optional[int]
    ray_remote_args: Mapping[str, Any]

    def __eq__(self, other: object) -> bool:
        """Return whether two configurations fit one physical replacement.

        Args:
            other: Configuration to compare with this one.

        Returns:
            ``True`` when both configurations can use the same fused operator.
        """

        if not isinstance(other, _CudfMapFusionConfig):
            return False
        if self.actor_pool != other.actor_pool:
            return False
        if self.batch_size != other.batch_size:
            return False
        if not self._values_match_exactly(
            self.min_rows_per_bundle,
            other.min_rows_per_bundle,
        ):
            return False
        return self._values_match_exactly(
            self.ray_remote_args,
            other.ray_remote_args,
        )

    @staticmethod
    def _values_match_exactly(left: Any, right: Any) -> bool:
        """Compare nested settings without type coercion.

        Values must have identical types and recursively equal contents. Comparisons
        that raise or return an array-like result are treated as unequal so ambiguous
        settings never enable fusion.

        Args:
            left: First configuration value.
            right: Second configuration value.

        Returns:
            ``True`` when both values have the same types and equal contents.

        Examples:
            Matching nested settings compare equal:

            >>> _CudfMapFusionConfig._values_match_exactly(
            ...     {"num_gpus": 1, "resources": {"worker": 1}},
            ...     {"num_gpus": 1, "resources": {"worker": 1}},
            ... )
            True

            Values with different types do not compare equal:

            >>> _CudfMapFusionConfig._values_match_exactly(
            ...     {"num_gpus": 1}, {"num_gpus": True}
            ... )
            False
        """

        if type(left) is not type(right):
            return False

        try:
            if isinstance(left, Mapping):
                if len(left) != len(right):
                    return False
                for key in left:
                    if key not in right:
                        return False
                    if not _CudfMapFusionConfig._values_match_exactly(
                        left[key], right[key]
                    ):
                        return False
                return True

            if isinstance(left, (list, tuple)):
                if len(left) != len(right):
                    return False
                for left_value, right_value in zip(left, right):
                    if not _CudfMapFusionConfig._values_match_exactly(
                        left_value, right_value
                    ):
                        return False
                return True

            result = left == right
        except Exception:
            return False

        if type(result) is not bool:
            return False
        return result


@dataclass(frozen=True, eq=False)
class _CudfMapStage:
    """Serializable representation of one original callable-class UDF.

    Keeps constructor arguments separate from per-batch call arguments so the fused
    actor can reproduce the original ``map_batches`` stage. ``error_label`` identifies
    that stage if construction or execution fails.
    """

    udf_class: type
    error_label: str
    constructor_args: Iterable[Any] = ()
    constructor_kwargs: Dict[str, Any] = field(default_factory=dict)
    call_args: Iterable[Any] = ()
    call_kwargs: Dict[str, Any] = field(default_factory=dict)


class _FusedCudfMapBatches:
    """Run several cuDF UDFs as one Ray actor-map operation.

    Ray's map planner expects one callable class for each physical map operator. This
    wrapper makes several original UDF classes look like one callable class, allowing
    Ray to use a single actor pool for the entire fused group.

    Each actor constructs its own UDF instances and passes each cuDF result directly to
    the next UDF. This avoids the actor, conversion, and rebatching boundaries that
    would otherwise exist between separate ``map_batches`` operations.
    """

    def __init__(self, stages: Tuple[_CudfMapStage, ...]):
        """Construct every original UDF inside the actor that owns its state.

        Args:
            stages: UDF construction and call descriptions in execution order.

        Raises:
            RuntimeError: If a UDF constructor fails. The stage is identified in the
                message and the original exception is retained as the cause.
        """

        self._stages = tuple(stages)
        udf_instances = []
        for stage in self._stages:
            try:
                udf_instance = stage.udf_class(
                    *stage.constructor_args,
                    **stage.constructor_kwargs,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Fused cuDF map_batches stage {stage.error_label!r} failed "
                    "during construction."
                ) from exc
            udf_instances.append(udf_instance)
        self._udf_instances = tuple(udf_instances)

    def __call__(self, batch: Any) -> Any:
        """Run ``batch`` through every UDF and return the final cuDF DataFrame.

        Each intermediate result must remain a cuDF DataFrame because fusion removes
        the Ray boundary that normally validates and converts batches between stages.

        Args:
            batch: cuDF DataFrame provided to the first UDF.

        Returns:
            The cuDF DataFrame returned by the final UDF.

        Raises:
            RuntimeError: If a UDF raises. The stage is identified in the message and
                the original exception is retained as the cause.
            TypeError: If any UDF returns an object other than a cuDF DataFrame.
        """

        for stage, udf_instance in zip(self._stages, self._udf_instances):
            try:
                batch = udf_instance(batch, *stage.call_args, **stage.call_kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"Fused cuDF map_batches stage {stage.error_label!r} failed."
                ) from exc

            if not _is_cudf_dataframe(batch):
                returned_type = type(batch)
                output_type = f"{returned_type.__module__}.{returned_type.__qualname__}"
                raise TypeError(
                    f"Fused cuDF map_batches stage {stage.error_label!r} returned "
                    f"{output_type}; expected cudf.DataFrame."
                )
        return batch
