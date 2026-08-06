"""Run compatible cuDF ``map_batches`` UDFs in the same Ray actor.

Ray normally gives every actor-based ``map_batches`` call its own actor pool::

    Input -> MapBatches(Udf1) -> MapBatches(Udf2) -> MapBatches(Udf3)

Moving a cuDF batch between those pools introduces an execution boundary. This rule
removes that boundary by replacing compatible consecutive calls with one actor-pool
operator::

    Input -> MapBatches(Udf1->Udf2->Udf3)

Each fused actor constructs all of the original UDFs and passes each returned cuDF
DataFrame directly to the next UDF. The rule uses Ray's normal UDF map planner to
create the replacement so actor setup and batch conversion keep their standard
behavior.
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

# Choose which physical operators can safely share one actor pool on the driver.


# A fused operator has one set of actor options. Skip fusion when an option is not on
# this list, because silently dropping or combining an unknown option could change how
# the user's workload runs.
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
        """Replace eligible consecutive cuDF actor maps in a physical plan.

        The plan has already been lowered into executable physical operators. This
        method walks backward from the final operator, replaces each safe linear group
        of cuDF actor maps, and reconnects the surrounding operators.

        Returns the original plan if no replacement is made. Otherwise, returns a new
        ``PhysicalPlan`` wrapper around the rewritten graph while preserving the plan
        context and original logical lineage.
        """

        if not self._context_allows_fusion(plan.context):
            return plan

        # The rewritten plan needs a matching physical-to-logical lookup. Work on a
        # copy so removing absorbed operators does not alter the input plan's lookup.
        rewritten_op_map = plan.op_map.copy()

        # An operator cannot be absorbed if another downstream operator still needs it.
        # Ray planning can duplicate a logical operator in the physical graph, so check
        # for sharing in both representations before deciding that a path is linear.
        (
            logical_consumer_counts,
            physical_consumer_counts,
        ) = self._count_consumers_in_both_plan_views(rewritten_op_map, plan.dag)

        # A DAG can reach the same operator through multiple paths. Remember the
        # replacement chosen for each original operator so every path reuses it.
        rewritten_operators: Dict[PhysicalOperator, PhysicalOperator] = {}

        # Keeping the created replacements also tells us whether anything changed.
        fused_operators: set[PhysicalOperator] = set()

        def rewrite_subdag(physical_op: PhysicalOperator) -> PhysicalOperator:
            """Rewrite ``physical_op`` and all operators that feed into it.

            If ``physical_op`` ends a group that can share one actor pool, replace that
            group and continue from the operator immediately before it. Otherwise,
            continue through every direct input. Starting this process at ``plan.dag``
            reaches every operator in the physical plan.

            Operators outside a fused group are reused. Their input links may be updated
            to point to replacements. Returns the final operator for this rewritten
            portion of the graph.
            """
            # If another path already reached this operator, return the same replacement
            # so shared parts of the DAG stay shared after the rewrite.
            if physical_op in rewritten_operators:
                return rewritten_operators[physical_op]

            # Look only for one consecutive group ending at the current operator. The
            # recursive calls find any other groups earlier in the plan. The helper also
            # returns the output block-size setting the replacement must preserve.
            fusible_operators, target_max_block_size = self._collect_fusible_chain(
                physical_op,
                rewritten_op_map,
                logical_consumer_counts,
                physical_consumer_counts,
            )

            if len(fusible_operators) >= 2:
                # These operators will become one replacement. First rewrite the graph
                # feeding the group so fusion here does not skip earlier work or another
                # input path.
                rewritten_input = rewrite_subdag(
                    fusible_operators[0].input_dependencies[0]
                )

                # Ray needs two views of the replacement: an executable physical
                # operator for the DAG and a synthetic logical operator for op_map.
                fused_physical_op, fused_logical_op = self._plan_fused_map_operator(
                    fusible_operators,
                    rewritten_op_map,
                    rewritten_input,
                    plan.context,
                    target_max_block_size,
                )

                # Remove the physical-to-logical entries for operators that no longer
                # exist in the rewritten DAG, then add the entry for their replacement.
                # The physical operator separately retains all original logical
                # operators for lineage and stats.
                for stage_op in fusible_operators:
                    rewritten_op_map.pop(stage_op)
                rewritten_op_map[fused_physical_op] = fused_logical_op
                rewritten_operators[physical_op] = fused_physical_op
                fused_operators.add(fused_physical_op)
                return fused_physical_op

            # This operator cannot end a fused group. Keep it, but keep walking through
            # every operator that provides its input so the rest of the DAG is checked.
            rewritten_inputs = [
                rewrite_subdag(input_op) for input_op in physical_op.input_dependencies
            ]

            # A recursive call may have replaced an input group. Point this reused
            # operator at the returned replacement when that happens.
            if any(
                rewritten is not original
                for rewritten, original in zip(
                    rewritten_inputs,
                    physical_op.input_dependencies,
                )
            ):
                # Ray executes from input_dependencies, so changing these links is
                # enough during traversal. Reverse output links are rebuilt afterward.
                physical_op._input_dependencies = rewritten_inputs
            rewritten_operators[physical_op] = physical_op
            return physical_op

        # plan.dag is the final physical operator, not a separate graph container.
        # Following its input_dependencies backward reaches the entire physical plan.
        new_dag = rewrite_subdag(plan.dag)

        # Avoid returning a new plan object when the rewrite made no replacements.
        if not fused_operators:
            return plan
        self._rebuild_output_dependencies(new_dag)
        return PhysicalPlan(new_dag, rewritten_op_map, plan.context)

    @staticmethod
    def _context_allows_fusion(context: Any) -> bool:
        """Return whether the current Ray Data settings preserve fusion semantics."""

        # Fusion is experimental and must be explicitly enabled by the user before the
        # Dataset is created.
        if getattr(context, "enable_cudf_actor_fusion", False) is not True:
            return False

        # The combined UDF adds the stage name by wrapping failures in RuntimeError.
        # Skip fusion when Ray is expected to expose the user's original exception.
        if context.raise_original_map_exception is not False:
            return False

        # Class-specific retry policies inspect the exception type, which the wrapper
        # changes. Boolean policies are safe because they mean retry all or retry none.
        if (
            type(context.actor_task_retry_on_errors) is not bool
            or type(context.retried_map_errors) is not bool
        ):
            return False

        # Multiple original pools and one combined pool can admit tasks differently.
        # Until that behavior is preserved explicitly, use fusion only with Ray's
        # default actor task-admission setting.
        return context.max_tasks_in_flight_per_actor is None

    @staticmethod
    def _count_consumers_in_both_plan_views(
        rewritten_op_map: Mapping[PhysicalOperator, LogicalOperator],
        physical_root: PhysicalOperator,
    ) -> Tuple[Dict[LogicalOperator, int], Dict[PhysicalOperator, int]]:
        """Count how many downstream operators consume each logical and physical op.

        Fusion may remove an operator only when exactly one downstream operator uses
        it. Ray planning can represent one logical operator with multiple physical
        operators, so checking only one view could incorrectly remove work needed by a
        different path.
        """

        # Count consumers in the logical plan captured by the operator map.
        logical_counts: Dict[LogicalOperator, int] = {}
        for op in set(rewritten_op_map.values()):
            for input_op in op.input_dependencies:
                logical_counts[input_op] = logical_counts.get(input_op, 0) + 1

        # Count consumers again in the executable physical plan.
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
        """Collect consecutive operators that can be fused with ``op``.

        Starting at ``op``, walk toward its inputs until fusion would change behavior.
        The returned tuple is ordered as the operators execute, from upstream to
        downstream. An empty tuple means ``op`` is not eligible; a one-item tuple means
        it is eligible but has no compatible neighbor to fuse with. The recursive
        caller discovers other groups elsewhere in the DAG.
        """

        chain_config = cls._fusion_config_if_eligible(
            op,
            rewritten_op_map.get(op),
        )
        if chain_config is None:
            return (), None

        chain = [op]
        upstream_cursor = op

        # One fused operator can expose only one output block-size override. Start with
        # the downstream setting and inherit an upstream setting only if none is set.
        target_max_block_size = op.target_max_block_size_override

        # Actor-based map operators have one input. Stop if the path is no longer
        # linear instead of guessing how multiple inputs should be combined.
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

            # Stop before the first operator that cannot safely share the replacement:
            # another path needs it, it is ineligible, its actor/batch settings differ,
            # or it requires a different output block size.
            if (
                upstream_is_shared
                or upstream_config is None
                or not chain_config.is_compatible_with(upstream_config)
                or block_sizes_conflict
            ):
                break
            chain.append(upstream)
            upstream_cursor = upstream

            # Preserve the only explicit block-size override found in the group.
            if target_max_block_size is None:
                target_max_block_size = upstream_target_size

        return tuple(reversed(chain)), target_max_block_size

    @staticmethod
    def _fusion_config_if_eligible(
        physical_op: PhysicalOperator,
        logical_op: Optional[LogicalOperator],
    ) -> Optional["_CudfMapFusionConfig"]:
        """Return settings needed to compare an eligible cuDF actor map.

        Returns ``None`` whenever one combined actor map cannot preserve the original
        operator's execution behavior.
        """

        # Fusion applies only to physical actor maps created from logical MapBatches.
        # Do not fuse a replacement produced by this rule again on a later optimizer
        # pass; it already represents all of its original operators.
        if (
            not isinstance(physical_op, ActorPoolMapOperator)
            or not physical_op.supports_fusion()
            or not isinstance(logical_op, MapBatches)
            or logical_op.fn is _FusedCudfMapBatches
        ):
            return None

        # The replacement has one input conversion and one batching policy for the
        # entire group. Require a fixed cuDF batch size and zero-copy input so removing
        # intermediate boundaries does not remove a requested copy or change batching.
        # Dynamic actor options and per-block limits cannot be represented by one
        # replacement. A callable class is required because these maps run in actors.
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

        # The combined executor calls each UDF synchronously and expects one DataFrame.
        # Inspect the class without constructing it on the driver, because construction
        # may allocate GPU state or run other user setup intended for the actor.
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

        # One fused actor owns all UDF instances. Reject actor-pool modes that could call
        # those instances concurrently, and reject remote options whose behavior this
        # rule does not explicitly know how to preserve.
        if (
            type(actor_pool) is not ActorPoolStrategy
            or actor_pool.enable_true_multi_threading is not False
            or actor_pool.max_tasks_in_flight_per_actor is not None
            or type(remote_args) is not dict
            or not remote_args.keys() <= _SUPPORTED_REMOTE_ARGS
        ):
            return None

        # Run one batch at a time through the shared UDF instances. The current feature
        # also supports exactly one full GPU per fused actor, matching its intended cuDF
        # execution model.
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
        """Create one logical and physical replacement for consecutive actor maps.

        The logical replacement describes one callable that runs all original UDFs.
        Ray's standard planner then turns that description into the executable physical
        operator. Both are returned because ``PhysicalPlan`` stores the physical DAG and
        a physical-to-logical operator map.
        """

        # UDF classes and arguments live on the logical operators, so recover those
        # operators from the physical-to-logical lookup before building the replacement.
        logical_chain = tuple(
            rewritten_op_map[physical_op] for physical_op in physical_chain
        )

        # Send each UDF class and its original arguments to the fused actor as data. The
        # actor, rather than the driver, constructs the UDF so GPU setup and mutable
        # state remain local to the worker that will use them.
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

        # Ray's normal planner expects one logical MapBatches per physical map. Create a
        # synthetic logical map whose callable runs every original stage. Eligibility
        # checks guarantee compatible execution settings, while the first map supplies
        # the group's input and the batching policy used when data enters the group.
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

        # Show the original UDF order in plans and stats. MapBatches is frozen, so use
        # object.__setattr__ for this display-only private field.
        object.__setattr__(fused_logical_op, "_name", f"MapBatches({names})")

        # Let Ray's existing UDF planner create the executable actor map. Reimplementing
        # that logic here could diverge from normal actor setup, cuDF conversion, and
        # BatchMapTransformFn behavior.
        fused_physical_op = create_udf_map_operator(
            fused_logical_op,
            input_physical_dag,
            context,
            target_max_block_size_override=target_max_block_size,
        )

        # Keep all original logical operators on the physical replacement so lineage,
        # stats, and plan inspection still show the work that was combined.
        fused_physical_op.set_logical_operators(*logical_chain)
        return fused_physical_op, fused_logical_op

    @staticmethod
    def _rebuild_output_dependencies(root: PhysicalOperator) -> None:
        """Rebuild producer-to-consumer links after input links are rewritten.

        Ray stores each edge twice: a consumer points to its inputs and each input points
        back to its consumers. The rewrite changes only the consumer-side links while it
        traverses the graph, so the old reverse links must be discarded and recreated.
        """

        operators = set(root.post_order_iter())

        # Remove reverse links that may still point to operators absorbed by fusion.
        for op in operators:
            op._output_dependencies = []

        # Treat input_dependencies as the source of truth and recreate every reverse
        # producer-to-consumer link from them.
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
    """Compare configuration values without Python's surprising coercions.

    Configuration values can contain nested dictionaries, lists, or objects whose
    ``==`` operator returns an array instead of one boolean. Fusion requires a definite,
    exact match; an ambiguous comparison is treated as incompatible.
    """

    # Keep values such as True and 1 distinct even though Python considers them equal.
    if type(left) is not type(right):
        return False

    # Compare containers recursively so a nested array-like value cannot turn the
    # result into an array of booleans.
    if isinstance(left, Mapping):
        if left.keys() != right.keys():
            return False
        return all(_equal_config_values(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _equal_config_values(left_value, right_value)
            for left_value, right_value in zip(left, right)
        )

    # User-provided configuration objects may raise from equality. In that case, skip
    # fusion instead of risking different settings in one replacement.
    try:
        result = left == right
    except Exception:
        return False

    # Accept only a real bool; numpy and similar libraries may return array-like values.
    return type(result) is bool and result


# Execute the combined user UDFs inside each fused Ray actor.


@dataclass(frozen=True, eq=False)
class _CudfMapStage:
    """Serializable instructions for constructing and calling one original UDF.

    The driver creates this description, then the fused actor uses it to construct the
    user's class locally and call it with the same arguments as the original
    ``map_batches`` stage.
    """

    udf_class: type
    error_label: str
    constructor_args: Iterable[Any] = ()
    constructor_kwargs: Dict[str, Any] = field(default_factory=dict)
    call_args: Iterable[Any] = ()
    call_kwargs: Dict[str, Any] = field(default_factory=dict)


class _FusedCudfMapBatches:
    """Present several original UDFs to Ray as one callable actor transform.

    Ray's map planner accepts one callable class per actor map. This adapter lets the
    planner create one actor pool while still running every original UDF in order.
    """

    def __init__(self, stages: Tuple[_CudfMapStage, ...]):
        """Construct each original UDF inside the actor that will execute it.

        Constructing here keeps GPU resources and mutable UDF state on the worker instead
        of creating them on the driver and trying to serialize them.
        """

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
                # Give each UDF's exact return value to the next UDF. Ray performs batch
                # conversion only before the first stage and after the last, which is the
                # performance benefit of removing the intermediate actor boundaries.
                batch = instance(batch, *stage.call_args, **stage.call_kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"Fused cuDF map_batches stage {stage.error_label!r} failed."
                ) from exc

            # Without an intermediate Ray boundary, no normal batch conversion runs
            # between stages. Require cuDF here so the next UDF receives the format it
            # was promised by its original map_batches call.
            if not _is_cudf_dataframe(batch):
                output_type = f"{type(batch).__module__}.{type(batch).__qualname__}"
                raise TypeError(
                    f"Fused cuDF map_batches stage {stage.error_label!r} returned "
                    f"{output_type}; expected cudf.DataFrame."
                )
        return batch
