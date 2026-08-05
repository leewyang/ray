"""Fuse consecutive actor-based cuDF ``map_batches`` after physical planning.

This rule receives a ``PhysicalPlan``; it does not rewrite the Dataset's logical
plan. For example, it changes the executable DAG from::

    Input -> MapBatches(Udf1) -> MapBatches(Udf2) -> MapBatches(Udf3)

to::

    Input -> MapBatches(Udf1->Udf2->Udf3)

The replacement is still a normal Ray ``ActorPoolMapOperator``. The rule creates a
temporary logical ``MapBatches`` describing the combined UDFs, then asks Ray's
existing map planner to build the physical operator. Inside each actor, the composite
instantiates the user UDF objects and passes each returned cuDF DataFrame directly to
the next stage.
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

# Only allow actor options whose behavior fusion knows how to preserve. Any
# new or unsupported option prevents fusion by default.
_SUPPORTED_REMOTE_ARGS = frozenset(
    "_labels accelerator_type allow_out_of_order_execution enable_task_events "
    "fallback_strategy label_selector max_concurrency max_restarts "
    "max_task_retries memory num_cpus num_gpus placement_group "
    "placement_group_bundle_index placement_group_capture_child_tasks resources "
    "runtime_env scheduling_strategy".split()
)


@dataclass(frozen=True)
class _CudfMapStageDefinition:
    """Store everything a fused actor needs for one user-defined function (UDF).

    For a ``Udf1 -> Udf2`` chain, the optimizer creates one definition for ``Udf1``
    and another for ``Udf2``. Each definition stores the callable class supplied to
    ``map_batches``, the arguments needed to construct it, the arguments passed on
    every batch, and a name for error messages. ``_FusedCudfMapBatches`` reads these
    definitions inside each actor. This class only stores data; it never calls user
    code.
    """

    # The user's callable class and the stage name shown in errors.
    udf_class: type
    error_label: str
    # Arguments used once when each actor constructs this stage's UDF instance.
    # User values may be unhashable, so they are stored but excluded from equality.
    constructor_args: Iterable[Any] = field(default=(), compare=False, hash=False)
    constructor_kwargs: Dict[str, Any] = field(
        default_factory=dict, compare=False, hash=False
    )
    # Additional arguments passed every time this stage processes a batch.
    call_args: Iterable[Any] = field(default=(), compare=False, hash=False)
    call_kwargs: Dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


class _FusedCudfMapBatches:
    """Run all UDFs in a fused chain inside one Ray actor.

    For ``Udf1 -> Udf2``, Ray creates one instance of this class in every pool actor.
    Its constructor uses the stage definitions above to create separate ``Udf1`` and
    ``Udf2`` objects. For each input batch, ``__call__`` runs ``Udf1`` and passes the
    exact returned cuDF DataFrame to ``Udf2``. This is the class that runs user code;
    the stage-definition class above only stores the information it needs.
    """

    def __init__(self, stage_definitions: Tuple[_CudfMapStageDefinition, ...]):
        self._stage_definitions = tuple(stage_definitions)
        instances = []
        # Create a separate UDF instance for every stage, even when the same class
        # appears more than once.
        for stage in self._stage_definitions:
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
        for stage, instance in zip(self._stage_definitions, self._instances):
            try:
                # The surrounding BatchMapTransformFn handles ingress batching and
                # format conversion, then final output shaping. Between stages, pass
                # the exact returned object: no copy, conversion, normalization, or
                # rebatching.
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


@dataclass(frozen=True)
class _CudfMapFusionSettings:
    """Store the settings used to decide whether two UDF maps can be fused.

    The optimizer creates one of these objects for each ``MapBatches`` that might be
    fused. ``compatible_with`` checks that two maps use the same actor-pool strategy,
    batch size, minimum rows grouped into an input task, and Ray actor options. This
    class only compares settings; it does not create actors or run UDFs.
    """

    # How Ray creates and scales the actors that execute this map.
    actor_pool: ActorPoolStrategy = field(compare=False)
    # How input rows are grouped before one actor call.
    batch_size: int
    min_rows_per_bundle: Optional[int]
    # CPU, GPU, memory, scheduling, retry, and other Ray actor options.
    ray_remote_args: Mapping[str, Any] = field(compare=False)

    def compatible_with(self, other: "_CudfMapFusionSettings") -> bool:
        if (
            self.actor_pool != other.actor_pool
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
    """Compare execution settings conservatively and without type coercion.

    Mappings, lists, and tuples are compared recursively. Values such as
    ``True`` and ``1`` must not compare as the same resource setting. Only a plain
    boolean equality result is accepted; array-like results are incompatible.
    """

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
    """Replace compatible cuDF actor maps with one executable actor-map operator.

    For ``Input -> Udf1 -> Udf2 -> Output``, ``apply`` checks that the UDF maps form
    a linear chain, do not cross a shared upstream node, and use compatible settings.
    It then creates their stage definitions, asks Ray's normal planner to build one
    ``MapBatches(Udf1->Udf2)``, and reconnects ``Input`` and ``Output`` around that
    replacement. This optimizer rule runs while Ray builds the physical plan; it
    never processes a user batch itself.
    """

    def apply(self, plan: PhysicalPlan) -> PhysicalPlan:
        if not self._context_allows_fusion(plan.context):
            return plan

        op_map = plan.op_map.copy()
        # ``op_map`` records the logical origin of each physical node. Count
        # consumers in both representations before rewriting, and never absorb an
        # upstream node that another branch still consumes. The logical check is
        # needed because the logical and physical DAGs can represent sharing
        # differently after planning.
        logical_consumer_counts = self._logical_consumer_counts(op_map)
        physical_consumer_counts = self._physical_consumer_counts(plan.dag)
        # A physical plan is a DAG, not necessarily a tree. Memoization rewrites a
        # shared node once and gives every consumer the same replacement.
        rewritten_operators: Dict[PhysicalOperator, PhysicalOperator] = {}
        did_fuse = False

        def rewrite_operator(op: PhysicalOperator) -> PhysicalOperator:
            """Rewrite ``op`` and everything upstream, returning its replacement.

            If a compatible cuDF actor-map chain ends here, replace it with one
            fused operator. Otherwise, keep ``op`` and rewrite its inputs.
            """
            nonlocal did_fuse
            if op in rewritten_operators:
                return rewritten_operators[op]

            fusible_chain, effective_target_size = self._collect_fusible_run(
                op,
                op_map,
                logical_consumer_counts,
                physical_consumer_counts,
            )

            if len(fusible_chain) >= 2:
                # Collection is downstream-first, so the final entry is the first
                # map that receives data. That map's input feeds the replacement.
                logical_chain_downstream_first = tuple(
                    op_map[stage_op] for stage_op in fusible_chain
                )
                fused_logical_op = self._create_fused_logical_op(
                    logical_chain_downstream_first
                )
                first_stage_physical_op = fusible_chain[-1]
                # Lower the temporary combined MapBatches with Ray's normal planner
                # so actor setup, cuDF conversion, and execution settings stay stock.
                fused_physical_op = create_udf_map_operator(
                    fused_logical_op,
                    rewrite_operator(first_stage_physical_op.input_dependencies[0]),
                    plan.context,
                    target_max_block_size_override=effective_target_size,
                )
                # Execute the composite while retaining the original logical maps
                # for plan lineage and reporting.
                fused_physical_op.set_logical_operators(
                    *reversed(logical_chain_downstream_first)
                )
                for stage_op in fusible_chain:
                    op_map.pop(stage_op)
                op_map[fused_physical_op] = fused_logical_op
                rewritten_operators[op] = fused_physical_op
                did_fuse = True
                return fused_physical_op

            rewritten_inputs = [
                rewrite_operator(child) for child in op.input_dependencies
            ]
            if any(
                rewritten is not original
                for rewritten, original in zip(
                    rewritten_inputs,
                    op.input_dependencies,
                )
            ):
                # This operator was not fused, but one of its inputs was. Reconnect
                # the retained operator now; reverse output links are rebuilt once
                # after the complete traversal.
                op._input_dependencies = rewritten_inputs
            rewritten_operators[op] = op
            return op

        new_dag = rewrite_operator(plan.dag)
        if not did_fuse:
            return plan
        self._rebuild_output_dependencies(new_dag)
        return PhysicalPlan(new_dag, op_map, plan.context)

    @staticmethod
    def _context_allows_fusion(context: Any) -> bool:
        """Whether context-wide settings preserve this rule's runtime semantics."""

        if getattr(context, "enable_cudf_actor_fusion", False) is not True:
            return False
        # Adding the stage name wraps the original exception. Original-exception
        # propagation and exception-class retry matching would therefore change.
        if context.raise_original_map_exception is not False:
            return False
        # Boolean retry policies still mean retry-all or retry-none after wrapping;
        # policies that match particular exception classes do not.
        if (
            type(context.actor_task_retry_on_errors) is not bool
            or type(context.retried_map_errors) is not bool
        ):
            return False
        # The current fusion contract uses the actor pool's default tasks-in-flight
        # behavior, without a context-wide override.
        return context.max_tasks_in_flight_per_actor is None

    @classmethod
    def _collect_fusible_run(
        cls,
        op: PhysicalOperator,
        op_map: Mapping[PhysicalOperator, LogicalOperator],
        logical_consumer_counts: Mapping[LogicalOperator, int],
        physical_consumer_counts: Mapping[PhysicalOperator, int],
    ) -> Tuple[Tuple[PhysicalOperator, ...], Optional[int]]:
        """Collect the longest candidate chain ending at ``op``.

        The walk moves toward the source, so returned operators are ordered
        downstream to upstream. The second return value is the single output block
        size override that the fused operator should inherit. Two different
        non-``None`` overrides end the chain. An ineligible ``op`` produces a
        singleton, which the caller leaves unfused.
        """

        chain_settings = cls._get_fusion_settings(op, op_map.get(op))
        chain = [op]
        upstream_cursor = op
        effective_target_size = op.target_max_block_size_override
        if chain_settings is None:
            return tuple(chain), effective_target_size

        while len(upstream_cursor.input_dependencies) == 1:
            upstream = upstream_cursor.input_dependencies[0]
            upstream_logical_op = op_map.get(upstream)
            upstream_settings = cls._get_fusion_settings(upstream, upstream_logical_op)
            upstream_target_size = upstream.target_max_block_size_override
            upstream_is_shared = (
                logical_consumer_counts.get(upstream_logical_op, 0) != 1
                or physical_consumer_counts.get(upstream, 0) != 1
            )
            block_sizes_conflict = (
                effective_target_size is not None
                and upstream_target_size is not None
                and effective_target_size != upstream_target_size
            )
            # Any of these conditions is a boundary. In particular, absorbing a
            # shared upstream map could remove work still needed by another branch.
            if (
                upstream_is_shared
                or upstream_settings is None
                or not chain_settings.compatible_with(upstream_settings)
                or block_sizes_conflict
            ):
                break
            chain.append(upstream)
            upstream_cursor = upstream
            if effective_target_size is None:
                effective_target_size = upstream_target_size
        return tuple(chain), effective_target_size

    @classmethod
    def _create_fused_logical_op(
        cls,
        logical_chain_downstream_first: Tuple[MapBatches, ...],
    ) -> MapBatches:
        """Build a temporary logical ``MapBatches`` for a downstream-first chain."""

        # The traversal found stages backward, so restore execution order.
        stage_definitions = tuple(
            cls._stage_definition_from_op(stage_op, stage_index)
            for stage_index, stage_op in enumerate(
                reversed(logical_chain_downstream_first), start=1
            )
        )
        ingress_logical_op = logical_chain_downstream_first[-1]
        # Compatibility was already established for the full chain. The first map
        # therefore supplies the shared actor settings and the one ingress batching
        # policy. Its former input bypasses every map that the composite replaces.
        fused_logical_op = MapBatches(
            _FusedCudfMapBatches,
            input_dependencies=[ingress_logical_op.input_dependencies[0]],
            can_modify_num_rows=any(
                stage_op.can_modify_num_rows
                for stage_op in logical_chain_downstream_first
            ),
            batch_size=ingress_logical_op.batch_size,
            batch_format="cudf",
            zero_copy_batch=True,
            fn_constructor_args=(stage_definitions,),
            min_rows_per_bundled_input=ingress_logical_op.min_rows_per_bundled_input,
            compute=ingress_logical_op.compute,
            ray_remote_args=dict(ingress_logical_op.ray_remote_args),
        )
        cls._set_fused_name(fused_logical_op, stage_definitions)
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
    def _get_fusion_settings(
        cls,
        physical_op: PhysicalOperator,
        logical_op: Optional[LogicalOperator],
    ) -> Optional[_CudfMapFusionSettings]:
        """Return compatibility settings, or ``None`` for a fusion boundary.

        Eligibility is deliberately allowlisted. If the rule cannot prove that an
        option keeps the same behavior after fusion, the existing Ray operator is
        left unchanged without warning.
        """

        # Start with a normal actor-pool map and its MapBatches lineage. Rejecting
        # the private composite makes repeated optimizer passes idempotent.
        if (
            not isinstance(physical_op, ActorPoolMapOperator)
            or not physical_op.supports_fusion()
            or not isinstance(logical_op, MapBatches)
            or logical_op.fn is _FusedCudfMapBatches
        ):
            return None
        # The replacement physical operator has one BatchMapTransformFn, so it needs
        # one explicit cuDF ingress batch size. Requiring zero_copy_batch=True avoids
        # silently removing a defensive copy from each original boundary. Per-block
        # limits and dynamically generated actor options cannot be represented by
        # one operator either.
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

        # Inspect the class definition without constructing it or invoking a user
        # descriptor. Reject __call__ implementations declared async or as
        # generators; the runtime cuDF check catches other unsupported return types.
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

        actor_pool = cls._supported_actor_pool(logical_op.compute)
        remote_args = logical_op.ray_remote_args
        # Unknown remote options opt out. Supported options are compared exactly
        # against the chain's settings through
        # _CudfMapFusionSettings.compatible_with().
        if (
            actor_pool is None
            or type(remote_args) is not dict
            or not remote_args.keys() <= _SUPPORTED_REMOTE_ARGS
        ):
            return None
        max_concurrency = remote_args.get("max_concurrency", 1)
        # One actor task at a time preserves single-threaded access to every stage's
        # UDF instance. One full GPU per actor is the current scope.
        if type(max_concurrency) is not int or max_concurrency != 1:
            return None
        num_gpus = remote_args.get("num_gpus")
        if type(num_gpus) not in (int, float) or num_gpus != 1:
            return None

        return _CudfMapFusionSettings(
            actor_pool=actor_pool,
            batch_size=logical_op.batch_size,
            min_rows_per_bundle=logical_op.min_rows_per_bundled_input,
            ray_remote_args=remote_args,
        )

    @staticmethod
    def _supported_actor_pool(compute: Any) -> Optional[ActorPoolStrategy]:
        """Return a supported fixed or autoscaling, single-threaded actor pool."""

        if (
            type(compute) is not ActorPoolStrategy
            or compute.enable_true_multi_threading is not False
            or compute.max_tasks_in_flight_per_actor is not None
        ):
            return None
        return compute

    @staticmethod
    def _set_fused_name(
        op: MapBatches,
        stage_definitions: Tuple[_CudfMapStageDefinition, ...],
    ) -> None:
        names = "->".join(stage.udf_class.__name__ for stage in stage_definitions)
        # MapBatches is frozen; _name affects display only.
        object.__setattr__(op, "_name", f"MapBatches({names})")

    @staticmethod
    def _stage_definition_from_op(
        op: MapBatches, stage_index: int
    ) -> _CudfMapStageDefinition:
        # Keep the user's argument containers untouched. The fused actor unpacks
        # them only when it constructs or calls the stage. The numbered label lets
        # constructor and runtime errors identify the exact stage.
        return _CudfMapStageDefinition(
            udf_class=op.fn,
            error_label=f"{stage_index}: {op.name}",
            constructor_args=(
                () if op.fn_constructor_args is None else op.fn_constructor_args
            ),
            constructor_kwargs=(
                {} if op.fn_constructor_kwargs is None else op.fn_constructor_kwargs
            ),
            call_args=() if op.fn_args is None else op.fn_args,
            call_kwargs={} if op.fn_kwargs is None else op.fn_kwargs,
        )

    @staticmethod
    def _rebuild_output_dependencies(root: PhysicalOperator) -> None:
        # Physical operators store each edge twice: as an input on the consumer and
        # as an output on the producer. Recursive rewriting updates the authoritative
        # input edges first, so rebuild reverse edges for the rewritten DAG at the
        # end.
        operators = set(root.post_order_iter())
        for op in operators:
            op._output_dependencies = []
        for op in operators:
            for input_op in op.input_dependencies:
                input_op._output_dependencies.append(op)
