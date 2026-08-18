"""Physical Ray Data operator for the spillable distributed GPU sort.

The driver keeps Plasma references until all input is known::

    input refs -> deterministic sample -> ordered range boundaries
               -> bounded GPU shuffle waves -> ordered output partitions

Input assignment prefers the actor on the same Plasma node. A logical block
ordinal, rather than actor placement, seeds sampling so retries and locality
changes produce the same boundaries.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import ray
from ray.actor import ActorHandle
from ray.data import ExecutionOptions
from ray.data._internal.execution.bundle_queue import ReorderingBundleQueue
from ray.data._internal.execution.interfaces import (
    ExecutionResources,
    PhysicalOperator,
    RefBundle,
)
from ray.data._internal.execution.interfaces.physical_operator import DataOpTask, OpTask
from ray.data._internal.execution.operators.hash_shuffle import (
    _get_total_cluster_resources,
)
from ray.data._internal.execution.operators.sub_progress import SubProgressBarMixin
from ray.data._internal.gpu_sort.config import GPUSortConfig
from ray.data.block import BlockStats, to_stats
from ray.data.context import DataContext

if TYPE_CHECKING:
    from ray.data._internal.execution.block_ref_counter import BlockRefCounter

logger = logging.getLogger(__name__)


def _validate_gpu_schema(schema: Any, key_columns: Sequence[str]) -> None:
    """Validate the supported Arrow schema before starting GPU work."""

    import pyarrow as pa

    from ray.data._internal.gpu_sort.backend import _validate_payload_schema

    schema = getattr(schema, "base_schema", schema)
    if not isinstance(schema, pa.Schema):
        raise NotImplementedError("GPU sort currently requires Arrow-backed blocks.")
    missing = [name for name in key_columns if name not in schema.names]
    if missing:
        raise ValueError(
            f"GPU sort keys {missing} are absent from schema columns {schema.names}."
        )
    _validate_payload_schema(schema)

    key_types = [schema.field(name).type for name in key_columns]
    bad_keys = [
        name
        for name, typ in zip(key_columns, key_types)
        if not (
            pa.types.is_boolean(typ)
            or pa.types.is_string(typ)
            or pa.types.is_large_string(typ)
            or pa.types.is_integer(typ)
            or pa.types.is_floating(typ)
            or pa.types.is_date(typ)
            or pa.types.is_timestamp(typ)
        )
    ]
    if bad_keys:
        raise NotImplementedError(
            "GPU sort keys must be boolean, string, integer, float, date, or "
            f"timestamp columns; unsupported keys: {bad_keys}."
        )


@dataclass(frozen=True)
class _InputBlock:
    value: Any
    size_bytes: int
    num_rows: int
    ordinal: int = -1


def _allocate_stratified_sample_quotas(
    blocks: Sequence[_InputBlock], configured_sample_rows: int
) -> Tuple[List[int], int]:
    """Allocate the exact global sample budget across logical input blocks.

    Every nonempty block receives one row. The remaining budget is assigned in
    proportion to each block's remaining rows, with largest remainders and then
    logical block ordinals breaking ties.
    """

    configured_sample_rows = int(configured_sample_rows)
    if configured_sample_rows < 1:
        raise ValueError("GPU sort sample size must be positive.")
    nonempty = [block for block in blocks if int(block.num_rows) > 0]
    ordinals = [int(block.ordinal) for block in blocks]
    if any(ordinal < 0 for ordinal in ordinals) or len(set(ordinals)) != len(ordinals):
        raise ValueError("GPU sort requires unique nonnegative logical block ordinals.")
    total_rows = sum(int(block.num_rows) for block in nonempty)
    if total_rows == 0:
        return [0] * len(blocks), 0

    target = min(total_rows, max(configured_sample_rows, len(nonempty)))
    quotas = {int(block.ordinal): 1 for block in nonempty}
    remaining = target - len(nonempty)
    capacities = {int(block.ordinal): int(block.num_rows) - 1 for block in nonempty}
    total_capacity = sum(capacities.values())
    if remaining and total_capacity:
        remainders = []
        assigned = 0
        for block in nonempty:
            ordinal = int(block.ordinal)
            numerator = remaining * capacities[ordinal]
            extra, remainder = divmod(numerator, total_capacity)
            quotas[ordinal] += extra
            assigned += extra
            remainders.append((remainder, ordinal))
        leftover = remaining - assigned
        for _, ordinal in sorted(remainders, key=lambda item: (-item[0], item[1]))[
            :leftover
        ]:
            quotas[ordinal] += 1

    result = [quotas.get(int(block.ordinal), 0) for block in blocks]
    if sum(result) != target or any(
        quota < 0 or quota > int(block.num_rows) for block, quota in zip(blocks, result)
    ):
        raise RuntimeError("GPU sort could not allocate its exact sample budget.")
    return result, target


def _underlying_object_ref(block: _InputBlock) -> Any:
    value = block.value
    return value[0] if isinstance(value, tuple) and len(value) == 3 else value


def _assign_blocks_by_locality(
    blocks: Sequence[_InputBlock],
    actor_node_ids: Sequence[str],
    object_locations: Mapping[Any, Mapping[str, Any]],
) -> Tuple[
    List[List[_InputBlock]],
    List[int],
    List[int],
    List[int],
    List[int],
]:
    """Assign blocks locally, with deterministic decoded-byte balancing.

    An object may have multiple replicas, no reported location (for example an
    inline object), or a location on a node without a GPU-sort actor.  The same
    size-first rule is used within the local candidates and as the global
    fallback, so location lookup is never a correctness dependency.
    """

    nranks = len(actor_node_ids)
    if nranks < 1:
        raise ValueError("GPU sort requires at least one actor for block assignment.")
    blocks_by_rank: List[List[_InputBlock]] = [[] for _ in range(nranks)]
    assigned_bytes = [0] * nranks
    assigned_blocks = [0] * nranks
    local_bytes = [0] * nranks
    local_blocks = [0] * nranks

    for block in blocks:
        ref = _underlying_object_ref(block)
        try:
            location = object_locations.get(ref, {})
        except TypeError:
            location = {}
        node_ids = set(location.get("node_ids", ()) or ())
        candidates = [
            rank
            for rank, node_id in enumerate(actor_node_ids)
            if node_id and node_id in node_ids
        ]
        is_local = bool(candidates)
        if not candidates:
            candidates = list(range(nranks))
        rank = min(
            candidates,
            key=lambda item: (
                assigned_bytes[item],
                assigned_blocks[item],
                item,
            ),
        )
        blocks_by_rank[rank].append(block)
        assigned_bytes[rank] += int(block.size_bytes)
        assigned_blocks[rank] += 1
        if is_local:
            local_bytes[rank] += int(block.size_bytes)
            local_blocks[rank] += 1

    return (
        blocks_by_rank,
        assigned_bytes,
        assigned_blocks,
        local_bytes,
        local_blocks,
    )


def _make_waves(
    blocks_by_rank: Sequence[Sequence[_InputBlock]],
    target_bytes_per_rank: Optional[int],
) -> List[List[List[Any]]]:
    """Create the same number of deterministic, bounded waves for every rank."""

    rank_waves: List[List[List[Any]]] = []
    for blocks in blocks_by_rank:
        if not blocks:
            rank_waves.append([])
            continue
        if target_bytes_per_rank is None:
            rank_waves.append([[block.value for block in blocks]])
            continue
        waves: List[List[Any]] = []
        current: List[Any] = []
        current_bytes = 0
        for block in blocks:
            if current and current_bytes + block.size_bytes > target_bytes_per_rank:
                waves.append(current)
                current = []
                current_bytes = 0
            current.append(block.value)
            current_bytes += block.size_bytes
        if current:
            waves.append(current)
        rank_waves.append(waves)

    count = max((len(waves) for waves in rank_waves), default=0)
    return [
        [waves[index] if index < len(waves) else [] for waves in rank_waves]
        for index in range(count)
    ]


def _wave_target_bytes(
    blocks_by_rank: Sequence[Sequence[_InputBlock]],
    *,
    explicit_residency_budget_bytes: Optional[int],
    actor_usable_budgets: Sequence[int],
    auto_wave_fraction: float,
) -> Optional[int]:
    """Return a bounded-wave target, or ``None`` for the resident fast path."""

    if explicit_residency_budget_bytes is not None:
        # Bound source waves as well as retained destination data when callers
        # choose a smaller residency budget.
        return max(256 << 20, int(explicit_residency_budget_bytes) // 2)

    budgets = [int(value) for value in actor_usable_budgets if int(value) > 0]
    if not budgets:
        raise RuntimeError("GPU sort actors did not report usable memory budgets.")
    target = max(256 << 20, int(min(budgets) * float(auto_wave_fraction)))
    largest_rank_input = max(
        (sum(block.size_bytes for block in blocks) for blocks in blocks_by_rank),
        default=0,
    )
    return None if largest_rank_input <= target else target


def _derive_num_ranks(data_context: DataContext) -> int:
    configured = data_context.gpu_shuffle_num_actors
    if configured is not None:
        if configured < 1:
            raise ValueError("gpu_shuffle_num_actors must be positive.")
        return int(configured)
    ranks = int(_get_total_cluster_resources().gpu or 0)
    if ranks < 1:
        raise RuntimeError(
            "GPU sort requires at least one Ray GPU resource. Set "
            "DataContext.gpu_shuffle_num_actors to override detection."
        )
    return ranks


def _operator_config(data_context: DataContext) -> Dict[str, Any]:
    sample_size = max(
        65_536, int(data_context.get_config("gpu_sort_sample_size", 65_536))
    )
    return GPUSortConfig(
        sample_size=sample_size,
        sample_seed=int(data_context.get_config("gpu_sort_sample_seed", 0)),
        residency_budget_bytes=data_context.get_config(
            "gpu_sort_memory_budget_bytes", None
        ),
        auto_wave_fraction=float(
            data_context.get_config("gpu_sort_auto_wave_fraction", 0.50)
        ),
        exchange_batch_bytes=int(
            data_context.get_config("gpu_sort_exchange_batch_bytes", 512 << 20)
        ),
        run_chunk_bytes=int(
            data_context.get_config("gpu_sort_run_chunk_bytes", 512 << 20)
        ),
        setup_timeout_s=float(
            data_context.get_config("gpu_sort_setup_timeout_s", 300.0)
        ),
    ).to_actor_dict()


class _RankPool:
    """Own one short-lived actor per GPU and order them by actual MPF rank.

    GPU sort needs the node and memory receipt returned during MPF setup for
    Plasma-local input assignment and bounded-wave admission. It also needs to
    reorder actors because MPF ranks can differ from Ray actor creation order.
    The existing generic GPU shuffle pool doesn't expose either result.
    """

    def __init__(
        self,
        nranks: int,
        key_columns: List[str],
        ascending: List[bool],
        config: Dict[str, Any],
        label_selector: Optional[Dict[str, str]],
    ) -> None:
        self.nranks = nranks
        self.key_columns = key_columns
        self.ascending = ascending
        self.config = config
        self.label_selector = label_selector
        self.actors: List[ActorHandle] = []
        self.rank_infos: List[Dict[str, Any]] = []
        self._shutdown_lock = threading.Lock()
        self._is_shutdown = False

    @property
    def is_shutdown(self) -> bool:
        return self._is_shutdown

    def start(self) -> None:
        from ray.data._internal.gpu_sort.actor import GPUSortActor

        if self._is_shutdown:
            raise RuntimeError("A stopped GPU sort rank pool cannot be restarted.")
        if self.actors:
            raise RuntimeError("GPU sort ranks have already been started.")
        actor_options: Dict[str, Any] = {
            "num_gpus": 1,
            "num_cpus": 1,
            "scheduling_strategy": "SPREAD",
        }
        if self.label_selector:
            actor_options["label_selector"] = self.label_selector
        self.actors = [
            GPUSortActor.options(**actor_options).remote(
                nranks=self.nranks,
                index=rank,
                key_columns=self.key_columns,
                ascending=self.ascending,
                num_partitions=self.nranks,
                config=self.config,
            )
            for rank in range(self.nranks)
        ]
        timeout = self.config["setup_timeout_s"]
        root_rank, address = ray.get(
            self.actors[0].setup_root.remote(), timeout=timeout
        )
        if int(root_rank) != 0:
            raise RuntimeError(
                f"GPU sort communicator assigned root rank {root_rank}, expected 0."
            )
        setup = ray.get(
            [actor.setup_worker.remote(address) for actor in self.actors],
            timeout=timeout,
        )
        actors_by_rank: List[Optional[ActorHandle]] = [None] * self.nranks
        infos_by_rank: List[Optional[Dict[str, Any]]] = [None] * self.nranks
        for actor, result in zip(self.actors, setup):
            rank = int(result["rank"])
            if not 0 <= rank < self.nranks or actors_by_rank[rank] is not None:
                raise RuntimeError(
                    f"GPU sort communicator returned invalid rank {rank}."
                )
            actors_by_rank[rank] = actor
            infos_by_rank[rank] = dict(result)
        if any(actor is None for actor in actors_by_rank):
            raise RuntimeError("GPU sort communicator rank assignment is incomplete.")
        self.actors = [actor for actor in actors_by_rank if actor is not None]
        self.rank_infos = [info for info in infos_by_rank if info is not None]

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._is_shutdown:
                return
            actors, self.actors = self.actors, []
            self._is_shutdown = True
        if not actors:
            return
        try:
            ray.get([actor.release.remote() for actor in actors], timeout=10)
        except Exception:
            logger.debug(
                "GPU sort actor release failed; terminating the actors.",
                exc_info=True,
            )
        finally:
            for actor in actors:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception:
                    logger.debug(
                        "GPU sort actor termination failed during cleanup.",
                        exc_info=True,
                    )

    def shutdown_async(self) -> None:
        """Release successful one-shot ranks after their outputs are sealed."""

        thread = threading.Thread(
            target=self.shutdown,
            name="gpu-sort-rank-reaper",
            daemon=True,
        )
        thread.start()


class GPUSortOperator(PhysicalOperator, SubProgressBarMixin):
    """Unified resident/spillable distributed GPU range sort."""

    def __init__(
        self,
        input_op: PhysicalOperator,
        data_context: DataContext,
        *,
        sort_key: Any,
    ) -> None:
        if sort_key.boundaries is not None:
            raise ValueError("GPU sort does not support explicit `boundaries`.")
        key_columns = list(sort_key.get_columns())
        if not key_columns:
            raise ValueError("GPU sort requires at least one sort key.")
        nranks = _derive_num_ranks(data_context)
        descending = list(sort_key.get_descending())
        config = _operator_config(data_context)
        super().__init__(
            name=f"GPUSort(keys={key_columns}, ranks={nranks})",
            input_dependencies=[input_op],
            data_context=data_context,
        )
        self._sort_key = sort_key
        self._key_columns = key_columns
        self._config = config
        self._rank_pool = _RankPool(
            nranks,
            key_columns,
            [not value for value in descending],
            config,
            data_context.execution_options.label_selector,
        )
        self._input_blocks: List[_InputBlock] = []
        self._blocks_by_rank: List[List[_InputBlock]] = [[] for _ in range(nranks)]
        self._input_bundles: List[RefBundle] = []
        self._input_rows = 0
        self._input_schema = None
        self._input_stats: List[BlockStats] = []
        self._output_stats: List[BlockStats] = []
        self._output_queue = ReorderingBundleQueue()
        self._extraction_tasks: Dict[int, DataOpTask] = {}
        self._finalization_started = False
        self._sample_quotas: List[int] = []
        self._sample_target_rows = 0
        self._progress = {"GPU Sort/Merge": None}

    def start(
        self,
        options: ExecutionOptions,
        block_ref_counter: "BlockRefCounter",
    ) -> None:
        if not options.preserve_order:
            raise RuntimeError(
                "GPU sort requires preserve_order=True so logical block "
                "ordinals remain reproducible."
            )
        super().start(options, block_ref_counter)

    def _add_input_inner(self, bundle: RefBundle, input_index: int) -> None:
        if input_index != 0:
            raise ValueError("GPU sort accepts exactly one input dependency.")
        self._input_bundles.append(bundle)
        self._metrics.on_input_queued(bundle, input_index=0)
        self._input_stats.extend(to_stats(bundle.metadata))
        if self._input_schema is None and bundle.schema is not None:
            self._input_schema = bundle.schema

        for entry, block_slice in zip(bundle.blocks, bundle.slices):
            block_ref = entry.ref
            metadata = entry.metadata
            if block_slice is None:
                if metadata.num_rows is None:
                    raise ValueError(
                        "GPU sort deterministic sampling requires exact block "
                        "row metadata."
                    )
                value: Any = block_ref
                rows = int(metadata.num_rows or 0)
                size_bytes = int(metadata.size_bytes or 0)
            else:
                value = (
                    block_ref,
                    int(block_slice.start_offset),
                    int(block_slice.end_offset),
                )
                rows = int(block_slice.num_rows)
                full_rows = int(metadata.num_rows or 0)
                size_bytes = (
                    max(1, math.ceil((metadata.size_bytes or 0) * rows / full_rows))
                    if full_rows
                    else int(metadata.size_bytes or 0)
                )
            # Actor placement and communicator rank order are not known until
            # MPF bootstrap, so assignment is deliberately deferred.
            self._input_blocks.append(
                _InputBlock(
                    value=value,
                    size_bytes=size_bytes,
                    num_rows=rows,
                    ordinal=len(self._input_blocks),
                )
            )
            self._input_rows += rows

    def _assign_input_blocks(self) -> None:
        actor_node_ids = [
            str(info.get("node_id", "")) for info in self._rank_pool.rank_infos
        ]
        refs = []
        seen = set()
        for block in self._input_blocks:
            ref = _underlying_object_ref(block)
            if isinstance(ref, ray.ObjectRef) and ref not in seen:
                seen.add(ref)
                refs.append(ref)
        try:
            locations = ray.experimental.get_object_locations(refs) if refs else {}
        except Exception:
            # The API is experimental and excludes some valid objects.  A
            # deterministic non-local assignment is always safe.
            logger.debug(
                "GPU sort couldn't read Plasma locations; using balanced "
                "non-local assignment.",
                exc_info=True,
            )
            locations = {}
        self._blocks_by_rank, _, _, local_bytes, _ = _assign_blocks_by_locality(
            self._input_blocks, actor_node_ids, locations
        )
        total_bytes = sum(block.size_bytes for block in self._input_blocks)
        logger.debug(
            "GPU sort assigned %.1f%% of input bytes to a local Plasma node.",
            100 * sum(local_bytes) / total_bytes if total_bytes else 100.0,
        )

    def _prepare_sampling_plan(self) -> None:
        self._sample_quotas, self._sample_target_rows = (
            _allocate_stratified_sample_quotas(
                self._input_blocks, int(self._config["sample_size"])
            )
        )

    def _sample(self) -> Tuple[Any, Any]:
        seed = self._config["sample_seed"]
        quotas_by_ordinal = {
            block.ordinal: quota
            for block, quota in zip(self._input_blocks, self._sample_quotas)
        }
        refs = [
            actor.sample_blocks.remote(
                [block.value for block in blocks],
                [block.ordinal for block in blocks],
                [quotas_by_ordinal[block.ordinal] for block in blocks],
                seed,
            )
            for actor, blocks in zip(self._rank_pool.actors, self._blocks_by_rank)
        ]
        manifests = ray.get(refs, timeout=self._config["setup_timeout_s"])
        schema = self._input_schema or next(
            (
                item.get("schema")
                for item in manifests
                if item.get("schema") is not None
            ),
            None,
        )
        if schema is None:
            raise ValueError("GPU sort could not determine the input Arrow schema.")
        _validate_gpu_schema(schema, self._key_columns)
        self._sort_key.validate_schema(getattr(schema, "base_schema", schema))
        samples = [
            item["sample"]
            for item in manifests
            if item.get("sample") is not None and item["sample"].num_rows
        ]
        result = ray.get(
            self._rank_pool.actors[0].compute_boundaries.remote(samples, schema),
            timeout=self._config["setup_timeout_s"],
        )
        sample_rows = int(result.get("sample_rows", 0) or 0)
        if sample_rows != self._sample_target_rows:
            raise RuntimeError(
                "GPU sort sampling did not produce its exact global target: "
                f"expected {self._sample_target_rows}, got {sample_rows}."
            )
        logger.debug(
            "GPU sort selected %s planning rows from %s nonempty blocks.",
            sample_rows,
            sum(quota > 0 for quota in self._sample_quotas),
        )
        return schema, result["boundaries"]

    def _plan_waves(self) -> List[List[List[Any]]]:
        budget = self._config["residency_budget_bytes"]
        target_bytes = _wave_target_bytes(
            self._blocks_by_rank,
            explicit_residency_budget_bytes=budget,
            actor_usable_budgets=[
                int(info.get("usable_memory_budget_bytes", 0) or 0)
                for info in self._rank_pool.rank_infos
            ],
            auto_wave_fraction=float(self._config["auto_wave_fraction"]),
        )
        waves = _make_waves(self._blocks_by_rank, target_bytes)
        logger.debug(
            "GPU sort planned %s shuffle waves with a per-rank target of %s bytes.",
            len(waves),
            target_bytes,
        )
        return waves

    def _release_inputs(self) -> None:
        """Release input references after every rank has consumed its last wave."""

        for bundle in self._input_bundles:
            self._metrics.on_input_dequeued(bundle, input_index=0)
            bundle.destroy_if_owned()
        self._input_bundles.clear()
        self._input_blocks.clear()
        for blocks in self._blocks_by_rank:
            blocks.clear()

    def _try_finalize(self) -> None:
        if self._finalization_started or not self._inputs_complete:
            return
        self._finalization_started = True
        if not self._input_blocks:
            self._release_inputs()
            return

        try:
            self._rank_pool.start()
            self._prepare_sampling_plan()
            self._assign_input_blocks()
            schema, boundaries = self._sample()
            ray.get(
                [
                    actor.install_plan.remote(schema, boundaries)
                    for actor in self._rank_pool.actors
                ],
                timeout=self._config["setup_timeout_s"],
            )
            waves = self._plan_waves()
            for wave_id, blocks_for_ranks in enumerate(waves):
                ray.get(
                    [
                        actor.process_wave.remote(wave_id, blocks)
                        for actor, blocks in zip(
                            self._rank_pool.actors, blocks_for_ranks
                        )
                    ]
                )
            self._release_inputs()
            self._schedule_extraction()
        except Exception:
            self._rank_pool.shutdown()
            self._release_inputs()
            raise

    def _schedule_extraction(self) -> None:
        for rank, actor in enumerate(self._rank_pool.actors):

            def _on_bundle_ready(bundle: RefBundle, rank: int = rank) -> None:
                self._output_queue.add(bundle, key=rank)
                self._metrics.on_output_queued(bundle)
                self._metrics.on_task_output_generated(rank, bundle)
                progress = self._progress["GPU Sort/Merge"]
                if progress is not None:
                    progress.update(
                        increment=bundle.num_rows() or 0, total=self._input_rows
                    )

            def _on_done(
                exc: Optional[Exception],
                worker_stats=None,
                driver_stats=None,
                rank: int = rank,
            ) -> None:
                self._extraction_tasks.pop(rank, None)
                self._output_queue.finalize(key=rank)
                self._metrics.on_task_finished(
                    task_index=rank,
                    exception=exc,
                    task_exec_stats=worker_stats,
                    task_exec_driver_stats=driver_stats,
                )
                if exc is not None:
                    self._rank_pool.shutdown()
                    return
                if not self._extraction_tasks:
                    self._rank_pool.shutdown_async()

            generator = actor.finish_and_extract.options(
                num_returns="streaming"
            ).remote()
            task = DataOpTask(
                task_index=rank,
                streaming_gen=generator,
                block_ref_counter=self._block_ref_counter,
                producer_id=self.id,
                output_ready_callback=_on_bundle_ready,
                task_done_callback=_on_done,
                operator_name=self.name,
            )
            self._extraction_tasks[rank] = task
            self._metrics.on_task_submitted(
                rank, RefBundle([], schema=None, owns_blocks=False), task.get_task_id()
            )

    def has_next(self) -> bool:
        self._try_finalize()
        return self._output_queue.has_next()

    def _get_next_inner(self) -> RefBundle:
        bundle = self._output_queue.get_next()
        self._metrics.on_output_dequeued(bundle)
        self._output_stats.extend(to_stats(bundle.metadata))
        return bundle

    def get_active_tasks(self) -> List[OpTask]:
        self._try_finalize()
        return list(self._extraction_tasks.values())

    def has_completed(self) -> bool:
        return (
            self._finalization_started
            and not self._extraction_tasks
            and super().has_completed()
        )

    def _do_shutdown(self, force: bool = False) -> None:
        self._rank_pool.shutdown()
        super()._do_shutdown(force)
        self._release_inputs()
        self._extraction_tasks.clear()
        self._output_queue.clear()

    def current_logical_usage(self) -> ExecutionResources:
        if self._rank_pool.is_shutdown:
            return ExecutionResources.zero()
        return ExecutionResources(
            cpu=(len(self._rank_pool.actors) or self._rank_pool.nranks),
            gpu=(len(self._rank_pool.actors) or self._rank_pool.nranks),
        )

    @property
    def base_resource_usage(self) -> ExecutionResources:
        return ExecutionResources(
            cpu=self._rank_pool.nranks, gpu=self._rank_pool.nranks
        )

    def incremental_resource_usage(self) -> ExecutionResources:
        return ExecutionResources(cpu=1, gpu=1)

    def get_sub_progress_bar_names(self) -> List[str]:
        return list(self._progress)

    def set_sub_progress_bar(self, name: str, pg: Any) -> None:
        if name in self._progress:
            self._progress[name] = pg

    def get_stats(self) -> Dict[str, List[BlockStats]]:
        return {
            f"{self.name}_input": self._input_stats,
            f"{self.name}_output": self._output_stats,
        }
