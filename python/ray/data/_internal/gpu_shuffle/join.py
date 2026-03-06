"""GPU-native join operator using RAPIDS MPF (rapidsmpf) + UCXX.

Two-phase hash-join that routes both left and right datasets through GPU actors:

  Phase 1 — right shuffle:
      Each right block is hash-partitioned by ``right_key_columns`` and sent
      to its assigned actor rank via UCXX.  When all right blocks are
      inserted, each actor extracts its portion of the right data and stores
      it in GPU memory (``self._stored_right_df``).

  Phase 2 — left shuffle:
      Each left block is hash-partitioned by ``left_key_columns``.  Because
      both sides are partitioned by the same key hash space, a left row with
      key K lands on the same rank that holds the right rows with key K.

  Phase 3 — join:
      Each actor calls ``cudf.merge`` between its extracted left partitions
      and its stored right ``cudf.DataFrame``.  Results are streamed back as
      Arrow Tables via the Ray Data streaming generator protocol.

Dependencies (all optional at import time — deferred to actor construction):
    rapidsmpf, cudf, ucxx
"""

from __future__ import annotations

import logging
from collections import deque
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import pyarrow as pa

import ray
from ray.actor import ActorHandle
from ray.data._internal.execution.interfaces import (
    ExecutionResources,
    PhysicalOperator,
    RefBundle,
)
from ray.data._internal.execution.interfaces.physical_operator import (
    DataOpTask,
    MetadataOpTask,
    OpTask,
)
from ray.data._internal.execution.operators.hash_shuffle import (
    _get_total_cluster_resources,
)
from ray.data._internal.execution.operators.sub_progress import SubProgressBarMixin
from ray.data._internal.logical.operators import JoinType
from ray.data._internal.stats import OpRuntimeMetrics
from ray.data.block import BlockAccessor, BlockMetadataWithSchema, BlockStats, to_stats
from ray.data.context import DataContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Join type mapping: Ray Data JoinType → cuDF merge ``how`` string
# ---------------------------------------------------------------------------

_CUDF_JOIN_MAP: Dict[JoinType, str] = {
    JoinType.INNER: "inner",
    JoinType.LEFT_OUTER: "left",
    JoinType.RIGHT_OUTER: "right",
    JoinType.FULL_OUTER: "outer",
    JoinType.LEFT_SEMI: "leftsemi",
    JoinType.RIGHT_SEMI: "rightsemi",
    JoinType.LEFT_ANTI: "leftanti",
    JoinType.RIGHT_ANTI: "rightanti",
}


def _normalize_schema_for_cudf(schema: pa.Schema) -> pa.Schema:
    """Replace utf8/string fields with large_utf8/large_string to match cuDF output.

    cuDF's ``to_arrow()`` always emits string columns as ``large_string``
    (64-bit offsets).  If the captured schema contains ``string`` (32-bit
    offsets), the subsequent ``Table.cast()`` will fail with
    "input array too large" even when the data is small.
    """
    fields = []
    for field in schema:
        if pa.types.is_string(field.type):
            fields.append(field.with_type(pa.large_utf8()))
        else:
            fields.append(field)
    return pa.schema(fields)


# ---------------------------------------------------------------------------
# GPU join actor
# ---------------------------------------------------------------------------


@ray.remote(num_gpus=1)
class GPUJoinActor:
    """One GPU rank in a two-phase RAPIDS MPF join.

    Wraps ``BulkRapidsMPFJoinShuffler`` by composition so that importing this
    module does not require ``rapidsmpf`` at module load time.

    Lifecycle::

        setup_root()                  # rank 0 only
        setup_worker(root_address)    # all ranks
        [insert_right_batch(...)]* → right_insert_finished()
        [insert_left_batch(...)]*  → left_insert_finished()
        execute_join()                # streaming generator
    """

    def __init__(
        self,
        nranks: int,
        total_nparts: int,
        left_key_columns: List[str],
        right_key_columns: List[str],
        join_type: JoinType,
        rmm_pool_size: Union[int, str, None] = None,
        spill_memory_limit: Union[int, str, None] = "auto",
    ):
        from ray.data._internal.gpu_shuffle.rapidsmpf_backend import (
            BulkRapidsMPFJoinShuffler,
        )

        # Initialise shuffler with RIGHT keys for Phase 1.
        self._shuffler = BulkRapidsMPFJoinShuffler(
            nranks=nranks,
            total_nparts=total_nparts,
            shuffle_on=right_key_columns,
            rmm_pool_size=rmm_pool_size,
            spill_memory_limit=spill_memory_limit,
        )
        self._left_key_columns: List[str] = list(left_key_columns)
        self._right_key_columns: List[str] = list(right_key_columns)
        self._join_type: JoinType = join_type
        self._stored_right_df = None  # cudf.DataFrame after Phase 1 extraction
        self._right_schema: Optional[pa.Schema] = None
        self._left_schema: Optional[pa.Schema] = None

    # ------------------------------------------------------------------
    # UCXX communicator setup
    # ------------------------------------------------------------------

    def setup_root(self) -> "tuple[int, bytes]":
        """Initialise root communicator; return ``(rank, root_address_bytes)``."""
        return self._shuffler.setup_root()

    def setup_worker(self, root_address: bytes) -> None:
        """Complete UCXX setup and create the Phase-1 shuffler."""
        self._shuffler.setup_worker(root_address)

    # ------------------------------------------------------------------
    # Column-schema helpers (needed for ranks that receive no batches)
    # ------------------------------------------------------------------

    def set_right_schema(self, schema: "pa.Schema") -> None:
        """Receive the right-side Arrow schema from the operator.

        Called for every rank before ``right_insert_finished`` so that ranks
        which received no right batches can still create a properly-typed empty
        DataFrame instead of a null-typed one.
        """
        if self._right_schema is None:
            self._right_schema = schema

    def set_left_schema(self, schema: "pa.Schema") -> None:
        """Receive the left-side Arrow schema from the operator.

        Called for every rank before ``left_insert_finished`` so that ranks
        which received no left batches can still create a properly-typed empty
        DataFrame instead of a null-typed one.
        """
        if self._left_schema is None:
            self._left_schema = schema

    # ------------------------------------------------------------------
    # Phase 1 — right dataset shuffle
    # ------------------------------------------------------------------

    def insert_right_batch(self, batch) -> int:
        """Hash-partition a right batch and route shards to peer ranks."""
        import cudf
        import pyarrow as pa

        if isinstance(batch, pa.Table):
            df = cudf.DataFrame.from_arrow(batch)
        else:
            df = batch

        if self._right_schema is None:
            self._right_schema = BlockAccessor.for_block(batch).to_arrow().schema

        self._shuffler.insert_chunk(table=df, column_names=self._right_schema.names)
        return len(df)

    def right_insert_finished(self) -> None:
        """Finalise Phase 1: extract right partitions, store, and reset for Phase 2."""
        import cudf
        from rapidsmpf.utils.cudf import pylibcudf_to_cudf_dataframe

        self._shuffler.insert_finished()

        right_columns = self._right_schema.names if self._right_schema else []

        right_parts = []
        for _, partition in self._shuffler.extract():
            cdf = pylibcudf_to_cudf_dataframe(partition, column_names=right_columns)
            right_parts.append(cdf)

        if right_parts:
            self._stored_right_df = cudf.concat(right_parts, ignore_index=True)
            right_parts.clear()  # free individual partition DFs now that concat is done
        else:
            # Build a properly-typed empty DataFrame so that merges on this
            # rank yield correct Arrow column types instead of null-typed ones.
            empty_table = self._right_schema.empty_table()
            self._stored_right_df = cudf.DataFrame.from_arrow(empty_table)

        # Transition to Phase 2 — updates shuffle_on + creates new Shuffler.
        self._shuffler.reset_for_left_shuffle(self._left_key_columns)

    # ------------------------------------------------------------------
    # Phase 2 — left dataset shuffle
    # ------------------------------------------------------------------

    def insert_left_batch(self, batch) -> int:
        """Hash-partition a left batch and route shards to peer ranks."""
        import cudf
        import pyarrow as pa

        if isinstance(batch, pa.Table):
            df = cudf.DataFrame.from_arrow(batch)
        else:
            df = batch

        if self._left_schema is None:
            self._left_schema = BlockAccessor.for_block(batch).to_arrow().schema

        self._shuffler.insert_chunk(table=df, column_names=self._left_schema.names)
        return len(df)

    def left_insert_finished(self) -> None:
        """Signal no more left batches will be inserted."""
        self._shuffler.insert_finished()

    # ------------------------------------------------------------------
    # Phase 3 — join
    # ------------------------------------------------------------------

    def execute_join(self) -> Iterator:
        """Extract left partitions and merge with stored right data.

        Streaming generator: yields alternating ``pa.Table`` /
        ``BlockMetadataWithSchema`` pairs following the Ray Data protocol.

        For FULL_OUTER and RIGHT_OUTER we must not emit right-only rows per
        partition: if an empty left partition is merged first we would emit all
        of stored_right as right-only, then a later non-empty partition would
        emit matched rows, duplicating those right rows. So we only yield
        matched + left-only rows in the loop and emit right-only rows after
        processing all left partitions (minus matched keys).
        """
        import cudf

        from rapidsmpf.utils.cudf import pylibcudf_to_cudf_dataframe

        from ray.data.block import BlockExecStats, BlockMetadataWithSchema

        how = _CUDF_JOIN_MAP[self._join_type]

        left_columns = self._left_schema.names if self._left_schema is not None else []
        left_value_columns = [c for c in left_columns if c not in self._left_key_columns]

        # construct merged schema from left and right schemas
        merged_schema = pa.unify_schemas([self._left_schema, self._right_schema])

        deferred_right_only = self._join_type in (
            JoinType.FULL_OUTER,
            JoinType.RIGHT_OUTER,
        )
        # Set of cudf.DataFrames for matched keys and right-only rows
        matched_keys_cdfs = []
        right_only_cdfs = []

        for _, left_partition in self._shuffler.extract():
            exec_stats_builder = BlockExecStats.builder()
            left_cdf = pylibcudf_to_cudf_dataframe(
                left_partition, column_names=left_columns
            )

            result_cdf = left_cdf.merge(
                self._stored_right_df,
                left_on=self._left_key_columns,
                right_on=self._right_key_columns,
                how=how,
            )
            del left_cdf  # free left partition GPU memory immediately after merge
            if len(result_cdf) == 0:
                continue

            if deferred_right_only:
                right_only_mask = result_cdf[left_value_columns].isna().all(axis=1)
                matched_mask = ~right_only_mask
                if right_only_mask.any():
                    # save right-only rows for later
                    right_only_cdf = result_cdf[right_only_mask]
                    right_only_cdfs.append(right_only_cdf)
                if not matched_mask.any():
                    # skip if no matched rows
                    continue

                matched_cdf = result_cdf[matched_mask]
                matched_keys = matched_cdf[self._left_key_columns].drop_duplicates()
                matched_keys_cdfs.append(matched_keys)
                result_cdf = matched_cdf
                matched_cdf = None

            # explicitly cast the result to the merged schema to avoid null type conversion errors
            final_schema = _normalize_schema_for_cudf(
                pa.schema([field for field in merged_schema if field.name in result_cdf.columns])
            )
            block = result_cdf.to_arrow(preserve_index=False).cast(final_schema)
            del result_cdf  # free GPU memory; data is now in Arrow (CPU/object store)
            exec_stats = exec_stats_builder.build()
            stats = yield block
            if stats:
                exec_stats.block_ser_time_s = stats.object_creation_dur_s
            yield BlockMetadataWithSchema.from_block(block, stats=exec_stats)

        # Emit right-only rows whose key never matched after all left partitions have been processed
        if deferred_right_only and len(self._stored_right_df) > 0:
            matched_keys_cdf = cudf.concat(matched_keys_cdfs).drop_duplicates()
            matched_keys_cdfs.clear()
            right_only_cdf = cudf.concat(right_only_cdfs).drop_duplicates()
            right_only_cdfs.clear()
            right_only_cdf = right_only_cdf.merge(
                matched_keys_cdf, on=self._right_key_columns, how="leftanti"
            )
            del matched_keys_cdf

            if len(right_only_cdf) > 0:
                block = right_only_cdf.to_arrow(preserve_index=False).cast(merged_schema)
                del right_only_cdf  # free GPU memory after Arrow conversion
                exec_stats = BlockExecStats.builder().build()
                stats = yield block
                if stats:
                    exec_stats.block_ser_time_s = stats.object_creation_dur_s
                yield BlockMetadataWithSchema.from_block(block, stats=exec_stats)

        # All joins complete — release right dataset GPU memory.
        del self._stored_right_df


# ---------------------------------------------------------------------------
# GPUJoinRankPool — lifecycle manager
# ---------------------------------------------------------------------------


class GPUJoinRankPool:
    """Manages the lifecycle of ``GPUJoinActor`` instances."""

    def __init__(
        self,
        nranks: int,
        total_nparts: int,
        left_key_columns: List[str],
        right_key_columns: List[str],
        join_type: JoinType,
        rmm_pool_size: Union[int, str, None],
        spill_memory_limit: Union[int, str, None],
    ):
        self._nranks = nranks
        self._total_nparts = total_nparts
        self._left_key_columns = left_key_columns
        self._right_key_columns = right_key_columns
        self._join_type = join_type
        self._rmm_pool_size = rmm_pool_size
        self._spill_memory_limit = spill_memory_limit
        self._actors: List[ActorHandle] = []

    @property
    def nranks(self) -> int:
        return self._nranks

    @property
    def actors(self) -> List[ActorHandle]:
        return self._actors

    def start(self) -> None:
        """Create actors and coordinate UCXX setup (blocks until complete)."""
        self._actors = [
            GPUJoinActor.options(
                num_gpus=1,
                scheduling_strategy="SPREAD",
            ).remote(
                nranks=self._nranks,
                total_nparts=self._total_nparts,
                left_key_columns=self._left_key_columns,
                right_key_columns=self._right_key_columns,
                join_type=self._join_type,
                rmm_pool_size=self._rmm_pool_size,
                spill_memory_limit=self._spill_memory_limit,
            )
            for _ in range(self._nranks)
        ]

        _, root_address_bytes = ray.get(self._actors[0].setup_root.remote())
        ray.get(
            [actor.setup_worker.remote(root_address_bytes) for actor in self._actors]
        )

    def get_actor_for_block(self, block_idx: int) -> ActorHandle:
        """Round-robin distribution of input blocks across ranks."""
        return self._actors[block_idx % self._nranks]

    def shutdown(self, force: bool = False) -> None:
        if force:
            for actor in self._actors:
                try:
                    ray.kill(actor)
                except Exception:
                    pass
        self._actors.clear()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _derive_num_gpu_ranks(data_context: DataContext) -> int:
    """Return the configured or auto-detected number of GPU ranks."""
    if data_context.gpu_shuffle_num_actors is not None:
        return data_context.gpu_shuffle_num_actors

    total_resources = _get_total_cluster_resources()
    num_gpus = int(total_resources.gpu or 0)
    if num_gpus == 0:
        raise RuntimeError(
            "ShuffleStrategy.GPU_SHUFFLE requires GPU resources in the cluster. "
            "Set DataContext.gpu_shuffle_num_actors to override the number of ranks."
        )
    return num_gpus


# ---------------------------------------------------------------------------
# GPUJoinOperator
# ---------------------------------------------------------------------------


class GPUJoinOperator(PhysicalOperator, SubProgressBarMixin):
    """GPU-native join operator using RAPIDS MPF + UCXX.

    Takes two input operators (left at index 0, right at index 1) and
    orchestrates a three-phase GPU join:

    Phase 1 — right shuffle:
        Right blocks are immediately routed to actors via ``insert_right_batch``.
        Left blocks received before the right phase completes are buffered as
        object refs (no data copy) in ``_pending_left_bundles``.

    Phase 2 — left shuffle:
        Triggered by ``input_done(input_index=1)``.  ``right_insert_finished``
        is fire-and-forgot to all actors; then buffered left blocks and any
        subsequent left blocks are submitted via ``insert_left_batch``.

    Phase 3 — join:
        Triggered by ``all_inputs_done()``.  ``left_insert_finished`` +
        ``execute_join`` (streaming) are submitted fire-and-forget.  Ray's
        per-actor FIFO ordering guarantees correct sequencing.
    """

    def __init__(
        self,
        data_context: DataContext,
        left_input_op: PhysicalOperator,
        right_input_op: PhysicalOperator,
        join_type: JoinType,
        left_key_columns: List[str],
        right_key_columns: List[str],
        num_partitions: Optional[int] = None,
    ):
        nranks = _derive_num_gpu_ranks(data_context)
        target_nparts = max(
            num_partitions or data_context.default_hash_shuffle_parallelism,
            nranks,
        )

        super().__init__(
            name=(
                f"GPUJoin("
                f"type={join_type.value}, "
                f"left_keys={left_key_columns}, "
                f"right_keys={right_key_columns}, "
                f"num_partitions={target_nparts})"
            ),
            input_dependencies=[left_input_op, right_input_op],
            data_context=data_context,
        )

        self._join_type = join_type
        self._left_key_columns = left_key_columns
        self._right_key_columns = right_key_columns
        self._num_partitions = target_nparts
        self._rank_pool = GPUJoinRankPool(
            nranks=nranks,
            total_nparts=target_nparts,
            left_key_columns=left_key_columns,
            right_key_columns=right_key_columns,
            join_type=join_type,
            rmm_pool_size=data_context.gpu_shuffle_rmm_pool_size,
            spill_memory_limit=data_context.gpu_shuffle_spill_memory_limit,
        )

        # Round-robin block counters (separate for right and left so both
        # distribute evenly across actors).
        self._right_block_idx: int = 0
        self._left_block_idx: int = 0

        # Left blocks are buffered as RefBundles until the right phase
        # completes.  Only object refs are stored — no data copies.
        self._right_phase_done: bool = False
        self._pending_left_bundles: Deque[RefBundle] = deque()

        # Full Arrow schema captured from the first bundle of each side
        # (used to broadcast typed schema to actors that receive no batches).
        self._right_schema: Optional[pa.Schema] = None
        self._left_schema: Optional[pa.Schema] = None

        self._insert_tasks: Dict[int, MetadataOpTask] = {}
        self._extraction_tasks: Dict[int, DataOpTask] = {}
        self._finalization_started: bool = False
        self._output_queue: Deque[RefBundle] = deque()

        self._insert_stats: List[BlockStats] = []
        self._output_stats: List[BlockStats] = []

        self._shuffle_bar = None
        self._reduce_bar = None

        self._shuffle_metrics = OpRuntimeMetrics(self)
        self._reduce_metrics = OpRuntimeMetrics(self)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, options) -> None:
        super().start(options)
        self._rank_pool.start()

    # ------------------------------------------------------------------
    # Input routing
    # ------------------------------------------------------------------

    def _add_input_inner(self, bundle: RefBundle, input_index: int) -> None:
        self._shuffle_metrics.on_input_received(bundle)
        self._insert_stats.extend(to_stats(bundle.metadata))

        if input_index == 1:
            # Right side — submit immediately.
            if self._right_schema is None and bundle.schema is not None:
                self._right_schema = bundle.schema
            self._submit_right_bundle(bundle)
        else:
            # Left side — buffer if right phase not yet complete.
            if self._left_schema is None and bundle.schema is not None:
                self._left_schema = bundle.schema
            if self._right_phase_done:
                self._submit_left_bundle(bundle)
            else:
                self._pending_left_bundles.append(bundle)

    def _submit_right_bundle(self, bundle: RefBundle) -> None:
        for block_ref, _metadata in zip(bundle.block_refs, bundle.metadata):
            actor = self._rank_pool.get_actor_for_block(self._right_block_idx)
            insert_ref = actor.insert_right_batch.remote(block_ref)
            task_idx = self._right_block_idx
            self._right_block_idx += 1

            def _on_done(idx: int = task_idx) -> None:
                self._insert_tasks.pop(idx, None)

            task = MetadataOpTask(
                task_index=task_idx,
                object_ref=insert_ref,
                task_done_callback=_on_done,
                task_resource_bundle=ExecutionResources(),
            )
            self._insert_tasks[task_idx] = task

            if self._shuffle_bar is not None:
                self._shuffle_bar.update(total=self._right_block_idx)

    def _submit_left_bundle(self, bundle: RefBundle) -> None:
        # Use a separate counter offset so right and left task IDs don't clash.
        for block_ref, _metadata in zip(bundle.block_refs, bundle.metadata):
            actor = self._rank_pool.get_actor_for_block(self._left_block_idx)
            insert_ref = actor.insert_left_batch.remote(block_ref)
            # Offset left indices past right indices to keep task IDs unique.
            task_idx = 10_000_000 + self._left_block_idx
            self._left_block_idx += 1

            def _on_done(idx: int = task_idx) -> None:
                self._insert_tasks.pop(idx, None)

            task = MetadataOpTask(
                task_index=task_idx,
                object_ref=insert_ref,
                task_done_callback=_on_done,
                task_resource_bundle=ExecutionResources(),
            )
            self._insert_tasks[task_idx] = task

    # ------------------------------------------------------------------
    # Phase transitions
    # ------------------------------------------------------------------

    def input_done(self, input_index: int) -> None:
        """Called when one input operator has no more blocks.

        When the right input (index 1) is done we:
        1. Broadcast right Arrow schema to actors that received no batches.
        2. Fire-and-forget ``right_insert_finished`` on all actors.
        3. Flush any buffered left bundles (now safe to submit).
        """
        super().input_done(input_index)

        if input_index == 1 and not self._right_phase_done:
            for actor in self._rank_pool.actors:
                if self._right_schema is not None:
                    actor.set_right_schema.remote(self._right_schema)
                actor.right_insert_finished.remote()

            self._right_phase_done = True

            # Drain any left blocks that arrived before the right phase ended.
            while self._pending_left_bundles:
                self._submit_left_bundle(self._pending_left_bundles.popleft())

    def _is_inserting_done(self) -> bool:
        # _inputs_complete is set when all_inputs_done() has been called,
        # meaning both input_done(0) and input_done(1) have fired.
        return self._inputs_complete

    def _try_finalize(self) -> None:
        """Submit left_insert_finished + execute_join once all inserts are in."""
        if self._finalization_started or not self._is_inserting_done():
            return
        self._finalization_started = True

        for rank_idx, actor in enumerate(self._rank_pool.actors):
            if self._left_schema is not None:
                actor.set_left_schema.remote(self._left_schema)
            actor.left_insert_finished.remote()
            block_gen = actor.execute_join.options(num_returns="streaming").remote()

            def _on_bundle_ready(
                bundle: RefBundle, _rank: int = rank_idx
            ) -> None:
                self._output_queue.append(bundle)
                self._reduce_metrics.on_output_queued(bundle)

            def _on_extraction_done(
                exc: Optional[Exception],
                _rank: int = rank_idx,
                actor_handle: ActorHandle = actor,
            ) -> None:
                self._extraction_tasks.pop(_rank, None)
                try:
                    ray.kill(actor_handle, no_restart=True)
                except Exception:
                    pass

            data_task = DataOpTask(
                task_index=rank_idx,
                streaming_gen=block_gen,
                output_ready_callback=_on_bundle_ready,
                task_done_callback=_on_extraction_done,
            )
            self._extraction_tasks[rank_idx] = data_task

            empty_bundle = RefBundle([], schema=None, owns_blocks=False)
            self._reduce_metrics.on_task_submitted(
                rank_idx, empty_bundle, task_id=data_task.get_task_id()
            )

    # ------------------------------------------------------------------
    # Output interface
    # ------------------------------------------------------------------

    def has_next(self) -> bool:
        self._try_finalize()
        return len(self._output_queue) > 0

    def _get_next_inner(self) -> RefBundle:
        bundle = self._output_queue.popleft()
        self._reduce_metrics.on_output_dequeued(bundle)
        self._reduce_metrics.on_output_taken(bundle)
        self._output_stats.extend(to_stats(bundle.metadata))
        return bundle

    # ------------------------------------------------------------------
    # Task / completion tracking
    # ------------------------------------------------------------------

    def get_active_tasks(self) -> List[OpTask]:
        return list(self._insert_tasks.values()) + list(
            self._extraction_tasks.values()
        )

    def has_completed(self) -> bool:
        return (
            self._finalization_started
            and len(self._extraction_tasks) == 0
            and super().has_completed()
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _do_shutdown(self, force: bool = False) -> None:
        self._rank_pool.shutdown(force=True)
        super()._do_shutdown(force)
        self._insert_tasks.clear()
        self._extraction_tasks.clear()

    # ------------------------------------------------------------------
    # Resource accounting
    # ------------------------------------------------------------------

    @property
    def base_resource_usage(self) -> ExecutionResources:
        return ExecutionResources(gpu=self._rank_pool.nranks)

    def incremental_resource_usage(self) -> ExecutionResources:
        return ExecutionResources()

    def current_logical_usage(self) -> ExecutionResources:
        return self.base_resource_usage

    # ------------------------------------------------------------------
    # SubProgressBarMixin
    # ------------------------------------------------------------------

    def get_sub_progress_bar_names(self) -> List[str]:
        return ["GPU Join Shuffle", "GPU Join Reduce"]

    def set_sub_progress_bar(self, name: str, pg: Any) -> None:
        if name == "GPU Join Shuffle":
            self._shuffle_bar = pg
        elif name == "GPU Join Reduce":
            self._reduce_bar = pg

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self):
        return {
            f"{self._name}_shuffle": self._insert_stats,
            f"{self._name}_finalize": self._output_stats,
        }
