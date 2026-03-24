import logging
import time
import typing
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
import ray.exceptions
from ray.actor import ActorHandle
from ray.data import ExecutionOptions
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
from ray.data._internal.stats import OpRuntimeMetrics
from ray.data.block import BlockMetadataWithSchema, BlockStats, to_stats
from ray.data.context import DataContext

if typing.TYPE_CHECKING:
    from ray.data._internal.progress.base_progress import BaseProgressBar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU shuffle actor
# ---------------------------------------------------------------------------


@ray.remote(num_gpus=1)
class GPUShuffleActor:
    """One GPU rank in a RAPIDS MPF-based distributed shuffle.

    Each instance wraps a ``BulkRapidsMPFShuffler`` via composition rather than
    inheritance to keep CPU-only environments unaffected.

    Actors are arranged in a virtual communicator ring coordinated
    through UCXX; data never passes through the Ray object store or the CPU
    after initial ingestion.

    Constructor is intentionally lightweight — expensive UCXX setup happens in
    :meth:`setup_worker`, which is called once from :class:`GPURankPool`.
    """

    def __init__(
        self,
        nranks: int,
        total_nparts: int,
        key_columns: List[str],
        columns: Optional[List[str]] = None,
        rmm_pool_size: Union[int, str, None] = None,
        spill_memory_limit: Union[int, str, None] = "auto",
    ):
        from ray.data._internal.gpu_shuffle.rapidsmpf_backend import (
            BulkRapidsMPFShuffler,
        )

        self._shuffler = BulkRapidsMPFShuffler(
            nranks=nranks,
            total_nparts=total_nparts,
            shuffle_on=key_columns,
            rmm_pool_size=rmm_pool_size,
            spill_memory_limit=spill_memory_limit,
        )
        self._columns: Optional[List[str]] = None
        self._key_columns: List[str] = list(key_columns)

    # ------------------------------------------------------------------
    # UCXX communicator setup
    # ------------------------------------------------------------------

    def setup_root(self) -> tuple[int, bytes]:
        """Initialize the root communicator and return ``(rank, root_address_bytes)``.

        Only called on rank 0; the returned address is broadcast to all ranks
        via :meth:`setup_worker`.
        """
        logger.info("UCXX setup_root starting on rank 0.")
        t0 = time.perf_counter()
        result = self._shuffler.setup_root()
        elapsed = time.perf_counter() - t0
        logger.info("UCXX setup_root completed in %.2fs (rank=%d).", elapsed, result[0])
        return result

    def setup_worker(self, root_address: bytes) -> None:
        """Finish UCXX communicator setup and create the internal shuffler.

        Must be called on *every* rank (including rank 0) after
        :meth:`get_root_address` has been called on rank 0 and its result
        broadcast to all ranks.
        """
        logger.info(
            "UCXX setup_worker starting (root_address=%d bytes).",
            len(root_address),
        )
        t0 = time.perf_counter()
        self._shuffler.setup_worker(root_address)
        elapsed = time.perf_counter() - t0
        logger.info("UCXX setup_worker completed in %.2fs.", elapsed)

    def shutdown_and_exit(self) -> None:
        """Gracefully tear down UCXX/rapidsmpf state, then exit the actor."""
        try:
            self._shuffler.cleanup()
        except Exception:
            pass
        ray.actor.exit_actor()

    # ------------------------------------------------------------------
    # Insert / extract interface (called by GPUShuffleOperator)
    # ------------------------------------------------------------------

    def set_columns(self, columns: List[str]) -> None:
        """Set column names externally (e.g. from the operator).

        Needed when a rank receives shuffled data from peers but never gets
        a direct ``insert_batch`` call, which is the only other place where
        ``_columns`` is populated.
        """
        if self._columns is None:
            self._columns = list(columns)

    def insert_batch(self, batch: pa.Table) -> int:
        """Hash-partition *batch* and route shards to peers.

        Returns the number of rows in the incoming batch so the driver can
        track throughput without serialising the data back.
        """
        import cudf

        df = cudf.DataFrame.from_arrow(batch)
        # This is a fallback in case `infer_schema` is None, we need to then
        # infer from the first batch.
        if self._columns is None:
            self._columns = list(df.columns)
        self._shuffler.insert_chunk(table=df, column_names=self._columns)
        return len(df)

    def insert_finished(self) -> None:
        """Signal that no more batches will be inserted into this rank."""
        self._shuffler.insert_finished()

    def extract_partitions(self) -> Iterator:
        """Yield one Arrow Table per output partition assigned to this rank.

        This is a streaming generator; it must be called *after*
        :meth:`insert_finished` (Ray's per-actor task ordering guarantees
        this when both calls are submitted to the same actor handle).

        Follows the Ray Data streaming generator protocol: yield block then
        BlockMetadataWithSchema for each output partition.
        """
        from rapidsmpf.utils.cudf import pylibcudf_to_cudf_dataframe

        from ray.data.block import BlockExecStats, BlockMetadataWithSchema

        # columns = self._columns or []
        for _, partition in self._shuffler.extract():
            exec_stats_builder = BlockExecStats.builder()
            cdf = pylibcudf_to_cudf_dataframe(
                partition, column_names=self._columns
            ).copy(deep=True)
            # Caveat: The following operation copies the data to CPU memory, unless we use Arrow CUDA.
            block = cdf.to_arrow(preserve_index=False)
            exec_stats = exec_stats_builder.build()
            stats = yield block
            if stats:
                import dataclasses

                exec_stats = dataclasses.replace(
                    exec_stats,
                    block_ser_time_s=stats.object_creation_dur_s,
                )
            yield BlockMetadataWithSchema.from_block(block, block_exec_stats=exec_stats)

    def extract_and_apply(
        self,
        udf: Callable,
        keys: List[str],
        batch_format: Optional[str],
        fn_args: Tuple,
        fn_kwargs: Dict[str, Any],
        fn_constructor_args: Optional[Tuple],
        fn_constructor_kwargs: Optional[Dict[str, Any]],
    ) -> Iterator[Tuple[pa.Table, BlockMetadataWithSchema]]:
        """Fused extract + per-group UDF application on GPU.

        Applies *udf* directly on each GPU-resident cuDF partition right after
        extraction.  Only the (typically much smaller) UDF results are
        converted to Arrow for the Ray object store, eliminating the
        full-partition cuDF→Arrow→cuDF round-trip that would otherwise occur
        between ``extract_partitions`` and a downstream ``MapBatches`` task.

        Follows the same streaming generator protocol as ``extract_partitions``:
        yields alternating ``block`` / ``BlockMetadataWithSchema`` pairs.

        Args:
            udf: The user-defined function (or callable class) to apply.
            keys: Group-by key column names (empty list = global group).
            batch_format: Format passed to each group before calling *udf*
                (e.g. ``"cudf"``, ``"pandas"``, ``"pyarrow"``).
            fn_args: Positional arguments forwarded to *udf*.
            fn_kwargs: Keyword arguments forwarded to *udf*.
            fn_constructor_args: If *udf* is a callable class, these are
                passed to its ``__init__``.  One instance is created per rank.
            fn_constructor_kwargs: Keyword arguments for the callable class
                ``__init__``.

        Yields:
            Tuple[pa.Table, BlockMetadataWithSchema]:
                A tuple of the extracted partition and its metadata.
        """
        from collections.abc import Iterator as IteratorABC

        import pyarrow as pa
        from rapidsmpf.utils.cudf import pylibcudf_to_cudf_dataframe

        from ray.data._internal.cudf_block import CudfBlockAccessor
        from ray.data.block import (
            BlockAccessor,
            BlockExecStats,
            BlockMetadataWithSchema,
        )

        # Instantiate callable-class UDFs once per rank, mirroring the
        # ActorPoolMapOperator behaviour for fn_constructor_args.
        if fn_constructor_args is not None or fn_constructor_kwargs is not None:
            udf = udf(*(fn_constructor_args or ()), **(fn_constructor_kwargs or {}))

        columns = self._columns or []
        for _, partition in self._shuffler.extract():
            exec_stats_builder = BlockExecStats.builder()
            cdf = pylibcudf_to_cudf_dataframe(partition, column_names=columns).copy(
                deep=True
            )
            if self._key_columns:
                cdf = cdf.sort_values(by=self._key_columns)

            # Apply UDF to the full sorted partition.  The partition is
            # already hash-partitioned by group keys and sorted, so the UDF
            # receives all groups for its key-space contiguously.
            import cudf as _cudf

            if len(cdf) > 0:
                partition_batch = CudfBlockAccessor(cdf).to_batch_format(batch_format)
                result = udf(partition_batch, *fn_args, **fn_kwargs)
                results = list(result) if isinstance(result, IteratorABC) else [result]
                cudf_parts: List["_cudf.DataFrame"] = []
                for r in results:
                    if isinstance(r, _cudf.DataFrame):
                        cudf_parts.append(r)
                    else:
                        arrow_r = BlockAccessor.for_block(r).to_arrow()
                        cudf_parts.append(_cudf.DataFrame.from_arrow(arrow_r))
                combined = _cudf.concat(cudf_parts, ignore_index=True)
                block = combined.to_arrow(preserve_index=False)
            else:
                block = pa.table({})

            exec_stats = exec_stats_builder.build()
            stats = yield block
            if stats:
                import dataclasses

                exec_stats = dataclasses.replace(
                    exec_stats,
                    block_ser_time_s=stats.object_creation_dur_s,
                )
            yield BlockMetadataWithSchema.from_block(block, block_exec_stats=exec_stats)


def _wait_for_refs_with_timeout(
    refs: List[ray.ObjectRef],
    timeout_s: float,
    task_name: str,
) -> None:
    """Poll ``refs`` in a loop, raising on timeout or task failure.

    Logs incremental progress as tasks complete and raises any exceptions
    from completed tasks eagerly (via ``ray.get``).
    """
    total = len(refs)
    pending = list(refs)
    t_start = time.perf_counter()

    while pending:
        elapsed = time.perf_counter() - t_start
        if elapsed >= timeout_s:
            pending_indices = [i for i, ref in enumerate(refs) if ref in pending]
            raise TimeoutError(
                f"{task_name} did not complete on {len(pending)}/{total} "
                f"rank(s) within {timeout_s}s "
                f"(pending ranks: {pending_indices}). "
                f"Check GPU/network health."
            )
        ready, pending = ray.wait(pending, num_returns=len(pending), timeout=1)
        if ready:
            ray.get(ready)
            logger.info(
                "GPURankPool: %d/%d rank(s) completed %s.",
                total - len(pending),
                total,
                task_name,
            )


# ---------------------------------------------------------------------------
# GPURankPool — lifecycle manager for a set of GPUShuffleActors
# ---------------------------------------------------------------------------


class GPURankPool:
    """Manages the lifecycle of ``GPUShuffleActor`` instances.

    Analogous to ``AggregatorPool`` in the CPU hash-shuffle path, but for GPU
    ranks coordinated through UCXX.
    """

    def __init__(
        self,
        nranks: int,
        total_nparts: int,
        key_columns: List[str],
        columns: Optional[List[str]],
        rmm_pool_size: Union[int, str, None],
        spill_memory_limit: Union[int, str, None],
        setup_timeout_s: float,
    ):
        self._nranks = nranks
        self._total_nparts = total_nparts
        self._key_columns = key_columns
        self._columns = columns
        self._rmm_pool_size = rmm_pool_size
        self._spill_memory_limit = spill_memory_limit
        self._setup_timeout_s = setup_timeout_s
        self._actors: List[ActorHandle] = []

    @property
    def nranks(self) -> int:
        return self._nranks

    @property
    def actors(self) -> List[ActorHandle]:
        return self._actors

    def start(self) -> None:
        """Create actors and coordinate UCXX setup.

        This call *blocks* until all actors have finished UCXX initialisation.
        It is invoked once from ``GPUShuffleOperator.start()`` before any data
        flows through the pipeline.

        Raises:
            TimeoutError: If UCXX setup does not complete within
                ``setup_timeout_s`` seconds.
        """
        timeout = self._setup_timeout_s
        t_start = time.perf_counter()

        logger.info(
            "GPURankPool: creating %d GPUShuffleActor(s) "
            "(total_nparts=%d, key_columns=%s).",
            self._nranks,
            self._total_nparts,
            self._key_columns,
        )
        self._actors = [
            GPUShuffleActor.options(num_gpus=1, scheduling_strategy="SPREAD",).remote(
                nranks=self._nranks,
                total_nparts=self._total_nparts,
                key_columns=self._key_columns,
                rmm_pool_size=self._rmm_pool_size,
                spill_memory_limit=self._spill_memory_limit,
            )
            for _ in range(self._nranks)
        ]
        t_actors = time.perf_counter()
        logger.info(
            "GPURankPool: %d actor(s) created in %.2fs.",
            self._nranks,
            t_actors - t_start,
        )

        # Rank 0 establishes the root communicator; all ranks connect to it.
        remaining = max(0, timeout - (time.perf_counter() - t_start))
        logger.info("GPURankPool: calling setup_root on rank 0.")
        try:
            _, root_address_bytes = ray.get(
                self._actors[0].setup_root.remote(), timeout=remaining
            )
        except ray.exceptions.GetTimeoutError:
            raise TimeoutError(
                f"UCXX setup_root on rank 0 did not complete within "
                f"{timeout}s. Check GPU/network health."
            )
        t_root = time.perf_counter()
        logger.info(
            "GPURankPool: setup_root completed in %.2fs, "
            "broadcasting root address (%d bytes) to %d worker(s).",
            t_root - t_actors,
            len(root_address_bytes),
            self._nranks,
        )

        remaining = max(0, timeout - (time.perf_counter() - t_start))
        worker_refs = [
            actor.setup_worker.remote(root_address_bytes) for actor in self._actors
        ]
        _wait_for_refs_with_timeout(worker_refs, remaining, "setup_worker")
        t_done = time.perf_counter()
        logger.info(
            "GPURankPool: all %d worker(s) setup completed in %.2fs "
            "(total UCXX init: %.2fs).",
            self._nranks,
            t_done - t_root,
            t_done - t_start,
        )

    def get_actor_for_block(self, block_idx: int) -> ActorHandle:
        """Round-robin distribution of input blocks across ranks."""
        return self._actors[block_idx % self._nranks]

    def shutdown(self, force: bool = False) -> None:
        for actor in self._actors:
            if force:
                ray.kill(actor)
            else:
                try:
                    actor.shutdown_and_exit.remote()
                except Exception:
                    pass
        self._actors.clear()


# ---------------------------------------------------------------------------
# Helper: derive number of GPU ranks from the cluster
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
# GPUShuffleOperator
# ---------------------------------------------------------------------------


class GPUShuffleOperator(PhysicalOperator, SubProgressBarMixin):
    """GPU-native shuffle operator using RAPIDS MPF + UCXX.

    Unlike the CPU ``HashShuffleOperator``, this operator:

    * Uses UCXX point-to-point communication instead of the Ray object store
      for inter-rank data movement.
    * Accepts Arrow Tables from upstream (converting to cuDF on the actor) so
      it remains compatible with non-GPU upstream operators.
    * Supports repartition-only (no reduce/aggregate phase on the driver side).

    Lifecycle::

        start()                    # creates actors, blocks for UCXX setup
        _add_input_inner(bundle)   # routes blocks to actors round-robin
        [inputs_done()]            # called by the executor
        has_next() / _get_next_inner()   # streams output bundles

    The ``insert_finished`` + ``extract_partitions`` actor tasks are submitted
    as fire-and-forget once all inserts complete; Ray's per-actor task ordering
    guarantees correct sequencing without blocking the driver.
    """

    def __init__(
        self,
        input_op: PhysicalOperator,
        data_context: DataContext,
        *,
        key_columns: Tuple[str, ...],
        columns: Optional[List[str]] = None,
        num_partitions: Optional[int] = None,
        post_shuffle_udf: Optional[Callable] = None,
        keys_for_udf: Optional[List[str]] = None,
        batch_format: Optional[str] = None,
        fn_args: Optional[Tuple] = None,
        fn_kwargs: Optional[Dict[str, Any]] = None,
        fn_constructor_args: Optional[Tuple] = None,
        fn_constructor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        nranks = _derive_num_gpu_ranks(data_context)
        target_num_partitions = (
            num_partitions or data_context.default_hash_shuffle_parallelism
        )
        # rapidsmpf requires total_nparts >= nranks
        target_num_partitions = max(target_num_partitions, nranks)

        super().__init__(
            name=(
                f"GPUShuffle("
                f"key_columns={key_columns}, "
                f"num_partitions={target_num_partitions})"
            ),
            input_dependencies=[input_op],
            data_context=data_context,
        )

        self._key_columns = key_columns
        self._num_partitions = target_num_partitions
        self._rank_pool = GPURankPool(
            nranks=nranks,
            total_nparts=target_num_partitions,
            key_columns=list(key_columns),
            columns=columns,
            rmm_pool_size=data_context.gpu_shuffle_rmm_pool_size,
            spill_memory_limit=data_context.gpu_shuffle_spill_memory_limit,
            setup_timeout_s=data_context.gpu_shuffle_setup_timeout_s,
        )

        # Optional fused UDF (set when planning GPUShuffleMapGroups).
        self._post_shuffle_udf = post_shuffle_udf
        self._keys_for_udf: List[str] = keys_for_udf or []
        self._batch_format = batch_format
        self._fn_args: Tuple = fn_args or ()
        self._fn_kwargs: Dict[str, Any] = fn_kwargs or {}
        self._fn_constructor_args = fn_constructor_args
        self._fn_constructor_kwargs = fn_constructor_kwargs

        self._schema_columns: Optional[List[str]] = None
        self._next_block_idx: int = 0
        self._insert_tasks: Dict[int, MetadataOpTask] = {}
        self._extraction_tasks: Dict[int, DataOpTask] = {}
        self._finalization_started: bool = False
        self._output_queue: Deque[RefBundle] = deque()
        self._shuffled_blocks_stats: List[BlockStats] = []
        self._output_blocks_stats: List[BlockStats] = []

        # Progress bars (populated by SubProgressBarMixin callbacks)
        self._shuffle_bar = None
        self._reduce_bar = None

        # Metrics
        self._shuffle_metrics = OpRuntimeMetrics(self)
        self._reduce_metrics = OpRuntimeMetrics(self)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, options: ExecutionOptions) -> None:
        super().start(options)
        self._rank_pool.start()

    def _add_input_inner(self, bundle: RefBundle, input_index: int) -> None:
        self._shuffle_metrics.on_input_received(bundle)
        self._shuffled_blocks_stats.extend(to_stats(bundle.metadata))

        # Capture column names from the first input bundle so we can
        # broadcast them to all actors (including those that never
        # receive a direct insert_batch call).
        if self._schema_columns is None and bundle.schema is not None:
            self._schema_columns = list(bundle.schema.names)

        for block_ref, metadata in zip(bundle.block_refs, bundle.metadata):
            actor = self._rank_pool.get_actor_for_block(self._next_block_idx)
            insert_ref = actor.insert_batch.remote(block_ref)
            task_idx = self._next_block_idx
            self._next_block_idx += 1

            def _on_insert_done(idx: int = task_idx) -> None:
                self._insert_tasks.pop(idx, None)

            task = MetadataOpTask(
                task_index=task_idx,
                object_ref=insert_ref,
                task_done_callback=_on_insert_done,
                task_resource_bundle=ExecutionResources(),
            )
            self._insert_tasks[task_idx] = task
            self._shuffle_metrics.on_task_submitted(
                task_idx,
                RefBundle([(block_ref, metadata)], schema=None, owns_blocks=False),
                task_id=task.get_task_id(),
            )

            if self._shuffle_bar is not None:
                self._shuffle_bar.update(total=self._next_block_idx)

    def _is_inserting_done(self) -> bool:
        # Only require that all inserts have been *submitted*, not completed.
        # Ray's per-actor task ordering guarantees that insert_finished (and
        # extract) run after all pending insert_batch calls on each actor.
        return self._inputs_complete

    def _try_finalize(self) -> None:
        """Schedule extraction once all inserts have completed."""
        if self._finalization_started or not self._is_inserting_done():
            return

        self._finalization_started = True

        for rank_idx, actor in enumerate(self._rank_pool.actors):
            # Ensure every actor knows the column names, even those that
            # never received a direct insert_batch (fewer blocks than ranks).
            if self._schema_columns is not None:
                actor.set_columns.remote(self._schema_columns)
            # Fire-and-forget: Ray serialises actor tasks per actor, so
            # insert_finished is guaranteed to run before extract_partitions.
            actor.insert_finished.remote()
            if self._post_shuffle_udf is not None:
                block_gen = actor.extract_and_apply.options(
                    num_returns="streaming"
                ).remote(
                    self._post_shuffle_udf,
                    self._keys_for_udf,
                    self._batch_format,
                    self._fn_args,
                    self._fn_kwargs,
                    self._fn_constructor_args,
                    self._fn_constructor_kwargs,
                )
            else:
                block_gen = actor.extract_partitions.options(
                    num_returns="streaming"
                ).remote()

            def _on_bundle_ready(bundle: RefBundle, rank: int = rank_idx) -> None:
                self._output_queue.append(bundle)
                self._reduce_metrics.on_output_queued(bundle)

            def _on_extraction_done(
                exc: Optional[Exception],
                worker_stats=None,
                driver_stats=None,
                _rank: int = rank_idx,
                _actor_handle: ActorHandle = actor,
            ) -> None:
                self._extraction_tasks.pop(_rank, None)
                try:
                    _actor_handle.shutdown_and_exit.remote()
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
        self._output_blocks_stats.extend(to_stats(bundle.metadata))
        return bundle

    # ------------------------------------------------------------------
    # Task / completion tracking
    # ------------------------------------------------------------------

    def get_active_tasks(self) -> List[OpTask]:
        return list(self._insert_tasks.values()) + list(self._extraction_tasks.values())

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
        return ["GPU Shuffle", "GPU Reduce"]

    def set_sub_progress_bar(self, name: str, pg: "BaseProgressBar") -> None:
        if name == "GPU Shuffle":
            self._shuffle_bar = pg
        elif name == "GPU Reduce":
            self._reduce_bar = pg

    # ------------------------------------------------------------------
    # Actor info
    # ------------------------------------------------------------------

    def get_actor_info(self):
        from ray.data._internal.execution.interfaces.physical_operator import (
            ActorPoolInfo,
        )

        n = len(self._rank_pool.actors)
        return ActorPoolInfo(running=n, pending=0, restarting=0)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self):
        shuffle_name = f"{self._name}_shuffle"
        reduce_name = f"{self._name}_finalize"
        return {
            shuffle_name: self._shuffled_blocks_stats,
            reduce_name: self._output_blocks_stats,
        }
