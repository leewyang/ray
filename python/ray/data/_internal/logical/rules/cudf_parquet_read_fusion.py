"""Fuse a Parquet read with its downstream cuDF ``map_batches`` actor.

Normally, the read materializes host Arrow blocks::

    ListFiles -> ReadFiles(CPU) -> MapBatches(GPU)

Fusion sends the same file manifests to one GPU actor::

    ListFiles -> ReadFiles+MapBatches(GPU)

The actor decodes each assigned row-group range with cuDF and immediately runs the
already-planned UDF. Unsupported reads keep the original plan.
"""

from __future__ import annotations

import copy
import functools
import importlib
import inspect
import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Tuple

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from ray._common.retry import call_with_retry
from ray.data._internal.compute import ActorPoolStrategy, TaskPoolStrategy
from ray.data._internal.datasource_v2.chunkers.file_chunker import (
    ChunkMetadata,
    FileChunker,
    ParquetFileChunker,
)
from ray.data._internal.datasource_v2.listing.file_indexer import (
    NonSamplingFileIndexer,
)
from ray.data._internal.datasource_v2.listing.file_manifest import FileManifest
from ray.data._internal.datasource_v2.scanners.parquet_scanner import ParquetScanner
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.operators.actor_pool_map_operator import (
    ActorPoolMapOperator,
)
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
    BatchMapTransformFn,
    BlockMapTransformFn,
    MapTransformer,
)
from ray.data._internal.execution.operators.task_pool_map_operator import (
    TaskPoolMapOperator,
)
from ray.data._internal.logical.interfaces import (
    LogicalOperator,
    LogicalPlan,
    PhysicalPlan,
    Rule,
)
from ray.data._internal.logical.operators import ListFiles, MapBatches, ReadFiles
from ray.data._internal.object_extensions.arrow import _contains_pickle_object_type
from ray.data.block import Block, DataBatch
from ray.data.datasource.partitioning import PathPartitionParser

logger = logging.getLogger(__name__)

_DIAGNOSTIC_LOGGED_ATTR = "_cudf_parquet_read_fusion_diagnostic_logged"


@dataclass(frozen=True)
class _CudfParquetReadConfig:
    """Read settings captured by the logical rule for one fused actor."""

    scanner: ParquetScanner
    columns: Tuple[str, ...]
    batch_size: int
    retried_io_errors: Tuple[str, ...]


class ConfigureCudfParquetReadForFusion(Rule):
    """Choose exact row-group work units before physical planning.

    Ray's planner reads the ListFiles chunker when it builds the ReadFiles operator.
    This logical pass runs first so an eligible linear read-map pair is planned with
    exact row-group ranges.
    """

    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        if getattr(plan.context, "enable_cudf_parquet_read_fusion", False) is not True:
            return plan

        consumers = _count_consumers(set(plan.dag.post_order_iter()))

        def configure_map_input(operator: LogicalOperator) -> LogicalOperator:
            if not isinstance(operator, MapBatches):
                return operator
            if len(operator.input_dependencies) != 1:
                return operator
            read = operator.input_dependencies[0]
            if (
                not isinstance(read, ReadFiles)
                or consumers.get(operator, 0) > 1
                or consumers.get(read, 0) != 1
            ):
                return operator

            config = _logical_config_if_eligible(read, operator, plan.context)
            if config is None:
                return operator
            list_files = read.input_dependencies[0]
            if consumers.get(list_files, 0) != 1:
                return operator

            # Copy the input chain so this optimization never mutates the logical DAG.
            indexer = copy.deepcopy(list_files.file_indexer)
            indexer._file_chunker = _CudfParquetFileChunker(config)
            list_files = replace(list_files, file_indexer=indexer)
            read = replace(read, input_dependencies=[list_files])
            return replace(operator, input_dependencies=[read])

        rewritten_dag = plan.dag._apply_transform(configure_map_input)
        return (
            plan
            if rewritten_dag is plan.dag
            else LogicalPlan(rewritten_dag, plan.context)
        )


class FuseCudfParquetReadIntoMapBatches(Rule):
    """Fuse an eligible V2 Parquet read into its downstream GPU map actor."""

    def apply(self, plan: PhysicalPlan) -> PhysicalPlan:
        """Rewrite all eligible Parquet read-map pairs in ``plan``.

        Traversal starts at the plan's final physical operator and follows input
        dependencies upstream. Unaffected physical operators are reused and may be
        relinked to fused replacements.

        Args:
            plan: Physical plan to inspect and rewrite.

        Returns:
            The original plan when fusion is disabled or no eligible pair exists;
            otherwise, a new ``PhysicalPlan`` with an updated DAG and operator map.
        """

        if getattr(plan.context, "enable_cudf_parquet_read_fusion", False) is not True:
            return plan

        # Build a separate map so the input plan keeps its operator map.
        rewritten_op_map = plan.op_map.copy()

        # plan.dag is the final operator. Following its inputs reaches the whole plan.
        physical_consumer_counts = _count_consumers(set(plan.dag.post_order_iter()))

        # Reuse the first rewrite when two paths reach the same operator.
        rewritten_operator_by_original: Dict[PhysicalOperator, PhysicalOperator] = {}
        did_fuse = False

        def rewrite_upstream_graph(physical_op: PhysicalOperator) -> PhysicalOperator:
            """Rewrite the subgraph ending at ``physical_op``.

            If a fusible pair ends here, first rewrite the input before that pair and
            then replace the pair. Otherwise, rewrite every direct input. Cached results
            keep shared operators shared. Returns the downstream operator of the
            rewritten subgraph.
            """

            nonlocal did_fuse

            if physical_op in rewritten_operator_by_original:
                return rewritten_operator_by_original[physical_op]

            # Check only the pair ending here; recursion finds earlier pairs.
            candidate_pair = self._candidate_pair(physical_op, rewritten_op_map)
            if candidate_pair is not None:
                read_physical_op, read_logical_op, map_logical_op = candidate_pair
                config = self._fusion_config_if_eligible(
                    read_physical_op,
                    read_logical_op,
                    physical_op,
                    map_logical_op,
                    physical_consumer_counts,
                )
                if config is not None:
                    assert isinstance(read_physical_op, TaskPoolMapOperator)

                    # Continue before this pair so fusion does not skip upstream work.
                    rewritten_pair_input = rewrite_upstream_graph(
                        read_physical_op.input_dependency
                    )

                    fused_physical_op, fused_logical = self._create_fused_map_operator(
                        read_physical_op,
                        read_logical_op,
                        physical_op,
                        map_logical_op,
                        rewritten_pair_input,
                        plan.context,
                        config,
                    )

                    del rewritten_op_map[read_physical_op]
                    del rewritten_op_map[physical_op]
                    rewritten_op_map[fused_physical_op] = fused_logical
                    rewritten_operator_by_original[physical_op] = fused_physical_op
                    did_fuse = True
                    return fused_physical_op
                self._log_skip_once(physical_op, read_logical_op, map_logical_op)

            # No pair ends here, so continue through every input.
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

        rewritten_final_operator = rewrite_upstream_graph(plan.dag)

        if not did_fuse:
            return plan
        self._rebuild_output_dependencies(rewritten_final_operator)
        return PhysicalPlan(rewritten_final_operator, rewritten_op_map, plan.context)

    @staticmethod
    def _candidate_pair(
        downstream: PhysicalOperator,
        op_map: Mapping[PhysicalOperator, LogicalOperator],
    ) -> Optional[Tuple[MapOperator, ReadFiles, MapBatches]]:
        """Return the logical read-map pair ending at ``downstream``, if present."""

        logical = op_map.get(downstream)
        if (
            not isinstance(logical, MapBatches)
            or logical.batch_format != "cudf"
            or len(downstream.input_dependencies) != 1
        ):
            return None
        upstream = downstream.input_dependencies[0]
        upstream_logical = op_map.get(upstream)
        if not isinstance(upstream_logical, ReadFiles) or not isinstance(
            upstream, MapOperator
        ):
            return None
        return upstream, upstream_logical, logical

    @staticmethod
    def _fusion_config_if_eligible(
        read_physical: MapOperator,
        read_logical: ReadFiles,
        downstream_physical: PhysicalOperator,
        downstream_logical: MapBatches,
        physical_consumers: Mapping[PhysicalOperator, int],
    ) -> Optional[_CudfParquetReadConfig]:
        """Return the read config when planning preserved the eligible pair.

        The configured chunker proves that the logical rule accepted the pair. This
        method checks only state added or changed by physical planning.
        """

        # The custom chunker proves that the logical rule accepted this pair.
        list_files = read_logical.input_dependencies[0]
        chunker = list_files.file_indexer.file_chunker
        if type(chunker) is not _CudfParquetFileChunker:
            return None

        # Physical planning must leave one task-pool read with one consumer.
        read_transforms = read_physical.get_map_transformer().get_transform_fns()
        if (
            not isinstance(read_physical, TaskPoolMapOperator)
            or not read_physical.supports_fusion()
            or read_physical.get_max_concurrency_limit() is not None
            or read_physical.get_additional_split_factor() != 1
            or read_physical._on_start is not None
            or read_physical._map_task_kwargs
            or read_physical._map_task_kwargs_fns
            or len(read_transforms) != 1
            or type(read_transforms[0]) is not BlockMapTransformFn
            or physical_consumers.get(read_physical, 0) != 1
        ):
            return None

        # The downstream map must remain one unfused actor-map transform.
        if (
            not isinstance(downstream_physical, ActorPoolMapOperator)
            or not downstream_physical.supports_fusion()
            or downstream_physical.get_additional_split_factor() != 1
            or downstream_physical._on_start is not None
            or downstream_physical._map_task_kwargs
            or downstream_physical._map_task_kwargs_fns
        ):
            return None

        downstream_transformer = downstream_physical.get_map_transformer()
        transforms = downstream_transformer.get_transform_fns()
        # Preserve only the UDF transform; fusion supplies its reader and batcher.
        if len(transforms) != 1 or type(transforms[0]) is not BatchMapTransformFn:
            return None
        planned_batch_transform = transforms[0]
        if (
            planned_batch_transform._batch_format != "cudf"
            or planned_batch_transform._batch_size != downstream_logical.batch_size
        ):
            return None

        # Manifest rows and decoded rows have different sizes, so neither block-size
        # override transfers safely to the replacement.
        if (
            read_physical.target_max_block_size_override is not None
            or downstream_physical.target_max_block_size_override is not None
        ):
            return None

        return chunker._config

    @staticmethod
    def _create_fused_map_operator(
        read_physical: MapOperator,
        read_logical: ReadFiles,
        downstream_physical: PhysicalOperator,
        downstream_logical: MapBatches,
        input_physical_op: PhysicalOperator,
        context: Any,
        config: _CudfParquetReadConfig,
    ) -> Tuple[MapOperator, MapBatches]:
        """Create one actor map that reads manifests and runs the planned GPU UDF.

        The fused reader yields cuDF frames instead of decoded Ray blocks. Ray's
        ordinary batcher cannot consume those frames directly, so preserve the planned
        UDF transform and replace only its input reader and batcher.

        Args:
            read_physical: Planned Parquet read operator.
            read_logical: Logical Parquet read represented by that operator.
            downstream_physical: Planned GPU map operator.
            downstream_logical: Logical GPU map represented by that operator.
            input_physical_op: Rewritten input to the read operator.
            context: Ray Data context captured by the plan.
            config: Direct-read settings chosen by the logical rule.

        Returns:
            The executable physical replacement and its synthetic logical operator.
        """

        assert isinstance(downstream_physical, ActorPoolMapOperator)
        downstream_transformer = downstream_physical.get_map_transformer()
        (planned_batch_transform,) = downstream_transformer.get_transform_fns()
        output_option = downstream_transformer._output_block_size_option_override

        fused_map_transformer = MapTransformer(
            [
                BlockMapTransformFn(
                    _CudfParquetReader(config),
                    disable_block_shaping=True,
                ),
                _CudfBatchMapTransformFn(planned_batch_transform),
            ],
            init_fn=functools.partial(
                _init_fused_actor, downstream_transformer._init_fn
            ),
            output_block_size_option_override=(
                replace(output_option) if output_option is not None else None
            ),
        )

        name = f"{read_physical.name}->{downstream_physical.name}"

        fused_physical_op = MapOperator.create(
            fused_map_transformer,
            input_physical_op,
            context,
            name=name,
            compute_strategy=downstream_logical.compute,
            # The input rows are manifest entries, not decoded dataset rows.
            min_rows_per_bundle=None,
            # Stock fusion cannot interpret manifest rows as dataset rows.
            supports_fusion=False,
            ray_remote_args=dict(downstream_logical.ray_remote_args),
        )

        # The caller maps this physical operator to ``fused_logical_op`` in
        # ``op_map``. Keep the replaced logical operators separately as lineage
        # metadata for stats and plan inspection; they aren't dataflow inputs.
        fused_physical_op.set_logical_operators(read_logical, downstream_logical)
        fused_logical_op = replace(
            downstream_logical,
            input_dependencies=list(read_logical.input_dependencies),
            # Relative to ListFiles, this operator expands manifest entries into
            # decoded dataset rows even when the user UDF preserves row count.
            can_modify_num_rows=True,
            min_rows_per_bundled_input=None,
            ray_remote_args=dict(downstream_logical.ray_remote_args),
        )
        object.__setattr__(fused_logical_op, "_name", name)
        return fused_physical_op, fused_logical_op

    @staticmethod
    def _log_skip_once(
        physical: PhysicalOperator,
        read: ReadFiles,
        downstream: MapBatches,
    ) -> None:
        if getattr(physical, _DIAGNOSTIC_LOGGED_ATTR, False):
            return
        logger.debug(
            "Skipping direct cuDF Parquet read fusion for %s -> %s. Fusion "
            "requires a default local read or an S3 read using ambient AWS "
            "credentials, followed by a synchronous callable-class map_batches with "
            "batch_format='cudf', a positive integer batch_size, "
            "zero_copy_batch=True, num_gpus=1, and max_concurrency between 1 and "
            "4 when specified. The physical pair must also be "
            "linear, callback-free, and have no block-size override.",
            read.name,
            downstream.name,
        )
        setattr(physical, _DIAGNOSTIC_LOGGED_ATTR, True)

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


class _CudfParquetFileChunker(FileChunker):
    """Group contiguous Parquet row groups near the requested map batch size."""

    def __init__(self, config: _CudfParquetReadConfig):
        self._config = config
        self._partition_parser = (
            PathPartitionParser(config.scanner.partitioning)
            if config.scanner.partitioning is not None
            else None
        )

    def generate_chunk_metadatas(
        self, path: str, file_size: int
    ) -> Iterable[Tuple[Optional[ChunkMetadata], int]]:
        """Yield row-group ranges with estimated compressed-byte weights."""

        footer = call_with_retry(
            lambda: pq.read_metadata(path, filesystem=self._config.scanner.filesystem),
            description=f"read Parquet footer for {path}",
            match=self._config.retried_io_errors,
        )
        if footer.num_rows == 0:
            return

        row_group_ranges = []
        start_row_group = 0
        num_rows = 0
        for end_row_group in range(1, footer.num_row_groups + 1):
            num_rows += footer.row_group(end_row_group - 1).num_rows
            if num_rows >= self._config.batch_size:
                row_group_ranges.append((start_row_group, end_row_group))
                start_row_group = end_row_group
                num_rows = 0
        if start_row_group < footer.num_row_groups:
            row_group_ranges.append((start_row_group, footer.num_row_groups))

        needs_arrow_fallback = self._needs_arrow_fallback(path, footer)
        num_chunks = len(row_group_ranges)
        # Divide the listed file size across ranges so their scheduling weights still
        # sum to the original input size.
        chunk_size, remainder = divmod(file_size, num_chunks)
        for chunk_idx, (start_row_group, end_row_group) in enumerate(row_group_ranges):
            yield (
                {
                    "chunk_idx": chunk_idx,
                    "total_num_chunks": num_chunks,
                    "row_group_start_idx": start_row_group,
                    "row_group_end_idx": end_row_group,
                    "needs_arrow_fallback": needs_arrow_fallback,
                },
                chunk_size + (chunk_idx < remainder),
            )

    def _needs_arrow_fallback(self, path: str, footer: pq.FileMetaData) -> bool:
        """Return whether direct cuDF decoding would change the scanner output."""

        # Arrow synthesizes partition columns that a direct cuDF read would miss.
        if self._partition_parser is not None and self._partition_parser(path):
            return True

        physical_schema = footer.schema.to_arrow_schema()
        expected_schema = self._config.scanner.schema
        for name in self._config.columns:
            physical_index = physical_schema.get_field_index(name)
            if physical_index < 0:
                return True
            physical_type = physical_schema.field(physical_index).type
            expected_type = expected_schema.field(name).type
            if physical_type != expected_type or _contains_pickle_object_type(
                physical_type
            ):
                return True
        return False


class _CudfParquetReader:
    """Decode manifest row-group ranges inside the fused GPU actor."""

    def __init__(self, config: _CudfParquetReadConfig):
        self._config = config
        self._is_s3 = type(config.scanner.filesystem) is pafs.S3FileSystem
        self._can_read_s3_directly: Optional[bool] = None

    def __call__(self, manifest_blocks: Iterable[Block], _: Any) -> Iterator[DataBatch]:
        # Importing cuDF initializes CUDA and is deliberately deferred to the actor.
        import cudf

        can_read_s3_directly = self._direct_s3_read_is_available()
        storage_options = (
            {"client_kwargs": {"region_name": self._config.scanner.filesystem.region}}
            if self._is_s3 and can_read_s3_directly
            else None
        )
        for manifest_block in manifest_blocks:
            manifest = FileManifest(manifest_block)
            for path, size, chunk_metadata in zip(
                manifest.paths,
                manifest.file_sizes,
                manifest.file_chunk_metadatas,
            ):
                path = str(path)
                if chunk_metadata["needs_arrow_fallback"] or not can_read_s3_directly:
                    # Preserve Arrow's schema, partition, and S3 behavior inside this
                    # actor.
                    single_file = FileManifest.construct_manifest(
                        [path], [int(size)], [chunk_metadata]
                    )
                    for table in self._config.scanner.create_reader().read(single_file):
                        if table.num_rows:
                            yield table
                    continue

                # PyArrow stores S3 paths without a scheme; cuDF expects an S3 URI.
                cudf_path = f"s3://{path}" if self._is_s3 else path
                row_groups = list(
                    range(
                        int(chunk_metadata["row_group_start_idx"]),
                        int(chunk_metadata["row_group_end_idx"]),
                    )
                )
                frame = call_with_retry(
                    lambda: cudf.read_parquet(
                        cudf_path,
                        columns=list(self._config.columns),
                        row_groups=row_groups,
                        storage_options=storage_options,
                        use_pandas_metadata=False,
                        dataset_kwargs={"partitioning": None},
                    ),
                    description=f"read Parquet row groups from {path}",
                    match=self._config.retried_io_errors,
                )
                if len(frame):
                    yield frame

    def _direct_s3_read_is_available(self) -> bool:
        """Return whether cuDF can open S3 paths directly in this actor."""

        if not self._is_s3:
            return True
        if self._can_read_s3_directly is None:
            try:
                importlib.import_module("s3fs")
            except ImportError:
                # Without s3fs, use the scanner's PyArrow S3 filesystem inside this
                # actor.
                logger.debug("s3fs is unavailable; decoding S3 Parquet with Arrow.")
                self._can_read_s3_directly = False
            else:
                self._can_read_s3_directly = True
        return self._can_read_s3_directly


class _CudfBatchMapTransformFn(BatchMapTransformFn):
    """Run the planned UDF without a separate object-store Arrow boundary."""

    def __init__(self, planned: BatchMapTransformFn):
        output_option = planned._output_block_size_option
        super().__init__(
            planned._fn,
            is_udf=planned._is_udf,
            batch_size=planned._batch_size,
            batch_format=planned._batch_format,
            zero_copy_batch=planned._zero_copy_batch,
            output_block_size_option=(
                replace(output_option) if output_option is not None else None
            ),
        )

    def _pre_process(self, frames: Iterable[Any]) -> Iterator[DataBatch]:
        """Rebatch cuDF frames to the requested size, leaving the final batch short."""

        import cudf

        batch_size = self._batch_size
        batch_pieces = []
        num_buffered_rows = 0

        def combine_batch_pieces() -> Any:
            if len(batch_pieces) == 1:
                return batch_pieces[0].reset_index(drop=True)
            return cudf.concat(batch_pieces, ignore_index=True)

        for frame in frames:
            if isinstance(frame, pa.Table):
                # Arrow fallbacks rejoin the GPU path before the user UDF runs.
                frame = cudf.DataFrame.from_arrow(frame)
            elif not isinstance(frame, cudf.DataFrame):
                raise TypeError(
                    "Direct cuDF Parquet read produced an unsupported value: "
                    f"{type(frame)!r}"
                )
            frame_rows = len(frame)
            # Avoid slicing and concatenating an already complete batch.
            if not batch_pieces and frame_rows == batch_size:
                yield frame.reset_index(drop=True)
                continue

            offset = 0
            while offset < frame_rows:
                rows_to_take = min(batch_size - num_buffered_rows, frame_rows - offset)
                batch_pieces.append(frame.iloc[offset : offset + rows_to_take])
                offset += rows_to_take
                num_buffered_rows += rows_to_take
                if num_buffered_rows == batch_size:
                    yield combine_batch_pieces()
                    batch_pieces = []
                    num_buffered_rows = 0

        if num_buffered_rows:
            yield combine_batch_pieces()


def _init_fused_actor(init_udf: Any) -> None:
    """Best-effort configure reusable RMM allocations, then initialize the UDF."""

    try:
        import rmm

        current = rmm.mr.get_current_device_resource()
        # Leave custom memory resources unchanged.
        if type(current) is rmm.mr.CudaMemoryResource:
            available, _ = rmm.mr.available_device_memory()
            # Grow on demand, then release unused memory above one quarter of what
            # was available when the actor started.
            resource = rmm.mr.CudaAsyncMemoryResource(
                initial_pool_size=0,
                release_threshold=available // 4,
            )
            rmm.mr.set_current_device_resource(resource)
    except (AttributeError, ImportError, RuntimeError, TypeError):
        # Allocator tuning is optional; continue with the available allocator.
        logger.debug("Async RMM setup unavailable; using the available allocator.")

    init_udf()


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


def _count_consumers(operators: Iterable[Any]) -> Dict[Any, int]:
    """Count how many direct consumers refer to each operator."""

    counts = {}
    for operator in operators:
        for input_op in operator.input_dependencies:
            counts[input_op] = counts.get(input_op, 0) + 1
    return counts


def _has_default_read_remote_args(read: ReadFiles, context: Any) -> bool:
    """Return whether the read uses only the inherited scheduling strategy."""

    remote_args = read.ray_remote_args
    if set(remote_args) - {"scheduling_strategy"}:
        return False
    if "scheduling_strategy" not in remote_args:
        return True
    # Treat exception-raising or non-boolean equality as incompatible.
    try:
        matches_context = (
            remote_args["scheduling_strategy"] == context.scheduling_strategy
        )
    except Exception:
        return False
    return type(matches_context) is bool and matches_context


def _logical_config_if_eligible(
    read: ReadFiles,
    downstream: MapBatches,
    context: Any,
) -> Optional[_CudfParquetReadConfig]:
    """Return direct-read settings for one eligible logical read-map pair.

    Returning ``None`` leaves the ordinary Ray Data plan unchanged.
    """

    # Fusion makes read and map one retry unit, so require compatible failure settings.
    if (
        context.checkpoint_config is not None
        or context.raise_original_map_exception is not False
        or context.actor_init_retry_on_errors is not False
        or not isinstance(context.actor_task_retry_on_errors, bool)
        or not isinstance(context.retried_map_errors, bool)
        or context.max_errored_blocks != 0
    ):
        return None

    # Require an uncapped task-pool read directly after one ListFiles operator.
    if (
        type(read.compute) is not TaskPoolStrategy
        or read.compute.size is not None
        or read.block_udf is not None
        or not _has_default_read_remote_args(read, context)
        or len(read.input_dependencies) != 1
        or not isinstance(read.input_dependencies[0], ListFiles)
    ):
        return None

    indexer = read.input_dependencies[0].file_indexer
    chunker = indexer.file_chunker
    # Preserve caller-supplied sampling and chunking choices.
    if (
        type(indexer) is not NonSamplingFileIndexer
        or context.parquet_chunker_target_chunk_size is not None
        or type(chunker) is not ParquetFileChunker
        or chunker._target_chunk_size != ParquetFileChunker._DEFAULT_TARGET_CHUNK_SIZE
    ):
        return None

    # Reject scanner options that the direct cuDF call below cannot reproduce.
    scanner = read.scanner
    if type(scanner) is not ParquetScanner:
        return None
    filesystem = scanner.filesystem
    if type(filesystem) not in (pafs.LocalFileSystem, pafs.S3FileSystem):
        return None
    if (
        scanner.has_custom_read_behavior
        or scanner.parquet_format_kwargs
        or scanner.predicate is not None
        or scanner.partition_predicate is not None
        or scanner.partition_columns
        or scanner.limit is not None
        or scanner.include_paths
        or scanner.include_row_hash
        or scanner.batch_size is not None
        or scanner.ignore_prefixes
    ):
        return None

    if type(filesystem) is pafs.S3FileSystem:
        # Recreate only the region; credentials use the actor's ambient AWS chain.
        if not filesystem.region:
            return None

    columns = (
        tuple(scanner.columns)
        if scanner.columns is not None
        else tuple(scanner.schema.names)
    )
    # cuDF needs a nonempty, unique projection present in the planned schema.
    if (
        not columns
        or len(set(columns)) != len(columns)
        or any(scanner.schema.get_field_index(column) < 0 for column in columns)
    ):
        return None

    # Keep GPU UDF calls serial even when actor tasks overlap reading.
    if (
        downstream.batch_format != "cudf"
        or type(downstream.compute) is not ActorPoolStrategy
        or downstream.compute.enable_true_multi_threading is not False
        or type(downstream.batch_size) is not int
        or downstream.batch_size <= 0
        or downstream.zero_copy_batch is not True
        or downstream.per_block_limit is not None
        or downstream.ray_remote_args_fn is not None
        or not _is_synchronous_callable_class(downstream.fn)
    ):
        return None

    remote_args = downstream.ray_remote_args
    # Actor-option allow-list: add an option only when fusion preserves its behavior.
    if type(remote_args) is not dict or set(remote_args) - {
        "max_concurrency",
        "num_gpus",
    }:
        return None
    max_concurrency = remote_args.get("max_concurrency", 1)
    # Cap overlap at four tasks to bound how many decoded frames one GPU can retain.
    if type(max_concurrency) is not int or max_concurrency not in (1, 2, 3, 4):
        return None
    num_gpus = remote_args.get("num_gpus")
    if type(num_gpus) not in (int, float) or num_gpus != 1:
        return None

    return _CudfParquetReadConfig(
        scanner=scanner,
        columns=columns,
        batch_size=downstream.batch_size,
        retried_io_errors=tuple(context.retried_io_errors),
    )
