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
    ParquetFileChunkMetadata,
    create_chunk_metadata,
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


@dataclass(frozen=True)
class _CudfParquetReadConfig:
    """Settings used by one fused actor."""

    scanner: ParquetScanner
    columns: Tuple[str, ...]
    batch_size: int
    retried_io_errors: Tuple[str, ...]


class _CudfParquetFileChunkMetadata(ParquetFileChunkMetadata):
    """An exact row-group range for one fused GPU read."""

    row_group_start_idx: int
    row_group_end_idx: int
    needs_arrow_fallback: bool


class _CudfParquetFileChunker(FileChunker):
    """Split one Parquet file into GPU-sized row-group ranges."""

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
        footer = call_with_retry(
            lambda: pq.read_metadata(path, filesystem=self._config.scanner.filesystem),
            description=f"read Parquet footer for {path}",
            match=self._config.retried_io_errors,
        )
        if footer.num_rows == 0:
            return

        ranges = []
        start = 0
        rows = 0
        for end in range(1, footer.num_row_groups + 1):
            rows += footer.row_group(end - 1).num_rows
            if rows >= self._config.batch_size:
                ranges.append((start, end))
                start = end
                rows = 0
        if start < footer.num_row_groups:
            ranges.append((start, footer.num_row_groups))

        needs_arrow_fallback = self._needs_arrow_fallback(path, footer)
        num_chunks = len(ranges)
        chunk_size, remainder = divmod(file_size, num_chunks)
        for chunk_idx, (start, end) in enumerate(ranges):
            yield (
                create_chunk_metadata(
                    _CudfParquetFileChunkMetadata,
                    chunk_idx=chunk_idx,
                    total_num_chunks=num_chunks,
                    row_group_start_idx=start,
                    row_group_end_idx=end,
                    needs_arrow_fallback=needs_arrow_fallback,
                ),
                chunk_size + (chunk_idx < remainder),
            )

    def _needs_arrow_fallback(self, path: str, footer: pq.FileMetaData) -> bool:
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
    """Read V2 manifest entries into cuDF frames."""

    def __init__(self, config: _CudfParquetReadConfig):
        self._config = config
        self._is_s3 = type(config.scanner.filesystem) is pafs.S3FileSystem

    def __call__(self, blocks: Iterable[Block], _: Any) -> Iterator[DataBatch]:
        # Importing cuDF initializes CUDA and is deliberately deferred to the actor.
        import cudf

        for block in blocks:
            manifest = FileManifest(block)
            for path, size, chunk_metadata in zip(
                manifest.paths,
                manifest.file_sizes,
                manifest.file_chunk_metadatas,
            ):
                path = str(path)
                if chunk_metadata["needs_arrow_fallback"]:
                    # Preserve Arrow's schema and partition handling in this actor.
                    yield from self._read_with_arrow(path, int(size), chunk_metadata)
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
                        storage_options=self._s3_storage_options(),
                        use_pandas_metadata=False,
                        dataset_kwargs={"partitioning": None},
                    ),
                    description=f"read Parquet row groups from {path}",
                    match=self._config.retried_io_errors,
                )
                if len(frame):
                    yield frame

    def _s3_storage_options(self) -> Optional[Dict[str, Any]]:
        if not self._is_s3:
            return None
        # Let s3fs discover and refresh the actor's ambient AWS credentials.
        return {
            "client_kwargs": {"region_name": self._config.scanner.filesystem.region},
        }

    def _read_with_arrow(
        self,
        path: str,
        size: int,
        chunk_metadata: _CudfParquetFileChunkMetadata,
    ) -> Iterator[pa.Table]:
        """Run one manifest entry through the original V2 reader in this actor."""

        single_file = FileManifest.construct_manifest([path], [size], [chunk_metadata])
        for table in self._config.scanner.create_reader().read(single_file):
            if table.num_rows:
                yield table


class _CudfBatchMapTransformFn(BatchMapTransformFn):
    """Reuse a planned batch UDF with device-native input batching."""

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
            target_batch_size_bytes=planned._target_batch_size_bytes,
            reports_custom_op_stats=planned._reports_custom_op_stats,
        )

    def _pre_process(self, frames: Iterable[Any]) -> Iterator[DataBatch]:
        """Form exact row batches because Ray's batcher can't consume cuDF frames."""

        import cudf

        batch_size = self._batch_size
        pieces = []
        rows = 0

        def combine() -> Any:
            if len(pieces) == 1:
                return pieces[0].reset_index(drop=True)
            return cudf.concat(pieces, ignore_index=True)

        for frame in frames:
            if isinstance(frame, pa.Table):
                frame = cudf.DataFrame.from_arrow(frame)
            elif not isinstance(frame, cudf.DataFrame):
                raise TypeError(
                    "Direct cuDF Parquet read produced an unsupported value: "
                    f"{type(frame)!r}"
                )
            frame_rows = len(frame)
            if not pieces and frame_rows == batch_size:
                yield frame.reset_index(drop=True)
                continue

            offset = 0
            while offset < frame_rows:
                count = min(batch_size - rows, frame_rows - offset)
                pieces.append(frame.iloc[offset : offset + count])
                offset += count
                rows += count
                if rows == batch_size:
                    yield combine()
                    pieces = []
                    rows = 0

        if rows:
            yield combine()


def _init_fused_actor(init_udf: Any) -> None:
    """Reuse RMM device allocations inside the fused cuDF actor."""

    import rmm

    current = rmm.mr.get_current_device_resource()
    # Leave custom memory resources unchanged.
    if type(current) is rmm.mr.CudaMemoryResource:
        available, _ = rmm.mr.available_device_memory()
        # Grow on demand, then release unused memory above one quarter of what
        # was available when the actor started.
        try:
            resource = rmm.mr.CudaAsyncMemoryResource(
                initial_pool_size=0,
                release_threshold=available // 4,
            )
        except RuntimeError:
            logger.debug("CUDA asynchronous allocation is unavailable.")
        else:
            rmm.mr.set_current_device_resource(resource)

    init_udf()


def _is_synchronous_callable_class(fn: Any) -> bool:
    if not inspect.isclass(fn):
        return False
    call = inspect.getattr_static(fn, "__call__", None)
    if isinstance(call, (classmethod, staticmethod)):
        call = call.__func__
    return bool(
        call is not None
        and not inspect.iscoroutinefunction(call)
        and not inspect.isasyncgenfunction(call)
        and not inspect.isgeneratorfunction(call)
    )


def _count_consumers(operators: Iterable[Any]) -> Dict[Any, int]:
    counts = {}
    for operator in operators:
        for input_op in operator.input_dependencies:
            counts[input_op] = counts.get(input_op, 0) + 1
    return counts


def _has_default_read_remote_args(read: ReadFiles, context: Any) -> bool:
    args = read.ray_remote_args
    if set(args) - {"scheduling_strategy"}:
        return False
    if "scheduling_strategy" not in args:
        return True
    try:
        result = args["scheduling_strategy"] == context.scheduling_strategy
    except Exception:
        return False
    return type(result) is bool and result


def _logical_config_if_eligible(
    read: ReadFiles,
    downstream: MapBatches,
    context: Any,
) -> Optional[_CudfParquetReadConfig]:
    # Keep checkpointing, retries, errors, and actor task admission unchanged.
    if (
        context.checkpoint_config is not None
        or context.raise_original_map_exception is not False
        or context.actor_init_retry_on_errors is not False
        or not isinstance(context.actor_task_retry_on_errors, bool)
        or not isinstance(context.retried_map_errors, bool)
        or context.max_errored_blocks != 0
    ):
        return None

    if (
        downstream.batch_format != "cudf"
        or type(read.compute) is not TaskPoolStrategy
        or read.compute.size is not None
        or read.block_udf is not None
        or not _has_default_read_remote_args(read, context)
        or len(read.input_dependencies) != 1
        or not isinstance(read.input_dependencies[0], ListFiles)
    ):
        return None

    indexer = read.input_dependencies[0].file_indexer
    if type(indexer) is not NonSamplingFileIndexer or type(
        indexer.file_chunker
    ) not in (ParquetFileChunker, _CudfParquetFileChunker):
        return None

    # Reject scanner features that a direct cuDF read would skip.
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
        # Accept only the default ambient-credential configuration.
        region = filesystem.region
        if not region:
            return None
        try:
            state = filesystem.__reduce__()[1][0]
            default_state = pafs.S3FileSystem(region=region).__reduce__()[1][0]
        except Exception:
            return None
        if type(state) is not dict or state != default_state:
            return None

    columns = (
        tuple(scanner.columns)
        if scanner.columns is not None
        else tuple(scanner.schema.names)
    )
    if (
        not columns
        or len(set(columns)) != len(columns)
        or any(scanner.schema.get_field_index(column) < 0 for column in columns)
    ):
        return None

    # Reuse only a serial callable-class actor that owns one GPU.
    if (
        type(downstream.compute) is not ActorPoolStrategy
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
    if type(remote_args) is not dict or set(remote_args) - _FUSION_SAFE_REMOTE_ARGS:
        return None
    max_concurrency = remote_args.get("max_concurrency", 1)
    if type(max_concurrency) is not int or max_concurrency != 1:
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


class ConfigureCudfParquetReadForFusion(Rule):
    """Size Parquet work units for an eligible fused GPU map."""

    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        if getattr(plan.context, "enable_cudf_parquet_read_fusion", False) is not True:
            return plan

        consumers = _count_consumers(set(plan.dag.post_order_iter()))

        def rewrite(operator: LogicalOperator) -> LogicalOperator:
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
            if consumers.get(
                list_files, 0
            ) != 1 or not self._uses_default_parquet_chunker(list_files, plan.context):
                return operator

            indexer = copy.deepcopy(list_files.file_indexer)
            indexer._file_chunker = _CudfParquetFileChunker(config)
            list_files = replace(list_files, file_indexer=indexer)
            read = replace(read, input_dependencies=[list_files])
            return replace(operator, input_dependencies=[read])

        rewritten = plan.dag._apply_transform(rewrite)
        return plan if rewritten is plan.dag else LogicalPlan(rewritten, plan.context)

    @staticmethod
    def _uses_default_parquet_chunker(list_files: ListFiles, context: Any) -> bool:
        indexer = list_files.file_indexer
        if type(indexer) is not NonSamplingFileIndexer:
            return False
        chunker = indexer.file_chunker
        return (
            context.parquet_chunker_target_chunk_size is None
            and type(chunker) is ParquetFileChunker
            and chunker._target_chunk_size
            == ParquetFileChunker._DEFAULT_TARGET_CHUNK_SIZE
        )


class FuseCudfParquetReadIntoMapBatches(Rule):
    """Fuse one compatible V2 Parquet read into its GPU actor consumer."""

    def apply(self, plan: PhysicalPlan) -> PhysicalPlan:
        """Rewrite every eligible read-map pair without changing the logical plan.

        Return ``plan`` itself when nothing fuses.
        """

        if getattr(plan.context, "enable_cudf_parquet_read_fusion", False) is not True:
            return plan

        # Keep the input plan's operator map unchanged.
        rewritten_op_map = plan.op_map.copy()
        # Check both graphs because physical planning can copy logical operators.
        logical_consumers = _count_consumers(set(rewritten_op_map.values()))
        physical_consumers = _count_consumers(set(plan.dag.post_order_iter()))
        # Cache rewrites so shared operators stay shared.
        rewritten: Dict[PhysicalOperator, PhysicalOperator] = {}
        did_fuse = False

        def rewrite(physical_op: PhysicalOperator) -> PhysicalOperator:
            nonlocal did_fuse
            if physical_op in rewritten:
                return rewritten[physical_op]

            # Check only the pair ending here; recursion finds earlier pairs.
            pair = self._candidate_pair(physical_op, rewritten_op_map)
            if pair is not None:
                read_physical, read_logical, map_logical = pair
                config = self._config_if_eligible(
                    read_physical,
                    read_logical,
                    physical_op,
                    map_logical,
                    plan,
                    logical_consumers,
                    physical_consumers,
                )
                if config is not None:
                    # Rewrite work before the pair, then replace the pair.
                    fused_input = rewrite(read_physical.input_dependency)
                    fused_physical, fused_logical = self._create_fused_operator(
                        read_physical,
                        read_logical,
                        physical_op,
                        map_logical,
                        fused_input,
                        plan,
                        config,
                    )
                    rewritten_op_map.pop(read_physical)
                    rewritten_op_map.pop(physical_op)
                    rewritten_op_map[fused_physical] = fused_logical
                    rewritten[physical_op] = fused_physical
                    did_fuse = True
                    return fused_physical
                self._log_skip_once(physical_op, read_logical, map_logical)

            # No replacement ends here, so rewrite each input.
            new_inputs = []
            inputs_changed = False
            for input_op in physical_op.input_dependencies:
                new_input = rewrite(input_op)
                new_inputs.append(new_input)
                inputs_changed |= new_input is not input_op
            if inputs_changed:
                # Output links are rebuilt after all input links are final.
                physical_op._input_dependencies = new_inputs
            rewritten[physical_op] = physical_op
            return physical_op

        final_operator = rewrite(plan.dag)
        if not did_fuse:
            return plan
        self._rebuild_output_dependencies(final_operator)
        return PhysicalPlan(final_operator, rewritten_op_map, plan.context)

    @staticmethod
    def _candidate_pair(
        downstream: PhysicalOperator,
        op_map: Mapping[PhysicalOperator, LogicalOperator],
    ) -> Optional[Tuple[MapOperator, ReadFiles, MapBatches]]:
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
    def _config_if_eligible(
        read_physical: MapOperator,
        read_logical: ReadFiles,
        downstream_physical: PhysicalOperator,
        downstream_logical: MapBatches,
        plan: PhysicalPlan,
        logical_consumers: Mapping[LogicalOperator, int],
        physical_consumers: Mapping[PhysicalOperator, int],
    ) -> Optional[_CudfParquetReadConfig]:
        context = plan.context
        config = _logical_config_if_eligible(read_logical, downstream_logical, context)
        if config is None:
            return None

        # Exact ranges are added by the logical rule before physical planning.
        list_files = read_logical.input_dependencies[0]
        if type(list_files.file_indexer.file_chunker) is not _CudfParquetFileChunker:
            return None

        # The read must be an ordinary, unshared V2 read.
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
            or logical_consumers.get(read_logical, 0) != 1
            or physical_consumers.get(read_physical, 0) != 1
        ):
            return None

        # The physical pair must still match the logical actor-read shape.
        if (
            not isinstance(downstream_physical, ActorPoolMapOperator)
            or not downstream_physical.supports_fusion()
            or downstream_physical.get_additional_split_factor() != 1
            or downstream_physical._on_start is not None
        ):
            return None

        downstream_transformer = downstream_physical.get_map_transformer()
        transforms = downstream_transformer.get_transform_fns()
        if len(transforms) != 1 or type(transforms[0]) is not BatchMapTransformFn:
            return None
        planned_batch_transform = transforms[0]
        if (
            planned_batch_transform._batch_format != "cudf"
            or planned_batch_transform._batch_size != downstream_logical.batch_size
        ):
            return None

        read_target = read_physical.target_max_block_size_override
        downstream_target = downstream_physical.target_max_block_size_override
        # A fused operator can preserve only one block-size override.
        if (
            read_target is not None
            and downstream_target is not None
            and read_target != downstream_target
        ):
            return None

        return config

    @staticmethod
    def _create_fused_operator(
        read_physical: MapOperator,
        read_logical: ReadFiles,
        downstream_physical: PhysicalOperator,
        downstream_logical: MapBatches,
        input_physical: PhysicalOperator,
        plan: PhysicalPlan,
        config: _CudfParquetReadConfig,
    ) -> Tuple[MapOperator, MapBatches]:
        assert isinstance(downstream_physical, ActorPoolMapOperator)
        downstream_transformer = downstream_physical.get_map_transformer()
        (planned_batch_transform,) = downstream_transformer.get_transform_fns()
        output_option = downstream_transformer._output_block_size_option_override

        # The standard planner can't consume file manifests in a GPU actor. Reuse
        # its planned UDF transform and replace only the input batcher.
        fused_transformer = MapTransformer(
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

        read_target = read_physical.target_max_block_size_override
        downstream_target = downstream_physical.target_max_block_size_override
        target_max_block_size = (
            read_target if read_target is not None else downstream_target
        )
        name = f"{read_physical.name}->{downstream_physical.name}"

        fused_physical = MapOperator.create(
            fused_transformer,
            input_physical,
            plan.context,
            target_max_block_size_override=target_max_block_size,
            name=name,
            compute_strategy=downstream_logical.compute,
            # The input rows are manifest entries, not decoded dataset rows.
            min_rows_per_bundle=None,
            # Do not offer this custom transform to another fusion rule.
            supports_fusion=False,
            map_task_kwargs=dict(downstream_physical._map_task_kwargs),
            ray_remote_args=dict(downstream_logical.ray_remote_args),
        )
        for callback in downstream_physical._map_task_kwargs_fns:
            fused_physical.add_map_task_kwargs_fn(callback)

        # Keep original operators as lineage only; op_map gets a synthetic node.
        fused_physical.set_logical_operators(read_logical, downstream_logical)
        fused_logical = replace(
            downstream_logical,
            input_dependencies=list(read_logical.input_dependencies),
            # Relative to ListFiles, this operator expands manifest entries into
            # decoded dataset rows even when the user UDF preserves row count.
            can_modify_num_rows=True,
            min_rows_per_bundled_input=None,
            ray_remote_args=dict(downstream_logical.ray_remote_args),
        )
        object.__setattr__(fused_logical, "_name", name)
        return fused_physical, fused_logical

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
            "credentials, followed by a serial callable-class map_batches with "
            "batch_format='cudf', "
            "zero_copy_batch=True, and num_gpus=1.",
            read.name,
            downstream.name,
        )
        setattr(physical, _DIAGNOSTIC_LOGGED_ATTR, True)

    @staticmethod
    def _rebuild_output_dependencies(final_operator: PhysicalOperator) -> None:
        """Rebuild producer links after rewriting consumer inputs."""

        operators = set(final_operator.post_order_iter())
        for operator in operators:
            operator._output_dependencies = []
        for operator in operators:
            for input_op in operator.input_dependencies:
                input_op._output_dependencies.append(operator)


__all__ = [
    "ConfigureCudfParquetReadForFusion",
    "FuseCudfParquetReadIntoMapBatches",
]
