"""CPU-only tests for direct cuDF Parquet read-to-map fusion."""

import builtins
import os
import sys
from dataclasses import replace
from types import ModuleType

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

import ray
from ray.data._internal.compute import ActorPoolStrategy, TaskPoolStrategy
from ray.data._internal.datasource_v2.chunkers.file_chunker import (
    ParquetFileChunker,
    WholeFileChunker,
)
from ray.data._internal.datasource_v2.listing.file_manifest import FileManifest
from ray.data._internal.datasource_v2.listing.listing_utils import sample_files
from ray.data._internal.datasource_v2.parquet_datasource_v2 import (
    ParquetDatasourceV2,
)
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.execution.operators.actor_pool_map_operator import (
    ActorPoolMapOperator,
)
from ray.data._internal.logical.interfaces import LogicalPlan
from ray.data._internal.logical.operators import ListFiles, MapBatches, ReadFiles, Union
from ray.data._internal.logical.optimizers import get_execution_plan
from ray.data._internal.logical.rules import cudf_parquet_read_fusion
from ray.data._internal.logical.rules.cudf_parquet_read_fusion import (
    ConfigureCudfParquetReadForFusion,
    FuseCudfParquetReadIntoMapBatches,
    _CudfBatchMapTransformFn,
    _CudfParquetFileChunker,
    _CudfParquetReader,
    _init_fused_actor,
)
from ray.data._internal.object_extensions.arrow import ArrowPythonObjectArray
from ray.data._internal.output_buffer import OutputBlockSizeOption
from ray.data._internal.planner import create_planner
from ray.data.context import (
    DEFAULT_ENABLE_CUDF_PARQUET_READ_FUSION,
    DataContext,
)
from ray.data.expressions import col


class _Identity:
    def __call__(self, batch):
        return batch


class _AsyncIdentity:
    async def __call__(self, batch):
        return batch


class _RecordBatches:
    def __init__(self, seen):
        self._seen = seen

    def __call__(self, batch):
        self._seen.append((len(batch), tuple(batch.to_arrow().column_names)))
        return batch


def _identity_function(batch):
    return batch


class _FakeIloc:
    def __init__(self, frame):
        self._frame = frame

    def __getitem__(self, item):
        assert isinstance(item, slice)
        start = item.start or 0
        stop = len(self._frame) if item.stop is None else item.stop
        return _FakeDataFrame(self._frame._table.slice(start, stop - start))


class _FakeDataFrame:
    """Minimal device-frame stand-in used after planning."""

    def __init__(self, table):
        self._table = table

    def __len__(self):
        return self._table.num_rows

    @property
    def iloc(self):
        return _FakeIloc(self)

    @classmethod
    def from_arrow(cls, table):
        return cls(table)

    def reset_index(self, *, drop):
        assert drop is True
        return _FakeDataFrame(self._table)

    def copy(self, *, deep):
        assert deep is True
        return _FakeDataFrame(self._table)

    def to_arrow(self, preserve_index=False):
        assert preserve_index is False
        return self._table


@pytest.fixture
def fake_cudf(monkeypatch):
    calls = []
    module = ModuleType("cudf")
    module.DataFrame = _FakeDataFrame
    module._path_overrides = {}

    def read_parquet(path, **kwargs):
        call = {
            "path": path,
            "columns": tuple(kwargs["columns"]),
            "row_groups": tuple(kwargs["row_groups"]),
        }
        if kwargs.get("storage_options") is not None:
            call["storage_options"] = kwargs["storage_options"]
        calls.append(call)
        table = pq.ParquetFile(module._path_overrides.get(path, path)).read_row_groups(
            kwargs["row_groups"], columns=kwargs["columns"]
        )
        return _FakeDataFrame(table)

    def concat(frames, *, ignore_index):
        assert ignore_index is True
        return _FakeDataFrame(pa.concat_tables([frame._table for frame in frames]))

    module.read_parquet = read_parquet
    module.concat = concat
    monkeypatch.setitem(sys.modules, "cudf", module)
    return calls


@pytest.fixture
def fake_rmm(monkeypatch):
    module = ModuleType("rmm")
    module.mr = ModuleType("rmm.mr")

    class CudaMemoryResource:
        pass

    class CudaAsyncMemoryResource:
        def __init__(self, *, initial_pool_size, release_threshold):
            self.initial_pool_size = initial_pool_size
            self.release_threshold = release_threshold
            module.mr.async_constructions += 1

    module.mr.CudaMemoryResource = CudaMemoryResource
    module.mr.CudaAsyncMemoryResource = CudaAsyncMemoryResource
    module.mr.async_constructions = 0
    module.mr.current = CudaMemoryResource()
    module.mr.get_current_device_resource = lambda: module.mr.current
    module.mr.available_device_memory = lambda: (8_192, 16_384)

    def set_current_device_resource(resource):
        module.mr.current = resource

    module.mr.set_current_device_resource = set_current_device_resource
    monkeypatch.setitem(sys.modules, "rmm", module)
    return module.mr


def _write_parquet(path, *, rows=12, row_group_size=3, start=0):
    values = list(range(start, start + rows))
    table = pa.table(
        {
            "value": values,
            "unused": [f"row-{value}" for value in values],
        }
    )
    pq.write_table(table, path, row_group_size=row_group_size)
    return table


def _write_uneven_parquet(path, row_group_sizes=(2, 5, 1, 6)):
    tables = []
    start = 0
    writer = None
    try:
        for size in row_group_sizes:
            table = pa.table(
                {
                    "value": range(start, start + size),
                    "unused": [f"row-{value}" for value in range(start, start + size)],
                }
            )
            writer = writer or pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
            tables.append(table)
            start += size
    finally:
        if writer is not None:
            writer.close()
    return pa.concat_tables(tables)


def _make_read_files(
    path,
    *,
    scanner_overrides=None,
    read_overrides=None,
    default_partitioning=False,
):
    paths = path if isinstance(path, (list, tuple)) else [path]
    datasource_kwargs = {} if default_partitioning else {"partitioning": None}
    datasource = ParquetDatasourceV2([str(path) for path in paths], **datasource_kwargs)
    indexer = datasource._get_file_indexer()
    sample = sample_files(
        indexer, datasource.paths, datasource.filesystem, pruners=None
    )
    schema = datasource.infer_schema(sample)
    scanner = datasource.create_scanner(
        schema=schema,
        filesystem=datasource.filesystem,
        partitioning=datasource.resolve_partitioning(sample),
    )
    if scanner_overrides:
        scanner = replace(scanner, **scanner_overrides)

    list_files = ListFiles(
        paths=list(datasource.paths),
        file_indexer=indexer,
        filesystem=datasource.filesystem,
        source_paths=list(datasource.paths),
        file_extensions=datasource.file_extensions,
    )
    options = {
        "datasource_name": datasource.name,
        "scanner": scanner,
        "schema": schema,
        "parallelism": -1,
        "compute": TaskPoolStrategy(),
        "ray_remote_args": {},
        "block_udf": None,
        "input_dependencies": [list_files],
    }
    options.update(read_overrides or {})
    return ReadFiles(**options)


def _make_gpu_map(input_op, fn=_Identity, **overrides):
    options = {
        "batch_size": 4,
        "batch_format": "cudf",
        "zero_copy_batch": True,
        "compute": ActorPoolStrategy(size=1),
        "ray_remote_args": {"num_gpus": 1},
    }
    options.update(overrides)
    return MapBatches(fn, input_dependencies=[input_op], **options)


def _context(*, enabled=True):
    context = DataContext.get_current().copy()
    context.enable_cudf_parquet_read_fusion = enabled
    return context


def _configure(root, *, enabled=True, context=None):
    context = context or _context(enabled=enabled)
    return ConfigureCudfParquetReadForFusion().apply(LogicalPlan(root, context))


def _plan(root, *, enabled=True, context=None):
    plan, _ = create_planner().plan(_configure(root, enabled=enabled, context=context))
    return plan


def _is_direct_cudf_read(operator):
    if not isinstance(operator, ActorPoolMapOperator):
        return False
    transforms = operator.get_map_transformer().get_transform_fns()
    return (
        len(transforms) == 2
        and isinstance(transforms[0]._fn, _CudfParquetReader)
        and isinstance(transforms[1], _CudfBatchMapTransformFn)
    )


def _find_read_physical(plan, read_logical):
    del read_logical  # Planning copies the configured logical operators.
    return next(
        physical
        for physical, logical in plan.op_map.items()
        if isinstance(logical, ReadFiles)
    )


def test_default_off_is_identity_and_planning_does_not_import_cudf(
    ray_start_regular_shared_2_cpus, tmp_path, monkeypatch
):
    path = tmp_path / "input.parquet"
    _write_parquet(path)
    root = _make_gpu_map(_make_read_files(path))
    real_import = builtins.__import__

    def import_without_cudf(name, *args, **kwargs):
        if name == "cudf" or name.startswith("cudf."):
            raise AssertionError("physical planning must not import cuDF")
        return real_import(name, *args, **kwargs)

    assert DEFAULT_ENABLE_CUDF_PARQUET_READ_FUSION is False
    assert DataContext().enable_cudf_parquet_read_fusion is False
    monkeypatch.setattr(builtins, "__import__", import_without_cudf)
    raw = _plan(root, enabled=False)
    result = FuseCudfParquetReadIntoMapBatches().apply(raw)
    assert result is raw
    assert not _is_direct_cudf_read(result.dag)


def test_logical_rule_installs_exact_chunker_and_keeps_cpu_fallback(tmp_path):
    path = tmp_path / "input.parquet"
    expected = _write_uneven_parquet(path)
    read = _make_read_files(path)
    mapped = _make_gpu_map(read, batch_size=6)
    original_list = read.input_dependencies[0]
    original_indexer = original_list.file_indexer
    configured = _configure(mapped)
    configured_read = configured.dag.input_dependencies[0]
    configured_list = configured_read.input_dependencies[0]
    chunker = configured_list.file_indexer.file_chunker
    file_size = os.path.getsize(path)
    chunks = list(chunker.generate_chunk_metadatas(str(path), file_size))

    assert configured.dag is not mapped
    assert configured_read is not read
    assert configured_list is not original_list
    assert configured_list.file_indexer is not original_indexer
    assert isinstance(chunker, _CudfParquetFileChunker)
    assert isinstance(original_indexer.file_chunker, ParquetFileChunker)
    assert [
        (metadata["row_group_start_idx"], metadata["row_group_end_idx"])
        for metadata, _ in chunks
    ] == [(0, 2), (2, 4)]
    assert all(
        metadata["total_num_chunks"] == 2 and not metadata["needs_arrow_fallback"]
        for metadata, _ in chunks
    )
    assert sum(size for _, size in chunks) == file_size
    chunk_metadata, chunk_sizes = zip(*chunks)
    manifest = FileManifest.construct_manifest(
        [str(path)] * len(chunks), list(chunk_sizes), list(chunk_metadata)
    )
    # A later physical-rule rejection still sends ordinary chunk keys to Arrow.
    tables = list(configured_read.scanner.create_reader().read(manifest))
    assert pa.concat_tables(tables).sort_by("value").to_pydict() == expected.to_pydict()


@pytest.mark.parametrize(
    "case",
    [
        "custom-chunker",
        "custom-size",
        "context-chunker",
    ],
)
def test_logical_rule_preserves_nondefault_configuration(tmp_path, case):
    path = tmp_path / "input.parquet"
    _write_parquet(path)
    read = _make_read_files(path)
    context = _context()

    if case == "custom-chunker":
        read.input_dependencies[0].file_indexer._file_chunker = WholeFileChunker()
    elif case == "custom-size":
        read.input_dependencies[0].file_indexer._file_chunker = ParquetFileChunker(
            target_chunk_size=64
        )
    elif case == "context-chunker":
        context.parquet_chunker_target_chunk_size = 64
    plan = LogicalPlan(_make_gpu_map(read), context)

    assert ConfigureCudfParquetReadForFusion().apply(plan) is plan


@pytest.mark.parametrize(
    ("max_concurrency", "expected_tasks_in_flight"),
    [(None, 2), (2, 4), (3, 6), (4, 8)],
)
def test_public_api_shape_selects_direct_cudf_decode(
    ray_start_regular_shared_2_cpus,
    tmp_path,
    monkeypatch,
    max_concurrency,
    expected_tasks_in_flight,
):
    path = tmp_path / "input.parquet"
    _write_parquet(path)
    context = DataContext.get_current()
    monkeypatch.setattr(context, "use_datasource_v2", True)
    monkeypatch.setattr(context, "enable_cudf_parquet_read_fusion", True)

    remote_args = {"num_gpus": 1}
    if max_concurrency is not None:
        remote_args["max_concurrency"] = max_concurrency
    dataset = ray.data.read_parquet(str(path), columns=["value"]).map_batches(
        _Identity,
        batch_format="cudf",
        batch_size=4,
        compute=ActorPoolStrategy(size=1),
        **remote_args,
    )
    physical, _ = get_execution_plan(dataset._logical_plan)
    configured_list = dataset._logical_plan.dag.input_dependencies[
        0
    ].input_dependencies[0]

    assert isinstance(
        configured_list.file_indexer.file_chunker, _CudfParquetFileChunker
    )
    assert _is_direct_cudf_read(physical.dag)
    reader = physical.dag.get_map_transformer().get_transform_fns()[0]._fn
    assert reader._config.columns == ("value",)
    assert physical.dag._ray_remote_args["num_gpus"] == 1
    if max_concurrency is None:
        assert "max_concurrency" not in physical.dag._ray_remote_args
    else:
        assert physical.dag._ray_remote_args["max_concurrency"] == max_concurrency
    assert physical.dag._actor_pool.max_actor_concurrency() == (max_concurrency or 1)
    assert (
        physical.dag._actor_pool.max_tasks_in_flight_per_actor()
        == expected_tasks_in_flight
    )


def test_fused_actor_uses_async_device_memory(fake_rmm):
    initialized_with = []

    _init_fused_actor(lambda: initialized_with.append(fake_rmm.current))

    resource = fake_rmm.current
    assert type(resource) is fake_rmm.CudaAsyncMemoryResource
    assert resource.initial_pool_size == 0
    assert resource.release_threshold == 2_048
    assert initialized_with == [resource]


def test_fused_actor_preserves_configured_device_memory_resource(fake_rmm):
    configured = object()
    fake_rmm.current = configured
    initialized = []

    _init_fused_actor(lambda: initialized.append(True))

    assert fake_rmm.current is configured
    assert initialized == [True]


def test_fused_actor_reuses_async_resource_after_init_retry(fake_rmm):
    attempts = 0

    def init():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("retry actor initialization")

    with pytest.raises(RuntimeError, match="retry actor initialization"):
        _init_fused_actor(init)
    resource = fake_rmm.current

    _init_fused_actor(init)

    assert fake_rmm.current is resource
    assert fake_rmm.async_constructions == 1
    assert attempts == 2


def test_fused_actor_allows_udf_to_replace_device_memory_resource(fake_rmm):
    configured = object()

    _init_fused_actor(lambda: setattr(fake_rmm, "current", configured))
    _init_fused_actor(lambda: None)

    assert fake_rmm.current is configured
    assert fake_rmm.async_constructions == 1


def test_fused_actor_keeps_default_resource_when_async_is_unavailable(
    fake_rmm, monkeypatch
):
    default = fake_rmm.current

    def unsupported_async_resource(**_):
        raise RuntimeError("unsupported")

    monkeypatch.setattr(
        fake_rmm,
        "CudaAsyncMemoryResource",
        unsupported_async_resource,
    )
    initialized = []

    _init_fused_actor(lambda: initialized.append(True))

    assert fake_rmm.current is default
    assert initialized == [True]


@pytest.mark.parametrize("error_type", [AttributeError, RuntimeError, TypeError])
@pytest.mark.parametrize("failure_point", ["memory-query", "resource-install"])
def test_fused_actor_ignores_optional_allocator_setup_failures(
    fake_rmm, monkeypatch, failure_point, error_type
):
    default = fake_rmm.current

    def unavailable(*_, **__):
        raise error_type("unavailable")

    if failure_point == "memory-query":
        monkeypatch.setattr(fake_rmm, "available_device_memory", unavailable)
    else:
        monkeypatch.setattr(fake_rmm, "set_current_device_resource", unavailable)
    initialized = []

    _init_fused_actor(lambda: initialized.append(True))

    assert fake_rmm.current is default
    assert initialized == [True]


def test_fused_actor_initializes_without_rmm(monkeypatch):
    monkeypatch.setitem(sys.modules, "rmm", None)
    initialized = []

    _init_fused_actor(lambda: initialized.append(True))

    assert initialized == [True]


def test_fake_cudf_execution_preserves_projection_row_groups_and_batching(
    ray_start_regular_shared_2_cpus, tmp_path, monkeypatch, fake_cudf, fake_rmm
):
    path = tmp_path / "input.parquet"
    _write_parquet(path)
    read = _make_read_files(path, scanner_overrides={"columns": ("value",)})
    seen = []
    mapped = _make_gpu_map(
        read,
        _RecordBatches,
        fn_constructor_args=(seen,),
    )
    fused = FuseCudfParquetReadIntoMapBatches().apply(_plan(mapped))
    assert _is_direct_cudf_read(fused.dag)
    reader = fused.dag.get_map_transformer().get_transform_fns()[0]._fn
    chunks = list(
        _CudfParquetFileChunker(reader._config).generate_chunk_metadatas(
            str(path), os.path.getsize(path)
        )
    )
    chunk_metadata, chunk_sizes = zip(*chunks)
    manifest = FileManifest.construct_manifest(
        [str(path)] * len(chunks), list(chunk_sizes), list(chunk_metadata)
    )
    monkeypatch.setattr(
        cudf_parquet_read_fusion.pq,
        "read_metadata",
        lambda *_, **__: pytest.fail("the actor must not read a footer"),
    )

    previous_actor_context = ray.data._map_actor_context
    try:
        ray.data._map_actor_context = None
        transformer = fused.dag.get_map_transformer()
        transformer.init()
        output = list(
            transformer.apply_transform(
                [manifest.as_block()], TaskContext(task_idx=0, op_name="test")
            )
        )
    finally:
        ray.data._map_actor_context = previous_actor_context

    assert fake_cudf == [
        {"path": str(path), "columns": ("value",), "row_groups": (0, 1)},
        {"path": str(path), "columns": ("value",), "row_groups": (2, 3)},
    ]
    assert seen == [(4, ("value",)), (4, ("value",)), (4, ("value",))]
    table = pa.concat_tables(output)
    assert table.column("value").to_pylist() == list(range(12))


@pytest.mark.parametrize(
    "filesystem",
    [
        pytest.param(pafs.S3FileSystem(), id="inferred-region"),
        pytest.param(
            pafs.S3FileSystem(region="us-east-1"),
            id="explicit-region",
        ),
    ],
)
def test_ambient_s3_read_passes_uri_and_region_to_cudf(
    ray_start_regular_shared_2_cpus, tmp_path, fake_cudf, filesystem, monkeypatch
):
    monkeypatch.setitem(sys.modules, "s3fs", ModuleType("s3fs"))
    local_path = tmp_path / "input.parquet"
    _write_parquet(local_path, rows=2, row_group_size=2)
    s3_path = "bucket/folder/input file.parquet"
    cudf_path = f"s3://{s3_path}"

    read = _make_read_files(
        local_path,
        scanner_overrides={"filesystem": filesystem},
    )
    fused = FuseCudfParquetReadIntoMapBatches().apply(_plan(_make_gpu_map(read)))
    assert _is_direct_cudf_read(fused.dag)
    reader = fused.dag.get_map_transformer().get_transform_fns()[0]._fn

    # Keep this offline and exercise only the actor's S3 URI and region.
    sys.modules["cudf"]._path_overrides[cudf_path] = str(local_path)

    manifest = FileManifest.construct_manifest(
        [s3_path],
        [os.path.getsize(local_path)],
        [
            {
                "chunk_idx": 0,
                "total_num_chunks": 1,
                "row_group_start_idx": 0,
                "row_group_end_idx": 1,
                "needs_arrow_fallback": False,
            }
        ],
    )
    assert len(list(reader([manifest.as_block()], None))) == 1

    assert [call["path"] for call in fake_cudf] == [cudf_path]
    assert fake_cudf[0]["storage_options"] == {
        "client_kwargs": {"region_name": filesystem.region},
    }


def test_s3_without_s3fs_decodes_with_arrow_in_actor(
    ray_start_regular_shared_2_cpus, tmp_path, fake_cudf, monkeypatch
):
    path = tmp_path / "input.parquet"
    _write_parquet(path, rows=2, row_group_size=2)
    read = _make_read_files(path)
    fused = FuseCudfParquetReadIntoMapBatches().apply(_plan(_make_gpu_map(read)))
    reader = fused.dag.get_map_transformer().get_transform_fns()[0]._fn

    # Exercise the transport fallback offline with the scanner's local path.
    reader._is_s3 = True
    monkeypatch.setitem(sys.modules, "s3fs", None)
    manifest = FileManifest.construct_manifest(
        [str(path)],
        [os.path.getsize(path)],
        [
            {
                "chunk_idx": 0,
                "total_num_chunks": 1,
                "row_group_start_idx": 0,
                "row_group_end_idx": 1,
                "needs_arrow_fallback": False,
            }
        ],
    )

    table = pa.concat_tables(reader([manifest.as_block()], None))
    assert table.column("value").to_pylist() == [0, 1]
    assert fake_cudf == []


@pytest.mark.parametrize(
    "case", ["hive-partition", "schema-mismatch", "nested-pickled-object"]
)
def test_incompatible_file_uses_arrow_fallback(
    ray_start_regular_shared_2_cpus, tmp_path, fake_cudf, case
):
    sample_path = tmp_path / "sample.parquet"
    if case == "hive-partition":
        _write_parquet(sample_path, rows=2, row_group_size=2)
        input_path = tmp_path / "group=blue" / "part.parquet"
        input_path.parent.mkdir()
        _write_parquet(input_path, rows=2, row_group_size=2, start=2)
        read = _make_read_files([sample_path, input_path], default_partitioning=True)
        assert read.scanner.partition_columns == set()
    elif case == "schema-mismatch":
        _write_parquet(sample_path, rows=2, row_group_size=2)
        input_path = tmp_path / "input.parquet"
        pq.write_table(
            pa.table(
                {
                    "value": pa.array([2, 3], type=pa.int32()),
                    "unused": ["row-2", "row-3"],
                }
            ),
            input_path,
        )
        read = _make_read_files(sample_path)
    else:
        objects = ArrowPythonObjectArray.from_objects([{"a": 1}, {"b": 2}])
        nested = pa.ListArray.from_arrays(pa.array([0, 1, 2]), objects)
        input_path = sample_path
        pq.write_table(pa.table({"value": nested}), input_path)
        read = _make_read_files(input_path)

    fused = FuseCudfParquetReadIntoMapBatches().apply(_plan(_make_gpu_map(read)))
    reader = fused.dag.get_map_transformer().get_transform_fns()[0]._fn
    metadata, size = next(
        iter(
            _CudfParquetFileChunker(reader._config).generate_chunk_metadatas(
                str(input_path), os.path.getsize(input_path)
            )
        )
    )
    manifest = FileManifest.construct_manifest([str(input_path)], [size], [metadata])
    assert metadata["needs_arrow_fallback"] is True
    if case == "nested-pickled-object":
        assert fake_cudf == []
        return
    table = pa.concat_tables(reader([manifest.as_block()], None))
    assert table.column("value").to_pylist() == [2, 3]
    if case == "hive-partition":
        assert table.column("group").to_pylist() == ["blue", "blue"]
    else:
        assert table.schema.field("value").type == pa.int64()
    assert fake_cudf == []


@pytest.mark.parametrize("tasks_in_flight_source", ["actor-pool", "context"])
@pytest.mark.parametrize("max_concurrency", [1, 2, 3, 4])
def test_fusion_preserves_planned_udf_and_actor_pool(
    ray_start_regular_shared_2_cpus,
    tmp_path,
    tasks_in_flight_source,
    max_concurrency,
):
    path = tmp_path / "input.parquet"
    _write_parquet(path)
    read = _make_read_files(path)
    context = _context()
    compute = ActorPoolStrategy(size=1)
    if tasks_in_flight_source == "actor-pool":
        compute = ActorPoolStrategy(size=1, max_tasks_in_flight_per_actor=3)
    else:
        context.max_tasks_in_flight_per_actor = 3
    remote_args = {
        "num_gpus": 1,
        "max_concurrency": max_concurrency,
    }
    mapped = _make_gpu_map(read, compute=compute, ray_remote_args=remote_args)
    raw = _plan(mapped, context=context)
    downstream_transformer = raw.dag.get_map_transformer()
    (planned_batch_transform,) = downstream_transformer.get_transform_fns()

    original_physical_remote_args = dict(raw.dag._ray_remote_args)
    fused = FuseCudfParquetReadIntoMapBatches().apply(raw)
    fused_transformer = fused.dag.get_map_transformer()
    fused_batch_transform = fused_transformer.get_transform_fns()[1]

    assert isinstance(fused_batch_transform, _CudfBatchMapTransformFn)
    assert fused_batch_transform._fn is planned_batch_transform._fn
    for field in (
        "_is_udf",
        "_batch_size",
        "_batch_format",
        "_zero_copy_batch",
    ):
        assert getattr(fused_batch_transform, field) == getattr(
            planned_batch_transform, field
        )
    assert fused_transformer._init_fn.func is _init_fused_actor
    assert fused_transformer._init_fn.args == (downstream_transformer._init_fn,)
    assert fused.dag._ray_remote_args == original_physical_remote_args
    fused_pool = fused.dag._actor_pool
    assert fused_pool.max_tasks_in_flight_per_actor() == 3
    assert fused_pool.max_actor_concurrency() == max_concurrency
    assert fused_pool.per_actor_resource_usage().gpu == 1


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("checkpoint_config", True),
        ("raise_original_map_exception", True),
        ("actor_init_retry_on_errors", True),
        ("actor_task_retry_on_errors", ["ValueError"]),
        ("retried_map_errors", ["ValueError"]),
        ("max_errored_blocks", 1),
    ],
)
def test_nondefault_context_settings_fail_closed(
    tmp_path,
    setting,
    value,
):
    path = tmp_path / "input.parquet"
    _write_parquet(path)
    context = _context()
    if setting == "checkpoint_config":
        context._checkpoint_config = object()
    else:
        setattr(context, setting, value)
    plan = LogicalPlan(_make_gpu_map(_make_read_files(path)), context)

    assert ConfigureCudfParquetReadForFusion().apply(plan) is plan


@pytest.mark.parametrize(
    "remote_args",
    [
        {},
        {"num_gpus": 1, "future_actor_option": True},
        {"num_gpus": 1, "memory": 1_048_576},
        {"num_gpus": 1, "enable_task_events": False},
        {"num_gpus": 1, "runtime_env": {}},
        {"num_gpus": 0.25},
        {"num_gpus": float("nan")},
        {"num_gpus": 1, "max_concurrency": True},
        {"num_gpus": 1, "max_concurrency": 2.0},
        {"num_gpus": 1, "max_concurrency": 5},
    ],
)
def test_unsupported_remote_args_fail_closed(
    ray_start_regular_shared_2_cpus,
    tmp_path,
    remote_args,
):
    path = tmp_path / "input.parquet"
    _write_parquet(path)
    raw = _plan(_make_gpu_map(_make_read_files(path), ray_remote_args=remote_args))

    assert FuseCudfParquetReadIntoMapBatches().apply(raw) is raw


@pytest.mark.parametrize(
    "filesystem",
    [
        pytest.param(
            pafs.S3FileSystem(region="us-east-1", anonymous=True),
            id="anonymous",
        ),
        pytest.param(
            pafs.S3FileSystem(
                region="us-east-1", access_key="key", secret_key="secret"
            ),
            id="static-credentials",
        ),
        pytest.param(
            pafs.S3FileSystem(region="us-east-1", endpoint_override="localhost:9000"),
            id="custom-endpoint",
        ),
        pytest.param(
            pafs.S3FileSystem(region="us-east-1", scheme="http"),
            id="plain-http",
        ),
    ],
)
def test_nonambient_s3_configuration_fails_closed(
    ray_start_regular_shared_2_cpus, tmp_path, filesystem
):
    path = tmp_path / "input.parquet"
    _write_parquet(path)
    read = _make_read_files(
        path,
        scanner_overrides={
            "filesystem": filesystem,
            "has_custom_read_behavior": True,
        },
    )
    raw = _plan(_make_gpu_map(read))

    assert FuseCudfParquetReadIntoMapBatches().apply(raw) is raw


@pytest.mark.parametrize(
    "case",
    [
        "custom-read",
        "predicate",
        "synthetic-column",
        "block-udf",
        "batch-size",
        "task-map",
        "async-map",
        "dynamic-args",
        "read-split",
        "read-on-start",
        "read-task-args",
        "map-on-start",
        "map-task-args",
        "map-task-args-fn",
        "read-target",
        "map-target",
    ],
)
def test_ineligible_candidate_fails_closed(
    ray_start_regular_shared_2_cpus,
    tmp_path,
    case,
):
    path = tmp_path / "input.parquet"
    _write_parquet(path)
    scanner_overrides = {}
    read_overrides = {}
    map_overrides = {}
    fn = _Identity

    if case == "custom-read":
        scanner_overrides["has_custom_read_behavior"] = True
    elif case == "predicate":
        scanner_overrides["predicate"] = col("value") > 0
    elif case == "synthetic-column":
        scanner_overrides["include_paths"] = True
    elif case == "block-udf":
        read_overrides["block_udf"] = lambda block: block
    elif case == "batch-size":
        map_overrides["batch_size"] = True
    elif case == "task-map":
        fn = _identity_function
        map_overrides["compute"] = TaskPoolStrategy()
    elif case == "async-map":
        fn = _AsyncIdentity
    elif case == "dynamic-args":
        map_overrides["ray_remote_args_fn"] = lambda: {"num_gpus": 1}

    read = _make_read_files(
        path,
        scanner_overrides=scanner_overrides,
        read_overrides=read_overrides,
    )
    raw = _plan(_make_gpu_map(read, fn, **map_overrides))
    if case == "read-split":
        _find_read_physical(raw, read).set_additional_split_factor(2)
    elif case == "read-on-start":
        _find_read_physical(raw, read)._on_start = lambda schema: None
    elif case == "read-task-args":
        _find_read_physical(raw, read)._map_task_kwargs = {"read": True}
    elif case == "map-on-start":
        raw.dag._on_start = lambda schema: None
    elif case == "map-task-args":
        raw.dag._map_task_kwargs = {"map": True}
    elif case == "map-task-args-fn":
        raw.dag.add_map_task_kwargs_fn(lambda: {"map": True})
    elif case == "read-target":
        read_physical = _find_read_physical(raw, read)
        read_physical._output_block_size_option_override = OutputBlockSizeOption.of(
            target_max_block_size=34_567
        )
    elif case == "map-target":
        raw.dag._output_block_size_option_override = OutputBlockSizeOption.of(
            target_max_block_size=34_567
        )

    assert FuseCudfParquetReadIntoMapBatches().apply(raw) is raw


def test_skip_diagnostic_is_logged_once(
    ray_start_regular_shared_2_cpus, tmp_path, monkeypatch
):
    path = tmp_path / "input.parquet"
    _write_parquet(path)
    raw = _plan(_make_gpu_map(_make_read_files(path), fn=_AsyncIdentity))
    calls = []
    monkeypatch.setattr(
        cudf_parquet_read_fusion.logger,
        "debug",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    rule = FuseCudfParquetReadIntoMapBatches()
    assert rule.apply(raw) is raw
    rule.apply(raw)
    assert len(calls) == 1
    message = calls[0][0][0]
    assert "positive integer batch_size" in message
    assert "max_concurrency between 1 and 4" in message


def test_lineage_op_map_and_replanning_are_stable(
    ray_start_regular_shared_2_cpus, tmp_path
):
    path = tmp_path / "input.parquet"
    _write_parquet(path)
    read = _make_read_files(path)
    mapped = _make_gpu_map(read)
    original_list = read.input_dependencies[0]

    raw = _plan(mapped)
    raw_map_logical = raw.op_map[raw.dag]
    raw_read_physical = _find_read_physical(raw, read)
    raw_read_logical = raw.op_map[raw_read_physical]
    fused = FuseCudfParquetReadIntoMapBatches().apply(raw)
    actor = fused.dag
    synthetic = fused.op_map[actor]

    assert _is_direct_cudf_read(actor)
    assert actor._logical_operators == [raw_read_logical, raw_map_logical]
    assert synthetic.can_modify_num_rows is True
    assert synthetic.input_dependencies == list(raw_read_logical.input_dependencies)
    assert set(fused.op_map) == set(actor.post_order_iter())
    assert actor.input_dependency.output_dependencies == [actor]
    assert FuseCudfParquetReadIntoMapBatches().apply(fused) is fused

    # Physical planning and rewriting never mutate the caller's logical chain.
    second = FuseCudfParquetReadIntoMapBatches().apply(_plan(mapped))
    assert _is_direct_cudf_read(second.dag)
    assert mapped.input_dependencies == [read]
    assert read.input_dependencies == [original_list]


@pytest.mark.parametrize(
    ("case", "expected_fused"),
    [
        ("shared-list", 0),
        ("shared-read", 0),
        ("shared-map", 0),
        ("independent", 2),
    ],
)
def test_branch_topology_rewrite_safety(
    ray_start_regular_shared_2_cpus, tmp_path, case, expected_fused
):
    paths = [tmp_path / "left.parquet", tmp_path / "right.parquet"]
    for path in paths:
        _write_parquet(path)

    if case == "shared-list":
        left_read = _make_read_files(paths[0])
        right_read = replace(
            _make_read_files(paths[1]),
            input_dependencies=list(left_read.input_dependencies),
        )
        branches = [_make_gpu_map(left_read), _make_gpu_map(right_read)]
    elif case == "shared-read":
        read = _make_read_files(paths[0])
        branches = [_make_gpu_map(read), _make_gpu_map(read)]
    elif case == "shared-map":
        mapped = _make_gpu_map(_make_read_files(paths[0]))
        branches = [mapped, mapped]
    else:
        branches = [_make_gpu_map(_make_read_files(path)) for path in paths]

    result = FuseCudfParquetReadIntoMapBatches().apply(_plan(Union(branches)))
    actors = [
        operator
        for operator in result.dag.post_order_iter()
        if isinstance(operator, ActorPoolMapOperator)
    ]
    assert sum(_is_direct_cudf_read(actor) for actor in actors) == expected_fused
    assert set(result.op_map) == set(result.dag.post_order_iter())


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-v", __file__]))
