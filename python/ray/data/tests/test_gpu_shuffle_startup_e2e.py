import copy
import os
import threading
import time
from pathlib import Path

import pyarrow as pa
import pytest

import ray
from ray._private.accelerators import NvidiaGPUAcceleratorManager
from ray.data import ActorPoolStrategy
from ray.data._internal.execution.block_ref_counter import BlockRefCounter
from ray.data._internal.execution.gpu_shuffle_startup_policy import (
    build_gpu_shuffle_segment_topologies,
)
from ray.data._internal.execution.streaming_executor_state import (
    Topology,
    build_streaming_topology,
)
from ray.data._internal.logical.optimizers import get_execution_plan
from ray.data.context import DataContext, ShuffleStrategy

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.timeout(300),
]

_GPU_SHUFFLE_RANKS = 4
_DEFAULT_OBJECT_STORE_BYTES = 256 * 1024**2
_LOW_OBJECT_STORE_BYTES = 90_000_000
_EXECUTION_TIMEOUT_S = 240


def _identity_batch(batch: pa.Table) -> pa.Table:
    return batch


def _make_dedup_bands(batch):
    import cudf

    normalized = batch["text"].str.strip().str.lower()
    base = cudf.DataFrame({"doc_id": batch["doc_id"], "band_key": normalized})
    bands = []
    for band_id in range(2):
        band = base.copy(deep=True)
        band["band_id"] = band_id
        bands.append(band[["doc_id", "band_id", "band_key"]])
    return cudf.concat(bands, ignore_index=True)


def _create_collision_edges(batch):
    ids = batch[["doc_id"]].drop_duplicates().sort_values("doc_id")
    edges = ids.rename(columns={"doc_id": "dst"})
    edges["src"] = edges["dst"].min()
    return edges[["src", "dst"]]


def _local_distinct_edges(batch):
    return batch.drop_duplicates(subset=["src", "dst"]).reset_index(drop=True)


def _global_distinct_edge(batch):
    return batch[["src", "dst"]].head(1).reset_index(drop=True)


class _IdentityActor:
    def __call__(self, batch: pa.Table) -> pa.Table:
        return batch


def _widen_batch(batch: pa.Table) -> pa.Table:
    ids = batch.column("id").combine_chunks().to_numpy(zero_copy_only=False)
    columns = {"id": pa.array(ids.copy())}
    for index in range(8):
        columns[f"value_{index}"] = pa.array(ids.copy() + index)
    return pa.table(columns)


@pytest.fixture
def gpu_cluster(request):
    required_gpus, object_store_memory = request.param
    pytest.importorskip("cudf", reason="cudf is required for GPU shuffle")
    pytest.importorskip("rapidsmpf", reason="rapidsmpf is required for GPU shuffle")
    pytest.importorskip("ucxx", reason="ucxx is required for GPU shuffle")

    visible_gpus = NvidiaGPUAcceleratorManager.get_current_node_num_accelerators()
    if visible_gpus < required_gpus:
        pytest.skip(
            f"test requires {required_gpus} GPUs, but only {visible_gpus} are visible"
        )

    ray.shutdown()
    ray_tmpdir = Path(
        os.environ.get("RAY_GPU_ACCEPTANCE_TMPDIR", "/dev/shm/ray-admission")
    )
    ray_tmpdir.mkdir(parents=True, exist_ok=True)
    previous_ray_tmpdir = os.environ.get("RAY_TMPDIR")
    os.environ["RAY_TMPDIR"] = str(ray_tmpdir)

    ray.init(
        num_cpus=max(8, required_gpus * 2),
        num_gpus=required_gpus,
        object_store_memory=object_store_memory,
        object_spilling_directory=str(ray_tmpdir),
        _temp_dir=str(ray_tmpdir),
        include_dashboard=False,
    )
    try:
        assert int(ray.cluster_resources().get("GPU", 0)) == required_gpus
        yield required_gpus
    finally:
        ray.shutdown()
        if previous_ray_tmpdir is None:
            os.environ.pop("RAY_TMPDIR", None)
        else:
            os.environ["RAY_TMPDIR"] = previous_ray_tmpdir


@pytest.fixture
def isolated_data_context():
    context = DataContext.get_current()
    original = copy.deepcopy(context)
    try:
        yield context
    finally:
        DataContext._set_current(original)


def _configure_gpu_shuffle(context: DataContext) -> None:
    context.shuffle_strategy = ShuffleStrategy.GPU_SHUFFLE
    context.gpu_shuffle_num_actors = _GPU_SHUFFLE_RANKS
    context.gpu_shuffle_setup_timeout_s = 60
    context.wait_for_min_actors_s = -1


def _startup_segment_topologies(dataset) -> list[Topology]:
    physical_plan, _ = get_execution_plan(dataset._logical_plan)
    context = DataContext.get_current()
    topology = build_streaming_topology(
        physical_plan.dag,
        context.execution_options,
        BlockRefCounter(add_object_out_of_scope_callback=lambda *_: True),
        start_operators=False,
    )
    return build_gpu_shuffle_segment_topologies(topology, context.execution_options)


def _materialize_with_timeout(dataset):
    result = {}
    errors = []

    def materialize() -> None:
        try:
            result["dataset"] = dataset.materialize()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=materialize, daemon=True)
    thread.start()
    deadline = time.monotonic() + _EXECUTION_TIMEOUT_S
    while thread.is_alive() and time.monotonic() < deadline:
        thread.join(timeout=0.05)

    if thread.is_alive():
        ray.shutdown()
        thread.join(timeout=10)
        pytest.fail(
            f"GPU shuffle execution did not finish within {_EXECUTION_TIMEOUT_S}s"
        )
    if errors:
        raise errors[0]
    return result["dataset"]


def _assert_exact_rows_and_partitions(dataset, num_rows: int) -> None:
    assert dataset.num_blocks() == _GPU_SHUFFLE_RANKS
    assert sorted(row["id"] for row in dataset.take_all()) == list(range(num_rows))


def _wait_for_gpu_release(expected_gpus: int, timeout_s: float = 30) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if ray.available_resources().get("GPU", 0) >= expected_gpus:
            return
        time.sleep(0.05)
    pytest.fail(
        f"GPU resources were not released: available={ray.available_resources()}"
    )


@pytest.mark.parametrize(
    "gpu_cluster",
    [pytest.param((4, _DEFAULT_OBJECT_STORE_BYTES), id="4gpu")],
    indirect=True,
)
def test_task_map_to_full_rank_shuffle_without_intermediate_materialize(
    gpu_cluster, isolated_data_context
):
    _configure_gpu_shuffle(isolated_data_context)
    num_rows = 256

    # Intentionally construct one lazy Dataset: there is no materialize between
    # the GPU task map and the full-rank shuffle.
    dataset = (
        ray.data.range(num_rows, override_num_blocks=8)
        .map_batches(
            _identity_batch,
            batch_size=32,
            batch_format="pyarrow",
            num_gpus=1,
        )
        .repartition(keys=["id"], num_blocks=_GPU_SHUFFLE_RANKS)
    )
    assert len(_startup_segment_topologies(dataset)) == 2

    result = _materialize_with_timeout(dataset)

    _assert_exact_rows_and_partitions(result, num_rows)
    _wait_for_gpu_release(gpu_cluster)


@pytest.mark.parametrize(
    "gpu_cluster",
    [pytest.param((4, _DEFAULT_OBJECT_STORE_BYTES), id="4gpu")],
    indirect=True,
)
def test_two_cudf_dedup_phases_without_materialize(gpu_cluster, isolated_data_context):
    _configure_gpu_shuffle(isolated_data_context)
    records = []
    for doc_id in range(256):
        cluster = doc_id // 4
        canonical = f"duplicate cluster {cluster:03d}"
        text = f" {canonical.upper()} " if doc_id % 2 else canonical
        records.append({"doc_id": doc_id, "text": text})

    dataset = (
        ray.data.from_items(records, override_num_blocks=8)
        .map_batches(
            _make_dedup_bands,
            batch_size=32,
            batch_format="cudf",
            num_gpus=1,
        )
        .groupby(
            ["band_id", "band_key"],
            num_partitions=_GPU_SHUFFLE_RANKS,
        )
        .map_groups(_create_collision_edges, batch_format="cudf", num_gpus=1)
        .map_batches(
            _local_distinct_edges,
            batch_size=32,
            batch_format="cudf",
            num_gpus=1,
        )
        .groupby(["src", "dst"], num_partitions=_GPU_SHUFFLE_RANKS)
        .map_groups(_global_distinct_edge, batch_format="cudf", num_gpus=1)
    )
    assert len(_startup_segment_topologies(dataset)) == 3

    result = _materialize_with_timeout(dataset)

    actual = {(row["src"], row["dst"]) for row in result.take_all()}
    expected = {(4 * (doc_id // 4), doc_id) for doc_id in range(256)}
    assert actual == expected
    assert result.count() == 256
    _wait_for_gpu_release(gpu_cluster)


@pytest.mark.parametrize(
    "gpu_cluster",
    [pytest.param((4, _DEFAULT_OBJECT_STORE_BYTES), id="4gpu")],
    indirect=True,
)
def test_fixed_gpu_actors_around_full_rank_shuffle_complete_and_release(
    gpu_cluster, isolated_data_context
):
    _configure_gpu_shuffle(isolated_data_context)
    num_rows = 256

    dataset = (
        ray.data.range(num_rows, override_num_blocks=8)
        .map_batches(
            _IdentityActor,
            compute=ActorPoolStrategy(size=1),
            batch_size=32,
            batch_format="pyarrow",
            num_gpus=1,
        )
        .repartition(keys=["id"], num_blocks=_GPU_SHUFFLE_RANKS)
        .map_batches(
            _IdentityActor,
            compute=ActorPoolStrategy(size=1),
            batch_size=32,
            batch_format="pyarrow",
            num_gpus=1,
        )
    )
    assert len(_startup_segment_topologies(dataset)) == 2

    result = _materialize_with_timeout(dataset)

    _assert_exact_rows_and_partitions(result, num_rows)
    _wait_for_gpu_release(gpu_cluster)


@pytest.mark.parametrize(
    "gpu_cluster",
    [pytest.param((4, _LOW_OBJECT_STORE_BYTES), id="4gpu-low-object-store")],
    indirect=True,
)
def test_low_object_store_prefix_spills_and_completes(
    gpu_cluster, isolated_data_context
):
    _configure_gpu_shuffle(isolated_data_context)
    num_rows = 2_000_000

    dataset = (
        ray.data.range(num_rows, override_num_blocks=16)
        .map_batches(
            _widen_batch,
            batch_size=125_000,
            batch_format="pyarrow",
            zero_copy_batch=True,
            num_gpus=1,
        )
        .repartition(keys=["id"], num_blocks=_GPU_SHUFFLE_RANKS)
    )
    assert len(_startup_segment_topologies(dataset)) == 2

    result = _materialize_with_timeout(dataset)

    assert result.num_blocks() == _GPU_SHUFFLE_RANKS
    assert result.count() == num_rows
    stats = result._get_stats_summary()
    assert stats.global_bytes_spilled > 0
    assert stats.global_bytes_restored > 0
    _wait_for_gpu_release(gpu_cluster)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
