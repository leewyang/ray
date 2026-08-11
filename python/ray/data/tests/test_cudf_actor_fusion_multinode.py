"""Run fused cuDF map_batches actors on two local Ray nodes and real GPUs."""

import os

import pyarrow as pa
import pytest

import ray
from ray.data._internal.execution.operators.actor_pool_map_operator import (
    ActorPoolMapOperator,
)
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.logical.optimizers import get_execution_plan
from ray.cluster_utils import Cluster

cudf = pytest.importorskip("cudf")
cupy = pytest.importorskip("cupy")


def _get_visible_gpu_tokens():
    try:
        device_count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA device discovery failed: {exc}")

    if device_count < 2:
        pytest.skip("cuDF multi-node fusion requires two visible GPUs")

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        return tuple(str(index) for index in range(device_count))

    tokens = tuple(token.strip() for token in visible_devices.split(",") if token)
    if len(tokens) < 2:
        pytest.skip("cuDF multi-node fusion requires two visible GPU tokens")
    return tokens


@pytest.fixture
def two_node_cudf_fusion_cluster():
    gpu_tokens = _get_visible_gpu_tokens()[:2]
    ray.shutdown()
    cluster = Cluster()
    context = None
    try:
        for index, gpu_token in enumerate(gpu_tokens):
            # Give each local raylet one distinct physical GPU. Ray maps the
            # node's logical GPU 0 through this visibility token.
            node_args = {
                "node_name": f"cudf-fusion-gpu-{index}",
                "num_cpus": 2,
                "num_gpus": 1,
                "resources": {f"cudf_fusion_node_{index}": 1},
                "env_vars": {"CUDA_VISIBLE_DEVICES": gpu_token},
            }
            if index == 0:
                node_args["include_dashboard"] = False
            cluster.add_node(**node_args)

        cluster.wait_for_nodes()
        ray.init(address=cluster.address)

        context = ray.data.DataContext.get_current()
        previous_fusion = context.enable_cudf_actor_fusion
        previous_wait = context.wait_for_min_actors_s
        context.enable_cudf_actor_fusion = True
        context.wait_for_min_actors_s = 60
        yield gpu_tokens
    finally:
        if context is not None:
            context.enable_cudf_actor_fusion = previous_fusion
            context.wait_for_min_actors_s = previous_wait
        ray.shutdown()
        cluster.shutdown()


@pytest.mark.gpu
def test_cudf_actor_fusion_across_two_nodes(two_node_cudf_fusion_cluster):
    gpu_tokens = two_node_cudf_fusion_cluster

    @ray.remote(num_cpus=0)
    def make_arrow_block(start):
        return pa.table({"id": range(start, start + 8)})

    # Keep one input block on each node so actor-locality scheduling sends work
    # through both actors in the fixed pool.
    block_refs = [
        make_arrow_block.options(resources={f"cudf_fusion_node_{index}": 0.01}).remote(
            index * 8
        )
        for index in range(2)
    ]

    class FirstStage:
        def __call__(self, batch):
            assert isinstance(batch, cudf.DataFrame)
            runtime_context = ray.get_runtime_context()
            gpu_id = str(ray.get_gpu_ids()[0])

            batch["actor_id"] = str(runtime_context.get_actor_id())
            batch["node_id"] = str(runtime_context.get_node_id())
            batch["gpu_id"] = gpu_id
            batch["visible_gpu"] = os.environ["CUDA_VISIBLE_DEVICES"]
            batch["pci_bus_id"] = cupy.cuda.Device().pci_bus_id
            batch["first_frame_id"] = id(batch)
            batch["value"] = batch["id"] * 2
            return batch

    class FilterStage:
        def __call__(self, batch):
            runtime_context = ray.get_runtime_context()
            assert bool(
                (batch["actor_id"] == str(runtime_context.get_actor_id())).all()
            )
            assert bool((batch["node_id"] == str(runtime_context.get_node_id())).all())
            assert bool((batch["gpu_id"] == str(ray.get_gpu_ids()[0])).all())
            assert bool(
                (batch["visible_gpu"] == os.environ["CUDA_VISIBLE_DEVICES"]).all()
            )
            assert bool((batch["first_frame_id"] == id(batch)).all())

            filtered = batch[batch["id"] % 2 == 0]
            filtered["filtered_frame_id"] = id(filtered)
            filtered["value"] = filtered["value"] + 1
            return filtered

    class FinalStage:
        def __call__(self, batch):
            runtime_context = ray.get_runtime_context()
            assert bool(
                (batch["actor_id"] == str(runtime_context.get_actor_id())).all()
            )
            assert bool((batch["node_id"] == str(runtime_context.get_node_id())).all())
            assert bool((batch["gpu_id"] == str(ray.get_gpu_ids()[0])).all())
            assert bool(
                (batch["visible_gpu"] == os.environ["CUDA_VISIBLE_DEVICES"]).all()
            )
            assert bool((batch["filtered_frame_id"] == id(batch)).all())

            batch["result"] = batch["value"] + 10
            return batch

    map_kwargs = {
        "batch_format": "cudf",
        "batch_size": 8,
        "zero_copy_batch": True,
        "compute": ray.data.ActorPoolStrategy(size=2),
        "num_gpus": 1,
    }
    ds = (
        ray.data.from_arrow_refs(block_refs)
        .map_batches(FirstStage, **map_kwargs)
        .map_batches(FilterStage, **map_kwargs)
        .map_batches(FinalStage, **map_kwargs)
    )

    physical_plan, _ = get_execution_plan(ds._logical_plan)
    operators = []
    stack = [physical_plan.dag]
    while stack:
        operator = stack.pop()
        operators.append(operator)
        stack.extend(operator.input_dependencies)
    map_operators = [op for op in operators if isinstance(op, MapOperator)]
    assert len(map_operators) == 1
    assert isinstance(map_operators[0], ActorPoolMapOperator)
    assert map_operators[0].name == "MapBatches(FirstStage->FilterStage->FinalStage)"

    rows = sorted(ds.take_all(), key=lambda row: row["id"])
    assert [(row["id"], row["result"]) for row in rows] == [
        (value, value * 2 + 11) for value in range(0, 16, 2)
    ]
    assert {row["visible_gpu"] for row in rows} == set(gpu_tokens)
    assert len({row["pci_bus_id"] for row in rows}) == 2
    assert len({row["node_id"] for row in rows}) == 2
    assert len({row["actor_id"] for row in rows}) == 2
    assert len({(row["node_id"], row["actor_id"], row["gpu_id"]) for row in rows}) == 2


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
