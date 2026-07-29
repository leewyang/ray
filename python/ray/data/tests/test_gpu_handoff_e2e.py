"""GPU handoff acceptance tests with tight spill and output budgets."""

import os
import shutil
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

import ray
import ray.data._internal.execution.streaming_executor as streaming_executor
import ray.data._internal.gpu_shuffle.hash_shuffle as hash_shuffle
from ray.data._internal.compute import ActorPoolStrategy
from ray.data._internal.execution.interfaces import ExecutionResources
from ray.data._internal.execution.operators.actor_pool_map_operator import (
    ActorPoolMapOperator,
)
from ray.data._internal.execution.operators.task_pool_map_operator import (
    TaskPoolMapOperator,
)
from ray.data._internal.gpu_shuffle.hash_shuffle import (
    GPURankPool,
    GPUShuffleOperator,
)
from ray.data.context import DataContext, ShuffleStrategy


_NUM_ROWS = 16_000_000
_NUM_BLOCKS = 8
_NUM_KEYS = 2_000_000


class _GPUAddKey:
    def __call__(self, batch):
        result = batch.copy(deep=True)
        result["key"] = result["id"] % 64
        return result


class _GPUAddOne:
    def __call__(self, batch):
        result = batch.copy(deep=True)
        result["id"] = result["id"] + 1
        return result


class _GPUAddTwo:
    def __call__(self, batch):
        result = batch.copy(deep=True)
        result["id"] = result["id"] + 2
        return result


def _cpu_identity(batch):
    return batch


def _cpu_add_key(batch):
    result = batch.copy(deep=True)
    result["key"] = result["id"] % _NUM_KEYS
    return result


def _require_gpu_stack():
    pytest.importorskip("cudf")
    pytest.importorskip("rapidsmpf")
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA GPU is visible")
    except Exception as exc:
        pytest.skip(f"CUDA driver is unavailable: {exc}")


def _start_tight_cluster(allocator_enabled):
    base = Path(os.environ.get("RAY_GPU_ACCEPTANCE_TMPDIR", "/dev/shm/ray-admission"))
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / f"run-{uuid.uuid4().hex[:8]}"
    spill_dir = run_dir / "spill"
    spill_dir.mkdir(parents=True)
    ray.init(
        num_cpus=4,
        num_gpus=1,
        object_store_memory=100e6,
        object_spilling_directory=str(spill_dir),
        include_dashboard=False,
        _temp_dir=str(run_dir / "ray"),
    )
    ctx = DataContext.get_current()
    ctx._enable_gpu_handoff = True
    ctx.op_resource_reservation_enabled = allocator_enabled
    ctx.wait_for_min_actors_s = 0
    ctx.shuffle_strategy = ShuffleStrategy.GPU_SHUFFLE
    ctx.gpu_shuffle_num_actors = 1
    ctx.enable_get_object_locations_for_metrics = True
    ctx.execution_options.resource_limits = ExecutionResources.for_limits(
        cpu=4,
        gpu=1,
        object_store_memory=20e6,
    )
    return run_dir


def _assert_materialized_and_spilled(
    ds, expected_rows=_NUM_ROWS, *, sum_column="id", expected_sum=None
):
    materialized = ds.materialize()
    assert materialized.count() == expected_rows
    assert materialized.get_stats_summary().global_bytes_spilled > 0
    if expected_sum is not None:
        actual_sum = sum(
            int(batch[sum_column].sum())
            for batch in materialized.iter_batches(batch_format="numpy")
        )
        assert actual_sum == expected_sum
    return materialized


def _capture_topologies():
    topologies = []
    build = streaming_executor.build_streaming_topology

    def capture(*args, **kwargs):
        topology = build(*args, **kwargs)
        topologies.append(topology)
        return topology

    return topologies, capture


@pytest.mark.gpu
@pytest.mark.timeout(240)
@pytest.mark.parametrize("allocator_enabled", [True, False])
def test_actor_to_gpu_shuffle_handoff_spills(
    shutdown_only, restore_data_context, allocator_enabled
):
    _require_gpu_stack()
    run_dir = _start_tight_cluster(allocator_enabled)
    observed = []
    topologies, capture_topology = _capture_topologies()
    create_placement_group = hash_shuffle.placement_group

    def checked_create(*args, **kwargs):
        topology = topologies[-1]
        op = next(op for op in topology if isinstance(op, GPUShuffleOperator))
        predecessor = op.input_dependencies[0]
        predecessor_state = topology[predecessor]
        observed.append(
            (
                predecessor.has_execution_finished(),
                predecessor._actor_pool.current_size(),
                predecessor_state.output_queue.num_blocks,
                topology[op].input_queues[0] is predecessor_state.output_queue,
                op._rank_pool.owns_resources,
            )
        )
        return create_placement_group(*args, **kwargs)

    try:
        with (
            patch.object(
                streaming_executor, "build_streaming_topology", capture_topology
            ),
            patch.object(hash_shuffle, "placement_group", checked_create),
        ):
            ds = (
                ray.data.range(_NUM_ROWS, override_num_blocks=_NUM_BLOCKS)
                .map_batches(
                    _GPUAddKey,
                    batch_size=_NUM_ROWS // _NUM_BLOCKS,
                    batch_format="cudf",
                    compute=ActorPoolStrategy(size=1),
                    num_cpus=0,
                    num_gpus=1,
                    scheduling_strategy="SPREAD",
                )
                .repartition(keys=["key"], num_blocks=1)
            )
            _assert_materialized_and_spilled(
                ds, expected_sum=_NUM_ROWS * (_NUM_ROWS - 1) // 2
            )
        assert len(observed) == 1
        finished, pool_size, queued_blocks, queue_is_wired, successor_owned = observed[0]
        assert finished and pool_size == 0
        assert queued_blocks > 0 and queue_is_wired
        assert not successor_owned
    finally:
        ray.shutdown()
        shutil.rmtree(run_dir, ignore_errors=True)


@pytest.mark.gpu
@pytest.mark.timeout(240)
@pytest.mark.parametrize("allocator_enabled", [True, False])
def test_cpu_bridge_finishes_before_gpu_shuffle_creation(
    shutdown_only, restore_data_context, allocator_enabled
):
    _require_gpu_stack()
    run_dir = _start_tight_cluster(allocator_enabled)
    observed = []
    topologies, capture_topology = _capture_topologies()
    create_placement_group = hash_shuffle.placement_group

    def checked_create(*args, **kwargs):
        topology = topologies[-1]
        operators = list(topology)
        successor_index = next(
            i for i, op in enumerate(operators) if isinstance(op, GPUShuffleOperator)
        )
        predecessor_index = max(
            i
            for i, op in enumerate(operators[:successor_index])
            if isinstance(op, ActorPoolMapOperator)
        )
        segment = operators[predecessor_index + 1 : successor_index]
        assert len(segment) == 2
        assert all(isinstance(op, TaskPoolMapOperator) for op in segment)
        final_cpu_op = segment[-1]
        final_state = topology[final_cpu_op]
        successor = operators[successor_index]
        observed.append(
            (
                all(op.has_execution_finished() for op in segment),
                final_cpu_op.has_execution_finished(),
                final_cpu_op.has_completed(),
                final_state.output_queue.num_blocks,
                topology[successor].input_queues[0] is final_state.output_queue,
                successor._rank_pool.owns_resources,
            )
        )
        return create_placement_group(*args, **kwargs)

    try:
        with (
            patch.object(
                streaming_executor, "build_streaming_topology", capture_topology
            ),
            patch.object(hash_shuffle, "placement_group", checked_create),
        ):
            ds = (
                ray.data.range(_NUM_ROWS, override_num_blocks=_NUM_BLOCKS)
                .map_batches(
                    _GPUAddKey,
                    batch_size=_NUM_ROWS // _NUM_BLOCKS,
                    batch_format="cudf",
                    compute=ActorPoolStrategy(size=1),
                    num_cpus=0,
                    num_gpus=1,
                    scheduling_strategy="SPREAD",
                )
                .map_batches(
                    _cpu_identity,
                    batch_size=_NUM_ROWS // _NUM_BLOCKS,
                    batch_format="pyarrow",
                    num_cpus=1,
                )
                .map_batches(
                    _cpu_identity,
                    batch_size=_NUM_ROWS // _NUM_BLOCKS,
                    batch_format="pyarrow",
                    num_cpus=2,
                )
                .repartition(keys=["key"], num_blocks=1)
            )
            _assert_materialized_and_spilled(
                ds, expected_sum=_NUM_ROWS * (_NUM_ROWS - 1) // 2
            )

        assert len(observed) == 1
        (
            segment_finished,
            final_finished,
            _,
            queued_blocks,
            queue_is_wired,
            owned_before_creation,
        ) = observed[0]
        assert segment_finished and final_finished
        assert queued_blocks > 0 and queue_is_wired
        assert not owned_before_creation
    finally:
        ray.shutdown()
        shutil.rmtree(run_dir, ignore_errors=True)
@pytest.mark.gpu
@pytest.mark.timeout(240)
@pytest.mark.parametrize("allocator_enabled", [True, False])
def test_actor_to_actor_handoff_spills(
    shutdown_only, restore_data_context, allocator_enabled
):
    _require_gpu_stack()
    run_dir = _start_tight_cluster(allocator_enabled)
    observed = []
    original_start_actor = ActorPoolMapOperator._start_actor

    def checked_start_actor(op, *args, **kwargs):
        predecessor = op.input_dependencies[0]
        if isinstance(predecessor, ActorPoolMapOperator):
            observed.append(
                (
                    predecessor.has_execution_finished(),
                    predecessor._actor_pool.current_size(),
                )
            )
        return original_start_actor(op, *args, **kwargs)

    try:
        with patch.object(ActorPoolMapOperator, "_start_actor", checked_start_actor):
            ds = (
                ray.data.range(_NUM_ROWS, override_num_blocks=_NUM_BLOCKS)
                .map_batches(
                    _GPUAddOne,
                    batch_size=_NUM_ROWS // _NUM_BLOCKS,
                    batch_format="cudf",
                    compute=ActorPoolStrategy(size=1),
                    num_cpus=0,
                    num_gpus=1,
                    scheduling_strategy="SPREAD",
                )
                .map_batches(
                    _GPUAddTwo,
                    batch_size=_NUM_ROWS // _NUM_BLOCKS,
                    batch_format="cudf",
                    compute=ActorPoolStrategy(size=1),
                    num_cpus=0,
                    num_gpus=1,
                    scheduling_strategy="DEFAULT",
                )
            )
            _assert_materialized_and_spilled(
                ds,
                expected_sum=_NUM_ROWS * (_NUM_ROWS - 1) // 2 + 3 * _NUM_ROWS,
            )
        assert observed == [(True, 0)]
    finally:
        ray.shutdown()
        shutil.rmtree(run_dir, ignore_errors=True)


@pytest.mark.gpu
@pytest.mark.timeout(240)
@pytest.mark.parametrize("allocator_enabled", [True, False])
def test_rank_gang_to_rank_gang_handoff_spills(
    shutdown_only, restore_data_context, allocator_enabled
):
    _require_gpu_stack()
    run_dir = _start_tight_cluster(allocator_enabled)
    activated = []
    active_pool = []
    topologies, capture_topology = _capture_topologies()
    original_activate = GPURankPool.activate
    create_placement_group = hash_shuffle.placement_group

    def checked_activate(pool):
        active_pool.append(pool)
        try:
            result = original_activate(pool)
        finally:
            active_pool.pop()
        activated.append(pool)
        return result

    def checked_create(*args, **kwargs):
        topology = topologies[-1]
        owners = [op for op in topology if isinstance(op, GPUShuffleOperator)]
        assert len(owners) == 2
        indices = [list(topology).index(op) for op in owners]
        assert indices[1] == indices[0] + 1
        pool = active_pool[-1]
        owner_index = next(i for i, op in enumerate(owners) if op._rank_pool is pool)
        if owner_index:
            predecessor = owners[owner_index - 1]
            predecessor_state = topology[predecessor]
            assert predecessor.has_execution_finished()
            assert not predecessor._rank_pool.owns_resources
            assert predecessor_state.output_queue.num_blocks > 0
            assert (
                topology[owners[owner_index]].input_queues[0]
                is predecessor_state.output_queue
            )
        return create_placement_group(*args, **kwargs)

    try:
        with (
            patch.object(
                streaming_executor, "build_streaming_topology", capture_topology
            ),
            patch.object(GPURankPool, "activate", checked_activate),
            patch.object(hash_shuffle, "placement_group", checked_create),
        ):
            ds = (
                ray.data.range(_NUM_ROWS, override_num_blocks=_NUM_BLOCKS)
                .map_batches(
                    _cpu_add_key,
                    batch_size=_NUM_ROWS // _NUM_BLOCKS,
                    batch_format="pandas",
                    num_cpus=1,
                )
                .groupby("key", num_partitions=1)
                .count()
                .repartition(keys=["key"], num_blocks=1)
            )
            _assert_materialized_and_spilled(
                ds,
                expected_rows=_NUM_KEYS,
                sum_column="count()",
                expected_sum=_NUM_ROWS,
            )
        assert len(activated) == 2
    finally:
        ray.shutdown()
        shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
