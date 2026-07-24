"""Unit tests for ShuffleStrategy.GPU_SHUFFLE.

These tests do NOT require GPUs or the rapidsmpf/cudf/ucxx packages.
All Ray actor calls are mocked so the tests run on a standard CPU cluster.
"""

import time
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pytest

import ray
import ray.data._internal.gpu_shuffle.hash_shuffle as hash_shuffle
from ray.data import ActorPoolStrategy, ExecutionOptions
from ray.data._internal.execution.interfaces import (
    BlockEntry,
    ExecutionResources,
    PhysicalOperator,
    RefBundle,
)
from ray.data._internal.execution.operators.actor_pool_map_operator import (
    ActorPoolMapOperator,
)
from ray.data._internal.execution.resource_admission import ResourceAdmissionGrant
from ray.data._internal.gpu_shuffle.hash_shuffle import (
    GPURankPool,
    GPURankPoolState,
    GPUShuffleActor,
    GPUShuffleOperator,
    _derive_num_gpu_ranks,
)
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.logical.operators import Repartition
from ray.data._internal.logical.optimizers import get_execution_plan
from ray.data._internal.planner.plan_all_to_all_op import plan_all_to_all_op
from ray.data._internal.util import explain_plan
from ray.data.block import BlockMetadata
from ray.data.context import DataContext, ShuffleStrategy
from ray.data.tests.conftest import noop_counter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_input_op_mock(num_blocks=None, size_bytes=None):
    """Return a minimal PhysicalOperator mock compatible with HashShufflingOperatorBase."""
    logical_mock = MagicMock(LogicalOperator)
    logical_mock.infer_metadata.return_value = BlockMetadata(
        num_rows=None,
        size_bytes=size_bytes,
        exec_stats=None,
        input_files=None,
    )
    logical_mock.estimated_num_outputs.return_value = num_blocks

    op_mock = MagicMock(PhysicalOperator)
    op_mock._output_dependencies = []
    op_mock._logical_operators = [logical_mock]
    op_mock.num_output_splits.return_value = 1
    return op_mock


def _make_bundle(num_blocks: int = 1) -> RefBundle:
    """Return a RefBundle with *num_blocks* placeholder block refs."""
    meta = BlockMetadata(num_rows=10, size_bytes=100, exec_stats=None, input_files=None)
    blocks = [
        BlockEntry(ray.ObjectRef(bytes([i % 256]) * 28), meta)
        for i in range(num_blocks)
    ]
    return RefBundle(blocks, schema=None, owns_blocks=False)


def _make_data_context(
    *,
    gpu_shuffle_num_actors: int = 4,
    gpu_shuffle_rmm_pool_size="auto",
    gpu_shuffle_spill_memory_limit="auto",
) -> DataContext:
    ctx = DataContext()
    ctx._shuffle_strategy = ShuffleStrategy.GPU_SHUFFLE
    ctx.gpu_shuffle_num_actors = gpu_shuffle_num_actors
    ctx.gpu_shuffle_rmm_pool_size = gpu_shuffle_rmm_pool_size
    ctx.gpu_shuffle_spill_memory_limit = gpu_shuffle_spill_memory_limit
    return ctx


# ---------------------------------------------------------------------------
# Enum / DataContext field tests (no Ray required)
# ---------------------------------------------------------------------------


class TestDataContextGpuFields:
    def test_gpu_shuffle_default_values(self):
        ctx = DataContext()
        assert ctx.gpu_shuffle_num_actors is None
        assert ctx.gpu_shuffle_rmm_pool_size is None
        assert ctx.gpu_shuffle_spill_memory_limit == "auto"

    def test_gpu_shuffle_fields_settable(self):
        ctx = DataContext()
        ctx.shuffle_strategy = ShuffleStrategy.GPU_SHUFFLE
        ctx.gpu_shuffle_num_actors = 8
        ctx.gpu_shuffle_rmm_pool_size = 4 * 1024**3
        ctx.gpu_shuffle_spill_memory_limit = None
        assert ctx.shuffle_strategy == ShuffleStrategy.GPU_SHUFFLE
        assert ctx.gpu_shuffle_num_actors == 8
        assert ctx.gpu_shuffle_rmm_pool_size == 4 * 1024**3
        assert ctx.gpu_shuffle_spill_memory_limit is None


# ---------------------------------------------------------------------------
# Import isolation — gpu_shuffle.py must be importable without GPU packages
# ---------------------------------------------------------------------------


class TestImportIsolation:
    def test_module_importable_without_rapidsmpf(self):
        """The gpu_shuffle module must not import rapidsmpf at module level."""

        import ray.data._internal.gpu_shuffle.hash_shuffle as mod

        # If we got here the import succeeded on a CPU-only env.
        assert hasattr(mod, "GPUShuffleOperator")
        assert hasattr(mod, "GPURankPool")
        assert hasattr(mod, "GPUShuffleActor")

    def test_ray_data_importable_without_gpu_packages(self):
        import ray.data  # noqa: F401 — must not raise


# ---------------------------------------------------------------------------
# _derive_num_gpu_ranks
# ---------------------------------------------------------------------------


class TestDeriveNumGpuRanks:
    def test_explicit_count_used(self):
        ctx = DataContext()
        ctx.gpu_shuffle_num_actors = 7
        assert _derive_num_gpu_ranks(ctx) == 7

    def test_auto_detect_from_cluster(self):
        ctx = DataContext()
        ctx.gpu_shuffle_num_actors = None

        with patch(
            "ray.data._internal.gpu_shuffle.hash_shuffle._get_total_cluster_resources"
        ) as mock_res:
            mock_res.return_value = ExecutionResources(cpu=8, gpu=4)
            assert _derive_num_gpu_ranks(ctx) == 4

    def test_zero_gpus_raises(self):
        ctx = DataContext()
        ctx.gpu_shuffle_num_actors = None

        with patch(
            "ray.data._internal.gpu_shuffle.hash_shuffle._get_total_cluster_resources"
        ) as mock_res:
            mock_res.return_value = ExecutionResources(cpu=8, gpu=0)
            with pytest.raises(RuntimeError, match="GPU resources"):
                _derive_num_gpu_ranks(ctx)

    def test_fractional_gpu_count_truncated(self):
        """ExecutionResources.gpu may be fractional; int() truncates."""
        ctx = DataContext()
        ctx.gpu_shuffle_num_actors = None

        with patch(
            "ray.data._internal.gpu_shuffle.hash_shuffle._get_total_cluster_resources"
        ) as mock_res:
            mock_res.return_value = ExecutionResources(cpu=4, gpu=3.9)
            assert _derive_num_gpu_ranks(ctx) == 3


# ---------------------------------------------------------------------------
# GPURankPool
# ---------------------------------------------------------------------------


class TestGPURankPool:
    def _make_pool(self, nranks=4, total_nparts=8):
        return GPURankPool(
            nranks=nranks,
            total_nparts=total_nparts,
            setup_timeout_s=60.0,
            actor_cls_factory=lambda: hash_shuffle.GPUShuffleActor,
            actor_kwargs={
                "key_columns": ["user_id"],
                "columns": None,
                "rmm_pool_size": "auto",
                "spill_memory_limit": "auto",
            },
            log_label="GPUShufflePool",
        )

    def test_actors_empty_before_start(self):
        pool = self._make_pool()
        assert pool.actors == []

    def test_activate_uses_exact_bundle_indexes_and_refresh_waits_for_setup(self):
        pool = self._make_pool(nranks=3)
        pool._label_selector = {"accelerator": "h100"}
        fake_pg = MagicMock()
        pg_ready_ref = MagicMock(name="pg_ready")
        fake_pg.ready.return_value = pg_ready_ref
        rank_ref = MagicMock(name="rank")
        root_ref = MagicMock(name="root")
        worker_refs = [MagicMock(name=f"worker_{i}") for i in range(3)]
        actor_handles = [MagicMock(name=f"actor_{i}") for i in range(3)]
        actor_handles[0].setup_root.options.return_value.remote.return_value = (
            rank_ref,
            root_ref,
        )
        for actor, worker_ref in zip(actor_handles, worker_refs):
            actor.setup_worker.remote.return_value = worker_ref

        with (
            patch(
                "ray.data._internal.gpu_shuffle.hash_shuffle.placement_group",
                return_value=fake_pg,
            ) as mock_pg,
            patch(
                "ray.data._internal.gpu_shuffle.hash_shuffle.GPUShuffleActor"
            ) as mock_actor_cls,
            patch(
                "ray.data._internal.gpu_shuffle.hash_shuffle.ray.get"
            ) as mock_ray_get,
            patch(
                "ray.data._internal.gpu_shuffle.hash_shuffle.ray.wait"
            ) as mock_ray_wait,
        ):
            mock_actor_cls.options.return_value.remote.side_effect = actor_handles
            mock_ray_get.return_value = [None, None, None]
            mock_ray_wait.side_effect = [
                ([pg_ready_ref], []),
                ([], worker_refs),
                (worker_refs, []),
            ]

            pool.activate()
            mock_pg.assert_called_once_with(
                bundles=[{"CPU": 1.0, "GPU": 1.0}] * 3,
                strategy="SPREAD",
                bundle_label_selector=[{"accelerator": "h100"}] * 3,
            )
            assert pool.state is GPURankPoolState.RESERVING
            assert pool.pending_setup_refs == (pg_ready_ref,)
            pool.refresh()
            assert pool.state is GPURankPoolState.STARTING
            pool.refresh()

        assert len(pool.actors) == 3
        assert mock_actor_cls.options.call_count == 3
        assert pool.state is GPURankPoolState.READY
        strategies = [
            call.kwargs["scheduling_strategy"]
            for call in mock_actor_cls.options.call_args_list
        ]
        assert [strategy.placement_group for strategy in strategies] == [fake_pg] * 3
        assert [strategy.placement_group_bundle_index for strategy in strategies] == [
            0,
            1,
            2,
        ]
        for actor in actor_handles:
            actor.setup_worker.remote.assert_called_once_with(root_ref)

    def test_setup_timeout_starts_after_placement_group_is_ready(self):
        pool = self._make_pool(nranks=2)
        fake_pg = MagicMock()
        pg_ready_ref = MagicMock(name="pg_ready")
        fake_pg.ready.return_value = pg_ready_ref
        worker_refs = [MagicMock(name=f"worker_{rank}") for rank in range(2)]
        actors = [MagicMock(name=f"actor_{rank}") for rank in range(2)]
        actors[0].setup_root.options.return_value.remote.return_value = (
            MagicMock(name="rank"),
            MagicMock(name="root"),
        )

        with (
            patch(
                "ray.data._internal.gpu_shuffle.hash_shuffle.placement_group",
                return_value=fake_pg,
            ),
            patch(
                "ray.data._internal.gpu_shuffle.hash_shuffle.remove_placement_group"
            ) as mock_remove,
            patch(
                "ray.data._internal.gpu_shuffle.hash_shuffle.GPUShuffleActor"
            ) as mock_actor_cls,
            patch("ray.data._internal.gpu_shuffle.hash_shuffle.ray.wait") as mock_wait,
            patch("ray.data._internal.gpu_shuffle.hash_shuffle.ray.get"),
        ):
            mock_actor_cls.options.return_value.remote.side_effect = actors
            for actor, worker_ref in zip(actors, worker_refs):
                actor.setup_worker.remote.return_value = worker_ref
            pool.activate()
            mock_wait.return_value = ([], [pg_ready_ref])
            pool.refresh()
            assert pool._setup_start_time is None

            mock_wait.side_effect = [([pg_ready_ref], []), ([], worker_refs)]
            pool.refresh()
            assert pool._setup_start_time is not None
            pool._setup_start_time = hash_shuffle.time.perf_counter() - 61
            with pytest.raises(TimeoutError, match="UCXX setup"):
                pool.refresh()

        mock_remove.assert_called_once_with(fake_pg)
        assert pool.state is GPURankPoolState.FAILED

    def test_activate_and_wait_blocks_for_reservation_and_rank_setup(self):
        pool = self._make_pool(nranks=2)
        pg_ref = MagicMock(name="pg_ready")
        worker_refs = [MagicMock(name=f"worker_{rank}") for rank in range(2)]
        pool.activate = MagicMock(
            side_effect=lambda: (
                setattr(pool, "_state", GPURankPoolState.RESERVING),
                setattr(pool, "_pending_setup_refs", [pg_ref]),
            )
        )

        def refresh():
            if pool._state is GPURankPoolState.RESERVING:
                pool._state = GPURankPoolState.STARTING
                pool._setup_start_time = time.perf_counter()
                pool._pending_setup_refs = worker_refs
            else:
                pool._pending_setup_refs = []
                pool._state = GPURankPoolState.READY

        pool.refresh = MagicMock(side_effect=refresh)

        with (
            patch("ray.data._internal.gpu_shuffle.hash_shuffle.ray.get") as mock_get,
        ):
            pool.activate_and_wait()

        assert mock_get.call_count == 2
        assert mock_get.call_args_list[0].args == ([pg_ref],)
        assert mock_get.call_args_list[1].args == (worker_refs,)
        assert pool.refresh.call_count == 2
        assert pool.state is GPURankPoolState.READY

    def test_activate_and_wait_bounds_unmanaged_reservation(self):
        pool = self._make_pool(nranks=1)
        placement_group = MagicMock()
        pool._state = GPURankPoolState.RESERVING
        pool._placement_group = placement_group
        pool._pending_setup_refs = [MagicMock(name="pg_ready")]
        pool.activate = MagicMock()

        with (
            patch(
                "ray.data._internal.gpu_shuffle.hash_shuffle.ray.get",
                side_effect=ray.exceptions.GetTimeoutError("not ready"),
            ),
            patch(
                "ray.data._internal.gpu_shuffle.hash_shuffle.remove_placement_group"
            ) as mock_remove,
            pytest.raises(TimeoutError, match="activation timed out"),
        ):
            pool.activate_and_wait()

        mock_remove.assert_called_once_with(placement_group)
        assert pool.state is GPURankPoolState.FAILED

    def test_setup_failure_cleans_up_every_rank_and_placement_group(self):
        pool = self._make_pool(nranks=2)
        fake_pg = MagicMock()
        worker_refs = [MagicMock(name=f"worker_{rank}") for rank in range(2)]
        actors = [MagicMock(name=f"actor_{rank}") for rank in range(2)]
        failure = RuntimeError("rank setup failed")
        pool._placement_group = fake_pg
        pool._actors = actors
        pool._pending_setup_refs = worker_refs
        pool._setup_start_time = hash_shuffle.time.perf_counter()
        pool._state = GPURankPoolState.STARTING

        with (
            patch(
                "ray.data._internal.gpu_shuffle.hash_shuffle.ray.wait",
                return_value=(worker_refs, []),
            ),
            patch(
                "ray.data._internal.gpu_shuffle.hash_shuffle.ray.get",
                side_effect=failure,
            ),
            patch("ray.data._internal.gpu_shuffle.hash_shuffle.ray.kill") as mock_kill,
            patch(
                "ray.data._internal.gpu_shuffle.hash_shuffle.remove_placement_group"
            ) as mock_remove,
        ):
            with pytest.raises(RuntimeError, match="rank setup failed"):
                pool.refresh()

        mock_kill.assert_not_called()
        mock_remove.assert_called_once_with(fake_pg)
        assert pool.actors == []
        assert pool._placement_group is None
        assert pool.state is GPURankPoolState.FAILED

    def test_get_actor_for_block_round_robin(self):
        pool = self._make_pool(nranks=3)
        mock_actors = [MagicMock(name=f"actor_{i}") for i in range(3)]
        pool._actors = mock_actors
        pool._state = GPURankPoolState.READY

        # Blocks 0,1,2,3,4 should map to actors 0,1,2,0,1
        expected = [mock_actors[i % 3] for i in range(5)]
        actual = [pool.get_actor_for_block(i) for i in range(5)]
        assert actual == expected

    def test_shutdown_kills_actors(self):
        pool = self._make_pool(nranks=2)
        mock_actors = [MagicMock(), MagicMock()]
        pool._actors = mock_actors

        with patch("ray.data._internal.gpu_shuffle.hash_shuffle.ray.kill") as mock_kill:
            pool.shutdown()

        assert mock_kill.call_count == 2
        assert pool.actors == []
        assert pool.state is GPURankPoolState.CLOSED

    def test_shutdown_retries_transient_cleanup_failures(self):
        pool = self._make_pool(nranks=1)
        actor = MagicMock()
        placement_group = MagicMock()
        pool._actors = [actor]
        pool._placement_group = placement_group

        with (
            patch(
                "ray.data._internal.gpu_shuffle.hash_shuffle.ray.kill",
                side_effect=[RuntimeError("kill failed"), None],
            ),
            patch(
                "ray.data._internal.gpu_shuffle.hash_shuffle.remove_placement_group",
                side_effect=[RuntimeError("remove failed"), None],
            ),
        ):
            pool.shutdown()

        assert pool.actors == []
        assert pool._placement_group is None
        assert pool.state is GPURankPoolState.CLOSED

    def test_reactivation_finishes_partial_cleanup_before_reserving(self):
        pool = self._make_pool(nranks=1)
        old_placement_group = MagicMock()
        new_placement_group = MagicMock()
        new_placement_group.ready.return_value = MagicMock()
        pool._placement_group = old_placement_group
        pool._actors = [MagicMock()]
        pool._state = GPURankPoolState.READY

        with (
            patch(
                "ray.data._internal.gpu_shuffle.hash_shuffle.remove_placement_group",
                side_effect=[RuntimeError("remove failed"), None],
            ) as mock_remove,
            patch("ray.data._internal.gpu_shuffle.hash_shuffle.ray.kill"),
            patch(
                "ray.data._internal.gpu_shuffle.hash_shuffle.placement_group",
                return_value=new_placement_group,
            ),
        ):
            pool.deactivate()
            assert pool.state is GPURankPoolState.INACTIVE
            assert pool._placement_group is old_placement_group

            pool.activate()

        assert mock_remove.call_count == 2
        assert pool._placement_group is new_placement_group
        assert pool.state is GPURankPoolState.RESERVING


# ---------------------------------------------------------------------------
# GPUShuffleOperator constructor
# ---------------------------------------------------------------------------


class TestGPUShuffleOperatorConstructor:
    def _make_op(self, num_partitions=None, nranks=4, default_parallelism=200):
        ctx = _make_data_context(gpu_shuffle_num_actors=nranks)
        ctx.default_hash_shuffle_parallelism = default_parallelism
        input_op = _make_input_op_mock()

        return GPUShuffleOperator(
            input_op,
            ctx,
            key_columns=("user_id",),
            num_partitions=num_partitions,
        )

    def test_name_contains_key_columns(self):
        op = self._make_op(num_partitions=8)
        assert "user_id" in op.name

    def test_name_contains_num_partitions(self):
        op = self._make_op(num_partitions=8)
        assert "8" in op.name

    def test_num_partitions_clamped_to_nranks(self):
        """When requested partitions < nranks, partitions is raised to nranks."""
        op = self._make_op(num_partitions=2, nranks=8)
        assert op._num_partitions == 8

    def test_num_partitions_exceeds_nranks_unchanged(self):
        op = self._make_op(num_partitions=16, nranks=4)
        assert op._num_partitions == 16

    def test_num_partitions_defaults_to_context_parallelism(self):
        op = self._make_op(num_partitions=None, nranks=4, default_parallelism=200)
        assert op._num_partitions == 200

    def test_resource_requirements_are_complete_fixed_rank_gang(self):
        op = self._make_op(nranks=6, num_partitions=6)
        assert op.base_resource_usage == ExecutionResources(cpu=6, gpu=6)
        minimum, maximum = op.min_max_resource_requirements()
        assert minimum == ExecutionResources(cpu=6, gpu=6)
        assert maximum.cpu == maximum.gpu == 6
        assert maximum.memory == 0
        assert maximum.object_store_memory == float("inf")

    def test_current_logical_usage_is_zero_before_admission(self):
        op = self._make_op(nranks=5, num_partitions=5)
        assert op.current_logical_usage().gpu == 0

    def test_current_logical_usage_is_full_gang_while_reserved(self):
        op = self._make_op(nranks=4, num_partitions=4)
        op._rank_pool._placement_group = MagicMock()
        assert op.current_logical_usage().gpu == 4
        assert op.current_logical_usage().cpu == 4

    def test_current_logical_usage_zero_after_pool_shutdown(self):
        op = self._make_op(nranks=4, num_partitions=4)
        op._rank_pool._actors = [MagicMock() for _ in range(4)]
        op._rank_pool._state = GPURankPoolState.READY
        with patch("ray.data._internal.gpu_shuffle.hash_shuffle.ray.kill"):
            op._rank_pool.shutdown()
        assert op.current_logical_usage().gpu == 0

    def test_actor_info_reports_atomic_startup_as_pending(self):
        op = self._make_op(nranks=4, num_partitions=4)
        op._rank_pool._state = GPURankPoolState.RESERVING

        info = op.get_actor_info()
        assert (info.running, info.pending, info.active) == (0, 4, 0)

        op._rank_pool._actors = [MagicMock() for _ in range(4)]
        op._rank_pool._state = GPURankPoolState.READY
        info = op.get_actor_info()
        assert (info.running, info.pending, info.active) == (4, 0, 4)

    def test_resource_admission_spec_is_fixed_complete_gang(self):
        op = self._make_op(nranks=4, num_partitions=4)
        spec = op.resource_admission_spec()

        assert spec.max_units == 1
        assert spec.unit_resources is None
        assert spec.minimum_resources == ExecutionResources(cpu=4, gpu=4)

    def test_start_does_not_acquire_resources_without_grant(self):
        op = self._make_op()
        op.apply_resource_admission_grant(
            ResourceAdmissionGrant(max_units=0, may_submit=False)
        )
        with patch.object(op._rank_pool, "activate") as mock_activate:
            op.start(ExecutionOptions(), noop_counter())
        mock_activate.assert_not_called()

    def test_controller_fallback_without_grant_uses_legacy_activation(self):
        op = self._make_op()
        with patch.object(op._rank_pool, "activate_and_wait") as mock_activate_and_wait:
            op.start(ExecutionOptions(), noop_counter())
        mock_activate_and_wait.assert_called_once()

    def test_disabled_resource_admission_uses_legacy_activation(self):
        op = self._make_op()
        op.data_context._enable_resource_admission_control = False
        assert op.resource_admission_spec() is None
        with patch.object(op._rank_pool, "activate_and_wait") as mock_activate_and_wait:
            op.start(ExecutionOptions(), noop_counter())
        mock_activate_and_wait.assert_called_once()

    def test_grant_before_start_activates_only_after_start(self):
        op = self._make_op()
        with patch.object(op._rank_pool, "activate") as mock_activate:
            op.apply_resource_admission_grant(
                ResourceAdmissionGrant(max_units=1, may_submit=True)
            )
            mock_activate.assert_not_called()
            op.start(ExecutionOptions(), noop_counter())
        mock_activate.assert_called_once()

    def test_frontier_grant_acquires_gang_without_accepting_input(self):
        op = self._make_op()
        op.apply_resource_admission_grant(
            ResourceAdmissionGrant(max_units=0, may_submit=False)
        )
        op.start(ExecutionOptions(), noop_counter())

        with (
            patch.object(op._rank_pool, "activate") as mock_activate,
            patch.object(op._rank_pool, "refresh") as mock_refresh,
        ):
            op.apply_resource_admission_grant(
                ResourceAdmissionGrant(max_units=1, may_submit=False)
            )
            op._rank_pool._state = GPURankPoolState.RESERVING
            op.refresh_state()

            mock_activate.assert_called_once()
            mock_refresh.assert_called_once()
            assert not op.can_add_input()
            mock_refresh.assert_called_once()

    def test_zero_grant_refresh_retries_cleanup(self):
        op = self._make_op()
        with patch.object(op._rank_pool, "deactivate") as mock_deactivate:
            op.apply_resource_admission_grant(
                ResourceAdmissionGrant(max_units=0, may_submit=False)
            )
            mock_deactivate.reset_mock()

            op.refresh_state()

        mock_deactivate.assert_called_once()

    def test_refresh_propagates_setup_timeout_without_input(self):
        op = self._make_op()
        pool = op._rank_pool
        pool._state = GPURankPoolState.STARTING
        pool._setup_start_time = time.perf_counter() - pool._setup_timeout_s
        pool._placement_group = MagicMock()

        with (
            patch("ray.data._internal.gpu_shuffle.hash_shuffle.remove_placement_group"),
            pytest.raises(TimeoutError, match="UCXX setup"),
        ):
            op.refresh_state()

        assert op._next_block_idx == 0
        assert pool.state is GPURankPoolState.FAILED

    def test_incremental_resource_usage_reuses_rank_actor_resources(self):
        op = self._make_op()
        assert op.incremental_resource_usage() == ExecutionResources.zero()

    def test_progress_bar_names(self):
        op = self._make_op()
        names = op.get_sub_progress_bar_names()
        assert names == ["GPU Shuffle", "GPU Reduce"]

    def test_set_sub_progress_bar_shuffle(self):
        op = self._make_op()
        mock_bar = MagicMock()
        op.set_sub_progress_bar("GPU Shuffle", mock_bar)
        assert op._shuffle_bar is mock_bar

    def test_set_sub_progress_bar_reduce(self):
        op = self._make_op()
        mock_bar = MagicMock()
        op.set_sub_progress_bar("GPU Reduce", mock_bar)
        assert op._reduce_bar is mock_bar

    def test_initial_state(self):
        op = self._make_op()
        assert op._next_block_idx == 0
        assert op._insert_tasks == {}
        assert op._extraction_tasks == {}
        assert not op._finalization_started
        assert len(op._output_queue) == 0


# ---------------------------------------------------------------------------
# GPUShuffleOperator: _add_input_inner block routing
# ---------------------------------------------------------------------------


class TestGPUShuffleOperatorInputRouting:
    def _make_op_with_mock_pool(self, nranks=3, num_partitions=6):
        ctx = _make_data_context(gpu_shuffle_num_actors=nranks)
        input_op = _make_input_op_mock()
        op = GPUShuffleOperator(
            input_op, ctx, key_columns=("k",), num_partitions=num_partitions
        )
        # Replace the real pool with a mock
        mock_actors = [MagicMock(name=f"actor_{i}") for i in range(nranks)]
        for actor in mock_actors:
            actor.insert_batch.remote.return_value = MagicMock()
        op._rank_pool._actors = mock_actors
        op._rank_pool._nranks = nranks
        op._rank_pool._state = GPURankPoolState.READY
        return op, mock_actors

    def test_single_block_routed_to_first_actor(self):
        op, actors = self._make_op_with_mock_pool(nranks=3)
        bundle = _make_bundle(num_blocks=1)
        op._add_input_inner(bundle, input_index=0)

        actors[0].insert_batch.remote.assert_called_once()

    def test_round_robin_across_three_ranks(self):
        op, actors = self._make_op_with_mock_pool(nranks=3)

        # Submit 6 single-block bundles
        for _ in range(6):
            op._add_input_inner(_make_bundle(1), input_index=0)

        # Each actor should have received exactly 2 blocks
        for actor in actors:
            assert actor.insert_batch.remote.call_count == 2

    def test_block_idx_increments_per_block(self):
        op, actors = self._make_op_with_mock_pool(nranks=3)

        bundle_with_2 = _make_bundle(num_blocks=2)
        op._add_input_inner(bundle_with_2, input_index=0)

        assert op._next_block_idx == 2

    def test_insert_tasks_tracked(self):
        op, actors = self._make_op_with_mock_pool(nranks=2)
        op._add_input_inner(_make_bundle(1), 0)
        assert len(op._insert_tasks) == 1

    def test_insert_task_callback_removes_task(self):
        op, actors = self._make_op_with_mock_pool(nranks=2)
        op._add_input_inner(_make_bundle(1), 0)
        # Grab the callback and invoke it
        task = list(op._insert_tasks.values())[0]
        assert 0 in op._insert_tasks
        with patch(
            "ray.data._internal.gpu_shuffle.hash_shuffle.ray.get",
            return_value=1,
        ):
            task._task_done_callback()
        assert 0 not in op._insert_tasks

    def test_insert_failure_fails_and_cleans_up_gang(self):
        op, _ = self._make_op_with_mock_pool(nranks=2)
        op._rank_pool._placement_group = MagicMock()
        op._add_input_inner(_make_bundle(1), 0)
        task = op._insert_tasks[0]
        with patch(
            "ray.data._internal.gpu_shuffle.hash_shuffle.ray.get",
            side_effect=RuntimeError("insert failed"),
        ):
            task._task_done_callback()
        op._inputs_complete = True

        with (
            patch("ray.data._internal.gpu_shuffle.hash_shuffle.ray.kill"),
            patch("ray.data._internal.gpu_shuffle.hash_shuffle.remove_placement_group"),
            pytest.raises(RuntimeError, match="insert failed"),
        ):
            op._try_finalize()

        assert op._rank_pool.state is GPURankPoolState.FAILED
        assert op._rank_pool._placement_group is None
        assert op._insert_tasks == {}

    def test_idle_gang_can_be_revoked_before_first_input(self):
        op, _ = self._make_op_with_mock_pool(nranks=2)
        op._admission_grant = ResourceAdmissionGrant(max_units=1, may_submit=True)

        with patch.object(op._rank_pool, "deactivate") as mock_deactivate:
            op.apply_resource_admission_grant(
                ResourceAdmissionGrant(max_units=0, may_submit=False)
            )

        mock_deactivate.assert_called_once()
        assert op._admission_grant.max_units == 0

    def test_gang_cannot_release_after_first_input(self):
        op, _ = self._make_op_with_mock_pool(nranks=2)
        op._admission_grant = ResourceAdmissionGrant(max_units=1, may_submit=True)
        op._add_input_inner(_make_bundle(1), 0)

        assert not op.can_release_resource_admission()


# ---------------------------------------------------------------------------
# GPUShuffleOperator: finalization and completion
# ---------------------------------------------------------------------------


class TestGPUShuffleOperatorFinalization:
    def _make_op(self, nranks=2, num_partitions=4):
        ctx = _make_data_context(gpu_shuffle_num_actors=nranks)
        input_op = _make_input_op_mock()
        op = GPUShuffleOperator(
            input_op, ctx, key_columns=("k",), num_partitions=num_partitions
        )
        mock_actors = [MagicMock(name=f"actor_{i}") for i in range(nranks)]
        for actor in mock_actors:
            actor.finish_and_extract.options.return_value.remote.return_value = (
                MagicMock()
            )
        op._rank_pool._actors = mock_actors
        op._rank_pool._nranks = nranks
        op._rank_pool._state = GPURankPoolState.READY
        op._block_ref_counter = noop_counter()
        return op, mock_actors

    def test_finalization_not_started_until_inputs_complete(self):
        op, _ = self._make_op()
        op._inputs_complete = False
        op._try_finalize()
        assert not op._finalization_started

    def test_finalization_not_started_while_inserts_pending(self):
        op, _ = self._make_op()
        op._inputs_complete = True
        op._insert_tasks[0] = MagicMock()  # fake pending insert
        op._try_finalize()
        assert not op._finalization_started

    def test_finalization_starts_after_all_inserts_done(self):
        op, mock_actors = self._make_op(nranks=2)
        op._inputs_complete = True
        # No pending inserts

        with patch.object(op._reduce_metrics, "on_task_submitted"):
            op._try_finalize()

        assert op._finalization_started

    def test_finish_and_extract_called_on_all_ranks(self):
        op, mock_actors = self._make_op(nranks=2)
        op._inputs_complete = True

        with patch.object(op._reduce_metrics, "on_task_submitted"):
            op._try_finalize()

        for actor in mock_actors:
            actor.finish_and_extract.options.assert_called_once()

    def test_try_finalize_idempotent(self):
        op, mock_actors = self._make_op(nranks=2)
        op._inputs_complete = True

        with patch.object(op._reduce_metrics, "on_task_submitted"):
            op._try_finalize()
            op._try_finalize()  # second call should be no-op

        # finish_and_extract should only be called once per actor
        for actor in mock_actors:
            assert actor.finish_and_extract.options.call_count == 1

    def test_successful_extraction_releases_sticky_gang(self):
        op, _ = self._make_op(nranks=2)
        op._next_block_idx = 1
        op._inputs_complete = True

        with patch.object(op._reduce_metrics, "on_task_submitted"):
            op._try_finalize()

        callbacks = [
            task._task_done_callback for task in list(op._extraction_tasks.values())
        ]
        with patch("ray.data._internal.gpu_shuffle.hash_shuffle.ray.kill"):
            for callback in callbacks:
                callback(None, None, None)

        assert op.can_release_resource_admission()
        assert op._rank_pool.state is GPURankPoolState.CLOSED

    def test_empty_input_finalization_cannot_release_admission(self):
        op, _ = self._make_op(nranks=2)
        op._admission_grant = ResourceAdmissionGrant(max_units=1, may_submit=True)
        op._inputs_complete = True

        with patch.object(op._reduce_metrics, "on_task_submitted"):
            op._try_finalize()

        assert op._finalization_started
        assert op._next_block_idx == 0
        assert not op.can_release_resource_admission()

    def test_extraction_failure_fails_and_cleans_up_complete_gang(self):
        op, actors = self._make_op(nranks=2)
        fake_pg = MagicMock()
        op._rank_pool._placement_group = fake_pg
        op._next_block_idx = 1
        op._inputs_complete = True

        with patch.object(op._reduce_metrics, "on_task_submitted"):
            op._try_finalize()

        failure = RuntimeError("rank extraction failed")
        failed_callback = op._extraction_tasks[0]._task_done_callback
        with (
            patch("ray.data._internal.gpu_shuffle.hash_shuffle.ray.kill") as mock_kill,
            patch(
                "ray.data._internal.gpu_shuffle.hash_shuffle.remove_placement_group"
            ) as mock_remove,
        ):
            failed_callback(failure, None, None)

        mock_kill.assert_not_called()
        mock_remove.assert_called_once_with(fake_pg)
        assert op.can_release_resource_admission()
        assert op._rank_pool.actors == []
        assert op._rank_pool._placement_group is None
        assert op._rank_pool.state is GPURankPoolState.FAILED

    def test_failed_gang_is_not_releasable_while_cleanup_handles_remain(self):
        op, _ = self._make_op(nranks=2)
        op._rank_pool._state = GPURankPoolState.FAILED

        assert not op.can_release_resource_admission()
        op._rank_pool._actors.clear()
        assert op.can_release_resource_admission()

    def test_has_next_false_initially(self):
        op, _ = self._make_op()
        op._inputs_complete = False
        assert not op.has_next()

    def test_has_next_true_when_output_queued(self):
        op, _ = self._make_op()
        bundle = _make_bundle(1)
        op._output_queue.add(bundle, key=0)
        op._output_queue.finalize(key=0)
        op._finalization_started = True
        assert op.has_next()

    def test_get_next_inner_dequeues(self):
        op, _ = self._make_op()
        b1 = _make_bundle(1)
        b2 = _make_bundle(1)
        op._output_queue.add(b1, key=0)
        op._output_queue.finalize(key=0)
        op._output_queue.add(b2, key=1)
        op._output_queue.finalize(key=1)

        with (
            patch.object(op._reduce_metrics, "on_output_dequeued"),
            patch.object(op._reduce_metrics, "on_output_taken"),
        ):
            result = op._get_next_inner()

        assert result is b1
        assert op._output_queue.has_next()

    def test_has_completed_false_while_extracting(self):
        op, _ = self._make_op()
        op._finalization_started = True
        op._extraction_tasks[0] = MagicMock()  # still running
        assert not op.has_completed()

    def test_output_order_is_partition_order_regardless_of_arrival(self):
        """Bundles arriving out of order must be emitted in ascending partition order."""
        op, _ = self._make_op(nranks=2, num_partitions=4)
        op._finalization_started = True

        # Build 4 bundles, insert in reverse order (3, 2, 1, 0)
        bundles = {}
        for partition_id in reversed(range(4)):
            meta = BlockMetadata(
                num_rows=1, size_bytes=8, exec_stats=None, input_files=None
            )
            bundle = RefBundle(
                [BlockEntry(ray.ObjectRef(bytes([partition_id]) * 28), meta)],
                schema=None,
                owns_blocks=False,
            )
            bundles[partition_id] = bundle
            op._output_queue.add(bundle, key=partition_id)
            op._output_queue.finalize(key=partition_id)

        with (
            patch.object(op._reduce_metrics, "on_output_dequeued"),
            patch.object(op._reduce_metrics, "on_output_taken"),
        ):
            results = [op._get_next_inner() for _ in range(4)]

        # Output must be in partition order 0, 1, 2, 3 — not insertion order 3, 2, 1, 0
        assert results == [bundles[i] for i in range(4)]

    def test_get_active_tasks_combines_both_phases(self):
        op, _ = self._make_op()
        insert_task = MagicMock()
        extract_task = MagicMock()
        op._insert_tasks[0] = insert_task
        op._extraction_tasks[0] = extract_task

        active = op.get_active_tasks()
        assert insert_task in active
        assert extract_task in active
        assert len(active) == 2

    def test_pending_rank_setup_refs_are_active_tasks(self):
        op, _ = self._make_op()
        setup_refs = [MagicMock(name=f"setup_{rank}") for rank in range(2)]
        op._rank_pool._pending_setup_refs = setup_refs

        assert [task.get_waitable() for task in op.get_active_tasks()] == setup_refs

    def test_shutdown_clears_tasks_and_kills_actors(self):
        op, mock_actors = self._make_op(nranks=2)
        op._insert_tasks[0] = MagicMock()
        op._extraction_tasks[0] = MagicMock()
        expected_kill_count = len(mock_actors)

        with (
            patch("ray.data._internal.gpu_shuffle.hash_shuffle.ray.kill") as mock_kill,
            patch.object(PhysicalOperator, "_do_shutdown", return_value=None),
        ):
            op._do_shutdown(force=True)

        assert op._insert_tasks == {}
        assert op._extraction_tasks == {}
        assert mock_kill.call_count == expected_kill_count


# ---------------------------------------------------------------------------
# plan_all_to_all_op routing
# ---------------------------------------------------------------------------


class TestPlanAllToAllOpRouting:
    """Verify that plan_all_to_all_op routes GPU_SHUFFLE to GPUShuffleOperator."""

    def _make_repartition_op(self, keys=("user_id",), num_outputs=8):
        return Repartition(
            num_outputs=num_outputs,
            input_dependencies=[MagicMock(LogicalOperator)],
            shuffle=True,
            keys=list(keys),
        )

    def test_gpu_shuffle_routes_to_gpu_operator(self):
        ctx = DataContext()
        ctx.gpu_shuffle_num_actors = 4
        ctx._shuffle_strategy = ShuffleStrategy.GPU_SHUFFLE

        logical_op = self._make_repartition_op(keys=["user_id"], num_outputs=8)
        input_physical_op = _make_input_op_mock()

        op = plan_all_to_all_op(logical_op, [input_physical_op], ctx)

        assert isinstance(op, GPUShuffleOperator)

    def test_hash_shuffle_still_routes_to_hash_shufflle_v1(self, restore_data_context):
        from ray.data._internal.execution.operators.hash_shuffle import (
            HashShuffleOperator,
        )

        ctx = DataContext()
        ctx._shuffle_strategy = ShuffleStrategy.HASH_SHUFFLE

        with patch(
            "ray.data._internal.execution.operators.hash_shuffle"
            "._get_total_cluster_resources",
            return_value=ExecutionResources(cpu=4, gpu=0),
        ):
            logical_op = self._make_repartition_op(keys=["user_id"], num_outputs=8)
            input_physical_op = _make_input_op_mock()
            op = plan_all_to_all_op(logical_op, [input_physical_op], ctx)

        assert isinstance(op, HashShuffleOperator)

    def test_hash_shuffle_still_routes_to_hash_shufflle_v2(self, restore_data_context):
        """V2 hash shuffle is a two-op DAG; planner returns the ShuffleReduceOp
        with the ShuffleMapOp as its upstream input dependency."""
        from ray.data._internal.execution.operators.shuffle_operators.shuffle_map_operator import (  # noqa: E501
            ShuffleMapOp,
        )
        from ray.data._internal.execution.operators.shuffle_operators.shuffle_reduce_operator import (  # noqa: E501
            ShuffleReduceOp,
        )

        ctx = DataContext()
        ctx.use_hash_shuffle_v2 = True
        ctx._shuffle_strategy = ShuffleStrategy.HASH_SHUFFLE

        logical_op = self._make_repartition_op(keys=["user_id"], num_outputs=8)
        input_physical_op = _make_input_op_mock()
        op = plan_all_to_all_op(logical_op, [input_physical_op], ctx)

        assert isinstance(op, ShuffleReduceOp)
        assert isinstance(op.input_dependencies[0], ShuffleMapOp)

    def test_unsupported_strategy_with_keys_raises(self):
        ctx = DataContext()
        ctx._shuffle_strategy = ShuffleStrategy.SORT_SHUFFLE_PULL_BASED

        logical_op = self._make_repartition_op(keys=["user_id"], num_outputs=8)
        input_physical_op = _make_input_op_mock()

        with pytest.raises(ValueError, match="HASH_SHUFFLE"):
            plan_all_to_all_op(logical_op, [input_physical_op], ctx)

    def test_gpu_shuffle_respects_num_outputs(self):
        ctx = DataContext()
        ctx.gpu_shuffle_num_actors = 4
        ctx._shuffle_strategy = ShuffleStrategy.GPU_SHUFFLE

        logical_op = self._make_repartition_op(keys=["id"], num_outputs=16)
        input_physical_op = _make_input_op_mock()
        op = plan_all_to_all_op(logical_op, [input_physical_op], ctx)

        assert op._num_partitions == 16

    def test_gpu_shuffle_key_columns_normalised(self):
        """Key columns from SortKey.get_columns() should propagate correctly."""
        ctx = DataContext()
        ctx.gpu_shuffle_num_actors = 4
        ctx._shuffle_strategy = ShuffleStrategy.GPU_SHUFFLE

        logical_op = self._make_repartition_op(keys=["col_a", "col_b"], num_outputs=8)
        input_physical_op = _make_input_op_mock()
        op = plan_all_to_all_op(logical_op, [input_physical_op], ctx)

        assert "col_a" in op._key_columns
        assert "col_b" in op._key_columns


# ---------------------------------------------------------------------------
# GPUShuffleActor: deferred import guard
# ---------------------------------------------------------------------------


class TestGPUShuffleActorImportGuard:
    """GPUShuffleActor.__init__ must raise ImportError with a helpful message
    when rapidsmpf is not installed, not a generic ModuleNotFoundError."""

    def test_missing_rapidsmpf_raises_import_error(self):
        from ray.data._internal.gpu_shuffle.hash_shuffle import GPUShuffleActor

        # Access the underlying class (bypass Ray actor wrapper)
        ActorClass = GPUShuffleActor.__ray_actor_class__

        with patch.dict(
            "sys.modules",
            {"ray.data._internal.gpu_shuffle.rapidsmpf_backend": None},
        ):
            with pytest.raises(ImportError, match="rapidsmpf"):
                ActorClass(
                    nranks=2,
                    total_nparts=4,
                    key_columns=["k"],
                )


# ---------------------------------------------------------------------------
# GPU fixtures — shared by all real-GPU test classes below
# ---------------------------------------------------------------------------


def _num_cluster_gpus() -> int:
    """Return the number of GPUs in the Ray cluster (0 if Ray not initialised)."""
    if not ray.is_initialized():
        return 0
    return int(ray.cluster_resources().get("GPU", 0))


@pytest.fixture(scope="module")
def ray_with_gpu():
    """Skip the test if GPU packages or GPU hardware are absent.

    Imports ``cudf``, ``rapidsmpf``, and ``ucxx`` with
    ``pytest.importorskip`` so the skip message clearly names the missing
    package.  Also initialises Ray (if not already running) and checks that
    at least one GPU is visible in the cluster.
    """
    pytest.importorskip("cudf", reason="cudf (GPU DataFrame library) not installed")
    pytest.importorskip("rapidsmpf", reason="rapidsmpf not installed")
    pytest.importorskip("ucxx", reason="ucxx not installed")
    cupy = pytest.importorskip("cupy", reason="cupy not installed")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA GPU is visible")
    except Exception as exc:
        pytest.skip(f"CUDA driver is unavailable: {exc}")

    if not ray.is_initialized():
        ray.init()

    num_gpus = _num_cluster_gpus()
    if num_gpus < 1:
        pytest.skip("No GPU resources found in the Ray cluster")

    yield num_gpus


# ---------------------------------------------------------------------------
# GPUShuffleActor — real GPU paths (conditional)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestGPUShuffleActorReal:
    """Exercises GPU actor methods on actual hardware.

    All tests are skipped automatically when GPU packages or GPU resources are
    absent.  Run them explicitly with ``pytest -m gpu``.
    """

    def _make_setup_actor(self, total_nparts: int = 2, key_columns=None):
        """Create, UCXX-initialise, and return a single-rank GPUShuffleActor."""
        key_columns = key_columns or ["id"]
        actor = GPUShuffleActor.options(num_gpus=1).remote(
            nranks=1,
            total_nparts=total_nparts,
            key_columns=key_columns,
        )
        _, root_address = ray.get(actor.setup_root.remote())
        ray.get(actor.setup_worker.remote(root_address))
        return actor

    def test_actor_init_succeeds(self, ray_with_gpu):
        """GPUShuffleActor.__init__ succeeds (rapidsmpf import guard passes)."""
        actor = GPUShuffleActor.options(num_gpus=1).remote(
            nranks=1,
            total_nparts=2,
            key_columns=["id"],
        )
        ray.kill(actor)

    def test_setup_root_returns_rank_and_address(self, ray_with_gpu):
        """setup_root() returns the rank and address needed for UCXX setup."""
        actor = GPUShuffleActor.options(num_gpus=1).remote(
            nranks=1, total_nparts=1, key_columns=["k"]
        )
        rank, address = ray.get(actor.setup_root.remote())
        assert isinstance(rank, int)
        assert isinstance(address, bytes)
        assert len(address) > 0
        ray.kill(actor)

    def test_insert_batch(self, ray_with_gpu):
        actor = self._make_setup_actor()
        table = pa.table({"id": [1, 2, 3], "val": [0.1, 0.2, 0.3]})
        assert ray.get(actor.insert_batch.remote(table)) == table.num_rows
        ray.kill(actor)

    def test_insert_batch_large_table(self, ray_with_gpu):
        """insert_batch handles a larger Arrow Table without error."""

        n = 5_000
        actor = self._make_setup_actor(total_nparts=4)
        table = pa.table(
            {
                "id": pa.array(np.arange(n, dtype=np.int64)),
                "val": pa.array(np.random.rand(n)),
            }
        )
        assert ray.get(actor.insert_batch.remote(table)) == table.num_rows
        ray.kill(actor)

    def test_insert_batch_multiple_batches(self, ray_with_gpu):
        """Multiple insert_batch calls are accepted."""
        actor = self._make_setup_actor(total_nparts=2)
        sizes = [3, 7, 5]
        offset = 0
        for size in sizes:
            table = pa.table(
                {
                    "id": list(range(offset, offset + size)),
                    "label": ["x"] * size,
                }
            )
            assert ray.get(actor.insert_batch.remote(table)) == table.num_rows
            offset += size
        ray.kill(actor)

    def test_finish_and_extract_succeeds_after_inserts(self, ray_with_gpu):
        """finish_and_extract() completes without error after a batch insert."""
        actor = self._make_setup_actor()
        table = pa.table({"id": [0, 1, 2], "v": [10, 20, 30]})
        ray.get(actor.insert_batch.remote(table))
        gen = actor.finish_and_extract.options(num_returns="streaming").remote()
        # Drain the generator to ensure it completes.
        for ref in gen:
            ray.get(ref)
        ray.kill(actor)


# ---------------------------------------------------------------------------
# Single-rank end-to-end roundtrip (conditional)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestGPUSingleRankRoundtrip:
    """Full insert → finish_and_extract roundtrip (1 GPU)."""

    @staticmethod
    def _collect_partitions(actor) -> List[pa.Table]:
        """Drain a streaming finish_and_extract generator.

        finish_and_extract follows the Ray Data streaming protocol: each
        partition yields a block (pa.Table) followed by a BlockMetadataWithSchema.
        Collect only the blocks.
        """

        gen = actor.finish_and_extract.options(num_returns="streaming").remote()
        return [
            item for ref in gen for item in [ray.get(ref)] if isinstance(item, pa.Table)
        ]

    def _actor_with_data(
        self,
        table: pa.Table,
        key_columns: List[str],
        total_nparts: int = 2,
    ):
        """Create a single-rank actor, feed *table* into it, return ready actor."""
        actor = GPUShuffleActor.options(num_gpus=1).remote(
            nranks=1,
            total_nparts=total_nparts,
            key_columns=key_columns,
        )
        _, root_address = ray.get(actor.setup_root.remote())
        ray.get(actor.setup_worker.remote(root_address))
        assert ray.get(actor.insert_batch.remote(table)) == table.num_rows
        return actor

    def test_roundtrip_preserves_row_count(self, ray_with_gpu):
        """All inserted rows appear in the extracted partitions."""
        n_rows = 30
        table = pa.table(
            {
                "key": list(range(n_rows)),
                "data": [float(i) for i in range(n_rows)],
            }
        )
        actor = self._actor_with_data(table, ["key"], total_nparts=3)
        partitions = self._collect_partitions(actor)
        assert sum(t.num_rows for t in partitions) == n_rows
        ray.kill(actor)

    def test_roundtrip_output_is_arrow_tables(self, ray_with_gpu):
        """finish_and_extract yields pyarrow.Table objects."""
        table = pa.table({"id": [1, 2, 3, 4], "name": ["a", "b", "c", "d"]})
        actor = self._actor_with_data(table, ["id"], total_nparts=2)
        partitions = self._collect_partitions(actor)
        for part in partitions:
            assert isinstance(part, pa.Table)
        ray.kill(actor)

    def test_roundtrip_multiple_batches_no_rows_lost(self, ray_with_gpu):
        """Rows from multiple insert_batch calls are all recovered."""
        actor = GPUShuffleActor.options(num_gpus=1).remote(
            nranks=1, total_nparts=2, key_columns=["k"]
        )
        _, root_address = ray.get(actor.setup_root.remote())
        ray.get(actor.setup_worker.remote(root_address))

        batch_sizes = [4, 6, 10]
        offset = 0
        for size in batch_sizes:
            table = pa.table({"k": list(range(offset, offset + size)), "v": [0] * size})
            assert ray.get(actor.insert_batch.remote(table)) == table.num_rows
            offset += size

        partitions = self._collect_partitions(actor)
        assert sum(t.num_rows for t in partitions) == sum(batch_sizes)
        ray.kill(actor)

    def test_roundtrip_column_names_preserved(self, ray_with_gpu):
        """Column names in extracted partitions match the inserted schema."""
        col_names = ["alpha", "beta", "gamma"]
        table = pa.table({"alpha": [1, 2], "beta": [3.0, 4.0], "gamma": ["x", "y"]})
        actor = self._actor_with_data(table, ["alpha"], total_nparts=1)
        partitions = self._collect_partitions(actor)
        for part in partitions:
            if part.num_rows > 0:
                for name in col_names:
                    assert name in part.schema.names
        ray.kill(actor)

    def test_roundtrip_key_column_hash_partitions_consistently(self, ray_with_gpu):
        """Each key value is always routed to exactly one partition."""
        # Hash partitioning guarantees that all rows sharing a key land in the
        # same partition, but makes no promise that *distinct* keys go to
        # *distinct* partitions (collisions are possible, especially with few
        # partitions).  Test the actual guarantee: no key is split across
        # multiple partitions.
        n_rows, n_keys = 100, 10
        table = pa.table(
            {
                "group": [i % n_keys for i in range(n_rows)],
                "val": list(range(n_rows)),
            }
        )
        actor = self._actor_with_data(table, ["group"], total_nparts=2)
        all_partitions = self._collect_partitions(actor)
        ray.kill(actor)

        # For each key, collect the set of partition indices it appears in.
        key_to_part_indices: dict = {}
        for idx, part in enumerate(all_partitions):
            for key in part.column("group").unique().to_pylist():
                key_to_part_indices.setdefault(key, set()).add(idx)

        for key, part_indices in key_to_part_indices.items():
            assert (
                len(part_indices) == 1
            ), f"Key {key!r} was split across partitions {part_indices}"


# ---------------------------------------------------------------------------
# GPURankPool — real GPU lifecycle (conditional)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestGPURankPoolReal:
    """Tests that exercise GPURankPool with actual GPU actors."""

    def _make_pool(self, nranks: int = 1, total_nparts: int = 2) -> GPURankPool:
        return GPURankPool(
            nranks=nranks,
            total_nparts=total_nparts,
            setup_timeout_s=60.0,
            actor_cls_factory=lambda: hash_shuffle.GPUShuffleActor,
            actor_kwargs={
                "key_columns": ["id"],
                "columns": None,
                "rmm_pool_size": "auto",
                "spill_memory_limit": "auto",
            },
            log_label="GPUShufflePool",
        )

    @staticmethod
    def _activate_and_wait(pool: GPURankPool) -> None:
        pool.activate()
        while pool.state is not GPURankPoolState.READY:
            pool.refresh()
            time.sleep(0.01)

    def test_pool_shutdown_clears_actors(self, ray_with_gpu):
        """GPURankPool.shutdown() kills actors and empties the list."""
        pool = self._make_pool(nranks=1)
        self._activate_and_wait(pool)
        pool.shutdown()
        assert pool.actors == []

    def test_pool_actors_respond_after_activation(self, ray_with_gpu):
        pool = self._make_pool(nranks=1, total_nparts=1)
        self._activate_and_wait(pool)
        actor = pool.actors[0]
        table = pa.table({"id": [1], "v": [2]})
        ray.get(actor.insert_batch.remote(table))
        pool.shutdown()


# ---------------------------------------------------------------------------
# GPU Hash Shuffle - end to end
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestGPUHashShuffle:
    def test_hash_shuffle(self, ray_with_gpu):
        """Test that hash shuffle works end to end."""
        # ray.init(num_gpus=1)
        num_gpus = ray_with_gpu
        ray.data.context.DataContext.get_current().shuffle_strategy = (
            ShuffleStrategy.GPU_SHUFFLE
        )

        num_rows = 10000
        parallelism = 1000
        num_blocks = int(parallelism / 10)

        ds = ray.data.range(num_rows, parallelism=parallelism).materialize()
        ds = ds.repartition(keys=["id"], num_blocks=num_blocks)
        assert "GPUShuffle" in explain_plan(ds._logical_plan)
        ds = ds.materialize()
        assert ds.num_blocks() == max(num_blocks, num_gpus)
        assert ds.count() == num_rows

    def test_one_gpu_actor_shuffle_map_groups_handoff(self, ray_with_gpu):
        class AddOne:
            def __call__(self, batch):
                result = batch.copy(deep=True)
                result["value"] = result["value"] + 1
                return result

        class SumGroup:
            def __call__(self, batch):
                result = batch.iloc[:1].copy(deep=True)
                result["value"] = batch["value"].sum()
                return result[["key", "value"]]

        class AddHundred:
            def __call__(self, batch):
                result = batch.copy(deep=True)
                result["value"] = result["value"] + 100
                return result

        ctx = DataContext.get_current()
        old_strategy = ctx.shuffle_strategy
        old_rank_count = ctx.gpu_shuffle_num_actors
        old_limits = ctx.execution_options.resource_limits
        ctx.shuffle_strategy = ShuffleStrategy.GPU_SHUFFLE
        ctx.gpu_shuffle_num_actors = 1
        ctx.execution_options.resource_limits = ExecutionResources.for_limits(gpu=1)

        try:
            ds = (
                ray.data.from_items(
                    [
                        {"key": 0, "value": 1},
                        {"key": 0, "value": 2},
                        {"key": 0, "value": 3},
                        {"key": 1, "value": 4},
                        {"key": 1, "value": 5},
                        {"key": 1, "value": 6},
                    ],
                    override_num_blocks=2,
                )
                .map_batches(
                    AddOne,
                    batch_size=1024,
                    batch_format="cudf",
                    compute=ActorPoolStrategy(size=1),
                    num_cpus=0,
                    num_gpus=1,
                )
                .groupby("key", num_partitions=1)
                .map_groups(
                    SumGroup,
                    batch_format="cudf",
                    compute=ActorPoolStrategy(size=1),
                    num_cpus=0,
                    num_gpus=1,
                )
                .map_batches(
                    AddHundred,
                    batch_size=1024,
                    batch_format="cudf",
                    compute=ActorPoolStrategy(size=1),
                    num_cpus=0,
                    num_gpus=1,
                )
            )

            physical_plan, _ = get_execution_plan(ds._logical_plan)
            stack = [physical_plan.dag]
            actor_ops = []
            shuffle_ops = []
            while stack:
                op = stack.pop()
                if isinstance(op, ActorPoolMapOperator):
                    actor_ops.append(op)
                if isinstance(op, GPUShuffleOperator):
                    shuffle_ops.append(op)
                stack.extend(op.input_dependencies)

            assert len(actor_ops) == 3, physical_plan.dag.dag_str
            assert len(shuffle_ops) == 1, physical_plan.dag.dag_str
            assert all(
                op.resource_admission_spec().unit_resources is not None
                for op in actor_ops
            )
            assert shuffle_ops[0].resource_admission_spec().unit_resources is None

            assert sorted(ds.take_all(), key=lambda row: row["key"]) == [
                {"key": 0, "value": 109},
                {"key": 1, "value": 118},
            ]

            total_gpus = ray.cluster_resources()["GPU"]
            for _ in range(100):
                if ray.available_resources().get("GPU", 0) >= total_gpus:
                    break
                time.sleep(0.05)
            assert ray.available_resources().get("GPU", 0) >= total_gpus
        finally:
            ctx.shuffle_strategy = old_strategy
            ctx.gpu_shuffle_num_actors = old_rank_count
            ctx.execution_options.resource_limits = old_limits

    def test_one_gpu_actor_shuffle_map_groups_handoff_with_spilling(
        self, shutdown_only, tmp_path
    ):
        pytest.importorskip("cudf")
        pytest.importorskip("rapidsmpf")
        pytest.importorskip("ucxx")
        cupy = pytest.importorskip("cupy")
        try:
            if cupy.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("No CUDA GPU is visible")
        except Exception as exc:
            pytest.skip(f"CUDA driver is unavailable: {exc}")

        class AddKey:
            def __call__(self, batch):
                result = batch.copy(deep=True)
                result["key"] = result["id"] % 64
                return result

        class SumGroup:
            def __call__(self, batch):
                result = batch.iloc[:1].copy(deep=True)
                result["id"] = batch["id"].sum()
                return result[["key", "id"]]

        class Identity:
            def __call__(self, batch):
                return batch

        ray.shutdown()
        ray.init(
            num_cpus=4,
            num_gpus=1,
            object_store_memory=100e6,
            object_spilling_directory=str(tmp_path),
            include_dashboard=False,
        )
        ctx = DataContext.get_current()
        ctx.shuffle_strategy = ShuffleStrategy.GPU_SHUFFLE
        ctx.gpu_shuffle_num_actors = 1
        ctx.execution_options.resource_limits = ExecutionResources.for_limits(gpu=1)
        ctx.enable_get_object_locations_for_metrics = True

        num_rows = 16_000_000
        ds = (
            ray.data.range(num_rows, override_num_blocks=8)
            .map_batches(
                AddKey,
                batch_size=1_000_000,
                batch_format="cudf",
                compute=ActorPoolStrategy(size=1),
                num_cpus=0,
                num_gpus=1,
            )
            .groupby("key", num_partitions=1)
            .map_groups(
                SumGroup,
                batch_format="cudf",
                compute=ActorPoolStrategy(size=1),
                num_cpus=0,
                num_gpus=1,
            )
            .map_batches(
                Identity,
                batch_size=64,
                batch_format="cudf",
                compute=ActorPoolStrategy(size=1),
                num_cpus=0,
                num_gpus=1,
            )
        )

        materialized = ds.materialize()
        assert materialized.count() == 64
        assert materialized.get_stats_summary().global_bytes_spilled > 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
