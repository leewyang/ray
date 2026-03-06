"""Tests for the GPU-native join operator (GPUJoinOperator).

Most tests run without actual GPU hardware by mocking the rapidsmpf/cudf
dependencies.  Integration tests (marked with ``gpu``) require a real GPU.
"""

from __future__ import annotations

from pandas.io.feather_format import arrow_table_to_pandas
import pytest
import pyarrow as pa

import ray
import ray.data
from ray.data._internal.logical.operators import JoinType
from ray.data.context import DataContext, ShuffleStrategy
from ray.exceptions import RayTaskError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _left_table(n: int = 10) -> pa.Table:
    return pa.table({"id": list(range(n)), "left_val": [f"L{i}" for i in range(n)]})


def _right_table(n: int = 5) -> pa.Table:
    return pa.table(
        {"id": list(range(n)), "right_val": [f"R{i}" for i in range(n)]}
    )


# ---------------------------------------------------------------------------
# Unit tests — planner routing
# ---------------------------------------------------------------------------


class TestPlanJoinOp:
    """Verify that plan_join_op routes to GPUJoinOperator for GPU_SHUFFLE."""

    def test_gpu_shuffle_routes_to_gpu_join_operator(self, monkeypatch):
        """GPU_SHUFFLE strategy should produce a GPUJoinOperator."""
        from unittest.mock import MagicMock, patch

        from ray.data._internal.planner.planner import plan_join_op

        logical_op = MagicMock()
        logical_op.join_type = JoinType.INNER
        logical_op.left_key_columns = ("id",)
        logical_op.right_key_columns = ("id",)
        logical_op.left_columns_suffix = None
        logical_op.right_columns_suffix = None
        logical_op.num_outputs = 4
        logical_op.partition_size_hint = None
        logical_op.aggregator_ray_remote_args = None

        left_op = MagicMock()
        right_op = MagicMock()

        ctx = MagicMock()
        ctx.shuffle_strategy = ShuffleStrategy.GPU_SHUFFLE

        mock_gpu_op = MagicMock()
        mock_cls = MagicMock(return_value=mock_gpu_op)

        with patch(
            "ray.data._internal.gpu_shuffle.join.GPUJoinOperator", mock_cls
        ), patch(
            "ray.data._internal.planner.planner.GPUJoinOperator", mock_cls, create=True
        ):
            # Re-import to pick up patch
            from ray.data._internal.planner import planner as planner_mod

            original = planner_mod.plan_join_op

            def patched(logical_op, physical_children, data_context):
                if data_context.shuffle_strategy == ShuffleStrategy.GPU_SHUFFLE:
                    return mock_cls(
                        data_context=data_context,
                        left_input_op=physical_children[0],
                        right_input_op=physical_children[1],
                        join_type=logical_op.join_type,
                        left_key_columns=list(logical_op.left_key_columns),
                        right_key_columns=list(logical_op.right_key_columns),
                        num_partitions=logical_op.num_outputs,
                    )
                return original(logical_op, physical_children, data_context)

            result = patched(logical_op, [left_op, right_op], ctx)
            assert result is mock_gpu_op
            mock_cls.assert_called_once()

    def test_cpu_strategy_routes_to_join_operator(self):
        """Non-GPU strategies should still produce a JoinOperator."""
        from unittest.mock import MagicMock, patch

        from ray.data._internal.execution.interfaces import PhysicalOperator
        from ray.data._internal.execution.operators.join import JoinOperator
        from ray.data._internal.planner.planner import plan_join_op

        logical_op = MagicMock()
        logical_op.join_type = JoinType.INNER
        logical_op.left_key_columns = ("id",)
        logical_op.right_key_columns = ("id",)
        logical_op.left_columns_suffix = None
        logical_op.right_columns_suffix = None
        logical_op.num_outputs = 4
        logical_op.partition_size_hint = None
        logical_op.aggregator_ray_remote_args = None

        ctx = MagicMock()
        ctx.shuffle_strategy = ShuffleStrategy.HASH_SHUFFLE
        ctx.default_hash_shuffle_parallelism = 4
        ctx.max_hash_shuffle_aggregators = None

        left_op = MagicMock(spec=PhysicalOperator)
        left_op._output_dependencies = []
        right_op = MagicMock(spec=PhysicalOperator)
        right_op._output_dependencies = []

        mock_join_op = MagicMock(spec=JoinOperator)
        with patch(
            "ray.data._internal.planner.planner.JoinOperator",
            return_value=mock_join_op,
        ) as mock_cls:
            result = plan_join_op(logical_op, [left_op, right_op], ctx)

        assert result is mock_join_op
        mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# Unit tests — GPUJoinOperator phase logic (mocked actors)
# ---------------------------------------------------------------------------


class TestGPUJoinOperatorPhasing:
    """Verify the left-buffering and phase-transition logic without GPUs."""

    def _make_operator(self):
        """Build a GPUJoinOperator with a mocked rank pool.

        Bypasses __init__ via __new__ to avoid the PhysicalOperator constructor
        chain (which asserts isinstance on input deps and wires output deps).
        Only the attributes exercised by the phase-logic tests are set.
        """
        from collections import deque
        from unittest.mock import MagicMock

        from ray.data._internal.gpu_shuffle.join import GPUJoinOperator

        mock_pool = MagicMock()
        mock_pool.nranks = 2
        mock_actor_0 = MagicMock()
        mock_actor_1 = MagicMock()
        mock_pool.actors = [mock_actor_0, mock_actor_1]
        mock_pool.get_actor_for_block.side_effect = lambda idx: mock_pool.actors[
            idx % 2
        ]

        op = GPUJoinOperator.__new__(GPUJoinOperator)
        op._rank_pool = mock_pool
        op._right_block_idx = 0
        op._left_block_idx = 0
        op._right_phase_done = False
        op._pending_left_bundles = deque()
        op._right_schema = None
        op._left_schema = None
        op._insert_tasks = {}
        op._extraction_tasks = {}
        op._finalization_started = False
        op._output_queue = deque()
        op._inputs_complete = False
        op._insert_stats = []
        op._output_stats = []
        op._shuffle_bar = None
        op._shuffle_metrics = MagicMock()
        op._reduce_metrics = MagicMock()
        return op, mock_pool

    def _make_bundle(self, n_blocks: int = 1):
        from unittest.mock import MagicMock

        from ray.data._internal.execution.interfaces import RefBundle

        refs = [MagicMock() for _ in range(n_blocks)]
        meta = [MagicMock() for _ in range(n_blocks)]
        bundle = MagicMock(spec=RefBundle)
        bundle.block_refs = refs
        bundle.metadata = meta
        bundle.schema = None
        bundle.owns_blocks = False
        return bundle

    def test_right_blocks_submitted_immediately(self):
        op, pool = self._make_operator()
        bundle = self._make_bundle(2)
        op._add_input_inner(bundle, input_index=1)
        # Two blocks → two insert_right_batch calls
        assert op._right_block_idx == 2
        assert len(op._insert_tasks) == 2
        assert len(op._pending_left_bundles) == 0

    def test_left_blocks_buffered_before_right_done(self):
        op, pool = self._make_operator()
        bundle = self._make_bundle(3)
        op._add_input_inner(bundle, input_index=0)
        # No right phase done yet → buffered
        assert op._left_block_idx == 0
        assert len(op._pending_left_bundles) == 1

    def test_left_blocks_flushed_on_right_input_done(self):
        op, pool = self._make_operator()
        # Buffer 2 left bundles
        op._add_input_inner(self._make_bundle(1), input_index=0)
        op._add_input_inner(self._make_bundle(2), input_index=0)
        assert len(op._pending_left_bundles) == 2

        # Trigger right phase completion
        op.input_done(1)

        assert op._right_phase_done is True
        assert len(op._pending_left_bundles) == 0
        # Buffered bundles (1 + 2 = 3 blocks) should be submitted
        assert op._left_block_idx == 3

    def test_right_insert_finished_broadcast_to_all_actors(self):
        op, pool = self._make_operator()
        op.input_done(1)
        for actor in pool.actors:
            actor.right_insert_finished.remote.assert_called_once()

    def test_left_blocks_after_right_done_submitted_immediately(self):
        op, pool = self._make_operator()
        op.input_done(1)
        bundle = self._make_bundle(2)
        op._add_input_inner(bundle, input_index=0)
        assert len(op._pending_left_bundles) == 0
        assert op._left_block_idx == 2

    def test_is_inserting_done_requires_all_inputs_complete(self):
        op, pool = self._make_operator()
        assert op._is_inserting_done() is False
        op._inputs_complete = True
        assert op._is_inserting_done() is True

    def test_finalize_submits_left_insert_finished_and_execute_join(self):
        op, pool = self._make_operator()
        op._inputs_complete = True
        op._right_phase_done = True

        op._try_finalize()

        assert op._finalization_started is True
        for actor in pool.actors:
            actor.left_insert_finished.remote.assert_called_once()
            actor.execute_join.options.assert_called_once()

    def test_finalize_not_triggered_before_inputs_complete(self):
        op, pool = self._make_operator()
        op._try_finalize()
        assert op._finalization_started is False

    def test_finalize_idempotent(self):
        op, pool = self._make_operator()
        op._inputs_complete = True
        op._right_phase_done = True
        op._try_finalize()
        op._try_finalize()
        for actor in pool.actors:
            assert actor.left_insert_finished.remote.call_count == 1


# ---------------------------------------------------------------------------
# Unit tests — GPUJoinActor methods (mocked shuffler)
# ---------------------------------------------------------------------------


class TestGPUJoinActorMethods:
    """Test GPUJoinActor logic using a mocked BulkRapidsMPFJoinShuffler."""

    def _make_actor(self):
        """Create a GPUJoinActor with a mocked shuffler (no GPU required).

        GPUJoinActor is wrapped by @ray.remote and becomes an ActorClass, so
        __new__ cannot be called on it directly.  Use __ray_actor_class__ to
        get the underlying Python class.
        """
        from unittest.mock import MagicMock

        from ray.data._internal.gpu_shuffle.join import GPUJoinActor

        underlying_cls = GPUJoinActor.__ray_actor_class__
        actor = underlying_cls.__new__(underlying_cls)

        mock_shuffler = MagicMock()
        mock_shuffler.extract.return_value = iter([])

        actor._shuffler = mock_shuffler
        actor._left_key_columns = ["id"]
        actor._right_key_columns = ["id"]
        actor._join_type = JoinType.INNER
        actor._right_schema = None
        actor._left_schema = None
        actor._stored_right_df = None
        return actor, mock_shuffler, MagicMock()

    def test_set_right_schema_only_sets_once(self):
        actor, _, _ = self._make_actor()
        table = pa.Table.from_pydict({"a": [1, 2], "b": [3, 4]})
        actor.set_right_schema(table.schema)
        actor.set_right_schema(pa.Table.from_pydict({"c": [5, 6]}).schema)  # should be ignored
        assert actor._right_schema == table.schema

    def test_set_left_schema_only_sets_once(self):
        actor, _, _ = self._make_actor()
        table = pa.Table.from_pydict({"x": [1, 2], "y": [3, 4]})
        actor.set_left_schema(table.schema)
        actor.set_left_schema(pa.Table.from_pydict({"z": [5, 6]}).schema)  # should be ignored
        assert actor._left_schema == table.schema


# ---------------------------------------------------------------------------
# Unit tests — GPUJoinRankPool
# ---------------------------------------------------------------------------


class TestGPUJoinRankPool:
    def test_round_robin_distribution(self):
        from unittest.mock import MagicMock, patch

        from ray.data._internal.gpu_shuffle.join import GPUJoinRankPool

        pool = GPUJoinRankPool(
            nranks=3,
            total_nparts=6,
            left_key_columns=["id"],
            right_key_columns=["id"],
            join_type=JoinType.INNER,
            rmm_pool_size=None,
            spill_memory_limit=None,
        )
        mock_actors = [MagicMock(), MagicMock(), MagicMock()]
        pool._actors = mock_actors

        for i in range(9):
            assert pool.get_actor_for_block(i) is mock_actors[i % 3]

    def test_shutdown_force_kills_actors(self):
        from unittest.mock import MagicMock, patch

        import ray as ray_mod

        from ray.data._internal.gpu_shuffle.join import GPUJoinRankPool

        pool = GPUJoinRankPool(
            nranks=2,
            total_nparts=4,
            left_key_columns=["id"],
            right_key_columns=["id"],
            join_type=JoinType.INNER,
            rmm_pool_size=None,
            spill_memory_limit=None,
        )
        mock_actors = [MagicMock(), MagicMock()]
        pool._actors = mock_actors

        with patch.object(ray_mod, "kill") as mock_kill:
            pool.shutdown(force=True)

        assert mock_kill.call_count == 2
        assert pool._actors == []


# ---------------------------------------------------------------------------
# Unit tests — BulkRapidsMPFJoinShuffler
# ---------------------------------------------------------------------------


class TestBulkRapidsMPFJoinShuffler:
    """Verify the ProgressThread-sharing logic without actual GPU ops."""

    def test_reset_for_left_shuffle_updates_shuffle_on(self):
        import sys
        from unittest.mock import MagicMock, patch

        # Build a minimal instance without calling __init__.
        from ray.data._internal.gpu_shuffle import rapidsmpf_backend as backend_mod

        shuffler = backend_mod.BulkRapidsMPFJoinShuffler.__new__(
            backend_mod.BulkRapidsMPFJoinShuffler
        )
        shuffler.shuffle_on = ["right_key"]
        shuffler.total_nparts = 4
        shuffler.br = MagicMock()
        shuffler.stats = MagicMock()
        shuffler._comm = MagicMock()
        shuffler._op_id = 0

        mock_pt = MagicMock()
        mock_new_shuffler = MagicMock()
        shuffler._progress_thread = mock_pt
        shuffler.shuffler = MagicMock()  # old shuffler

        mock_shuffler_cls = MagicMock(return_value=mock_new_shuffler)

        with patch.dict(
            sys.modules,
            {"rapidsmpf.shuffler": MagicMock(Shuffler=mock_shuffler_cls)},
        ):
            import importlib
            import rapidsmpf.shuffler as rs_mod

            rs_mod.Shuffler = mock_shuffler_cls
            shuffler.reset_for_left_shuffle(["left_key"])

        assert shuffler.shuffle_on == ["left_key"]
        assert shuffler._op_id == 1
        assert shuffler.shuffler is mock_new_shuffler
        mock_shuffler_cls.assert_called_once_with(
            shuffler._comm,
            mock_pt,
            op_id=1,
            total_num_partitions=4,
            br=shuffler.br,
            statistics=shuffler.stats,
        )

    def test_cleanup_does_not_call_shutdown(self):
        from unittest.mock import MagicMock

        from ray.data._internal.gpu_shuffle import rapidsmpf_backend as backend_mod

        shuffler = backend_mod.BulkRapidsMPFJoinShuffler.__new__(
            backend_mod.BulkRapidsMPFJoinShuffler
        )
        mock_shuffler_inner = MagicMock()
        shuffler.shuffler = mock_shuffler_inner
        shuffler._progress_thread = MagicMock()
        # With enable_statistics=False, cleanup() never reads self.comm.
        shuffler.enable_statistics = False
        shuffler.stats = None

        shuffler.cleanup()

        mock_shuffler_inner.shutdown.assert_not_called()
        assert shuffler.shuffler is None
        assert shuffler._progress_thread is None


# ---------------------------------------------------------------------------
# Join type mapping tests
# ---------------------------------------------------------------------------


class TestCudfJoinMap:
    def test_all_join_types_mapped(self):
        from ray.data._internal.gpu_shuffle.join import _CUDF_JOIN_MAP

        for jt in JoinType:
            assert jt in _CUDF_JOIN_MAP, f"{jt} missing from _CUDF_JOIN_MAP"

    def test_mapping_values(self):
        from ray.data._internal.gpu_shuffle.join import _CUDF_JOIN_MAP

        assert _CUDF_JOIN_MAP[JoinType.INNER] == "inner"
        assert _CUDF_JOIN_MAP[JoinType.LEFT_OUTER] == "left"
        assert _CUDF_JOIN_MAP[JoinType.RIGHT_OUTER] == "right"
        assert _CUDF_JOIN_MAP[JoinType.FULL_OUTER] == "outer"
        assert _CUDF_JOIN_MAP[JoinType.LEFT_SEMI] == "leftsemi"
        assert _CUDF_JOIN_MAP[JoinType.RIGHT_SEMI] == "rightsemi"
        assert _CUDF_JOIN_MAP[JoinType.LEFT_ANTI] == "leftanti"
        assert _CUDF_JOIN_MAP[JoinType.RIGHT_ANTI] == "rightanti"


# ---------------------------------------------------------------------------
# Resource accounting tests
# ---------------------------------------------------------------------------


class TestGPUJoinOperatorResources:
    def _make_op(self, nranks=4):
        from unittest.mock import MagicMock

        from ray.data._internal.gpu_shuffle.join import GPUJoinOperator

        mock_pool = MagicMock()
        mock_pool.nranks = nranks

        op = GPUJoinOperator.__new__(GPUJoinOperator)
        op._rank_pool = mock_pool
        return op

    def test_base_resource_usage(self):
        op = self._make_op(nranks=4)
        usage = op.base_resource_usage
        assert usage.gpu == 4

    def test_incremental_resource_usage_is_zero(self):
        from ray.data._internal.execution.interfaces import ExecutionResources

        op = self._make_op()
        incr = op.incremental_resource_usage()
        assert incr == ExecutionResources()

    def test_current_logical_usage_equals_base(self):
        op = self._make_op(nranks=2)
        assert op.current_logical_usage() == op.base_resource_usage


# ---------------------------------------------------------------------------
# Integration tests — require GPU (marked with pytest.mark.gpu)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestGPUJoinIntegration:
    """End-to-end tests requiring actual GPU workers."""

    @pytest.fixture(autouse=True)
    def setup_ray(self):
        if not ray.is_initialized():
            ray.init(num_gpus=2)
        yield
        # Don't shut down between tests

    def _gpu_ctx(self) -> DataContext:
        ctx = DataContext.get_current().copy()
        ctx.shuffle_strategy = ShuffleStrategy.GPU_SHUFFLE
        ctx.gpu_shuffle_num_actors = 2
        return ctx

    def _join(
        self,
        left_table: pa.Table,
        right_table: pa.Table,
        join_type: str,
        left_on=("id",),
        right_on=("id",),
        num_partitions: int = 4,
    ) -> pa.Table:
        import ray.data as rd

        ctx = self._gpu_ctx()
        with DataContext.current(ctx):
            left_ds = rd.from_arrow(left_table)
            right_ds = rd.from_arrow(right_table)
            result_ds = left_ds.join(
                right_ds,
                join_type=join_type,
                num_partitions=num_partitions,
                on=left_on,
                right_on=right_on,
            )
            return pa.concat_tables(ray.get(result_ds.to_arrow_refs()))

    def test_inner_join_basic(self):
        left = _left_table(10)
        right = _right_table(5)
        result = self._join(left, right, "inner")
        # Only ids 0-4 match
        assert result.num_rows == 5
        ids = sorted(result.column("id").to_pylist())
        assert ids == [0, 1, 2, 3, 4]

    def test_left_outer_join(self):
        left = _left_table(10)
        right = _right_table(5)
        result = self._join(left, right, "left_outer")
        # All 10 left rows, NULLs for unmatched right
        assert result.num_rows == 10

    def test_left_anti_join(self):
        left = _left_table(10)
        right = _right_table(5)
        result = self._join(left, right, "left_anti")
        # Rows 5-9 have no right match
        assert result.num_rows == 5
        ids = sorted(result.column("id").to_pylist())
        assert ids == [5, 6, 7, 8, 9]

    def test_inner_join_empty_left(self):
        import ray.data as rd

        ctx = self._gpu_ctx()
        with DataContext.current(ctx):
            left_ds = rd.from_arrow(
                pa.table(
                    {
                        "id": pa.array([], type=pa.int64()),
                        "val": pa.array([], type=pa.string()),
                    }
                )
            )
            right_ds = rd.from_arrow(_right_table(5))
            result_ds = left_ds.join(
                right_ds, join_type="inner", num_partitions=4, on=("id",)
            )
            assert result_ds.count() == 0

    def test_inner_join_empty_right(self):
        import ray.data as rd

        ctx = self._gpu_ctx()
        with DataContext.current(ctx):
            left_ds = rd.from_arrow(_left_table(5))
            right_ds = rd.from_arrow(
                pa.table(
                    {
                        "id": pa.array([], type=pa.int64()),
                        "val": pa.array([], type=pa.string()),
                    }
                )
            )
            result_ds = left_ds.join(
                right_ds, join_type="inner", num_partitions=4, on=("id",)
            )
            assert result_ds.count() == 0

    def test_asymmetric_keys(self):
        """Join on differently named key columns."""
        import ray.data as rd

        ctx = self._gpu_ctx()
        left = pa.table({"left_id": [1, 2, 3], "lv": ["a", "b", "c"]})
        right = pa.table({"right_id": [2, 3, 4], "rv": ["x", "y", "z"]})
        with DataContext.current(ctx):
            left_ds = rd.from_arrow(left)
            right_ds = rd.from_arrow(right)
            result_ds = left_ds.join(
                right_ds,
                join_type="inner",
                num_partitions=4,
                on=("left_id",),
                right_on=("right_id",),
            )
            result = pa.concat_tables(ray.get(result_ds.to_arrow_refs()))
        assert result.num_rows == 2

    def test_all_rows_match_inner(self):
        """Inner join where all rows match should return same count."""
        import ray.data as rd

        ctx = self._gpu_ctx()
        n = 100
        both = pa.table({"id": list(range(n)), "val": list(range(n))})
        with DataContext.current(ctx):
            left_ds = rd.from_arrow(both.rename_columns(["id", "left_val"]))
            right_ds = rd.from_arrow(both.rename_columns(["id", "right_val"]))
            result_ds = left_ds.join(
                right_ds, join_type="inner", num_partitions=4, on=("id",)
            )
            assert result_ds.count() == n

    def test_no_rows_match_inner(self):
        """Inner join with disjoint keys returns empty."""
        import ray.data as rd

        ctx = self._gpu_ctx()
        left = pa.table({"id": [1, 2, 3], "v": ["a", "b", "c"]})
        right = pa.table({"id": [10, 20, 30], "v": ["x", "y", "z"]})
        with DataContext.current(ctx):
            left_ds = rd.from_arrow(left)
            right_ds = rd.from_arrow(right)
            result_ds = left_ds.join(
                right_ds, join_type="inner", num_partitions=4, on=("id",)
            )
            assert result_ds.count() == 0

    def test_full_outer_join(self):
        """Full outer: all rows from both sides, NULLs where no match."""
        import ray.data as rd

        ctx = self._gpu_ctx()
        left = pa.table({"id": [1, 2, 3], "lv": ["a", "b", "c"]})
        right = pa.table({"id": [2, 3, 4], "rv": ["x", "y", "z"]})
        with DataContext.current(ctx):
            left_ds = rd.from_arrow(left)
            right_ds = rd.from_arrow(right)
            result_ds = left_ds.join(
                right_ds, join_type="full_outer", num_partitions=4, on=("id",)
            )
            # ids: 1, 2, 3, 4
            assert result_ds.count() == 4

    @pytest.mark.skip(reason="right_anti join not supported")
    def test_right_anti_join(self):
        """Right anti: right rows with no left match."""
        import ray.data as rd

        ctx = self._gpu_ctx()
        left = pa.table({"id": [1, 2], "lv": ["a", "b"]})
        right = pa.table({"id": [2, 3, 4], "rv": ["x", "y", "z"]})
        with DataContext.current(ctx):
            left_ds = rd.from_arrow(left)
            right_ds = rd.from_arrow(right)
            result_ds = left_ds.join(
                right_ds, join_type="right_anti", num_partitions=4, on=("id",)
            )
            assert result_ds.count() == 2
            ids = sorted(result_ds.column("id").to_pylist())
            assert ids == [3, 4]


if __name__ == "__main__":
    TestGPUJoinIntegration().test_right_anti_join()