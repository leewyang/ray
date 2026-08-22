"""Focused state-machine tests for streaming GPU-sort input ingestion.

These tests intentionally construct the operator with lightweight fakes.  They
exercise driver-side lifecycle and ownership rules without starting Ray or
requiring RAPIDS/GPU dependencies.
"""

from __future__ import annotations

import gc
from typing import Any
import weakref

import pytest

import ray
from ray.data import ExecutionOptions
from ray.data._internal.execution.block_ref_counter import BlockRefCounter
from ray.data._internal.execution.interfaces import BlockEntry, RefBundle
from ray.data._internal.gpu_sort import operator as gpu_sort_operator
from ray.data._internal.gpu_sort.operator import GPUSortOperator
from ray.data._internal.gpu_sort.operator import (
    _InputBlock,
    _ordered_exchange_receipts,
    _validate_source_wave_commit,
)
from ray.data.block import BlockMetadata


class _RemoteMethod:
    def __init__(self, result: Any):
        self._result = result
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._result

    def options(self, **kwargs):
        return self


class _Actor:
    def __init__(self, result_ref: ray.ObjectRef):
        self.ingest_blocks = _RemoteMethod(result_ref)
        self.diagnostics = _RemoteMethod(object())
        self.release = _RemoteMethod(object())
        self.finish_and_extract = _RemoteMethod(object())


class _RankPool:
    def __init__(self, actors):
        self._actors_on_start = list(actors)
        self.nranks = len(self._actors_on_start)
        self.actors = []
        self.start_calls = 0
        self.shutdown_calls = 0
        self.shutdown_async_calls = 0

    def start(self):
        self.start_calls += 1
        self.actors = list(self._actors_on_start)

    def shutdown_async(self):
        self.shutdown_async_calls += 1

    def shutdown(self):
        self.shutdown_calls += 1
        self.actors = []


class _Metrics:
    def __init__(self):
        self.queued = []
        self.dequeued = []
        self.submitted = []

    def on_input_queued(self, bundle, input_index):
        self.queued.append((bundle, input_index))

    def on_input_dequeued(self, bundle, input_index):
        self.dequeued.append((bundle, input_index))

    def on_task_submitted(self, task_index, bundle, task_id):
        self.submitted.append((task_index, bundle, task_id))


def _bundle(block_ref, *, size_bytes: int, num_rows: int) -> RefBundle:
    metadata = BlockMetadata(
        num_rows=num_rows,
        size_bytes=size_bytes,
        exec_stats=None,
        input_files=None,
    )
    return RefBundle(
        (BlockEntry(ref=block_ref, metadata=metadata),),
        schema=None,
        owns_blocks=True,
    )


def _block_ref_counter() -> BlockRefCounter:
    return BlockRefCounter(add_object_out_of_scope_callback=lambda *_args: True)


def _object_ref(byte: int) -> ray.ObjectRef:
    return ray.ObjectRef(bytes([byte]) * 28)


def _new_operator(*, input_buffer_budget_bytes: int = 100):
    result_ref = _object_ref(250)
    actor = _Actor(result_ref)
    rank_pool = _RankPool([actor])
    rank_pool.start()

    operator = object.__new__(GPUSortOperator)
    operator._rank_pool = rank_pool
    operator._metrics = _Metrics()
    operator._name = "GPU sort test"
    operator._id = "gpu-sort-test-operator"
    operator._block_ref_counter = _block_ref_counter()
    operator._config = {
        "input_buffer_budget_bytes": input_buffer_budget_bytes,
        "streaming_sample_rows_per_block": 4,
    }
    operator._started = True
    operator._inputs_complete = False
    operator._input_stats = []
    operator._input_schema = None
    operator._input_blocks = []
    operator._sample_quotas = []
    operator._next_block_ordinal = 0
    operator._input_bytes = 0
    operator._input_rows = 0
    operator._assigned_bytes = [0]
    operator._assigned_blocks = [0]
    operator._buffered_input_bytes = 0
    operator._buffered_input_blocks = 0
    operator._peak_buffered_input_bytes = 0
    operator._peak_buffered_input_blocks = 0
    operator._max_input_block_bytes = 0
    operator._input_ref_ids_received = set()
    operator._released_input_ref_ids = set()
    operator._released_input_object_refs = 0
    operator._first_input_received_at_ns = None
    operator._last_input_received_at_ns = None
    operator._inputs_complete_at_ns = None
    operator._first_gpu_run_committed_at_ns = None
    operator._last_gpu_run_committed_at_ns = None
    operator._ranks_started_at_ns = None
    operator._ingest_tasks = {}
    operator._extraction_tasks = {}
    operator._next_ingest_task_id = 0
    operator._finalization_started = False
    operator._finalization_succeeded = False
    operator._controller_phases = {}
    return operator, actor


def _receipt(*, rows: int, input_blocks: int = 1):
    return {
        "rank": 0,
        "rows": rows,
        "input_blocks": input_blocks,
        "first_run_committed_at_ns": 101,
        "last_run_committed_at_ns": 111,
    }


def test_start_starts_gpu_ranks_before_accepting_input():
    actor = _Actor(_object_ref(249))
    operator = object.__new__(GPUSortOperator)
    operator._rank_pool = _RankPool([actor])
    operator._controller_phases = {}
    operator._started = False

    block_ref_counter = _block_ref_counter()
    operator.start(ExecutionOptions(preserve_order=True), block_ref_counter)

    assert operator._started
    assert operator._block_ref_counter is block_ref_counter
    assert operator._rank_pool.start_calls == 1
    assert operator._rank_pool.actors == [actor]
    assert operator._ranks_started_at_ns is not None
    assert operator._controller_phases["startup"] >= 0


def test_start_requires_preserved_input_order():
    operator = object.__new__(GPUSortOperator)

    with pytest.raises(RuntimeError, match="preserve_order=True"):
        operator.start(ExecutionOptions(preserve_order=False), _block_ref_counter())


def test_add_input_submits_ingestion_without_blocking(monkeypatch):
    operator, actor = _new_operator()
    bundle = _bundle(_object_ref(1), size_bytes=60, num_rows=7)
    destroyed = []
    monkeypatch.setattr(
        RefBundle, "destroy_if_owned", lambda self: destroyed.append(self)
    )

    def fail_if_waited(*args, **kwargs):
        raise AssertionError("input submission must not synchronously call ray.get")

    monkeypatch.setattr(gpu_sort_operator.ray, "get", fail_if_waited)

    operator._add_input_inner(bundle, input_index=0)

    assert actor.ingest_blocks.calls == [(([_object_ref(1)], [0]), {})]
    assert len(operator._ingest_tasks) == 1
    assert operator._buffered_input_bytes == 60
    assert operator._buffered_input_blocks == 1
    assert destroyed == []


def test_can_add_input_allows_only_one_block_past_budget():
    operator, _ = _new_operator(input_buffer_budget_bytes=100)

    assert operator.can_add_input()
    operator._add_input_inner(
        _bundle(_object_ref(2), size_bytes=60, num_rows=6), input_index=0
    )
    assert operator.can_add_input()

    operator._add_input_inner(
        _bundle(_object_ref(3), size_bytes=60, num_rows=6), input_index=0
    )

    assert operator._buffered_input_bytes == 120
    assert operator._peak_buffered_input_bytes == 120
    assert operator._max_input_block_bytes == 60
    assert operator._peak_buffered_input_bytes <= 100 + 60
    assert not operator.can_add_input()


def test_ingest_completion_releases_bundle_and_buffered_accounting(monkeypatch):
    operator, _ = _new_operator()
    block_ref = _object_ref(4)
    bundle = _bundle(block_ref, size_bytes=60, num_rows=7)
    destroyed = []
    monkeypatch.setattr(
        RefBundle, "destroy_if_owned", lambda self: destroyed.append(self)
    )
    operator._add_input_inner(bundle, input_index=0)
    task = next(iter(operator._ingest_tasks.values()))
    monkeypatch.setattr(
        gpu_sort_operator.ray,
        "get",
        lambda result_ref: _receipt(rows=7),
    )

    task.on_task_finished()

    assert operator._ingest_tasks == {}
    assert operator._buffered_input_bytes == 0
    assert operator._buffered_input_blocks == 0
    assert operator._metrics.dequeued == [(bundle, 0)]
    assert destroyed == [bundle]
    assert operator._released_input_ref_ids == {block_ref.hex()}
    assert operator._released_input_object_refs == 1
    assert operator._first_gpu_run_committed_at_ns == 101
    assert operator._last_gpu_run_committed_at_ns == 111


def test_ingest_completion_severs_callback_bundle_ownership(monkeypatch):
    operator, _ = _new_operator()
    bundle = _bundle(_object_ref(7), size_bytes=60, num_rows=7)
    bundle_weakref = weakref.ref(bundle)
    operator._add_input_inner(bundle, input_index=0)
    task = next(iter(operator._ingest_tasks.values()))
    monkeypatch.setattr(gpu_sort_operator.ray, "get", lambda _ref: _receipt(rows=7))

    del bundle
    task.on_task_finished()
    # The fake metrics object intentionally retains dequeued bundles. Remove
    # that unrelated test-only owner, while retaining the completed task and
    # its callback just like the executor's active-task snapshot does.
    operator._metrics.queued.clear()
    operator._metrics.dequeued.clear()
    gc.collect()

    assert bundle_weakref() is None


def test_source_wave_receipts_prove_row_conservation_and_exact_release():
    blocks = [
        [_InputBlock(value=4, size_bytes=20, num_rows=2)],
        [_InputBlock(value=7, size_bytes=30, num_rows=3)],
    ]
    receipts = [
        {"rank": 1, "wave": 5, "exchange": 9, "received_rows": 3},
        {"rank": 0, "wave": 5, "exchange": 9, "received_rows": 2},
    ]
    ordered = _ordered_exchange_receipts(receipts, nranks=2, wave_id=5, exchange_id=9)
    assert [item["rank"] for item in ordered] == [0, 1]

    commits = [
        {
            "rank": 0,
            "wave": 5,
            "released_source_runs": 1,
            "released_source_run_ids": [4],
            "released_source_rows": 2,
            "released_source_bytes": 20,
        },
        {
            "rank": 1,
            "wave": 5,
            "released_source_runs": 1,
            "released_source_run_ids": [7],
            "released_source_rows": 3,
            "released_source_bytes": 30,
        },
    ]
    _validate_source_wave_commit(
        commits,
        blocks,
        wave_id=5,
        received_rows=sum(r["received_rows"] for r in receipts),
    )

    with pytest.raises(RuntimeError, match="did not conserve rows"):
        _validate_source_wave_commit(commits, blocks, wave_id=5, received_rows=4)

    bad_commits = [dict(commits[0]), dict(commits[1])]
    bad_commits[1]["released_source_run_ids"] = [8]
    with pytest.raises(RuntimeError, match="does not match"):
        _validate_source_wave_commit(bad_commits, blocks, wave_id=5, received_rows=5)


def test_finalization_waits_for_pending_ingestion(monkeypatch):
    operator, _ = _new_operator()
    bundle = _bundle(_object_ref(5), size_bytes=60, num_rows=7)
    operator._add_input_inner(bundle, input_index=0)
    task = next(iter(operator._ingest_tasks.values()))
    published = []
    operator._publish_diagnostics = published.append

    def fake_get(value):
        if isinstance(value, list):
            return [{"rank": 0}]
        return _receipt(rows=7)

    monkeypatch.setattr(gpu_sort_operator.ray, "get", fake_get)
    operator.all_inputs_done()

    operator._try_finalize()
    assert not operator._finalization_started
    assert published == []
    assert operator._rank_pool.shutdown_async_calls == 0

    task.on_task_finished()
    # Use the empty-input finalization path to isolate the pending-ingest gate.
    operator._input_blocks.clear()
    operator._try_finalize()

    assert operator._finalization_started
    assert operator._finalization_succeeded
    assert published == [[{"rank": 0}]]
    assert operator._rank_pool.actors[0].release.calls == [((), {"strict": True})]
    assert operator._rank_pool.shutdown_async_calls == 1


def test_empty_success_requires_strict_rank_cleanup(monkeypatch):
    operator, actor = _new_operator()
    operator._inputs_complete = True
    published = []
    operator._publish_diagnostics = published.append

    def fail_cleanup(_refs):
        raise RuntimeError("synthetic persistent local-run leak")

    monkeypatch.setattr(gpu_sort_operator.ray, "get", fail_cleanup)
    with pytest.raises(RuntimeError, match="persistent local-run leak"):
        operator._try_finalize()

    assert actor.release.calls == [((), {"strict": True})]
    assert operator._rank_pool.shutdown_calls == 1
    assert operator._rank_pool.shutdown_async_calls == 0
    assert not operator._finalization_succeeded
    assert published == []


def test_input_history_retains_metadata_and_string_ids_not_object_refs():
    operator, _ = _new_operator()
    block_ref = _object_ref(6)

    operator._add_input_inner(
        _bundle(block_ref, size_bytes=60, num_rows=7), input_index=0
    )

    assert [
        (block.value, block.size_bytes, block.num_rows, block.ordinal)
        for block in operator._input_blocks
    ] == [(None, 60, 7, 0)]
    assert operator._input_ref_ids_received == {block_ref.hex()}
    assert all(isinstance(ref_id, str) for ref_id in operator._input_ref_ids_received)
    assert all(
        not isinstance(block.value, ray.ObjectRef) for block in operator._input_blocks
    )
    assert not hasattr(operator, "_input_bundles")


def test_shutdown_cannot_reenter_finalization_after_rank_teardown(monkeypatch):
    operator, _ = _new_operator()
    operator._inputs_complete = True

    monkeypatch.setattr(
        gpu_sort_operator.PhysicalOperator,
        "_do_shutdown",
        lambda self, force: self.get_active_tasks(),
    )
    operator._do_shutdown(force=True)

    assert operator._finalization_started
    assert operator._rank_pool.shutdown_calls == 1
    assert operator.current_logical_usage().gpu == 0


def test_extraction_task_tracks_output_blocks_for_the_operator(monkeypatch):
    operator, _ = _new_operator()
    captured = {}

    class _DataOpTask:
        def __init__(
            self,
            *,
            block_ref_counter,
            producer_id,
            **kwargs,
        ):
            captured["block_ref_counter"] = block_ref_counter
            captured["producer_id"] = producer_id

        def get_task_id(self):
            return "task-id"

    monkeypatch.setattr(gpu_sort_operator, "DataOpTask", _DataOpTask)

    operator._schedule_extraction()

    assert captured == {
        "block_ref_counter": operator._block_ref_counter,
        "producer_id": operator.id,
    }
