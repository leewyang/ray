from contextlib import contextmanager
import sys
import types

import pyarrow as pa
import pytest

from ray.data._internal.gpu_sort.backend import (
    _ExternalRun,
    _RunChunk,
    _duplicate_boundary_cuts,
    lazy_load_backend,
)
from ray.data._internal.gpu_sort.config import GPUSortConfig


def _stub_module(monkeypatch, name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _new_backend(backend_class, **attributes):
    backend = object.__new__(backend_class)
    backend.__dict__.update(attributes)
    return backend


@pytest.fixture
def gpu_backend_class(monkeypatch):
    """Load backend methods without importing optional RAPIDS packages."""

    _stub_module(
        monkeypatch,
        "rapidsmpf.utils.ray_utils",
        BaseShufflingActor=type("BaseShufflingActor", (), {}),
    )
    return lazy_load_backend()


def test_source_run_transaction_releases_all_prior_commits(
    gpu_backend_class, monkeypatch
):
    _stub_module(
        monkeypatch,
        "cupy",
        cuda=types.SimpleNamespace(
            runtime=types.SimpleNamespace(deviceSynchronize=lambda: None)
        ),
    )
    _stub_module(
        monkeypatch,
        "cudf",
        DataFrame=types.SimpleNamespace(from_arrow=lambda arrow: arrow),
    )
    _stub_module(
        monkeypatch,
        "rapidsmpf.utils.cudf",
        cudf_to_pylibcudf_table=lambda frame: frame,
    )

    backend = _new_backend(
        gpu_backend_class,
        _config=GPUSortConfig(exchange_batch_bytes=64),
        _payload_limit_bytes=64,
        _source_runs={},
        _source_run_rows={},
        _next_source_run_id=0,
        _started_at=0.0,
        _stats={
            "h2d_bytes": 0,
            "first_run_committed_at_ns": None,
            "last_run_committed_at_ns": None,
            "first_externalize_s": None,
            "first_externalize_wave": None,
            "run_sort_oom_retry_count": 0,
            "phases_s": {"run_sort": 0.0},
        },
    )
    backend._augment_table = lambda table, names: (table, names)
    backend._sort_table = lambda table: table
    backend._bounded_arrow_prefix = lambda table, _target: (
        table.slice(0, min(2, table.num_rows)),
        table.slice(min(2, table.num_rows)),
    )

    first_chunk = _RunChunk(ref="source-0", rows=2, size_bytes=16)
    committed_runs = []

    def store(_table, *, initial):
        assert initial
        if committed_runs:
            # The first run remains actor-owned until the complete input block
            # has committed, then the outer transaction rolls it back.
            assert list(backend._source_runs) == [0]
            assert backend._source_runs[0] is committed_runs[0]
            raise OSError("synthetic second source-run failure")
        run = _ExternalRun([first_chunk])
        committed_runs.append(run)
        return run

    released = []
    backend._store_table_as_run = store
    backend._release_run_chunks = lambda chunks: released.extend(chunks)

    arrow = pa.table({"Origin": ["ATL", "BOS", "LAX", "SFO"]})
    with pytest.raises(OSError, match="second source-run failure"):
        backend._commit_source_arrow(arrow)

    assert released == [first_chunk]
    assert committed_runs[0].chunks == []
    assert backend._source_runs == {}
    assert backend._source_run_rows == {}
    # Published IDs are never reused, including after rollback.
    assert backend._next_source_run_id == 1


def test_source_run_commit_accepts_unsplit_prefix_without_remainder(
    gpu_backend_class, monkeypatch
):
    _stub_module(
        monkeypatch,
        "cupy",
        cuda=types.SimpleNamespace(
            runtime=types.SimpleNamespace(deviceSynchronize=lambda: None)
        ),
    )
    _stub_module(
        monkeypatch,
        "cudf",
        DataFrame=types.SimpleNamespace(from_arrow=lambda arrow: arrow),
    )
    _stub_module(
        monkeypatch,
        "rapidsmpf.utils.cudf",
        cudf_to_pylibcudf_table=lambda frame: frame,
    )
    backend = _new_backend(
        gpu_backend_class,
        _config=GPUSortConfig(exchange_batch_bytes=1 << 20),
        _payload_limit_bytes=1 << 20,
        _source_runs={},
        _source_run_rows={},
        _next_source_run_id=0,
        _started_at=0.0,
        _stats={
            "h2d_bytes": 0,
            "first_run_committed_at_ns": None,
            "last_run_committed_at_ns": None,
            "first_externalize_s": None,
            "first_externalize_wave": None,
            "run_sort_oom_retry_count": 0,
            "phases_s": {"run_sort": 0.0},
        },
    )
    backend._augment_table = lambda table, names: (table, names)
    backend._sort_table = lambda table: table
    backend._bounded_arrow_prefix = lambda table, _target: (table, None)
    backend._store_table_as_run = lambda table, *, initial: _ExternalRun(
        [_RunChunk(ref="source", rows=table.num_rows, size_bytes=table.nbytes)]
    )

    arrow = pa.table({"Origin": ["ATL", "BOS"]})
    assert backend._commit_source_arrow(arrow) == [0]
    assert backend._source_run_rows == {0: 2}


def _make_ingest_backend(backend_class):
    weight_name = "__weight"
    block_name = "__block"
    stratum_name = "__stratum"
    index_name = "__index"
    backend = _new_backend(
        backend_class,
        _config=GPUSortConfig(sample_size=3, sample_seed=17),
        _key_columns=["Origin"],
        _weight_name=weight_name,
        _sample_block_name=block_name,
        _sample_stratum_name=stratum_name,
        _sample_index_name=index_name,
        _arrow_schema=None,
        _source_runs={},
        _source_run_rows={},
        _next_source_run_id=0,
        _streaming_sample=None,
        _streaming_sample_candidate_rows=0,
        _streaming_sampled_blocks=0,
        _ingest_finalized=False,
        _stats={
            "input_rows": 0,
            "input_bytes": 0,
            "input_block_count": 0,
            "ingest_rpc_count": 0,
            "source_run_count": 0,
            "source_run_bytes": 0,
            "source_run_rows": 0,
            "first_ingest_started_at_ns": None,
            "first_run_committed_at_ns": 11,
            "last_run_committed_at_ns": 22,
            "ingest_wall_s": 0.0,
            "streaming_sample_construction_s": 0.0,
            "phases_s": {"sampling": 0.0},
        },
    )
    backend.rank = lambda: 0
    backend._to_arrow_table = lambda block: block

    def set_schema(schema):
        if backend._arrow_schema is None:
            backend._arrow_schema = schema
        else:
            assert backend._arrow_schema.equals(schema)

    backend._set_schema = set_schema

    def sample(arrow, *, block_ordinal):
        rows = int(arrow.num_rows)
        return pa.table(
            {
                "Origin": arrow["Origin"],
                weight_name: pa.array([1] * rows, type=pa.uint64()),
                block_name: pa.array([block_ordinal] * rows, type=pa.uint64()),
                stratum_name: pa.array(range(rows), type=pa.uint64()),
                index_name: pa.array(range(rows), type=pa.uint64()),
            }
        )

    backend._streaming_sample_for_arrow = sample

    def commit(arrow):
        run_id = backend._next_source_run_id
        backend._next_source_run_id += 1
        chunk = _RunChunk(
            ref=f"source-{run_id}",
            rows=int(arrow.num_rows),
            size_bytes=int(arrow.nbytes),
        )
        backend._source_runs[run_id] = _ExternalRun([chunk])
        backend._source_run_rows[run_id] = int(arrow.num_rows)
        return [run_id]

    backend._commit_source_arrow = commit
    backend._release_run_chunks = lambda chunks: pytest.fail(
        f"successful ingest unexpectedly released {chunks!r}"
    )
    backend._update_peak = lambda: None
    return backend


def test_finalize_ingest_manifest_and_sample_are_batching_independent(
    gpu_backend_class,
):
    blocks = [
        pa.table(
            {
                "Origin": [f"O{ordinal * 2}", f"O{ordinal * 2 + 1}"],
                "payload": [ordinal, ordinal],
            }
        )
        for ordinal in range(4)
    ]

    batched = _make_ingest_backend(gpu_backend_class)
    first = batched.ingest_blocks(blocks[:2], block_ordinals=[0, 1])
    second = batched.ingest_blocks(blocks[2:], block_ordinals=[2, 3])
    batched_manifest = batched.finalize_ingest()

    combined = _make_ingest_backend(gpu_backend_class)
    combined.ingest_blocks(blocks, block_ordinals=[0, 1, 2, 3])
    combined_manifest = combined.finalize_ingest()

    assert first["source_runs"] == second["source_runs"] == 2
    assert first["rows"] == second["rows"] == 4
    assert batched_manifest["sample_candidate_rows"] == 8
    assert batched_manifest["sampled_block_count"] == 4
    assert batched_manifest["sample"].num_rows == 3
    assert batched_manifest["sample"].equals(combined_manifest["sample"])
    assert batched_manifest["runs"] == combined_manifest["runs"]
    assert [run["run_id"] for run in batched_manifest["runs"]] == [0, 1, 2, 3]
    assert [run["rows"] for run in batched_manifest["runs"]] == [2, 2, 2, 2]
    assert batched._stats["ingest_rpc_count"] == 2
    assert combined._stats["ingest_rpc_count"] == 1
    with pytest.raises(RuntimeError, match="already finalized"):
        batched.ingest_blocks([], block_ordinals=[])


def test_source_wave_releases_runs_only_after_explicit_commit(
    gpu_backend_class,
):
    chunks = [
        _RunChunk(ref="source-4", rows=2, size_bytes=20),
        _RunChunk(ref="source-7", rows=3, size_bytes=30),
    ]
    arrows = [
        pa.table({"Origin": ["ATL", "BOS"]}),
        pa.table({"Origin": ["SEA", "SFO", "SJC"]}),
    ]

    class Store:
        mode = "local_disk"

        def __init__(self):
            self.reads = []
            self.released = []

        @contextmanager
        def read_many(self, selected):
            self.reads.append(list(selected))
            yield arrows

        def release(self, selected):
            self.released.extend(selected)

        def diagnostics(self):
            return {}

    store = Store()
    source_runs = {
        4: _ExternalRun([chunks[0]]),
        7: _ExternalRun([chunks[1]]),
    }
    backend = _new_backend(
        gpu_backend_class,
        _ingest_finalized=True,
        _active_source_wave_id=None,
        _active_source_run_ids=[],
        _source_runs=source_runs,
        _source_run_rows={4: 2, 7: 3},
        _run_store=store,
        _prepared_wave_id=None,
        _prepared_batches={},
        _stats={},
    )
    backend.rank = lambda: 0

    def fail_prepare(*_args, **_kwargs):
        raise RuntimeError("synthetic preparation failure")

    backend.prepare_wave = fail_prepare
    with pytest.raises(RuntimeError, match="preparation failure"):
        backend.prepare_ingested_runs(5, [4, 7])
    assert backend._active_source_wave_id is None
    assert set(backend._source_runs) == {4, 7}
    assert backend._source_runs[4].chunks == [chunks[0]]
    assert backend._source_runs[7].chunks == [chunks[1]]
    assert store.released == []

    prepared = {}

    def prepare(wave_id, restored, **options):
        prepared.update(
            wave_id=wave_id,
            restored=list(restored),
            options=options,
        )
        return {"rank": 0, "wave": wave_id, "batches": []}

    backend.prepare_wave = prepare
    manifest = backend.prepare_ingested_runs(5, [4, 7])
    assert manifest["source_run_ids"] == [4, 7]
    assert prepared["restored"] == arrows
    assert prepared["options"] == {
        "count_input_blocks": False,
        "inputs_are_sorted": True,
    }
    assert backend._active_source_wave_id == 5
    assert store.released == []
    assert set(backend._source_runs) == {4, 7}

    backend._prepared_wave_id = 5
    with pytest.raises(RuntimeError, match="unfinished source wave"):
        backend.commit_source_wave(5)
    assert store.released == []
    assert set(backend._source_runs) == {4, 7}

    backend._prepared_wave_id = None
    committed = backend.commit_source_wave(5)
    assert committed["released_source_runs"] == 2
    assert committed["released_source_run_ids"] == [4, 7]
    assert committed["released_source_rows"] == 5
    assert committed["released_source_bytes"] == 50
    assert store.released == chunks
    assert backend._source_runs == {}
    assert backend._source_run_rows == {}
    assert backend._active_source_wave_id is None
    assert backend._active_source_run_ids == []


def test_duplicate_boundaries_produce_contiguous_balanced_cuts(
    gpu_backend_class, monkeypatch
):
    class DeviceTable:
        def __init__(self, rows, columns=("Origin",)):
            self._rows = int(rows)
            self.columns = list(columns)

        def num_rows(self):
            return self._rows

    class Series:
        def __init__(self, values):
            self._values = values

        @classmethod
        def from_pylibcudf(cls, values):
            return cls(values)

        def to_arrow(self):
            return pa.array(self._values, type=pa.int32())

    class Packer:
        def __init__(self, table):
            self._table = table

        def get_total_contiguous_size(self):
            return 10 * self._table.num_rows()

    class ChunkedPack:
        @staticmethod
        def create(table, _chunk_bytes, _stream):
            return Packer(table)

    lower_bounds = [2, 2, 2, 9]
    upper_bounds = [8, 8, 8, 10]
    _stub_module(
        monkeypatch,
        "pylibcudf",
        search=types.SimpleNamespace(
            lower_bound=lambda *_args: lower_bounds,
            upper_bound=lambda *_args: upper_bounds,
        ),
        contiguous_split=types.SimpleNamespace(ChunkedPack=ChunkedPack),
    )
    _stub_module(
        monkeypatch,
        "cudf",
        DataFrame=types.SimpleNamespace(
            from_arrow=lambda arrow: DeviceTable(arrow.num_rows, arrow.column_names)
        ),
        Series=Series,
    )
    captured = {}

    def split_and_pack(table, cuts, _stream, _buffer_resource):
        captured["table"] = table
        captured["cuts"] = list(cuts)
        return "packed"

    _stub_module(
        monkeypatch,
        "rapidsmpf.integrations.cudf.partition",
        split_and_pack=split_and_pack,
    )
    _stub_module(
        monkeypatch,
        "rapidsmpf.utils.cudf",
        cudf_to_pylibcudf_table=lambda frame: frame,
    )
    _stub_module(monkeypatch, "rmm.pylibrmm.stream", DEFAULT_STREAM=object())

    backend = _new_backend(
        gpu_backend_class,
        _num_partitions=5,
        _payload_limit_bytes=123,
        _run_chunk_bytes=45,
        _buffer_resource=object(),
        _work_names=["Origin"],
        _row_ordinal=0,
        _stats={"h2d_bytes": 0, "planning_h2d_bytes": 0},
    )
    backend.rank = lambda: 0
    backend._set_schema = lambda _schema: None
    augment_calls = []

    def augment(table, names):
        augment_calls.append(list(names))
        return table, names

    backend._augment_table = augment
    backend._comparison_table = lambda table, _names: table
    backend._initialize_run_limits = lambda: None
    backend._order_and_nulls = lambda: (object(), object())
    backend._slice_table = lambda _table, start, end: DeviceTable(end - start)

    boundaries = pa.table({"Origin": ["ATL", "ATL", "ATL", "SFO"]})
    installed = backend.install_plan(boundaries.schema, boundaries)
    assert installed["boundary_rows"] == 4
    assert backend._boundary_group_spans == [(0, 3), (3, 4)]
    assert backend._boundary_keys.num_rows() == 4
    assert augment_calls == [["Origin"]]

    packed, destination_bytes = backend._partition_sorted_and_pack(
        DeviceTable(10), wave_id=0
    )
    assert packed == "packed"
    # Three equal ATL boundaries split their [2, 8) equal-key interval across
    # the four adjacent ranges. The one SFO row is assigned to one of its two
    # adjacent ranges using the deterministic cross-run phase.
    assert captured["cuts"] == [3, 4, 6, 10]
    assert destination_bytes == (30, 10, 20, 40, 0)
    assert captured["table"].num_rows() == 10
    # Restored source runs already use work-schema columns, so replay must not
    # append float comparator columns a second time.
    assert augment_calls == [["Origin"]]

    # Arrow treats separate NaN scalars as unequal and install_plan therefore
    # may expose singleton groups. Equal GPU comparator intervals still merge
    # them into one balanced all-equal range.
    assert _duplicate_boundary_cuts(
        [0, 0],
        [9, 9],
        [(0, 1), (1, 2)],
        num_partitions=3,
        rows=9,
    ) == [3, 6]

    destinations = []
    for phase in range(16):
        cuts = _duplicate_boundary_cuts(
            [0] * 15,
            [1] * 15,
            [(0, 15)],
            num_partitions=16,
            rows=1,
            phase=phase,
        )
        destinations.append(next((index for index, cut in enumerate(cuts) if cut), 15))
    assert sorted(destinations) == list(range(16))
