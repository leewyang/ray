import hashlib

import numpy as np
import pyarrow as pa
import pytest

from ray.data._internal.gpu_sort.streaming_sample import (
    finalize_priority_sample,
    select_priority_sample,
)


WEIGHT = "__weight"
BLOCK = "__block"
STRATUM = "__stratum"
INDEX = "__index"


def _candidates(rows=257):
    values = np.arange(rows, dtype=np.int64)
    schema = pa.schema(
        [
            pa.field("origin", pa.string()),
            pa.field("sequence", pa.int64()),
            pa.field(WEIGHT, pa.uint64(), nullable=False),
            pa.field(BLOCK, pa.uint64(), nullable=False),
            pa.field(STRATUM, pa.uint64(), nullable=False),
            pa.field(INDEX, pa.uint64(), nullable=False),
        ],
        metadata={b"contract": b"streaming-sample"},
    )
    return pa.Table.from_arrays(
        [
            pa.array([f"airport-{value % 23:02d}" for value in values]),
            pa.array((values * 37) % 101, type=pa.int64()),
            pa.array((values % 9) + 1, type=pa.uint64()),
            pa.array(values // 11, type=pa.uint64()),
            pa.array(values % 11, type=pa.uint64()),
            pa.array(values * 7 + 3, type=pa.uint64()),
        ],
        schema=schema,
    )


def _nullable_candidates(rows=257):
    values = np.arange(rows, dtype=np.int64)
    schema = pa.schema(
        [
            pa.field("origin", pa.string()),
            pa.field("sequence", pa.int64()),
            pa.field(WEIGHT, pa.uint64(), nullable=False),
            pa.field(BLOCK, pa.uint64(), nullable=False),
            pa.field(STRATUM, pa.uint64(), nullable=False),
            pa.field(INDEX, pa.uint64(), nullable=False),
        ],
        metadata={b"contract": b"nullable-multiple-key-sample"},
    )
    return pa.Table.from_arrays(
        [
            pa.array(
                [
                    None if value % 2 == 0 else f"airport-{value % 19:02d}"
                    for value in values
                ],
                type=pa.string(),
            ),
            pa.array(
                [int(value) if value % 2 == 0 else None for value in values],
                type=pa.int64(),
            ),
            pa.array((values % 7) + 1, type=pa.uint64()),
            pa.array(values // 13, type=pa.uint64()),
            pa.array(values % 13, type=pa.uint64()),
            pa.array(values * 5 + 1, type=pa.uint64()),
        ],
        schema=schema,
    )


def _select(samples, *, capacity=31, seed=1234):
    return select_priority_sample(
        samples,
        capacity=capacity,
        seed=seed,
        block_name=BLOCK,
        stratum_name=STRATUM,
        index_name=INDEX,
    )


def _finalize(samples, *, candidate_rows, capacity=31, seed=1234):
    return finalize_priority_sample(
        samples,
        candidate_rows=candidate_rows,
        capacity=capacity,
        seed=seed,
        weight_name=WEIGHT,
        block_name=BLOCK,
        stratum_name=STRATUM,
        index_name=INDEX,
    )


def _slices(table, widths):
    chunks = []
    offset = 0
    for width in widths:
        chunks.append(table.slice(offset, width))
        offset += width
    if offset < table.num_rows:
        chunks.append(table.slice(offset))
    assert sum(chunk.num_rows for chunk in chunks) == table.num_rows
    return chunks


def _incremental_reservoir(chunks, *, capacity, seed):
    reservoir = None
    for chunk in chunks:
        inputs = [chunk] if reservoir is None else [reservoir, chunk]
        reservoir = _select(inputs, capacity=capacity, seed=seed)
        assert reservoir.num_rows <= capacity
    return reservoir


def _coordinates(table):
    return set(zip(*(table[name].to_pylist() for name in (BLOCK, STRATUM, INDEX))))


def _digest(table):
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def test_priority_sample_is_invariant_to_arrival_order_and_batching():
    candidates = _candidates()
    capacity = 31
    seed = 0xC0FFEE
    expected = _select([candidates], capacity=capacity, seed=seed)

    uneven = _slices(candidates, [1, 19, 3, 47, 2, 61, 7, 89])
    single_rows = _slices(candidates, [1] * candidates.num_rows)

    assert _select(uneven, capacity=capacity, seed=seed).equals(expected)
    assert _select(list(reversed(uneven)), capacity=capacity, seed=seed).equals(
        expected
    )
    assert _incremental_reservoir(uneven, capacity=capacity, seed=seed).equals(expected)
    assert _incremental_reservoir(
        list(reversed(single_rows)), capacity=capacity, seed=seed
    ).equals(expected)


@pytest.mark.parametrize("rank_count", [1, 2, 5, 17])
def test_bottom_k_is_bounded_mergeable_and_rank_invariant(rank_count):
    candidates = _candidates()
    capacity = 29
    seed = 987654321
    expected = _select([candidates], capacity=capacity, seed=seed)

    rank_reservoirs = []
    for rank in range(rank_count):
        row_indices = np.arange(rank, candidates.num_rows, rank_count, dtype=np.int64)
        rank_input = candidates.take(pa.array(row_indices, type=pa.int64()))
        reservoir = _select([rank_input], capacity=capacity, seed=seed)
        assert reservoir.num_rows <= capacity
        rank_reservoirs.append(reservoir)

    merged = _select(list(reversed(rank_reservoirs)), capacity=capacity, seed=seed)
    assert merged.num_rows == capacity
    assert merged.equals(expected)


def test_priority_sample_is_reproducible_and_seeded():
    candidates = _candidates()

    first = _select([candidates], capacity=37, seed=42)
    second = _select([candidates], capacity=37, seed=42)
    different_seed = _select([candidates], capacity=37, seed=43)

    assert first.equals(second)
    assert _digest(first) == _digest(second)
    assert _coordinates(first) != _coordinates(different_seed)
    assert first.schema == different_seed.schema == candidates.schema


def test_finalize_scales_weights_by_global_candidate_inclusion_rate():
    candidates = _candidates()
    capacity = 31
    selected = _select([candidates], capacity=capacity, seed=77)
    finalized = _finalize(
        [candidates],
        candidate_rows=candidates.num_rows,
        capacity=capacity,
        seed=77,
    )

    assert finalized.schema == selected.schema
    for before, after in zip(selected.to_pylist(), finalized.to_pylist()):
        assert {key: value for key, value in after.items() if key != WEIGHT} == {
            key: value for key, value in before.items() if key != WEIGHT
        }
        assert after[WEIGHT] == (
            before[WEIGHT] * candidates.num_rows // selected.num_rows
        )

    unscaled = _finalize(
        [selected],
        candidate_rows=selected.num_rows,
        capacity=capacity,
        seed=77,
    )
    assert unscaled.equals(selected)


def test_string_null_and_multiple_sort_keys_are_preserved():
    candidates = _nullable_candidates()
    selected = _select([candidates], capacity=200, seed=2024)
    source_rows = {
        (row[BLOCK], row[STRATUM], row[INDEX]): row for row in candidates.to_pylist()
    }

    assert selected.schema == candidates.schema
    assert selected["origin"].null_count > 0
    assert selected["origin"].null_count < selected.num_rows
    assert selected["sequence"].null_count > 0
    assert selected["sequence"].null_count < selected.num_rows
    for row in selected.to_pylist():
        coordinate = (row[BLOCK], row[STRATUM], row[INDEX])
        assert row["origin"] == source_rows[coordinate]["origin"]
        assert row["sequence"] == source_rows[coordinate]["sequence"]


@pytest.mark.parametrize("capacity", [0, -1])
def test_priority_sample_rejects_nonpositive_capacity(capacity):
    with pytest.raises(ValueError, match="capacity must be positive"):
        _select([_candidates(3)], capacity=capacity)


@pytest.mark.parametrize("seed", [-1, 1 << 64])
def test_priority_sample_rejects_seed_outside_uint64(seed):
    with pytest.raises(ValueError, match="seeds must fit in unsigned 64 bits"):
        _select([_candidates(3)], capacity=1, seed=seed)


@pytest.mark.parametrize("missing", [BLOCK, STRATUM, INDEX])
def test_priority_sample_requires_all_canonical_coordinates(missing):
    candidates = _candidates(3)
    incomplete = candidates.select(
        [name for name in candidates.column_names if name != missing]
    )
    with pytest.raises(ValueError, match="missing coordinate column"):
        _select([incomplete], capacity=1)


def test_finalize_validates_candidate_count_and_weight_column():
    candidates = _candidates(3)

    assert _finalize([], candidate_rows=0) is None
    with pytest.raises(ValueError, match="candidates are missing"):
        _finalize([], candidate_rows=1)
    with pytest.raises(ValueError, match="candidate count is inconsistent"):
        _finalize([candidates], candidate_rows=2)

    without_weight = candidates.select(
        [name for name in candidates.column_names if name != WEIGHT]
    )
    with pytest.raises(ValueError, match="missing weight column"):
        _finalize(
            [without_weight],
            candidate_rows=without_weight.num_rows,
            capacity=1,
        )
