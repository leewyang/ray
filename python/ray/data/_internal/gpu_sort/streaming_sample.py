"""Deterministic bounded sampling for one-pass GPU-sort ingestion."""

from __future__ import annotations

from typing import Any, Iterable


def _mix_uint64(values: Any) -> Any:
    """Vectorized SplitMix64 finalizer with defined unsigned overflow."""

    import numpy as np

    values = np.asarray(values, dtype=np.uint64)
    values = values ^ (values >> np.uint64(30))
    values = values * np.uint64(0xBF58476D1CE4E5B9)
    values = values ^ (values >> np.uint64(27))
    values = values * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def _priorities(
    table: Any,
    *,
    seed: int,
    block_name: str,
    stratum_name: str,
    index_name: str,
) -> tuple[Any, Any, Any, Any]:
    """Return deterministic priorities and canonical coordinate columns."""

    import numpy as np

    if not 0 <= int(seed) < 1 << 64:
        raise ValueError("GPU sort sample seeds must fit in unsigned 64 bits.")
    coordinates = []
    for name in (block_name, stratum_name, index_name):
        if name not in table.column_names:
            raise ValueError(f"GPU sort sample is missing coordinate column {name!r}.")
        coordinates.append(
            table[name]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .astype(np.uint64, copy=False)
        )
    blocks, strata, indices = coordinates
    # Mix each coordinate independently before combining it. This avoids the
    # simple linear relationships that raw block/row ordinals contain.
    priorities = _mix_uint64(
        _mix_uint64(blocks + np.uint64(0x9E3779B97F4A7C15))
        ^ _mix_uint64(strata + np.uint64(0xD1B54A32D192ED03))
        ^ _mix_uint64(indices + np.uint64(0x94D049BB133111EB))
        ^ np.uint64(seed)
    )
    return priorities, blocks, strata, indices


def select_priority_sample(
    samples: Iterable[Any],
    *,
    capacity: int,
    seed: int,
    block_name: str,
    stratum_name: str,
    index_name: str,
) -> Any:
    """Select the globally smallest deterministic row priorities.

    Taking the bottom ``capacity`` rows from each rank and applying this
    function again to their union is identical to selecting once from the
    complete candidate stream.  The result therefore does not depend on actor
    placement, arrival order, or batching.
    """

    import numpy as np
    import pyarrow as pa

    capacity = int(capacity)
    if capacity < 1:
        raise ValueError("GPU sort sample capacity must be positive.")
    nonempty = [sample for sample in samples if int(sample.num_rows) > 0]
    if not nonempty:
        return None
    table = nonempty[0] if len(nonempty) == 1 else pa.concat_tables(nonempty)
    if int(table.num_rows) <= capacity:
        return table
    priorities, blocks, strata, indices = _priorities(
        table,
        seed=seed,
        block_name=block_name,
        stratum_name=stratum_name,
        index_name=index_name,
    )
    # ``lexsort`` uses the last key as primary. Coordinates make the extremely
    # unlikely priority collision deterministic as well.
    order = np.lexsort((indices, strata, blocks, priorities))[:capacity]
    return table.take(pa.array(order, type=pa.int64()))


def finalize_priority_sample(
    samples: Iterable[Any],
    *,
    candidate_rows: int,
    capacity: int,
    seed: int,
    weight_name: str,
    block_name: str,
    stratum_name: str,
    index_name: str,
) -> Any:
    """Reduce rank reservoirs and apply inverse-inclusion row weights."""

    import numpy as np
    import pyarrow as pa

    candidate_rows = int(candidate_rows)
    selected = select_priority_sample(
        samples,
        capacity=capacity,
        seed=seed,
        block_name=block_name,
        stratum_name=stratum_name,
        index_name=index_name,
    )
    if selected is None:
        if candidate_rows:
            raise ValueError("GPU sort sample candidates are missing.")
        return None
    selected_rows = int(selected.num_rows)
    if not selected_rows or candidate_rows < selected_rows:
        raise ValueError("GPU sort sample candidate count is inconsistent.")
    if weight_name not in selected.column_names:
        raise ValueError(f"GPU sort sample is missing weight column {weight_name!r}.")
    if candidate_rows == selected_rows:
        return selected
    weights = (
        selected[weight_name]
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
        .astype(np.uint64, copy=False)
    )
    whole, remainder = divmod(candidate_rows, selected_rows)
    scaled = weights * np.uint64(whole)
    if remainder:
        scaled = scaled + (weights * np.uint64(remainder)) // np.uint64(selected_rows)
    column = selected.schema.get_field_index(weight_name)
    return selected.set_column(
        column,
        selected.schema.field(weight_name),
        pa.array(scaled, type=pa.uint64()),
    )
