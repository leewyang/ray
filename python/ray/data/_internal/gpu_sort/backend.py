"""Compact RAPIDS backend for a spillable distributed GPU range sort.

The controller admits exact destination bytes to synchronized RAPIDS-MPF rounds.
Destination ranks retain received tables while they fit their residency budget.
Crossing that watermark converts complete GPU-sorted tables to immutable Arrow
runs in a configured actor-local store; those runs are later merged with
bounded pylibcudf operations. The CPU orders only the bounded planning sample,
never dataset rows or runs.
"""

from __future__ import annotations

import gc
import hashlib
import os
import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from ray.data._internal.gpu_sort.config import (
    MPF_PROGRESS_RESERVE_BYTES,
    MPF_RECEIVE_BUFFER_OWNERS,
    GPUSortCapacityError,
    GPUSortConfig,
)
from ray.data._internal.gpu_sort.run_store import (
    LocalDiskRunStore,
    PlasmaRunStore,
    RunStore,
    _RunChunk,
)

GPU_SORT_PARTITION_ID_KEY = b"ray-data-gpu-sort-partition"
GPU_SORT_DIAGNOSTICS_KEY = b"ray-data-gpu-sort-diagnostics"

_WEIGHT_BASE = "__ray_gpu_sort_byte_weight"
_HIDDEN_BASE = "__ray_gpu_sort_cmp"
_SAMPLE_BLOCK_BASE = "__ray_gpu_sort_sample_block"
_SAMPLE_STRATUM_BASE = "__ray_gpu_sort_sample_stratum"
_SAMPLE_INDEX_BASE = "__ray_gpu_sort_sample_index"


@dataclass
class _ExternalRun:
    chunks: List[_RunChunk] = dc_field(default_factory=list)


@dataclass
class _PreparedBatch:
    chunks: Dict[int, Any]
    destination_bytes: Tuple[int, ...]


def _align_down_256(value: int) -> int:
    return (int(value) >> 8) << 8


def _private_name(base: str, names: List[str]) -> str:
    candidate = base
    suffix = 0
    while candidate in names:
        suffix += 1
        candidate = f"{base}_{suffix}"
    return candidate


def _safe_stat(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _safe_stat(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_stat(child) for child in value]
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _scale_sample_weights(weights: Any, population_rows: int, sample_rows: int) -> Any:
    """Apply inverse-inclusion weighting with integer-only arithmetic."""

    population_rows = int(population_rows)
    sample_rows = int(sample_rows)
    if sample_rows <= 0 or population_rows < sample_rows:
        raise ValueError("Sample rows must be in [1, population_rows].")
    if population_rows == sample_rows:
        return weights

    whole, remainder = divmod(population_rows, sample_rows)
    scaled = weights * whole
    if remainder:
        scaled = scaled + (weights * remainder) // sample_rows
    return scaled.astype("uint64")


def _stratified_sample_indices(
    population_rows: int,
    sample_rows: int,
    *,
    seed: int,
    block_ordinal: int,
) -> tuple[Any, Any]:
    """Select one deterministic PCG64 row from every equal integer stratum.

    The stream depends only on the user-controlled global seed and the logical
    input block ordinal.  It is therefore independent of Ray actor, node, GPU,
    communicator rank, and locality assignment.
    """

    import numpy as np

    population_rows = int(population_rows)
    sample_rows = int(sample_rows)
    block_ordinal = int(block_ordinal)
    if sample_rows <= 0 or population_rows < sample_rows:
        raise ValueError("Sample rows must be in [1, population_rows].")
    if block_ordinal < 0:
        raise ValueError("Logical block ordinals must be nonnegative.")
    if block_ordinal >= 1 << 64:
        raise ValueError("Logical block ordinals must fit in unsigned 64 bits.")
    if not 0 <= int(seed) < 1 << 64:
        raise ValueError("GPU sort sample seeds must fit in unsigned 64 bits.")

    if population_rows > np.iinfo(np.int64).max:
        raise ValueError("Arrow row indices must fit in signed 64 bits.")

    # Use Python integers for the boundary products. A vectorized uint64
    # ``j * population_rows`` can overflow for an otherwise valid large block.
    # This is only the bounded control-plane sample (normally 65K entries).
    boundaries = np.fromiter(
        (j * population_rows // sample_rows for j in range(sample_rows + 1)),
        dtype=np.int64,
        count=sample_rows + 1,
    )
    lows = boundaries[:-1]
    highs = boundaries[1:]
    widths = (highs - lows).astype(np.uint64)
    if np.any(widths == 0):
        raise RuntimeError("GPU sort sampling produced an empty stratum.")

    # SeedSequence accepts an integer vector without depending on process hash
    # randomization. Split the validated uint64 inputs into fixed 32-bit words
    # so no two supported seeds or ordinals alias the same PCG64 stream.
    seed_word = int(seed)
    ordinal_word = block_ordinal
    entropy = [
        seed_word & 0xFFFFFFFF,
        seed_word >> 32,
        ordinal_word & 0xFFFFFFFF,
        ordinal_word >> 32,
    ]
    raw = np.random.PCG64(np.random.SeedSequence(entropy)).random_raw(sample_rows)
    indices = lows.astype(np.uint64) + raw % widths
    return indices.astype(np.int64), widths.astype(np.uint64)


def _scale_sample_weights_by_stratum(weights: Any, stratum_widths: Any) -> Any:
    """Apply exact inverse-inclusion weights for unequal integer strata."""

    import numpy as np

    weights = np.asarray(weights, dtype=np.uint64)
    stratum_widths = np.asarray(stratum_widths, dtype=np.uint64)
    if weights.shape != stratum_widths.shape or np.any(stratum_widths == 0):
        raise ValueError("Every sampled row must have one nonempty stratum.")
    return np.multiply(weights, stratum_widths, dtype=np.uint64)


def _arrow_table_digest(table: Any) -> str:
    """Hash an Arrow table including its schema and ordered physical values."""

    import pyarrow as pa

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _sample_index_digest(
    sample: Any,
    *,
    block_name: str,
    stratum_name: str,
    index_name: str,
) -> str:
    """Hash canonical ``(block, stratum, row)`` sample coordinates."""

    import numpy as np

    columns = []
    for name in (block_name, stratum_name, index_name):
        columns.append(
            sample[name]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .astype("<u8", copy=False)
        )
    coordinates = np.column_stack(columns).astype("<u8", copy=False)
    return hashlib.sha256(coordinates.tobytes(order="C")).hexdigest()


def _sampled_arrow_row_weights(sampled: Any) -> Any:
    """Return decoded row weights for an already sampled Arrow table.

    Sampling before calling this helper is important: variable-width lengths
    are evaluated for only the control-plane rows, never for the full input.
    The accounting intentionally matches the GPU planner it replaces.
    """

    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    if not isinstance(sampled, pa.Table):
        raise TypeError("GPU sort sampling requires a PyArrow table.")
    fixed = 0
    variable: List[tuple[str, int]] = []
    for field in sampled.schema:
        typ = field.type
        if pa.types.is_string(typ) or pa.types.is_binary(typ):
            variable.append((field.name, 4))
        elif pa.types.is_large_string(typ) or pa.types.is_large_binary(typ):
            variable.append((field.name, 8))
        elif pa.types.is_boolean(typ):
            fixed += 1
        elif pa.types.is_fixed_size_binary(typ):
            fixed += int(typ.byte_width)
        elif hasattr(typ, "bit_width"):
            fixed += max(1, int(typ.bit_width) // 8)
        else:
            fixed += 8
        if field.nullable:
            fixed += 1

    weights = np.full(sampled.num_rows, max(1, fixed), dtype=np.uint64)
    for name, offset_width in variable:
        lengths = pc.fill_null(pc.binary_length(sampled[name]), 0).to_numpy(
            zero_copy_only=False
        )
        weights += lengths.astype(np.uint64, copy=False) + offset_width
    return weights


def _cpu_sample_boundaries(
    samples: List[Any],
    *,
    schema: Any,
    key_columns: List[str],
    ascending: List[bool],
    num_partitions: int,
    null_position: str,
    weight_name: str,
    sample_block_name: Optional[str] = None,
    sample_stratum_name: Optional[str] = None,
    sample_index_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Sort the small weighted planning sample and choose ordered ranges."""

    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    key_schema = pa.schema([schema.field(name) for name in key_columns])
    if not samples:
        boundaries = key_schema.empty_table()
        return {
            "boundaries": boundaries,
            "sample_rows": 0,
            "sample_bytes": 0,
            "planning_sample_bytes": 0,
            "boundary_sort_s": 0.0,
            "boundary_select_s": 0.0,
            "sample_index_digest": hashlib.sha256(b"").hexdigest(),
            "boundary_digest": _arrow_table_digest(boundaries),
        }

    sort_started = time.perf_counter()
    arrow = pa.concat_tables(samples)
    ordering_names = (sample_block_name, sample_stratum_name, sample_index_name)
    if all(ordering_names):
        missing = [name for name in ordering_names if name not in arrow.column_names]
        if missing:
            raise ValueError(
                "GPU sort samples are missing deterministic coordinates: " f"{missing}."
            )
        canonical_indices = pc.sort_indices(
            arrow,
            sort_keys=[
                (sample_block_name, "ascending"),
                (sample_stratum_name, "ascending"),
            ],
        )
        arrow = arrow.take(canonical_indices)
        index_digest = _sample_index_digest(
            arrow,
            block_name=sample_block_name,
            stratum_name=sample_stratum_name,
            index_name=sample_index_name,
        )
    else:
        index_digest = hashlib.sha256(b"").hexdigest()
    planning_arrow = arrow.select(key_columns + [weight_name])
    indices = pc.sort_indices(
        planning_arrow,
        sort_keys=[
            (name, "ascending" if direction else "descending")
            for name, direction in zip(key_columns, ascending)
        ],
        null_placement="at_start" if null_position == "first" else "at_end",
    )
    sorted_arrow = planning_arrow.take(indices)
    select_started = time.perf_counter()
    rows = int(sorted_arrow.num_rows)
    if num_partitions == 1 or rows == 0:
        boundaries = key_schema.empty_table()
    else:
        weights = (
            sorted_arrow[weight_name]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .astype(np.uint64, copy=False)
        )
        cumulative = np.cumsum(weights, dtype=np.uint64)
        total_weight = int(cumulative[-1])
        targets = np.asarray(
            [
                max(0, int(total_weight * part / num_partitions) - 1)
                for part in range(1, num_partitions)
            ],
            dtype=np.uint64,
        )
        positions = np.searchsorted(cumulative, targets, side="left")
        boundaries = sorted_arrow.select(key_columns).take(pa.array(positions))
    boundary_select_s = time.perf_counter() - select_started
    boundary_sort_s = time.perf_counter() - sort_started
    return {
        "boundaries": boundaries,
        "sample_rows": int(planning_arrow.num_rows),
        "sample_bytes": int(arrow.nbytes),
        "planning_sample_bytes": int(planning_arrow.nbytes),
        "boundary_sort_s": boundary_sort_s,
        "boundary_select_s": boundary_select_s,
        "sample_index_digest": index_digest,
        "boundary_digest": _arrow_table_digest(boundaries),
    }


def _sum_spill_bytes(value: Any, path: str = "") -> int:
    """Extract byte-valued spill counters from nested RAPIDS statistics."""

    if isinstance(value, Mapping):
        if (
            "spill" in path.lower()
            and "byte" in path.lower()
            and isinstance(value.get("value"), (int, float))
        ):
            return int(value["value"])
        return sum(
            _sum_spill_bytes(child, f"{path}.{key}") for key, child in value.items()
        )
    if (
        "spill" in path.lower()
        and "byte" in path.lower()
        and isinstance(value, (int, float))
    ):
        return int(value)
    return 0


def _workspace_bounded_payload_bytes(
    pool_max_bytes: int,
    current_allocated_bytes: int,
    workspace_factor: float,
    reserve_bytes: int,
) -> int:
    """Return payload that can be sorted without exceeding the RMM pool.

    ``workspace_factor`` includes the input payload itself.  The input is
    already reflected in ``current_allocated_bytes``. A bounded concatenate
    needs one additional payload, while sorting needs ``factor - 1``; the
    stricter of those two allocations must fit in the available headroom.
    """

    available = max(
        0,
        int(pool_max_bytes) - int(current_allocated_bytes) - max(0, int(reserve_bytes)),
    )
    additional_factor = max(1.0, float(workspace_factor) - 1.0)
    return max(0, int(available / additional_factor))


def _input_batch_target_bytes(
    configured_bytes: int,
    pool_max_bytes: int,
    current_allocated_bytes: int,
    workspace_factor: float,
) -> int:
    """Cap one Arrow-to-GPU batch by current allocator headroom."""

    available = max(0, int(pool_max_bytes) - int(current_allocated_bytes))
    memory_bound = int(available / float(workspace_factor))
    return max(1, min(int(configured_bytes), memory_bound))


def _largest_bounded_arrow_prefix(table: Any, target_bytes: int) -> int:
    """Return the largest nonempty row prefix within ``target_bytes``."""

    first_row_bytes = int(table.slice(0, 1).nbytes)
    if first_row_bytes > target_bytes:
        raise GPUSortCapacityError(
            "One Arrow input row exceeds the available GPU input-batch memory."
        )
    low = 1
    high = int(table.num_rows)
    while low < high:
        middle = (low + high + 1) // 2
        if int(table.slice(0, middle).nbytes) <= target_bytes:
            low = middle
        else:
            high = middle - 1
    return low


def _coalesce_arrow_inputs(tables: List[Any], target_bytes: int) -> Iterator[Any]:
    """Coalesce adjacent inputs into byte-bounded Arrow batches."""

    import pyarrow as pa

    if target_bytes <= 0:
        raise ValueError("GPU sort input batch target must be positive.")
    pending: List[Any] = []
    pending_bytes = 0
    for table in tables:
        remaining = table
        while remaining.num_rows:
            available = target_bytes - pending_bytes
            first_row_bytes = int(remaining.slice(0, 1).nbytes)
            if pending and first_row_bytes > available:
                yield pending[0] if len(pending) == 1 else pa.concat_tables(pending)
                pending = []
                pending_bytes = 0
                continue
            take_rows = (
                int(remaining.num_rows)
                if int(remaining.nbytes) <= available
                else _largest_bounded_arrow_prefix(remaining, available)
            )
            piece = remaining.slice(0, take_rows)
            pending.append(piece)
            pending_bytes += int(piece.nbytes)
            remaining = remaining.slice(take_rows)
            if pending_bytes >= target_bytes or remaining.num_rows:
                yield pending[0] if len(pending) == 1 else pa.concat_tables(pending)
                pending = []
                pending_bytes = 0
    if pending:
        yield pending[0] if len(pending) == 1 else pa.concat_tables(pending)


def _duplicate_boundary_cuts(
    lowers: List[int],
    uppers: List[int],
    group_spans: List[Tuple[int, int]],
    *,
    num_partitions: int,
    rows: int,
    phase: int = 0,
) -> List[int]:
    """Split equal-key intervals evenly across repeated adjacent ranges."""

    boundary_count = int(num_partitions) - 1
    if (
        boundary_count < 0
        or len(lowers) != boundary_count
        or len(uppers) != boundary_count
    ):
        raise ValueError("GPU sort boundary search results have the wrong size.")
    covered = [False] * boundary_count
    for start, end in group_spans:
        start, end = int(start), int(end)
        if not 0 <= start < end <= boundary_count:
            raise ValueError("GPU sort boundary group spans are invalid.")
        if any(covered[start:end]):
            raise ValueError("GPU sort boundary group spans overlap.")
        if any(
            lowers[index] != lowers[start] or uppers[index] != uppers[start]
            for index in range(start, end)
        ):
            raise ValueError("Equal GPU sort boundaries returned unequal intervals.")
        for boundary_index in range(start, end):
            covered[boundary_index] = True
    if not all(covered) and boundary_count:
        raise ValueError("GPU sort boundary groups do not cover every boundary.")

    # Arrow equality deliberately distinguishes NaN values, while the sort
    # comparator groups them into one category. GPU search intervals are the
    # authoritative comparator result, so coalesce adjacent boundaries that
    # returned the same interval. This is also harmless when a source run has
    # no rows between two distinct boundaries: the interval width is zero.
    effective_spans: List[Tuple[int, int]] = []
    effective_start = 0
    for index in range(1, boundary_count + 1):
        if index < boundary_count and (
            lowers[index],
            uppers[index],
        ) == (lowers[effective_start], uppers[effective_start]):
            continue
        if effective_start < index:
            effective_spans.append((effective_start, index))
        effective_start = index

    cuts = [0] * boundary_count
    for start, end in effective_spans:
        lower, upper = int(lowers[start]), int(uppers[start])
        divisions = end - start + 1
        base, remainder = divmod(upper - lower, divisions)
        counts = [base] * divisions
        remainder_start = (int(phase) + lower + start) % divisions
        for offset in range(remainder):
            counts[(remainder_start + offset) % divisions] += 1
        cumulative = lower
        for boundary_index, count in zip(range(start, end), counts):
            cumulative += count
            cuts[boundary_index] = cumulative
    if any(left > right for left, right in zip(cuts, cuts[1:])) or (
        cuts and (cuts[0] < 0 or cuts[-1] > int(rows))
    ):
        raise RuntimeError(f"GPU sort produced invalid contiguous range cuts: {cuts}.")
    return cuts


def lazy_load_backend() -> type[Any]:
    """Build the implementation only inside a one-GPU Ray actor."""

    from rapidsmpf.utils.ray_utils import BaseShufflingActor

    class GPURangeSortBackend(BaseShufflingActor):  # pragma: no cover - GPU only
        # RAPIDS-MPF encodes operation IDs as uint8. Each wave fully shuts its
        # shuffler down before the next one, so IDs can safely cycle.
        DATA_OP_BASE = 29

        def __init__(
            self,
            *,
            nranks: int,
            index: int,
            key_columns: List[str],
            ascending: List[bool],
            num_partitions: int,
            config: Dict[str, Any],
        ) -> None:
            super().__init__(nranks)
            if len(key_columns) != len(ascending):
                raise ValueError("GPU sort needs one direction per sort key.")
            self._index = int(index)
            self._key_columns = list(key_columns)
            self._ascending = [bool(value) for value in ascending]
            self._num_partitions = int(num_partitions)
            self._config = GPUSortConfig.from_actor_dict(config)

            self._arrow_schema = None
            self._work_schema = None
            self._column_names: List[str] = []
            self._work_names: List[str] = []
            self._float_hidden: Dict[str, tuple[str, str]] = {}
            self._merge_key_names: List[str] = []
            self._merge_key_indices: List[int] = []
            self._weight_name = ""
            self._sample_block_name = ""
            self._sample_stratum_name = ""
            self._sample_index_name = ""
            self._boundary_keys = None
            self._boundary_group_spans: List[Tuple[int, int]] = []

            self._mr = None
            self._buffer_resource = None
            self._statistics = None
            self._pool_max_bytes = 0
            self._total_vram_bytes = 0
            self._memory_budget_bytes = 0
            self._payload_limit_bytes = 0
            self._run_chunk_bytes = 0
            self._run_store: Optional[RunStore] = None

            self._device_tables: Dict[int, List[Any]] = {
                partition: [] for partition in range(index, num_partitions, nranks)
            }
            self._device_bytes: Dict[int, int] = {
                partition: 0 for partition in self._device_tables
            }
            self._runs: Dict[int, List[_ExternalRun]] = {
                partition: [] for partition in self._device_tables
            }
            # Source runs are full-key sorted during one-pass ingestion before
            # global range boundaries are known. They remain actor-local until
            # post-EOS redistribution commits their destination ranges.
            self._source_runs: Dict[int, _ExternalRun] = {}
            self._source_run_rows: Dict[int, int] = {}
            self._next_source_run_id = 0
            self._active_source_run_ids: List[int] = []
            self._active_source_wave_id: Optional[int] = None
            self._streaming_sample = None
            self._streaming_sample_candidate_rows = 0
            self._streaming_sampled_blocks = 0
            self._ingest_finalized = False
            self._row_ordinal = 0
            self._duplicate_kernel = None
            self._prepared_wave_id: Optional[int] = None
            self._prepared_batches: Dict[int, _PreparedBatch] = {}
            self._pending_arrow_owners: List[Any] = []
            self._pending_arrow_batches: Optional[Iterator[Any]] = None
            self._next_arrow_batch = None
            self._source_input_complete = False
            self._pending_inputs_sorted = False
            self._next_prepared_batch_id = 0
            self._started_at = time.perf_counter()
            self._ray_spill_start = 0

            self._stats: Dict[str, Any] = {
                "rank": index,
                "state": "DEVICE_ACCUMULATING",
                "mode": "resident",
                "memory_budget_bytes": 0,
                "peak_device_bytes": 0,
                "peak_live_bytes": 0,
                "input_bytes": 0,
                "input_rows": 0,
                "input_block_count": 0,
                "ingest_rpc_count": 0,
                "source_run_count": 0,
                "source_run_bytes": 0,
                "source_run_rows": 0,
                "redistributed_run_count": 0,
                "redistributed_run_bytes": 0,
                "redistributed_run_rows": 0,
                "first_ingest_started_at_ns": None,
                "first_run_committed_at_ns": None,
                "last_run_committed_at_ns": None,
                "ingest_wall_s": 0.0,
                "streaming_sample_construction_s": 0.0,
                "input_batch_requested_bytes": self._config.exchange_batch_bytes,
                "input_batch_count": 0,
                "min_input_batch_target_bytes": 0,
                "max_input_batch_target_bytes": 0,
                "max_input_batch_bytes": 0,
                "prepared_batch_count": 0,
                "prepared_bytes": 0,
                "preparation_retry_count": 0,
                "exchange_subround_count": 0,
                "forced_subround_run_count": 0,
                "output_bytes": 0,
                "externalized_bytes": 0,
                "externalized_rows": 0,
                "first_externalize_s": None,
                "first_externalize_wave": None,
                "initial_run_count": 0,
                "merge_pass_count": 0,
                "replacement_run_count": 0,
                "direct_final_merge_count": 0,
                "resident_final_fallback_count": 0,
                "concat_oom_fallback_count": 0,
                "run_sort_oom_retry_count": 0,
                "h2d_bytes": 0,
                "planning_h2d_bytes": 0,
                "d2h_bytes": 0,
                "run_store": self._config.external_run_store,
                "plasma_read_bytes": 0,
                "plasma_write_bytes": 0,
                "plasma_intermediate_read_bytes": 0,
                "plasma_intermediate_write_bytes": 0,
                "plasma_intermediate_read_calls": 0,
                "plasma_intermediate_write_calls": 0,
                "plasma_intermediate_read_s": 0.0,
                "plasma_intermediate_write_s": 0.0,
                "plasma_output_write_bytes": 0,
                "plasma_output_write_calls": 0,
                "plasma_output_write_s": 0.0,
                "local_run_write_bytes": 0,
                "local_run_read_bytes": 0,
                "local_run_physical_write_bytes": 0,
                "local_run_physical_read_bytes": 0,
                "local_run_write_calls": 0,
                "local_run_read_calls": 0,
                "local_run_write_s": 0.0,
                "local_run_read_s": 0.0,
                "local_run_restore_s": 0.0,
                "local_run_live_bytes": 0,
                "local_run_peak_bytes": 0,
                "local_run_live_files": 0,
                "local_run_peak_files": 0,
                "local_run_write_errors": 0,
                "local_run_read_errors": 0,
                "local_run_cleanup_errors": 0,
                "local_run_cleanup_pending_files": 0,
                "mpf_host_spill_bytes": 0,
                "ray_disk_spill_bytes": 0,
                "cpu_sort_rows": 0,
                "cpu_merge_rows": 0,
                "fallback_count": 0,
                "phases_s": {
                    "sampling": 0.0,
                    "partition": 0.0,
                    "mpf_shuffle": 0.0,
                    "run_sort": 0.0,
                    "gpu_merge": 0.0,
                    "arrow_conversion": 0.0,
                    "plasma_seal": 0.0,
                    "orchestration": 0.0,
                },
            }

        # -- setup and schemas -------------------------------------------

        def setup_worker(self, root_address_bytes: bytes) -> Dict[str, Any]:
            import rmm
            from rapidsmpf.memory.buffer_resource import BufferResource
            from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
            from rapidsmpf.statistics import Statistics

            started = time.perf_counter()
            super().setup_worker(root_address_bytes)
            # Communicator ranks are assigned during bootstrap and need not
            # match Ray actor creation order. Partition ownership must follow
            # the actual MPF rank.
            actual_rank = int(self.rank())
            self._device_tables = {
                partition: []
                for partition in range(actual_rank, self._num_partitions, self.nranks())
            }
            self._device_bytes = {partition: 0 for partition in self._device_tables}
            self._runs = {partition: [] for partition in self._device_tables}
            self._stats["rank"] = actual_rank
            free_bytes, total_bytes = rmm.mr.available_device_memory()
            maximum = _align_down_256(
                min(
                    int(total_bytes * self._config.rmm_max_fraction),
                    int(free_bytes * 0.95),
                )
            )
            initial = _align_down_256(
                min(maximum, int(total_bytes * self._config.rmm_initial_fraction))
            )
            if initial <= 0 or maximum <= 0:
                raise GPUSortCapacityError("GPU sort could not reserve an RMM pool.")
            pool = rmm.mr.PoolMemoryResource(
                rmm.mr.CudaMemoryResource(),
                initial_pool_size=initial,
                maximum_pool_size=maximum,
            )
            self._mr = RmmResourceAdaptor(pool)
            rmm.mr.set_current_device_resource(self._mr)
            self._statistics = Statistics(enable=True, mr=self._mr)
            # The controller reserves device headroom so MPF should not need
            # its host pressure-spill path. Explicit Arrow runs are the sort's
            # capacity mechanism, and diagnostics surface any MPF host spill.
            self._buffer_resource = BufferResource(
                device_mr=self._mr,
                pinned_mr=None,
                memory_available=None,
                statistics=self._statistics,
            )
            self._pool_max_bytes = maximum
            self._total_vram_bytes = int(total_bytes)
            requested = self._config.residency_budget_bytes
            self._memory_budget_bytes = min(int(requested or maximum), maximum)
            if self._memory_budget_bytes < 16 << 20:
                raise GPUSortCapacityError(
                    "GPU sort residency budget must be at least 16 MiB."
                )
            self._stats["memory_budget_bytes"] = self._memory_budget_bytes
            self._initialize_run_limits()
            try:
                import cudf

                cudf.set_option("spill", False)
                cudf.set_option("spill_on_demand", False)
                cudf.set_option("spill_device_limit", None)
            except (KeyError, ValueError):
                pass
            if self._config.external_run_store == "local_disk":
                actor_directory = os.path.join(
                    str(self._config.external_run_directory),
                    str(self._config.external_run_id),
                    f"rank-{actual_rank}",
                )
                self._run_store = LocalDiskRunStore(
                    actor_directory,
                    min_free_bytes=self._config.external_run_min_free_bytes,
                    max_live_bytes=self._config.external_run_max_live_bytes,
                )
                self._stats["local_run_directory"] = actor_directory
            else:
                self._run_store = PlasmaRunStore()
            self._ray_spill_start = self._ray_spilled_bytes()
            self._stats["phases_s"]["orchestration"] += time.perf_counter() - started
            self._update_peak()
            return {
                "rank": self.rank(),
                "pool_max_bytes": maximum,
                "memory_budget_bytes": self._memory_budget_bytes,
            }

        def is_ready(self) -> bool:
            return self.is_initialized() and self._buffer_resource is not None

        def _initialize_run_limits(self) -> None:
            """Set run-sort and merge bounds before streaming input arrives."""

            self._payload_limit_bytes = max(
                4 << 20,
                int(
                    self._memory_budget_bytes / self._config.final_sort_workspace_factor
                ),
            )
            merge_bound = max(
                1 << 20,
                int(
                    self._memory_budget_bytes
                    / (
                        2
                        * self._config.merge_fan_in
                        * self._config.merge_workspace_factor
                    )
                ),
            )
            self._run_chunk_bytes = min(self._config.run_chunk_bytes, merge_bound)

        def _set_schema(self, schema: Any) -> None:
            import pyarrow as pa

            schema = getattr(schema, "base_schema", schema)
            if not isinstance(schema, pa.Schema):
                raise TypeError("GPU sort requires a PyArrow schema.")
            if self._arrow_schema is not None:
                if not self._arrow_schema.equals(schema, check_metadata=False):
                    raise TypeError("GPU sort input blocks do not share one schema.")
                return
            self._arrow_schema = schema
            self._column_names = list(schema.names)
            missing = [name for name in self._key_columns if name not in schema.names]
            if missing:
                raise ValueError(f"GPU sort key columns are missing: {missing}.")
            names = list(schema.names)
            self._weight_name = _private_name(_WEIGHT_BASE, names)
            names.append(self._weight_name)
            self._sample_block_name = _private_name(_SAMPLE_BLOCK_BASE, names)
            names.append(self._sample_block_name)
            self._sample_stratum_name = _private_name(_SAMPLE_STRATUM_BASE, names)
            names.append(self._sample_stratum_name)
            self._sample_index_name = _private_name(_SAMPLE_INDEX_BASE, names)
            work_fields = list(schema)
            merge_names: List[str] = []
            for key in self._key_columns:
                if pa.types.is_floating(schema.field(key).type):
                    null_name = _private_name(f"{_HIDDEN_BASE}_null", names)
                    names.append(null_name)
                    nan_name = _private_name(f"{_HIDDEN_BASE}_nan", names)
                    names.append(nan_name)
                    self._float_hidden[key] = (null_name, nan_name)
                    work_fields.extend(
                        [
                            pa.field(null_name, pa.bool_()),
                            pa.field(nan_name, pa.bool_(), nullable=True),
                        ]
                    )
                    merge_names.extend([null_name, nan_name])
                merge_names.append(key)
            self._work_schema = pa.schema(work_fields)
            self._work_names = list(self._work_schema.names)
            self._merge_key_names = merge_names
            self._merge_key_indices = [
                self._work_names.index(name) for name in merge_names
            ]

        def _to_arrow_table(self, block: Any):
            import pyarrow as pa

            if isinstance(block, pa.Table):
                return block
            from ray.data.block import BlockAccessor

            return BlockAccessor.for_block(block).to_arrow()

        def _augment_table(self, table: Any, names: List[str]):
            """Append physical float-category columns used by every comparator."""

            import pylibcudf as plc

            columns = list(table.columns())
            output_names = list(names)
            for key in self._key_columns:
                hidden = self._float_hidden.get(key)
                if hidden is None:
                    continue
                column = table.columns()[names.index(key)]
                columns.extend([plc.unary.is_null(column), plc.unary.is_nan(column)])
                output_names.extend(hidden)
            return plc.Table(columns), output_names

        def _comparison_table(self, table: Any, names: List[str]):
            import pylibcudf as plc

            columns = [
                table.columns()[names.index(name)] for name in self._merge_key_names
            ]
            return plc.Table(columns)

        def _order_and_nulls(self) -> tuple[List[Any], List[Any]]:
            import pylibcudf as plc

            order: List[Any] = []
            nulls: List[Any] = []

            def null_order(ascending: bool) -> Any:
                # libcudf applies BEFORE/AFTER in the column's natural
                # ascending order, then reverses it for DESCENDING. Choose the
                # physical enum that preserves Arrow's direction-independent
                # null placement.
                null_first = self._config.null_position == "first"
                return (
                    plc.types.NullOrder.BEFORE
                    if null_first == bool(ascending)
                    else plc.types.NullOrder.AFTER
                )

            for key, ascending in zip(self._key_columns, self._ascending):
                if key in self._float_hidden:
                    category_ascending = self._config.null_position == "last"
                    category_order = (
                        plc.types.Order.ASCENDING
                        if category_ascending
                        else plc.types.Order.DESCENDING
                    )
                    category_nulls = null_order(category_ascending)
                    order.extend([category_order, category_order])
                    nulls.extend([category_nulls, category_nulls])
                value_order = (
                    plc.types.Order.ASCENDING
                    if ascending
                    else plc.types.Order.DESCENDING
                )
                value_nulls = null_order(ascending)
                order.append(value_order)
                nulls.append(value_nulls)
            return order, nulls

        # -- bounded GPU sampling and boundary selection -----------------

        def sample_blocks(
            self,
            blocks: Iterable[Any],
            *,
            block_ordinals: List[int],
            sample_quotas: List[int],
            seed: int,
        ) -> Dict[str, Any]:
            import numpy as np
            import pyarrow as pa

            if len(block_ordinals) != len(sample_quotas):
                raise ValueError("GPU sort sampling plan does not match its blocks.")
            block_iterator = iter(blocks)
            started = time.perf_counter()
            samples = []
            total_rows = 0
            total_bytes = 0
            sampled_blocks = 0
            for block_ordinal, quota in zip(block_ordinals, sample_quotas):
                try:
                    block = next(block_iterator)
                except StopIteration as exc:
                    raise ValueError(
                        "GPU sort sampling plan does not match its blocks."
                    ) from exc
                arrow = self._to_arrow_table(block)
                self._set_schema(arrow.schema)
                rows = int(arrow.num_rows)
                total_rows += rows
                total_bytes += int(arrow.nbytes)
                if rows == 0:
                    if int(quota) != 0:
                        raise ValueError("Empty GPU sort blocks cannot be sampled.")
                    continue
                take = int(quota)
                if not 1 <= take <= rows:
                    raise ValueError(
                        "Every nonempty GPU sort block needs between one and "
                        "all of its rows sampled."
                    )
                sampled_blocks += 1
                indices, stratum_widths = _stratified_sample_indices(
                    rows,
                    take,
                    seed=seed,
                    block_ordinal=int(block_ordinal),
                )
                sampled = arrow.take(pa.array(indices, type=pa.int64()))
                weights = _scale_sample_weights_by_stratum(
                    _sampled_arrow_row_weights(sampled), stratum_widths
                )
                selected = sampled.select(self._key_columns)
                selected = (
                    selected.append_column(
                        self._weight_name, pa.array(weights, type=pa.uint64())
                    )
                    .append_column(
                        self._sample_block_name,
                        pa.array(
                            np.full(take, int(block_ordinal), dtype=np.uint64),
                            type=pa.uint64(),
                        ),
                    )
                    .append_column(
                        self._sample_stratum_name,
                        pa.array(np.arange(take, dtype=np.uint64), type=pa.uint64()),
                    )
                    .append_column(
                        self._sample_index_name,
                        pa.array(indices.astype(np.uint64), type=pa.uint64()),
                    )
                )
                sample_schema = pa.schema(
                    [self._arrow_schema.field(name) for name in self._key_columns]
                    + [
                        pa.field(self._weight_name, pa.uint64(), nullable=False),
                        pa.field(self._sample_block_name, pa.uint64(), nullable=False),
                        pa.field(
                            self._sample_stratum_name, pa.uint64(), nullable=False
                        ),
                        pa.field(self._sample_index_name, pa.uint64(), nullable=False),
                    ]
                )
                samples.append(selected.cast(sample_schema))
            exhausted = object()
            if next(block_iterator, exhausted) is not exhausted:
                raise ValueError("GPU sort sampling plan does not match its blocks.")
            sample = pa.concat_tables(samples) if samples else None
            self._stats["input_rows"] = total_rows
            self._stats["input_bytes"] = total_bytes
            elapsed = time.perf_counter() - started
            self._stats["phases_s"]["sampling"] += elapsed
            self._update_peak()
            return {
                "rank": self.rank(),
                "schema": self._arrow_schema,
                "sample": sample,
                "rows": total_rows,
                "input_bytes": total_bytes,
                "sample_rows": 0 if sample is None else int(sample.num_rows),
                "sample_bytes": 0 if sample is None else int(sample.nbytes),
                "sampled_block_count": sampled_blocks,
                "cpu_sample_construction_s": elapsed,
                "planning_h2d_bytes": 0,
            }

        def _streaming_sample_for_arrow(self, arrow: Any, *, block_ordinal: int) -> Any:
            """Build one bounded, weighted sample without replaying a block."""

            import numpy as np
            import pyarrow as pa

            rows = int(arrow.num_rows)
            if rows == 0:
                return None
            take = min(rows, int(self._config.streaming_sample_rows_per_block))
            indices, stratum_widths = _stratified_sample_indices(
                rows,
                take,
                seed=int(self._config.sample_seed),
                block_ordinal=int(block_ordinal),
            )
            sampled = arrow.take(pa.array(indices, type=pa.int64()))
            weights = _scale_sample_weights_by_stratum(
                _sampled_arrow_row_weights(sampled), stratum_widths
            )
            selected = sampled.select(self._key_columns)
            selected = (
                selected.append_column(
                    self._weight_name, pa.array(weights, type=pa.uint64())
                )
                .append_column(
                    self._sample_block_name,
                    pa.array(
                        np.full(take, int(block_ordinal), dtype=np.uint64),
                        type=pa.uint64(),
                    ),
                )
                .append_column(
                    self._sample_stratum_name,
                    pa.array(np.arange(take, dtype=np.uint64), type=pa.uint64()),
                )
                .append_column(
                    self._sample_index_name,
                    pa.array(indices.astype(np.uint64), type=pa.uint64()),
                )
            )
            sample_schema = pa.schema(
                [self._arrow_schema.field(name) for name in self._key_columns]
                + [
                    pa.field(self._weight_name, pa.uint64(), nullable=False),
                    pa.field(self._sample_block_name, pa.uint64(), nullable=False),
                    pa.field(self._sample_stratum_name, pa.uint64(), nullable=False),
                    pa.field(self._sample_index_name, pa.uint64(), nullable=False),
                ]
            )
            return selected.cast(sample_schema)

        def _commit_source_arrow(self, arrow: Any) -> List[int]:
            """GPU-sort one Arrow block into transactional full-key source runs."""

            import cupy as cp
            import cudf
            from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table

            committed: List[int] = []
            remaining = arrow
            try:
                while remaining is not None and int(remaining.num_rows):
                    target = min(
                        int(self._config.exchange_batch_bytes),
                        int(self._payload_limit_bytes),
                    )
                    piece, remainder = self._bounded_arrow_prefix(remaining, target)
                    while True:
                        frame = None
                        table = None
                        sorted_table = None
                        sort_started = None
                        try:
                            arrow_bytes = int(piece.nbytes)
                            frame = cudf.DataFrame.from_arrow(piece)
                            self._stats["h2d_bytes"] += arrow_bytes
                            table = cudf_to_pylibcudf_table(frame)
                            table, _ = self._augment_table(table, list(frame.columns))
                            sort_started = time.perf_counter()
                            sorted_table = self._sort_table(table)
                            cp.cuda.runtime.deviceSynchronize()
                            self._stats["phases_s"]["run_sort"] += (
                                time.perf_counter() - sort_started
                            )
                            sort_started = None
                            run = self._store_table_as_run(sorted_table, initial=True)
                            run_id = self._next_source_run_id
                            self._next_source_run_id += 1
                            self._source_runs[run_id] = run
                            self._source_run_rows[run_id] = int(piece.num_rows)
                            committed.append(run_id)
                            now_ns = time.time_ns()
                            if self._stats["first_run_committed_at_ns"] is None:
                                self._stats["first_run_committed_at_ns"] = now_ns
                                self._stats["first_externalize_s"] = (
                                    time.perf_counter() - self._started_at
                                )
                                self._stats["first_externalize_wave"] = -1
                            self._stats["last_run_committed_at_ns"] = now_ns
                            break
                        except MemoryError as exc:
                            if sort_started is not None:
                                self._stats["phases_s"]["run_sort"] += max(
                                    0.0, time.perf_counter() - sort_started
                                )
                            self._stats["run_sort_oom_retry_count"] += 1
                            frame = table = sorted_table = None
                            gc.collect()
                            cp.cuda.runtime.deviceSynchronize()
                            rows = int(piece.num_rows)
                            if rows <= 1:
                                raise GPUSortCapacityError(
                                    "One GPU sort source row cannot fit in the RMM pool."
                                ) from exc
                            take = max(1, rows // 2)
                            piece = remaining.slice(0, take)
                            remainder = remaining.slice(take)
                            continue
                        finally:
                            frame = table = sorted_table = None
                    remaining = remainder
                return committed
            except BaseException:
                for run_id in committed:
                    run = self._source_runs.pop(run_id, None)
                    self._source_run_rows.pop(run_id, None)
                    if run is not None:
                        self._release_run_chunks(run.chunks)
                        run.chunks.clear()
                raise

        def ingest_blocks(
            self,
            blocks: Iterable[Any],
            *,
            block_ordinals: List[int],
        ) -> Dict[str, Any]:
            """Consume, sample, and durably externalize one bounded input RPC."""

            from ray.data._internal.gpu_sort.streaming_sample import (
                select_priority_sample,
            )

            if self._ingest_finalized:
                raise RuntimeError("GPU sort input was already finalized.")
            block_iterator = iter(blocks)
            created_ids: List[int] = []
            samples = []
            rows = 0
            input_bytes = 0
            nonempty_blocks = 0
            if self._stats["first_ingest_started_at_ns"] is None:
                self._stats["first_ingest_started_at_ns"] = time.time_ns()
            started = time.perf_counter()
            sample_construction_s = 0.0
            try:
                for block_ordinal in block_ordinals:
                    try:
                        block = next(block_iterator)
                    except StopIteration as exc:
                        raise ValueError(
                            "GPU sort ingestion ordinals do not match its blocks."
                        ) from exc
                    arrow = self._to_arrow_table(block)
                    self._set_schema(arrow.schema)
                    block_rows = int(arrow.num_rows)
                    rows += block_rows
                    input_bytes += int(arrow.nbytes)
                    sample_started = time.perf_counter()
                    sample = self._streaming_sample_for_arrow(
                        arrow, block_ordinal=int(block_ordinal)
                    )
                    sample_construction_s += time.perf_counter() - sample_started
                    if sample is not None:
                        samples.append(sample)
                        nonempty_blocks += 1
                    created_ids.extend(self._commit_source_arrow(arrow))
                exhausted = object()
                if next(block_iterator, exhausted) is not exhausted:
                    raise ValueError(
                        "GPU sort ingestion ordinals do not match its blocks."
                    )

                if samples:
                    sample_started = time.perf_counter()
                    candidate_rows = sum(int(sample.num_rows) for sample in samples)
                    self._streaming_sample_candidate_rows += candidate_rows
                    combined = [self._streaming_sample, *samples]
                    self._streaming_sample = select_priority_sample(
                        [sample for sample in combined if sample is not None],
                        capacity=int(self._config.sample_size),
                        seed=int(self._config.sample_seed),
                        block_name=self._sample_block_name,
                        stratum_name=self._sample_stratum_name,
                        index_name=self._sample_index_name,
                    )
                    self._streaming_sampled_blocks += nonempty_blocks
                    sample_construction_s += time.perf_counter() - sample_started
                committed_bytes = sum(
                    chunk.size_bytes
                    for run_id in created_ids
                    for chunk in self._source_runs[run_id].chunks
                )
                self._stats["input_rows"] += rows
                self._stats["input_bytes"] += input_bytes
                self._stats["input_block_count"] += len(block_ordinals)
                self._stats["ingest_rpc_count"] += 1
                self._stats["source_run_count"] += len(created_ids)
                self._stats["source_run_bytes"] += committed_bytes
                self._stats["source_run_rows"] += rows
                elapsed = time.perf_counter() - started
                self._stats["ingest_wall_s"] += elapsed
                self._stats["streaming_sample_construction_s"] += sample_construction_s
                self._stats["phases_s"]["sampling"] += sample_construction_s
                self._update_peak()
                return {
                    "rank": self.rank(),
                    "rows": rows,
                    "input_bytes": input_bytes,
                    "input_blocks": len(block_ordinals),
                    "source_runs": len(created_ids),
                    "source_run_bytes": committed_bytes,
                    "first_run_committed_at_ns": self._stats[
                        "first_run_committed_at_ns"
                    ],
                    "last_run_committed_at_ns": self._stats["last_run_committed_at_ns"],
                }
            except BaseException:
                for run_id in created_ids:
                    run = self._source_runs.pop(run_id, None)
                    self._source_run_rows.pop(run_id, None)
                    if run is not None:
                        self._release_run_chunks(run.chunks)
                        run.chunks.clear()
                raise

        def finalize_ingest(self) -> Dict[str, Any]:
            """Return the bounded reservoir and source-run geometry at EOS."""

            self._ingest_finalized = True
            return {
                "rank": self.rank(),
                "schema": self._arrow_schema,
                "sample": self._streaming_sample,
                "sample_candidate_rows": self._streaming_sample_candidate_rows,
                "sampled_block_count": self._streaming_sampled_blocks,
                "runs": [
                    {
                        "run_id": run_id,
                        "rows": int(self._source_run_rows[run_id]),
                        "bytes": sum(
                            int(chunk.size_bytes)
                            for chunk in self._source_runs[run_id].chunks
                        ),
                    }
                    for run_id in sorted(self._source_runs)
                ],
            }

        def compute_boundaries(self, samples: List[Any], schema: Any) -> Dict[str, Any]:
            started = time.perf_counter()
            self._set_schema(schema)
            result = _cpu_sample_boundaries(
                samples,
                schema=self._arrow_schema,
                key_columns=self._key_columns,
                ascending=self._ascending,
                num_partitions=self._num_partitions,
                null_position=self._config.null_position,
                weight_name=self._weight_name,
                sample_block_name=self._sample_block_name,
                sample_stratum_name=self._sample_stratum_name,
                sample_index_name=self._sample_index_name,
            )
            self._stats["phases_s"]["sampling"] += time.perf_counter() - started
            self._update_peak()
            return {
                "rank": self.rank(),
                **result,
                "planning_h2d_bytes": 0,
            }

        def install_plan(self, schema: Any, boundaries: Any) -> Dict[str, Any]:
            import cudf
            from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table

            self._set_schema(schema)
            boundary_bytes = int(boundaries.nbytes)
            boundary_frame = cudf.DataFrame.from_arrow(boundaries)
            self._stats["h2d_bytes"] += boundary_bytes
            self._stats["planning_h2d_bytes"] += boundary_bytes
            boundary_table = cudf_to_pylibcudf_table(boundary_frame)
            boundary_table, names = self._augment_table(
                boundary_table, list(boundary_frame.columns)
            )
            self._boundary_keys = self._comparison_table(boundary_table, names)
            self._boundary_group_spans = []
            group_start = 0
            for index in range(1, int(boundaries.num_rows) + 1):
                if index < int(boundaries.num_rows) and boundaries.slice(
                    group_start, 1
                ).equals(boundaries.slice(index, 1)):
                    continue
                if group_start < index:
                    self._boundary_group_spans.append((group_start, index))
                group_start = index
            self._initialize_run_limits()
            return {
                "rank": self.rank(),
                "payload_limit_bytes": self._payload_limit_bytes,
                "run_chunk_bytes": self._run_chunk_bytes,
                "boundary_rows": int(boundaries.num_rows),
            }

        # -- synchronized range shuffle waves ----------------------------

        @staticmethod
        def _slice_table(table: Any, start: int, end: int):
            import pylibcudf as plc

            if start == 0 and end == table.num_rows():
                return table
            return plc.copying.slice(table, [int(start), int(end)])[0]

        def _table_bytes(self, table: Any) -> int:
            from rapidsmpf.utils.cudf import pylibcudf_to_cudf_dataframe

            if table is None or table.num_rows() == 0:
                return 0
            frame = pylibcudf_to_cudf_dataframe(table, self._work_names)
            return int(frame.memory_usage(index=False, deep=True).sum())

        def _sort_table(self, table: Any):
            import pylibcudf as plc

            order, nulls = self._order_and_nulls()
            if len(self._work_names) == 1 and self._merge_key_indices == [0]:
                return plc.sorting.sort(table, order, nulls)
            keys = self._comparison_table(table, self._work_names)
            return plc.sorting.sort_by_key(table, keys, order, nulls)

        def _partition_and_pack(self, frame: Any, wave_id: int):
            import cupy as cp
            import cudf
            import pylibcudf as plc
            from rapidsmpf.integrations.cudf.partition import split_and_pack
            from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table
            from rmm.pylibrmm.stream import DEFAULT_STREAM

            table = cudf_to_pylibcudf_table(frame)
            table, names = self._augment_table(table, list(frame.columns))
            rows = int(table.num_rows())
            if rows == 0:
                return {}, (0,) * self._num_partitions
            keys = self._comparison_table(table, names)
            order, nulls = self._order_and_nulls()
            lower_column = plc.search.lower_bound(
                self._boundary_keys, keys, order, nulls
            )
            upper_column = plc.search.upper_bound(
                self._boundary_keys, keys, order, nulls
            )
            lower = cudf.Series.from_pylibcudf(lower_column).values
            upper = cudf.Series.from_pylibcudf(upper_column).values
            if self._duplicate_kernel is None:
                self._duplicate_kernel = cp.ElementwiseKernel(
                    "int32 lower, int32 upper, uint64 base",
                    "int32 destination",
                    """
                    const unsigned long long width =
                        static_cast<unsigned long long>(upper - lower + 1);
                    destination = lower + static_cast<int>(
                        (base + static_cast<unsigned long long>(i)) % width);
                    """,
                    "ray_data_gpu_sort_duplicate_range",
                )
            ordinal = (
                (int(wave_id) << 48) + (int(self.rank()) << 40) + int(self._row_ordinal)
            )
            destinations = self._duplicate_kernel(
                lower, upper, cp.uint64(ordinal & ((1 << 64) - 1))
            )
            self._row_ordinal += rows
            partition_map, _ = cudf.Series(destinations).to_pylibcudf()
            partitioned, offsets = plc.partitioning.partition(
                table,
                partition_map,
                self._num_partitions,
            )
            normalized = [int(value) for value in offsets]
            if not normalized or normalized[0] != 0 or normalized[-1] != rows:
                raise RuntimeError(
                    "libcudf returned invalid GPU range-partition offsets: "
                    f"{normalized}."
                )
            destination_bytes = []
            for start, end in zip(normalized[:-1], normalized[1:]):
                if start == end:
                    destination_bytes.append(0)
                    continue
                view = self._slice_table(partitioned, start, end)
                packer = plc.contiguous_split.ChunkedPack.create(
                    view, 1 << 20, DEFAULT_STREAM
                )
                destination_bytes.append(int(packer.get_total_contiguous_size()))
                del packer, view
            packed = split_and_pack(
                partitioned,
                normalized[1:-1],
                DEFAULT_STREAM,
                self._buffer_resource,
            )
            return packed, tuple(destination_bytes)

        def _partition_sorted_and_pack(self, frame: Any, wave_id: int):
            """Split one sorted source run into contiguous ordered ranges."""

            import cudf
            import pylibcudf as plc
            from rapidsmpf.integrations.cudf.partition import split_and_pack
            from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table
            from rmm.pylibrmm.stream import DEFAULT_STREAM

            table = cudf_to_pylibcudf_table(frame)
            names = list(frame.columns)
            if names != self._work_names:
                raise TypeError(
                    "GPU sort restored source runs do not match the work schema."
                )
            rows = int(table.num_rows())
            if rows == 0:
                return {}, (0,) * self._num_partitions
            keys = self._comparison_table(table, names)
            order, nulls = self._order_and_nulls()
            lower_column = plc.search.lower_bound(
                keys, self._boundary_keys, order, nulls
            )
            upper_column = plc.search.upper_bound(
                keys, self._boundary_keys, order, nulls
            )
            lowers = [
                int(value)
                for value in cudf.Series.from_pylibcudf(lower_column)
                .to_arrow()
                .to_pylist()
            ]
            uppers = [
                int(value)
                for value in cudf.Series.from_pylibcudf(upper_column)
                .to_arrow()
                .to_pylist()
            ]
            cuts = _duplicate_boundary_cuts(
                lowers,
                uppers,
                self._boundary_group_spans,
                num_partitions=self._num_partitions,
                rows=rows,
                phase=(int(wave_id) * 0x9E3779B1)
                + int(self.rank())
                + int(self._row_ordinal),
            )
            self._row_ordinal += rows
            offsets = [0, *cuts, rows]
            destination_bytes = []
            for start, end in zip(offsets[:-1], offsets[1:]):
                if start == end:
                    destination_bytes.append(0)
                    continue
                view = self._slice_table(table, start, end)
                packer = plc.contiguous_split.ChunkedPack.create(
                    view, 1 << 20, DEFAULT_STREAM
                )
                destination_bytes.append(int(packer.get_total_contiguous_size()))
                del packer, view
            packed = split_and_pack(
                table,
                cuts,
                DEFAULT_STREAM,
                self._buffer_resource,
            )
            return packed, tuple(destination_bytes)

        def _unpack_one(self, chunk: Any):
            from rapidsmpf.integrations.cudf.partition import (
                unpack_and_concat,
                unspill_partitions,
            )
            from rmm.pylibrmm.stream import DEFAULT_STREAM

            device = unspill_partitions(
                [chunk],
                br=self._buffer_resource,
                allow_overbooking=False,
                statistics=self._statistics,
            )
            return unpack_and_concat(device, DEFAULT_STREAM, self._buffer_resource)

        def _externalize_before_next_wave(self, wave_id: int) -> None:
            """Free retained range tails before the next bounded shuffle."""

            if wave_id <= 0:
                return
            for partition in sorted(self._device_tables):
                if self._device_tables[partition]:
                    self._externalize_device_tables(partition, wave_id)

        def _estimated_device_bytes(self, arrow: Any) -> int:
            """Estimate cuDF payload bytes, including Arrow representation gaps."""

            import pyarrow as pa

            rows = int(arrow.num_rows)
            estimate = int(arrow.nbytes)
            for field in arrow.schema:
                if pa.types.is_boolean(field.type):
                    estimate += max(0, rows - (rows + 7) // 8)
            estimate += 2 * rows * len(self._float_hidden)
            return max(1, estimate)

        def _bounded_arrow_prefix(self, arrow: Any, target_device_bytes: int):
            """Split one actual Arrow batch by its estimated device footprint."""

            rows = int(arrow.num_rows)
            if rows == 0:
                return arrow, None
            if self._estimated_device_bytes(arrow) <= target_device_bytes:
                return arrow, None
            if self._estimated_device_bytes(arrow.slice(0, 1)) > target_device_bytes:
                raise GPUSortCapacityError(
                    "One Arrow row cannot fit a bounded GPU preparation batch."
                )
            low, high = 1, rows - 1
            while low < high:
                middle = (low + high + 1) // 2
                if (
                    self._estimated_device_bytes(arrow.slice(0, middle))
                    <= target_device_bytes
                ):
                    low = middle
                else:
                    high = middle - 1
            return arrow.slice(0, low), arrow.slice(low)

        def _take_pending_arrow_batch(self, target_device_bytes: int):
            """Peek at a bounded prefix without consuming it until GPU packing succeeds."""

            if self._next_arrow_batch is None:
                if self._pending_arrow_batches is None:
                    self._source_input_complete = True
                    return None
                try:
                    self._next_arrow_batch = next(self._pending_arrow_batches)
                except StopIteration:
                    self._pending_arrow_batches = None
                    self._pending_arrow_owners.clear()
                    self._source_input_complete = True
                    return None
            return self._bounded_arrow_prefix(
                self._next_arrow_batch, target_device_bytes
            )

        def _prepare_more(self) -> None:
            """Prepare one source group using live allocator headroom."""

            import cupy as cp
            import cudf

            if self._prepared_batches:
                raise RuntimeError("GPU sort must exchange prepared batches first.")
            baseline = int(self._mr.current_allocated)
            source_limit = max(
                1,
                (self._pool_max_bytes - baseline - MPF_PROGRESS_RESERVE_BYTES)
                // (1 + MPF_RECEIVE_BUFFER_OWNERS),
            )
            prepared_bytes = 0
            while prepared_bytes < source_limit and not self._source_input_complete:
                current = int(self._mr.current_allocated)
                transient_limit = _input_batch_target_bytes(
                    self._config.exchange_batch_bytes,
                    self._pool_max_bytes,
                    current + MPF_PROGRESS_RESERVE_BYTES,
                    self._config.final_sort_workspace_factor,
                )
                batch_limit = min(source_limit - prepared_bytes, transient_limit)
                if batch_limit <= 0:
                    break
                try:
                    pending = self._take_pending_arrow_batch(batch_limit)
                except GPUSortCapacityError:
                    if prepared_bytes and self._next_arrow_batch is not None:
                        # The next row does not fit the small remainder of this
                        # source group. Exchange accepted owners first, then
                        # retry the untouched row against fresh headroom.
                        self._update_peak()
                        return
                    raise
                if pending is None:
                    break
                arrow, remainder = pending
                while True:
                    arrow_bytes = int(arrow.nbytes)
                    self._stats["input_batch_count"] += 1
                    self._stats["max_input_batch_bytes"] = max(
                        int(self._stats["max_input_batch_bytes"]), arrow_bytes
                    )
                    minimum = int(self._stats["min_input_batch_target_bytes"])
                    self._stats["min_input_batch_target_bytes"] = (
                        min(minimum, batch_limit) if minimum else batch_limit
                    )
                    self._stats["max_input_batch_target_bytes"] = max(
                        int(self._stats["max_input_batch_target_bytes"]), batch_limit
                    )
                    frame = None
                    chunks = None
                    row_ordinal_before = int(self._row_ordinal)
                    preparation_error = None
                    try:
                        frame = cudf.DataFrame.from_arrow(arrow)
                        self._stats["h2d_bytes"] += arrow_bytes
                        if getattr(self, "_pending_inputs_sorted", False):
                            chunks, destination_bytes = self._partition_sorted_and_pack(
                                frame, int(self._prepared_wave_id)
                            )
                        else:
                            chunks, destination_bytes = self._partition_and_pack(
                                frame, int(self._prepared_wave_id)
                            )
                        physical_bytes = sum(destination_bytes)
                        # The decoded-payload cap does not include fixed pack
                        # metadata. Exact packed bytes instead have to fit the
                        # remaining source-group budget; controller admission
                        # subsequently accounts for their destinations.
                        if physical_bytes <= source_limit - prepared_bytes:
                            break
                    except MemoryError as exc:
                        preparation_error = exc
                    # Packing has real scratch and alignment costs that decoded
                    # Arrow bytes cannot predict exactly. Release the failed or
                    # oversized attempt, then retry a smaller prefix while the
                    # original unconsumed table remains in `_next_arrow_batch`.
                    self._row_ordinal = row_ordinal_before
                    if chunks is not None:
                        chunks.clear()
                    frame = None
                    chunks = None
                    gc.collect()
                    cp.cuda.runtime.deviceSynchronize()
                    rows = int(arrow.num_rows)
                    if rows <= 1:
                        if prepared_bytes:
                            # This row can be retried after the accepted group
                            # is exchanged and its packed owners are released.
                            self._update_peak()
                            return
                        raise GPUSortCapacityError(
                            "One Arrow row cannot be partitioned and packed "
                            "within the RMM transport reserve."
                        ) from preparation_error
                    self._stats["preparation_retry_count"] += 1
                    retry_rows = max(1, rows // 2)
                    retry_target = self._estimated_device_bytes(
                        self._next_arrow_batch.slice(0, retry_rows)
                    )
                    batch_limit = max(1, min(batch_limit - 1, retry_target))
                    arrow, remainder = self._bounded_arrow_prefix(
                        self._next_arrow_batch, batch_limit
                    )

                self._next_arrow_batch = remainder
                frame = None
                cp.cuda.runtime.deviceSynchronize()
                batch_id = self._next_prepared_batch_id
                self._prepared_batches[batch_id] = _PreparedBatch(
                    chunks=chunks,
                    destination_bytes=destination_bytes,
                )
                prepared_bytes += physical_bytes
                self._next_prepared_batch_id += 1
                self._stats["prepared_batch_count"] += 1
                self._stats["prepared_bytes"] += physical_bytes
            # Peek once so an exactly full final group can be marked complete
            # without adding a useless empty MPF collective.
            if (
                self._next_arrow_batch is None
                and self._pending_arrow_batches is not None
            ):
                try:
                    self._next_arrow_batch = next(self._pending_arrow_batches)
                except StopIteration:
                    self._pending_arrow_batches = None
                    self._pending_arrow_owners.clear()
                    self._source_input_complete = True
            self._update_peak()

        def _prepared_manifest(self, *, rows: int = 0, input_bytes: int = 0):
            return {
                "rank": self.rank(),
                "wave": int(self._prepared_wave_id),
                "rows": int(rows),
                "input_bytes": int(input_bytes),
                "batches": [
                    {
                        "batch_id": batch_id,
                        "destination_bytes": list(batch.destination_bytes),
                    }
                    for batch_id, batch in sorted(self._prepared_batches.items())
                ],
                "source_complete": bool(self._source_input_complete),
                "current_allocated_bytes": int(self._mr.current_allocated),
                "pool_max_bytes": self._pool_max_bytes,
            }

        def prepare_wave(
            self,
            wave_id: int,
            blocks: List[Any],
            *,
            count_input_blocks: bool = True,
            inputs_are_sorted: bool = False,
        ) -> Dict[str, Any]:
            """Begin a wave and prepare its first memory-bounded source group."""

            if self._boundary_keys is None:
                raise RuntimeError("GPU sort plan must be installed before a wave.")
            if self._prepared_wave_id is not None or self._prepared_batches:
                raise RuntimeError("GPU sort already has an unconsumed prepared wave.")
            # Seal the prior wave's resident tail before the next source payload
            # and MPF transport buffers share the RMM pool.
            self._externalize_before_next_wave(wave_id)
            self._prepared_wave_id = int(wave_id)
            self._source_input_complete = False
            self._pending_inputs_sorted = bool(inputs_are_sorted)
            self._next_prepared_batch_id = 0
            self._next_arrow_batch = None
            input_rows = 0
            input_bytes = 0
            partition_started = time.perf_counter()
            try:
                for block in blocks:
                    arrow = self._to_arrow_table(block)
                    if inputs_are_sorted:
                        if self._work_schema is None or not arrow.schema.equals(
                            self._work_schema, check_metadata=False
                        ):
                            raise TypeError(
                                "GPU sort restored source run schema changed "
                                "after ingestion."
                            )
                    else:
                        self._set_schema(arrow.schema)
                    self._pending_arrow_owners.append(arrow)
                    input_rows += int(arrow.num_rows)
                    input_bytes += int(arrow.nbytes)
                if count_input_blocks:
                    self._stats["input_block_count"] += len(blocks)
                self._pending_arrow_batches = (
                    iter(self._pending_arrow_owners)
                    if self._pending_inputs_sorted
                    else iter(
                        _coalesce_arrow_inputs(
                            self._pending_arrow_owners,
                            self._config.exchange_batch_bytes,
                        )
                    )
                )
                self._prepare_more()
                self._stats["phases_s"]["partition"] += (
                    time.perf_counter() - partition_started
                )
                return self._prepared_manifest(rows=input_rows, input_bytes=input_bytes)
            except BaseException:
                self._prepared_batches.clear()
                self._pending_arrow_owners.clear()
                self._pending_arrow_batches = None
                self._next_arrow_batch = None
                self._prepared_wave_id = None
                self._pending_inputs_sorted = False
                raise

        def prepare_ingested_runs(
            self, wave_id: int, run_ids: List[int]
        ) -> Dict[str, Any]:
            """Restore actor-local source runs for one redistribution wave."""

            if not self._ingest_finalized:
                raise RuntimeError("GPU sort ingestion must be finalized first.")
            if self._active_source_wave_id is not None:
                raise RuntimeError("GPU sort already has an active source-run wave.")
            normalized = [int(run_id) for run_id in run_ids]
            if len(normalized) != len(set(normalized)) or any(
                run_id not in self._source_runs for run_id in normalized
            ):
                raise RuntimeError("GPU sort selected an invalid source run.")
            chunks = [
                chunk
                for run_id in normalized
                for chunk in self._source_runs[run_id].chunks
            ]
            store = self._ensure_run_store()
            local_restore_started = (
                time.perf_counter() if store.mode == "local_disk" else None
            )
            try:
                with store.read_many(chunks) as arrows:
                    manifest = self.prepare_wave(
                        wave_id,
                        list(arrows),
                        count_input_blocks=False,
                        inputs_are_sorted=True,
                    )
            finally:
                if local_restore_started is not None:
                    self._stats["local_run_restore_s"] = float(
                        self._stats.get("local_run_restore_s", 0.0)
                    ) + (time.perf_counter() - local_restore_started)
                self._sync_run_store_stats()
            self._active_source_run_ids = normalized
            self._active_source_wave_id = int(wave_id)
            manifest["source_run_ids"] = list(normalized)
            return manifest

        def prepare_more(self, wave_id: int) -> Dict[str, Any]:
            """Prepare the next source group after prior batches were exchanged."""

            if self._prepared_wave_id != int(wave_id):
                raise RuntimeError("GPU sort continuation does not match its wave.")
            started = time.perf_counter()
            self._prepare_more()
            self._stats["phases_s"]["partition"] += time.perf_counter() - started
            return self._prepared_manifest()

        def exchange_prepared_round(
            self,
            wave_id: int,
            exchange_id: int,
            batch_ids: List[int],
            final_subround: bool,
        ) -> Dict[str, Any]:
            """Exchange an admitted subset and quiesce MPF before run sorting."""

            import cupy as cp

            if self._prepared_wave_id != int(wave_id):
                raise RuntimeError(
                    "GPU sort exchange does not match its prepared wave."
                )
            selected_ids = [int(value) for value in batch_ids]
            if len(selected_ids) != len(set(selected_ids)) or any(
                value not in self._prepared_batches for value in selected_ids
            ):
                raise RuntimeError("GPU sort exchange selected an invalid batch.")
            shuffler = self.create_shuffler(
                (self.DATA_OP_BASE + int(exchange_id)) % 256,
                total_num_partitions=self._num_partitions,
                buffer_resource=self._buffer_resource,
                statistics=self._statistics,
            )
            shuffler_active = True
            received_rows = 0
            selected_owners = [self._prepared_batches[value] for value in selected_ids]
            try:
                shuffle_started = time.perf_counter()
                for batch in selected_owners:
                    if batch.chunks:
                        shuffler.insert_chunks(batch.chunks)
                shuffler.insert_finished(list(range(self._num_partitions)))
                received: Dict[int, List[Any]] = {}
                while not shuffler.finished():
                    partition = int(shuffler.wait_any())
                    received[partition] = shuffler.extract(partition)
                # MPF's progress thread must release all transport ownership
                # before unpack or run sorting uses the remaining RMM headroom.
                shuffler.shutdown()
                shuffler_active = False
                for batch_id in selected_ids:
                    del self._prepared_batches[batch_id]
                selected_owners.clear()
                cp.cuda.runtime.deviceSynchronize()
                for partition in range(
                    self.rank(), self._num_partitions, self.nranks()
                ):
                    chunks = received.pop(partition, [])
                    while chunks:
                        chunk = chunks.pop()
                        table = self._unpack_one(chunk)
                        cp.cuda.runtime.deviceSynchronize()
                        del chunk
                        received_rows += int(table.num_rows())
                        if getattr(self, "_pending_inputs_sorted", False):
                            self._accept_sorted_received(partition, table)
                        else:
                            self._accept_received(partition, table, wave_id)
                        del table
                received.clear()
                cp.cuda.runtime.deviceSynchronize()
                self._stats["phases_s"]["mpf_shuffle"] += (
                    time.perf_counter() - shuffle_started
                )
                if not final_subround:
                    before = int(self._stats["initial_run_count"])
                    for partition in sorted(self._device_tables):
                        if self._device_tables[partition]:
                            self._externalize_device_tables(partition, wave_id)
                    self._stats["forced_subround_run_count"] += (
                        int(self._stats["initial_run_count"]) - before
                    )
                    cp.cuda.runtime.deviceSynchronize()
                elif self._prepared_batches:
                    raise RuntimeError(
                        "GPU sort final exchange left prepared batches unconsumed."
                    )
                elif not self._source_input_complete:
                    raise RuntimeError(
                        "GPU sort finalized a wave before preparing all input."
                    )
                else:
                    self._prepared_wave_id = None
                    self._pending_inputs_sorted = False
                self._stats["exchange_subround_count"] += 1
            finally:
                selected_owners.clear()
                if shuffler_active:
                    shuffler.shutdown()
            self._update_peak()
            return {
                "rank": self.rank(),
                "wave": int(wave_id),
                "exchange": int(exchange_id),
                "received_rows": received_rows,
                "remaining_batches": len(self._prepared_batches),
                "source_complete": bool(self._source_input_complete),
                "current_allocated_bytes": int(self._mr.current_allocated),
            }

        def commit_source_wave(self, wave_id: int) -> Dict[str, Any]:
            """Release source runs after every rank completed redistribution."""

            if self._active_source_wave_id != int(wave_id):
                raise RuntimeError("GPU sort source-wave commit does not match.")
            if self._prepared_wave_id is not None or self._prepared_batches:
                raise RuntimeError("GPU sort cannot commit an unfinished source wave.")
            released_bytes = 0
            released_runs = 0
            released_rows = 0
            released_run_ids = list(self._active_source_run_ids)
            for run_id in self._active_source_run_ids:
                run = self._source_runs.pop(run_id)
                released_rows += int(self._source_run_rows.pop(run_id))
                released_bytes += sum(int(chunk.size_bytes) for chunk in run.chunks)
                self._release_run_chunks(run.chunks)
                run.chunks.clear()
                released_runs += 1
            self._active_source_run_ids = []
            self._active_source_wave_id = None
            return {
                "rank": self.rank(),
                "wave": int(wave_id),
                "released_source_runs": released_runs,
                "released_source_run_ids": released_run_ids,
                "released_source_rows": released_rows,
                "released_source_bytes": released_bytes,
            }

        def _accept_received(self, partition: int, table: Any, wave_id: int) -> None:
            """Retain a table or transition the range to sorted external runs."""

            rows = int(table.num_rows())
            if rows == 0:
                return
            table_bytes = self._table_bytes(table)
            if table_bytes > self._payload_limit_bytes and rows > 1:
                rows_per_piece = max(
                    1,
                    int(rows * self._payload_limit_bytes / max(1, table_bytes)),
                )
                pieces = [
                    self._slice_table(table, start, min(rows, start + rows_per_piece))
                    for start in range(0, rows, rows_per_piece)
                ]
            else:
                pieces = [table]
            for piece in pieces:
                piece_bytes = self._table_bytes(piece)
                if (
                    self._device_tables[partition]
                    and self._device_bytes[partition] + piece_bytes
                    > self._payload_limit_bytes
                ):
                    self._externalize_device_tables(partition, wave_id)
                self._device_tables[partition].append(piece)
                self._device_bytes[partition] += piece_bytes
                live = sum(self._device_bytes.values())
                self._stats["peak_live_bytes"] = max(
                    int(self._stats["peak_live_bytes"]), live
                )
                if self._device_bytes[partition] >= self._payload_limit_bytes:
                    self._externalize_device_tables(partition, wave_id)

        def _accept_sorted_received(self, partition: int, table: Any) -> None:
            """Commit an already ordered range segment without sorting it again."""

            rows = int(table.num_rows())
            if rows == 0:
                return
            run = self._store_table_as_run(table, initial=False)
            self._runs[partition].append(run)
            run_bytes = sum(int(chunk.size_bytes) for chunk in run.chunks)
            self._stats["redistributed_run_count"] += 1
            self._stats["redistributed_run_bytes"] += run_bytes
            self._stats["redistributed_run_rows"] += rows
            self._stats["state"] = "EXTERNAL_RUNS"
            self._stats["mode"] = "external"
            self._update_peak()

        def _externalize_device_tables(self, partition: int, wave_id: int) -> None:
            import cupy as cp
            import pylibcudf as plc

            tables = self._device_tables[partition]
            if not tables:
                return
            self._device_tables[partition] = []
            self._device_bytes[partition] = 0
            while tables:
                cp.cuda.runtime.deviceSynchronize()
                # Retained tables and extracted receive buffers share the RMM
                # pool. They are included in ``current_allocated``; bound both
                # concatenate and final-sort workspace by actual headroom.
                current_allocated = int(self._mr.current_allocated)
                available = max(0, self._pool_max_bytes - current_allocated)
                group_limit = _workspace_bounded_payload_bytes(
                    self._pool_max_bytes,
                    current_allocated,
                    self._config.final_sort_workspace_factor,
                    0,
                )
                group_limit = min(self._payload_limit_bytes, available, group_limit)
                if group_limit <= 0:
                    raise GPUSortCapacityError(
                        "GPU sort cannot reserve run-sort workspace from the "
                        "current live device allocation."
                    )

                # Destructively remove only a group whose concatenate output
                # can coexist with all remaining sources. Once the new table
                # is synchronized, releasing ``group`` drops the old buffers
                # before sort workspace is allocated.
                group = []
                group_bytes = 0
                while tables:
                    candidate_bytes = self._table_bytes(tables[-1])
                    if group and group_bytes + candidate_bytes > group_limit:
                        break
                    group.append(tables.pop())
                    group_bytes += candidate_bytes
                    if group_bytes >= group_limit:
                        break

                # Recheck allocator headroom immediately before allocating the
                # concatenate output and return excess sources to the queue.
                if len(group) > 1:
                    cp.cuda.runtime.deviceSynchronize()
                    late_group_limit = _workspace_bounded_payload_bytes(
                        self._pool_max_bytes,
                        int(self._mr.current_allocated),
                        self._config.final_sort_workspace_factor,
                        0,
                    )
                    while len(group) > 1 and group_bytes > late_group_limit:
                        returned = group.pop()
                        group_bytes -= self._table_bytes(returned)
                        tables.append(returned)
                if len(group) == 1:
                    table = group.pop()
                else:
                    try:
                        table = plc.concatenate.concatenate(group)
                        cp.cuda.runtime.deviceSynchronize()
                    except MemoryError:
                        # Keep every source and make progress through the
                        # existing bounded single-table sort path.
                        self._stats["concat_oom_fallback_count"] = (
                            int(self._stats.get("concat_oom_fallback_count", 0)) + 1
                        )
                        table = group.pop()
                        tables.extend(group)
                    finally:
                        group.clear()

                rows = int(table.num_rows())
                start = 0
                while start < rows:
                    cp.cuda.runtime.deviceSynchronize()
                    piece_limit = min(
                        self._payload_limit_bytes,
                        _workspace_bounded_payload_bytes(
                            self._pool_max_bytes,
                            int(self._mr.current_allocated),
                            self._config.final_sort_workspace_factor,
                            0,
                        ),
                    )
                    if piece_limit <= 0:
                        raise GPUSortCapacityError(
                            "GPU sort cannot reserve run-sort workspace from the "
                            "current live device allocation."
                        )
                    remaining = self._slice_table(table, start, rows)
                    remaining_bytes = self._table_bytes(remaining)
                    piece_rows = max(
                        1,
                        int((rows - start) * piece_limit / max(1, remaining_bytes)),
                    )
                    end = min(rows, start + piece_rows)
                    del remaining
                    while True:
                        piece = self._slice_table(table, start, end)
                        piece_bytes = self._table_bytes(piece)
                        while piece_bytes > piece_limit and end - start > 1:
                            piece_rows = max(
                                1,
                                int((end - start) * piece_limit / piece_bytes),
                            )
                            end = start + piece_rows
                            piece = self._slice_table(table, start, end)
                            piece_bytes = self._table_bytes(piece)
                        if piece_bytes > piece_limit:
                            raise GPUSortCapacityError(
                                "One GPU sort row exceeds the available "
                                "run-sort workspace."
                            )

                        # Slicing changes allocator headroom after the first
                        # estimate. Recheck before sorting, and reduce this
                        # run instead of relying on a stale snapshot.
                        cp.cuda.runtime.deviceSynchronize()
                        live_limit = min(
                            self._payload_limit_bytes,
                            _workspace_bounded_payload_bytes(
                                self._pool_max_bytes,
                                int(self._mr.current_allocated),
                                self._config.final_sort_workspace_factor,
                                0,
                            ),
                        )
                        if piece_bytes > live_limit and end - start > 1:
                            next_rows = max(
                                1,
                                int((end - start) * live_limit / piece_bytes),
                            )
                            if next_rows >= end - start:
                                next_rows = end - start - 1
                            end = start + next_rows
                            del piece
                            continue
                        if piece_bytes > live_limit:
                            raise GPUSortCapacityError(
                                "One GPU sort row exceeds the live run-sort workspace."
                            )

                        sort_started = time.perf_counter()
                        try:
                            sorted_piece = self._sort_table(piece)
                            cp.cuda.runtime.deviceSynchronize()
                        except MemoryError as exc:
                            # Actual libcudf workspace can exceed the decoded-
                            # byte estimate. Retry the same rows as smaller GPU
                            # runs.
                            self._stats["run_sort_oom_retry_count"] = (
                                int(self._stats.get("run_sort_oom_retry_count", 0)) + 1
                            )
                            self._stats["phases_s"]["run_sort"] += (
                                time.perf_counter() - sort_started
                            )
                            del piece
                            cp.cuda.runtime.deviceSynchronize()
                            if end - start <= 1:
                                raise GPUSortCapacityError(
                                    "One GPU sort row cannot fit in the RMM pool."
                                ) from exc
                            end = start + max(1, (end - start) // 2)
                            continue
                        self._stats["phases_s"]["run_sort"] += (
                            time.perf_counter() - sort_started
                        )
                        break
                    run = self._store_table_as_run(sorted_piece, initial=True)
                    self._runs[partition].append(run)
                    del sorted_piece, piece
                    start = end
                del table
            self._stats["state"] = "EXTERNAL_RUNS"
            self._stats["mode"] = "external"
            if self._stats["first_externalize_s"] is None:
                self._stats["first_externalize_s"] = (
                    time.perf_counter() - self._started_at
                )
                self._stats["first_externalize_wave"] = int(wave_id)
            self._update_peak()

        def _iter_table_as_work_arrow(self, table: Any) -> Iterator[Any]:
            """Copy one GPU table to bounded, schema-faithful Arrow chunks."""

            import cupy as cp

            rows = int(table.num_rows())
            table_bytes = self._table_bytes(table)
            rows_per_chunk = max(
                1,
                int(rows * self._run_chunk_bytes / max(1, table_bytes)),
            )
            for start in range(0, rows, rows_per_chunk):
                part = self._slice_table(
                    table, start, min(rows, start + rows_per_chunk)
                )
                arrow_started = time.perf_counter()
                arrow = self._table_to_work_arrow(part)
                cp.cuda.runtime.deviceSynchronize()
                arrow_s = time.perf_counter() - arrow_started
                self._stats["phases_s"]["arrow_conversion"] += arrow_s
                size_bytes = int(arrow.nbytes)
                self._stats["d2h_bytes"] += size_bytes
                yield arrow

        def _ensure_run_store(self) -> RunStore:
            store = getattr(self, "_run_store", None)
            if store is None:
                # Unit-level backend method tests and the default production
                # path both retain the historical Plasma behavior.
                store = PlasmaRunStore()
                self._run_store = store
            return store

        def _sync_run_store_stats(self) -> None:
            store = getattr(self, "_run_store", None)
            if store is None:
                return
            diagnostics = store.diagnostics()
            self._stats.update(diagnostics)
            self._stats["run_store"] = store.mode
            self._stats["plasma_read_bytes"] = int(
                diagnostics.get("plasma_intermediate_read_bytes", 0)
            )
            self._stats["plasma_write_bytes"] = int(
                diagnostics.get("plasma_intermediate_write_bytes", 0)
            ) + int(self._stats.get("plasma_output_write_bytes", 0))

        def _release_run_chunks(self, chunks: List[_RunChunk]) -> None:
            if not chunks:
                return
            self._ensure_run_store().release(chunks)
            self._sync_run_store_stats()

        def _store_table_as_run(
            self, table: Any, *, initial: bool, replacement: bool = False
        ) -> _ExternalRun:
            """D2H and commit one sorted GPU table as bounded Arrow chunks."""

            rows = int(table.num_rows())
            run = _ExternalRun()
            store = self._ensure_run_store()
            before_write_s = float(
                store.diagnostics().get("plasma_intermediate_write_s", 0.0)
            )
            try:
                for arrow in self._iter_table_as_work_arrow(table):
                    run.chunks.append(store.write(arrow))
            except BaseException:
                # One GPU table is a transactional run: a later chunk failure
                # must not orphan the earlier committed fragments.
                store.release(run.chunks)
                run.chunks.clear()
                self._sync_run_store_stats()
                raise
            self._sync_run_store_stats()
            if store.mode == "plasma":
                self._stats["phases_s"]["plasma_seal"] += max(
                    0.0,
                    float(
                        store.diagnostics().get(
                            "plasma_intermediate_write_s", before_write_s
                        )
                    )
                    - before_write_s,
                )
            if initial:
                self._stats["initial_run_count"] += 1
                self._stats["externalized_rows"] += rows
                self._stats["externalized_bytes"] += sum(
                    chunk.size_bytes for chunk in run.chunks
                )
            if replacement:
                self._stats["replacement_run_count"] += 1
            return run

        # -- bounded GPU-only external merge -----------------------------

        def _load_run_chunks(self, chunks: List[_RunChunk]) -> List[Any]:
            """Restore one bounded set of merge heads in one store read."""

            import cudf
            from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table

            if not chunks:
                return []
            store = self._ensure_run_store()
            before_read_s = float(
                store.diagnostics().get("plasma_intermediate_read_s", 0.0)
            )
            local_restore_started = (
                time.perf_counter() if store.mode == "local_disk" else None
            )
            tables = []
            try:
                with store.read_many(chunks) as arrows:
                    for chunk, arrow in zip(chunks, arrows):
                        frame = cudf.DataFrame.from_arrow(arrow)
                        tables.append(cudf_to_pylibcudf_table(frame))
                        # Preserve completed transfer evidence if a later head
                        # fails conversion, for example at the RMM ceiling.
                        self._stats["h2d_bytes"] += int(chunk.size_bytes)
                    if store.mode == "local_disk":
                        # Do not let an mmap close while an asynchronous H2D
                        # copy still references its Arrow buffers.
                        import cupy as cp

                        cp.cuda.runtime.deviceSynchronize()
            finally:
                if local_restore_started is not None:
                    self._stats["local_run_restore_s"] = float(
                        self._stats.get("local_run_restore_s", 0.0)
                    ) + (time.perf_counter() - local_restore_started)
                self._sync_run_store_stats()
            if store.mode == "plasma":
                self._stats["phases_s"]["orchestration"] += max(
                    0.0,
                    float(
                        store.diagnostics().get(
                            "plasma_intermediate_read_s", before_read_s
                        )
                    )
                    - before_read_s,
                )
            return tables

        def _watermark(self, tables: List[Any]):
            """Return the earliest current-head last key in requested order."""

            import pylibcudf as plc

            last_keys = []
            for table in tables:
                keys = self._comparison_table(table, self._work_names)
                last_keys.append(
                    self._slice_table(keys, keys.num_rows() - 1, keys.num_rows())
                )
            candidates = (
                last_keys[0]
                if len(last_keys) == 1
                else plc.concatenate.concatenate(last_keys)
            )
            order, nulls = self._order_and_nulls()
            sorted_candidates = plc.sorting.sort_by_key(
                candidates, candidates, order, nulls
            )
            return self._slice_table(sorted_candidates, 0, 1)

        def _upper_bound(self, table: Any, watermark: Any) -> int:
            import cudf
            import pylibcudf as plc

            keys = self._comparison_table(table, self._work_names)
            order, nulls = self._order_and_nulls()
            result = plc.search.upper_bound(keys, watermark, order, nulls)
            return int(cudf.Series.from_pylibcudf(result).iloc[0])

        def _iter_merged_sources(self, group: List[_ExternalRun]) -> Iterator[Any]:
            """Yield a GPU-merged group as device tables or reusable run chunks."""

            import cupy as cp
            import pylibcudf as plc

            states = [{"run": run, "next": 0, "table": None} for run in group]
            while True:
                missing = []
                for state in states:
                    if state["table"] is None and state["next"] < len(
                        state["run"].chunks
                    ):
                        missing.append((state, state["run"].chunks[state["next"]]))
                if missing:
                    tables = self._load_run_chunks([chunk for _, chunk in missing])
                    for (state, _), table in zip(missing, tables):
                        state["table"] = table
                        state["next"] += 1
                active = [state for state in states if state["table"] is not None]
                if not active:
                    break
                if len(active) == 1:
                    # With no competing head, the remaining stream is already
                    # globally ordered. Yield the resident suffix and then
                    # reuse unread stored chunks without a GPU round-trip.
                    state = active[0]
                    yield state["table"]
                    state["table"] = None
                    yield from state["run"].chunks[state["next"] :]
                    state["next"] = len(state["run"].chunks)
                    continue

                gpu_started = time.perf_counter()
                watermark = self._watermark([state["table"] for state in active])
                prefixes = []
                for state in active:
                    table = state["table"]
                    count = self._upper_bound(table, watermark)
                    if count:
                        prefixes.append(self._slice_table(table, 0, count))
                    state["table"] = (
                        self._slice_table(table, count, table.num_rows())
                        if count < table.num_rows()
                        else None
                    )
                if not prefixes:
                    raise RuntimeError("GPU external merge made no forward progress.")
                if len(prefixes) == 1:
                    merged = prefixes[0]
                else:
                    order, nulls = self._order_and_nulls()
                    merged = plc.merge.merge(
                        prefixes,
                        self._merge_key_indices,
                        order,
                        nulls,
                    )
                cp.cuda.runtime.deviceSynchronize()
                self._stats["phases_s"]["gpu_merge"] += (
                    time.perf_counter() - gpu_started
                )
                yield merged
                self._update_peak()
            cp.cuda.runtime.deviceSynchronize()

        def _merge_group(self, group: List[_ExternalRun]) -> _ExternalRun:
            """Materialize one bounded intermediate GPU merge group."""

            output = _ExternalRun()
            created_chunks: List[_RunChunk] = []
            try:
                for source in self._iter_merged_sources(group):
                    if isinstance(source, _RunChunk):
                        output.chunks.append(source)
                    else:
                        replacement = self._store_table_as_run(source, initial=False)
                        output.chunks.extend(replacement.chunks)
                        created_chunks.extend(replacement.chunks)
            except BaseException:
                # Preserve every source so the group still represents a
                # complete run. Only partial replacement files are rolled back.
                self._release_run_chunks(created_chunks)
                output.chunks.clear()
                raise
            self._stats["replacement_run_count"] += 1
            # All output chunks are committed before prior-pass ownership is
            # released. Unread suffixes move by object identity to ``output``.
            transferred = {id(chunk) for chunk in output.chunks}
            consumed = [
                chunk
                for run in group
                for chunk in run.chunks
                if id(chunk) not in transferred
            ]
            self._release_run_chunks(consumed)
            for run in group:
                run.chunks.clear()
            return output

        def _merge_runs_until_final(
            self, runs: List[_ExternalRun]
        ) -> List[_ExternalRun]:
            """Materialize only merge passes required to reach one fan-in group."""

            fan_in = self._config.merge_fan_in
            current = list(runs)
            while len(current) > fan_in:
                self._stats["merge_pass_count"] += 1
                following: List[_ExternalRun] = []
                for start in range(0, len(current), fan_in):
                    group = current[start : start + fan_in]
                    following.append(
                        group[0] if len(group) == 1 else self._merge_group(group)
                    )
                current = following
            return current

        def _iter_final_work_arrow(
            self, final_group: List[_ExternalRun]
        ) -> Iterator[Any]:
            """Stream the last merge pass without a replacement stored run."""

            if len(final_group) == 1:
                sources: Iterator[Any] = iter(final_group[0].chunks)
            else:
                self._stats["merge_pass_count"] += 1
                sources = self._iter_merged_sources(final_group)

            for source in sources:
                if isinstance(source, _RunChunk):
                    store = self._ensure_run_store()
                    before_read_s = float(
                        store.diagnostics().get("plasma_intermediate_read_s", 0.0)
                    )
                    try:
                        # The yielded Arrow table retains its mmap buffers even
                        # after the NativeFile descriptor is closed. Source
                        # files themselves remain owned until output sealing.
                        with store.read_many([source]) as arrows:
                            yield arrows[0]
                    finally:
                        self._sync_run_store_stats()
                    if store.mode == "plasma":
                        self._stats["phases_s"]["orchestration"] += max(
                            0.0,
                            float(
                                store.diagnostics().get(
                                    "plasma_intermediate_read_s", before_read_s
                                )
                            )
                            - before_read_s,
                        )
                else:
                    yield from self._iter_table_as_work_arrow(source)

            if len(final_group) > 1:
                self._stats["direct_final_merge_count"] += 1

        # -- Arrow output -------------------------------------------------

        @staticmethod
        def _fixed_width_bytes(data_type: Any) -> Optional[int]:
            import pyarrow as pa

            if pa.types.is_boolean(data_type):
                return None
            if pa.types.is_fixed_size_binary(data_type):
                return int(data_type.byte_width)
            if (
                pa.types.is_integer(data_type)
                or pa.types.is_floating(data_type)
                or pa.types.is_decimal(data_type)
                or pa.types.is_date32(data_type)
                or pa.types.is_date64(data_type)
                or pa.types.is_time32(data_type)
                or pa.types.is_time64(data_type)
                or pa.types.is_timestamp(data_type)
                or pa.types.is_duration(data_type)
            ):
                return int(data_type.bit_width // 8)
            return None

        @staticmethod
        def _generic_frame_to_arrow(frame: Any, schema: Any):
            """Schema-faithful fallback, including cuDF's all-null corner."""

            import pyarrow as pa

            rows = len(frame)
            arrays = []
            for field in schema:
                column = frame[field.name]._column
                if rows and int(column.null_count) == rows:
                    array = pa.nulls(rows, type=field.type)
                else:
                    array = column.to_arrow()
                    if not array.type.equals(field.type):
                        array = array.cast(field.type)
                arrays.append(array)
            return pa.Table.from_arrays(arrays, schema=schema)

        def _frame_to_arrow(self, frame: Any, schema: Any):
            """Pinned asynchronous D2H for flat fixed-width/string columns."""

            import cupy as cp
            import pyarrow as pa

            rows = len(frame)
            if rows == 0:
                return schema.empty_table()
            if self._config.pinned_output_max_bytes == 0:
                return self._generic_frame_to_arrow(frame, schema)

            plan = []
            pinned_bytes = 0
            arrays: List[Any] = []
            for field in schema:
                column = frame[field.name]._column
                if int(column.null_count) == rows:
                    arrays.append(pa.nulls(rows, type=field.type))
                    plan.append(None)
                    continue
                width = self._fixed_width_bytes(field.type)
                kind = None
                data_bytes = 0
                extra_bytes = 0
                if width is not None and column.offset == 0 and column.data is not None:
                    kind = "fixed"
                    data_bytes = rows * width
                elif (
                    pa.types.is_string(field.type)
                    and column.offset == 0
                    and len(column.children) >= 2
                    and column.children[0].data is not None
                ):
                    kind = "string"
                    data_bytes = (rows + 1) * 4
                    extra_bytes = int(column.children[1].size)
                mask_bytes = (
                    (rows + 7) // 8
                    if kind is not None and int(column.null_count) > 0
                    else 0
                )
                pinned_bytes += data_bytes + extra_bytes + mask_bytes
                plan.append((field, column, kind, data_bytes, extra_bytes, mask_bytes))
                arrays.append(None)
            if pinned_bytes > self._config.pinned_output_max_bytes:
                return self._generic_frame_to_arrow(frame, schema)

            stream = cp.cuda.Stream(non_blocking=True)
            pending: Dict[int, Any] = {}
            try:
                for index, item in enumerate(plan):
                    if item is None:
                        continue
                    field, column, kind, data_bytes, extra_bytes, mask_bytes = item
                    if kind is None:
                        array = column.to_arrow()
                        arrays[index] = (
                            array
                            if array.type.equals(field.type)
                            else array.cast(field.type)
                        )
                        continue
                    host_data = cp.cuda.alloc_pinned_memory(data_bytes)
                    source_data = (
                        column.data if kind == "fixed" else column.children[0].data
                    )
                    with source_data.access(mode="read"):
                        cp.cuda.runtime.memcpyAsync(
                            host_data.ptr,
                            source_data.ptr,
                            data_bytes,
                            cp.cuda.runtime.memcpyDeviceToHost,
                            stream.ptr,
                        )
                    host_mask = None
                    if mask_bytes:
                        host_mask = cp.cuda.alloc_pinned_memory(mask_bytes)
                        if column.mask is None:
                            raise RuntimeError("Nullable cuDF column has no mask.")
                        with column.mask.access(mode="read"):
                            cp.cuda.runtime.memcpyAsync(
                                host_mask.ptr,
                                column.mask.ptr,
                                mask_bytes,
                                cp.cuda.runtime.memcpyDeviceToHost,
                                stream.ptr,
                            )
                    host_extra = None
                    if extra_bytes:
                        host_extra = cp.cuda.alloc_pinned_memory(extra_bytes)
                        chars = column.children[1].data
                        with chars.access(mode="read"):
                            cp.cuda.runtime.memcpyAsync(
                                host_extra.ptr,
                                chars.ptr,
                                extra_bytes,
                                cp.cuda.runtime.memcpyDeviceToHost,
                                stream.ptr,
                            )
                    pending[index] = (
                        field,
                        column,
                        kind,
                        host_data,
                        host_extra,
                        host_mask,
                    )
                stream.synchronize()
                for index, item in pending.items():
                    field, column, kind, host_data, host_extra, host_mask = item
                    data_buffer = pa.py_buffer(memoryview(host_data))
                    mask_buffer = (
                        pa.py_buffer(memoryview(host_mask))
                        if host_mask is not None
                        else None
                    )
                    if kind == "fixed":
                        buffers = [mask_buffer, data_buffer]
                    else:
                        buffers = [
                            mask_buffer,
                            data_buffer,
                            pa.py_buffer(memoryview(host_extra))
                            if host_extra is not None
                            else pa.py_buffer(b""),
                        ]
                    arrays[index] = pa.Array.from_buffers(
                        field.type,
                        rows,
                        buffers,
                        null_count=int(column.null_count),
                    )
                return pa.Table.from_arrays(arrays, schema=schema)
            except Exception:
                self._stats["fallback_count"] += 1
                try:
                    stream.synchronize()
                except Exception:
                    pass
                pending.clear()
                arrays.clear()
                return self._generic_frame_to_arrow(frame, schema)

        def _table_to_work_arrow(self, table: Any):
            from rapidsmpf.utils.cudf import pylibcudf_to_cudf_dataframe

            frame = pylibcudf_to_cudf_dataframe(table, self._work_names)
            return self._frame_to_arrow(frame, self._work_schema)

        def _output_arrow(self, work_arrow: Any):
            arrow = work_arrow.select(self._column_names)
            if not arrow.schema.equals(self._arrow_schema, check_metadata=False):
                arrow = arrow.cast(self._arrow_schema)
            return arrow.replace_schema_metadata(self._arrow_schema.metadata)

        def _iter_coalesced_output_blocks(
            self, work_tables: Iterator[Any], target_bytes: int
        ) -> Iterator[Any]:
            """Coalesce a stream while folding a tiny tail into its predecessor."""

            import pyarrow as pa

            previous = None
            current = []
            current_bytes = 0
            for work_arrow in work_tables:
                output = self._output_arrow(work_arrow)
                current.append(output)
                current_bytes += int(output.nbytes)
                if current_bytes < target_bytes:
                    continue
                block = current[0] if len(current) == 1 else pa.concat_tables(current)
                if previous is not None:
                    yield previous
                previous = block
                current = []
                current_bytes = 0

            if current:
                tail = current[0] if len(current) == 1 else pa.concat_tables(current)
                previous = (
                    tail if previous is None else pa.concat_tables([previous, tail])
                )
            if previous is not None:
                yield previous

        def _tagged_metadata(self, block: Any, partition: int, creation_stats: Any):
            from ray.data.block import BlockExecStats, BlockMetadataWithSchema

            stats = BlockExecStats.builder()
            serialization_s = (
                creation_stats.object_creation_dur_s
                if creation_stats is not None
                else 0
            )
            metadata = BlockMetadataWithSchema.from_block(
                block,
                block_exec_stats=stats.build(block_ser_time_s=serialization_s),
            )
            schema_metadata = block.schema.metadata or {}
            tagged_schema = block.schema.with_metadata(
                {
                    **schema_metadata,
                    GPU_SORT_PARTITION_ID_KEY: str(partition).encode(),
                }
            )
            return BlockMetadataWithSchema.from_metadata(
                metadata.metadata, schema=tagged_schema
            )

        def _finish_resident_partition(self, partition: int):
            """Final-sort one resident range or move it to bounded GPU runs."""

            import cupy as cp
            import pylibcudf as plc

            tables = self._device_tables[partition]
            if not tables:
                return None

            def externalize(owned_tables: List[Any]):
                self._device_tables[partition] = owned_tables
                self._device_bytes[partition] = sum(
                    self._table_bytes(table) for table in owned_tables
                )
                self._stats["resident_final_fallback_count"] += 1
                self._externalize_device_tables(partition, wave_id=-1)
                return None

            payload_bytes = sum(self._table_bytes(table) for table in tables)
            cp.cuda.runtime.deviceSynchronize()
            live_limit = _workspace_bounded_payload_bytes(
                self._pool_max_bytes,
                int(self._mr.current_allocated),
                self._config.final_sort_workspace_factor,
                0,
            )
            if payload_bytes > live_limit:
                return externalize(tables)

            try:
                if len(tables) == 1:
                    table = tables.pop()
                else:
                    table = plc.concatenate.concatenate(tables)
                    cp.cuda.runtime.deviceSynchronize()
                    # The concatenated table owns its buffers. Release every
                    # retained source before admitting final-sort workspace.
                    tables.clear()
            except MemoryError:
                gc.collect()
                cp.cuda.runtime.deviceSynchronize()
                return externalize(tables)

            self._device_tables[partition] = []
            self._device_bytes[partition] = 0
            table_bytes = self._table_bytes(table)
            cp.cuda.runtime.deviceSynchronize()
            live_limit = _workspace_bounded_payload_bytes(
                self._pool_max_bytes,
                int(self._mr.current_allocated),
                self._config.final_sort_workspace_factor,
                0,
            )
            if table_bytes > live_limit:
                return externalize([table])

            sort_started = time.perf_counter()
            try:
                sorted_table = self._sort_table(table)
                cp.cuda.runtime.deviceSynchronize()
            except MemoryError:
                self._stats["run_sort_oom_retry_count"] += 1
                self._stats["phases_s"]["run_sort"] += (
                    time.perf_counter() - sort_started
                )
                cp.cuda.runtime.deviceSynchronize()
                return externalize([table])
            self._stats["phases_s"]["run_sort"] += time.perf_counter() - sort_started
            return sorted_table

        def finish_and_extract(self) -> Iterator[Any]:
            """Finish resident ranges or GPU-merge external runs and stream blocks."""

            import cupy as cp

            for partition in sorted(self._device_tables):
                resident_table = None
                runs = self._runs[partition]
                if not runs and self._device_tables[partition]:
                    resident_table = self._finish_resident_partition(partition)
                    runs = self._runs[partition]
                if runs:
                    if self._device_tables[partition]:
                        self._externalize_device_tables(partition, wave_id=-1)
                        runs = self._runs[partition]
                    final_group = self._merge_runs_until_final(runs)
                    self._runs[partition] = final_group
                    # Materialize intermediate merge passes, but stream the
                    # final fan-in directly to Ray. This avoids writing the
                    # final replacement run only to read and seal it again as
                    # the operator output.
                    target_output_bytes = max(128 << 10, self._run_chunk_bytes)
                    work_tables = self._iter_final_work_arrow(final_group)
                    for block in self._iter_coalesced_output_blocks(
                        work_tables, target_output_bytes
                    ):
                        self._stats["output_bytes"] += int(block.nbytes)
                        self._stats["plasma_output_write_bytes"] = int(
                            self._stats.get("plasma_output_write_bytes", 0)
                        ) + int(block.nbytes)
                        self._stats["plasma_output_write_calls"] = (
                            int(self._stats.get("plasma_output_write_calls", 0)) + 1
                        )
                        self._stats["plasma_write_bytes"] += int(block.nbytes)
                        creation_stats = yield block
                        if creation_stats is not None:
                            self._stats["plasma_output_write_s"] = float(
                                self._stats.get("plasma_output_write_s", 0.0)
                            ) + float(creation_stats.object_creation_dur_s)
                            self._stats["phases_s"]["plasma_seal"] += float(
                                creation_stats.object_creation_dur_s
                            )
                        yield self._tagged_metadata(block, partition, creation_stats)
                    # Every creation-stats acknowledgement above proves the
                    # corresponding final block is sealed before its source is
                    # deleted. Reused chunks remain live throughout the
                    # streaming generator's one-block look-behind.
                    for run in final_group:
                        self._release_run_chunks(run.chunks)
                        run.chunks.clear()
                    self._runs[partition] = []
                elif resident_table is not None:
                    arrow_started = time.perf_counter()
                    work_arrow = self._table_to_work_arrow(resident_table)
                    cp.cuda.runtime.deviceSynchronize()
                    self._stats["phases_s"]["arrow_conversion"] += (
                        time.perf_counter() - arrow_started
                    )
                    self._stats["d2h_bytes"] += int(work_arrow.nbytes)
                    block = self._output_arrow(work_arrow)
                    self._stats["output_bytes"] += int(block.nbytes)
                    self._stats["plasma_output_write_bytes"] = int(
                        self._stats.get("plasma_output_write_bytes", 0)
                    ) + int(block.nbytes)
                    self._stats["plasma_output_write_calls"] = (
                        int(self._stats.get("plasma_output_write_calls", 0)) + 1
                    )
                    self._stats["plasma_write_bytes"] += int(block.nbytes)
                    creation_stats = yield block
                    if creation_stats is not None:
                        self._stats["plasma_output_write_s"] = float(
                            self._stats.get("plasma_output_write_s", 0.0)
                        ) + float(creation_stats.object_creation_dur_s)
                        self._stats["phases_s"]["plasma_seal"] += float(
                            creation_stats.object_creation_dur_s
                        )
                    yield self._tagged_metadata(block, partition, creation_stats)
                else:
                    block = self._arrow_schema.empty_table()
                    self._stats["plasma_output_write_calls"] = (
                        int(self._stats.get("plasma_output_write_calls", 0)) + 1
                    )
                    creation_stats = yield block
                    if creation_stats is not None:
                        self._stats["plasma_output_write_s"] = float(
                            self._stats.get("plasma_output_write_s", 0.0)
                        ) + float(creation_stats.object_creation_dur_s)
                        self._stats["phases_s"]["plasma_seal"] += float(
                            creation_stats.object_creation_dur_s
                        )
                    yield self._tagged_metadata(block, partition, creation_stats)
            self._update_peak()
            store = getattr(self, "_run_store", None)
            if store is not None:
                # This generator finishes only after Ray has acknowledged every
                # final output seal. Do not report successful extraction while
                # actor-local intermediate files remain on disk.
                store.close(strict=True)
                self._sync_run_store_stats()

        # -- compact diagnostics and cleanup -----------------------------

        def _rapids_stats(self) -> Dict[str, Any]:
            if self._statistics is None:
                return {}
            return {
                str(name): _safe_stat(self._statistics.get_stat(name))
                for name in self._statistics.list_stat_names()
            }

        def _ray_spilled_bytes(self) -> int:
            if self.is_initialized() and self.rank() != 0:
                return 0
            try:
                import ray
                from ray._private.internal_api import (
                    get_memory_info_reply,
                    get_state_from_address,
                )

                state = get_state_from_address(ray.get_runtime_context().gcs_address)
                reply = get_memory_info_reply(state)
                return int(reply.store_stats.spilled_bytes_total)
            except Exception:
                return 0

        def _update_peak(self) -> None:
            peak = 0
            if self._mr is not None:
                try:
                    peak = max(
                        int(self._mr.current_allocated),
                        int(self._mr.get_main_record().peak()),
                    )
                except Exception:
                    pass
            try:
                import cupy as cp

                free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
                peak = max(peak, int(total_bytes) - int(free_bytes))
            except Exception:
                pass
            self._stats["peak_device_bytes"] = max(
                int(self._stats["peak_device_bytes"]), peak
            )

        def diagnostics(self) -> Dict[str, Any]:
            self._update_peak()
            self._sync_run_store_stats()
            rapids_stats = self._rapids_stats()
            self._stats["mpf_host_spill_bytes"] = _sum_spill_bytes(rapids_stats)
            if self.rank() == 0:
                self._stats["ray_disk_spill_bytes"] = max(
                    0, self._ray_spilled_bytes() - self._ray_spill_start
                )
            return {
                **self._stats,
                "rank": self.rank(),
                "pool_max_bytes": self._pool_max_bytes,
                "total_vram_bytes": self._total_vram_bytes,
                "rapidsmpf": rapids_stats,
            }

        def release(self, *, strict: bool = False) -> None:
            self._prepared_batches.clear()
            self._pending_arrow_owners.clear()
            self._pending_arrow_batches = None
            self._next_arrow_batch = None
            self._prepared_wave_id = None
            for tables in self._device_tables.values():
                tables.clear()
            for runs in self._runs.values():
                for run in runs:
                    store = getattr(self, "_run_store", None)
                    if store is not None:
                        store.release(run.chunks)
                    run.chunks.clear()
                runs.clear()
            for run in self._source_runs.values():
                store = getattr(self, "_run_store", None)
                if store is not None:
                    store.release(run.chunks)
                run.chunks.clear()
            self._source_runs.clear()
            self._source_run_rows.clear()
            self._active_source_run_ids = []
            self._active_source_wave_id = None
            store = getattr(self, "_run_store", None)
            if store is not None:
                store.close(strict=strict)
                self._sync_run_store_stats()
            self._boundary_keys = None
            gc.collect()

    return GPURangeSortBackend


_BACKEND_CLASS: Optional[type[Any]] = None


def get_backend_class() -> type[Any]:
    global _BACKEND_CLASS
    if _BACKEND_CLASS is None:
        _BACKEND_CLASS = lazy_load_backend()
    return _BACKEND_CLASS
