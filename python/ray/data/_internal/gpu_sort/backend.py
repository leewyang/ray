"""RAPIDS backend for a spillable distributed GPU range sort.

Each GPU rank follows the same path::

    Arrow blocks -> range partition -> MPF exchange -> destination range
                                                       | fits: final GPU sort
                                                       | full: sorted runs
                                                               -> GPU merge

Sorted runs are immutable Arrow objects in Ray's object store. Ray may spill
those objects to its configured filesystem when Plasma fills. The CPU orders
only the bounded planning sample; it never sorts or merges dataset rows.
"""

from __future__ import annotations

import gc
from contextlib import contextmanager
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, Iterator, List, Optional

from ray.data._internal.gpu_sort.config import GPUSortCapacityError, GPUSortConfig

_WEIGHT_BASE = "__ray_gpu_sort_byte_weight"
_HIDDEN_BASE = "__ray_gpu_sort_cmp"
_SAMPLE_BLOCK_BASE = "__ray_gpu_sort_sample_block"
_SAMPLE_STRATUM_BASE = "__ray_gpu_sort_sample_stratum"


@dataclass
class _RunChunk:
    """One object-store chunk of a sorted external run."""

    ref: Any
    size_bytes: int


@dataclass
class _ExternalRun:
    """Ordered chunks produced by one bounded GPU sort or merge."""

    chunks: List[_RunChunk] = dc_field(default_factory=list)


def _align_down_256(value: int) -> int:
    return (int(value) >> 8) << 8


@contextmanager
def _read_device_buffer(buffer: Any) -> Iterator[int]:
    """Keep a cuDF buffer stable and yield its device pointer."""

    access = getattr(buffer, "access", None)
    if access is not None:
        with access(mode="read"):
            yield int(buffer.ptr)
        return

    # cuDF 25.12 exposed the same lifetime guarantee through a spill lock.
    from cudf.core.buffer import acquire_spill_lock

    with acquire_spill_lock():
        yield int(buffer.get_ptr(mode="read"))


def _private_name(base: str, names: List[str]) -> str:
    candidate = base
    suffix = 0
    while candidate in names:
        suffix += 1
        candidate = f"{base}_{suffix}"
    return candidate


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
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    indices = generator.integers(lows, highs, dtype=np.int64)
    return indices, widths.astype(np.uint64)


def _scale_sample_weights_by_stratum(weights: Any, stratum_widths: Any) -> Any:
    """Apply exact inverse-inclusion weights for unequal integer strata."""

    import numpy as np

    weights = np.asarray(weights, dtype=np.uint64)
    stratum_widths = np.asarray(stratum_widths, dtype=np.uint64)
    if weights.shape != stratum_widths.shape or np.any(stratum_widths == 0):
        raise ValueError("Every sampled row must have one nonempty stratum.")
    return np.multiply(weights, stratum_widths, dtype=np.uint64)


def _sampled_arrow_row_weights(sampled: Any) -> Any:
    """Estimate decoded byte weights for an already sampled Arrow table.

    Sampling before calling this helper is important: variable-width lengths
    are evaluated for only the control-plane rows, never for the full input.
    Fixed-width values use their byte width. Variable-width values also include
    one offset per row, and nullable fields include a conservative validity byte.
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
        if pa.types.is_string(typ):
            variable.append((field.name, 4))
        elif pa.types.is_large_string(typ):
            variable.append((field.name, 8))
        elif pa.types.is_boolean(typ):
            fixed += 1
        elif hasattr(typ, "bit_width"):
            fixed += max(1, int(typ.bit_width) // 8)
        else:
            raise NotImplementedError(
                f"GPU sort cannot calculate row weights for Arrow type {typ}."
            )
        if field.nullable:
            fixed += 1

    weights = np.full(sampled.num_rows, max(1, fixed), dtype=np.uint64)
    for name, offset_width in variable:
        lengths = pc.fill_null(pc.binary_length(sampled[name]), 0).to_numpy(
            zero_copy_only=False
        )
        weights += lengths.astype(np.uint64, copy=False) + offset_width
    return weights


def _validate_payload_schema(schema: Any) -> None:
    """Reject Arrow types that the GPU path does not handle end to end."""

    import pyarrow as pa

    def supported(data_type: Any) -> bool:
        return bool(
            pa.types.is_boolean(data_type)
            or pa.types.is_string(data_type)
            or pa.types.is_large_string(data_type)
            or pa.types.is_integer(data_type)
            or (pa.types.is_floating(data_type) and not pa.types.is_float16(data_type))
            or (
                pa.types.is_decimal(data_type) and not pa.types.is_decimal256(data_type)
            )
            or pa.types.is_date(data_type)
            or pa.types.is_timestamp(data_type)
            or pa.types.is_duration(data_type)
        )

    unsupported = [
        f"{field.name}: {field.type}"
        for field in schema
        if isinstance(field.type, pa.ExtensionType) or not supported(field.type)
    ]
    if unsupported:
        raise NotImplementedError(
            "GPU sort supports flat cuDF-compatible Arrow columns only; "
            f"unsupported columns: {unsupported}."
        )


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
        }

    arrow = pa.concat_tables(samples)
    ordering_names = (sample_block_name, sample_stratum_name)
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
                total_weight * part // num_partitions
                for part in range(1, num_partitions)
            ],
            dtype=np.uint64,
        )
        # A boundary is the first row that starts after the target byte.
        positions = np.searchsorted(cumulative, targets, side="right")
        positions = np.minimum(positions, rows - 1)
        boundaries = sorted_arrow.select(key_columns).take(pa.array(positions))
    return {
        "boundaries": boundaries,
        "sample_rows": int(planning_arrow.num_rows),
    }


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


def _coalesce_arrow_inputs(tables: List[Any], target_bytes: int) -> Iterator[Any]:
    """Yield ordered Arrow batches no larger than the GPU admission target.

    Ray blocks are normally smaller than ``target_bytes``, but block size is
    not an API guarantee. Oversized blocks are sliced by rows before H2D so a
    single block cannot bypass the device-memory bound.
    """

    import pyarrow as pa

    if target_bytes <= 0:
        raise ValueError("GPU sort input batch target must be positive.")

    def bounded_slices(table: Any) -> Iterator[Any]:
        start = 0
        while start < table.num_rows:
            remaining = table.slice(start)
            remaining_bytes = int(remaining.nbytes)
            if remaining_bytes <= target_bytes:
                yield remaining
                return

            rows = max(
                1,
                int(remaining.num_rows) * target_bytes // max(1, remaining_bytes),
            )
            part = remaining.slice(0, rows)
            while int(part.nbytes) > target_bytes and rows > 1:
                smaller = max(1, rows * target_bytes // int(part.nbytes))
                rows = min(rows - 1, smaller)
                part = remaining.slice(0, rows)
            if int(part.nbytes) > target_bytes:
                raise GPUSortCapacityError(
                    "One Arrow row exceeds the current GPU input-batch limit: "
                    f"row_bytes={part.nbytes}, limit_bytes={target_bytes}."
                )
            yield part
            start += rows

    pending: List[Any] = []
    pending_bytes = 0
    for original in tables:
        if original.num_rows == 0:
            continue
        for table in bounded_slices(original):
            table_bytes = int(table.nbytes)
            if pending and pending_bytes + table_bytes > target_bytes:
                yield pending[0] if len(pending) == 1 else pa.concat_tables(pending)
                pending = []
                pending_bytes = 0
            pending.append(table)
            pending_bytes += table_bytes
            if pending_bytes >= target_bytes:
                yield pending[0] if len(pending) == 1 else pa.concat_tables(pending)
                pending = []
                pending_bytes = 0
    if pending:
        yield pending[0] if len(pending) == 1 else pa.concat_tables(pending)


def lazy_load_backend() -> type[Any]:
    """Build the implementation only inside a one-GPU Ray actor."""

    from rapidsmpf.utils.ray_utils import BaseShufflingActor

    class GPURangeSortBackend(BaseShufflingActor):  # pragma: no cover - GPU only
        """One GPU rank of the distributed range sort."""

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
            if int(num_partitions) != int(nranks):
                raise ValueError(
                    "GPU sort requires exactly one ordered range per GPU rank."
                )
            self._key_columns = list(key_columns)
            self._ascending = [bool(value) for value in ascending]
            self._num_partitions = int(num_partitions)
            # A transport partition holds one source rank's contribution to a
            # logical range. This lets the receiver release skewed inputs one
            # source at a time instead of waiting for an all-source partition.
            self._transport_partitions = self._num_partitions * int(nranks)
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
            self._boundary_keys = None

            self._mr = None
            self._buffer_resource = None
            self._pool_max_bytes = 0
            self._memory_budget_bytes = 0
            self._payload_limit_bytes = 0
            self._run_chunk_bytes = 0

            self._device_tables: Dict[int, List[Any]] = {
                partition: [] for partition in range(index, num_partitions, nranks)
            }
            self._device_bytes: Dict[int, int] = dict.fromkeys(self._device_tables, 0)
            self._runs: Dict[int, List[_ExternalRun]] = {
                partition: [] for partition in self._device_tables
            }
            self._row_ordinal = 0
            self._duplicate_kernel = None

        # -- setup and schemas -------------------------------------------

        def setup_worker(self, root_address_bytes: bytes) -> Dict[str, Any]:
            import rmm
            from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor

            try:
                from rapidsmpf.memory.buffer_resource import BufferResource
            except ModuleNotFoundError as exc:
                if exc.name not in {
                    "rapidsmpf.memory",
                    "rapidsmpf.memory.buffer_resource",
                }:
                    raise
                # RAPIDS-MPF 25.12 used this package name and had no separate
                # pinned-memory resource argument.
                from rapidsmpf.buffer.resource import BufferResource

                buffer_resource_kwargs = {}
            else:
                buffer_resource_kwargs = {"pinned_mr": None}

            super().setup_worker(root_address_bytes)
            # Communicator ranks are assigned during bootstrap and need not
            # match Ray actor creation order. Partition ownership must follow
            # the actual MPF rank.
            actual_rank = int(self.rank())
            self._device_tables = {
                partition: []
                for partition in range(actual_rank, self._num_partitions, self.nranks())
            }
            self._device_bytes = dict.fromkeys(self._device_tables, 0)
            self._runs = {partition: [] for partition in self._device_tables}
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
            # RAPIDS-MPF row buffers remain device-only.  The sort's explicit
            # Arrow runs, not MPF pressure spill, are the recovery mechanism.
            self._buffer_resource = BufferResource(
                device_mr=self._mr,
                memory_available=None,
                **buffer_resource_kwargs,
            )
            self._pool_max_bytes = maximum
            requested = self._config.residency_budget_bytes
            self._memory_budget_bytes = min(int(requested or maximum), maximum)
            if self._memory_budget_bytes < 16 << 20:
                raise GPUSortCapacityError(
                    "GPU sort residency budget must be at least 16 MiB."
                )

            # Managed cuDF spilling would introduce a second, implicit host
            # spill path. External GPU runs are the only supported path here.
            spill_options = {
                "spill": False,
                "spill_on_demand": False,
                "spill_device_limit": None,
            }
            try:
                import cudf

                for name, value in spill_options.items():
                    cudf.set_option(name, value)
                observed = {name: cudf.get_option(name) for name in spill_options}
            except (AttributeError, KeyError, ValueError) as exc:
                raise RuntimeError(
                    "GPU sort requires cuDF options that disable managed spilling."
                ) from exc
            if observed != spill_options:
                raise RuntimeError(
                    "GPU sort could not disable cuDF managed spilling: "
                    f"expected={spill_options}, observed={observed}."
                )
            return {
                "rank": self.rank(),
                "pool_max_bytes": maximum,
                "memory_budget_bytes": self._memory_budget_bytes,
            }

        def _set_schema(self, schema: Any) -> None:
            import pyarrow as pa

            schema = getattr(schema, "base_schema", schema)
            if not isinstance(schema, pa.Schema):
                raise TypeError("GPU sort requires a PyArrow schema.")
            if self._arrow_schema is not None:
                if not self._arrow_schema.equals(schema, check_metadata=False):
                    raise TypeError("GPU sort input blocks do not share one schema.")
                return
            _validate_payload_schema(schema)
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
            for key, ascending in zip(self._key_columns, self._ascending):
                if key in self._float_hidden:
                    category_ascending = self._config.null_position == "last"
                    category_order = (
                        plc.types.Order.ASCENDING
                        if category_ascending
                        else plc.types.Order.DESCENDING
                    )
                    category_nulls = (
                        plc.types.NullOrder.AFTER
                        if category_ascending
                        else plc.types.NullOrder.BEFORE
                    )
                    order.extend([category_order, category_order])
                    nulls.extend([category_nulls, category_nulls])
                value_order = (
                    plc.types.Order.ASCENDING
                    if ascending
                    else plc.types.Order.DESCENDING
                )
                # libcudf's BEFORE/AFTER is the requested position relative to
                # non-null values, independent of the value sort direction.
                value_nulls = (
                    plc.types.NullOrder.BEFORE
                    if self._config.null_position == "first"
                    else plc.types.NullOrder.AFTER
                )
                order.append(value_order)
                nulls.append(value_nulls)
            return order, nulls

        # -- bounded GPU sampling and boundary selection -----------------

        def sample_blocks(
            self,
            blocks: List[Any],
            *,
            block_ordinals: List[int],
            sample_quotas: List[int],
            seed: int,
        ) -> Dict[str, Any]:
            import numpy as np
            import pyarrow as pa

            if not (len(blocks) == len(block_ordinals) == len(sample_quotas)):
                raise ValueError("GPU sort sampling plan does not match its blocks.")
            samples = []
            for block, block_ordinal, quota in zip(
                blocks, block_ordinals, sample_quotas
            ):
                arrow = self._to_arrow_table(block)
                self._set_schema(arrow.schema)
                rows = int(arrow.num_rows)
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
                )
                sample_schema = pa.schema(
                    [self._arrow_schema.field(name) for name in self._key_columns]
                    + [
                        pa.field(self._weight_name, pa.uint64(), nullable=False),
                        pa.field(self._sample_block_name, pa.uint64(), nullable=False),
                        pa.field(
                            self._sample_stratum_name, pa.uint64(), nullable=False
                        ),
                    ]
                )
                samples.append(selected.cast(sample_schema))
            sample = pa.concat_tables(samples) if samples else None
            return {
                "schema": self._arrow_schema,
                "sample": sample,
            }

        def compute_boundaries(self, samples: List[Any], schema: Any) -> Dict[str, Any]:
            self._set_schema(schema)
            return _cpu_sample_boundaries(
                samples,
                schema=self._arrow_schema,
                key_columns=self._key_columns,
                ascending=self._ascending,
                num_partitions=self._num_partitions,
                null_position=self._config.null_position,
                weight_name=self._weight_name,
                sample_block_name=self._sample_block_name,
                sample_stratum_name=self._sample_stratum_name,
            )

        def install_plan(self, schema: Any, boundaries: Any) -> None:
            import cudf
            from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table

            self._set_schema(schema)
            boundary_frame = cudf.DataFrame.from_arrow(boundaries)
            boundary_table = cudf_to_pylibcudf_table(boundary_frame)
            boundary_table, names = self._augment_table(
                boundary_table, list(boundary_frame.columns)
            )
            self._boundary_keys = self._comparison_table(boundary_table, names)
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

            keys = self._comparison_table(table, self._work_names)
            order, nulls = self._order_and_nulls()
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
                return {}
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
            packed = split_and_pack(
                partitioned,
                normalized[1:-1],
                DEFAULT_STREAM,
                self._buffer_resource,
            )
            lane_base = int(self.rank()) * self._num_partitions
            return {
                lane_base + int(destination): chunk
                for destination, chunk in packed.items()
            }

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
            )
            return unpack_and_concat(device, DEFAULT_STREAM, self._buffer_resource)

        def process_wave(self, wave_id: int, blocks: List[Any]) -> None:
            """Range-shuffle one bounded wave and consume completed source lanes."""

            import cupy as cp
            import cudf

            if self._boundary_keys is None:
                raise RuntimeError("GPU sort plan must be installed before a wave.")
            shuffler = self.create_shuffler(
                (self.DATA_OP_BASE + int(wave_id)) % 256,
                total_num_partitions=self._transport_partitions,
                buffer_resource=self._buffer_resource,
            )
            owners = []
            try:
                for block in blocks:
                    arrow = self._to_arrow_table(block)
                    self._set_schema(arrow.schema)
                    owners.append(arrow)
                batch_target = _input_batch_target_bytes(
                    self._config.exchange_batch_bytes,
                    self._pool_max_bytes,
                    int(self._mr.current_allocated),
                    self._config.final_sort_workspace_factor,
                )
                for arrow in _coalesce_arrow_inputs(owners, batch_target):
                    frame = cudf.DataFrame.from_arrow(arrow)
                    packed = self._partition_and_pack(frame, wave_id)
                    if packed:
                        shuffler.insert_chunks(packed)
                    del packed, frame
                # One barrier proves cuDF has stopped reading all Arrow owners.
                cp.cuda.runtime.deviceSynchronize()
                owners.clear()
                shuffler.insert_finished(list(range(self._transport_partitions)))

                # MPF makes a partition extractable only after all sources
                # finish it. Source-rank lanes keep each extract bounded to one
                # source contribution while preserving logical range ownership.
                while not shuffler.finished():
                    transport_partition = int(shuffler.wait_any())
                    partition = transport_partition % self._num_partitions
                    if partition != int(self.rank()):
                        raise RuntimeError(
                            "RAPIDS-MPF returned a GPU sort range owned by "
                            f"another rank: partition={partition}, rank={self.rank()}."
                        )
                    chunks = shuffler.extract(transport_partition)
                    while chunks:
                        chunk = chunks.pop()
                        table = self._unpack_one(chunk)
                        cp.cuda.runtime.deviceSynchronize()
                        # ``unpack_and_concat`` owns the resulting table. Drop
                        # the consumed packed receive buffer before a retained
                        # range may need its run-sort workspace.
                        del chunk
                        self._accept_received(partition, table)
                        del table
                cp.cuda.runtime.deviceSynchronize()
            finally:
                owners.clear()
                shuffler.shutdown()

        def _accept_received(self, partition: int, table: Any) -> None:
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
                    self._externalize_device_tables(partition)
                self._device_tables[partition].append(piece)
                self._device_bytes[partition] += piece_bytes
                if self._device_bytes[partition] >= self._payload_limit_bytes:
                    self._externalize_device_tables(partition)

        def _externalize_device_tables(self, partition: int) -> None:
            import cupy as cp
            import pylibcudf as plc

            tables = self._device_tables[partition]
            if not tables:
                return
            self._device_tables[partition] = []
            self._device_bytes[partition] = 0
            while tables:
                cp.cuda.runtime.deviceSynchronize()
                # Concurrent MPF receive buffers count against the same RMM
                # pool. They are already included in ``current_allocated``;
                # bound both concatenate and final-sort workspace by the
                # actual remaining headroom.
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
                        "GPU sort cannot reserve run-sort workspace while "
                        "shuffle buffers are live. Reduce the automatic wave size."
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
                if len(group) == 1:
                    table = group.pop()
                else:
                    table = plc.concatenate.concatenate(group)
                    cp.cuda.runtime.deviceSynchronize()
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
                            "GPU sort cannot reserve run-sort workspace while "
                            "shuffle buffers are live. Reduce the automatic wave size."
                        )
                    remaining = self._slice_table(table, start, rows)
                    remaining_bytes = self._table_bytes(remaining)
                    piece_rows = max(
                        1,
                        int((rows - start) * piece_limit / max(1, remaining_bytes)),
                    )
                    end = min(rows, start + piece_rows)
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
                            "One GPU sort row exceeds the available run-sort workspace."
                        )

                    sorted_piece = self._sort_table(piece)
                    cp.cuda.runtime.deviceSynchronize()
                    run = self._store_table_as_run(sorted_piece)
                    self._runs[partition].append(run)
                    del sorted_piece, piece, remaining
                    start = end
                del table

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
                arrow = self._table_to_work_arrow(part)
                cp.cuda.runtime.deviceSynchronize()
                yield arrow

        def _store_table_as_run(self, table: Any) -> _ExternalRun:
            """D2H and seal one sorted GPU table as bounded Arrow chunks."""

            import ray

            run = _ExternalRun()
            for arrow in self._iter_table_as_work_arrow(table):
                size_bytes = int(arrow.nbytes)
                ref = ray.put(arrow)
                run.chunks.append(_RunChunk(ref=ref, size_bytes=size_bytes))
            return run

        # -- bounded GPU-only external merge -----------------------------

        def _load_run_chunk(self, chunk: _RunChunk):
            import cudf
            import ray
            from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table

            arrow = ray.get(chunk.ref)
            frame = cudf.DataFrame.from_arrow(arrow)
            return cudf_to_pylibcudf_table(frame)

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
                for state in states:
                    if state["table"] is None and state["next"] < len(
                        state["run"].chunks
                    ):
                        state["table"] = self._load_run_chunk(
                            state["run"].chunks[state["next"]]
                        )
                        state["next"] += 1
                active = [state for state in states if state["table"] is not None]
                if not active:
                    break
                if len(active) == 1:
                    # With no competing head, the remaining stream is already
                    # globally ordered. Yield the resident suffix and then
                    # reuse unread ObjectRefs without a GPU round-trip.
                    state = active[0]
                    yield state["table"]
                    state["table"] = None
                    yield from state["run"].chunks[state["next"] :]
                    state["next"] = len(state["run"].chunks)
                    continue

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
                yield merged
            cp.cuda.runtime.deviceSynchronize()

        def _merge_group(self, group: List[_ExternalRun]) -> _ExternalRun:
            """Materialize one bounded intermediate GPU merge group."""

            output = _ExternalRun()
            for source in self._iter_merged_sources(group):
                if isinstance(source, _RunChunk):
                    output.chunks.append(source)
                else:
                    replacement = self._store_table_as_run(source)
                    output.chunks.extend(replacement.chunks)
            # All output references are sealed before the prior-pass owners
            # are released.  Reused chunks remain owned by ``output``.
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
            """Stream the last merge pass without a replacement Plasma run."""

            import ray

            if len(final_group) == 1:
                sources: Iterator[Any] = iter(final_group[0].chunks)
            else:
                sources = self._iter_merged_sources(final_group)

            for source in sources:
                if isinstance(source, _RunChunk):
                    arrow = ray.get(source.ref)
                    yield arrow
                else:
                    yield from self._iter_table_as_work_arrow(source)

        # -- Arrow output -------------------------------------------------

        @staticmethod
        def _fixed_width_bytes(data_type: Any) -> Optional[int]:
            import pyarrow as pa

            if pa.types.is_boolean(data_type):
                return None
            if (
                pa.types.is_integer(data_type)
                or pa.types.is_floating(data_type)
                or pa.types.is_decimal(data_type)
                or pa.types.is_date32(data_type)
                or pa.types.is_date64(data_type)
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
            for index, item in enumerate(plan):
                if item is None:
                    continue
                field, column, kind, *_ = item
                if kind is None:
                    array = column.to_arrow()
                    arrays[index] = (
                        array
                        if array.type.equals(field.type)
                        else array.cast(field.type)
                    )

            # Only pinned-host exhaustion falls back to cuDF's regular Arrow
            # conversion. CUDA copies, synchronization, and buffer invariants
            # must fail closed so they cannot be mistaken for a valid result.
            def allocate_pinned(size: int):
                try:
                    return cp.cuda.alloc_pinned_memory(size)
                except cp.cuda.runtime.CUDARuntimeError as exc:
                    # CUDA error 2 is cudaErrorMemoryAllocation. Other runtime
                    # errors indicate a broken CUDA operation and must escape.
                    if exc.status != 2:
                        raise
                    raise MemoryError("Pinned-host allocation failed.") from exc

            try:
                for index, item in enumerate(plan):
                    if item is None:
                        continue
                    field, column, kind, data_bytes, extra_bytes, mask_bytes = item
                    if kind is None:
                        continue
                    host_data = allocate_pinned(data_bytes)
                    host_mask = None
                    if mask_bytes:
                        host_mask = allocate_pinned(mask_bytes)
                        if column.mask is None:
                            raise RuntimeError("Nullable cuDF column has no mask.")
                    host_extra = None
                    if extra_bytes:
                        host_extra = allocate_pinned(extra_bytes)
                    pending[index] = (
                        field,
                        column,
                        kind,
                        host_data,
                        host_extra,
                        host_mask,
                    )
            except (MemoryError, cp.cuda.memory.OutOfMemoryError):
                pending.clear()
                arrays.clear()
                host_data = host_mask = host_extra = None
                return self._generic_frame_to_arrow(frame, schema)

            for item in pending.values():
                _, column, kind, host_data, host_extra, host_mask = item
                source_data = (
                    column.data if kind == "fixed" else column.children[0].data
                )
                with _read_device_buffer(source_data) as source_ptr:
                    cp.cuda.runtime.memcpyAsync(
                        host_data.ptr,
                        source_ptr,
                        memoryview(host_data).nbytes,
                        cp.cuda.runtime.memcpyDeviceToHost,
                        stream.ptr,
                    )
                if host_mask is not None:
                    with _read_device_buffer(column.mask) as mask_ptr:
                        cp.cuda.runtime.memcpyAsync(
                            host_mask.ptr,
                            mask_ptr,
                            memoryview(host_mask).nbytes,
                            cp.cuda.runtime.memcpyDeviceToHost,
                            stream.ptr,
                        )
                if host_extra is not None:
                    chars = column.children[1].data
                    with _read_device_buffer(chars) as chars_ptr:
                        cp.cuda.runtime.memcpyAsync(
                            host_extra.ptr,
                            chars_ptr,
                            memoryview(host_extra).nbytes,
                            cp.cuda.runtime.memcpyDeviceToHost,
                            stream.ptr,
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
                        (
                            pa.py_buffer(memoryview(host_extra))
                            if host_extra is not None
                            else pa.py_buffer(b"")
                        ),
                    ]
                arrays[index] = pa.Array.from_buffers(
                    field.type,
                    rows,
                    buffers,
                    null_count=int(column.null_count),
                )
            return pa.Table.from_arrays(arrays, schema=schema)

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

        @staticmethod
        def _block_metadata(block: Any, block_ser_time_s: Optional[float]):
            """Build metadata after Ray confirms that the block is sealed."""

            from ray.data.block import BlockExecStats, BlockMetadataWithSchema

            stats = BlockExecStats.builder()
            return BlockMetadataWithSchema.from_block(
                block,
                block_exec_stats=stats.build(block_ser_time_s=block_ser_time_s or 0),
            )

        def _yield_block(self, block: Any) -> Iterator[Any]:
            """Emit a block with Ray Data's streaming metadata protocol."""

            from ray.data._internal.execution.util import yield_block_with_stats

            yield from yield_block_with_stats(
                block,
                lambda block_ser_time_s: self._block_metadata(block, block_ser_time_s),
            )

        def finish_and_extract(self) -> Iterator[Any]:
            """Finish resident ranges or GPU-merge external runs and stream blocks."""

            import cupy as cp
            import pylibcudf as plc

            for partition in sorted(self._device_tables):
                runs = self._runs[partition]
                if runs:
                    if self._device_tables[partition]:
                        self._externalize_device_tables(partition)
                        runs = self._runs[partition]
                    final_group = self._merge_runs_until_final(runs)
                    self._runs[partition] = final_group
                    # Materialize intermediate merge passes, but stream the
                    # final fan-in directly to Ray. This avoids writing the
                    # final replacement run to Plasma only to read and seal it
                    # again as the operator output.
                    target_output_bytes = max(128 << 10, self._run_chunk_bytes)
                    work_tables = self._iter_final_work_arrow(final_group)
                    for block in self._iter_coalesced_output_blocks(
                        work_tables, target_output_bytes
                    ):
                        yield from self._yield_block(block)
                    # The final output has been sealed before its source refs
                    # are released. Reused chunks remain live throughout the
                    # streaming generator's one-block look-behind.
                    for run in final_group:
                        run.chunks.clear()
                    self._runs[partition] = []
                elif self._device_tables[partition]:
                    tables = self._device_tables[partition]
                    self._device_tables[partition] = []
                    self._device_bytes[partition] = 0
                    table = (
                        tables[0]
                        if len(tables) == 1
                        else plc.concatenate.concatenate(tables)
                    )
                    table = self._sort_table(table)
                    cp.cuda.runtime.deviceSynchronize()
                    work_arrow = self._table_to_work_arrow(table)
                    cp.cuda.runtime.deviceSynchronize()
                    block = self._output_arrow(work_arrow)
                    yield from self._yield_block(block)
                else:
                    block = self._arrow_schema.empty_table()
                    yield from self._yield_block(block)

        # -- cleanup -----------------------------------------------------

        def release(self) -> None:
            for tables in self._device_tables.values():
                tables.clear()
            for runs in self._runs.values():
                for run in runs:
                    run.chunks.clear()
                runs.clear()
            self._boundary_keys = None
            gc.collect()

    return GPURangeSortBackend


_BACKEND_CLASS: Optional[type[Any]] = None


def get_backend_class() -> type[Any]:
    global _BACKEND_CLASS
    if _BACKEND_CLASS is None:
        _BACKEND_CLASS = lazy_load_backend()
    return _BACKEND_CLASS
