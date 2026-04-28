# GPU Data Operations Design

Summary of changes in commit `cd42395df3` to enable GPU-native `group_by`/`map_groups` and `join` operations in Ray Data.

Both leverage RAPIDS MPF (rapidsmpf) for hash-based shuffle over UCXX (GPU-direct communication) so that data stays on GPU throughout, avoiding costly CPU round-trips. 17 files changed (+2,854 / -103 lines).

---

## Architecture Overview

### 1. New cuDF Block Abstraction (`cudf_block.py` — 306 lines, new file)

A full `BlockAccessor` / `BlockBuilder` implementation for cuDF DataFrames, paralleling the existing Arrow and Pandas block accessors:

- **`CudfBlockAccessor`** — Implements `slice`, `take`, `select`, `sort`, `to_arrow`, `to_pandas`, `to_numpy`, `to_cudf`, `schema`, `size_bytes`, `merge_sorted_blocks`, etc. Key design choice: `_get_group_boundaries_sorted()` detects group transitions entirely on GPU via cuDF `shift`/compare, transferring only the small boundary-index array to CPU.
- **`CudfBlockBuilder`** — Builds cuDF DataFrames from dicts/lists, supports `concat`, and reports `BlockType.CUDF`.
- **`CudfRow`** — A `Mapping`-compatible row wrapper that converts cuDF scalars to Python native types.

The new `BlockType.CUDF` enum variant is added to `block.py`, and `BlockAccessor.for_block()` now dispatches cuDF DataFrames to `CudfBlockAccessor`.

### 2. Fused GPU Shuffle + map_groups (`GPUShuffleMapGroups`)

**Problem**: The CPU path does shuffle -> Arrow object store -> downstream `MapBatches` UDF, causing a cuDF->Arrow->cuDF round-trip for each partition.

**Solution**: A fused operator that runs the UDF *inside* the GPU shuffle actor immediately after extraction.

#### Logical layer
- New **`GPUShuffleMapGroups`** logical operator (in `all_to_all_operator.py`) stores the UDF, key columns, batch format, and constructor args.
- `GroupedData.map_groups()` detects `ShuffleStrategy.GPU_SHUFFLE` and emits this operator instead of the usual Repartition + MapBatches chain.

#### Physical layer
- **`GPUShuffleOperator`** gains optional `post_shuffle_udf` parameters. When set, `_try_finalize()` calls `extract_and_apply` instead of `extract_partitions`.
- New actor method **`extract_and_apply()`** on `GPUShuffleActor`:
  1. Extracts partitions from rapidsmpf (GPU-resident cuDF).
  2. Sorts by key columns.
  3. Applies the user's UDF directly on the cuDF partition.
  4. Converts only the (typically smaller) UDF result to Arrow for the object store.
  5. Supports callable-class UDFs (instantiated once per rank, like `ActorPoolMapOperator`).

#### Refactoring of `GPUShuffleActor`
- `finish_and_extract()` split into separate `insert_finished()` + `extract_partitions()` — enables fire-and-forget submission relying on Ray's per-actor FIFO ordering.
- `_columns` deferred: set via `set_columns()` broadcast from the operator, handling ranks that receive no direct `insert_batch` calls.
- Graceful shutdown via `shutdown_and_exit()` (cleans up UCXX/rapidsmpf state before exiting).
- Finalization simplified: `_is_inserting_done()` now only requires inserts *submitted* (not completed), relying on actor task ordering.

### 3. GPU-Native Join Operator (`join.py` — 894 lines, new file)

A three-phase hash join:

#### Phase 1 — Right Shuffle
- Right-side blocks are hash-partitioned by `right_key_columns` via rapidsmpf and routed to GPU actors via UCXX.
- When all right blocks are inserted, each actor calls `right_insert_finished()`, extracts its partitions, and stores them as a single `cudf.DataFrame` (`_stored_right_df`).
- The shuffler is then reset for Phase 2 via `reset_for_left_shuffle()`.

#### Phase 2 — Left Shuffle
- Left blocks are hash-partitioned by `left_key_columns` using the *same* hash space, so co-located keys land on the same actor.
- Left blocks arriving before right-phase completion are buffered as object refs (no data copy) in `_pending_left_bundles`.
- **Chunked left shuffle** (optional): when `gpu_join_left_chunk_rows > 0`, left blocks are cut into chunks that trigger intermediate join executions, keeping GPU memory bounded.

#### Phase 3 — Join
- Each actor calls `cudf.merge()` between its extracted left partitions and `_stored_right_df`.
- Results stream back as Arrow Tables via the Ray Data streaming generator protocol.
- **All 8 join types** supported: INNER, LEFT_OUTER, RIGHT_OUTER, FULL_OUTER, LEFT_SEMI, RIGHT_SEMI, LEFT_ANTI, RIGHT_ANTI.
- **Deferred right-only rows** (for FULL_OUTER / RIGHT_OUTER): right-only candidates are accumulated across chunks and reconciled in the final flush, since a right row unmatched in chunk N may match in chunk N+1.

#### `GPUJoinActor`
- Wraps `BulkRapidsMPFJoinShuffler` by composition (deferred import).
- Schema broadcasting: `set_right_schema()` / `set_left_schema()` ensure ranks with no batches produce correctly-typed empty DataFrames.
- `_normalize_schema_for_cudf()` helper: cuDF's `to_arrow()` always emits `large_string` (64-bit offsets); the helper upgrades `string` fields to `large_utf8` to prevent cast failures.

#### `GPUJoinRankPool`
- Manages actor lifecycle: creation, UCXX communicator setup (root + worker), round-robin block distribution, and graceful/forced shutdown.

#### `GPUJoinOperator`
- Two-input `PhysicalOperator` with `SubProgressBarMixin`.
- Separate round-robin counters for right and left distribution.
- Phase transitions driven by `input_done(input_index)` and `all_inputs_done()`.
- Task IDs for left inserts offset by 10M to avoid collision with right task IDs.

### 4. RAPIDS MPF Backend Extension (`rapidsmpf_backend.py`)

New **`BulkRapidsMPFJoinShuffler`** class extends `BulkRapidsMPFShuffler`:
- Shares a **single `ProgressThread`** across both shuffle phases (creating a second one on the same communicator causes SIGABRT in rapidsmpf).
- `reset_for_left_shuffle(left_keys)` — creates a new `Shuffler` with an incremented `op_id` without calling `shutdown()` on the old one (preserves the shared `ProgressThread`).
- Custom `cleanup()` drops references without calling `shutdown()` to avoid invalidating the shared thread.

### 5. cuDF->Arrow Conversion Optimization (`map_operator.py` + `planner.py`)

**Problem**: In a chain of `MapBatches(batch_format="cudf")` operators, each step converts cuDF->Arrow for the object store and the next step converts Arrow->cuDF back.

**Solution**:
- `MapOperator` gains a `_convert_cudf_output` flag (default `True`).
- `_map_task()` checks `BlockType.CUDF` and the flag; when `False`, skips the cuDF->Arrow conversion.
- `Planner._configure_cudf_boundaries()` walks the physical DAG: for each cuDF-producing `MapOperator` whose *every* consumer is also a cuDF operator, sets `_convert_cudf_output = False`. Only the final cuDF operator in a chain performs the conversion.

### 6. Planner Integration (`planner.py` + `plan_all_to_all_op.py`)

- `plan_join_op()` checks `ShuffleStrategy.GPU_SHUFFLE` and returns `GPUJoinOperator` instead of the CPU `JoinOperator`.
- `plan_all_to_all_op()` dispatches `GPUShuffleMapGroups` to `_plan_gpu_shuffle_map_groups()`, which constructs a `GPUShuffleOperator` with the fused UDF parameters.

### 7. Configuration (`context.py`)

- New `gpu_join_left_chunk_rows: int = 0` — controls chunked left shuffle for GPU joins (0 = single pass).

### 8. Other Changes

- **`table_block.py`** — `to_batch_format()` now handles `BlockType.CUDF` -> `to_cudf()`.
- **`block.py`** — `batch_to_block()` uses `preserve_index=False` for cuDF->Arrow to avoid spurious index columns.
- **`grouped_data.py`** — `_apply_udf_to_groups()` bulk-converts Arrow->cuDF once when `batch_format="cudf"` instead of per-group.
- **`test_gpu_join.py`** (710 lines) — Comprehensive tests: planner routing, schema normalization, join type mapping, deferred right-only logic, chunked left shuffle, and GPU integration tests.
- **`cudf_transport_utils.py`** + **`test_gpu_objects_cudf.py`** — GPU object manager utilities for cuDF transport.
