# GPU Data Operations Design

## Problem

Ray Data's `group_by`/`map_groups` and `join` operations require all-to-all data shuffles. When data lives on GPUs as cuDF DataFrames, the existing CPU-based shuffle path forces repeated GPU-to-CPU-to-GPU round-trips: every operator boundary serializes cuDF to Arrow for the Ray object store, and the next operator deserializes Arrow back to cuDF. These conversions dominate wall-clock time for GPU workloads and waste GPU memory bandwidth, negating the speed advantage of keeping computation on-GPU.

There are three specific pain points:

1. **group_by/map_groups**: The shuffle and the UDF run as separate operators with an Arrow materialization step in between, even though the UDF's input is the shuffle's output.
2. **join**: Both sides are shuffled through CPU, joined on CPU, and the result is converted back to GPU — the GPU is idle during the most expensive phase.
3. **Chained cuDF MapBatches**: A pipeline of N consecutive `MapBatches(batch_format="cudf")` steps performs N-1 unnecessary cuDF->Arrow->cuDF round-trips at intermediate boundaries.

## Current Architecture

### group_by / map_groups (CPU path)

```
                        Ray Object Store (Arrow)
                        ~~~~~~~~~~~~~~~~~~~~~~~~
  cuDF blocks                                          cuDF blocks
  on GPU                                               on GPU
      |                                                    ^
      v                                                    |
 +---------+    +-------+    +---------------+    +----------------+
 | cuDF -> |    | CPU   |    | Arrow ->      |    | cuDF ->        |
 | Arrow   |--->| Hash  |--->| Object Store  |--->| MapBatches UDF |
 | (ser)   |    | Shuf. |    | (materialize) |    | (deser + run)  |
 +---------+    +-------+    +---------------+    +----------------+
      |                            |                       |
      v                            v                       v
    GPU->CPU                 CPU memory                 CPU->GPU
    transfer                 copy                       transfer
```

Each arrow in the diagram is a serialization boundary. The shuffle runs entirely on CPU, and the UDF's input must be deserialized from Arrow back to cuDF — even though both the source and destination are GPU memory.

### join (CPU path)

```
 Left (GPU)                                            Right (GPU)
     |                                                       |
     v                                                       v
 +--------+    +-------+                  +-------+    +--------+
 | cuDF-> |    | CPU   |                  | CPU   |    | cuDF-> |
 | Arrow  |--->| Hash  |---+          +---| Hash  |<---| Arrow  |
 | (ser)  |    | Shuf. |   |          |   | Shuf. |    | (ser)  |
 +--------+    +-------+   |          |   +-------+    +--------+
                           v          v
                     +----------------------+
                     |    CPU Join Op       |
                     | (Arrow object store) |
                     +----------------------+
                                |
                                v
                     +----------------------+
                     | Arrow -> cuDF (GPU)  |
                     +----------------------+
```

Both sides leave GPU memory for the shuffle, the join itself runs on CPU against Arrow tables, and the result must be transferred back to GPU.

### Chained cuDF MapBatches (current)

```
 +-------------+      +-------------+      +-------------+
 | MapBatches  |      | MapBatches  |      | MapBatches  |
 | (cudf UDF)  |      | (cudf UDF)  |      | (cudf UDF)  |
 +------+------+      +------+------+      +------+------+
        |                    |                    |
        v                    v                    v
   cuDF -> Arrow        cuDF -> Arrow        cuDF -> Arrow
   Arrow -> cuDF        Arrow -> cuDF             |
        \___________________/ \___________________/
          unnecessary             unnecessary
          round-trip              round-trip
```

## Proposal

Replace the CPU shuffle and join paths with GPU-native operators that use RAPIDS MPF (rapidsmpf) for hash-partitioned shuffles over UCXX (GPU-direct communication via NVLink/InfiniBand). Data stays in GPU memory from input through shuffle, UDF application or join, and only converts to Arrow at the final output boundary.

### group_by / map_groups (GPU path — fused operator)

```
                    GPU Memory (cuDF throughout)
                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 cuDF blocks ──> GPUShuffleActor (per rank)
                 ┌───────────────────────────────┐
                 │  1. insert_batch()            │
                 │     hash-partition by key     │
                 │     route shards via UCXX     │──> peer actors
                 │                               │    (GPU-direct)
                 │  2. insert_finished()         │
                 │                               │
                 │  3. extract_and_apply()       │
                 │     extract partition (cuDF)  │
                 │     sort by group keys        │
                 │     apply UDF on cuDF         │
                 │     convert RESULT to Arrow   │──> Object Store
                 └───────────────────────────────┘
                                                      (only UDF output
                                                       hits Arrow, not
                                                       the full partition)
```

The UDF runs inside the shuffle actor immediately after extraction. The full input partition never leaves GPU memory — only the (typically much smaller) UDF result is serialized to Arrow.

### join (GPU path — three-phase)

```
                    GPU Memory (cuDF throughout)
                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 Phase 1: Right Shuffle
 ┌──────────────────────────────────────────────────┐
 │  Right blocks ──> hash-partition by right_key    │
 │                   route via UCXX to actors       │
 │  Each actor: extract + store as _stored_right_df │
 │  Reset shuffler for Phase 2                      │
 └──────────────────────────────────────────────────┘
                          |
                          v
 Phase 2: Left Shuffle
 ┌─────────────────────────────────────────────────┐
 │  Left blocks ──> hash-partition by left_key     │
 │                  (same hash space as Phase 1)   │
 │  Co-located keys guaranteed on same actor       │
 └─────────────────────────────────────────────────┘
                          |
                          v
 Phase 3: Local Join
 ┌─────────────────────────────────────────────────┐
 │  Each actor: cudf.merge(left, _stored_right_df) │
 │  Stream results as Arrow ──> Object Store       │
 └─────────────────────────────────────────────────┘
```

Both sides are partitioned by the same key hash space using UCXX, so a left row with key K always lands on the same actor that holds the right rows with key K. The join is a local `cudf.merge` — no cross-actor data movement during the join itself. Supports all 8 join types (inner, left/right/full outer, left/right semi, left/right anti).

For large left sides, an optional **chunked left shuffle** (`gpu_join_left_chunk_rows`) splits the left into bounded chunks with intermediate join executions, keeping GPU memory usage under control while the right side stays resident.

### Chained cuDF MapBatches (optimized boundaries)

```
 +-------------+      +-------------+      +-------------+
 | MapBatches  |      | MapBatches  |      | MapBatches  |
 | (cudf UDF)  |      | (cudf UDF)  |      | (cudf UDF)  |
 +------+------+      +------+------+      +------+------+
        |                    |                    |
        v                    v                    v
   cuDF block ──────> cuDF block ────────> cuDF -> Arrow
   (no conversion)    (no conversion)     (final boundary only)
```

The planner walks the physical DAG and suppresses cuDF->Arrow conversion for any `MapBatches(batch_format="cudf")` operator whose every downstream consumer is also a cuDF operator. Only the last operator in a cuDF chain converts to Arrow.

### Key design decisions

- **Fused shuffle+UDF** rather than separate operators: eliminates the largest serialization cost (full partition through the object store) and is the default path when `ShuffleStrategy.GPU_SHUFFLE` is active.
- **Single shared `ProgressThread`** for join's two shuffle phases: creating a second `ProgressThread` on the same UCXX communicator causes a SIGABRT in rapidsmpf. The `BulkRapidsMPFJoinShuffler` reuses one thread and increments `op_id` to distinguish phases.
- **Fire-and-forget actor task submission**: relies on Ray's per-actor FIFO task ordering for correctness (e.g., `insert_finished` is guaranteed to run after all pending `insert_batch` calls) rather than blocking the driver on acknowledgments.
- **Schema broadcasting**: actors that receive no batches (fewer blocks than ranks) get the Arrow schema via `set_columns()`/`set_right_schema()`/`set_left_schema()` so they produce correctly-typed empty DataFrames instead of null-typed ones.
- **Deferred right-only rows** for outer joins: a right row unmatched in chunk N may be matched in chunk N+1, so right-only candidates are accumulated across all chunks and reconciled only on the final flush.
- **First-class cuDF block type** (`BlockType.CUDF`, `CudfBlockAccessor`, `CudfBlockBuilder`): enables the rest of the system to work with cuDF DataFrames natively, including GPU-native group boundary detection via cuDF `shift`/compare that transfers only the small boundary-index array to CPU.
