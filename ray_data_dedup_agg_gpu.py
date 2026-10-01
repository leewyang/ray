"""
Large-scale deduplication using Ray Data with Spark-like operations.

Based on the approach from: https://huggingface.co/blog/dedup

This implementation uses Ray Data's native operations (map_batches, aggregate, etc.)
to implement MinHash + LSH deduplication, similar to the Spark approach.

This GPU-focused version is derived from ray_data_dedup_agg.py, but rewrites
the hot aggregate stages to use built-in Count/Min AggregateFnV2 operations
and keyed GPU-shuffle/cuDF reductions for connected components.
With DataContext.shuffle_strategy=GPU_SHUFFLE, those groupby().aggregate()
calls are planned as GPUHashAggregateOperator instead of falling back to the
CPU hash aggregate path for custom Python list aggregators.

Architecture:
1. MinHash signature generation (map_batches)
2. LSH banding to generate candidate pairs (GPUHashAggregate Min/Count + join)
3. Connected components (keyed GPU shuffle + block-local cuDF star reductions)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import argparse
import hashlib
import logging
import os
import struct
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ray
from ray.data import ActorPoolStrategy
from ray.data.aggregate import AggregateFnV2, Count, Min
from ray.data.block import Block, BlockAccessor
from ray.data.context import DataContext, ShuffleStrategy
from scipy import integrate

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_HELPER_MODULE_DIRS = (
    _SCRIPT_DIR,
    os.environ.get("RAY_DEDUP_HELPER_DIR"),
    os.path.realpath(os.path.join(_SCRIPT_DIR, "..", "..", "deduplication")),
)
for _helper_module_dir in _HELPER_MODULE_DIRS:
    if (
        _helper_module_dir
        and os.path.exists(os.path.join(_helper_module_dir, "util.py"))
        and _helper_module_dir not in sys.path
    ):
        sys.path.insert(0, _helper_module_dir)
        break

try:
    import cudf
    from minhash_gpu import GPUMinHash
except:
    cudf = None
from util import check_path_exists, list_parquet_files


logger = logging.getLogger(__name__)

# Constants
MERSENNE_PRIME = np.uint64((1 << 61) - 1)
MAX_HASH = np.uint32((1 << 32) - 1)


# ---------------------------------------------------------------------------
# AggregateFnV2 subclasses
# ---------------------------------------------------------------------------

class EdgesFromCollisionsAgg(AggregateFnV2[List, List]):
    """Aggregate doc_ids within a (band_id, band_hash) group into edge pairs.

    Accumulator: flat list of doc_ids (Arrow-serializable as list<string>)
    Output: list of (src, dst) tuples where src = min_doc_id
    """

    _LOG_INTERVAL = 500_000

    def __init__(self):
        super().__init__(
            name="edges_from_collisions",
            on="doc_id",
            ignore_nulls=False,
            zero_factory=lambda: [],
        )
        self._total_groups = 0
        self._singleton_groups = 0
        self._empty_acc_groups = 0
        self._multi_groups = 0
        self._total_edges = 0
        self._total_unique_ids = 0
        self._total_raw_ids = 0
        self._max_group_size = 0

    def aggregate_block(self, block: Block) -> List:
        accessor = BlockAccessor.for_block(block)
        table = accessor.to_arrow()
        return table.column('doc_id').to_pylist()

    def combine(self, current: List, new: List) -> List:
        return current + new

    def finalize(self, acc: List) -> List:
        raw_len = len(acc)
        unique_ids = list(set(acc))
        unique_len = len(unique_ids)

        self._total_groups += 1
        self._total_raw_ids += raw_len
        self._total_unique_ids += unique_len
        if unique_len > self._max_group_size:
            self._max_group_size = unique_len

        if unique_len < 2:
            if raw_len == 0:
                self._empty_acc_groups += 1
            else:
                self._singleton_groups += 1
            if self._total_groups % self._LOG_INTERVAL == 0:
                self._log_stats()
            return []

        min_id = min(unique_ids)
        edges = [(min_id, did) for did in unique_ids]
        self._multi_groups += 1
        self._total_edges += len(edges)

        if self._total_groups % self._LOG_INTERVAL == 0:
            self._log_stats()
        return edges

    def __del__(self):
        if self._total_groups > 0:
            self._log_stats()

    def _log_stats(self):
        import sys
        print(
            f"[EdgesFromCollisionsAgg pid={os.getpid()}] "
            f"total_groups={self._total_groups} "
            f"empty_acc={self._empty_acc_groups} "
            f"singleton={self._singleton_groups} "
            f"multi={self._multi_groups} "
            f"total_raw_ids={self._total_raw_ids} "
            f"total_unique_ids={self._total_unique_ids} "
            f"total_edges={self._total_edges} "
            f"max_group_size={self._max_group_size}",
            file=sys.stderr, flush=True,
        )


class DistinctValuesAgg(AggregateFnV2[List, List]):
    """Collect unique values of a column within each group.

    Accumulator: flat list of values (Arrow-serializable); deduplicated in finalize
    Output: list of unique values
    """

    def __init__(self, on: str):
        self._col = on
        super().__init__(
            name=f"distinct({on})",
            on=on,
            ignore_nulls=False,
            zero_factory=lambda: [],
        )

    def aggregate_block(self, block: Block) -> List:
        accessor = BlockAccessor.for_block(block)
        table = accessor.to_arrow()
        return table.column(self._col).to_pylist()

    def combine(self, current: List, new: List) -> List:
        return current + new

    def finalize(self, acc: List) -> List:
        return list(set(acc))


class LargeStarAgg(AggregateFnV2[List, List]):
    """Large-star aggregation for connected components.

    For each node group, find mp = min(neighbors union {node}),
    then emit (v, mp) for all neighbors v > node.

    Accumulator: flat list [node_val, parent1, parent2, ...] (Arrow-serializable)
    Output: list of (node, parent) tuples
    """

    def __init__(self):
        super().__init__(
            name="large_star(node)",
            on="parent",
            ignore_nulls=False,
            zero_factory=lambda: [],
        )

    def aggregate_block(self, block: Block) -> List:
        accessor = BlockAccessor.for_block(block)
        table = accessor.to_arrow()
        nodes = table.column('node').to_pylist()
        parents = table.column('parent').to_pylist()
        node_val = nodes[0] if nodes else None
        return [node_val] + parents

    def combine(self, current: List, new: List) -> List:
        if not current:
            return new
        if not new:
            return current
        return [current[0]] + current[1:] + new[1:]

    def finalize(self, acc: List) -> List:
        if not acc:
            return []
        node = acc[0]
        parents = set(acc[1:])
        neighbors = parents | {node}
        mp = min(neighbors)
        large_neighbors = [v for v in neighbors if v > node]
        return [(v, mp) for v in large_neighbors]


class SmallStarAgg(AggregateFnV2[List, List]):
    """Small-star aggregation for connected components.

    For each node group, find small_neighbors = {v in parents | v <= node} union {node},
    then mp = min(small_neighbors), emit (v, mp) for all v in small_neighbors.

    Accumulator: flat list [node_val, parent1, parent2, ...] (Arrow-serializable)
    Output: list of (node, parent) tuples
    """

    def __init__(self):
        super().__init__(
            name="small_star(node)",
            on="parent",
            ignore_nulls=False,
            zero_factory=lambda: [],
        )

    def aggregate_block(self, block: Block) -> List:
        accessor = BlockAccessor.for_block(block)
        table = accessor.to_arrow()
        nodes = table.column('node').to_pylist()
        parents = table.column('parent').to_pylist()
        node_val = nodes[0] if nodes else None
        return [node_val] + parents

    def combine(self, current: List, new: List) -> List:
        if not current:
            return new
        if not new:
            return current
        return [current[0]] + current[1:] + new[1:]

    def finalize(self, acc: List) -> List:
        if not acc:
            return []
        node = acc[0]
        parents = set(acc[1:])
        small_neighbors = {v for v in parents if v <= node} | {node}
        mp = min(small_neighbors)
        return [(v, mp) for v in small_neighbors]


# ---------------------------------------------------------------------------
# Explode helpers for flat_map after aggregate
# ---------------------------------------------------------------------------

def explode_edge_pairs(row):
    pairs = row['edges_from_collisions']
    if not pairs:
        return []
    return [{'src': p[0], 'dst': p[1]} for p in pairs]


def explode_star_pairs(row, agg_col):
    pairs = row[agg_col]
    if not pairs:
        return []
    return [{'node': p[0], 'parent': p[1]} for p in pairs]


# ---------------------------------------------------------------------------
# GPUHashAggregate-friendly helpers
# ---------------------------------------------------------------------------

def gpu_hash_partitions(parallelism: int) -> int:
    return max(1, int(parallelism / 10))


def gpu_fused_cc_partitions(parallelism: int) -> int:
    """Size iterative CC partitions independently of the GPU actor count."""
    return gpu_hash_partitions(parallelism)


def project_collision_edges_gpu(batch: cudf.DataFrame) -> cudf.DataFrame:
    """Project joined band rows into min-doc star edges on GPU."""
    batch = batch[batch["band_group_size"] > 1]
    return cudf.DataFrame({
        "src": batch["min_doc_id"],
        "dst": batch["doc_id"],
    })


def empty_edge_batch_like(batch: cudf.DataFrame) -> cudf.DataFrame:
    return batch[["node", "parent"]].head(0)


def local_large_star_reduce_gpu(batch: cudf.DataFrame) -> cudf.DataFrame:
    """Run large-star reduction inside one keyed, sorted GPU shuffle block."""
    if len(batch) == 0:
        return empty_edge_batch_like(batch)

    nodes = batch[["node"]].drop_duplicates().reset_index(drop=True)
    identities = cudf.DataFrame({
        "node": nodes["node"],
        "parent": nodes["node"],
    })
    neighborhood = cudf.concat(
        [batch[["node", "parent"]], identities],
        ignore_index=True,
    ).drop_duplicates()

    min_parent = (
        neighborhood
        .groupby("node", as_index=False)
        .agg({"parent": "min"})
        .rename(columns={"parent": "min_parent"})
    )
    reduced = neighborhood.merge(min_parent, on="node", how="inner")
    reduced = reduced[reduced["parent"] > reduced["node"]]
    if len(reduced) == 0:
        return empty_edge_batch_like(batch)

    return cudf.DataFrame({
        "node": reduced["parent"],
        "parent": reduced["min_parent"],
    }).drop_duplicates().reset_index(drop=True)


def local_small_star_reduce_gpu(batch: cudf.DataFrame) -> cudf.DataFrame:
    """Run small-star reduction inside one keyed, sorted GPU shuffle block."""
    if len(batch) == 0:
        return empty_edge_batch_like(batch)

    nodes = batch[["node"]].drop_duplicates().reset_index(drop=True)
    identities = cudf.DataFrame({
        "node": nodes["node"],
        "parent": nodes["node"],
    })
    small_neighbors = batch[batch["parent"] <= batch["node"]][["node", "parent"]]
    neighborhood = cudf.concat(
        [small_neighbors, identities],
        ignore_index=True,
    ).drop_duplicates()

    min_parent = (
        neighborhood
        .groupby("node", as_index=False)
        .agg({"parent": "min"})
        .rename(columns={"parent": "min_parent"})
    )
    reduced = neighborhood.merge(min_parent, on="node", how="inner")
    if len(reduced) == 0:
        return empty_edge_batch_like(batch)

    return cudf.DataFrame({
        "node": reduced["parent"],
        "parent": reduced["min_parent"],
    }).drop_duplicates().reset_index(drop=True)


def map_gpu_blocks(
    ds: ray.data.Dataset,
    fn,
) -> ray.data.Dataset:
    """Map whole GPU shuffle output blocks.

    Public map_batches requires an explicit batch_size with num_gpus, but grouped
    star reductions need full key-partition blocks so keyed groups are not split.
    """
    return ds.map_batches_internal(
        fn,
        batch_size=None,
        compute=None,
        batch_format="cudf",
        zero_copy_batch=True,
        fn_args=None,
        fn_kwargs=None,
        fn_constructor_args=None,
        fn_constructor_kwargs=None,
        num_cpus=None,
        num_gpus=1,
        memory=None,
        concurrency=None,
        udf_modifying_row_count=True,
        ray_remote_args_fn=None,
    )


def gpu_distinct_2col(
    current_ds: ray.data.Dataset,
    col_1: str,
    col_2: str,
    parallelism: int,
) -> ray.data.Dataset:
    """Deduplicate two-column pairs using GPUHashAggregate Count()."""
    num_partitions = gpu_hash_partitions(parallelism)
    return (
        current_ds
        .select_columns([col_1, col_2])
        .groupby([col_1, col_2], num_partitions=num_partitions)
        .aggregate(Count())
        .select_columns([col_1, col_2])
        .materialize()
    )


def generate_edges_from_bands_gpu(
    bands_ds: ray.data.Dataset,
    hash_parallelism: int,
) -> ray.data.Dataset:
    """Create LSH candidate edges using GPUHashAggregate Min()+Count()."""
    num_partitions = gpu_hash_partitions(hash_parallelism)
    key_columns = ["band_id", "band_hash"]
    bands_ds = bands_ds.select_columns(key_columns + ["doc_id"])

    unique_bands_ds = (
        bands_ds
        .groupby(key_columns + ["doc_id"], num_partitions=num_partitions)
        .aggregate(Count(alias_name="band_doc_count"))
        .select_columns(key_columns + ["doc_id"])
        .materialize()
    )

    band_stats = (
        unique_bands_ds
        .groupby(key_columns, num_partitions=num_partitions)
        .aggregate(
            Min(on="doc_id", alias_name="min_doc_id"),
            Count(alias_name="band_group_size"),
        )
    )

    return (
        unique_bands_ds
        .join(
            band_stats,
            join_type="inner",
            num_partitions=num_partitions,
            on=tuple(key_columns),
        )
        .map_batches(
            project_collision_edges_gpu,
            batch_format="cudf",
            num_gpus=1,
            batch_size=100_000,
        )
        .materialize()
    )


def large_star_gpu_fused_blocks(
    current_ds: ray.data.Dataset,
    parallelism: int,
) -> ray.data.Dataset:
    num_partitions = gpu_fused_cc_partitions(parallelism)
    logger.info("large_star_gpu_fused_blocks partitions: %s", num_partitions)
    emitted = (
        current_ds
        .map_batches(
            large_star_emit_gpu,
            batch_format="cudf",
            num_gpus=1,
            batch_size=100_000,
        )
        .materialize()
    )

    # Keep GPU stages out of a single streaming pipeline. GPUShuffle reserves
    # gpu_shuffle_num_actors GPUs, so pipelining it with a 1-GPU map can request
    # num_gpus + 1 GPUs and stall under Ray Data's resource budget.
    shuffled = emitted.repartition(
        num_blocks=num_partitions,
        keys=["node"],
        sort=True,
    ).materialize()
    # Give the terminated GPU shuffle actors time to release their RMM pools
    # before scheduling downstream GPU tasks on the same devices.
    time.sleep(2)
    return map_gpu_blocks(shuffled, local_large_star_reduce_gpu).materialize()


def small_star_gpu_fused_blocks(
    current_ds: ray.data.Dataset,
    parallelism: int,
) -> ray.data.Dataset:
    num_partitions = gpu_fused_cc_partitions(parallelism)
    logger.info("small_star_gpu_fused_blocks partitions: %s", num_partitions)
    emitted = (
        current_ds
        .map_batches(
            small_star_emit_gpu,
            batch_format="cudf",
            num_gpus=1,
            batch_size=100_000,
        )
        .materialize()
    )

    shuffled = emitted.repartition(
        num_blocks=num_partitions,
        keys=["node"],
        sort=True,
    ).materialize()
    # Give the terminated GPU shuffle actors time to release their RMM pools
    # before scheduling downstream GPU tasks on the same devices.
    time.sleep(2)
    return map_gpu_blocks(shuffled, local_small_star_reduce_gpu).materialize()


# ---------------------------------------------------------------------------
# Original helper functions (unchanged)
# ---------------------------------------------------------------------------

def sha1_hash32(data: bytes) -> int:
    """Generate 32-bit hash from SHA1."""
    return struct.unpack('<I', hashlib.sha1(data).digest()[:4])[0]


def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
) -> Tuple[int, int]:
    """
    Compute optimal LSH parameters (bands, rows) that minimize weighted sum
    of false positive and false negative probabilities.

    Returns:
        (num_bands, rows_per_band)
    """
    def false_positive_probability(th: float, band: int, rows: int) -> float:
        def proba(s: float) -> float:
            return 1 - (1 - s ** float(rows)) ** float(band)
        a, _ = integrate.quad(proba, 0.0, th)
        return a

    def false_negative_probability(th: float, band: int, rows: int) -> float:
        def proba(s: float) -> float:
            return 1 - (1 - (1 - s ** float(rows)) ** float(band))
        a, _ = integrate.quad(proba, th, 1.0)
        return a

    min_error = float('inf')
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_probability(threshold, b, r)
            fn = false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


class MinHashGenerator:
    """Generates MinHash signatures from text using n-grams."""

    def __init__(
        self,
        num_perm: int = 128,
        ngram_size: int = 5,
        seed: int = 42,
        lowercase: bool = True,
    ):
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.lowercase = lowercase

        # Generate permutations for MinHash
        gen = np.random.RandomState(seed=seed)
        self.perm_a, self.perm_b = np.array(
            [
                (
                    gen.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                    gen.randint(0, MERSENNE_PRIME, dtype=np.uint64),
                )
                for _ in range(num_perm)
            ],
            dtype=np.uint64,
        ).T

    def _ngrams(self, text: str) -> Set[bytes]:
        """Generate character n-grams from text."""
        if self.lowercase:
            text = text.lower()
        return {
            text[i:i + self.ngram_size].encode('utf-8')
            for i in range(len(text) - self.ngram_size + 1)
        }

    def compute_minhash(self, text: str) -> np.ndarray:
        """
        Compute MinHash signature for a single text.

        Returns:
            Array of shape (num_perm,) with uint32 values
        """
        tokens = self._ngrams(text)

        if len(tokens) == 0:
            # Empty text gets max hash values
            return np.full(self.num_perm, MAX_HASH, dtype=np.uint32)

        # Hash all tokens
        hashes = np.array([sha1_hash32(token) for token in tokens], dtype=np.uint64)

        # Apply permutations: (h * a + b) % c
        # Broadcasting: hashes[:, None] has shape (num_tokens, 1)
        # perm_a[None, :] has shape (1, num_perm)
        phv = ((hashes[:, None] * self.perm_a[None, :] + self.perm_b) % MERSENNE_PRIME).astype(np.uint32)

        # Take minimum across all tokens for each permutation
        return phv.min(axis=0)


def generate_minhash_signatures_gpu(
    batch: cudf.DataFrame,
    text_column: str,
    num_perm: int,
    ngram_size: int,
    seed: int,
) -> cudf.DataFrame:
    """Generate MinHash signatures for a batch of documents using GPU."""
    generator = GPUMinHash(seed=seed, num_hashes=num_perm, char_ngrams=ngram_size)
    texts = batch[text_column]
    signatures = generator.compute_minhashes(texts.str.lower())
    return cudf.DataFrame({"id": batch["id"], "minhash": signatures})


def generate_minhash_signatures(
    batch: Dict[str, np.ndarray],
    text_column: str,
    num_perm: int,
    ngram_size: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Ray Data UDF to generate MinHash signatures for a batch of documents.

    This function is called by map_batches and processes documents in parallel.
    """
    generator = MinHashGenerator(num_perm=num_perm, ngram_size=ngram_size, seed=seed)

    texts = batch[text_column]
    signatures = np.array([generator.compute_minhash(text) for text in texts])

    # Drop the source columns (especially the large text column) once their
    # signatures have been computed.  Step 2 only consumes these two columns,
    # and retaining the full input rows causes Step 1's materialized output to
    # exhaust the Ray object-store budget.
    return {"id": batch["id"], "minhash": signatures}


def generate_lsh_bands(
    batch: Dict[str, np.ndarray],
    num_bands: int,
    rows_per_band: int,
) -> Dict[str, np.ndarray]:
    """
    Generate LSH bands from MinHash signatures.

    This creates multiple (band_id, band_hash) pairs per document,
    which will be used to find candidate duplicate pairs.

    Returns a flattened batch where each row represents one band of one document.

    Example:
    {
        'minhash': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        'id': np.array([1, 2, 3]),
    }
    ->
    {
        'doc_id': np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
        'band_id': np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]),
        'band_hash': np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']),
    }
    """
    minhashes = batch['minhash']
    num_docs = len(minhashes)

    # For each document, generate num_bands rows
    output_size = num_docs * num_bands

    # Replicate document IDs for each band
    doc_ids = np.repeat(batch['id'], num_bands)

    # np.tile repeats the entire array a given number of times.
    # [0, 1, ..., num_bands-1] * num_docs times
    band_ids = np.tile(np.arange(num_bands), num_docs)
    band_hashes = []

    for doc_idx, minhash in enumerate(minhashes):
        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            end = start + rows_per_band
            band_values = minhash[start:end]
            # Create a hash of the band
            band_hash = hashlib.sha256(band_values.tobytes()).hexdigest()[:16]
            band_hashes.append(band_hash)

    return {
        'doc_id': doc_ids,
        'band_id': band_ids,
        'band_hash': np.array(band_hashes),
    }


def large_star_emit_gpu(batch: cudf.DataFrame) -> cudf.DataFrame:
    """Vectorized implementation of large_star_emit using GPU."""
    # non-identity pairs
    non_identity_pairs = batch[batch['node'] != batch['parent']]

    # reverse the non-identity pairs
    reversed_non_identity_pairs = cudf.DataFrame({
        'node': non_identity_pairs['parent'],
        'parent': non_identity_pairs['node'],
    })

    # concatenate the identity pairs and the reversed non-identity pairs
    result = cudf.concat([batch, reversed_non_identity_pairs])
    return result


def large_star_emit(row):
    u, v = row["node"], row["parent"]
    if u == v:
        return [{"node": u, "parent": v}]
    return [{"node": u, "parent": v}, {"node": v, "parent": u}]


def small_star_emit_gpu(batch: cudf.DataFrame) -> cudf.DataFrame:
    """Vectorized implementation of small_star_emit using GPU."""
    node_greater = batch[batch['node'] >= batch['parent']]
    node_lesser = batch[batch['node'] < batch['parent']]
    node_lesser_reversed = cudf.DataFrame({
        'node': node_lesser['parent'],
        'parent': node_lesser['node'],
    })
    result = cudf.concat([node_greater, node_lesser_reversed])
    return result


def small_star_emit(row):
    """Emit (u, v) if u >= v else (v, u) -> (node, parent)"""
    u, v = row["node"], row["parent"]
    if u >= v:
        return [{"node": u, "parent": v}]
    else:
        return [{"node": v, "parent": u}]


def cast_node_parent_large_string(batch: pa.Table) -> pa.Table:
    """Cast node and parent to large string."""
    new_schema = pa.schema([
        pa.field('node', pa.large_string()),
        pa.field('parent', pa.large_string()),
    ])
    return batch.select(['node', 'parent']).cast(new_schema)


def anti_filter_with_cpu_join(
    ds: ray.data.Dataset,
    duplicate_components: ray.data.Dataset,
    id_column: str,
    duplicate_key_column: str = "node",
    num_partitions: int = 1_000,
) -> ray.data.Dataset:
    """Drop duplicate documents with a distributed CPU left anti-join.

    The duplicate keys remain distributed instead of being collected into a
    driver-side Python set and replicated to every filter worker. The join does
    shuffle the document rows, but bounds per-worker state by hash partitioning
    both inputs.
    """
    ds.context.shuffle_strategy = ShuffleStrategy.HASH_SHUFFLE
    duplicate_keys = duplicate_components.select_columns([duplicate_key_column])
    return ds.join(
        duplicate_keys,
        join_type="left_anti",
        num_partitions=num_partitions,
        on=(id_column,),
        right_on=(duplicate_key_column,),
    )


_UUID_KEY_DTYPE = np.dtype([("hi", "<u8"), ("lo", "<u8")])
_UUID_LENGTH = 47
_UUID_HEX_POSITIONS = np.array(
    [
        *range(10, 18),
        *range(19, 23),
        *range(24, 28),
        *range(29, 33),
        *range(34, 46),
    ],
    dtype=np.intp,
)
_UUID_HEX_LOOKUP = np.full(256, -1, dtype=np.int16)
_UUID_HEX_LOOKUP[ord("0") : ord("9") + 1] = np.arange(10)
_UUID_HEX_LOOKUP[ord("a") : ord("f") + 1] = np.arange(10, 16)
_UUID_HEX_LOOKUP[ord("A") : ord("F") + 1] = np.arange(10, 16)


def _uuid_halves(values: pa.ChunkedArray | pa.Array) -> Tuple[np.ndarray, np.ndarray]:
    """Decode canonical ``<urn:uuid:...>`` strings without Python objects."""
    if isinstance(values, pa.ChunkedArray):
        decoded = [_uuid_halves(chunk) for chunk in values.chunks]
        if not decoded:
            empty = np.empty(0, dtype=np.uint64)
            return empty, empty.copy()
        return (
            np.concatenate([item[0] for item in decoded]),
            np.concatenate([item[1] for item in decoded]),
        )

    if not (pa.types.is_string(values.type) or pa.types.is_large_string(values.type)):
        raise ValueError(f"UUID column must be a string column, got {values.type}")
    if values.null_count:
        raise ValueError("UUID column contains null values")
    if len(values) == 0:
        empty = np.empty(0, dtype=np.uint64)
        return empty, empty.copy()

    offsets_buffer, data_buffer = values.buffers()[1:]
    offset_dtype = np.int64 if pa.types.is_large_string(values.type) else np.int32
    all_offsets = np.frombuffer(offsets_buffer, dtype=offset_dtype)
    starts = all_offsets[values.offset : values.offset + len(values)]
    ends = all_offsets[values.offset + 1 : values.offset + len(values) + 1]
    if np.any(ends - starts != _UUID_LENGTH):
        raise ValueError("UUID values must use canonical <urn:uuid:...> formatting")

    # Canonical UUID values all have the same width, so their Arrow data is a
    # contiguous byte matrix. This avoids allocating an N x 32 matrix of offsets.
    raw_bytes = np.frombuffer(data_buffer, dtype=np.uint8)
    raw = raw_bytes[starts[0] : ends[-1]].reshape(len(values), _UUID_LENGTH)
    if not (
        np.all(raw[:, :10] == np.frombuffer(b"<urn:uuid:", dtype=np.uint8))
        and np.all(raw[:, 46] == ord(">"))
        and np.all(raw[:, [18, 23, 28, 33]] == ord("-"))
    ):
        raise ValueError("UUID values must use canonical <urn:uuid:...> formatting")

    nibbles = _UUID_HEX_LOOKUP[raw[:, _UUID_HEX_POSITIONS]]
    if np.any(nibbles < 0):
        raise ValueError("UUID values contain non-hexadecimal characters")
    nibbles = nibbles.astype(np.uint64, copy=False)
    hi = np.zeros(len(values), dtype=np.uint64)
    lo = np.zeros(len(values), dtype=np.uint64)
    for index in range(16):
        hi = (hi << 4) | nibbles[:, index]
        lo = (lo << 4) | nibbles[:, index + 16]
    return hi, lo


def _encode_uuid_keys(
    batch: pa.Table,
    duplicate_key_column: str,
) -> pa.Table:
    """Encode UUID strings as two fixed-width columns for distributed sorting."""
    hi, lo = _uuid_halves(batch.column(duplicate_key_column))
    return pa.table({"hi": hi, "lo": lo})


def _exact_uuid_membership(
    index_hi: np.ndarray,
    index_lo: np.ndarray,
    query_hi: np.ndarray,
    query_lo: np.ndarray,
) -> np.ndarray:
    """Test membership in a lexicographically sorted exact UUID index."""
    left = np.searchsorted(index_hi, query_hi, side="left")
    right = np.searchsorted(index_hi, query_hi, side="right")
    matched = np.zeros(len(query_hi), dtype=bool)
    present = left < right
    single = present & (right == left + 1)
    matched[single] = index_lo[left[single]] == query_lo[single]

    # Collisions in the high 64 bits are exceptionally rare for UUIDs, but
    # handle them exactly rather than relying on a probabilistic fingerprint.
    for row_index in np.flatnonzero(present & ~single):
        matched[row_index] = np.any(
            index_lo[left[row_index] : right[row_index]] == query_lo[row_index]
        )
    return matched


def _uuid_index_ref_metadata(table: pa.Table) -> List[Tuple[int, int, int, int, int]]:
    """Return row count and inclusive key range for each zero-copy batch."""
    metadata = []
    for batch in table.to_batches():
        if batch.num_rows == 0:
            continue
        hi = batch.column("hi").to_numpy(zero_copy_only=True)
        lo = batch.column("lo").to_numpy(zero_copy_only=True)
        metadata.append(
            (len(hi), int(hi[0]), int(lo[0]), int(hi[-1]), int(lo[-1]))
        )
    return metadata


def build_object_store_uuid_index(
    duplicate_components: ray.data.Dataset,
    duplicate_key_column: str = "node",
    num_shards: int = 32,
) -> Tuple[List[Any], np.ndarray, int]:
    """Build a range-sharded exact UUID index and retain its Plasma references.

    The returned boundaries are the inclusive maximum key for every Arrow
    record batch. Filter actors use them to route each lookup to exactly one
    zero-copy view of an immutable object-store buffer.
    """
    if num_shards < 1:
        raise ValueError("Object-store UUID index must have at least one shard")

    logger.info(
        "Building range-sorted object-store UUID index with %s shards",
        num_shards,
    )
    start = time.time()
    context = duplicate_components.context
    previous_shuffle_strategy = context.shuffle_strategy
    previous_use_polars_sort = context.use_polars_sort
    try:
        # Dataset.sort is a CPU range shuffle. Explicitly select its supported
        # shuffle implementation because the main pipeline may use GPU_SHUFFLE
        # for hash aggregates, and keep this path independent of Polars.
        context.shuffle_strategy = ShuffleStrategy.SORT_SHUFFLE_PULL_BASED
        context.use_polars_sort = False
        sorted_keys = (
            duplicate_components.select_columns([duplicate_key_column])
            .map_batches(
                _encode_uuid_keys,
                fn_kwargs={"duplicate_key_column": duplicate_key_column},
                batch_format="pyarrow",
            )
            .repartition(num_shards)
            .sort(["hi", "lo"])
            .materialize()
        )
    finally:
        context.shuffle_strategy = previous_shuffle_strategy
        context.use_polars_sort = previous_use_polars_sort

    shard_refs = sorted_keys.to_arrow_refs()
    metadata_refs = [ray.remote(_uuid_index_ref_metadata).remote(ref) for ref in shard_refs]
    metadata_by_ref = ray.get(metadata_refs)
    batch_metadata = [item for items in metadata_by_ref for item in items]
    key_count = sum(item[0] for item in batch_metadata)
    boundaries = np.empty(len(batch_metadata), dtype=_UUID_KEY_DTYPE)
    for index, (_, _first_hi, _first_lo, last_hi, last_lo) in enumerate(
        batch_metadata
    ):
        boundaries[index] = (last_hi, last_lo)

    # Range-sort output must be globally monotonic. Validate both ends of each
    # record batch before trusting boundary routing in the hot filter path.
    previous_last = None
    for _, first_hi, first_lo, last_hi, last_lo in batch_metadata:
        first = (first_hi, first_lo)
        last = (last_hi, last_lo)
        if first > last or (previous_last is not None and previous_last > first):
            raise RuntimeError("Object-store UUID index is not globally sorted")
        previous_last = last

    logger.info(
        "Built object-store UUID index with %s keys, %s Plasma blocks, and %s "
        "zero-copy batches in %.2f seconds",
        key_count,
        len(shard_refs),
        len(batch_metadata),
        time.time() - start,
    )
    return shard_refs, boundaries, key_count


def build_sorted_uuid_index(
    components_uri: str,
    index_uri: str,
    duplicate_key_column: str = "node",
) -> None:
    """Build an exact, memory-mappable index for URN UUID duplicate keys."""
    if os.path.exists(index_uri):
        logger.info("Using existing sorted UUID index: %s", index_uri)
        return

    component_files = list_parquet_files(components_uri)
    if not component_files:
        raise ValueError(f"No component parquet files found at {components_uri}")

    logger.info(
        "Building sorted UUID index from %s component files: %s",
        len(component_files),
        index_uri,
    )
    start = time.time()
    key_count = sum(pq.ParquetFile(path).metadata.num_rows for path in component_files)
    os.makedirs(os.path.dirname(os.path.abspath(index_uri)), exist_ok=True)
    temporary_uri = f"{index_uri}.tmp-{os.getpid()}"
    keys = np.lib.format.open_memmap(
        temporary_uri,
        mode="w+",
        dtype=_UUID_KEY_DTYPE,
        shape=(key_count,),
    )
    offset = 0
    try:
        for path in component_files:
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(
                batch_size=1_000_000,
                columns=[duplicate_key_column],
            ):
                hi, lo = _uuid_halves(batch.column(0))
                next_offset = offset + len(hi)
                keys["hi"][offset:next_offset] = hi
                keys["lo"][offset:next_offset] = lo
                offset = next_offset
        if offset != key_count:
            raise RuntimeError(f"Expected {key_count} keys, decoded {offset}")
        keys.sort(order=["hi", "lo"])
        keys.flush()
        os.replace(temporary_uri, index_uri)
    finally:
        if os.path.exists(temporary_uri):
            os.unlink(temporary_uri)
    logger.info(
        "Built sorted UUID index with %s keys in %.2f seconds",
        key_count,
        time.time() - start,
    )


class SortedUuidAntiFilter:
    """Exact anti-filter backed by a shared, memory-mapped sorted UUID index."""

    def __init__(self, index_uri: str, id_column: str):
        self._id_column = id_column
        self._keys = np.load(index_uri, mmap_mode="r")
        if self._keys.dtype != _UUID_KEY_DTYPE or self._keys.ndim != 1:
            raise ValueError(f"Unexpected sorted UUID index format at {index_uri}")
        self._hi = self._keys["hi"]
        self._lo = self._keys["lo"]

    def __call__(self, batch: pa.Table) -> pa.Table:
        query_hi, query_lo = _uuid_halves(batch.column(self._id_column))
        matched = _exact_uuid_membership(
            self._hi, self._lo, query_hi, query_lo
        )
        return batch.filter(pa.array(~matched))


def anti_filter_with_sorted_uuid_index(
    ds: ray.data.Dataset,
    index_uri: str,
    id_column: str,
    concurrency: int,
) -> ray.data.Dataset:
    """Stream an exact UUID anti-filter without shuffling document payloads."""
    return ds.map_batches(
        SortedUuidAntiFilter,
        fn_constructor_kwargs={"index_uri": index_uri, "id_column": id_column},
        batch_format="pyarrow",
        compute=ActorPoolStrategy(size=concurrency),
        num_cpus=1,
        memory=2 * 1024**3,
    )


class ObjectStoreUuidAntiFilter:
    """Exact UUID anti-filter over immutable range-sorted Plasma blocks."""

    def __init__(
        self,
        shard_refs: List[Any],
        boundaries: np.ndarray,
        id_column: str,
    ):
        self._id_column = id_column
        if shard_refs and isinstance(shard_refs[0], ray.ObjectRef):
            tables = ray.get(shard_refs)
        else:
            # Preserve compatibility if a Ray version dereferences nested refs.
            tables = shard_refs

        self._hi_batches = []
        self._lo_batches = []
        for table in tables:
            for batch in table.to_batches():
                if batch.num_rows == 0:
                    continue
                self._hi_batches.append(
                    batch.column("hi").to_numpy(zero_copy_only=True)
                )
                self._lo_batches.append(
                    batch.column("lo").to_numpy(zero_copy_only=True)
                )

        self._boundaries = np.asarray(boundaries, dtype=_UUID_KEY_DTYPE)
        if len(self._hi_batches) != len(self._boundaries):
            raise RuntimeError(
                "Object-store UUID index metadata does not match its Arrow batches"
            )

    def __call__(self, batch: pa.Table) -> pa.Table:
        query_hi, query_lo = _uuid_halves(batch.column(self._id_column))
        query_keys = np.empty(len(query_hi), dtype=_UUID_KEY_DTYPE)
        query_keys["hi"] = query_hi
        query_keys["lo"] = query_lo
        batch_indices = np.searchsorted(self._boundaries, query_keys, side="left")

        matched = np.zeros(len(query_hi), dtype=bool)
        valid = batch_indices < len(self._boundaries)
        for batch_index in np.unique(batch_indices[valid]):
            rows = np.flatnonzero(valid & (batch_indices == batch_index))
            matched[rows] = _exact_uuid_membership(
                self._hi_batches[batch_index],
                self._lo_batches[batch_index],
                query_hi[rows],
                query_lo[rows],
            )
        return batch.filter(pa.array(~matched))


def anti_filter_with_object_store_uuid_index(
    ds: ray.data.Dataset,
    shard_refs: List[Any],
    boundaries: np.ndarray,
    id_column: str,
    concurrency: int,
    num_gpus_per_actor: float = 0,
    batch_size: int = 100_000,
) -> ray.data.Dataset:
    """Stream exact filtering while sharing index buffers through Plasma."""
    return ds.map_batches(
        ObjectStoreUuidAntiFilter,
        fn_constructor_kwargs={
            "shard_refs": shard_refs,
            "boundaries": boundaries,
            "id_column": id_column,
        },
        batch_format="pyarrow",
        batch_size=batch_size,
        compute=ActorPoolStrategy(size=concurrency),
        num_cpus=1,
        num_gpus=num_gpus_per_actor,
        memory=2 * 1024**3,
    )


def distinct_2col(
    current_ds: ray.data.Dataset,
    col_1,
    col_2,
    parallelism: int = 100,
    num_gpus: int = 0,
) -> ray.data.Dataset:
    if num_gpus > 0:
        logger.info("distinct_2col using GPUHashAggregate Count()...")
        return gpu_distinct_2col(current_ds, col_1, col_2, parallelism)

    logger.info("distinct_2col using custom aggregate...")
    current_ds = current_ds.select_columns([col_1, col_2])
    current_ds = (
        current_ds
        .groupby(col_1, num_partitions=parallelism)
        .aggregate(DistinctValuesAgg(on=col_2))
    )
    agg_col = f"distinct({col_2})"
    current_ds = current_ds.flat_map(
        lambda row, _c1=col_1, _c2=col_2, _ac=agg_col: [
            {_c1: row[_c1], _c2: v} for v in row[_ac]
        ] if row[_ac] else []
    )
    current_ds = current_ds.materialize()
    return current_ds


def compute_connected_components_distributed(
    current_ds: ray.data.Dataset,
    max_iterations: int = 100,
    parallelism: int = 200,
    verbose=False,
    num_gpus: int = 0,
) -> ray.data.Dataset:
    """
    Compute connected components using distributed large-star/small-star algorithm.

    This iterative algorithm is suitable for large-scale graphs (billions of edges).
    Based on the paper: "Connected Components in MapReduce and Beyond"

    Algorithm:
    1. Large-star: For each edge (u,v), point the larger node to the smaller
        Emit (u, v) and (v, u) -> (node, parent)
        Group by node and take the minimum parent (mp)
        Emit (v, mp) where v > mp for v in neighborhood of node
    2. Small-star: Propagate parent pointers transitively
        Emit (u, v) if u >= v else (v, u) -> (node, parent)
        Group by node and take the minimum parent (mp)
        Emit (v, mp) for all v in neighborhood of node
    3. Repeat until convergence

    Returns:
        Dataset with columns: node, parent (where parent is the component root)
    """
    logger.info(
        "Computing connected components with distributed algorithm%s...",
        " (fused-blocks)" if num_gpus > 0 else "",
    )

    current_ds = current_ds.materialize()
    num_components = current_ds.count()
    logger.info("Initial count: %s", num_components)
    convergence_counter = 3
    for i in range(max_iterations):
        current_ds = current_ds.map_batches(cast_node_parent_large_string, batch_format="pyarrow")
        # Step 1: Large-star
        if num_gpus > 0:
            current_ds = large_star_gpu_fused_blocks(current_ds, parallelism)
            logger.info(
                "Length of large_star_gpu_fused_blocks: %s",
                current_ds.count(),
            )
        else:
            # CPU
            current_ds = current_ds.flat_map(large_star_emit, memory=8*2**30).materialize()
            logger.info("Length of large_star_emit: %s", current_ds.count())
            current_ds = (
                current_ds
                .groupby(['node'], num_partitions=parallelism)
                .aggregate(LargeStarAgg())
            )
            current_ds = current_ds.flat_map(
                lambda row: explode_star_pairs(row, 'large_star(node)')
            ).materialize()
            logger.info("Length of large_star_agg: %s", current_ds.count())
        if verbose:
            print(current_ds.to_pandas())

        # Step 2: Small-star
        if num_gpus > 0:
            current_ds = small_star_gpu_fused_blocks(current_ds, parallelism)
            logger.info(
                "Length of small_star_gpu_fused_blocks: %s",
                current_ds.count(),
            )
        else:
            # CPU
            current_ds = current_ds.flat_map(small_star_emit).materialize()
            logger.info("Length of small_star_emit: %s", current_ds.count())
            current_ds = (
                current_ds
                .groupby(['node'], num_partitions=parallelism)
                .aggregate(SmallStarAgg())
            )
            current_ds = current_ds.flat_map(
                lambda row: explode_star_pairs(row, 'small_star(node)')
            ).materialize()
            logger.info("Length of small_star_agg: %s", current_ds.count())

        if num_gpus > 0:
            new_num_components = (
                current_ds
                .groupby("parent", num_partitions=gpu_hash_partitions(parallelism))
                .aggregate(Count())
                .count()
            )
        else:
            new_num_components = current_ds.groupby("parent").count().count()
        logger.info("Number of new components: %s", new_num_components)

        if num_components == new_num_components:
            convergence_counter -= 1
        else:
            convergence_counter = 3
        if convergence_counter <= 0:
            break

        num_components = new_num_components

        print("-" * 10)
        print(f"Iteration {i}: {current_ds.count()}")
        print(f"number of components: {num_components}")
        print(f"convergence_counter: {convergence_counter}")
        print("-" * 10)

    if num_gpus > 0:
        current_ds = gpu_distinct_2col(current_ds, "node", "parent", parallelism)

    return current_ds


def get_or_create_minhash_bands(
        ds: ray.data.Dataset,
        minhash_checkpoint_uri: Optional[str],
        text_column: str,
        threshold: float,
        num_perm: int,
        ngram_size: int,
        seed: int,
        output_blocks: int = 100,
        num_gpus: int = 0,
        gpu_batch_size: int = 1000*100,
        num_gpus_per_task: float = 1.0) -> ray.data.Dataset:

    if minhash_checkpoint_uri is not None:
        if check_path_exists(minhash_checkpoint_uri):
            logger.info("Reading minhash bands from checkpoint: %s", minhash_checkpoint_uri)
            bands_ds = ray.data.read_parquet(minhash_checkpoint_uri)
            bands_ds = bands_ds.repartition(num_blocks=output_blocks)
            return bands_ds

    logger.info("Generating minhash bands")

    # Compute optimal LSH parameters
    num_bands, rows_per_band = optimal_param(threshold, num_perm)
    logger.info("LSH parameters: %s bands, %s rows per band", num_bands, rows_per_band)

    # Step 1: Generate MinHash signatures
    logger.info("Step 1: Generating MinHash signatures...")
    generate_minhash_start_time = time.time()
    if num_gpus > 0:
        ds_with_minhash = ds.map_batches(
            generate_minhash_signatures_gpu,
            fn_kwargs={
                'text_column': text_column,
                'num_perm': num_perm,
                'ngram_size': ngram_size,
                'seed': seed,
            },
            batch_format='cudf',
            batch_size=gpu_batch_size,
            num_gpus=num_gpus_per_task,
        )
    else:
        # Schema: dict_keys(['*', 'minhash'])
        ds_with_minhash = ds.map_batches(
            generate_minhash_signatures,
            fn_kwargs={
                'text_column': text_column,
                'num_perm': num_perm,
                'ngram_size': ngram_size,
                'seed': seed,
            },
        batch_format='numpy',
    )

    # Step 2: Generate LSH bands (creates multiple rows per document)
    logger.info("Step 2: Generating LSH bands...")
    # Schema: ['doc_id', 'band_id', 'band_hash'], non are unique
    bands_ds: ray.data.Dataset = ds_with_minhash.map_batches(
        generate_lsh_bands,
        fn_kwargs={
            'num_bands': num_bands,
            'rows_per_band': rows_per_band,
        },
        batch_format='numpy',
    )
    bands_ds = bands_ds.repartition(num_blocks=output_blocks)
    bands_ds = bands_ds.materialize()
    generate_lsh_bands_end_time = time.time()
    logger.info(
        "Steps 1-2 time: %s seconds",
        generate_lsh_bands_end_time - generate_minhash_start_time,
    )

    if minhash_checkpoint_uri is not None:
        bands_ds.write_parquet(minhash_checkpoint_uri)

    return bands_ds

def find_duplicate_components(
    bands_ds: ray.data.Dataset,
    max_cc_iterations: int = 100,
    validate_local: bool = False,
    hash_parallelism: int = 100,
    num_gpus: int = 0,
    edges_checkpoint_uri: Optional[str] = None,
    distinct_edges_checkpoint_uri: Optional[str] = None,
    parallelism_cc: Optional[int] = None,
) -> ray.data.Dataset:
    """
    Find duplicate components in a dataset of bands/hashes.

    Args:
        bands_ds: Input Ray dataset with bands/hashes
        max_cc_iterations: Maximum iterations for connected components
        validate_local: Whether to validate the local version of the algorithm
        hash_parallelism: Number of partitions for the hash step
        parallelism_cc: Parallelism for Step 6 connected components. Defaults to
            hash_parallelism when unset.
        edges_checkpoint_uri: Optional checkpoint for candidate edges before distinct
        distinct_edges_checkpoint_uri: Optional checkpoint for distinct edges

    Returns:
        Deduplicated dataset
    """
    if (
        distinct_edges_checkpoint_uri is not None
        and check_path_exists(distinct_edges_checkpoint_uri)
    ):
        logger.info(
            "Reading distinct edges from checkpoint: %s",
            distinct_edges_checkpoint_uri,
        )
        edges_ds = ray.data.read_parquet(distinct_edges_checkpoint_uri)
        distinct_edges_count = edges_ds.count()
        logger.info("Length of edges_ds after distinct: %s", distinct_edges_count)
    else:
        # Step 3: Group by band to find candidate pairs.
        create_edges_start_time = time.time()
        if (
            edges_checkpoint_uri is not None
            and check_path_exists(edges_checkpoint_uri)
        ):
            logger.info("Reading edges from checkpoint: %s", edges_checkpoint_uri)
            edges_ds = ray.data.read_parquet(edges_checkpoint_uri)
        else:
            logger.info("Generating edges")
            logger.info(
                "Step 3: Grouping by bands to find candidate pairs using aggregate..."
            )
            if num_gpus > 0:
                edges_ds = generate_edges_from_bands_gpu(bands_ds, hash_parallelism)
            else:
                edges_ds = (
                    bands_ds
                    .groupby(['band_id', 'band_hash'], num_partitions=hash_parallelism)
                    .aggregate(EdgesFromCollisionsAgg())
                )
                edges_ds = edges_ds.flat_map(explode_edge_pairs).materialize()
            if edges_checkpoint_uri:
                edges_ds.write_parquet(edges_checkpoint_uri)

        edges_count = edges_ds.count()
        create_edges_end_time = time.time()
        logger.info("Length of edges_ds: %s", edges_count)
        logger.info(
            "Step 3 time: %s seconds",
            create_edges_end_time - create_edges_start_time,
        )

        # Handle empty edges case (no collisions found)
        if edges_count == 0:
            logger.info("No candidate pairs found. No duplicates detected.")
            # Return empty dataset with expected schema
            empty_df = pd.DataFrame(
                {
                    "node": pd.Series(dtype=object),
                    "parent": pd.Series(dtype=object),
                }
            )
            return ray.data.from_pandas(empty_df)

        # Step 4: Deduplicate edges found in multiple bands.
        logger.info("Step 4: Deduplicating edges...")
        distinct_2col_start_time = time.time()
        edges_ds = distinct_2col(
            edges_ds,
            col_1='src',
            col_2='dst',
            parallelism=hash_parallelism,
            num_gpus=num_gpus,
        )
        distinct_edges_count = edges_ds.count()
        distinct_2col_end_time = time.time()
        logger.info("Length of edges_ds after distinct: %s", distinct_edges_count)
        logger.info(
            "Step 4 time: %s seconds",
            distinct_2col_end_time - distinct_2col_start_time,
        )
        if distinct_edges_checkpoint_uri is not None:
            logger.info(
                "Writing distinct edges checkpoint: %s",
                distinct_edges_checkpoint_uri,
            )
            edges_ds.write_parquet(distinct_edges_checkpoint_uri)

    if distinct_edges_count == 0:
        logger.info("No distinct candidate pairs found. No duplicates detected.")
        empty_df = pd.DataFrame(
            {
                "node": pd.Series(dtype=object),
                "parent": pd.Series(dtype=object),
            }
        )
        return ray.data.from_pandas(empty_df)

    # Step 6: Compute connected components (distributed algorithm)
    logger.info("Step 6: Computing connected components (distributed)...")
    connected_components_start_time = time.time()
    edges_ds = edges_ds.rename_columns({"src": "node", "dst": "parent"})
    if parallelism_cc is None:
        parallelism_cc = hash_parallelism
    components_ds = compute_connected_components_distributed(
        edges_ds,
        max_iterations=max_cc_iterations,
        parallelism=parallelism_cc,
        num_gpus=num_gpus,
    )
    logger.info("Length of components_ds: %s", components_ds.count())
    connected_components_end_time = time.time()
    logger.info("Step 6 time: %s seconds", connected_components_end_time - connected_components_start_time)

    # check local version
    if validate_local:
        compute_connected_components_pandas(edges_ds.to_pandas())

    # Step 7: Filter duplicates
    logger.info("Step 7: Filtering duplicates...")

    # Keep only documents where node != parent (extraneous components)
    filter_duplicates_start_time = time.time()
    if num_gpus > 0:
        def filter_gpu(batch: "cudf.DataFrame") -> "cudf.DataFrame":
            return batch[batch['node'] != batch['parent']]
        duplicate_components = components_ds.map_batches(
            filter_gpu,
            batch_format="cudf",
            num_gpus=1,
            batch_size=100_000
        ).materialize()
    else:
        duplicate_components = components_ds.filter(
            lambda row: row['node'] != row['parent']
        ).materialize()
    filter_duplicates_end_time = time.time()
    logger.info("Duplicated components count: %s", duplicate_components.count())
    logger.info("Step 7 time: %s seconds", filter_duplicates_end_time - filter_duplicates_start_time)
    return duplicate_components


def main():
    """CLI for large-scale deduplication (3TB+)."""
    parser = argparse.ArgumentParser(
        description='Large-scale deduplication with Ray Data (designed for 3TB+)'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input path (parquet files, can use wildcards or GCS paths)',
    )
    parser.add_argument(
        '--output',
        type=str,
        required=False,
        default=os.path.join(
            os.environ.get("ANYSCALE_ARTIFACT_STORAGE", "/raid/spark-team/leey/tmp"), "dedup-output"
        ),
        help='Output path for deduplicated data',
    )
    parser.add_argument(
        '--text-column',
        type=str,
        default='text',
        help='Name of text column',
    )
    parser.add_argument(
        '--id-column',
        type=str,
        default='id',
        help='Name of ID column (must be unique for each document)',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='Jaccard similarity threshold (0.0-1.0)',
    )
    parser.add_argument(
        '--num-perm',
        type=int,
        default=128,
        help='Number of MinHash permutations',
    )
    parser.add_argument(
        '--ngram-size',
        type=int,
        default=5,
        help='Character n-gram size',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )
    parser.add_argument(
        '--max-cc-iterations',
        type=int,
        default=100,
        help='Maximum iterations for connected components convergence',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of documents to process (for testing on subset)',
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--parallelism-cc",
        type=int,
        default=None,
        help=(
            "Parallelism for Step 6 connected components "
            "(default: use --parallelism)"
        ),
    )
    parser.add_argument(
        "--disable-progress-bars",
        action="store_true",
        default=False,
        help="Disable progress bars",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use in ray cluster",
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help=(
            "Address of an existing Ray cluster, for example 'auto'. If omitted, "
            "start a local Ray instance."
        ),
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=1000*100,
        help="Batch size for GPU processing (larger = better GPU utilization)",
    )
    parser.add_argument(
        "--num-gpus-per-task",
        type=float,
        default=1.0,
        help="Number of GPUs per task (default: 1.0). "
             "Set lower (e.g., 0.5) to improve GPU utilization through pipelining.",
    )
    parser.add_argument(
        "--minhash-checkpoint-uri",
        type=str,
        help="Checkpoint URI for minhash bands",
    )
    parser.add_argument(
        "--edges-checkpoint-uri",
        type=str,
        help="Checkpoint URI for candidate edges before deduplication",
    )
    parser.add_argument(
        "--distinct-edges-checkpoint-uri",
        type=str,
        help="Checkpoint URI for distinct edges after Step 4",
    )
    parser.add_argument(
        "--components-checkpoint-uri",
        type=str,
        help="Checkpoint URI for deduplicated connected components",
    )
    parser.add_argument(
        "--filter-parallelism",
        type=int,
        default=None,
        help=(
            "Number of CPU hash partitions for the Step 8 left anti-join "
            "(default: use --parallelism)"
        ),
    )
    parser.add_argument(
        "--filter-engine",
        choices=("cpu_join", "sorted_uuid", "object_store_uuid"),
        default="cpu_join",
        help=(
            "Step 8 implementation. 'sorted_uuid' streams input blocks through "
            "an exact memory-mapped index. 'object_store_uuid' builds the same "
            "exact index as range-sorted Plasma blocks shared by actors on each host. "
            "Both require canonical <urn:uuid:...> IDs."
        ),
    )
    parser.add_argument(
        "--filter-concurrency",
        type=int,
        default=32,
        help="Actor concurrency for UUID index filter engines (default: 32)",
    )
    parser.add_argument(
        "--filter-num-gpus-per-actor",
        type=float,
        default=None,
        help=(
            "GPU scheduling resources reserved per object-store UUID filter actor. "
            "By default, divide --num-gpus by --filter-concurrency; with 16 GPUs "
            "and 32 actors this places two actors on each single-GPU host. Set to "
            "0 to disable GPU-based placement."
        ),
    )
    parser.add_argument(
        "--filter-batch-size",
        type=int,
        default=100_000,
        help="Rows per object-store UUID filter batch (default: 100000)",
    )
    parser.add_argument(
        "--object-store-index-shards",
        type=int,
        default=32,
        help=(
            "Number of range-sorted Plasma blocks for object_store_uuid "
            "(default: 32)"
        ),
    )
    parser.add_argument(
        "--duplicate-key-index-uri",
        type=str,
        default=None,
        help=(
            "Sorted UUID index path. Defaults to a file alongside the components "
            "checkpoint and is built once when absent."
        ),
    )

    args = parser.parse_args()

    if args.ray_address is not None:
        ray.init(address=args.ray_address)
    else:
        ray.init(
            num_gpus=args.num_gpus,
            _temp_dir=os.environ.get("RAY_TMP_DIR", "/tmp/ray"),
            object_store_memory=400 * 1024**3,
        )
    # ray.init()
    ctx = ray.data.context.DataContext.get_current()
    # Leave half of Plasma available for shuffle intermediates instead of
    # allowing queued streaming-executor blocks to consume the full store.
    ctx.override_object_store_memory_limit_fraction = 0.5
    # ctx.max_hash_shuffle_aggregators = 48

    if args.disable_progress_bars:
        ray.data.DataContext.get_current().enable_progress_bars = False
    if args.num_gpus > 0:
        ctx = ray.data.context.DataContext.get_current()
        ctx.shuffle_strategy = ShuffleStrategy.GPU_SHUFFLE
        ctx.gpu_shuffle_num_actors = args.num_gpus
        ctx.gpu_join_left_chunk_rows = 100_000
        #ctx.gpu_shuffle_rmm_pool_size = None
        #ctx.gpu_shuffle_spill_memory_limit = 48 * 1024**3  # 32? 28?
        ctx.gpu_shuffle_rmm_pool_size = "auto"
        ctx.gpu_shuffle_spill_memory_limit = "auto"

    # Read input data
    logger.info("Reading data from %s", args.input)
    input_path = args.input

    # List all parquet files in the directory (supports both GCS and local paths)
    list_of_all_input_files = list_parquet_files(input_path)
    logger.info("Reading %s parquet files", len(list_of_all_input_files))

    ds = ray.data.read_parquet(list_of_all_input_files)
    input_count = ds.count()
    logger.info("Original input count: %s", input_count)
    if args.limit is not None:
        logger.info(f"Limiting input to {args.limit} documents")
        assert input_count >= args.limit
        # Materialization is needed here to make a limited input deterministic when
        # it is consumed once for MinHash generation and again for final filtering.
        ds = ds.limit(args.limit).materialize()
        input_count = args.limit

    logger.info("Starting large-scale deduplication with threshold=%s", args.threshold)

    bands_ds = get_or_create_minhash_bands(
        ds,
        text_column=args.text_column,
        threshold=args.threshold,
        num_perm=args.num_perm,
        ngram_size=args.ngram_size,
        seed=args.seed,
        minhash_checkpoint_uri=args.minhash_checkpoint_uri,
        output_blocks=args.parallelism,
        num_gpus=args.num_gpus,
        gpu_batch_size=args.gpu_batch_size,
        num_gpus_per_task=args.num_gpus_per_task
    )

    # Duplicate components: Schema: ['node', 'parent']
    if args.components_checkpoint_uri is not None and check_path_exists(args.components_checkpoint_uri):
        logger.info("Reading components from checkpoint: %s", args.components_checkpoint_uri)
        duplicate_components = ray.data.read_parquet(args.components_checkpoint_uri)
    else:
        duplicate_components = find_duplicate_components(
            bands_ds,
            max_cc_iterations=args.max_cc_iterations,
            hash_parallelism=args.parallelism,
            parallelism_cc=args.parallelism_cc,
            num_gpus=args.num_gpus,
            edges_checkpoint_uri=args.edges_checkpoint_uri,
            distinct_edges_checkpoint_uri=args.distinct_edges_checkpoint_uri,
        )
        if args.components_checkpoint_uri is not None:
            duplicate_components.write_parquet(args.components_checkpoint_uri)
    duplicate_count = duplicate_components.count()

    # Filter the original dataset with a distributed CPU left anti-join.
    logger.info("Step 8: Filtering original dataset with duplicate keys...")
    join_start_time = time.time()

    if duplicate_count == 0:
        # No duplicates found, skip the filter.
        logger.info("No duplicates found, skipping filter.")
        deduplicated_ds = ds
    elif args.filter_engine == "sorted_uuid":
        if args.components_checkpoint_uri is None:
            raise ValueError(
                "--filter-engine=sorted_uuid requires --components-checkpoint-uri"
            )
        index_uri = args.duplicate_key_index_uri or os.path.join(
            args.components_checkpoint_uri, "duplicate_keys_uuid128.sorted.npy"
        )
        build_sorted_uuid_index(args.components_checkpoint_uri, index_uri)
        logger.info(
            "Filtering original dataset with exact sorted UUID index: "
            "duplicate_count=%s, concurrency=%s",
            duplicate_count,
            args.filter_concurrency,
        )
        deduplicated_ds = anti_filter_with_sorted_uuid_index(
            ds,
            index_uri=index_uri,
            id_column=args.id_column,
            concurrency=args.filter_concurrency,
        )
    elif args.filter_engine == "object_store_uuid":
        if args.filter_concurrency < 1:
            raise ValueError("--filter-concurrency must be at least 1")
        if args.filter_batch_size < 1:
            raise ValueError("--filter-batch-size must be at least 1")
        filter_num_gpus_per_actor = args.filter_num_gpus_per_actor
        if filter_num_gpus_per_actor is None:
            filter_num_gpus_per_actor = (
                args.num_gpus / args.filter_concurrency if args.num_gpus > 0 else 0
            )
        shard_refs, boundaries, index_count = build_object_store_uuid_index(
            duplicate_components,
            duplicate_key_column="node",
            num_shards=args.object_store_index_shards,
        )
        if index_count != duplicate_count:
            raise RuntimeError(
                f"Expected {duplicate_count} duplicate keys, indexed {index_count}"
            )
        logger.info(
            "Filtering original dataset with exact object-store UUID index: "
            "duplicate_count=%s, shards=%s, concurrency=%s, GPUs/actor=%s",
            duplicate_count,
            len(shard_refs),
            args.filter_concurrency,
            filter_num_gpus_per_actor,
        )
        deduplicated_ds = anti_filter_with_object_store_uuid_index(
            ds,
            shard_refs=shard_refs,
            boundaries=boundaries,
            id_column=args.id_column,
            concurrency=args.filter_concurrency,
            num_gpus_per_actor=filter_num_gpus_per_actor,
            batch_size=args.filter_batch_size,
        )
    else:
        logger.info(
            "Filtering original dataset with CPU left anti-join: "
            "duplicate_count=%s, num_partitions=%s",
            duplicate_count,
            args.filter_parallelism or args.parallelism,
        )
        deduplicated_ds = anti_filter_with_cpu_join(
            ds,
            duplicate_components,
            id_column=args.id_column,
            duplicate_key_column="node",
            num_partitions=args.filter_parallelism or args.parallelism,
        )
    # Keep filtering and writing in one streaming execution. Materializing the
    # full filtered dataset first retains hundreds of GiB in Plasma/spill.
    logger.info("Step 9: Writing deduplicated data to %s", args.output)
    deduplicated_ds.write_parquet(args.output)
    join_end_time = time.time()
    logger.info(
        "Steps 8-9 time (streaming filter and write): %s seconds",
        join_end_time - join_start_time,
    )

    # Parquet metadata makes this validation inexpensive and avoids rerunning
    # the filter pipeline merely to obtain a row count.
    output_count = ray.data.read_parquet(args.output).count()
    write_end_time = time.time()
    logger.info("Output validation time: %s seconds", write_end_time - join_end_time)

    logger.info("Input dataset: %s documents", input_count)
    logger.info("Output dataset: %s documents", output_count)
    logger.info("Removed %s duplicates (%s%%)", input_count - output_count, 100*(input_count - output_count)/input_count)


def compute_connected_components_pandas(edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute connected components for a local dataframe using scipy's DisjointSet.

    This is an efficient local algorithm suitable for graphs that fit in memory.

    Args:
        edges_df: DataFrame with columns ['node', 'parent'] representing edges

    Returns:
        DataFrame with columns ['node', 'parent'] where parent is the component root
    """
    from scipy.cluster.hierarchy import DisjointSet
    edges = edges_df[['node', 'parent']].values

    # Create DisjointSet with all unique nodes
    all_nodes = set()
    for node, parent_node in edges:
        all_nodes.add(node)
        all_nodes.add(parent_node)

    ds = DisjointSet(all_nodes)

    # Merge edges
    for node, parent_node in edges:
        ds.merge(node, parent_node)
    # Create result dataframe
    result = pd.DataFrame({
        'node': list(all_nodes),
        'parent': [ds[node] for node in all_nodes]
    })
    print("Number of subsets:", ds.n_subsets)
    return result


def test_connected(local: bool = False):
    # Generate random graph
    n = 10000
    # Generate n random edges between 0 and n-1 (uniformly)
    edges = np.random.randint(0, n, size=(n * 2, 2))
    edges = pd.DataFrame(edges, columns=["node", "parent"])

    print("running distributed")
    ray.data.DataContext.get_current().enable_progress_bars = False
    edges_ds = ray.data.from_pandas(edges)
    edges_ds = edges_ds.map_batches(lambda batch: batch, batch_format="pyarrow")
    result = compute_connected_components_distributed(
        edges_ds, max_iterations=10, parallelism=10)

    print("Running local")
    result = compute_connected_components_pandas(edges)

    print(result)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    main()
    # test_connected()
    print("finished")
