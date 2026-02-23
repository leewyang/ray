from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    import cudf


def hash_partition_cudf(
    df: "cudf.DataFrame",
    hash_cols: List[str],
    num_partitions: int,
) -> Dict[int, "cudf.DataFrame"]:
    """Hash-partition a cuDF DataFrame using cuDF's native GPU hashing.

    Uses ``cudf.DataFrame.partition_by_hash`` which computes MurmurHash3 on the
    key columns entirely on-device, avoiding any CPU round-trips.

    Unlike the Arrow path (which does a Python-level ``hash(tuple(...))`` per
    row), this function delegates all hashing and splitting to the GPU.

    Args:
        df: The cuDF DataFrame to partition.
        hash_cols: Column names to hash on.
        num_partitions: Number of output partitions.

    Returns:
        Dict mapping partition index to a (possibly empty) cuDF DataFrame
        shard.  Empty partitions are omitted, matching the behaviour of the
        Arrow ``hash_partition`` helper.
    """
    assert num_partitions > 0

    if len(df) == 0:
        return {}
    if num_partitions == 1:
        return {0: df}

    partitions = df.partition_by_hash(columns=hash_cols, npartitions=num_partitions)
    return {i: part for i, part in enumerate(partitions) if len(part) > 0}


def _create_empty_cudf_table(reference_df: "cudf.DataFrame") -> "cudf.DataFrame":
    """Return an empty cuDF DataFrame with the same columns and dtypes as *reference_df*.

    Used by ``_shuffle_block`` to broadcast schema to aggregators that would
    otherwise receive no shards for a given partition.

    Args:
        reference_df: A cuDF DataFrame whose column names and dtypes are used
            to construct the empty result.

    Returns:
        An empty ``cudf.DataFrame`` with matching schema.
    """
    import cudf

    return cudf.DataFrame(
        {
            col: cudf.Series([], dtype=reference_df[col].dtype)
            for col in reference_df.columns
        }
    )
