from __future__ import annotations

import pickle
import types
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import pandas as pd

from ray.data._internal.compute import ActorPoolStrategy

if TYPE_CHECKING:
    import cudf
    import cupy

    from ray.data.preprocessors.gpu.base import GPUPreprocessor

_DEFAULT_GPU_BATCH_SIZE = 4096
_DEFAULT_WORD_PATTERN = r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?"
_DEFAULT_TOKEN_PATTERN = r"[A-Za-z]+|[0-9]+|[^A-Za-z0-9\s]"
_COMBINED_FIT_INDEX_COLUMN = "__preprocessor_index"
_COMBINED_FIT_STATS_COLUMN = "__fit_stats"
_GPU_ORDINAL_FIT_NUM_PARTITIONS = 256


def _import_cudf():
    try:
        import cudf
    except ImportError as exc:
        raise ImportError(
            "GPU preprocessors require cuDF. Install a RAPIDS cuDF package "
            "compatible with your CUDA runtime."
        ) from exc
    return cudf


def _import_cupy():
    try:
        import cupy as cp
    except ImportError as exc:
        raise ImportError(
            "GPU preprocessors require CuPy. Install a CuPy package compatible "
            "with your CUDA runtime."
        ) from exc
    return cp


def _str_count(series: cudf.Series, pattern: str) -> cudf.Series:
    if hasattr(series.str, "count"):
        return series.str.count(pattern)
    if hasattr(series.str, "count_re"):
        return series.str.count_re(pattern)
    raise AttributeError("cuDF string columns do not expose count/count_re.")


def _list_lengths(list_series: cudf.Series) -> cudf.Series:
    lengths = getattr(list_series.list, "len", None)
    if lengths is None:
        raise AttributeError("cuDF list columns do not expose list.len.")
    return lengths() if callable(lengths) else lengths


def _list_leaves(list_series: cudf.Series) -> cudf.Series:
    leaves = getattr(list_series.list, "leaves", None)
    if leaves is None:
        raise AttributeError("cuDF list columns do not expose list.leaves.")
    return leaves() if callable(leaves) else leaves


def _hash_string_series(series: cudf.Series) -> cudf.Series:
    if hasattr(series, "hash_values"):
        try:
            return series.hash_values(method="murmur3")
        except TypeError:
            return series.hash_values()
    raise AttributeError("cuDF Series does not expose hash_values.")


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    try:
        return bool(missing)
    except (TypeError, ValueError):
        return False


def _cudf_dataframe_from_cupy(
    values: cupy.ndarray, columns: Sequence[str], index: Any
) -> cudf.DataFrame:
    cudf = _import_cudf()
    result = cudf.DataFrame(values, columns=list(columns))
    result.index = index
    return result


def _power_transform_values(
    values: cupy.ndarray,
    power: float,
    method: str,
    cp: types.ModuleType,
) -> cupy.ndarray:
    """Apply a Yeo-Johnson or Box-Cox transform to a CuPy array."""
    if method == "yeo-johnson":
        positive = values >= 0
        if power != 0:
            positive_values = (cp.power(values + 1, power) - 1) / power
        else:
            positive_values = cp.log(values + 1)

        if power != 2:
            negative_values = -(cp.power(-values + 1, 2 - power) - 1) / (2 - power)
        else:
            negative_values = -cp.log(-values + 1)

        return cp.where(positive, positive_values, negative_values)

    if power != 0:
        return (cp.power(values, power) - 1) / power
    return cp.log(values)


def _assign_dataframe_columns(
    df: cudf.DataFrame, columns: Sequence[str], values: Any
) -> cudf.DataFrame:
    df[list(columns)] = values
    return df


def _to_pandas_dict(series: cudf.Series) -> Dict[Any, Any]:
    if hasattr(series, "to_pandas"):
        return series.to_pandas().to_dict()
    return dict(series)


def _ordinal_map_from_stats(stats: Dict[str, Any], column: str) -> Dict[Any, int]:
    stat_value = stats[f"unique_values({column})"]
    if isinstance(stat_value, dict):
        return stat_value
    keys_array, values_array = stat_value
    return {key.as_py(): value.as_py() for key, value in zip(keys_array, values_array)}


def _deserialize_pandas_fit_stats(
    payload: Union[bytes, memoryview],
) -> pd.DataFrame:
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    return pickle.loads(payload)


def _gpu_actor_compute_strategy(
    concurrency: Optional[int],
) -> Dict[str, ActorPoolStrategy]:
    if concurrency is None:
        return {}
    return {
        "compute": ActorPoolStrategy(
            size=concurrency,
            max_tasks_in_flight_per_actor=2,
        )
    }


class _GPUTransformContext:
    """Shared state for a fused GPU transform batch.

    GPU preprocessors in a fused chain often reuse intermediate string and token
    representations. The context memoizes those intermediates for one batch and
    invalidates cache entries when a preprocessor modifies their source columns.
    """

    def __init__(self) -> None:
        self._cache: Dict[Tuple[Any, ...], Any] = {}
        self._dependencies: Dict[Tuple[Any, ...], set[str]] = {}

    def get_or_compute(
        self,
        key: Tuple[Any, ...],
        dependencies: Sequence[str],
        compute: Callable[[], Any],
    ) -> Any:
        if key not in self._cache:
            self._cache[key] = compute()
            self._dependencies[key] = set(dependencies)
        return self._cache[key]

    def invalidate_columns(self, columns: Sequence[str]) -> None:
        modified = set(columns)
        if not modified:
            return
        stale_keys = [
            key
            for key, dependencies in self._dependencies.items()
            if dependencies.intersection(modified)
        ]
        for key in stale_keys:
            self._cache.pop(key, None)
            self._dependencies.pop(key, None)

    def string_column(self, df: cudf.DataFrame, column: str) -> cudf.Series:
        return self.get_or_compute(
            ("string_column", column),
            (column,),
            lambda: df[column].fillna("").astype("str"),
        )

    def lowercase_string_column(self, df: cudf.DataFrame, column: str) -> cudf.Series:
        return self.get_or_compute(
            ("lowercase_string_column", column),
            (column,),
            lambda: self.string_column(df, column).str.lower(),
        )

    def tokenized_text(
        self, df: cudf.DataFrame, column: str, pattern: str
    ) -> Tuple[cudf.Series, cudf.Series]:
        def compute() -> Tuple[cudf.Series, cudf.Series]:
            text_lower = self.lowercase_string_column(df, column)
            tokens = text_lower.str.findall(pattern)
            lengths = _list_lengths(tokens).fillna(0).astype("int32")
            return tokens, lengths

        return self.get_or_compute(
            ("tokenized_text", column, pattern),
            (column,),
            compute,
        )


class _GPUPhysicalOp(Protocol):
    """Interface implemented by planned GPU transforms."""

    def _prepare_gpu_state(self) -> None:
        ...

    def _transform_cudf(
        self, df: cudf.DataFrame, context: Optional[_GPUTransformContext] = None
    ) -> cudf.DataFrame:
        ...

    def _gpu_modified_columns(self) -> List[str]:
        ...


def _append_hashing_columns_from_tokens(
    df: cudf.DataFrame,
    tokens: cudf.Series,
    lengths: cudf.Series,
    output_col: str,
    num_features: int,
) -> cudf.DataFrame:
    """Append dense hashing-vectorizer columns from tokenized cuDF strings."""
    cudf = _import_cudf()
    cp = _import_cupy()

    lengths = lengths.fillna(0).astype("int32")
    lengths_gpu = lengths.to_cupy()
    dense = cp.zeros((len(df), num_features), dtype=cp.int32)
    total_tokens = int(lengths.sum())
    if total_tokens:
        flat_tokens = _list_leaves(tokens)
        token_hashes = (
            _hash_string_series(flat_tokens).astype("uint64").to_cupy() % num_features
        )
        offsets = cp.cumsum(lengths_gpu)
        token_offsets = cp.arange(total_tokens, dtype=cp.int64)
        row_ids = cp.searchsorted(offsets, token_offsets, side="right").astype(cp.int32)
        cp.add.at(dense, (row_ids, token_hashes.astype(cp.int32)), 1)
    for idx in range(num_features):
        df[f"{output_col}_{idx}"] = cudf.Series(dense[:, idx], index=df.index)
    return df


def _apply_gpu_physical_ops(
    batch: cudf.DataFrame,
    ops: Sequence[_GPUPhysicalOp],
    *,
    prepare: bool = True,
) -> cudf.DataFrame:
    df = batch.copy(deep=False)
    context = _GPUTransformContext()
    for op in ops:
        if prepare:
            op._prepare_gpu_state()
        result = op._transform_cudf(df, context)
        if result is not None:
            df = result
        context.invalidate_columns(op._gpu_modified_columns())
    return df


def _apply_gpu_ops(
    batch: cudf.DataFrame,
    preprocessors: Sequence["GPUPreprocessor"],
    *,
    prepare: bool = True,
) -> cudf.DataFrame:
    return _apply_gpu_physical_ops(batch, preprocessors, prepare=prepare)


def _apply_gpu_transform_ops(
    batch: cudf.DataFrame,
    preprocessors: Sequence["GPUPreprocessor"],
    *,
    prepare: bool = True,
) -> cudf.DataFrame:
    from ray.data.preprocessors.gpu._fusion import _plan_gpu_transform_ops

    return _apply_gpu_physical_ops(
        batch,
        _plan_gpu_transform_ops(preprocessors),
        prepare=prepare,
    )
