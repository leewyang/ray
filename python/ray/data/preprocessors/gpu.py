from __future__ import annotations

import math
import pickle
import types
from collections import Counter
from numbers import Number
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
from ray.data.preprocessor import Preprocessor, SerializablePreprocessorBase
from ray.data.preprocessors.scaler import _EPSILON
from ray.data.preprocessors.utils import (
    _Computed,
    _PublicField,
    migrate_private_fields,
)
from ray.data.preprocessors.version_support import SerializablePreprocessor
from ray.util.annotations import DeveloperAPI, PublicAPI

if TYPE_CHECKING:
    import cudf
    import cupy

    from ray.data.dataset import Dataset


_DEFAULT_GPU_BATCH_SIZE = 4096
_DEFAULT_WORD_PATTERN = r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?"
_DEFAULT_TOKEN_PATTERN = r"[A-Za-z]+|[0-9]+|[^A-Za-z0-9\s]"
_COMBINED_FIT_INDEX_COLUMN = "__preprocessor_index"
_COMBINED_FIT_STATS_COLUMN = "__fit_stats"


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
    return _apply_gpu_physical_ops(
        batch,
        _plan_gpu_transform_ops(preprocessors),
        prepare=prepare,
    )


def _is_in_place_column_transform(preprocessor: "GPUPreprocessor") -> bool:
    return preprocessor.get_input_columns() == preprocessor.get_output_columns()


def _is_constant_numeric_imputer(preprocessor: "GPUPreprocessor") -> bool:
    if not isinstance(preprocessor, GPUSimpleImputer):
        return False
    return (
        preprocessor.strategy == "constant"
        and isinstance(preprocessor.fill_value, Number)
        and _is_in_place_column_transform(preprocessor)
    )


def _is_fusible_numeric_transform(preprocessor: "GPUPreprocessor") -> bool:
    if isinstance(preprocessor, (GPUPowerTransformer, GPUStandardScaler)):
        return _is_in_place_column_transform(preprocessor)
    return _is_constant_numeric_imputer(preprocessor)


class _FusedGPUNumericColumnOp:
    def __init__(self, preprocessors: Sequence["GPUPreprocessor"]):
        self._preprocessors = tuple(preprocessors)
        self._columns = self._preprocessors[0].get_input_columns()
        self._output_columns = self._preprocessors[-1].get_output_columns()

    def _prepare_gpu_state(self) -> None:
        for preprocessor in self._preprocessors:
            preprocessor._prepare_gpu_state()

    def _transform_cudf(
        self, df: cudf.DataFrame, context: Optional[_GPUTransformContext] = None
    ) -> cudf.DataFrame:
        """Apply a compatible sequence of numeric transforms in one CuPy array."""
        cp = _import_cupy()
        values = df[self._columns].astype("float64").to_cupy(na_value=cp.nan)

        for preprocessor in self._preprocessors:
            if isinstance(preprocessor, GPUPowerTransformer):
                values = _power_transform_values(
                    values, preprocessor.power, preprocessor.method, cp
                )
            elif isinstance(preprocessor, GPUStandardScaler):
                means = cp.asarray(
                    [
                        (
                            float(preprocessor.stats_.get(f"mean({column})"))
                            if preprocessor.stats_.get(f"mean({column})") is not None
                            else cp.nan
                        )
                        for column in preprocessor.columns
                    ],
                    dtype=cp.float64,
                )
                stds = cp.asarray(
                    [
                        (
                            float(preprocessor.stats_.get(f"std({column})"))
                            if preprocessor.stats_.get(f"std({column})") is not None
                            and preprocessor.stats_.get(f"std({column})") >= _EPSILON
                            else 1.0
                        )
                        for column in preprocessor.columns
                    ],
                    dtype=cp.float64,
                )
                values = (values - means) / stds
            elif isinstance(preprocessor, GPUSimpleImputer):
                fill_values: List[Any] = []
                for column in preprocessor.columns:
                    value = preprocessor._get_fill_value(column)
                    if value is None:
                        raise ValueError(
                            f"Column {column} has no fill value. "
                            "Check the data used to fit the SimpleImputer."
                        )
                    fill_values.append(value)
                fill_array = cp.asarray(fill_values, dtype=values.dtype).reshape(1, -1)
                values = cp.where(cp.isnan(values), fill_array, values)
            else:
                raise TypeError(
                    f"Unsupported fused GPU numeric transform: {preprocessor!r}."
                )

            output_dtype = getattr(preprocessor, "output_dtype", None)
            if output_dtype is not None:
                values = values.astype(output_dtype)

        output = _cudf_dataframe_from_cupy(values, self._output_columns, df.index)
        return _assign_dataframe_columns(df, self._output_columns, output)

    def _gpu_modified_columns(self) -> List[str]:
        return list(self._output_columns)


class _FusedGPUCategoricalOrdinalOp:
    def __init__(
        self,
        preprocessors: Sequence["GPUPreprocessor"],
        caster: Optional["GPUColumnCaster"],
        imputer: Optional["GPUSimpleImputer"],
        encoder: "GPUOrdinalEncoder",
    ):
        self._preprocessors = tuple(preprocessors)
        self._caster = caster
        self._imputer = imputer
        self._encoder = encoder
        self._columns = encoder.get_input_columns()
        self._output_columns = encoder.get_output_columns()
        self._gpu_maps: Dict[str, Dict[Any, int]] = {}

    def _prepare_gpu_state(self) -> None:
        for preprocessor in self._preprocessors:
            preprocessor._prepare_gpu_state()
        self._gpu_maps = {
            column: _ordinal_map_from_stats(self._encoder.stats_, column)
            for column in self._columns
        }

    def _transform_cudf(
        self, df: cudf.DataFrame, context: Optional[_GPUTransformContext] = None
    ) -> cudf.DataFrame:
        """Apply cast, imputation, and ordinal encoding to one working frame."""
        if any(column not in df.columns for column in self._columns):
            return _apply_gpu_ops(df, self._preprocessors)

        cudf = _import_cudf()
        working = df[self._columns].copy(deep=False)

        if self._caster is not None:
            cast_map = dict(zip(self._caster.columns, self._caster.output_dtypes))
            working = working.astype(cast_map)

        if self._imputer is not None:
            fill_values: Dict[str, Any] = {}
            for column in self._imputer.columns:
                value = self._imputer._get_fill_value(column)
                if value is None:
                    raise ValueError(
                        f"Column {column} has no fill value. "
                        "Check the data used to fit the SimpleImputer."
                    )
                fill_values[column] = value
            working = working.fillna(fill_values)
            if self._imputer.output_dtype is not None:
                working = working.astype(self._imputer.output_dtype)

        if self._encoder.encode_lists:
            for column in self._columns:
                if hasattr(working[column].dtype, "element_type"):
                    raise ValueError(
                        "GPUOrdinalEncoder doesn't support list columns yet."
                    )

        output_columns: List[str] = []
        encoded_columns: List[Any] = []
        for input_col, output_col in zip(
            self._encoder.columns, self._encoder.output_columns
        ):
            missing = working[input_col].isnull()
            if bool(missing.any()) and self._encoder.encoded_missing_value is None:
                raise ValueError(
                    f"Unable to transform column {input_col!r} because it contains "
                    "null values. Consider imputing missing values first."
                )

            mapping = self._gpu_maps.get(input_col) or _ordinal_map_from_stats(
                self._encoder.stats_, input_col
            )
            codes = working[input_col].map(mapping)
            unknown = codes.isnull() & ~missing
            if self._encoder.unknown_value is not None:
                codes = codes.mask(unknown, self._encoder.unknown_value)
            if self._encoder.encoded_missing_value is not None:
                codes = codes.mask(missing, self._encoder.encoded_missing_value)
            if self._encoder.output_dtype is not None:
                codes = codes.astype(self._encoder.output_dtype)
            output_columns.append(output_col)
            encoded_columns.append(codes)

        encoded = cudf.DataFrame(dict(zip(output_columns, encoded_columns)))
        encoded.index = df.index
        return _assign_dataframe_columns(df, output_columns, encoded)

    def _gpu_modified_columns(self) -> List[str]:
        return list(self._output_columns)


def _match_fused_numeric_ops(
    preprocessors: Sequence["GPUPreprocessor"],
    start: int,
) -> Optional[Tuple[_FusedGPUNumericColumnOp, int]]:
    """Match a contiguous run of compatible in-place numeric transforms."""
    if not _is_fusible_numeric_transform(preprocessors[start]):
        return None

    ops: List[GPUPreprocessor] = []
    columns: Optional[List[str]] = None
    index = start
    while index < len(preprocessors):
        preprocessor = preprocessors[index]
        if not _is_fusible_numeric_transform(preprocessor):
            break
        input_columns = preprocessor.get_input_columns()
        if columns is None:
            columns = input_columns
        elif input_columns != columns:
            break
        ops.append(preprocessor)
        index += 1

    if len(ops) < 2:
        return None
    if not any(isinstance(op, (GPUPowerTransformer, GPUStandardScaler)) for op in ops):
        return None
    return _FusedGPUNumericColumnOp(ops), index


def _match_fused_categorical_ordinal_ops(
    preprocessors: Sequence["GPUPreprocessor"],
    start: int,
) -> Optional[Tuple[_FusedGPUCategoricalOrdinalOp, int]]:
    """Match an optional cast/impute prefix followed by ordinal encoding."""
    ops: List[GPUPreprocessor] = []
    index = start
    caster = None
    imputer = None

    if isinstance(preprocessors[index], GPUColumnCaster):
        caster = preprocessors[index]
        if not _is_in_place_column_transform(caster):
            return None
        ops.append(caster)
        index += 1
        if index >= len(preprocessors):
            return None

    if isinstance(preprocessors[index], GPUSimpleImputer):
        imputer = preprocessors[index]
        if not _is_in_place_column_transform(imputer):
            return None
        ops.append(imputer)
        index += 1
        if index >= len(preprocessors):
            return None

    if not isinstance(preprocessors[index], GPUOrdinalEncoder):
        return None

    encoder = preprocessors[index]
    if not _is_in_place_column_transform(encoder):
        return None

    encoder_columns = encoder.get_input_columns()
    if caster is not None and caster.get_input_columns() != encoder_columns:
        return None
    if imputer is not None and imputer.get_input_columns() != encoder_columns:
        return None

    ops.append(encoder)
    if len(ops) < 2:
        return None
    return _FusedGPUCategoricalOrdinalOp(ops, caster, imputer, encoder), index + 1


def _plan_gpu_transform_ops(
    preprocessors: Sequence["GPUPreprocessor"],
) -> Tuple[_GPUPhysicalOp, ...]:
    """Plan logical GPU preprocessors into fused physical transform operations."""
    planned: List[_GPUPhysicalOp] = []
    index = 0
    while index < len(preprocessors):
        categorical_match = _match_fused_categorical_ordinal_ops(preprocessors, index)
        if categorical_match is not None:
            op, index = categorical_match
            planned.append(op)
            continue

        numeric_match = _match_fused_numeric_ops(preprocessors, index)
        if numeric_match is not None:
            op, index = numeric_match
            planned.append(op)
            continue

        planned.append(preprocessors[index])
        index += 1
    return tuple(planned)


class _FusedGPUChainUDF:
    def __init__(self, preprocessors: Sequence["GPUPreprocessor"]):
        self._preprocessors = tuple(preprocessors)
        self._ops = _plan_gpu_transform_ops(self._preprocessors)
        for op in self._ops:
            op._prepare_gpu_state()

    def __call__(self, batch: cudf.DataFrame) -> cudf.DataFrame:
        return _apply_gpu_physical_ops(batch, self._ops, prepare=False)


class _SingleGPUPreprocessorUDF(_FusedGPUChainUDF):
    def __init__(self, preprocessor: "GPUPreprocessor"):
        super().__init__((preprocessor,))


class _GPUFitStatsUDF:
    def __init__(
        self,
        fit_entries: Sequence[
            Tuple[int, "GPUPreprocessor", Sequence["GPUPreprocessor"]]
        ],
    ) -> None:
        self._fit_entries = tuple(
            (index, preprocessor, tuple(prefix))
            for index, preprocessor, prefix in fit_entries
        )
        for _, _, prefix in self._fit_entries:
            for prefix_preprocessor in prefix:
                prefix_preprocessor._prepare_gpu_state()

    def __call__(self, batch: cudf.DataFrame) -> pd.DataFrame:
        """Compute serialized partial fit statistics for one cuDF batch."""
        prefix_cache: Dict[Tuple["GPUPreprocessor", ...], cudf.DataFrame] = {}
        rows: List[Dict[str, Any]] = []
        for index, preprocessor, prefix in self._fit_entries:
            if prefix not in prefix_cache:
                prefix_cache[prefix] = _apply_gpu_ops(batch, prefix)
            stats = preprocessor._gpu_fit_stats_cudf(prefix_cache[prefix])
            if stats.empty:
                continue
            rows.append(
                {
                    _COMBINED_FIT_INDEX_COLUMN: index,
                    _COMBINED_FIT_STATS_COLUMN: pickle.dumps(
                        stats, protocol=pickle.HIGHEST_PROTOCOL
                    ),
                }
            )

        return pd.DataFrame(
            rows, columns=[_COMBINED_FIT_INDEX_COLUMN, _COMBINED_FIT_STATS_COLUMN]
        )


@DeveloperAPI
class GPUPreprocessor(SerializablePreprocessorBase):
    """Base class for preprocessors that transform cuDF batches on GPU.

    Args:
        batch_size: Number of rows per cuDF batch during fit and transform.
        num_gpus_per_worker: GPUs allocated to each ``map_batches`` worker.
        concurrency: Maximum number of concurrent GPU workers. If ``None``,
            Ray Data chooses concurrency automatically.
    """

    _is_fittable = False

    def __init__(
        self,
        *,
        batch_size: int = _DEFAULT_GPU_BATCH_SIZE,
        num_gpus_per_worker: float = 1,
        concurrency: Optional[int] = None,
    ):
        super().__init__()
        self._batch_size = batch_size
        self._num_gpus_per_worker = num_gpus_per_worker
        self._concurrency = concurrency

    def _fit(self, dataset: "Dataset") -> "GPUPreprocessor":
        """Fit this preprocessor on a dataset.

        Subclasses should override :meth:`_fit_gpu` instead of this method.
        """
        return self._fit_gpu(dataset, ())

    def _fit_gpu(
        self, dataset: "Dataset", prefix: Sequence["GPUPreprocessor"]
    ) -> "GPUPreprocessor":
        """Fit this preprocessor on GPU, optionally after a prefix chain.

        When fitting inside a :class:`GPUChain`, ``prefix`` contains the
        preprocessors that must run on each batch before this one computes fit
        statistics.

        Args:
            dataset: The dataset to fit on.
            prefix: Preprocessors that transform each batch before fitting.

        Returns:
            This fitted preprocessor.
        """
        return self

    def _fit_gpu_with_stats(
        self, dataset: "Dataset", prefix: Sequence["GPUPreprocessor"]
    ) -> "GPUPreprocessor":
        """Fit from per-batch statistics computed by a GPU actor UDF."""
        partials = dataset.map_batches(
            _GPUFitStatsUDF,
            fn_constructor_args=(((0, self, tuple(prefix)),),),
            batch_format="cudf",
            batch_size=self._batch_size,
            num_gpus=self._num_gpus_per_worker,
            zero_copy_batch=True,
            udf_modifying_row_count=True,
            **_gpu_actor_compute_strategy(self._concurrency),
        )

        partial_batches: List[pd.DataFrame] = []
        for batch in partials.iter_batches(batch_size=None, batch_format="pandas"):
            if batch.empty or _COMBINED_FIT_STATS_COLUMN not in batch:
                continue
            for index, payload in batch[
                [_COMBINED_FIT_INDEX_COLUMN, _COMBINED_FIT_STATS_COLUMN]
            ].itertuples(index=False, name=None):
                if int(index) == 0:
                    partial_batches.append(_deserialize_pandas_fit_stats(payload))

        stats = (
            pd.concat(partial_batches, ignore_index=True, sort=False)
            if partial_batches
            else pd.DataFrame()
        )
        self._finalize_gpu_fit_stats(stats)
        return self

    def _prepare_gpu_state(self) -> None:
        """Prepare per-worker GPU state before fit or transform.

        Called once when a ``map_batches`` worker is constructed. Subclasses
        can override this to load fit statistics or other state onto the GPU.
        """
        pass

    def _transform_cudf(
        self, df: cudf.DataFrame, context: Optional[_GPUTransformContext] = None
    ) -> cudf.DataFrame:
        """Transform a single cuDF batch.

        Args:
            df: Input cuDF DataFrame batch.
            context: Shared cache for the current fused transform batch. When
                this preprocessor runs inside a :class:`GPUChain`, the chain
                passes a shared context so intermediate representations (such
                as tokenized text) can be reused across preprocessors. It is
                ``None`` when the preprocessor is applied standalone.

        Returns:
            Transformed cuDF DataFrame batch.
        """
        raise NotImplementedError

    def _gpu_modified_columns(self) -> List[str]:
        """Return columns modified by :meth:`_transform_cudf`.

        Used by fused GPU chains to invalidate cached intermediates when a
        preprocessor changes its source columns.

        Returns:
            Column names written or updated by this preprocessor.
        """
        return self.get_output_columns()

    def _supports_gpu_combined_fit(self) -> bool:
        """Return whether this preprocessor supports combined GPU fitting.

        When ``True``, a :class:`GPUChain` may fuse fit-statistics collection
        for this preprocessor with other fittable preprocessors in a single
        GPU ``map_batches`` pass.

        Returns:
            ``True`` if combined GPU fitting is supported, else ``False``.
        """
        return False

    def _gpu_fit_stats_cudf(self, df: cudf.DataFrame) -> pd.DataFrame:
        """Compute partial fit statistics for one cuDF batch.

        Args:
            df: Input cuDF DataFrame batch, after any required prefix
                preprocessors have been applied.

        Returns:
            Pandas DataFrame of partial statistics for this batch.
        """
        raise NotImplementedError

    def _finalize_gpu_fit_stats(self, partials: pd.DataFrame) -> None:
        """Aggregate partial fit statistics into ``stats_``.

        Args:
            partials: Concatenated partial statistics from all batches.
        """
        raise NotImplementedError

    def _transform(
        self,
        ds: "Dataset",
        batch_size: Optional[int],
        num_cpus: Optional[float] = None,
        memory: Optional[float] = None,
        concurrency: Optional[int] = None,
    ) -> "Dataset":
        """Transform a dataset by applying this preprocessor on GPU.

        Args:
            ds: The dataset to transform.
            batch_size: Rows per cuDF batch. Defaults to this preprocessor's
                configured batch size.
            num_cpus: Not supported for GPU preprocessors.
            memory: Not supported for GPU preprocessors.
            concurrency: Maximum number of concurrent GPU workers. Defaults to
                this preprocessor's configured concurrency.

        Returns:
            The transformed dataset.

        Raises:
            ValueError: If ``num_cpus`` or ``memory`` is provided.
        """
        if num_cpus is not None:
            raise ValueError("GPUPreprocessor does not support transform num_cpus.")
        if memory is not None:
            raise ValueError("GPUPreprocessor does not support transform memory.")

        kwargs: Dict[str, Any] = {
            "batch_format": "cudf",
            "batch_size": batch_size or self._batch_size,
            "num_gpus": self._num_gpus_per_worker,
            "zero_copy_batch": True,
            "udf_modifying_row_count": False,
        }
        effective_concurrency = concurrency or self._concurrency
        kwargs.update(_gpu_actor_compute_strategy(effective_concurrency))

        return ds.map_batches(
            _SingleGPUPreprocessorUDF,
            fn_constructor_args=(self,),
            **kwargs,
        )

    def _base_serializable_fields(self) -> Dict[str, Any]:
        """Return GPU execution settings shared by all GPU preprocessors.

        Returns:
            Dictionary of base GPU preprocessor configuration fields.
        """
        return {
            "batch_size": self._batch_size,
            "num_gpus_per_worker": self._num_gpus_per_worker,
            "concurrency": self._concurrency,
            "_fitted": getattr(self, "_fitted", None),
        }

    def _set_base_serializable_fields(self, fields: Dict[str, Any]) -> None:
        """Restore GPU execution settings from serialized data.

        Args:
            fields: Dictionary containing base GPU preprocessor fields.
        """
        self._batch_size = fields.get("batch_size", _DEFAULT_GPU_BATCH_SIZE)
        self._num_gpus_per_worker = fields.get("num_gpus_per_worker", 1)
        self._concurrency = fields.get("concurrency")
        self._fitted = fields.get("_fitted")

    def _get_serializable_fields(self) -> Dict[str, Any]:
        """Return instance fields that should be serialized.

        Returns:
            Dictionary mapping field names to their values.
        """
        return self._base_serializable_fields()

    def _set_serializable_fields(self, fields: Dict[str, Any], version: int):
        """Restore instance fields from deserialized data.

        Args:
            fields: Dictionary of field names to values.
            version: Version of the serialized data.
        """
        self._set_base_serializable_fields(fields)


@PublicAPI(stability="alpha")
@SerializablePreprocessor(version=1, identifier="io.ray.preprocessors.gpu_chain")
class GPUChain(SerializablePreprocessorBase):
    """Fuse GPU preprocessors into one cuDF ``map_batches`` stage."""

    def __init__(
        self,
        *preprocessors: GPUPreprocessor,
        batch_size: int = _DEFAULT_GPU_BATCH_SIZE,
        num_gpus_per_worker: float = 1,
        concurrency: Optional[int] = None,
    ):
        super().__init__()
        for preprocessor in preprocessors:
            if not isinstance(preprocessor, GPUPreprocessor):
                raise TypeError(
                    "GPUChain only supports GPUPreprocessor instances; got "
                    f"{type(preprocessor)!r}."
                )
        self._preprocessors = tuple(preprocessors)
        self._batch_size = batch_size
        self._num_gpus_per_worker = num_gpus_per_worker
        self._concurrency = concurrency

    @property
    def preprocessors(self) -> Tuple[GPUPreprocessor, ...]:
        """Return the preprocessors in execution order."""
        return self._preprocessors

    def fit_status(self) -> Preprocessor.FitStatus:
        """Return the aggregate fit status of the GPU preprocessors."""
        fittable_count = 0
        fitted_count = 0

        for preprocessor in self._preprocessors:
            status = preprocessor.fit_status()
            if status == Preprocessor.FitStatus.FITTED:
                fittable_count += 1
                fitted_count += 1
            elif status in (
                Preprocessor.FitStatus.NOT_FITTED,
                Preprocessor.FitStatus.PARTIALLY_FITTED,
            ):
                fittable_count += 1
            else:
                assert status == Preprocessor.FitStatus.NOT_FITTABLE

        if fittable_count == 0:
            return Preprocessor.FitStatus.NOT_FITTABLE
        if fitted_count == fittable_count:
            return Preprocessor.FitStatus.FITTED
        if fitted_count > 0:
            return Preprocessor.FitStatus.PARTIALLY_FITTED
        return Preprocessor.FitStatus.NOT_FITTED

    @staticmethod
    def _required_prefix(
        preprocessor: "GPUPreprocessor", prefix: Sequence["GPUPreprocessor"]
    ) -> List["GPUPreprocessor"]:
        """Select earlier transforms that produce inputs needed for fitting."""
        required = set(preprocessor.get_input_columns())
        selected: List[GPUPreprocessor] = []
        for candidate in reversed(prefix):
            if set(candidate.get_output_columns()).intersection(required):
                selected.append(candidate)
                required.update(candidate.get_input_columns())
        selected.reverse()
        return selected

    def _fit_combined(self, ds: "Dataset") -> bool:
        """Fit multiple compatible preprocessors in one dataset pass."""
        prefix: List[GPUPreprocessor] = []
        fit_entries: List[Tuple[int, GPUPreprocessor, Tuple[GPUPreprocessor, ...]]] = []

        for index, preprocessor in enumerate(self._preprocessors):
            if preprocessor.fit_status() != Preprocessor.FitStatus.NOT_FITTABLE:
                if not preprocessor._supports_gpu_combined_fit():
                    return False
                required_prefix = tuple(self._required_prefix(preprocessor, prefix))
                fit_entries.append((index, preprocessor, required_prefix))
            prefix.append(preprocessor)

        if len(fit_entries) < 2:
            return False

        fittable_preprocessors = {preprocessor for _, preprocessor, _ in fit_entries}
        if any(
            preprocessor in fittable_preprocessors
            for _, _, required_prefix in fit_entries
            for preprocessor in required_prefix
        ):
            return False

        kwargs = _gpu_actor_compute_strategy(self._concurrency)

        partials = ds.map_batches(
            _GPUFitStatsUDF,
            fn_constructor_args=(tuple(fit_entries),),
            batch_format="cudf",
            batch_size=self._batch_size,
            num_gpus=self._num_gpus_per_worker,
            zero_copy_batch=True,
            udf_modifying_row_count=True,
            **kwargs,
        )

        partial_batches: Dict[int, List[pd.DataFrame]] = {
            index: [] for index, _, _ in fit_entries
        }

        for batch in partials.iter_batches(batch_size=None, batch_format="pandas"):
            if batch.empty or _COMBINED_FIT_STATS_COLUMN not in batch:
                continue
            for index, payload in batch[
                [_COMBINED_FIT_INDEX_COLUMN, _COMBINED_FIT_STATS_COLUMN]
            ].itertuples(index=False, name=None):
                index = int(index)
                if index not in partial_batches:
                    continue
                partial_batches[index].append(_deserialize_pandas_fit_stats(payload))

        for index, preprocessor, _ in fit_entries:
            preprocessor_partials = partial_batches[index]
            if preprocessor_partials:
                stats = pd.concat(preprocessor_partials, ignore_index=True, sort=False)
            else:
                stats = pd.DataFrame()
            preprocessor._finalize_gpu_fit_stats(stats)
            preprocessor._fitted = True
        return True

    def _fit(self, ds: "Dataset") -> "GPUChain":
        """Fit each fittable preprocessor, combining statistics when possible."""
        if self._fit_combined(ds):
            return self

        prefix: List[GPUPreprocessor] = []
        for preprocessor in self._preprocessors:
            if preprocessor.fit_status() != Preprocessor.FitStatus.NOT_FITTABLE:
                original_concurrency = preprocessor._concurrency
                if original_concurrency is None and self._concurrency is not None:
                    preprocessor._concurrency = self._concurrency
                try:
                    preprocessor._fit_gpu(
                        ds, self._required_prefix(preprocessor, prefix)
                    )
                finally:
                    preprocessor._concurrency = original_concurrency
                preprocessor._fitted = True
            prefix.append(preprocessor)
        return self

    def _transform(
        self,
        ds: "Dataset",
        batch_size: Optional[int],
        num_cpus: Optional[float] = None,
        memory: Optional[float] = None,
        concurrency: Optional[int] = None,
        output_batch_format: Optional[str] = None,
    ) -> "Dataset":
        """Transform a dataset in one fused GPU actor stage."""
        if num_cpus is not None:
            raise ValueError("GPUChain does not support transform num_cpus.")
        if memory is not None:
            raise ValueError("GPUChain does not support transform memory.")

        kwargs: Dict[str, Any] = {
            "batch_format": "cudf",
            "batch_size": batch_size or self._batch_size,
            "num_gpus": self._num_gpus_per_worker,
            "output_batch_format": output_batch_format,
            "zero_copy_batch": True,
            "udf_modifying_row_count": False,
        }
        effective_concurrency = concurrency or self._concurrency
        kwargs.update(_gpu_actor_compute_strategy(effective_concurrency))

        return ds.map_batches(
            _FusedGPUChainUDF,
            fn_constructor_args=(self._preprocessors,),
            **kwargs,
        )

    def transform(
        self,
        ds: "Dataset",
        *,
        batch_size: Optional[int] = None,
        num_cpus: Optional[float] = None,
        memory: Optional[float] = None,
        concurrency: Optional[int] = None,
        output_batch_format: Optional[str] = None,
    ) -> "Dataset":
        """Transform a dataset and optionally convert GPU outputs to a block format."""
        fit_status = self.fit_status()
        if fit_status in (
            Preprocessor.FitStatus.PARTIALLY_FITTED,
            Preprocessor.FitStatus.NOT_FITTED,
        ):
            from ray.data.preprocessor import PreprocessorNotFittedException

            raise PreprocessorNotFittedException(
                "`fit` must be called before `transform`, "
                "or simply use fit_transform() to run both steps"
            )
        return self._transform(
            ds,
            batch_size=batch_size,
            num_cpus=num_cpus,
            memory=memory,
            concurrency=concurrency,
            output_batch_format=output_batch_format,
        )

    def transform_cudf(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """Transform one cuDF DataFrame eagerly."""
        return _apply_gpu_transform_ops(df, self._preprocessors)

    def _transform_batch(self, df: cudf.DataFrame) -> cudf.DataFrame:
        return self.transform_cudf(df)

    def __repr__(self) -> str:
        arguments = ", ".join(repr(p) for p in self._preprocessors)
        return f"{self.__class__.__name__}({arguments})"

    def _get_serializable_fields(self) -> Dict[str, Any]:
        return {
            "preprocessors": self._preprocessors,
            "batch_size": self._batch_size,
            "num_gpus_per_worker": self._num_gpus_per_worker,
            "concurrency": self._concurrency,
            "_fitted": getattr(self, "_fitted", None),
        }

    def _set_serializable_fields(self, fields: Dict[str, Any], version: int):
        self._preprocessors = tuple(fields["preprocessors"])
        self._batch_size = fields.get("batch_size", _DEFAULT_GPU_BATCH_SIZE)
        self._num_gpus_per_worker = fields.get("num_gpus_per_worker", 1)
        self._concurrency = fields.get("concurrency")
        self._fitted = fields.get("_fitted")


@PublicAPI(stability="alpha")
@SerializablePreprocessor(version=1, identifier="io.ray.preprocessors.gpu_text_stats")
class GPUTextStatsPreprocessor(GPUPreprocessor):
    """Append GPU-computed text count features for string columns."""

    _is_fittable = False

    def __init__(
        self,
        text_column: str,
        *,
        word_pattern: str = _DEFAULT_WORD_PATTERN,
        token_pattern: str = _DEFAULT_TOKEN_PATTERN,
        batch_size: int = _DEFAULT_GPU_BATCH_SIZE,
        num_gpus_per_worker: float = 1,
        concurrency: Optional[int] = None,
    ):
        super().__init__(
            batch_size=batch_size,
            num_gpus_per_worker=num_gpus_per_worker,
            concurrency=concurrency,
        )
        self._text_column = text_column
        self._word_pattern = word_pattern
        self._token_pattern = token_pattern

    def _transform_cudf(
        self, df: cudf.DataFrame, context: Optional[_GPUTransformContext] = None
    ) -> cudf.DataFrame:
        """Append word, line, and tokenizer-token counts for one text column."""
        context = context or _GPUTransformContext()
        text = context.string_column(df, self._text_column)
        _, token_lengths = context.tokenized_text(
            df, self._text_column, self._token_pattern
        )
        df["word_count"] = (
            _str_count(text, self._word_pattern).fillna(0).astype("int64")
        )
        newline_count = _str_count(text, r"\n").fillna(0).astype("int64")
        non_empty = text.str.len().fillna(0) > 0
        df["line_count"] = (newline_count + 1).where(non_empty, 0).astype("int64")
        df["tokenizer_token_count"] = token_lengths.astype("int64")
        return df

    def get_input_columns(self) -> List[str]:
        return [self._text_column]

    def get_output_columns(self) -> List[str]:
        return ["word_count", "line_count", "tokenizer_token_count"]

    def _get_serializable_fields(self) -> Dict[str, Any]:
        return {
            **self._base_serializable_fields(),
            "text_column": self._text_column,
            "word_pattern": self._word_pattern,
            "token_pattern": self._token_pattern,
        }

    def _set_serializable_fields(self, fields: Dict[str, Any], version: int):
        self._set_base_serializable_fields(fields)
        self._text_column = fields["text_column"]
        self._word_pattern = fields.get("word_pattern", _DEFAULT_WORD_PATTERN)
        self._token_pattern = fields.get("token_pattern", _DEFAULT_TOKEN_PATTERN)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(text_column={self._text_column!r})"


@PublicAPI(stability="alpha")
@SerializablePreprocessor(
    version=1, identifier="io.ray.preprocessors.gpu_standard_scaler"
)
class GPUStandardScaler(GPUPreprocessor):
    """GPU-native variant of :class:`StandardScaler`."""

    _is_fittable = True

    def __init__(
        self,
        columns: List[str],
        output_columns: Optional[List[str]] = None,
        *,
        output_dtype: Optional[Any] = None,
        batch_size: int = _DEFAULT_GPU_BATCH_SIZE,
        num_gpus_per_worker: float = 1,
        concurrency: Optional[int] = None,
    ):
        super().__init__(
            batch_size=batch_size,
            num_gpus_per_worker=num_gpus_per_worker,
            concurrency=concurrency,
        )
        self._columns = columns
        self._output_columns = Preprocessor._derive_and_validate_output_columns(
            columns, output_columns
        )
        self._output_dtype = output_dtype

    @property
    def columns(self) -> List[str]:
        return self._columns

    @property
    def output_columns(self) -> List[str]:
        return self._output_columns

    @property
    def output_dtype(self) -> Optional[Any]:
        return self._output_dtype

    def _supports_gpu_combined_fit(self) -> bool:
        return True

    def _gpu_fit_stats_cudf(self, df: cudf.DataFrame) -> pd.DataFrame:
        numeric = df[self._columns].astype("float64")
        counts = _to_pandas_dict(numeric.count())
        sums = _to_pandas_dict(numeric.sum())
        sum_sqs = _to_pandas_dict((numeric * numeric).sum())
        rows = [
            {
                "column": column,
                "count": int(counts.get(column, 0) or 0),
                "sum": float(sums.get(column, 0.0) or 0.0),
                "sum_sq": float(sum_sqs.get(column, 0.0) or 0.0),
            }
            for column in self._columns
        ]
        return pd.DataFrame(rows, columns=["column", "count", "sum", "sum_sq"])

    def _finalize_gpu_fit_stats(self, partials: pd.DataFrame) -> None:
        """Combine partial count, sum, and squared-sum statistics."""
        counts = {column: 0 for column in self._columns}
        sums = {column: 0.0 for column in self._columns}
        sum_sqs = {column: 0.0 for column in self._columns}
        if not partials.empty:
            for column, count, col_sum, col_sum_sq in partials[
                ["column", "count", "sum", "sum_sq"]
            ].itertuples(index=False, name=None):
                if column not in counts:
                    continue
                if _is_missing_value(count):
                    continue
                counts[column] += int(count)
                sums[column] += float(col_sum)
                sum_sqs[column] += float(col_sum_sq)

        self.stats_ = {}
        for column in self._columns:
            count = counts[column]
            if count == 0:
                self.stats_[f"mean({column})"] = None
                self.stats_[f"std({column})"] = None
                continue
            mean = sums[column] / count
            variance = max((sum_sqs[column] / count) - (mean * mean), 0.0)
            self.stats_[f"mean({column})"] = mean
            self.stats_[f"std({column})"] = math.sqrt(variance)

    def _fit_gpu(
        self, dataset: "Dataset", prefix: Sequence[GPUPreprocessor]
    ) -> "GPUStandardScaler":
        return self._fit_gpu_with_stats(dataset, prefix)

    def _transform_cudf(
        self, df: cudf.DataFrame, context: Optional[_GPUTransformContext] = None
    ) -> cudf.DataFrame:
        """Standardize configured columns using fitted GPU statistics."""
        cp = _import_cupy()
        values = df[self._columns].astype("float64").to_cupy(na_value=cp.nan)
        means = cp.asarray(
            [
                (
                    float(self.stats_.get(f"mean({column})"))
                    if self.stats_.get(f"mean({column})") is not None
                    else cp.nan
                )
                for column in self._columns
            ],
            dtype=cp.float64,
        )
        stds = cp.asarray(
            [
                (
                    float(self.stats_.get(f"std({column})"))
                    if self.stats_.get(f"std({column})") is not None
                    and self.stats_.get(f"std({column})") >= _EPSILON
                    else 1.0
                )
                for column in self._columns
            ],
            dtype=cp.float64,
        )
        scaled = (values - means) / stds
        if self._output_dtype is not None:
            scaled = scaled.astype(self._output_dtype)
        output = _cudf_dataframe_from_cupy(scaled, self._output_columns, df.index)
        return _assign_dataframe_columns(df, self._output_columns, output)

    def get_input_columns(self) -> List[str]:
        return list(self._columns)

    def get_output_columns(self) -> List[str]:
        return list(self._output_columns)

    def _get_serializable_fields(self) -> Dict[str, Any]:
        return {
            **self._base_serializable_fields(),
            "columns": self._columns,
            "output_columns": self._output_columns,
            "output_dtype": self._output_dtype,
        }

    def _set_serializable_fields(self, fields: Dict[str, Any], version: int):
        self._set_base_serializable_fields(fields)
        self._columns = fields["columns"]
        self._output_columns = fields["output_columns"]
        self._output_dtype = fields.get("output_dtype")

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)
        migrate_private_fields(
            self,
            fields={
                "_columns": _PublicField(public_field="columns"),
                "_output_columns": _PublicField(
                    public_field="output_columns",
                    default=_Computed(lambda obj: obj._columns),
                ),
                "_output_dtype": _PublicField(
                    public_field="output_dtype",
                    default=None,
                ),
            },
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(columns={self._columns!r}, "
            f"output_columns={self._output_columns!r}, "
            f"output_dtype={self._output_dtype!r})"
        )


@PublicAPI(stability="alpha")
@SerializablePreprocessor(
    version=1, identifier="io.ray.preprocessors.gpu_power_transformer"
)
class GPUPowerTransformer(GPUPreprocessor):
    """GPU-native variant of :class:`PowerTransformer`."""

    _valid_methods = ["yeo-johnson", "box-cox"]
    _is_fittable = False

    def __init__(
        self,
        columns: List[str],
        power: float,
        method: str = "yeo-johnson",
        *,
        output_columns: Optional[List[str]] = None,
        output_dtype: Optional[Any] = None,
        batch_size: int = _DEFAULT_GPU_BATCH_SIZE,
        num_gpus_per_worker: float = 1,
        concurrency: Optional[int] = None,
    ):
        super().__init__(
            batch_size=batch_size,
            num_gpus_per_worker=num_gpus_per_worker,
            concurrency=concurrency,
        )
        if method not in self._valid_methods:
            raise ValueError(
                f"Method {method} is not supported."
                f"Supported values are: {self._valid_methods}"
            )
        self._columns = columns
        self._method = method
        self._power = power
        self._output_columns = Preprocessor._derive_and_validate_output_columns(
            columns, output_columns
        )
        self._output_dtype = output_dtype

    @property
    def columns(self) -> List[str]:
        return self._columns

    @property
    def method(self) -> str:
        return self._method

    @property
    def power(self) -> float:
        return self._power

    @property
    def output_columns(self) -> List[str]:
        return self._output_columns

    @property
    def output_dtype(self) -> Optional[Any]:
        return self._output_dtype

    def _transform_cudf(
        self, df: cudf.DataFrame, context: Optional[_GPUTransformContext] = None
    ) -> cudf.DataFrame:
        """Apply the configured Yeo-Johnson or Box-Cox power transform."""
        cp = _import_cupy()
        values = df[self._columns].astype("float64").to_cupy(na_value=cp.nan)
        transformed = _power_transform_values(values, self._power, self._method, cp)

        if self._output_dtype is not None:
            transformed = transformed.astype(self._output_dtype)
        output = _cudf_dataframe_from_cupy(transformed, self._output_columns, df.index)
        return _assign_dataframe_columns(df, self._output_columns, output)

    def get_input_columns(self) -> List[str]:
        return list(self._columns)

    def get_output_columns(self) -> List[str]:
        return list(self._output_columns)

    def _get_serializable_fields(self) -> Dict[str, Any]:
        return {
            **self._base_serializable_fields(),
            "columns": self._columns,
            "power": self._power,
            "method": self._method,
            "output_columns": self._output_columns,
            "output_dtype": self._output_dtype,
        }

    def _set_serializable_fields(self, fields: Dict[str, Any], version: int):
        self._set_base_serializable_fields(fields)
        self._columns = fields["columns"]
        self._power = fields["power"]
        self._method = fields.get("method", "yeo-johnson")
        self._output_columns = fields["output_columns"]
        self._output_dtype = fields.get("output_dtype")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(columns={self._columns!r}, "
            f"power={self._power!r}, method={self._method!r}, "
            f"output_columns={self._output_columns!r}, "
            f"output_dtype={self._output_dtype!r})"
        )


@PublicAPI(stability="alpha")
@SerializablePreprocessor(
    version=1, identifier="io.ray.preprocessors.gpu_simple_imputer"
)
class GPUSimpleImputer(GPUPreprocessor):
    """GPU-native variant of :class:`SimpleImputer`."""

    _valid_strategies = ["mean", "most_frequent", "constant"]

    def __init__(
        self,
        columns: List[str],
        strategy: str = "mean",
        fill_value: Optional[Number] = None,
        *,
        output_columns: Optional[List[str]] = None,
        output_dtype: Optional[Any] = None,
        batch_size: int = _DEFAULT_GPU_BATCH_SIZE,
        num_gpus_per_worker: float = 1,
        concurrency: Optional[int] = None,
    ):
        super().__init__(
            batch_size=batch_size,
            num_gpus_per_worker=num_gpus_per_worker,
            concurrency=concurrency,
        )
        if strategy not in self._valid_strategies:
            raise ValueError(
                f"Strategy {strategy} is not supported."
                f"Supported values are: {self._valid_strategies}"
            )
        if strategy == "constant" and fill_value is None:
            raise ValueError('`fill_value` must be set when using "constant" strategy.')
        self._is_fittable = strategy != "constant"
        self._columns = columns
        self._strategy = strategy
        self._fill_value = fill_value
        self._output_columns = Preprocessor._derive_and_validate_output_columns(
            columns, output_columns
        )
        self._output_dtype = output_dtype

    @property
    def columns(self) -> List[str]:
        return self._columns

    @property
    def strategy(self) -> str:
        return self._strategy

    @property
    def fill_value(self) -> Optional[Number]:
        return self._fill_value

    @property
    def output_columns(self) -> List[str]:
        return self._output_columns

    @property
    def output_dtype(self) -> Optional[Any]:
        return self._output_dtype

    def _supports_gpu_combined_fit(self) -> bool:
        return self._strategy in {"mean", "most_frequent"}

    def _gpu_fit_stats_cudf(self, df: cudf.DataFrame) -> pd.DataFrame:
        """Compute per-batch mean or most-frequent imputation statistics."""
        if self._strategy == "mean":
            numeric = df[self._columns].astype("float64")
            counts = _to_pandas_dict(numeric.count())
            sums = _to_pandas_dict(numeric.sum())
            rows = [
                {
                    "column": column,
                    "count": int(counts.get(column, 0) or 0),
                    "sum": float(sums.get(column, 0.0) or 0.0),
                }
                for column in self._columns
            ]
            return pd.DataFrame(rows, columns=["column", "count", "sum"])

        rows: List[Dict[str, Any]] = []
        for column in self._columns:
            counts = df[column].value_counts(dropna=False)
            keys = counts.index.to_pandas().tolist()
            values = counts.to_pandas().tolist()
            rows.extend(
                {"column": column, "value": key, "count": int(value)}
                for key, value in zip(keys, values)
                if not _is_missing_value(key) and not _is_missing_value(value)
            )
        return pd.DataFrame(rows, columns=["column", "value", "count"])

    def _finalize_gpu_fit_stats(self, partials: pd.DataFrame) -> None:
        """Combine partials into one fitted fill value per input column."""
        self.stats_ = {}
        if self._strategy == "mean":
            counts = {column: 0 for column in self._columns}
            sums = {column: 0.0 for column in self._columns}
            if not partials.empty:
                for column, count, col_sum in partials[
                    ["column", "count", "sum"]
                ].itertuples(index=False, name=None):
                    if column not in counts or _is_missing_value(count):
                        continue
                    counts[column] += int(count)
                    sums[column] += float(col_sum)
            for column in self._columns:
                count = counts[column]
                self.stats_[f"mean({column})"] = sums[column] / count if count else None
            return

        counters: Dict[str, Counter] = {column: Counter() for column in self._columns}
        if not partials.empty:
            for column, value, count in partials[
                ["column", "value", "count"]
            ].itertuples(index=False, name=None):
                if column in counters and not _is_missing_value(value):
                    counters[column][value] += int(count)
        for column in self._columns:
            self.stats_[f"most_frequent({column})"] = (
                counters[column].most_common(1)[0][0] if counters[column] else None
            )

    def _fit_gpu(
        self, dataset: "Dataset", prefix: Sequence[GPUPreprocessor]
    ) -> "GPUSimpleImputer":
        return self._fit_gpu_with_stats(dataset, prefix)

    def _get_fill_value(self, column: str) -> Any:
        if self._strategy == "mean":
            return self.stats_[f"mean({column})"]
        if self._strategy == "most_frequent":
            return self.stats_[f"most_frequent({column})"]
        if self._strategy == "constant":
            return self._fill_value
        raise ValueError(
            f"Strategy {self._strategy} is not supported. "
            f"Supported values are: {self._valid_strategies}"
        )

    def _transform_cudf(
        self, df: cudf.DataFrame, context: Optional[_GPUTransformContext] = None
    ) -> cudf.DataFrame:
        """Fill nulls in configured columns and write the requested outputs."""
        existing_columns: List[str] = []
        existing_outputs: List[str] = []
        fill_values: Dict[str, Any] = {}
        for column, output_column in zip(self._columns, self._output_columns):
            value = self._get_fill_value(column)
            if value is None:
                raise ValueError(
                    f"Column {column} has no fill value. "
                    "Check the data used to fit the SimpleImputer."
                )
            if column in df.columns:
                existing_columns.append(column)
                existing_outputs.append(output_column)
                fill_values[column] = value
            else:
                df[output_column] = value

        if existing_columns:
            filled = df[existing_columns].copy(deep=False).fillna(fill_values)
            if self._output_dtype is not None:
                filled = filled.astype(self._output_dtype)
            filled.columns = existing_outputs
            df = _assign_dataframe_columns(df, existing_outputs, filled)
        return df

    def get_input_columns(self) -> List[str]:
        return list(self._columns)

    def get_output_columns(self) -> List[str]:
        return list(self._output_columns)

    def _get_serializable_fields(self) -> Dict[str, Any]:
        return {
            **self._base_serializable_fields(),
            "columns": self._columns,
            "strategy": self._strategy,
            "fill_value": self._fill_value,
            "output_columns": self._output_columns,
            "output_dtype": self._output_dtype,
        }

    def _set_serializable_fields(self, fields: Dict[str, Any], version: int):
        self._set_base_serializable_fields(fields)
        self._columns = fields["columns"]
        self._strategy = fields["strategy"]
        self._fill_value = fields.get("fill_value")
        self._output_columns = fields["output_columns"]
        self._output_dtype = fields.get("output_dtype")
        self._is_fittable = self._strategy != "constant"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(columns={self._columns!r}, "
            f"strategy={self._strategy!r}, fill_value={self._fill_value!r}, "
            f"output_columns={self._output_columns!r}, "
            f"output_dtype={self._output_dtype!r})"
        )


@PublicAPI(stability="alpha")
@SerializablePreprocessor(
    version=1, identifier="io.ray.preprocessors.gpu_ordinal_encoder"
)
class GPUOrdinalEncoder(GPUPreprocessor):
    """GPU-native variant of :class:`OrdinalEncoder` for scalar columns."""

    _is_fittable = True

    def __init__(
        self,
        columns: List[str],
        *,
        encode_lists: bool = True,
        output_columns: Optional[List[str]] = None,
        unknown_value: Optional[Number] = None,
        encoded_missing_value: Optional[Number] = None,
        output_dtype: Optional[Any] = None,
        encoded_value_offset: int = 0,
        batch_size: int = _DEFAULT_GPU_BATCH_SIZE,
        num_gpus_per_worker: float = 1,
        concurrency: Optional[int] = None,
    ):
        super().__init__(
            batch_size=batch_size,
            num_gpus_per_worker=num_gpus_per_worker,
            concurrency=concurrency,
        )
        self._columns = columns
        self._encode_lists = encode_lists
        self._output_columns = Preprocessor._derive_and_validate_output_columns(
            columns, output_columns
        )
        self._unknown_value = unknown_value
        self._encoded_missing_value = encoded_missing_value
        self._output_dtype = output_dtype
        self._encoded_value_offset = encoded_value_offset
        self._gpu_maps: Dict[str, Dict[Any, int]] = {}

    @property
    def columns(self) -> List[str]:
        return self._columns

    @property
    def encode_lists(self) -> bool:
        return self._encode_lists

    @property
    def output_columns(self) -> List[str]:
        return self._output_columns

    @property
    def unknown_value(self) -> Optional[Number]:
        return self._unknown_value

    @property
    def encoded_missing_value(self) -> Optional[Number]:
        return self._encoded_missing_value

    @property
    def output_dtype(self) -> Optional[Any]:
        return self._output_dtype

    @property
    def encoded_value_offset(self) -> int:
        return self._encoded_value_offset

    def _supports_gpu_combined_fit(self) -> bool:
        return True

    def _gpu_fit_stats_cudf(self, df: cudf.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for column in self._columns:
            if bool(df[column].isnull().any()) and self._encoded_missing_value is None:
                raise ValueError(
                    "Unable to fit column because it contains null values. "
                    "Consider imputing missing values first."
                )
            counts = df[column].dropna().value_counts(dropna=False)
            keys = counts.index.to_pandas().tolist()
            values = counts.to_pandas().tolist()
            rows.extend(
                {"column": column, "value": key, "count": int(value)}
                for key, value in zip(keys, values)
                if not _is_missing_value(key) and not _is_missing_value(value)
            )
        return pd.DataFrame(rows, columns=["column", "value", "count"])

    def _finalize_gpu_fit_stats(self, partials: pd.DataFrame) -> None:
        """Combine category counts into deterministic ordinal mappings."""
        counters: Dict[str, Counter] = {column: Counter() for column in self._columns}
        if not partials.empty:
            for column, value, count in partials[
                ["column", "value", "count"]
            ].itertuples(index=False, name=None):
                if column in counters and not _is_missing_value(value):
                    counters[column][value] += int(count)

        self.stats_ = {}
        for column in self._columns:
            self.stats_[f"unique_values({column})"] = {
                value: index + self._encoded_value_offset
                for index, value in enumerate(sorted(counters[column]))
            }
        self._gpu_maps = {}

    def _fit_gpu(
        self, dataset: "Dataset", prefix: Sequence[GPUPreprocessor]
    ) -> "GPUOrdinalEncoder":
        return self._fit_gpu_with_stats(dataset, prefix)

    def _prepare_gpu_state(self) -> None:
        self._gpu_maps = {
            column: _ordinal_map_from_stats(self.stats_, column)
            for column in self._columns
        }

    def _transform_cudf(
        self, df: cudf.DataFrame, context: Optional[_GPUTransformContext] = None
    ) -> cudf.DataFrame:
        """Map configured categorical columns to fitted ordinal codes."""
        if self._encode_lists:
            for column in self._columns:
                if hasattr(df[column].dtype, "element_type"):
                    raise ValueError(
                        "GPUOrdinalEncoder doesn't support list columns yet."
                    )

        output_columns: List[str] = []
        encoded_columns: List[Any] = []
        for input_col, output_col in zip(self._columns, self._output_columns):
            missing = df[input_col].isnull()
            if bool(missing.any()) and self._encoded_missing_value is None:
                raise ValueError(
                    f"Unable to transform column {input_col!r} because it contains "
                    "null values. Consider imputing missing values first."
                )

            mapping = self._gpu_maps.get(input_col) or _ordinal_map_from_stats(
                self.stats_, input_col
            )
            codes = df[input_col].map(mapping)
            unknown = codes.isnull() & ~missing
            if self._unknown_value is not None:
                codes = codes.mask(unknown, self._unknown_value)
            if self._encoded_missing_value is not None:
                codes = codes.mask(missing, self._encoded_missing_value)
            if self._output_dtype is not None:
                codes = codes.astype(self._output_dtype)
            output_columns.append(output_col)
            encoded_columns.append(codes)

        cudf = _import_cudf()
        encoded = cudf.DataFrame(dict(zip(output_columns, encoded_columns)))
        encoded.index = df.index
        return _assign_dataframe_columns(df, output_columns, encoded)

    def get_input_columns(self) -> List[str]:
        return list(self._columns)

    def get_output_columns(self) -> List[str]:
        return list(self._output_columns)

    def _get_serializable_fields(self) -> Dict[str, Any]:
        return {
            **self._base_serializable_fields(),
            "columns": self._columns,
            "encode_lists": self._encode_lists,
            "output_columns": self._output_columns,
            "unknown_value": self._unknown_value,
            "encoded_missing_value": self._encoded_missing_value,
            "output_dtype": self._output_dtype,
            "encoded_value_offset": self._encoded_value_offset,
        }

    def _set_serializable_fields(self, fields: Dict[str, Any], version: int):
        self._set_base_serializable_fields(fields)
        self._columns = fields["columns"]
        self._encode_lists = fields.get("encode_lists", True)
        self._output_columns = fields["output_columns"]
        self._unknown_value = fields.get("unknown_value")
        self._encoded_missing_value = fields.get("encoded_missing_value")
        self._output_dtype = fields.get("output_dtype")
        self._encoded_value_offset = fields.get("encoded_value_offset", 0)
        self._gpu_maps = {}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(columns={self._columns!r}, "
            f"encode_lists={self._encode_lists!r}, "
            f"output_columns={self._output_columns!r}, "
            f"unknown_value={self._unknown_value!r}, "
            f"encoded_missing_value={self._encoded_missing_value!r}, "
            f"output_dtype={self._output_dtype!r}, "
            f"encoded_value_offset={self._encoded_value_offset!r})"
        )


@PublicAPI(stability="alpha")
@SerializablePreprocessor(
    version=1, identifier="io.ray.preprocessors.gpu_column_caster"
)
class GPUColumnCaster(GPUPreprocessor):
    """Cast one or more columns inside a fused GPU preprocessing stage."""

    _is_fittable = False

    def __init__(
        self,
        columns: List[str],
        output_dtype: Any,
        *,
        output_columns: Optional[List[str]] = None,
        batch_size: int = _DEFAULT_GPU_BATCH_SIZE,
        num_gpus_per_worker: float = 1,
        concurrency: Optional[int] = None,
    ):
        super().__init__(
            batch_size=batch_size,
            num_gpus_per_worker=num_gpus_per_worker,
            concurrency=concurrency,
        )
        self._columns = columns
        if isinstance(output_dtype, list):
            if len(output_dtype) != len(columns):
                raise ValueError(
                    f"Expected {len(columns)} values, but got "
                    f"{len(output_dtype)} values: {output_dtype!r}."
                )
            self._output_dtypes = output_dtype
        else:
            self._output_dtypes = [output_dtype for _ in columns]
        self._output_columns = Preprocessor._derive_and_validate_output_columns(
            columns, output_columns
        )

    @property
    def columns(self) -> List[str]:
        return self._columns

    @property
    def output_columns(self) -> List[str]:
        return self._output_columns

    @property
    def output_dtypes(self) -> List[Any]:
        return self._output_dtypes

    def _transform_cudf(
        self, df: cudf.DataFrame, context: Optional[_GPUTransformContext] = None
    ) -> cudf.DataFrame:
        """Cast configured columns to their requested cuDF dtypes."""
        cast_map = dict(zip(self._columns, self._output_dtypes))
        casted = df[self._columns].astype(cast_map)
        casted.columns = self._output_columns
        return _assign_dataframe_columns(df, self._output_columns, casted)

    def get_input_columns(self) -> List[str]:
        return list(self._columns)

    def get_output_columns(self) -> List[str]:
        return list(self._output_columns)

    def _get_serializable_fields(self) -> Dict[str, Any]:
        return {
            **self._base_serializable_fields(),
            "columns": self._columns,
            "output_dtypes": self._output_dtypes,
            "output_columns": self._output_columns,
        }

    def _set_serializable_fields(self, fields: Dict[str, Any], version: int):
        self._set_base_serializable_fields(fields)
        self._columns = fields["columns"]
        self._output_dtypes = fields["output_dtypes"]
        self._output_columns = fields["output_columns"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(columns={self._columns!r}, "
            f"output_dtypes={self._output_dtypes!r}, "
            f"output_columns={self._output_columns!r})"
        )


@PublicAPI(stability="alpha")
@SerializablePreprocessor(
    version=1, identifier="io.ray.preprocessors.gpu_one_hot_encoder"
)
class GPUOneHotEncoder(GPUPreprocessor):
    """GPU-native one-hot encoder that emits dense scalar indicator columns."""

    _is_fittable = True

    def __init__(
        self,
        columns: List[str],
        *,
        max_categories: Optional[Dict[str, int]] = None,
        output_columns: Optional[List[str]] = None,
        batch_size: int = _DEFAULT_GPU_BATCH_SIZE,
        num_gpus_per_worker: float = 1,
        concurrency: Optional[int] = None,
    ):
        super().__init__(
            batch_size=batch_size,
            num_gpus_per_worker=num_gpus_per_worker,
            concurrency=concurrency,
        )
        self._columns = columns
        self._max_categories = max_categories or {}
        self._output_columns = Preprocessor._derive_and_validate_output_columns(
            columns, output_columns
        )
        self._gpu_maps: Dict[str, Dict[Any, int]] = {}

    @property
    def columns(self) -> List[str]:
        return self._columns

    @property
    def output_columns(self) -> List[str]:
        return self._output_columns

    def _supports_gpu_combined_fit(self) -> bool:
        return True

    def _gpu_fit_stats_cudf(self, df: cudf.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for column in self._columns:
            counts = df[column].value_counts(dropna=False)
            keys = counts.index.to_pandas().tolist()
            vals = counts.to_pandas().tolist()
            rows.extend(
                {"column": column, "value": key, "count": int(value)}
                for key, value in zip(keys, vals)
                if not _is_missing_value(key) and not _is_missing_value(value)
            )
        return pd.DataFrame(rows, columns=["column", "value", "count"])

    def _finalize_gpu_fit_stats(self, partials: pd.DataFrame) -> None:
        """Combine category counts into deterministic one-hot mappings."""
        counters: Dict[str, Counter] = {column: Counter() for column in self._columns}
        if not partials.empty:
            for column, value, count in partials[
                ["column", "value", "count"]
            ].itertuples(index=False, name=None):
                if column not in counters:
                    continue
                if _is_missing_value(value) or _is_missing_value(count):
                    continue
                counters[column][value] += int(count)

        self.stats_ = {}
        for column in self._columns:
            counter = counters[column]
            if column in self._max_categories:
                values = list(dict(counter.most_common(self._max_categories[column])))
            else:
                values = list(counter.keys())
            self.stats_[f"unique_values({column})"] = {
                value: index for index, value in enumerate(sorted(values))
            }
        self._gpu_maps = {}

    def _fit_gpu(
        self, dataset: "Dataset", prefix: Sequence[GPUPreprocessor]
    ) -> "GPUOneHotEncoder":
        return self._fit_gpu_with_stats(dataset, prefix)

    def _prepare_gpu_state(self) -> None:
        self._gpu_maps = {
            column: self.stats_.get(f"unique_values({column})", {})
            for column in self._columns
        }

    def _transform_cudf(
        self, df: cudf.DataFrame, context: Optional[_GPUTransformContext] = None
    ) -> cudf.DataFrame:
        """Expand categorical columns into one-hot indicator columns."""
        cudf = _import_cudf()
        cp = _import_cupy()
        for input_col, output_col in zip(self._columns, self._output_columns):
            if bool(df[input_col].isnull().any()):
                raise ValueError(
                    f"Unable to transform column {input_col!r} because it contains "
                    "null values. Consider imputing missing values first."
                )
            mapping = (
                self._gpu_maps.get(input_col)
                or self.stats_[f"unique_values({input_col})"]
            )
            num_categories = len(mapping)
            codes = df[input_col].map(mapping).fillna(-1).astype("int32")
            codes_gpu = codes.to_cupy()
            one_hot = cp.zeros((len(df), num_categories), dtype=cp.uint8)
            if len(df) and num_categories:
                rows = cp.arange(len(df), dtype=cp.int32)
                valid = codes_gpu >= 0
                one_hot[rows[valid], codes_gpu[valid]] = 1
            for idx in range(num_categories):
                df[f"{output_col}_{idx}"] = cudf.Series(one_hot[:, idx], index=df.index)
            if output_col == input_col and input_col in df.columns:
                df = df.drop(columns=[input_col])
        return df

    def get_input_columns(self) -> List[str]:
        return list(self._columns)

    def get_output_columns(self) -> List[str]:
        outputs: List[str] = []
        for input_col, output_col in zip(self._columns, self._output_columns):
            mapping = self.stats_.get(f"unique_values({input_col})")
            if mapping is None:
                outputs.append(output_col)
            else:
                outputs.extend(f"{output_col}_{idx}" for idx in range(len(mapping)))
        return outputs

    def _gpu_modified_columns(self) -> List[str]:
        modified = self.get_output_columns()
        modified.extend(
            input_col
            for input_col, output_col in zip(self._columns, self._output_columns)
            if input_col == output_col
        )
        return modified

    def _get_serializable_fields(self) -> Dict[str, Any]:
        return {
            **self._base_serializable_fields(),
            "columns": self._columns,
            "max_categories": self._max_categories,
            "output_columns": self._output_columns,
        }

    def _set_serializable_fields(self, fields: Dict[str, Any], version: int):
        self._set_base_serializable_fields(fields)
        self._columns = fields["columns"]
        self._max_categories = fields.get("max_categories", {})
        self._output_columns = fields["output_columns"]
        self._gpu_maps = {}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(columns={self._columns!r}, "
            f"output_columns={self._output_columns!r})"
        )


@PublicAPI(stability="alpha")
@SerializablePreprocessor(
    version=1, identifier="io.ray.preprocessors.gpu_hashing_vectorizer"
)
class GPUHashingVectorizer(GPUPreprocessor):
    """GPU hashing vectorizer that emits dense scalar feature columns."""

    _is_fittable = False

    def __init__(
        self,
        columns: List[str],
        num_features: int,
        *,
        token_pattern: str = _DEFAULT_TOKEN_PATTERN,
        output_columns: Optional[List[str]] = None,
        batch_size: int = _DEFAULT_GPU_BATCH_SIZE,
        num_gpus_per_worker: float = 1,
        concurrency: Optional[int] = None,
    ):
        super().__init__(
            batch_size=batch_size,
            num_gpus_per_worker=num_gpus_per_worker,
            concurrency=concurrency,
        )
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        self._columns = columns
        self._num_features = num_features
        self._token_pattern = token_pattern
        self._output_columns = Preprocessor._derive_and_validate_output_columns(
            columns, output_columns
        )

    def _transform_cudf(
        self, df: cudf.DataFrame, context: Optional[_GPUTransformContext] = None
    ) -> cudf.DataFrame:
        """Tokenize string columns and append dense hashed-count features."""
        context = context or _GPUTransformContext()
        for input_col, output_col in zip(self._columns, self._output_columns):
            tokens, lengths = context.tokenized_text(df, input_col, self._token_pattern)
            df = _append_hashing_columns_from_tokens(
                df, tokens, lengths, output_col, self._num_features
            )
            if output_col == input_col and input_col in df.columns:
                df = df.drop(columns=[input_col])
        return df

    def _gpu_modified_columns(self) -> List[str]:
        modified = self.get_output_columns()
        modified.extend(
            input_col
            for input_col, output_col in zip(self._columns, self._output_columns)
            if input_col == output_col
        )
        return modified

    def get_input_columns(self) -> List[str]:
        return list(self._columns)

    def get_output_columns(self) -> List[str]:
        return [
            f"{output_col}_{idx}"
            for output_col in self._output_columns
            for idx in range(self._num_features)
        ]

    def _get_serializable_fields(self) -> Dict[str, Any]:
        return {
            **self._base_serializable_fields(),
            "columns": self._columns,
            "num_features": self._num_features,
            "token_pattern": self._token_pattern,
            "output_columns": self._output_columns,
        }

    def _set_serializable_fields(self, fields: Dict[str, Any], version: int):
        self._set_base_serializable_fields(fields)
        self._columns = fields["columns"]
        self._num_features = fields["num_features"]
        self._token_pattern = fields.get("token_pattern", _DEFAULT_TOKEN_PATTERN)
        self._output_columns = fields["output_columns"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(columns={self._columns!r}, "
            f"num_features={self._num_features!r}, "
            f"output_columns={self._output_columns!r})"
        )


@PublicAPI(stability="alpha")
@SerializablePreprocessor(
    version=1, identifier="io.ray.preprocessors.gpu_column_dropper"
)
class GPUColumnDropper(GPUPreprocessor):
    """Drop columns inside a fused GPU preprocessing stage."""

    _is_fittable = False

    def __init__(
        self,
        columns: List[str],
        *,
        batch_size: int = _DEFAULT_GPU_BATCH_SIZE,
        num_gpus_per_worker: float = 1,
        concurrency: Optional[int] = None,
    ):
        super().__init__(
            batch_size=batch_size,
            num_gpus_per_worker=num_gpus_per_worker,
            concurrency=concurrency,
        )
        self._columns = columns

    def _transform_cudf(
        self, df: cudf.DataFrame, context: Optional[_GPUTransformContext] = None
    ) -> cudf.DataFrame:
        """Drop configured columns that are present in the batch."""
        existing = [column for column in self._columns if column in df.columns]
        if not existing:
            return df
        return df.drop(columns=existing)

    def get_input_columns(self) -> List[str]:
        return list(self._columns)

    def get_output_columns(self) -> List[str]:
        return []

    def _gpu_modified_columns(self) -> List[str]:
        return list(self._columns)

    def _get_serializable_fields(self) -> Dict[str, Any]:
        return {**self._base_serializable_fields(), "columns": self._columns}

    def _set_serializable_fields(self, fields: Dict[str, Any], version: int):
        self._set_base_serializable_fields(fields)
        self._columns = fields["columns"]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(columns={self._columns!r})"
