import math
import pickle
from collections import Counter
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

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


def _str_count(series: Any, pattern: str):
    if hasattr(series.str, "count"):
        return series.str.count(pattern)
    if hasattr(series.str, "count_re"):
        return series.str.count_re(pattern)
    raise AttributeError("cuDF string columns do not expose count/count_re.")


def _list_lengths(list_series: Any):
    lengths = getattr(list_series.list, "len", None)
    if lengths is None:
        raise AttributeError("cuDF list columns do not expose list.len.")
    return lengths() if callable(lengths) else lengths


def _list_leaves(list_series: Any):
    leaves = getattr(list_series.list, "leaves", None)
    if leaves is None:
        raise AttributeError("cuDF list columns do not expose list.leaves.")
    return leaves() if callable(leaves) else leaves


def _hash_string_series(series: Any):
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


def _copy_for_gpu_transform(batch: Any):
    return batch.copy(deep=False) if hasattr(batch, "copy") else batch


def _serialize_pandas_fit_stats(stats: pd.DataFrame) -> bytes:
    return pickle.dumps(stats, protocol=pickle.HIGHEST_PROTOCOL)


def _deserialize_pandas_fit_stats(payload: Any) -> pd.DataFrame:
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    return pickle.loads(payload)


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

    def string_column(self, df: Any, column: str) -> Any:
        return self.get_or_compute(
            ("string_column", column),
            (column,),
            lambda: df[column].fillna("").astype("str"),
        )

    def lowercase_string_column(self, df: Any, column: str) -> Any:
        return self.get_or_compute(
            ("lowercase_string_column", column),
            (column,),
            lambda: self.string_column(df, column).str.lower(),
        )

    def tokenized_text(self, df: Any, column: str, pattern: str) -> Tuple[Any, Any]:
        def compute() -> Tuple[Any, Any]:
            text_lower = self.lowercase_string_column(df, column)
            tokens = text_lower.str.findall(pattern)
            lengths = _list_lengths(tokens).fillna(0).astype("int32")
            return tokens, lengths

        return self.get_or_compute(
            ("tokenized_text", column, pattern),
            (column,),
            compute,
        )


def _append_hashing_columns_from_tokens(
    df: Any,
    tokens: Any,
    lengths: Any,
    output_col: str,
    num_features: int,
) -> Any:
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


def _apply_gpu_ops(batch: Any, preprocessors: Sequence["GPUPreprocessor"]):
    df = _copy_for_gpu_transform(batch)
    context = _GPUTransformContext()
    for preprocessor in preprocessors:
        preprocessor._prepare_gpu_state()
        result = preprocessor._transform_cudf(df, context)
        if result is not None:
            df = result
        context.invalidate_columns(preprocessor._gpu_modified_columns())
    return df


def _required_prefix(
    preprocessor: "GPUPreprocessor", prefix: Sequence["GPUPreprocessor"]
) -> List["GPUPreprocessor"]:
    required = set(preprocessor.get_input_columns())
    selected: List[GPUPreprocessor] = []
    for candidate in reversed(prefix):
        outputs = set(candidate.get_output_columns())
        if outputs.intersection(required):
            selected.append(candidate)
            required.update(candidate.get_input_columns())
    selected.reverse()
    return selected


class _FusedGPUChainUDF:
    def __init__(self, preprocessors: Sequence["GPUPreprocessor"]):
        self._preprocessors = tuple(preprocessors)
        for preprocessor in self._preprocessors:
            preprocessor._prepare_gpu_state()

    def __call__(self, batch: Any):
        return _apply_gpu_ops(batch, self._preprocessors)


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

    def __call__(self, batch: Any) -> pd.DataFrame:
        prefix_cache: Dict[Tuple["GPUPreprocessor", ...], Any] = {}
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
                    _COMBINED_FIT_STATS_COLUMN: _serialize_pandas_fit_stats(stats),
                }
            )

        return pd.DataFrame(
            rows, columns=[_COMBINED_FIT_INDEX_COLUMN, _COMBINED_FIT_STATS_COLUMN]
        )


def _fit_gpu_with_stats_udf(
    preprocessor: "GPUPreprocessor",
    dataset: "Dataset",
    prefix: Sequence["GPUPreprocessor"],
) -> "GPUPreprocessor":
    kwargs: Dict[str, Any] = {}
    if preprocessor._concurrency is not None:
        kwargs["concurrency"] = preprocessor._concurrency

    partials = dataset.map_batches(
        _GPUFitStatsUDF,
        fn_constructor_args=(((0, preprocessor, tuple(prefix)),),),
        batch_format="cudf",
        batch_size=preprocessor._batch_size,
        num_gpus=preprocessor._num_gpus_per_worker,
        zero_copy_batch=True,
        udf_modifying_row_count=True,
        **kwargs,
    )

    partial_batches: List[pd.DataFrame] = []
    for batch in partials.iter_batches(batch_size=None, batch_format="pandas"):
        if batch.empty or _COMBINED_FIT_STATS_COLUMN not in batch:
            continue
        for index, payload in batch[
            [_COMBINED_FIT_INDEX_COLUMN, _COMBINED_FIT_STATS_COLUMN]
        ].itertuples(index=False, name=None):
            if int(index) != 0:
                continue
            partial_batches.append(_deserialize_pandas_fit_stats(payload))
    if partial_batches:
        stats = pd.concat(partial_batches, ignore_index=True, sort=False)
    else:
        stats = pd.DataFrame()
    preprocessor._finalize_gpu_fit_stats(stats)
    return preprocessor


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

    def _prepare_gpu_state(self) -> None:
        """Prepare per-worker GPU state before fit or transform.

        Called once when a ``map_batches`` worker is constructed. Subclasses
        can override this to load fit statistics or other state onto the GPU.
        """
        pass

    def _transform_cudf(
        self, df: Any, context: Optional[_GPUTransformContext] = None
    ) -> Any:
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

    def _gpu_fit_stats_cudf(self, df: Any) -> pd.DataFrame:
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
        if effective_concurrency is not None:
            kwargs["concurrency"] = effective_concurrency

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
        return self._preprocessors

    def fit_status(self) -> Preprocessor.FitStatus:
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

    def _fit_combined(self, ds: "Dataset") -> bool:
        prefix: List[GPUPreprocessor] = []
        fit_entries: List[Tuple[int, GPUPreprocessor, Tuple[GPUPreprocessor, ...]]] = []

        for index, preprocessor in enumerate(self._preprocessors):
            if preprocessor.fit_status() != Preprocessor.FitStatus.NOT_FITTABLE:
                if not preprocessor._supports_gpu_combined_fit():
                    return False
                required_prefix = tuple(_required_prefix(preprocessor, prefix))
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

        kwargs: Dict[str, Any] = {}
        if self._concurrency is not None:
            kwargs["concurrency"] = self._concurrency

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
        if self._fit_combined(ds):
            return self

        prefix: List[GPUPreprocessor] = []
        for preprocessor in self._preprocessors:
            if preprocessor.fit_status() != Preprocessor.FitStatus.NOT_FITTABLE:
                original_concurrency = preprocessor._concurrency
                if original_concurrency is None and self._concurrency is not None:
                    preprocessor._concurrency = self._concurrency
                try:
                    preprocessor._fit_gpu(ds, _required_prefix(preprocessor, prefix))
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
    ) -> "Dataset":
        if num_cpus is not None:
            raise ValueError("GPUChain does not support transform num_cpus.")
        if memory is not None:
            raise ValueError("GPUChain does not support transform memory.")

        kwargs: Dict[str, Any] = {
            "batch_format": "cudf",
            "batch_size": batch_size or self._batch_size,
            "num_gpus": self._num_gpus_per_worker,
            "zero_copy_batch": True,
            "udf_modifying_row_count": False,
        }
        effective_concurrency = concurrency or self._concurrency
        if effective_concurrency is not None:
            kwargs["concurrency"] = effective_concurrency

        return ds.map_batches(
            _FusedGPUChainUDF,
            fn_constructor_args=(self._preprocessors,),
            **kwargs,
        )

    def transform_cudf(self, df: Any) -> Any:
        return _apply_gpu_ops(df, self._preprocessors)

    def _transform_batch(self, df: Any) -> Any:
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
        self, df: Any, context: Optional[_GPUTransformContext] = None
    ) -> Any:
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

    @property
    def columns(self) -> List[str]:
        return self._columns

    @property
    def output_columns(self) -> List[str]:
        return self._output_columns

    def _supports_gpu_combined_fit(self) -> bool:
        return True

    def _gpu_fit_stats_cudf(self, df: Any) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for column in self._columns:
            col = df[column].astype("float64")
            valid = col.dropna()
            count = int(valid.count())
            if count:
                col_sum = float(valid.sum())
                col_sum_sq = float((valid * valid).sum())
            else:
                col_sum = 0.0
                col_sum_sq = 0.0
            rows.append(
                {
                    "column": column,
                    "count": count,
                    "sum": col_sum,
                    "sum_sq": col_sum_sq,
                }
            )
        return pd.DataFrame(rows, columns=["column", "count", "sum", "sum_sq"])

    def _finalize_gpu_fit_stats(self, partials: pd.DataFrame) -> None:
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
        return _fit_gpu_with_stats_udf(self, dataset, prefix)

    def _transform_cudf(
        self, df: Any, context: Optional[_GPUTransformContext] = None
    ) -> Any:
        cudf = _import_cudf()
        for input_col, output_col in zip(self._columns, self._output_columns):
            mean = self.stats_.get(f"mean({input_col})")
            std = self.stats_.get(f"std({input_col})")
            if mean is None or std is None:
                df[output_col] = cudf.Series(
                    [None] * len(df), index=df.index, dtype="float64"
                )
                continue
            if std < _EPSILON:
                std = 1
            df[output_col] = (df[input_col].astype("float64") - mean) / std
        return df

    def get_input_columns(self) -> List[str]:
        return list(self._columns)

    def get_output_columns(self) -> List[str]:
        return list(self._output_columns)

    def _get_serializable_fields(self) -> Dict[str, Any]:
        return {
            **self._base_serializable_fields(),
            "columns": self._columns,
            "output_columns": self._output_columns,
        }

    def _set_serializable_fields(self, fields: Dict[str, Any], version: int):
        self._set_base_serializable_fields(fields)
        self._columns = fields["columns"]
        self._output_columns = fields["output_columns"]

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
            },
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(columns={self._columns!r}, "
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

    def _gpu_fit_stats_cudf(self, df: Any) -> pd.DataFrame:
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
        return _fit_gpu_with_stats_udf(self, dataset, prefix)

    def _prepare_gpu_state(self) -> None:
        self._gpu_maps = {
            column: self.stats_.get(f"unique_values({column})", {})
            for column in self._columns
        }

    def _transform_cudf(
        self, df: Any, context: Optional[_GPUTransformContext] = None
    ) -> Any:
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
        self, df: Any, context: Optional[_GPUTransformContext] = None
    ) -> Any:
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
        self, df: Any, context: Optional[_GPUTransformContext] = None
    ) -> Any:
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
