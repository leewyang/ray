import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ray.data.preprocessor import (
    Preprocessor,
    PreprocessorNotFittedException,
    SerializablePreprocessorBase,
)
from ray.data.preprocessors.utils import (
    _PublicField,
    migrate_private_fields,
)
from ray.data.preprocessors.version_support import SerializablePreprocessor
from ray.data.util.data_batch_conversion import BatchFormat

if TYPE_CHECKING:
    from ray.air.data_batch_type import DataBatchType
    from ray.data.dataset import Dataset


@SerializablePreprocessor(version=1, identifier="io.ray.preprocessors.chain")
class Chain(SerializablePreprocessorBase):
    """Combine multiple preprocessors into a single :py:class:`Preprocessor`.

    When you call ``fit``, each preprocessor is fit on the dataset produced by the
    preceeding preprocessor's ``fit_transform``.

    Example:
        >>> import pandas as pd
        >>> import ray
        >>> from ray.data.preprocessors import *
        >>>
        >>> df = pd.DataFrame({
        ...     "X0": [0, 1, 2],
        ...     "X1": [3, 4, 5],
        ...     "Y": ["orange", "blue", "orange"],
        ... })
        >>> ds = ray.data.from_pandas(df)  # doctest: +SKIP
        >>>
        >>> preprocessor = Chain(
        ...     StandardScaler(columns=["X0", "X1"]),
        ...     Concatenator(columns=["X0", "X1"], output_column_name="X"),
        ...     LabelEncoder(label_column="Y")
        ... )
        >>> preprocessor.fit_transform(ds).to_pandas()  # doctest: +SKIP
           Y                                         X
        0  1  [-1.224744871391589, -1.224744871391589]
        1  0                                [0.0, 0.0]
        2  1    [1.224744871391589, 1.224744871391589]

    Args:
        *preprocessors: The preprocessors to sequentially compose.
    """

    def fit_status(self):
        fittable_count = 0
        fitted_count = 0

        for p in self._preprocessors:
            if p.fit_status() == Preprocessor.FitStatus.FITTED:
                fittable_count += 1
                fitted_count += 1
            elif p.fit_status() in (
                Preprocessor.FitStatus.NOT_FITTED,
                Preprocessor.FitStatus.PARTIALLY_FITTED,
            ):
                fittable_count += 1
            else:
                assert p.fit_status() == Preprocessor.FitStatus.NOT_FITTABLE
        if fittable_count > 0:
            if fitted_count == fittable_count:
                return Preprocessor.FitStatus.FITTED
            elif fitted_count > 0:
                return Preprocessor.FitStatus.PARTIALLY_FITTED
            else:
                return Preprocessor.FitStatus.NOT_FITTED
        else:
            return Preprocessor.FitStatus.NOT_FITTABLE

    def __init__(self, *preprocessors: SerializablePreprocessorBase):
        super().__init__()
        self._preprocessors = preprocessors
        self._gpu_chain = None

    @property
    def preprocessors(self) -> Tuple[SerializablePreprocessorBase, ...]:
        return self._preprocessors

    def _fit(self, ds: "Dataset") -> SerializablePreprocessorBase:
        for preprocessor in self._preprocessors[:-1]:
            ds = preprocessor.fit_transform(ds)
        self._preprocessors[-1].fit(ds)
        return self

    def fit(
        self,
        ds: "Dataset",
        *,
        accelerator: Optional[str] = None,
        batch_size: Optional[int] = None,
        num_gpus: Optional[int] = None,
        num_gpus_per_worker: float = 1,
        concurrency: Optional[int] = None,
    ) -> "Chain":
        if not self._use_gpu_accelerator(accelerator):
            self._gpu_chain = None
            return super().fit(ds)

        fit_status = self.fit_status()
        if fit_status in (
            Preprocessor.FitStatus.FITTED,
            Preprocessor.FitStatus.PARTIALLY_FITTED,
        ):
            warnings.warn(
                "`fit` has already been called on the preprocessor (or at least one "
                "contained preprocessors if this is a chain). "
                "All previously fitted state will be overwritten!"
            )

        self._stat_computation_plan.reset()
        self.stats_ = {}
        gpu_chain = self._build_gpu_chain(
            batch_size=batch_size,
            num_gpus_per_worker=num_gpus_per_worker,
            concurrency=self._gpu_concurrency(
                num_gpus=num_gpus,
                concurrency=concurrency,
            ),
            copy_fitted_state=False,
        )
        gpu_chain.fit(ds)
        self._sync_fitted_state_from_gpu_chain(gpu_chain)
        self._gpu_chain = gpu_chain
        self._fitted = True
        return self

    def fit_transform(
        self,
        ds: "Dataset",
        *,
        transform_num_cpus: Optional[float] = None,
        transform_memory: Optional[float] = None,
        transform_batch_size: Optional[int] = None,
        transform_concurrency: Optional[int] = None,
        accelerator: Optional[str] = None,
        num_gpus: Optional[int] = None,
        num_gpus_per_worker: float = 1,
    ) -> "Dataset":
        if not self._use_gpu_accelerator(accelerator):
            for preprocessor in self._preprocessors:
                ds = preprocessor.fit_transform(
                    ds,
                    transform_num_cpus=transform_num_cpus,
                    transform_memory=transform_memory,
                    transform_batch_size=transform_batch_size,
                    transform_concurrency=transform_concurrency,
                )
            return ds

        if transform_num_cpus is not None:
            raise ValueError("GPU Chain preprocessing does not support num_cpus.")
        if transform_memory is not None:
            raise ValueError("GPU Chain preprocessing does not support memory.")

        self.fit(
            ds,
            accelerator=accelerator,
            batch_size=transform_batch_size,
            num_gpus=num_gpus,
            num_gpus_per_worker=num_gpus_per_worker,
            concurrency=transform_concurrency,
        )
        return self.transform(
            ds,
            batch_size=transform_batch_size,
            accelerator=accelerator,
            num_gpus=num_gpus,
            num_gpus_per_worker=num_gpus_per_worker,
            concurrency=transform_concurrency,
        )

    @staticmethod
    def _use_gpu_accelerator(accelerator: Optional[str]) -> bool:
        if accelerator is None:
            return False
        if accelerator != "gpu":
            raise ValueError(
                f"Unsupported accelerator {accelerator!r}. Only 'gpu' is supported."
            )
        return True

    @staticmethod
    def _gpu_concurrency(
        *,
        num_gpus: Optional[int],
        concurrency: Optional[int],
    ) -> Optional[int]:
        if num_gpus is None:
            return concurrency
        if num_gpus <= 0:
            raise ValueError("num_gpus must be positive when accelerator='gpu'.")
        if concurrency is not None and concurrency != num_gpus:
            raise ValueError(
                "Specify either num_gpus or concurrency for GPU Chain preprocessing, "
                "or set them to the same value."
            )
        return num_gpus

    def _build_gpu_chain(
        self,
        *,
        batch_size: Optional[int],
        num_gpus_per_worker: float,
        concurrency: Optional[int],
        copy_fitted_state: bool,
    ) -> SerializablePreprocessorBase:
        from ray.data.preprocessors.encoder import OrdinalEncoder
        from ray.data.preprocessors.gpu import (
            _DEFAULT_GPU_BATCH_SIZE,
            GPUChain,
            GPUOrdinalEncoder,
            GPUPowerTransformer,
            GPUSimpleImputer,
            GPUStandardScaler,
        )
        from ray.data.preprocessors.imputer import SimpleImputer
        from ray.data.preprocessors.scaler import StandardScaler
        from ray.data.preprocessors.transformer import PowerTransformer

        gpu_batch_size = batch_size or _DEFAULT_GPU_BATCH_SIZE
        gpu_preprocessors: List[SerializablePreprocessorBase] = []
        for preprocessor in self._preprocessors:
            if isinstance(preprocessor, PowerTransformer):
                gpu_preprocessor = GPUPowerTransformer(
                    columns=preprocessor.columns,
                    power=preprocessor.power,
                    method=preprocessor.method,
                    output_columns=preprocessor.output_columns,
                    batch_size=gpu_batch_size,
                    num_gpus_per_worker=num_gpus_per_worker,
                    concurrency=concurrency,
                )
            elif isinstance(preprocessor, StandardScaler):
                gpu_preprocessor = GPUStandardScaler(
                    columns=preprocessor.columns,
                    output_columns=preprocessor.output_columns,
                    batch_size=gpu_batch_size,
                    num_gpus_per_worker=num_gpus_per_worker,
                    concurrency=concurrency,
                )
            elif isinstance(preprocessor, SimpleImputer):
                gpu_preprocessor = GPUSimpleImputer(
                    columns=preprocessor.columns,
                    strategy=preprocessor.strategy,
                    fill_value=preprocessor.fill_value,
                    output_columns=preprocessor.output_columns,
                    batch_size=gpu_batch_size,
                    num_gpus_per_worker=num_gpus_per_worker,
                    concurrency=concurrency,
                )
            elif isinstance(preprocessor, OrdinalEncoder):
                gpu_preprocessor = GPUOrdinalEncoder(
                    columns=preprocessor.columns,
                    encode_lists=preprocessor.encode_lists,
                    output_columns=preprocessor.output_columns,
                    batch_size=gpu_batch_size,
                    num_gpus_per_worker=num_gpus_per_worker,
                    concurrency=concurrency,
                )
            else:
                raise TypeError(
                    "Chain accelerator='gpu' supports PowerTransformer, "
                    "StandardScaler, SimpleImputer, and OrdinalEncoder. "
                    f"Got {type(preprocessor).__name__}."
                )

            if (
                copy_fitted_state
                and preprocessor.fit_status() == Preprocessor.FitStatus.FITTED
                and gpu_preprocessor.fit_status() != Preprocessor.FitStatus.NOT_FITTABLE
            ):
                gpu_preprocessor.stats_ = dict(preprocessor.stats_)
                gpu_preprocessor._fitted = True
            gpu_preprocessors.append(gpu_preprocessor)

        return GPUChain(
            *gpu_preprocessors,
            batch_size=gpu_batch_size,
            num_gpus_per_worker=num_gpus_per_worker,
            concurrency=concurrency,
        )

    def _sync_fitted_state_from_gpu_chain(
        self, gpu_chain: SerializablePreprocessorBase
    ) -> None:
        for preprocessor, gpu_preprocessor in zip(
            self._preprocessors, gpu_chain.preprocessors
        ):
            if (
                gpu_preprocessor.fit_status() == Preprocessor.FitStatus.FITTED
                and preprocessor.fit_status() != Preprocessor.FitStatus.NOT_FITTABLE
            ):
                preprocessor.stats_ = dict(gpu_preprocessor.stats_)
                preprocessor._fitted = True

    def _transform(
        self,
        ds: "Dataset",
        batch_size: Optional[int],
        num_cpus: Optional[float] = None,
        memory: Optional[float] = None,
        concurrency: Optional[int] = None,
    ) -> "Dataset":
        for preprocessor in self._preprocessors:
            ds = preprocessor.transform(
                ds,
                batch_size=batch_size,
                num_cpus=num_cpus,
                memory=memory,
                concurrency=concurrency,
            )
        return ds

    def transform(
        self,
        ds: "Dataset",
        *,
        batch_size: Optional[int] = None,
        num_cpus: Optional[float] = None,
        memory: Optional[float] = None,
        concurrency: Optional[int] = None,
        accelerator: Optional[str] = None,
        num_gpus: Optional[int] = None,
        num_gpus_per_worker: float = 1,
    ) -> "Dataset":
        if not self._use_gpu_accelerator(accelerator):
            return super().transform(
                ds,
                batch_size=batch_size,
                num_cpus=num_cpus,
                memory=memory,
                concurrency=concurrency,
            )

        if num_cpus is not None:
            raise ValueError("GPU Chain preprocessing does not support num_cpus.")
        if memory is not None:
            raise ValueError("GPU Chain preprocessing does not support memory.")

        fit_status = self.fit_status()
        if fit_status in (
            Preprocessor.FitStatus.PARTIALLY_FITTED,
            Preprocessor.FitStatus.NOT_FITTED,
        ):
            raise PreprocessorNotFittedException(
                "`fit` must be called before `transform`, "
                "or simply use fit_transform() to run both steps"
            )

        gpu_chain = self._build_gpu_chain(
            batch_size=batch_size,
            num_gpus_per_worker=num_gpus_per_worker,
            concurrency=self._gpu_concurrency(
                num_gpus=num_gpus,
                concurrency=concurrency,
            ),
            copy_fitted_state=True,
        )
        return gpu_chain.transform(ds, batch_size=batch_size)

    def _transform_batch(self, df: "DataBatchType") -> "DataBatchType":
        for preprocessor in self._preprocessors:
            df = preprocessor.transform_batch(df)
        return df

    def __repr__(self):
        arguments = ", ".join(
            repr(preprocessor) for preprocessor in self._preprocessors
        )
        return f"{self.__class__.__name__}({arguments})"

    def _determine_transform_to_use(self) -> BatchFormat:
        # This is relevant for BatchPrediction.
        # For Chain preprocessor, we picked the first one as entry point.
        # TODO (jiaodong): We should revisit if our Chain preprocessor is
        # still optimal with context of lazy execution.
        return self._preprocessors[0]._determine_transform_to_use()

    def _get_serializable_fields(self) -> Dict[str, Any]:
        return {
            "preprocessors": self._preprocessors,
        }

    def _set_serializable_fields(self, fields: Dict[str, Any], version: int):
        # required fields
        self._preprocessors = fields["preprocessors"]
        self._gpu_chain = None

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Handle backwards compatibility for old pickled objects."""
        super().__setstate__(state)
        migrate_private_fields(
            self,
            fields={
                "_preprocessors": _PublicField(public_field="preprocessors"),
            },
        )
