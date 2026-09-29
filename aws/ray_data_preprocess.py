#!/usr/bin/env python3
"""Ray Data benchmark app for GPU preprocessing."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import ray
import ray.data
from ray.data.context import DataContext, ShuffleStrategy
from ray.data.preprocessor import SerializablePreprocessorBase
from ray.data.preprocessors import (
    Chain,
    GPUChain,
    GPUColumnCaster,
    GPUOrdinalEncoder,
    GPUSimpleImputer,
    GPUStandardScaler,
    OrdinalEncoder,
    SimpleImputer,
    StandardScaler,
)

TRAIN_OUTPUT_BLOCKS = 50
VAL_OUTPUT_BLOCKS = 4
DEFAULT_READ_MEMORY_BYTES = 12 * 1024**3
DEFAULT_CPU_TRANSFORM_MEMORY_BYTES = 4 * 1024**3
DEFAULT_TRANSFORM_BATCH_SIZE = 4096
DEFAULT_FIT_DOWNSAMPLE = 0.1
DEFAULT_FIT_DOWNSAMPLE_SEED = 12_345
DEFAULT_TRANSFORM_DOWNSAMPLE = 1.0
DEFAULT_TRANSFORM_DOWNSAMPLE_SEED = 12_345
DEFAULT_MIN_EVIDENCE = 750
DEFAULT_ISOLATE_READ_WORKERS = True
DEFAULT_ENABLE_DEFAULT_MAP_LOGICAL_MEMORY = True
CATEGORICAL_NULL_SENTINEL = "__ray_null__"
PREPROCESSOR_STATE_FILENAME = "preprocessor.bin"
PREPROCESSOR_METADATA_FILENAME = "metadata.json"
PARQUET_FILE_EXTENSIONS = ["parquet", "pq"]


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative integer, got {value!r}"
        )
    return parsed


def _fraction(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a float, got {value!r}") from exc
    if parsed <= 0 or parsed > 1:
        raise argparse.ArgumentTypeError(
            f"expected a value in the range (0, 1], got {value!r}"
        )
    return parsed


def _join_input_path(root: str, split: str) -> str:
    return f"{root.rstrip('/')}/{split}"


def _infer_columns(schema: Any) -> tuple[List[str], List[str]]:
    names = getattr(schema, "names", None)
    if names is None:
        names = list(schema)
    numeric_columns = [name for name in names if name.startswith("num_")]
    categorical_columns = [name for name in names if name.startswith("cat_")]
    if not numeric_columns:
        raise ValueError(
            "No numeric columns found in parquet schema; expected columns named num_*"
        )
    if not categorical_columns:
        raise ValueError(
            "No categorical columns found in parquet schema; expected columns named cat_*"
        )
    return numeric_columns, categorical_columns


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _list_parquet_files(path: str) -> tuple[List[str], Any]:
    from ray.data._internal.util import RetryingPyFileSystem
    from ray.data.datasource.file_meta_provider import _list_files
    from ray.data.datasource.path_util import _resolve_paths_and_filesystem

    paths, filesystem = _resolve_paths_and_filesystem(path)
    filesystem = RetryingPyFileSystem.wrap(
        filesystem,
        retryable_errors=DataContext.get_current().retried_io_errors,
    )
    listed_files = _list_files(
        paths,
        filesystem,
        partition_filter=None,
        file_extensions=PARQUET_FILE_EXTENSIONS,
    )
    files = sorted(file_path for file_path, _ in listed_files)
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")
    return files, filesystem


def _select_files(
    files: Sequence[str],
    *,
    downsample: float,
    seed: int,
) -> List[str]:
    sample_size = max(1, int(math.ceil(len(files) * downsample)))
    if sample_size >= len(files):
        return list(files)
    rng = random.Random(seed)
    return sorted(rng.sample(list(files), sample_size))


def _read_parquet_dataset(
    path: Any,
    columns: Optional[Sequence[str]],
    override_num_blocks: Optional[int],
    read_concurrency: Optional[int],
    filesystem: Optional[Any] = None,
) -> "ray.data.Dataset":
    read_kwargs: Dict[str, Any] = {"memory": DEFAULT_READ_MEMORY_BYTES}
    if filesystem is not None:
        read_kwargs["filesystem"] = filesystem
    if columns is not None:
        read_kwargs["columns"] = list(columns)
    if override_num_blocks is not None:
        read_kwargs["override_num_blocks"] = override_num_blocks
    if read_concurrency is not None:
        read_kwargs["concurrency"] = read_concurrency
    return ray.data.read_parquet(path, **read_kwargs)


def _configure_data_context(*, num_gpus: int) -> None:
    data_context = DataContext.get_current()
    data_context.isolate_read_workers = DEFAULT_ISOLATE_READ_WORKERS
    data_context.default_map_logical_memory_enabled = (
        DEFAULT_ENABLE_DEFAULT_MAP_LOGICAL_MEMORY
    )
    if num_gpus > 0:
        data_context.shuffle_strategy = ShuffleStrategy.GPU_SHUFFLE


def _prepare_categorical_batch(
    df: pd.DataFrame,
    *,
    categorical_columns: Sequence[str],
    null_sentinel: str,
) -> pd.DataFrame:
    for column in categorical_columns:
        df[column] = df[column].astype("string").fillna(null_sentinel).astype("object")
    return df


def _finalize_encoded_batch(
    df: pd.DataFrame,
    *,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> pd.DataFrame:
    for column in numeric_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        df[column] = np.nan_to_num(values.to_numpy(dtype=np.float32), nan=0.0)

    for column in categorical_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        # OrdinalEncoder emits 0..N-1 for known categories and NaN/null for OOV.
        # Shift known values by one so 0 is reserved for OOV, matching the lineage.
        shifted = values + 1
        df[column] = shifted.where(values.notna(), 0).astype(np.int32)
    return df


def _prepare_for_encoding(
    ds: "ray.data.Dataset",
    *,
    categorical_columns: Sequence[str],
    batch_size: int,
) -> "ray.data.Dataset":
    return ds.map_batches(
        _prepare_categorical_batch,
        batch_format="pandas",
        batch_size=batch_size,
        memory=DEFAULT_CPU_TRANSFORM_MEMORY_BYTES,
        fn_kwargs={
            "categorical_columns": list(categorical_columns),
            "null_sentinel": CATEGORICAL_NULL_SENTINEL,
        },
    )


def _finalize_encoded_dataset(
    ds: "ray.data.Dataset",
    *,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    batch_size: int,
) -> "ray.data.Dataset":
    return ds.map_batches(
        _finalize_encoded_batch,
        batch_format="pandas",
        batch_size=batch_size,
        memory=DEFAULT_CPU_TRANSFORM_MEMORY_BYTES,
        fn_kwargs={
            "numeric_columns": list(numeric_columns),
            "categorical_columns": list(categorical_columns),
        },
    )


def _import_cudf() -> Any:
    try:
        import cudf
    except ImportError as exc:
        raise ImportError(
            "GPU mode requires cuDF. Use an environment with RAPIDS installed."
        ) from exc
    return cudf


def _import_cupy() -> Any:
    try:
        import cupy as cp
    except ImportError as exc:
        raise ImportError(
            "GPU mode requires CuPy. Use an environment with RAPIDS installed."
        ) from exc
    return cp


def _check_gpu_dependencies() -> Dict[str, str]:
    cudf = _import_cudf()
    cp = _import_cupy()
    return {
        "cudf_version": getattr(cudf, "__version__", "unknown"),
        "cupy_version": getattr(cp, "__version__", "unknown"),
    }


def _build_preprocessor(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    *,
    min_evidence: int,
) -> Chain:
    return Chain(
        #        PowerTransformer(
        #            columns=list(numeric_columns),
        #            power=0,
        #            method="yeo-johnson",
        #        ),
        StandardScaler(columns=list(numeric_columns)),
        SimpleImputer(
            columns=list(numeric_columns),
            strategy="constant",
            fill_value=0.0,
        ),
        OrdinalEncoder(
            columns=list(categorical_columns),
            encode_lists=False,
            min_evidence=min_evidence,
        ),
    )


def _build_gpu_preprocessor(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    *,
    batch_size: int,
    concurrency: int,
    min_evidence: int,
) -> GPUChain:
    gpu_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "num_gpus_per_worker": 1,
        "concurrency": concurrency,
    }
    return GPUChain(
        #        GPUPowerTransformer(
        #            columns=list(numeric_columns),
        #            power=0,
        #            method="yeo-johnson",
        #            **gpu_kwargs,
        #        ),
        GPUStandardScaler(
            columns=list(numeric_columns),
            output_dtype="float32",
            **gpu_kwargs,
        ),
        GPUSimpleImputer(
            columns=list(numeric_columns),
            strategy="constant",
            fill_value=0.0,
            output_dtype="float32",
            **gpu_kwargs,
        ),
        GPUColumnCaster(
            columns=list(categorical_columns),
            output_dtype="str",
            **gpu_kwargs,
        ),
        GPUSimpleImputer(
            columns=list(categorical_columns),
            strategy="constant",
            fill_value=CATEGORICAL_NULL_SENTINEL,
            output_dtype="str",
            **gpu_kwargs,
        ),
        GPUOrdinalEncoder(
            columns=list(categorical_columns),
            unknown_value=0,
            encoded_missing_value=0,
            output_dtype="int32",
            encoded_value_offset=1,
            min_evidence=min_evidence,
            **gpu_kwargs,
        ),
        batch_size=batch_size,
        num_gpus_per_worker=1,
        concurrency=concurrency,
    )


def _save_preprocessor(
    preprocessor: SerializablePreprocessorBase,
    preprocessor_dir: Path,
    metadata: Dict[str, Any],
) -> None:
    _remove_path(preprocessor_dir)
    preprocessor_dir.mkdir(parents=True, exist_ok=True)

    state = preprocessor.serialize()
    state_path = preprocessor_dir / PREPROCESSOR_STATE_FILENAME
    if isinstance(state, str):
        state_path.write_text(state, encoding="utf-8")
    else:
        state_path.write_bytes(state)

    metadata_path = preprocessor_dir / PREPROCESSOR_METADATA_FILENAME
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_preprocessor(preprocessor_dir: Path) -> SerializablePreprocessorBase:
    state = (preprocessor_dir / PREPROCESSOR_STATE_FILENAME).read_bytes()
    return SerializablePreprocessorBase.deserialize(state)


def _load_preprocessor_metadata(preprocessor_dir: Path) -> Dict[str, Any]:
    metadata_path = preprocessor_dir / PREPROCESSOR_METADATA_FILENAME
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _load_gpu_preprocessor(preprocessor_dir: Path) -> GPUChain:
    preprocessor = _load_preprocessor(preprocessor_dir)
    if not isinstance(preprocessor, GPUChain):
        raise TypeError(f"Expected saved GPUChain, found {type(preprocessor).__name__}")
    return preprocessor


def _categorical_map_sizes(preprocessor: GPUChain) -> Dict[str, int]:
    for child in preprocessor.preprocessors:
        if isinstance(child, GPUOrdinalEncoder):
            return {
                column: len(child.stats_.get(f"unique_values({column})", {}))
                for column in child.columns
            }
    return {}


def _write_encoded_split(
    *,
    split: str,
    input_path: str,
    input_files: Sequence[str],
    filesystem: Any,
    output_dir: Path,
    preprocessor: SerializablePreprocessorBase,
    columns: Sequence[str],
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    override_num_blocks: Optional[int],
    read_concurrency: Optional[int],
    batch_size: int,
    output_blocks: int,
) -> Dict[str, Any]:
    split_start = time.perf_counter()
    split_output = output_dir / split
    _remove_path(split_output)

    ds = _read_parquet_dataset(
        input_files,
        columns,
        override_num_blocks,
        read_concurrency,
        filesystem,
    )
    ds = _prepare_for_encoding(
        ds,
        categorical_columns=categorical_columns,
        batch_size=batch_size,
    )
    transformed = preprocessor.transform(
        ds,
        batch_size=batch_size,
        memory=DEFAULT_CPU_TRANSFORM_MEMORY_BYTES,
    )
    transformed = _finalize_encoded_dataset(
        transformed,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        batch_size=batch_size,
    )
    # transformed = transformed.repartition(output_blocks)
    # transformed = transformed.repartition(1, shuffle=False)
    transformed.write_parquet(os.fspath(split_output), mode="overwrite")

    return {
        "input_path": input_path,
        "input_files": len(input_files),
        "output_path": os.fspath(split_output),
        "output_blocks": output_blocks,
        "repartitioned": True,
        "elapsed_s": time.perf_counter() - split_start,
    }


def _write_gpu_encoded_split(
    *,
    split: str,
    input_path: str,
    input_files: Sequence[str],
    filesystem: Any,
    output_dir: Path,
    preprocessor: GPUChain,
    columns: Sequence[str],
    override_num_blocks: Optional[int],
    read_concurrency: Optional[int],
    batch_size: int,
    concurrency: int,
    output_blocks: int,
) -> Dict[str, Any]:
    split_start = time.perf_counter()
    split_output = output_dir / split
    _remove_path(split_output)

    ds = _read_parquet_dataset(
        input_files,
        columns,
        override_num_blocks,
        read_concurrency,
        filesystem,
    )
    transformed = preprocessor.transform(
        ds,
        batch_size=batch_size,
        concurrency=concurrency,
        output_batch_format="pyarrow",
    )
    # transformed = transformed.repartition(1, shuffle=False)
    # transformed.write_parquet(os.fspath(split_output), mode="overwrite", min_rows_per_file=22000000)
    transformed.write_parquet(os.fspath(split_output), mode="overwrite")

    return {
        "input_path": input_path,
        "input_files": len(input_files),
        "output_path": os.fspath(split_output),
        "output_blocks": None,
        "requested_output_blocks": output_blocks,
        "repartitioned": False,
        "gpu_output_batch_format": "pyarrow",
        "elapsed_s": time.perf_counter() - split_start,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run Ray Data preprocessing workflow using preprocessor APIs.")
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Input dataset root containing train/ and optionally val/ parquet files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output root for encoded train/ and optional val/ parquet files.",
    )
    parser.add_argument(
        "--preprocessor-dir",
        type=Path,
        default=None,
        help="Directory for serialized fitted preprocessor state.",
    )
    parser.add_argument(
        "--val",
        action="store_true",
        help="Also encode the validation split from <input>/val.",
    )
    parser.add_argument(
        "--num-gpus",
        type=_nonnegative_int,
        default=0,
        help="Use the GPU code path with this many GPUs. Defaults to 0.",
    )
    parser.add_argument(
        "--fit-downsample",
        type=_fraction,
        default=DEFAULT_FIT_DOWNSAMPLE,
        help=(
            "Fraction of train parquet files used for fitting. "
            f"Defaults to {DEFAULT_FIT_DOWNSAMPLE}."
        ),
    )
    parser.add_argument(
        "--transform-downsample",
        type=_fraction,
        default=DEFAULT_TRANSFORM_DOWNSAMPLE,
        help=(
            "Fraction of parquet files in each split used for transforming. "
            f"Defaults to {DEFAULT_TRANSFORM_DOWNSAMPLE}."
        ),
    )
    parser.add_argument(
        "--min-evidence",
        type=_positive_int,
        default=DEFAULT_MIN_EVIDENCE,
        help=(
            "Minimum global fit-sample count required to retain a categorical "
            f"value. Defaults to {DEFAULT_MIN_EVIDENCE}."
        ),
    )
    parser.add_argument(
        "--override-num-blocks",
        type=_positive_int,
        default=None,
        help="Optional Ray Data parquet read block override.",
    )
    parser.add_argument(
        "--read-concurrency",
        type=_positive_int,
        default=None,
        help=(
            "Optional maximum number of concurrent parquet read tasks. "
            "This does not change the number of output blocks."
        ),
    )
    parser.add_argument(
        "--transform-batch-size",
        type=_positive_int,
        default=DEFAULT_TRANSFORM_BATCH_SIZE,
        help=f"Batch size for transform map tasks. Defaults to {DEFAULT_TRANSFORM_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--gpu-fit-batch-rows",
        type=_positive_int,
        default=None,
        help="Rows per GPU fit batch. Defaults to --transform-batch-size.",
    )
    parser.add_argument(
        "--gpu-transform-batch-rows",
        type=_positive_int,
        default=None,
        help="Rows per GPU transform batch. Defaults to --transform-batch-size.",
    )
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument(
        "--fit-only",
        action="store_true",
        help="Fit and save the preprocessor without transforming any splits.",
    )
    stage_group.add_argument(
        "--transform-only",
        action="store_true",
        help="Load a saved preprocessor and transform splits without fitting.",
    )
    parser.add_argument(
        "--materialize-fit-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Materialize the prepared fit dataset before fitting.",
    )
    args = parser.parse_args()
    if args.gpu_fit_batch_rows is None:
        args.gpu_fit_batch_rows = args.transform_batch_size
    if args.gpu_transform_batch_rows is None:
        args.gpu_transform_batch_rows = args.transform_batch_size
    return args


def run(args: argparse.Namespace) -> int:
    input_root = os.path.expanduser(args.input).rstrip("/")
    output_root = args.output.expanduser().resolve()
    preprocessor_dir = (
        args.preprocessor_dir.expanduser().resolve()
        if args.preprocessor_dir is not None
        else output_root / "preprocessor"
    )

    if input_root.endswith(".parquet"):
        # pass thru file URLs as-is
        train_input = input_root
    else:
        # otherwise, assume root directory containing train/val dirs
        train_input = _join_input_path(input_root, "train")

    val_input = _join_input_path(input_root, "val")
    train_files, train_filesystem = _list_parquet_files(train_input)
    fit_files: Optional[List[str]] = None
    if not args.transform_only:
        fit_files = _select_files(
            train_files,
            downsample=args.fit_downsample,
            seed=DEFAULT_FIT_DOWNSAMPLE_SEED,
        )

    train_transform_files: Optional[List[str]] = None
    val_files: Optional[List[str]] = None
    val_filesystem: Optional[Any] = None
    val_transform_files: Optional[List[str]] = None
    if not args.fit_only:
        train_transform_files = _select_files(
            train_files,
            downsample=args.transform_downsample,
            seed=DEFAULT_TRANSFORM_DOWNSAMPLE_SEED,
        )
        if args.val:
            val_files, val_filesystem = _list_parquet_files(val_input)
            val_transform_files = _select_files(
                val_files,
                downsample=args.transform_downsample,
                seed=DEFAULT_TRANSFORM_DOWNSAMPLE_SEED,
            )

    if not args.fit_only:
        _remove_path(output_root / "val")
    output_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "input": os.fspath(input_root),
        "output": os.fspath(output_root),
        "preprocessor_dir": os.fspath(preprocessor_dir),
        "stage": (
            "fit"
            if args.fit_only
            else "transform"
            if args.transform_only
            else "fit_transform"
        ),
        "execution_path": "ray_data_cudf_gpu" if args.num_gpus > 0 else "ray_data_cpu",
        "num_gpus": args.num_gpus,
        "fit_downsample": args.fit_downsample,
        "fit_downsample_seed": DEFAULT_FIT_DOWNSAMPLE_SEED,
        "transform_downsample": args.transform_downsample,
        "transform_downsample_seed": DEFAULT_TRANSFORM_DOWNSAMPLE_SEED,
        "min_evidence": args.min_evidence,
        "override_num_blocks": args.override_num_blocks,
        "read_concurrency": args.read_concurrency,
        "read_memory_bytes": DEFAULT_READ_MEMORY_BYTES,
        "cpu_transform_memory_bytes": (
            DEFAULT_CPU_TRANSFORM_MEMORY_BYTES if args.num_gpus == 0 else None
        ),
        "transform_batch_size": args.transform_batch_size,
        "isolate_read_workers": DEFAULT_ISOLATE_READ_WORKERS,
        "default_map_logical_memory_enabled": (
            DEFAULT_ENABLE_DEFAULT_MAP_LOGICAL_MEMORY
        ),
        "gpu_fit_batch_rows": args.gpu_fit_batch_rows if args.num_gpus > 0 else None,
        "gpu_transform_batch_rows": (
            args.gpu_transform_batch_rows if args.num_gpus > 0 else None
        ),
        "gpu_concurrency": args.num_gpus if args.num_gpus > 0 else None,
        "gpu_output_batch_format": "pyarrow" if args.num_gpus > 0 else None,
        "materialize_fit_data": args.materialize_fit_data,
        "train_input": train_input,
        "val_input": val_input if args.val else None,
        "train_input_files": len(train_files),
        "fit_input": fit_files,
        "fit_input_files": len(fit_files) if fit_files is not None else None,
        "train_transform_input_files": (
            len(train_transform_files) if train_transform_files is not None else None
        ),
        "val_input_files": len(val_files) if val_files is not None else None,
        "val_transform_input_files": (
            len(val_transform_files) if val_transform_files is not None else None
        ),
        "categorical_null_sentinel": CATEGORICAL_NULL_SENTINEL,
    }

    start = time.perf_counter()
    try:
        if args.num_gpus > 0:
            summary.update(_check_gpu_dependencies())

        #        ray.init(
        #            num_gpus=args.num_gpus,
        #            _temp_dir=os.environ.get("RAY_TMP_DIR", "/tmp/ray"),
        #        )
        ray.init()
        _configure_data_context(num_gpus=args.num_gpus)

        if args.transform_only:
            metadata = _load_preprocessor_metadata(preprocessor_dir)
            numeric_columns = metadata["numeric_columns"]
            categorical_columns = metadata["categorical_columns"]
        else:
            fit_read_start = time.perf_counter()
            fit_ds = _read_parquet_dataset(
                fit_files,
                None,
                args.override_num_blocks,
                args.read_concurrency,
                train_filesystem,
            )
            numeric_columns, categorical_columns = _infer_columns(fit_ds.schema())
            fit_ds = fit_ds.select_columns(numeric_columns + categorical_columns)
            if args.num_gpus == 0:
                fit_ds = _prepare_for_encoding(
                    fit_ds,
                    categorical_columns=categorical_columns,
                    batch_size=args.transform_batch_size,
                )
            summary["fit_read_prepare_elapsed_s"] = time.perf_counter() - fit_read_start
        all_columns = numeric_columns + categorical_columns
        summary.update(
            {
                "numeric_columns": len(numeric_columns),
                "categorical_columns": len(categorical_columns),
                "total_columns": len(all_columns),
            }
        )

        splits: Dict[str, Any] = {}
        if args.num_gpus > 0:
            if not args.transform_only:
                if args.materialize_fit_data:
                    materialize_start = time.perf_counter()
                    fit_ds = fit_ds.materialize()
                    summary["fit_materialize_elapsed_s"] = (
                        time.perf_counter() - materialize_start
                    )

                preprocessor = _build_gpu_preprocessor(
                    numeric_columns,
                    categorical_columns,
                    batch_size=args.gpu_fit_batch_rows,
                    concurrency=args.num_gpus,
                    min_evidence=args.min_evidence,
                )
                fit_start = time.perf_counter()
                preprocessor.fit(fit_ds)
                summary["fit_elapsed_s"] = time.perf_counter() - fit_start

                save_start = time.perf_counter()
                metadata = {
                    "numeric_columns": numeric_columns,
                    "categorical_columns": categorical_columns,
                    "categorical_map_sizes": _categorical_map_sizes(preprocessor),
                    "fit_input": fit_files,
                    "config": summary,
                    "preprocessors": [
                        type(child).__name__ for child in preprocessor.preprocessors
                    ],
                }
                _save_preprocessor(preprocessor, preprocessor_dir, metadata)
                summary["save_preprocessor_elapsed_s"] = (
                    time.perf_counter() - save_start
                )

            if not args.fit_only:
                load_start = time.perf_counter()
                fitted_preprocessor = _load_gpu_preprocessor(preprocessor_dir)
                summary["load_preprocessor_elapsed_s"] = (
                    time.perf_counter() - load_start
                )
                splits["train"] = _write_gpu_encoded_split(
                    split="train",
                    input_path=train_input,
                    input_files=train_transform_files,
                    filesystem=train_filesystem,
                    output_dir=output_root,
                    preprocessor=fitted_preprocessor,
                    columns=all_columns,
                    override_num_blocks=args.override_num_blocks,
                    read_concurrency=args.read_concurrency,
                    batch_size=args.gpu_transform_batch_rows,
                    concurrency=args.num_gpus,
                    output_blocks=TRAIN_OUTPUT_BLOCKS,
                )
            if not args.fit_only and args.val:
                splits["val"] = _write_gpu_encoded_split(
                    split="val",
                    input_path=val_input,
                    input_files=val_transform_files,
                    filesystem=val_filesystem,
                    output_dir=output_root,
                    preprocessor=fitted_preprocessor,
                    columns=all_columns,
                    override_num_blocks=args.override_num_blocks,
                    read_concurrency=args.read_concurrency,
                    batch_size=args.gpu_transform_batch_rows,
                    concurrency=args.num_gpus,
                    output_blocks=VAL_OUTPUT_BLOCKS,
                )
        else:
            if not args.transform_only:
                if args.materialize_fit_data:
                    materialize_start = time.perf_counter()
                    fit_ds = fit_ds.materialize()
                    summary["fit_materialize_elapsed_s"] = (
                        time.perf_counter() - materialize_start
                    )

                preprocessor = _build_preprocessor(
                    numeric_columns,
                    categorical_columns,
                    min_evidence=args.min_evidence,
                )
                fit_start = time.perf_counter()
                preprocessor.fit(fit_ds)
                summary["fit_elapsed_s"] = time.perf_counter() - fit_start

                save_start = time.perf_counter()
                metadata = {
                    "numeric_columns": numeric_columns,
                    "categorical_columns": categorical_columns,
                    "fit_input": fit_files,
                    "config": summary,
                    "preprocessors": [
                        type(child).__name__ for child in preprocessor.preprocessors
                    ],
                }
                _save_preprocessor(preprocessor, preprocessor_dir, metadata)
                summary["save_preprocessor_elapsed_s"] = (
                    time.perf_counter() - save_start
                )

            if not args.fit_only:
                load_start = time.perf_counter()
                fitted_preprocessor = _load_preprocessor(preprocessor_dir)
                summary["load_preprocessor_elapsed_s"] = (
                    time.perf_counter() - load_start
                )
                splits["train"] = _write_encoded_split(
                    split="train",
                    input_path=train_input,
                    input_files=train_transform_files,
                    filesystem=train_filesystem,
                    output_dir=output_root,
                    preprocessor=fitted_preprocessor,
                    columns=all_columns,
                    numeric_columns=numeric_columns,
                    categorical_columns=categorical_columns,
                    override_num_blocks=args.override_num_blocks,
                    read_concurrency=args.read_concurrency,
                    batch_size=args.transform_batch_size,
                    output_blocks=TRAIN_OUTPUT_BLOCKS,
                )
            if not args.fit_only and args.val:
                splits["val"] = _write_encoded_split(
                    split="val",
                    input_path=val_input,
                    input_files=val_transform_files,
                    filesystem=val_filesystem,
                    output_dir=output_root,
                    preprocessor=fitted_preprocessor,
                    columns=all_columns,
                    numeric_columns=numeric_columns,
                    categorical_columns=categorical_columns,
                    override_num_blocks=args.override_num_blocks,
                    read_concurrency=args.read_concurrency,
                    batch_size=args.transform_batch_size,
                    output_blocks=VAL_OUTPUT_BLOCKS,
                )

        summary["splits"] = splits
        summary["status"] = "OK"
        summary["elapsed_s"] = time.perf_counter() - start
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - keep benchmark failures readable.
        summary["status"] = "FAILED"
        summary["elapsed_s"] = time.perf_counter() - start
        summary["error"] = repr(exc)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1
    finally:
        if ray.is_initialized():
            ray.shutdown()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
