"""
Tests for RDT (Ray Direct Transport) with cuDF DataFrames.

These tests require:
  - A CUDA-capable GPU
  - cuDF installed (``pip install cudf-cu12`` or similar)
  - NIXL installed

Run with::

    pytest python/ray/tests/gpu_objects/test_gpu_objects_cudf.py -sv
"""

import atexit
import os
import sys

import pytest

import ray

# Skip the entire module when cuDF is not installed.
cudf = pytest.importorskip("cudf")


# TODO: remove this if the UCX destructor crash is fixed
def _exit_before_ucx_destructor():
    """Skip shared-library destructors that crash on process exit.

    UCX (bundled inside nixl_cu12) registers a libnl-route-3 destructor
    (rtnl_tc_unregister) via __attribute__((destructor)) that segfaults during
    normal process teardown.  os._exit() terminates the process via the
    _exit(2) syscall, bypassing all C-level library destructors.

    Registering this handler first means it executes last (atexit is LIFO),
    so all other cleanup — including Ray shutdown — runs before we bail out.
    """
    os._exit(0)


atexit.register(_exit_before_ucx_destructor)


@ray.remote(num_gpus=1, num_cpus=0, enable_tensor_transport=True)
class CuDFProducer:
    @ray.method(tensor_transport="nixl")
    def echo(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """Return the DataFrame unchanged."""
        return df

    @ray.method(tensor_transport="nixl")
    def produce_two_col(self) -> cudf.DataFrame:
        """Return a DataFrame with an int and a float column."""
        return cudf.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})

    @ray.method(tensor_transport="nixl")
    def produce_multi_col(self) -> cudf.DataFrame:
        """Return a DataFrame with several column types."""
        return cudf.DataFrame(
            {
                "i32": cudf.Series([10, 20, 30], dtype="int32"),
                "i64": cudf.Series([100, 200, 300], dtype="int64"),
                "f32": cudf.Series([1.1, 2.2, 3.3], dtype="float32"),
                "f64": cudf.Series([10.1, 20.2, 30.3], dtype="float64"),
                "str": cudf.Series(["hello", "world", "ray"], dtype="object"),
            }
        )

    @ray.method(tensor_transport="nixl")
    def produce_with_nulls(self) -> cudf.DataFrame:
        """Return a DataFrame that has a column with null values."""
        return cudf.DataFrame(
            {
                "x": cudf.Series([1, None, 3], dtype="int32"),
                "y": cudf.Series([4.0, 5.0, None], dtype="float64"),
                "str": cudf.Series(["hello", None, "ray"], dtype="object"),
            }
        )


@ray.remote(num_gpus=1, num_cpus=0, enable_tensor_transport=True)
class CuDFConsumer:
    def col_sum(self, df: cudf.DataFrame, col: str) -> float:
        """Sum a single column; assert the DataFrame is on the GPU."""
        assert isinstance(
            df, cudf.DataFrame
        ), f"expected cudf.DataFrame, got {type(df)}"
        return float(df[col].sum())

    def sum_all(self, df: cudf.DataFrame) -> dict:
        """Return the column-wise sum for numeric columns and list of values for others."""
        assert isinstance(
            df, cudf.DataFrame
        ), f"expected cudf.DataFrame, got {type(df)}"
        result = {}
        for col in df.columns:
            if df[col].dtype == "object":
                result[col] = df[col].to_arrow().to_pylist()
            else:
                result[col] = float(df[col].sum())
        return result

    def null_count(self, df: cudf.DataFrame) -> dict:
        """Return null counts per column."""
        assert isinstance(df, cudf.DataFrame)
        return {col: int(df[col].isnull().sum()) for col in df.columns}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cudf_frames_equal(a: cudf.DataFrame, b: cudf.DataFrame) -> bool:
    """Return True when two cuDF DataFrames have the same columns, dtypes, and values."""
    if list(a.columns) != list(b.columns):
        return False
    for col in a.columns:
        if not (a[col] == b[col]).all():
            return False
    return True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 1}], indirect=True)
def test_cudf_df_round_trip_nixl(ray_start_regular):
    """A single actor echoes a cuDF DataFrame back through NIXL; values must be preserved."""
    actor = CuDFProducer.remote()
    original = cudf.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    ref = actor.echo.remote(original)
    result = ray.get(ref)
    assert isinstance(result, cudf.DataFrame)
    assert _cudf_frames_equal(original, result)


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 2}], indirect=True)
def test_cudf_df_multi_column_nixl(ray_start_regular):
    """A DataFrame with mixed int and float columns is transferred intact."""
    producer = CuDFProducer.remote()
    consumer = CuDFConsumer.remote()

    ref = producer.produce_multi_col.remote()
    result = ray.get(consumer.sum_all.remote(ref))

    assert result["i32"] == pytest.approx(60.0)
    assert result["i64"] == pytest.approx(600.0)
    assert result["f32"] == pytest.approx(6.6, abs=1e-4)
    assert result["f64"] == pytest.approx(60.6, abs=1e-6)
    assert result["str"] == ["hello", "world", "ray"]


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 2}], indirect=True)
def test_cudf_df_producer_consumer_nixl(ray_start_regular):
    """Producer returns a cuDF DataFrame; consumer receives it and computes column sums."""
    producer = CuDFProducer.remote()
    consumer = CuDFConsumer.remote()

    ref = producer.produce_two_col.remote()
    result = ray.get(consumer.sum_all.remote(ref))
    assert result == {"a": pytest.approx(6.0), "b": pytest.approx(15.0)}


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 2}], indirect=True)
def test_cudf_df_multiple_refs(ray_start_regular):
    """The same RDT ref can be consumed by multiple independent consumers."""
    producer = CuDFProducer.remote()
    # Use fractional GPU allocation so both consumers can share the second GPU,
    # allowing all 3 actors to be scheduled within a 2-GPU cluster.
    consumers = [CuDFConsumer.options(num_gpus=0.5).remote() for _ in range(2)]

    ref = producer.produce_two_col.remote()

    # Both consumers receive the same object ref.
    results = ray.get([c.sum_all.remote(ref) for c in consumers])
    for result in results:
        assert result["a"] == pytest.approx(6.0)
        assert result["b"] == pytest.approx(15.0)


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 2}], indirect=True)
def test_cudf_df_null_columns(ray_start_regular):
    """Columns containing null values round-trip correctly."""
    producer = CuDFProducer.remote()
    consumer = CuDFConsumer.remote()

    ref = producer.produce_with_nulls.remote()
    null_counts = ray.get(consumer.null_count.remote(ref))
    # Column "x": one null; column "y": one null; column "str": one null.
    assert null_counts["x"] == 1
    assert null_counts["y"] == 1
    assert null_counts["str"] == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-sv"]))
