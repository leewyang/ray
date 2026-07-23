from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pyarrow as pa

from ray.data._internal.table_block import TableBlockAccessor, TableBlockBuilder
from ray.data.block import (
    Block,
    BlockAccessor,
    BlockColumn,
    BlockColumnAccessor,
    BlockExecStats,
    BlockMetadataWithSchema,
    BlockType,
    U,
)
from ray.data.expressions import Expr

if TYPE_CHECKING:
    import cudf
    import pandas
    import pyarrow

    from ray.data._internal.planner.exchange.sort_task_spec import SortKey


def _import_cudf():
    try:
        import cudf
    except ImportError as exc:
        raise ImportError(
            "cuDF blocks require cuDF. Install a RAPIDS cuDF package compatible "
            "with your CUDA runtime."
        ) from exc
    return cudf


def _as_py(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return value.as_py()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return value


class CudfRow(Mapping):
    """Row of a tabular Dataset backed by a cuDF DataFrame block."""

    def __init__(self, row: "cudf.DataFrame"):
        self._row = row

    def __getitem__(self, key: Union[str, List[str]]) -> Any:
        if isinstance(key, str):
            return _as_py(self._row[key].iloc[0])
        return tuple(_as_py(self._row[column].iloc[0]) for column in key)

    def __iter__(self) -> Iterator:
        for key in self._row.columns:
            yield key

    def __len__(self):
        return len(self._row.columns)

    def as_pydict(self) -> Dict[str, Any]:
        if len(self._row) == 0:
            return {}
        return self._row.head(1).to_pandas().iloc[0].to_dict()


class CudfBlockBuilder(TableBlockBuilder):
    def __init__(self):
        cudf = _import_cudf()
        super().__init__(cudf.DataFrame)

    @staticmethod
    def _table_from_pydict(columns: Dict[str, List[Any]]) -> "cudf.DataFrame":
        cudf = _import_cudf()
        return cudf.DataFrame(columns)

    @staticmethod
    def _combine_tables(tables: List["cudf.DataFrame"]) -> "cudf.DataFrame":
        cudf = _import_cudf()
        if len(tables) > 1:
            return cudf.concat(tables, ignore_index=True)
        return tables[0]

    @staticmethod
    def _concat_would_copy() -> bool:
        return True

    @staticmethod
    def _empty_table() -> "cudf.DataFrame":
        cudf = _import_cudf()
        return cudf.DataFrame()

    def block_type(self) -> BlockType:
        return BlockType.CUDF


class CudfBlockAccessor(TableBlockAccessor):
    ROW_TYPE = CudfRow

    def __init__(self, table: "cudf.DataFrame"):
        super().__init__(table)

    def _get_row(self, index: int) -> CudfRow:
        return CudfRow(self.slice(index, index + 1, copy=False))

    def column_names(self) -> List[str]:
        return list(self._table.columns)

    def fill_column(self, name: str, value: Any) -> Block:
        return self.upsert_column(name, value)

    def slice(self, start: int, end: int, copy: bool = False) -> "cudf.DataFrame":
        view = self._table.iloc[start:end]
        if copy or start != 0 or end != len(self._table):
            # cuDF string/list slices can retain parent buffers. Ray Data often
            # slices read outputs into smaller blocks before putting them in the
            # object store, so compact sub-slices to avoid keeping the full
            # source block alive behind each small slice.
            view = view.copy(deep=True)
        return view

    def take(self, indices: List[int]) -> "cudf.DataFrame":
        return self._table.iloc[indices].reset_index(drop=True)

    def drop(self, columns: List[str]) -> Block:
        return self._table.drop(columns=columns)

    def select(self, columns: List[str]) -> "cudf.DataFrame":
        if not all(isinstance(col, str) for col in columns):
            raise ValueError(
                "Columns must be a list of column name strings when operating on "
                f"cuDF blocks, but got: {columns}."
            )
        if len(columns) == 0:
            return self._table.iloc[:, 0:0]
        return self._table[columns]

    def rename_columns(self, columns_rename: Dict[str, str]) -> "cudf.DataFrame":
        return self._table.rename(columns=columns_rename)

    def upsert_column(
        self, column_name: str, column_data: BlockColumn
    ) -> "cudf.DataFrame":
        cudf = _import_cudf()
        df = self._table.copy(deep=False)
        if isinstance(column_data, (pa.Array, pa.ChunkedArray)):
            column_data = cudf.Series.from_arrow(column_data)
        df[column_name] = column_data
        return df

    def random_shuffle(self, random_seed: Optional[int]) -> "cudf.DataFrame":
        return self._table.sample(frac=1, random_state=random_seed).reset_index(
            drop=True
        )

    def schema(self) -> "pyarrow.lib.Schema":
        return self._table.head(0).to_arrow(preserve_index=False).schema

    def to_default(self) -> Block:
        return self._table

    def to_cudf(self) -> "cudf.DataFrame":
        return self._table

    def to_pandas(self) -> "pandas.DataFrame":
        return self._table.to_pandas()

    def to_numpy(
        self, columns: Optional[Union[str, List[str]]] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if columns is None:
            columns = self.column_names()
            should_be_single_ndarray = False
        elif isinstance(columns, list):
            should_be_single_ndarray = False
        else:
            columns = [columns]
            should_be_single_ndarray = True

        arrays = [self._table[column].to_numpy() for column in columns]
        if should_be_single_ndarray:
            return arrays[0]
        return dict(zip(columns, arrays))

    def to_arrow(self) -> "pyarrow.Table":
        return self._table.to_arrow(preserve_index=False)

    def num_rows(self) -> int:
        return len(self._table)

    def size_bytes(self) -> int:
        usage = self._table.memory_usage(deep=True)
        total = usage.sum() if hasattr(usage, "sum") else sum(usage)
        return int(_as_py(total))

    def _zip(self, acc: BlockAccessor) -> "cudf.DataFrame":
        r = self._table.copy(deep=False)
        s = acc.to_cudf()
        for col_name in s.columns:
            output_name = col_name
            if output_name in r.columns:
                i = 1
                while output_name in r.columns:
                    output_name = f"{col_name}_{i}"
                    i += 1
            r[output_name] = s[col_name]
        return r

    @staticmethod
    def builder() -> CudfBlockBuilder:
        return CudfBlockBuilder()

    @staticmethod
    def _empty_table() -> "cudf.DataFrame":
        return CudfBlockBuilder._empty_table()

    def _sample(self, n_samples: int, sort_key: "SortKey") -> "cudf.DataFrame":
        return self._table[sort_key.get_columns()].sample(n=n_samples)

    def sort(self, sort_key: "SortKey"):
        assert (
            sort_key.get_columns()
        ), f"Sorting columns couldn't be empty (got {sort_key.get_columns()})"
        if len(self._table) == 0:
            return self._empty_table()
        columns, ascending = sort_key.to_pandas_sort_args()
        return self._table.sort_values(by=columns, ascending=ascending)

    def sort_and_partition(
        self, boundaries: List[U], sort_key: "SortKey"
    ) -> List[Block]:
        return BlockAccessor.for_block(self.to_arrow()).sort_and_partition(
            boundaries, sort_key
        )

    @staticmethod
    def merge_sorted_blocks(
        blocks: List[Block], sort_key: "SortKey"
    ) -> Tuple[Block, BlockMetadataWithSchema]:
        stats = BlockExecStats.builder()
        if not blocks:
            ret = CudfBlockAccessor._empty_table()
            return ret, BlockMetadataWithSchema.from_block(
                ret, block_exec_stats=stats.build()
            )
        arrow_blocks = [BlockAccessor.for_block(block).to_arrow() for block in blocks]
        ret, _ = BlockAccessor.for_block(arrow_blocks[0]).merge_sorted_blocks(
            arrow_blocks, sort_key
        )
        return ret, BlockMetadataWithSchema.from_block(
            ret, block_exec_stats=stats.build()
        )

    def block_type(self) -> BlockType:
        return BlockType.CUDF

    def iter_rows(
        self, public_row_format: bool
    ) -> Iterator[Union[Mapping, np.ndarray]]:
        if public_row_format:
            yield from self._table.to_pandas().to_dict("records")
            return
        for i in range(self.num_rows()):
            yield self._get_row(i)

    def filter(self, predicate_expr: Expr) -> "cudf.DataFrame":
        if len(self._table) == 0:
            return self._table

        from ray.data._internal.planner.plan_expression.expression_evaluator import (
            eval_expr,
        )

        mask = eval_expr(predicate_expr, self._table)
        return self._table[mask]

    def hstack(self, other_block: Block) -> Block:
        result = self._table.copy(deep=False)
        other = BlockAccessor.for_block(other_block).to_cudf()
        for name in other.columns:
            result[name] = other[name]
        return result


class CudfBlockColumnAccessor(BlockColumnAccessor):
    def __init__(self, col: "cudf.Series"):
        super().__init__(col)

    def count(self, *, ignore_nulls: bool, as_py: bool = True) -> Optional[U]:
        result = self._column.count() if ignore_nulls else len(self._column)
        return _as_py(result) if as_py else result

    def sum(self, *, ignore_nulls: bool, as_py: bool = True) -> Optional[U]:
        result = self._column.sum(skipna=ignore_nulls)
        return _as_py(result) if as_py else result

    def min(self, *, ignore_nulls: bool, as_py: bool = True) -> Optional[U]:
        result = self._column.min(skipna=ignore_nulls)
        return _as_py(result) if as_py else result

    def max(self, *, ignore_nulls: bool, as_py: bool = True) -> Optional[U]:
        result = self._column.max(skipna=ignore_nulls)
        return _as_py(result) if as_py else result

    def mean(self, *, ignore_nulls: bool, as_py: bool = True) -> Optional[U]:
        result = self._column.mean(skipna=ignore_nulls)
        return _as_py(result) if as_py else result

    def quantile(
        self, *, q: float, ignore_nulls: bool, as_py: bool = True
    ) -> Optional[U]:
        result = self._column.quantile(q=q, interpolation="linear")
        return _as_py(result) if as_py else result

    def unique(self) -> BlockColumn:
        return self._column.unique()

    def value_counts(self) -> Optional[Dict[str, List]]:
        value_counts = self._column.value_counts()
        if len(value_counts) == 0:
            return None
        return {
            "values": value_counts.index.to_pandas().tolist(),
            "counts": value_counts.to_pandas().tolist(),
        }

    def hash(self) -> BlockColumn:
        if hasattr(self._column, "hash_values"):
            try:
                return self._column.hash_values(method="murmur3")
            except TypeError:
                return self._column.hash_values()
        return BlockColumnAccessor.for_column(self._column.to_pandas()).hash()

    def flatten(self) -> BlockColumn:
        leaves = getattr(self._column.list, "leaves", None)
        if leaves is None:
            raise TypeError("cuDF list column does not expose list.leaves.")
        return leaves() if callable(leaves) else leaves

    def dropna(self) -> BlockColumn:
        return self._column.dropna()

    def is_composed_of_lists(self) -> bool:
        return hasattr(self._column.dtype, "element_type")

    def sum_of_squared_diffs_from_mean(
        self,
        *,
        ignore_nulls: bool,
        mean: Optional[U] = None,
        as_py: bool = True,
    ) -> Optional[U]:
        if mean is None:
            mean = self.mean(ignore_nulls=ignore_nulls)
        if mean is None:
            return None
        result = ((self._column - mean) ** 2).sum(skipna=ignore_nulls)
        return _as_py(result) if as_py else result

    def to_pylist(self) -> List[Any]:
        return self._column.to_arrow().to_pylist()

    def to_numpy(self, zero_copy_only: bool = False) -> np.ndarray:
        return self._column.to_numpy()

    def _to_arrow_compatible_container(self) -> Union[List[Any], "pyarrow.Array"]:
        return self._column.to_arrow()
