from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

from ray.data._internal.table_block import TableBlockAccessor, TableBlockBuilder
from ray.data.block import (
    Block,
    BlockAccessor,
    BlockExecStats,
    BlockType,
)

if TYPE_CHECKING:
    import cudf
    import pandas
    import pyarrow

    from ray.data._internal.planner.exchange.sort_task_spec import SortKey
    from ray.data.block import BlockMetadataWithSchema

T = TypeVar("T")


class CudfRow(Mapping):
    """Row of a tabular Dataset backed by a cuDF DataFrame block."""

    def __init__(self, row: Any):
        self._row = row

    def __getitem__(self, key: str) -> Any:
        val = self._row[key].iloc[0]
        # Convert cuDF scalar to Python native type.
        try:
            return val.item()
        except AttributeError:
            return val

    def __iter__(self) -> Iterator:
        for k in self._row.columns:
            yield k

    def __len__(self):
        return len(self._row.columns)

    def as_pydict(self) -> Dict[str, Any]:
        return {k: self[k] for k in self._row.columns}


class CudfBlockBuilder(TableBlockBuilder):
    def __init__(self):
        try:
            import cudf
        except ImportError:
            raise ImportError(
                "cuDF is required to use batch_format='cudf'. "
                "Install it with: pip install cudf-cu12"
            )
        super().__init__(cudf.DataFrame)

    @staticmethod
    def _table_from_pydict(columns: Dict[str, List[Any]]) -> "cudf.DataFrame":
        import cudf

        return cudf.DataFrame(columns)

    @staticmethod
    def _combine_tables(tables: List["cudf.DataFrame"]) -> "cudf.DataFrame":
        import cudf

        if len(tables) > 1:
            df = cudf.concat(tables, ignore_index=True)
        else:
            df = tables[0]
        return df

    @staticmethod
    def _concat_would_copy() -> bool:
        return True

    @staticmethod
    def _empty_table() -> "cudf.DataFrame":
        import cudf

        return cudf.DataFrame()

    def block_type(self) -> BlockType:
        return BlockType.CUDF


class CudfBlockAccessor(TableBlockAccessor):
    ROW_TYPE = CudfRow

    def __init__(self, table: "cudf.DataFrame"):
        super().__init__(table)

    def _get_row(self, index: int) -> CudfRow:
        base_row = self.slice(index, index + 1, copy=False)
        return CudfRow(base_row)

    def column_names(self) -> List[str]:
        return self._table.columns.tolist()

    def fill_column(self, name: str, value: Any) -> Block:
        return self._table.assign(**{name: value})

    def slice(self, start: int, end: int, copy: bool = False) -> "cudf.DataFrame":
        view = self._table.iloc[start:end]
        if copy:
            view = view.copy(deep=True)
        return view

    def take(self, indices: List[int]) -> "cudf.DataFrame":
        return self._table.iloc[indices].reset_index(drop=True)

    def drop(self, columns: List[str]) -> Block:
        return self._table.drop(columns=columns)

    def select(self, columns: List[str]) -> "cudf.DataFrame":
        if not all(isinstance(col, str) for col in columns):
            raise ValueError(
                "Columns must be a list of column name strings when aggregating on "
                f"cuDF blocks, but got: {columns}."
            )
        return self._table[columns]

    def rename_columns(self, columns_rename: Dict[str, str]) -> "cudf.DataFrame":
        return self._table.rename(columns=columns_rename)

    def upsert_column(self, column_name: str, column_data: Any) -> "cudf.DataFrame":
        return self._table.assign(**{column_name: column_data})

    def random_shuffle(self, random_seed: Optional[int]) -> "cudf.DataFrame":
        return self._table.sample(frac=1, random_state=random_seed).reset_index(
            drop=True
        )

    def schema(self) -> "pyarrow.Schema":
        return self._table.to_arrow().schema

    def to_pandas(self) -> "pandas.DataFrame":
        return self._table.to_pandas()

    def to_numpy(
        self, columns: Optional[Union[str, List[str]]] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if columns is None:
            columns = self._table.columns.tolist()
            should_be_single_ndarray = False
        elif isinstance(columns, list):
            should_be_single_ndarray = False
        else:
            columns = [columns]
            should_be_single_ndarray = True

        column_names_set = set(self._table.columns)
        for column in columns:
            if column not in column_names_set:
                raise ValueError(
                    f"Cannot find column {column}, available columns: "
                    f"{self._table.columns.tolist()}"
                )

        arrays = [self._table[col].to_numpy() for col in columns]

        if should_be_single_ndarray:
            return arrays[0]
        return dict(zip(columns, arrays))

    def to_arrow(self) -> "pyarrow.Table":
        return self._table.to_arrow()

    def to_cudf(self) -> "cudf.DataFrame":
        return self._table

    def to_block(self) -> "cudf.DataFrame":
        return self._table

    def num_rows(self) -> int:
        return len(self._table)

    def size_bytes(self) -> int:
        return int(self._table.memory_usage(deep=True).sum())

    def block_type(self) -> BlockType:
        return BlockType.CUDF

    @staticmethod
    def builder() -> CudfBlockBuilder:
        return CudfBlockBuilder()

    @staticmethod
    def _empty_table() -> "cudf.DataFrame":
        return CudfBlockBuilder._empty_table()

    def _zip(self, acc: BlockAccessor) -> "cudf.DataFrame":
        r = self._table.copy(deep=False)
        s = acc.to_cudf()
        for col_name in s.columns:
            col = s[col_name]
            column_names = list(r.columns)
            if col_name in column_names:
                i = 1
                new_name = col_name
                while new_name in column_names:
                    new_name = "{}_{}".format(col_name, i)
                    i += 1
                col_name = new_name
            r[col_name] = col
        return r

    def _sample(self, n_samples: int, sort_key: "SortKey") -> "cudf.DataFrame":
        return self._table[sort_key.get_columns()].sample(n_samples)

    def sort(self, sort_key: "SortKey") -> "cudf.DataFrame":
        assert (
            sort_key.get_columns()
        ), f"Sorting columns couldn't be empty (got {sort_key.get_columns()})"

        if len(self._table) == 0:
            return self._empty_table()

        columns, ascending = sort_key.to_pandas_sort_args()
        return self._table.sort_values(by=columns, ascending=ascending)

    def sort_and_partition(
        self, boundaries: List[T], sort_key: "SortKey"
    ) -> List[Block]:
        table = self.sort(sort_key)

        if len(table) == 0:
            return [self._empty_table() for _ in range(len(boundaries) + 1)]
        elif len(boundaries) == 0:
            return [table]

        return BlockAccessor.for_block(table)._find_partitions_sorted(
            boundaries, sort_key
        )

    @staticmethod
    def merge_sorted_blocks(
        blocks: List[Block], sort_key: "SortKey"
    ) -> Tuple[Block, "BlockMetadataWithSchema"]:
        import cudf

        stats = BlockExecStats.builder()
        blocks = [b for b in blocks if len(b) > 0]
        if len(blocks) == 0:
            ret = CudfBlockAccessor._empty_table()
        else:
            blocks = TableBlockAccessor.normalize_block_types(blocks, BlockType.CUDF)
            ret = cudf.concat(blocks, ignore_index=True)
            columns, ascending = sort_key.to_pandas_sort_args()
            ret = ret.sort_values(by=columns, ascending=ascending)
        from ray.data.block import BlockMetadataWithSchema

        return ret, BlockMetadataWithSchema.from_block(ret, stats=stats.build())

    def _get_group_boundaries_sorted(self, keys: List[str]) -> np.ndarray:
        """GPU-native group boundary detection.

        Detects transitions between groups entirely on GPU using cuDF
        shift/compare, transferring only the small boundary-index array to CPU.

        NOTE: NaN key values follow IEEE 754 semantics (NaN != NaN), so
        consecutive NaN rows are treated as *distinct* groups. This differs
        from the base-class numpy implementation which treats consecutive NaN
        values as the *same* group.
        """
        if self.num_rows() == 0:
            return np.array([], dtype=np.int32)
        if not keys:
            return np.array([0, self.num_rows()])

        # Detect where key values change — runs entirely on GPU.
        # shift(1) introduces a null in row 0, and key columns may themselves
        # contain nulls, so (key_df != shifted) can produce a null-valued
        # boolean DataFrame.  cuDF's any(axis=1) does not accept nulls, so we
        # fill them with True (null comparison → treat as a group boundary).
        key_df = self._table[keys]
        shifted = key_df.shift(1)
        changed = (key_df != shifted).fillna(True).any(axis=1)
        # Transfer only boundary indices (small int array) to CPU
        boundary_starts = changed.to_numpy().nonzero()[0]
        return np.append(boundary_starts, self.num_rows()).astype(np.int32)

    def iter_rows(
        self, public_row_format: bool
    ) -> Iterator[Union[Mapping, np.ndarray]]:
        num_rows = self.num_rows()
        for i in range(num_rows):
            row = self._get_row(i)
            if public_row_format:
                yield row.as_pydict()
            else:
                yield row
