from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from ray.data._internal.logical.interfaces import (
    LogicalOperator,
    LogicalOperatorSupportsPredicatePassThrough,
    PredicatePassThroughBehavior,
)
from ray.data._internal.planner.exchange.interfaces import ExchangeTaskSpec
from ray.data._internal.planner.exchange.shuffle_task_spec import ShuffleTaskSpec
from ray.data._internal.planner.exchange.sort_task_spec import SortKey, SortTaskSpec
from ray.data.aggregate import AggregateFn
from ray.data.block import BlockMetadata

if TYPE_CHECKING:

    from ray.data.block import Schema

__all__ = [
    "AbstractAllToAll",
    "Aggregate",
    "RandomShuffle",
    "RandomizeBlocks",
    "Repartition",
    "Sort",
]


class AbstractAllToAll(LogicalOperator):
    """Abstract class for logical operators should be converted to physical
    AllToAllOperator.
    """

    def __init__(
        self,
        input_op: LogicalOperator,
        num_outputs: Optional[int] = None,
        sub_progress_bar_names: Optional[List[str]] = None,
        ray_remote_args: Optional[Dict[str, Any]] = None,
        *,
        name: Optional[str] = None,
    ):
        """Initialize an ``AbstractAllToAll`` logical operator.

        Args:
            input_op: The operator preceding this operator in the plan DAG. The outputs
                of `input_op` will be the inputs to this operator.
            num_outputs: The number of expected output bundles outputted by this
                operator.
            sub_progress_bar_names: Optional sub-stage progress bar names for this
                operator.
            ray_remote_args: Args to provide to :func:`ray.remote`.
            name: Name for this operator. This is the name that will appear when
                inspecting the logical plan of a Dataset.
        """
        super().__init__(
            input_dependencies=[input_op],
            num_outputs=num_outputs,
            name=name,
        )
        self.ray_remote_args = ray_remote_args or {}
        self.sub_progress_bar_names = sub_progress_bar_names

    @property
    def num_outputs(self) -> Optional[int]:
        return self._num_outputs


class RandomizeBlocks(AbstractAllToAll, LogicalOperatorSupportsPredicatePassThrough):
    """Logical operator for randomize_block_order."""

    def __init__(
        self,
        input_op: LogicalOperator,
        seed: Optional[int] = None,
    ):
        super().__init__(
            input_op,
            name="RandomizeBlockOrder",
        )
        self.seed = seed

    def infer_metadata(self) -> "BlockMetadata":
        assert len(self.input_dependencies) == 1, len(self.input_dependencies)
        assert isinstance(self.input_dependencies[0], LogicalOperator)
        return self.input_dependencies[0].infer_metadata()

    def infer_schema(
        self,
    ) -> Optional["Schema"]:
        assert len(self.input_dependencies) == 1, len(self.input_dependencies)
        assert isinstance(self.input_dependencies[0], LogicalOperator)
        return self.input_dependencies[0].infer_schema()

    def predicate_passthrough_behavior(self) -> PredicatePassThroughBehavior:
        # Randomizing block order doesn't affect filtering correctness
        return PredicatePassThroughBehavior.PASSTHROUGH


class RandomShuffle(AbstractAllToAll, LogicalOperatorSupportsPredicatePassThrough):
    """Logical operator for random_shuffle."""

    def __init__(
        self,
        input_op: LogicalOperator,
        name: str = "RandomShuffle",
        seed: Optional[int] = None,
        ray_remote_args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            input_op,
            sub_progress_bar_names=[
                ExchangeTaskSpec.MAP_SUB_PROGRESS_BAR_NAME,
                ExchangeTaskSpec.REDUCE_SUB_PROGRESS_BAR_NAME,
            ],
            ray_remote_args=ray_remote_args,
            name=name,
        )
        self.seed = seed

    def infer_metadata(self) -> "BlockMetadata":
        assert len(self.input_dependencies) == 1, len(self.input_dependencies)
        assert isinstance(self.input_dependencies[0], LogicalOperator)
        return self.input_dependencies[0].infer_metadata()

    def infer_schema(
        self,
    ) -> Optional["Schema"]:
        assert len(self.input_dependencies) == 1, len(self.input_dependencies)
        assert isinstance(self.input_dependencies[0], LogicalOperator)
        return self.input_dependencies[0].infer_schema()

    def predicate_passthrough_behavior(self) -> PredicatePassThroughBehavior:
        # Random shuffle doesn't affect filtering correctness
        return PredicatePassThroughBehavior.PASSTHROUGH


class Repartition(AbstractAllToAll, LogicalOperatorSupportsPredicatePassThrough):
    """Logical operator for repartition."""

    def __init__(
        self,
        input_op: LogicalOperator,
        num_outputs: int,
        shuffle: bool,
        keys: Optional[List[str]] = None,
        sort: bool = False,
    ):
        if shuffle:
            sub_progress_bar_names = [
                ExchangeTaskSpec.MAP_SUB_PROGRESS_BAR_NAME,
                ExchangeTaskSpec.REDUCE_SUB_PROGRESS_BAR_NAME,
            ]
        else:
            sub_progress_bar_names = [
                ShuffleTaskSpec.SPLIT_REPARTITION_SUB_PROGRESS_BAR_NAME,
            ]
        super().__init__(
            input_op,
            num_outputs=num_outputs,
            sub_progress_bar_names=sub_progress_bar_names,
        )
        self.shuffle = shuffle
        self.keys = keys
        self.sort = sort

    def infer_metadata(self) -> "BlockMetadata":
        assert len(self.input_dependencies) == 1, len(self.input_dependencies)
        assert isinstance(self.input_dependencies[0], LogicalOperator)
        return self.input_dependencies[0].infer_metadata()

    def infer_schema(
        self,
    ) -> Optional["Schema"]:
        assert len(self.input_dependencies) == 1, len(self.input_dependencies)
        assert isinstance(self.input_dependencies[0], LogicalOperator)
        return self.input_dependencies[0].infer_schema()

    def predicate_passthrough_behavior(self) -> PredicatePassThroughBehavior:
        # Repartition doesn't affect filtering correctness
        return PredicatePassThroughBehavior.PASSTHROUGH


class Sort(AbstractAllToAll, LogicalOperatorSupportsPredicatePassThrough):
    """Logical operator for sort."""

    def __init__(
        self,
        input_op: LogicalOperator,
        sort_key: SortKey,
        batch_format: Optional[str] = "default",
    ):
        super().__init__(
            input_op,
            sub_progress_bar_names=[
                SortTaskSpec.SORT_SAMPLE_SUB_PROGRESS_BAR_NAME,
                ExchangeTaskSpec.MAP_SUB_PROGRESS_BAR_NAME,
                ExchangeTaskSpec.REDUCE_SUB_PROGRESS_BAR_NAME,
            ],
        )
        self.sort_key = sort_key
        self.batch_format = batch_format

    def infer_metadata(self) -> "BlockMetadata":
        assert len(self.input_dependencies) == 1, len(self.input_dependencies)
        assert isinstance(self.input_dependencies[0], LogicalOperator)
        return self.input_dependencies[0].infer_metadata()

    def infer_schema(
        self,
    ) -> Optional["Schema"]:
        assert len(self.input_dependencies) == 1, len(self.input_dependencies)
        assert isinstance(self.input_dependencies[0], LogicalOperator)
        return self.input_dependencies[0].infer_schema()

    def predicate_passthrough_behavior(self) -> PredicatePassThroughBehavior:
        # Sort doesn't affect filtering correctness
        return PredicatePassThroughBehavior.PASSTHROUGH


class Aggregate(AbstractAllToAll):
    """Logical operator for aggregate."""

    def __init__(
        self,
        input_op: LogicalOperator,
        key: Optional[str],
        aggs: List[AggregateFn],
        num_partitions: Optional[int] = None,
        batch_format: Optional[str] = "default",
    ):
        super().__init__(
            input_op,
            sub_progress_bar_names=[
                SortTaskSpec.SORT_SAMPLE_SUB_PROGRESS_BAR_NAME,
                ExchangeTaskSpec.MAP_SUB_PROGRESS_BAR_NAME,
                ExchangeTaskSpec.REDUCE_SUB_PROGRESS_BAR_NAME,
            ],
        )
        self.key = key
        self.aggs = aggs
        self.num_partitions = num_partitions
        self.batch_format = batch_format


class GPUShuffleMapGroups(AbstractAllToAll):
    """Logical operator for fused GPU shuffle + per-group UDF application.

    Produced by ``GroupedData.map_groups()`` when
    ``DataContext.shuffle_strategy == ShuffleStrategy.GPU_SHUFFLE``.

    The UDF is applied directly inside the ``GPUShuffleActor`` immediately
    after extraction, so the full-size input partition never passes through
    the Ray object store as Arrow. Only the (typically much smaller) UDF
    result is serialised to Arrow and stored.
    """

    def __init__(
        self,
        input_op: LogicalOperator,
        key_columns: List[str],
        udf: Callable,
        num_partitions: int,
        batch_format: Optional[str],
        fn_args: Tuple,
        fn_kwargs: Dict[str, Any],
        fn_constructor_args: Optional[Tuple],
        fn_constructor_kwargs: Optional[Dict[str, Any]],
    ):
        super().__init__(
            input_op,
            num_outputs=num_partitions,
            name=f"GPUShuffleMapGroups(key_columns={key_columns})",
        )
        self.key_columns = key_columns
        self.udf = udf
        self.batch_format = batch_format
        self.fn_args = fn_args
        self.fn_kwargs = fn_kwargs
        self.fn_constructor_args = fn_constructor_args
        self.fn_constructor_kwargs = fn_constructor_kwargs

    def infer_metadata(self) -> "BlockMetadata":
        assert len(self.input_dependencies) == 1
        return self.input_dependencies[0].infer_metadata()

    def infer_schema(self) -> Optional["Schema"]:
        # Output schema depends on the UDF return type and is not statically known.
        return None
