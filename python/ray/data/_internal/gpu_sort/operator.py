"""Physical Ray Data operator for the streaming spillable GPU range sort.

GPU ranks turn bounded input into durable actor-local sorted runs while the
upstream source is still producing blocks.  Only range planning and the final
redistribution wait for end-of-input.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import ray
from ray.actor import ActorHandle
from ray.data import ExecutionOptions
from ray.data._internal.execution.bundle_queue import ReorderingBundleQueue
from ray.data._internal.execution.interfaces import (
    ExecutionResources,
    PhysicalOperator,
    RefBundle,
)
from ray.data._internal.execution.interfaces.physical_operator import (
    DataOpTask,
    MetadataOpTask,
    OpTask,
)
from ray.data._internal.execution.operators.hash_shuffle import (
    _get_total_cluster_resources,
)
from ray.data._internal.execution.operators.sub_progress import SubProgressBarMixin
from ray.data._internal.gpu_sort.config import (
    MPF_PROGRESS_RESERVE_BYTES,
    MPF_RECEIVE_BUFFER_OWNERS,
    GPUSortCapacityError,
    GPUSortConfig,
)
from ray.data.block import BlockStats, to_stats
from ray.data.context import DataContext

if TYPE_CHECKING:
    from ray.data._internal.execution.block_ref_counter import BlockRefCounter

GPU_SORT_PARTITION_ID_KEY = b"ray-data-gpu-sort-partition"
GPU_SORT_DIAGNOSTICS_KEY = b"ray-data-gpu-sort-diagnostics"
_MPF_PROGRESS_RESERVE_BYTES = MPF_PROGRESS_RESERVE_BYTES
_MPF_RECEIVE_BUFFER_OWNERS = MPF_RECEIVE_BUFFER_OWNERS

# Driver-local benchmark hook.  A JSON round trip in the accessor prevents a
# caller from mutating the record while another Dataset is being constructed.
LAST_RUN_STATS: Dict[str, Any] = {}


def get_last_run_stats() -> Dict[str, Any]:
    return json.loads(json.dumps(LAST_RUN_STATS))


def _validate_gpu_schema(schema: Any, key_columns: Sequence[str]) -> None:
    """Validate the intentionally small, Arrow-only first-PR type surface."""

    import pyarrow as pa

    schema = getattr(schema, "base_schema", schema)
    if not isinstance(schema, pa.Schema):
        raise NotImplementedError("GPU sort currently requires Arrow-backed blocks.")
    missing = [name for name in key_columns if name not in schema.names]
    if missing:
        raise ValueError(
            f"GPU sort keys {missing} are absent from schema columns {schema.names}."
        )
    unsupported = [
        field.name
        for field in schema
        if (
            pa.types.is_nested(field.type)
            or pa.types.is_union(field.type)
            or pa.types.is_dictionary(field.type)
            or isinstance(field.type, pa.ExtensionType)
        )
    ]
    if unsupported:
        raise NotImplementedError(
            "GPU sort supports flat Arrow scalar columns only; unsupported "
            f"columns: {unsupported}."
        )

    key_types = [schema.field(name).type for name in key_columns]
    bad_keys = [
        name
        for name, typ in zip(key_columns, key_types)
        if not (
            pa.types.is_boolean(typ)
            or pa.types.is_string(typ)
            or pa.types.is_large_string(typ)
            or pa.types.is_integer(typ)
            or pa.types.is_floating(typ)
            or pa.types.is_date(typ)
            or pa.types.is_time(typ)
            or pa.types.is_timestamp(typ)
        )
    ]
    if bad_keys:
        raise NotImplementedError(
            "GPU sort keys must be boolean, string, integer, float, date, "
            f"time, or timestamp columns; unsupported keys: {bad_keys}."
        )


@dataclass(frozen=True)
class _InputBlock:
    value: Any
    size_bytes: int
    num_rows: int
    ordinal: int = -1


@dataclass(frozen=True)
class _ExchangeRound:
    """A deterministic set of prepared batches safe for one MPF exchange."""

    batch_ids_by_rank: Tuple[Tuple[int, ...], ...]
    outgoing_bytes: Tuple[int, ...]
    incoming_bytes: Tuple[int, ...]
    modeled_headroom_bytes: Tuple[int, ...]


def _allocate_stratified_sample_quotas(
    blocks: Sequence[_InputBlock], configured_sample_rows: int
) -> Tuple[List[int], int]:
    """Globally allocate an exact sample budget across logical input blocks."""

    configured_sample_rows = int(configured_sample_rows)
    if configured_sample_rows < 1:
        raise ValueError("GPU sort sample size must be positive.")
    nonempty = [block for block in blocks if int(block.num_rows) > 0]
    ordinals = [int(block.ordinal) for block in blocks]
    if any(ordinal < 0 for ordinal in ordinals) or len(set(ordinals)) != len(ordinals):
        raise ValueError("GPU sort requires unique nonnegative logical block ordinals.")
    total_rows = sum(int(block.num_rows) for block in nonempty)
    if total_rows == 0:
        return [0] * len(blocks), 0

    target = min(total_rows, max(configured_sample_rows, len(nonempty)))
    quotas = {int(block.ordinal): 1 for block in nonempty}
    remaining = target - len(nonempty)
    capacities = {int(block.ordinal): int(block.num_rows) - 1 for block in nonempty}
    total_capacity = sum(capacities.values())
    if remaining and total_capacity:
        remainders = []
        assigned = 0
        for block in nonempty:
            ordinal = int(block.ordinal)
            numerator = remaining * capacities[ordinal]
            extra, remainder = divmod(numerator, total_capacity)
            quotas[ordinal] += extra
            assigned += extra
            remainders.append((remainder, ordinal))
        leftover = remaining - assigned
        for _, ordinal in sorted(remainders, key=lambda item: (-item[0], item[1]))[
            :leftover
        ]:
            quotas[ordinal] += 1

    result = [quotas.get(int(block.ordinal), 0) for block in blocks]
    if sum(result) != target or any(
        quota < 0 or quota > int(block.num_rows) for block, quota in zip(blocks, result)
    ):
        raise RuntimeError("GPU sort could not allocate its exact sample budget.")
    return result, target


def _sampling_plan_digest(
    blocks: Sequence[_InputBlock],
    quotas: Sequence[int],
    *,
    seed: int,
    target_rows: int,
) -> str:
    """Hash the scheduling-independent logical sampling plan."""

    if len(blocks) != len(quotas):
        raise ValueError("GPU sort blocks and sample quotas must have equal length.")
    payload = {
        "scheme": "deterministic_stratified_random",
        "version": 1,
        "seed": int(seed),
        "target_rows": int(target_rows),
        "blocks": sorted(
            [
                [int(block.ordinal), int(block.num_rows), int(quota)]
                for block, quota in zip(blocks, quotas)
            ]
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sample_quota_summary(quotas: Sequence[int]) -> Dict[str, Any]:
    """Return compact distribution telemetry over sampled nonempty blocks."""

    positive = sorted(int(quota) for quota in quotas if int(quota) > 0)
    if not positive:
        return {"min": 0, "median": 0, "max": 0}
    middle = len(positive) // 2
    median: Any
    if len(positive) % 2:
        median = positive[middle]
    else:
        total = positive[middle - 1] + positive[middle]
        median = total // 2 if total % 2 == 0 else total / 2
    return {"min": positive[0], "median": median, "max": positive[-1]}


def _underlying_object_ref(block: _InputBlock) -> Any:
    value = block.value
    return value[0] if isinstance(value, tuple) and len(value) == 3 else value


def _assign_blocks_by_locality(
    blocks: Sequence[_InputBlock],
    actor_node_ids: Sequence[str],
    object_locations: Mapping[Any, Mapping[str, Any]],
) -> Tuple[
    List[List[_InputBlock]],
    List[int],
    List[int],
    List[int],
    List[int],
]:
    """Assign blocks locally, with deterministic decoded-byte balancing.

    An object may have multiple replicas, no reported location (for example an
    inline object), or a location on a node without a GPU-sort actor.  The same
    size-first rule is used within the local candidates and as the global
    fallback, so location lookup is never a correctness dependency.
    """

    nranks = len(actor_node_ids)
    if nranks < 1:
        raise ValueError("GPU sort requires at least one actor for block assignment.")
    blocks_by_rank: List[List[_InputBlock]] = [[] for _ in range(nranks)]
    assigned_bytes = [0] * nranks
    assigned_blocks = [0] * nranks
    local_bytes = [0] * nranks
    local_blocks = [0] * nranks

    for block in blocks:
        ref = _underlying_object_ref(block)
        try:
            location = object_locations.get(ref, {})
        except TypeError:
            location = {}
        node_ids = set(location.get("node_ids", ()) or ())
        candidates = [
            rank
            for rank, node_id in enumerate(actor_node_ids)
            if node_id and node_id in node_ids
        ]
        is_local = bool(candidates)
        if not candidates:
            candidates = list(range(nranks))
        rank = min(
            candidates,
            key=lambda item: (
                assigned_bytes[item],
                assigned_blocks[item],
                item,
            ),
        )
        blocks_by_rank[rank].append(block)
        assigned_bytes[rank] += int(block.size_bytes)
        assigned_blocks[rank] += 1
        if is_local:
            local_bytes[rank] += int(block.size_bytes)
            local_blocks[rank] += 1

    return (
        blocks_by_rank,
        assigned_bytes,
        assigned_blocks,
        local_bytes,
        local_blocks,
    )


def _make_waves(
    blocks_by_rank: Sequence[Sequence[_InputBlock]],
    target_bytes_per_rank: Optional[int],
) -> List[List[List[Any]]]:
    """Create the same number of deterministic, bounded waves for every rank."""

    rank_waves: List[List[List[Any]]] = []
    for blocks in blocks_by_rank:
        if not blocks:
            rank_waves.append([])
            continue
        if target_bytes_per_rank is None:
            rank_waves.append([[block.value for block in blocks]])
            continue
        waves: List[List[Any]] = []
        current: List[Any] = []
        current_bytes = 0
        for block in blocks:
            if current and current_bytes + block.size_bytes > target_bytes_per_rank:
                waves.append(current)
                current = []
                current_bytes = 0
            current.append(block.value)
            current_bytes += block.size_bytes
        if current:
            waves.append(current)
        rank_waves.append(waves)

    count = max((len(waves) for waves in rank_waves), default=0)
    return [
        [waves[index] if index < len(waves) else [] for waves in rank_waves]
        for index in range(count)
    ]


def _ordered_exchange_receipts(
    receipts: Sequence[Mapping[str, Any]],
    *,
    nranks: int,
    wave_id: int,
    exchange_id: int,
) -> List[Dict[str, Any]]:
    """Validate and order one all-rank exchange acknowledgement."""

    by_rank: List[Optional[Dict[str, Any]]] = [None] * int(nranks)
    for raw in receipts:
        item = dict(raw)
        rank = int(item.get("rank", -1))
        if (
            not 0 <= rank < nranks
            or by_rank[rank] is not None
            or int(item.get("wave", -1)) != int(wave_id)
            or int(item.get("exchange", -1)) != int(exchange_id)
        ):
            raise RuntimeError("GPU sort exchange receipts are invalid or duplicated.")
        received_rows = int(item.get("received_rows", -1))
        if received_rows < 0:
            raise RuntimeError("GPU sort exchange receipt has an invalid row count.")
        by_rank[rank] = item
    if any(item is None for item in by_rank):
        raise RuntimeError("GPU sort exchange receipts are incomplete.")
    return [item for item in by_rank if item is not None]


def _validate_source_wave_rows(
    blocks_for_ranks: Sequence[Sequence[_InputBlock]],
    *,
    wave_id: int,
    received_rows: int,
) -> None:
    """Prove redistribution conserved a wave before deleting source runs."""

    expected_wave_rows = sum(
        int(block.num_rows) for blocks in blocks_for_ranks for block in blocks
    )
    if int(received_rows) != expected_wave_rows:
        raise RuntimeError(
            "GPU sort source wave did not conserve rows before commit: "
            f"wave={wave_id}, expected={expected_wave_rows}, "
            f"received={received_rows}."
        )


def _validate_source_wave_commit(
    receipts: Sequence[Mapping[str, Any]],
    blocks_for_ranks: Sequence[Sequence[_InputBlock]],
    *,
    wave_id: int,
    received_rows: int,
) -> None:
    """Validate exact actor ownership release after a conserved exchange."""

    _validate_source_wave_rows(
        blocks_for_ranks, wave_id=wave_id, received_rows=received_rows
    )

    by_rank: List[Optional[Dict[str, Any]]] = [None] * len(blocks_for_ranks)
    for raw in receipts:
        item = dict(raw)
        rank = int(item.get("rank", -1))
        if (
            not 0 <= rank < len(by_rank)
            or by_rank[rank] is not None
            or int(item.get("wave", -1)) != int(wave_id)
        ):
            raise RuntimeError(
                "GPU sort source-wave commit receipts are invalid or duplicated."
            )
        by_rank[rank] = item
    if any(item is None for item in by_rank):
        raise RuntimeError("GPU sort source-wave commit receipts are incomplete.")

    for rank, (item, blocks) in enumerate(zip(by_rank, blocks_for_ranks)):
        assert item is not None
        expected_ids = [int(block.value) for block in blocks]
        expected_rows = sum(int(block.num_rows) for block in blocks)
        expected_bytes = sum(int(block.size_bytes) for block in blocks)
        actual_ids = [int(value) for value in item.get("released_source_run_ids", ())]
        if (
            actual_ids != expected_ids
            or int(item.get("released_source_runs", -1)) != len(expected_ids)
            or int(item.get("released_source_rows", -1)) != expected_rows
            or int(item.get("released_source_bytes", -1)) != expected_bytes
        ):
            raise RuntimeError(
                "GPU sort source-wave commit receipt does not match its owned "
                f"runs for rank {rank}."
            )


def _plan_exchange_round(
    batches_by_rank: Sequence[Sequence[Mapping[str, Any]]],
    *,
    current_allocated_bytes: Sequence[int],
    pool_max_bytes: Sequence[int],
) -> _ExchangeRound:
    """Select the largest deterministic MPF round that preserves headroom.

    Prepared packed buffers are already included in ``current_allocated_bytes``
    and move into MPF without another full-payload send allocation. Conservatively
    reserve two incoming owners. Exact destination sizes therefore let balanced
    traffic stay together while an all-to-one exchange is split before MPF starts
    its C++ progress thread.
    """

    ranks = len(batches_by_rank)
    if (
        ranks == 0
        or len(current_allocated_bytes) != ranks
        or len(pool_max_bytes) != ranks
    ):
        raise ValueError("GPU sort exchange admission requires one budget per rank.")
    current = [int(value) for value in current_allocated_bytes]
    maximum = [int(value) for value in pool_max_bytes]
    if any(value < 0 for value in current) or any(value <= 0 for value in maximum):
        raise ValueError("GPU sort exchange memory measurements must be positive.")

    selected: List[List[int]] = [[] for _ in range(ranks)]
    outgoing = [0] * ranks
    incoming = [0] * ranks
    # Interleave source ranks at each batch ordinal so a hot destination does
    # not let the first rank monopolize a bounded round.
    candidates = sorted(
        (
            int(batch["batch_id"]),
            source,
            batch,
        )
        for source, batches in enumerate(batches_by_rank)
        for batch in batches
    )
    for batch_id, source, batch in candidates:
        destinations = [int(value) for value in batch["destination_bytes"]]
        if len(destinations) != ranks or any(value < 0 for value in destinations):
            raise ValueError(
                "Every prepared GPU batch needs nonnegative bytes for every rank."
            )
        source_bytes = sum(destinations)
        candidate_outgoing = list(outgoing)
        candidate_incoming = list(incoming)
        candidate_outgoing[source] += source_bytes
        for destination, value in enumerate(destinations):
            candidate_incoming[destination] += value
        required = [
            current[rank]
            + _MPF_RECEIVE_BUFFER_OWNERS * candidate_incoming[rank]
            + _MPF_PROGRESS_RESERVE_BYTES
            for rank in range(ranks)
        ]
        if all(required[rank] <= maximum[rank] for rank in range(ranks)):
            selected[source].append(batch_id)
            outgoing = candidate_outgoing
            incoming = candidate_incoming

    if candidates and not any(selected):
        smallest = min(maximum[rank] - current[rank] for rank in range(ranks))
        raise GPUSortCapacityError(
            "No prepared GPU shuffle batch fits with the required MPF transport "
            f"reserve; smallest measured headroom is {smallest} bytes."
        )
    modeled_headroom = tuple(
        maximum[rank] - current[rank] - _MPF_RECEIVE_BUFFER_OWNERS * incoming[rank]
        for rank in range(ranks)
    )
    return _ExchangeRound(
        batch_ids_by_rank=tuple(tuple(values) for values in selected),
        outgoing_bytes=tuple(outgoing),
        incoming_bytes=tuple(incoming),
        modeled_headroom_bytes=modeled_headroom,
    )


def _wave_target_bytes(
    blocks_by_rank: Sequence[Sequence[_InputBlock]],
    *,
    explicit_residency_budget_bytes: Optional[int],
    actor_usable_budgets: Sequence[int],
    auto_wave_fraction: float,
) -> Optional[int]:
    """Return a bounded-wave target, or ``None`` for the resident fast path."""

    if explicit_residency_budget_bytes is not None:
        # Preserve the original capacity-study behavior exactly.
        return max(256 << 20, int(explicit_residency_budget_bytes) // 2)

    budgets = [int(value) for value in actor_usable_budgets if int(value) > 0]
    if not budgets:
        raise RuntimeError("GPU sort actors did not report usable memory budgets.")
    smallest_budget = min(budgets)
    requested = max(256 << 20, int(smallest_budget * float(auto_wave_fraction)))
    # Four copies is the balanced-case MPF ownership ceiling. Actor admission
    # later subtracts its measured allocator baseline before applying the same
    # transport reserve.
    balanced_target = max(1, (smallest_budget - _MPF_PROGRESS_RESERVE_BYTES) // 4)
    resident_target = min(requested, balanced_target)
    largest_rank_input = max(
        (sum(block.size_bytes for block in blocks) for blocks in blocks_by_rank),
        default=0,
    )
    return None if largest_rank_input <= resident_target else resident_target


def _derive_num_ranks(data_context: DataContext) -> int:
    configured = data_context.gpu_shuffle_num_actors
    if configured is not None:
        if configured < 1:
            raise ValueError("gpu_shuffle_num_actors must be positive.")
        return int(configured)
    ranks = int(_get_total_cluster_resources().gpu or 0)
    if ranks < 1:
        raise RuntimeError(
            "GPU sort requires at least one Ray GPU resource. Set "
            "DataContext.gpu_shuffle_num_actors to override detection."
        )
    return ranks


def _operator_config(data_context: DataContext) -> Dict[str, Any]:
    env_budget = os.environ.get("RAY_DATA_GPU_SORT_MEMORY_BUDGET_BYTES")
    context_budget = data_context.get_config("gpu_sort_memory_budget_bytes", None)
    sample_size = max(
        65_536, int(data_context.get_config("gpu_sort_sample_size", 65_536))
    )
    external_run_store = str(
        data_context.get_config("gpu_sort_external_run_store", "plasma")
    )
    return GPUSortConfig(
        sample_size=sample_size,
        sample_seed=int(data_context.get_config("gpu_sort_sample_seed", 0)),
        input_buffer_budget_bytes=int(
            data_context.get_config("gpu_sort_input_buffer_budget_bytes", 16 << 30)
        ),
        streaming_sample_rows_per_block=int(
            data_context.get_config("gpu_sort_streaming_sample_rows_per_block", 64)
        ),
        residency_budget_bytes=(
            env_budget if env_budget is not None else context_budget
        ),
        auto_wave_fraction=float(
            data_context.get_config("gpu_sort_auto_wave_fraction", 0.50)
        ),
        exchange_batch_bytes=int(
            data_context.get_config("gpu_sort_exchange_batch_bytes", 512 << 20)
        ),
        run_chunk_bytes=int(
            data_context.get_config("gpu_sort_run_chunk_bytes", 512 << 20)
        ),
        merge_fan_in=int(data_context.get_config("gpu_sort_merge_fan_in", 4)),
        external_run_store=external_run_store,
        external_run_directory=data_context.get_config(
            "gpu_sort_external_run_directory", None
        ),
        external_run_id=(
            uuid.uuid4().hex if external_run_store == "local_disk" else None
        ),
        external_run_min_free_bytes=data_context.get_config(
            "gpu_sort_external_run_min_free_bytes", 0
        ),
        external_run_max_live_bytes=data_context.get_config(
            "gpu_sort_external_run_max_live_bytes", None
        ),
        setup_timeout_s=float(
            data_context.get_config("gpu_sort_setup_timeout_s", 300.0)
        ),
    ).to_actor_dict()


def _communication_environment(data_context: DataContext) -> Dict[str, str]:
    """Resolve the transport environment before any rank imports UCXX.

    Ray installs this mapping as an actor ``runtime_env``.  In particular,
    RAPIDS-MPF must use non-threaded polling on the 16-GPU DGX so lazy
    endpoint creation stays inline while MPF advances the worker explicitly.
    """

    def resolve(
        context_name: str,
        environment_names: Sequence[str],
        default: str,
    ) -> str:
        value = data_context.get_config(context_name, None)
        if value is None:
            value = next(
                (
                    os.environ[name]
                    for name in environment_names
                    if os.environ.get(name) is not None
                ),
                default,
            )
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{context_name} must be a nonempty string.")
        return value

    return {
        "UCX_TLS": resolve(
            "gpu_sort_ucx_tls",
            ("RAY_DATA_GPU_SORT_UCX_TLS", "UCX_TLS"),
            "cuda_copy,cuda_ipc,sm,tcp",
        ),
        "UCX_SOCKADDR_TLS_PRIORITY": resolve(
            "gpu_sort_ucx_sockaddr_tls_priority",
            (
                "RAY_DATA_GPU_SORT_UCX_SOCKADDR_TLS_PRIORITY",
                "UCX_SOCKADDR_TLS_PRIORITY",
            ),
            "tcp",
        ),
        "UCX_MEMTYPE_CACHE": "n",
        "UCX_LOG_LEVEL": "warn",
        "RAPIDSMPF_LOG": "WARN",
        "RAPIDSMPF_UCXX_PROGRESS_MODE": resolve(
            "gpu_sort_ucxx_progress_mode",
            (
                "RAY_DATA_GPU_SORT_UCXX_PROGRESS_MODE",
                "RAPIDSMPF_UCXX_PROGRESS_MODE",
            ),
            # MPF's progress loop advances UCXX explicitly in non-threaded
            # modes.  Keeping endpoint creation inline avoids depending on a
            # separate UCXX progress thread to service its own creation
            # callback before the UCXX deadline.
            "polling",
        ),
        "CUDF_SPILL": "0",
    }


class _RankPool:
    """Fresh, non-detached one-actor-per-GPU rank pool."""

    def __init__(
        self,
        nranks: int,
        key_columns: List[str],
        ascending: List[bool],
        config: Dict[str, Any],
        communication_environment: Dict[str, str],
    ) -> None:
        self.nranks = nranks
        self.key_columns = key_columns
        self.ascending = ascending
        self.config = config
        self.communication_environment = dict(communication_environment)
        self.actors: List[ActorHandle] = []
        self.rank_infos: List[Dict[str, Any]] = []
        self._shutdown_lock = threading.Lock()

    def start(self) -> None:
        from ray.data._internal.gpu_sort.actor import GPUSortActor

        self.actors = [
            GPUSortActor.options(
                num_gpus=1,
                num_cpus=1,
                scheduling_strategy="SPREAD",
                runtime_env={"env_vars": self.communication_environment},
            ).remote(
                nranks=self.nranks,
                index=rank,
                key_columns=self.key_columns,
                ascending=self.ascending,
                num_partitions=self.nranks,
                config=self.config,
                communication_environment=self.communication_environment,
            )
            for rank in range(self.nranks)
        ]
        timeout = self.config["setup_timeout_s"]
        root_rank, address = ray.get(
            self.actors[0].setup_root.remote(), timeout=timeout
        )
        if int(root_rank) != 0:
            raise RuntimeError(
                f"GPU sort communicator assigned root rank {root_rank}, expected 0."
            )
        setup = ray.get(
            [actor.setup_worker.remote(address) for actor in self.actors],
            timeout=timeout,
        )
        actors_by_rank: List[Optional[ActorHandle]] = [None] * self.nranks
        infos_by_rank: List[Optional[Dict[str, Any]]] = [None] * self.nranks
        for actor, result in zip(self.actors, setup):
            rank = int(result["rank"])
            if not 0 <= rank < self.nranks or actors_by_rank[rank] is not None:
                raise RuntimeError(
                    f"GPU sort communicator returned invalid rank {rank}."
                )
            actors_by_rank[rank] = actor
            infos_by_rank[rank] = dict(result)
        if any(actor is None for actor in actors_by_rank):
            raise RuntimeError("GPU sort communicator rank assignment is incomplete.")
        self.actors = [actor for actor in actors_by_rank if actor is not None]
        self.rank_infos = [info for info in infos_by_rank if info is not None]
        ready = ray.get(
            [actor.is_ready.remote() for actor in self.actors], timeout=timeout
        )
        if not all(ready):
            raise RuntimeError("One or more GPU sort ranks failed initialization.")

    def shutdown(self) -> None:
        with self._shutdown_lock:
            actors, self.actors = self.actors, []
        if not actors:
            return
        try:
            ray.get([actor.release.remote() for actor in actors], timeout=10)
        except Exception:
            pass
        finally:
            for actor in actors:
                ray.kill(actor, no_restart=True)

    def shutdown_async(self) -> None:
        """Reap successful one-shot ranks without extending output sealing time."""

        thread = threading.Thread(
            target=self.shutdown,
            name="gpu-sort-rank-reaper",
            daemon=True,
        )
        thread.start()


class GPUSortOperator(PhysicalOperator, SubProgressBarMixin):
    """Unified resident/spillable distributed GPU range sort."""

    def __init__(
        self,
        input_op: PhysicalOperator,
        data_context: DataContext,
        *,
        sort_key: Any,
    ) -> None:
        if sort_key.boundaries is not None:
            raise ValueError("GPU sort does not support explicit `boundaries`.")
        key_columns = list(sort_key.get_columns())
        if not key_columns:
            raise ValueError("GPU sort requires at least one sort key.")
        nranks = _derive_num_ranks(data_context)
        descending = list(sort_key.get_descending())
        config = _operator_config(data_context)
        communication_environment = _communication_environment(data_context)
        super().__init__(
            name=f"GPUSort(keys={key_columns}, ranks={nranks})",
            input_dependencies=[input_op],
            data_context=data_context,
        )
        self._sort_key = sort_key
        self._key_columns = key_columns
        self._config = config
        self._rank_pool = _RankPool(
            nranks,
            key_columns,
            [not value for value in descending],
            config,
            communication_environment,
        )
        self._input_blocks: List[_InputBlock] = []
        self._blocks_by_rank: List[List[_InputBlock]] = [[] for _ in range(nranks)]
        self._assigned_bytes = [0] * nranks
        self._assigned_blocks = [0] * nranks
        self._local_bytes = [0] * nranks
        self._local_blocks = [0] * nranks
        self._wave_target_bytes: Optional[int] = None
        self._wave_count = 0
        self._exchange_subround_count = 0
        self._exchange_rounds_per_wave: List[int] = []
        self._exchange_plan_s = 0.0
        self._prepared_batch_count = 0
        self._prepared_bytes = 0
        self._minimum_modeled_mpf_headroom_bytes: Optional[int] = None
        self._maximum_exchange_destination_bytes = 0
        self._next_exchange_id = 0
        self._ingest_tasks: Dict[int, MetadataOpTask] = {}
        self._next_ingest_task_id = 0
        self._next_block_ordinal = 0
        self._buffered_input_bytes = 0
        self._buffered_input_blocks = 0
        self._peak_buffered_input_bytes = 0
        self._peak_buffered_input_blocks = 0
        self._max_input_block_bytes = 0
        # Store only hexadecimal IDs for proof counters. Keeping ObjectRef
        # instances here would pin every released input in Plasma.
        self._input_ref_ids_received: set[str] = set()
        self._released_input_ref_ids: set[str] = set()
        self._released_input_object_refs = 0
        self._first_input_received_at_ns: Optional[int] = None
        self._last_input_received_at_ns: Optional[int] = None
        self._inputs_complete_at_ns: Optional[int] = None
        self._first_gpu_run_committed_at_ns: Optional[int] = None
        self._last_gpu_run_committed_at_ns: Optional[int] = None
        self._ranks_started_at_ns: Optional[int] = None
        self._finalization_started_at_ns: Optional[int] = None
        self._first_output_bundle_ready_at_ns: Optional[int] = None
        self._last_output_bundle_ready_at_ns: Optional[int] = None
        self._finalization_complete_at_ns: Optional[int] = None
        self._input_rows = 0
        self._input_bytes = 0
        self._input_schema = None
        self._input_stats: List[BlockStats] = []
        self._output_stats: List[BlockStats] = []
        self._output_queue = ReorderingBundleQueue()
        self._extraction_tasks: Dict[int, DataOpTask] = {}
        self._finalization_started = False
        self._finalization_succeeded = False
        self._run_started_at: Optional[float] = None
        self._controller_phases: Dict[str, float] = {}
        self._sample_manifests: List[Dict[str, Any]] = []
        self._sample_quotas: List[int] = []
        self._sample_target_rows = 0
        self._sample_rows = 0
        self._sample_bytes = 0
        self._planning_sample_bytes = 0
        self._sampled_block_count = 0
        self._sample_quota_rows: Dict[str, Any] = {
            "min": 0,
            "median": 0,
            "max": 0,
        }
        self._sample_plan_digest = hashlib.sha256(b"").hexdigest()
        self._sample_index_digest = hashlib.sha256(b"").hexdigest()
        self._boundary_digest = hashlib.sha256(b"").hexdigest()
        self._sampling_subphases = {
            "cpu_sample_construction": 0.0,
            "boundary_sort": 0.0,
            "orchestration_remainder": 0.0,
        }
        self._progress = {"GPU Sample": None, "GPU Sort/Merge": None}

    def start(
        self,
        options: ExecutionOptions,
        block_ref_counter: "BlockRefCounter",
    ) -> None:
        if not options.preserve_order:
            raise RuntimeError(
                "GPU sort requires preserve_order=True so logical block "
                "ordinals remain reproducible."
            )
        super().start(options, block_ref_counter)
        self._run_started_at = time.perf_counter()
        started = time.perf_counter()
        self._rank_pool.start()
        self._controller_phases["startup"] = time.perf_counter() - started
        self._ranks_started_at_ns = time.time_ns()

    def can_add_input(self) -> bool:
        """Apply real upstream backpressure at the configured decoded-byte cap."""

        return (
            self._started
            and not self._inputs_complete
            and self._buffered_input_bytes
            < int(self._config["input_buffer_budget_bytes"])
        )

    def _add_input_inner(self, bundle: RefBundle, input_index: int) -> None:
        if input_index != 0:
            raise ValueError("GPU sort accepts exactly one input dependency.")
        if len(bundle.blocks) != 1:
            raise ValueError(
                "Streaming GPU sort requires singleton RefBundles so its input "
                "budget is bounded by the configured limit plus one input block."
            )
        if not self._rank_pool.actors:
            raise RuntimeError("GPU sort ranks must start before accepting input.")
        self._metrics.on_input_queued(bundle, input_index=0)
        self._input_stats.extend(to_stats(bundle.metadata))
        if self._input_schema is None and bundle.schema is not None:
            self._input_schema = bundle.schema

        blocks: List[_InputBlock] = []
        for entry, block_slice in zip(bundle.blocks, bundle.slices):
            block_ref = entry.ref
            metadata = entry.metadata
            if block_slice is None:
                if metadata.num_rows is None:
                    raise ValueError(
                        "GPU sort deterministic sampling requires exact block "
                        "row metadata."
                    )
                value: Any = block_ref
                rows = int(metadata.num_rows or 0)
                size_bytes = int(metadata.size_bytes or 0)
            else:
                value = (
                    block_ref,
                    int(block_slice.start_offset),
                    int(block_slice.end_offset),
                )
                rows = int(block_slice.num_rows)
                full_rows = int(metadata.num_rows or 0)
                size_bytes = (
                    max(1, math.ceil((metadata.size_bytes or 0) * rows / full_rows))
                    if full_rows
                    else int(metadata.size_bytes or 0)
                )
            block = _InputBlock(
                value=value,
                size_bytes=size_bytes,
                num_rows=rows,
                ordinal=self._next_block_ordinal,
            )
            self._next_block_ordinal += 1
            blocks.append(block)
            # Keep only metadata for deterministic sample-plan telemetry. The
            # ObjectRef itself lives solely in the outstanding actor RPC.
            self._input_blocks.append(
                _InputBlock(
                    value=None,
                    size_bytes=size_bytes,
                    num_rows=rows,
                    ordinal=block.ordinal,
                )
            )
            self._sample_quotas.append(
                min(rows, int(self._config["streaming_sample_rows_per_block"]))
                if rows
                else 0
            )
            self._input_bytes += size_bytes
            self._input_rows += rows

        bundle_bytes = sum(block.size_bytes for block in blocks)
        bundle_blocks = len(blocks)
        rank = min(
            range(len(self._rank_pool.actors)),
            key=lambda item: (
                self._assigned_bytes[item],
                self._assigned_blocks[item],
                item,
            ),
        )
        self._assigned_bytes[rank] += bundle_bytes
        self._assigned_blocks[rank] += bundle_blocks
        self._buffered_input_bytes += bundle_bytes
        self._buffered_input_blocks += bundle_blocks
        self._peak_buffered_input_bytes = max(
            self._peak_buffered_input_bytes, self._buffered_input_bytes
        )
        self._peak_buffered_input_blocks = max(
            self._peak_buffered_input_blocks, self._buffered_input_blocks
        )
        self._max_input_block_bytes = max(
            self._max_input_block_bytes,
            max((block.size_bytes for block in blocks), default=0),
        )
        received_at_ns = time.time_ns()
        if self._first_input_received_at_ns is None:
            self._first_input_received_at_ns = received_at_ns
        self._last_input_received_at_ns = received_at_ns
        self._input_ref_ids_received.update(ref.hex() for ref in bundle.block_refs)

        task_index = self._next_ingest_task_id
        self._next_ingest_task_id += 1
        expected_rows = sum(block.num_rows for block in blocks)
        input_ref_ids = [ref.hex() for ref in bundle.block_refs]
        # The task callback can outlive this stack frame. Keep its one bundle
        # in a mutable holder so completion can explicitly sever the closure's
        # ObjectRef ownership before the executor enters synchronous EOS work.
        bundle_holder = [bundle]
        result_ref = self._rank_pool.actors[rank].ingest_blocks.remote(
            [block.value for block in blocks],
            [block.ordinal for block in blocks],
        )

        def _on_ingest_done() -> None:
            owned_bundle = bundle_holder[0]
            try:
                receipt = dict(ray.get(result_ref))
                if int(receipt.get("rank", -1)) != rank:
                    raise RuntimeError("GPU sort ingest receipt has the wrong rank.")
                if int(receipt.get("rows", -1)) != expected_rows:
                    raise RuntimeError(
                        "GPU sort ingest receipt has the wrong row count."
                    )
                if int(receipt.get("input_blocks", -1)) != bundle_blocks:
                    raise RuntimeError(
                        "GPU sort ingest receipt has the wrong block count."
                    )
                committed_at = receipt.get("first_run_committed_at_ns")
                if committed_at is not None:
                    committed_at = int(committed_at)
                    self._first_gpu_run_committed_at_ns = (
                        committed_at
                        if self._first_gpu_run_committed_at_ns is None
                        else min(self._first_gpu_run_committed_at_ns, committed_at)
                    )
                last_committed = receipt.get("last_run_committed_at_ns")
                if last_committed is not None:
                    self._last_gpu_run_committed_at_ns = max(
                        self._last_gpu_run_committed_at_ns or 0,
                        int(last_committed),
                    )
                for ref_id in input_ref_ids:
                    if ref_id not in self._released_input_ref_ids:
                        self._released_input_ref_ids.add(ref_id)
                        self._released_input_object_refs += 1
            finally:
                self._buffered_input_bytes -= bundle_bytes
                self._buffered_input_blocks -= bundle_blocks
                self._metrics.on_input_dequeued(owned_bundle, input_index=0)
                owned_bundle.destroy_if_owned()
                bundle_holder.clear()
                self._ingest_tasks.pop(task_index, None)

        self._ingest_tasks[task_index] = MetadataOpTask(
            task_index=task_index,
            object_ref=result_ref,
            task_done_callback=_on_ingest_done,
        )

    def all_inputs_done(self) -> None:
        self._inputs_complete_at_ns = time.time_ns()
        super().all_inputs_done()

    def _finalize_streaming_sample(self) -> Tuple[Any, Any]:
        """Merge rank reservoirs and freeze range boundaries after EOS."""

        from ray.data._internal.gpu_sort.streaming_sample import (
            finalize_priority_sample,
        )

        construction_started = time.perf_counter()
        manifests = ray.get(
            [actor.finalize_ingest.remote() for actor in self._rank_pool.actors],
            timeout=self._config["setup_timeout_s"],
        )
        manifests_by_rank = {int(item["rank"]): dict(item) for item in manifests}
        if len(manifests_by_rank) != len(self._rank_pool.actors):
            raise RuntimeError("GPU sort ingest manifests are incomplete.")
        self._sample_manifests = [
            manifests_by_rank[rank] for rank in range(len(self._rank_pool.actors))
        ]
        schema = self._input_schema or next(
            (
                item.get("schema")
                for item in self._sample_manifests
                if item.get("schema") is not None
            ),
            None,
        )
        if schema is None:
            raise ValueError("GPU sort could not determine the input Arrow schema.")
        _validate_gpu_schema(schema, self._key_columns)
        self._sort_key.validate_schema(getattr(schema, "base_schema", schema))
        candidate_rows = sum(
            int(item.get("sample_candidate_rows", 0) or 0)
            for item in self._sample_manifests
        )
        self._sample_target_rows = min(int(self._config["sample_size"]), candidate_rows)
        rank_samples = [
            item["sample"]
            for item in self._sample_manifests
            if item.get("sample") is not None and item["sample"].num_rows
        ]
        if rank_samples:
            first_sample = rank_samples[0]
            coordinate_names = first_sample.column_names[-4:]
            weight_name, block_name, stratum_name, index_name = coordinate_names
            sample = finalize_priority_sample(
                rank_samples,
                candidate_rows=candidate_rows,
                capacity=int(self._config["sample_size"]),
                seed=int(self._config["sample_seed"]),
                weight_name=weight_name,
                block_name=block_name,
                stratum_name=stratum_name,
                index_name=index_name,
            )
            samples = [sample]
        else:
            samples = []

        self._sampled_block_count = sum(
            int(item.get("sampled_block_count", 0) or 0)
            for item in self._sample_manifests
        )
        self._sample_quota_rows = _sample_quota_summary(self._sample_quotas)
        sample_plan = {
            "scheme": "deterministic_streaming_priority_stratified",
            "version": 1,
            "seed": int(self._config["sample_seed"]),
            "capacity": int(self._config["sample_size"]),
            "candidate_rows": candidate_rows,
            "blocks": [
                [int(block.ordinal), int(block.num_rows), int(quota)]
                for block, quota in zip(self._input_blocks, self._sample_quotas)
            ],
        }
        self._sample_plan_digest = hashlib.sha256(
            json.dumps(sample_plan, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

        self._blocks_by_rank = []
        run_ordinal = 0
        for rank, item in enumerate(self._sample_manifests):
            rank_runs = []
            for run in item.get("runs", ()):
                rank_runs.append(
                    _InputBlock(
                        value=int(run["run_id"]),
                        size_bytes=int(run["bytes"]),
                        num_rows=int(run["rows"]),
                        ordinal=run_ordinal,
                    )
                )
                run_ordinal += 1
            self._blocks_by_rank.append(rank_runs)

        self._sampling_subphases["cpu_sample_construction"] = (
            time.perf_counter() - construction_started
        )
        result = ray.get(
            self._rank_pool.actors[0].compute_boundaries.remote(samples, schema),
            timeout=self._config["setup_timeout_s"],
        )
        self._sample_rows = int(result.get("sample_rows", 0) or 0)
        self._sample_bytes = int(result.get("sample_bytes", 0) or 0)
        self._planning_sample_bytes = int(result.get("planning_sample_bytes", 0) or 0)
        if self._sample_rows != self._sample_target_rows:
            raise RuntimeError(
                "GPU sort sampling did not produce its exact global target: "
                f"expected {self._sample_target_rows}, got {self._sample_rows}."
            )
        self._sample_index_digest = str(result["sample_index_digest"])
        self._boundary_digest = str(result["boundary_digest"])
        self._sampling_subphases["boundary_sort"] = float(
            result.get("boundary_sort_s", 0.0) or 0.0
        )
        return schema, result["boundaries"]

    def _plan_waves(self) -> List[List[List[Any]]]:
        budget = self._config["residency_budget_bytes"]
        self._wave_target_bytes = _wave_target_bytes(
            self._blocks_by_rank,
            explicit_residency_budget_bytes=budget,
            actor_usable_budgets=[
                int(info.get("usable_memory_budget_bytes", 0) or 0)
                for info in self._rank_pool.rank_infos
            ],
            auto_wave_fraction=float(self._config["auto_wave_fraction"]),
        )
        waves = _make_waves(self._blocks_by_rank, self._wave_target_bytes)
        self._wave_count = len(waves)
        return waves

    def _try_finalize(self) -> None:
        if (
            self._finalization_started
            or not self._inputs_complete
            or self._ingest_tasks
        ):
            return
        self._finalization_started = True
        self._finalization_started_at_ns = time.time_ns()
        if not self._input_blocks:
            try:
                diagnostics = ray.get(
                    [
                        actor.release.remote(strict=True)
                        for actor in self._rank_pool.actors
                    ]
                )
            except Exception:
                self._rank_pool.shutdown()
                raise
            self._finalization_succeeded = True
            self._finalization_complete_at_ns = time.time_ns()
            self._publish_diagnostics(diagnostics)
            self._rank_pool.shutdown_async()
            return

        try:
            started = time.perf_counter()
            schema, boundaries = self._finalize_streaming_sample()
            self._controller_phases["sampling"] = time.perf_counter() - started
            self._sampling_subphases["orchestration_remainder"] = max(
                0.0,
                self._controller_phases["sampling"]
                - self._sampling_subphases["cpu_sample_construction"]
                - self._sampling_subphases["boundary_sort"],
            )

            started = time.perf_counter()
            ray.get(
                [
                    actor.install_plan.remote(schema, boundaries)
                    for actor in self._rank_pool.actors
                ],
                timeout=self._config["setup_timeout_s"],
            )
            waves = self._plan_waves()
            source_run_geometry = [
                {int(block.value): block for block in blocks}
                for blocks in self._blocks_by_rank
            ]
            for wave_id, blocks_for_ranks in enumerate(waves):
                wave_geometry = [
                    [source_run_geometry[rank][int(run_id)] for run_id in run_ids]
                    for rank, run_ids in enumerate(blocks_for_ranks)
                ]
                received_rows_for_wave = 0
                prepared = ray.get(
                    [
                        actor.prepare_ingested_runs.remote(wave_id, blocks)
                        for actor, blocks in zip(
                            self._rank_pool.actors, blocks_for_ranks
                        )
                    ]
                )
                rounds_for_wave = 0
                while True:
                    prepared_by_rank: List[Optional[Dict[str, Any]]] = [None] * len(
                        self._rank_pool.actors
                    )
                    for item in prepared:
                        rank = int(item["rank"])
                        if (
                            not 0 <= rank < len(prepared_by_rank)
                            or prepared_by_rank[rank] is not None
                        ):
                            raise RuntimeError(
                                f"GPU sort received an invalid prepared rank {rank}."
                            )
                        prepared_by_rank[rank] = dict(item)
                    if any(item is None for item in prepared_by_rank):
                        raise RuntimeError(
                            "GPU sort prepared-wave ranks are incomplete."
                        )
                    manifests = [item for item in prepared_by_rank if item is not None]
                    pending = [list(item.get("batches", ())) for item in manifests]
                    allocated = [
                        int(item.get("current_allocated_bytes", 0))
                        for item in manifests
                    ]
                    pool_max = [
                        int(item.get("pool_max_bytes", 0)) for item in manifests
                    ]
                    source_complete = [
                        bool(item.get("source_complete", False)) for item in manifests
                    ]
                    self._prepared_batch_count += sum(len(items) for items in pending)
                    self._prepared_bytes += sum(
                        sum(int(value) for value in batch["destination_bytes"])
                        for items in pending
                        for batch in items
                    )
                    if not any(pending):
                        if not all(source_complete):
                            raise GPUSortCapacityError(
                                "GPU sort could not prepare one bounded source batch."
                            )
                        exchange_id = self._next_exchange_id
                        receipts = ray.get(
                            [
                                actor.exchange_prepared_round.remote(
                                    wave_id, exchange_id, [], True
                                )
                                for actor in self._rank_pool.actors
                            ]
                        )
                        ordered_receipts = _ordered_exchange_receipts(
                            receipts,
                            nranks=len(self._rank_pool.actors),
                            wave_id=wave_id,
                            exchange_id=exchange_id,
                        )
                        received_rows_for_wave += sum(
                            int(item["received_rows"]) for item in ordered_receipts
                        )
                        self._next_exchange_id += 1
                        rounds_for_wave += 1
                    while any(pending):
                        plan_started = time.perf_counter()
                        plan = _plan_exchange_round(
                            pending,
                            current_allocated_bytes=allocated,
                            pool_max_bytes=pool_max,
                        )
                        self._exchange_plan_s += time.perf_counter() - plan_started
                        selected = [set(values) for values in plan.batch_ids_by_rank]
                        remaining = [
                            [
                                batch
                                for batch in items
                                if int(batch["batch_id"]) not in selected[rank]
                            ]
                            for rank, items in enumerate(pending)
                        ]
                        final_subround = not any(remaining) and all(source_complete)
                        exchange_id = self._next_exchange_id
                        receipts = ray.get(
                            [
                                actor.exchange_prepared_round.remote(
                                    wave_id,
                                    exchange_id,
                                    list(plan.batch_ids_by_rank[rank]),
                                    final_subround,
                                )
                                for rank, actor in enumerate(self._rank_pool.actors)
                            ]
                        )
                        self._next_exchange_id += 1
                        rounds_for_wave += 1
                        ordered_receipts = _ordered_exchange_receipts(
                            receipts,
                            nranks=len(self._rank_pool.actors),
                            wave_id=wave_id,
                            exchange_id=exchange_id,
                        )
                        received_rows_for_wave += sum(
                            int(item["received_rows"]) for item in ordered_receipts
                        )
                        allocated = [
                            int(item["current_allocated_bytes"])
                            for item in ordered_receipts
                        ]
                        pending = remaining
                        self._minimum_modeled_mpf_headroom_bytes = min(
                            plan.modeled_headroom_bytes
                            if self._minimum_modeled_mpf_headroom_bytes is None
                            else (
                                self._minimum_modeled_mpf_headroom_bytes,
                                *plan.modeled_headroom_bytes,
                            )
                        )
                        self._maximum_exchange_destination_bytes = max(
                            self._maximum_exchange_destination_bytes,
                            max(plan.incoming_bytes, default=0),
                        )
                    if all(source_complete):
                        break
                    prepared = ray.get(
                        [
                            actor.prepare_more.remote(wave_id)
                            for actor in self._rank_pool.actors
                        ]
                    )
                _validate_source_wave_rows(
                    wave_geometry,
                    wave_id=wave_id,
                    received_rows=received_rows_for_wave,
                )
                commit_receipts = ray.get(
                    [
                        actor.commit_source_wave.remote(wave_id)
                        for actor in self._rank_pool.actors
                    ]
                )
                _validate_source_wave_commit(
                    commit_receipts,
                    wave_geometry,
                    wave_id=wave_id,
                    received_rows=received_rows_for_wave,
                )
                self._exchange_subround_count += rounds_for_wave
                self._exchange_rounds_per_wave.append(rounds_for_wave)
            self._controller_phases["partition_and_exchange"] = (
                time.perf_counter() - started
            )
            self._schedule_extraction()
        except Exception:
            self._rank_pool.shutdown()
            raise

    def _schedule_extraction(self) -> None:
        for rank, actor in enumerate(self._rank_pool.actors):

            def _on_bundle_ready(bundle: RefBundle, rank: int = rank) -> None:
                ready_at_ns = time.time_ns()
                if self._first_output_bundle_ready_at_ns is None:
                    self._first_output_bundle_ready_at_ns = ready_at_ns
                self._last_output_bundle_ready_at_ns = ready_at_ns
                schema = bundle.schema
                metadata = schema.metadata if schema is not None else None
                partition_id = rank
                if metadata and GPU_SORT_PARTITION_ID_KEY in metadata:
                    partition_id = int(metadata[GPU_SORT_PARTITION_ID_KEY].decode())
                if partition_id != rank:
                    raise RuntimeError(
                        "GPU sort rank emitted an output for a nonlocal ordered "
                        f"partition: rank={rank}, partition={partition_id}."
                    )
                if metadata:
                    clean = {
                        key: value
                        for key, value in metadata.items()
                        if key
                        not in (GPU_SORT_PARTITION_ID_KEY, GPU_SORT_DIAGNOSTICS_KEY)
                    }
                    schema = schema.with_metadata(clean or None)
                    bundle = RefBundle(
                        bundle.blocks,
                        schema=schema,
                        owns_blocks=bundle.owns_blocks,
                        slices=bundle.slices,
                    )
                self._output_queue.add(bundle, key=rank)
                self._metrics.on_output_queued(bundle)
                self._metrics.on_task_output_generated(rank, bundle)
                progress = self._progress["GPU Sort/Merge"]
                if progress is not None:
                    progress.update(
                        increment=bundle.num_rows() or 0, total=self._input_rows
                    )

            def _on_done(
                exc: Optional[Exception],
                worker_stats=None,
                driver_stats=None,
                rank: int = rank,
            ) -> None:
                self._extraction_tasks.pop(rank, None)
                self._output_queue.finalize(key=rank)
                self._metrics.on_task_finished(
                    task_index=rank,
                    exception=exc,
                    task_exec_stats=worker_stats,
                    task_exec_driver_stats=driver_stats,
                )
                if exc is not None:
                    self._rank_pool.shutdown()
                    return
                if not self._extraction_tasks:
                    diagnostics = ray.get(
                        [actor.diagnostics.remote() for actor in self._rank_pool.actors]
                    )
                    self._finalization_succeeded = True
                    self._finalization_complete_at_ns = time.time_ns()
                    self._publish_diagnostics(diagnostics)
                    self._rank_pool.shutdown_async()

            generator = actor.finish_and_extract.options(
                num_returns="streaming"
            ).remote()
            task = DataOpTask(
                task_index=rank,
                streaming_gen=generator,
                block_ref_counter=self._block_ref_counter,
                producer_id=self.id,
                output_ready_callback=_on_bundle_ready,
                task_done_callback=functools.partial(_on_done, rank=rank),
                operator_name=self.name,
            )
            self._extraction_tasks[rank] = task
            self._metrics.on_task_submitted(
                rank, RefBundle([], schema=None, owns_blocks=False), task.get_task_id()
            )

    def _publish_diagnostics(self, diagnostics: List[Dict[str, Any]]) -> None:
        global LAST_RUN_STATS

        ranks: List[Dict[str, Any]] = []
        for default_rank, raw in enumerate(diagnostics):
            item = dict(raw or {})
            item.setdefault("rank", default_rank)
            item.setdefault(
                "node_id",
                self._rank_pool.rank_infos[default_rank].get("node_id", "")
                if default_rank < len(self._rank_pool.rank_infos)
                else "",
            )
            item.setdefault(
                "usable_memory_budget_bytes",
                self._rank_pool.rank_infos[default_rank].get(
                    "usable_memory_budget_bytes", 0
                )
                if default_rank < len(self._rank_pool.rank_infos)
                else 0,
            )
            item.setdefault("peak_device_bytes", 0)
            item.setdefault("run_store", self._config["external_run_store"])
            item.setdefault("input_bytes", self._assigned_bytes[default_rank])
            item.setdefault("input_blocks", self._assigned_blocks[default_rank])
            item.setdefault("local_input_bytes", self._local_bytes[default_rank])
            item.setdefault("local_input_blocks", self._local_blocks[default_rank])
            for name in (
                "output_bytes",
                "externalized_bytes",
                "externalized_rows",
                "ingest_rpc_count",
                "source_run_count",
                "source_run_bytes",
                "source_run_rows",
                "initial_run_count",
                "merge_pass_count",
                "replacement_run_count",
                "direct_final_merge_count",
                "resident_final_fallback_count",
                "concat_oom_fallback_count",
                "run_sort_oom_retry_count",
                "prepared_batch_count",
                "prepared_bytes",
                "preparation_retry_count",
                "exchange_subround_count",
                "forced_subround_run_count",
                "h2d_bytes",
                "planning_h2d_bytes",
                "d2h_bytes",
                "plasma_read_bytes",
                "plasma_write_bytes",
                "plasma_intermediate_read_bytes",
                "plasma_intermediate_write_bytes",
                "plasma_intermediate_read_calls",
                "plasma_intermediate_write_calls",
                "plasma_output_write_bytes",
                "plasma_output_write_calls",
                "plasma_output_write_s",
                "local_run_write_bytes",
                "local_run_read_bytes",
                "local_run_physical_write_bytes",
                "local_run_physical_read_bytes",
                "local_run_write_calls",
                "local_run_read_calls",
                "local_run_sync_calls",
                "local_run_restore_s",
                "local_run_live_bytes",
                "local_run_peak_bytes",
                "local_run_live_files",
                "local_run_peak_files",
                "local_run_write_errors",
                "local_run_read_errors",
                "local_run_cleanup_errors",
                "local_run_cleanup_pending_files",
                "local_run_cleanup_pending_bytes",
                "local_run_cleanup_calls",
                "local_run_cleanup_files",
                "local_run_cleanup_bytes",
                "mpf_host_spill_bytes",
                "ray_disk_spill_bytes",
                "cpu_sort_rows",
                "cpu_merge_rows",
                "fallback_count",
            ):
                item.setdefault(name, 0)
            item.setdefault("phases_s", {})
            ranks.append(item)

        def total(name: str) -> int:
            return sum(int(item.get(name, 0) or 0) for item in ranks)

        def total_float(name: str) -> float:
            return sum(float(item.get(name, 0.0) or 0.0) for item in ranks)

        phase_names = (
            "partition",
            "mpf_shuffle",
            "run_sort",
            "gpu_merge",
            "arrow_conversion",
            "plasma_seal",
        )
        phases = {
            name: max(
                (float(item.get("phases_s", {}).get(name, 0) or 0) for item in ranks),
                default=0.0,
            )
            for name in phase_names
        }
        phases["sampling"] = float(self._controller_phases.get("sampling", 0.0))
        elapsed = (
            time.perf_counter() - self._run_started_at
            if self._run_started_at is not None
            else 0.0
        )
        phases["orchestration"] = max(0.0, elapsed - sum(phases.values()))
        externalized_bytes = total("externalized_bytes")
        budgets = [int(item.get("memory_budget_bytes", 0) or 0) for item in ranks]
        configured_budget = int(self._config["residency_budget_bytes"] or 0)
        first_times = [
            float(item["first_externalize_s"])
            for item in ranks
            if item.get("first_externalize_s") is not None
        ]
        first_waves = [
            int(item["first_externalize_wave"])
            for item in ranks
            if item.get("first_externalize_wave") is not None
        ]
        LAST_RUN_STATS = {
            "mode": "external" if externalized_bytes else "resident",
            "run_store": self._config["external_run_store"],
            "streaming_input": True,
            "sampling_mode": "one_pass_rank_reservoir",
            "sampling_scheme": "deterministic_streaming_priority_stratified",
            "sampling_scheme_version": 1,
            "sample_seed": int(self._config["sample_seed"]),
            "sample_target_rows": self._sample_target_rows,
            "sample_rows": self._sample_rows,
            "sample_bytes": self._sample_bytes,
            "planning_sample_bytes": self._planning_sample_bytes,
            "sampled_block_count": self._sampled_block_count,
            "sample_quota_rows": dict(self._sample_quota_rows),
            "sample_plan_digest": self._sample_plan_digest,
            "sample_index_digest": self._sample_index_digest,
            "boundary_digest": self._boundary_digest,
            "planning_h2d_bytes": total("planning_h2d_bytes"),
            "sampling_subphases_s": dict(self._sampling_subphases),
            "memory_budget_bytes": max(budgets, default=0) or configured_budget,
            "peak_device_bytes": max(
                (int(item["peak_device_bytes"]) for item in ranks), default=0
            ),
            "input_rows": self._input_rows,
            "input_bytes": self._input_bytes,
            "input_buffer_budget_bytes": int(self._config["input_buffer_budget_bytes"]),
            "peak_buffered_input_bytes": self._peak_buffered_input_bytes,
            "peak_buffered_input_blocks": self._peak_buffered_input_blocks,
            "max_input_block_bytes": self._max_input_block_bytes,
            "input_buffer_within_bound": self._peak_buffered_input_bytes
            <= int(self._config["input_buffer_budget_bytes"])
            + self._max_input_block_bytes,
            "input_object_refs_received": len(self._input_ref_ids_received),
            "released_input_object_refs": self._released_input_object_refs,
            "all_input_object_refs_released": self._released_input_object_refs
            == len(self._input_ref_ids_received),
            "ranks_started_at_ns": self._ranks_started_at_ns,
            "first_input_received_at_ns": self._first_input_received_at_ns,
            "last_input_received_at_ns": self._last_input_received_at_ns,
            "inputs_complete_at_ns": self._inputs_complete_at_ns,
            "first_gpu_run_committed_at_ns": (self._first_gpu_run_committed_at_ns),
            "last_gpu_run_committed_at_ns": self._last_gpu_run_committed_at_ns,
            "finalization_started_at_ns": self._finalization_started_at_ns,
            "first_output_bundle_ready_at_ns": (self._first_output_bundle_ready_at_ns),
            "last_output_bundle_ready_at_ns": self._last_output_bundle_ready_at_ns,
            "finalization_complete_at_ns": self._finalization_complete_at_ns,
            "gpu_processing_began_before_eos": (
                self._first_gpu_run_committed_at_ns is not None
                and self._inputs_complete_at_ns is not None
                and self._first_gpu_run_committed_at_ns < self._inputs_complete_at_ns
            ),
            "auto_wave_fraction": float(self._config["auto_wave_fraction"]),
            "wave_target_bytes": self._wave_target_bytes,
            "wave_count": self._wave_count,
            "exchange_subround_count": self._exchange_subround_count,
            "exchange_rounds_per_wave": {
                "min": min(self._exchange_rounds_per_wave, default=0),
                "median": (
                    sorted(self._exchange_rounds_per_wave)[
                        len(self._exchange_rounds_per_wave) // 2
                    ]
                    if self._exchange_rounds_per_wave
                    else 0
                ),
                "max": max(self._exchange_rounds_per_wave, default=0),
            },
            "exchange_plan_s": self._exchange_plan_s,
            "prepared_batch_count": self._prepared_batch_count,
            "prepared_bytes": self._prepared_bytes,
            "preparation_retry_count": total("preparation_retry_count"),
            "minimum_modeled_mpf_headroom_bytes": (
                self._minimum_modeled_mpf_headroom_bytes
            ),
            "maximum_exchange_destination_bytes": (
                self._maximum_exchange_destination_bytes
            ),
            "forced_subround_run_count": total("forced_subround_run_count"),
            "ranks": ranks,
            "externalized_bytes": externalized_bytes,
            "externalized_rows": total("externalized_rows"),
            "first_externalize_s": min(first_times) if first_times else None,
            "first_externalize_wave": min(first_waves) if first_waves else None,
            "initial_run_count": total("initial_run_count"),
            "source_run_count": total("source_run_count"),
            "source_run_bytes": total("source_run_bytes"),
            "source_run_rows": total("source_run_rows"),
            "ingest_rpc_count": total("ingest_rpc_count"),
            "ingest_wall_s_rank_sum": total_float("ingest_wall_s"),
            "streaming_sample_construction_s_rank_sum": total_float(
                "streaming_sample_construction_s"
            ),
            "merge_pass_count": max(
                (int(item["merge_pass_count"]) for item in ranks), default=0
            ),
            "replacement_run_count": total("replacement_run_count"),
            "direct_final_merge_count": total("direct_final_merge_count"),
            "resident_final_fallback_count": total("resident_final_fallback_count"),
            "concat_oom_fallback_count": total("concat_oom_fallback_count"),
            "run_sort_oom_retry_count": total("run_sort_oom_retry_count"),
            "h2d_bytes": total("h2d_bytes"),
            "d2h_bytes": total("d2h_bytes"),
            "plasma_read_bytes": total("plasma_read_bytes"),
            "plasma_write_bytes": total("plasma_write_bytes"),
            "plasma_intermediate_read_bytes": total("plasma_intermediate_read_bytes"),
            "plasma_intermediate_write_bytes": total("plasma_intermediate_write_bytes"),
            "plasma_intermediate_read_calls": total("plasma_intermediate_read_calls"),
            "plasma_intermediate_write_calls": total("plasma_intermediate_write_calls"),
            "plasma_intermediate_read_s": total_float("plasma_intermediate_read_s"),
            "plasma_intermediate_write_s": total_float("plasma_intermediate_write_s"),
            "plasma_output_write_bytes": total("plasma_output_write_bytes"),
            "plasma_output_write_calls": total("plasma_output_write_calls"),
            "plasma_output_write_s": total_float("plasma_output_write_s"),
            "local_run_write_bytes": total("local_run_write_bytes"),
            "local_run_read_bytes": total("local_run_read_bytes"),
            "local_run_physical_write_bytes": total("local_run_physical_write_bytes"),
            "local_run_physical_read_bytes": total("local_run_physical_read_bytes"),
            "local_run_write_calls": total("local_run_write_calls"),
            "local_run_read_calls": total("local_run_read_calls"),
            "local_run_write_s": total_float("local_run_write_s"),
            "local_run_read_s": total_float("local_run_read_s"),
            "local_run_sync_s": total_float("local_run_sync_s"),
            "local_run_sync_calls": total("local_run_sync_calls"),
            "local_run_restore_s": total_float("local_run_restore_s"),
            "local_run_live_bytes": total("local_run_live_bytes"),
            "local_run_peak_bytes": total("local_run_peak_bytes"),
            "local_run_live_files": total("local_run_live_files"),
            "local_run_peak_files": total("local_run_peak_files"),
            "local_run_write_errors": total("local_run_write_errors"),
            "local_run_read_errors": total("local_run_read_errors"),
            "local_run_cleanup_errors": total("local_run_cleanup_errors"),
            "local_run_cleanup_pending_files": total("local_run_cleanup_pending_files"),
            "local_run_cleanup_pending_bytes": total("local_run_cleanup_pending_bytes"),
            "local_run_cleanup_calls": total("local_run_cleanup_calls"),
            "local_run_cleanup_files": total("local_run_cleanup_files"),
            "local_run_cleanup_bytes": total("local_run_cleanup_bytes"),
            "mpf_host_spill_bytes": total("mpf_host_spill_bytes"),
            "ray_disk_spill_bytes": total("ray_disk_spill_bytes"),
            "cpu_sort_rows": total("cpu_sort_rows"),
            "cpu_merge_rows": total("cpu_merge_rows"),
            "fallback_count": total("fallback_count"),
            "phases_s": phases,
            "controller_phases_s": dict(self._controller_phases),
            "total_s": elapsed,
        }

    def has_next(self) -> bool:
        self._try_finalize()
        return self._output_queue.has_next()

    def _get_next_inner(self) -> RefBundle:
        bundle = self._output_queue.get_next()
        self._metrics.on_output_dequeued(bundle)
        self._output_stats.extend(to_stats(bundle.metadata))
        return bundle

    def get_active_tasks(self) -> List[OpTask]:
        self._try_finalize()
        return list(self._ingest_tasks.values()) + list(self._extraction_tasks.values())

    def has_completed(self) -> bool:
        return (
            self._finalization_started
            and not self._ingest_tasks
            and not self._extraction_tasks
            and super().has_completed()
        )

    def _do_shutdown(self, force: bool = False) -> None:
        # ``PhysicalOperator._do_shutdown`` asks for active tasks while
        # cancelling them. Prevent that error path from re-entering EOS
        # finalization after this method has already torn down the rank pool.
        self._finalization_started = True
        self._rank_pool.shutdown()
        self._ingest_tasks.clear()
        self._extraction_tasks.clear()
        super()._do_shutdown(force)

    def current_logical_usage(self) -> ExecutionResources:
        return ExecutionResources(gpu=len(self._rank_pool.actors))

    @property
    def base_resource_usage(self) -> ExecutionResources:
        return ExecutionResources(gpu=self._rank_pool.nranks)

    def incremental_resource_usage(self) -> ExecutionResources:
        # Ingest RPCs run on the already-reserved one-GPU rank actors; they do
        # not require an additional schedulable GPU per input bundle.
        return ExecutionResources()

    def get_sub_progress_bar_names(self) -> List[str]:
        return list(self._progress)

    def set_sub_progress_bar(self, name: str, pg: Any) -> None:
        if name in self._progress:
            self._progress[name] = pg

    def get_stats(self) -> Dict[str, List[BlockStats]]:
        return {
            f"{self.name}_input": self._input_stats,
            f"{self.name}_output": self._output_stats,
        }
