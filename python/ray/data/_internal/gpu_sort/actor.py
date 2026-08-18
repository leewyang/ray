"""Ray actor wrapper that imports GPU libraries only after GPU placement."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List

import ray

logger = logging.getLogger(__name__)


def _resolve_blocks(entries: List[Any]) -> List[Any]:
    """Resolve block refs and apply optional ``(ref, start, end)`` row slices."""

    refs = []
    locations = []
    resolved = list(entries)
    for index, entry in enumerate(entries):
        value = entry[0] if isinstance(entry, tuple) and len(entry) == 3 else entry
        if isinstance(value, ray.ObjectRef):
            refs.append(value)
            locations.append(index)
    if refs:
        for index, value in zip(locations, ray.get(refs)):
            entry = entries[index]
            resolved[index] = (
                (value, entry[1], entry[2])
                if isinstance(entry, tuple) and len(entry) == 3
                else value
            )

    result = []
    for entry in resolved:
        if isinstance(entry, tuple) and len(entry) == 3:
            block, start, end = entry
            from ray.data.block import BlockAccessor

            table = BlockAccessor.for_block(block).to_arrow()
            result.append(table.slice(int(start), int(end) - int(start)))
        else:
            result.append(entry)
    return result


@ray.remote(num_gpus=1)
class GPUSortActor:
    """Run one GPU-sort rank after Ray assigns this actor a GPU."""

    def __init__(
        self,
        *,
        nranks: int,
        index: int,
        key_columns: List[str],
        ascending: List[bool],
        num_partitions: int,
        config: Dict[str, Any],
    ) -> None:
        try:
            from ray.data._internal.gpu_sort.backend import get_backend_class

            backend = get_backend_class()
        except ImportError as exc:
            raise ImportError(
                "Ray Data GPU sort requires RAPIDS-MPF on every GPU node."
            ) from exc
        self._backend = backend(
            nranks=nranks,
            index=index,
            key_columns=key_columns,
            ascending=ascending,
            num_partitions=num_partitions,
            config=config,
        )
        self._node_id = ray.get_runtime_context().get_node_id()

    def setup_root(self) -> tuple[int, bytes]:
        return self._backend.setup_root()

    def setup_worker(self, root_address_bytes: bytes) -> Dict[str, Any]:
        result = dict(self._backend.setup_worker(root_address_bytes))
        result["node_id"] = self._node_id
        result["usable_memory_budget_bytes"] = int(
            result.get("memory_budget_bytes", 0) or 0
        )
        return result

    def sample_blocks(
        self,
        blocks: List[Any],
        block_ordinals: List[int],
        sample_quotas: List[int],
        seed: int,
    ) -> Dict[str, Any]:
        return self._backend.sample_blocks(
            _resolve_blocks(blocks),
            block_ordinals=block_ordinals,
            sample_quotas=sample_quotas,
            seed=seed,
        )

    def compute_boundaries(self, samples: List[Any], schema: Any) -> Dict[str, Any]:
        return self._backend.compute_boundaries(samples, schema)

    def install_plan(self, schema: Any, boundaries: Any) -> None:
        self._backend.install_plan(schema, boundaries)

    def process_wave(self, wave_id: int, blocks: List[Any]) -> None:
        self._backend.process_wave(wave_id, _resolve_blocks(blocks))

    def finish_and_extract(self) -> Iterator[Any]:
        try:
            yield from self._backend.finish_and_extract()
        except BaseException:
            # Cleanup failures must not hide the sort or cancellation error.
            try:
                self._backend.release()
            except BaseException:
                logger.debug(
                    "GPU sort cleanup failed after output extraction failed.",
                    exc_info=True,
                )
            raise

    def release(self) -> None:
        self._backend.release()
