"""Small, serializable configuration for Ray Data's GPU range sort."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional


# The observed MPF progress-thread failure requested 16.762 MiB. Ninety-six
# MiB preserves that allocation plus more than the selected 64 MiB margin.
MPF_PROGRESS_RESERVE_BYTES = 96 << 20
# PackedData send owners move into MPF. Keep room for the receive buffer and
# its unpack handoff in addition to the separately reserved progress memory.
MPF_RECEIVE_BUFFER_OWNERS = 2


def _positive_bytes(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("GPU sort memory budget must be a byte count.")
    if isinstance(value, int):
        result = value
    else:
        text = str(value).strip().lower().replace(" ", "")
        units = {
            "tib": 1 << 40,
            "tb": 1 << 40,
            "gib": 1 << 30,
            "gb": 1 << 30,
            "mib": 1 << 20,
            "mb": 1 << 20,
            "kib": 1 << 10,
            "kb": 1 << 10,
            "b": 1,
        }
        result = None
        for suffix, multiplier in units.items():
            if text.endswith(suffix):
                result = int(float(text[: -len(suffix)]) * multiplier)
                break
        if result is None:
            result = int(text)
    if result <= 0:
        raise ValueError("GPU sort memory budget must be positive.")
    return result


def _nonnegative_bytes(value: Any) -> int:
    if value is None or value == "":
        return 0
    if isinstance(value, bool):
        raise ValueError("GPU sort memory budget must be a byte count.")
    if value == 0:
        return 0
    result = _positive_bytes(value)
    assert result is not None
    return result


@dataclass(frozen=True)
class GPUSortConfig:
    """Internal controls shared by the controller and one-GPU actors.

    ``residency_budget_bytes`` is a proactive payload-residency watermark, not
    a second allocator pool.  RMM retains enough room to sort or merge the run
    that caused the transition to external execution.
    """

    sample_size: int = 1 << 16
    sample_seed: int = 0
    # Maximum decoded input bytes whose actor calls may be outstanding on the
    # driver. ``PhysicalOperator.can_add_input`` applies backpressure at this
    # watermark; one already-produced upstream block may take the live total
    # above it.
    input_buffer_budget_bytes: int = 16 << 30
    # Each logical block contributes a small deterministic stratified sample.
    # A bounded global priority reservoir reduces these candidates to
    # ``sample_size`` after end-of-input without replaying the source.
    streaming_sample_rows_per_block: int = 64
    # Preallocate the bounded pool as one arena. Growing from a smaller pool
    # can leave enough total bytes but no contiguous segment for run sorting.
    rmm_initial_fraction: float = 0.85
    rmm_max_fraction: float = 0.85
    residency_budget_bytes: Optional[int] = None
    # Fraction of the smallest actor-reported usable memory budget assigned to
    # one automatic source wave.  This is deliberately internal: callers that
    # need an exact capacity experiment should continue to use the explicit
    # residency budget.
    auto_wave_fraction: float = 0.50
    exchange_batch_bytes: int = 512 << 20
    merge_fan_in: int = 4
    run_chunk_bytes: int = 512 << 20
    # Intermediate external runs remain in Plasma unless a benchmark or
    # deployment explicitly selects actor-local Arrow IPC files.
    external_run_store: str = "plasma"
    external_run_directory: Optional[str] = None
    external_run_id: Optional[str] = None
    external_run_min_free_bytes: int = 0
    external_run_max_live_bytes: Optional[int] = None
    setup_timeout_s: float = 300.0
    null_position: str = "last"
    pinned_output_max_bytes: int = 12 << 30
    # Payload-to-workspace admission factors.  They are intentionally
    # conservative: spilling early is recoverable, an allocator OOM is not.
    # Keep enough room for libcudf's final ordering permutation plus the
    # resident input, keys, and output buffers. A 2.7 factor can leave less
    # than one small allocation of headroom at the RMM pool ceiling.
    final_sort_workspace_factor: float = 3.0
    merge_workspace_factor: float = 2.5

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "residency_budget_bytes",
            _positive_bytes(self.residency_budget_bytes),
        )
        object.__setattr__(
            self,
            "external_run_min_free_bytes",
            _nonnegative_bytes(self.external_run_min_free_bytes),
        )
        object.__setattr__(
            self,
            "external_run_max_live_bytes",
            _positive_bytes(self.external_run_max_live_bytes),
        )
        if self.sample_size < 1:
            raise ValueError("GPU sort sample_size must be positive.")
        if self.input_buffer_budget_bytes < 1:
            raise ValueError("GPU sort input buffer budget must be positive.")
        if self.streaming_sample_rows_per_block < 1:
            raise ValueError(
                "GPU sort streaming sample rows per block must be positive."
            )
        if (
            not isinstance(self.sample_seed, int)
            or isinstance(self.sample_seed, bool)
            or not 0 <= self.sample_seed < 1 << 64
        ):
            raise ValueError("GPU sort sample_seed must be an unsigned 64-bit integer.")
        if not 0 < self.rmm_initial_fraction <= self.rmm_max_fraction < 1:
            raise ValueError(
                "GPU sort requires 0 < rmm_initial_fraction <= " "rmm_max_fraction < 1."
            )
        if not 0 < self.auto_wave_fraction <= 1:
            raise ValueError("GPU sort automatic wave fraction must be in (0, 1].")
        if self.exchange_batch_bytes <= 0 or self.run_chunk_bytes <= 0:
            raise ValueError("GPU sort batch and run sizes must be positive.")
        if self.merge_fan_in < 2:
            raise ValueError("GPU sort merge_fan_in must be at least two.")
        if self.external_run_store not in ("plasma", "local_disk"):
            raise ValueError(
                "GPU sort external_run_store must be 'plasma' or 'local_disk'."
            )
        if self.external_run_store == "local_disk":
            if not self.external_run_directory:
                raise ValueError(
                    "GPU sort local_disk runs require external_run_directory."
                )
            if not os.path.isabs(self.external_run_directory):
                raise ValueError(
                    "GPU sort external_run_directory must be an absolute path."
                )
            run_id = self.external_run_id or ""
            if not run_id or not all(
                character.isalnum() or character in ("-", "_") for character in run_id
            ):
                raise ValueError(
                    "GPU sort local_disk runs require an alphanumeric external_run_id."
                )
        if self.setup_timeout_s <= 0:
            raise ValueError("GPU sort setup timeout must be positive.")
        if self.null_position not in ("first", "last"):
            raise ValueError("GPU sort null_position must be 'first' or 'last'.")
        if self.pinned_output_max_bytes < 0:
            raise ValueError("GPU sort pinned output limit cannot be negative.")
        if self.final_sort_workspace_factor < 1 or self.merge_workspace_factor < 1:
            raise ValueError("GPU sort workspace factors must be at least one.")

    @classmethod
    def from_actor_dict(cls, values: Mapping[str, Any]) -> "GPUSortConfig":
        known = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in known})

    @classmethod
    def from_env(cls) -> "GPUSortConfig":
        return cls(
            residency_budget_bytes=os.environ.get(
                "RAY_DATA_GPU_SORT_MEMORY_BUDGET_BYTES"
            )
        )

    def to_actor_dict(self) -> dict[str, Any]:
        return asdict(self)


class GPUSortCapacityError(RuntimeError):
    """Raised when even one bounded GPU run cannot be admitted safely."""
