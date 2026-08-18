"""Small, serializable configuration for Ray Data's GPU range sort."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional


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


@dataclass(frozen=True)
class GPUSortConfig:
    """Internal controls shared by the controller and one-GPU actors.

    ``residency_budget_bytes`` is a proactive payload-residency watermark, not
    a second allocator pool.  RMM retains enough room to sort or merge the run
    that caused the transition to external execution.
    """

    sample_size: int = 1 << 16
    sample_seed: int = 0
    rmm_initial_fraction: float = 0.50
    rmm_max_fraction: float = 0.85
    residency_budget_bytes: Optional[int] = None
    # Bound one source contribution by the smallest actor-reported memory
    # budget. Source-specific MPF lanes let receivers drain contributions
    # independently instead of concatenating an entire logical range first.
    auto_wave_fraction: float = 0.50
    exchange_batch_bytes: int = 512 << 20
    merge_fan_in: int = 4
    run_chunk_bytes: int = 512 << 20
    setup_timeout_s: float = 300.0
    null_position: str = "last"
    pinned_output_max_bytes: int = 12 << 30
    # Leave room for libcudf's ordering permutation, keys, and output buffers.
    # Externalizing a run early is recoverable; an allocator OOM is not.
    final_sort_workspace_factor: float = 3.0
    merge_workspace_factor: float = 2.5

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "residency_budget_bytes",
            _positive_bytes(self.residency_budget_bytes),
        )
        if self.sample_size < 1:
            raise ValueError("GPU sort sample_size must be positive.")
        if (
            not isinstance(self.sample_seed, int)
            or isinstance(self.sample_seed, bool)
            or not 0 <= self.sample_seed < 1 << 64
        ):
            raise ValueError("GPU sort sample_seed must be an unsigned 64-bit integer.")
        if not 0 < self.rmm_initial_fraction <= self.rmm_max_fraction < 1:
            raise ValueError(
                "GPU sort requires 0 < rmm_initial_fraction " "<= rmm_max_fraction < 1."
            )
        if not 0 < self.auto_wave_fraction <= 1:
            raise ValueError("GPU sort automatic wave fraction must be in (0, 1].")
        if self.exchange_batch_bytes <= 0 or self.run_chunk_bytes <= 0:
            raise ValueError("GPU sort batch and run sizes must be positive.")
        if self.merge_fan_in < 2:
            raise ValueError("GPU sort merge_fan_in must be at least two.")
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
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise ValueError(f"Unknown GPU sort configuration fields: {unknown}.")
        return cls(**dict(values))

    def to_actor_dict(self) -> dict[str, Any]:
        return asdict(self)


class GPUSortCapacityError(RuntimeError):
    """Raised when even one bounded GPU run cannot be admitted safely."""
