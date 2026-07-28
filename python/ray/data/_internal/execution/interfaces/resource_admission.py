from dataclasses import dataclass
from typing import Optional

from ray.data._internal.execution.interfaces.execution_options import ExecutionResources


@dataclass(frozen=True)
class ResourceAdmissionSpec:
    """Static resource shape for a persistent physical operator.

    ``minimum_resources`` is the complete progress floor. For an elastic
    owner, ``unit_resources`` describes one worker and ``min_units`` describes
    how many workers form that floor. ``unit_resources=None`` denotes an
    indivisible fixed gang whose only valid grant is one complete gang.
    ``max_units=None`` leaves elastic growth unbounded.
    """

    minimum_resources: ExecutionResources
    unit_resources: Optional[ExecutionResources]
    min_units: int
    max_units: Optional[int]


@dataclass(frozen=True)
class ResourceAdmissionGrant:
    """Executor-owned capacity and submission grant for a persistent operator."""

    max_units: int
    may_submit: bool

    def __post_init__(self) -> None:
        if type(self.max_units) is not int or self.max_units < 0:
            raise ValueError(
                "Resource admission max_units must be a non-negative integer"
            )
        if type(self.may_submit) is not bool:
            raise ValueError("Resource admission may_submit must be a boolean")
        if self.may_submit and self.max_units == 0:
            raise ValueError(
                "Resource admission may_submit requires at least one granted unit"
            )
