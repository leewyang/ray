"""Optional distributed GPU range sort for Ray Data."""

from ray.data._internal.gpu_sort.config import GPUSortConfig

__all__ = ["GPUSortConfig", "GPUSortOperator"]


def __getattr__(name: str):
    if name == "GPUSortOperator":
        from ray.data._internal.gpu_sort.operator import GPUSortOperator

        return GPUSortOperator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
