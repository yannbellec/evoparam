"""Utilities shared by the modules of evoparam (optional imports, names, interpolation)."""

from ._estimator import NotFittedError
from ._optional import require

__all__ = ["NotFittedError", "require"]
