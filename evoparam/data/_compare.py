# SPDX-License-Identifier: Apache-2.0
"""Deep equality used by the container and result ``__eq__`` methods.

Arrays are equal when they have the same shape, the same dtype kind and the
same values, NaN being equal to NaN. Tables are equal when they have the same
columns, dtypes, index and values, with missing values (NaN or None) equal to
each other. Dataclasses are compared field by field.
"""

import math
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from typing import Any, ClassVar

import numpy as np

from evoparam.data._table import Table


def _is_dataframe(obj: object) -> bool:
    """Return whether ``obj`` is a pandas DataFrame (without importing pandas)."""
    return type(obj).__name__ == "DataFrame" and type(obj).__module__.startswith("pandas")


def _float_equal(a: float, b: float) -> bool:
    """Return whether two numbers are equal, NaN being equal to NaN."""
    if math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _array_equal(a: Any, b: Any) -> bool:
    """Return whether two arrays have the same shape, dtype kind and values (NaN == NaN)."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or a.dtype.kind != b.dtype.kind:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _frame_equal(a: Any, b: Any) -> bool:
    """Return whether two DataFrames have the same columns, index, dtypes and values."""
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    if not deep_equal(list(a.index), list(b.index)):
        return False
    return all(_series_equal(a[col], b[col]) for col in a.columns)


def _series_equal(ca: Any, cb: Any) -> bool:
    """Return whether two columns have the same dtype kind, missing values and values."""
    if ca.dtype.kind != cb.dtype.kind:
        return False
    na, nb = ca.isna().to_numpy(), cb.isna().to_numpy()
    if not np.array_equal(na, nb):
        return False
    va, vb = ca.to_numpy()[~na], cb.to_numpy()[~nb]
    if ca.dtype.kind in "fc":
        return bool(np.array_equal(va.astype(float), vb.astype(float)))
    return all(deep_equal(x, y) for x, y in zip(va, vb, strict=False))


def _table_equal(a: Table, b: Table) -> bool:
    """Return whether two :class:`Table` have the same columns and values."""
    if a.columns != b.columns or len(a) != len(b):
        return False
    for col in a.columns:
        ca, cb = a[col].to_numpy(), b[col].to_numpy()
        if ca.dtype.kind != cb.dtype.kind:
            return False
        if ca.dtype.kind == "O":
            if not all(deep_equal(x, y) for x, y in zip(ca, cb, strict=False)):
                return False
        elif not _array_equal(ca, cb):
            return False
    return True


def _dataclass_equal(a: Any, b: Any) -> bool:
    """Return whether two dataclasses have the same type and equal fields."""
    return type(a) is type(b) and all(
        deep_equal(getattr(a, f.name), getattr(b, f.name)) for f in fields(a)
    )


def _frames_equal(a: Any, b: Any) -> bool:
    """Return whether both are DataFrames with the same contents."""
    return _is_dataframe(a) and _is_dataframe(b) and _frame_equal(a, b)


def _tables_equal(a: Any, b: Any) -> bool:
    """Return whether both are :class:`Table` with the same contents."""
    return isinstance(a, Table) and isinstance(b, Table) and _table_equal(a, b)


def _dicts_equal(a: Any, b: Any) -> bool:
    """Return whether two dicts have the same keys and equal values."""
    return bool(a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a))


def _sequences_equal(a: Any, b: Any) -> bool:
    """Return whether two lists (or two tuples) have equal items."""
    return (
        type(a) is type(b)
        and len(a) == len(b)
        and all(deep_equal(x, y) for x, y in zip(a, b, strict=False))
    )


def _bools_equal(a: Any, b: Any) -> bool:
    """Return whether both are booleans with the same value."""
    return (
        isinstance(a, (bool, np.bool_)) and isinstance(b, (bool, np.bool_)) and bool(a) == bool(b)
    )


def _numbers_equal(a: Any, b: Any) -> bool:
    """Return whether two numbers are equal as floats, NaN being equal to NaN."""
    return _float_equal(float(a), float(b))


_NUMBERS = (int, float, np.integer, np.floating)
_BOOLS = (bool, np.bool_)
_SEQUENCES = (list, tuple)

# (applies to the pair, comparison), tried in this order after the identity and None checks
_RULES: tuple[tuple[Callable[[object, object], bool], Callable[[Any, Any], bool]], ...] = (
    (lambda a, _b: is_dataclass(a) and not isinstance(a, type), _dataclass_equal),
    (lambda a, b: isinstance(a, np.ndarray) or isinstance(b, np.ndarray), _array_equal),
    (lambda a, b: _is_dataframe(a) or _is_dataframe(b), _frames_equal),
    (lambda a, b: isinstance(a, Table) or isinstance(b, Table), _tables_equal),
    (lambda a, b: isinstance(a, dict) and isinstance(b, dict), _dicts_equal),
    (lambda a, b: isinstance(a, _SEQUENCES) and isinstance(b, _SEQUENCES), _sequences_equal),
    (lambda a, b: isinstance(a, _BOOLS) or isinstance(b, _BOOLS), _bools_equal),
    (lambda a, b: isinstance(a, _NUMBERS) and isinstance(b, _NUMBERS), _numbers_equal),
)


def deep_equal(a: object, b: object) -> bool:
    """Return True when ``a`` and ``b`` hold the same values (see module docstring)."""
    if a is b:
        return True
    if a is None or b is None:
        return False
    for applies, equal in _RULES:
        if applies(a, b):
            return equal(a, b)
    return a == b


class DeepEqualMixin:
    """Gives a dataclass a NaN-aware, array-aware ``__eq__``."""

    def __eq__(self, other: object) -> bool:
        """Compare the contents with :func:`deep_equal` (only objects of the same type)."""
        if type(self) is not type(other):
            return NotImplemented
        return deep_equal(self, other)

    __hash__: ClassVar[None] = None  # type: ignore[assignment]  # unhashable: equality compares the contents
