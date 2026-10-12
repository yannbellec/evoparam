# SPDX-License-Identifier: Apache-2.0
"""Deep equality used by the container and result ``__eq__`` methods.

Arrays are equal when they have the same shape, the same dtype kind and the
same values, NaN being equal to NaN. Tables are equal when they have the same
columns, dtypes, index and values, with missing values (NaN or None) equal to
each other. Dataclasses are compared field by field.
"""

import math
from dataclasses import fields, is_dataclass

import numpy as np


def _is_dataframe(obj):
    return type(obj).__name__ == "DataFrame" and type(obj).__module__.startswith("pandas")


def _float_equal(a, b):
    if math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _array_equal(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or a.dtype.kind != b.dtype.kind:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _frame_equal(a, b):
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    if not deep_equal(list(a.index), list(b.index)):
        return False
    for col in a.columns:
        ca, cb = a[col], b[col]
        if ca.dtype.kind != cb.dtype.kind:
            return False
        na, nb = ca.isna().to_numpy(), cb.isna().to_numpy()
        if not np.array_equal(na, nb):
            return False
        va, vb = ca.to_numpy()[~na], cb.to_numpy()[~nb]
        if ca.dtype.kind in "fc":
            if not np.array_equal(va.astype(float), vb.astype(float)):
                return False
        elif not all(deep_equal(x, y) for x, y in zip(va, vb)):
            return False
    return True


def deep_equal(a, b):
    """Return True when ``a`` and ``b`` hold the same values (see module docstring)."""

    if a is b:
        return True
    if a is None or b is None:
        return False
    if is_dataclass(a) and not isinstance(a, type):
        if type(a) is not type(b):
            return False
        return all(deep_equal(getattr(a, f.name), getattr(b, f.name)) for f in fields(a))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return _array_equal(a, b)
    if _is_dataframe(a) or _is_dataframe(b):
        return _is_dataframe(a) and _is_dataframe(b) and _frame_equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return (
            type(a) is type(b) and len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
        )
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return (
            isinstance(a, (bool, np.bool_))
            and isinstance(b, (bool, np.bool_))
            and bool(a) == bool(b)
        )
    if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(
        b, (int, float, np.integer, np.floating)
    ):
        return _float_equal(float(a), float(b))
    return a == b


class DeepEqualMixin:
    """Gives a dataclass a NaN-aware, array-aware ``__eq__``."""

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return deep_equal(self, other)

    __hash__ = None
