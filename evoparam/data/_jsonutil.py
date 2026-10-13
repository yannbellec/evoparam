# SPDX-License-Identifier: Apache-2.0
"""Strict JSON helpers.

Every JSON file written by evoparam is valid RFC 8259 JSON: NaN and infinities
are written as ``null`` (``json.dumps(..., allow_nan=False)`` after a conversion
pass), numpy scalars and arrays are converted to Python numbers and lists, and
floats keep full float64 precision.
"""

import json
import math
from typing import Any, NoReturn

import numpy as np


def to_jsonable(obj: object) -> Any:
    """Convert ``obj`` to plain JSON types; non-finite floats become None."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable.")


def dumps(obj: object, **kwargs: Any) -> str:
    """Strict ``json.dumps``: converts first, then refuses any NaN token."""
    return json.dumps(to_jsonable(obj), allow_nan=False, **kwargs)


def _refuse_constant(name: str) -> NoReturn:
    raise ValueError(f"Invalid JSON constant {name!r}.")


def loads(text: str | bytes) -> Any:
    """Strict ``json.loads``: NaN, Infinity and -Infinity tokens are refused."""
    return json.loads(text, parse_constant=_refuse_constant)
