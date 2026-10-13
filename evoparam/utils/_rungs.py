# SPDX-License-Identifier: Apache-2.0
"""Rungs between a unit of work and joblib in a worker process (``evoparam.utils._parallel``).

A warning issued in a unit with a ``stacklevel`` reaching past the unit lands on one of
these frames; the rung it lands on tells how many frames above the unit it points to,
and the calling process locates it as many frames above the caller of ``parallel_map``.
Each rung is its own function, so that the count depends neither on the frames of joblib
nor on the Python version (comprehensions are frames of their own before Python 3.12).
"""

from collections.abc import Callable
from typing import Any

__all__ = ["LEVELS", "climb"]


def _rung_1(function: Callable[[Any], Any], item: Any) -> Any:
    return function(item)


def _rung_2(function: Callable[[Any], Any], item: Any) -> Any:
    return _rung_1(function, item)


def _rung_3(function: Callable[[Any], Any], item: Any) -> Any:
    return _rung_2(function, item)


def _rung_4(function: Callable[[Any], Any], item: Any) -> Any:
    return _rung_3(function, item)


def _rung_5(function: Callable[[Any], Any], item: Any) -> Any:
    return _rung_4(function, item)


def _rung_6(function: Callable[[Any], Any], item: Any) -> Any:
    return _rung_5(function, item)


#: ``function(item)`` below six rungs
climb = _rung_6

#: line of each rung's call -> how many frames above the unit it is
LEVELS = {
    max(line for _, _, line in rung.__code__.co_lines() if line is not None): level
    for level, rung in enumerate((_rung_1, _rung_2, _rung_3, _rung_4, _rung_5, _rung_6), 1)
}
FILE = _rung_1.__code__.co_filename
