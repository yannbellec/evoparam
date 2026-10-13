# SPDX-License-Identifier: Apache-2.0
"""Type aliases of the package's annotations (spec §9.6, REQ-301); no run-time role."""

import os
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "Any",
    "ArrayLike",
    "BoolArray",
    "FloatArray",
    "IntArray",
    "NDArray",
    "Picks",
    "RandomState",
    "StrPath",
]

#: floating-point array (times, data, parameters): NDArray[np.float64] (spec REQ-301)
FloatArray = NDArray[np.float64]
#: index array (``np.flatnonzero``, ``np.arange``)
IntArray = NDArray[np.intp]
#: boolean mask
BoolArray = NDArray[np.bool_]
#: a file name or a path
StrPath = str | os.PathLike[str]
#: seed or generator (``np.random.default_rng``)
RandomState = int | np.random.Generator | None
#: channels by name or index, a boolean mask, or None for all
Picks = str | int | Sequence[str | int] | NDArray[Any] | None
