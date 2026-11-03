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

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
BoolArray = NDArray[np.bool_]
StrPath = str | os.PathLike[str]
RandomState = int | np.random.Generator | None
Picks = str | int | Sequence[str | int] | NDArray[Any] | None
