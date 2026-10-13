# SPDX-License-Identifier: Apache-2.0
"""IEEE arithmetic without floating-point warnings, for documented NaN and inf results.

Each helper returns, element by element, exactly what the plain NumPy operation returns
(same bits), but computes the operation only where it is defined and writes the IEEE value
of the other elements directly, so that no ``divide`` or ``invalid`` floating-point
warning is raised and the floating-point error state is never changed to hide one (spec
REQ-307 iv). An overflow still warns, as it does without the helpers.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["divide", "log", "log10", "sqrt"]


def divide(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Return ``a / b`` without a floating-point warning.

    ``±inf`` for a non-zero numerator over zero, NaN for ``0 / 0`` and ``inf / inf``.

    Examples
    --------
    >>> divide([1.0, 0.0, -2.0, 6.0], [0.0, 0.0, 0.0, 3.0]).tolist()
    [inf, nan, -inf, 2.0]
    """
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    undefined = b == 0
    infinite = np.isinf(b)
    if infinite.any():
        undefined |= infinite & np.isinf(a)
    if not undefined.any():
        return np.asarray(np.divide(a, b))
    special = np.where(
        np.isnan(a) | (a == 0) | ~(b == 0), np.nan, np.copysign(np.inf, a) * np.copysign(1, b)
    )
    out: NDArray[np.float64] = np.divide(a, b, out=special, where=~undefined)
    return out


def sqrt(x: ArrayLike) -> NDArray[np.float64]:
    """``np.sqrt(x)`` with NaN for negative values, without a floating-point warning."""
    x = np.asarray(x, dtype=float)
    negative = x < 0
    if not negative.any():
        return np.asarray(np.sqrt(x))
    out: NDArray[np.float64] = np.sqrt(x, out=np.full(x.shape, np.nan), where=~negative)
    return out


def _logarithm(func: np.ufunc, x: ArrayLike) -> NDArray[np.float64]:
    x = np.asarray(x, dtype=float)
    undefined = x <= 0
    if not undefined.any():
        return np.asarray(func(x))
    special = np.where(x == 0, -np.inf, np.nan)
    out: NDArray[np.float64] = func(x, out=special, where=~undefined)
    return out


def log(x: ArrayLike) -> NDArray[np.float64]:
    """``np.log(x)``: ``-inf`` at zero, NaN for negative values, without a warning."""
    return _logarithm(np.log, x)


def log10(x: ArrayLike) -> NDArray[np.float64]:
    """``np.log10(x)``: ``-inf`` at zero, NaN for negative values, without a warning."""
    return _logarithm(np.log10, x)
