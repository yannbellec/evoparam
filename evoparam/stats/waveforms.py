# SPDX-License-Identifier: Apache-2.0
"""Summaries of a set of waveforms for display.

The average with a spread band, and the positions of a strip plot.

The plots of :mod:`evoparam.plots` compute no number of their own (REQ-349): the shaded
average of :func:`evoparam.plots.plot_signals_yshade` and the horizontal jitter of
:func:`evoparam.plots.plot_scatter_1` come from these functions.
"""

from collections.abc import Callable
from typing import cast

import numpy as np

from evoparam._typing import Any, ArrayLike, FloatArray, RandomState
from evoparam.errors import SettingsError

__all__ = ["waveform_band", "strip_jitter", "AVERAGES", "SPREADS"]


def _sem(values: Any, axis: int = 0) -> Any:
    values = np.asarray(values, dtype=float)
    return np.std(values, axis=axis, ddof=1) / np.sqrt(values.shape[axis])


AVERAGES: dict[str, Callable[..., Any]] = {"mean": np.mean, "median": np.median}
SPREADS: dict[str, Callable[..., Any]] = {"std": np.std, "sem": _sem}


def waveform_band(
    signals: ArrayLike, *, spread: str = "std", average: str = "mean", scale: float = 1.0
) -> tuple[Any, Any]:
    """Average waveform and half-width of the band drawn around it.

    Parameters
    ----------
    signals : 1-D or 2-D array
        Waveforms (trials × times). A 1-D waveform is its own average and needs ``spread``
        given as an array.
    spread : {"std", "sem"}, 1-D array or callable
        Half-width of the band: the standard deviation or the standard error of the mean
        over the trials (axis 0), an array of half-widths, or a function applied to the
        2-D array.
    average : {"mean", "median"} or callable
        Average of a 2-D array along axis 0.
    scale : float
        Factor applied to the half-width.

    Returns
    -------
    center, half_width : 1-D arrays
    """
    signals = np.asarray(signals, dtype=float)
    if (isinstance(spread, str) or callable(spread)) and signals.ndim != 2:
        raise SettingsError("signals must be 2-D when the spread is not given as an array.")
    if isinstance(average, str) and average not in AVERAGES:
        raise SettingsError(f"average must be one of {tuple(AVERAGES)} or a function.")
    if isinstance(spread, str) and spread not in SPREADS:
        raise SettingsError(f"spread must be one of {tuple(SPREADS)}, an array or a function.")
    if signals.ndim == 2:
        center = (
            AVERAGES[average](signals, axis=0)
            if isinstance(average, str)
            else np.asarray(average(signals), dtype=float)
        )
    else:
        center = signals
    if isinstance(spread, str):
        half = SPREADS[spread](signals, axis=0)
    elif callable(spread):
        half = np.asarray(spread(signals), dtype=float)
    else:
        half = np.asarray(spread, dtype=float)
    return center, scale * half


def strip_jitter(
    n_points: Any, *, x: float = 0.0, sd: float = 0.025, random_state: RandomState = 0
) -> FloatArray:
    """Horizontal positions of a strip plot: ``x`` plus a normal jitter of SD ``sd``.

    The jitter comes from ``numpy.random.default_rng(random_state)`` (spec §9.4), never from
    the global state: the same ``random_state`` gives the same positions. A ``Generator``
    is used as is (and advanced).

    Parameters
    ----------
    n_points : int or tuple of int
        Number (or shape) of the positions.
    x : float
        Centre of the strip.
    sd : float
        Standard deviation of the jitter.
    random_state : int, Generator or None

    Returns
    -------
    ndarray
    """
    rng = np.random.default_rng(random_state)
    return cast("FloatArray", x + rng.normal(0.0, sd, n_points))
