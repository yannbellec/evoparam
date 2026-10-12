# SPDX-License-Identifier: Apache-2.0
"""Replace a time window by an interpolation between the samples around it.

Used by :mod:`evoparam.artifact` (method ``"interpolate"``).
"""

from collections.abc import Sequence

import numpy as np

from evoparam._typing import ArrayLike, FloatArray, IntArray
from evoparam.errors import ArtifactError, SettingsError

__all__ = ["window_indices", "interpolate_window", "INTERPOLATIONS"]

INTERPOLATIONS = ("linear", "cubic", "pchip")


def window_indices(times: ArrayLike, window: Sequence[float], buffer: int = 0) -> IntArray:
    """Return the indices of the samples inside ``window`` (inclusive), checked against the buffer.

    Raises
    ------
    ArtifactError
        Empty window, or fewer than ``buffer`` samples on either side of it.
    """
    times = np.asarray(times)
    if window[1] < window[0]:
        raise ArtifactError(f"window {tuple(window)} ends before it starts.")
    inds = np.flatnonzero((times >= window[0]) & (times <= window[1]))
    if len(inds) == 0:
        raise ArtifactError(
            f"The window {tuple(window)} s contains no sample "
            f"(epoch {times[0]:.4f} to {times[-1]:.4f} s)."
        )
    if inds[0] - buffer < 0 or inds[-1] + 1 + buffer > len(times):
        raise ArtifactError(
            f"The window {tuple(window)} s needs {buffer} samples on each side inside the epoch."
        )
    return inds


def interpolate_window(
    times: ArrayLike,
    data: ArrayLike,
    window: Sequence[float],
    buffer: int = 3,
    kind: str = "linear",
    anchor: str = "mean",
) -> tuple[FloatArray, IntArray]:
    """Replace ``window`` (time is the last axis of ``data``) by an interpolation.

    - ``linear``: a straight line between the anchors, one on each side, each the
      ``anchor`` statistic (mean or median) of ``buffer`` samples and of their times:
      ``x(t) = x(a) + (x(b) - x(a)) (t - a) / (b - a)``.
    - ``cubic`` / ``pchip``: a cubic spline / a shape-preserving cubic (PCHIP) through
      the ``buffer`` samples on each side.

    Parameters
    ----------
    times : 1d array
        Time of each sample (s).
    data : array
        Values, time on the last axis.
    window : (float, float)
        Start and end of the replaced window (s), inclusive.
    buffer : int, default: 3
        Samples on each side used as anchors.
    kind : {"linear", "cubic", "pchip"}
    anchor : {"mean", "median"}
        Statistic of the anchor samples for ``linear``.

    Returns
    -------
    cleaned : array
        Copy of ``data`` with the window replaced.
    inds : 1d array
        Indices of the replaced samples.
    """
    if kind not in INTERPOLATIONS:
        raise SettingsError(f"interpolation must be one of {INTERPOLATIONS}, not {kind!r}.")
    if anchor not in ("mean", "median"):
        raise SettingsError("anchor must be 'mean' or 'median'.")
    if buffer < 1 or (kind != "linear" and buffer < 2):
        raise SettingsError("buffer must be at least 1 (2 for cubic and pchip).")
    times = np.asarray(times, dtype=float)
    inds = window_indices(times, window, buffer)
    first, last = inds[0], inds[-1]
    left = np.arange(first - buffer, first)
    right = np.arange(last + 1, last + 1 + buffer)
    cleaned = np.array(data, dtype=float, copy=True)

    if kind == "linear":
        stat = np.mean if anchor == "mean" else np.median
        x_left, x_right = stat(times[left]), stat(times[right])
        y_left = stat(cleaned[..., left], axis=-1)
        y_right = stat(cleaned[..., right], axis=-1)
        slope = (y_right - y_left) / (x_right - x_left)
        cleaned[..., inds] = y_left[..., None] + slope[..., None] * (times[inds] - x_left)
    else:
        from scipy.interpolate import CubicSpline, PchipInterpolator

        knots = np.r_[left, right]
        build = CubicSpline if kind == "cubic" else PchipInterpolator
        curve = build(times[knots], cleaned[..., knots], axis=-1)
        cleaned[..., inds] = curve(times[inds])
    return cleaned, inds
