# SPDX-License-Identifier: Apache-2.0
"""Quality score of each trial after artifact removal."""

from typing import TYPE_CHECKING

import numpy as np

from evoparam._typing import Any, ArrayLike, BoolArray, FloatArray
from evoparam.utils._optional import require

from ._common import check_trialset, window_slice

if TYPE_CHECKING:
    from evoparam.data.containers import TrialSet

__all__ = ["artifact_quality"]

# Windows of the penalties, in seconds after the end of the artifact window.
RESIDUAL_WINDOW = 0.010
SATURATION_WINDOW = 0.020


def artifact_quality(
    ts: "TrialSet",
    *,
    window: tuple[float, float],
    baseline: tuple[float, float] = (-0.5, -0.05),
    template: FloatArray | None = None,
    saturation: BoolArray | None = None,
    fit_r2: ArrayLike | None = None,
) -> Any:
    """Quality of each trial, from 0 (unusable) to 1 (clean).

    ``q = 1 - clip(mean(p1, p2, p3, p4), 0, 1)`` with:

    - ``p1 = 0.5 * min(RMS(x on [b, b + 10 ms]) / RMS(baseline), 1)``: residual energy just
      after the window (``residual_ratio``);
    - ``p2 = min(|step at the window edges| / (3 * noise step), 1)``: discontinuity at the
      edges, against the scaled MAD of the baseline's sample-to-sample changes
      (``edge_discontinuity``);
    - ``p3``: fraction of saturated samples in ``[a, b + 20 ms]`` (``saturation_fraction``);
    - ``p4 = 1 - r2`` of the template fit, 0 without a template (``template_fit_r2``
      reports r2).

    Parameters
    ----------
    ts : TrialSet
        Cleaned trials.
    window : (float, float)
        Artifact window ``(a, b)``, in seconds.
    baseline : (float, float)
        Baseline window, in seconds.
    template : array, optional
        Unused except to mark that a template was used when ``fit_r2`` is None.
    saturation : bool array like ``ts.data``, optional
        Saturated samples (default: ``ts.masks``).
    fit_r2 : 1d array, optional
        r2 of the template (or basis) fit of each trial.

    Returns
    -------
    pandas.DataFrame
        One row per trial: ``residual_ratio``, ``edge_discontinuity``,
        ``saturation_fraction``, ``template_fit_r2`` and ``quality``.
    """
    pd = require("pandas", extra="data", caller="evoparam.artifact.artifact_quality")
    check_trialset(ts)
    times, data = ts.times, ts.data
    win = window_slice(ts, window)
    base = window_slice(ts, baseline)
    first, last = win[0], win[-1]
    b = times[last]

    residual_inds = np.flatnonzero((times > b) & (times <= b + RESIDUAL_WINDOW))
    base_part = data[:, :, base] - np.nanmean(data[:, :, base], axis=-1, keepdims=True)
    base_rms = np.sqrt(np.nanmean(base_part**2, axis=(1, 2)))
    if len(residual_inds):
        res = data[:, :, residual_inds] - np.nanmean(data[:, :, base], axis=-1, keepdims=True)
        res_rms = np.sqrt(np.nanmean(res**2, axis=(1, 2)))
    else:
        res_rms = np.zeros(ts.n_trials)
    ratio = np.divide(res_rms, base_rms, out=np.ones_like(res_rms), where=base_rms > 0)
    p1 = 0.5 * np.minimum(ratio, 1.0)

    steps = np.abs(np.diff(data[:, :, base], axis=-1))
    noise = 1.4826 * np.nanmedian(
        np.abs(steps - np.nanmedian(steps, axis=-1, keepdims=True)), axis=-1
    )
    noise = np.maximum(noise, 1e-12 + 1e-9 * np.nanmax(np.abs(data), axis=-1))
    edges = []
    if first >= 1:
        edges.append(np.abs(data[:, :, first] - data[:, :, first - 1]))
    if last + 1 < ts.n_times:
        edges.append(np.abs(data[:, :, last + 1] - data[:, :, last]))
    jump = np.nanmean(np.mean(edges, axis=0) / noise, axis=1) if edges else np.zeros(ts.n_trials)
    p2 = np.minimum(jump / 3.0, 1.0)

    sat = ts.masks if saturation is None else saturation
    sat_inds = np.flatnonzero((times >= times[first]) & (times <= b + SATURATION_WINDOW))
    p3 = (
        np.zeros(ts.n_trials) if sat is None else np.asarray(sat)[:, :, sat_inds].mean(axis=(1, 2))
    )

    r2 = np.full(ts.n_trials, np.nan) if fit_r2 is None else np.asarray(fit_r2, dtype=float)
    p4 = np.where(np.isnan(r2), 0.0, 1.0 - np.clip(r2, 0, 1))

    quality = 1.0 - np.clip(np.mean([p1, p2, p3, p4], axis=0), 0.0, 1.0)
    return pd.DataFrame(
        {
            "residual_ratio": ratio,
            "edge_discontinuity": jump,
            "saturation_fraction": p3,
            "template_fit_r2": r2,
            "quality": quality,
        }
    )
