# SPDX-License-Identifier: Apache-2.0
"""Robust rejection of bad trials (flagged, never deleted)."""

import time as _time
import warnings

import numpy as np

from evoparam.errors import SettingsError
from evoparam.errors import warn as _warn
from evoparam.utils._optional import require

from ._common import check_trialset, masked_values, new, window_mask

__all__ = ["reject_trials"]

MAD_TO_SD = 1.4826


def _robust_z(values):
    """Robust z-score over trials (axis 0), per channel."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        center = np.nanmedian(values, axis=0, keepdims=True)
        spread = MAD_TO_SD * np.nanmedian(np.abs(values - center), axis=0, keepdims=True)
    spread = np.where(spread > 0, spread, np.nan)
    return (values - center) / spread


def reject_trials(
    ts,
    *,
    response_window=(0.01, 0.5),
    baseline=(-0.5, -0.05),
    ptp_z=5.0,
    rms_z=5.0,
    flat_threshold=1e-3,
    quality_min=0.5,
    max_masked_fraction=0.2,
    max_fraction=0.3,
    all_channels_fraction=0.5,
):
    """Flag bad trials; trials are never deleted.

    For each trial and channel, robust z-scores over trials (median and scaled MAD) of the
    peak-to-peak amplitude and of the RMS, in the response window and in the baseline. A
    trial-channel is flagged when a z-score exceeds ``ptp_z`` / ``rms_z``, when it is flat
    (MAD over time below ``flat_threshold``, in the data units), or when masks cover more
    than ``max_masked_fraction`` of the response window. A trial whose
    ``metadata["quality"]`` is below ``quality_min`` is flagged on every channel.

    On a channel, at most ``max_fraction`` of the trials are flagged (the most extreme
    ones); above it, a warning says the channel, not the trials, is the problem.

    Flagged trial-channels are masked (later stages skip them). A trial flagged on at least
    ``all_channels_fraction`` of its channels, or for quality, gets ``rejected = True`` and
    ``reject_reason`` (``"ptp"``, ``"rms"``, ``"flat"``, ``"masked"``, ``"quality"``, joined
    by ``"+"``) in ``metadata``; ``TrialSet.average`` leaves it out and
    ``ts.select(query="not rejected")`` filters it.

    Returns
    -------
    TrialSet
        Same trials, with ``metadata["rejected"]``, ``metadata["reject_reason"]``,
        ``metadata["n_flagged_channels"]`` and the masks of the flagged trial-channels.
    """
    pd = require("pandas", extra="data", caller="evoparam.prep.reject_trials")
    start = _time.perf_counter()
    check_trialset(ts)
    if not 0 < max_fraction <= 1:
        raise SettingsError("max_fraction must be in (0, 1].")
    resp = window_mask(ts.times, response_window, "response_window")
    base = window_mask(ts.times, baseline, "baseline")
    values = masked_values(ts)
    n_trials, n_channels, _ = values.shape

    reasons = np.zeros((n_trials, n_channels), dtype=object)
    reasons[:] = ""
    score = np.zeros((n_trials, n_channels))

    def flag(cond, name, strength):
        nonlocal score
        cond = np.nan_to_num(cond, nan=0).astype(bool)
        for k, c in zip(*np.nonzero(cond), strict=False):
            reasons[k, c] = name if not reasons[k, c] else reasons[k, c] + "+" + name
        score = np.where(cond, np.maximum(score, strength), score)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for window in (resp, base):
            seg = values[:, :, window]
            ptp = np.nanmax(seg, axis=-1) - np.nanmin(seg, axis=-1)
            rms = np.sqrt(np.nanmean((seg - np.nanmean(seg, -1, keepdims=True)) ** 2, axis=-1))
            z_ptp, z_rms = _robust_z(ptp), _robust_z(rms)
            flag(z_ptp > ptp_z, "ptp", np.nan_to_num(z_ptp))
            flag(z_rms > rms_z, "rms", np.nan_to_num(z_rms))
        seg = values[:, :, resp | base]
        mad = MAD_TO_SD * np.nanmedian(np.abs(seg - np.nanmedian(seg, -1, keepdims=True)), -1)
        flag(mad < flat_threshold, "flat", np.inf)
    if ts.masks is not None:
        masked = ts.masks[:, :, resp].mean(axis=-1)
        flag(masked > max_masked_fraction, "masked", np.inf)

    # Cap the share of flagged trials per channel
    capped = []
    limit = int(np.floor(max_fraction * n_trials))
    for c in range(n_channels):
        rows = np.flatnonzero(reasons[:, c] != "")
        if len(rows) > limit:
            capped.append(ts.ch_names[c])
            keep = rows[np.argsort(-score[rows, c], kind="stable")[:limit]]
            drop = np.setdiff1d(rows, keep)
            reasons[drop, c] = ""
    if capped:
        _warn(
            f"reject_trials: more than {max_fraction:.0%} of the trials are flagged on "
            f"{capped}; the channel, not the trials, is probably the problem (only "
            "the most extreme trials are flagged).",
            UserWarning,
            stacklevel=2,
        )

    flagged = reasons != ""
    meta = ts.metadata.copy() if ts.metadata is not None else pd.DataFrame(index=range(n_trials))
    low_quality = np.zeros(n_trials, dtype=bool)
    if "quality" in meta.columns:
        low_quality = meta["quality"].to_numpy(dtype=float) < quality_min
    share = flagged.mean(axis=1)
    rejected = low_quality | (share >= all_channels_fraction)
    reason_text = []
    for k in range(n_trials):
        parts = sorted({part for r in reasons[k] if r for part in r.split("+")})
        if low_quality[k]:
            parts.append("quality")
        reason_text.append("+".join(parts) if rejected[k] else "")
    previous = (
        meta["rejected"].to_numpy(dtype=bool)
        if "rejected" in meta.columns
        else np.zeros(n_trials, dtype=bool)
    )
    meta["rejected"] = previous | rejected
    meta["reject_reason"] = reason_text
    meta["n_flagged_channels"] = flagged.sum(axis=1).astype("int64")

    masks = np.zeros(ts.data.shape, dtype=bool) if ts.masks is None else ts.masks.copy()
    masks |= flagged[:, :, None]
    masks |= low_quality[:, None, None]
    settings = {
        "response_window": list(response_window),
        "baseline": list(baseline),
        "ptp_z": ptp_z,
        "rms_z": rms_z,
        "flat_threshold": flat_threshold,
        "quality_min": quality_min,
        "max_masked_fraction": max_masked_fraction,
        "max_fraction": max_fraction,
        "all_channels_fraction": all_channels_fraction,
        "n_rejected": int(rejected.sum()),
        "capped_channels": capped,
    }
    return new(
        ts,
        "evoparam.prep.reject_trials",
        settings,
        start,
        metadata=meta,
        masks=masks if masks.any() else None,
    )
