# SPDX-License-Identifier: Apache-2.0
"""Local cubic fit and subtraction, with recovery after amplifier saturation.

Implemented from the published local-curve-fitting approach to artifact suppression in
multi-electrode recordings (Wagenaar & Potter, 2002, J Neurosci Methods, "SALPA"): at
every sample, a cubic is fitted by least squares on ``[t - tau, t + tau]`` and its value
at ``t`` is subtracted. The value at the centre of a least-squares polynomial fit is a
Savitzky-Golay filter, so the whole series is processed in one linear filter. While the
amplifier is railed the output is zero; after it leaves the rail, one-sided windows are
used ("depegging") and the samples stay masked until the residual looks like noise.

Extent. The published method subtracts the fit at every sample: it is a high-pass filter of
the whole recording (cut-off near 280 Hz for ``tau = 3 ms`` at 2048 Hz), which changes the
baseline's RMS by 9 % on white noise at 2048 Hz and 18 % at 1 kHz (spec REQ-060 allows 2 %).
``extent="local"`` (default) subtracts it only from the artifact window to the end of the
artifact, found as the first time after the window (and after any rail) where the fitted
curve stays within ``TRUST_SD`` robust SD of its own baseline level for half a fit window,
with half-cosine ramps of ``tau`` on both sides; ``extent="global"`` is the published
method, at every sample.
"""

import numpy as np

from evoparam.errors import SettingsError

__all__ = []

# Residual RMS, in units of the series' noise level, below which a recovered sample is
# trusted again.
TRUST_SD = 3.0

# Series processed together by the local extent (bounds the memory of the weights).
CHUNK = 512


def _poly_rows(n_window, order, positions):
    """Rows of the least-squares projection: fit on 0..n_window-1, evaluated at positions."""
    x = np.arange(n_window, dtype=float)
    x = (x - x.mean()) / max(x.std(), 1.0)
    design = np.vander(x, order + 1)
    evaluate = np.vander(x[positions], order + 1)
    return evaluate @ np.linalg.pinv(design)


def _local_weights(fit, window_inds, baseline_inds, railed, half):
    """Weight of the subtraction at each sample (rows of ``fit``).

    1 from the window start to the end of the artifact, half-cosine ramps of ``half`` samples, 0
    elsewhere.
    """
    n_rows, n_times = fit.shape
    weights = np.zeros(fit.shape)
    lo = int(window_inds[0])
    first_after = int(window_inds[-1]) + 1
    if baseline_inds is not None and len(baseline_inds) >= 8:
        base = fit[:, baseline_inds]
    else:
        outside = np.ones(n_times, dtype=bool)
        outside[lo:] = False
        base = fit[:, outside] if outside.sum() >= 8 else fit
    level = np.median(base, axis=1)
    spread = 1.4826 * np.median(np.abs(base - level[:, None]), axis=1)
    spread = np.maximum(spread, 1e-12 * np.maximum(np.max(np.abs(fit), axis=1), 1.0))
    hold = max(1, half // 2)
    starts = np.full(n_rows, first_after)
    if railed is not None:
        tail = railed[:, lo:]
        has = tail.any(axis=1)
        last = tail.shape[1] - 1 - np.argmax(tail[:, ::-1], axis=1)
        starts = np.where(has, np.maximum(starts, lo + last + 1), starts)
    # First run of ``hold`` quiet samples at or after the start: its first sample ends the
    # artifact (the end of the series when there is none)
    quiet = np.abs(fit - level[:, None]) <= TRUST_SD * spread[:, None]
    count = np.cumsum(np.c_[np.zeros(n_rows, dtype=int), quiet.astype(int)], axis=1)
    runs = count[:, hold:] - count[:, :-hold]  # runs[:, t]: quiet in t..t+hold-1
    candidate = (runs == hold) & (np.arange(runs.shape[1])[None] >= starts[:, None])
    ends = np.where(candidate.any(axis=1), np.argmax(candidate, axis=1), n_times)
    positions = np.arange(n_times)[None]
    weights = ((positions >= lo) & (positions < ends[:, None])).astype(float)
    # Half-cosine ramps of ``half`` samples before the window and after the end
    if half > 0:
        rise = positions - (lo - half - 1)  # 1..half before the window
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.clip(rise, 0, half + 1) / (half + 1))
        weights = np.where((positions >= lo - half) & (positions < lo), ramp, weights)
        fall = (ends[:, None] + half) - positions  # half..1 after the end
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.clip(fall, 0, half + 1) / (half + 1))
        after = (positions >= ends[:, None]) & (positions < ends[:, None] + half)
        weights = np.where(after, ramp, weights)
    return weights


def polynomial_clean(
    data,
    sfreq,
    railed=None,
    *,
    halfwidth=0.003,
    order=3,
    extent="global",
    window_inds=None,
    baseline_inds=None,
):
    """Subtract a local polynomial fit from each series (time last).

    Parameters
    ----------
    data : array, time last
    sfreq : float
    railed : bool array like ``data``, optional
        Samples at the amplifier's rail (saturation masks).
    halfwidth : float, default: 0.003
        Half-width ``tau`` of the fit window, in seconds.
    order : int, default: 3
    extent : {"global", "local"}
        ``"global"``: the fit is subtracted at every sample (the published method).
        ``"local"``: only from the artifact window to the end of the artifact (module
        docstring); needs ``window_inds``.
    window_inds : 1d int array, optional
        Samples of the artifact window (``"local"``).
    baseline_inds : 1d int array, optional
        Baseline samples, for the level of the fitted curve (``"local"``; default: the
        samples before the window).

    Returns
    -------
    cleaned : array like ``data``
    masked : bool array like ``data``
        Samples set to zero (railed) or not yet trusted after a rail.
    """
    from scipy.signal import savgol_filter

    half = int(round(halfwidth * sfreq))
    if half < (order + 1) // 2 + 1:
        raise SettingsError(
            f"halfwidth {halfwidth} s is too short at {sfreq} Hz for a "
            f"polynomial of order {order}."
        )
    if extent not in ("global", "local"):
        raise SettingsError("extent must be 'global' or 'local'.")
    if extent == "local" and (window_inds is None or len(window_inds) == 0):
        raise SettingsError("extent='local' needs the artifact window.")
    n_window = 2 * half + 1
    data = np.asarray(data, dtype=float)
    shape = data.shape
    flat = data.reshape(-1, shape[-1])
    fit = savgol_filter(flat, n_window, order, axis=-1, mode="interp")
    cleaned = flat - fit
    masked = np.zeros(flat.shape, dtype=bool)
    rails = None if railed is None else np.asarray(railed, dtype=bool).reshape(flat.shape)
    if rails is not None and np.any(rails):
        _depeg(flat, cleaned, masked, rails, half, n_window, order)
    if extent == "local":
        window_inds = np.asarray(window_inds)
        for lo in range(0, len(flat), CHUNK):
            rows = slice(lo, lo + CHUNK)
            weights = _local_weights(
                fit[rows], window_inds, baseline_inds, None if rails is None else rails[rows], half
            )
            cleaned[rows] = flat[rows] + weights * (cleaned[rows] - flat[rows])
            masked[rows] &= weights > 0
    return cleaned.reshape(shape), masked.reshape(shape)


def _depeg(flat, cleaned, masked, rails, half, n_window, order):
    """Zeros while railed, one-sided fits next to the rail, masked until trusted (in place)."""
    forward = _poly_rows(n_window, order, np.arange(half))  # fit on [t0, t0 + 2 tau]
    backward = _poly_rows(n_window, order, np.arange(half + 1, n_window))
    trust = max(1, half // 2)
    n_times = flat.shape[1]
    for row in np.flatnonzero(rails.any(axis=1)):
        x = flat[row]
        noise = 1.4826 * np.median(np.abs(cleaned[row] - np.median(cleaned[row])))
        edges = np.diff(np.r_[0, rails[row].astype(np.int8), 0])
        for first, last in zip(
            np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=False
        ):
            cleaned[row, first:last] = 0.0
            masked[row, first:last] = True
            # Depegging: one-sided fits next to the rail
            after = last
            if after + n_window <= n_times:
                cleaned[row, after : after + half] = (
                    x[after : after + half] - forward @ x[after : after + n_window]
                )
            before = first - n_window
            if before >= 0:
                cleaned[row, first - half : first] = (
                    x[first - half : first] - backward @ x[before:first]
                )
            # Masked until the residual over 0.5 tau looks like noise
            t = after
            while t + trust <= n_times:
                if np.sqrt(np.mean(cleaned[row, t : t + trust] ** 2)) <= TRUST_SD * noise:
                    break
                masked[row, t : t + trust] = True
                t += trust
