"""Local cubic fit and subtraction, with recovery after amplifier saturation.

Implemented from the published local-curve-fitting approach to artifact suppression in
multi-electrode recordings (Wagenaar & Potter, 2002, J Neurosci Methods, "SALPA"): at
every sample, a cubic is fitted by least squares on ``[t - tau, t + tau]`` and its value
at ``t`` is subtracted. The value at the centre of a least-squares polynomial fit is a
Savitzky-Golay filter, so the whole series is processed in one linear filter. While the
amplifier is railed the output is zero; after it leaves the rail, one-sided windows are
used ("depegging") and the samples stay masked until the residual looks like noise.
"""

import numpy as np

from evoparam.errors import SettingsError

__all__ = []

# Residual RMS, in units of the series' noise level, below which a recovered sample is
# trusted again.
TRUST_SD = 3.0


def _poly_rows(n_window, order, positions):
    """Rows of the least-squares projection: fit on 0..n_window-1, evaluated at positions."""

    x = np.arange(n_window, dtype=float)
    x = (x - x.mean()) / max(x.std(), 1.0)
    design = np.vander(x, order + 1)
    evaluate = np.vander(x[positions], order + 1)
    return evaluate @ np.linalg.pinv(design)


def polynomial_clean(data, sfreq, railed=None, *, halfwidth=0.003, order=3):
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

    Returns
    -------
    cleaned : array like ``data``
    masked : bool array like ``data``
        Samples set to zero (railed) or not yet trusted after a rail.
    """

    from scipy.signal import savgol_filter

    half = int(round(halfwidth * sfreq))
    if half < (order + 1) // 2 + 1:
        raise SettingsError(f"halfwidth {halfwidth} s is too short at {sfreq} Hz for a "
                            f"polynomial of order {order}.")
    n_window = 2 * half + 1
    data = np.asarray(data, dtype=float)
    shape = data.shape
    flat = data.reshape(-1, shape[-1])
    fit = savgol_filter(flat, n_window, order, axis=-1, mode="interp")
    cleaned = flat - fit
    masked = np.zeros(flat.shape, dtype=bool)
    if railed is None or not np.any(railed):
        return cleaned.reshape(shape), masked.reshape(shape)

    rails = np.asarray(railed, dtype=bool).reshape(flat.shape)
    forward = _poly_rows(n_window, order, np.arange(half))           # fit on [t0, t0 + 2 tau]
    backward = _poly_rows(n_window, order, np.arange(half + 1, n_window))
    trust = max(1, half // 2)
    n_times = flat.shape[1]
    for row in np.flatnonzero(rails.any(axis=1)):
        x = flat[row]
        noise = 1.4826 * np.median(np.abs(cleaned[row] - np.median(cleaned[row])))
        edges = np.diff(np.r_[0, rails[row].astype(np.int8), 0])
        for first, last in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            cleaned[row, first:last] = 0.0
            masked[row, first:last] = True
            # Depegging: one-sided fits next to the rail
            after = last
            if after + n_window <= n_times:
                cleaned[row, after:after + half] = (x[after:after + half]
                                                    - forward @ x[after:after + n_window])
            before = first - n_window
            if before >= 0:
                cleaned[row, first - half:first] = (x[first - half:first]
                                                    - backward @ x[before:first])
            # Masked until the residual over 0.5 tau looks like noise
            t = after
            while t + trust <= n_times:
                if np.sqrt(np.mean(cleaned[row, t:t + trust] ** 2)) <= TRUST_SD * noise:
                    break
                masked[row, t:t + trust] = True
                t += trust
    return cleaned.reshape(shape), masked.reshape(shape)
