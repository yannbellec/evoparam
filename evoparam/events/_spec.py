# SPDX-License-Identifier: Apache-2.0
"""The pulse detector of the specification as written (spec §5.2, methods 1 and 2).

``detect_pulses(..., procedure="spec")`` uses these steps instead of the default ones:

1. Robust difference detector: ``d[n] = median_c |x_c[n] - x_c[n - 1]|`` over the picked
   channels (a true median, every channel), ``z = (d - median(d)) / (1.4826 MAD(d))``; an
   onset is the first ``n`` with ``z[n] > threshold_sd`` after the refractory period
   ``min_interval``; it is refined to the sample of the largest ``d`` within +/-2 ms, minus
   one sample (the last pre-artifact sample).
2. Template detector (``"template"`` and ``"threshold+template"``): a template ``T`` from
   the first ``n_template`` detections (-1 ms to +3 ms around the onset, all picked channels,
   unit norm per channel); the normalised cross-correlation
   ``rho[n] = <T, X[n:n+L]> / (||T|| ||X[n:n+L]||)`` is scanned over the whole recording and
   its peaks above ``rho_min`` that respect ``min_interval`` are onsets. As a second pass it
   rejects threshold detections that do not match the template and adds the template onsets
   that the threshold missed.
"""

from typing import cast

import numpy as np

from evoparam._typing import Any, FloatArray, NDArray

from ._score import _CHUNK, MAD_TO_SD

__all__ = []


def spec_score(data: Any, index: Any) -> FloatArray:
    """Return the spectral score ``d[n] = median_c |x_c[n] - x_c[n - 1]|``.

    For n = 1 .. n_times - 1 (element n - 1), the median of all the picked channels (NaN changes
    count as zero).
    """
    n_times = data.shape[1]
    index = np.asarray(index)
    score = np.empty(n_times - 1)
    for start in range(0, n_times - 1, _CHUNK):
        stop = min(start + _CHUNK, n_times - 1)
        block = np.abs(np.diff(data[index, start : stop + 1], axis=1))
        np.nan_to_num(block, copy=False, nan=0.0)
        score[start:stop] = np.median(block, axis=0)
    return score


def spec_z(score: Any) -> Any:
    """Robust z-score of the specification (median and 1.4826 MAD; MAD 0 gives SD)."""
    center = np.median(score)
    spread = MAD_TO_SD * np.median(np.abs(score - center))
    if not spread > 0:
        spread = np.std(score) or 1.0
    return (score - center) / spread


def spec_threshold(
    score: Any, z: Any, sfreq: float, threshold_sd: Any, min_samples: Any, refine: bool = True
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Detect with method 1: onsets (sample indices) and the largest z of each artifact.

    ``score[i]`` is ``d[i + 1]``. The first crossing after the refractory period is kept;
    with ``refine`` the onset is the sample of the largest ``d`` within +/-2 ms of the
    crossing, minus one (else the crossing sample itself).
    """
    above = np.flatnonzero(z > threshold_sd)
    onsets, strength = [], []
    last = -np.inf
    half = max(1, int(round(0.002 * sfreq)))
    n = len(score)
    for i in above:
        sample = i + 1  # d[sample] crosses the threshold
        if sample - last < min_samples:
            continue
        lo, hi = max(0, i - half), min(n, i + half + 1)
        if refine:
            best = lo + int(np.argmax(score[lo:hi]))  # d[best + 1] is the largest
            onset = best  # (best + 1) - 1: last pre-artifact
        else:
            onset = sample
        onsets.append(max(onset, 0))
        strength.append(float(np.max(z[lo:hi])))
        last = sample
    return np.array(onsets, dtype=int), np.array(strength)


def spec_template(data: Any, index: Any, onsets: Any, sfreq: float, n_template: Any) -> Any:
    """Mean of the first ``n_template`` snippets (-1 ms to +3 ms), unit norm per channel.

    Returns (template [n_picked, L], n_pre) or (None, n_pre) when no snippet fits.
    """
    n_pre, n_post = max(1, int(round(0.001 * sfreq))), max(1, int(round(0.003 * sfreq)))
    n_times = data.shape[1]
    rows = [
        data[index, onset - n_pre : onset + n_post]
        for onset in onsets
        if onset - n_pre >= 0 and onset + n_post <= n_times
    ][: int(n_template)]
    if not rows:
        return None, n_pre
    template = np.nan_to_num(np.mean(rows, axis=0))
    norms = np.linalg.norm(template, axis=1, keepdims=True)
    template = np.divide(template, norms, out=np.zeros_like(template), where=norms > 0)
    return template, n_pre


def spec_rho(data: Any, index: Any, template: Any) -> NDArray[Any]:
    """Normalised cross-correlation of the template with every window of the picked channels.

    ``rho[n]`` for windows starting at sample n (n_times - L + 1 values).
    """
    from scipy.signal import fftconvolve

    length = template.shape[1]
    n_times = data.shape[1]
    numerator = np.zeros(n_times - length + 1)
    energy = np.zeros(n_times - length + 1)
    kernel = np.ones(length)
    for row, ch in enumerate(index):
        x = np.nan_to_num(np.asarray(data[ch], dtype=float))
        if np.any(template[row]):
            numerator += fftconvolve(x, template[row][::-1], mode="valid")
        energy += fftconvolve(x * x, kernel, mode="valid")
    denominator = np.linalg.norm(template) * np.sqrt(np.maximum(energy, 0.0))
    return cast(
        "NDArray[Any]",
        np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0),
    )


def spec_template_peaks(rho: Any, rho_min: Any, min_samples: Any) -> NDArray[Any]:
    """Local maxima of ``rho`` above ``rho_min``, the larger kept within ``min_samples``."""
    if len(rho) < 3:
        return np.array([], dtype=int)
    inner = (rho[1:-1] > rho_min) & (rho[1:-1] >= rho[:-2]) & (rho[1:-1] > rho[2:])
    peaks = np.flatnonzero(inner) + 1
    order = peaks[np.argsort(-rho[peaks], kind="stable")]
    kept = []
    taken: Any
    taken = np.zeros(0, dtype=int)
    for peak in order:
        if len(taken) and np.min(np.abs(taken - peak)) < min_samples:
            continue
        kept.append(peak)
        taken = np.append(taken, peak)
    return np.sort(np.array(kept, dtype=int))
