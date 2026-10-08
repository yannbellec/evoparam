"""Pixel statistics of time-frequency maps (spec §5.10, method 4).

The tested quantity is, per trial, the normalised power minus its baseline level (see
:func:`evoparam.tf.normalize.normalize`). Pixel-wise one-sample ``t`` across trials, then:

- ``fdr``: Benjamini-Hochberg over the pixels of the test window;
- ``cluster``: sign-flip permutation, cluster mass (sum of ``|t|``) of the clusters of
  pixels above the two-sided ``t`` threshold at ``cluster_alpha`` (Maris and Oostenveld 2007);
  family-wise error by the maximum mass of each permutation;
- ``tfce``: threshold-free cluster enhancement (``E = 0.5``, ``H = 2``, step 0.2 in ``t``;
  Smith and Nichols 2009) with the maximum of each permutation.
"""

import numpy as np
from scipy import ndimage, stats as sstats

from evoparam.errors import SettingsError

__all__ = ["STATS", "t_map", "fdr_significant", "test_map"]

STATS = ("fdr", "cluster", "tfce", "none")
MIN_TRIALS = 3
# TFCE step in t units (the default of MNE-Python's threshold-free clustering).
TFCE_E, TFCE_H, TFCE_DH = 0.5, 2.0, 0.2
PERM_CHUNK = 64


def t_map(d):
    """One-sample ``t`` across trials (axis 0), NaN-aware. Returns (t, n)."""

    n = np.sum(np.isfinite(d), axis=0)
    filled = np.nan_to_num(d)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = filled.sum(0) / n
        var = (np.sum(filled ** 2, 0) - n * mean ** 2) / (n - 1)
        t = mean / np.sqrt(var / n)
    t[n < MIN_TRIALS] = np.nan
    return t, n


def p_values(t, n):
    """Two-sided p-values of ``t`` with ``n - 1`` degrees of freedom."""

    with np.errstate(invalid="ignore"):
        p = 2 * sstats.t.sf(np.abs(t), np.maximum(n - 1, 1))
    p[~np.isfinite(t)] = np.nan
    return p


def fdr_significant(p, alpha):
    """Benjamini-Hochberg: True where ``p`` is at or below the step-up threshold."""

    flat = p[np.isfinite(p)]
    out = np.zeros(p.shape, bool)
    if flat.size == 0:
        return out
    ranked = np.sort(flat)
    m = ranked.size
    below = ranked <= alpha * np.arange(1, m + 1) / m
    if not below.any():
        return out
    threshold = ranked[np.flatnonzero(below).max()]
    out[np.isfinite(p)] = p[np.isfinite(p)] <= threshold
    return out


def _flipped_t(signs, filled, sumsq, n):
    """``t`` maps of sign-flipped trials: signs (B, n_trials) -> (B, n_pixels)."""

    with np.errstate(divide="ignore", invalid="ignore"):
        mean = (signs @ filled) / n
        var = (sumsq - n * mean ** 2) / (n - 1)
        return mean / np.sqrt(var / n)


def _cluster_masses(tmap, threshold):
    """Masses and labels of the supra-threshold clusters of both signs."""

    masses, labels = [], []
    for sign in (1, -1):
        lab, n_lab = ndimage.label(np.nan_to_num(sign * tmap) > threshold)
        if n_lab:
            mass = ndimage.sum_labels(np.abs(np.nan_to_num(tmap)), lab, np.arange(1, n_lab + 1))
        else:
            mass = np.zeros(0)
        masses.append(mass)
        labels.append(lab)
    return masses, labels


# Connectivity within each (frequency x time) slice of a stack, none across slices.
_STACK = np.zeros((3, 3, 3), bool)
_STACK[1] = ndimage.generate_binary_structure(2, 1)


def _null_cluster_max(stack, threshold):
    """Largest cluster mass of each map of a stack (B, F, T), both signs."""

    out = np.zeros(stack.shape[0])
    filled = np.nan_to_num(stack)
    owner = np.repeat(np.arange(stack.shape[0]), stack[0].size)
    for sign in (1, -1):
        lab, n_lab = ndimage.label(sign * filled > threshold, structure=_STACK)
        if not n_lab:
            continue
        mass = ndimage.sum_labels(np.abs(filled), lab, np.arange(1, n_lab + 1))
        which = np.zeros(n_lab + 1, int)
        which[lab.ravel()] = owner
        np.maximum.at(out, which[1:], mass)
    return out


def _tfce(stack, dh):
    """TFCE score of each pixel of each map of a stack (B, F, T), signed (both tails)."""

    out = np.zeros(stack.shape)
    filled = np.nan_to_num(stack)
    for sign in (1, -1):
        values = sign * filled
        top = values.max()
        if top <= 0:
            continue
        for h in np.arange(dh, top + dh / 2, dh):
            lab, n_lab = ndimage.label(values >= h, structure=_STACK)
            if not n_lab:
                break
            extent = np.bincount(lab.ravel())
            extent[0] = 0
            out += sign * (extent[lab] ** TFCE_E) * (h ** TFCE_H) * dh
    return out


def test_map(d, window, *, method="fdr", alpha=0.05, n_permutations=1000, rng=None,
             cluster_alpha=0.05):
    """Statistics of one map.

    Parameters
    ----------
    d : ndarray (n_trials, n_freqs, n_times)
        Tested quantity per trial (NaN for invalid pixels).
    window : bool array (n_freqs, n_times)
        Pixels that are tested (inside the test window and valid).

    Returns
    -------
    t : ndarray (n_freqs, n_times)
        ``t`` map (NaN outside ``window``).
    significant : bool ndarray (n_freqs, n_times)
    p : ndarray (n_freqs, n_times)
        Pixel p-values (uncorrected for ``fdr``; family-wise for ``tfce``; the cluster's
        p-value for ``cluster``).
    """

    if method not in STATS:
        raise SettingsError(f"stats must be one of {STATS}, not {method!r}.")
    d = np.where(window[None], d, np.nan)
    t, n = t_map(d)
    tested = window & np.isfinite(t)
    t = np.where(tested, t, np.nan)
    if method in ("fdr", "none"):
        p = p_values(t, n)
        sig = fdr_significant(p, alpha) if method == "fdr" else np.zeros(t.shape, bool)
        return t, sig & tested, p
    rows = np.flatnonzero(tested.any(axis=1))
    cols = np.flatnonzero(tested.any(axis=0))
    if not rows.size:
        return t, np.zeros(t.shape, bool), np.full(t.shape, np.nan)
    box = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    t_box, sig_box, p_box = _permutation(d[:, box[0], box[1]], t[box], n[box], tested[box],
                                         method, alpha, n_permutations, rng, cluster_alpha)
    sig = np.zeros(t.shape, bool)
    p = np.full(t.shape, np.nan)
    sig[box], p[box] = sig_box, p_box
    return t, sig, p


def _permutation(d, t, n, tested, method, alpha, n_permutations, rng, cluster_alpha):
    rng = np.random.default_rng(rng)
    n_trials = d.shape[0]
    shape = t.shape
    filled = np.nan_to_num(d).reshape(n_trials, -1)
    sumsq = np.sum(filled ** 2, 0)
    n_flat = n.reshape(-1).astype(float)
    keep = tested.reshape(-1)
    if method == "cluster":
        df = max(int(np.median(n[tested])) - 1, 1) if tested.any() else 1
        threshold = sstats.t.ppf(1 - cluster_alpha / 2, df)
        null = np.zeros(n_permutations)
    else:
        dh = TFCE_DH
        null = np.zeros(n_permutations)
    signs = rng.choice((-1.0, 1.0), size=(n_permutations, n_trials))
    signs[0] = 1.0
    for start in range(0, n_permutations, PERM_CHUNK):
        block = _flipped_t(signs[start:start + PERM_CHUNK], filled, sumsq, n_flat)
        block[:, ~keep] = np.nan
        stack = block.reshape((-1,) + shape)
        if method == "cluster":
            null[start:start + len(stack)] = _null_cluster_max(stack, threshold)
        else:
            null[start:start + len(stack)] = np.abs(_tfce(stack, dh)).reshape(
                len(stack), -1).max(axis=1)
    if method == "cluster":
        masses, labels = _cluster_masses(t, threshold)
        p = np.full(shape, np.nan)
        sig = np.zeros(shape, bool)
        for mass, lab in zip(masses, labels):
            for k, m in enumerate(mass, start=1):
                pk = (1 + np.sum(null >= m)) / (1 + n_permutations)
                region = lab == k
                p[region] = pk
                if pk <= alpha:
                    sig |= region
        return t, sig & tested, p
    score = _tfce(t[None], dh)[0]
    p = (1 + np.sum(null[:, None] >= np.abs(score).ravel()[None], 0)) / (1 + n_permutations)
    p = p.reshape(shape)
    p[~tested] = np.nan
    return t, (p <= alpha) & tested, p
