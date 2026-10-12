# SPDX-License-Identifier: Apache-2.0
"""Shared helpers: the artifact score, robust scaling, channel picks and artifact snippets."""

import numpy as np

from evoparam.errors import DataError

__all__ = []

# Number of samples processed at once by the score, to bound memory.
_CHUNK = 1 << 14

# Scale factor from the median absolute deviation to the standard deviation of a Gaussian.
MAD_TO_SD = 1.4826


def as_continuous(data):
    """Return ``data`` as a float 2d array [n_channels, n_times]."""

    data = np.asarray(data)
    if data.ndim == 1:
        data = data[np.newaxis]
    if data.ndim != 2:
        raise DataError("data must be continuous, as [n_times] or [n_channels, n_times].")
    if data.shape[1] < 3:
        raise DataError("data must have at least 3 samples.")
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(float)
    return data


def pick_index(n_channels, picks, ch_names=None):
    """Channel indices from ``picks`` (None, indices, bool mask or names)."""

    if picks is None:
        return np.arange(n_channels)
    picks = np.atleast_1d(np.asarray(picks))
    if picks.dtype == bool:
        if picks.size != n_channels:
            raise DataError("picks: boolean mask must have one value per channel.")
        index = np.flatnonzero(picks)
    elif picks.dtype.kind in "iu":
        index = picks.astype(int)
    else:
        if ch_names is None:
            raise DataError("picks: channel names need ch_names.")
        names = list(ch_names)
        missing = [name for name in picks if name not in names]
        if missing:
            raise DataError(f"picks: unknown channel(s) {missing}.")
        index = np.array([names.index(name) for name in picks])
    if index.size == 0:
        raise DataError("picks: no channel selected.")
    if index.min() < 0 or index.max() >= n_channels:
        raise DataError("picks: channel index out of range.")
    return index


# The score uses at most this many channels, evenly spaced among the picks: the median
# of 16 channels already ignores transients on a few of them, and the median over all
# channels of a large grid would cost seconds per minute of data.
SCORE_CHANNELS = 16


def score_index(index, n_score=SCORE_CHANNELS):
    """Evenly spaced subset of ``index`` used for the score."""

    if n_score is None or len(index) <= n_score:
        return np.asarray(index)
    keep = np.unique(np.linspace(0, len(index) - 1, n_score).round().astype(int))
    return np.asarray(index)[keep]


def difference_score(data, index):
    """Median across channels of the absolute sample-to-sample change.

    Element ``i`` measures the change from sample ``i`` to sample ``i + 1``, so the
    score has ``n_times - 1`` values. The data are processed in chunks, so the memory
    used is a few times one chunk, whatever the recording length. NaN samples
    contribute a change of zero. With an even number of channels the upper of the two
    middle values is used (one partition instead of a full median).
    """

    n_times = data.shape[1]
    index = np.asarray(index)
    middle = len(index) // 2
    score = np.empty(n_times - 1)
    for start in range(0, n_times - 1, _CHUNK):
        stop = min(start + _CHUNK, n_times - 1)
        block = np.abs(np.diff(data[index, start : stop + 1], axis=1))
        np.nan_to_num(block, copy=False, nan=0.0)
        if len(index) == 1:
            score[start:stop] = block[0]
        else:
            score[start:stop] = np.partition(block, middle, axis=0)[middle]
    return score


def robust_z(values):
    """Values centred on their median and scaled by their MAD (as a standard deviation)."""

    center = np.median(values)
    spread = MAD_TO_SD * np.median(np.abs(values - center))
    if not spread > 0:
        spread = np.std(values)
    if not spread > 0:
        spread = 1.0
    return (values - center) / spread


def crossings(z, threshold):
    """Indices where ``z`` goes from at or below ``threshold`` to above it."""

    above = z > threshold
    index = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    if above[0]:
        index = np.r_[0, index]
    return index


def far_from(positions, others, min_samples):
    """Boolean mask: positions at least ``min_samples`` away from every sorted ``others``."""

    positions = np.asarray(positions)
    if len(others) == 0 or len(positions) == 0:
        return np.ones(len(positions), dtype=bool)
    others = np.asarray(others)
    right = np.searchsorted(others, positions)
    dist = np.full(len(positions), np.inf)
    has_right = right < len(others)
    dist[has_right] = others[right[has_right]] - positions[has_right]
    has_left = right > 0
    dist[has_left] = np.minimum(dist[has_left], positions[has_left] - others[right[has_left] - 1])
    return dist >= min_samples


def snippet_lengths(sfreq):
    """Samples before and after the onset used for artifact snippets (-1 ms to +3 ms),
    and the largest shift tried when matching (0.5 ms, at least one sample)."""

    return (
        max(1, int(round(0.001 * sfreq))),
        max(3, int(round(0.003 * sfreq))),
        max(1, int(round(0.0005 * sfreq))),
    )


def shifted_snippets(data, index, onsets, n_pre, n_post, max_shift):
    """Snippets at every shift from ``-max_shift`` to ``+max_shift`` samples.

    Returns a 3d array [n_onsets, 2 * max_shift + 1, n_features]; the middle row of the
    second axis is the unshifted snippet.
    """

    onsets = np.asarray(onsets, dtype=int)
    shifts = np.arange(-max_shift, max_shift + 1)
    out = np.zeros((len(onsets), len(shifts), len(index) * (n_pre + n_post)))
    for col, shift in enumerate(shifts):
        out[:, col] = snippets(data, index, onsets + shift, n_pre, n_post)
    return out


def snippets(data, index, onsets, n_pre, n_post):
    """Unit-norm artifact snippets: the first difference of each channel around each onset.

    Using the first difference removes the channel offsets, so the normalised
    cross-correlation compares the shape and the spatial pattern of the artifact.
    Snippets that would run past either end of the recording are zero.

    Returns
    -------
    features : 2d array, [n_onsets, len(index) * (n_pre + n_post)]
    """

    n_times = data.shape[1]
    width = n_pre + n_post
    features = np.zeros((len(onsets), len(index) * width))
    for row, onset in enumerate(onsets):
        start, stop = onset - n_pre - 1, onset + n_post
        if start < 0 or stop > n_times:
            continue
        values = np.diff(data[index, start:stop], axis=1).ravel()
        np.nan_to_num(values, copy=False, nan=0.0)
        norm = np.linalg.norm(values)
        if norm > 0:
            features[row] = values / norm
    return features
