# SPDX-License-Identifier: Apache-2.0
"""Sequential low-rank regression across channels, pulses and trials.

Implemented from the published method for estimating and removing stimulation artifacts
by regression across channels, pulses and trials (O'Shea & Shenoy, 2018, J Neural Eng,
"ERAASR"): an artifact shared by many channels, pulses and trials is low-rank, while a
neural response local to one channel and not locked to the pulses is not.

One addition to the published steps: in the across-channel step, the principal-component
scores of the other channels are denoised by a low-rank approximation across pulse
segments (rank ``K_P``) before the regression. With few channels (24 rather than about
100), regressing on raw scores injects the other channels' noise into the cleaned channel
(errors in the regressors); the denoised scores keep the artifact and drop that noise.
"""

import numpy as np

from evoparam._typing import Any, ArrayLike, FloatArray, NDArray
from evoparam.errors import ArtifactError, SettingsError

__all__ = []

# Fewest samples of a pulse period, and of the window after the train fitted for the slow
# transient.
_MIN_PERIOD_SAMPLES = 2
_MIN_POST_SAMPLES = 2

MIN_CHANNELS, MIN_PULSES, MIN_TRIALS = 8, 4, 4


def _top_scores(matrix: Any, k: Any) -> NDArray[Any]:
    """Scores (n_rows x k) of the top-k principal components of the columns of ``matrix``."""
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    k = min(k, len(s))
    scores: NDArray[Any] = u[:, :k] * s[:k]
    return scores


def _regress_out(target: Any, scores: Any) -> Any:
    """Residual of ``target`` (n_rows,) after least squares on ``scores`` and an intercept."""
    design = np.c_[np.ones(len(target)), scores]
    coefs, *_ = np.linalg.lstsq(design, target, rcond=None)
    return target - design @ coefs


def _denoise_scores(scores: Any, blocks: Any, rank: Any) -> Any:
    """Low-rank approximation of each score series seen as (blocks x samples per block).

    The artifact repeats over pulses and trials, so each component's time course is
    low-rank once cut into pulse segments; the noise is not, and is largely removed.
    """
    out = np.empty_like(scores)
    for k in range(scores.shape[1]):
        mat = scores[:, k].reshape(blocks, -1)
        u, s, vt = np.linalg.svd(mat, full_matrices=False)
        r = max(1, min(rank, len(s)))
        out[:, k] = ((u[:, :r] * s[:r]) @ vt[:r]).ravel()
    return out


def _leave_out_pass(
    matrix: Any, k: Any, exclude: Any, blocks: Any = None, denoise_rank: Any = None
) -> Any:
    """For each column j, regress out the top-k PCs of the columns not in ``exclude(j)``.

    With ``blocks``, the PC scores are first denoised by a rank-``denoise_rank``
    approximation across the blocks (pulse segments).
    """
    out = matrix.copy()
    n_cols = matrix.shape[1]
    for j in range(n_cols):
        keep = [c for c in range(n_cols) if c not in exclude(j)]
        if not keep:
            continue
        scores = _top_scores(matrix[:, keep], k)
        if blocks is not None:
            scores = _denoise_scores(scores, blocks, denoise_rank)
        out[:, j] = _regress_out(matrix[:, j], scores)
    return out


def lowrank_clean(
    data: NDArray[Any],
    sfreq: float,
    times: FloatArray,
    pulse_times: ArrayLike,
    *,
    ranks: Any = (4, 2, 4),
    post_window: float = 0.03,
    neighbours: int = 1,
) -> tuple[NDArray[Any], dict[str, Any]]:
    """Remove an artifact train by sequential low-rank regression.

    Parameters
    ----------
    data : array [n_trials, n_channels, n_times]
    sfreq : float
    times : 1d array
    pulse_times : 1d array
        Pulse onsets within each trial, in seconds (relative to the trial onset).
    ranks : (int, int, int)
        Components removed across channels, pulses and trials ``(K_C, K_P, K_R)``.
    post_window : float
        Window after the last pulse cleaned across trials (the slow transient), in s.
    neighbours : int
        Channels on each side (by index) left out with the channel itself when its
        across-channel components are computed.

    Returns
    -------
    cleaned : array like ``data``
    info : dict
        ``segment_inds`` (samples of the pulse train), ``post_inds`` and
        ``removed_fraction`` (fraction of the train-window variance removed).
    """
    k_c, k_p, k_r = (int(k) for k in ranks)
    if min(k_c, k_p, k_r) < 0:
        raise SettingsError("ranks must be non-negative.")
    n_trials, n_channels, n_times = data.shape
    pulses = np.sort(np.asarray(pulse_times, dtype=float))
    if n_channels < MIN_CHANNELS:
        raise ArtifactError(
            f"method 'lowrank' needs at least {MIN_CHANNELS} channels, got {n_channels}."
        )
    if len(pulses) < MIN_PULSES:
        raise ArtifactError(
            f"method 'lowrank' needs at least {MIN_PULSES} pulses per train, got {len(pulses)}."
        )
    if n_trials < MIN_TRIALS:
        raise ArtifactError(
            f"method 'lowrank' needs at least {MIN_TRIALS} trials, got {n_trials}."
        )
    starts = np.searchsorted(times, pulses)
    length = int(np.min(np.diff(starts)))
    if starts[0] < 0 or starts[-1] + length > n_times or length < _MIN_PERIOD_SAMPLES:
        raise ArtifactError("The pulse train does not fit inside the epoch.")

    cleaned = np.array(data, dtype=float, copy=True)
    seg_inds = (starts[:, None] + np.arange(length)[None]).ravel()  # P x L
    seg = cleaned[:, :, seg_inds].reshape(n_trials, n_channels, len(pulses), length)
    before = seg.copy()

    # 1. Across channels: (R P L) x C
    mat: Any
    mat = seg.transpose(0, 2, 3, 1).reshape(-1, n_channels)
    mat = _leave_out_pass(
        mat,
        k_c,
        lambda j: set(range(j - neighbours, j + neighbours + 1)),
        blocks=n_trials * len(pulses),
        denoise_rank=max(1, k_p),
    )
    seg = mat.reshape(n_trials, len(pulses), length, n_channels).transpose(0, 3, 1, 2)

    # 2. Across pulses, per channel: (R L) x P
    for ch in range(n_channels):
        mat = seg[:, ch].transpose(0, 2, 1).reshape(-1, len(pulses))
        mat = _leave_out_pass(mat, k_p, lambda j: {j})
        seg[:, ch] = mat.reshape(n_trials, length, len(pulses)).transpose(0, 2, 1)

    # 3. Across trials, per channel: (P L) x R
    for ch in range(n_channels):
        mat = seg[:, ch].reshape(n_trials, -1).T
        seg[:, ch] = _leave_out_pass(mat, k_r, lambda j: {j}).T.reshape(
            n_trials, len(pulses), length
        )

    cleaned[:, :, seg_inds] = seg.reshape(n_trials, n_channels, -1)

    # 4. Slow transient after the train, across trials
    post_start = starts[-1] + length
    post_inds = np.arange(post_start, min(n_times, post_start + int(round(post_window * sfreq))))
    if len(post_inds) >= _MIN_POST_SAMPLES and k_r > 0:
        for ch in range(n_channels):
            mat = cleaned[:, ch][:, post_inds].T
            cleaned[:, ch][:, post_inds] = _leave_out_pass(mat, k_r, lambda j: {j}).T

    total = np.var(before)
    removed = 1 - np.var(seg) / total if total > 0 else 0.0
    return cleaned, {
        "segment_inds": seg_inds,
        "post_inds": post_inds,
        "removed_fraction": float(removed),
    }
