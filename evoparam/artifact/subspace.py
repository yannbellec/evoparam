# SPDX-License-Identifier: Apache-2.0
"""Pre-whitening and projection away from the artifact subspace.

Implemented from the published pre-whitening and null-projection approach to
stimulation-artifact removal in multichannel EEG: the channels are whitened with the
baseline covariance, the few spatial directions that carry most of the artifact-window
energy are found by SVD, and the data are projected away from them.
"""

import numpy as np

from evoparam.errors import ArtifactError, SettingsError

__all__ = []

# Energy of the artifact window removed when the number of directions is automatic.
ENERGY = 0.95
MAX_DIRECTIONS = 6
# ``n_components="noise"``: directions whose whitened window energy per sample exceeds this
# multiple of the largest eigenvalue expected from white noise (Marchenko-Pastur edge)
NOISE_FACTOR = 2.0


def _sqrtm(cov, inverse=False):
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, vals.max() * 1e-12, None)
    power = -0.5 if inverse else 0.5
    return (vecs * vals ** power) @ vecs.T


def subspace_clean(data, window_inds, baseline_inds, *, n_components=None, extent="all"):
    """Remove the artifact directions found in the window from every sample.

    ``X_w = S^{-1/2} (X - mu)`` with ``S`` the baseline covariance across channels and
    ``mu`` the baseline mean of each trial and channel; ``U_a`` holds the top ``k`` left
    singular vectors of ``X_w`` on the window (all trials), ``k`` the smallest number
    that reaches 95 % of the window energy (at most 6) unless ``n_components`` is given;
    ``X_hat = S^{1/2} (I - U_a U_a^T) X_w + mu``.

    ``n_components="noise"`` keeps every direction whose window energy per sample is above
    ``NOISE_FACTOR`` times the largest eigenvalue of whitened noise,
    ``(1 + sqrt(n_channels / n_samples))^2`` (at most ``n_channels - 1``).
    ``extent="window"`` projects the samples of the window only (the rest of the epoch is
    left as recorded); ``"all"`` (the specification) projects every sample.

    Returns
    -------
    cleaned : array like ``data``
    info : dict
        ``n_directions``, ``directions`` (whitened space, channels x k) and
        ``window_energy_removed``.
    """

    n_trials, n_channels, _ = data.shape
    if n_channels < 2:
        raise ArtifactError("method 'subspace' needs at least 2 channels.")
    if len(baseline_inds) < n_channels + 1:
        raise ArtifactError(f"method 'subspace' needs a baseline longer than the number of "
                            f"channels ({n_channels}) to estimate their covariance.")
    if extent not in ("all", "window"):
        raise SettingsError("extent must be 'all' or 'window'.")
    if n_components not in (None, "energy", "noise") and not (
            isinstance(n_components, (int, np.integer)) and 1 <= n_components < n_channels):
        raise SettingsError("n_components must be None, 'energy', 'noise' or an integer "
                            "between 1 and n_channels - 1.")
    mu = data[:, :, baseline_inds].mean(axis=-1, keepdims=True)
    centred = data - mu
    base = centred[:, :, baseline_inds].transpose(1, 0, 2).reshape(n_channels, -1)
    cov = base @ base.T / base.shape[1]
    cov += np.eye(n_channels) * 1e-10 * np.trace(cov) / n_channels
    w_inv, w = _sqrtm(cov, inverse=True), _sqrtm(cov)

    white = np.einsum("ij,kjt->kit", w_inv, centred)
    win = white[:, :, window_inds].transpose(1, 0, 2).reshape(n_channels, -1)
    u, s, _ = np.linalg.svd(win, full_matrices=False)
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    if n_components in (None, "energy"):
        k = int(min(np.searchsorted(energy, ENERGY) + 1, MAX_DIRECTIONS, n_channels - 1))
    elif n_components == "noise":
        edge = (1.0 + np.sqrt(n_channels / win.shape[1])) ** 2
        k = int(np.clip(np.sum(s ** 2 / win.shape[1] > NOISE_FACTOR * edge), 1,
                        n_channels - 1))
    else:
        k = int(n_components)
    directions = u[:, :k]
    projector = directions @ directions.T
    if extent == "all":
        projected = white - np.einsum("ij,kjt->kit", projector, white)
    else:
        projected = white.copy()
        projected[:, :, window_inds] -= np.einsum("ij,kjt->kit", projector,
                                                  white[:, :, window_inds])
    cleaned = np.einsum("ij,kjt->kit", w, projected) + mu
    return cleaned, {"n_directions": k, "directions": directions,
                     "window_energy_removed": float(energy[k - 1])}
