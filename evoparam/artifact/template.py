"""Artifact templates: group median, moving (local) median, and learned bases."""

import warnings

import numpy as np

from evoparam.errors import ArtifactError, SettingsError

from ._common import GROUP_BY, check_trialset, groups, window_slice

__all__ = ["estimate_artifact_template"]

# Fewest pulses from which a group template is estimated.
MIN_PULSES = 3

# Correlation between a template and the mean response just after the window above which
# the template probably contains response, not only artifact.
RESPONSE_CORRELATION_WARN = 0.5


def _window_values(ts, inds, masks):
    """Window samples, with masked (for example saturated) samples set to NaN."""

    values = ts.data[:, :, inds].copy()
    if masks is not None:
        values[masks[:, :, inds]] = np.nan
    return values


def _nan_statistic(values, statistic, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN columns
        out = (np.nanmedian if statistic == "median" else np.nanmean)(values, axis=axis)
    return np.nan_to_num(out, nan=0.0)


def estimate_artifact_template(ts, window=(-0.002, 0.008), *, group_by=GROUP_BY,
                               statistic="median"):
    """Artifact template of each group of pulses.

    ``T_g(t) = median_{k in g} x_k(t)`` on the window, per channel (median rather than
    mean to resist outliers). Masked samples (for example saturated) are left out.

    Parameters
    ----------
    ts : TrialSet
    window : (float, float)
        Artifact window, in seconds relative to the onset.
    group_by : tuple of str
        Metadata columns that define the groups (missing columns are ignored).
    statistic : {"median", "mean"}

    Returns
    -------
    templates : dict
        Group key (tuple of values) -> array [n_channels, n_window_samples].
    """

    check_trialset(ts)
    if statistic not in ("median", "mean"):
        raise SettingsError("statistic must be 'median' or 'mean'.")
    inds = window_slice(ts, window)
    values = _window_values(ts, inds, ts.masks)
    return {key: _nan_statistic(values[rows], statistic) for key, rows in groups(ts, group_by)}


def _neighbours(times, row, n):
    """Indices (into ``times``) of the ``n`` pulses closest in time to ``row``, itself excluded."""

    order = np.argsort(np.abs(times - times[row]), kind="stable")
    order = order[order != row]
    return order[:n]


def subtract_template(ts, inds, *, template="global", group_by=GROUP_BY, n_neighbors=10,
                      scale=True, fallback=False, interpolate=None):
    """Template subtraction on the window ``inds``.

    Returns (data, info). ``info`` holds the template(s), the per-trial fit r²
    (``fit_r2``), the response correlation, and the trials that fell back to
    interpolation (``fallback_trials``).
    """

    data = ts.data.copy()
    values = _window_values(ts, inds, ts.masks)
    raw = ts.data[:, :, inds]
    n_trials, n_channels, n_window = raw.shape
    fit_r2 = np.ones(n_trials)
    fallback_trials = []
    templates = {}
    after = inds[-1] + 1 + np.arange(n_window)
    after = after[after < ts.n_times]

    if isinstance(template, np.ndarray) or isinstance(template, (list, tuple)):
        user = np.asarray(template, dtype=float)
        if user.ndim == 1:
            user = np.broadcast_to(user, (n_channels, n_window))
        if user.shape != (n_channels, n_window):
            raise SettingsError(f"template must have shape ({n_channels}, {n_window}) or "
                                f"({n_window},), got {user.shape}.")
        per_trial = np.broadcast_to(user, raw.shape)
        templates[("user",)] = user
    elif template in ("global", "local"):
        per_trial = np.zeros(raw.shape)
        order_times = (ts.event_times if ts.event_times is not None
                       else np.arange(n_trials, dtype=float))
        for key, rows in groups(ts, group_by):
            if len(rows) < MIN_PULSES:
                if not fallback:
                    raise ArtifactError(
                        f"Template group {key} has {len(rows)} pulse(s); at least {MIN_PULSES} "
                        "are needed. Use fallback=True to interpolate these trials instead, "
                        "or group_by=() for one template over all trials.")
                warnings.warn(f"Template group {key} has {len(rows)} pulse(s): these trials "
                              "are interpolated instead.", stacklevel=3)
                fallback_trials.extend(rows.tolist())
                continue
            if template == "global":
                tpl = _nan_statistic(values[rows], "median")
                per_trial[rows] = tpl
                templates[key] = tpl
            else:
                local_times = order_times[rows]
                for pos, row in enumerate(rows):
                    near = rows[_neighbours(local_times, pos, n_neighbors)]
                    per_trial[row] = _nan_statistic(values[near], "median")
                templates[key] = _nan_statistic(values[rows], "median")
    else:
        raise SettingsError("template must be 'global', 'local' or an array.")

    if scale:
        num = np.sum(raw * per_trial, axis=-1)
        den = np.sum(per_trial * per_trial, axis=-1)
        alpha = np.divide(num, den, out=np.ones_like(num), where=den > 0)
    else:
        alpha = np.ones(raw.shape[:2])
    fitted = alpha[..., None] * per_trial
    residual = raw - fitted
    data[:, :, inds] = residual

    total = np.sum((raw - raw.mean(axis=-1, keepdims=True)) ** 2, axis=(1, 2))
    fit_r2 = 1 - np.divide(np.sum(residual ** 2, axis=(1, 2)), total,
                           out=np.zeros(n_trials), where=total > 0)
    fit_r2 = np.clip(fit_r2, 0.0, 1.0)

    # Response protection: a template that looks like the response just after it
    correlation = 0.0
    if len(after) == n_window and templates:
        response = ts.data[:, :, after].mean(axis=0)
        for tpl in templates.values():
            a, b = np.ravel(tpl), np.ravel(response)
            if np.std(a) > 0 and np.std(b) > 0:
                correlation = max(correlation, float(np.corrcoef(a, b)[0, 1]))
        if correlation > RESPONSE_CORRELATION_WARN:
            warnings.warn(f"The artifact template correlates with the response after the window "
                          f"(r = {correlation:.2f}): the window may end inside the response.",
                          stacklevel=3)

    if fallback_trials and interpolate is not None:
        rows = np.array(fallback_trials)
        data[rows] = interpolate(ts.data[rows])
        fit_r2[rows] = 0.0

    stacked = np.stack(list(templates.values())) if templates else None
    return data, {"templates": templates, "template": stacked, "fit_r2": fit_r2,
                  "template_response_correlation": correlation,
                  "fallback_trials": sorted(fallback_trials), "alpha": alpha}


def subtract_learned(ts, inds, *, group_by=GROUP_BY, n_components=None, ridge=1e-3,
                     variance=0.95, max_components=8):
    """Learned artifact basis per group: PCA (uncentred SVD) of the window waveforms of
    every pulse and channel of the group; each window is ridge-regressed on the basis and
    the fit is subtracted.

    Returns (data, info) with ``basis`` [n_basis_total, n_window] (the groups' bases
    stacked), ``basis_groups`` (group index of each basis row) and ``fit_r2``.
    """

    if ridge < 0:
        raise SettingsError("ridge must be >= 0.")
    data = ts.data.copy()
    raw = ts.data[:, :, inds]
    n_trials, n_channels, n_window = raw.shape
    values = _window_values(ts, inds, ts.masks)
    fit_r2 = np.zeros(n_trials)
    bases, owners = [], []
    for g, (key, rows) in enumerate(groups(ts, group_by)):
        waves = values[rows].reshape(-1, n_window)
        waves = waves[~np.isnan(waves).any(axis=1)]
        if len(waves) < 2:
            raise ArtifactError(f"Group {key}: not enough unmasked pulses to learn a basis.")
        _, sing, vt = np.linalg.svd(waves, full_matrices=False)
        if n_components is None:
            energy = np.cumsum(sing ** 2) / np.sum(sing ** 2)
            k = int(min(np.searchsorted(energy, variance) + 1, max_components, len(sing)))
        else:
            k = int(min(n_components, len(sing)))
        basis = vt[:k]                                        # orthonormal rows
        coefs = (raw[rows] @ basis.T) / (1.0 + ridge)
        fitted = coefs @ basis
        residual = raw[rows] - fitted
        data[np.ix_(rows, np.arange(n_channels), inds)] = residual
        total = np.sum((raw[rows] - raw[rows].mean(axis=-1, keepdims=True)) ** 2, axis=(1, 2))
        fit_r2[rows] = np.clip(1 - np.divide(np.sum(residual ** 2, axis=(1, 2)), total,
                                             out=np.zeros(len(rows)), where=total > 0), 0, 1)
        bases.append(basis)
        owners.extend([g] * k)
    return data, {"basis": np.vstack(bases), "basis_groups": owners, "fit_r2": fit_r2}
