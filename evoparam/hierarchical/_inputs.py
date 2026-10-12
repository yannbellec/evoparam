# SPDX-License-Identifier: Apache-2.0
"""Per-subject inputs of the hierarchical model: channel fits, trial fits, templates.

For every subject: the channel fits of spec §5.5 (``evoparam.models.fit``), the trial fits of
§5.7 (``evoparam.trials.fit_trials``, mode ``shift_scale``) of the channels that carry the
component, and, for every channel (responding or not), the per-trial amplitude of the
subject's template of the component (least squares with an offset, the template's
latency and width fixed). The template amplitude is the existence statistic: it has a
known null (mean zero) on channels without a response, which the fitted amplitude has not
(a fit is only reported where a component was found).
"""

import warnings
from dataclasses import dataclass, field

import numpy as np

from evoparam.data import TrialSet
from evoparam.errors import DataError, SettingsError
from evoparam.models.component_shapes import get_shape

__all__ = []

# Template window: the component's latency +/- this many FWHM.
TEMPLATE_HALF_WIDTH = 1.5
# Fewer channels with the component than this in a subject: the group template is used.
MIN_TEMPLATE_CHANNELS = 3


@dataclass
class SubjectData:
    """The inputs of one subject."""

    name: str
    ts: TrialSet
    group_fit: object
    trial_fits: object = None
    covariates: dict = field(default_factory=dict)


def split_subjects(ts, subject_column):
    """List of (name, TrialSet): one per element of a sequence, or one per value of the
    metadata column ``subject_column`` of a single TrialSet.
    """
    if isinstance(ts, TrialSet):
        meta = ts.metadata
        if meta is not None and subject_column in meta.columns:
            out = []
            labels = meta[subject_column].astype(str).to_numpy()
            for name in dict.fromkeys(labels):
                out.append((name, ts.select(trials=np.flatnonzero(labels == name))))
            return out
        return [("sub-1", ts)]
    sets = list(ts)
    if not sets or not all(isinstance(s, TrialSet) for s in sets):
        raise DataError("fit_hierarchical expects a TrialSet or a sequence of TrialSets.")
    out = []
    for i, s in enumerate(sets):
        meta = s.metadata
        if meta is not None and subject_column in meta.columns and len(meta):
            values = meta[subject_column].astype(str).unique()
            if len(values) > 1:
                raise DataError(
                    f"TrialSet {i} holds several subjects ({list(values)}); pass one "
                    f"TrialSet per subject or a single TrialSet."
                )
            name = values[0]
        else:
            name = f"sub-{i + 1}"
        out.append((name, s))
    names = [n for n, _ in out]
    if len(set(names)) != len(names):
        raise DataError(f"Subject names repeat: {names}.")
    return out


def subject_covariates(name, ts, columns):
    """Values of subject-level covariates: constant metadata columns of the subject."""
    out = {}
    for column in columns:
        meta = ts.metadata
        if meta is None or column not in meta.columns:
            raise SettingsError(
                f"formula: the column {column!r} is not in the metadata of subject {name!r}."
            )
        values = meta[column].dropna().unique()
        if len(values) != 1:
            raise SettingsError(
                f"formula: {column!r} must be constant within a subject "
                f"(subject {name!r} has {len(values)} values)."
            )
        out[column] = values[0]
    return out


def channel_fits(ts, preset, picks=None, n_jobs=1):
    from evoparam.models import fit as fit_models

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_models(ts, preset, picks=picks, n_jobs=n_jobs)


def trial_fits(ts, group_fit, label, max_shift, random_state=None):
    from evoparam.trials import fit_trials

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_trials(
            ts,
            group_fit,
            components=[label],
            by=(),
            max_shift=max_shift,
            random_state=random_state,
            verbose=False,
        )


def run_subjects(function, arguments, n_jobs):
    """``function(*args)`` for each subject's arguments, in processes when ``n_jobs != 1``
    (``-1``: one per core).
    """
    if n_jobs == 1 or len(arguments) < 2:
        return [function(*args) for args in arguments]
    import os
    from concurrent.futures import ProcessPoolExecutor

    workers = os.cpu_count() if n_jobs is None or n_jobs < 0 else n_jobs
    with ProcessPoolExecutor(max_workers=min(workers, len(arguments))) as pool:
        return list(pool.map(function, *zip(*arguments, strict=False)))


def component_rows(group_fit, label):
    """For each channel of a GroupFit: the component carrying ``label`` (or None)."""
    out = {}
    for name in group_fit.ch_names:
        fit = group_fit.get(name)
        comps = [c for c in fit.components if c.label == label] if fit.status == "ok" else []
        out[name] = max(comps, key=lambda c: abs(c.amplitude)) if comps else None
    return out


def comp_window(fit, label):
    """The label window ``(low, high)`` (s) of a channel fit: the latency of a component
    carrying ``label`` lies in it (``(-inf, inf)`` when the fit has no such window).
    """
    windows = (getattr(fit, "settings", None) or {}).get("windows") or {}
    window = windows.get(label)
    if window is None or len(window) < 2:
        return -np.inf, np.inf
    return float(window[0]), float(window[1])


def wald_se(comp, name):
    """Standard error from a component's Wald interval (NaN when absent)."""
    ci = (comp.ci or {}).get(name)
    if ci is None or not np.all(np.isfinite(ci)):
        return np.nan
    return float(ci[1] - ci[0]) / (2 * 1.959963984540054)


def template(components):
    """Shape name and median parameters of a list of components."""
    names = [c.shape for c in components]
    shape = max(set(names), key=names.count)
    params = np.median(np.array([c.params for c in components if c.shape == shape]), axis=0)
    latency = float(np.median([c.latency for c in components if c.shape == shape]))
    fwhm = float(np.median([c.fwhm for c in components if c.shape == shape]))
    polarity = int(np.sign(np.median([c.amplitude for c in components])))
    return {
        "shape": shape,
        "params": params,
        "latency": latency,
        "fwhm": fwhm,
        "polarity": polarity,
    }


def template_amplitudes(ts, tmpl):
    """Per-trial amplitude of a fixed template on every channel: (n_trials, n_channels).

    Least squares of the samples within ``latency +/- 1.5 FWHM`` on the template (unit peak)
    and an offset; masked samples are left out.
    """
    times = ts.times
    curve = get_shape(tmpl["shape"]).evaluate(times, tmpl["params"])
    peak = np.max(np.abs(curve))
    if not np.isfinite(peak) or peak == 0:
        return np.full((ts.n_trials, ts.n_channels), np.nan)
    curve = curve / peak
    half = TEMPLATE_HALF_WIDTH * max(tmpl["fwhm"], 2.0 / ts.sfreq)
    win = (times >= tmpl["latency"] - half) & (times <= tmpl["latency"] + half)
    x = ts.data[:, :, win]
    design = np.column_stack([curve[win], np.ones(win.sum())])
    if ts.masks is None:
        pinv = np.linalg.pinv(design)
        return np.einsum("j,icj->ic", pinv[0], np.nan_to_num(x))
    out = np.full((ts.n_trials, ts.n_channels), np.nan)
    masks = ts.masks[:, :, win]
    for i in range(ts.n_trials):
        for c in range(ts.n_channels):
            ok = ~masks[i, c] & np.isfinite(x[i, c])
            if ok.sum() >= 3:
                coef, *_ = np.linalg.lstsq(design[ok], x[i, c, ok], rcond=None)
                out[i, c] = coef[0]
    return out
