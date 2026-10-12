# SPDX-License-Identifier: Apache-2.0
"""Reliability of scores and trial-count planning (spec §5.11, methods 4 and 5).

Split-half reliability: each unit's trials (the units are subjects, or stimulation trains
for intracranial data) are split in two halves, the score is computed on each half's
average, the two scores are correlated across units and the correlation ``r`` is stepped
up to the full length by the Spearman-Brown formula ``2 r / (1 + r)``; odd versus even
trials, and the mean over random splits with its 95 % range (Parsons, Kruijt and Fox
2019).

Dependability (generalisability theory, as in the ERA toolbox of Clayson and Miller 2017):
the single-trial scores are decomposed into a variance between units ``s2_u`` and a
variance between trials within units ``s2_e`` by restricted maximum likelihood; the
dependability of a unit's mean over ``n`` trials is ``Phi(n) = s2_u / (s2_u + s2_e / n)``,
and the decision study gives the trials needed for ``Phi >= target``:
``n = target / (1 - target) * s2_e / s2_u``.

The standardised measurement error of each unit's score (Luck et al. 2021) is
``SD(trial scores) / sqrt(n)`` for a mean amplitude and a bootstrap over trials for other
scores; the root mean square over units is reported.
"""

import time as _time
import warnings

import numpy as np

from evoparam.data import Evoked, TrialSet
from evoparam.errors import DataError, SettingsError
from evoparam.errors import warn as _warn
from evoparam.existence import test_response
from evoparam.utils._optional import require

from ._scorers import TIME_SCORES, default_window, resolve_scorer
from .variance import reml

__all__ = [
    "reliability",
    "trials_needed",
    "spearman_brown",
    "dependability",
    "RELIABILITY_COLUMNS",
    "TRIALS_NEEDED_COLUMNS",
    "EXTRAPOLATIONS",
]

METHODS = ("split_half", "dependability", "both")

RELIABILITY_COLUMNS = {
    "channel": "",
    "scorer": "",
    "facet": "",
    "n_units": "",
    "n_trials_mean": "",
    "split_half_odd_even": "",
    "split_half_random": "",
    "split_half_low": "",
    "split_half_high": "",
    "var_unit": "score²",
    "var_trial": "score²",
    "dependability": "",
    "trials_needed": "",
    "target_phi": "",
    "sme_rms": "score",
    "n_units_dropped": "",
}

TRIALS_NEEDED_COLUMNS = {
    "channel": "",
    "scorer": "",
    "n_trials": "",
    "sd": "score",
    "p_significant": "",
    "extrapolated": "",
    "n_resamples": "",
}

# Random splits processed at once for scores that need the half averages (bounds memory).
_CHUNK = 64

# Bootstrap resamples per unit for the measurement error of a non-linear score.
_SME_BOOT = 200

# Laws that extrapolate a score's SD beyond the recorded trials (trials_needed).
EXTRAPOLATIONS = ("auto", "sqrt_n", "power")
# Bounds of the fitted exponent of the power law: the SD of a score does not grow with the
# trial count, and a latency approaches the 1/sqrt(n) law from below (its SD is bounded by
# the window while the average is noisy), so its local exponent lies in [-1/2, 0].
_POWER_EXPONENT = (-0.5, 0.0)
# Scores extrapolated by the power law with extrapolation="auto": the two latencies whose
# 80-trial SD the power law predicted better than the 1/sqrt(n) law on development
# simulations (pink, white and AR(1) noise, per-trial SNR 0.5 to 2; CHANGELOG). The
# fractional peak latency is not among them (the power law over-predicted it).
_POWER_SCORES = ("peak_latency", "fractional_area_latency")


###################################################################################################
## Formulas


def spearman_brown(r, factor=2.0):
    """Reliability of a test ``factor`` times longer: ``k r / (1 + (k - 1) r)``.

    Examples
    --------
    >>> round(spearman_brown(0.6), 6)
    0.75
    """
    r = np.asarray(r, float)
    out = factor * r / (1 + (factor - 1) * r)
    return float(out) if out.ndim == 0 else out


def dependability(var_unit, var_trial, n_trials):
    """Dependability ``Phi = var_unit / (var_unit + var_trial / n)`` of a mean over n trials.

    Examples
    --------
    >>> round(dependability(1.0, 10.0, 30), 3)
    0.75
    """
    n = np.asarray(n_trials, float)
    total = var_unit + var_trial / n
    out = np.divide(var_unit, total, out=np.full(np.shape(total), np.nan), where=total > 0)
    return float(out) if np.ndim(out) == 0 else out


def _trials_for(target, var_unit, var_trial):
    if var_unit <= 0:
        return np.inf
    return float(np.ceil(target / (1 - target) * var_trial / var_unit - 1e-9))


def _pearson(x, y):
    """Correlation of the columns of x and y [n_units, k] (NaN rows left out per column)."""
    ok = np.isfinite(x) & np.isfinite(y)
    n = ok.sum(axis=0)
    xs, ys = np.where(ok, x, 0.0), np.where(ok, y, 0.0)
    mx = np.divide(xs.sum(axis=0), n, out=np.full(n.shape, np.nan), where=n > 0)
    my = np.divide(ys.sum(axis=0), n, out=np.full(n.shape, np.nan), where=n > 0)
    dx, dy = np.where(ok, x - mx, 0.0), np.where(ok, y - my, 0.0)
    den = np.sqrt((dx**2).sum(axis=0) * (dy**2).sum(axis=0))
    return np.divide(
        (dx * dy).sum(axis=0), den, out=np.full(n.shape, np.nan), where=(den > 0) & (n >= 3)
    )


###################################################################################################
## Helpers


def _check_trialset(ts, caller):
    if isinstance(ts, Evoked):
        raise DataError(f"{caller} needs single trials: give a TrialSet, not an Evoked.")
    if not isinstance(ts, TrialSet):
        raise DataError(f"{caller} expects a TrialSet.")


def _trial_data(ts, index):
    """Trials of the picked channels with masked samples as NaN, and the kept trial rows."""
    keep = np.ones(ts.n_trials, bool)
    if ts.metadata is not None and "rejected" in ts.metadata.columns:
        keep &= ~ts.metadata["rejected"].fillna(False).astype(bool).to_numpy()
    data = ts.data[:, index]
    if ts.masks is not None:
        data = np.where(ts.masks[:, index], np.nan, data)
    return data, keep


def _scorer_on_window(scorer, times, window, scorer_kw):
    """Return the scorer of waveforms cropped to the window, or of whole waveforms.

    Cropped for a named score, whole for a callable; returns the scorer, its name, linearity, the
    window and the samples kept.
    """
    if isinstance(scorer, str):
        window = default_window(times) if window is None else window
        inside = (times >= window[0]) & (times <= window[1])
        func, name, linear, window = resolve_scorer(scorer, times[inside], window, **scorer_kw)
        return func, name, linear, window, inside
    func, name, linear, window = resolve_scorer(scorer, times, window, **scorer_kw)
    return func, name, linear, window, np.ones(len(times), bool)


def _nanmean(x, axis):
    finite = np.isfinite(x)
    count = finite.sum(axis=axis)
    total = np.where(finite, x, 0.0).sum(axis=axis)
    return np.divide(total, count, out=np.full(count.shape, np.nan), where=count > 0)


###################################################################################################
## Reliability


def reliability(
    ts,
    *,
    scorer="mean_amplitude",
    facets=("trial", "subject"),
    method="both",
    n_splits=1000,
    target_phi=0.70,
    random_state=None,
    window=None,
    picks=None,
    **scorer_kw,
):
    """Split-half reliability, dependability and the trials needed for a target dependability.

    Parameters
    ----------
    ts : TrialSet
        Trials of several units (subjects, or stimulation trains), the unit of each trial in
        the metadata column ``facets[1]``. Trials flagged ``rejected`` are left out.
    scorer : str or callable
        A classical score (``"mean_amplitude"``, ``"peak_amplitude"``, ``"peak_latency"``,
        ``"area"``, ``"fractional_area_latency"``, ``"fractional_peak_latency"``) in
        ``window`` with the options ``scorer_kw`` of
        :func:`evoparam.models.classical_measures`, or ``scorer(times, waveforms)`` taking
        an array [n_waveforms, n_times] and returning [n_waveforms] scores.
    facets : (str, str)
        ``("trial", <unit column>)``: trials nested in units. For intracranial data use
        the stimulation train (``("trial", "train_id")``) to measure train-to-train
        reproducibility on the same scale.
    method : {"both", "split_half", "dependability"}
    n_splits : int
        Random splits of the split-half estimate.
    target_phi : float
        Dependability the decision study aims at.
    random_state : int, Generator or None
    window : (float, float), optional
        Scoring window (s); default the post-stimulus epoch.
    picks : sequence of str, optional
        Channels (default all).

    Returns
    -------
    DataFrame
        One row per channel: ``channel``, ``scorer``, ``facet``, ``n_units``,
        ``n_trials_mean`` (mean trials per unit), ``split_half_odd_even``,
        ``split_half_random`` (mean of the Spearman-Brown corrected correlations over the
        random splits) with ``split_half_low`` / ``split_half_high`` (their 2.5 and 97.5
        percentiles), ``var_unit`` and ``var_trial`` (REML variance components of the
        single-trial scores), ``dependability`` (at ``n_trials_mean``), ``trials_needed``
        (for ``target_phi``; inf when there is no variance between units), ``target_phi``,
        ``sme_rms`` (root mean square of the units' standardised measurement errors),
        ``n_units_dropped`` (units with fewer than 2 trials). ``attrs["d_study"]`` holds
        the dependability for 1 to ``2 x`` the largest trial count, per channel.

    Raises
    ------
    DataError
        Fewer than 3 units with at least 2 trials (one trial per unit cannot separate
        the trial variance from the unit variance).

    Examples
    --------
    >>> import pandas as pd
    >>> from evoparam.data import TrialSet
    >>> rng = np.random.default_rng(0)
    >>> times = np.linspace(-0.1, 0.3, 41)
    >>> subject = np.repeat(np.arange(20), 30)
    >>> data = (rng.normal(0, 1, 20)[subject][:, None, None]
    ...         + rng.normal(0, 3, (600, 1, 41)))
    >>> ts = TrialSet(data=data, times=times, sfreq=100., ch_names=["Cz"],
    ...               metadata=pd.DataFrame({"subject": subject.astype(str)}))
    >>> out = reliability(ts, window=(0.0, 0.3), n_splits=200, random_state=0)
    >>> out[["n_units", "n_trials_mean"]].values.tolist()
    [[20.0, 30.0]]
    """
    start = _time.perf_counter()
    pd = require("pandas", extra="data", caller="evoparam.stats.reliability")
    _check_trialset(ts, "reliability")
    if method not in METHODS:
        raise SettingsError(f"method must be one of {METHODS}, not {method!r}.")
    facets = tuple(facets)
    if len(facets) != 2 or facets[0] != "trial":
        raise SettingsError(
            "facets must be ('trial', <unit column>), for example ('trial', 'subject')."
        )
    unit_col = facets[1]
    if ts.metadata is None or unit_col not in ts.metadata.columns:
        raise SettingsError(f"reliability: the metadata has no column '{unit_col}'.")
    if not 0 < target_phi < 1:
        raise SettingsError("target_phi must be in (0, 1).")
    rng = np.random.default_rng(random_state)
    names = list(ts.ch_names)
    index = (
        list(range(len(names)))
        if picks is None
        else [names.index(p) if isinstance(p, str) else int(p) for p in picks]
    )
    func, name, linear, window, inside = _scorer_on_window(scorer, ts.times, window, scorer_kw)
    data, keep = _trial_data(ts, index)
    data = data[..., inside]
    labels = ts.metadata[unit_col].astype(str).to_numpy()
    units = [np.flatnonzero(keep & (labels == u)) for u in dict.fromkeys(labels[keep])]
    dropped = sum(len(u) < 2 for u in units)
    units = [u for u in units if len(u) >= 2]
    if len(units) < 3:
        raise DataError(
            f"reliability needs at least 3 units ('{unit_col}') with 2 or more "
            f"trials; found {len(units)}."
        )
    n_units, n_ch = len(units), len(index)
    sizes = np.array([len(u) for u in units])
    trial_scores = func(data)  # (n_trials, n_ch)
    unit_of = np.full(ts.n_trials, -1)
    for k, rows in enumerate(units):
        unit_of[rows] = k

    def half_scores(selector):
        """Scores of the averages of a subset of each unit's trials -> (n_units, n_ch)."""
        if linear:
            return np.stack([_nanmean(trial_scores[rows[selector(rows)]], 0) for rows in units])
        avgs = np.stack([_nanmean(data[rows[selector(rows)]], 0) for rows in units])
        return func(avgs)

    out = {
        "split_half_odd_even": np.full(n_ch, np.nan),
        "split_half_random": np.full(n_ch, np.nan),
        "split_half_low": np.full(n_ch, np.nan),
        "split_half_high": np.full(n_ch, np.nan),
        "var_unit": np.full(n_ch, np.nan),
        "var_trial": np.full(n_ch, np.nan),
        "dependability": np.full(n_ch, np.nan),
        "trials_needed": np.full(n_ch, np.nan),
        "sme_rms": np.full(n_ch, np.nan),
    }
    if method in ("split_half", "both"):
        odd = half_scores(lambda rows: np.arange(len(rows)) % 2 == 0)
        even = half_scores(lambda rows: np.arange(len(rows)) % 2 == 1)
        out["split_half_odd_even"] = spearman_brown(_pearson(odd, even))
        out.update(_random_splits(units, data, trial_scores, func, linear, n_splits, rng))
    if method in ("dependability", "both"):
        for c in range(n_ch):
            score = trial_scores[:, c]
            ok = (unit_of >= 0) & np.isfinite(score)
            if np.unique(unit_of[ok]).size < 3:
                continue
            fit = reml(score[ok], [unit_of[ok]])
            var_u, var_e = fit["variances"][0], fit["residual"]
            n_mean = float(np.mean(np.bincount(unit_of[ok])[np.unique(unit_of[ok])]))
            out["var_unit"][c], out["var_trial"][c] = var_u, var_e
            out["dependability"][c] = dependability(var_u, var_e, n_mean)
            out["trials_needed"][c] = _trials_for(target_phi, var_u, var_e)
    out["sme_rms"] = _sme_rms(units, data, trial_scores, func, linear, rng)

    table = pd.DataFrame(
        {
            "channel": [names[i] for i in index],
            "scorer": name,
            "facet": unit_col,
            "n_units": float(n_units),
            "n_trials_mean": float(sizes.mean()),
            **out,
            "target_phi": float(target_phi),
            "n_units_dropped": int(dropped),
        },
        columns=list(RELIABILITY_COLUMNS),
    )
    n_max = int(2 * sizes.max())
    grid = np.arange(1, n_max + 1)
    table.attrs["d_study"] = {
        "n_trials": grid.tolist(),
        **{
            names[i]: dependability(out["var_unit"][c], out["var_trial"][c], grid).tolist()
            for c, i in enumerate(index)
        },
    }
    unit = "s" if name in TIME_SCORES else ("data·s" if name == "area" else ts.units)
    table.attrs["units"] = {k: v.replace("score", unit) for k, v in RELIABILITY_COLUMNS.items()}
    settings = {
        "scorer": name,
        "facets": list(facets),
        "method": method,
        "n_splits": int(n_splits),
        "target_phi": target_phi,
        "window": list(window),
        "random_state": random_state
        if isinstance(random_state, (int, type(None)))
        else "generator",
        **{k: v for k, v in scorer_kw.items() if isinstance(v, (int, float, str, bool))},
    }
    table.attrs["settings"] = settings
    table.attrs["provenance"] = ts.provenance.with_step(
        "evoparam.stats.reliability", settings, duration_s=_time.perf_counter() - start
    ).to_dict()
    return table


def _random_splits(units, data, trial_scores, func, linear, n_splits, rng):
    """Spearman-Brown corrected correlations of random halves: mean and 95 % range."""
    n_ch = trial_scores.shape[1]
    r_sb = np.full((n_splits, n_ch), np.nan)
    for start in range(0, n_splits, _CHUNK):
        m = min(_CHUNK, n_splits - start)
        first = np.empty((m, len(units), n_ch))
        second = np.empty((m, len(units), n_ch))
        for k, rows in enumerate(units):
            n = len(rows)
            half = n // 2
            order = np.argsort(rng.random((m, n)), axis=1)
            weights = np.zeros((m, n))
            np.put_along_axis(weights, order[:, :half], 1.0, axis=1)
            if linear:
                s = trial_scores[rows]
                fin = np.isfinite(s)
                sv = np.where(fin, s, 0.0)
                for target, w in ((first, weights), (second, 1.0 - weights)):
                    num, den = w @ sv, w @ fin.astype(float)
                    target[:, k] = np.divide(
                        num, den, out=np.full(num.shape, np.nan), where=den > 0
                    )
            else:
                x = data[rows]
                flat = x.reshape(n, -1)
                fin = np.isfinite(flat)
                xv = np.where(fin, flat, 0.0)
                for target, w in ((first, weights), (second, 1.0 - weights)):
                    num, den = w @ xv, w @ fin.astype(float)
                    avg = np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)
                    target[:, k] = func(avg.reshape((m,) + x.shape[1:]))
        for j in range(m):
            r_sb[start + j] = spearman_brown(_pearson(first[j], second[j]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN channels stay NaN
        return {
            "split_half_random": np.nanmean(r_sb, axis=0),
            "split_half_low": np.nanpercentile(r_sb, 2.5, axis=0),
            "split_half_high": np.nanpercentile(r_sb, 97.5, axis=0),
        }


def _sme_rms(units, data, trial_scores, func, linear, rng):
    """Root mean square over units of the standardised measurement error of their score."""
    n_ch = trial_scores.shape[1]
    sme = np.full((len(units), n_ch), np.nan)
    for k, rows in enumerate(units):
        n = len(rows)
        if linear:
            s = trial_scores[rows]
            fin = np.isfinite(s)
            count = fin.sum(axis=0)
            mean = _nanmean(s, 0)
            dev = np.where(fin, s - mean, 0.0)
            sd = np.sqrt(
                np.divide(
                    (dev**2).sum(axis=0), count - 1, out=np.full(n_ch, np.nan), where=count > 1
                )
            )
            sme[k] = sd / np.sqrt(np.maximum(count, 1))
        else:
            pick = rng.integers(0, n, (_SME_BOOT, n))
            weights = np.zeros((_SME_BOOT, n))
            np.add.at(weights, (np.repeat(np.arange(_SME_BOOT), n), pick.ravel()), 1.0)
            x = data[rows]
            flat = x.reshape(n, -1)
            fin = np.isfinite(flat)
            num, den = weights @ np.where(fin, flat, 0.0), weights @ fin.astype(float)
            avg = np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)
            boot = func(avg.reshape((_SME_BOOT,) + x.shape[1:]))
            sme[k] = np.nanstd(boot, axis=0, ddof=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.sqrt(np.nanmean(sme**2, axis=0))


###################################################################################################
## Trial-count planning


def _extrapolate(sd, n_recorded, targets, law):
    """SD at ``targets`` (> n_recorded) from the resampled SDs ``sd`` {count: (n_ch,)}.

    ``sqrt_n``: ``c / sqrt(n)``, ``c`` the mean of ``sd(m) * sqrt(m)`` over the counts
    ``m >= N / 2``. ``power``: ``log sd`` regressed on ``log m`` over the counts
    ``m >= N / 4`` (at least two; else ``sqrt_n``), exponent clipped to [-1/2, 0]. Both
    are multiplied by ``sqrt(N / (N - 1))`` (a bootstrap of N trials uses their variance
    with divisor N). Returns {target: (n_ch,)} and the exponent per channel.
    """
    n_total = n_recorded
    resampled = sorted(sd)
    correction = np.sqrt(n_total / max(n_total - 1, 1))
    fit = [m for m in resampled if m >= n_total / 4]
    if law == "power" and len(fit) >= 2:
        logm = np.log(np.asarray(fit, float))
        dm = logm - logm.mean()
        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            logsd = np.log(np.stack([sd[m] for m in fit]))  # (n_counts, n_ch)
            slope = np.clip((dm @ (logsd - logsd.mean(axis=0))) / (dm @ dm), *_POWER_EXPONENT)
            level = np.mean(logsd - slope[None] * logm[:, None], axis=0)
            return ({n: np.exp(level + slope * np.log(n)) * correction for n in targets}, slope)
    base = [m for m in resampled if m >= n_total / 2] or resampled
    c = np.mean([sd[m] * np.sqrt(m) for m in base], axis=0) * correction
    return {n: c / np.sqrt(n) for n in targets}, np.full(np.shape(c), -0.5)


def trials_needed(
    ts,
    *,
    scorer="existence",
    steps=None,
    n_resamples=200,
    random_state=None,
    window=None,
    picks=None,
    existence_kwargs=None,
    extrapolation="auto",
    **scorer_kw,
):
    """How a score's precision, or the chance of detecting the response, grows with trials.

    Parameters
    ----------
    ts : TrialSet
        The trials of one recording (stratify by calling it per condition).
    scorer : str or callable
        ``"existence"``: the probability that the existence test (§5.6,
        :func:`evoparam.existence.test_response`, settings ``existence_kwargs``) is
        significant on ``n`` trials drawn without replacement. Otherwise a classical score
        or callable as in :func:`reliability`: the SD of the score of the average of ``n``
        trials drawn with replacement (a bootstrap of size ``n``), extrapolated beyond the
        recorded trials by the ``1 / sqrt(n)`` law.
    steps : sequence of int, optional
        Trial counts (default 5, 10, ... up to the number of trials, which is included).
        Counts above the number of trials are extrapolated (scores only).
    n_resamples : int
        Resamples per count.
    random_state : int, Generator or None
    window : (float, float), optional
        Scoring window (s); default the post-stimulus epoch (or the existence window).
    picks : sequence of str, optional
    existence_kwargs : dict, optional
        Arguments of :func:`~evoparam.existence.test_response` (default: 200 permutations,
        no correction, no bootstrap of the duration, no grouping).
    extrapolation : {"auto", "sqrt_n", "power"}
        Law of a score's SD beyond the recorded trials. ``"sqrt_n"``: the ``1 / sqrt(n)``
        law (spec §5.11, method 5). ``"power"``: ``sd(n) = a * n**b`` fitted on the
        resampled counts, ``b`` in [-1/2, 0]. ``"auto"`` (default): ``"power"`` for
        ``"peak_latency"`` and ``"fractional_area_latency"``, ``"sqrt_n"`` for the other
        scores and for callables. These two latencies have an SD that falls more slowly
        than ``1 / sqrt(n)`` while the average is noisy: from 20 trials to 80, on
        development simulations (per-trial SNR 0.5 to 2), the ``1 / sqrt(n)`` law
        predicted a median 0.55 to 0.89 of the true SD, the power law 0.83 to 1.18
        (CHANGELOG).

    Returns
    -------
    DataFrame
        One row per channel and count: ``channel``, ``scorer``, ``n_trials``, ``sd`` (NaN
        for the existence test), ``p_significant`` (NaN for a score), ``extrapolated``
        (True beyond the recorded trials), ``n_resamples``.

        The extrapolation (``N`` trials): ``sqrt_n``, ``sd(n) = c / sqrt(n)`` with ``c``
        the mean of ``sd(m) * sqrt(m)`` over the resampled counts ``m >= N / 2``;
        ``power``, least squares of ``log sd(m)`` on ``log m`` over the resampled counts
        ``m >= N / 4``, exponent clipped to [-1/2, 0]; both times ``sqrt(N / (N - 1))``
        (a bootstrap of the N trials uses their variance with divisor N).
        ``attrs["settings"]["extrapolation"]`` is the law used and ``attrs["exponents"]``
        the exponent per channel.

    Examples
    --------
    >>> from evoparam.sim import simulate_erp
    >>> ts, _ = simulate_erp(20, components=["N1"], noise="white", snr=2., seed=0)
    >>> table = trials_needed(ts, scorer="mean_amplitude", window=(0.08, 0.12),
    ...                       steps=[5, 10, 20, 80], n_resamples=100, random_state=0)
    >>> table[["n_trials", "extrapolated"]].values.tolist()
    [[5, False], [10, False], [20, False], [80, True]]
    """
    start = _time.perf_counter()
    pd = require("pandas", extra="data", caller="evoparam.stats.trials_needed")
    _check_trialset(ts, "trials_needed")
    rng = np.random.default_rng(random_state)
    names = list(ts.ch_names)
    index = (
        list(range(len(names)))
        if picks is None
        else [names.index(p) if isinstance(p, str) else int(p) for p in picks]
    )
    data, keep = _trial_data(ts, index)
    rows_kept = np.flatnonzero(keep)
    n_total = rows_kept.size
    if n_total < 2:
        raise DataError("trials_needed needs at least 2 trials.")
    if steps is None:
        steps = list(range(5, n_total + 1, 5))
        if not steps or steps[-1] != n_total:
            steps.append(n_total)
    steps = sorted({int(s) for s in steps})
    if steps[0] < 1:
        raise SettingsError("steps must be positive trial counts.")
    n_resamples = int(n_resamples)
    if extrapolation not in EXTRAPOLATIONS:
        raise SettingsError(
            f"extrapolation must be one of {EXTRAPOLATIONS}, not {extrapolation!r}."
        )
    records = []
    law, exponents = None, None
    if scorer == "existence":
        kwargs = {"n_permutations": 200, "correction": "none", "n_boot": 0, "by": ()}
        if window is not None:
            kwargs["window"] = window
        kwargs.update(existence_kwargs or {})
        picked = [names[i] for i in index]
        for n in steps:
            if n > n_total:
                _warn(
                    f"trials_needed: {n} trials exceed the {n_total} recorded; the "
                    "detection probability is not extrapolated.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            hits = np.zeros(len(index))
            draws = 1 if n == n_total else n_resamples
            for _ in range(draws):
                rows = np.sort(rng.choice(rows_kept, n, replace=False))
                sub = ts.select(trials=rows.tolist(), picks=picked)
                test = test_response(sub, random_state=int(rng.integers(2**31)), **kwargs)
                hits += np.asarray(test.significant, bool).reshape(len(index), -1).any(axis=1)
            for c, i in enumerate(index):
                records.append(
                    {
                        "channel": names[i],
                        "scorer": "existence",
                        "n_trials": n,
                        "sd": np.nan,
                        "p_significant": hits[c] / draws,
                        "extrapolated": False,
                        "n_resamples": draws,
                    }
                )
        name, unit = "existence", ""
    else:
        func, name, linear, window, inside = _scorer_on_window(scorer, ts.times, window, scorer_kw)
        x = data[rows_kept][..., inside]
        flat = x.reshape(n_total, -1)
        fin = np.isfinite(flat)
        xv, xc = np.where(fin, flat, 0.0), fin.astype(float)
        sd = {}
        for n in [s for s in steps if s <= n_total]:
            weights = np.zeros((n_resamples, n_total))
            pick = rng.integers(0, n_total, (n_resamples, n))
            np.add.at(weights, (np.repeat(np.arange(n_resamples), n), pick.ravel()), 1.0)
            num, den = weights @ xv, weights @ xc
            avg = np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)
            scores = func(avg.reshape((n_resamples,) + x.shape[1:]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sd[n] = np.nanstd(scores, axis=0, ddof=1)
        law = extrapolation
        if law == "auto":
            law = "power" if isinstance(scorer, str) and name in _POWER_SCORES else "sqrt_n"
        beyond, exponents = _extrapolate(sd, n_total, [s for s in steps if s > n_total], law)
        for n in steps:
            values = sd[n] if n in sd else beyond[n]
            for k, i in enumerate(index):
                records.append(
                    {
                        "channel": names[i],
                        "scorer": name,
                        "n_trials": n,
                        "sd": float(values[k]),
                        "p_significant": np.nan,
                        "extrapolated": n > n_total,
                        "n_resamples": n_resamples if n <= n_total else 0,
                    }
                )
        unit = "s" if name in TIME_SCORES else ("data·s" if name == "area" else ts.units)
    table = pd.DataFrame(records, columns=list(TRIALS_NEEDED_COLUMNS))
    table = table.sort_values(
        ["channel", "n_trials"],
        kind="stable",
        key=lambda col: (
            col.map({n: i for i, n in enumerate(names)}) if col.name == "channel" else col
        ),
    ).reset_index(drop=True)
    table.attrs["units"] = {k: v.replace("score", unit) for k, v in TRIALS_NEEDED_COLUMNS.items()}
    settings = {
        "scorer": name,
        "steps": steps,
        "n_resamples": n_resamples,
        "n_recorded": int(n_total),
        "window": None if window is None else list(window),
        "extrapolation": law,
        "existence_kwargs": {
            k: v
            for k, v in (existence_kwargs or {}).items()
            if isinstance(v, (int, float, str, bool))
        },
        "random_state": random_state
        if isinstance(random_state, (int, type(None)))
        else "generator",
    }
    table.attrs["settings"] = settings
    if exponents is not None:
        table.attrs["exponents"] = {names[i]: float(exponents[k]) for k, i in enumerate(index)}
    table.attrs["provenance"] = ts.provenance.with_step(
        "evoparam.stats.trials_needed", settings, duration_s=_time.perf_counter() - start
    ).to_dict()
    return table
