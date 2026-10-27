# SPDX-License-Identifier: Apache-2.0
"""Trial -> channel -> subject model with uncertainty at every level (spec §5.8).

``fit_hierarchical`` takes one TrialSet per subject (or one TrialSet with a ``subject``
metadata column). For every subject it fits the channels (§5.5) and the trials of the
channels that carry each component (§5.7), then combines them:

- ``engine="eb"`` (default): empirical Bayes, see :mod:`evoparam.hierarchical.eb`;
  deterministic, no dependency beyond numpy and scipy.
- ``engine="numpyro"`` or ``"pymc"``: the full model sampled with NUTS (extras), see
  :mod:`evoparam.hierarchical.bayes`.
"""

import contextlib
import functools
import logging
import time as _time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from evoparam._typing import Any
from evoparam.data import HierarchicalFit
from evoparam.errors import DataError, SettingsError
from evoparam.errors import warn as _warn
from evoparam.utils._optional import require
from evoparam.version import __version__

from . import eb
from ._inputs import (
    MIN_TEMPLATE_CHANNELS,
    channel_fits,
    comp_window,
    component_rows,
    run_subjects,
    split_subjects,
    subject_covariates,
    template,
    template_amplitudes,
    trial_fits,
    wald_se,
)

logger = logging.getLogger(__name__)

# spec §4.2: hierarchical imports data, models, trials and stats only (utils.require aside),
# so the verbose= of fit_hierarchical sets the level of the evoparam logger here, with the
# values of evoparam.utils.verbose (REQ-304); "progress" acts as "info" (no bar)
_LEVELS = {
    None: None,
    False: None,
    True: logging.INFO,
    "info": logging.INFO,
    "progress": logging.INFO,
    "debug": logging.DEBUG,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


@contextlib.contextmanager
def _verbose_level(value: Any) -> Iterator[None]:
    """Set the level of the ``evoparam`` logger for a block (None, False: unchanged)."""
    level: int | None
    key = value.lower() if isinstance(value, str) else value
    if isinstance(key, int) and not isinstance(key, bool):
        level = key
    elif key in _LEVELS:
        level = _LEVELS[key]
    else:
        raise SettingsError(
            "verbose must be one of None, False, True, 'info', 'debug', 'warning', 'error', "
            f"'progress' or a logging level; got {value!r}."
        )
    root = logging.getLogger("evoparam")
    saved = root.level
    if level is not None:
        root.setLevel(level)
    try:
        yield
    finally:
        root.setLevel(saved)


def _verbose(function: Callable[..., Any]) -> Callable[..., Any]:
    """Run ``function`` under the logger level of its ``verbose`` keyword."""

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Call ``function`` with the logger level of ``verbose``."""
        with _verbose_level(kwargs.get("verbose")):
            return function(*args, **kwargs)

    return wrapper


if TYPE_CHECKING:
    from evoparam.data.results import GroupFit

__all__ = ["fit_hierarchical", "posterior_summary", "compare_hierarchical", "ENGINES", "LEVELS"]

ENGINES = ("eb", "numpyro", "pymc")
LEVELS = ("trial", "channel", "subject")
MIN_TRUSTED = 3

# Fewest channels with a value from which a parameter is shrunk.
MIN_SHRINK_CHANNELS = 2
# Channel amplitude from the trusted trial fits (jitter flattens the average: SD 11 % low);
# channel latency from the fit of the average (unbiased under symmetric jitter).
TRIAL_PARAMETERS = ("amplitude",)
# Latency SE = Wald error of the average plus the mean trial jitter (jitter SD beyond noise
# / sqrt(n_trials)), truncated to the label window (REQ-171, CHANGELOG). The trial spread
# is kept only for a channel without a Wald error.
TRIAL_SE_PARAMETERS = ("latency",)
# Largest trial shift, in FWHM of the component (see the trial fits below).
JITTER_RANGE = 0.75
MODALITY_PRESETS = {
    "spes": "ccep",
    "erp": "erp",
    "tep": "tep",
    "sep": "sep",
    "vep": "vep",
    "erna": "erna",
    "ecap": "ecap",
    "mep": "mep",
    "baep": "baep",
    "lfp_ep": "lfp_ep",
}

CHANNEL_COLUMNS = {
    "subject": "",
    "channel": "",
    "component": "",
    "has_component": "",
    "n_trials": "",
    "n_trusted": "",
    "p_response": "",
    "existence_amplitude": "data",
    "existence_amplitude_se": "data",
    "flag": "",
}


def _preset_for(ts: Any, preset: Any) -> Any:
    """Return ``preset``, else the preset of the modality of ``ts``."""
    if preset is not None:
        return preset
    if ts.modality in MODALITY_PRESETS:
        return MODALITY_PRESETS[ts.modality]
    raise SettingsError("Give preset= (the TrialSet has no modality).")


@_verbose
def fit_hierarchical(  # noqa: PLR0913  (spec §5.8 fixes 14 parameters, plus preset, group_fits and subject_column; DECISIONS.md §7.2)
    ts: Any,
    *,
    levels: Any = LEVELS,
    components: Sequence[str] | None = None,
    engine: str = "eb",
    existence: bool = True,
    formula: str | None = None,
    latency_dist: str = "exgaussian",
    n_chains: int = 4,
    n_warmup: int = 1000,
    n_draws: int = 1000,
    target_accept: float = 0.9,
    preset: str | None = None,
    group_fits: Sequence["GroupFit"] | None = None,
    subject_column: str = "subject",
    random_state: Any = None,
    n_jobs: int = 1,
    verbose: Any = None,  # noqa: ARG001  (read by the @_verbose decorator)
) -> "HierarchicalFit":
    """Hierarchical estimates of evoked components: trial -> channel -> subject.

    Parameters
    ----------
    ts : TrialSet or sequence of TrialSet
        One TrialSet per subject, or one with a ``subject_column`` metadata column (one
        subject when the column is absent: the subject level is then dropped, with a
        note in ``diagnostics``).
    components : sequence of str, optional
        Component labels (default: every label found in the channel fits).
    engine : {"eb", "numpyro", "pymc"}
    existence : bool
        Compute ``P(response)`` per channel and component.
    formula : str, optional
        ``"parameter ~ term + term"`` with subject-level metadata columns (constant within
        a subject), for example ``"latency ~ age"``; the effects are in ``effects``.
    latency_dist : {"exgaussian", "normal"}
        Trial latency distribution (the EB engine reports the ex-Gaussian ``tau`` from the
        third moment; the Bayesian engines use it in the model).
    n_chains, n_warmup, n_draws, target_accept
        NUTS settings of the Bayesian engines.
    preset : str, optional
        Fitting preset of the channel fits (default from the modality).
    n_jobs : int
        Subjects fitted in parallel processes (``-1``: one per core).
    verbose : bool, str or None
        Level of the ``evoparam`` logger for the call (spec §9.8): one INFO record at the
        end, the channel fits' records at DEBUG; ``"progress"`` is ``"info"`` here.
    group_fits : sequence of GroupFit, optional
        Channel fits already made, one per subject in the same order.

    Returns
    -------
    HierarchicalFit
        ``level_tables`` (``"trial"``, ``"channel"``, ``"subject"``, ``"group"``),
        ``p_response``, ``variance_components``, ``effects``, ``summary``, ``diagnostics``.
    """
    start = _time.perf_counter()
    _check_arguments(engine, latency_dist, levels)
    parameter, terms = eb.parse_formula(formula) if formula else (None, [])

    subjects = split_subjects(ts, subject_column)
    notes = []
    if len(subjects) == 1:
        notes.append("one subject: the subject level is dropped.")
    if group_fits is not None and len(group_fits) != len(subjects):
        raise SettingsError(f"group_fits has {len(group_fits)} fits for {len(subjects)} subjects.")
    covariates = {name: subject_covariates(name, s, terms) for name, s in subjects}

    clock = {"channel_fits": 0.0, "trial_fits": 0.0, "model": 0.0}
    t0 = _time.perf_counter()
    if group_fits is not None:
        fits = list(group_fits)
    else:
        fits = run_subjects(
            channel_fits, [(s, _preset_for(s, preset)) for _, s in subjects], n_jobs
        )
    clock["channel_fits"] = _time.perf_counter() - t0
    labels = _component_labels(fits, components)

    t0 = _time.perf_counter()
    require("pandas", extra="data", caller="evoparam.hierarchical.api.fit_hierarchical")
    import pandas as pd

    run = _Run(
        subjects=subjects,
        fits=fits,
        covariates=covariates,
        formula=(parameter, terms),
        options={
            "existence": existence,
            "latency_dist": latency_dist,
            "random_state": random_state,
            "n_jobs": n_jobs,
        },
        clock=clock,
        diagnostics={"notes": notes, "em": {}, "reml": {}, "jitter": {}},
    )
    for label in labels:
        _fit_component(run, label, pd)

    level_tables, variance, effects, p_response = _level_tables(run, pd)
    group_table = level_tables["group"]
    clock["model"] = _time.perf_counter() - t0 - clock["trial_fits"]
    diagnostics = run.diagnostics
    diagnostics["timing_s"] = clock
    settings = {
        "levels": list(levels),
        "components": list(labels),
        "engine": engine,
        "existence": existence,
        "formula": formula,
        "latency_dist": latency_dist,
        "preset": preset,
        "subjects": [n for n, _ in subjects],
        "subject_column": subject_column,
        "random_state": random_state,
        "package_version": __version__,
    }
    if engine != "eb":
        from .bayes import NutsSettings, sample

        result = sample(
            engine,
            level_tables,
            settings,
            nuts=NutsSettings(n_chains, n_warmup, n_draws, target_accept, random_state),
            formula=(parameter, terms),
            covariates=covariates,
            eb_variance=variance,
        )
        level_tables.update(result["level_tables"])
        diagnostics.update(result["diagnostics"])
        variance = result.get("variance_components", variance)
        effects = result.get("effects", effects)
        p_response = result.get("p_response", p_response)
        settings.update(
            n_chains=n_chains, n_warmup=n_warmup, n_draws=n_draws, target_accept=target_accept
        )
    logger.info(
        "fit_hierarchical (%s): %d subject(s), component(s) %s, %.2f s",
        engine,
        len(subjects),
        ", ".join(map(str, labels)),
        _time.perf_counter() - start,
    )
    provenance = subjects[0][1].provenance.with_step(
        "evoparam.hierarchical.fit_hierarchical",
        dict(settings.items()),
        duration_s=_time.perf_counter() - start,
    )
    return HierarchicalFit(
        levels=tuple(levels),
        summary=group_table,
        level_tables=level_tables,
        p_response=p_response,
        variance_components=variance,
        effects=effects,
        engine=engine,
        diagnostics=diagnostics,
        settings=settings,
        provenance=provenance,
    )


def _check_arguments(engine: str, latency_dist: str, levels: Any) -> None:
    """Raise SettingsError for an unknown engine, latency distribution or level."""
    if engine not in ENGINES:
        raise SettingsError(f"engine must be one of {ENGINES}, not {engine!r}.")
    if latency_dist not in ("exgaussian", "normal"):
        raise SettingsError("latency_dist must be 'exgaussian' or 'normal'.")
    if engine != "eb":
        from .bayes import require_engine

        require_engine(engine)
    unknown = set(levels) - set(LEVELS)
    if unknown:
        raise SettingsError(f"Unknown levels {sorted(unknown)}; use {LEVELS}.")


def _component_labels(fits: Any, components: Any) -> Any:
    """Return ``components``, else every component label of the channel fits (in order)."""
    labels = components
    if labels is None:
        found = [c.label for gf in fits for f in gf.fits for c in f.components if c.label]
        labels = sorted(set(found), key=found.index)
    if not labels:
        raise SettingsError("No labelled component in the channel fits; give components=.")
    return labels


@dataclass
class _Run:
    """What the steps of :func:`fit_hierarchical` share: its inputs and the rows built.

    ``formula`` is the parsed formula (the parameter and the terms); ``options`` holds
    ``existence``, ``latency_dist``, ``random_state`` and ``n_jobs``.
    """

    subjects: Any
    fits: Any
    covariates: Any
    formula: Any
    options: Any
    clock: Any
    diagnostics: Any
    channel_rows: Any = field(default_factory=list)
    trial_tables: Any = field(default_factory=list)
    subject_rows: Any = field(default_factory=list)
    group_rows: Any = field(default_factory=list)
    var_rows: Any = field(default_factory=list)
    effect_tables: Any = field(default_factory=list)

    @property
    def notes(self) -> Any:
        """The notes of ``diagnostics``."""
        return self.diagnostics["notes"]

    def with_terms(self, p: str) -> bool:
        """Whether the formula models parameter ``p`` with covariates."""
        parameter, terms = self.formula
        return bool(parameter == p and terms)


def _fit_component(run: _Run, label: str, pd: Any) -> None:
    """Fit the trials of component ``label`` and combine the levels; add the rows to ``run``."""
    comps = [component_rows(gf, label) for gf in run.fits]
    pooled = [c for rows in comps for c in rows.values() if c is not None]
    if not pooled:
        run.notes.append(f"component {label!r}: found on no channel; left out.")
        return
    group_tmpl = template(pooled)
    # Trial shifts up to 0.75 FWHM: with the default quarter FWHM of §5.7, trials of a
    # 4 ms jitter reach the bound, are not trusted, and the jitter SD comes out halved
    t1 = _time.perf_counter()
    trial_sets = run_subjects(
        trial_fits,
        [
            (s, gf, label, JITTER_RANGE * group_tmpl["fwhm"], run.options["random_state"])
            for (_, s), gf in zip(run.subjects, run.fits, strict=False)
        ],
        run.options["n_jobs"],
    )
    run.clock["trial_fits"] += _time.perf_counter() - t1
    rows = []
    for si, (gf, crow, tset) in enumerate(zip(run.fits, comps, trial_sets, strict=False)):
        rows += _subject_channel_rows(run, label, si, (gf, crow, tset), group_tmpl)
    frame = pd.DataFrame(rows)
    jit = _trial_jitter(run, label, frame, pd)
    jitter_sd = jit["sd"] if jit is not None and np.isfinite(jit["sd"]) else 0.0
    _latency_standard_errors(frame, jitter_sd, pd)
    pi, mu = _existence_mixture(run, label, frame, group_tmpl)

    frame["flag"] = ""
    usable = frame["has_component"] & (frame["n_trusted"] >= MIN_TRUSTED)
    frame.loc[frame["has_component"] & ~usable, "flag"] = "prior_dominated"
    subject_est: dict[Any, Any] = {}
    for p in eb.PARAMETERS:
        _shrink_parameter(run, (label, p), frame, usable, subject_est)
    _subject_table_rows(run, label, frame, (pi, mu), subject_est)
    if jit is not None:
        _jitter_rows(run, label, jit)
    run.channel_rows.append(
        frame.drop(columns=["subject_index"] + [c for c in frame.columns if c.startswith("_")])
    )


def _subject_channel_rows(
    run: _Run, label: str, si: int, fitted: Any, group_tmpl: Any
) -> list[dict[str, Any]]:
    """Return the rows of the channels of subject ``si`` for component ``label``.

    ``fitted`` holds the subject's channel fits, its component rows and its trial fits.
    The trusted trials go to ``run.trial_tables``.
    """
    gf, crow, tset = fitted
    name, s = run.subjects[si]
    own = [c for c in crow.values() if c is not None]
    tmpl = template(own) if len(own) >= MIN_TEMPLATE_CHANNELS else group_tmpl
    amps = template_amplitudes(s, tmpl) if run.options["existence"] else None
    table = tset.table
    sub = table[table["label"] == label] if len(table) else table
    rows = []
    for ci, ch in enumerate(s.ch_names):
        comp = crow.get(ch)
        trials_ch = sub[sub["channel"] == ch] if len(sub) else sub
        trusted = trials_ch[trials_ch["trusted"]] if len(trials_ch) else trials_ch
        noise = _latency_noise_sd(tset, ch, trials_ch)
        row: Any
        row = {
            "subject": name,
            "subject_index": si,
            "channel": ch,
            "component": label,
            "has_component": comp is not None,
            "n_trials": s.n_trials,
            "n_trusted": int(len(trusted)),
            "latency_noise_sd": noise,
        }
        if amps is not None:
            a = amps[:, ci]
            a = a[np.isfinite(a)]
            row["existence_amplitude"] = float(a.mean()) if a.size else np.nan
            row["existence_amplitude_se"] = (
                float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else np.nan
            )
        _add_parameters(row, comp, trusted)
        window = comp_window(gf.get(ch), label) if comp is not None else (np.nan, np.nan)
        row["_window_low"], row["_window_high"] = window
        rows.append(row)
    if len(sub):
        part = sub[sub["trusted"]].copy()
        if len(part):
            part = part.drop(columns=["subject"], errors="ignore")
            part.insert(0, "subject", name)
            run.trial_tables.append(part)
    return rows


def _latency_noise_sd(tset: Any, ch: str, trials_ch: Any) -> float:
    """Return the null jitter SD of channel ``ch`` (NaN without trials of the component)."""
    noise = np.nan
    for tf in tset.fits:
        if tf.channel == ch and len(trials_ch):
            k = int(trials_ch["component"].iloc[0])
            noise = float(tf.jitter_sd_null.get(k, np.nan))
    return noise


def _add_parameters(row: Any, comp: Any, trusted: Any) -> None:
    """Add the fitted and raw values of every parameter, with their standard errors."""
    for p in eb.PARAMETERS:
        row[f"{p}_fit"] = getattr(comp, p) if comp is not None else np.nan
        row[f"{p}_fit_se"] = wald_se(comp, p) if comp is not None else np.nan
        # the sufficient statistics (§5.8 step 2): mean and standard error of
        # the trusted trial estimates (see TRIAL_PARAMETERS)
        values = (
            trusted[p].to_numpy(float)
            if p in TRIAL_PARAMETERS + TRIAL_SE_PARAMETERS and len(trusted)
            else np.array([])
        )
        row[f"{p}_raw"], row[f"{p}_raw_se"] = row[f"{p}_fit"], row[f"{p}_fit_se"]
        if comp is not None and values.size >= MIN_TRUSTED:
            se_trials = float(values.std(ddof=1) / np.sqrt(values.size))
            if p in TRIAL_PARAMETERS:
                row[f"{p}_raw"] = float(values.mean())
                row[f"{p}_raw_se"] = se_trials
            else:
                # completed below, once the jitter SD is known
                row[f"_{p}_se_trials"] = se_trials


def _trial_jitter(run: _Run, label: str, frame: Any, pd: Any) -> Any:
    """Return the trial jitter beyond noise of component ``label`` (None without trials)."""
    trial_part = [t for t in run.trial_tables if len(t) and t["label"].iloc[0] == label]
    if not trial_part:
        return None
    tt = pd.concat(trial_part, ignore_index=True)
    noise = frame.set_index(["subject", "channel"])["latency_noise_sd"]
    sd = noise.reindex(list(zip(tt["subject"], tt["channel"], strict=False))).to_numpy()
    return eb.jitter_components(
        tt["latency"].to_numpy(), sd, tt["subject"].to_numpy(), tt["channel"].to_numpy()
    )


def _latency_standard_errors(frame: Any, jitter_sd: float, pd: Any) -> None:
    """Set the raw standard errors of TRIAL_SE_PARAMETERS, once the jitter SD is known."""
    for p in TRIAL_SE_PARAMETERS:
        frame[f"{p}_raw_se"] = eb.latency_standard_error(
            frame[f"{p}_fit"].to_numpy(float),
            frame[f"{p}_fit_se"].to_numpy(float),
            jitter_sd,
            frame["n_trials"].to_numpy(float),
            frame["_window_low"].to_numpy(float),
            frame["_window_high"].to_numpy(float),
            fallback=frame.get(f"_{p}_se_trials", pd.Series(np.nan, index=frame.index)).to_numpy(
                float
            ),
        )


def _existence_mixture(run: _Run, label: str, frame: Any, group_tmpl: Any) -> tuple[Any, Any]:
    """Set ``p_response`` from the two-component mixture; return its pi and mu per subject.

    Without the existence step, ``p_response`` is NaN and both are empty.
    """
    if not run.options["existence"]:
        frame["p_response"] = np.nan
        return {}, {}
    em = eb.mixture_em(
        frame["existence_amplitude"].to_numpy(),
        frame["existence_amplitude_se"].to_numpy(),
        frame["subject_index"].to_numpy(),
        group_tmpl["polarity"],
    )
    frame["p_response"] = em["p_response"]
    run.diagnostics["em"][label] = {
        k: em[k] for k in ("n_iter", "converged", "loglik", "tau", "mu_group", "sd_group")
    }
    pi = dict(zip(range(len(run.subjects)), em["pi"], strict=False))
    mu = dict(zip(range(len(run.subjects)), em["mu"], strict=False))
    run.var_rows.append(
        {
            "component": label,
            "parameter": "amplitude",
            "level": "channel (existence mixture)",
            "sd": em["tau"],
            "method": "EM, two-component mixture",
        }
    )
    return pi, mu


def _shrink_parameter(
    run: _Run, key: tuple[str, str], frame: Any, usable: Any, subject_est: Any
) -> None:
    """Shrink parameter ``p`` of component ``label`` (``key``) by REML; fill ``frame``.

    A numerical failure is noted and warned, and the raw values are kept.
    """
    label, p = key
    use = (
        usable
        & np.isfinite(frame[f"{p}_raw"])
        & np.isfinite(frame[f"{p}_raw_se"])
        & (frame[f"{p}_raw_se"] > 0)
    )
    frame[p] = np.nan
    frame[f"{p}_se"] = np.nan
    frame[f"{p}_shrinkage"] = np.nan
    if use.sum() < MIN_SHRINK_CHANNELS:
        run.notes.append(f"component {label!r}, {p}: fewer than 2 channels; not shrunk.")
        return
    part = frame[use]
    design, names = None, ["Intercept"]
    if run.with_terms(p):
        design, names = eb.design_matrix(
            [run.covariates[n] for n in part["subject"]], run.formula[1]
        )
    try:
        out, lev = eb.shrink(
            part[f"{p}_raw"].to_numpy(),
            part[f"{p}_raw_se"].to_numpy(),
            part["subject"].to_numpy(),
            part["channel"].to_numpy(),
            X=design,
        )
    except (DataError, ArithmeticError, ValueError, np.linalg.LinAlgError) as err:
        # a numerical failure of this parameter (singular design, too few channels
        # for the covariates): recorded and warned (REQ-307), the raw values kept
        message = f"component {label!r}, {p}: REML failed ({type(err).__name__}: {err})."
        run.notes.append(message)
        # stacklevel 4: the caller of fit_hierarchical, as when this was its own code
        _warn(f"fit_hierarchical: {message} The raw values are kept.", UserWarning, stacklevel=4)
        return
    run.diagnostics["reml"][f"{label}/{p}"] = {
        "converged": bool(out["converged"]),
        "n_iter": int(out["n_iter"]),
    }
    index = {k: i for i, k in enumerate(out["labels"][lev])}
    keys = [
        tuple(k)
        for k in (
            part[["subject", "channel"]].to_numpy() if lev == 1 else part[["channel"]].to_numpy()
        )
    ]
    pos = [index[k] for k in keys]
    frame.loc[use, p] = out["unit_mean"][lev][pos]
    frame.loc[use, f"{p}_se"] = out["unit_mean_se"][lev][pos]
    frame.loc[use, f"{p}_shrinkage"] = out["shrinkage"][lev][pos]
    var_channel = float(out["variances"][lev])
    run.var_rows.append(
        {
            "component": label,
            "parameter": p,
            "level": "channel",
            "sd": float(np.sqrt(var_channel)),
            "method": "REML",
        }
    )
    beta, beta_cov = np.asarray(out["beta"]), np.asarray(out["beta_cov"])
    if lev == 1:
        _subject_level(run, key, out, frame, subject_est)
    run.group_rows.append(
        {
            "component": label,
            "parameter": p,
            "estimate": float(beta[0]),
            "se": float(np.sqrt(beta_cov[0, 0])),
            "n_channels": int(use.sum()),
            "n_subjects": int(part["subject"].nunique()),
        }
    )
    if run.with_terms(p):
        table = eb.effects_table(out, names)
        table.insert(0, "parameter", p)
        table.insert(0, "component", label)
        run.effect_tables.append(table)


def _subject_level(
    run: _Run, key: tuple[str, str], out: Any, frame: Any, subject_est: Any
) -> None:
    """Record the subject level of a three-level REML fit of parameter ``p`` (``key``).

    Every subject gets an estimate (the group prediction when it has no usable channel),
    and the prior-dominated channels take their subject's value.
    """
    label, p = key
    var_channel = float(out["variances"][1])
    beta, beta_cov = np.asarray(out["beta"]), np.asarray(out["beta_cov"])
    run.var_rows.append(
        {
            "component": label,
            "parameter": p,
            "level": "subject",
            "sd": float(np.sqrt(out["variances"][0])),
            "method": "REML",
        }
    )
    for (sname,), m, se in zip(
        out["labels"][0], out["unit_mean"][0], out["unit_mean_se"][0], strict=False
    ):
        subject_est[(sname, p)] = (float(m), float(se))
    # a subject without usable channels: its prediction is the group value
    for sname, _ in run.subjects:
        if (sname, p) not in subject_est:
            row = (
                eb.design_matrix([run.covariates[sname]], run.formula[1])[0][0]
                if run.with_terms(p)
                else np.ones(1)
            )
            subject_est[(sname, p)] = (
                float(row @ beta),
                float(np.sqrt(row @ beta_cov @ row + out["variances"][0])),
            )
    # prior-only estimate of the flagged channels: their subject's value
    flagged = frame["flag"].eq("prior_dominated")
    for idx in frame.index[flagged]:
        est = subject_est.get((frame.at[idx, "subject"], p))
        if est is not None:
            frame.at[idx, p] = est[0]
            frame.at[idx, f"{p}_se"] = float(np.sqrt(est[1] ** 2 + var_channel))
            frame.at[idx, f"{p}_shrinkage"] = 0.0


def _subject_table_rows(
    run: _Run, label: str, frame: Any, mixture: tuple[Any, Any], subject_est: Any
) -> None:
    """Add the subject rows of component ``label`` (``mixture``: pi and mu per subject)."""
    pi, mu = mixture
    for si, (name, _s) in enumerate(run.subjects):
        row = {
            "subject": name,
            "component": label,
            "n_channels_with_component": int(
                frame[(frame["subject"] == name) & frame["has_component"]].shape[0]
            ),
            "pi": float(pi.get(si, np.nan)),
            "existence_mean_amplitude": float(mu.get(si, np.nan)),
        }
        for p in eb.PARAMETERS:
            est = subject_est.get((name, p), (np.nan, np.nan))
            row[p], row[f"{p}_se"] = est
        row.update(run.covariates[name])
        run.subject_rows.append(row)


def _jitter_rows(run: _Run, label: str, jit: Any) -> None:
    """Record the trial jitter of component ``label`` (and its ex-Gaussian parts)."""
    run.diagnostics["jitter"][label] = jit
    run.var_rows.append(
        {
            "component": label,
            "parameter": "latency",
            "level": "trial",
            "sd": jit["sd"],
            "method": "REML (residual beyond noise)",
        }
    )
    if run.options["latency_dist"] == "exgaussian":
        run.var_rows.append(
            {
                "component": label,
                "parameter": "latency",
                "level": "trial (ex-Gaussian sigma)",
                "sd": jit["sigma"],
                "method": "moments",
            }
        )
        run.var_rows.append(
            {
                "component": label,
                "parameter": "latency",
                "level": "trial (ex-Gaussian tau)",
                "sd": jit["tau"],
                "method": "moments",
            }
        )


def _level_tables(run: _Run, pd: Any) -> tuple[Any, Any, Any, Any]:
    """Return the level tables, the variance components, the effects and ``p_response``."""
    channel_table = (
        pd.concat(run.channel_rows, ignore_index=True)
        if run.channel_rows
        else pd.DataFrame(columns=list(CHANNEL_COLUMNS))
    )
    trial_table = (
        pd.concat(run.trial_tables, ignore_index=True) if run.trial_tables else pd.DataFrame()
    )
    subject_table = pd.DataFrame(run.subject_rows)
    group_table = pd.DataFrame(run.group_rows)
    variance = pd.DataFrame(
        run.var_rows, columns=["component", "parameter", "level", "sd", "method"]
    )
    effects = pd.concat(run.effect_tables, ignore_index=True) if run.effect_tables else None
    p_response = (
        channel_table[["subject", "channel", "component", "p_response"]].copy()
        if len(channel_table)
        else pd.DataFrame(columns=["subject", "channel", "component", "p_response"])
    )
    level_tables = {
        "trial": trial_table,
        "channel": channel_table,
        "subject": subject_table,
        "group": group_table,
    }
    return level_tables, variance, effects, p_response


def posterior_summary(fit: Any, level: str = "channel") -> Any:
    """Return the table of one level (``trial``, ``channel``, ``subject`` or ``group``)."""
    if level not in fit.level_tables:
        raise SettingsError(f"level must be one of {sorted(fit.level_tables)}, not {level!r}.")
    return fit.level_tables[level]


def compare_hierarchical(fits: Any) -> Any:
    """Leave-one-out comparison (PSIS-LOO) of Bayesian fits: one row per fit, best first."""
    from .bayes import loo_table

    if any(f.engine == "eb" for f in fits):
        raise SettingsError(
            "compare_hierarchical needs fits of a Bayesian engine (numpyro or "
            "pymc): the leave-one-out comparison uses posterior draws."
        )
    return loo_table(fits)
