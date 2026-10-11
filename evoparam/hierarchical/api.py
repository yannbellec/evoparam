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

import time as _time

import numpy as np

from evoparam.data import HierarchicalFit
from evoparam.errors import SettingsError
from evoparam.version import __version__

from . import eb
from ._inputs import (MIN_TEMPLATE_CHANNELS, channel_fits, component_rows, run_subjects,
                      split_subjects, subject_covariates, template, template_amplitudes,
                      trial_fits, wald_se)

__all__ = ["fit_hierarchical", "posterior_summary", "compare_hierarchical", "ENGINES",
           "LEVELS"]

ENGINES = ("eb", "numpyro", "pymc")
LEVELS = ("trial", "channel", "subject")
MIN_TRUSTED = 3
# Channel statistics taken from the trusted trial estimates (shift-and-scale trial fits)
# rather than from the fit of the average: the amplitude, because latency jitter flattens
# the average (the channel SD of the amplitude came out 11 % low). The latency stays the
# fit's: the average is not biased by a symmetric jitter, and its estimate beat the mean
# of the trial latencies (latency RMSE lower in 2 of 3 simulated datasets).
TRIAL_PARAMETERS = ("amplitude",)
# Channel statistics whose standard error comes from the spread of the trusted trials: the
# Wald error of the fitted latency is unstable (its z-scores had an SD of 1.2 to 1.5 on
# simulations, with some errors ten times too large, which shrank good channels away).
TRIAL_SE_PARAMETERS = ("latency",)
# Largest trial shift, in FWHM of the component (see the trial fits below).
JITTER_RANGE = 0.75
MODALITY_PRESETS = {"spes": "ccep", "erp": "erp", "tep": "tep", "sep": "sep", "vep": "vep",
                    "erna": "erna", "ecap": "ecap", "mep": "mep", "baep": "baep",
                    "lfp_ep": "lfp_ep"}

CHANNEL_COLUMNS = {
    "subject": "", "channel": "", "component": "", "has_component": "", "n_trials": "",
    "n_trusted": "", "p_response": "", "existence_amplitude": "data",
    "existence_amplitude_se": "data", "flag": "",
}


def _preset_for(ts, preset):
    if preset is not None:
        return preset
    if ts.modality in MODALITY_PRESETS:
        return MODALITY_PRESETS[ts.modality]
    raise SettingsError("Give preset= (the TrialSet has no modality).")


def fit_hierarchical(ts, *, levels=LEVELS, components=None, engine="eb", existence=True,
                     formula=None, latency_dist="exgaussian", n_chains=4, n_warmup=1000,
                     n_draws=1000, target_accept=0.9, preset=None, group_fits=None,
                     subject_column="subject", random_state=None, n_jobs=1, verbose=None):
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
    group_fits : sequence of GroupFit, optional
        Channel fits already made, one per subject in the same order.

    Returns
    -------
    HierarchicalFit
        ``level_tables`` (``"trial"``, ``"channel"``, ``"subject"``, ``"group"``),
        ``p_response``, ``variance_components``, ``effects``, ``summary``, ``diagnostics``.
    """

    start = _time.perf_counter()
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
    parameter, terms = eb.parse_formula(formula) if formula else (None, [])

    subjects = split_subjects(ts, subject_column)
    notes = []
    if len(subjects) == 1:
        notes.append("one subject: the subject level is dropped.")
    if group_fits is not None and len(group_fits) != len(subjects):
        raise SettingsError(f"group_fits has {len(group_fits)} fits for {len(subjects)} "
                            f"subjects.")
    covariates = {name: subject_covariates(name, s, terms) for name, s in subjects}

    clock = {"channel_fits": 0.0, "trial_fits": 0.0, "model": 0.0}
    t0 = _time.perf_counter()
    if group_fits is not None:
        fits = list(group_fits)
    else:
        fits = run_subjects(channel_fits, [(s, _preset_for(s, preset)) for _, s in subjects],
                            n_jobs)
    clock["channel_fits"] = _time.perf_counter() - t0
    labels = components
    if labels is None:
        found = [c.label for gf in fits for f in gf.fits for c in f.components if c.label]
        labels = sorted(set(found), key=found.index)
    if not labels:
        raise SettingsError("No labelled component in the channel fits; give components=.")

    t0 = _time.perf_counter()
    import pandas as pd

    channel_rows, trial_tables, subject_rows, group_rows, var_rows, effect_tables = \
        [], [], [], [], [], []
    diagnostics = {"notes": notes, "em": {}, "reml": {}, "jitter": {}}
    for label in labels:
        comps = [component_rows(gf, label) for gf in fits]
        pooled = [c for rows in comps for c in rows.values() if c is not None]
        if not pooled:
            notes.append(f"component {label!r}: found on no channel; left out.")
            continue
        group_tmpl = template(pooled)
        # Trial shifts up to 0.75 FWHM: with the default quarter FWHM of §5.7, trials of a
        # 4 ms jitter reach the bound, are not trusted, and the jitter SD comes out halved
        t1 = _time.perf_counter()
        trial_sets = run_subjects(trial_fits, [(s, gf, label, JITTER_RANGE * group_tmpl["fwhm"],
                                                random_state)
                                               for (_, s), gf in zip(subjects, fits)], n_jobs)
        clock["trial_fits"] += _time.perf_counter() - t1
        rows = []
        for si, ((name, s), gf, crow, tset) in enumerate(zip(subjects, fits, comps,
                                                            trial_sets)):
            own = [c for c in crow.values() if c is not None]
            tmpl = template(own) if len(own) >= MIN_TEMPLATE_CHANNELS else group_tmpl
            amps = template_amplitudes(s, tmpl) if existence else None
            table = tset.table
            sub = table[table["label"] == label] if len(table) else table
            for ci, ch in enumerate(s.ch_names):
                comp = crow.get(ch)
                trials_ch = sub[sub["channel"] == ch] if len(sub) else sub
                trusted = trials_ch[trials_ch["trusted"]] if len(trials_ch) else trials_ch
                noise = np.nan
                for tf in tset.fits:
                    if tf.channel == ch and len(trials_ch):
                        k = int(trials_ch["component"].iloc[0])
                        noise = float(tf.jitter_sd_null.get(k, np.nan))
                row = {"subject": name, "subject_index": si, "channel": ch, "component": label,
                       "has_component": comp is not None, "n_trials": s.n_trials,
                       "n_trusted": int(len(trusted)), "latency_noise_sd": noise}
                if amps is not None:
                    a = amps[:, ci]
                    a = a[np.isfinite(a)]
                    row["existence_amplitude"] = float(a.mean()) if a.size else np.nan
                    row["existence_amplitude_se"] = float(a.std(ddof=1) / np.sqrt(a.size)) \
                        if a.size > 1 else np.nan
                for p in eb.PARAMETERS:
                    row[f"{p}_fit"] = getattr(comp, p) if comp is not None else np.nan
                    row[f"{p}_fit_se"] = wald_se(comp, p) if comp is not None else np.nan
                    # the sufficient statistics (§5.8 step 2): mean and standard error of
                    # the trusted trial estimates (see TRIAL_PARAMETERS)
                    values = trusted[p].to_numpy(float) if p in TRIAL_PARAMETERS + \
                        TRIAL_SE_PARAMETERS and len(trusted) else np.array([])
                    row[f"{p}_raw"], row[f"{p}_raw_se"] = row[f"{p}_fit"], row[f"{p}_fit_se"]
                    if comp is not None and values.size >= MIN_TRUSTED:
                        se_trials = float(values.std(ddof=1) / np.sqrt(values.size))
                        if p in TRIAL_PARAMETERS:
                            row[f"{p}_raw"] = float(values.mean())
                            row[f"{p}_raw_se"] = se_trials
                        else:
                            # the fitted value keeps both errors: the noise of the average
                            # (Wald) and the mean of the trial jitter (spread of the trusted
                            # trials); either alone gave z-scores with an SD of 1.5 to 1.8 on
                            # the reference simulations, their sum in quadrature 1.0
                            row[f"{p}_raw_se"] = float(np.hypot(row[f"{p}_fit_se"], se_trials)) \
                                if np.isfinite(row[f"{p}_fit_se"]) else se_trials
                rows.append(row)
            if len(sub):
                part = sub[sub["trusted"]].copy()
                if len(part):
                    part = part.drop(columns=["subject"], errors="ignore")
                    part.insert(0, "subject", name)
                    trial_tables.append(part)
        frame = pd.DataFrame(rows)

        if existence:
            em = eb.mixture_em(frame["existence_amplitude"].to_numpy(),
                               frame["existence_amplitude_se"].to_numpy(),
                               frame["subject_index"].to_numpy(), group_tmpl["polarity"])
            frame["p_response"] = em["p_response"]
            diagnostics["em"][label] = {k: em[k] for k in ("n_iter", "converged", "loglik",
                                                           "tau", "mu_group", "sd_group")}
            pi = dict(zip(range(len(subjects)), em["pi"]))
            mu = dict(zip(range(len(subjects)), em["mu"]))
            var_rows.append({"component": label, "parameter": "amplitude",
                             "level": "channel (existence mixture)", "sd": em["tau"],
                             "method": "EM, two-component mixture"})
        else:
            frame["p_response"] = np.nan
            pi, mu = {}, {}

        frame["flag"] = ""
        usable = frame["has_component"] & (frame["n_trusted"] >= MIN_TRUSTED)
        frame.loc[frame["has_component"] & ~usable, "flag"] = "prior_dominated"
        subject_est = {}
        for p in eb.PARAMETERS:
            use = usable & np.isfinite(frame[f"{p}_raw"]) & np.isfinite(frame[f"{p}_raw_se"]) \
                & (frame[f"{p}_raw_se"] > 0)
            frame[p] = np.nan
            frame[f"{p}_se"] = np.nan
            frame[f"{p}_shrinkage"] = np.nan
            if use.sum() < 2:
                notes.append(f"component {label!r}, {p}: fewer than 2 channels; not shrunk.")
                continue
            part = frame[use]
            X, names = None, ["Intercept"]
            if parameter == p and terms:
                X, names = eb.design_matrix([covariates[n] for n in part["subject"]], terms)
            try:
                out, lev = eb.shrink(part[f"{p}_raw"].to_numpy(), part[f"{p}_raw_se"].to_numpy(),
                                     part["subject"].to_numpy(), part["channel"].to_numpy(), X=X)
            except Exception as err:  # numerical failure: report, keep the raw values
                notes.append(f"component {label!r}, {p}: REML failed ({err}).")
                continue
            diagnostics["reml"][f"{label}/{p}"] = {"converged": bool(out["converged"]),
                                                   "n_iter": int(out["n_iter"])}
            index = {key: i for i, key in enumerate(out["labels"][lev])}
            keys = [tuple(k) for k in (part[["subject", "channel"]].to_numpy() if lev == 1
                                       else part[["channel"]].to_numpy())]
            pos = [index[k] for k in keys]
            frame.loc[use, p] = out["unit_mean"][lev][pos]
            frame.loc[use, f"{p}_se"] = out["unit_mean_se"][lev][pos]
            frame.loc[use, f"{p}_shrinkage"] = out["shrinkage"][lev][pos]
            var_channel = float(out["variances"][lev])
            var_rows.append({"component": label, "parameter": p, "level": "channel",
                             "sd": float(np.sqrt(var_channel)), "method": "REML"})
            beta, beta_cov = np.asarray(out["beta"]), np.asarray(out["beta_cov"])
            if lev == 1:
                var_rows.append({"component": label, "parameter": p, "level": "subject",
                                 "sd": float(np.sqrt(out["variances"][0])), "method": "REML"})
                for (sname,), m, se in zip(out["labels"][0], out["unit_mean"][0],
                                           out["unit_mean_se"][0]):
                    subject_est[(sname, p)] = (float(m), float(se))
                # a subject without usable channels: its prediction is the group value
                for sname, _ in subjects:
                    if (sname, p) not in subject_est:
                        row = eb.design_matrix([covariates[sname]], terms)[0][0] \
                            if parameter == p and terms else np.ones(1)
                        subject_est[(sname, p)] = (
                            float(row @ beta),
                            float(np.sqrt(row @ beta_cov @ row + out["variances"][0])))
                # prior-only estimate of the flagged channels: their subject's value
                flagged = frame["flag"].eq("prior_dominated")
                for idx in frame.index[flagged]:
                    est = subject_est.get((frame.at[idx, "subject"], p))
                    if est is not None:
                        frame.at[idx, p] = est[0]
                        frame.at[idx, f"{p}_se"] = float(np.sqrt(est[1] ** 2 + var_channel))
                        frame.at[idx, f"{p}_shrinkage"] = 0.0
            group_rows.append({"component": label, "parameter": p, "estimate": float(beta[0]),
                               "se": float(np.sqrt(beta_cov[0, 0])),
                               "n_channels": int(use.sum()),
                               "n_subjects": int(part["subject"].nunique())})
            if parameter == p and terms:
                table = eb.effects_table(out, names)
                table.insert(0, "parameter", p)
                table.insert(0, "component", label)
                effect_tables.append(table)

        for si, (name, s) in enumerate(subjects):
            row = {"subject": name, "component": label,
                   "n_channels_with_component": int(frame[(frame["subject"] == name)
                                                          & frame["has_component"]].shape[0]),
                   "pi": float(pi.get(si, np.nan)),
                   "existence_mean_amplitude": float(mu.get(si, np.nan))}
            for p in eb.PARAMETERS:
                est = subject_est.get((name, p), (np.nan, np.nan))
                row[p], row[f"{p}_se"] = est
            row.update(covariates[name])
            subject_rows.append(row)

        trial_part = [t for t in trial_tables if len(t) and t["label"].iloc[0] == label]
        if trial_part:
            tt = pd.concat(trial_part, ignore_index=True)
            noise = frame.set_index(["subject", "channel"])["latency_noise_sd"]
            sd = noise.reindex(list(zip(tt["subject"], tt["channel"]))).to_numpy()
            jit = eb.jitter_components(tt["latency"].to_numpy(), sd, tt["subject"].to_numpy(),
                                       tt["channel"].to_numpy())
            diagnostics["jitter"][label] = jit
            var_rows.append({"component": label, "parameter": "latency", "level": "trial",
                             "sd": jit["sd"], "method": "REML (residual beyond noise)"})
            if latency_dist == "exgaussian":
                var_rows.append({"component": label, "parameter": "latency",
                                 "level": "trial (ex-Gaussian sigma)", "sd": jit["sigma"],
                                 "method": "moments"})
                var_rows.append({"component": label, "parameter": "latency",
                                 "level": "trial (ex-Gaussian tau)", "sd": jit["tau"],
                                 "method": "moments"})
        channel_rows.append(frame.drop(columns=["subject_index"]))

    channel_table = pd.concat(channel_rows, ignore_index=True) if channel_rows else \
        pd.DataFrame(columns=list(CHANNEL_COLUMNS))
    trial_table = pd.concat(trial_tables, ignore_index=True) if trial_tables else pd.DataFrame()
    subject_table = pd.DataFrame(subject_rows)
    group_table = pd.DataFrame(group_rows)
    variance = pd.DataFrame(var_rows, columns=["component", "parameter", "level", "sd",
                                               "method"])
    effects = pd.concat(effect_tables, ignore_index=True) if effect_tables else None
    p_response = channel_table[["subject", "channel", "component", "p_response"]].copy() \
        if len(channel_table) else pd.DataFrame(columns=["subject", "channel", "component",
                                                         "p_response"])
    level_tables = {"trial": trial_table, "channel": channel_table, "subject": subject_table,
                    "group": group_table}
    clock["model"] = _time.perf_counter() - t0 - clock["trial_fits"]
    diagnostics["timing_s"] = clock
    settings = {"levels": list(levels), "components": list(labels), "engine": engine,
                "existence": existence, "formula": formula, "latency_dist": latency_dist,
                "preset": preset, "subjects": [n for n, _ in subjects],
                "subject_column": subject_column, "random_state": random_state,
                "package_version": __version__}
    if engine != "eb":
        from .bayes import sample

        result = sample(engine, level_tables, settings, n_chains=n_chains, n_warmup=n_warmup,
                        n_draws=n_draws, target_accept=target_accept,
                        random_state=random_state, formula=(parameter, terms),
                        covariates=covariates, eb_variance=variance)
        level_tables.update(result["level_tables"])
        diagnostics.update(result["diagnostics"])
        variance = result.get("variance_components", variance)
        effects = result.get("effects", effects)
        p_response = result.get("p_response", p_response)
        settings.update(n_chains=n_chains, n_warmup=n_warmup, n_draws=n_draws,
                        target_accept=target_accept)
    provenance = subjects[0][1].provenance.with_step("evoparam.hierarchical.fit_hierarchical",
                                            {k: v for k, v in settings.items()},
                                            duration_s=_time.perf_counter() - start)
    return HierarchicalFit(levels=tuple(levels), summary=group_table, level_tables=level_tables,
                           p_response=p_response, variance_components=variance,
                           effects=effects, engine=engine, diagnostics=diagnostics,
                           settings=settings, provenance=provenance)


def posterior_summary(fit, level="channel"):
    """The table of one level (``trial``, ``channel``, ``subject`` or ``group``)."""

    if level not in fit.level_tables:
        raise SettingsError(f"level must be one of {sorted(fit.level_tables)}, not {level!r}.")
    return fit.level_tables[level]


def compare_hierarchical(fits):
    """Leave-one-out comparison (PSIS-LOO) of Bayesian fits: one row per fit, best first."""

    from .bayes import loo_table

    if any(f.engine == "eb" for f in fits):
        raise SettingsError("compare_hierarchical needs fits of a Bayesian engine (numpyro or "
                            "pymc): the leave-one-out comparison uses posterior draws.")
    return loo_table(fits)
