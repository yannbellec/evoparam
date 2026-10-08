"""Is there a response? test_response and its companions (spec §5.6)."""

import dataclasses
import itertools
import logging
import time as _time
import warnings

import numpy as np

from evoparam.data import Evoked, Provenance, ResponseTest, TrialSet
from evoparam.errors import DataError, SettingsError
from evoparam.utils._optional import require
from evoparam.version import __version__

from .corrections import CORRECTIONS, correct
from .permutation import permutation_test, shuffle_test
from .projection import projection_test
from .rules import apply_rule, load_rules, resolve_rule

__all__ = ["test_response", "response_duration", "compare_detectors", "list_rules",
           "train_discordance", "methods_text", "detectors", "fit", "attach_test", "METHODS",
           "HUMAN_KAPPA"]

METHODS = ("baseline_permutation", "projection", "model_comparison", "threshold")

# Inter-rater agreement of human experts on CCEP N1 presence (van Blooijs et al.).
HUMAN_KAPPA = 0.60


def _groups(ts, by):
    meta = ts.metadata
    columns = [c for c in (by or ()) if meta is not None and c in meta.columns]
    if not columns:
        return [("", np.arange(ts.n_trials))]
    keys = meta[columns].astype(str).agg("/".join, axis=1)
    return [(key, np.flatnonzero(keys.to_numpy() == key)) for key in dict.fromkeys(keys)]


def _check_artifact(ts, window, allow):
    if ts.masks is None or allow:
        return
    win = (ts.times >= window[0]) & (ts.times <= window[1])
    if ts.masks[:, :, win].all(axis=0).any():
        raise SettingsError("The test window overlaps samples replaced in every trial (the "
                            "artifact window); move the window, or allow_artifact_overlap=True.")


def _model_comparison(avg, times, window, baseline, n_permutations, rng):
    from evoparam.models import ResponseModel

    if n_permutations > 200:
        warnings.warn("model_comparison refits the model for every bootstrap sample; "
                      f"{n_permutations} samples may take long.", UserWarning, stacklevel=3)
    model = ResponseModel(fit_window=window, baseline=baseline, classical=False)
    background = ResponseModel(fit_window=window, baseline=baseline, classical=False,
                               max_components=0)
    out = {k: np.full(len(avg), np.nan) for k in ("statistic", "p_value", "onset", "offset",
                                                    "duration", "snr_db")}
    win = (times >= window[0]) & (times <= window[1])
    for c, y in enumerate(avg):
        fit = model.fit_waveform(times, y)
        if fit.gof is None:
            continue
        steps = fit.diagnostics["steps"]
        delta = steps[-1]["criterion"] - steps[0]["criterion"]
        null_fit = background.fit_waveform(times, y)
        sigma, rho = fit.diagnostics["sigma"], fit.diagnostics["rho"]
        null = []
        for _ in range(n_permutations):
            e = rng.normal(0, 1, len(times))
            noise = np.empty_like(e)
            noise[0] = e[0]
            for i in range(1, len(e)):
                noise[i] = rho * noise[i - 1] + np.sqrt(1 - rho ** 2) * e[i]
            synthetic = y.copy()
            synthetic[win] = null_fit.model + sigma * noise[win]
            synthetic[~win] = sigma * noise[~win]
            f = model.fit_waveform(times, synthetic)
            if f.gof is not None:
                s = f.diagnostics["steps"]
                null.append(s[-1]["criterion"] - s[0]["criterion"])
        null = np.array(null)
        out["statistic"][c] = delta
        out["p_value"][c] = (1 + np.sum(null <= delta)) / (1 + len(null))
        if fit.components:
            out["onset"][c] = min(comp.onset for comp in fit.components)
            out["offset"][c] = max(comp.offset for comp in fit.components)
            out["duration"][c] = out["offset"][c] - out["onset"][c]
    out["duration_ci"] = np.full((len(avg), 2), np.nan)
    return out


def test_response(ts, *, method="baseline_permutation", window=(0.009, 0.5),
                  baseline=(-0.5, -0.05), alpha=0.05, correction="fdr_bh",
                  n_permutations=1000, statistic="max_t", cluster_threshold=None,
                  polarity="both", rule=None, rule_overrides=None, picks=None,
                  by=("stim_site",), return_null=False, allow_artifact_overlap=False,
                  n_boot=200, random_state=None, n_jobs=1, verbose=None):
    """Is there a response in the window, on each channel (and stimulation site)?

    Methods:

    - ``"baseline_permutation"`` (default): pointwise ``t`` of the trials, summarised by the
      largest ``|t|`` over the window (``statistic="max_t"``) or the largest cluster mass
      (``"cluster"``); null distribution from random sign flips of whole trials (all
      patterns when there are few trials). Onset and offset: the first and last significant
      sample (or cluster), with a bootstrap interval of the duration over trials. With an
      Evoked (no trials), baseline windows of the average form a weaker null (warning).
    - ``"projection"``: cross-projections between trials (canonical response
      parameterisation, re-implemented from the published description): the response
      duration and a one-sided t-test of the projections at its end.
    - ``"model_comparison"``: change of BIC between the fitted model and the background-only
      model, with a parametric bootstrap under the background-only model.
    - ``"threshold"``: a published rule of the registry (``rule``: name or dict), to the
      letter; fields the paper does not state must be supplied in ``rule_overrides``.

    p-values are corrected across channels and groups (``correction``: ``"fdr_bh"``,
    ``"holm"``, ``"bonferroni"``, ``"none"``); ``significant`` is ``p_value <= alpha``
    (the rule's decision for ``"threshold"``).

    Returns
    -------
    ResponseTest
        Arrays [n_channels] (or [n_channels, n_groups] when ``by`` splits the trials into
        several groups, the group names in ``settings["groups"]``).
    """

    start = _time.perf_counter()
    if method not in METHODS:
        raise SettingsError(f"method must be one of {METHODS}, not {method!r}.")
    if correction not in CORRECTIONS:
        raise SettingsError(f"correction must be one of {CORRECTIONS}.")
    if statistic not in ("max_t", "cluster"):
        raise SettingsError("statistic must be 'max_t' or 'cluster'.")
    if polarity not in ("both", "negative", "positive"):
        raise SettingsError("polarity must be 'both', 'negative' or 'positive'.")
    rng = np.random.default_rng(random_state)
    spec = None
    if method == "threshold":
        spec = resolve_rule(rule if rule is not None else "custom", **(rule_overrides or {}))
        window, baseline = spec["window"], spec["baseline"]

    if isinstance(ts, Evoked):
        evoked, trials_ts = ts, None
    elif isinstance(ts, TrialSet):
        evoked, trials_ts = None, ts
        _check_artifact(ts, window, allow_artifact_overlap)
    else:
        raise DataError("test_response expects a TrialSet or an Evoked.")
    base_obj = trials_ts if trials_ts is not None else evoked
    names = list(base_obj.ch_names)
    index = list(range(len(names))) if picks is None else \
        [names.index(p) if isinstance(p, str) else int(p) for p in picks]
    times = base_obj.times

    groups = _groups(trials_ts, by) if trials_ts is not None else [("", None)]
    results = []
    for _, rows in groups:
        if trials_ts is not None:
            data = trials_ts.data[rows][:, index]
            masks = None if trials_ts.masks is None else trials_ts.masks[rows][:, index]
        if method == "baseline_permutation":
            if trials_ts is None:
                res = shuffle_test(evoked.data[index], times, window, baseline,
                                   n_permutations=n_permutations, polarity=polarity,
                                   alpha=alpha, rng=rng)
            else:
                res = permutation_test(data, times, window, baseline,
                                       n_permutations=n_permutations, statistic=statistic,
                                       cluster_threshold=cluster_threshold, polarity=polarity,
                                       alpha=alpha, rng=rng, n_boot=n_boot, masks=masks)
        elif method == "projection":
            if trials_ts is None:
                raise DataError("method 'projection' needs trials.")
            res = projection_test(data, times, window, baseline, rng=rng)
        elif method == "model_comparison":
            avg = evoked.data[index] if trials_ts is None else data.mean(axis=0)
            res = _model_comparison(avg, times, window, baseline, n_permutations, rng)
        else:
            avg = evoked.data[index] if trials_ts is None else np.nanmean(data, axis=0)
            res = apply_rule(avg, times, base_obj.sfreq, spec, base_obj.units)
            res.update(p_value=np.full(len(index), np.nan), duration=np.full(len(index), np.nan),
                       snr_db=np.full(len(index), np.nan),
                       duration_ci=np.full((len(index), 2), np.nan))
        results.append(res)

    stack = lambda key: (np.asarray(results[0][key], float) if len(results) == 1
                         else np.stack([np.asarray(r[key], float) for r in results], axis=1))
    raw = stack("p_value")
    adjusted = correct(raw, correction)
    if method == "threshold":
        significant = np.asarray(results[0]["significant"]) if len(results) == 1 else \
            np.stack([r["significant"] for r in results], axis=1)
    else:
        significant = np.nan_to_num(adjusted, nan=1.0) <= alpha
    settings = {"method": method, "window": list(window), "baseline": list(baseline),
                "statistic": statistic, "polarity": polarity, "alpha": alpha,
                "correction": correction, "n_permutations": n_permutations,
                "random_state": random_state, "groups": [g for g, _ in groups],
                "channels": [names[i] for i in index], "by": list(by or ()),
                "duration_ci": (results[0]["duration_ci"].tolist() if len(results) == 1
                                else [r["duration_ci"].tolist() for r in results]),
                "exact": bool(results[0].get("exact", False)),
                "package_version": __version__}
    if spec is not None:
        settings["rule_spec"] = {k: (list(v) if isinstance(v, tuple) else v)
                                 for k, v in spec.items() if k != "verified"}
    if return_null:
        settings["critical"] = [np.asarray(r.get("critical", np.nan)).tolist() for r in results]
    provenance = Provenance().with_step("evoparam.existence.test_response",
                                        {k: v for k, v in settings.items()
                                         if k not in ("duration_ci", "critical")},
                                        duration_s=_time.perf_counter() - start)
    return ResponseTest(method=method, statistic=stack("statistic"), p_value=adjusted,
                        p_value_raw=raw, significant=significant, duration=stack("duration"),
                        onset=stack("onset"), offset=stack("offset"), snr_db=stack("snr_db"),
                        rule=spec["name"] if spec else None, alpha=alpha, correction=correction,
                        n_permutations=n_permutations, window=window, settings=settings,
                        provenance=provenance)


def response_duration(ts, *, method="baseline_permutation", **kwargs):
    """Onset, offset and duration of the response of each channel, as a table."""

    pd = require("pandas", extra="data", caller="evoparam.existence.response_duration")
    test = test_response(ts, method=method, **kwargs)
    ci = np.asarray(test.settings["duration_ci"], float).reshape(-1, 2)
    return pd.DataFrame({"channel": test.settings["channels"],
                         "onset": np.ravel(test.onset), "offset": np.ravel(test.offset),
                         "duration": np.ravel(test.duration),
                         "duration_ci_low": ci[:, 0], "duration_ci_high": ci[:, 1],
                         "significant": np.ravel(test.significant)})


def list_rules():
    """The registry of published rules, as a table."""

    pd = require("pandas", extra="data", caller="evoparam.existence.list_rules")
    rows = []
    for name, spec in load_rules().items():
        rows.append({"name": name, "statistic": spec["statistic"], "k": spec["k"],
                     "window": spec["window"], "baseline": spec["baseline"],
                     "unspecified": ",".join(spec.get("unspecified", [])),
                     "source": spec["source"]})
    return pd.DataFrame(rows)


def _kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def compare_detectors(ts, rules=("baseline_permutation", "vanblooijs2023", "hays2021",
                                 "projection"), *, rule_overrides=None, **kwargs):
    """Decisions of several methods and rules on every channel, and their agreement.

    Parameters
    ----------
    rules : sequence of str
        Method names (``"baseline_permutation"``, ``"projection"``, ``"model_comparison"``)
        or registry rule names.
    rule_overrides : dict, optional
        Rule name -> overrides (for the fields a paper does not state, such as ``baseline``).
    **kwargs
        Passed to :func:`test_response` for the methods (window, baseline, ...).

    Returns
    -------
    pandas.DataFrame
        One row per channel and one boolean column per detector; ``attrs["kappa"]`` holds the
        pairwise Cohen's kappa table and ``attrs["caption"]`` quotes the human inter-rater
        agreement (0.60) as the reference.
    """

    pd = require("pandas", extra="data", caller="evoparam.existence.compare_detectors")
    decisions = {}
    for name in rules:
        if name in METHODS:
            test = test_response(ts, method=name, by=(), **kwargs)
        else:
            test = test_response(ts, method="threshold", rule=name, by=(),
                                 rule_overrides=(rule_overrides or {}).get(name),
                                 random_state=kwargs.get("random_state"))
        decisions[name] = np.ravel(test.significant)
    table = pd.DataFrame(decisions, index=test.settings["channels"])
    kappa = pd.DataFrame(np.eye(len(rules)), index=list(rules), columns=list(rules))
    for a, b in itertools.combinations(rules, 2):
        kappa.loc[a, b] = kappa.loc[b, a] = _kappa(decisions[a], decisions[b])
    table.attrs["kappa"] = kappa
    table.attrs["caption"] = (f"Pairwise Cohen's kappa between detectors; for reference, "
                              f"human experts agree on N1 presence with kappa about "
                              f"{HUMAN_KAPPA:.2f} (van Blooijs et al.).")
    return table


def train_discordance(ts, test):
    """Agreement of the decision across trains: ``d = 1 - 2 |P(significant) - 0.5|``.

    The test of ``test`` is re-run on the trials of each train
    (``metadata["train_id"]``). ``d`` is 0 when all trains agree and 1 when half do.
    """

    pd = require("pandas", extra="data", caller="evoparam.existence.train_discordance")
    if ts.metadata is None or "train_id" not in ts.metadata.columns:
        raise DataError("train_discordance needs metadata['train_id'] (events.group_trains).")
    s = test.settings
    decisions = []
    for _, rows in ts.metadata.groupby("train_id").indices.items():
        sub = ts.select(trials=list(rows))
        kwargs = dict(method=s["method"], window=tuple(s["window"]),
                      baseline=tuple(s["baseline"]), alpha=s["alpha"], correction="none",
                      n_permutations=s["n_permutations"], statistic=s["statistic"],
                      polarity=s["polarity"], by=(), picks=s["channels"],
                      random_state=s["random_state"], n_boot=0)
        if s["method"] == "threshold":
            kwargs.update(rule=test.rule, rule_overrides={"baseline": tuple(s["baseline"]),
                                                          "window": tuple(s["window"])})
        decisions.append(np.ravel(test_response(sub, **kwargs).significant))
    share = np.mean(decisions, axis=0)
    return pd.DataFrame({"channel": s["channels"], "n_trains": len(decisions),
                         "p_significant": share, "discordance": 1 - 2 * np.abs(share - 0.5)})


def methods_text(test):
    """Methods sentence for a ResponseTest (spec REQ-140)."""

    s = test.settings
    window = f"{s['window'][0] * 1000:g} to {s['window'][1] * 1000:g} ms"
    baseline = f"{s['baseline'][0] * 1000:g} to {s['baseline'][1] * 1000:g} ms"
    if test.method == "threshold":
        spec = s.get("rule_spec", {})
        return (f"Responses were detected with the published rule '{test.rule}' "
                f"({spec.get('source', 'custom')}): {spec.get('statistic')} above "
                f"{spec.get('k')} in {window}, baseline {baseline} (evoparam "
                f"{s['package_version']}).")
    what = {"baseline_permutation": f"a sign-flip permutation test of the trials "
                                    f"({s['statistic']} statistic, {s['n_permutations']} "
                                    f"permutations{', all patterns' if s['exact'] else ''})",
            "projection": "the cross-projection test of the trials",
            "model_comparison": f"a model comparison (change of BIC) with a parametric "
                                f"bootstrap ({s['n_permutations']} samples)"}[test.method]
    return (f"The presence of a response in {window} was tested with {what} against the "
            f"baseline {baseline}, {s['polarity']} polarity, alpha {s['alpha']:g}, "
            f"{s['correction']} correction across channels, random seed "
            f"{s['random_state']} (evoparam {s['package_version']}).")


class detectors:
    """The registry rules as functions: ``detectors.vanblooijs2023(ts, baseline=(-1, -0.1))``."""


def _make_detector(name):
    def detector(ts, **overrides):
        return test_response(ts, method="threshold", rule=name, rule_overrides=overrides, by=())
    detector.__name__ = name
    detector.__doc__ = f"Published rule '{name}' ({load_rules()[name]['source']})."
    return staticmethod(detector)


for _name in load_rules():
    setattr(detectors, _name, _make_detector(_name))


def fit(data, preset=None, *, existence=False, existence_kwargs=None, **kwargs):
    """:func:`evoparam.models.fit`, and with ``existence=True`` the existence test of every
    channel attached to its fit (``ResponseFit.existence``; ``existence_p`` and
    ``existence_significant`` in ``GroupFit.to_dataframe()``)."""

    from evoparam.models import fit as fit_models

    group = fit_models(data, preset, **kwargs)
    if not existence:
        _log_methods(group, kwargs.get("verbose"))
        return group
    if not isinstance(data, TrialSet):
        raise SettingsError("existence=True needs a TrialSet (the trials).")
    settings = group.settings
    test_kwargs = dict(window=tuple(settings["fit_window"]),
                       baseline=tuple(settings["baseline"]) if settings.get("baseline") else
                       (-0.5, -0.05), by=(), random_state=kwargs.get("random_state"))
    test_kwargs.update(existence_kwargs or {})
    test = test_response(data, **test_kwargs)
    group = attach_test(group, test)
    _log_methods(group, kwargs.get("verbose"))
    return group


def attach_test(group, test):
    """The GroupFit with the existence test of each channel attached to its fit
    (``ResponseFit.existence``) and the test recorded as a provenance step of the group and
    of every fit."""

    channels = test.settings["channels"]
    step = {k: v for k, v in test.settings.items() if k not in _PER_CHANNEL_SETTINGS}
    fits = []
    for fit_, ch in zip(group.fits, group.index["channel"]):
        i = channels.index(ch)
        one = dataclasses.replace(
            test, statistic=test.statistic[i:i + 1], p_value=test.p_value[i:i + 1],
            p_value_raw=test.p_value_raw[i:i + 1], significant=test.significant[i:i + 1],
            duration=test.duration[i:i + 1], onset=test.onset[i:i + 1],
            offset=test.offset[i:i + 1], snr_db=test.snr_db[i:i + 1])
        fits.append(dataclasses.replace(
            fit_, existence=one,
            provenance=fit_.provenance.with_step("evoparam.existence.test_response", step)))
    return dataclasses.replace(
        group, fits=fits,
        provenance=group.provenance.with_step("evoparam.existence.test_response", step))


# Settings of a test that hold one value per channel; the provenance step of the test
# attached to a fit keeps the settings that apply to every channel.
_PER_CHANNEL_SETTINGS = ("channels", "duration_ci", "groups", "critical")


def _log_methods(group, verbose):
    """Log the methods paragraph at INFO (spec §5.13: printed when verbose is INFO)."""

    logger = logging.getLogger("evoparam")
    explicit = verbose in (True, "info", "INFO", logging.INFO)
    if not explicit and (verbose is not None or not logger.isEnabledFor(logging.INFO)):
        return
    previous = logger.level
    if explicit and not logger.isEnabledFor(logging.INFO):
        logger.setLevel(logging.INFO)
    try:
        logger.info("Methods: %s", group.methods_text())
    finally:
        logger.setLevel(previous)
