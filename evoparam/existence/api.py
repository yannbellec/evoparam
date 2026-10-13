# SPDX-License-Identifier: Apache-2.0
"""Is there a response? test_response and its companions (spec §5.6)."""

import copy
import dataclasses
import itertools
import logging
import time as _time
import warnings
from collections.abc import Sequence

import numpy as np

from evoparam._typing import Any, NDArray
from evoparam.data import Evoked, Provenance, ResponseTest, TrialSet
from evoparam.data._table import row_labels
from evoparam.errors import DataError, SettingsError
from evoparam.errors import warn as _warn
from evoparam.utils._failures import PROPAGATE, check_on_error, fit_error, record_failure
from evoparam.utils._logging import verbose
from evoparam.utils._optional import require
from evoparam.utils._parallel import parallel_map
from evoparam.version import __version__

from .corrections import CORRECTIONS, correct
from .permutation import permutation_test, shuffle_test
from .projection import projection_test
from .rules import apply_rule, load_rules, resolve_rule

__all__ = [
    "test_response",
    "existence_detector",
    "response_duration",
    "compare_detectors",
    "list_rules",
    "train_discordance",
    "methods_text",
    "detectors",
    "fit",
    "attach_test",
    "METHODS",
    "HUMAN_KAPPA",
]

logger = logging.getLogger(__name__)

METHODS = ("baseline_permutation", "projection", "model_comparison", "threshold")

# Inter-rater agreement of human experts on CCEP N1 presence (van Blooijs et al.).
HUMAN_KAPPA = 0.60


def _groups(ts: Any, by: Any) -> Any:
    meta = ts.metadata
    columns = [c for c in (by or ()) if meta is not None and c in meta.columns]
    if not columns:
        return [("", np.arange(ts.n_trials))]
    keys = row_labels(meta, columns)
    return [(key, np.flatnonzero(keys == key)) for key in dict.fromkeys(keys)]


def _check_artifact(ts: Any, window: Any, allow: Any) -> None:
    if ts.masks is None or allow:
        return
    win = (ts.times >= window[0]) & (ts.times <= window[1])
    if ts.masks[:, :, win].all(axis=0).any():
        raise SettingsError(
            "The test window overlaps samples replaced in every trial (the "
            "artifact window); move the window, or allow_artifact_overlap=True."
        )


def _comparison_fits(unit: tuple[Any, ...]) -> dict[str, Any] | None:
    """Fit the model and the background-only model to one channel (model_comparison unit)."""
    from evoparam.models import ResponseModel

    times, y, window, baseline = unit
    model = ResponseModel(fit_window=window, baseline=baseline, classical=False)
    background = ResponseModel(
        fit_window=window, baseline=baseline, classical=False, max_components=0
    )
    # a failed fit is the failure of the channel (REQ-307 ii: recorded by test_response)
    fit = model.fit_waveform(times, y, on_error="raise")
    if fit.gof is None:
        return None
    steps = fit.diagnostics["steps"]
    null_fit = background.fit_waveform(times, y, on_error="raise")
    return {
        "delta": steps[-1]["criterion"] - steps[0]["criterion"],
        "null_model": null_fit.model,
        "sigma": fit.diagnostics["sigma"],
        "rho": fit.diagnostics["rho"],
        "onset": min(comp.onset for comp in fit.components) if fit.components else None,
        "offset": max(comp.offset for comp in fit.components) if fit.components else None,
    }


def _comparison_null(unit: tuple[Any, ...]) -> NDArray[Any]:
    """Return the parametric-bootstrap null of one channel (second model_comparison unit).

    The noise is drawn from a copy of the generator taken when the serial loop reached
    this channel, so the null does not depend on n_jobs.
    """
    from evoparam.models import ResponseModel

    times, y, window, baseline, first, n_permutations, rng = unit
    model = ResponseModel(fit_window=window, baseline=baseline, classical=False)
    win = (times >= window[0]) & (times <= window[1])
    sigma, rho = first["sigma"], first["rho"]
    null = []
    for _ in range(n_permutations):
        e = rng.normal(0, 1, len(times))
        noise = np.empty_like(e)
        noise[0] = e[0]
        for i in range(1, len(e)):
            noise[i] = rho * noise[i - 1] + np.sqrt(1 - rho**2) * e[i]
        synthetic = y.copy()
        synthetic[win] = first["null_model"] + sigma * noise[win]
        synthetic[~win] = sigma * noise[~win]
        f = model.fit_waveform(times, synthetic)
        if f.gof is not None:
            s = f.diagnostics["steps"]
            null.append(s[-1]["criterion"] - s[0]["criterion"])
    return np.array(null)


def _model_comparison(
    avg: Any,
    times: Any,
    window: Any,
    baseline: Any,
    n_permutations: Any,
    rng: np.random.Generator,
    n_jobs: int | None = 1,
) -> dict[str, NDArray[Any]]:
    if n_permutations > 200:
        _warn(
            "model_comparison refits the model for every bootstrap sample; "
            f"{n_permutations} samples may take long.",
            UserWarning,
            stacklevel=3,
        )
    out: dict[str, NDArray[Any]] = {
        k: np.full(len(avg), np.nan)
        for k in ("statistic", "p_value", "onset", "offset", "duration", "snr_db")
    }
    firsts = parallel_map(
        _comparison_fits,
        [(times, y, window, baseline) for y in avg],
        n_jobs,
        desc="test_response",
    )
    # a copy of the generator when the serial loop reaches each fitted channel; the
    # draws of the channel are then skipped here (made again in its unit)
    units = []
    for c, first in enumerate(firsts):
        if first is None:
            continue
        units.append(
            (
                times,
                avg[c],
                window,
                baseline,
                first,
                n_permutations,
                copy.deepcopy(rng),
            )
        )
        for _ in range(n_permutations):
            rng.normal(0, 1, len(times))
    nulls = parallel_map(_comparison_null, units, n_jobs, desc="test_response")
    fitted = [(c, first) for c, first in enumerate(firsts) if first is not None]
    for (c, first), null in zip(fitted, nulls, strict=True):
        delta = first["delta"]
        out["statistic"][c] = delta
        out["p_value"][c] = (1 + np.sum(null <= delta)) / (1 + len(null))
        if first["onset"] is not None:
            out["onset"][c] = first["onset"]
            out["offset"][c] = first["offset"]
            out["duration"][c] = out["offset"][c] - out["onset"][c]
    out["duration_ci"] = np.full((len(avg), 2), np.nan)
    return out


@verbose
def test_response(  # noqa: C901, PLR0912, PLR0913, PLR0915  (public API, signature fixed by spec §5.6)
    ts: Any,
    *,
    method: str = "baseline_permutation",  # noqa: PT028  (public API test_response, not a pytest test)
    window: Any = (0.009, 0.5),  # noqa: PT028  (public API test_response, not a pytest test)
    baseline: Any = (-0.5, -0.05),  # noqa: PT028  (public API test_response, not a pytest test)
    alpha: float = 0.05,  # noqa: PT028  (public API test_response, not a pytest test)
    correction: str = "fdr_bh",  # noqa: PT028  (public API test_response, not a pytest test)
    n_permutations: int = 1000,  # noqa: PT028  (public API test_response, not a pytest test)
    statistic: str = "max_t",  # noqa: PT028  (public API test_response, not a pytest test)
    cluster_threshold: Any = None,  # noqa: PT028  (public API test_response, not a pytest test)
    polarity: str = "both",  # noqa: PT028  (public API test_response, not a pytest test)
    rule: Any = None,  # noqa: PT028  (public API test_response, not a pytest test)
    rule_overrides: Any = None,  # noqa: PT028  (public API test_response, not a pytest test)
    picks: Any = None,  # noqa: PT028  (public API test_response, not a pytest test)
    by: Any = ("stim_site",),  # noqa: PT028  (public API test_response, not a pytest test)
    return_null: bool = False,  # noqa: PT028  (public API test_response, not a pytest test)
    allow_artifact_overlap: bool = False,  # noqa: PT028  (public API test_response, not a pytest test)
    n_boot: int = 200,  # noqa: PT028  (public API test_response, not a pytest test)
    random_state: Any = None,  # noqa: PT028  (public API test_response, not a pytest test)
    n_jobs: int = 1,  # noqa: PT028  (public API test_response, not a pytest test)
    verbose: Any = None,  # noqa: ARG001, PT028  (read by @verbose; public API, not a pytest test)
    projection_profile: str = "published",  # noqa: PT028  (public API test_response, not a pytest test)
    projection_pairs: str = "signflip",  # noqa: PT028  (public API test_response, not a pytest test)
    on_error: str = "record",  # noqa: PT028  (public API test_response, not a pytest test)
) -> "ResponseTest":
    """Test whether there is a response in the window, on each channel (and stimulation site).

    Methods
    -------
    - ``"baseline_permutation"`` (default): pointwise ``t`` of the trials, summarised by the
      largest ``|t|`` over the window (``statistic="max_t"``) or the largest cluster mass
      (``"cluster"``); null distribution from random sign flips of whole trials (all
      patterns when there are few trials). Onset and offset: the first and last significant
      sample (or cluster), with a bootstrap interval of the duration over trials. With an
      Evoked (no trials), baseline windows of the average form a weaker null (warning),
      and the intervals add baseline segments of the average to its window (a baseline
      bootstrap, conservative; REQ-006).
    - ``"projection"``: cross-projections between trials (canonical response
      parameterisation, re-implemented from the published description): the response
      duration and the significance of the projections.
      ``projection_profile`` chooses the response duration: ``"published"`` (default, the
      peak of the mean projection, as the paper defines it) or ``"normalised"`` (the peak
      of the mean projection divided by the square root of the window length, §5.6 item 2
      as reviewed; see ``docs/spec/DECISIONS.md``). ``projection_pairs`` chooses the test:
      ``"signflip"`` (default: every cross-projection, the peak of the mean-projection
      profile against its null from sign flips of whole trials; rejects pure noise at
      ``alpha`` and gives the paper's decisions on its examples), ``"disjoint"`` (a t-test,
      each trial in one pair) or ``"half"`` (a t-test on one projection per unordered pair,
      the paper's set, which rejects pure noise at 0.33 to 0.44 at ``alpha = 0.05``).
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
        several groups, the group names in ``settings["groups"]``). ``curve`` holds the
        statistic over time in the window (the pointwise t of the trials; the baseline z of
        the average without trials or for a registry rule; the projection profile
        for ``"projection"``; None for ``"model_comparison"``) on ``curve_times``, and
        ``null_band`` the level it must cross: the family-wise critical value of the null
        maximum (``max_t``), the cluster-forming threshold (``cluster``), the rule's ``k``.

    Notes
    -----
    ``n_jobs`` runs the channels in parallel (``joblib`` processes when the extra
    ``parallel`` is installed, else threads): the p-values and the bootstrap of the
    permutation tests, the channels of ``"projection"`` and ``"model_comparison"``. Every
    random number is drawn in the order of the serial run, so the result does not depend
    on ``n_jobs``. ``verbose`` sets the level of the ``evoparam`` logger for the call (the
    summary of the test at INFO; ``"progress"`` adds a progress bar, extra ``progress``).

    A channel whose test fails (REQ-307 ii) is kept with ``status="failed"`` and the
    exception in ``message`` (arrays shaped as ``statistic``), NaN numbers and
    ``significant`` False, and one WARNING record names it (``on_error="record"``, the
    default); ``on_error="raise"`` raises the first :class:`~evoparam.errors.FitError`.
    The channels are found by testing them one at a time after the test of all of them
    failed, each from the random state of the group; errors of the data or of the settings
    always propagate.
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
    check_on_error(on_error)
    if random_state is None:
        # a seed drawn from the operating system and recorded in the settings, so that the
        # test can be regenerated from its provenance (REQ-007, REQ-333)
        random_state = int(np.random.default_rng().integers(2**31 - 1))
    rng = np.random.default_rng(random_state)
    spec = None
    if method == "threshold":
        spec = resolve_rule(rule if rule is not None else "custom", **(rule_overrides or {}))
        window, baseline = spec["window"], spec["baseline"]

    evoked: Any  # an Evoked whenever trials_ts is None
    base_obj: TrialSet | Evoked
    if isinstance(ts, Evoked):
        evoked, trials_ts = ts, None
    elif isinstance(ts, TrialSet):
        evoked, trials_ts = None, ts
        _check_artifact(ts, window, allow_artifact_overlap)
    else:
        raise DataError("test_response expects a TrialSet or an Evoked.")
    base_obj = trials_ts if trials_ts is not None else evoked
    names = list(base_obj.ch_names)
    index = (
        list(range(len(names)))
        if picks is None
        else [names.index(p) if isinstance(p, str) else int(p) for p in picks]
    )
    times = base_obj.times

    groups = _groups(trials_ts, by) if trials_ts is not None else [("", None)]

    def run(rows: Any, chans: Any, gen: np.random.Generator) -> dict[str, Any]:
        """Test the channels ``chans`` on the trials ``rows`` of one group."""
        if trials_ts is not None:
            data = trials_ts.data[rows][:, chans]
            masks = None if trials_ts.masks is None else trials_ts.masks[rows][:, chans]
        if method == "baseline_permutation":
            if trials_ts is None:
                res = shuffle_test(
                    evoked.data[chans],
                    times,
                    window,
                    baseline,
                    n_permutations=n_permutations,
                    polarity=polarity,
                    alpha=alpha,
                    rng=gen,
                    n_jobs=n_jobs,
                    n_boot=n_boot,
                )
            else:
                res = permutation_test(
                    data,
                    times,
                    window,
                    baseline,
                    n_permutations=n_permutations,
                    statistic=statistic,
                    cluster_threshold=cluster_threshold,
                    polarity=polarity,
                    alpha=alpha,
                    rng=gen,
                    n_boot=n_boot,
                    masks=masks,
                    n_jobs=n_jobs,
                )
        elif method == "projection":
            if trials_ts is None:
                raise DataError("method 'projection' needs trials.")
            res = projection_test(
                data,
                times,
                window,
                baseline,
                rng=gen,
                profile=projection_profile,
                pairs=projection_pairs,
                n_jobs=n_jobs,
            )
        elif method == "model_comparison":
            avg = evoked.data[chans] if trials_ts is None else data.mean(axis=0)
            res = _model_comparison(
                avg, times, window, baseline, n_permutations, gen, n_jobs=n_jobs
            )
        else:
            avg = evoked.data[chans] if trials_ts is None else np.nanmean(data, axis=0)
            res = apply_rule(avg, times, base_obj.sfreq, spec, base_obj.units)
            res.update(
                p_value=np.full(len(chans), np.nan),
                duration=np.full(len(chans), np.nan),
                snr_db=np.full(len(chans), np.nan),
                duration_ci=np.full((len(chans), 2), np.nan),
            )
        return res

    results, failures = [], []
    for group, rows in groups:
        snapshot = copy.deepcopy(rng)
        try:
            res, failed = run(rows, index, rng), dict[int, str]()
        except PROPAGATE:
            raise
        except Exception:
            # the channels that fail on their own (REQ-307 ii); none: not a channel's failure
            res, failed, rng = _channel_by_channel(
                run, rows, index, names, group, snapshot, on_error
            )
            if not failed:
                raise
        results.append(res)
        failures.append(failed)

    def stack(key: Any) -> Any:
        return (
            np.asarray(results[0][key], float)
            if len(results) == 1
            else np.stack([np.asarray(r[key], float) for r in results], axis=1)
        )

    raw = stack("p_value")
    adjusted = correct(raw, correction)
    if method == "threshold":
        significant = (
            np.asarray(results[0]["significant"])
            if len(results) == 1
            else np.stack([r["significant"] for r in results], axis=1)
        )
    else:
        significant = np.nan_to_num(adjusted, nan=1.0) <= alpha
    settings = {
        "method": method,
        "window": list(window),
        "baseline": list(baseline),
        "statistic": statistic,
        "polarity": polarity,
        "alpha": alpha,
        "correction": correction,
        "n_permutations": n_permutations,
        "random_state": random_state,
        "groups": [g for g, _ in groups],
        "channels": [names[i] for i in index],
        "by": list(by or ()),
        "duration_ci": (
            results[0]["duration_ci"].tolist()
            if len(results) == 1
            else [r["duration_ci"].tolist() for r in results]
        ),
        **{
            key: (
                _interval(results[0], key)
                if len(results) == 1
                else [_interval(r, key) for r in results]
            )
            for key in ("onset_ci", "offset_ci")
        },
        "exact": bool(results[0].get("exact", False)),
        "package_version": __version__,
    }
    if method == "projection":
        settings["projection_profile"] = projection_profile
        settings["projection_pairs"] = projection_pairs
    if spec is not None:
        settings["rule_spec"] = {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items() if k != "verified"
        }
    if return_null:
        settings["critical"] = [np.asarray(r.get("critical", np.nan)).tolist() for r in results]
    provenance = Provenance().with_step(
        "evoparam.existence.test_response",
        {k: v for k, v in settings.items() if k not in _PER_CHANNEL_SETTINGS},
        duration_s=_time.perf_counter() - start,
    )
    logger.info(
        "test_response (%s): %d channel(s) x %d group(s), %d significant at alpha=%g, %.2f s",
        method,
        len(index),
        len(groups),
        int(np.sum(significant)),
        alpha,
        _time.perf_counter() - start,
    )
    status = np.full((len(index), len(groups)), "ok", dtype=object)
    message = np.full((len(index), len(groups)), "", dtype=object)
    for g, failed in enumerate(failures):
        for pos, text in failed.items():
            status[pos, g], message[pos, g] = "failed", text
    if len(groups) == 1:
        status, message = status[:, 0], message[:, 0]
    curve_times = curve = band = None
    if all(r.get("curve") is not None for r in results):
        # the statistic over time and its null band (spec §5.13, plot_existence)
        curve_times = np.asarray(results[0]["curve_times"], float)
        curve, band = stack("curve"), stack("null_band")
    return ResponseTest(
        method=method,
        statistic=stack("statistic"),
        p_value=adjusted,
        p_value_raw=raw,
        significant=significant,
        duration=stack("duration"),
        onset=stack("onset"),
        offset=stack("offset"),
        snr_db=stack("snr_db"),
        rule=spec["name"] if spec else None,
        alpha=alpha,
        correction=correction,
        n_permutations=n_permutations,
        window=window,
        settings=settings,
        provenance=provenance,
        curve_times=curve_times,
        curve=curve,
        null_band=band,
        status=status.astype(str),
        message=message.astype(str),
    )


# the entries of a test's result that hold one value (or row) per channel
_PER_CHANNEL = (
    "statistic",
    "p_value",
    "onset",
    "offset",
    "duration",
    "snr_db",
    "significant",
    "duration_ci",
    "onset_ci",
    "offset_ci",
    "critical",
    "curve",
    "null_band",
)


def _channel_by_channel(
    run: Any,
    rows: Any,
    index: Any,
    names: Any,
    group: str,
    snapshot: np.random.Generator,
    on_error: str,
) -> tuple[dict[str, Any], dict[int, str], np.random.Generator]:
    """Test the channels one at a time, each from the random state ``snapshot``.

    Returns the merged result, {position in ``index``: message} of the channels that failed
    (recorded, or raised as :class:`FitError` with ``on_error="raise"``), and the generator
    of the last channel, which the next group continues.
    """
    parts: list[dict[str, Any] | None] = []
    failed: dict[int, str] = {}
    gen = snapshot
    for pos, ch in enumerate(index):
        gen = copy.deepcopy(snapshot)
        try:
            parts.append(run(rows, [ch], gen))
        except PROPAGATE:
            raise
        except Exception as exc:  # the failure of this channel only
            unit = f"{names[ch]} ({group})" if group else str(names[ch])
            if on_error == "raise":
                raise fit_error("test_response", unit, exc) from exc
            failed[pos] = record_failure(logger, "test_response", unit, exc)
            parts.append(None)
    return _merge(parts), failed, gen


def _merge(parts: list[dict[str, Any] | None]) -> dict[str, Any]:
    """Merge the results of single channels into one (None: a failed channel, NaN)."""
    done = [part for part in parts if part is not None]
    if not done:
        n = len(parts)
        out: dict[str, Any] = {
            key: np.full(n, np.nan)
            for key in ("statistic", "p_value", "onset", "offset", "duration", "snr_db")
        }
        out["significant"] = np.zeros(n, bool)
        out["duration_ci"] = np.full((n, 2), np.nan)
        return out
    first = done[0]
    out = dict(first)
    for key in _PER_CHANNEL:
        if key not in first or first[key] is None:
            continue
        value = np.asarray(first[key])
        if value.ndim == 0:
            continue
        empty = np.full(value.shape, False if value.dtype == bool else np.nan)
        out[key] = np.concatenate(
            [empty if part is None else np.asarray(part[key]) for part in parts], axis=0
        )
    return out


def _interval(result: Any, key: Any) -> Any:
    """Return the [n_channels, 2] interval ``key`` of one result, NaN when the method has none."""
    n = len(np.atleast_1d(result["p_value"]))
    return np.asarray(result.get(key, np.full((n, 2), np.nan)), float).tolist()


def existence_detector(
    ts: Any,
    *,
    window: Any = (0.009, 0.5),
    by: Any = ("stim_site",),
    allow_artifact_overlap: bool = True,
    random_state: int = 0,
    **kwargs: Any,
) -> Any:
    """Run the existence test as a detector for :func:`evoparam.prep.reference_sensitivity`.

    It is the default there (``detector="existence"``).

    Runs :func:`test_response` (default method, ``by`` stimulation site; masked samples are
    left out of the statistic, so a window touching the artifact mask is allowed) and
    returns, per channel and group, whether the response is significant and its latency:
    the time of the largest absolute value of the trial average between the test's onset
    and offset (NaN when not significant).

    Returns
    -------
    detected : bool array [n_channels] (one group) or [n_channels, n_groups]
    latency : float array, same shape, seconds
    groups : list of str
        Group names (``[""]`` without grouping).
    """
    test = test_response(
        ts,
        window=window,
        by=by,
        allow_artifact_overlap=allow_artifact_overlap,
        random_state=random_state,
        **kwargs,
    )
    groups = list(test.settings["groups"])
    index = [ts.ch_names.index(name) for name in test.settings["channels"]]
    significant = np.asarray(test.significant, dtype=bool).reshape(len(index), -1)
    onset = np.asarray(test.onset, dtype=float).reshape(len(index), -1)
    offset = np.asarray(test.offset, dtype=float).reshape(len(index), -1)
    latency = np.full(significant.shape, np.nan)
    times = ts.times
    for g, (_, rows) in enumerate(_groups(ts, by)):
        data = ts.data[rows][:, index]
        if ts.masks is not None:
            data = np.where(ts.masks[rows][:, index], np.nan, data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            avg = np.nanmean(data, axis=0)
        for c in np.flatnonzero(significant[:, g]):
            inside = (times >= onset[c, g]) & (times <= offset[c, g])
            if not inside.any() or not np.isfinite(avg[c, inside]).any():
                continue
            values = np.where(np.isfinite(avg[c, inside]), np.abs(avg[c, inside]), -np.inf)
            latency[c, g] = float(times[inside][int(np.argmax(values))])
    if significant.shape[1] == 1:
        return significant[:, 0], latency[:, 0], groups
    return significant, latency, groups


def response_duration(ts: Any, *, method: str = "baseline_permutation", **kwargs: Any) -> Any:
    """Onset, offset and duration of the response of each channel, as a table.

    Each value with its interval (bootstrap over trials, NaN when the method gives none) and the
    decision with its p-value (REQ-006).
    """
    pd = require("pandas", extra="data", caller="evoparam.existence.response_duration")
    test = test_response(ts, method=method, **kwargs)
    columns = {"channel": test.settings["channels"]}
    for name in ("onset", "offset", "duration"):
        ci = np.asarray(test.settings.get(f"{name}_ci", np.nan), float)
        ci = (
            np.broadcast_to(ci, (len(columns["channel"]), 2))
            if ci.size == 1
            else ci.reshape(-1, 2)
        )
        columns.update(
            {
                name: np.ravel(getattr(test, name)),
                f"{name}_ci_low": ci[:, 0],
                f"{name}_ci_high": ci[:, 1],
            }
        )
    columns.update(significant=np.ravel(test.significant), p_value=np.ravel(test.p_value))
    return pd.DataFrame(columns)


def list_rules() -> Any:
    """Return the registry of published rules, as a table."""
    pd = require("pandas", extra="data", caller="evoparam.existence.list_rules")
    rows = []
    for name, spec in load_rules().items():
        rows.append(
            {
                "name": name,
                "statistic": spec["statistic"],
                "k": spec["k"],
                "window": spec["window"],
                "baseline": spec["baseline"],
                "unspecified": ",".join(spec.get("unspecified", [])),
                "source": spec["source"],
            }
        )
    return pd.DataFrame(rows)


def _kappa(a: Any, b: Any) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def compare_detectors(
    ts: Any,
    rules: Sequence[str] = ("baseline_permutation", "vanblooijs2023", "hays2021", "projection"),
    *,
    rule_overrides: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
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
            test = test_response(
                ts,
                method="threshold",
                rule=name,
                by=(),
                rule_overrides=(rule_overrides or {}).get(name),
                random_state=kwargs.get("random_state"),
            )
        decisions[name] = np.ravel(test.significant)
    table = pd.DataFrame(decisions, index=test.settings["channels"])
    kappa = pd.DataFrame(np.eye(len(rules)), index=list(rules), columns=list(rules))
    for a, b in itertools.combinations(rules, 2):
        kappa.loc[a, b] = kappa.loc[b, a] = _kappa(decisions[a], decisions[b])
    table.attrs["kappa"] = kappa
    table.attrs["caption"] = (
        f"Pairwise Cohen's kappa between detectors; for reference, "
        f"human experts agree on N1 presence with kappa about "
        f"{HUMAN_KAPPA:.2f} (van Blooijs et al.)."
    )
    return table


def train_discordance(ts: Any, test: Any) -> Any:
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
        kwargs = {
            "method": s["method"],
            "window": tuple(s["window"]),
            "baseline": tuple(s["baseline"]),
            "alpha": s["alpha"],
            "correction": "none",
            "n_permutations": s["n_permutations"],
            "statistic": s["statistic"],
            "polarity": s["polarity"],
            "by": (),
            "picks": s["channels"],
            "random_state": s["random_state"],
            "n_boot": 0,
        }
        if s["method"] == "threshold":
            kwargs.update(
                rule=test.rule,
                rule_overrides={"baseline": tuple(s["baseline"]), "window": tuple(s["window"])},
            )
        decisions.append(np.ravel(test_response(sub, **kwargs).significant))
    share = np.mean(decisions, axis=0)
    return pd.DataFrame(
        {
            "channel": s["channels"],
            "n_trains": len(decisions),
            "p_significant": share,
            "discordance": 1 - 2 * np.abs(share - 0.5),
        }
    )


def methods_text(test: Any) -> str:
    """Methods sentence for a ResponseTest (spec REQ-140)."""
    s = test.settings
    window = f"{s['window'][0] * 1000:g} to {s['window'][1] * 1000:g} ms"
    baseline = f"{s['baseline'][0] * 1000:g} to {s['baseline'][1] * 1000:g} ms"
    if test.method == "threshold":
        spec = s.get("rule_spec", {})
        return (
            f"Responses were detected with the published rule '{test.rule}' "
            f"({spec.get('source', 'custom')}): {spec.get('statistic')} above "
            f"{spec.get('k')} in {window}, baseline {baseline} (evoparam "
            f"{s['package_version']})."
        )
    what = {
        "baseline_permutation": f"a sign-flip permutation test of the trials "
        f"({s['statistic']} statistic, {s['n_permutations']} "
        f"permutations{', all patterns' if s['exact'] else ''})",
        "projection": "the cross-projection test of the trials",
        "model_comparison": f"a model comparison (change of BIC) with a parametric "
        f"bootstrap ({s['n_permutations']} samples)",
    }[test.method]
    return (
        f"The presence of a response in {window} was tested with {what} against the "
        f"baseline {baseline}, {s['polarity']} polarity, alpha {s['alpha']:g}, "
        f"{s['correction']} correction across channels, random seed "
        f"{s['random_state']} (evoparam {s['package_version']})."
    )


class detectors:  # noqa: N801  (public namespace evoparam.existence.detectors)
    """The registry rules as functions: ``detectors.vanblooijs2023(ts, baseline=(-1, -0.1))``."""


def _make_detector(name: Any) -> Any:
    def detector(ts: Any, **overrides: Any) -> Any:
        return test_response(ts, method="threshold", rule=name, rule_overrides=overrides, by=())

    detector.__name__ = name
    detector.__doc__ = f"Published rule '{name}' ({load_rules()[name]['source']})."
    return staticmethod(detector)


for _name in load_rules():
    setattr(detectors, _name, _make_detector(_name))


@verbose
def fit(
    data: Any,
    preset: Any = None,
    *,
    existence: Any = None,
    existence_kwargs: Any = None,
    **kwargs: Any,
) -> Any:
    """Fit with :func:`evoparam.models.fit` and attach the existence test of every channel.

    The test is attached to each channel's fit (``ResponseFit.existence``; ``existence_p`` and
    ``existence_significant`` in ``GroupFit.to_dataframe()``).

    ``data``: a TrialSet, an Evoked, MNE Epochs or Evoked, or an array [n_channels, n_times]
    with ``times``. With no other argument the result has existence results, components,
    their intervals and a methods paragraph (REQ-002).

    ``existence``: None (default) runs the test and, when it cannot run (too few trials for
    the method, a list of MNE Evoked), returns the fit with a warning and NaN existence
    columns; True runs it and raises instead; False skips it. With an Evoked (no trials) the
    test uses baseline windows of the average, a weaker null (warning). The test window is
    the fit window; its baseline is the fit's, else (-0.5, -0.05) s when the data start
    before -0.5 s, else the pre-stimulus samples. The seed of the test is ``random_state``,
    or the seed drawn and recorded by the fit (REQ-331).

    ``verbose`` (in ``kwargs``) sets the level of the ``evoparam`` logger for the whole call,
    the fit and the test (spec §9.8); at INFO the methods paragraph is logged at the end.
    ``n_jobs`` (in ``kwargs``) is used by the fit and by the test.
    """
    from evoparam.models import fit as fit_models
    from evoparam.models.api import _from_mne

    data = _from_mne(data)
    if isinstance(data, np.ndarray):
        # the same Evoked as evoparam.models.fit builds, so that the test sees its channels
        times = kwargs.pop("times", None)
        sfreq = kwargs.pop("sfreq", None)
        if times is None:
            raise SettingsError("times is required when data is an array.")
        values = np.atleast_2d(data)
        times = np.asarray(times, dtype=float)
        rate = sfreq or 1.0 / np.median(np.diff(times))
        data = Evoked(
            data=values, times=times, sfreq=rate, ch_names=[f"ch{i}" for i in range(len(values))]
        )
    group = fit_models(data, preset, **kwargs)
    if existence is False:
        _log_methods(group, kwargs.get("verbose"))
        return group
    if not isinstance(data, (TrialSet, Evoked)):
        if existence:
            raise SettingsError("existence=True needs a TrialSet or one Evoked.")
        _warn(
            "evoparam.fit: the existence test was not run (one test per condition needs "
            "the trials; pass a TrialSet with by=). The existence columns are NaN.",
            UserWarning,
            stacklevel=2,
        )
        _log_methods(group, kwargs.get("verbose"))
        return group
    settings = group.settings
    seed = kwargs.get("random_state")
    if seed is None:
        seed = settings.get("random_state")
    window = tuple(settings["fit_window"])
    if settings.get("baseline"):
        baseline = tuple(settings["baseline"])
    elif data.times[0] <= -0.5:
        baseline = (-0.5, -0.05)
    else:
        # a preset without a baseline (short epochs: ECAP, ERNA): the pre-stimulus samples
        baseline = (float(data.times[0]), min(0.0, window[0]))
    test_kwargs: dict[str, Any] = {
        "window": window,
        "baseline": baseline,
        "by": (),
        "random_state": seed,
    }
    # the channels and the workers of the fit
    test_kwargs.update({k: kwargs[k] for k in ("picks", "n_jobs") if kwargs.get(k) is not None})
    test_kwargs.update(existence_kwargs or {})
    try:
        test = test_response(data, **test_kwargs)
    except (DataError, SettingsError) as exc:
        if existence:
            raise
        _warn(
            f"evoparam.fit: the existence test could not run ({exc}). The existence "
            "columns are NaN; pass existence_kwargs to choose another method.",
            UserWarning,
            stacklevel=2,
        )
        _log_methods(group, kwargs.get("verbose"))
        return group
    group = attach_test(group, test)
    _log_methods(group, kwargs.get("verbose"))
    return group


def attach_test(group: Any, test: Any) -> Any:
    """Return the GroupFit with the existence test of each channel attached to its fit.

    The test is attached as ``ResponseFit.existence`` and recorded as a provenance step of the
    group and of every fit.
    """
    channels = test.settings["channels"]
    step = {k: v for k, v in test.settings.items() if k not in _PER_CHANNEL_SETTINGS}
    fits = []
    for fit_, ch in zip(group.fits, group.index["channel"], strict=False):
        i = channels.index(ch)
        one = dataclasses.replace(
            test,
            statistic=test.statistic[i : i + 1],
            p_value=test.p_value[i : i + 1],
            p_value_raw=test.p_value_raw[i : i + 1],
            significant=test.significant[i : i + 1],
            duration=test.duration[i : i + 1],
            onset=test.onset[i : i + 1],
            offset=test.offset[i : i + 1],
            snr_db=test.snr_db[i : i + 1],
            curve=None if test.curve is None else test.curve[i : i + 1],
            null_band=None if test.null_band is None else test.null_band[i : i + 1],
            status=test.status[i : i + 1],
            message=test.message[i : i + 1],
        )
        fits.append(
            dataclasses.replace(
                fit_,
                existence=one,
                provenance=fit_.provenance.with_step("evoparam.existence.test_response", step),
            )
        )
    return dataclasses.replace(
        group,
        fits=fits,
        provenance=group.provenance.with_step("evoparam.existence.test_response", step),
    )


# Settings of a test that hold one value per channel; the provenance step of the test
# attached to a fit keeps the settings that apply to every channel.
_PER_CHANNEL_SETTINGS = ("channels", "duration_ci", "onset_ci", "offset_ci", "groups", "critical")


def _log_methods(group: Any, verbose: Any) -> None:
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
