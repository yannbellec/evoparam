# SPDX-License-Identifier: Apache-2.0
"""The N1 case on OpenNeuro ds004080 (REQ-112 on real baselines, REQ-113, REQ-114, REQ-273).

REQ-112 and the regression floor use the 40 s segments cached by the events validation.
The agreement with the published rule uses every trial of the stimulation pair (both
polarities, as the published pipeline merges them), read by byte ranges and cached under
the evoparam data folder: run-021448 of sub-ccepAgeUMCU01 (the example of spec §1) and the
six subjects of validation/ds004080_subjects.txt (REQ-371, owner decision D14), each with
its listed run and the first stimulation pair of that run with at least 10 trials.

REQ-113 to the letter: the preset's own decision (a negative component with latency in
9-50 ms, the component the ccep preset labels N1), no threshold borrowed from the rule;
"the published rule" is read both ways (owner decision D8): the registry row of spec §5.6
(the paper's description, rule ``vanblooijs2023``) and the published code
(``vanblooijs2023_code``), each in its own frame (:data:`FRAMES`: the registry row on the
data's time axis with the registry's baseline, the code with its own baseline and time
axis). Runs only with EVOPARAM_NETWORK_TESTS=1.
"""

import io
import os
import re
import warnings
from typing import Any

import numpy as np
import pytest

from evoparam.artifact import remove_artifact
from evoparam.data import TrialSet
from evoparam.io import epoch_data
from evoparam.models import ResponseModel
from evoparam.prep import baseline_correct, rereference

pd = pytest.importorskip("pandas")

# Baseline (s) of the averages, and the N1 window of the detection rule.
BASELINE_START = -0.5
BASELINE_END = -0.02
N1_START = 0.009
RULE_END = 0.1
# N1 latency window end (s) of the CCEP preset.
N1_END = 0.05
# Largest latency difference (s) counted as agreeing with the rule (REQ-114).
LATENCY_TOLERANCE = 0.002
# mnl_ccepAge baselines (s): per trial (-1, -0.1), on the average (-2, -0.1).
AVERAGE_BASELINE_START = -2
RULE_BASELINE_END = -0.1
# Non-stimulated PT contacts of the example run, the kappa required (REQ-113), the other
# subjects of the REQ-371 list, and how many of them must reproduce it.
N_PT_CONTACTS = 46
MIN_KAPPA = 0.8
N_OTHER_RUNS = 6
MIN_REPRODUCED = 5
# Fewest channels where both find an N1, and the fraction within the tolerance (REQ-114).
MIN_BOTH = 10
MIN_CLOSE = 0.9
# Floor of the regression on the example run: N1s found and kappa.
MIN_N1_FOUND = 8
MIN_REGRESSION_KAPPA = 0.3
# Fraction of real baselines with no response (REQ-112).
MIN_SILENT = 0.95

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("EVOPARAM_NETWORK_TESTS") != "1",
        reason="set EVOPARAM_NETWORK_TESTS=1 to read OpenNeuro",
    ),
]

RUN = "sub-ccepAgeUMCU01/ses-1/ieeg/sub-ccepAgeUMCU01_ses-1_task-SPESclin_run-021448"


def _contacts(names: Any, site: Any = "") -> Any:
    return [
        i
        for i, n in enumerate(names)
        if re.match(r"^[A-Za-z]+\d+$", n)
        and not n.upper().startswith(("MKR", "ECG", "EMG", "EOG"))
        and n not in site.split("-")
    ]


def _average(run: Any) -> Any:
    from evoparam.tests.events.test_ds004080_events import _segment

    data, sfreq, names, t0, onsets, sites = _segment(run)
    site = pd.Series(sites).mode()[0]
    onsets = onsets[np.asarray(sites) == site]
    keep = _contacts(names, site)
    times, epochs, _ = epoch_data(data[keep], sfreq, onsets, -1.0, 1.0, t0=t0)
    ts = TrialSet(
        data=epochs,
        times=times,
        sfreq=sfreq,
        ch_names=[names[i] for i in keep],
        metadata=pd.DataFrame({"stim_site": [site] * len(epochs)}),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = remove_artifact(
            ts, method="template", window=(-0.002, 0.008), baseline=(-0.5, -0.02)
        ).cleaned
        ts = rereference(ts, "car_adaptive", random_state=0, n_boot=50)
    return baseline_correct(ts, (-0.5, -0.02), check=False).average()


def _agreement() -> Any:
    evoked = _average(RUN)
    times, avg = evoked.times, evoked.data
    base = (times >= BASELINE_START) & (times <= BASELINE_END)
    win = (times >= N1_START) & (times <= RULE_END)
    sd = np.maximum(avg[:, base].std(axis=1), 50.0)
    rule = avg[:, win].min(axis=1) < -3.4 * sd
    rule_latency = times[win][avg[:, win].argmin(axis=1)]
    group = ResponseModel("ccep").fit_group(evoked)
    model, latency = [], []
    for fit in group.fits:
        n1 = [c for c in fit.components if N1_START <= c.latency <= N1_END and c.amplitude < 0]
        model.append(bool(n1))
        latency.append(min(n1, key=lambda c: c.latency).latency if n1 else np.nan)
    model, latency = np.array(model), np.array(latency)
    po = np.mean(rule == model)
    pe = rule.mean() * model.mean() + (1 - rule.mean()) * (1 - model.mean())
    kappa = (po - pe) / (1 - pe)
    both = rule & model
    close = (
        np.mean(np.abs(latency[both] - rule_latency[both]) <= LATENCY_TOLERANCE)
        if both.any()
        else np.nan
    )
    return rule, model, kappa, close


###################################################################################################
## Agreement with the published rule, applied as the published pipeline applies it

LIST = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "validation", "ds004080_subjects.txt"
)
OPENNEURO = "https://s3.amazonaws.com/openneuro.org/ds004080"
EXAMPLE = ("ccepAgeUMCU01", "021448")


def _first_pair(subject: Any, stem: Any, min_trials: Any = 10) -> Any:
    """The first stimulation pair of a run, in event order, with at least ``min_trials``
    trials (the rule of the subject list); the events file is cached."""

    from evoparam.io._fetch import data_folder, fetch_bytes

    cache = os.path.join(data_folder(), "n1-validation", "ds004080", f"{stem}_events.tsv")
    if not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        raw = fetch_bytes(f"{OPENNEURO}/{subject}/ses-1/ieeg/{stem}_events.tsv")
        tmp = f"{cache}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            f.write(raw)
        os.replace(tmp, cache)
    with open(cache, "rb") as f:
        events = pd.read_csv(io.BytesIO(f.read()), sep="\t")
    sites = [s for s in events["electrical_stimulation_site"] if isinstance(s, str)]
    counts = pd.Series(sites).value_counts()
    return next(s for s in dict.fromkeys(sites) if counts[s] >= min_trials)


def req371_pairs() -> Any:
    """(subject, run, pair) of the REQ-371 list (validation/ds004080_subjects.txt): each
    subject's first qualifying run and its first pair of >= 10 trials."""

    with open(LIST, encoding="utf-8") as f:
        rows = [line.strip().split("\t") for line in f if not line.startswith("#")]
    return [
        (subject[len("sub-") :], stem.split("_run-")[1], _first_pair(subject, stem))
        for subject, stem in rows
    ]


def _pair_trials(subject: Any, run: Any, site: Any) -> Any:
    """Every trial of one pair (both polarities), -2 to 3 s, the good non-stimulated ECoG
    channels of channels.tsv (the published channel selection), raw monopolar data."""

    from evoparam.io import read_bids
    from evoparam.io._fetch import data_folder

    cache = os.path.join(data_folder(), "n1-validation", "ds004080")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"sub-{subject}_run-{run}_{site}.npz")
    if not os.path.exists(path):
        a, b = site.split("-")
        ts = read_bids(
            "ds004080",
            subject,
            session="1",
            task="SPESclin",
            run=run,
            epoch=(-2.0, 3.0),
            select={"electrical_stimulation_site": [f"{a}-{b}", f"{b}-{a}"]},
            remote=True,
        )
        info = ts.ch_info
        keep = (
            (info["type"].str.lower() == "ecog")
            & ~info["bad"].astype(bool)
            & ~info["name"].isin([a, b])
        ).to_numpy()
        tmp = f"{path[:-4]}.{os.getpid()}.tmp.npz"  # atomic: concurrent runs share the cache
        np.savez_compressed(
            tmp,
            data=ts.data[:, keep].astype(np.float32),
            times=ts.times,
            sfreq=ts.sfreq,
            names=np.array(ts.ch_names)[keep],
        )
        os.replace(tmp, path)
    seg = np.load(path)
    names = [str(n) for n in seg["names"]]
    data = seg["data"].astype(float)
    return TrialSet(
        data=data,
        times=seg["times"],
        sfreq=float(seg["sfreq"]),
        ch_names=names,
        metadata=pd.DataFrame({"stim_site": [site] * len(data)}),
    )


# The two readings of "the published rule" (owner decision D8): the registry row of spec
# §5.6 as written ("|x| > 3.4·SD_baseline with a negative peak, >= 1 sample", 9-100 ms: the
# deepest strict local minimum of the window beyond 3.4 SD, no SD floor, no selectivity)
# and the published code (mnl_ccepAge: 50 uV SD floor, local minimum with 20 uV
# selectivity, first sample excluded, saturation at 3000 uV). The tests are parametrised
# by these two; "extremum" (the window's most negative sample, peak or not: the registry
# row as read until 2026-10-13) is measured too.
READINGS = {"paper": "vanblooijs2023", "code": "vanblooijs2023_code"}
RULE_NAMES = {**READINGS, "extremum": "vanblooijs2023"}

# Owner decision (D8 revised, docs/spec/DECISIONS.md §5.8): the requirements are judged on
# the spec's registry row ("paper"); the published code's reading ("code") is measured and
# reported by the same tests but does not gate (no decision can satisfy both: the exact
# joint bound of test_d8_req113_cannot_hold_under_both_readings).
GATING = "paper"

# Where each reading reads the average: the SD baseline and the time axis. The registry
# row reads the data's own time axis (stimulation at t = 0) with the baseline the registry
# names for its unspecified one (the ER-detect default, -1 to -0.1 s, as REQ-135's test
# does). The published code takes its SD over (-2, -0.1) s and labels the stimulation
# sample t = 1/sfreq (MATLAB indexing), so its 9 ms is one sample earlier on the data's
# axis. Until 2026-10-13 the registry row was read with the code's baseline and axis too
# (CHANGELOG, "the comparator trail"). Third field: the peak operator (registry overrides).
# The row's "negative peak" is a peak, a local minimum: a sample at the window's edge on a
# monotone slope is not one (DECISIONS §5.8).
FRAMES = {
    "paper": ((-1.0, -0.1), 0, {"peak": "strict"}),
    "extremum": ((-1.0, -0.1), 0, {"peak": "extremum"}),
    "code": ((-2.0, -0.1), 1, {}),
}


def _frame(ts: Any, reading: Any) -> Any:
    """The reading's SD baseline and its time shift (s) relative to the data's axis."""

    baseline, samples, _ = FRAMES[reading]
    return baseline, samples / ts.sfreq


def _rule_spec(reading: Any) -> Any:
    """The complete registry rule of a reading, in its frame."""

    from evoparam.existence.rules import resolve_rule

    baseline, _, overrides = FRAMES[reading]
    return resolve_rule(RULE_NAMES[reading], baseline=baseline, **overrides)


def _rule_average(ts: Any) -> Any:
    """mnl_ccepAge: median of (-1, -0.1) s removed from each trial, mean, median of (-2, -0.1)
    s removed from the average."""

    t = ts.times
    x = ts.data - np.median(
        ts.data[..., (t > -1) & (t < RULE_BASELINE_END)], axis=-1, keepdims=True
    )
    avg = x.mean(axis=0)
    return avg - np.median(
        avg[:, (t > AVERAGE_BASELINE_START) & (t < RULE_BASELINE_END)], axis=1, keepdims=True
    )


def _published_rule(ts: Any, reading: Any) -> Any:
    """The average of :func:`_rule_average`; the registry rule of the reading, in the
    reading's frame (:data:`FRAMES`). Latencies are returned on the data's axis. Also the
    code's amplitude criterion, 3.4 x max(SD over (-2, -0.1) s, 50 uV)."""

    from evoparam.existence.rules import apply_rule

    t = ts.times
    avg = _rule_average(ts)
    base = (t > AVERAGE_BASELINE_START) & (t < RULE_BASELINE_END)
    baseline, shift = _frame(ts, reading)
    spec = _rule_spec(reading)
    out = apply_rule(avg, t + shift, ts.sfreq, spec, "uV")
    floor = 3.4 * np.maximum(avg[:, base].std(axis=1), 50.0)
    return out["significant"], out["onset"] - shift, floor


def _rule_peak_is_n1_trough(ts: Any, rule_lat: Any, window: Any = (0.009, 0.05)) -> Any:
    """Per channel: the rule's peak is an N1 trough of the average it reads, i.e. the
    deepest negative interior local minimum of :func:`_rule_average` in 9-50 ms (within one
    sample). Not so: the first samples of the window on the recovery tail, a slow
    deflection at 50-100 ms, a positive-valued minimum."""

    t = ts.times
    m = (t >= window[0]) & (t <= window[1])
    out = []
    for y, lat in zip(_rule_average(ts)[:, m], rule_lat, strict=True):
        inner = np.flatnonzero((y[1:-1] <= y[:-2]) & (y[1:-1] < y[2:])) + 1
        inner = inner[y[inner] < 0]
        out.append(
            bool(inner.size)
            and np.isfinite(lat)
            and abs(t[m][inner[np.argmin(y[inner])]] - lat) <= 1.0 / ts.sfreq + 1e-9
        )
    return np.array(out, bool)


def _no_n1_trough(ts: Any, reading: Any, window: Any = (0.009, 0.05)) -> Any:
    """Per channel: the average the rule reads (:func:`_rule_average`) has no negative
    interior local minimum in 9-50 ms on the rule's time axis (no N1 trough of any depth:
    the average is monotone there or stays above its baseline level)."""

    baseline, shift = _frame(ts, reading)
    t = ts.times + shift
    avg = _rule_average(ts)
    base = (t >= baseline[0]) & (t <= baseline[1])
    m = (t >= window[0]) & (t <= window[1])
    out = []
    for y, level in zip(avg[:, m], avg[:, base].mean(axis=1), strict=True):
        inner = np.flatnonzero((y[1:-1] <= y[:-2]) & (y[1:-1] < y[2:])) + 1
        out.append(not np.any(y[inner] < level))
    return np.array(out, bool)


def _preset_fits(ts: Any) -> Any:
    """The ccep preset pipeline (artifact "template+tail:joint": template subtraction, the
    tail left to the exponential background of the fit; adjusted common average; baseline;
    average; fit). Returns the fits, and the common average the re-referencing subtracted
    (baseline-corrected and averaged like the data: one curve, the same for every
    channel) with its times."""

    sub = ts.select(tmin=-1.0, tmax=2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sub = remove_artifact(
            sub, method="template", window=(-0.002, 0.008), baseline=(-0.5, -0.02)
        ).cleaned
        car = rereference(sub, "car_adaptive", random_state=0)
    evoked = baseline_correct(car, (-0.5, -0.02), check=False).average()
    recorded = baseline_correct(sub, (-0.5, -0.02), check=False).average()
    reference = np.median(recorded.data - evoked.data, axis=0)
    return ResponseModel("ccep").fit_group(evoked).fits, evoked.times, reference


def _model_fits(ts: Any) -> Any:
    """The fits of :func:`_preset_fits`."""

    return _preset_fits(ts)[0]


def _model_n1(fits: Any) -> Any:
    """The preset's N1 (REQ-113 to the letter): a negative component with latency in 9-50 ms
    (the preset labels the largest |A| there N1; its latency is the model latency of
    REQ-114). Also the trough of the fitted curve in 9-50 ms (``fitted_peak``, K3)."""

    from evoparam.models import fitted_peak

    det, lat, trough, trough_amp = [], [], [], []
    for fit in fits:
        n1 = [c for c in fit.components if N1_START <= c.latency <= N1_END and c.amplitude < 0]
        det.append(bool(n1))
        lat.append(max(n1, key=lambda c: abs(c.amplitude)).latency if n1 else np.nan)
        peak = fitted_peak(fit, (0.009, 0.05))
        trough.append(peak["latency"])
        trough_amp.append(peak["amplitude"])
    return np.array(det), np.array(lat), np.array(trough), np.array(trough_amp)


def _rule_peak_on_fitted_curve(ts: Any, fits: Any, reading: Any, reference: Any = None) -> Any:
    """The published rule's peak read on the fitted curve (a reading of "the model latency"
    of REQ-114): the reading's own peak operator (the most negative sample of 9-100 ms for
    the registry row; the code's largest local minimum with its 20 uV selectivity, its
    first sample excluded) applied to the curve of the ccep fit (components and
    background) from the rule's first sample on; before it, the rule's average, whose
    baseline gives the SD and the level the operator uses. NaN where the curve has no such
    peak.

    ``reference`` (times, curve): the common average the preset's re-referencing
    subtracted, added back to every fitted curve, so that the fit is read in the reference
    the rule reads (the recording's; the re-referencing is linear: data = re-referenced
    data + that one curve)."""

    from evoparam.existence.rules import apply_rule
    from evoparam.models import evaluate_fit

    t = ts.times
    curve = _rule_average(ts)
    baseline, shift = _frame(ts, reading)
    # from the rule's first sample on (in its frame)
    late = t >= 0.009 - shift - 0.01 / ts.sfreq
    common = np.interp(t[late], *reference) if reference is not None else 0.0
    for c, fit in enumerate(fits):
        curve[c, late] = (evaluate_fit(fit, t[late]) if fit.status == "ok" else 0.0) + common
    spec = _rule_spec(reading)
    out = apply_rule(curve, t + shift, ts.sfreq, spec, "uV")
    return out["onset"] - shift


def joint_kappa_bound(a: Any, b: Any) -> float:
    """The largest min(kappa(M, a), kappa(M, b)) over every decision M of the channels.

    Kappa depends only on the counts of M = 1 in the four cells of the table of a against
    b, so every combination of counts is searched (exact, not a heuristic)."""

    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n = len(a)
    cells = [int(np.sum(x)) for x in (a & b, a & ~b, ~a & b, ~a & ~b)]
    k11, k10, k01, k00 = np.meshgrid(*[np.arange(c + 1) for c in cells], indexing="ij")
    ones = (k11 + k10 + k01 + k00) / n
    agree_a = (k11 + k10 + (cells[2] - k01) + (cells[3] - k00)) / n
    agree_b = (k11 + k01 + (cells[1] - k10) + (cells[3] - k00)) / n
    out = []
    for agree, rate in ((agree_a, a.mean()), (agree_b, b.mean())):
        pe = ones * rate + (1 - ones) * (1 - rate)
        with np.errstate(divide="ignore", invalid="ignore"):
            out.append(np.where(pe < 1, (agree - pe) / (1 - pe), 1.0))
    return float(np.nanmax(np.minimum(*out)))


def _kappa(a: Any, b: Any) -> Any:
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


@pytest.fixture(scope="module")
def agreement() -> Any:
    """Per run (the example, then the REQ-371 runs) and reading: the rule's decision and
    latency, the model's decision and latencies, the channel names."""

    out = {}
    for subject, run, site in [EXAMPLE + ("PT01-PT02",)] + req371_pairs():
        ts = _pair_trials(subject, run, site)
        fits, ref_times, reference = _preset_fits(ts)
        det, lat, trough, trough_amp = _model_n1(fits)
        for reading in FRAMES:
            rule, rule_lat, floor = _published_rule(ts, reading)
            out[(subject, run, reading)] = {
                "rule": rule,
                "rule_lat": rule_lat,
                "model": det,
                "lat": lat,
                "trough": trough,
                "curve_peak": _rule_peak_on_fitted_curve(ts, fits, reading),
                "curve_peak_recording": _rule_peak_on_fitted_curve(
                    ts, fits, reading, (ref_times, reference)
                ),
                "n1_trough": _rule_peak_is_n1_trough(ts, rule_lat),
                "no_n1_trough": _no_n1_trough(ts, reading),
                # the decision of the earlier proposed wording (not the REQ): the trough of
                # the fitted curve at least as deep as the code's amplitude criterion
                "trough_floor": np.isfinite(trough) & (-np.nan_to_num(trough_amp) >= floor),
                "names": ts.ch_names,
            }
    return out


def _scores(r: Any, mask: Any = None, decision: Any = "model", latency: Any = "lat") -> Any:
    mask = np.ones(len(r["rule"]), bool) if mask is None else mask
    rule, model = r["rule"][mask], r[decision][mask]
    both = rule & model
    close = np.abs(r[latency][mask][both] - r["rule_lat"][mask][both]) <= LATENCY_TOLERANCE
    return {
        "kappa": _kappa(rule, model),
        "missed": int(np.sum(rule & ~model)),
        "close": close,
        "n_both": int(both.sum()),
    }


def _runs(agreement: Any, reading: Any, example: Any = False) -> Any:
    return [
        v
        for (s, r, rd), v in agreement.items()
        if rd == reading and ((s, r) == EXAMPLE) == example
    ]


AS_WRITTEN = (
    "REQ-113 is judged on the registry row (D8 revised, DECISIONS §5.8), read as written: "
    "the deepest negative peak (a strict local minimum) of 9-100 ms beyond 3.4 SD of the "
    "baseline (-1 to -0.1 s, the data's time axis); the code's reading is reported. "
    "REQ-113 as written is not met under either reading of the published rule "
    "(ccep v1.1: kappa on the 46 PT contacts 0.01 with the registry row, 0.19 with the "
    "code; pooled over the REQ-371 runs 0.44 and 0.40, per run 0.37, 0.60, 0.09, 0.35, "
    "0.42, 0.32 with the registry row; read as the window's most negative sample, peak or "
    "not, 0.24 and 0.44; in the code's frame, until 2026-10-13, 0.46 and 0.46; CHANGELOG, "
    "'The comparator trail'). Of the registry row's 11 PT detections 10 peak after 50 ms, "
    "outside the N1 window of 9-50 ms that REQ-113 names; over the REQ-371 runs the 54 "
    "channels the preset misses are 24 such late peaks, 11 peaks that the preset's input "
    "(the adjusted common average) does not have and 19 that it has without an N1 "
    "component; of the 33 N1s the rule does not find, 9 are peaks of the preset's input "
    "and 24 stay below 3.4 SD there. The kappa of 1.00 recorded on 2026-10-08 was the "
    "code's reading against a model decision that borrowed the code's amplitude "
    "criterion, not the preset's N1 (test_n1_trough_with_the_code_floor_regression keeps "
    "it). No decision reaches 0.8 under both readings (exact bound 0.378 on the PT "
    "contacts, 0.387 to 0.783 per REQ-371 run: "
    "test_d8_req113_cannot_hold_under_both_readings); the code's 170 uV floor leaves out "
    "small N1s the preset finds. Per reading, the variants measured on 2026-10-13 "
    "(recording reference, delta 6, max_components 8, fit window to 0.3 s, double "
    "exponential, constant or linear background, plain common average, the N1 gated by "
    "the existence test) reach at most 0.51 (registry row) and 0.56 (code) pooled "
    "(CHANGELOG, 'N1 case: the two readings')"
)


@pytest.mark.xfail(strict=False, reason=AS_WRITTEN)
@pytest.mark.parametrize("reading", list(READINGS))
def test_req113_as_written(agreement: Any, reading: Any) -> None:
    """REQ-113: the preset's N1 on every channel where the rule finds one minus at most one,
    and kappa >= 0.8 on the 46 non-stimulated PT contacts of the example run; reproduced
    (at most one channel missed, kappa >= 0.8) on at least five subjects of the REQ-371
    list."""

    r = _runs(agreement, reading, example=True)[0]
    grid = np.array([n.startswith("PT") for n in r["names"]])
    assert grid.sum() == N_PT_CONTACTS
    assert _scores(r)["missed"] <= 1
    assert _scores(r, grid)["kappa"] >= MIN_KAPPA
    others = _runs(agreement, reading)
    assert len(others) == N_OTHER_RUNS
    ok = [_scores(o)["missed"] <= 1 and _scores(o)["kappa"] >= MIN_KAPPA for o in others]
    assert sum(ok) >= MIN_REPRODUCED, [round(_scores(o)["kappa"], 2) for o in others]


# Why REQ-114 is not met (CHANGELOG, "N1 latency agreement"): on the REQ runs the
# rule's peak is an N1 trough of the average it reads (the deepest interior local minimum
# in 9-50 ms) on 44 of the 57 channels where the registry row and the preset both find an
# N1 (77 %) and on 24 of 30 with the code (80 %). The others peak after 50 ms, outside the
# N1 window (registry row 13; code 2, with 3 at the window start and one positive-valued
# minimum).
RULE_PEAKS = (
    "the rule's peak is an N1 trough of its average on 77 % (registry row) and 80 % "
    "(code) of these channels; elsewhere it is a deflection at 51-100 ms, after the N1 "
    "window (and, for the code, the window start on the recovery tail)"
)


def _close(agreement: Any, reading: Any, latency: Any, n1_trough: Any = False) -> Any:
    runs = _runs(agreement, reading, example=True) + _runs(agreement, reading)
    close = []
    for r in runs:
        mask = r["n1_trough"] if n1_trough else None
        close.append(_scores(r, mask, latency=latency)["close"])
    return np.concatenate(close)


@pytest.mark.xfail(
    strict=False,
    reason="REQ-114 is judged on the registry row (D8 revised, DECISIONS §5.8). "
    "REQ-114 as written (latency of the N1 component) is "
    "not met: within 2 ms on 39 % (registry row, 22 of 57) and 37 % (code, 11 "
    "of 30) of the channels where both find an N1 on the REQ-371 runs and the "
    "example; naming the N1 by the trough it makes (the negative component that "
    "contributes most at the fitted trough) gives 40 % and 40 %; "
    + RULE_PEAKS
    + "; read as the rule's peak on the fitted curve, 72 % and "
    "90.0 %, and in the rule's own reference 82 % and 90.0 % (met with the code only: "
    "test_req114_rule_peak_on_the_fitted_curve_in_the_rules_reference)",
)
@pytest.mark.parametrize("reading", list(READINGS))
def test_req114_as_written(agreement: Any, reading: Any) -> None:
    """REQ-114: on the channels where both find an N1, the model latency (the N1
    component's) within 2 ms of the rule's peak on >= 90 %, over all runs."""

    close = _close(agreement, reading, "lat")
    assert close.size >= MIN_BOTH
    assert np.mean(close) >= MIN_CLOSE, (np.mean(close), close.size)


@pytest.mark.xfail(
    strict=False,
    reason="REQ-114 with the latency read as the trough of the "
    "fitted curve (K3) is not met over all runs: 61 % (registry row, 35 of 57), "
    "80 % (code, 24 of 30); " + RULE_PEAKS,
)
@pytest.mark.parametrize("reading", list(READINGS))
def test_req114_trough_latency(agreement: Any, reading: Any) -> None:
    """REQ-114 with the model latency read as the trough of the fitted curve (K3), on the
    channels where the preset's N1 and the rule agree, over all runs."""

    close = _close(agreement, reading, "trough")
    assert close.size >= MIN_BOTH
    assert np.mean(close) >= MIN_CLOSE, (np.mean(close), close.size)


@pytest.mark.xfail(
    strict=False,
    reason="REQ-114 with the model latency read as the published rule's peak on the "
    "fitted curve (the reading's own operator on the ccep fit) is not met with the "
    "registry row: 72 % (41 of 57); with the code 90.0 % (27 of 30), at the threshold "
    "(development runs 66 % and 86 %). Read in the rule's reference: 82 % and 90.0 % "
    "(test_req114_rule_peak_on_the_fitted_curve_in_the_rules_reference)",
)
@pytest.mark.parametrize("reading", list(READINGS))
def test_req114_rule_peak_on_the_fitted_curve(agreement: Any, reading: Any) -> None:
    """REQ-114 with "the model latency" read as the published rule's peak measured on the
    fitted curve instead of the noisy average (:func:`_rule_peak_on_fitted_curve`): the same
    operator on both sides, so the window-start peaks on the recovery tail and the late
    deflections of 50-100 ms the rule counts are compared with the model's curve there."""

    close = _close(agreement, reading, "curve_peak")
    assert close.size >= MIN_BOTH
    assert np.mean(close) >= MIN_CLOSE, (np.mean(close), close.size)


@pytest.mark.parametrize(
    "reading",
    [
        pytest.param(
            "paper",
            marks=pytest.mark.xfail(
                strict=False,
                reason="REQ-114 read as the rule's peak on the fitted curve in the rule's "
                "reference is not met with the registry row read as written (a negative "
                "peak): 82 % (47 of 57) on the REQ runs, 84 % (27 of 32) on the "
                "development runs. 7 of the 10 misses are the rule's peaks at 87-100 ms, "
                "small local minima of the noisy average on a slope that the fitted curve "
                "follows without a local minimum. With the window's most negative sample "
                "(peak or not) it is 93 % (test_req114_extremum_reading_in_the_rules_"
                "reference)",
            ),
        ),
        "code",
    ],
)
def test_req114_rule_peak_on_the_fitted_curve_in_the_rules_reference(
    agreement: Any, reading: Any
) -> None:
    """REQ-114 with "the model latency" read as the published rule's peak on the fitted
    curve, in the reference the rule reads. The preset fits the adjusted common average and
    the published pipeline reads the recording reference; the re-referencing subtracts one
    curve from every channel (the common average of its members), which is added back to
    every fitted curve (:func:`_rule_peak_on_fitted_curve`). Nothing is fitted again and
    the preset's results are unchanged: the comparison is made in one reference.
    Measured: 82 % (47 of 57, registry row) and 90.0 % (27 of 30, code) on the REQ runs;
    84 % (27 of 32) and 95 % (20 of 21) on the development runs."""

    close = _close(agreement, reading, "curve_peak_recording")
    assert close.size >= MIN_BOTH
    assert np.mean(close) >= MIN_CLOSE, (np.mean(close), close.size)


def test_req114_extremum_reading_in_the_rules_reference(agreement: Any) -> None:
    """Measured, not gating: the registry row read as the window's most negative sample
    (peak or not; the reading used until 2026-10-13), REQ-114 with the rule's peak on the
    fitted curve in the rule's reference: 93 % (64 of 69) on the REQ runs, 89 % (32 of 36)
    on the development runs. The window-edge samples it takes on a slope are read on the
    fitted curve's slope as well."""

    close = _close(agreement, "extremum", "curve_peak_recording")
    assert close.size >= MIN_BOTH
    assert np.mean(close) >= MIN_CLOSE, (np.mean(close), close.size)


@pytest.mark.xfail(
    strict=False,
    reason="REQ-113 with the registry row read as the window's most negative sample (peak "
    "or not; measured, not gating): kappa 0.24 on the 46 PT contacts, 0.44 pooled over "
    "the REQ-371 runs (per run 0.44, 0.58, 0.25, 0.33, 0.47, 0.31); 25 of its detections "
    "missed on the example run",
)
def test_req113_extremum_reading(agreement: Any) -> None:
    """REQ-113 as in :func:`test_req113_as_written`, the registry row read as the window's
    most negative sample (FRAMES "extremum")."""

    test_req113_as_written(agreement, "extremum")


def test_d8_req113_cannot_hold_under_both_readings(agreement: Any) -> None:
    """Not a REQ (owner decision D8, measured): REQ-113 asks one decision (the preset's N1)
    with kappa >= 0.8 against the published rule, and D8 asks it under both readings of
    the rule. For every possible decision of the channels, the smaller of the two kappas
    is at most :func:`joint_kappa_bound`: 0.378 on the 46 PT contacts of the example run,
    0.387 to 0.783 on the six REQ-371 runs (0.655 pooled). No detector can meet REQ-113
    under both readings on these data; the readings agree with each other at kappa 0.13 on
    the PT contacts (11 and 1 N1s)."""

    paper = _runs(agreement, "paper", example=True)[0]
    code = _runs(agreement, "code", example=True)[0]
    grid = np.array([n.startswith("PT") for n in paper["names"]])
    assert joint_kappa_bound(paper["rule"][grid], code["rule"][grid]) < MIN_KAPPA
    bounds = [
        joint_kappa_bound(p["rule"], c["rule"])
        for p, c in zip(_runs(agreement, "paper"), _runs(agreement, "code"), strict=True)
    ]
    assert len(bounds) == N_OTHER_RUNS
    assert sum(b >= MIN_KAPPA for b in bounds) < MIN_REPRODUCED, bounds


def test_req113_registry_row_counts_channels_without_an_n1_trough(agreement: Any) -> None:
    """Not a REQ (measured, under the gating reading, the registry row): REQ-113 asks the
    preset's N1 (a negative component with latency 9-50 ms) on every channel where the row
    detects one, minus at most one. On the example run 9 of the row's 11 detections on the
    46 PT contacts are channels whose own average has no negative local minimum at all in
    9-50 ms: the row's peak is a deflection after 50 ms. On the REQ-371 runs there are 0,
    1, 1, 2, 5 and 12 such channels. A decision that names an N1 only where the data
    have
    an N1 trough therefore misses more than one channel on the example and on three of the
    six runs, so it can meet REQ-113 on at most three of them (five are asked)."""

    example = _runs(agreement, "paper", example=True)[0]
    grid = np.array([n.startswith("PT") for n in example["names"]])
    assert int(np.sum(example["rule"] & example["no_n1_trough"] & grid)) > 1
    counts = [int(np.sum(r["rule"] & r["no_n1_trough"])) for r in _runs(agreement, "paper")]
    assert len(counts) == N_OTHER_RUNS
    assert sum(c <= 1 for c in counts) < MIN_REPRODUCED, counts


def test_n1_trough_where_the_rule_peak_is_an_n1_trough_regression(agreement: Any) -> None:
    """Not a REQ (a regression floor for the latency of REQ-114): on the channels where both
    find an N1 and the rule's peak is an N1 trough of its average
    (:func:`_rule_peak_is_n1_trough`), the trough of the fitted curve is within 2 ms of the
    rule's peak on >= 90 % with the code's reading (24 of 24 on the REQ runs; the N1
    component's own latency 11 of 24: the N1 is fitted by overlapping components and the
    largest |A|, which names it, is often a broad one). With the registry row 31 of 38
    (component 18 of 38): the rule reads the monopolar average, the preset the adjusted
    common average, which moves shallow troughs (sub-ccepAgeUMCU03: the registry row's peak
    is at 32.7 ms on 35 of the 70 channels, a common-mode deflection the common average
    removes)."""

    close = _close(agreement, "code", "trough", n1_trough=True)
    assert close.size >= MIN_BOTH
    assert np.mean(close) >= MIN_CLOSE, (np.mean(close), close.size)


@pytest.mark.xfail(
    strict=False,
    reason="REQ-273 (judged on the registry row, D8 revised; the code's reading is also "
    "checked here and reported) needs REQ-113 and REQ-114 as written: REQ-113 is not met "
    "under either "
    "reading (and no decision can meet it under both, D8); REQ-114 is not met with the "
    "registry row in any reading of the model latency (at most 82 %: the rule's peak on "
    "the fitted curve in the rule's reference; 90.0 % with the code)",
)
def test_req273_ccep_meets_req113_and_req114(agreement: Any) -> None:
    """REQ-273: the ccep preset meets REQ-113 and REQ-114 on ds004080 (both readings)."""

    for reading in READINGS:
        test_req113_as_written(agreement, reading)
        test_req114_as_written(agreement, reading)


def test_n1_trough_with_the_code_floor_regression(agreement: Any) -> None:
    """Not a REQ (the earlier proposed wording, kept as a regression floor): the trough of
    the fitted curve judged by the code's own amplitude criterion, against the code's
    reading: at most one missed channel on the example, pooled kappa >= 0.8 over the
    REQ-371 runs and each run at least the inter-rater kappa of human reviewers.

    This is the comparison that gave kappa 1.00 on the PT contacts and 0.91 pooled on
    2026-10-08 (73788a2), recorded then as REQ-113 met; it still gives 1.00 and 0.91. It is
    not REQ-113: the rule is the code's reading (50 uV SD floor, local minimum), not the
    registry row of §5.6, and finds 1 N1 on the 46 PT contacts; the decision is not the
    preset's N1 but the fitted trough judged by the rule's own amplitude criterion on the
    rule's own average (CHANGELOG, 'The comparator trail')."""

    from evoparam.existence import HUMAN_KAPPA

    example = _runs(agreement, "code", example=True)[0]
    assert _scores(example, decision="trough_floor")["missed"] <= 1
    others = _runs(agreement, "code")
    rule = np.concatenate([r["rule"] for r in others])
    model = np.concatenate([r["trough_floor"] for r in others])
    assert _kappa(rule, model) >= MIN_KAPPA
    assert all(_scores(r, decision="trough_floor")["kappa"] >= HUMAN_KAPPA for r in others)


def test_req113_engine_finds_n1_on_the_example_run() -> None:
    """Floor for the regression: the inherited engine found 2 N1s on this run."""

    rule, model, kappa, _ = _agreement()
    assert model.sum() >= MIN_N1_FOUND
    assert kappa >= MIN_REGRESSION_KAPPA


@pytest.mark.parametrize("preset", ["ccep", None])
def test_req112_real_baselines_mostly_silent(preset: Any) -> None:
    """Pre-stimulus windows of the two REQ runs: >= 95 % without a component, with the
    ``ccep`` preset and with the package defaults (both at the fixed delta 10; DECISIONS §4
    "REQ-112 on real baselines")."""

    from evoparam.tests.events.test_ds004080_events import _segment

    statuses = []
    for run in [
        RUN,
        "sub-ccepAgeUMCU04/ses-1/ieeg/sub-ccepAgeUMCU04_ses-1_task-SPESclin_run-031148",
    ]:
        data, sfreq, names, t0, onsets, _ = _segment(run)
        keep = _contacts(names)[:40]
        times, epochs, _ = epoch_data(data[keep], sfreq, onsets, -2.0, 0.0, t0=t0)
        model = ResponseModel(preset, fit_window=(-0.49, -0.01), baseline=(-2.0, -0.5))
        statuses += [model.fit_waveform(times, ch).status for ch in epochs.mean(axis=0)]
    assert np.mean([s == "no_response" for s in statuses]) >= MIN_SILENT
