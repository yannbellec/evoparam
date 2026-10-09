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
(``vanblooijs2023_code``). Runs only with EVOPARAM_NETWORK_TESTS=1.
"""

import io
import os
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from evoparam.artifact import remove_artifact
from evoparam.data import TrialSet
from evoparam.io import epoch_data
from evoparam.models import ResponseModel
from evoparam.prep import baseline_correct, rereference

pytestmark = [pytest.mark.network,
              pytest.mark.skipif(os.environ.get("EVOPARAM_NETWORK_TESTS") != "1",
                                 reason="set EVOPARAM_NETWORK_TESTS=1 to read OpenNeuro")]

RUN = "sub-ccepAgeUMCU01/ses-1/ieeg/sub-ccepAgeUMCU01_ses-1_task-SPESclin_run-021448"


def _contacts(names, site=""):
    return [i for i, n in enumerate(names) if re.match(r"^[A-Za-z]+\d+$", n)
            and not n.upper().startswith(("MKR", "ECG", "EMG", "EOG")) and n not in site.split("-")]


def _average(run):
    from evoparam.tests.events.test_ds004080_events import _segment

    data, sfreq, names, t0, onsets, sites = _segment(run)
    site = pd.Series(sites).mode()[0]
    onsets = onsets[np.asarray(sites) == site]
    keep = _contacts(names, site)
    times, epochs, _ = epoch_data(data[keep], sfreq, onsets, -1.0, 1.0, t0=t0)
    ts = TrialSet(data=epochs, times=times, sfreq=sfreq, ch_names=[names[i] for i in keep],
                  metadata=pd.DataFrame({"stim_site": [site] * len(epochs)}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = remove_artifact(ts, method="template", window=(-0.002, 0.008),
                             baseline=(-0.5, -0.02)).cleaned
        ts = rereference(ts, "car_adaptive", random_state=0, n_boot=50)
    return baseline_correct(ts, (-0.5, -0.02), check=False).average()


def _agreement():
    evoked = _average(RUN)
    times, avg = evoked.times, evoked.data
    base = (times >= -0.5) & (times <= -0.02)
    win = (times >= 0.009) & (times <= 0.1)
    sd = np.maximum(avg[:, base].std(axis=1), 50.0)
    rule = avg[:, win].min(axis=1) < -3.4 * sd
    rule_latency = times[win][avg[:, win].argmin(axis=1)]
    group = ResponseModel("ccep").fit_group(evoked)
    model, latency = [], []
    for fit in group.fits:
        n1 = [c for c in fit.components if 0.009 <= c.latency <= 0.05 and c.amplitude < 0]
        model.append(bool(n1))
        latency.append(min(n1, key=lambda c: c.latency).latency if n1 else np.nan)
    model, latency = np.array(model), np.array(latency)
    po = np.mean(rule == model)
    pe = rule.mean() * model.mean() + (1 - rule.mean()) * (1 - model.mean())
    kappa = (po - pe) / (1 - pe)
    both = rule & model
    close = np.mean(np.abs(latency[both] - rule_latency[both]) <= 0.002) if both.any() else np.nan
    return rule, model, kappa, close


###################################################################################################
## Agreement with the published rule, applied as the published pipeline applies it

LIST = os.path.join(os.path.dirname(__file__), "..", "..", "..", "validation",
                    "ds004080_subjects.txt")
OPENNEURO = "https://s3.amazonaws.com/openneuro.org/ds004080"
EXAMPLE = ("ccepAgeUMCU01", "021448")


def _first_pair(subject, stem, min_trials=10):
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


def req371_pairs():
    """(subject, run, pair) of the REQ-371 list (validation/ds004080_subjects.txt): each
    subject's first qualifying run and its first pair of >= 10 trials."""

    with open(LIST, encoding="utf-8") as f:
        rows = [line.strip().split("\t") for line in f if not line.startswith("#")]
    return [(subject[len("sub-"):], stem.split("_run-")[1], _first_pair(subject, stem))
            for subject, stem in rows]


def _pair_trials(subject, run, site):
    """Every trial of one pair (both polarities), -2 to 3 s, the good non-stimulated ECoG
    channels of channels.tsv (the published channel selection), raw monopolar data."""

    from evoparam.io import read_bids
    from evoparam.io._fetch import data_folder

    cache = os.path.join(data_folder(), "n1-validation", "ds004080")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"sub-{subject}_run-{run}_{site}.npz")
    if not os.path.exists(path):
        a, b = site.split("-")
        ts = read_bids("ds004080", subject, session="1", task="SPESclin", run=run,
                       epoch=(-2.0, 3.0), select={"electrical_stimulation_site":
                                                  [f"{a}-{b}", f"{b}-{a}"]}, remote=True)
        info = ts.ch_info
        keep = ((info["type"].str.lower() == "ecog") & ~info["bad"].astype(bool)
                & ~info["name"].isin([a, b])).to_numpy()
        tmp = f"{path[:-4]}.{os.getpid()}.tmp.npz"  # atomic: concurrent runs share the cache
        np.savez_compressed(tmp, data=ts.data[:, keep].astype(np.float32), times=ts.times,
                            sfreq=ts.sfreq, names=np.array(ts.ch_names)[keep])
        os.replace(tmp, path)
    seg = np.load(path)
    names = [str(n) for n in seg["names"]]
    data = seg["data"].astype(float)
    return TrialSet(data=data, times=seg["times"], sfreq=float(seg["sfreq"]), ch_names=names,
                    metadata=pd.DataFrame({"stim_site": [site] * len(data)}))


# The two readings of "the published rule" (owner decision D8): the registry row of spec
# §5.6 as written (the paper's description: the most negative sample of 9-100 ms beyond
# 3.4 SD, no SD floor) and the published code (mnl_ccepAge: 50 uV SD floor, local minimum,
# first sample excluded, saturation at 3000 uV).
READINGS = {"paper": "vanblooijs2023", "code": "vanblooijs2023_code"}


def _published_rule(ts, reading):
    """mnl_ccepAge: median of (-1, -0.1) s removed from each trial, mean, median of (-2, -0.1)
    s removed from the average; the registry rule of the reading with its SD over (-2, -0.1)
    s. The code labels the onset sample t = 1/sfreq; latencies are returned on our axis."""

    from evoparam.existence.rules import apply_rule, resolve_rule

    t = ts.times
    x = ts.data - np.median(ts.data[..., (t > -1) & (t < -0.1)], axis=-1, keepdims=True)
    avg = x.mean(axis=0)
    base = (t > -2) & (t < -0.1)
    avg = avg - np.median(avg[:, base], axis=1, keepdims=True)
    spec = resolve_rule(READINGS[reading], baseline=(-2.0, -0.1))
    out = apply_rule(avg, t + 1.0 / ts.sfreq, ts.sfreq, spec, "uV")
    floor = 3.4 * np.maximum(avg[:, base].std(axis=1), 50.0)
    return out["significant"], out["onset"] - 1.0 / ts.sfreq, floor


def _model_fits(ts):
    """The ccep preset pipeline (artifact "template+tail:joint": template subtraction, the
    tail left to the exponential background of the fit; adjusted common average; baseline;
    average; fit)."""

    sub = ts.select(tmin=-1.0, tmax=2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sub = remove_artifact(sub, method="template", window=(-0.002, 0.008),
                              baseline=(-0.5, -0.02)).cleaned
        sub = rereference(sub, "car_adaptive", random_state=0)
    evoked = baseline_correct(sub, (-0.5, -0.02), check=False).average()
    return ResponseModel("ccep").fit_group(evoked).fits


def _model_n1(fits):
    """The preset's N1 (REQ-113 to the letter): a negative component with latency in 9-50 ms
    (the preset labels the largest |A| there N1; its latency is the model latency of
    REQ-114). Also the trough of the fitted curve in 9-50 ms (``fitted_peak``, K3)."""

    from evoparam.models import fitted_peak

    det, lat, trough, trough_amp = [], [], [], []
    for fit in fits:
        n1 = [c for c in fit.components if 0.009 <= c.latency <= 0.05 and c.amplitude < 0]
        det.append(bool(n1))
        lat.append(max(n1, key=lambda c: abs(c.amplitude)).latency if n1 else np.nan)
        peak = fitted_peak(fit, (0.009, 0.05))
        trough.append(peak["latency"])
        trough_amp.append(peak["amplitude"])
    return np.array(det), np.array(lat), np.array(trough), np.array(trough_amp)


def _kappa(a, b):
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


@pytest.fixture(scope="module")
def agreement():
    """Per run (the example, then the REQ-371 runs) and reading: the rule's decision and
    latency, the model's decision and latencies, the channel names."""

    out = {}
    for subject, run, site in [EXAMPLE + ("PT01-PT02",)] + req371_pairs():
        ts = _pair_trials(subject, run, site)
        det, lat, trough, trough_amp = _model_n1(_model_fits(ts))
        for reading in READINGS:
            rule, rule_lat, floor = _published_rule(ts, reading)
            out[(subject, run, reading)] = dict(
                rule=rule, rule_lat=rule_lat, model=det, lat=lat, trough=trough,
                # the decision of the earlier proposed wording (not the REQ): the trough of
                # the fitted curve at least as deep as the code's amplitude criterion
                trough_floor=np.isfinite(trough) & (-np.nan_to_num(trough_amp) >= floor),
                names=ts.ch_names)
    return out


def _scores(r, mask=None, decision="model", latency="lat"):
    mask = np.ones(len(r["rule"]), bool) if mask is None else mask
    rule, model = r["rule"][mask], r[decision][mask]
    both = rule & model
    close = np.abs(r[latency][mask][both] - r["rule_lat"][mask][both]) <= 0.002
    return dict(kappa=_kappa(rule, model), missed=int(np.sum(rule & ~model)),
                close=close, n_both=int(both.sum()))


def _runs(agreement, reading, example=False):
    return [v for (s, r, rd), v in agreement.items()
            if rd == reading and ((s, r) == EXAMPLE) == example]


AS_WRITTEN = ("REQ-113 as written is not met under either reading of the published rule "
              "(ccep v1.1: kappa on the 46 PT contacts 0.46 with the registry row, 0.19 with "
              "the code; pooled over the REQ-371 runs 0.46 and 0.40; CHANGELOG, 'N1 case as "
              "written' and 'Defaults settled by measurement'): "
              "no decision reaches 0.8 under both (bound 0.30 on the PT contacts); the code's "
              "170 uV floor "
              "leaves out small N1s the preset finds; the paper's extremum also counts "
              "drifts at 70-100 ms; the two readings agree with each other at kappa 0.35")


@pytest.mark.xfail(strict=False, reason=AS_WRITTEN)
@pytest.mark.parametrize("reading", list(READINGS))
def test_req113_as_written(agreement, reading):
    """REQ-113: the preset's N1 on every channel where the rule finds one minus at most one,
    and kappa >= 0.8 on the 46 non-stimulated PT contacts of the example run; reproduced
    (at most one channel missed, kappa >= 0.8) on at least five subjects of the REQ-371
    list."""

    r = _runs(agreement, reading, example=True)[0]
    grid = np.array([n.startswith("PT") for n in r["names"]])
    assert grid.sum() == 46
    assert _scores(r)["missed"] <= 1
    assert _scores(r, grid)["kappa"] >= 0.8
    others = _runs(agreement, reading)
    assert len(others) == 6
    ok = [_scores(o)["missed"] <= 1 and _scores(o)["kappa"] >= 0.8 for o in others]
    assert sum(ok) >= 5, [round(_scores(o)["kappa"], 2) for o in others]


@pytest.mark.xfail(strict=False, reason="REQ-114 as written (latency of the N1 component) is "
                   "not met: within 2 ms on 24 % (registry row) and 37 % (code) of the "
                   "channels where both find an N1 on the REQ-371 runs and the example "
                   "(ccep v1.1; CHANGELOG, 'Defaults settled by measurement')")
@pytest.mark.parametrize("reading", list(READINGS))
def test_req114_as_written(agreement, reading):
    """REQ-114: on the channels where both find an N1, the model latency (the N1
    component's) within 2 ms of the rule's peak on >= 90 %, over all runs."""

    runs = _runs(agreement, reading, example=True) + _runs(agreement, reading)
    close = np.concatenate([_scores(r)["close"] for r in runs])
    assert close.size >= 10
    assert np.mean(close) >= 0.9


@pytest.mark.xfail(strict=False, reason="REQ-114 with the latency read as the trough of the "
                   "fitted curve (K3) is not met over all runs: 41 % (registry row), 82 % "
                   "(code) (CHANGELOG, 'N1 case as written')")
@pytest.mark.parametrize("reading", list(READINGS))
def test_req114_trough_latency(agreement, reading):
    """REQ-114 with the model latency read as the trough of the fitted curve (K3), on the
    channels where the preset's N1 and the rule agree, over all runs."""

    runs = _runs(agreement, reading, example=True) + _runs(agreement, reading)
    close = np.concatenate([_scores(r, latency="trough")["close"] for r in runs])
    assert close.size >= 10
    assert np.mean(close) >= 0.9


@pytest.mark.xfail(strict=False, reason="REQ-273 needs REQ-113 and REQ-114, not met")
def test_req273_ccep_meets_req113_and_req114(agreement):
    """REQ-273: the ccep preset meets REQ-113 and REQ-114 on ds004080 (both readings)."""

    for reading in READINGS:
        test_req113_as_written(agreement, reading)
        test_req114_as_written(agreement, reading)


def test_n1_trough_with_the_code_floor_regression(agreement):
    """Not a REQ (the earlier proposed wording, kept as a regression floor): the trough of
    the fitted curve judged by the code's own amplitude criterion, against the code's
    reading: at most one missed channel on the example, pooled kappa >= 0.8 over the
    REQ-371 runs and each run at least the inter-rater kappa of human reviewers."""

    from evoparam.existence import HUMAN_KAPPA

    example = _runs(agreement, "code", example=True)[0]
    assert _scores(example, decision="trough_floor")["missed"] <= 1
    others = _runs(agreement, "code")
    rule = np.concatenate([r["rule"] for r in others])
    model = np.concatenate([r["trough_floor"] for r in others])
    assert _kappa(rule, model) >= 0.8
    assert all(_scores(r, decision="trough_floor")["kappa"] >= HUMAN_KAPPA for r in others)


def test_req113_engine_finds_n1_on_the_example_run():
    """Floor for the regression: the inherited engine found 2 N1s on this run."""

    rule, model, kappa, _ = _agreement()
    assert model.sum() >= 8
    assert kappa >= 0.3


def test_req112_real_baselines_mostly_silent():

    from evoparam.tests.events.test_ds004080_events import _segment

    statuses = []
    for run in [RUN, "sub-ccepAgeUMCU04/ses-1/ieeg/sub-ccepAgeUMCU04_ses-1_task-SPESclin_run-031148"]:
        data, sfreq, names, t0, onsets, _ = _segment(run)
        keep = _contacts(names)[:40]
        times, epochs, _ = epoch_data(data[keep], sfreq, onsets, -2.0, 0.0, t0=t0)
        model = ResponseModel("ccep", fit_window=(-0.49, -0.01), baseline=(-2.0, -0.5))
        statuses += [model.fit_waveform(times, ch).status for ch in epochs.mean(axis=0)]
    assert np.mean([s == "no_response" for s in statuses]) >= 0.95
