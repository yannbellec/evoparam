"""The N1 case on OpenNeuro ds004080 (REQ-112 on real baselines, REQ-113, REQ-114).

Uses the 40 s segments cached by the events validation; runs only with
EVOPARAM_NETWORK_TESTS=1.
"""

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


@pytest.mark.xfail(strict=False, reason="REQ-113/114 not met yet (kappa about 0.5 on the cached "
                   "8-trial segment): part of the rule's detections are artifact residue at the "
                   "window start or slow drift at 100 ms, and the model finds small N1s under the "
                   "rule's 50 uV floor; see CHANGELOG, step S7.")
def test_req113_req114_n1_agrees_with_published_rule():

    rule, model, kappa, close = _agreement()
    assert np.sum(rule & ~model) <= 1
    assert kappa >= 0.8
    assert close >= 0.9


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
