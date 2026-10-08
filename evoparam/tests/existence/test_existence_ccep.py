"""Agreement with the field on OpenNeuro ds004080 (REQ-135); runs only with
EVOPARAM_NETWORK_TESTS=1 (uses the segments cached by the events validation)."""

import os
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from evoparam.artifact import remove_artifact
from evoparam.data import TrialSet
from evoparam.existence import test_response as run_test
from evoparam.io import epoch_data
from evoparam.prep import baseline_correct

pytestmark = [pytest.mark.network,
              pytest.mark.skipif(os.environ.get("EVOPARAM_NETWORK_TESTS") != "1",
                                 reason="set EVOPARAM_NETWORK_TESTS=1 to read OpenNeuro")]

RUNS = [
    "sub-ccepAgeUMCU01/ses-1/ieeg/sub-ccepAgeUMCU01_ses-1_task-SPESclin_run-021448",
    "sub-ccepAgeUMCU07/ses-1/ieeg/sub-ccepAgeUMCU07_ses-1_task-SPESclin_run-021752",
    "sub-ccepAgeUMCU19/ses-1/ieeg/sub-ccepAgeUMCU19_ses-1_task-SPESclin_run-040955",
    "sub-ccepAgeUMCU28/ses-1/ieeg/sub-ccepAgeUMCU28_ses-1_task-SPESclin_run-021305",
    "sub-ccepAgeUMCU40/ses-1/ieeg/sub-ccepAgeUMCU40_ses-1_task-SPESclin_run-031553",
    "sub-ccepAgeUMCU49/ses-1/ieeg/sub-ccepAgeUMCU49_ses-1_task-SPESclin_run-031049",
]
BASELINE = (-1.0, -0.1)


def _trials(run):
    from evoparam.tests.events.test_ds004080_events import _segment

    data, sfreq, names, t0, onsets, sites = _segment(run)
    site = pd.Series(sites).mode()[0]
    onsets = onsets[np.asarray(sites) == site]
    keep = [i for i, n in enumerate(names) if re.match(r"^[A-Za-z]+\d+$", n)
            and not n.upper().startswith(("MKR", "ECG")) and n not in site.split("-")]
    times, epochs, _ = epoch_data(data[keep], sfreq, onsets, -1.0, 1.0, t0=t0)
    ts = TrialSet(data=epochs, times=times, sfreq=sfreq, ch_names=[names[i] for i in keep],
                  metadata=pd.DataFrame({"stim_site": [site] * len(epochs)}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = remove_artifact(ts, method="template", window=(-0.002, 0.008),
                             baseline=BASELINE).cleaned
    return baseline_correct(ts, BASELINE, check=False)


def test_req135_permutation_agrees_with_vanblooijs2023():

    assert len({r.split("/")[0] for r in RUNS}) >= 5
    a_all, b_all = [], []
    for run in RUNS:
        ts = _trials(run)
        a = run_test(ts, window=(0.009, 0.1), baseline=BASELINE, polarity="negative",
                     n_permutations=1000, n_boot=0, by=(), random_state=0).significant
        b = run_test(ts, method="threshold", rule="vanblooijs2023",
                     rule_overrides={"baseline": BASELINE}, by=()).significant
        a_all += list(a)
        b_all += list(b)
    a, b = np.array(a_all), np.array(b_all)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    assert (po - pe) / (1 - pe) >= 0.7
