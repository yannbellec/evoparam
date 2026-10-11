# SPDX-License-Identifier: Apache-2.0
"""API, groups, text (REQ-140), integration with evoparam.fit, edge cases."""

import numpy as np
import pandas as pd
import pytest

import evoparam
from evoparam.artifact import apply_mask
from evoparam.errors import DataError, SettingsError
from evoparam.existence import compare_detectors, methods_text, response_duration
from evoparam.existence import test_response as run_test

from .helpers import TIMES, gaussian, noise_trials, trialset

###################################################################################################
###################################################################################################

def _responsive(seed=0, n=20, channels=4):
    rng = np.random.default_rng(seed)
    data = noise_trials(rng, n, channels)
    data[:, 0] += 3 * gaussian(0.03, 0.01)
    return data


def test_groups_by_site_and_methods_text():

    data = _responsive()
    meta = pd.DataFrame({"stim_site": ["A-B"] * 10 + ["C-D"] * 10})
    test = run_test(trialset(data, meta), n_permutations=200, n_boot=0, random_state=0)
    assert test.p_value.shape == (4, 2)
    assert test.settings["groups"] == ["A-B", "C-D"]
    text = methods_text(test)
    assert "sign-flip permutation test" in text and "fdr_bh" in text and "seed 0" in text
    table = response_duration(trialset(data), n_permutations=200, n_boot=20, by=(),
                              random_state=0)
    assert table["significant"].tolist() == [True, False, False, False]


def test_evoked_without_trials_warns():

    ts = trialset(_responsive())
    with pytest.warns(UserWarning, match="weaker"):
        test = run_test(ts.average(), n_permutations=200, random_state=0)
    assert test.significant[0]


def test_edge_cases():

    with pytest.raises(DataError, match="at least 5"):
        run_test(trialset(_responsive(n=4)), by=())
    ts = trialset(_responsive())
    mask = np.zeros(ts.data.shape, bool)
    mask[..., (TIMES >= 0.0) & (TIMES <= 0.02)] = True
    with pytest.raises(SettingsError, match="artifact"):
        run_test(apply_mask(ts, mask), window=(0.0, 0.1), by=())
    run_test(apply_mask(ts, mask), window=(0.0, 0.1), by=(), allow_artifact_overlap=True,
                  n_permutations=100, n_boot=0)
    with pytest.raises(SettingsError):
        run_test(ts, method="magic")


def test_exact_enumeration_with_few_trials():

    test = run_test(trialset(_responsive(n=6)), n_permutations=1000, n_boot=0, by=(),
                         random_state=0)
    assert test.settings["exact"]


def test_compare_detectors_caption():

    table = compare_detectors(trialset(_responsive()),
                              rules=("baseline_permutation", "keller2014", "projection"),
                              n_permutations=200, random_state=0)
    assert "0.60" in table.attrs["caption"]
    assert table.attrs["kappa"].shape == (3, 3)


def test_fit_with_existence():

    from evoparam.data import TrialSet
    rng = np.random.default_rng(1)
    t = np.arange(-0.5, 1.0, 1 / 1000.)
    sig = -100 * np.exp(-0.5 * ((t - 0.025) / 0.005) ** 2)
    data = rng.normal(0, 10, (15, 2, len(t)))
    data[:, 0] += sig
    ts = TrialSet(data=data, times=t, sfreq=1000., ch_names=["a", "b"])
    group = evoparam.fit(ts, "ccep", existence=True,
                         existence_kwargs={"n_permutations": 200, "n_boot": 0})
    table = group.to_dataframe()
    assert {"existence_p", "existence_significant"} <= set(table.columns)
    assert group.get("a").existence.significant[0]
    assert not group.get("b").existence.significant[0]


def test_statistic_over_time_and_null_band():
    """Spec §5.13 (plot_existence): the test keeps its statistic over time and the band it
    must leave, consistent with the decision, the onset and the offset."""

    data = _responsive()
    test = run_test(trialset(data), n_permutations=200, n_boot=0, by=(), random_state=0)
    inside = (TIMES >= test.window[0]) & (TIMES <= test.window[1])
    assert test.curve.shape == (4, inside.sum()) and test.null_band.shape == (4, inside.sum(), 2)
    assert np.allclose(test.curve_times, TIMES[inside])
    # max_t: the pointwise t of the trials; the band is +/- the critical value of the null
    # maximum, so a channel is significant when its curve leaves the band
    x = data[:, 0][:, inside]
    t = x.mean(axis=0) / (x.std(axis=0, ddof=1) / np.sqrt(len(x)))
    assert np.allclose(test.curve[0], t)
    assert np.allclose(test.null_band[..., 0], -test.null_band[..., 1])
    outside = np.abs(test.curve) > test.null_band[..., 1]
    assert outside[0].any() and test.significant[0]
    hits = TIMES[inside][outside[0]]
    assert test.onset[0] == hits[0] and test.offset[0] == hits[-1]
    # one-sided tests look at one side only; the band of a split test follows the groups
    neg = run_test(trialset(data), n_permutations=200, n_boot=0, by=(), polarity="negative",
                   random_state=0)
    assert np.isnan(neg.null_band[..., 1]).all() and (neg.null_band[..., 0] < 0).all()
    meta = pd.DataFrame({"stim_site": ["A-B"] * 10 + ["C-D"] * 10})
    split = run_test(trialset(data, meta), n_permutations=200, n_boot=0, random_state=0)
    assert split.curve.shape == (4, 2, inside.sum())
    # the attached per-channel tests keep their own curve
    group = evoparam.fit(trialset(data), "ccep", existence_kwargs={"n_permutations": 100})
    one = group.fits[0].existence
    assert one.curve.shape == (1, inside.sum()) and one.null_band.shape == (1, inside.sum(), 2)
    # the other methods: the projection profile, the rule's z with its k
    proj = run_test(trialset(data), method="projection", by=(), random_state=0)
    assert proj.curve.shape == (4, inside.sum()) and np.isnan(proj.null_band).all()
    rule = run_test(trialset(data), method="threshold", rule="hays2021",
                    rule_overrides={"baseline": (-0.5, -0.05)}, by=())
    assert rule.curve.shape == (4, len(rule.curve_times))
    assert rule.curve_times[0] >= 0.010 and rule.curve_times[-1] <= 0.050
    # a negative-peak rule: z below -k (here 6) in the window
    assert (rule.null_band[..., 0] == -6).all() and np.isnan(rule.null_band[..., 1]).all()
    assert rule.significant[0] == (rule.curve[0] < -6).any()
    # a saved test keeps them
    import tempfile

    from evoparam.data import ResponseTest

    with tempfile.TemporaryDirectory() as folder:
        loaded = ResponseTest.load(test.save(folder + "/t.evoparam"))
    assert np.allclose(loaded.curve, test.curve) and np.allclose(loaded.null_band,
                                                                 test.null_band)
