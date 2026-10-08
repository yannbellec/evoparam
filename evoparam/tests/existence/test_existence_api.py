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
