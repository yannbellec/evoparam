"""Recovery tail, matching pursuit and method comparison."""

import warnings

import numpy as np
import pytest

from evoparam.artifact import compare_methods, remove_artifact
from evoparam.errors import SettingsError

from .helpers import make_overlay

###################################################################################################
###################################################################################################

def test_tail_subtraction_removes_slow_recovery():

    noisy, truth = make_overlay(onset=0.03, art_tail=600.0, art_tau=0.006, n_channels=4)
    plain = remove_artifact(noisy, method="interpolate")
    tailed = remove_artifact(noisy, method="tail", tail="subtract", tail_length=0.02)
    seg = (noisy.times > 0.008) & (noisy.times < 0.025)
    err = lambda res: np.sqrt(np.mean((res.cleaned.data - truth.data)[..., seg] ** 2))
    assert err(tailed) < 0.3 * err(plain)
    taus = tailed.settings["_info"]["tail_tau1"]
    assert taus.shape == noisy.data.shape[:2]


def test_tail_joint_leaves_tail_for_the_model():

    noisy, _ = make_overlay()
    joint = remove_artifact(noisy, method="tail", tail="joint")
    plain = remove_artifact(noisy, method="interpolate")
    assert np.array_equal(joint.cleaned.data, plain.cleaned.data)
    with pytest.raises(SettingsError):
        remove_artifact(noisy, method="tail", tail="other")


def test_matching_pursuit_removes_decaying_artifact():

    noisy, truth = make_overlay(onset=0.03, art_amplitude=0.0, art_tail=800.0, art_tau=0.004,
                                n_channels=4)
    with pytest.warns(UserWarning, match="patent"):
        from evoparam.artifact import pursuit
        pursuit._WARNED = False
        result = remove_artifact(noisy, method="matching_pursuit")
    seg = (noisy.times > 0.0) & (noisy.times < 0.02)
    before = np.sum((noisy.data - truth.data)[..., seg] ** 2)
    after = np.sum((result.cleaned.data - truth.data)[..., seg] ** 2)
    assert after < 0.1 * before


def test_compare_methods_with_truth():

    noisy, truth = make_overlay()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = compare_methods(noisy, [{"method": "interpolate"},
                                        {"method": "template", "label": "tpl"}], truth=truth)
    assert table["label"].tolist() == ["interpolate", "tpl"]
    assert {"rrmse", "sir_improvement_db", "median_quality", "runtime_s"} <= set(table.columns)
    assert (table["sir_improvement_db"] > 10).all()
    without = compare_methods(noisy, [{"method": "none"}])
    assert "rrmse" not in without.columns


def test_unknown_method():

    noisy, _ = make_overlay()
    with pytest.raises(SettingsError):
        remove_artifact(noisy, method="magic")
