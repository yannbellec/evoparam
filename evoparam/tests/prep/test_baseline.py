# SPDX-License-Identifier: Apache-2.0
"""Baseline correction and its checks (REQ-080)."""

import warnings

import numpy as np
import pytest

from evoparam.artifact import apply_mask
from evoparam.errors import DataError, SettingsError
from evoparam.prep import baseline_correct

from .helpers import epochs

###################################################################################################
###################################################################################################

def test_mean_median_linear():

    ts = epochs()
    base = (ts.times >= -0.5) & (ts.times <= -0.05)
    for mode in ("mean", "median"):
        out = baseline_correct(ts, mode=mode)
        stat = np.mean if mode == "mean" else np.median
        assert np.allclose(stat(out.data[..., base], axis=-1), 0, atol=1e-9)
    from dataclasses import replace
    trend = replace(ts, data=ts.data + 30 * ts.times)
    out = baseline_correct(trend, mode="linear")
    slope = np.polyfit(ts.times[base], out.data[0, 0, base], 1)[0]
    assert abs(slope) < 5
    assert out.provenance.steps[-1]["function"] == "evoparam.prep.baseline_correct"
    with pytest.raises(SettingsError):
        baseline_correct(ts, mode="mode")


def test_req080_one_hertz_design_warns_and_suggests():

    ts = epochs(n_trials=30, event_times=np.arange(30) * 1.0)
    with pytest.warns(UserWarning, match="Suggested: baseline_source='first_pulse'") as rec:
        baseline_correct(ts, baseline=(-0.5, -0.05))
    assert "shortest interval between pulses 1.000 s" in str(rec[0].message)
    # The suggested alternative is valid: no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = baseline_correct(ts, baseline=(-0.5, -0.05), baseline_source="first_pulse")
    base = (ts.times >= -0.5) & (ts.times <= -0.05)
    first_level = ts.data[0][..., base].mean(axis=-1)
    assert np.allclose(out.data[5], ts.data[5] - first_level[:, None])


def test_req080_suggests_shorter_baseline_when_possible():

    ts = epochs(n_trials=10, event_times=np.arange(10) * 1.3)
    with pytest.warns(UserWarning, match=r"Suggested: baseline=\(-0.300, -0.050\)"):
        baseline_correct(ts, baseline=(-0.5, -0.05))


def test_req080_no_warning_at_0_2_hz():

    ts = epochs(n_trials=10, event_times=np.arange(10) * 5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        baseline_correct(ts, baseline=(-0.5, -0.05))


def test_masked_baseline_is_respected():

    ts = epochs(n_trials=2)
    base = np.flatnonzero((ts.times >= -0.5) & (ts.times <= -0.05))
    mask = np.zeros(ts.data.shape, dtype=bool)
    mask[0, 0, base[:100]] = True
    data = ts.data.copy()
    data[0, 0, base[:100]] = 1e6
    from dataclasses import replace
    masked = apply_mask(replace(ts, data=data), mask)
    out = baseline_correct(masked)
    assert abs(np.mean(out.data[0, 0, base[100:]])) < 1.0
    mask[0, 0, base] = True
    with pytest.raises(DataError, match="fewer than half"):
        baseline_correct(apply_mask(ts, mask))
