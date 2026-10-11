# SPDX-License-Identifier: Apache-2.0
"""Tests for TrialSet and Evoked (spec §4.3.1, §4.3.2)."""

import numpy as np
import pandas as pd
import pytest

from evoparam.data import TrialSet, Evoked
from evoparam.data.containers import auto_chunk
from evoparam.errors import DataError
from evoparam.tests.data.factories import make_trialset, SFREQ


def test_construction_and_repr():
    trials = make_trialset()
    assert (trials.n_trials, trials.n_channels, trials.n_times) == (6, 3, 400)
    assert trials.data.dtype == np.float64
    assert "6 trials × 3 channels × 400 times" in repr(trials)
    assert trials.provenance.data_hash is not None


@pytest.mark.parametrize("change, field", [
    (dict(data=np.zeros((3, 400))), "data"),
    (dict(times=np.arange(399) / SFREQ), "times"),
    (dict(times=np.arange(400)[::-1] / SFREQ), "times"),
    (dict(sfreq=999.0), "sfreq"),
    (dict(ch_names=["A", "A", "B"]), "ch_names"),
    (dict(ch_names=["A", "B"]), "ch_names"),
    (dict(units="microvolt"), "units"),
    (dict(modality="fmri"), "modality"),
    (dict(species="Human"), "species"),
    (dict(masks=np.zeros((6, 3, 10), bool)), "masks"),
    (dict(event_times=np.zeros(4)), "event_times"),
    (dict(metadata=pd.DataFrame({"trial": range(5)})), "metadata"),
    (dict(metadata=pd.DataFrame({"stim_current_mA": ["1"] * 6})), "stim_current_mA"),
    (dict(ch_info=pd.DataFrame({"name": ["X", "Y", "Z"]})), "ch_info"),
])
def test_invariants(change, field):
    base = make_trialset(with_tables=False)
    kwargs = dict(data=base.data, times=base.times, sfreq=base.sfreq, ch_names=base.ch_names)
    kwargs.update(change)
    with pytest.raises(DataError, match=field):
        TrialSet(**kwargs)


def test_data_error_is_value_error():
    with pytest.raises(ValueError):
        TrialSet(np.zeros((1, 1, 1)), np.zeros(1), 1.0, ["A"])


def test_nan_allowed_only_in_masks():
    trials = make_trialset(with_masks=True)
    assert np.isnan(trials.data).any()
    data = trials.data.copy()
    data[0, 0, 300] = np.nan
    with pytest.raises(DataError, match="NaN"):
        TrialSet(data, trials.times, trials.sfreq, trials.ch_names, masks=trials.masks)


def test_select_and_crop_window_convention():
    trials = make_trialset()
    sub = trials.select(trials=[4, 1], picks=["CH02", 0], tmin=0.0, tmax=0.1)
    assert sub.ch_names == ["CH02", "CH00"]
    assert sub.times[0] == pytest.approx(0.0) and sub.times[-1] == pytest.approx(0.099)
    assert sub.n_times == 100
    np.testing.assert_array_equal(sub.data[0, 0], trials.data[4, 2, 100:200])
    assert list(sub.metadata["trial"]) == [4, 1]
    assert list(sub.ch_info["name"]) == ["CH02", "CH00"]
    assert sub.provenance.steps[-1]["function"] == "evoparam.data.TrialSet.select"
    assert sub.provenance.data_hash != trials.provenance.data_hash
    cropped = trials.crop(-0.05, 0.05)
    assert cropped.n_times == 100 and cropped.n_trials == trials.n_trials


def test_select_query_and_errors():
    trials = make_trialset()
    high = trials.select(query="condition == 'high'")
    assert set(high.metadata["condition"]) == {"high"}
    with pytest.raises(DataError, match="unknown channel"):
        trials.select(picks="NOPE")
    with pytest.raises(DataError, match="empty"):
        trials.select(query="condition == 'none'")
    with pytest.raises(DataError, match="fewer than 2"):
        trials.select(tmin=0.2995)


def test_original_unchanged():
    trials = make_trialset()
    before = trials.copy()
    trials.select(trials=[0], picks=[0]).average()
    trials.convert_units("mV")
    assert trials == before


def test_average_mean_and_sem():
    trials = make_trialset()
    evoked = trials.average()
    assert isinstance(evoked, Evoked)
    np.testing.assert_allclose(evoked.data, trials.data.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(evoked.sem, trials.data.std(axis=0, ddof=1) / np.sqrt(6),
                               rtol=1e-12)
    assert evoked.n_trials == 6 and evoked.masks is None


def test_average_excludes_rejected_and_groups():
    trials = make_trialset()
    trials.metadata.loc[0, "rejected"] = True
    evoked = trials.average()
    assert evoked.n_trials == 5
    np.testing.assert_allclose(evoked.data, trials.data[1:].mean(axis=0), rtol=1e-12)
    groups = trials.average(by="condition")
    assert [g.metadata["condition"][0] for g in groups] == ["high", "low"]
    assert [g.n_trials for g in groups] == [3, 2]


def test_average_with_masks():
    trials = make_trialset(with_masks=True)
    masks = trials.masks.copy()
    masks[1:, :, 98:106] = False
    data = np.where(masks, np.nan, np.nan_to_num(trials.data, nan=1.0))
    masks[0, :, 98:106] = True
    data[0, :, 98:106] = np.nan
    data[1:, :, 98:106] = 1.0
    partial = TrialSet(data, trials.times, trials.sfreq, trials.ch_names, masks=masks)
    evoked = partial.average()
    np.testing.assert_allclose(evoked.data[:, 98:106], 1.0)
    full = trials.average()
    assert full.masks is not None and full.masks[:, 98:106].all()
    assert np.isnan(full.data[:, 98:106]).all()


def test_average_robust_matches_scipy():
    from scipy.stats import trim_mean
    trials = make_trialset(n_trials=11)
    evoked = trials.average(robust=True)
    np.testing.assert_allclose(evoked.data, trim_mean(trials.data, 0.2, axis=0), rtol=1e-10)
    assert np.all(np.isfinite(evoked.sem))


def test_chunk_does_not_change_result():
    trials = make_trialset(n_channels=5)
    reference = trials.average(robust=True)
    for chunk in (1, 2, "auto"):
        assert np.array_equal(trials.average(robust=True, chunk=chunk).data, reference.data)
    doubled = trials.apply(lambda x: 2 * x, chunk=2)
    np.testing.assert_array_equal(doubled.data, 2 * trials.data)
    with pytest.raises(DataError, match="chunk"):
        trials.average(chunk=0)


def test_auto_chunk_rule():
    assert auto_chunk(500, 4096, 256, max_memory_gb=1.0) == int(1024 ** 3 // (500 * 4096 * 24))
    assert auto_chunk(1, 10, 3, max_memory_gb=100) == 3


def test_apply_times_axis_and_provenance():
    trials = make_trialset()
    detrended = trials.apply(lambda x: x - x.mean(axis=-1, keepdims=True), axis="times")
    np.testing.assert_allclose(detrended.data.mean(axis=-1), 0.0, atol=1e-12)
    assert detrended.provenance.steps[-1]["settings"]["axis"] == "times"
    with pytest.raises(DataError, match="shape"):
        trials.apply(lambda x: x[:, :1])


def test_convert_units():
    trials = make_trialset()
    millivolts = trials.convert_units("mV")
    np.testing.assert_allclose(millivolts.data, trials.data * 1e-3)
    assert millivolts.units == "mV"
    evoked = trials.average().convert_units("V")
    np.testing.assert_allclose(evoked.sem, trials.average().sem * 1e-6)
    with pytest.raises(DataError):
        TrialSet(trials.data, trials.times, trials.sfreq, trials.ch_names, units="au") \
            .convert_units("uV")


def test_as_evoked_and_as_trialset():
    trials = make_trialset(with_tables=False)
    one = trials.select(trials=[2])
    evoked = one.as_evoked()
    np.testing.assert_array_equal(evoked.data, trials.data[2])
    back = evoked.as_trialset()
    assert back.n_trials == 1
    np.testing.assert_array_equal(back.data[0], trials.data[2])
    assert trials.as_evoked().n_trials == 6


def test_evoked_validation_and_select():
    evoked = make_trialset().average()
    with pytest.raises(DataError, match="n_trials"):
        Evoked(evoked.data, evoked.times, evoked.sfreq, evoked.ch_names, n_trials=0)
    with pytest.raises(DataError, match="sem"):
        Evoked(evoked.data, evoked.times, evoked.sfreq, evoked.ch_names, sem=np.zeros((1, 2)))
    per_channel = Evoked(evoked.data, evoked.times, evoked.sfreq, evoked.ch_names,
                         n_trials=np.array([5, 6, 4]))
    sub = per_channel.select(picks=["CH02"], tmin=0.0, tmax=0.05)
    assert list(sub.n_trials) == [4] and sub.n_times == 50


def test_equality_and_copy():
    trials = make_trialset()
    clone = trials.copy()
    assert clone == trials and clone is not trials
    clone.data[0, 0, 0] += 1.0
    assert clone != trials
