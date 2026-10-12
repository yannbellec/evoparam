# SPDX-License-Identifier: Apache-2.0
"""Tests for evoparam.io.epochs."""

import numpy as np
import pytest

from evoparam.data import TrialSet
from evoparam.errors import DataError, EventError
from evoparam.io.epochs import epoch, epoch_data

###################################################################################################
###################################################################################################


def test_epoch_data():

    sfreq = 100.0
    data = np.vstack([np.arange(1000), -np.arange(1000)]).astype(float)

    times, epochs, kept = epoch_data(data, sfreq, [1.0, 5.0, 9.95], -0.1, 0.2)
    assert np.allclose(times, np.arange(-10, 21) / sfreq)
    assert epochs.shape == (2, 2, 31)
    assert list(kept) == [True, True, False]
    assert epochs[0, 0, 10] == 100
    assert epochs[1, 1, 10] == -500

    # With an offset clock, and 1d data
    times, epochs, kept = epoch_data(data[0], sfreq, [11.0], -0.1, 0.2, t0=10.0)
    assert epochs.shape == (1, 31)
    assert epochs[0, 10] == 100


def test_epoch_returns_a_trialset():

    sfreq = 100.0
    data = np.vstack([np.arange(1000), -np.arange(1000)]).astype(float)
    onsets = [0.05, 1.0, 1.2, 5.004, 9.95]
    trials = epoch(data, sfreq, onsets, -0.1, 0.2, ch_names=["x", "y"])
    assert isinstance(trials, TrialSet)
    trials.validate()

    # Window [-0.1, 0.2): 30 samples, onset sample at t = 0
    assert trials.n_trials == 3
    assert trials.n_times == 30
    assert np.allclose(trials.times, np.arange(-10, 20) / sfreq)
    assert trials.data[0, 0, 10] == 100
    assert trials.data[2, 1, 10] == -500
    assert np.allclose(trials.event_times, [1.0, 1.2, 5.004])

    step = trials.provenance.steps[-1]
    assert step["function"] == "evoparam.io.epoch"
    assert step["kept"] == [False, True, True, True, False]
    assert step["overlaps_previous"] == [False, True, False]

    pytest.importorskip("pandas")
    assert trials.metadata["trial"].tolist() == [1, 2, 3]
    assert trials.metadata["overlaps_previous"].tolist() == [False, True, False]


def test_epoch_with_metadata_and_clock():

    pd = pytest.importorskip("pandas")
    data = np.arange(500, dtype=float)
    metadata = pd.DataFrame({"condition": ["a", "b", "c"]}, index=[7, 8, 9])
    trials = epoch(
        data, 100.0, [10.5, 12.0, 14.99], -0.05, 0.05, t0=10.0, metadata=metadata, units="mV"
    )
    assert trials.ch_names == ["CH001"]
    assert trials.units == "mV"
    assert list(trials.metadata["condition"]) == ["a", "b"]
    assert trials.data[0, 0, 5] == 50
    assert trials.data[1, 0, 5] == 200

    with pytest.raises(DataError, match="rows"):
        epoch(data, 100.0, [10.5], -0.05, 0.05, t0=10.0, metadata=metadata)


def test_epoch_errors():

    data = np.zeros((2, 100))
    with pytest.raises(EventError, match="fits inside"):
        epoch(data, 100.0, [5.0], -0.1, 0.1)
    with pytest.raises(DataError, match="greater"):
        epoch(data, 100.0, [0.5], 0.1, -0.1)
    with pytest.raises(DataError, match="2d"):
        epoch(np.zeros((2, 2, 2)), 100.0, [0.5], -0.1, 0.1)
    with pytest.raises(EventError, match="finite"):
        epoch(data, 100.0, [np.nan], -0.1, 0.1)
