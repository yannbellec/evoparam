# SPDX-License-Identifier: Apache-2.0
"""REQ-009 (units and time): all times in seconds, all amplitudes with an explicit unit;
mixing units raises ``DataError``."""

import warnings

import numpy as np
import pytest

import evoparam
from evoparam import sim
from evoparam.data import Evoked, TrialSet
from evoparam.data.containers import UNITS
from evoparam.data.results import COMPONENT_COLUMNS, INTERVAL_COLUMNS
from evoparam.errors import DataError

TIME_COLUMNS = ("latency", "fwhm", "onset", "offset", "rise_time", "decay_time")


def test_req009_containers_have_times_in_seconds_and_an_explicit_unit() -> None:
    trials, _ = sim.simulate_ccep(n_trials=4, n_channels=3, seed=0)
    assert trials.units in UNITS
    assert trials.units != ""
    # times in seconds: an epoch of -0.5 to 1 s at 2048 Hz
    assert trials.times[0] == pytest.approx(-0.5)
    assert trials.times[-1] < 1.0
    assert np.diff(trials.times) == pytest.approx(1 / trials.sfreq)
    evoked = trials.average()
    assert evoked.units == trials.units
    with pytest.raises(DataError):
        TrialSet(
            data=trials.data,
            times=trials.times,
            sfreq=trials.sfreq,
            ch_names=trials.ch_names,
            units="microvolts",
        )
    with pytest.raises(DataError):
        Evoked(
            data=evoked.data,
            times=evoked.times,
            sfreq=evoked.sfreq,
            ch_names=evoked.ch_names,
            units="",
        )


def test_req009_result_tables_give_times_in_seconds() -> None:
    for name in TIME_COLUMNS:
        assert COMPONENT_COLUMNS[name] == "s", name
    for name in ("latency", "fwhm", "onset", "offset"):
        assert INTERVAL_COLUMNS[f"{name}_ci_low"] == "s"


def test_req009_unit_conversion_is_explicit() -> None:
    trials, _ = sim.simulate_ccep(n_trials=4, n_channels=3, seed=0)
    mv = trials.convert_units("mV")
    assert mv.units == "mV"
    assert np.allclose(mv.data * 1000.0, trials.data)
    with pytest.raises(DataError):
        trials.convert_units("au")


def test_req009_result_amplitudes_have_an_explicit_unit() -> None:
    pytest.importorskip("pandas")
    trials, _ = sim.simulate_erp(n_trials=10, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = evoparam.fit(trials, preset="erp", random_state=0).to_dataframe()
        mv = evoparam.fit(trials.convert_units("mV"), preset="erp", random_state=0)
    assert table.attrs["units"]["amplitude"] == trials.units
    assert table.attrs["units"]["area"] == f"{trials.units}·s"
    assert mv.to_dataframe().attrs["units"]["amplitude"] == "mV"
    assert mv.to_dataframe("fit").attrs["units"]["rmse"] == "mV"
    assert mv.fits[0].to_dataframe().attrs["units"]["amplitude"] == "mV"


def test_req009_a_group_of_fits_in_different_units_raises() -> None:
    pytest.importorskip("pandas")
    from evoparam.data.results import GroupFit

    trials, _ = sim.simulate_erp(n_trials=10, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        uv = evoparam.fit(trials, preset="erp", random_state=0)
        mv = evoparam.fit(trials.convert_units("mV"), preset="erp", random_state=0)
    with pytest.raises(DataError, match="different units"):
        GroupFit.from_fits(uv.fits + mv.fits, ["a"] * len(uv.fits) + ["b"] * len(mv.fits))


def test_req009_mixing_units_raises() -> None:
    mne = pytest.importorskip("mne")
    pytest.importorskip("pandas")
    info = mne.create_info(["EEG1", "MAG1"], 1000.0, ["eeg", "mag"])
    data = np.random.default_rng(0).normal(size=(5, 2, 300)) * 1e-6
    epochs = mne.EpochsArray(data, info, tmin=-0.1, verbose=False)
    with pytest.raises(DataError):
        evoparam.io.read_mne(epochs)
    with pytest.raises(DataError):
        evoparam.fit(epochs, preset="erp")
