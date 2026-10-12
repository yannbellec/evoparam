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


def test_req009_containers_have_times_in_seconds_and_an_explicit_unit():
    trials, _ = sim.simulate_ccep(n_trials=4, n_channels=3, seed=0)
    assert trials.units in UNITS and trials.units != ""
    # times in seconds: an epoch of -0.5 to 1 s at 2048 Hz
    assert trials.times[0] == pytest.approx(-0.5) and trials.times[-1] < 1.0
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


def test_req009_result_tables_give_times_in_seconds():
    for name in TIME_COLUMNS:
        assert COMPONENT_COLUMNS[name] == "s", name
    for name in ("latency", "fwhm", "onset", "offset"):
        assert INTERVAL_COLUMNS[f"{name}_ci_low"] == "s"


def test_req009_unit_conversion_is_explicit():
    trials, _ = sim.simulate_ccep(n_trials=4, n_channels=3, seed=0)
    mv = trials.convert_units("mV")
    assert mv.units == "mV"
    assert np.allclose(mv.data * 1000.0, trials.data)
    with pytest.raises(DataError):
        trials.convert_units("au")


@pytest.mark.xfail(
    strict=True,
    reason="The fit tables give the amplitude unit as 'data' (the "
    "unit of the fitted data), not the unit itself: fits do not record the unit "
    "of their data (measured: GroupFit.to_dataframe().attrs['units']['amplitude']"
    " == 'data' for a TrialSet in uV). evoparam/models and data/results are "
    "being changed by other builders; reported.",
)
def test_req009_result_amplitudes_have_an_explicit_unit():
    pytest.importorskip("pandas")
    trials, _ = sim.simulate_erp(n_trials=10, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = evoparam.fit(trials, preset="erp", random_state=0).to_dataframe()
    assert table.attrs["units"]["amplitude"] == trials.units


def test_req009_mixing_units_raises():
    mne = pytest.importorskip("mne")
    pytest.importorskip("pandas")
    info = mne.create_info(["EEG1", "MAG1"], 1000.0, ["eeg", "mag"])
    data = np.random.default_rng(0).normal(size=(5, 2, 300)) * 1e-6
    epochs = mne.EpochsArray(data, info, tmin=-0.1, verbose=False)
    with pytest.raises(DataError):
        evoparam.io.read_mne(epochs)
    with pytest.raises(DataError):
        evoparam.fit(epochs, preset="erp")
