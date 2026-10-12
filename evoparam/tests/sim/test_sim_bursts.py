# SPDX-License-Identifier: Apache-2.0
"""simulate_bursts: induced bursts with a random phase per trial (benchmark task T7)."""

import numpy as np
import pytest

from evoparam import sim
from evoparam.errors import SettingsError

BURST = {"freq": 80.0, "centre": 0.25, "duration": 0.1, "amplitude": 20.0, "channel": "c1"}


def test_bursts_truth_and_determinism():
    ts, truth = sim.simulate_bursts(20, n_channels=2, bursts=[BURST], seed=3)
    again, _ = sim.simulate_bursts(20, n_channels=2, bursts=[BURST], seed=3)
    np.testing.assert_allclose(ts.data, again.data, rtol=1e-9, atol=0)
    record = truth.settings["bursts"][0]
    assert record["channel"] == "c1"
    assert record["t_start"] == pytest.approx(0.2)
    assert record["t_end"] == pytest.approx(0.3)
    assert len(truth.settings["phases"][0]) == 20
    assert ts.data.shape == (20, 2, 1400)
    assert ts.units == "uV"
    null, _ = sim.simulate_bursts(20, n_channels=2, seed=3)  # same noise, no burst
    np.testing.assert_allclose(ts.data - truth.signal, null.data, rtol=1e-9, atol=1e-9)
    assert not np.any(truth.signal[:, 0])  # channel c0 has no burst


def test_bursts_are_induced_not_evoked():
    """The random phase leaves (almost) nothing in the average, all of it in the power."""

    ts, truth = sim.simulate_bursts(200, bursts=[dict(BURST, channel=0)], noise_sd=0.0, seed=1)
    times = ts.times
    inside = np.abs(times - 0.25) <= 0.02
    power = np.mean(truth.signal[:, 0, inside] ** 2)
    average = np.mean(truth.signal[:, 0].mean(axis=0)[inside] ** 2)
    envelope = 0.5 + 0.5 * np.cos(2 * np.pi * (times[inside] - 0.25) / 0.1)
    assert power == pytest.approx(np.mean((20.0 * envelope) ** 2) / 2, rel=0.1)
    assert average < 0.05 * power
    assert not np.any(truth.signal[:, 0, np.abs(times - 0.25) > 0.05 + 1e-9])


def test_bursts_waveform_and_checks():
    times = np.arange(0.0, 0.5, 0.001)
    one = sim.burst_waveform(times, freq=10.0, centre=0.25, duration=0.2, amplitude=2.0)
    assert one.shape == (500,)
    assert one[250] == pytest.approx(2.0 * np.cos(2 * np.pi * 2.5))
    many = sim.burst_waveform(
        times, freq=10.0, centre=0.25, duration=0.2, amplitude=2.0, phase=np.zeros(3)
    )
    assert many.shape == (3, 500)
    with pytest.raises(SettingsError, match="lacks"):
        sim.simulate_bursts(2, bursts=[{"freq": 10.0}])
    with pytest.raises(SettingsError, match="sfreq/2"):
        sim.simulate_bursts(2, bursts=[dict(BURST, channel=0, freq=600.0)])
    with pytest.raises(SettingsError, match="unknown channel"):
        sim.simulate_bursts(2, bursts=[dict(BURST, channel="x")])
    null, truth = sim.simulate_bursts(5, seed=0)
    assert truth.settings["bursts"] == []
    assert null.data.shape == (5, 1, 1400)
