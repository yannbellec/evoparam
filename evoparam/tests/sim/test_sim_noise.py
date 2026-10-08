"""REQ-251 (noise models) and REQ-252 (requested SNR is the measured SNR)."""

import numpy as np
import pytest
from scipy import signal

from evoparam import sim
from evoparam.errors import SettingsError
from evoparam.tests.sim._helpers import (SMALL, measured_noise_sd, measured_snr,
                                         real_baselines)

###################################################################################################
## REQ-251: noise models

@pytest.mark.parametrize("chi", [0.0, 0.5, 1.0, 1.5, 2.0])
def test_pink_spectral_exponent(chi):
    """Log-log slope over 2-100 Hz of the mean spectrum of 100 realisations within ±0.1."""

    sfreq = 1000.
    noise = sim.make_noise(4000, sfreq, "pink", chi=chi, size=100, seed=1)
    freqs, power = signal.welch(noise, fs=sfreq, nperseg=1000, axis=-1)
    keep = (freqs >= 2) & (freqs <= 100)
    slope = np.polyfit(np.log10(freqs[keep]), np.log10(power.mean(axis=0)[keep]), 1)[0]
    assert abs(-slope - chi) <= 0.1, (chi, -slope)
    assert abs(noise.std() - 1.0) < 0.05


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9, -0.5])
def test_ar1_lag1_autocorrelation(rho):
    noise = sim.make_noise(2000, 500., "ar1", rho=rho, size=100, sd=2.0, seed=2)
    centred = noise - noise.mean(axis=-1, keepdims=True)
    lag1 = np.sum(centred[:, 1:] * centred[:, :-1], axis=-1) / np.sum(centred ** 2, axis=-1)
    assert abs(lag1.mean() - rho) <= 0.02, (rho, lag1.mean())
    assert abs(noise.std() - 2.0) < 0.1


def test_white_and_line():
    white = sim.make_noise(5000, 1000., "white", size=(4,), sd=3.0, seed=0)
    assert white.shape == (4, 5000)
    assert abs(white.std() - 3.0) < 0.05
    lag1 = np.mean(white[:, 1:] * white[:, :-1]) / white.var()
    assert abs(lag1) < 0.03

    noise, info = sim.make_noise(2000, 1000., "white+line", sd=0.1, line_freq=60.,
                                 line_harmonics=3, line_amplitude=5.0, seed=0,
                                 return_info=True)
    freqs, power = signal.periodogram(noise, fs=1000.)
    for harmonic in (1, 2, 3):
        near = np.abs(freqs - 60. * harmonic) < 1.0
        assert power[near].max() > 100 * np.median(power)
    assert info["line"] == {"freq": 60.0, "harmonics": 3, "amplitude": 5.0}
    line_only = sim.make_noise(1000, 1000., "line", line_amplitude=2.0, line_harmonics=1, seed=1)
    assert abs(np.max(np.abs(line_only)) - 2.0) < 0.01


def test_harmonics_above_nyquist_are_left_out():
    noise = sim.make_noise(1000, 200., "line", line_freq=50., line_harmonics=5,
                           line_amplitude=1.0, seed=0)
    freqs, power = signal.periodogram(noise, fs=200.)
    assert power[np.argmin(np.abs(freqs - 50.))] > 1e3 * power[np.argmin(np.abs(freqs - 80.))]


def test_make_noise_errors():
    with pytest.raises(SettingsError, match="noise kind"):
        sim.make_noise(100, 100., "brown")
    with pytest.raises(SettingsError, match="at most one"):
        sim.make_noise(100, 100., "pink+white")
    with pytest.raises(SettingsError, match="rho"):
        sim.make_noise(100, 100., "ar1", rho=1.0)
    with pytest.raises(SettingsError, match="chi"):
        sim.make_noise(100, 100., "pink", chi=-1.0)
    with pytest.raises(SettingsError, match="n_times"):
        sim.make_noise(1, 100., "white")
    with pytest.raises(SettingsError, match="sd"):
        sim.make_noise(100, 100., "white", sd=-1.0)
    with pytest.raises(SettingsError, match="source"):
        sim.make_noise(100, 100., "real")
    with pytest.raises(SettingsError):
        sim.make_noise(100, 100., "none")


def test_real_noise_segments_come_from_the_source():
    real = real_baselines(n_trials=6, n_channels=3, sfreq=500.)
    noise, info = sim.make_noise(400, 500., "real", size=(5, 3), source=real, seed=4,
                                 return_info=True)
    assert noise.shape == (5, 3, 400)
    assert len(info["segments"]) == 5
    for index, segment in enumerate(info["segments"]):
        start, stop = segment["epoch_samples"]
        piece = real.data[segment["source_trial"], :, start:stop]
        assert stop - start == 400
        assert real.times[stop - 1] < 0                     # pre-stimulus window by default
        np.testing.assert_allclose(noise[index], piece - piece.mean(axis=-1, keepdims=True),
                                   atol=1e-10)


###################################################################################################
## REQ-252: SNR

@pytest.mark.parametrize("name", sorted(sim.GENERATORS))
@pytest.mark.parametrize("snr", [2.0, 5.0])
def test_measured_snr_equals_requested(name, snr):
    ts, truth = sim.GENERATORS[name](seed=21, snr=snr, **SMALL[name])
    checked = 0
    for index, comp in enumerate(truth.components):
        if comp["snr"] is None or not truth.present[:, index].any():
            continue
        measured, requested = measured_snr(ts, truth, index)
        assert abs(measured / requested - 1) <= 0.05, (comp["label"], measured, requested)
        checked += 1
    assert checked > 0


def test_snr_with_variability_and_every_noise_model():
    for noise in ("white", "ar1", "pink", "pink+line"):
        ts, truth = sim.simulate_erp(seed=3, n_trials=40, noise=noise, snr=4.0,
                                     amplitude_cv=0.2, jitter_sd=0.01)
        for index in range(truth.n_components):
            measured, requested = measured_snr(ts, truth, index)
            assert abs(measured / requested - 1) <= 0.05


def test_snr_with_real_noise():
    real = real_baselines(n_trials=10, n_channels=2, sfreq=500., tmin=-3.0)
    ts, truth = sim.simulate_erp(seed=3, n_trials=8, n_channels=2, noise="real",
                                 noise_kw={"source": real}, snr=3.0)
    assert truth.noise["kind"] == "real"
    for index in range(truth.n_components):
        measured, requested = measured_snr(ts, truth, index)
        assert abs(measured / requested - 1) <= 0.05


def test_snr_forms():
    ts, truth = sim.simulate_erp(seed=1, n_trials=10, components=["N1", "P3"], snr=[2.0, 6.0])
    table = truth.to_dataframe()
    assert dict(zip(table["label"], table["snr"])) == pytest.approx({"N1": 2.0, "P3": 6.0})
    _, truth = sim.simulate_erp(seed=1, n_trials=10, components=["N1", "P3"], snr={"P3": 3.0})
    snrs = {c["label"]: c["snr"] for c in truth.components}
    assert snrs["P3"] == 3.0 and snrs["N1"] is None
    _, truth = sim.simulate_erp(seed=1, n_trials=10, n_channels=2, snr={"Pz": 4.0},
                                components=[{"label": "P3", "channels": ("Pz",)}])
    assert [c["snr"] for c in truth.components] == [4.0]
    # The SNR is reached on the channel of largest weight and scaled by the weight elsewhere.
    ts, truth = sim.simulate_erp(seed=1, n_trials=10, n_channels=2, snr=5.0,
                                 components=[{"label": "P3", "weights": (1.0, -0.5)}])
    assert [c["snr"] for c in truth.components] == [5.0, 2.5]
    assert truth.components[1]["amplitude"] < 0 < truth.components[0]["amplitude"]
    for index in range(2):
        measured, requested = measured_snr(ts, truth, index)
        assert abs(measured / requested - 1) <= 0.05


def test_snr_for_a_channel_without_component_raises():
    with pytest.raises(SettingsError, match="no component"):
        sim.simulate_erp(seed=1, n_trials=4, n_channels=2, snr={"Pz": 3.0},
                         components=[{"label": "N1", "channels": ("Cz",)}])
    _, truth = sim.simulate_ccep(seed=2, **SMALL["ccep"])
    silent = [name for name in truth.ch_names if name not in truth.settings["responders"]]
    with pytest.raises(SettingsError, match="no component"):
        sim.simulate_ccep(seed=2, snr={silent[0]: 4.0}, **SMALL["ccep"])
    with pytest.raises(SettingsError, match="neither"):
        sim.simulate_erp(seed=1, n_trials=4, snr={"N2pc": 3.0})
    with pytest.raises(SettingsError, match="values for the"):
        sim.simulate_erp(seed=1, n_trials=4, snr=[1.0, 2.0])
    with pytest.raises(SettingsError, match="positive"):
        sim.simulate_erp(seed=1, n_trials=4, snr=-1.0)
    with pytest.raises(SettingsError, match="no noise"):
        sim.simulate_erp(seed=1, n_trials=4, noise="none", snr=2.0)


def test_amplitudes_in_data_units_without_snr():
    ts, truth = sim.simulate_erp(seed=1, n_trials=5, noise="none", snr=None,
                                 components=[{"label": "P3", "latency": 0.3, "fwhm": 0.1,
                                              "amplitude": 7.5}])
    assert np.allclose(ts.data, truth.signal)
    assert np.isclose(np.max(ts.data), 7.5, rtol=1e-6)
    assert truth.components[0]["snr"] is None
    assert np.allclose(measured_noise_sd(ts, truth), 0.0)
