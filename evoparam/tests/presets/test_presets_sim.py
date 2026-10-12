# SPDX-License-Identifier: Apache-2.0
"""Presets on simulations: ERNA (REQ-276) and SEP (REQ-277)."""

import warnings

import numpy as np

from evoparam.data import Evoked, TrialSet
from evoparam.models import fit
from evoparam.shapes import get_shape

DAMPED = get_shape("damped_oscillation")


def _erna_case(seed):
    """Per-pulse epochs of the last five pulses of simulated bursts (amplitude at least 95 %
    of its final value), channel at SNR 3; ERNA onset 2 ms after each pulse."""

    from evoparam.sim import simulate_erna

    ts, truth = simulate_erna(
        10, n_trials=20, seed=seed, snr=3, artifact=False, tmax=0.09, onset_delay=0.002
    )
    table = truth.to_dataframe()
    last = table[(table["event"] == 9) & (table["channel"] == "L01")].iloc[0]
    n = int(round(0.0105 * ts.sfreq))
    epochs = []
    for k in range(5, 10):
        start = np.searchsorted(ts.times, k / 130.0 - 0.002)
        epochs.append(ts.data[:, 1:2, start : start + n])
    pulse = 9 / 130.0
    start = np.searchsorted(ts.times, pulse - 0.002)
    times = ts.times[start : start + n] - pulse
    sub = TrialSet(
        data=np.concatenate(epochs), times=times, sfreq=ts.sfreq, ch_names=["L01"], modality="erna"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit(sub, "erna").fits[0]
    comps = [c for c in result.components if c.shape == "damped_oscillation"]
    if not comps:
        return np.nan, np.nan
    comp = max(comps, key=lambda c: abs(c.amplitude))
    true = np.array([last["param_t0"], last["param_tau"], last["param_freq"], last["param_phase"]])
    first_peak = float(DAMPED.extremum(np.asarray(comp.params))) - (
        float(DAMPED.extremum(true)) - pulse
    )
    return comp.params[2] - last["param_freq"], first_peak * 1e3


def test_req276_erna_frequency_and_first_peak():

    errors = np.array([_erna_case(seed) for seed in range(20)])
    ok = (np.abs(errors[:, 0]) <= 5) & (np.abs(errors[:, 1]) <= 0.1)
    assert ok.mean() >= 0.9


SEP_SFREQ = 5000.0
SEP_TIMES = np.arange(-0.05, 0.1, 1 / SEP_SFREQ)


def _gauss(latency, fwhm):
    return np.exp(-0.5 * ((SEP_TIMES - latency) / (fwhm / 2.3548)) ** 2)


def _sep_case(seed, snr=5.0, phi=0.0):
    """Median-nerve SEP average: N20 (FWHM 4 ms), P25, N33; noise SD = N20 peak / snr."""

    rng = np.random.default_rng(seed)
    latency = rng.uniform(0.0185, 0.0215)
    y = (
        -_gauss(latency, 0.004)
        + 0.8 * _gauss(latency + 0.005, 0.006)
        - 0.6 * _gauss(latency + 0.013, 0.008)
    )
    sd = 1.0 / snr
    e = rng.normal(0, sd * np.sqrt(1 - phi**2), SEP_TIMES.size)
    noise = np.empty_like(e)
    noise[0] = rng.normal(0, sd)
    for k in range(1, SEP_TIMES.size):
        noise[k] = phi * noise[k - 1] + e[k]
    ev = Evoked(data=(y + noise)[None], times=SEP_TIMES, sfreq=SEP_SFREQ, ch_names=["C3'"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit(ev, "sep").fits[0]
    n20 = [c for c in result.components if c.label == "N20"]
    return (n20[0].latency - latency) * 1e3 if n20 else np.nan


def test_req277_sep_n20_latency():

    errors = np.array([_sep_case(seed) for seed in range(20)])
    assert np.mean(np.abs(errors) <= 0.5) >= 0.9


def test_sep_with_coloured_noise_is_a_documented_limit():
    """AR(1) noise at the same SNR: N20 within 0.5 ms in 16 of 20 runs at rho 0.5 (and 8 of
    20 at rho 0.8; CHANGELOG, step S15)."""

    errors = np.array([_sep_case(seed, phi=0.5) for seed in range(20)])
    assert np.mean(np.abs(errors) <= 0.5) >= 0.75
