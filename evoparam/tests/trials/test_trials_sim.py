"""Single-trial estimation on simulations (REQ-150 to REQ-155)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from evoparam.data import TrialSet
from evoparam.errors import SettingsError
from evoparam.trials import dose_response, fit_trials, jitter, prestimulus_dependence

from .helpers import SFREQ, TIMES, channel_fit, gauss, simulate

###################################################################################################
###################################################################################################

def _run(ts, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_trials(ts, channel_fit(ts), **kwargs)


@pytest.mark.parametrize("sd", [0.002, 0.005])
def test_req150_latency_amplitude_and_jitter(sd):

    ratios, lat, amp = [], [], []
    for seed in range(6):
        ts, shifts, scales = simulate(jitter=sd, seed=seed)       # per-trial SNR 3
        tf = _run(ts, max_shift=0.015, r2_min=0.0)
        fit = tf.fits[0]
        lat.append(np.mean(np.abs(fit.shift[:, 0] - (shifts - shifts.mean()))) * SFREQ)
        amp.append(np.mean(np.abs(fit.amplitude[:, 0] / (-30 * scales) - 1)))
        ratios.append(jitter(tf).iloc[0]["jitter_sd_excess"] / np.std(shifts, ddof=1))
    assert np.median(lat) <= 1.0
    assert np.median(amp) <= 0.15
    assert abs(np.median(ratios) - 1) <= 0.2


def test_req151_no_invented_jitter_and_realign_refused():

    ratios, refused = [], 0
    for seed in range(20):
        ts, _, _ = simulate(jitter=0.0, cv=0.0, seed=100 + seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tf = fit_trials(ts, channel_fit(ts), realign=True)
        fit = tf.fits[0]
        ratios.append(fit.jitter_sd_null[0] / fit.jitter_sd[0])
        refused += fit.realigned is None and any("refused" in str(w.message) for w in caught)
    # jitter_sd_null >= 0.8 jitter_sd: the SD of 40 shifts varies by about 11 % from run to
    # run, so the bound holds in 90 % of runs or more, and in the median
    assert np.median(ratios) >= 0.8
    assert np.mean(np.array(ratios) >= 0.8 - 1e-9) >= 0.9
    assert refused / 20 >= 0.95


def test_req152_jitter_corrected_amplitude():

    ts, shifts, _ = simulate(n=100, jitter=0.006, cv=0.0, noise=3.0, fwhm=0.03, seed=7)
    tf = _run(ts, max_shift=0.02, r2_min=0.0)
    corrected = jitter(tf).iloc[0]["amplitude_jitter_corrected"]
    assert abs(corrected / -30 - 1) <= 0.10


def test_req153_regularized_beats_free_at_low_snr():

    better = 0
    for seed in range(10):
        ts, shifts, _ = simulate(jitter=0.003, amp=-30.0, noise=30.0, seed=200 + seed)
        group = channel_fit(TrialSet(data=ts.data - 0, times=ts.times, sfreq=ts.sfreq,
                                     ch_names=ts.ch_names))
        if group.fits[0].status != "ok":
            continue
        errs = {}
        for mode in ("free", "regularized"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = fit_trials(ts, group, mode=mode, r2_min=0.0).fits[0]
            errs[mode] = np.mean(np.abs(fit.shift[:, 0] - (shifts - shifts.mean())))
        better += errs["regularized"] <= errs["free"]
    assert better / 10 >= 0.9


def test_req154_dose_response_hill():

    rng = np.random.default_rng(8)
    currents = np.repeat([1.0, 2.0, 3.0, 5.0, 8.0], 10)
    i50, h = 3.0, 2.5
    amps = -60 * currents ** h / (i50 ** h + currents ** h)
    data = rng.normal(0, 60 / 3 / 3, (len(currents), 1, len(TIMES)))
    for i, a in enumerate(amps):
        data[i, 0] += a * gauss(0.08, 0.03)
    meta = pd.DataFrame({"stim_current_mA": currents})
    ts = TrialSet(data=data, times=TIMES, sfreq=SFREQ, ch_names=["c0"], metadata=meta)
    tf = _run(ts, r2_min=0.0, covariates=("dose",))
    table = dose_response(tf, model="hill")
    row = table.iloc[0]
    assert abs(row["i50"] / i50 - 1) <= 0.15
    assert abs(row["h"] / h - 1) <= 0.30
    assert np.isclose(row["i_thr"], row["i50"] * (0.1 / 0.9) ** (1 / row["h"]))
    with pytest.raises(SettingsError):
        dose_response(tf.table.assign(stim_current_mA=1.0))


def _phase_trials(modulation, seed, n=100):
    rng = np.random.default_rng(seed)
    phases = rng.uniform(-np.pi, np.pi, n)
    data = rng.normal(0, 1.0, (n, 1, len(TIMES)))
    pre = TIMES < 0
    for i, phi in enumerate(phases):
        # 10 Hz oscillation whose phase at t = -0.01 s is phi
        data[i, 0, pre] += 5 * np.cos(2 * np.pi * 10 * (TIMES[pre] + 0.01) + phi)
        gain = 1 + modulation * np.cos(phi)
        data[i, 0] += -40 * gain * gauss(0.08, 0.03)
    return TrialSet(data=data, times=TIMES, sfreq=SFREQ, ch_names=["c0"])


def test_req155_prestimulus_phase():

    false = []
    for seed in range(40):
        tf = _run(_phase_trials(0.0, seed), r2_min=0.0, covariates=("prestim",))
        false.append(prestimulus_dependence(tf, n_permutations=200, random_state=0)
                     .iloc[0]["p_value"] <= 0.05)
    assert np.mean(false) <= 0.06 + 2 * np.sqrt(0.05 * 0.95 / 40)
    hits = []
    for seed in range(10):
        tf = _run(_phase_trials(0.3, 1000 + seed), r2_min=0.0, covariates=("prestim",))
        hits.append(prestimulus_dependence(tf, n_permutations=200, random_state=0)
                    .iloc[0]["p_value"] <= 0.05)
    assert np.mean(hits) >= 0.8
