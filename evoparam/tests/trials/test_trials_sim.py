# SPDX-License-Identifier: Apache-2.0
"""Single-trial estimation on simulations (REQ-150 to REQ-155)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from evoparam.data import GroupFit, TrialSet
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
    """REQ-150 at the level reached: per-trial SNR 3, medians over six runs, the excess jitter
    (``jitter_sd_excess``). The REQ as written (SNR 2, every run, ``jitter_sd``) is
    ``test_req150_as_written``."""

    ratios, lat, amp = [], [], []
    for seed in range(6):
        ts, shifts, scales = simulate(jitter=sd, seed=seed)  # per-trial SNR 3
        tf = _run(ts, max_shift=0.015, r2_min=0.0)
        fit = tf.fits[0]
        lat.append(np.mean(np.abs(fit.shift[:, 0] - (shifts - shifts.mean()))) * SFREQ)
        amp.append(np.mean(np.abs(fit.amplitude[:, 0] / (-30 * scales) - 1)))
        ratios.append(jitter(tf).iloc[0]["jitter_sd_excess"] / np.std(shifts, ddof=1))
    assert np.median(lat) <= 1.0
    assert np.median(amp) <= 0.15
    assert abs(np.median(ratios) - 1) <= 0.2


# REQ-150 as written: per-trial SNR 2 (the lowest SNR the REQ covers), jitter_sd 1, 2 and 5 ms,
# amplitudes with CV 30 %, every run. The width is the one that balances the two Cramer-Rao
# limits below (FWHM 11 samples; 1 kHz as the other tests).
FWHM_150 = 0.011
AS_WRITTEN_150 = (
    "REQ-150 at SNR 2 is out of reach: no width makes the latency MAE <= 1 sample and the "
    "amplitude relative MAE <= 15 % together (test_req150_cramer_rao_bound). Measured with "
    "shift_scale at FWHM 11 samples, six runs: see CHANGELOG (trials, F36)."
)


@pytest.mark.xfail(strict=True, reason=AS_WRITTEN_150)
@pytest.mark.parametrize("sd", [0.001, 0.002, 0.005])
def test_req150_as_written(sd):

    for seed in range(6):
        ts, shifts, scales = simulate(
            jitter=sd, amp=-20.0, noise=10.0, fwhm=FWHM_150, seed=5000 + seed
        )  # per-trial SNR 2
        tf = _run(ts, max_shift=max(0.015, 3 * sd), r2_min=0.0)
        fit = tf.fits[0]
        lat = np.mean(np.abs(fit.shift[:, 0] - (shifts - shifts.mean()))) * SFREQ
        amp = np.mean(np.abs(fit.amplitude[:, 0] / (-20 * scales) - 1))
        ratio = fit.jitter_sd[fit.settings["picks"][0]] / np.std(shifts, ddof=1)
        assert lat <= 1.0
        assert amp <= 0.15
        assert abs(ratio - 1) <= 0.2


def _cramer_rao(fwhm, snr=2.0, n_win=491):
    """Cramer-Rao SDs of the latency (samples) and of the relative amplitude of a Gaussian of
    unit amplitude at ``snr`` (peak over the SD of white noise), with a per-trial constant
    and slope (the linear background of the trial model), on ``n_win`` samples."""

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    t = np.arange(n_win) - n_win / 3.0
    g = np.exp(-0.5 * (t / sigma) ** 2)
    design = np.column_stack([g, snr * g * t / sigma**2, np.ones(n_win), t / n_win])
    cov = np.linalg.inv(design.T @ design)
    return np.sqrt(cov[1, 1]), np.sqrt(cov[0, 0]) / snr


def test_req150_cramer_rao_bound():
    """Evidence for the xfail above: even an efficient unbiased estimator (Gaussian errors,
    MAE = sqrt(2/pi) SD) cannot meet both limits of REQ-150 at SNR 2 for any width. The
    latency SD grows as sqrt(width) and the amplitude SD falls as 1/sqrt(width); their
    product does not depend on the width. Both errors scale as 1/a for a trial of amplitude
    a, so their means carry E[1/a] = exp(log(1 + CV^2)) = 1.09 for log-normal amplitudes of
    CV 30 % (the simulator of spec 5.12), and more for normal ones."""

    factor = np.sqrt(2 / np.pi) * (1 + 0.3**2)
    worst = []
    for fwhm in np.arange(2.0, 60.0, 0.05):  # samples
        lat_sd, amp_sd = _cramer_rao(fwhm)
        worst.append(max(factor * lat_sd / 1.0, factor * amp_sd / 0.15))
    # the best width misses one of the limits by 1.6 % (FWHM 11.35 samples)
    assert min(worst) > 1.0
    assert np.argmin(worst) == pytest.approx((11.35 - 2.0) / 0.05, abs=2)


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


def _req151_runs(seeds, n_trials, null_rule):
    ratios, refused = [], 0
    for seed in seeds:
        ts, _, _ = simulate(n=n_trials, jitter=0.0, cv=0.0, seed=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tf = fit_trials(ts, channel_fit(ts), realign=True, null_rule=null_rule)
        fit = tf.fits[0]
        ratios.append(fit.jitter_sd_null[0] / fit.jitter_sd[0])
        refused += fit.realigned is None and any("refused" in str(w.message) for w in caught)
    return np.array(ratios), refused / len(seeds)


@pytest.mark.xfail(
    strict=True,
    reason="REQ-151 in every run with 40 trials per run: the SD "
    "of 40 shifts varies by about 11 % from run to run, so jitter_sd exceeds "
    "1.25 jitter_sd_null in a few runs (seed 104: ratio 0.768; 7 of 200 runs "
    "below 0.8, smallest 0.756). See CHANGELOG (trials, F37).",
)
def test_req151_as_written_40_trials():

    ratios, refused = _req151_runs(range(100, 120), 40, "segments")
    assert np.all(ratios >= 0.8)
    assert refused >= 0.95


@pytest.mark.parametrize("null_rule", ["segments", "spec"])
def test_req151_as_written(null_rule):
    """REQ-151 as written, in every run (20 runs), with 200 trials per run, where the SD of
    the shifts of one run is known to about 5 %; under both null rules (the default five
    baseline segments per trial with the F-quantile refusal, and the spec's one segment with
    the 1.2 x rule)."""

    ratios, refused = _req151_runs(range(2000, 2020), 200, null_rule)
    assert np.all(ratios >= 0.8)
    assert refused >= 0.95


def test_req152_jitter_corrected_amplitude():

    ts, shifts, _ = simulate(n=100, jitter=0.006, cv=0.0, noise=3.0, fwhm=0.03, seed=7)
    tf = _run(ts, max_shift=0.02, r2_min=0.0)
    corrected = jitter(tf).iloc[0]["amplitude_jitter_corrected"]
    assert abs(corrected / -30 - 1) <= 0.10


@pytest.mark.parametrize(
    ("ridge_cv", "trial_model"), [("loto", "all"), ("split", "all"), ("loto", "selected")]
)
def test_req153_regularized_beats_free_at_low_snr(ridge_cv, trial_model):
    """REQ-153 with the default penalty choice (leave-one-trial-out, the spec's) and split
    samples, and with the spec's trial model (selected components only)."""

    better = 0
    for seed in range(10):
        ts, shifts, _ = simulate(jitter=0.003, amp=-30.0, noise=30.0, seed=200 + seed)
        group = channel_fit(
            TrialSet(data=ts.data - 0, times=ts.times, sfreq=ts.sfreq, ch_names=ts.ch_names)
        )
        if group.fits[0].status != "ok":
            continue
        errs = {}
        for mode in ("free", "regularized"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = fit_trials(
                    ts, group, mode=mode, r2_min=0.0, ridge_cv=ridge_cv, trial_model=trial_model
                ).fits[0]
            errs[mode] = np.mean(np.abs(fit.shift[:, 0] - (shifts - shifts.mean())))
        better += errs["regularized"] <= errs["free"]
    assert better / 10 >= 0.9


def test_req154_dose_response_hill():

    rng = np.random.default_rng(8)
    currents = np.repeat([1.0, 2.0, 3.0, 5.0, 8.0], 10)
    i50, h = 3.0, 2.5
    amps = -60 * currents**h / (i50**h + currents**h)
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


def _phase_block(seed, n_runs, n=100):
    """``n_runs`` independent runs of ``_phase_trials(0.0, ...)`` as the channels of one
    TrialSet (each channel its own phases and noise), so that one fit_trials call scores
    them all."""

    rng = np.random.default_rng(seed)
    phases = rng.uniform(-np.pi, np.pi, (n, n_runs))
    data = rng.normal(0, 1.0, (n, n_runs, len(TIMES)))
    pre = TIMES < 0
    data[:, :, pre] += 5 * np.cos(
        2 * np.pi * 10 * (TIMES[pre] + 0.01)[None, None] + phases[..., None]
    )
    data += -40 * gauss(0.08, 0.03)[None, None]
    return TrialSet(data=data, times=TIMES, sfreq=SFREQ, ch_names=[f"c{i}" for i in range(n_runs)])


def test_req155_type_one_error():
    """REQ-155 type-I error <= 0.06 at alpha 0.05 as written, on 1000 runs of 100 trials with
    no phase dependence (the binomial SD of the rate is then 0.007, so 0.06 is 1.4 SD above
    the nominal 0.05). Every run uses the channel model of one pilot run of the same design
    (fitted once), so that the runs differ only by their trials."""

    pilot = channel_fit(_phase_trials(0.0, 99_999)).fits[0]
    hits = []
    for block in range(10):
        ts = _phase_block(10_000 + block, 100)
        group = GroupFit.from_fits([pilot] * len(ts.ch_names), ts.ch_names)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tf = fit_trials(ts, group, r2_min=0.0, covariates=("prestim",))
        table = prestimulus_dependence(tf, n_permutations=200, random_state=block)
        assert len(table) == 100
        hits += list(table["p_value"] <= 0.05)
    assert np.mean(hits) <= 0.06


def test_req155_power():
    """REQ-155 power >= 0.8 for a 30 % modulation with 100 trials (the type-I part is
    test_req155_type_one_error)."""

    hits = []
    for seed in range(10):
        tf = _run(_phase_trials(0.3, 1000 + seed), r2_min=0.0, covariates=("prestim",))
        hits.append(
            prestimulus_dependence(tf, n_permutations=200, random_state=0).iloc[0]["p_value"]
            <= 0.05
        )
    assert np.mean(hits) >= 0.8
