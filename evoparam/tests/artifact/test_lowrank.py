# SPDX-License-Identifier: Apache-2.0
"""Sequential low-rank regression (REQ-063)."""

import numpy as np
import pytest

from evoparam.artifact import remove_artifact
from evoparam.data import TrialSet
from evoparam.errors import ArtifactError

###################################################################################################
###################################################################################################

SFREQ = 10000.0


def spike(sfreq):
    t = np.arange(int(0.0015 * sfreq)) / sfreq
    return -60 * np.sin(np.pi * t / t[-1]) ** 2 * np.cos(2 * np.pi * t / t[-1] * 0.8)


def make_train(n_channels=24, n_pulses=20, n_trials=10, interval=0.01, seed=0):
    """An artifact train shared (low-rank) across channels, pulses and trials, and evoked
    spikes on 6 channels at jittered latencies (not locked to the pulses)."""

    rng = np.random.default_rng(seed)
    times = np.arange(int(-0.1 * SFREQ), int(0.35 * SFREQ)) / SFREQ
    n_times = len(times)
    i0 = np.searchsorted(times, 0.0)
    step = int(round(interval * SFREQ))
    rel = np.arange(step) / SFREQ
    shapes = np.array(
        [np.exp(-rel / 0.0008) * np.sin(2 * np.pi * rel / 0.002), np.exp(-rel / 0.003)]
    )
    mixing = rng.normal(0, 1, (n_channels, 2)) * np.array([800.0, 200.0])

    clean = rng.normal(0, 5, (n_trials, n_channels, n_times))
    spikes = []
    active = rng.choice(n_channels, 6, replace=False)
    wave = spike(SFREQ)
    for trial in range(n_trials):
        for pulse in range(n_pulses):
            for ch in active:
                if rng.random() < 0.5:
                    at = i0 + pulse * step + int(rng.uniform(0.002, 0.008) * SFREQ)
                    clean[trial, ch, at : at + len(wave)] += wave
                    spikes.append((trial, ch, at))
    noisy = clean.copy()
    for trial in range(n_trials):
        for pulse in range(n_pulses):
            gain = rng.uniform(0.95, 1.05) * (1 + 0.02 * pulse)
            start = i0 + pulse * step
            noisy[trial, :, start : start + step] += gain * (mixing @ shapes)
    names = [f"e{i:02d}" for i in range(n_channels)]

    def make(data):
        return TrialSet(data=data, times=times, sfreq=SFREQ, ch_names=names)

    silent = np.setdiff1d(np.arange(n_channels), active)
    return make(noisy), make(clean), spikes, silent, np.arange(n_pulses) * interval


def test_req063_spikes_kept_and_silent_channels_clean():

    noisy, truth, spikes, silent, pulse_times = make_train()
    result = remove_artifact(
        noisy,
        method="lowrank",
        pulse_times=pulse_times,
        baseline=(-0.09, -0.01),
        window=(0.0, 0.0005),
        quality=False,
    )
    cleaned = result.cleaned.data
    length = len(spike(SFREQ))
    got = np.mean([cleaned[t, c, a : a + length] for t, c, a in spikes], axis=0)
    ref = np.mean([truth.data[t, c, a : a + length] for t, c, a in spikes], axis=0)
    assert np.corrcoef(got, ref)[0, 1] >= 0.95

    train = (noisy.times >= 0) & (noisy.times < pulse_times[-1] + 0.01)
    pre = noisy.times < 0
    for ch in silent:
        during = np.sqrt(np.mean(cleaned[:, ch][:, train] ** 2))
        before = np.sqrt(np.mean(cleaned[:, ch][:, pre] ** 2))
        assert abs(during / before - 1) <= 0.10
    assert 0 < result.settings["_info"]["removed_fraction"] <= 1


def test_lowrank_requirements():

    noisy, _, _, _, pulse_times = make_train(n_channels=8, n_trials=4, n_pulses=4)
    with pytest.raises(ArtifactError, match="channels"):
        remove_artifact(
            noisy.select(picks=list(range(7))),
            method="lowrank",
            pulse_times=pulse_times,
            baseline=(-0.09, -0.01),
            quality=False,
        )
    with pytest.raises(ArtifactError, match="pulses"):
        remove_artifact(
            noisy,
            method="lowrank",
            pulse_times=pulse_times[:3],
            baseline=(-0.09, -0.01),
            quality=False,
        )
    with pytest.raises(ArtifactError, match="trials"):
        remove_artifact(
            noisy.select(trials=[0, 1, 2]),
            method="lowrank",
            pulse_times=pulse_times,
            baseline=(-0.09, -0.01),
            quality=False,
        )
    with pytest.raises(ArtifactError, match="pulse_times or expected_rate"):
        remove_artifact(noisy, method="lowrank", baseline=(-0.09, -0.01), quality=False)
