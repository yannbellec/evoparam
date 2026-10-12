# SPDX-License-Identifier: Apache-2.0
"""Overlay benchmark for the artifact tests: artifact-free epochs with known responses,
plus an artifact of known shape (spec §5.3 acceptance criteria)."""

import numpy as np
import pandas as pd

from evoparam.data import TrialSet

###################################################################################################
###################################################################################################


def times_axis(sfreq, tmin=-0.5, tmax=0.5):
    return np.arange(int(round(tmin * sfreq)), int(round(tmax * sfreq))) / sfreq


def response(times, onset=0.012, amplitude=-80.0, width=0.004):
    """A negative component that starts at ``onset`` (gamma-like rise, smooth decay)."""

    rel = np.clip(times - onset, 0, None)
    shape = (rel / width) ** 2 * np.exp(-rel / width)
    return amplitude * shape / shape.max()


def artifact_shape(times, sfreq, amplitude=3000.0, tail=500.0, tau=0.002):
    """Sharp biphasic pulse at t = 0 and an exponential recovery tail."""

    art = np.zeros_like(times)
    i0 = int(np.searchsorted(times, 0.0))
    pulse = np.array([1.0, -0.7, 0.3, -0.1])
    width = max(1, int(round(0.0005 * sfreq)))
    pulse = np.repeat(pulse, width)
    art[i0 : i0 + len(pulse)] = amplitude * pulse
    start = i0 + len(pulse)
    art[start:] += tail * np.exp(-(times[start:] - times[start]) / tau)
    return art


def make_overlay(
    sfreq=2048.0,
    n_trials=20,
    n_channels=8,
    onset=0.012,
    amplitude=-80.0,
    width=0.004,
    noise=5.0,
    art_amplitude=3000.0,
    art_tail=500.0,
    art_tau=0.002,
    jitter=0.1,
    channel_jitter=True,
    seed=0,
    metadata=True,
):
    """Return (with_artifact, truth) TrialSets that differ only by the artifact.

    With ``channel_jitter`` the artifact size varies independently per trial and channel;
    without it, it varies per trial only (a spatially fixed artifact, as from one
    stimulation site).
    """

    rng = np.random.default_rng(seed)
    times = times_axis(sfreq)
    gains = rng.uniform(0.5, 1.5, n_channels)
    clean = (
        rng.normal(0, noise, (n_trials, n_channels, len(times)))
        + gains[None, :, None] * response(times, onset, amplitude, width)[None, None]
    )
    art = artifact_shape(times, sfreq, art_amplitude, art_tail, art_tau)
    scale = rng.uniform(1 - jitter, 1 + jitter, (n_trials, n_channels if channel_jitter else 1, 1))
    spatial = rng.uniform(0.3, 1.0, (1, n_channels, 1))
    noisy = clean + scale * spatial * art[None, None]
    meta = None
    if metadata:
        meta = pd.DataFrame(
            {
                "stim_site": ["C1-C2"] * n_trials,
                "stim_current_mA": [4.0] * n_trials,
                "stim_polarity": ["biphasic_cathodic_first"] * n_trials,
            }
        )
    names = [f"ch{i:02d}" for i in range(n_channels)]

    def make(data):
        return TrialSet(
            data=data,
            times=times,
            sfreq=sfreq,
            ch_names=names,
            metadata=None if meta is None else meta.copy(),
            event_times=np.arange(n_trials) * 5.0,
        )

    return make(noisy), make(clean)
