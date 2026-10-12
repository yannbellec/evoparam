# SPDX-License-Identifier: Apache-2.0
"""Synthetic continuous recordings with known stimulation pulses, for the events tests."""

import numpy as np

###################################################################################################
###################################################################################################

# Biphasic artifact shape, one value per sample from the onset (normalised to a peak of 1).
PULSE_SHAPE = np.array([0.45, 1.0, 0.7, -0.5, -0.25, -0.08])


def make_noise(
    n_channels, duration, sfreq, sd=1.0, line_hz=None, line_amp=0.0, seed=0, dtype=np.float32
):
    """Gaussian background noise, with an optional line-noise sinusoid on every channel."""

    rng = np.random.default_rng(seed)
    n_times = int(round(duration * sfreq))
    data = rng.normal(0, sd, (n_channels, n_times)).astype(dtype)
    if line_hz:
        times = np.arange(n_times) / sfreq
        phases = rng.uniform(0, 2 * np.pi, n_channels)
        data += (line_amp * np.sin(2 * np.pi * line_hz * times[None] + phases[:, None])).astype(
            dtype
        )
    return data


def add_pulse(data, onset_sample, amplitude, gains, polarity=1.0, shape=PULSE_SHAPE):
    """Add one artifact: ``amplitude * gains[c] * polarity * shape`` from ``onset_sample``."""

    stop = min(data.shape[1], onset_sample + len(shape))
    seg = shape[: stop - onset_sample]
    data[:, onset_sample:stop] += (polarity * amplitude * gains[:, None] * seg[None]).astype(
        data.dtype
    )


def spatial_gains(n_channels, stim_pair, rng, floor=0.3):
    """Artifact gain per channel: 1 on the stimulated pair, decaying with index distance,
    and never below ``floor`` (volume conduction reaches every contact)."""

    distance = np.min(np.abs(np.arange(n_channels)[:, None] - np.asarray(stim_pair)[None]), axis=1)
    return np.maximum(floor, 1.0 / (1.0 + distance)) * rng.uniform(0.9, 1.1, n_channels)


def make_recording(
    n_channels=16,
    sfreq=2048.0,
    duration=20.0,
    interval=1.0,
    first=0.5,
    amplitude=(200.0, 400.0),
    line_hz=None,
    line_amp=0.0,
    n_transients=0,
    transient_sd=5.0,
    seed=0,
):
    """Noise plus regularly spaced pulses (and optional movement transients).

    Returns
    -------
    data : 2d array [n_channels, n_times]
    onsets : 1d array of int, onset samples of the pulses
    transients : 1d array of int, onset samples of the transients
    """

    rng = np.random.default_rng(seed)
    data = make_noise(
        n_channels, duration, sfreq, line_hz=line_hz, line_amp=line_amp, seed=seed + 1
    )
    n_times = data.shape[1]
    onsets = np.arange(first, duration - 0.1, interval)
    onsets = np.round(onsets * sfreq).astype(int)
    gains = spatial_gains(n_channels, (n_channels // 2, n_channels // 2 + 1), rng)
    for onset in onsets:
        add_pulse(data, onset, rng.uniform(*amplitude), gains)

    transients = []
    width = int(round(0.03 * sfreq))
    bump = np.hanning(2 * width)
    for _ in range(n_transients):
        while True:
            start = int(rng.integers(width, n_times - 3 * width))
            if np.min(np.abs(onsets - start)) > 0.1 * sfreq:
                break
        # A movement transient: a smooth bump of transient_sd on most channels
        chans = rng.random(n_channels) < 0.8
        data[chans, start : start + 2 * width] += (transient_sd * bump).astype(data.dtype)
        transients.append(start)
    return data, onsets, np.array(transients, dtype=int)
