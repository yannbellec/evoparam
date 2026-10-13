# SPDX-License-Identifier: Apache-2.0
"""Synthetic continuous recordings with known stimulation pulses, for the events tests."""

from dataclasses import dataclass
from typing import Any

import numpy as np

# Fraction of the channels a movement transient reaches.
TRANSIENT_FRACTION = 0.8

###################################################################################################
###################################################################################################

# Biphasic artifact shape, one value per sample from the onset (normalised to a peak of 1).
PULSE_SHAPE = np.array([0.45, 1.0, 0.7, -0.5, -0.25, -0.08])


def make_noise(
    n_channels: Any,
    duration: Any,
    sfreq: Any,
    sd: Any = 1.0,
    line_hz: Any = None,
    line_amp: Any = 0.0,
    seed: Any = 0,
    dtype: Any = np.float32,
) -> Any:
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


def add_pulse(
    data: Any,
    onset_sample: Any,
    amplitude: Any,
    gains: Any,
    polarity: Any = 1.0,
    shape: Any = PULSE_SHAPE,
) -> None:
    """Add one artifact: ``amplitude * gains[c] * polarity * shape`` from ``onset_sample``."""

    stop = min(data.shape[1], onset_sample + len(shape))
    seg = shape[: stop - onset_sample]
    data[:, onset_sample:stop] += (polarity * amplitude * gains[:, None] * seg[None]).astype(
        data.dtype
    )


def spatial_gains(n_channels: Any, stim_pair: Any, rng: Any, floor: Any = 0.3) -> Any:
    """Artifact gain per channel: 1 on the stimulated pair, decaying with index distance,
    and never below ``floor`` (volume conduction reaches every contact)."""

    distance = np.min(np.abs(np.arange(n_channels)[:, None] - np.asarray(stim_pair)[None]), axis=1)
    return np.maximum(floor, 1.0 / (1.0 + distance)) * rng.uniform(0.9, 1.1, n_channels)


@dataclass(frozen=True)
class RecordingDesign:
    """The keywords of :func:`make_recording`: the simulated continuous recording."""

    n_channels: Any = 16
    sfreq: Any = 2048.0
    duration: Any = 20.0
    interval: Any = 1.0
    first: Any = 0.5
    amplitude: Any = (200.0, 400.0)
    line_hz: Any = None
    line_amp: Any = 0.0
    n_transients: Any = 0
    transient_sd: Any = 5.0
    seed: Any = 0


def make_recording(**keywords: Any) -> Any:
    """Noise plus regularly spaced pulses (and optional movement transients).

    Returns
    -------
    data : 2d array [n_channels, n_times]
    onsets : 1d array of int, onset samples of the pulses
    transients : 1d array of int, onset samples of the transients

    Keywords: the fields of :class:`RecordingDesign`.
    """

    design = RecordingDesign(**keywords)
    rng = np.random.default_rng(design.seed)
    data = make_noise(
        design.n_channels,
        design.duration,
        design.sfreq,
        line_hz=design.line_hz,
        line_amp=design.line_amp,
        seed=design.seed + 1,
    )
    n_times = data.shape[1]
    onsets = np.arange(design.first, design.duration - 0.1, design.interval)
    onsets = np.round(onsets * design.sfreq).astype(int)
    gains = spatial_gains(
        design.n_channels, (design.n_channels // 2, design.n_channels // 2 + 1), rng
    )
    for onset in onsets:
        add_pulse(data, onset, rng.uniform(*design.amplitude), gains)

    transients = []
    width = int(round(0.03 * design.sfreq))
    bump = np.hanning(2 * width)
    for _ in range(design.n_transients):
        while True:
            start = int(rng.integers(width, n_times - 3 * width))
            if np.min(np.abs(onsets - start)) > 0.1 * design.sfreq:
                break
        # A movement transient: a smooth bump of transient_sd on most channels
        chans = rng.random(design.n_channels) < TRANSIENT_FRACTION
        data[chans, start : start + 2 * width] += (design.transient_sd * bump).astype(data.dtype)
        transients.append(start)
    return data, onsets, np.array(transients, dtype=int)
