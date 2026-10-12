# SPDX-License-Identifier: Apache-2.0
"""Synthetic epochs for the preparation tests."""

from typing import Any

import numpy as np

from evoparam.data import TrialSet

###################################################################################################
###################################################################################################


def epochs(
    n_trials: Any = 20,
    names: Any = None,
    sfreq: Any = 1000.0,
    tmin: Any = -0.5,
    tmax: Any = 1.0,
    noise: Any = 5.0,
    seed: Any = 0,
    event_times: Any = None,
    metadata: Any = None,
    ch_info: Any = None,
) -> Any:
    rng = np.random.default_rng(seed)
    names = names or [f"A{i}" for i in range(1, 9)]
    times = np.arange(int(round(tmin * sfreq)), int(round(tmax * sfreq))) / sfreq
    data = rng.normal(0, noise, (n_trials, len(names), len(times)))
    return TrialSet(
        data=data,
        times=times,
        sfreq=sfreq,
        ch_names=names,
        event_times=event_times,
        metadata=metadata,
        ch_info=ch_info,
    )


def ccep(times: Any, latency: Any = 0.025, amplitude: Any = -100.0, width: Any = 0.008) -> Any:
    return amplitude * np.exp(-0.5 * ((times - latency) / width) ** 2)


def sim_carla(
    fraction: Any, seed: Any, n_channels: Any = 50, n_trials: Any = 12, sfreq: Any = 1000.0
) -> Any:
    """Simulation of the adjusted-common-average paper: 50 channels, 12 trials, a fraction
    of responsive channels with random CCEP shapes (N1 and N2, either polarity)."""

    from scipy.signal import lfilter

    rng = np.random.default_rng(seed)
    times = np.arange(-0.5, 1.0, 1 / sfreq)
    data = (
        lfilter([1], [1, -0.9], rng.normal(0, 1, (n_trials, n_channels, len(times))), axis=-1)
        * 0.5
    )
    responsive = rng.choice(n_channels, int(round(fraction * n_channels)), replace=False)
    for ch in responsive:
        wave = -rng.uniform(5, 20) * np.exp(
            -0.5 * ((times - rng.uniform(0.015, 0.04)) / 0.008) ** 2
        ) - rng.uniform(3, 10) * np.exp(-0.5 * ((times - rng.uniform(0.08, 0.25)) / 0.04) ** 2)
        if rng.random() < 0.5:
            wave = -wave
        data[:, ch] += wave * rng.uniform(0.8, 1.2, (n_trials, 1))
    names = [f"S{i:02d}" for i in range(n_channels)]
    return TrialSet(data=data, times=times, sfreq=sfreq, ch_names=names), set(responsive)
