# SPDX-License-Identifier: Apache-2.0
"""Speed of the time-frequency parameterisation (REQ-218).

Run with ``python benchmarks/bench_tf.py``. Target, with one job: at most 60 s for
100 channels x 50 trials x 2 048 samples x 40 frequencies with ``stats="fdr"``.
``evoparam/tests/tf/test_performance.py`` checks the same number when EVOPARAM_PERF_TESTS=1.
"""

import time
import warnings

import numpy as np

from evoparam.data import TrialSet
from evoparam.tf import tf_fit


def make_data(n_trials=50, n_channels=100, n_times=2048, sfreq=2048.0, seed=0):
    rng = np.random.default_rng(seed)
    times = (np.arange(n_times) - n_times // 2) / sfreq
    data = rng.normal(0, 10, (n_trials, n_channels, n_times))
    # a response on a tenth of the channels, so that zones are found and measured
    burst = np.where(np.abs(times - 0.2) < 0.05, 1.0, 0.0) * np.cos(2 * np.pi * 80 * times)
    data[:, : n_channels // 10] += 15 * burst
    return TrialSet(
        data=data, times=times, sfreq=sfreq, ch_names=[f"c{i}" for i in range(n_channels)]
    )


def run(transform="morse", stats="fdr", zones="auto"):
    ts = make_data()
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tf = tf_fit(
            ts,
            transform=transform,
            freqs=(2.0, 250.0, 40),
            window=(0.0, 0.45),
            baseline=(-0.45, -0.05),
            stats=stats,
            zones=zones,
            random_state=0,
        )
    return time.perf_counter() - start, tf


if __name__ == "__main__":
    for transform in ("morse", "morlet", "multitaper", "stft"):
        duration, tf = run(transform)
        print(f"{transform:>10}: {duration:6.1f} s  ({len(tf.zones)} zones)")
