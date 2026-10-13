# SPDX-License-Identifier: Apache-2.0
"""Speed of every artifact-removal method (REQ-068).

Run with ``python benchmarks/bench_artifact.py``. Target, on one core: at most 2 s for
50 trials x 128 channels x 3 s at 2048 Hz, except ``lowrank`` and ``subspace`` (20 s).
``evoparam/tests/artifact/test_performance.py`` checks the same numbers when
EVOPARAM_PERF_TESTS=1.
"""

import contextlib
import time
import warnings

import numpy as np
import pandas as pd

from evoparam.artifact import remove_artifact
from evoparam.data import TrialSet

LIMITS = {"lowrank": 20.0, "subspace": 20.0}
CASES = [
    ("interpolate", {}),
    ("template", {}),
    ("template", {"template": "local"}),
    ("learned", {}),
    ("tail", {"tail": "subtract"}),
    ("polynomial", {}),
    ("matching_pursuit", {}),
    ("period_template", {"period": 1 / 130.0}),
    ("paradigm", {"scheme": "alternating"}),
    ("subspace", {}),
    ("lowrank", {"pulse_times": np.arange(0, 0.04, 0.01)}),
]


def make_data(n_trials=50, n_channels=128, duration=3.0, sfreq=2048.0, seed=0):
    """Simulate the benchmark data."""
    rng = np.random.default_rng(seed)
    times = np.arange(int(-1.0 * sfreq), int((duration - 1.0) * sfreq)) / sfreq
    data = rng.normal(0, 5, (n_trials, n_channels, len(times)))
    i0 = np.searchsorted(times, 0.0)
    gains = rng.uniform(0.3, 1.0, (1, n_channels, 1))
    data[:, :, i0 : i0 + 4] += gains * np.array([3000.0, -2000.0, 800.0, -200.0])
    meta = pd.DataFrame(
        {
            "stim_site": ["A-B"] * n_trials,
            "stim_polarity": ["monophasic_cathodic", "monophasic_anodic"] * (n_trials // 2),
        }
    )
    return TrialSet(
        data=data,
        times=times,
        sfreq=sfreq,
        ch_names=[f"c{i:03d}" for i in range(n_channels)],
        metadata=meta,
    )


def run(ts):
    """Time every artifact-removal case on ``ts``; returns (label, method, s, limit) rows.

    Each case first runs once on a small set of the same layout, untimed: the modules a
    method imports on first use (scipy.signal for ``polynomial``, about 1 s) are not the
    method's runtime (REQ-068); the import of the package is measured by REQ-290 P6.
    """
    small = make_data(n_trials=4, n_channels=8, duration=3.0, sfreq=ts.sfreq)
    rows = []
    for method, kwargs in CASES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # a warm-up only: the timed run below reports the errors
            with contextlib.suppress(Exception):
                remove_artifact(small, method=method, **kwargs)
            start = time.perf_counter()
            remove_artifact(ts, method=method, **kwargs)
            elapsed = time.perf_counter() - start
        label = method + ("" if not kwargs else f" {list(kwargs)[0]}")
        rows.append((label, method, elapsed, LIMITS.get(method, 2.0)))
    return rows


if __name__ == "__main__":
    for label, _, elapsed, limit in run(make_data()):
        print(f"{label:28s} {elapsed:6.2f} s   (limit {limit:g} s)")
