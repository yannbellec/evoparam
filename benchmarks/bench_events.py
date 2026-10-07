"""Speed and memory of pulse detection (REQ-045).

Run with ``python benchmarks/bench_events.py``. Target: at most 1 s per 10 minutes of
100-channel data at 2048 Hz on one core, and at most 2 times the data size in extra
memory. ``evoparam/tests/events/test_performance.py`` checks the same numbers when
EVOPARAM_PERF_TESTS=1.
"""

import time
import tracemalloc

import numpy as np

from evoparam.events import detect_pulses


def make_data(n_channels=100, sfreq=2048., duration=600., interval=5., seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 10, (n_channels, int(duration * sfreq)))
    shape = np.array([900., 2000., 1400., -1000., -500., -150.])
    gains = rng.uniform(0.3, 1.0, n_channels)[:, None]
    for onset in np.arange(1.0, duration - 1.0, interval):
        ind = int(onset * sfreq)
        data[:, ind:ind + len(shape)] += gains * shape
    return data, sfreq


def measure(data, sfreq, **kwargs):
    """Wall time (s, best of 3) and peak extra memory (bytes) of detect_pulses.

    Memory tracing slows numpy down, so time and memory are measured in separate runs.
    """

    detect_pulses(data[:, :int(2 * sfreq)], sfreq, method="threshold")      # warm-up
    times = []
    for _ in range(3):
        start = time.perf_counter()
        events = detect_pulses(data, sfreq, **kwargs)
        times.append(time.perf_counter() - start)
    tracemalloc.start()
    detect_pulses(data, sfreq, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return min(times), peak, events


if __name__ == "__main__":
    data, sfreq = make_data()
    names = [f"C{i:03d}" for i in range(data.shape[0])]
    for label, kwargs in (("default", {}), ("with site guess", {"ch_names": names})):
        elapsed, peak, events = measure(data, sfreq, **kwargs)
        print(f"{label:16s} {elapsed:6.2f} s   peak extra memory {peak / data.nbytes:4.2f} x data"
              f"   {len(events)} pulses")
