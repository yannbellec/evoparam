"""Latency of the online mode (REQ-324, REQ-347).

Run with ``OMP_NUM_THREADS=1 python benchmarks/stream_latency.py``. A simulated stream of
64 channels at 2 048 Hz with one pulse per second (stimulation artifact on every channel, a
response on half of them) goes through :class:`evoparam.stream.StreamFitter` in chunks of
32 samples: threshold detection of the pulses, a 0.5 s response window (0 to 0.5 s, the
artifact window interpolated), ``max_shift`` 5 ms, the running existence test. The latency
of a pulse is the wall time of the ``push`` call that delivers the last sample of its window
(detection, fit, existence and the estimates included). Target: p99 < 20 ms on one core of
the reference machine (``benchmarks/README.md``). The response has one component (N1) or,
with ``n_components=2``, two (N1 and N2, jittered independently), fitted in each ``mode``
of the fitter.
``evoparam/tests/stream/test_stream.py`` checks the same number when EVOPARAM_PERF_TESTS=1.
"""

import os
import platform
import time
import warnings

import numpy as np

from evoparam.data import TrialSet
from evoparam.models import fit as fit_models
from evoparam.stream import StreamFitter

SFREQ = 2048.0
N_CHANNELS = 64
PERIOD = 1.0                    # s between pulses
ARTIFACT = (-0.002, 0.008)
WINDOW = (0.0, 0.5)             # a 0.5 s response window (REQ-324), artifact interpolated
N1 = (-40.0, 0.03, 0.02)        # amplitude (uV), latency (s), FWHM (s)
N2 = (30.0, 0.12, 0.05)


def gauss(times, latency, fwhm):
    return np.exp(-0.5 * ((times - latency) / (fwhm / 2.3548)) ** 2)


def template(seed=0, n_components=1):
    """Channel fit of the simulated response (ccep preset), from 40 trials."""

    rng = np.random.default_rng(seed)
    times = np.arange(-1024, 2048) / SFREQ
    data = rng.normal(0, 10, (40, 1, len(times)))
    for amp, lat, fwhm in (N1, N2)[:n_components]:
        data[:, 0] += amp * gauss(times, lat, fwhm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = fit_models(TrialSet(data=data, times=times, sfreq=SFREQ, ch_names=["c0"]),
                           "ccep")
    return group.fits[0]


def stream_blocks(n_pulses, n_channels=N_CHANNELS, seed=1, n_components=1):
    """One block of samples per pulse period: noise, artifact spike at the onset, response
    (jittered, scaled) on the first half of the channels. Yields (block, onset sample)."""

    rng = np.random.default_rng(seed)
    n = int(PERIOD * SFREQ)
    onset = int(0.45 * SFREQ)
    rel = (np.arange(n) - onset) / SFREQ
    for _ in range(n_pulses):
        block = rng.normal(0, 10, (n, n_channels))
        block[onset:onset + 3] += np.array([3000.0, -2000.0, 500.0])[:, None]
        for amp, lat, fwhm in (N1, N2)[:n_components]:
            shift = rng.normal(0, 0.0015)
            scale = 1 + 0.2 * rng.normal()
            block[:, :n_channels // 2] += (scale * amp * gauss(rel, lat + shift, fwhm))[:, None]
        yield block, onset


def run(n_pulses=1000, chunk_size=32, n_channels=N_CHANNELS, fit=None, seed=1, n_components=1,
        mode="shift_scale"):
    """Replay ``n_pulses`` simulated pulses; returns the latency summary (ms) and the fitter."""

    fit = template(n_components=n_components) if fit is None else fit
    names = [f"c{i}" for i in range(n_channels)]
    fitter = StreamFitter(fit, sfreq=SFREQ, ch_names=names, window=WINDOW,
                          artifact_window=ARTIFACT, max_shift=0.005,
                          existence_max_pulses=n_pulses, random_state=0, mode=mode)
    push_ms, n_found, n_estimates = [], 0, 0
    for block, _ in stream_blocks(n_pulses, n_channels, seed, n_components):
        for first in range(0, len(block), chunk_size):
            chunk = block[first:first + chunk_size]
            start = time.perf_counter()
            estimates = fitter.push(chunk)
            elapsed = (time.perf_counter() - start) * 1e3
            if estimates:
                push_ms.append(elapsed)
                n_found += len({e.pulse_time for e in estimates})
                n_estimates += len(estimates)
    push_ms = np.array(push_ms)
    summary = {"n_pulses": n_pulses, "pulses_estimated": n_found, "estimates": n_estimates,
               "chunk_size": chunk_size, "n_channels": n_channels,
               "n_components": len(fit.components), "mode": mode,
               "push_p50_ms": float(np.percentile(push_ms, 50)),
               "push_p95_ms": float(np.percentile(push_ms, 95)),
               "push_p99_ms": float(np.percentile(push_ms, 99)),
               "push_max_ms": float(push_ms.max()),
               "compute": fitter.latency_stats,
               "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
               "load_average": os.getloadavg()[0],
               "machine": f"{platform.processor() or platform.machine()} "
                          f"({os.cpu_count()} CPUs), Python {platform.python_version()}"}
    return summary, fitter


if __name__ == "__main__":
    runs = [(1, "shift_scale", 1), (1, "shift_scale", 32)]
    runs += [(2, mode, 32) for mode in ("common", "scale", "shift_scale")]
    fits = {}
    for n_components, mode, chunk_size in runs:
        if n_components not in fits:
            fits[n_components] = template(n_components=n_components)
        summary, _ = run(1000, chunk_size=chunk_size, fit=fits[n_components], mode=mode)
        print(f"{summary['n_components']} component(s), {mode:<11} chunk {chunk_size:>2}: "
              f"pulses {summary['pulses_estimated']}/{summary['n_pulses']}"
              f"  push p50 {summary['push_p50_ms']:.2f}  p95 {summary['push_p95_ms']:.2f}"
              f"  p99 {summary['push_p99_ms']:.2f}  max {summary['push_max_ms']:.2f} ms"
              f"  (compute p99 {summary['compute']['p99']:.2f} ms;"
              f" load {summary['load_average']:.1f})")
    print(summary["machine"], "OMP_NUM_THREADS =", summary["omp_num_threads"])
