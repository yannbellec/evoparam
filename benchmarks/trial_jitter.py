# SPDX-License-Identifier: Apache-2.0
"""Latency jitter found by ``fit_trials`` on a slow P3-like component (REQ-151).

40 trials of a Gaussian component (latency 350 ms, FWHM 200 ms, amplitude 10) in white
noise of SD 5, 10 and 20 (per-trial SNR 2, 1 and 0.5), 500 Hz, preset ``erp``, with no
jitter and with a true jitter of 10 ms. For each null rule, the mean over the runs of the
raw ``jitter_sd``, of ``jitter_sd_eb`` (the jitter once the part of the noise is removed,
0 when the test does not detect a jitter), the fraction of runs where a jitter is detected,
and the smallest detectable jitter. Under a minute per rule::

    python benchmarks/trial_jitter.py --runs 10 --null-rule ar segments
"""

import argparse
import warnings

import numpy as np

from evoparam.data import TrialSet
from evoparam.models import fit as fit_models
from evoparam.trials import fit_trials, jitter

SFREQ = 500.0
TIMES = np.arange(-0.5, 1.0, 1 / SFREQ)


def simulate(noise: float, sd: float, seed: int, n: int = 40) -> TrialSet:
    """Return ``n`` trials of the P3-like component, shifted by N(0, ``sd``), in noise."""
    rng = np.random.default_rng(seed)
    shifts = rng.normal(0, sd, n) if sd else np.zeros(n)
    data = rng.normal(0, noise, (n, 1, len(TIMES)))
    for i in range(n):
        data[i, 0] += 10 * np.exp(-0.5 * ((TIMES - 0.35 - shifts[i]) / (0.2 / 2.3548)) ** 2)
    return TrialSet(data=data, times=TIMES, sfreq=SFREQ, ch_names=["c0"])


def main() -> None:
    """Print the table of the bench, one line per null rule, noise and true jitter."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--null-rule", nargs="+", default=["ar", "segments"])
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    print("null_rule | noise | true jitter | jitter_sd | jitter_sd_eb | detected | min detectable")
    for rule in args.null_rule:
        for noise in (5.0, 10.0, 20.0):
            for sd in (0.0, 0.01):
                rows = []
                for seed in range(1000, 1000 + args.runs):
                    ts = simulate(noise, sd, seed)
                    tf = fit_trials(ts, fit_models(ts, "erp"), r2_min=0.0, null_rule=rule)
                    rows.append(jitter(tf).iloc[0])
                raw = np.mean([r["jitter_sd"] for r in rows]) * 1e3
                eb = np.mean([r["jitter_sd_eb"] for r in rows]) * 1e3
                found = np.mean([r["jitter_detected"] for r in rows]) * 100
                limit = np.mean([r["jitter_min_detectable"] for r in rows]) * 1e3
                print(
                    f"{rule} | {noise:.0f} | {sd * 1e3:.0f} ms | {raw:.1f} ms | {eb:.1f} ms | "
                    f"{found:.0f} % | {limit:.1f} ms",
                    flush=True,
                )


if __name__ == "__main__":
    main()
