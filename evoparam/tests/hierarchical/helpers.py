# SPDX-License-Identifier: Apache-2.0
"""Three-level simulations (trial -> channel -> subject) with a known truth."""

import numpy as np
import pandas as pd

from evoparam.data import TrialSet

SFREQ = 1000.
TIMES = np.arange(-0.3, 0.55, 1 / SFREQ)
NOISE_SD = 10.0
PHI = 0.6


def ar1(rng, shape, phi=PHI, sd=NOISE_SD):
    e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), shape)
    out = np.empty(shape)
    out[..., 0] = rng.normal(0, sd, shape[:-1])
    for k in range(1, shape[-1]):
        out[..., k] = phi * out[..., k - 1] + e[..., k]
    return out


def gauss(latency, fwhm, times=TIMES):
    return np.exp(-0.5 * ((times[None] - np.asarray(latency)[:, None]) / (fwhm / 2.3548)) ** 2)


def simulate_group(n_subjects=20, n_channels=50, n_trials=20, p_respond=0.3, snr=2.0,
                   amp_sd=0.3, subject_amp_sd=0.15, latency=0.03, latency_sd_channel=0.004,
                   latency_sd_subject=0.003, jitter=0.002, fwhm=0.02, age_slope=0.0,
                   n2=0.0, seed=0):
    """List of TrialSets (one per subject) and the truth table (one row per channel).

    Amplitudes are negative (an N1): subject mean ``-snr * NOISE_SD * (1 + subject
    deviation)``, channel amplitudes around it with relative SD ``amp_sd``; latencies around
    ``latency`` (plus ``age_slope`` s/year x age) with channel and subject SDs; per-trial
    latency jitter SD ``jitter``. ``n2 > 0`` adds, on the responding channels, a positive
    N2 at 150 ms (FWHM 60 ms) of ``n2`` times the N1 amplitude, with the same trial jitter.
    """

    rng = np.random.default_rng(seed)
    sets, truth = [], []
    ages = rng.uniform(5, 50, n_subjects)
    for s in range(n_subjects):
        base_amp = -snr * NOISE_SD * (1 + subject_amp_sd * rng.normal())
        base_lat = latency + age_slope * (ages[s] - 25) + latency_sd_subject * rng.normal()
        respond = rng.random(n_channels) < p_respond
        amps = np.where(respond, base_amp * (1 + amp_sd * rng.normal(size=n_channels)), 0.0)
        lats = base_lat + latency_sd_channel * rng.normal(size=n_channels)
        data = ar1(rng, (n_trials, n_channels, len(TIMES)))
        for c in np.flatnonzero(respond):
            shifts = jitter * rng.normal(size=n_trials)
            data[:, c] += amps[c] * gauss(lats[c] + shifts, fwhm)
            if n2:
                data[:, c] -= n2 * amps[c] * gauss(0.15 + 3 * (lats[c] - base_lat) + shifts,
                                                   0.06)
        names = [f"s{s}c{c}" for c in range(n_channels)]
        meta = pd.DataFrame({"subject": [f"sub{s:02d}"] * n_trials, "age": ages[s]})
        sets.append(TrialSet(data=data, times=TIMES, sfreq=SFREQ, ch_names=names,
                             metadata=meta, modality="spes"))
        for c in range(n_channels):
            truth.append({"subject": f"sub{s:02d}", "channel": names[c],
                          "respond": bool(respond[c]), "amplitude": amps[c],
                          "latency": lats[c], "subject_amplitude": base_amp,
                          "subject_latency": base_lat, "age": ages[s]})
    return sets, pd.DataFrame(truth)


def auc(score, label):
    """Area under the ROC curve (ties counted half)."""

    from scipy.stats import rankdata

    label = np.asarray(label, bool)
    ranks = rankdata(score)
    n1, n0 = label.sum(), (~label).sum()
    return (ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def reliability_slope(prob, label, n_bins=10):
    """Slope of the reliability diagram: observed frequency on mean predicted probability
    per bin, weighted by the bin counts."""

    prob, label = np.asarray(prob, float), np.asarray(label, float)
    bins = np.clip((prob * n_bins).astype(int), 0, n_bins - 1)
    x, y, w = [], [], []
    for k in range(n_bins):
        sel = bins == k
        if sel.any():
            x.append(prob[sel].mean())
            y.append(label[sel].mean())
            w.append(sel.sum())
    x, y, w = map(np.asarray, (x, y, w))
    design = np.column_stack([np.ones_like(x), x])
    coef = np.linalg.solve(design.T @ (w[:, None] * design), design.T @ (w * y))
    return float(coef[1])


def merged(fit, truth):
    table = fit.level_tables["channel"]
    return table.merge(truth, on=["subject", "channel"], suffixes=("", "_true"))
