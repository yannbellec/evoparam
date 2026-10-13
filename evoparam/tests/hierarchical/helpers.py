# SPDX-License-Identifier: Apache-2.0
"""Three-level simulations (trial -> channel -> subject) with a known truth."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from evoparam.data import TrialSet

pd = pytest.importorskip("pandas")

SFREQ = 1000.0
TIMES = np.arange(-0.3, 0.55, 1 / SFREQ)
NOISE_SD = 10.0
PHI = 0.6


def ar1(rng: Any, shape: Any, phi: Any = PHI, sd: Any = NOISE_SD) -> Any:
    e = rng.normal(0, sd * np.sqrt(1 - phi**2), shape)
    out = np.empty(shape)
    out[..., 0] = rng.normal(0, sd, shape[:-1])
    for k in range(1, shape[-1]):
        out[..., k] = phi * out[..., k - 1] + e[..., k]
    return out


def gauss(latency: Any, fwhm: Any, times: Any = TIMES) -> Any:
    return np.exp(-0.5 * ((times[None] - np.asarray(latency)[:, None]) / (fwhm / 2.3548)) ** 2)


@dataclass(frozen=True)
class GroupDesign:
    """The keywords of :func:`simulate_group`: the simulated design and the seed."""

    n_subjects: Any = 20
    n_channels: Any = 50
    n_trials: Any = 20
    p_respond: Any = 0.3
    snr: Any = 2.0
    amp_sd: Any = 0.3
    subject_amp_sd: Any = 0.15
    latency: Any = 0.03
    latency_sd_channel: Any = 0.004
    latency_sd_subject: Any = 0.003
    jitter: Any = 0.002
    fwhm: Any = 0.02
    age_slope: Any = 0.0
    n2: Any = 0.0
    seed: Any = 0


def simulate_group(**keywords: Any) -> Any:
    """List of TrialSets (one per subject) and the truth table (one row per channel).

    Amplitudes are negative (an N1): subject mean ``-snr * NOISE_SD * (1 + subject
    deviation)``, channel amplitudes around it with relative SD ``amp_sd``; latencies around
    ``latency`` (plus ``age_slope`` s/year x age) with channel and subject SDs; per-trial
    latency jitter SD ``jitter``. ``n2 > 0`` adds, on the responding channels, a positive
    N2 at 150 ms (FWHM 60 ms) of ``n2`` times the N1 amplitude, with the same trial jitter.

    Keywords: the fields of :class:`GroupDesign`.
    """

    design = GroupDesign(**keywords)
    rng = np.random.default_rng(design.seed)
    sets, truth = [], []
    ages = rng.uniform(5, 50, design.n_subjects)
    for s in range(design.n_subjects):
        base_amp = -design.snr * NOISE_SD * (1 + design.subject_amp_sd * rng.normal())
        base_lat = (
            design.latency
            + design.age_slope * (ages[s] - 25)
            + design.latency_sd_subject * rng.normal()
        )
        respond = rng.random(design.n_channels) < design.p_respond
        amps = np.where(
            respond, base_amp * (1 + design.amp_sd * rng.normal(size=design.n_channels)), 0.0
        )
        lats = base_lat + design.latency_sd_channel * rng.normal(size=design.n_channels)
        data = ar1(rng, (design.n_trials, design.n_channels, len(TIMES)))
        for c in np.flatnonzero(respond):
            shifts = design.jitter * rng.normal(size=design.n_trials)
            data[:, c] += amps[c] * gauss(lats[c] + shifts, design.fwhm)
            if design.n2:
                data[:, c] -= (
                    design.n2 * amps[c] * gauss(0.15 + 3 * (lats[c] - base_lat) + shifts, 0.06)
                )
        names = [f"s{s}c{c}" for c in range(design.n_channels)]
        meta = pd.DataFrame({"subject": [f"sub{s:02d}"] * design.n_trials, "age": ages[s]})
        sets.append(
            TrialSet(
                data=data, times=TIMES, sfreq=SFREQ, ch_names=names, metadata=meta, modality="spes"
            )
        )
        for c in range(design.n_channels):
            truth.append(
                {
                    "subject": f"sub{s:02d}",
                    "channel": names[c],
                    "respond": bool(respond[c]),
                    "amplitude": amps[c],
                    "latency": lats[c],
                    "subject_amplitude": base_amp,
                    "subject_latency": base_lat,
                    "age": ages[s],
                }
            )
    return sets, pd.DataFrame(truth)


def auc(score: Any, label: Any) -> Any:
    """Area under the ROC curve (ties counted half)."""

    from scipy.stats import rankdata

    label = np.asarray(label, bool)
    ranks = rankdata(score)
    n1, n0 = label.sum(), (~label).sum()
    return (ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def reliability_slope(prob: Any, label: Any, n_bins: Any = 10) -> Any:
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


def merged(fit: Any, truth: Any) -> Any:
    table = fit.level_tables["channel"]
    return table.merge(truth, on=["subject", "channel"], suffixes=("", "_true"))
