"""Simulated multi-site and multi-channel data with known shapes, latencies and sources."""

import warnings

import numpy as np
import pandas as pd

from evoparam.data import Evoked, TrialSet
from evoparam.models import ResponseModel
from evoparam.sim.artifacts import seeg_layout

SFREQ = 1000.
TIMES = np.arange(-0.3, 0.6, 1 / SFREQ)
BASELINE = (-0.3, -0.01)


def gauss(latency, fwhm, times=TIMES):
    return np.exp(-0.5 * ((times - latency) / (fwhm / 2.3548)) ** 2)


# Three response shapes with little overlap (an N1-P2, a broad N2, a late positivity)
SHAPES = (-gauss(0.025, 0.015) + 0.6 * gauss(0.06, 0.03),
          -gauss(0.15, 0.08),
          gauss(0.30, 0.10) - 0.5 * gauss(0.12, 0.04))


def multi_site(seed, n_sites=20, n_trials=10, noise=30.0, amp=(40.0, 120.0), shapes=SHAPES):
    """One recording channel, ``n_sites`` stimulation sites, each driving one of ``shapes``
    (about equal numbers) with its own amplitude; per-trial amplitude CV 0.2, white noise."""

    rng = np.random.default_rng(seed)
    which = np.arange(n_sites) % len(shapes)
    rng.shuffle(which)
    data, sites = [], []
    for s in range(n_sites):
        a = rng.uniform(*amp)
        for _ in range(n_trials):
            data.append(a * rng.normal(1, 0.2) * shapes[which[s]] +
                        rng.normal(0, noise, len(TIMES)))
            sites.append(f"S{s:02d}-T{s:02d}")
    ts = TrialSet(data=np.array(data)[:, None], times=TIMES, sfreq=SFREQ, ch_names=["R01"],
                  metadata=pd.DataFrame({"stim_site": sites}))
    return ts, which


def shared_channels(seed, n=30, snr=2.0, latency=0.03, fwhm=0.02, offsets=None):
    """``n`` channels with one Gaussian component of the same width and latency (plus
    ``offsets``), amplitudes 25 to 75 in absolute value with random signs, and white noise
    of SD |amplitude| / snr per channel; contacts on sEEG shafts of 10."""

    rng = np.random.default_rng(seed)
    names, coords, groups = seeg_layout(n, contacts_per_shaft=10)
    amps = rng.uniform(-1, 1, n)
    amps = np.sign(amps) * (0.5 + np.abs(amps)) * 50
    offsets = np.zeros(n) if offsets is None else np.asarray(offsets, float)
    clean = amps[:, None] * np.array([gauss(latency + d, fwhm) for d in offsets])
    info = pd.DataFrame({"name": names, "x": coords[:, 0], "y": coords[:, 1],
                         "z": coords[:, 2], "group": groups})

    def draw():
        noise = rng.normal(0, 1, clean.shape) * (np.abs(amps) / snr)[:, None]
        return Evoked(data=clean + noise, times=TIMES, sfreq=SFREQ, ch_names=names,
                      ch_info=info)
    return draw, clean, info


def independent_fits(evoked, preset="ccep"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ResponseModel(preset, baseline=BASELINE).fit_group(evoked)


def volume_scene(seed, n_trials=10, noise=10.0, source_amp=1000.0, min_level=5.0):
    """sEEG scene with one stimulation site (A04-A05) whose response spreads passively,
    amplitude ``source_amp (3.5 mm / r)²`` with the same sign and shape everywhere, and two
    local dipolar sources (opposite polarities on two adjacent contacts, a third of the
    amplitude on the next ones, another waveform) on far shafts.

    Returns the TrialSet and the names of the volume-conducted channels whose expected
    amplitude is at least ``min_level`` noise SD of the average, and of the local-source contacts.
    """

    rng = np.random.default_rng(seed)
    names, coords, groups = seeg_layout(60, contacts_per_shaft=10)
    site = ("A04", "A05")
    centre = coords[[names.index(s) for s in site]].mean(axis=0)
    r = np.maximum(np.linalg.norm(coords - centre, axis=1), 1.75)
    source = -gauss(0.02, 0.012) + 0.5 * gauss(0.05, 0.03)
    gain = (3.5 / r) ** 2
    local_wave = -gauss(0.06 + 0.01 * rng.uniform(), 0.03) + 0.4 * gauss(0.14, 0.05)
    clean = gain[:, None] * source_amp * source[None]
    local = []
    for shaft in rng.choice(["D", "E", "F"], 2, replace=False):
        k = int(rng.integers(3, 7))
        amp = rng.uniform(80, 150)
        for offset, weight in ((-1, 0.3), (0, 1.0), (1, -1.0), (2, -0.3)):
            name = f"{shaft}{k + offset:02d}"
            clean[names.index(name)] += weight * amp * local_wave
        local += [f"{shaft}{k:02d}", f"{shaft}{k + 1:02d}"]
    data = clean[None] + rng.normal(0, noise, (n_trials,) + clean.shape)
    for s in site:                                     # stimulating contacts: saturated
        data[:, names.index(s)] = 3000.0
    info = pd.DataFrame({"name": names, "x": coords[:, 0], "y": coords[:, 1],
                         "z": coords[:, 2], "group": groups, "bad": False,
                         "stim_contact": [n in site for n in names]})
    ts = TrialSet(data=data, times=TIMES, sfreq=SFREQ, ch_names=names, ch_info=info,
                  metadata=pd.DataFrame({"stim_site": ["A04-A05"] * n_trials}))
    level = gain * source_amp * np.max(np.abs(source)) / (noise / np.sqrt(n_trials))
    volume = [n for n, lv in zip(names, level) if lv >= min_level and n not in site and
              n not in local and not any(n[0] == loc[0] for loc in local)]
    return ts, volume, local
