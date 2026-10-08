"""Synthetic waveforms with known components for the engine tests."""

import numpy as np
from scipy.signal import lfilter

from evoparam.models import ResponseModel

###################################################################################################
###################################################################################################

SFREQ = 1000.
TIMES = np.arange(-0.5, 1.0, 1 / SFREQ)
WINDOW = (0.02, 0.6)
BASELINE = (-0.5, -0.02)


def noise(rng, rho=0.0, sd=1.0, n=None):
    e = rng.normal(0, 1, len(TIMES) if n is None else n)
    x = lfilter([1], [1, -rho], e) if rho else e
    return sd * x / np.std(x)


def gauss(latency, amplitude, fwhm, times=TIMES):
    sigma = fwhm / 2.3548
    return amplitude * np.exp(-0.5 * ((times - latency) / sigma) ** 2)


def model(**kwargs):
    settings = dict(fit_window=WINDOW, baseline=BASELINE, shapes=("gaussian", "skewed_gaussian"),
                    background="constant", min_fwhm=0.008, max_fwhm=0.3)
    settings.update(kwargs)
    return ResponseModel(**settings)


def random_components(rng, k_max=3, snr=(5, 10), width=(0.02, 0.08)):
    k = int(rng.integers(1, k_max + 1))
    while True:
        lats = np.sort(rng.uniform(0.06, 0.5, k))
        if k == 1 or np.min(np.diff(lats)) >= 0.08:
            break
    return [(lat, rng.choice([-1, 1]) * rng.uniform(*snr), rng.uniform(*width)) for lat in lats]


def match(fit, truth):
    """(true positives, false negatives, false positives, latency errors, amplitude ratios)."""

    used, tp, fn, lat, amp = set(), 0, 0, [], []
    for latency, amplitude, fwhm in truth:
        hits = [c for c in fit.components if abs(c.latency - latency) <= fwhm / 2
                and np.sign(c.amplitude) == np.sign(amplitude) and id(c) not in used]
        if hits:
            comp = min(hits, key=lambda c: abs(c.latency - latency))
            used.add(id(comp))
            tp += 1
            lat.append(comp.latency - latency)
            amp.append(comp.amplitude / amplitude)
        else:
            fn += 1
    fp = sum(1 for c in fit.components if id(c) not in used)
    return tp, fn, fp, lat, amp
