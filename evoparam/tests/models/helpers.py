# SPDX-License-Identifier: Apache-2.0
"""Synthetic waveforms with known components for the engine tests."""

import numpy as np
from scipy.signal import lfilter

from evoparam.models import ResponseModel

###################################################################################################
###################################################################################################

SFREQ = 1000.0
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
    settings = {
        "fit_window": WINDOW,
        "baseline": BASELINE,
        "shapes": ("gaussian", "skewed_gaussian"),
        "background": "constant",
        "min_fwhm": 0.008,
        "max_fwhm": 0.3,
    }
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
        hits = [
            c
            for c in fit.components
            if abs(c.latency - latency) <= fwhm / 2
            and np.sign(c.amplitude) == np.sign(amplitude)
            and id(c) not in used
        ]
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


###################################################################################################
## Designs of the acceptance tests on simulate_erp waveforms (REQ-104 to REQ-106)

# Levels of spec §7.2: SNR (peak / noise SD of the waveform) and width (samples)
SNR_LEVELS = (3, 5, 10, 20)
WIDTH_SAMPLES = (3, 6, 12, 25)
TRUTH_SHAPES = ("gaussian", "skewed_gaussian", "exgaussian", "gamma", "biexponential")


def erp_waveform(
    seed,
    n_components,
    snr,
    widths,
    shape="gaussian",
    sfreq=SFREQ,
    span=(0.06, 0.55),
    noise="white",
    noise_kw=None,
):
    """One waveform of ``evoparam.sim.simulate_erp`` (a single trial, so that the SNR of the
    waveform is the requested one) on TIMES, with ``n_components`` components of the given
    widths (samples) and random signs, latencies drawn in ``span`` at least three widths
    apart. Returns (times, y, truth rows (latency, amplitude, fwhm))."""

    from evoparam.sim import simulate_erp

    rng = np.random.default_rng(seed)
    fwhm = [w / sfreq for w in widths[:n_components]]
    for _ in range(1000):
        lats = np.sort(rng.uniform(*span, n_components))
        gaps = [3 * max(fwhm[i], fwhm[i + 1]) for i in range(n_components - 1)]
        if all(b - a >= g for a, b, g in zip(lats[:-1], lats[1:], gaps, strict=False)):
            break
    comps = [
        {
            "label": f"C{j}",
            "shape": shape,
            "latency": float(lats[j]),
            "fwhm": fwhm[j],
            "amplitude": float(rng.choice([-1.0, 1.0])),
        }
        for j in range(n_components)
    ]
    ts, truth = simulate_erp(
        1,
        sfreq=sfreq,
        tmin=TIMES[0],
        tmax=TIMES[-1] + 0.5 / sfreq,
        components=comps,
        snr=snr if n_components else None,
        noise=noise,
        noise_sd=1.0,
        noise_kw=noise_kw,
        seed=seed,
    )
    table = truth.to_dataframe()
    rows = [(float(r.latency), float(r.amplitude), float(r.fwhm)) for r in table.itertuples()]
    return ts.times, ts.data[0, 0], rows


def same_decisions(a, b):
    """Two fits accept the same components: as many, of the same signs, at latencies within
    half the larger width."""

    if len(a.components) != len(b.components):
        return False
    for x, y in zip(
        sorted(a.components, key=lambda c: c.latency),
        sorted(b.components, key=lambda c: c.latency),
        strict=False,
    ):
        if x.polarity != y.polarity or abs(x.latency - y.latency) > max(x.fwhm, y.fwhm) / 2:
            return False
    return True
