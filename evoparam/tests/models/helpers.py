# SPDX-License-Identifier: Apache-2.0
"""Synthetic waveforms with known components for the engine tests."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import lfilter

from evoparam.models import ResponseModel

# Smallest latency gap (s) between two simulated components.
MIN_COMPONENT_GAP = 0.08

###################################################################################################
###################################################################################################

SFREQ = 1000.0
TIMES = np.arange(-0.5, 1.0, 1 / SFREQ)
WINDOW = (0.02, 0.6)
BASELINE = (-0.5, -0.02)


def noise(rng: Any, rho: Any = 0.0, sd: Any = 1.0, n: Any = None) -> Any:
    e = rng.normal(0, 1, len(TIMES) if n is None else n)
    x = lfilter([1], [1, -rho], e) if rho else e
    return sd * x / np.std(x)


def gauss(latency: Any, amplitude: Any, fwhm: Any, times: Any = TIMES) -> Any:
    sigma = fwhm / 2.3548
    return amplitude * np.exp(-0.5 * ((times - latency) / sigma) ** 2)


def model(**kwargs: Any) -> Any:
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


def random_components(
    rng: Any, k_max: Any = 3, snr: Any = (5, 10), width: Any = (0.02, 0.08)
) -> Any:
    k = int(rng.integers(1, k_max + 1))
    while True:
        lats = np.sort(rng.uniform(0.06, 0.5, k))
        if k == 1 or np.min(np.diff(lats)) >= MIN_COMPONENT_GAP:
            break
    return [(lat, rng.choice([-1, 1]) * rng.uniform(*snr), rng.uniform(*width)) for lat in lats]


def match(fit: Any, truth: Any) -> Any:
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


@dataclass(frozen=True)
class WaveformOptions:
    """The keywords of :func:`erp_waveform`: shape, sampling, latency span and noise."""

    shape: Any = "gaussian"
    sfreq: Any = SFREQ
    span: Any = (0.06, 0.55)
    noise: Any = "white"
    noise_kw: Any = None


def erp_waveform(seed: Any, n_components: Any, snr: Any, widths: Any, **keywords: Any) -> Any:
    """One waveform of ``evoparam.sim.simulate_erp`` (a single trial, so that the SNR of the
    waveform is the requested one) on TIMES, with ``n_components`` components of the given
    widths (samples) and random signs, latencies drawn in ``span`` at least three widths
    apart. Returns (times, y, truth rows (latency, amplitude, fwhm)).

    Keywords: the fields of :class:`WaveformOptions`.
    """

    from evoparam.sim import simulate_erp

    options = WaveformOptions(**keywords)
    rng = np.random.default_rng(seed)
    fwhm = [w / options.sfreq for w in widths[:n_components]]
    for _ in range(1000):
        lats = np.sort(rng.uniform(*options.span, n_components))
        gaps = [3 * max(fwhm[i], fwhm[i + 1]) for i in range(n_components - 1)]
        if all(b - a >= g for a, b, g in zip(lats[:-1], lats[1:], gaps, strict=False)):
            break
    comps = [
        {
            "label": f"C{j}",
            "shape": options.shape,
            "latency": float(lats[j]),
            "fwhm": fwhm[j],
            "amplitude": float(rng.choice([-1.0, 1.0])),
        }
        for j in range(n_components)
    ]
    ts, truth = simulate_erp(
        1,
        sfreq=options.sfreq,
        tmin=TIMES[0],
        tmax=TIMES[-1] + 0.5 / options.sfreq,
        components=comps,
        snr=snr if n_components else None,
        noise=options.noise,
        noise_sd=1.0,
        noise_kw=options.noise_kw,
        seed=seed,
    )
    table = truth.to_dataframe()
    rows = [(float(r.latency), float(r.amplitude), float(r.fwhm)) for r in table.itertuples()]
    return ts.times, ts.data[0, 0], rows


def same_decisions(a: Any, b: Any) -> Any:
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
