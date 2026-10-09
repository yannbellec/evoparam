"""Classical measures against an independent implementation of the ERPLAB definitions
(REQ-117).

The reference below is written from the definitions only (ERPLAB's measurement tool:
local peak with neighbours, mean amplitude, area, fractional area latency, fractional peak
latency), with loops instead of array operations, and shares no code with
:mod:`evoparam.models.classical`. REQ-117 asks for agreement to 1e-9 on the ERP CORE N170
and P3 averages: that test reads the local copy named by ``EVOPARAM_ERPCORE`` (layout of
``validation/erpcore.py``; 40 participants, the published channel and measurement window).
The same comparison runs on every build on simulated N170 and P3 averages of 40
participants (``evoparam.sim.simulate_erp``).
"""

import importlib.util
import math
import os
import pathlib

import numpy as np
import pytest

from evoparam.models import classical_measures

TOL = 1e-9


###################################################################################################
## Independent reference (definitions only)

def _ref_signed(v, polarity):
    return -v if polarity == "negative" else (abs(v) if polarity == "absolute" else v)


def ref_local_peak(t, y, polarity, n):
    """Index of the largest sample that exceeds each of its n neighbours on both sides and
    the mean of the neighbours on each side; the simple extremum when none does."""

    s = [_ref_signed(v, polarity) for v in y]
    best = None
    for i in range(n, len(s) - n):
        left, right = s[i - n:i], s[i + 1:i + n + 1]
        ok = all(s[i] > v for v in left + right) and s[i] > sum(left) / n \
            and s[i] > sum(right) / n
        if ok and (best is None or s[i] > s[best]):
            best = i
    if best is None:
        best = max(range(len(s)), key=lambda i: (s[i], -i))
    return best


def ref_area(t, y, kind, base):
    total = 0.0
    for i in range(len(t) - 1):
        a, b = y[i] - base, y[i + 1] - base
        if kind == "positive":
            a, b = max(a, 0.0), max(b, 0.0)
        elif kind == "negative":
            a, b = min(a, 0.0), min(b, 0.0)
        elif kind == "rectified":
            a, b = abs(a), abs(b)
        total += (a + b) / 2 * (t[i + 1] - t[i])
    return total


def ref_fractional_area_latency(t, y, fraction, kind, base):
    def mag(v):
        v = v - base
        if kind == "positive":
            return max(v, 0.0)
        if kind == "negative":
            return -min(v, 0.0)
        return abs(v)

    pieces = [(mag(y[i]) + mag(y[i + 1])) / 2 * (t[i + 1] - t[i]) for i in range(len(t) - 1)]
    total = sum(pieces)
    if not total > 0:
        return math.nan
    target, acc = fraction * total, 0.0
    for i, piece in enumerate(pieces):
        if acc + piece >= target and piece > 0:
            return t[i] + (target - acc) / piece * (t[i + 1] - t[i])
        acc += piece
    return t[-1]


def ref_fractional_peak_latency(t, y, peak, fraction, polarity):
    s = [_ref_signed(v, polarity) for v in y]
    level = fraction * s[peak]
    for j in range(peak, -1, -1):
        if s[j] < level:
            if j == peak:
                return t[peak]
            step = s[j + 1] - s[j]
            return t[j] + ((level - s[j]) / step if step else 0.0) * (t[j + 1] - t[j])
    return t[0]


def reference(times, y, window, polarity, fraction=0.5, area="total", area_baseline="zero",
              n_neighbors=3):
    idx = [i for i, v in enumerate(times) if window[0] <= v <= window[1]]
    t = [float(times[i]) for i in idx]
    w = [float(y[i]) for i in idx]
    peak = ref_local_peak(t, w, polarity, n_neighbors)
    base = 0.0 if area_baseline == "zero" else fraction * w[peak]
    return {"peak_amplitude": w[peak], "peak_latency": t[peak],
            "mean_amplitude": sum(w) / len(w), "area": ref_area(t, w, area, base),
            "fractional_area_latency": ref_fractional_area_latency(t, w, fraction, area, base),
            "fractional_peak_latency": ref_fractional_peak_latency(t, w, peak, fraction,
                                                                   polarity)}


MEASURES = ("peak", "mean", "area", "fractional_area_latency", "fractional_peak_latency")
VARIANTS = [dict(area="total", area_baseline="zero", fraction=0.5),
            dict(area="positive", area_baseline="zero", fraction=0.5),
            dict(area="negative", area_baseline="fraction_of_peak", fraction=0.3),
            dict(area="rectified", area_baseline="zero", fraction=0.5)]


def _compare(times, average, window, polarity):
    for variant in VARIANTS:
        got = classical_measures(times, average, window=window, polarity=polarity,
                                 measures=MEASURES, **variant)
        want = reference(times, average, window, polarity, **variant)
        for name, value in want.items():
            if math.isnan(value):
                assert math.isnan(got[name]), name
            else:
                assert abs(got[name] - value) <= TOL * max(1.0, abs(value)), \
                    (name, variant, got[name], value)


###################################################################################################
## Simulated N170 and P3 averages (every build)

@pytest.mark.parametrize("component, channel_window, polarity",
                         [("N170", (0.110, 0.150), "negative"),
                          ("P3", (0.300, 0.600), "positive")])
def test_req117_classical_measures_match_independent_definitions(component, channel_window,
                                                                  polarity):
    """40 simulated participants' averages (60 trials, pink noise), the ERP CORE
    measurement window of the component; every measure and the four area variants."""

    from evoparam.sim import simulate_erp

    for participant in range(40):
        ts, _ = simulate_erp(60, components=["P1", component], snr=1.0, noise="pink",
                             sfreq=256.0, seed=1700 + participant)
        _compare(ts.times, ts.data.mean(axis=0)[0], channel_window, polarity)


###################################################################################################
## ERP CORE N170 and P3 averages (local copy)

ERPCORE = os.environ.get("EVOPARAM_ERPCORE", "")
SCRIPT = pathlib.Path(__file__).parents[3] / "validation" / "erpcore.py"


@pytest.mark.skipif(not (ERPCORE and os.path.isdir(ERPCORE) and SCRIPT.exists()),
                    reason="set EVOPARAM_ERPCORE to the local ERP CORE copy")
@pytest.mark.parametrize("component", ["N170", "P3"])
def test_req117_on_erp_core_averages(component):
    """REQ-117 as written: the 40 ERP CORE participants' difference waves of N170 and P3 on
    the published channel, in the published measurement window."""

    pytest.importorskip("mne")
    spec = importlib.util.spec_from_file_location("erpcore", SCRIPT)
    erpcore = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(erpcore)
    design = erpcore.DESIGNS[component]
    polarity = "negative" if design["polarity"] < 0 else "positive"
    loaded = compared = 0
    for participant in range(1, 41):
        times, data, names, bins = erpcore.load(component, participant)
        loaded += 1
        wave, _ = erpcore.difference_wave(component, times, data, names, bins)
        if not np.all(np.isfinite(wave)):
            continue                  # no difference wave (as in the published analysis)
        _compare(np.asarray(times), np.asarray(wave), design["window"], polarity)
        compared += 1
    assert loaded == 40 and compared > 0
