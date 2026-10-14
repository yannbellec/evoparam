# SPDX-License-Identifier: Apache-2.0
"""Variable projection, masks and DC invariance (REQ-103, REQ-108, REQ-109)."""

from typing import Any

import numpy as np

from evoparam.models import engine as eng
from evoparam.shapes import get_background, get_shape

from .helpers import TIMES, gauss, model, noise

# Fit window end (s), the masked span (s) of REQ-103, and the 2 ms gap at the peak (REQ-108).
WINDOW_END = 0.5
MASK_START = 0.4
MASK_END = 0.45
GAP_START = 0.199
GAP_END = 0.201
# Latency tolerance (s) around the simulated 0.2 s, runs, and fraction found (REQ-108).
LATENCY_TOLERANCE = 0.02
N_RUNS = 20
MIN_FOUND = 0.9

###################################################################################################
###################################################################################################


def test_req103_linear_solution_is_weighted_least_squares() -> None:

    rng = np.random.default_rng(0)
    tw = TIMES[(TIMES >= 0) & (TIMES <= WINDOW_END)]
    y = gauss(0.1, -5, 0.03, tw) + gauss(0.3, 3, 0.06, tw) + 0.4 + noise(rng, n=len(tw)) * 0.2
    shapes = [get_shape("gaussian"), get_shape("gaussian")]
    background = get_background("constant")
    scale = rng.uniform(0.5, 2.0, len(tw))
    white = eng.Whitener(scale, [1.0], 1.0, np.ones(len(tw), bool))
    struct = eng._Structure(shapes, background, tw, white, y)
    theta = np.array([0.1, 0.03 / 2.3548, 0.3, 0.06 / 2.3548])
    beta, *_ = struct.solve(theta)
    m = struct.design(theta)
    w = 1 / scale
    expected = np.linalg.lstsq(m * w[:, None], y * w, rcond=None)[0]
    assert np.allclose(beta, expected, rtol=1e-10, atol=1e-12)


def test_req103_masked_samples_have_no_influence() -> None:

    rng = np.random.default_rng(1)
    y = gauss(0.2, -8, 0.04) + noise(rng)
    mask = (TIMES > MASK_START) & (TIMES < MASK_END)
    a = model().fit_waveform(TIMES, y, mask=mask)
    y2 = y.copy()
    y2[mask] = rng.normal(0, 100, mask.sum())
    b = model().fit_waveform(TIMES, y2, mask=mask)
    assert len(a.components) == len(b.components) == 1
    assert np.isclose(a.components[0].latency, b.components[0].latency, rtol=1e-9)
    assert np.isclose(a.components[0].amplitude, b.components[0].amplitude, rtol=1e-9)


def test_req109_dc_invariance() -> None:

    rng = np.random.default_rng(2)
    y = gauss(0.15, -7, 0.03) + gauss(0.35, 4, 0.08) + noise(rng)
    a = model().fit_waveform(TIMES, y)
    b = model().fit_waveform(TIMES, y + 1234.5)
    assert len(a.components) == len(b.components) > 0
    for ca, cb in zip(a.components, b.components, strict=False):
        for name in ("latency", "amplitude", "fwhm", "onset", "offset"):
            assert np.isclose(getattr(ca, name), getattr(cb, name), rtol=1e-6, atol=1e-9), name


def test_req108_component_inside_a_masked_gap() -> None:

    hits = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        y = gauss(0.2, -8, 0.04) + noise(rng)
        mask = (TIMES >= GAP_START) & (TIMES <= GAP_END)  # a 2 ms gap at the peak
        fit = model().fit_waveform(TIMES, y, mask=mask)
        hits += any(
            abs(c.latency - 0.2) <= LATENCY_TOLERANCE and c.amplitude < 0 for c in fit.components
        )
    assert hits / N_RUNS >= MIN_FOUND


# A gap (s) around the 10 % onset of a Gaussian at 0.1 s with FWHM 40 ms (onset about
# 0.064 s), and a gap after its support.
ONSET_GAP = (0.055, 0.075)
LATE_GAP = (0.4, 0.45)


def _without_widening(monkeypatch: Any) -> None:
    from evoparam.models import api

    monkeypatch.setattr(api, "_widen_masked", lambda fit, _times, _mask: fit)


def test_masked_onset_widens_the_intervals(monkeypatch: Any) -> None:
    """Spec §5.3 item 12: a component whose onset lies inside a masked interval has its
    ``Component.ci`` inflated by the masked fraction of its support (``1 + f`` times the
    width, around the same centre; DECISIONS §4); a component whose onset is not masked keeps
    its intervals."""

    y = gauss(0.1, -8, 0.04) + noise(np.random.default_rng(7))
    m = model(shapes=("gaussian",), random_state=0)
    gap_start, gap_end = ONSET_GAP
    mask = (gap_start <= TIMES) & (gap_end >= TIMES)
    widened = m.fit_waveform(TIMES, y, mask=mask)
    with monkeypatch.context() as patch:
        _without_widening(patch)
        plain = m.fit_waveform(TIMES, y, mask=mask)
    assert len(widened.components) == len(plain.components) == 1
    comp, ref = widened.components[0], plain.components[0]
    assert gap_start <= comp.onset <= gap_end
    support = (ref.onset <= TIMES) & (ref.offset >= TIMES)
    fraction = mask[support].mean()
    assert 0 < fraction < 1
    assert widened.diagnostics["masked_support"] == [
        {"component": 0, "masked_fraction": fraction, "factor": 1 + fraction}
    ]
    assert "masked_onset:0" in widened.warnings
    assert "masked_onset:0" not in plain.warnings
    assert set(comp.ci) == set(ref.ci)
    assert len(ref.ci) > len(ref.param_names)
    for name, (low, high) in ref.ci.items():
        w_low, w_high = comp.ci[name]
        assert np.isclose(w_high - w_low, (1 + fraction) * (high - low), rtol=1e-12), name
        assert np.isclose(w_low + w_high, low + high, rtol=1e-12, atol=1e-15), name
        assert np.isclose(comp.se[name], (1 + fraction) * ref.se[name], rtol=1e-12), name
    # the estimates themselves do not change
    assert comp.latency == ref.latency
    assert comp.amplitude == ref.amplitude

    late_start, late_end = LATE_GAP
    late = (late_start <= TIMES) & (late_end >= TIMES)
    kept = m.fit_waveform(TIMES, y, mask=late)
    assert "masked_support" not in kept.diagnostics
    assert not any(w.startswith("masked_onset") for w in kept.warnings)
    with monkeypatch.context() as patch:
        _without_widening(patch)
        same = m.fit_waveform(TIMES, y, mask=late)
    assert kept.components[0].ci == same.components[0].ci
