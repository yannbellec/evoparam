# SPDX-License-Identifier: Apache-2.0
"""Variable projection, masks and DC invariance (REQ-103, REQ-108, REQ-109)."""

import numpy as np

from evoparam.models import engine as eng
from evoparam.shapes import get_background, get_shape

from .helpers import TIMES, gauss, model, noise

###################################################################################################
###################################################################################################


def test_req103_linear_solution_is_weighted_least_squares():

    rng = np.random.default_rng(0)
    tw = TIMES[(TIMES >= 0) & (TIMES <= 0.5)]
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


def test_req103_masked_samples_have_no_influence():

    rng = np.random.default_rng(1)
    y = gauss(0.2, -8, 0.04) + noise(rng)
    mask = (TIMES > 0.4) & (TIMES < 0.45)
    a = model().fit_waveform(TIMES, y, mask=mask)
    y2 = y.copy()
    y2[mask] = rng.normal(0, 100, mask.sum())
    b = model().fit_waveform(TIMES, y2, mask=mask)
    assert len(a.components) == len(b.components) == 1
    assert np.isclose(a.components[0].latency, b.components[0].latency, rtol=1e-9)
    assert np.isclose(a.components[0].amplitude, b.components[0].amplitude, rtol=1e-9)


def test_req109_dc_invariance():

    rng = np.random.default_rng(2)
    y = gauss(0.15, -7, 0.03) + gauss(0.35, 4, 0.08) + noise(rng)
    a = model().fit_waveform(TIMES, y)
    b = model().fit_waveform(TIMES, y + 1234.5)
    assert len(a.components) == len(b.components) > 0
    for ca, cb in zip(a.components, b.components):
        for name in ("latency", "amplitude", "fwhm", "onset", "offset"):
            assert np.isclose(getattr(ca, name), getattr(cb, name), rtol=1e-6, atol=1e-9), name


def test_req108_component_inside_a_masked_gap():

    hits = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        y = gauss(0.2, -8, 0.04) + noise(rng)
        mask = (TIMES >= 0.199) & (TIMES <= 0.201)  # a 2 ms gap at the peak
        fit = model().fit_waveform(TIMES, y, mask=mask)
        hits += any(abs(c.latency - 0.2) <= 0.02 and c.amplitude < 0 for c in fit.components)
    assert hits / 20 >= 0.9
