# SPDX-License-Identifier: Apache-2.0
"""The fast paths of the engine compute what the reference paths compute (REQ-115 work):
the QR-reduced residual and Jacobian of the variable projection, the fused design columns,
the Gaussian factors left out where they are exactly zero, and the features scaled from
the unit-amplitude ones."""

import numpy as np
import pytest

from evoparam.models import _columns, api
from evoparam.models import engine as eng
from evoparam.shapes import compute_features, get_background, get_shape

from .helpers import TIMES, gauss, noise

###################################################################################################
###################################################################################################

def _struct(background="exponential", ar=True, prior=True, ordered=True):
    rng = np.random.default_rng(0)
    tw = TIMES[(TIMES >= 0.02) & (TIMES <= 0.6)]
    y = gauss(0.1, -5, 0.03, tw) + gauss(0.3, 3, 0.06, tw) + 0.4 + noise(rng, 0.8, n=len(tw))
    shapes = [get_shape("skewed_gaussian"), get_shape("gaussian"), get_shape("exgaussian")]
    a, innovation, acov = eng.ar_model(noise(rng, 0.8, n=400), return_acov=True) if ar else \
        (np.array([1.0]), 1.0, None)
    white = eng.Whitener(np.full(len(tw), 0.7), a, innovation, np.ones(len(tw), bool),
                         acov=acov)
    struct = eng._Structure(shapes, get_background(background), tw, white, y,
                            prior=2.0 if prior else None, scale=30.0 if prior else None,
                            edge=0.03, level=0.1, ordered=ordered, memo={})
    theta = np.r_[0.1, 0.012, 1.5, 0.3, 0.025, 0.45, 0.01, 0.02]
    if struct.n_bg:
        theta = np.r_[theta, 0.08]
    return struct, theta


@pytest.mark.parametrize("background", ["exponential", "constant", "linear"])
@pytest.mark.parametrize("ar, prior", [(True, True), (False, False), (True, False)])
def test_reduced_residual_and_jacobian_match_the_svd_route(background, ar, prior):

    struct, theta = _struct(background, ar, prior)
    beta, rw, _, q = struct.solve(theta)
    jac = struct.kaufman_jacobian(theta, beta, q)
    z, ctx = struct.reduced(theta)
    assert ctx[0] == "qr" and len(z) == len(rw) - len(beta)
    assert np.isclose(z @ z, rw @ rw, rtol=1e-10)
    jr = struct.reduced_jacobian(theta, ctx)
    # the same inner products: those are all the optimiser uses
    assert np.allclose(jr.T @ jr, jac.T @ jac, rtol=1e-9, atol=1e-9 * np.abs(jac.T @ jac).max())
    assert np.allclose(jr.T @ z, jac.T @ rw, rtol=1e-8, atol=1e-9 * np.abs(jac.T @ rw).max())
    beta2, rw2 = struct.reduced_solution(theta, ctx)
    assert np.allclose(beta2, beta, rtol=1e-9) and np.allclose(rw2, rw, rtol=1e-9, atol=1e-9)


def test_a_rank_deficient_design_takes_the_svd_route():

    base, _ = _struct()
    # two identical Gaussians: two equal columns
    struct = eng._Structure([get_shape("gaussian"), get_shape("gaussian")], base.background,
                            base.tw, base.white, base.y)
    theta = np.r_[0.1, 0.012, 0.1, 0.012, 0.08]
    z, ctx = struct.reduced(theta)
    assert ctx[0] == "svd"
    beta, rw, _, _ = struct.solve(theta)
    assert np.array_equal(z, rw)


@pytest.mark.parametrize("name", ["gaussian", "skewed_gaussian", "exgaussian", "gamma"])
def test_fused_columns_are_the_shapes_values(name):

    rng = np.random.default_rng(1)
    shape = get_shape(name)
    tw = np.linspace(-0.2, 0.8, 1501)
    lo, hi = shape.default_bounds(tw)
    for _ in range(50):
        p = lo + (hi - lo) * rng.uniform(0.01, 0.99, len(lo))
        values, aux = _columns.component(shape, tw, p)
        assert np.array_equal(values, shape.evaluate(tw, p))
        assert np.array_equal(_columns.component_derivatives(shape, tw, p, aux),
                              shape.jacobian(tw, p))


def test_exponential_background_columns():

    bg = get_background("exponential")
    tw = np.linspace(0.01, 0.5, 900)
    out = np.empty((2, len(tw)))
    aux = _columns.background(bg, tw, np.array([0.03]), out)
    assert np.array_equal(out, bg.basis(tw, np.array([0.03])))
    coefs = np.array([0.3, -2.0])
    assert np.array_equal(_columns.background_derivatives(bg, tw, np.array([0.03]), coefs, aux),
                          np.einsum("l,lnj->nj", coefs, bg.basis_jacobian(tw, np.array([0.03]))))


@pytest.mark.parametrize("name", ["gaussian", "skewed_gaussian"])
def test_gaussian_tails_left_out_are_exactly_zero(name):
    """Single parameter vectors skip the samples where exp(-z^2/2) underflows; the values
    are those of the batched evaluation, which computes every sample."""

    rng = np.random.default_rng(2)
    shape = get_shape(name)
    t = np.linspace(-1.0, 2.0, 30001)
    for _ in range(50):
        p = np.r_[rng.uniform(-0.5, 1.5), 10 ** rng.uniform(-4, -0.5),
                  rng.uniform(-10, 10, shape.n_params - 2)]
        single = shape.evaluate(t, p)
        batched = shape.evaluate(t, p[None, :])[0]
        assert np.array_equal(single, batched)
        assert np.array_equal(np.signbit(single), np.signbit(batched))


@pytest.mark.parametrize("name", ["gaussian", "skewed_gaussian", "exgaussian"])
def test_features_scale_with_the_amplitude(name):

    rng = np.random.default_rng(3)
    shape = get_shape(name, min_fwhm=0.003, max_fwhm=0.25)
    tw = np.linspace(0.009, 0.5, 1006)
    lo, hi = shape.default_bounds(tw)
    for _ in range(20):
        p = lo + (hi - lo) * rng.uniform(0.2, 0.8, len(lo))
        unit = compute_features(shape, tw, p, 1.0, sfreq=2048., window=(tw[0], tw[-1]))
        for amp in (rng.normal(0, 50), -3.0, 1e-3):
            direct = compute_features(shape, tw, p, amp, sfreq=2048., window=(tw[0], tw[-1]))
            scaled = api._scaled_features(unit, amp)
            for a, b in zip(direct.to_dict().values(), scaled.to_dict().values()):
                assert a == b or (np.isnan(a) and np.isnan(b))
