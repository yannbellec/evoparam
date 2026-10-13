# SPDX-License-Identifier: Apache-2.0
"""The fast paths of the engine compute what the reference paths compute (REQ-115 work):
the QR-reduced residual and Jacobian of the variable projection, the fused design columns,
the Gaussian factors left out where they are exactly zero, and the features scaled from
the unit-amplitude ones."""

from typing import Any

import numpy as np
import pytest

from evoparam.models import _columns, api
from evoparam.models import engine as eng
from evoparam.shapes import compute_features, get_background, get_shape

from .helpers import TIMES, gauss, noise

# Fit window (s) of the fast-path tests.
WINDOW_START = 0.02
WINDOW_END = 0.6

###################################################################################################
###################################################################################################


def _struct(
    background: Any = "exponential", ar: Any = True, prior: Any = True, ordered: Any = True
) -> Any:
    rng = np.random.default_rng(0)
    tw = TIMES[(TIMES >= WINDOW_START) & (TIMES <= WINDOW_END)]
    y = gauss(0.1, -5, 0.03, tw) + gauss(0.3, 3, 0.06, tw) + 0.4 + noise(rng, 0.8, n=len(tw))
    shapes = [get_shape("skewed_gaussian"), get_shape("gaussian"), get_shape("exgaussian")]
    a, innovation, acov = (
        eng.ar_model(noise(rng, 0.8, n=400), return_acov=True)
        if ar
        else (np.array([1.0]), 1.0, None)
    )
    white = eng.Whitener(np.full(len(tw), 0.7), a, innovation, np.ones(len(tw), bool), acov=acov)
    struct = eng._Structure(
        shapes,
        get_background(background),
        tw,
        white,
        y,
        prior=2.0 if prior else None,
        scale=30.0 if prior else None,
        edge=0.03,
        level=0.1,
        ordered=ordered,
        memo={},
    )
    theta = np.r_[0.1, 0.012, 1.5, 0.3, 0.025, 0.45, 0.01, 0.02]
    if struct.n_bg:
        theta = np.r_[theta, 0.08]
    return struct, theta


@pytest.mark.parametrize("background", ["exponential", "constant", "linear"])
@pytest.mark.parametrize(("ar", "prior"), [(True, True), (False, False), (True, False)])
def test_reduced_residual_and_jacobian_match_the_svd_route(
    background: Any, ar: Any, prior: Any
) -> None:

    struct, theta = _struct(background, ar, prior)
    beta, rw, _, q = struct.solve(theta)
    jac = struct.kaufman_jacobian(theta, beta, q)
    z, ctx = struct.reduced(theta)
    assert ctx[0] == "qr"
    assert len(z) == len(rw) - len(beta)
    assert np.isclose(z @ z, rw @ rw, rtol=1e-10)
    jr = struct.reduced_jacobian(theta, ctx)
    # the same inner products: those are all the optimiser uses
    assert np.allclose(jr.T @ jr, jac.T @ jac, rtol=1e-9, atol=1e-9 * np.abs(jac.T @ jac).max())
    assert np.allclose(jr.T @ z, jac.T @ rw, rtol=1e-8, atol=1e-9 * np.abs(jac.T @ rw).max())
    beta2, rw2 = struct.reduced_solution(theta, ctx)
    assert np.allclose(beta2, beta, rtol=1e-9)
    assert np.allclose(rw2, rw, rtol=1e-9, atol=1e-9)


def test_a_rank_deficient_design_takes_the_svd_route() -> None:

    base, _ = _struct()
    # two identical Gaussians: two equal columns
    struct = eng._Structure(
        [get_shape("gaussian"), get_shape("gaussian")],
        base.background,
        base.tw,
        base.white,
        base.y,
    )
    theta = np.r_[0.1, 0.012, 0.1, 0.012, 0.08]
    z, ctx = struct.reduced(theta)
    assert ctx[0] == "svd"
    beta, rw, _, _ = struct.solve(theta)
    assert np.array_equal(z, rw)


@pytest.mark.parametrize("name", ["gaussian", "skewed_gaussian", "exgaussian", "gamma"])
def test_fused_columns_are_the_shapes_values(name: Any) -> None:

    rng = np.random.default_rng(1)
    shape = get_shape(name)
    tw = np.linspace(-0.2, 0.8, 1501)
    lo, hi = shape.default_bounds(tw)
    for _ in range(50):
        p = lo + (hi - lo) * rng.uniform(0.01, 0.99, len(lo))
        values, aux = _columns.component(shape, tw, p)
        assert np.array_equal(values, shape.evaluate(tw, p))
        assert np.array_equal(
            _columns.component_derivatives(shape, tw, p, aux), shape.jacobian(tw, p)
        )


def test_exponential_background_columns() -> None:

    bg = get_background("exponential")
    tw = np.linspace(0.01, 0.5, 900)
    out = np.empty((2, len(tw)))
    aux = _columns.background(bg, tw, np.array([0.03]), out)
    assert np.array_equal(out, bg.basis(tw, np.array([0.03])))
    coefs = np.array([0.3, -2.0])
    assert np.array_equal(
        _columns.background_derivatives(bg, tw, np.array([0.03]), coefs, aux),
        np.einsum("l,lnj->nj", coefs, bg.basis_jacobian(tw, np.array([0.03]))),
    )


@pytest.mark.parametrize("name", ["gaussian", "skewed_gaussian"])
def test_gaussian_tails_left_out_are_exactly_zero(name: Any) -> None:
    """Single parameter vectors skip the samples where exp(-z^2/2) underflows; the values
    are those of the batched evaluation, which computes every sample."""

    rng = np.random.default_rng(2)
    shape = get_shape(name)
    t = np.linspace(-1.0, 2.0, 30001)
    for _ in range(50):
        p = np.r_[
            rng.uniform(-0.5, 1.5),
            10 ** rng.uniform(-4, -0.5),
            rng.uniform(-10, 10, shape.n_params - 2),
        ]
        single = shape.evaluate(t, p)
        batched = shape.evaluate(t, p[None, :])[0]
        assert np.array_equal(single, batched)
        assert np.array_equal(np.signbit(single), np.signbit(batched))


@pytest.mark.parametrize("name", ["gaussian", "skewed_gaussian", "exgaussian", "gamma"])
def test_features_scale_with_the_amplitude(name: Any) -> None:

    rng = np.random.default_rng(3)
    shape = get_shape(name, min_fwhm=0.003, max_fwhm=0.25)
    tw = np.linspace(0.009, 0.5, 1006)
    lo, hi = shape.default_bounds(tw)
    for _ in range(20):
        p = lo + (hi - lo) * rng.uniform(0.2, 0.8, len(lo))
        unit = compute_features(shape, tw, p, 1.0, sfreq=2048.0, window=(tw[0], tw[-1]))
        for amp in (rng.normal(0, 50), -3.0, 1e-3):
            direct = compute_features(shape, tw, p, amp, sfreq=2048.0, window=(tw[0], tw[-1]))
            scaled = api._scaled_features(unit, amp)
            for a, b in zip(direct.to_dict().values(), scaled.to_dict().values(), strict=False):
                assert a == b or (np.isnan(a) and np.isnan(b))


class _WholeGrid:
    """A shape without ``support``: its features are measured on the whole fine grid."""

    def __init__(self, shape: Any) -> None:
        self._shape = shape

    def __getattr__(self, name: str) -> Any:
        if name == "support":
            raise AttributeError(name)
        return getattr(self._shape, name)


@pytest.mark.parametrize("name", ["gaussian", "skewed_gaussian"])
def test_features_on_the_cut_grid_are_those_of_the_whole_grid(name: Any) -> None:
    """The grid is built only where the curve reaches the levels the features read, and the
    area reuses the curve on it: the same numbers, bit for bit."""

    rng = np.random.default_rng(4)
    shape = get_shape(name, min_fwhm=0.003, max_fwhm=0.25)
    tw = np.arange(0.009, 0.5, 1 / 2048.0)
    for _ in range(150):
        p = [rng.uniform(-0.1, 0.7), 10 ** rng.uniform(-3.5, -0.8)]
        p += [rng.uniform(-10, 10)] if name == "skewed_gaussian" else []
        for sfreq, onset in ((2048.0, 0.1), (512.0, 0.1), (2048.0, 0.01), (1024.0, 0.95)):
            kwargs = {"sfreq": sfreq, "window": (tw[0], tw[-1]), "onset_fraction": onset}
            cut = compute_features(shape, tw, p, -3.0, **kwargs).to_dict()
            whole = compute_features(_WholeGrid(shape), tw, p, -3.0, **kwargs).to_dict()
            for a, b in zip(cut.values(), whole.values(), strict=True):
                assert a == b or (np.isnan(a) and np.isnan(b))


def test_grid_search_is_searchsorted_on_the_built_grid() -> None:

    from evoparam.shapes._base import _grid_search, _grid_spec, feature_grid

    rng = np.random.default_rng(5)
    for _ in range(40):
        tw = np.arange(rng.uniform(-0.5, 0.1), rng.uniform(0.2, 2.0), 1 / rng.choice([256, 2048]))
        origin, step, n_points, _ = _grid_spec(tw, None, None)
        grid = feature_grid(tw)[0]
        picks = grid[rng.integers(0, len(grid), 20)]
        values = np.r_[
            rng.uniform(grid[0] - 1, grid[-1] + 1, 20),
            picks,
            np.nextafter(picks, np.inf),
            np.nextafter(picks, -np.inf),
            np.nan,
            np.inf,
            -np.inf,
        ]
        for value in values:
            for side in ("left", "right"):
                found = _grid_search(origin, step, n_points, float(value), side == "right")
                assert found == np.searchsorted(grid, value, side=side)


def test_copies_with_other_data_whiten_their_own() -> None:
    """The whitened data are shared by the structures of a fit, not by the copies that the
    bootstrap and the cross-validation make with other data."""

    import dataclasses

    struct, _ = _struct()
    struct._target()
    other = struct.y + noise(np.random.default_rng(6), 0.5, n=len(struct.y))
    copy = dataclasses.replace(struct, y=other)
    n = len(struct.tw)
    assert np.array_equal(copy._target()[0][:n], copy.white(other))
    assert np.array_equal(struct._target()[0][:n], struct.white(struct.y))


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9, 0.99])
def test_ar_orders_from_one_qr_are_the_least_squares_ones(rho: Any) -> None:
    """ar_model solves the nested least-squares problems from one QR factorisation."""

    rng = np.random.default_rng(int(rho * 100))
    for n in (60, 300, 1000):
        z = noise(rng, rho, n=n)
        a, innovation = eng.ar_model(z)
        p = len(a) - 1
        zc = z - np.mean(z)
        p_max = int(min(eng.MAX_AR_ORDER, n // eng.SAMPLES_PER_ORDER))
        if p:
            lagged = np.column_stack([zc[p_max - k : n - k] for k in range(1, p + 1)])
            phi = np.linalg.lstsq(lagged, zc[p_max:], rcond=None)[0]
            np.testing.assert_allclose(-a[1:], phi, rtol=1e-9, atol=1e-12)
