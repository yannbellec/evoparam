# SPDX-License-Identifier: Apache-2.0
"""Tests for the backgrounds (spec §5.5.2; REQ-100 for the equations)."""

import numpy as np
import pytest

from evoparam import shapes
from evoparam.errors import DataError, SettingsError
from evoparam.shapes import backgrounds as bg
from evoparam.tests.shapes import references
from evoparam.tests.shapes.helpers import TIMES, sample_background_params

WINDOW = (float(TIMES[0]), float(TIMES[-1]))


def test_linear_and_exponential_equations():
    t = TIMES
    t0 = t[0]
    np.testing.assert_array_equal(bg.NoBackground().evaluate(t, np.zeros(0)), np.zeros(t.size))
    assert bg.NoBackground().jacobian(t, np.zeros((4, 0))).shape == (4, t.size, 0)
    np.testing.assert_array_equal(bg.ConstantBackground().evaluate(t, [3.5]), np.full(t.size, 3.5))
    np.testing.assert_allclose(
        bg.LinearBackground().evaluate(t, [2.0, -5.0]), 2.0 - 5.0 * (t - t0), rtol=1e-15
    )
    np.testing.assert_allclose(
        bg.LinearBackground(t_ref=0.0).evaluate(t, [2.0, -5.0]), 2.0 - 5.0 * t, rtol=1e-15
    )
    np.testing.assert_allclose(
        bg.ExponentialBackground().evaluate(t, [1.0, 40.0, 0.01]),
        1.0 + 40.0 * np.exp(-(t - t0) / 0.01),
        rtol=1e-14,
    )
    double = bg.DoubleExponentialBackground(t_ref=0.0)
    np.testing.assert_allclose(
        double.evaluate(t, [1.0, 40.0, -8.0, 0.002, 0.05]),
        1.0 + 40.0 * np.exp(-t / 0.002) - 8.0 * np.exp(-t / 0.05),
        rtol=1e-13,
    )


def test_long_double_exponential():
    params = sample_background_params(
        bg.DoubleExponentialBackground(), TIMES, 20, np.random.default_rng(1), data_range=(-20, 20)
    )
    t = TIMES.astype(np.longdouble)
    for row in params:
        p = row.astype(np.longdouble)
        lag = t - t[0]
        expected = (p[0] + p[1] * np.exp(-lag / p[3]) + p[2] * np.exp(-lag / p[4])).astype(float)
        got = bg.DoubleExponentialBackground().evaluate(TIMES, row)
        assert np.all(np.abs(got - expected) <= 1e-12 * np.abs(expected) + 1e-12)


def test_parameter_names_and_linearity():
    assert bg.ExponentialBackground().param_names == ("c0", "c1", "tau")
    assert bg.DoubleExponentialBackground().nonlinear_names == ("tau1", "tau2")
    assert bg.ConstantBackground().is_linear and not bg.ExponentialBackground().is_linear
    spline = bg.PSplineBackground(WINDOW, max_fwhm=0.03)
    assert spline.is_linear and spline.n_params == spline.n_basis
    for name, cls in shapes.BACKGROUNDS.items():
        assert cls.name == name
    for background in (bg.NoBackground(), bg.ExponentialBackground(), spline):
        assert isinstance(background, shapes.Background)


def test_vectorised_backgrounds():
    rng = np.random.default_rng(2)
    for background in (
        bg.LinearBackground(),
        bg.ExponentialBackground(),
        bg.DoubleExponentialBackground(),
    ):
        params = sample_background_params(background, TIMES, 6, rng).reshape(3, 2, -1)
        values = background.evaluate(TIMES, params)
        jac = background.jacobian(TIMES, params)
        assert values.shape == (3, 2, TIMES.size)
        assert jac.shape == (3, 2, TIMES.size, background.n_params)
        np.testing.assert_allclose(
            values[2, 1], background.evaluate(TIMES, params[2, 1]), rtol=1e-14
        )
        np.testing.assert_allclose(jac[0, 1], background.jacobian(TIMES, params[0, 1]), rtol=1e-14)


def test_pspline_knot_rule():
    with pytest.raises(SettingsError, match="knot spacing"):
        bg.PSplineBackground(WINDOW, max_fwhm=0.03, knot_spacing=0.05)
    with pytest.raises(SettingsError):
        shapes.get_background("pspline", window=WINDOW, max_fwhm=0.03, knot_spacing=0.0599)
    for max_fwhm, spacing in [(0.03, None), (0.03, 0.06), (0.02, 0.07), (0.2, None)]:
        spline = bg.PSplineBackground(WINDOW, max_fwhm=max_fwhm, knot_spacing=spacing)
        assert spline.spacing >= 2 * max_fwhm
        np.testing.assert_allclose(np.diff(spline.knots), spline.spacing)
        assert spline.knots[3] == WINDOW[0] and spline.knots[-4] >= WINDOW[1]
    with pytest.raises(SettingsError):
        bg.PSplineBackground((0.1, 0.0), max_fwhm=0.03)


def test_pspline_basis():
    spline = bg.PSplineBackground(WINDOW, max_fwhm=0.03)
    basis = spline.basis(TIMES)
    assert basis.shape == (spline.n_basis, TIMES.size)
    # Partition of unity inside the window, zero outside.
    np.testing.assert_allclose(basis.sum(axis=0), 1.0, rtol=1e-13)
    assert np.all(spline.basis(np.array([WINDOW[0] - 0.01, WINDOW[1] + 1.0])) == 0)
    grid = TIMES[::23]
    for j in range(spline.n_basis):
        expected = [
            references.bspline(references.MATH, list(spline.knots), j, 3, float(t)) for t in grid
        ]
        np.testing.assert_allclose(basis[j, ::23], expected, rtol=1e-12, atol=1e-15)
    penalty = spline.penalty_matrix()
    assert penalty.shape == (spline.n_basis,) * 2
    # The penalty vanishes on straight lines.
    np.testing.assert_allclose(penalty @ np.arange(spline.n_basis), 0.0, atol=1e-12)


def test_pspline_basis_mpmath():
    """REQ-100 for the P-spline: Cox-de Boor recursion in mpmath at 1e-10."""

    mpmath = pytest.importorskip("mpmath")
    backend = references.mp_backend(mpmath, dps=40)
    spline = bg.PSplineBackground(WINDOW, max_fwhm=0.03)
    basis = spline.basis(TIMES)
    knots = [backend.mpf(float(k)) for k in spline.knots]
    for j in range(spline.n_basis):
        for i in range(0, TIMES.size, 37):
            expected = float(
                references.bspline(backend, knots, j, 3, backend.mpf(float(TIMES[i])))
            )
            assert abs(basis[j, i] - expected) <= 1e-10 * abs(expected) + 1e-15


def test_pspline_cannot_absorb_a_narrow_component():
    # A component of width max_fwhm is mostly left in the residual of the spline fit.
    spline = bg.PSplineBackground(WINDOW, max_fwhm=0.03)
    component = shapes.Gaussian().evaluate(TIMES, [0.15, 0.03 / 2.3548200450309493])
    design = spline.basis(TIMES).T
    fitted = design @ np.linalg.lstsq(design, component, rcond=None)[0]
    assert np.max(np.abs(component - fitted)) > 0.25


def test_pspline_select_lambda():
    rng = np.random.default_rng(0)
    times = np.arange(-0.5, 0.0, 1 / 1000)
    slow = 3.0 * np.sin(2 * np.pi * 1.5 * times)
    spline = bg.PSplineBackground(WINDOW, max_fwhm=0.02)
    lam = spline.select_lambda(times, slow + rng.normal(0, 1.0, times.size))
    assert 1e-6 <= lam <= 1e6
    rough = spline.select_lambda(times, rng.normal(0, 1.0, times.size))
    assert rough >= lam
    with pytest.raises(SettingsError):
        bg.PSplineBackground(WINDOW, max_fwhm=0.02, lam=-1.0)


def test_artifact_basis():
    basis_times = TIMES[:50]
    vectors = np.vstack([np.exp(-np.arange(50) / 10.0), np.cos(np.arange(50) / 4.0)])
    background = bg.ArtifactBasisBackground(vectors, basis_times)
    values = background.evaluate(TIMES, [2.0, -1.0])
    np.testing.assert_allclose(values[:50], 2.0 * vectors[0] - vectors[1], rtol=1e-15)
    assert np.all(values[50:] == 0)
    assert background.param_names == ("c0", "c1")
    with pytest.raises(DataError):
        bg.ArtifactBasisBackground(vectors[:, :10], basis_times)


@pytest.mark.parametrize(
    "background",
    [
        bg.ConstantBackground(),
        bg.LinearBackground(),
        bg.ExponentialBackground(),
        bg.DoubleExponentialBackground(),
    ],
    ids=lambda b: b.name,
)
def test_initial_guess_and_bounds(background):
    lower, upper = background.default_bounds(TIMES, data_range=(-30.0, 60.0))
    assert np.all(np.isfinite(lower)) and np.all(lower < upper)
    nonlinear = background._nonlinear_guess(TIMES)
    coefs = np.array([5.0, -12.0, 3.0][: background.n_linear])
    waveform = background.evaluate(TIMES, np.concatenate([coefs, nonlinear]))
    guess = background.initial_guess(TIMES, waveform)
    np.testing.assert_allclose(guess[: background.n_linear], coefs, rtol=1e-8, atol=1e-8)


def test_invalid_settings():
    with pytest.raises(SettingsError):
        bg.ExponentialBackground(tau_min=0.0)
    with pytest.raises(SettingsError):
        bg.DoubleExponentialBackground(tau_min=0.01, tau_max=0.02)
    with pytest.raises(DataError):
        bg.LinearBackground().evaluate(TIMES, [1.0])
