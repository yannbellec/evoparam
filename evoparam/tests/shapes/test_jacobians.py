# SPDX-License-Identifier: Apache-2.0
"""REQ-101: analytic Jacobians against central finite differences.

For every shape and background, 100 random parameter sets inside the default
bounds; the error of each Jacobian column is measured relative to the largest
entry of that column and must stay below 1e-6. The finite difference is
central with Richardson extrapolation (steps h and h/2). Samples within 100
steps of a point where the curve is not smooth in the perturbed parameter (the
onset of a causal shape, the edges of a template) are left out there, where
the finite difference itself is wrong.
"""

import numpy as np
import pytest

from evoparam import shapes
from evoparam.shapes import backgrounds as bg
from evoparam.tests.shapes.helpers import (
    TIMES,
    make_template,
    sample_background_params,
    sample_params,
)

N_SETS = 100
RTOL = 1e-6
REL_STEP = 1e-5

SHAPE_CASES = {
    "gaussian": shapes.Gaussian(),
    "gaussian_narrow": shapes.Gaussian(min_fwhm=0.004, max_fwhm=0.03),
    "skewed_gaussian": shapes.SkewedGaussian(),
    "exgaussian": shapes.ExGaussian(),
    "gamma": shapes.Gamma(),
    "biexponential": shapes.Biexponential(),
    "damped_oscillation": shapes.DampedOscillation(),
    "damped_oscillation_erna": shapes.DampedOscillation(freq_min=200.0, freq_max=450.0),
    "gabor": shapes.Gabor(),
    "template": make_template(),
    "template_stretch": make_template(stretch=True),
}

BACKGROUND_CASES = {
    "none": bg.NoBackground(),
    "constant": bg.ConstantBackground(),
    "linear": bg.LinearBackground(),
    "exponential": bg.ExponentialBackground(),
    "exponential_tref": bg.ExponentialBackground(t_ref=0.0),
    "double_exponential": bg.DoubleExponentialBackground(),
    "pspline": bg.PSplineBackground((TIMES[0], TIMES[-1]), max_fwhm=0.03),
    "artifact_basis": bg.ArtifactBasisBackground(
        np.vstack([np.exp(-np.arange(40) / 8.0), np.sin(np.arange(40) / 3.0)]), TIMES[:40]
    ),
}


def _steps(params, bounds):
    width = bounds[1] - bounds[0]
    return REL_STEP * np.maximum(np.abs(params), 1e-3 * width)


def _richardson(func, params, index, step):
    def central(h):
        delta = np.zeros(params.size)
        delta[index] = h
        return (func(params + delta) - func(params - delta)) / (2.0 * h)

    return (4.0 * central(step / 2.0) - central(step)) / 3.0


def _kinks(shape, params, steps):
    """Times at which the curve is not smooth, with the half-width to leave out."""

    if isinstance(shape, shapes.Template):
        # A step h in shift moves an edge by h, a step h in stretch by h·|edge|.
        stretch = params[1] if shape.stretch else 1.0
        edges = shape.template_times[[0, -1]]
        moves = [steps[0] + (steps[1] * abs(edge) if shape.stretch else 0.0) for edge in edges]
        return [
            (params[0] + stretch * edge, 100.0 * move)
            for edge, move in zip(edges, moves, strict=False)
        ]
    if shape.causal:
        index = shape.param_names.index(shape.onset_param)
        return [(params[index], 100.0 * steps[index])]
    return []


def _column_errors(func, analytic, params, bounds, kinks=()):
    steps = _steps(params, bounds)
    keep = np.ones(TIMES.size, bool)
    for time, half_width in kinks:
        keep &= np.abs(TIMES - time) > half_width
    errors = []
    for index in range(params.size):
        numeric = _richardson(func, params, index, steps[index])
        column = analytic[:, index]
        scale = max(np.max(np.abs(numeric[keep])), np.max(np.abs(column[keep])))
        if scale == 0:
            assert np.all(column[keep] == 0)
            continue
        errors.append(np.max(np.abs(column - numeric)[keep]) / scale)
    return errors


@pytest.mark.parametrize("case", sorted(SHAPE_CASES))
def test_shape_jacobian_matches_finite_differences(case):
    shape = SHAPE_CASES[case]
    bounds = shape.default_bounds(TIMES)
    params = sample_params(shape, TIMES, N_SETS, np.random.default_rng(101))
    analytic = shape.jacobian(TIMES, params)
    assert analytic.shape == (N_SETS, TIMES.size, shape.n_params)
    worst = 0.0
    for row in range(N_SETS):
        errors = _column_errors(
            lambda p: shape.evaluate(TIMES, p),
            analytic[row],
            params[row],
            bounds,
            _kinks(shape, params[row], _steps(params[row], bounds)),
        )
        worst = max([worst] + errors)
    assert worst < RTOL, f"{case}: worst relative Jacobian error {worst:.3g}"


@pytest.mark.parametrize("case", sorted(BACKGROUND_CASES))
def test_background_jacobian_matches_finite_differences(case):
    background = BACKGROUND_CASES[case]
    bounds = background.default_bounds(TIMES, data_range=(-50.0, 80.0))
    params = sample_background_params(
        background, TIMES, N_SETS, np.random.default_rng(7), data_range=(-50.0, 80.0)
    )
    analytic = background.jacobian(TIMES, params)
    assert analytic.shape == (N_SETS, TIMES.size, background.n_params)
    worst = 0.0
    for row in range(N_SETS):
        errors = _column_errors(
            lambda p: background.evaluate(TIMES, p), analytic[row], params[row], bounds
        )
        worst = max([worst] + errors)
    assert worst < RTOL, f"{case}: worst relative Jacobian error {worst:.3g}"


@pytest.mark.parametrize("case", ["exponential", "double_exponential"])
def test_basis_jacobian_matches_finite_differences(case):
    background = BACKGROUND_CASES[case]
    lower, upper = background.default_bounds(TIMES)
    params = sample_background_params(background, TIMES, 20, np.random.default_rng(3))
    nonlinear = params[:, background.n_linear :]
    bounds = (lower[background.n_linear :], upper[background.n_linear :])
    d_basis = background.basis_jacobian(TIMES, nonlinear)
    assert d_basis.shape == (20, background.n_linear, TIMES.size, background.n_nonlinear)
    for row in range(20):
        for j in range(background.n_linear):
            errors = _column_errors(
                lambda p: background.basis(TIMES, p)[j],  # noqa: B023  (called within this iteration only)
                d_basis[row, j],
                nonlinear[row],
                bounds,
            )
            assert max(errors, default=0.0) < RTOL


def test_registry_check_agrees_on_builtin_shapes():
    from evoparam.shapes.registry import jacobian_error

    for case, shape in SHAPE_CASES.items():
        bounds = shape.default_bounds(TIMES)
        params = sample_params(shape, TIMES, 30, np.random.default_rng(5))
        error = jacobian_error(shape, TIMES, params, bounds=bounds)
        assert np.nanmax(error) < RTOL, case
        assert np.mean(np.isnan(error)) < 0.2, case
