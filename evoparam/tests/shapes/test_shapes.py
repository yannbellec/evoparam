# SPDX-License-Identifier: Apache-2.0
"""Tests for the component shapes (spec §5.5.1; REQ-100).

REQ-100 compares every shape with values computed independently in mpmath at
1e-10 relative error; that test is skipped when mpmath is not installed. The
same scalar references run in float64 (``references.MATH``) and the closed
forms run in ``np.longdouble``, so the equations are also checked without
mpmath.
"""

import math
from typing import Any

import numpy as np
import pytest
from scipy.optimize import brentq

from evoparam import shapes
from evoparam.errors import DataError, SettingsError
from evoparam.shapes.components import gamma_fwhm_factor
from evoparam.tests.shapes import references
from evoparam.tests.shapes.helpers import SFREQ, TIMES, make_template, sample_params

ANALYTIC = {
    "gaussian": shapes.Gaussian(),
    "skewed_gaussian": shapes.SkewedGaussian(),
    "exgaussian": shapes.ExGaussian(),
    "gamma": shapes.Gamma(),
    "biexponential": shapes.Biexponential(),
    "damped_oscillation": shapes.DampedOscillation(),
    "gabor": shapes.Gabor(),
}
ALL = dict(ANALYTIC, template=make_template(), template_stretch=make_template(stretch=True))

# Parameter sets of moderate range, where the float64 references are accurate.
MODERATE = {
    "gaussian": [[0.1, 0.02], [0.012, 0.004]],
    "skewed_gaussian": [[0.1, 0.02, 3.0], [0.05, 0.01, -6.0], [0.08, 0.015, 0.0]],
    "exgaussian": [[0.1, 0.02, 0.03], [0.05, 0.008, 0.002]],
    "gamma": [[0.02, 3.0, 0.02], [-0.01, 1.6, 0.01], [0.0, 12.0, 0.005]],
    "biexponential": [[0.02, 0.005, 0.03], [0.0, 0.002, 0.1]],
    "damped_oscillation": [[0.02, 0.05, 30.0, 1.0], [0.01, 0.01, 300.0, -2.5]],
    "gabor": [[0.1, 0.02, 20.0, 2.0], [0.05, 0.004, 250.0, -0.4]],
}


def test_registry_has_the_eight_shapes_and_seven_backgrounds() -> None:
    assert set(shapes.SHAPES) == {
        "gaussian",
        "skewed_gaussian",
        "exgaussian",
        "gamma",
        "biexponential",
        "damped_oscillation",
        "gabor",
        "template",
    }
    assert set(shapes.BACKGROUNDS) == {
        "none",
        "constant",
        "linear",
        "exponential",
        "double_exponential",
        "pspline",
        "artifact_basis",
    }
    for _name, shape in ALL.items():
        assert isinstance(shape, shapes.ComponentShape)
        assert shape.latency_param in shape.param_names
        assert shape.n_params == len(shape.param_names)


@pytest.mark.parametrize("name", sorted(MODERATE))
def test_values_match_float_references(name: Any) -> None:
    shape = ANALYTIC[name]
    for params in MODERATE[name]:
        expected = np.array(references.reference_curve(references.MATH, name, TIMES, params))
        np.testing.assert_allclose(shape.evaluate(TIMES, params), expected, rtol=1e-9, atol=1e-13)


@pytest.mark.parametrize("name", sorted(ANALYTIC))
def test_values_match_mpmath_references(name: Any) -> None:
    """REQ-100: values on a reference grid at 1e-10 relative error against mpmath."""

    mpmath = pytest.importorskip("mpmath")
    backend = references.mp_backend(mpmath, dps=40)
    shape = ANALYTIC[name]
    params = np.vstack(
        [MODERATE[name], sample_params(shape, TIMES, 4, np.random.default_rng(2026))]
    )
    grid = TIMES[::8]
    for row in params:
        expected = np.array(references.reference_curve(backend, name, grid, row))
        got = shape.evaluate(grid, row)
        assert np.all(np.abs(got - expected) <= 1e-10 * np.abs(expected) + 1e-13), (
            f"{name} {row.tolist()}"
        )


def test_closed_forms_in_long_double() -> None:
    rng = np.random.default_rng(11)
    t = TIMES.astype(np.longdouble)
    for name in ("gaussian", "gamma", "biexponential"):
        shape = ANALYTIC[name]
        for row in sample_params(shape, TIMES, 30, rng):
            p = row.astype(np.longdouble)
            if name == "gaussian":
                expected = np.exp(-0.5 * ((t - p[0]) / p[1]) ** 2)
            elif name == "gamma":
                u = np.maximum(t - p[0], 0)
                a = p[1] - 1
                expected = np.where(t > p[0], (u / (a * p[2])) ** a * np.exp(a - u / p[2]), 0)
            else:
                u = np.maximum(t - p[0], 0)
                delay = p[1] * p[2] / (p[2] - p[1]) * np.log(p[2] / p[1])
                peak = np.exp(-delay / p[2]) - np.exp(-delay / p[1])
                expected = np.where(t > p[0], (np.exp(-u / p[2]) - np.exp(-u / p[1])) / peak, 0)
            got = shape.evaluate(TIMES, row)
            assert np.all(
                np.abs(got - expected.astype(float))
                <= 1e-12 * np.abs(expected.astype(float)) + 1e-15
            ), name


def test_template_equation() -> None:
    shape = make_template()
    nodes, values = shape.template_times, shape.template_values
    # The interpolant goes through the samples; the norm is the maximum of |T|.
    np.testing.assert_allclose(shape.evaluate(nodes, [0.0]) * shape.norm, values, atol=1e-15)
    assert np.max(np.abs(values)) <= shape.norm <= 1.001 * np.max(np.abs(values))
    stretched = make_template(stretch=True)
    for shift, stretch in [(0.01, 1.0), (-0.02, 0.85), (0.03, 1.2)]:
        np.testing.assert_allclose(
            stretched.evaluate(TIMES, [shift, stretch]),
            stretched.evaluate((TIMES - shift) / stretch, [0.0, 1.0]),
            rtol=1e-13,
            atol=1e-15,
        )
    assert np.all(shape.evaluate(np.array([-0.5, 0.5]), [0.0]) == 0.0)


@pytest.mark.parametrize("name", sorted(ALL))
def test_unit_amplitude(name: Any) -> None:
    shape = ALL[name]
    params = sample_params(shape, TIMES, 100, np.random.default_rng(3))
    peak = shape.extremum(params)
    values = np.array(
        [shape.evaluate(np.array([t]), p)[0] for t, p in zip(peak, params, strict=False)]
    )
    np.testing.assert_allclose(np.abs(values), 1.0, rtol=0, atol=1e-12)
    for row, center in zip(params, peak, strict=False):
        fine = center + np.linspace(-0.05, 0.05, 20001)
        assert np.max(np.abs(shape.evaluate(fine, row))) <= 1.0 + 1e-12


@pytest.mark.parametrize("name", sorted(ALL))
def test_extremum_is_a_stationary_point(name: Any) -> None:
    # Every shape is a function of t minus its latency parameter, so ds/dt = -ds/d(latency).
    shape = ALL[name]
    params = sample_params(shape, TIMES, 50, np.random.default_rng(4))
    index = shape.param_names.index(shape.latency_param)
    for row in params:
        peak = float(shape.extremum(row))
        if shape.causal and peak == row[index]:
            continue  # extremum at the onset of a damped oscillation
        span = 0.01 * np.linspace(-1, 1, 201)
        slope = shape.jacobian(np.array([peak]), row)[0, index]
        scale = np.max(np.abs(shape.jacobian(peak + span, row)[:, index]))
        assert abs(slope) <= 1e-7 * scale


def test_latency_formulas() -> None:
    gaussian = shapes.Gaussian()
    params = sample_params(gaussian, TIMES, 20, np.random.default_rng(0))
    np.testing.assert_array_equal(gaussian.latency(params), params[:, 0])

    gamma = shapes.Gamma()
    params = sample_params(gamma, TIMES, 20, np.random.default_rng(0))
    np.testing.assert_allclose(
        gamma.latency(params), params[:, 0] + (params[:, 1] - 1) * params[:, 2], rtol=1e-15
    )

    biexp = shapes.Biexponential()
    params = sample_params(biexp, TIMES, 20, np.random.default_rng(0))
    t0, tau_r, tau_d = params.T
    np.testing.assert_allclose(
        biexp.latency(params),
        t0 + tau_r * tau_d / (tau_d - tau_r) * np.log(tau_d / tau_r),
        rtol=1e-14,
    )

    for name in ("damped_oscillation", "gabor"):
        shape = ANALYTIC[name]
        params = sample_params(shape, TIMES, 20, np.random.default_rng(0))
        np.testing.assert_array_equal(shape.latency(params), params[:, 0])

    template = make_template(stretch=True)
    peak = template.template_times[np.argmax(np.abs(template.template_values))]
    np.testing.assert_allclose(template.latency([0.02, 1.1]), 0.02 + 1.1 * peak, atol=2e-3)
    assert template.evaluate(np.array([float(template.latency([0.02, 1.1]))]), [0.02, 1.1])[
        0
    ] == pytest.approx(-1.0, abs=1e-12)


def test_skewed_gaussian_reduces_to_gaussian_and_skews() -> None:
    skewed, gaussian = shapes.SkewedGaussian(), shapes.Gaussian()
    np.testing.assert_allclose(
        skewed.evaluate(TIMES, [0.1, 0.02, 0.0]), gaussian.evaluate(TIMES, [0.1, 0.02]), rtol=1e-15
    )
    right = skewed.features(TIMES, [0.1, 0.02, 4.0])
    left = skewed.features(TIMES, [0.1, 0.02, -4.0])
    assert right.symmetry > 0.1
    assert left.symmetry == pytest.approx(-right.symmetry, abs=1e-6)
    assert right.latency > 0.1 > left.latency


def test_exgaussian_mode_and_limits() -> None:
    shape = shapes.ExGaussian()
    # Short tau: the shape tends to the Gaussian, the mode to mu.
    np.testing.assert_allclose(
        shape.evaluate(TIMES, [0.1, 0.02, 1e-5]),
        shapes.Gaussian().evaluate(TIMES, [0.1 + 1e-5, 0.02]),
        atol=1e-6,
    )
    features = shape.features(TIMES, [0.1, 0.01, 0.03])
    assert features.symmetry > 0.2
    assert features.latency > 0.1


def test_gaussian_features_closed_forms() -> None:
    mu, sigma, amplitude = 0.1, 0.012, -7.5
    features = shapes.Gaussian().features(TIMES, [mu, sigma], amplitude)
    half_width = sigma * math.sqrt(2 * math.log(10))
    assert features.latency == mu
    assert features.peak_time == mu
    assert features.amplitude == amplitude
    assert features.polarity == -1
    assert features.fwhm == pytest.approx(2.3548200450309493 * sigma, rel=1e-5)
    assert features.onset == pytest.approx(mu - half_width, abs=1e-7)
    assert features.offset == pytest.approx(mu + half_width, abs=1e-7)
    assert features.rise_time == pytest.approx(half_width, abs=1e-7)
    assert features.symmetry == pytest.approx(0.0, abs=1e-6)
    area = amplitude * sigma * math.sqrt(2 * math.pi) * math.erf(math.sqrt(math.log(10)))
    assert features.area == pytest.approx(area, rel=1e-6)
    assert features.sharpness == pytest.approx(8 * math.log(2), rel=1e-5)
    assert features.peak_times == pytest.approx((mu,), abs=1e-9)
    assert features.warnings == ()
    assert set(features.to_dict()) >= {"latency", "fwhm", "onset", "offset", "area"}


def test_onset_fraction_and_sampling_rate() -> None:
    shape = shapes.Gaussian()
    sigma = 0.01
    features = shape.features(TIMES, [0.1, sigma], onset_fraction=0.5)
    assert features.offset - features.onset == pytest.approx(features.fwhm, rel=1e-6)
    coarse = shape.features(TIMES, [0.1, sigma], sfreq=SFREQ / 4)
    assert coarse.fwhm == pytest.approx(2.3548200450309493 * sigma, rel=1e-4)
    with pytest.raises(SettingsError):
        shape.features(TIMES, [0.1, sigma], onset_fraction=1.5)


def test_gamma_features_closed_forms() -> None:
    t0, kappa, theta = 0.01, 4.0, 0.008
    features = shapes.Gamma().features(TIMES, [t0, kappa, theta], 3.0)
    a = kappa - 1

    def half(x):
        return a * math.log(x / a) + a - x + math.log(2)

    width = theta * (brentq(half, a, 50 * a) - brentq(half, 1e-9, a))
    assert features.latency == pytest.approx(t0 + a * theta, abs=1e-15)
    assert features.fwhm == pytest.approx(width, rel=1e-5)
    assert float(gamma_fwhm_factor(kappa)) * theta == pytest.approx(width, rel=1e-12)
    assert features.amplitude == pytest.approx(3.0, rel=1e-12)
    assert features.onset > t0
    assert features.symmetry > 0


def test_biexponential_features() -> None:
    features = shapes.Biexponential().features(TIMES, [0.0, 0.002, 0.02])
    delay = 0.002 * 0.02 / 0.018 * math.log(10)
    assert features.latency == pytest.approx(delay, rel=1e-14)
    assert features.amplitude == pytest.approx(1.0, abs=1e-12)
    assert features.rise_time < features.decay_time


def test_damped_oscillation_features() -> None:
    t0, tau, freq = 0.02, 0.03, 300.0
    shape = shapes.DampedOscillation()
    features = shape.features(TIMES, [t0, tau, freq, 0.0], 2.0)
    assert features.latency == t0
    assert features.peak_time == t0
    assert features.amplitude == pytest.approx(2.0, abs=1e-12)
    # Successive extrema are half a period apart.
    np.testing.assert_allclose(np.diff(features.peak_times[1:]), 1 / (2 * freq), atol=2e-6)
    # Phase pi/2: the onset is a zero crossing, the extremum comes later.
    shifted = shape.features(TIMES, [t0, tau, freq, np.pi / 2])
    assert shifted.latency == t0
    assert shifted.peak_time > t0
    assert abs(shifted.amplitude) == pytest.approx(1.0, abs=1e-12)


def test_gabor_features() -> None:
    shape = shapes.Gabor()
    features = shape.features(TIMES, [0.1, 0.01, 50.0, 0.0])
    assert features.latency == 0.1
    assert features.peak_time == pytest.approx(0.1, abs=1e-12)
    flipped = shape.features(TIMES, [0.1, 0.01, 50.0, np.pi], 2.0)
    assert flipped.amplitude == pytest.approx(-2.0, abs=1e-12)
    assert flipped.polarity == -1


def test_latency_outside_window_is_reported_not_dropped() -> None:
    features = shapes.Gaussian().features(TIMES, [TIMES[0] - 0.004, 0.003], 5.0)
    assert features.latency == TIMES[0] - 0.004
    assert "latency_outside_window" in features.warnings
    assert np.isfinite(features.fwhm)


def test_features_of_an_extremum_after_the_feature_grid() -> None:
    """A gamma inside its default bounds can peak after the feature grid (here at 1.4 s,
    the grid ends near 0.98 s): the half-height search read one sample past the grid
    (IndexError). The width is not measurable there; the rest is reported."""

    tw = np.linspace(0.009, 0.5, 1006)
    shape = shapes.Gamma(min_fwhm=0.003, max_fwhm=0.25)
    params = np.array([0.0, 15.0, 0.1])
    lower, upper = shape.default_bounds(tw)
    assert np.all((lower <= params) & (params <= upper))
    features = shape.features(tw, params, 2.0, sfreq=2048.0)
    assert features.latency == pytest.approx(1.4, abs=1e-12)
    assert np.isnan(features.fwhm)
    assert np.isnan(features.offset)
    assert {"fwhm_outside_grid", "offset_outside_grid", "latency_outside_window"} <= set(
        features.warnings
    )
    assert tw[-1] < features.onset < 1.4
    assert features.amplitude == pytest.approx(2.0)
    # the same for a Gaussian placed after the grid (IndexError as well)
    late = shapes.Gaussian().features(tw, [2.0, 0.05], sfreq=2048.0)
    assert np.isnan(late.fwhm)
    assert "fwhm_outside_grid" in late.warnings


def test_features_of_an_extremum_before_the_feature_grid() -> None:
    """Before the grid, the right-hand crossing used sample -1, which wraps to the end of
    the grid (FWHM -1.5e23 s here); the width is now NaN with its warning."""

    tw = np.linspace(0.009, 0.5, 1006)
    features = shapes.Gaussian().features(tw, [-1.0, 0.05], sfreq=2048.0)
    assert np.isnan(features.fwhm)
    assert np.isnan(features.sharpness)
    assert {"fwhm_outside_grid", "latency_outside_window"} <= set(features.warnings)


@pytest.mark.parametrize("sigma", [1e-5, 5e-6])
def test_fwhm_of_a_lobe_narrower_than_the_feature_grid(sigma: Any) -> None:
    """No grid sample next to the extremum reaches half its height: each crossing is
    interpolated between the extremum and its neighbouring sample, so the width is within
    one grid step of the truth (it was 5 times the truth, or 0.41 s)."""

    tw = np.linspace(0.009, 0.5, 1006)
    step = 1 / (10 * 2048.0)
    assert 2.3548200450309493 * sigma < step
    features = shapes.Gaussian().features(tw, [0.20001, sigma], sfreq=2048.0)
    assert features.fwhm > 0
    assert abs(features.fwhm - 2.3548200450309493 * sigma) < step
    assert "fwhm_outside_grid" not in features.warnings


@pytest.mark.parametrize("name", sorted(ALL))
def test_features_on_the_whole_box_of_parameters(name: Any) -> None:
    """Every shape gives features (no exception) on parameters drawn over its default
    bounds, corners included, in two windows; a width that is not NaN is positive and
    NaN widths, onsets and offsets carry their warnings."""

    shape = ALL[name]
    rng = np.random.default_rng(31)
    for tw, sfreq in ((np.linspace(0.009, 0.5, 1006), 2048.0), (TIMES, SFREQ)):
        lower, upper = shape.default_bounds(tw)
        for i in range(120):
            u = rng.random(shape.n_params) if i < 80 else rng.integers(0, 2, shape.n_params)
            features = shape.features(tw, lower + (upper - lower) * u, -3.0, sfreq=sfreq)
            assert features.fwhm > 0 or (
                "fwhm_outside_grid" in features.warnings and np.isnan(features.fwhm)
            )
            for key in ("onset", "offset"):
                assert (
                    np.isfinite(getattr(features, key))
                    or f"{key}_outside_grid" in features.warnings
                )


def test_vectorised_shapes() -> None:
    rng = np.random.default_rng(8)
    for name, shape in ALL.items():
        params = sample_params(shape, TIMES, 6, rng).reshape(2, 3, shape.n_params)
        values = shape.evaluate(TIMES, params)
        jac = shape.jacobian(TIMES, params)
        assert values.shape == (2, 3, TIMES.size), name
        assert jac.shape == (2, 3, TIMES.size, shape.n_params), name
        np.testing.assert_allclose(
            values[1, 2], shape.evaluate(TIMES, params[1, 2]), rtol=1e-14, atol=1e-300
        )
        np.testing.assert_allclose(
            jac[0, 1], shape.jacobian(TIMES, params[0, 1]), rtol=1e-14, atol=1e-300
        )
        assert shape.latency(params).shape == (2, 3)


def test_default_bounds_follow_the_settings() -> None:
    lower, upper = shapes.Gaussian(min_fwhm=0.004, max_fwhm=0.05).default_bounds(TIMES)
    np.testing.assert_allclose([lower[1], upper[1]], np.array([0.004, 0.05]) / 2.3548200450309493)
    assert lower[0] == pytest.approx(TIMES[0] - upper[1] / 2)
    lower, upper = shapes.DampedOscillation(freq_min=200, freq_max=450).default_bounds(TIMES)
    assert (lower[2], upper[2]) == (200, 450)
    assert (lower[3], upper[3]) == (-np.pi, np.pi)
    lower, upper = shapes.Gamma(min_fwhm=0.004, max_fwhm=0.05).default_bounds(TIMES)
    assert 0.004 / gamma_fwhm_factor(20) == pytest.approx(lower[2])
    lower, upper = make_template(stretch=True).default_bounds(TIMES)
    assert list(lower[1:]) == [0.8]
    assert list(upper[1:]) == [1.25]


@pytest.mark.parametrize(
    "make",
    [
        lambda: shapes.Gaussian(min_fwhm=0.05, max_fwhm=0.01),
        lambda: shapes.Gaussian(min_fwhm=-1.0),
        lambda: shapes.SkewedGaussian(alpha_max=0.0),
        lambda: shapes.Gamma(kappa_min=1.0),
        lambda: shapes.Gabor(freq_min=400, freq_max=100),
        lambda: shapes.Template(np.arange(10.0), np.zeros(10)),
    ],
)
def test_invalid_settings(make: Any) -> None:
    with pytest.raises((SettingsError, DataError)):
        make()


def test_invalid_params_and_times() -> None:
    shape = shapes.Gaussian()
    with pytest.raises(DataError):
        shape.evaluate(TIMES, [0.1, 0.01, 1.0])
    with pytest.raises(DataError):
        shape.evaluate(np.zeros((2, 3)), [0.1, 0.01])
    with pytest.raises(DataError):
        shape.features(TIMES, np.zeros((2, 2)))
    with pytest.raises(SettingsError):
        shapes.Gaussian(max_fwhm=0.001).default_bounds(TIMES)


@pytest.mark.parametrize("name", sorted(ALL))
def test_initial_guess_inside_bounds(name: Any) -> None:
    shape = ALL[name]
    waveform = -4.0 * shapes.Gaussian().evaluate(TIMES, [0.08, 0.01])
    peak = int(np.argmax(np.abs(waveform)))
    guess = shape.initial_guess(TIMES, waveform, peak)
    lower, upper = shape.default_bounds(TIMES)
    assert guess.shape == (shape.n_params,)
    assert np.all(guess >= lower)
    assert np.all(guess <= upper)


def test_initial_guess_recovers_a_gaussian() -> None:
    shape = shapes.Gaussian()
    waveform = 3.0 * shape.evaluate(TIMES, [0.0805, 0.01])
    guess = shape.initial_guess(TIMES, waveform, int(np.argmax(waveform)))
    assert guess[0] == pytest.approx(0.0805, abs=0.5 / SFREQ)
    assert guess[1] == pytest.approx(0.01, rel=0.02)
    # Edge component: the window starts at the peak, so the width comes from one side.
    start = int(np.argmax(waveform))
    edge = shape.initial_guess(TIMES[start:], waveform[start:], 0)
    assert edge[1] == pytest.approx(0.01, rel=0.05)
    # No half maximum at all inside the window: fallback to min_fwhm.
    flat = shape.initial_guess(TIMES[:40], np.linspace(1.0, 0.9, 40), 0)
    assert flat[1] == pytest.approx(2.0 / SFREQ / 2.3548200450309493)
