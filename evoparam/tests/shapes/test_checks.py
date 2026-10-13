# SPDX-License-Identifier: Apache-2.0
"""REQ-298 (evoparam.shapes covered at 100 %, lines and branches): the input checks, the
fallbacks and the edge cases of the shapes, the backgrounds and the registry, each with the
result or the error that it must give.

REQ-102: every check of ``register_shape`` and of the plugin entry points names the check
that failed.
"""

import dataclasses
import warnings
from typing import Any, ClassVar

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal, assert_equal

from evoparam import _plugins, shapes
from evoparam.errors import DataError, SettingsError
from evoparam.shapes import _base, components, registry
from evoparam.shapes import backgrounds as bg
from evoparam.tests.shapes.helpers import TIMES
from evoparam.tests.shapes.test_registry import Lorentzian

pytestmark = pytest.mark.core


@pytest.fixture
def clean_registry() -> Any:
    before = dict(shapes.SHAPES)
    yield
    shapes.SHAPES.clear()
    shapes.SHAPES.update(before)


###################################################################################################
## _base: time axes, waveforms, bounds, crossings, grids


def test_time_axis_checks() -> None:
    assert _base.as_times(0.5).shape == (1,)
    with pytest.raises(DataError, match="1-D"):
        _base.as_times(np.zeros((2, 2)))
    with pytest.raises(DataError, match="finite"):
        _base.as_times([0.0, np.nan])
    with pytest.raises(DataError, match="at least 2 samples"):
        _base.window_info([0.0])
    with pytest.raises(DataError, match="strictly increasing"):
        _base.window_info([0.0, 0.2, 0.1])


def test_waveform_and_bounds_checks() -> None:
    with pytest.raises(DataError, match="differs from times"):
        _base.check_waveform([0.0, 0.1, 0.2], [1.0, 2.0])
    with pytest.raises(DataError, match=r"peak_index: 3 is outside 0\.\.2"):
        _base.check_waveform([0.0, 0.1, 0.2], [1.0, 2.0, 1.0], peak_index=3)
    with pytest.raises(SettingsError, match="empty parameter bounds"):
        _base.check_bounds("gaussian", [0.0, 1.0], [1.0, 1.0])


def test_peak_fwhm_of_a_zero_peak_is_none() -> None:
    times = np.linspace(0.0, 0.1, 11)
    assert _base.peak_fwhm(times, np.zeros(11), 5) is None


def test_crossings() -> None:
    grid = np.array([0.0, 1.0, 2.0])
    with pytest.raises(IndexError, match="samples -1 and 0 of a grid of 3"):
        _base._interp_cross(grid, np.array([0.0, 1.0, 0.0]), -1, 0, 0.5)
    # a flat segment: the crossing is at its first sample
    assert_equal(_base._interp_cross(grid, np.array([1.0, 1.0, 0.0]), 0, 1, 1.0), 0.0)
    assert_equal(_base._interp_points(0.25, 2.0, 0.75, 2.0, 2.0), 0.25)
    assert_equal(_base._interp_points(0.0, 0.0, 1.0, 2.0, 1.0), 0.5)


def test_feature_grid_checks() -> None:
    with pytest.raises(SettingsError, match="sfreq: must be a positive number"):
        _base.feature_grid(TIMES, sfreq=0.0)
    with pytest.raises(SettingsError, match="sfreq: must be a positive number"):
        _base.feature_grid(TIMES, sfreq=np.inf)
    with pytest.raises(SettingsError, match="window: end must be after start"):
        _base.feature_grid(TIMES, window=(0.1, 0.1))


def test_grid_extremum_at_an_edge_and_on_a_flat_top() -> None:
    grid = np.arange(5, dtype=float)
    # the largest value on the first or the last sample: that sample, not refined
    assert_equal(_base._grid_extremum(grid, np.array([3.0, 2.0, 1.0, 0.0, 0.0])), 0.0)
    assert_equal(_base._grid_extremum(grid, np.array([0.0, 0.0, 1.0, 2.0, -3.0])), 4.0)
    # a top whose second difference rounds to exactly 0 (1 - 2^-53, 1, 1): the sample itself
    top = np.array([0.0, 1.0 - 2.0**-53, 1.0, 1.0, 0.0])
    assert_equal(top[1] - 2.0 * top[2] + top[3], 0.0)
    assert_equal(_base._grid_extremum(grid, top), 2.0)
    # elsewhere the parabola through the three samples refines it
    assert_equal(_base._grid_extremum(grid, np.array([0.0, 1.0, 2.0, 1.0, 0.5])), 2.0)
    refined = _base._grid_extremum(grid, np.array([0.0, 1.0, 2.0, 1.5, 0.5]))
    assert_allclose(refined, 2.0 + 0.5 / 3.0, rtol=1e-15)


def test_features_of_a_curve_above_the_onset_level_at_the_grid_start() -> None:
    # a Gaussian much wider than the window: above 10 % of its peak over the whole grid
    feats = shapes.Gaussian().features(TIMES, [0.0, 10.0])
    assert_array_equal(np.isnan([feats.onset, feats.offset, feats.area, feats.symmetry]), True)
    assert "onset_outside_grid" in feats.warnings
    assert "offset_outside_grid" in feats.warnings


###################################################################################################
## components: helpers, settings, fallbacks


def test_bisection_of_an_undefined_bracket_returns_nan() -> None:
    # a NaN bound never converges: the iteration stops after its fixed number of steps
    at = 0.3
    root = components._bisect(lambda x: x > at, [0.0, np.nan], [1.0, 1.0])
    assert_allclose(root[0], at, rtol=1e-12)
    assert np.isnan(root[1])


def test_gaussian_factor_outside_its_support_is_zero() -> None:
    z = np.array([50.0, -60.0, 100.0])

    def value(_z: Any) -> Any:
        raise AssertionError("evaluated outside the support")

    assert_array_equal(components.gaussian_factor(z, value), np.zeros(3))
    # a few points outside: those are 0, the others computed
    z = np.array([0.0, 1.0, 2.0, 40.0, 0.5])
    out = components.gaussian_factor(z, lambda x: np.exp(-0.5 * x**2))
    assert_allclose(out, np.exp(-0.5 * z**2), rtol=1e-15)
    assert_equal(out[3], 0.0)


def test_skew_mode_of_one_value_equals_the_array_path() -> None:
    for alpha in (2.5, -4.0, 1e12):  # 1e12: the iteration stops after its 60 steps
        one = components.skew_mode(alpha)
        many = components.skew_mode(np.array([alpha, alpha]))
        for a, b in zip(one, many, strict=True):
            assert_allclose(float(a), b, rtol=1e-12, atol=0.0)
        # the value at the mode is above that of the Gaussian (1)
        assert float(one[1]) > 1.0


def test_skew_mode_of_a_non_finite_or_huge_skewness() -> None:
    # NaN gives NaN (it recursed without end before), without a warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = components.skew_mode(np.nan)
    assert_array_equal(np.isnan(result), True)
    z, _, _ = components.skew_mode(np.array([np.nan, 1.0]))
    assert_array_equal(np.isfinite(z), [False, True])
    # a skewness whose square overflows: the half-normal limit (mode 0, value 2), on both
    # paths (the scalar path raised OverflowError before)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # IEEE overflow warnings of NumPy
        one = components.skew_mode(1e155)
        many = components.skew_mode(np.array([1e155, 1e155]))
    assert_allclose(float(one[0]), 0.0, atol=1e-14)
    assert_equal(float(one[1]), 2.0)
    assert_array_equal(many[0], [float(one[0])] * 2)
    assert_array_equal(many[1], [2.0, 2.0])


def test_skew_mode_cache_is_bounded() -> None:
    size = components._MODE_CACHE_SIZE
    components._MODE_CACHE.clear()
    for k in range(size + 1):
        components.skew_mode(0.5 + k * 1e-3)
    assert 1 <= len(components._MODE_CACHE) <= size
    for k in range(size + 1):
        components.skew_mode(np.array([0.5, 1.0 + k * 1e-3]))
    assert 1 <= len(components._MODE_CACHE) <= size


def test_oscillation_settings_are_checked() -> None:
    with pytest.raises(SettingsError, match="freq_min must be a positive number"):
        shapes.DampedOscillation(freq_min=-1.0)
    with pytest.raises(SettingsError, match="freq_max must be a positive number"):
        shapes.Gabor(freq_max=np.inf)
    # freq_min above sfreq/4 of the window: no admissible frequency
    with pytest.raises(SettingsError, match=r"freq_min \(600.0\) must be smaller"):
        shapes.Gabor(freq_min=600.0).default_bounds(TIMES)


@pytest.mark.parametrize("cls", [shapes.DampedOscillation, shapes.Gabor])
def test_frequency_guess_falls_back_to_the_middle_of_the_range(cls: Any) -> None:
    shape = cls(freq_min=10.0, freq_max=100.0)
    lower, upper = shape.default_bounds(TIMES)
    # no periodogram peak (a flat waveform) or a waveform with a NaN
    flat = shape.initial_guess(TIMES, np.zeros(TIMES.size), 100)
    assert_equal(flat[2], 0.5 * (lower[2] + upper[2]))
    holed = np.sin(2 * np.pi * 30.0 * TIMES)
    holed[5] = np.nan
    assert_equal(shape.initial_guess(TIMES, holed, 100)[2], 0.5 * (lower[2] + upper[2]))
    # too few samples for a periodogram
    short = cls(freq_min=10.0, freq_max=1000.0, min_fwhm=1e-4, max_fwhm=1.0)
    times = np.array([0.0, 0.001, 0.002])
    lower, upper = short.default_bounds(times)
    guess = short.initial_guess(times, np.array([0.0, 1.0, 0.0]), 1)
    assert_equal(guess[2], 0.5 * (lower[2] + upper[2]))
    # a clean oscillation: its frequency
    clean = shape.initial_guess(TIMES, np.sin(2 * np.pi * 30.0 * TIMES), 100)
    assert_allclose(clean[2], 30.0, atol=1.0)


def test_template_checks() -> None:
    t = np.arange(10) / 100.0
    v = np.sin(np.pi * t / 0.09)
    with pytest.raises(DataError, match="same length"):
        shapes.Template(t, v[:-1])
    with pytest.raises(DataError, match="must be finite"):
        shapes.Template(t, np.where(t > t[5], np.nan, v))
    with pytest.raises(DataError, match="strictly increasing"):
        shapes.Template(t[::-1], v)
    with pytest.raises(DataError, match="all zero"):
        shapes.Template(t, np.zeros(10))
    with pytest.raises(SettingsError, match="max_shift must be positive"):
        shapes.Template(t, v, max_shift=0.0)
    with pytest.raises(SettingsError, match="0 < stretch_min < 1 < stretch_max"):
        shapes.Template(t, v, stretch_min=1.2)


###################################################################################################
## backgrounds


def test_linear_backgrounds_have_no_nonlinear_parameters() -> None:
    for background in (bg.NoBackground(), bg.ConstantBackground(), bg.LinearBackground()):
        assert background._nonlinear(None).shape == (0,)
        jac = background.basis_jacobian(TIMES)
        assert jac.shape == (background.n_linear, TIMES.size, 0)


def test_background_settings_are_checked() -> None:
    short = np.arange(5) / 1000.0  # 4 ms: tau2 >= 3 tau1 does not fit
    with pytest.raises(SettingsError, match="too short for tau2 >= 3 tau1"):
        bg.DoubleExponentialBackground(tau_min=0.002).default_bounds(short)
    with pytest.raises(SettingsError, match="max_fwhm must be positive"):
        bg.PSplineBackground((0.0, 0.3), max_fwhm=0.0)
    with pytest.raises(DataError, match="finite and basis_times strictly increasing"):
        bg.ArtifactBasisBackground(np.array([[0.0, np.inf, 1.0]]), np.array([0.0, 0.1, 0.2]))


def test_pspline_select_lambda_from_given_candidates() -> None:
    times = np.arange(-0.1, 0.0, 1e-3)
    values = np.random.default_rng(1).normal(0.0, 1.0, times.size)
    spline = bg.PSplineBackground((times[0], times[-1]), max_fwhm=0.02)
    assert_equal(spline.select_lambda(times, values, lambdas=[0.5]), 0.5)
    candidates = [1e-6, 1.0, 1e6]
    assert spline.select_lambda(times, values, lambdas=candidates) in candidates


###################################################################################################
## registry: plugins, settings, checks of register_shape


class PluginBackground(bg.ConstantBackground):
    kind: ClassVar[str] = "background"


class BareBackground:
    """Satisfies the Background protocol only: the fit cannot use it."""

    name = "bare"
    kind = "background"
    param_names = ("c0", "tau")
    n_params = 2
    n_nonlinear = 1

    def evaluate(self, times: Any, _params: Any) -> Any:
        return np.zeros(np.shape(times))

    def jacobian(self, times: Any, _params: Any) -> Any:
        return np.zeros((*np.shape(times), 2))

    def default_bounds(self, _times: Any, _data_range: Any = None) -> Any:
        return np.zeros(2), np.ones(2)

    def initial_guess(self, _times: Any, _waveform: Any) -> Any:
        return np.zeros(2)


def test_plugin_shapes_are_checked() -> None:
    origin = "pkg 1.0"
    assert registry._plugin_shape("constant", PluginBackground, origin) is _plugins.SKIP
    assert registry._plugin_shape("lorentzian", Lorentzian, origin) is Lorentzian
    instance = Lorentzian()
    assert registry._plugin_shape("lorentzian", instance, origin) is instance

    class NeedsArguments(Lorentzian):
        def __init__(self, scale: Any) -> None:
            self.scale = scale

    with pytest.raises(SettingsError, match="must build without arguments"):
        registry._plugin_shape("lorentzian", NeedsArguments, origin)
    with pytest.raises(SettingsError, match="does not satisfy ComponentShape"):
        registry._plugin_shape("thing", object(), origin)
    with pytest.raises(SettingsError, match="the entry-point name must be the same"):
        registry._plugin_shape("other", Lorentzian, origin)

    class Halved(Lorentzian):
        def evaluate(self, times: Any, params: Any) -> Any:
            return 0.5 * super().evaluate(times, params)

    with pytest.raises(SettingsError, match="plugin shape 'lorentzian' of pkg 1.0: .*unit_amp"):
        registry._plugin_shape("lorentzian", Halved, origin)


def test_plugin_backgrounds_are_checked() -> None:
    origin = "pkg 1.0"
    assert registry._plugin_background("lorentzian", Lorentzian, origin) is _plugins.SKIP
    assert registry._plugin_background("constant", PluginBackground, origin) is PluginBackground
    with pytest.raises(SettingsError, match="does not satisfy Background"):
        registry._plugin_background("other", PluginBackground, origin)
    with pytest.raises(SettingsError, match="plugin background 'bare' of pkg 1.0: .*lacks"):
        registry._plugin_background("bare", BareBackground, origin)


def test_check_background_names_what_the_fit_needs() -> None:
    registry.check_background(bg.ExponentialBackground())
    with pytest.raises(
        SettingsError, match="lacks linear_names, nonlinear_names, n_linear, basis, basis_jacobian"
    ):
        registry.check_background(BareBackground())


@pytest.mark.usefixtures("clean_registry")
def test_settings_of_a_shape_registered_as_an_instance() -> None:
    @dataclasses.dataclass(frozen=True)
    class Wide(shapes.Gaussian):
        name: ClassVar[str] = "wide_gaussian"

    entry = Wide(max_fwhm=0.1)
    shapes.register_shape(entry)
    built = shapes.get_shape("wide_gaussian", min_fwhm=0.01)
    assert isinstance(built, Wide)
    assert built == dataclasses.replace(entry, min_fwhm=0.01)
    assert shapes.get_shape("wide_gaussian") is entry
    instance = Lorentzian()
    instance.name = "lorentzian_instance"
    shapes.register_shape(instance)
    with pytest.raises(SettingsError, match="registered as an instance and takes no settings"):
        shapes.get_shape("lorentzian_instance", width=1.0)
    assert "wide_gaussian" in shapes.list_shapes()
    assert shapes.list_shapes() == sorted(shapes.SHAPES)
    assert shapes.list_backgrounds() == sorted(shapes.BACKGROUNDS)
    assert {"constant", "pspline"} <= set(shapes.list_backgrounds())


def test_unregister_an_unknown_shape() -> None:
    with pytest.raises(SettingsError, match="no shape named 'triangle'"):
        shapes.unregister_shape("triangle")


def _faulty(**members: Any) -> Any:
    """Return a Lorentzian with some members replaced."""
    return type("Faulty", (Lorentzian,), members)()


def _raise(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("user code failed")


def _not_finite(self: Any, times: Any, params: Any) -> Any:
    return np.where(
        np.asarray(times) > TIMES[-1] / 2, np.inf, Lorentzian.evaluate(self, times, params)
    )


def _batch_differs(self: Any, times: Any, params: Any) -> Any:
    return Lorentzian.evaluate(self, times, params) * (1.0 if np.ndim(params) > 1 else 0.99)


def _flat_jacobian(_self: Any, times: Any, params: Any) -> Any:
    return np.zeros((np.shape(params)[0], np.size(times)))


def _outside_guess(*_args: Any) -> Any:
    return np.array([10.0, 0.01])


def _no_features(*_args: Any, **_kwargs: Any) -> Any:
    return object()


@pytest.mark.parametrize(
    ("members", "check", "detail"),
    [
        ({"name": ""}, "protocol", "'name' must be a non-empty string"),
        ({"param_names": ("center", "center")}, "protocol", "distinct strings"),
        ({"n_params": 3}, "protocol", "'n_params' must equal"),
        ({"latency_param": "peak"}, "protocol", "'latency_param' must be one of"),
        ({"default_bounds": _raise}, "bounds", "default_bounds raised RuntimeError"),
        ({"evaluate": _raise}, "evaluate", "evaluate or jacobian raised"),
        ({"jacobian": _flat_jacobian}, "evaluate", "jacobian must have shape"),
        ({"evaluate": _not_finite}, "evaluate", "non-finite values"),
        ({"evaluate": _batch_differs}, "evaluate", "one parameter set and a batch give"),
        ({"initial_guess": _raise}, "initial_guess", "initial_guess raised"),
        ({"initial_guess": _outside_guess}, "initial_guess", "inside the bounds"),
        ({"features": _raise}, "features", "features raised"),
        ({"features": _no_features}, "features", "features lacks"),
    ],
)
def test_check_shape_names_the_failing_check(members: Any, check: str, detail: str) -> None:
    with pytest.raises(SettingsError, match=f"check '{check}' failed: .*{detail}"):
        registry.check_shape(_faulty(**members))


def test_check_shape_rejects_a_shape_whose_parameters_have_no_effect() -> None:
    class Fixed(Lorentzian):
        name = "fixed"

        def evaluate(self, times: Any, params: Any) -> Any:
            params = np.asarray(params, dtype=float)
            z = (np.asarray(times) - 0.1) / 0.01
            return np.broadcast_to(1.0 / (1.0 + z**2), (*params.shape[:-1], np.size(times)))

        def jacobian(self, times: Any, params: Any) -> Any:
            params = np.asarray(params, dtype=float)
            return np.zeros((*params.shape[:-1], np.size(times), 2))

    with pytest.raises(SettingsError, match="check 'jacobian' failed: no parameter"):
        registry.check_shape(Fixed())


def test_jacobian_error_leaves_out_a_column_that_is_not_smooth() -> None:
    class Rough(Lorentzian):
        """The width also enters through a fast oscillation: finite differences do not settle."""

        name = "rough"

        def evaluate(self, times: Any, params: Any) -> Any:
            params = np.asarray(params, dtype=float)
            base = super().evaluate(times, params)
            return base * (1.0 + 0.1 * np.sin(3e8 * params[..., 1, None]))

    error = registry.jacobian_error(Rough(), registry.CHECK_TIMES, np.array([[0.1, 0.01]]))
    assert np.isnan(error[0, 1])  # left out: not smooth
    # the centre column is measured: its analytic derivative lacks the oscillating factor
    assert np.isfinite(error[0, 0])
    assert_allclose(error[0, 0], 0.0, atol=0.1)
