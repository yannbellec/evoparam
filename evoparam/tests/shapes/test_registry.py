# SPDX-License-Identifier: Apache-2.0
"""REQ-102: the shape registry and the checks of ``register_shape``."""

import numpy as np
import pytest

from evoparam import shapes
from evoparam.errors import SettingsError
from evoparam.shapes import compute_features
from evoparam.shapes.registry import BUILTIN_SHAPES, check_shape
from evoparam.tests.shapes.helpers import TIMES, make_template


class Lorentzian:
    """User shape written against the protocol only: ``s = 1/(1 + ((t - c)/w)²)``."""

    name = "lorentzian"
    param_names = ("center", "width")
    n_params = 2
    latency_param = "center"

    def evaluate(self, times, params):
        params = np.asarray(params, dtype=float)
        z = (np.asarray(times) - params[..., 0, None]) / params[..., 1, None]
        return 1.0 / (1.0 + z ** 2)

    def jacobian(self, times, params):
        params = np.asarray(params, dtype=float)
        width = params[..., 1, None]
        z = (np.asarray(times) - params[..., 0, None]) / width
        s = 1.0 / (1.0 + z ** 2)
        return np.stack([2 * z * s ** 2 / width, 2 * z ** 2 * s ** 2 / width], axis=-1)

    def default_bounds(self, times, data_range=None):
        return np.array([times[0], 0.002]), np.array([times[-1], 0.05])

    def initial_guess(self, times, waveform, peak_index):
        return np.array([times[peak_index], 0.01])

    def features(self, times, params, amplitude=1.0):
        return compute_features(self, times, params, amplitude)


class WrongJacobian(Lorentzian):
    name = "lorentzian_wrong_jacobian"

    def jacobian(self, times, params):
        jac = super().jacobian(times, params)
        jac[..., 1] *= 0.5    # forgotten factor 2
        return jac


class NotNormalised(Lorentzian):
    name = "lorentzian_half"

    def evaluate(self, times, params):
        return 0.5 * super().evaluate(times, params)

    def jacobian(self, times, params):
        return 0.5 * super().jacobian(times, params)


class MissingMethod(Lorentzian):
    name = "lorentzian_no_features"
    features = None


class NeedsArguments(Lorentzian):
    name = "lorentzian_args"

    def __init__(self, scale):
        self.scale = scale


@pytest.fixture
def clean_registry():
    before = dict(shapes.SHAPES)
    yield
    shapes.SHAPES.clear()
    shapes.SHAPES.update(before)


def test_get_shape_and_background():
    shape = shapes.get_shape("gaussian", min_fwhm=0.004)
    assert isinstance(shape, shapes.Gaussian) and shape.min_fwhm == 0.004
    assert isinstance(shapes.get_background("exponential"), shapes.ExponentialBackground)
    with pytest.raises(SettingsError, match="unknown shape"):
        shapes.get_shape("triangle")
    with pytest.raises(SettingsError, match="unknown background"):
        shapes.get_background("polynomial")


@pytest.mark.parametrize("shape", [shapes.Gaussian(), shapes.SkewedGaussian(), shapes.ExGaussian(),
                                   shapes.Gamma(), shapes.Biexponential(),
                                   shapes.DampedOscillation(), shapes.Gabor(), make_template(),
                                   make_template(stretch=True)], ids=lambda s: s.name)
def test_builtin_shapes_pass_the_checks(shape):
    check_shape(shape)
    check_shape(shape, TIMES)


def test_register_a_user_shape(clean_registry):
    assert shapes.register_shape(Lorentzian) is Lorentzian
    assert isinstance(shapes.get_shape("lorentzian"), Lorentzian)
    instance = Lorentzian()
    instance.name = "lorentzian_instance"
    shapes.register_shape(instance)
    assert shapes.get_shape("lorentzian_instance") is instance
    with pytest.raises(SettingsError, match="check 'name'"):
        shapes.register_shape(Lorentzian)
    shapes.register_shape(Lorentzian, overwrite=True)
    shapes.unregister_shape("lorentzian")
    assert "lorentzian" not in shapes.SHAPES


def test_register_as_decorator(clean_registry):
    @shapes.register_shape
    class Decorated(Lorentzian):
        name = "decorated"

    assert shapes.SHAPES["decorated"] is Decorated


@pytest.mark.parametrize("cls, check", [
    (WrongJacobian, "jacobian"),
    (NotNormalised, "unit_amplitude"),
    (MissingMethod, "protocol"),
    (NeedsArguments, "instantiate"),
])
def test_register_rejects_a_faulty_shape(clean_registry, cls, check):
    with pytest.raises(SettingsError, match=f"check '{check}' failed"):
        shapes.register_shape(cls)
    assert cls.name not in shapes.SHAPES


def test_builtin_names_are_protected(clean_registry):
    class Impostor(Lorentzian):
        name = "gaussian"

    with pytest.raises(SettingsError, match="check 'name'"):
        shapes.register_shape(Impostor, overwrite=True)
    with pytest.raises(SettingsError):
        shapes.unregister_shape("gaussian")
    assert shapes.SHAPES["gaussian"] is shapes.Gaussian
    assert BUILTIN_SHAPES == frozenset(shapes.SHAPES) - {
        name for name in shapes.SHAPES if name not in BUILTIN_SHAPES}


def test_wrong_bounds_and_output_shapes_are_rejected():
    class BadBounds(Lorentzian):
        name = "bad_bounds"

        def default_bounds(self, times, data_range=None):
            return np.array([0.1, 0.05]), np.array([0.0, 0.01])

    class BadShape(Lorentzian):
        name = "bad_shape"

        def evaluate(self, times, params):
            return super().evaluate(times, params).T

    with pytest.raises(SettingsError, match="check 'bounds'"):
        check_shape(BadBounds())
    with pytest.raises(SettingsError, match="check 'evaluate'"):
        check_shape(BadShape())
