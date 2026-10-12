# SPDX-License-Identifier: Apache-2.0
"""Registries of shapes and backgrounds, and the checks run on user shapes (spec §5.5.1).

``SHAPES`` and ``BACKGROUNDS`` map a name to a class (or, for a user shape registered as an
instance, to that instance); they add the shapes and backgrounds of the entry-point group
``evoparam.shapes`` on first access (plugins, spec §10.8). :func:`register_shape` adds a user shape
after checking the protocol, the unit amplitude and the Jacobian numerically; a failed check raises
:class:`SettingsError` naming the check.
"""

import dataclasses
from typing import TYPE_CHECKING, NoReturn, TypeVar, cast

import numpy as np

from evoparam import _plugins
from evoparam._typing import Any, ArrayLike, FloatArray, RandomState
from evoparam.errors import SettingsError
from evoparam.shapes._base import FEATURE_OVERSAMPLING, Background, ComponentShape
from evoparam.shapes.backgrounds import (
    ArtifactBasisBackground,
    ConstantBackground,
    DoubleExponentialBackground,
    ExponentialBackground,
    LinearBackground,
    NoBackground,
    PSplineBackground,
)
from evoparam.shapes.components import (
    Biexponential,
    DampedOscillation,
    ExGaussian,
    Gabor,
    Gamma,
    Gaussian,
    SkewedGaussian,
    Template,
)

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

__all__ = [
    "SHAPES",
    "BACKGROUNDS",
    "BUILTIN_SHAPES",
    "list_shapes",
    "list_backgrounds",
    "register_shape",
    "unregister_shape",
    "check_shape",
    "get_shape",
    "get_background",
    "CHECK_TIMES",
]


_S = TypeVar("_S")


def _plugin_shape(name: str, obj: Any, origin: str) -> Any:
    """Check a shape of the entry-point group ``evoparam.shapes`` (spec §10.8)."""
    if getattr(obj, "kind", None) == "background":
        return _plugins.SKIP
    instance = obj
    if isinstance(obj, type):
        try:
            instance = obj()
        except Exception as err:
            raise SettingsError(
                f"plugin shape {name!r} of {origin}: the class must build "
                f"without arguments ({err!r})."
            ) from err
    if not isinstance(instance, ComponentShape):
        raise SettingsError(f"plugin shape {name!r} of {origin} does not satisfy ComponentShape.")
    if instance.name != name:
        raise SettingsError(
            f"plugin shape {name!r} of {origin}: its name attribute is "
            f"{instance.name!r}; the entry-point name must be the same."
        )
    try:
        check_shape(instance)
    except SettingsError as err:
        raise SettingsError(f"plugin shape {name!r} of {origin}: {err}") from err
    return obj


def _plugin_background(name: str, obj: Any, origin: str) -> Any:
    if getattr(obj, "kind", None) != "background":
        return _plugins.SKIP
    instance = obj() if isinstance(obj, type) else obj
    if not isinstance(instance, Background) or instance.name != name:
        raise SettingsError(
            f"plugin background {name!r} of {origin} does not satisfy "
            f"Background (or its name attribute differs)."
        )
    return obj


SHAPES = _plugins.LazyRegistry(
    "evoparam.shapes",
    {
        cls.name: cls
        for cls in (
            Gaussian,
            SkewedGaussian,
            ExGaussian,
            Gamma,
            Biexponential,
            DampedOscillation,
            Gabor,
            Template,
        )
    },
    _plugin_shape,
)
BUILTIN_SHAPES = SHAPES.builtin

BACKGROUNDS = _plugins.LazyRegistry(
    "evoparam.shapes",
    {
        cls.name: cls
        for cls in (
            NoBackground,
            ConstantBackground,
            LinearBackground,
            ExponentialBackground,
            DoubleExponentialBackground,
            PSplineBackground,
            ArtifactBasisBackground,
        )
    },
    _plugin_background,
)

# Reference axis of the registration checks: 300 ms at 1 kHz.
CHECK_TIMES = np.arange(-0.05, 0.25, 1e-3, dtype=np.float64)

FEATURE_FIELDS = (
    "latency",
    "amplitude",
    "polarity",
    "fwhm",
    "onset",
    "offset",
    "area",
    "rise_time",
    "decay_time",
    "symmetry",
    "sharpness",
)

METHODS = ("evaluate", "jacobian", "default_bounds", "initial_guess", "features")


def _instance(entry: Any, settings: dict[str, Any]) -> Any:
    if isinstance(entry, type):
        return entry(**settings)
    if settings:
        if not dataclasses.is_dataclass(entry):
            raise SettingsError(
                f"shape '{entry.name}' was registered as an instance and takes no settings."
            )
        return dataclasses.replace(cast("DataclassInstance", entry), **settings)
    return entry


def get_shape(name: str, **settings: Any) -> ComponentShape:
    """Return the shape registered under ``name``, built with ``settings``."""
    if name not in SHAPES:
        raise SettingsError(f"unknown shape '{name}'; registered: {sorted(SHAPES)}.")
    return cast("ComponentShape", _instance(SHAPES[name], settings))


def list_shapes() -> list[str]:
    """Names of the registered component shapes (built-in and plugins), sorted."""
    return sorted(SHAPES)


def list_backgrounds() -> list[str]:
    """Names of the registered backgrounds (built-in and plugins), sorted."""
    return sorted(BACKGROUNDS)


def get_background(name: str, **settings: Any) -> Background:
    """Return the background registered under ``name``, built with ``settings``."""
    if name not in BACKGROUNDS:
        raise SettingsError(f"unknown background '{name}'; registered: {sorted(BACKGROUNDS)}.")
    return cast("Background", _instance(BACKGROUNDS[name], settings))


###################################################################################################
## Checks


def _fail(name: object, check: str, detail: str) -> NoReturn:
    raise SettingsError(f"register_shape('{name}'): check '{check}' failed: {detail}")


def _finite_difference(
    shape: ComponentShape, times: FloatArray, params: FloatArray, index: int, step: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Central difference with Richardson extrapolation, and the two raw estimates."""

    def central(h: float) -> FloatArray:
        delta = np.zeros(params.shape[-1])
        delta[index] = h
        return (shape.evaluate(times, params + delta) - shape.evaluate(times, params - delta)) / (
            2.0 * h
        )

    coarse, fine = central(step), central(step / 2.0)
    return (4.0 * fine - coarse) / 3.0, coarse, fine


def jacobian_error(
    shape: ComponentShape,
    times: ArrayLike,
    params: ArrayLike,
    rel_step: float = 1e-5,
    bounds: tuple[FloatArray, FloatArray] | None = None,
) -> FloatArray:
    """Largest error of the analytic Jacobian against central finite differences.

    For each parameter set and each parameter, the error is
    ``max_t |J - FD| / max_t max(|J|, |FD|)``: relative to the largest entry of
    that Jacobian column. The step is ``rel_step · max(|p|, width/1000)``, with
    ``width`` the width of the bounds (1 without bounds), so that scale
    parameters close to zero get a step in proportion. Samples at which the finite
    difference is not smooth (``|FD(h) - FD(h/2)|`` large: the onset of a causal
    shape, a template edge) are left out, as are columns with no effect on the
    axis.

    Returns
    -------
    error : 2d array, shape (n_sets, n_params)
        NaN where a column was left out.
    """
    times = np.asarray(times, dtype=np.float64)
    params = np.atleast_2d(np.asarray(params, dtype=np.float64))
    analytic = shape.jacobian(times, params)
    n_sets, n_params = params.shape
    error = np.full((n_sets, n_params), np.nan)
    for index in range(n_params):
        width = 1.0 if bounds is None else float(bounds[1][index] - bounds[0][index])
        for row in range(n_sets):
            step = rel_step * max(abs(params[row, index]), 1e-3 * width)
            numeric, coarse, fine = _finite_difference(shape, times, params[row], index, step)
            column = analytic[row, :, index]
            scale = max(np.max(np.abs(numeric)), np.max(np.abs(column)))
            if not scale * width > 1e-9:
                continue
            smooth = np.abs(coarse - fine) <= 1e-3 * scale
            if np.mean(smooth) < 0.5:
                continue
            error[row, index] = np.max(np.abs(column - numeric)[smooth]) / scale
    return error


def check_shape(
    shape: ComponentShape,
    times: ArrayLike | None = None,
    *,
    n_sets: int = 20,
    rtol: float = 1e-6,
    random_state: RandomState = 0,
) -> None:
    """Run the registration checks on a shape instance.

    Checks, in order: ``protocol`` (attributes and methods of
    :class:`ComponentShape`), ``bounds`` (finite, lower < upper), ``evaluate``
    (output shapes, finite values, one set equals a batch), ``unit_amplitude``
    (``max |s| <= 1`` everywhere and ``≈ 1`` for most sets, on a grid ten times
    finer than the axis), ``jacobian`` (analytic against finite differences,
    ``rtol`` relative to the column maximum), ``initial_guess`` (inside the
    bounds) and ``features`` (the fields of :class:`ComponentFeatures`).

    Raises
    ------
    SettingsError
        Naming the first failing check.
    """
    name = getattr(shape, "name", None)
    if not isinstance(name, str) or not name:
        _fail(name, "protocol", "'name' must be a non-empty string.")
    names = getattr(shape, "param_names", None)
    if (
        not isinstance(names, (tuple, list))
        or not names
        or not all(isinstance(item, str) for item in names)
        or len(set(names)) != len(names)
    ):
        _fail(name, "protocol", "'param_names' must be a non-empty tuple of distinct strings.")
    if getattr(shape, "n_params", None) != len(names):
        _fail(name, "protocol", "'n_params' must equal len(param_names).")
    if getattr(shape, "latency_param", None) not in names:
        _fail(name, "protocol", "'latency_param' must be one of param_names.")
    for method in METHODS:
        if not callable(getattr(shape, method, None)):
            _fail(name, "protocol", f"method '{method}' is missing.")

    times = CHECK_TIMES if times is None else np.asarray(times, dtype=np.float64)
    n = len(names)
    try:
        lower, upper = (np.asarray(b, dtype=np.float64) for b in shape.default_bounds(times))
    except Exception as err:  # noqa: BLE001  (report any failure of user code as a failed check)
        _fail(name, "bounds", f"default_bounds raised {err!r}.")
    if (
        lower.shape != (n,)
        or upper.shape != (n,)
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or not np.all(lower < upper)
    ):
        _fail(
            name,
            "bounds",
            "default_bounds must return finite (lower, upper) arrays of "
            f"shape ({n},) with lower < upper.",
        )

    rng = np.random.default_rng(random_state)
    params = lower + (upper - lower) * rng.random((n_sets, n))
    try:
        values = np.asarray(shape.evaluate(times, params))
        single = np.asarray(shape.evaluate(times, params[0]))
        jac = np.asarray(shape.jacobian(times, params))
    except Exception as err:  # noqa: BLE001  (user code: any failure becomes a failed check (SettingsError))
        _fail(name, "evaluate", f"evaluate or jacobian raised {err!r}.")
    if values.shape != (n_sets, times.size) or single.shape != (times.size,):
        _fail(
            name,
            "evaluate",
            f"evaluate must map (..., {n}) to (..., n_times); got "
            f"{values.shape} for {params.shape}.",
        )
    if jac.shape != (n_sets, times.size, n):
        _fail(
            name,
            "evaluate",
            f"jacobian must have shape (..., n_times, {n}); got {jac.shape} for {params.shape}.",
        )
    if not (np.all(np.isfinite(values)) and np.all(np.isfinite(jac))):
        _fail(name, "evaluate", "evaluate or jacobian returned non-finite values.")
    if not np.allclose(single, values[0], rtol=1e-12, atol=1e-14):
        _fail(name, "evaluate", "one parameter set and a batch give different values.")

    step = float(np.median(np.diff(times))) / FEATURE_OVERSAMPLING
    span = times[-1] - times[0]
    fine = np.arange(times[0] - span, times[-1] + span, step)
    peaks = np.max(np.abs(shape.evaluate(fine, params)), axis=-1)
    if np.any(peaks > 1.0 + 1e-6) or np.median(peaks) < 1.0 - 5e-3:
        _fail(
            name,
            "unit_amplitude",
            f"max |s| must be 1; found values from {peaks.min():.6g} to {peaks.max():.6g}.",
        )

    error = jacobian_error(shape, times, params, bounds=(lower, upper))
    if np.all(np.isnan(error)):
        _fail(name, "jacobian", "no parameter has a measurable effect on the check axis.")
    worst = np.nanmax(error)
    if worst > rtol:
        row, col = np.unravel_index(np.nanargmax(error), error.shape)
        _fail(
            name,
            "jacobian",
            f"analytic derivative of '{names[col]}' differs from finite "
            f"differences by {worst:.3g} (relative), above {rtol:g}, at "
            f"params {params[row].tolist()}.",
        )

    middle = 0.5 * (lower + upper)
    waveform = shape.evaluate(times, middle)
    peak_index = int(np.argmax(np.abs(waveform)))
    try:
        guess = np.asarray(shape.initial_guess(times, waveform, peak_index), dtype=np.float64)
    except Exception as err:  # noqa: BLE001  (user code: any failure becomes a failed check (SettingsError))
        _fail(name, "initial_guess", f"initial_guess raised {err!r}.")
    if (
        guess.shape != (n,)
        or not np.all(np.isfinite(guess))
        or np.any(guess < lower)
        or np.any(guess > upper)
    ):
        _fail(
            name, "initial_guess", "initial_guess must return finite parameters inside the bounds."
        )

    try:
        features = shape.features(times, middle)
    except Exception as err:  # noqa: BLE001  (user code: any failure becomes a failed check (SettingsError))
        _fail(name, "features", f"features raised {err!r}.")
    missing = [item for item in FEATURE_FIELDS if not hasattr(features, item)]
    if missing:
        _fail(name, "features", f"features lacks {missing}.")


def register_shape(
    shape: _S, *, overwrite: bool = False, times: ArrayLike | None = None, **check_settings: Any
) -> _S:
    """Register a user shape after checking it (REQ-102).

    Parameters
    ----------
    shape : class or instance
        A class implementing :class:`ComponentShape` that can be built without
        arguments, or an instance. Usable as a class decorator.
    overwrite : bool
        Replace a user shape already registered under the same name. Built-in
        shapes are never replaced.
    times : 1d array, optional
        Axis of the checks (default ``CHECK_TIMES``, 300 ms at 1 kHz).
    **check_settings
        ``n_sets``, ``rtol``, ``random_state`` passed to :func:`check_shape`.

    Returns
    -------
    shape
        The argument, unchanged.

    Raises
    ------
    SettingsError
        When a check fails; the message names the check.
    """
    if isinstance(shape, type):
        try:
            instance = shape()
        except Exception as err:
            raise SettingsError(
                f"register_shape({shape.__name__}): check 'instantiate' "
                f"failed: the class must build without arguments ({err!r})."
            ) from err
    else:
        instance = shape
    check_shape(instance, times, **check_settings)
    name = instance.name
    if name in BUILTIN_SHAPES:
        _fail(name, "name", "a built-in shape has this name.")
    if name in SHAPES and not overwrite:
        _fail(name, "name", "a shape with this name is registered; pass overwrite=True.")
    SHAPES[name] = shape
    return shape


def unregister_shape(name: str) -> None:
    """Remove a user shape from ``SHAPES``; built-in shapes cannot be removed."""
    if name in BUILTIN_SHAPES:
        raise SettingsError(f"unregister_shape: '{name}' is a built-in shape.")
    if name not in SHAPES:
        raise SettingsError(f"unregister_shape: no shape named '{name}'.")
    del SHAPES[name]
