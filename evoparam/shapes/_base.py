# SPDX-License-Identifier: Apache-2.0
"""Protocols, the feature record and the helpers shared by shapes and backgrounds.

A component shape is a unit-amplitude function ``s(t; θ)`` with ``max_t |s| = 1``;
its signed amplitude is a separate linear parameter. A background is a sum of
basis functions with linear coefficients, some of which depend on non-linear
parameters (time constants). Both evaluate on a 1-D time axis (seconds) for a
parameter array of shape ``(..., n_params)`` and return ``(..., n_times)``;
their Jacobians have shape ``(..., n_times, n_params)``.
"""

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from evoparam._typing import ArrayLike, FloatArray
from evoparam.errors import DataError, SettingsError

__all__ = [
    "ComponentShape",
    "Background",
    "ComponentFeatures",
    "compute_features",
    "ShapeBase",
    "FWHM_PER_SIGMA",
    "ONSET_FRACTION",
]

# Full width at half maximum of a Gaussian in units of its standard deviation.
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))

# Default fraction of the peak that defines onset and offset (fractional-peak rule).
ONSET_FRACTION = 0.10

# Features are computed on a grid this many times finer than 1/sfreq.
FEATURE_OVERSAMPLING = 10


###################################################################################################
## Protocols


@runtime_checkable
class ComponentShape(Protocol):
    """Protocol of a component shape (spec §4.3.4, §5.5.1)."""

    # read-only members: a shape may give them as class variables or properties
    @property
    def name(self) -> str:
        """Registry name of the shape."""

    @property
    def param_names(self) -> tuple[str, ...]:
        """Names of the non-linear parameters, in order."""

    @property
    def n_params(self) -> int:
        """Number of non-linear parameters."""

    @property
    def latency_param(self) -> str:
        """Name of the parameter closest to the latency."""

    def evaluate(self, times: FloatArray, params: FloatArray) -> FloatArray:
        """Unit-amplitude shape, ``(..., n_params) -> (..., n_times)``."""

    def jacobian(self, times: FloatArray, params: FloatArray) -> FloatArray:
        """Analytic derivatives, ``(..., n_params) -> (..., n_times, n_params)``."""

    def default_bounds(
        self, times: FloatArray, data_range: Any = None
    ) -> tuple[FloatArray, FloatArray]:
        """Lower and upper bounds of the parameters for a window."""

    def initial_guess(
        self, times: FloatArray, waveform: FloatArray, peak_index: int
    ) -> FloatArray:
        """Return the starting parameters for a candidate extremum of ``waveform``."""

    def features(
        self, times: FloatArray, params: FloatArray, amplitude: float = 1.0
    ) -> "ComponentFeatures":
        """Features of the component curve (:class:`ComponentFeatures`)."""


@runtime_checkable
class Background(Protocol):
    """Protocol of a background (spec §4.3.4, §5.5.2)."""

    # read-only members: a background may give them as class variables or properties
    @property
    def name(self) -> str:
        """Registry name of the background."""

    @property
    def param_names(self) -> tuple[str, ...]:
        """Names of the parameters, in order."""

    @property
    def n_params(self) -> int:
        """Number of parameters."""

    def evaluate(self, times: FloatArray, params: FloatArray) -> FloatArray:
        """Background, ``(..., n_params) -> (..., n_times)``."""

    def jacobian(self, times: FloatArray, params: FloatArray) -> FloatArray:
        """Analytic derivatives, ``(..., n_params) -> (..., n_times, n_params)``."""

    def default_bounds(
        self, times: FloatArray, data_range: Any = None
    ) -> tuple[FloatArray, FloatArray]:
        """Lower and upper bounds of the parameters for a window."""

    def initial_guess(self, times: FloatArray, waveform: FloatArray) -> FloatArray:
        """Return the starting parameters for a waveform."""


###################################################################################################
## Helpers


def as_times(times: ArrayLike) -> FloatArray:
    """Return ``times`` as a finite 1-D float64 array."""
    times = np.asarray(times, dtype=np.float64)
    if times.ndim == 0:
        times = times.reshape(1)
    if times.ndim != 1:
        raise DataError(f"times: must be a 1-D array, got shape {times.shape}.")
    # a finite sum means finite entries (NaN and infinities propagate); the entries are
    # checked one by one only when the sum is not finite (an overflow of finite values)
    if not math.isfinite(times.sum()) and not np.isfinite(times).all():
        raise DataError("times: must be finite.")
    return times


def as_params(params: ArrayLike | None, n_params: int, name: str) -> FloatArray:
    """Return ``params`` as a float64 array whose last axis has ``n_params`` entries."""
    params = np.asarray(params, dtype=np.float64)
    if params.ndim == 0 or params.shape[-1] != n_params:
        raise DataError(f"{name}: params must have shape (..., {n_params}), got {params.shape}.")
    return params


# window_info of the last time axes, by content: a fit asks for the same window many times
# (bounds and starting values of every shape), and the median step is most of the cost
_WINDOW_CACHE: dict[bytes, tuple[float, float, float, float]] = {}
_WINDOW_CACHE_SIZE = 64


def window_info(times: ArrayLike) -> tuple[float, float, float, float]:
    """Window start, end, length and sampling rate of a strictly increasing time axis."""
    times = as_times(times)
    if times.size < 2:
        raise DataError("times: at least 2 samples are needed to define a window.")
    key = times.tobytes()
    known = _WINDOW_CACHE.get(key)
    if known is not None:
        return known
    steps = np.diff(times)
    if np.any(steps <= 0):
        raise DataError("times: must be strictly increasing.")
    out = (
        float(times[0]),
        float(times[-1]),
        float(times[-1] - times[0]),
        float(1.0 / np.median(steps)),
    )
    if len(_WINDOW_CACHE) >= _WINDOW_CACHE_SIZE:
        _WINDOW_CACHE.clear()
    _WINDOW_CACHE[key] = out
    return out


def check_waveform(
    times: ArrayLike, waveform: ArrayLike, peak_index: int | None = None
) -> tuple[FloatArray, FloatArray]:
    """Check that ``waveform`` matches ``times`` (and that ``peak_index`` is inside)."""
    times = as_times(times)
    waveform = np.asarray(waveform, dtype=np.float64)
    if waveform.shape != times.shape:
        raise DataError(f"waveform: shape {waveform.shape} differs from times {times.shape}.")
    if peak_index is not None and not 0 <= int(peak_index) < times.size:
        raise DataError(f"peak_index: {peak_index} is outside 0..{times.size - 1}.")
    return times, waveform


def peak_fwhm(times: FloatArray, waveform: FloatArray, peak_index: int) -> float | None:
    """Full width at half maximum of ``|waveform|`` around ``peak_index``, or None.

    The width is measured on each side where the half maximum is reached inside
    the window; when only one side is reached the width is twice that side.
    """
    values = np.abs(waveform)
    half = 0.5 * values[peak_index]
    if not half > 0:
        return None
    sides = []
    below = np.nonzero(values[: peak_index + 1] < half)[0]
    if below.size:
        j = below[-1]
        sides.append(times[peak_index] - _interp_cross(times, values, j, j + 1, half))
    below = np.nonzero(values[peak_index:] < half)[0]
    if below.size:
        j = peak_index + below[0]
        sides.append(_interp_cross(times, values, j - 1, j, half) - times[peak_index])
    if not sides:
        return None
    return 2.0 * float(np.mean(sides))


def clip_to_bounds(
    guess: ArrayLike | Sequence[Any], bounds: tuple[ArrayLike, ArrayLike]
) -> FloatArray:
    lower, upper = bounds
    return np.clip(np.asarray(guess, dtype=np.float64), lower, upper)


def check_bounds(
    name: str, lower: ArrayLike | Sequence[Any], upper: ArrayLike | Sequence[Any]
) -> tuple[FloatArray, FloatArray]:
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if not np.all(lower < upper):
        raise SettingsError(
            f"{name}: empty parameter bounds for this window "
            f"(lower {lower.tolist()}, upper {upper.tolist()})."
        )
    return lower, upper


def _interp_cross(grid: FloatArray, values: FloatArray, i: int, j: int, level: float) -> float:
    """Time at which ``values`` crosses ``level`` between samples ``i`` and ``j``.

    Both samples must be on the grid: a negative index would silently wrap to the
    other end, and one past the end does not exist.
    """
    if not (0 <= i < grid.size and 0 <= j < grid.size):
        raise IndexError(f"crossing between samples {i} and {j} of a grid of {grid.size}.")
    vi, vj = values[i], values[j]
    if vj == vi:
        return float(grid[i])
    return float(grid[i] + (level - vi) * (grid[j] - grid[i]) / (vj - vi))


def _half_height_edges(
    grid: FloatArray, lobe: FloatArray, peak_time: float, height: float
) -> tuple[float, float]:
    """Return the times at which ``lobe`` crosses ``height / 2`` before and after the extremum.

    ``lobe`` is the curve signed so that its extremum ``height`` at ``peak_time`` is positive.

    A side whose crossing is not on the grid is NaN; both are NaN when the extremum is
    off the grid, where the far side cannot be reached. When no grid sample next to the
    extremum reaches half the height (a lobe narrower than the grid step), each crossing
    is interpolated between the extremum itself and the grid sample next to it.
    """
    half = 0.5 * height
    if not grid[0] <= peak_time <= grid[-1]:
        return np.nan, np.nan
    k = int(np.clip(np.searchsorted(grid, peak_time), 1, grid.size - 1))
    k = k if lobe[k] >= lobe[k - 1] else k - 1
    if not lobe[k] >= half:
        i = min(int(np.searchsorted(grid, peak_time, side="right")) - 1, grid.size - 2)
        left = _interp_points(grid[i], lobe[i], peak_time, height, half)
        right = _interp_points(peak_time, height, grid[i + 1], lobe[i + 1], half)
        return left, right
    # lobe[k] >= half: the last sample below half before k and the first one after k
    # each have a neighbour on the lobe, inside the grid
    below = np.nonzero(lobe[:k] < half)[0]
    left = _interp_cross(grid, lobe, below[-1], below[-1] + 1, half) if below.size else np.nan
    below = np.nonzero(lobe[k + 1 :] < half)[0]
    right = (
        _interp_cross(grid, lobe, k + below[0], k + below[0] + 1, half) if below.size else np.nan
    )
    return left, right


def _interp_points(t0: float, v0: float, t1: float, v1: float, level: float) -> float:
    """Time at which the line through ``(t0, v0)`` and ``(t1, v1)`` reaches ``level``."""
    if v1 == v0:
        return float(t0)
    return float(t0 + (level - v0) * (t1 - t0) / (v1 - v0))


def _trapezoid(y: FloatArray, x: FloatArray) -> float:
    integrate = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined, unused-ignore]  # noqa: NPY201  (NumPy < 2.0 has no trapezoid)
    return float(integrate(y, x))


###################################################################################################
## Features


@dataclass(frozen=True)
class ComponentFeatures:
    """Features of one fitted component, measured on its curve (spec §5.5.1).

    Parameters
    ----------
    latency : float
        Reported latency (s), as defined per shape in spec §5.5.1: the time of
        the extremum for the non-oscillatory shapes, the onset ``t0`` for
        ``damped_oscillation`` and the envelope centre ``t0`` for ``gabor``.
    amplitude : float
        Signed value of the component curve at its extremum (data units).
    polarity : int
        Sign of ``amplitude``: +1 or -1.
    fwhm : float
        Full width at half the extremum (s), around ``peak_time``.
    onset, offset : float
        First and last time (s) at which ``|A s(t)| >= onset_fraction · |amplitude|``.
    area : float
        Signed trapezoidal integral of the curve between onset and offset (units·s).
    rise_time, decay_time : float
        ``peak_time - onset`` and ``offset - peak_time`` (s).
    symmetry : float
        ``(decay_time - rise_time) / (decay_time + rise_time)``; 0 for a symmetric shape.
    sharpness : float
        ``|s''(peak_time)| · fwhm²`` of the unit-amplitude shape (dimensionless).
    peak_time : float
        Time (s) of the largest ``|s|``; equal to ``latency`` except for the
        oscillatory shapes.
    peak_times : tuple of float
        All local extrema of the curve with ``|s| >= onset_fraction``.
    onset_fraction : float
        Fraction used for onset and offset.
    warnings : tuple of str
        For example ``"latency_outside_window"``, ``"onset_outside_grid"``.
    """

    latency: float
    amplitude: float
    polarity: int
    fwhm: float
    onset: float
    offset: float
    area: float
    rise_time: float
    decay_time: float
    symmetry: float
    sharpness: float
    peak_time: float
    peak_times: tuple[float, ...] = ()
    onset_fraction: float = ONSET_FRACTION
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the features as a plain dictionary."""
        out = asdict(self)
        out["peak_times"] = list(self.peak_times)
        out["warnings"] = list(self.warnings)
        return out


def feature_grid(
    times: ArrayLike, sfreq: float | None = None, window: Sequence[float] | None = None
) -> tuple[FloatArray, float, tuple[float, float]]:
    """Return a grid ten times finer than ``1/sfreq`` around the window.

    It covers the window and one window length on each side, so that a component cut by a window
    edge is still measured whole.
    """
    origin, step, n_points, bounds = _grid_spec(times, sfreq, window)
    return origin + step * np.arange(n_points, dtype=np.float64), step, bounds


def _grid_spec(
    times: ArrayLike, sfreq: float | None, window: Sequence[float] | None
) -> tuple[float, float, int, tuple[float, float]]:
    """``feature_grid`` without the grid: point ``i`` is ``origin + step * i``, ``i < n_points``."""
    times = as_times(times)
    if sfreq is None:
        sfreq = window_info(times)[3]
    if not (np.isfinite(sfreq) and sfreq > 0):
        raise SettingsError(f"sfreq: must be a positive number, got {sfreq}.")
    start, end = (float(times[0]), float(times[-1])) if window is None else map(float, window)
    if not end > start:
        raise SettingsError(f"window: end must be after start, got ({start}, {end}).")
    span = end - start
    step = 1.0 / (FEATURE_OVERSAMPLING * sfreq)
    n_points = int(np.ceil(3.0 * span / step)) + 1
    return start - span, step, n_points, (start, end)


def _grid_search(origin: float, step: float, n_points: int, value: float, right: bool) -> int:
    """``np.searchsorted(grid, value, side)`` on the grid of :func:`_grid_spec`, not built.

    The grid points are recomputed as NumPy computes them (``origin + step * i``); NaN is
    placed after every point, as NumPy sorts it.
    """
    if not math.isfinite(value):
        return 0 if value == -math.inf else n_points
    i = min(max(math.ceil((value - origin) / step), 0), n_points)

    def before(k: int) -> bool:  # grid[k] lies before the insertion point
        point = origin + step * float(k)
        return point <= value if right else point < value

    while i > 0 and not before(i - 1):
        i -= 1
    while i < n_points and before(i):
        i += 1
    return i


def _grid_extremum(grid: FloatArray, values: FloatArray) -> float:
    k = int(np.argmax(np.abs(values)))
    if 0 < k < grid.size - 1:
        num = values[k - 1] - values[k + 1]
        den = values[k - 1] - 2.0 * values[k] + values[k + 1]
        if den != 0:
            return float(grid[k] + 0.5 * (grid[1] - grid[0]) * np.clip(num / den, -1, 1))
    return float(grid[k])


def compute_features(
    shape: ComponentShape,
    times: ArrayLike,
    params: ArrayLike,
    amplitude: float = 1.0,
    *,
    onset_fraction: float = ONSET_FRACTION,
    sfreq: float | None = None,
    window: Sequence[float] | None = None,
) -> ComponentFeatures:
    """Features of the component curve ``amplitude · shape(t; params)`` (spec §5.5.1).

    Every feature is measured on the component curve evaluated on a grid ten
    times finer than ``1/sfreq``, never on data.

    Parameters
    ----------
    shape : ComponentShape
        The shape. When it has ``latency(params)`` and ``extremum(params)``
        methods, they give the reported latency and the peak time; otherwise
        both are the refined grid extremum.
    times : 1d array
        Time axis of the fit window (s).
    params : 1d array, shape (n_params,)
        Non-linear parameters of one component.
    amplitude : float
        Linear amplitude ``A`` (data units).
    onset_fraction : float
        Fraction of the peak defining onset and offset (default 0.10).
    sfreq : float, optional
        Sampling rate (Hz). Default: from ``times``.
    window : tuple of float, optional
        Fit window ``(start, end)`` (s). Default: the extent of ``times``.

    Returns
    -------
    ComponentFeatures
    """
    params = np.asarray(params, dtype=np.float64)
    if params.shape != (shape.n_params,):
        raise DataError(
            f"{shape.name}: features need params of shape ({shape.n_params},), got {params.shape}."
        )
    if not 0.0 < onset_fraction < 1.0:
        raise SettingsError(f"onset_fraction: must be in (0, 1), got {onset_fraction}.")
    amplitude = float(amplitude)
    origin, step, n_points, (start, end) = _grid_spec(times, sfreq, window)
    first, last = 0, n_points
    support = getattr(shape, "support", None)
    if support is not None:
        # Every feature reads the curve where it reaches a quarter of the onset level or of
        # half the extremum (the extremum is 1 for these shapes) and on the samples next to
        # them: the grid is cut to the interval outside which the curve is below that, two
        # samples of margin on each side (the same numbers as on the whole grid; only that
        # part of the grid is built)
        low, high = support(params, 0.25 * min(onset_fraction, 0.5))
        cut_first = max(_grid_search(origin, step, n_points, float(low), False) - 2, 0)
        cut_last = min(_grid_search(origin, step, n_points, float(high), True) + 2, n_points)
        if cut_last - cut_first >= 3:
            first, last = cut_first, cut_last
    grid = origin + step * np.arange(first, last, dtype=np.float64)
    curve = shape.evaluate(grid, params)
    warnings = []

    if hasattr(shape, "extremum"):
        peak_time = float(shape.extremum(params))
    else:
        peak_time = _grid_extremum(grid, curve)
    latency = float(shape.latency(params)) if hasattr(shape, "latency") else peak_time
    # Points of the second derivative at the peak; one-sided after the onset of a causal
    # shape.
    onset_param = getattr(shape, "onset_param", None)
    start_time = params[shape.param_names.index(onset_param)] if onset_param else -np.inf
    if peak_time - 2 * step < start_time:
        x_peak = peak_time + step * np.arange(3, dtype=np.float64)
    else:
        x_peak = peak_time + step * np.arange(-1, 2, dtype=np.float64)
    # A shape with ``support`` is evaluated point by point (the value at a time does not
    # depend on the other times asked for): the peak and the points next to it in one call,
    # and the area below reuses the values on the grid
    pointwise = support is not None
    if pointwise:
        at_peak = shape.evaluate(np.r_[peak_time, x_peak], params)
        peak_value, v = float(at_peak[0]), at_peak[1:]
    else:
        peak_value = float(shape.evaluate(np.array([peak_time]), params)[0])
    sign = 1.0 if peak_value >= 0 else -1.0
    height = abs(peak_value)

    # Full width at half the extremum, on the lobe that holds the extremum.
    left, right = _half_height_edges(grid, sign * curve, peak_time, height)
    fwhm = right - left
    if not np.isfinite(fwhm):
        warnings.append("fwhm_outside_grid")

    # Onset and offset: first and last time above the fraction of the peak.
    magnitude = np.abs(curve)
    level = onset_fraction * height
    above = np.nonzero(magnitude >= level)[0]
    onset = offset = np.nan
    if above.size:
        first, last = above[0], above[-1]
        if first > 0:
            onset = _interp_cross(grid, magnitude, first - 1, first, level)
        if last < grid.size - 1:
            offset = _interp_cross(grid, magnitude, last, last + 1, level)
    if not np.isfinite(onset):
        warnings.append("onset_outside_grid")
    if not np.isfinite(offset):
        warnings.append("offset_outside_grid")

    area = np.nan
    if np.isfinite(onset) and np.isfinite(offset):
        within = (grid > onset) & (grid < offset)
        x = np.concatenate([[onset], grid[within], [offset]])
        if pointwise:
            ends = shape.evaluate(np.array([onset, offset]), params)
            values = np.concatenate([ends[:1], curve[within], ends[1:]])
        else:
            values = shape.evaluate(x, params)
        area = amplitude * _trapezoid(values, x)

    rise_time = peak_time - onset
    decay_time = offset - peak_time
    total = rise_time + decay_time
    symmetry = (decay_time - rise_time) / total if total > 0 else np.nan

    # Second derivative at the peak.
    if not pointwise:
        v = shape.evaluate(x_peak, params)
    curvature = (v[0] - 2.0 * v[1] + v[2]) / step**2
    sharpness = abs(curvature) / height * fwhm**2 if height > 0 else np.nan

    # Local extrema of the curve above the onset fraction (oscillatory shapes have several).
    diff = np.diff(curve)
    turning = np.nonzero(diff[:-1] * diff[1:] < 0)[0] + 1
    turning = turning[magnitude[turning] >= level]
    peak_times = []
    for i in turning:
        den = curve[i - 1] - 2.0 * curve[i] + curve[i + 1]
        shift = 0.5 * (curve[i - 1] - curve[i + 1]) / den if den != 0 else 0.0
        peak_times.append(float(grid[i] + step * np.clip(shift, -0.5, 0.5)))
    if not peak_times:
        peak_times = [peak_time]

    if latency < start or latency > end:
        warnings.append("latency_outside_window")
    signed = amplitude * peak_value
    return ComponentFeatures(
        latency=latency,
        amplitude=signed,
        polarity=1 if signed >= 0 else -1,
        fwhm=float(fwhm),
        onset=float(onset),
        offset=float(offset),
        area=float(area),
        rise_time=float(rise_time),
        decay_time=float(decay_time),
        symmetry=float(symmetry),
        sharpness=float(sharpness),
        peak_time=peak_time,
        peak_times=tuple(peak_times),
        onset_fraction=float(onset_fraction),
        warnings=tuple(warnings),
    )


###################################################################################################
## Base class


class ShapeBase:
    """Common methods of the built-in shapes.

    Subclasses define ``name``, ``param_names``, ``latency_param``,
    ``onset_param`` (the onset parameter of a causal shape, else None),
    ``evaluate``, ``jacobian``, ``default_bounds``, ``initial_guess``,
    ``latency`` and ``extremum``.
    """

    name: ClassVar[str]
    onset_param: ClassVar[str | None] = None
    if TYPE_CHECKING:  # defined by each shape (param_names: a property of Template)
        latency_param: ClassVar[str]

        @property
        def param_names(self) -> tuple[str, ...]:
            """Names of the parameters, in order."""

        def evaluate(self, times: ArrayLike, params: ArrayLike) -> FloatArray:
            """Unit-amplitude shape, ``(..., n_params) -> (..., n_times)``."""
            ...

        def jacobian(self, times: ArrayLike, params: ArrayLike) -> FloatArray:
            """Analytic derivatives, ``(..., n_params) -> (..., n_times, n_params)``."""
            ...

        def default_bounds(
            self, times: ArrayLike, data_range: Any = None
        ) -> tuple[FloatArray, FloatArray]:
            """Lower and upper bounds of the parameters for a window."""
            ...

        def initial_guess(
            self, times: ArrayLike, waveform: ArrayLike, peak_index: int
        ) -> FloatArray:
            """Return the starting parameters for a candidate extremum of ``waveform``."""
            ...

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    @property
    def causal(self) -> bool:
        return self.onset_param is not None

    def _params(self, params: ArrayLike) -> FloatArray:
        return as_params(params, self.n_params, self.name)

    def _columns(self, params: ArrayLike) -> list[FloatArray]:
        """Parameters as columns of shape (..., 1), ready to broadcast against times."""
        params = self._params(params)
        return [params[..., i, None] for i in range(self.n_params)]

    def features(
        self,
        times: ArrayLike,
        params: ArrayLike,
        amplitude: float = 1.0,
        *,
        onset_fraction: float = ONSET_FRACTION,
        sfreq: float | None = None,
        window: Sequence[float] | None = None,
    ) -> ComponentFeatures:
        """Features of ``amplitude · s(t; params)``; see :func:`compute_features`."""
        return compute_features(
            self,
            times,
            params,
            amplitude,
            onset_fraction=onset_fraction,
            sfreq=sfreq,
            window=window,
        )
