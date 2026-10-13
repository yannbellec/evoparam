# SPDX-License-Identifier: Apache-2.0
"""Backgrounds with analytic Jacobians (spec §5.5.2).

A background is ``b(t) = Σ_j c_j B_j(t; τ)``: linear coefficients ``c_j`` on
basis functions that may depend on non-linear parameters ``τ`` (time
constants). The parameter vector is the linear coefficients followed by the
non-linear parameters. ``basis`` and ``basis_jacobian`` expose this structure
for variable projection; ``evaluate`` and ``jacobian`` act on the full vector.

``t_ref`` (the ``t_0`` of the equations) is a setting; when it is None the
first time of the axis passed to each call is used, so a fitted background
should be evaluated with an explicit ``t_ref`` on any other axis.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, SupportsFloat, cast

import numpy as np
from scipy.interpolate import BSpline

from evoparam._typing import ArrayLike, FloatArray
from evoparam.errors import DataError, SettingsError
from evoparam.shapes._base import (
    as_params,
    as_times,
    check_bounds,
    check_waveform,
    clip_to_bounds,
    window_info,
)

# Artifact basis: vectors [n_basis, n_samples], at least this many samples
_BASIS_NDIM = 2
_MIN_BASIS_SAMPLES = 2


__all__ = [
    "BackgroundBase",
    "NoBackground",
    "ConstantBackground",
    "LinearBackground",
    "ExponentialBackground",
    "DoubleExponentialBackground",
    "PSplineBackground",
    "ArtifactBasisBackground",
    "TAU_MIN",
]

# Shortest time constant of the exponential backgrounds (s).
TAU_MIN = 5e-4

# Linear coefficients are bounded by this multiple of the data scale (loose bounds:
# variable projection solves them without bounds).
COEF_BOUND = 10.0


class BackgroundBase:
    """Common methods of the backgrounds.

    Subclasses define ``name``, ``linear_names``, ``nonlinear_names``,
    ``basis`` and, when they have non-linear parameters, ``basis_jacobian``,
    ``_nonlinear_bounds`` and ``_nonlinear_guess``.
    """

    name: ClassVar[str]
    nonlinear_names: ClassVar[tuple[str, ...]] = ()
    if TYPE_CHECKING:  # defined by each background

        @property
        def linear_names(self) -> tuple[str, ...]:
            """Names of the linear coefficients (a class constant, or computed)."""
            ...

        def basis(self, times: ArrayLike, nonlinear: ArrayLike | None) -> FloatArray:
            """Basis functions, shape ``(..., n_linear, n_times)``."""
            ...

    else:
        linear_names = ()

    @property
    def param_names(self) -> tuple[str, ...]:
        """Names of the parameters, the linear coefficients first."""
        return tuple(self.linear_names) + tuple(self.nonlinear_names)

    @property
    def n_params(self) -> int:
        """Number of parameters."""
        return len(self.param_names)

    @property
    def n_linear(self) -> int:
        """Number of linear parameters."""
        return len(self.linear_names)

    @property
    def n_nonlinear(self) -> int:
        """Number of non-linear parameters."""
        return len(self.nonlinear_names)

    @property
    def is_linear(self) -> bool:
        """True when every parameter is linear."""
        return self.n_nonlinear == 0

    def _t_ref(self, times: FloatArray) -> float:
        """Return the time origin of the basis: ``t_ref``, else the first time."""
        t_ref = getattr(self, "t_ref", None)
        return float(times[0]) if t_ref is None else float(t_ref)

    def _nonlinear(self, nonlinear: ArrayLike | None) -> FloatArray:
        """Return the non-linear parameters checked (empty without any)."""
        if self.n_nonlinear == 0:
            return np.zeros(0)
        return as_params(nonlinear, self.n_nonlinear, self.name)

    def basis_jacobian(self, times: ArrayLike, nonlinear: ArrayLike | None = None) -> FloatArray:  # noqa: ARG002  (shape protocol: signature shared by every shape)
        """Return the derivatives of the basis, shape ``(..., n_linear, n_times, n_nonlinear)``."""
        times = as_times(times)
        return np.zeros((self.n_linear, times.size, 0))

    def evaluate(self, times: ArrayLike, params: ArrayLike) -> FloatArray:
        """Background, ``(..., n_params) -> (..., n_times)``."""
        times = as_times(times)
        params = as_params(params, self.n_params, self.name)
        basis = self.basis(times, params[..., self.n_linear :])
        return cast(
            "FloatArray", np.einsum("...j,...jt->...t", params[..., : self.n_linear], basis)
        )

    def jacobian(self, times: ArrayLike, params: ArrayLike) -> FloatArray:
        """Analytic derivatives, ``(..., n_params) -> (..., n_times, n_params)``."""
        times = as_times(times)
        params = as_params(params, self.n_params, self.name)
        lead = params.shape[:-1]
        nonlinear = params[..., self.n_linear :]
        basis = np.swapaxes(self.basis(times, nonlinear), -1, -2)
        linear = np.broadcast_to(basis, lead + (times.size, self.n_linear))
        if self.n_nonlinear == 0:
            return np.array(linear)
        d_basis = self.basis_jacobian(times, nonlinear)
        d_nonlinear = np.einsum("...j,...jtk->...tk", params[..., : self.n_linear], d_basis)
        return np.concatenate([linear, d_nonlinear], axis=-1)

    def _scale(self, data_range: Sequence[SupportsFloat] | None) -> float:
        """Return the scale of the data range (1 without a range)."""
        if data_range is None:
            return 1.0
        low, high = map(float, data_range)
        scale = max(abs(low), abs(high), high - low)
        return scale if scale > 0 else 1.0

    def default_bounds(
        self, times: ArrayLike, data_range: Sequence[SupportsFloat] | None = None
    ) -> tuple[FloatArray, FloatArray]:
        """Return the default bounds of the background parameters.

        Non-linear parameters as documented per background; linear coefficients within
        ``±10 · data scale / max|B_j|`` on the window.
        """
        times = as_times(times)
        nl_lower, nl_upper = self._nonlinear_bounds(times)
        basis = self.basis(times, self._nonlinear_guess(times))
        peak = np.max(np.abs(basis), axis=-1) if basis.size else np.zeros(self.n_linear)
        bound = COEF_BOUND * self._scale(data_range) / np.where(peak > 0, peak, 1.0)
        return check_bounds(
            self.name, np.concatenate([-bound, nl_lower]), np.concatenate([bound, nl_upper])
        )

    def _nonlinear_bounds(self, times: FloatArray) -> tuple[FloatArray, FloatArray]:  # noqa: ARG002  (shape protocol: signature shared by every shape)
        """Return the bounds of the non-linear parameters (none here)."""
        return np.zeros(0), np.zeros(0)

    def _nonlinear_guess(self, times: FloatArray) -> FloatArray:  # noqa: ARG002  (shape protocol: signature shared by every shape)
        """Return the starting non-linear parameters (none here)."""
        return np.zeros(0)

    def initial_guess(self, times: ArrayLike, waveform: ArrayLike) -> FloatArray:
        """Least-squares coefficients for the starting non-linear parameters."""
        times, waveform = check_waveform(times, waveform)
        nonlinear = self._nonlinear_guess(times)
        basis = self.basis(times, nonlinear)
        coefs = np.linalg.lstsq(basis.T, waveform, rcond=None)[0] if self.n_linear else np.zeros(0)
        return clip_to_bounds(
            np.concatenate([coefs, nonlinear]),
            self.default_bounds(times, (np.min(waveform), np.max(waveform))),
        )


###################################################################################################
## Linear backgrounds


@dataclass(frozen=True)
class NoBackground(BackgroundBase):
    """No background: ``b = 0``, no parameter."""

    name: ClassVar[str] = "none"

    def basis(self, times: ArrayLike, nonlinear: ArrayLike | None = None) -> FloatArray:  # noqa: ARG002  (shape protocol: signature shared by every shape)
        """Basis functions, shape ``(0, n_times)``."""
        return np.zeros((0, as_times(times).size))


@dataclass(frozen=True)
class ConstantBackground(BackgroundBase):
    """Constant: ``b = c0``."""

    name: ClassVar[str] = "constant"
    linear_names: ClassVar[tuple[str, ...]] = ("c0",)

    def basis(self, times: ArrayLike, nonlinear: ArrayLike | None = None) -> FloatArray:  # noqa: ARG002  (shape protocol: signature shared by every shape)
        """Basis functions, shape ``(1, n_times)``."""
        return np.ones((1, as_times(times).size))


@dataclass(frozen=True)
class LinearBackground(BackgroundBase):
    """Linear drift: ``b = c0 + c1 (t - t_ref)``."""

    t_ref: float | None = None
    name: ClassVar[str] = "linear"
    linear_names: ClassVar[tuple[str, ...]] = ("c0", "c1")

    def basis(self, times: ArrayLike, nonlinear: ArrayLike | None = None) -> FloatArray:  # noqa: ARG002  (shape protocol: signature shared by every shape)
        """Basis functions, shape ``(2, n_times)``."""
        times = as_times(times)
        return np.stack([np.ones_like(times), times - self._t_ref(times)])


###################################################################################################
## Exponential backgrounds


def _tau_limits(obj: Any, times: FloatArray) -> tuple[float, float]:
    """Return the smallest and largest time constant on ``times``."""
    _, _, length, _ = window_info(times)
    tau_max = length if obj.tau_max is None else float(obj.tau_max)
    return float(obj.tau_min), tau_max


@dataclass(frozen=True)
class ExponentialBackground(BackgroundBase):
    """Recovery tail: ``b = c0 + c1 exp(-(t - t_ref)/tau)``.

    ``tau`` in ``[tau_min, tau_max]``, default ``[0.5 ms, |W|]``.
    """

    t_ref: float | None = None
    tau_min: float = TAU_MIN
    tau_max: float | None = None
    name: ClassVar[str] = "exponential"
    linear_names: ClassVar[tuple[str, ...]] = ("c0", "c1")
    nonlinear_names: ClassVar[tuple[str, ...]] = ("tau",)

    def __post_init__(self) -> None:
        """Check and normalise the fields."""
        if not self.tau_min > 0 or (self.tau_max is not None and self.tau_max <= self.tau_min):
            raise SettingsError(
                f"{self.name}: need 0 < tau_min < tau_max, got {self.tau_min}, {self.tau_max}."
            )

    def basis(self, times: ArrayLike, nonlinear: ArrayLike | None) -> FloatArray:
        """Basis functions, shape ``(..., 2, n_times)``."""
        times = as_times(times)
        tau = self._nonlinear(nonlinear)[..., 0, None]
        decay = np.exp(-(times - self._t_ref(times)) / tau)
        return np.stack([np.ones_like(decay), decay], axis=-2)

    def basis_jacobian(self, times: ArrayLike, nonlinear: ArrayLike | None) -> FloatArray:  # type: ignore[override]  # needs its non-linear parameters
        """Return the derivatives of the basis, shape ``(..., n_linear, n_times, n_nonlinear)``."""
        times = as_times(times)
        tau = self._nonlinear(nonlinear)[..., 0, None]
        lag = times - self._t_ref(times)
        d_decay = np.exp(-lag / tau) * lag / tau**2
        return np.stack([np.zeros_like(d_decay), d_decay], axis=-2)[..., None]

    def _nonlinear_bounds(self, times: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return the bounds of the time constant."""
        low, high = _tau_limits(self, times)
        return np.array([low]), np.array([high])

    def _nonlinear_guess(self, times: FloatArray) -> FloatArray:
        """Return the starting time constant: a fifth of the window."""
        low, high = _tau_limits(self, times)
        _, _, length, _ = window_info(times)
        return np.array([np.clip(length / 5.0, low, high)])


@dataclass(frozen=True)
class DoubleExponentialBackground(BackgroundBase):
    """Two tails: ``b = c0 + c1 exp(-(t - t_ref)/tau1) + c2 exp(-(t - t_ref)/tau2)``.

    Box bounds ``tau1`` in ``[tau_min, tau_max/3]`` and ``tau2`` in
    ``[3 tau_min, tau_max]``; the constraint ``tau2 >= 3 tau1`` of the
    specification is not a box constraint and is listed in ``constraints``.
    """

    t_ref: float | None = None
    tau_min: float = TAU_MIN
    tau_max: float | None = None
    name: ClassVar[str] = "double_exponential"
    linear_names: ClassVar[tuple[str, ...]] = ("c0", "c1", "c2")
    nonlinear_names: ClassVar[tuple[str, ...]] = ("tau1", "tau2")
    constraints: ClassVar[tuple[str, ...]] = ("tau2 >= 3 * tau1",)

    def __post_init__(self) -> None:
        """Check and normalise the fields."""
        if not self.tau_min > 0 or (self.tau_max is not None and self.tau_max <= 3 * self.tau_min):
            raise SettingsError(
                f"{self.name}: need 0 < 3 tau_min < tau_max, got {self.tau_min}, {self.tau_max}."
            )

    def basis(self, times: ArrayLike, nonlinear: ArrayLike | None) -> FloatArray:
        """Basis functions, shape ``(..., 3, n_times)``."""
        times = as_times(times)
        nonlinear = self._nonlinear(nonlinear)
        lag = times - self._t_ref(times)
        first = np.exp(-lag / nonlinear[..., 0, None])
        second = np.exp(-lag / nonlinear[..., 1, None])
        first, second = np.broadcast_arrays(first, second)
        return np.stack([np.ones_like(first), first, second], axis=-2)

    def basis_jacobian(self, times: ArrayLike, nonlinear: ArrayLike | None) -> FloatArray:  # type: ignore[override]  # needs its non-linear parameters
        """Return the derivatives of the basis, shape ``(..., n_linear, n_times, n_nonlinear)``."""
        times = as_times(times)
        nonlinear = self._nonlinear(nonlinear)
        lag = times - self._t_ref(times)
        tau1, tau2 = nonlinear[..., 0, None], nonlinear[..., 1, None]
        d1 = np.exp(-lag / tau1) * lag / tau1**2
        d2 = np.exp(-lag / tau2) * lag / tau2**2
        d1, d2 = np.broadcast_arrays(d1, d2)
        zero = np.zeros_like(d1)
        # Rows: basis functions; last axis: (tau1, tau2).
        return np.stack(
            [np.stack([zero, zero], -1), np.stack([d1, zero], -1), np.stack([zero, d2], -1)],
            axis=-3,
        )

    def _nonlinear_bounds(self, times: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return the bounds of the two time constants (tau2 >= 3 tau1)."""
        low, high = _tau_limits(self, times)
        if not 3 * low < high:
            raise SettingsError(f"{self.name}: the window is too short for tau2 >= 3 tau1.")
        return np.array([low, 3 * low]), np.array([high / 3, high])

    def _nonlinear_guess(self, times: FloatArray) -> FloatArray:
        """Return the starting time constants: a twentieth and a third of the window."""
        lower, upper = self._nonlinear_bounds(times)
        _, _, length, _ = window_info(times)
        return np.clip([length / 20.0, length / 3.0], lower, upper)


###################################################################################################
## P-spline


@dataclass(frozen=True, eq=False)
class PSplineBackground(BackgroundBase):
    """Cubic B-spline background ``b = Σ_j c_j B_j(t)`` with a second-difference penalty.

    Knots are equally spaced over ``window`` with a spacing of at least
    ``knot_spacing``, which must be at least ``2 · max_fwhm`` so that the spline
    cannot represent a feature narrower than the widest admissible component
    (spec §5.5.2); a smaller spacing raises :class:`SettingsError`. The basis is
    zero outside ``window``. The penalty weight ``lam`` is usually chosen by
    :meth:`select_lambda` (generalised cross-validation on the baseline).

    Parameters
    ----------
    window : tuple of float
        ``(start, end)`` of the fit window (s).
    max_fwhm : float
        Largest admissible component FWHM (s).
    knot_spacing : float, optional
        Requested knot spacing (s). Default ``2 · max_fwhm``.
    lam : float, optional
        Penalty weight, None until chosen.
    """

    window: tuple[float, float]
    max_fwhm: float
    knot_spacing: float | None = None
    lam: float | None = None
    spacing: float = field(init=False)
    knots: FloatArray = field(init=False, repr=False)
    name: ClassVar[str] = "pspline"
    degree: ClassVar[int] = 3

    def __post_init__(self) -> None:
        """Check and normalise the fields."""
        start, end = map(float, self.window)
        if not (np.isfinite(start) and np.isfinite(end) and end > start):
            raise SettingsError(
                f"pspline: window must be (start, end) with end > start, got {self.window}."
            )
        if not (np.isfinite(self.max_fwhm) and self.max_fwhm > 0):
            raise SettingsError(f"pspline: max_fwhm must be positive, got {self.max_fwhm}.")
        requested = 2.0 * self.max_fwhm if self.knot_spacing is None else float(self.knot_spacing)
        if not requested >= 2.0 * self.max_fwhm:
            raise SettingsError(
                f"pspline: knot spacing {requested} s is smaller than "
                f"2 · max_fwhm = {2.0 * self.max_fwhm} s; the spline could "
                "absorb a component."
            )
        if self.lam is not None and not self.lam >= 0:
            raise SettingsError(f"pspline: lam must be non-negative, got {self.lam}.")
        n_intervals = max(1, int(np.floor((end - start) / requested)))
        spacing = max(requested, (end - start) / n_intervals)
        knots = start + spacing * np.arange(-self.degree, n_intervals + self.degree + 1)
        object.__setattr__(self, "window", (start, end))
        object.__setattr__(self, "spacing", float(spacing))
        object.__setattr__(self, "knots", knots)

    @property
    def n_basis(self) -> int:
        """Number of B-spline basis functions."""
        return self.knots.size - self.degree - 1

    @property
    def linear_names(self) -> tuple[str, ...]:
        """Names of the linear coefficients."""
        return tuple(f"c{j}" for j in range(self.n_basis))

    def basis(self, times: ArrayLike, nonlinear: ArrayLike | None = None) -> FloatArray:  # noqa: ARG002  (shape protocol: signature shared by every shape)
        """Basis functions, shape ``(n_basis, n_times)``; zero outside the spline support."""
        times = as_times(times)
        spline = BSpline(self.knots, np.eye(self.n_basis), self.degree, extrapolate=False)
        values = spline(times)
        return cast("FloatArray", np.nan_to_num(values, nan=0.0).T)

    def penalty_matrix(self) -> FloatArray:
        """Second-difference penalty ``DᵀD``, shape ``(n_basis, n_basis)``."""
        diff = np.diff(np.eye(self.n_basis), 2, axis=0)
        return cast("FloatArray", diff.T @ diff)

    def select_lambda(
        self,
        times: ArrayLike,
        values: ArrayLike,
        lambdas: Sequence[float] | FloatArray | None = None,
    ) -> float:
        """Penalty weight minimising generalised cross-validation on a segment.

        A P-spline with the same knot spacing is fitted to ``values`` (usually
        the baseline segment) for each candidate ``lam``;
        ``GCV = n · RSS / (n - tr H)²`` with ``H`` the hat matrix.

        Returns
        -------
        float
            The selected ``lam``.
        """
        times, values = check_waveform(times, values)
        if lambdas is None:
            lambdas = np.logspace(-6, 6, 49, dtype=np.float64)
        segment = PSplineBackground((times[0], times[-1]), self.max_fwhm, self.spacing)
        design = segment.basis(times).T
        penalty = segment.penalty_matrix()
        gram, rhs = design.T @ design, design.T @ values
        n = times.size
        scores = []
        for lam in lambdas:
            system = gram + lam * penalty
            coefs = np.linalg.solve(system, rhs)
            rss = float(np.sum((values - design @ coefs) ** 2))
            trace = float(np.trace(np.linalg.solve(system, gram)))
            scores.append(n * rss / (n - trace) ** 2 if trace < n else np.inf)
        return float(np.asarray(lambdas)[int(np.argmin(scores))])


###################################################################################################
## Artifact basis


@dataclass(frozen=True, eq=False)
class ArtifactBasisBackground(BackgroundBase):
    """Learned artifact basis: ``b = Σ_j c_j u_j(t)`` (``ArtifactResult.basis``, spec §5.3).

    The basis vectors are sampled at ``basis_times`` and linearly interpolated
    onto the axis of each call; they are zero outside ``basis_times``.

    Parameters
    ----------
    basis_vectors : 2d array, shape (n_basis, n_samples)
        Basis vectors ``u_j``.
    basis_times : 1d array, shape (n_samples,)
        Their time axis (s).
    """

    basis_vectors: FloatArray
    basis_times: FloatArray
    name: ClassVar[str] = "artifact_basis"

    def __post_init__(self) -> None:
        """Check and normalise the fields."""
        vectors = np.asarray(self.basis_vectors, dtype=np.float64)
        times = np.asarray(self.basis_times, dtype=np.float64)
        if (
            vectors.ndim != _BASIS_NDIM
            or vectors.shape[0] < 1
            or times.ndim != 1
            or vectors.shape[1] != times.size
            or times.size < _MIN_BASIS_SAMPLES
        ):
            raise DataError(
                "artifact_basis: basis_vectors must be (n_basis, n_samples) with "
                "n_basis >= 1 and basis_times of length n_samples >= 2."
            )
        if not (np.all(np.isfinite(vectors)) and np.all(np.isfinite(times))) or np.any(
            np.diff(times) <= 0
        ):
            raise DataError(
                "artifact_basis: basis must be finite and basis_times strictly increasing."
            )
        object.__setattr__(self, "basis_vectors", vectors)
        object.__setattr__(self, "basis_times", times)

    @property
    def linear_names(self) -> tuple[str, ...]:
        """Names of the linear coefficients."""
        return tuple(f"c{j}" for j in range(self.basis_vectors.shape[0]))

    def basis(self, times: ArrayLike, nonlinear: ArrayLike | None = None) -> FloatArray:  # noqa: ARG002  (shape protocol: signature shared by every shape)
        """Basis functions, shape ``(n_basis, n_times)``."""
        times = as_times(times)
        return np.stack(
            [
                np.interp(times, self.basis_times, row, left=0.0, right=0.0)
                for row in self.basis_vectors
            ]
        )
