# SPDX-License-Identifier: Apache-2.0
"""Settings of the fitting engine: package defaults, fit presets and validation."""

from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field, replace

import numpy as np

import evoparam.presets as _presets
from evoparam._typing import Any
from evoparam.errors import SettingsError

__all__ = ["FitSettings", "FIT_PRESETS", "resolve_settings"]

SELECTIONS = ("bic", "aicc", "cv", "fixed")
LOSSES = ("linear", "soft_l1", "huber", "cauchy")
UNCERTAINTIES = ("jacobian", "profile", "bootstrap", "posterior")
POLARITIES = ("both", "negative", "positive")
SHAPE_CHOICES = ("bic", "hqc", "aic", "penalised")
# ``delta`` value of the calibrated acceptance (a preset or a user asks for it with
# delta="calibrated"; the default is the fixed 10 of spec §5.5.4)
CALIBRATED = "calibrated"
# the fall of the criterion a component needs when the acceptance is calibrated but cannot
# be (a noise model other than "ar", a selection other than "bic"): spec §5.5.4
FIXED_DELTA = 10.0
# "ar": residuals whitened by an AR model of the baseline (default, measured better: K1);
# "neff": spec §5.5.3 step 1 (residuals over sigma_n, n_eff = n (1 - rho) / (1 + rho));
# "white": residuals over sigma_n with the raw sample count n
NOISE_MODELS = ("ar", "neff", "white")
# "parameterised": latencies ordered inside every optimisation, mu_{k+1} = mu_k + exp(v_k)
# (spec §5.5.3 step 5); "sorted": free latencies, components sorted after each step
LATENCY_ORDERS = ("parameterised", "sorted")
# "spec": candidates tried strongest first, the first accepted is kept (spec §5.5.3 step 4);
# "beam": the first qualifying candidates compared, split proposals, background restarts
CANDIDATE_SEARCHES = ("spec", "beam")


@dataclass(frozen=True)
class FitSettings:
    """Every setting of one fit (spec §5.5.8); the defaults are the package defaults."""

    shapes: tuple[str, ...] = ("gaussian", "skewed_gaussian")
    background: str = "exponential"
    max_components: int = 6
    min_fwhm: float = 0.002
    max_fwhm: float = 0.300
    min_amplitude: float = 0.0
    detection_threshold: float = 2.0
    polarity: str = "both"
    fit_window: tuple[float, float] | None = None
    baseline: tuple[float, float] | None = None
    selection: str = "bic"
    # BIC drop a component needs (spec §5.5.4: 10), or "calibrated" on each fit's noise
    # model at rate ``alpha`` (models/_calibration.py; ar noise and bic selection only)
    delta: float | str = 10.0
    alpha: float = 0.04
    # shape comparison: "bic" (forward search, every preset), "hqc" or "aic" (final
    # refit with every shape), "penalised" (default: hqc with an asymmetry prior, REQ-106)
    shape_choice: str = "penalised"
    # flag a component that a pair of overlapping components would explain as well
    # ("merged:k,k+1", REQ-111; engine._resolution_warnings); every preset sets False
    resolution_check: bool = True
    n_components: int | None = None
    overlap_c: float = 0.5
    loss: str = "linear"
    f_scale: float | None = None
    max_iter: int = 2000
    n_starts: int = 3
    onset_fraction: float = 0.10
    uncertainty: str = "jacobian"
    n_bootstrap: int = 200
    ci_level: float = 0.95
    classical: bool = True
    windows: dict[str, Any] | None = None
    max_masked_fraction: float = 0.5
    background_prior: float | None = 3.0
    amplitude_prior: float | None = 3.0
    edge_extension: float = 0.05
    background_tau_max: float | None = None
    freq_min: float | None = None
    freq_max: float | None = None
    noise_model: str = "ar"
    latency_order: str = "parameterised"  # spec §5.5.3 step 5 ("sorted": not better)
    # "beam" (measured more robust and faster than "spec", §5.5.3 step 4; see CHANGELOG)
    candidate_search: str = "beam"
    # True: refuse cancelling pairs. False (default): reseed, and mark the pair not resolvable
    # if it stays (better on ECAP simulations, CHANGELOG)
    reject_cancelling: bool = False
    # "order": every extremum is a candidate, detection_threshold only orders them (REQ-104);
    # "select": only the extrema >= detection_threshold are candidates (§5.5.3 step 3)
    threshold_role: str = "order"
    n_folds: int = 5
    # after the greedy path, refit a state holding a cancelling pair (opposite components
    # overlapping) or a nearly singular design from starts on the extrema of the data, and
    # keep it when it fits better; "always": for every state (slower; better on ERPs)
    reseed: Any = True
    # whitening noise source: "baseline", "plusminus" (>= 10 trials) or "auto" (plus-minus
    # when the baseline has fewer than 30 independent samples)
    noise_source: str = "auto"
    preset: str | None = None
    preset_note: str | None = None
    # the preset's peak decisions read on the fitted curve (Preset.detections; for the
    # methods text and the caller, the fit does not use them)
    detections: dict[str, Any] | None = None
    random_state: Any = None

    def validate(self) -> "FitSettings":
        """Normalise the collections and check every field; returns ``self``.

        Raises :class:`~evoparam.errors.SettingsError` naming the first invalid field.
        """
        if isinstance(self.shapes, str):
            object.__setattr__(self, "shapes", (self.shapes,))
        object.__setattr__(self, "shapes", tuple(self.shapes))
        if self.windows is not None:
            # a ComponentWindows (evoparam.models.windows), any mapping of windows, or the
            # lists of a settings record read back from JSON
            object.__setattr__(self, "windows", {k: tuple(v) for k, v in self.windows.items()})
        if not self.shapes:
            raise SettingsError("shapes: at least one shape is needed.")
        for name, allowed in (
            ("selection", SELECTIONS),
            ("loss", LOSSES),
            ("uncertainty", UNCERTAINTIES),
            ("polarity", POLARITIES),
        ):
            self._check_choice(name, allowed)
        if not 0 < self.min_fwhm < self.max_fwhm:
            raise SettingsError(
                f"need 0 < min_fwhm ({self.min_fwhm}) < max_fwhm ({self.max_fwhm})."
            )
        if self.max_components < 0:
            raise SettingsError("max_components must be >= 0.")
        self._check_choice("latency_order", LATENCY_ORDERS)
        if self.threshold_role not in ("select", "order"):
            raise SettingsError("threshold_role must be 'select' or 'order'.")
        self._check_choice("candidate_search", CANDIDATE_SEARCHES)
        self._check_choice("noise_model", NOISE_MODELS)
        self._check_choice("shape_choice", SHAPE_CHOICES)
        self._check_acceptance()
        self._check_counts()
        for name in ("fit_window", "baseline"):
            value = getattr(self, name)
            if value is not None:
                value = (float(value[0]), float(value[1]))
                if not value[1] > value[0]:
                    raise SettingsError(f"{name} {value}: the end must be after the start.")
                object.__setattr__(self, name, value)
        return self

    def _check_choice(self, name: str, allowed: Any) -> None:
        """Raise SettingsError when the field ``name`` is not one of ``allowed``."""
        if getattr(self, name) not in allowed:
            raise SettingsError(f"{name} must be one of {allowed}.")

    def _check_acceptance(self) -> None:
        """Check ``delta`` (a finite number, or ``"calibrated"``) and ``alpha``.

        ``delta=None`` reads as "not given": the fixed ``FIXED_DELTA`` (as in
        :class:`~evoparam.models.ResponseModel`, where None takes the preset's value).
        """
        if self.delta is None:
            object.__setattr__(self, "delta", FIXED_DELTA)
        elif isinstance(self.delta, str):
            if self.delta != CALIBRATED:
                raise SettingsError(f"delta must be a finite number or {CALIBRATED!r}.")
        else:
            if not np.isfinite(float(self.delta)):
                raise SettingsError(f"delta must be a finite number or {CALIBRATED!r}.")
            object.__setattr__(self, "delta", float(self.delta))
        if not 0 < self.alpha < 1:
            raise SettingsError("alpha must be in (0, 1).")

    def _check_counts(self) -> None:
        """Check ``n_components`` (fixed selection), ``n_starts``, ``max_iter``, ``ci_level``."""
        if self.selection == "fixed" and self.n_components is None:
            raise SettingsError("selection='fixed' needs n_components.")
        if self.n_starts < 1 or self.max_iter < 1:
            raise SettingsError("n_starts and max_iter must be >= 1.")
        if not 0 < self.ci_level < 1:
            raise SettingsError("ci_level must be in (0, 1).")

    def to_dict(self) -> dict[str, Any]:
        """Return the settings as a dictionary (``dataclasses.asdict``)."""
        return asdict(self)


class _FitPresets(Mapping[str, dict[str, Any]]):
    """Fit fields of every preset (spec §6.1), read from evoparam.presets on access.

    One source of truth; plugin presets included, and loaded only when first needed.
    """

    def __getitem__(self, name: str) -> dict[str, Any]:
        """Return the fit fields of the preset ``name``."""
        if name not in _presets.PRESETS:
            raise KeyError(name)
        return _presets.fit_fields(name)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the names of the presets."""
        return iter(_presets.PRESETS)

    def __len__(self) -> int:
        """Return the number of presets."""
        return len(_presets.PRESETS)


FIT_PRESETS = _FitPresets()


def resolve_settings(preset: Any = None, **overrides: Any) -> "FitSettings":
    """Package defaults, then the preset's fit fields, then the explicit overrides.

    Overrides left at None do not replace the preset value.
    """
    base = FitSettings()
    known = set(FitSettings.__dataclass_fields__)
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise SettingsError(f"Unknown setting(s) {unknown}.")
    values = {}
    if isinstance(preset, dict) and set(preset) == {"name", "version"}:
        # the preset as recorded in a result's settings (REQ-331): the same preset, which
        # must still have the recorded version
        recorded = preset
        preset = recorded["name"]
        if (
            preset in _presets.PRESETS
            and recorded["version"] is not None
            and _presets.get(preset).version != recorded["version"]
        ):
            from evoparam.errors import EvoparamVersionError

            raise EvoparamVersionError(
                f"The settings were made with preset {preset!r} v{recorded['version']}; this "
                f"evoparam has v{_presets.get(preset).version}."
            )
    if preset is not None:
        if isinstance(preset, dict):
            values.update(preset)
        elif isinstance(preset, _presets.Preset) or preset in _presets.PRESETS:
            values.update(_presets.fit_fields(preset))
            values["preset"] = _presets.get(preset).name
            values["preset_note"] = _presets.methods_note(preset)
            values["detections"] = dict(_presets.get(preset).detections) or None
        else:
            raise SettingsError(f"Unknown preset {preset!r}; known: {sorted(_presets.PRESETS)}.")
    values.update({key: value for key, value in overrides.items() if value is not None})
    return replace(base, **values).validate()


def settings_field(**kwargs: Any) -> Any:
    """Return a dataclass field (``dataclasses.field``) of the settings."""
    return field(**kwargs)
