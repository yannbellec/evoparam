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
SHAPE_CHOICES = ("bic", "aic")
# the fall of the criterion a component needs when delta is None and the acceptance cannot
# be calibrated (a noise model other than "ar", a selection other than "bic"): spec §5.5.4
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
    # None (default): the fall of BIC a component needs is calibrated on each fit's noise
    # model at the false-response rate ``alpha`` (models/_calibration.py; noise_model "ar"
    # and selection "bic"; otherwise FIXED_DELTA); a number: that fixed fall (spec §5.5.4,
    # delta = 10, which every preset sets). Measured: CHANGELOG, "Calibrated acceptance"
    delta: float | None = None
    alpha: float = 0.03
    # how the allowed shapes of a component are compared: "bic" (on the quick fits of the
    # forward search, spec §5.5.3 step 4; every preset), "aic" (default: every component
    # of the final model refitted with every allowed shape, the lowest AIC kept)
    shape_choice: str = "aic"
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
    # True: refuse every model with two overlapping components of opposite signs (a
    # cancelling pair). Default False: such a final state is refitted from the extrema of
    # the data (``reseed``) and, when it stays, its components are marked not resolvable;
    # measured better on ECAP simulations (CHANGELOG, "Defaults settled by measurement")
    reject_cancelling: bool = False
    # "order": every extremum is a candidate, detection_threshold only orders them (REQ-104);
    # "select": only the extrema >= detection_threshold are candidates (§5.5.3 step 3)
    threshold_role: str = "order"
    n_folds: int = 5
    # after the greedy path, refit a state holding a cancelling pair (opposite components
    # overlapping) or a nearly singular design from starts on the extrema of the data, and
    # keep it when it fits better; "always": for every state (slower; better on ERPs)
    reseed: Any = True
    # noise scale and AR model of the whitening: "baseline" (spec §5.5.3 step 1),
    # "plusminus" (the trials averaged with alternating signs over the whole epoch, when the
    # fit has at least 10 trials) or "auto": plus-minus when the baseline holds fewer than
    # 30 independent samples (n (1 - rho) / (1 + rho)), the baseline otherwise
    noise_source: str = "auto"
    preset: str | None = None
    preset_note: str | None = None
    random_state: Any = None

    def validate(self) -> "FitSettings":
        if isinstance(self.shapes, str):
            object.__setattr__(self, "shapes", (self.shapes,))
        object.__setattr__(self, "shapes", tuple(self.shapes))
        if self.windows is not None:
            # a ComponentWindows (evoparam.models.windows), any mapping of windows, or the
            # lists of a settings record read back from JSON
            object.__setattr__(self, "windows", {k: tuple(v) for k, v in self.windows.items()})
        if not self.shapes:
            raise SettingsError("shapes: at least one shape is needed.")
        if self.selection not in SELECTIONS:
            raise SettingsError(f"selection must be one of {SELECTIONS}.")
        if self.loss not in LOSSES:
            raise SettingsError(f"loss must be one of {LOSSES}.")
        if self.uncertainty not in UNCERTAINTIES:
            raise SettingsError(f"uncertainty must be one of {UNCERTAINTIES}.")
        if self.polarity not in POLARITIES:
            raise SettingsError(f"polarity must be one of {POLARITIES}.")
        if not 0 < self.min_fwhm < self.max_fwhm:
            raise SettingsError(
                f"need 0 < min_fwhm ({self.min_fwhm}) < max_fwhm ({self.max_fwhm})."
            )
        if self.max_components < 0:
            raise SettingsError("max_components must be >= 0.")
        if self.latency_order not in LATENCY_ORDERS:
            raise SettingsError(f"latency_order must be one of {LATENCY_ORDERS}.")
        if self.threshold_role not in ("select", "order"):
            raise SettingsError("threshold_role must be 'select' or 'order'.")
        if self.candidate_search not in CANDIDATE_SEARCHES:
            raise SettingsError(f"candidate_search must be one of {CANDIDATE_SEARCHES}.")
        if self.shape_choice not in SHAPE_CHOICES:
            raise SettingsError(f"shape_choice must be one of {SHAPE_CHOICES}.")
        if self.delta is not None:
            if not np.isfinite(float(self.delta)):
                raise SettingsError("delta must be a finite number or None (calibrated).")
            object.__setattr__(self, "delta", float(self.delta))
        if not 0 < self.alpha < 1:
            raise SettingsError("alpha must be in (0, 1).")
        if self.noise_model not in NOISE_MODELS:
            raise SettingsError(f"noise_model must be one of {NOISE_MODELS}.")
        if self.selection == "fixed" and self.n_components is None:
            raise SettingsError("selection='fixed' needs n_components.")
        if self.n_starts < 1 or self.max_iter < 1:
            raise SettingsError("n_starts and max_iter must be >= 1.")
        if not 0 < self.ci_level < 1:
            raise SettingsError("ci_level must be in (0, 1).")
        for name in ("fit_window", "baseline"):
            value = getattr(self, name)
            if value is not None:
                value = (float(value[0]), float(value[1]))
                if not value[1] > value[0]:
                    raise SettingsError(f"{name} {value}: the end must be after the start.")
                object.__setattr__(self, name, value)
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _FitPresets(Mapping[str, dict[str, Any]]):
    """Fit fields of every preset (spec §6.1), read from evoparam.presets on access.

    One source of truth; plugin presets included, and loaded only when first needed.
    """

    def __getitem__(self, name: str) -> dict[str, Any]:
        if name not in _presets.PRESETS:
            raise KeyError(name)
        return _presets.fit_fields(name)

    def __iter__(self) -> Iterator[str]:
        return iter(_presets.PRESETS)

    def __len__(self) -> int:
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
        else:
            raise SettingsError(f"Unknown preset {preset!r}; known: {sorted(_presets.PRESETS)}.")
    values.update({key: value for key, value in overrides.items() if value is not None})
    return replace(base, **values).validate()


def settings_field(**kwargs: Any) -> Any:
    return field(**kwargs)
