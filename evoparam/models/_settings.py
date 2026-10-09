"""Settings of the fitting engine: package defaults, fit presets and validation."""

from dataclasses import asdict, dataclass, field, replace

import evoparam.presets as _presets
from evoparam.errors import SettingsError

__all__ = ["FitSettings", "FIT_PRESETS", "resolve_settings"]

SELECTIONS = ("bic", "aicc", "cv", "fixed")
LOSSES = ("linear", "soft_l1", "huber", "cauchy")
UNCERTAINTIES = ("jacobian", "profile", "bootstrap", "posterior")
POLARITIES = ("both", "negative", "positive")
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

    shapes: tuple = ("gaussian", "skewed_gaussian")
    background: str = "exponential"
    max_components: int = 6
    min_fwhm: float = 0.002
    max_fwhm: float = 0.300
    min_amplitude: float = 0.0
    detection_threshold: float = 2.0
    polarity: str = "both"
    fit_window: tuple | None = None
    baseline: tuple | None = None
    selection: str = "bic"
    delta: float = 10.0
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
    windows: dict | None = None
    max_masked_fraction: float = 0.5
    background_prior: float | None = 3.0
    amplitude_prior: float | None = 3.0
    edge_extension: float = 0.05
    background_tau_max: float | None = None
    freq_min: float | None = None
    freq_max: float | None = None
    noise_model: str = "ar"
    latency_order: str = "parameterised"     # spec §5.5.3 step 5 ("sorted": not better)
    # "beam" (measured more robust and faster than "spec", §5.5.3 step 4; see CHANGELOG)
    candidate_search: str = "beam"
    # "order": every extremum is a candidate, detection_threshold only orders them (REQ-104);
    # "select": only the extrema >= detection_threshold are candidates (§5.5.3 step 3)
    threshold_role: str = "order"
    n_folds: int = 5
    preset: str | None = None
    preset_note: str | None = None
    random_state: object = None

    def validate(self):
        if isinstance(self.shapes, str):
            object.__setattr__(self, "shapes", (self.shapes,))
        object.__setattr__(self, "shapes", tuple(self.shapes))
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
            raise SettingsError(f"need 0 < min_fwhm ({self.min_fwhm}) < max_fwhm "
                                f"({self.max_fwhm}).")
        if self.max_components < 0:
            raise SettingsError("max_components must be >= 0.")
        if self.latency_order not in LATENCY_ORDERS:
            raise SettingsError(f"latency_order must be one of {LATENCY_ORDERS}.")
        if self.threshold_role not in ("select", "order"):
            raise SettingsError("threshold_role must be 'select' or 'order'.")
        if self.candidate_search not in CANDIDATE_SEARCHES:
            raise SettingsError(f"candidate_search must be one of {CANDIDATE_SEARCHES}.")
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

    def to_dict(self):
        return asdict(self)


# Fit fields of the presets of spec §6.1, read from evoparam.presets (one source of truth).
FIT_PRESETS = {name: _presets.fit_fields(name) for name in _presets.PRESETS}


def resolve_settings(preset=None, **overrides):
    """Package defaults, then the preset's fit fields, then the explicit overrides.

    Overrides left at None do not replace the preset value.
    """

    base = FitSettings()
    known = set(FitSettings.__dataclass_fields__)
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise SettingsError(f"Unknown setting(s) {unknown}.")
    values = {}
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


def settings_field(**kwargs):
    return field(**kwargs)
