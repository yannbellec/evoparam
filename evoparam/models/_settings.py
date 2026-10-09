"""Settings of the fitting engine: package defaults, fit presets and validation."""

from dataclasses import asdict, dataclass, field, replace

import evoparam.presets as _presets
from evoparam.errors import SettingsError

__all__ = ["FitSettings", "FIT_PRESETS", "resolve_settings"]

SELECTIONS = ("bic", "aicc", "cv", "fixed")
LOSSES = ("linear", "soft_l1", "huber", "cauchy")
UNCERTAINTIES = ("jacobian", "profile", "bootstrap", "posterior")
POLARITIES = ("both", "negative", "positive")


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
    n_folds: int = 5
    preset: str | None = None
    preset_note: str | None = None
    random_state: object = None

    def validate(self):
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
            raise SettingsError(f"need 0 < min_fwhm ({self.min_fwhm}) < max_fwhm "
                                f"({self.max_fwhm}).")
        if self.max_components < 0:
            raise SettingsError("max_components must be >= 0.")
        if self.noise_model not in ("ar", "white"):
            raise SettingsError("noise_model must be 'ar' or 'white'.")
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
    if isinstance(preset, dict) and set(preset) == {"name", "version"}:
        # the preset as recorded in a result's settings (REQ-331): the same preset, which
        # must still have the recorded version
        recorded = preset
        preset = recorded["name"]
        if preset in _presets.PRESETS and recorded["version"] is not None \
                and _presets.get(preset).version != recorded["version"]:
            from evoparam.errors import EvoparamVersionError

            raise EvoparamVersionError(
                f"The settings were made with preset {preset!r} v{recorded['version']}; this "
                f"evoparam has v{_presets.get(preset).version}.")
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
