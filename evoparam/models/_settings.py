"""Settings of the fitting engine: package defaults, fit presets and validation."""

from dataclasses import asdict, dataclass, field, replace

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
    noise_model: str = "ar"
    n_folds: int = 5
    preset: str | None = None
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


# Fit fields of the presets of spec §6.1 (the full presets, with artifact, reference and
# trial settings, are the subject of build step S15).
FIT_PRESETS = {
    "erp": dict(fit_window=(0.05, 0.7), baseline=(-0.2, 0.0),
                shapes=("gaussian", "skewed_gaussian"), background="constant",
                max_components=6, min_fwhm=0.010, max_fwhm=0.35,
                windows={"P1": (0.08, 0.13, 1), "N1": (0.13, 0.20, -1), "N170": (0.13, 0.20, -1),
                         "P2": (0.15, 0.27, 1), "P3": (0.25, 0.60, 1),
                         "N400": (0.30, 0.50, -1)}),
    "ccep": dict(fit_window=(0.009, 0.5), baseline=(-0.5, -0.02),
                 background_tau_max=0.05,
                 shapes=("skewed_gaussian", "gaussian"), background="exponential",
                 max_components=6, min_fwhm=0.003, max_fwhm=0.25,
                 windows={"N1": (0.009, 0.050, -1), "N2": (0.050, 0.300, -1)}),
    "ccep_seeg": dict(fit_window=(0.009, 0.5), baseline=(-0.5, -0.02),
                 background_tau_max=0.05,
                      shapes=("skewed_gaussian", "gaussian"), background="exponential",
                      max_components=6, min_fwhm=0.003, max_fwhm=0.25,
                      windows={"N1": (0.009, 0.050, -1), "N2": (0.050, 0.300, -1)}),
    "sep": dict(fit_window=(0.010, 0.060), baseline=(-0.05, -0.005),
                shapes=("gaussian", "skewed_gaussian"), background="linear",
                max_components=5, min_fwhm=0.002, max_fwhm=0.025,
                windows={"N20": (0.017, 0.024, -1), "P25": (0.022, 0.030, 1),
                         "N33": (0.028, 0.040, -1)}),
    "tep": dict(fit_window=(0.012, 0.3), baseline=(-0.8, -0.1),
                shapes=("gaussian", "skewed_gaussian", "gabor"),
                background="double_exponential", max_components=8, min_fwhm=0.005,
                max_fwhm=0.15,
                windows={"N15": (0.010, 0.020, -1), "P30": (0.022, 0.038, 1),
                         "N45": (0.038, 0.055, -1), "P60": (0.050, 0.075, 1),
                         "N100": (0.080, 0.140, -1), "P180": (0.150, 0.250, 1)}),
    "erna": dict(fit_window=(0.0015, 0.0077), baseline=None,
                 shapes=("damped_oscillation",), background="exponential",
                 max_components=2, min_fwhm=0.0005, max_fwhm=0.003,
                 windows={"ERNA": (0.0015, 0.0077, 0)}),
    "ecap": dict(fit_window=(0.0003, 0.003), baseline=(-0.0005, -0.00005),
                 shapes=("biexponential", "gaussian"), background="exponential",
                 max_components=3, min_fwhm=0.0001, max_fwhm=0.0015,
                 windows={"N1": (0.0003, 0.0012, -1), "P2": (0.0008, 0.0025, 1)}),
    "ecap_ci": dict(fit_window=(0.0001, 0.0015), baseline=None,
                    shapes=("biexponential", "gaussian"), background="linear",
                    max_components=2, min_fwhm=0.00005, max_fwhm=0.0007,
                    windows={"N1": (0.0001, 0.0005, -1), "P2": (0.0004, 0.0010, 1)}),
    "rodent_ep": dict(fit_window=(0.005, 0.3), baseline=(-0.1, -0.005),
                      shapes=("gamma", "skewed_gaussian"), background="exponential",
                      max_components=5, min_fwhm=0.003, max_fwhm=0.15,
                      windows={"early": (0.005, 0.060, 0), "late": (0.060, 0.300, 0)}),
    "primate_ep": dict(fit_window=(0.005, 0.3), baseline=(-0.1, -0.005),
                       shapes=("gamma", "skewed_gaussian"), background="exponential",
                       max_components=5, min_fwhm=0.003, max_fwhm=0.15,
                       windows={"early": (0.005, 0.060, 0), "late": (0.060, 0.300, 0)}),
    "p300_bci": dict(fit_window=(0.2, 0.7), baseline=(-0.2, 0.0),
                     shapes=("gaussian", "skewed_gaussian"), background="constant",
                     max_components=4, min_fwhm=0.020, max_fwhm=0.30,
                     windows={"P3": (0.25, 0.60, 1)}),
}


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
        elif preset in FIT_PRESETS:
            values.update(FIT_PRESETS[preset])
            values["preset"] = preset
        else:
            raise SettingsError(f"Unknown preset {preset!r}; known: {sorted(FIT_PRESETS)}.")
    values.update({key: value for key, value in overrides.items() if value is not None})
    return replace(base, **values).validate()


def settings_field(**kwargs):
    return field(**kwargs)
