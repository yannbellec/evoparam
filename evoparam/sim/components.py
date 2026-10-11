# SPDX-License-Identifier: Apache-2.0
"""Component specifications, presets, and the conversion of latency and FWHM to shape parameters.

A simulated component is described by a :class:`ComponentSpec`: a label, a shape
of :mod:`evoparam.shapes`, a latency, a full width at half maximum and a signed
amplitude (each a value or a ``(low, high)`` range drawn uniformly), and the
channels it appears on with their spatial weights. :func:`shape_params` turns a
latency and a FWHM into the parameters of the shape, so that the shape's
reported latency and FWHM are exactly the requested ones.
"""

import functools
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from evoparam.errors import SettingsError
from evoparam.shapes import FWHM_PER_SIGMA, compute_features, get_shape

__all__ = ["ComponentSpec", "ERP_COMPONENTS", "TEP_COMPONENTS", "CCEP_N1", "CCEP_N2",
           "CCEP_DELAYED", "shape_params", "as_spec", "ROLES"]

# Roles of a simulated component in the ground truth.
#   target: what the generator was asked to produce; overlap: response to a neighbouring
#   event; confound: a real response that is not the one studied (auditory click of a
#   TMS pulse); masker: response to the masker of a forward-masking paradigm.
ROLES = ("target", "overlap", "confound", "masker")

# Per shape: the location parameter (moved by latency jitter), the width parameter
# (scaled to reach the FWHM), dimensionless parameters with their defaults, and
# parameters given as a multiple of the width parameter (keyword ``<name>_ratio``).
SHAPE_FORMS = {
    "gaussian": {"loc": "mu", "scale": "sigma", "fixed": {}, "ratios": {}},
    "skewed_gaussian": {"loc": "xi", "scale": "omega", "fixed": {"alpha": 2.0}, "ratios": {}},
    "exgaussian": {"loc": "mu", "scale": "sigma", "fixed": {}, "ratios": {"tau": 1.0}},
    "gamma": {"loc": "t0", "scale": "theta", "fixed": {"kappa": 3.0}, "ratios": {}},
    "biexponential": {"loc": "t0", "scale": "tau_rise", "fixed": {},
                      "ratios": {"tau_decay": 4.0}},
}

# Oscillatory shapes: latency is the onset (damped) or the envelope centre (Gabor).
OSCILLATORY = ("damped_oscillation", "gabor")


###################################################################################################
## Latency and FWHM -> shape parameters

def _form_settings(name, form):
    """Split ``form`` into the fixed and ratio parameters of a shape, with defaults."""

    spec = SHAPE_FORMS[name]
    fixed = dict(spec["fixed"])
    ratios = dict(spec["ratios"])
    for key, value in form.items():
        if key in fixed:
            fixed[key] = float(value)
        elif key.endswith("_ratio") and key[:-6] in ratios:
            ratios[key[:-6]] = float(value)
        else:
            raise SettingsError(f"shape '{name}': unknown setting '{key}'; expected "
                                f"{sorted(fixed) + sorted(k + '_ratio' for k in ratios)}.")
    return tuple(sorted(fixed.items())), tuple(sorted(ratios.items()))


def _assemble(name, loc, scale, fixed, ratios):
    spec = SHAPE_FORMS[name]
    values = dict(fixed)
    values.update({key: ratio * scale for key, ratio in ratios})
    values[spec["loc"]] = loc
    values[spec["scale"]] = scale
    shape = get_shape(name)
    return np.array([values[key] for key in shape.param_names], dtype=np.float64)


@functools.lru_cache(maxsize=256)
def _unit_form(name, fixed, ratios):
    """Latency and FWHM of a shape with location 0 and width parameter 1.

    Both scale linearly with the width parameter and the latency moves with the
    location, so one evaluation per set of dimensionless parameters is enough.
    """

    shape = get_shape(name)
    params = _assemble(name, 0.0, 1.0, fixed, ratios)
    if name == "gaussian":
        return 0.0, float(FWHM_PER_SIGMA)
    times = np.linspace(-20.0, 60.0, 1601)
    features = compute_features(shape, times, params)
    if not np.isfinite(features.fwhm) or features.fwhm <= 0:
        raise SettingsError(f"shape '{name}': the FWHM of {dict(fixed + ratios)} cannot be "
                            "measured.")
    return float(shape.latency(params)), float(features.fwhm)


def shape_params(shape, latency, fwhm=None, **form):
    """Parameters of ``shape`` whose latency and FWHM are the requested values.

    Parameters
    ----------
    shape : str
        A shape of :data:`evoparam.shapes.SHAPES` except ``"template"``.
    latency : float
        Reported latency (s): the time of the extremum, or the onset ``t0`` of a
        ``damped_oscillation`` and the envelope centre ``t0`` of a ``gabor``.
    fwhm : float
        Full width at half maximum (s). For ``gabor`` it is the FWHM of the
        Gaussian envelope; for ``damped_oscillation`` it is the half-life of the
        envelope divided by ln 2 when ``tau`` is not given.
    **form
        Dimensionless settings: ``alpha`` (skewed Gaussian, default 2), ``kappa``
        (gamma, default 3), ``tau_ratio`` (ex-Gaussian ``tau / sigma``, default 1),
        ``tau_decay_ratio`` (biexponential ``tau_decay / tau_rise``, default 4); for
        the oscillatory shapes ``freq`` (Hz, required), ``phase`` (rad, default 0) and
        ``tau`` (s, damped oscillation).

    Returns
    -------
    params : ndarray, shape (n_params,)
        In the order of ``get_shape(shape).param_names``.

    Raises
    ------
    SettingsError
        Unknown shape or setting, non-positive FWHM, or a missing frequency.

    Examples
    --------
    A Gaussian peaking at 100 ms with a FWHM of 20 ms (``sigma = FWHM / 2.3548``):

    >>> [round(float(value), 6) for value in shape_params("gaussian", 0.1, 0.02)]
    [0.1, 0.008493]
    """

    latency = float(latency)
    if shape in OSCILLATORY:
        extra = set(form) - {"freq", "phase", "tau"}
        if extra or "freq" not in form:
            raise SettingsError(f"shape '{shape}' needs 'freq' (and optionally 'phase'"
                                f"{', tau' if shape == 'damped_oscillation' else ''}); "
                                f"got {sorted(form)}.")
        freq, phase = float(form["freq"]), float(form.get("phase", 0.0))
        if shape == "gabor":
            if "tau" in form or fwhm is None or not fwhm > 0:
                raise SettingsError("shape 'gabor' needs a positive fwhm (of its envelope) "
                                    "and no 'tau'.")
            return np.array([latency, float(fwhm) / FWHM_PER_SIGMA, freq, phase])
        tau = form.get("tau")
        if tau is None:
            if fwhm is None or not fwhm > 0:
                raise SettingsError("shape 'damped_oscillation' needs 'tau' or a positive "
                                    "fwhm.")
            tau = float(fwhm) / np.log(2.0)
        return np.array([latency, float(tau), freq, phase])
    if shape not in SHAPE_FORMS:
        raise SettingsError(f"shape '{shape}' cannot be placed by latency and FWHM; "
                            f"use one of {sorted(SHAPE_FORMS) + list(OSCILLATORY)} or give "
                            "its params.")
    if fwhm is None or not (np.isfinite(fwhm) and fwhm > 0):
        raise SettingsError(f"shape '{shape}': fwhm must be a positive number, got {fwhm}.")
    fixed, ratios = _form_settings(shape, form)
    unit_latency, unit_fwhm = _unit_form(shape, fixed, ratios)
    scale = float(fwhm) / unit_fwhm
    return _assemble(shape, latency - unit_latency * scale, scale, fixed, ratios)


###################################################################################################
## Specifications

def _draw(value, rng):
    """A number, or a uniform draw in a ``(low, high)`` range."""

    if isinstance(value, (tuple, list, np.ndarray)):
        if len(value) != 2:
            raise SettingsError(f"a range must be (low, high), got {value!r}.")
        low, high = float(value[0]), float(value[1])
        if high < low:
            raise SettingsError(f"a range must be (low, high) with low <= high, got {value!r}.")
        return low if high == low else float(rng.uniform(low, high))
    return None if value is None else float(value)


@dataclass(frozen=True)
class ComponentSpec:
    """Specification of one simulated component.

    Parameters
    ----------
    label : str
        Name, for example ``"N1"`` or ``"P3"``.
    latency : float or (float, float)
        Reported latency (s), or a range drawn uniformly once per simulation.
    fwhm : float or (float, float), optional
        Full width at half maximum (s), or a range (see :func:`shape_params`).
    amplitude : float or (float, float)
        Signed peak amplitude (data units), or a range. The sign is kept when a
        signal-to-noise ratio sets the magnitude.
    shape : str
        Shape name of :data:`evoparam.shapes.SHAPES`.
    form : dict
        Dimensionless shape settings passed to :func:`shape_params`.
    params : tuple of float, optional
        Explicit shape parameters; ``latency`` and ``fwhm`` are then ignored.
    channels : tuple of (str or int), optional
        Channels on which the component appears (names or indices); all when None.
    weights : tuple of float, optional
        Spatial gain on each channel of ``channels`` (or on every channel when
        ``channels`` is None). A negative weight reverses the polarity there.
    role : str
        One of :data:`ROLES`.
    """

    label: str
    latency: object = 0.1
    fwhm: object = 0.05
    amplitude: object = 1.0
    shape: str = "gaussian"
    form: dict = field(default_factory=dict)
    params: tuple | None = None
    channels: tuple | None = None
    weights: tuple | None = None
    role: str = "target"

    def __post_init__(self):
        if not isinstance(self.label, str) or not self.label:
            raise SettingsError(f"ComponentSpec: label must be a non-empty string, "
                                f"got {self.label!r}.")
        if self.role not in ROLES:
            raise SettingsError(f"ComponentSpec '{self.label}': role must be one of {ROLES}.")
        if self.weights is not None and self.channels is not None \
                and len(self.weights) != len(self.channels):
            raise SettingsError(f"ComponentSpec '{self.label}': {len(self.weights)} weights "
                                f"for {len(self.channels)} channels.")

    def to_dict(self):
        """Plain dictionary of the specification (JSON-serialisable)."""

        out = asdict(self)
        for key in ("latency", "fwhm", "amplitude", "params", "channels", "weights"):
            if isinstance(out[key], (tuple, np.ndarray)):
                out[key] = list(np.asarray(out[key]).tolist()) if key != "channels" \
                    else list(out[key])
        return out

    def resolve(self, rng):
        """Draw the ranges; return ``(params, amplitude)`` of the nominal component."""

        amplitude = _draw(self.amplitude, rng)
        if self.params is not None:
            shape = get_shape(self.shape)
            params = np.asarray(self.params, dtype=np.float64)
            if params.shape != (shape.n_params,):
                raise SettingsError(f"ComponentSpec '{self.label}': shape '{self.shape}' "
                                    f"takes {shape.n_params} params {shape.param_names}.")
            return params, amplitude
        latency = _draw(self.latency, rng)
        fwhm = _draw(self.fwhm, rng)
        return shape_params(self.shape, latency, fwhm, **self.form), amplitude


def as_spec(obj, presets=None):
    """A :class:`ComponentSpec` from a spec, a dict of its fields, or a preset name."""

    if isinstance(obj, ComponentSpec):
        return obj
    if isinstance(obj, str):
        table = ERP_COMPONENTS if presets is None else presets
        if obj not in table:
            raise SettingsError(f"unknown component preset '{obj}'; known: {sorted(table)}.")
        return table[obj]
    if isinstance(obj, dict):
        base = dict(obj)
        if "label" in base and presets is not None and base["label"] in presets \
                and base.get("preset", True):
            base.pop("preset", None)
            return replace(presets[base["label"]], **base)
        base.pop("preset", None)
        return ComponentSpec(**base)
    raise SettingsError(f"cannot read a component specification from {obj!r}.")


###################################################################################################
## Presets

# ERP components. Latency ranges follow the ERP CORE measurement windows where ERP CORE
# defines the component (N170 110-150 ms, MMN 125-225 ms, N400 300-500 ms, P3 300-600 ms,
# ERN 0-100 ms after the response) [LIT-erpmeas-39]; P1 and N1 (not ERP CORE components)
# use the usual visual ranges. Widths and amplitudes (µV) are typical values; they set the
# simulation, not a claim about the literature.
ERP_COMPONENTS = {
    "P1": ComponentSpec("P1", latency=(0.080, 0.130), fwhm=(0.030, 0.050), amplitude=(2.0, 5.0)),
    "N1": ComponentSpec("N1", latency=(0.130, 0.200), fwhm=(0.040, 0.070),
                        amplitude=(-6.0, -2.0)),
    "N170": ComponentSpec("N170", latency=(0.110, 0.150), fwhm=(0.030, 0.050),
                          amplitude=(-8.0, -3.0)),
    "MMN": ComponentSpec("MMN", latency=(0.125, 0.225), fwhm=(0.050, 0.100),
                         amplitude=(-4.0, -1.0)),
    "N400": ComponentSpec("N400", latency=(0.300, 0.500), fwhm=(0.150, 0.250),
                          amplitude=(-8.0, -3.0)),
    "P3": ComponentSpec("P3", latency=(0.300, 0.600), fwhm=(0.150, 0.300),
                        amplitude=(5.0, 15.0)),
    "ERN": ComponentSpec("ERN", latency=(0.000, 0.100), fwhm=(0.040, 0.080),
                         amplitude=(-12.0, -4.0)),
}

# TMS-evoked components and their latency ranges [LIT-domains-1]; widths and amplitudes
# (µV at the stimulated site) are typical values.
TEP_COMPONENTS = {
    "N15": ComponentSpec("N15", latency=(0.010, 0.020), fwhm=(0.006, 0.010),
                         amplitude=(-3.0, -1.0)),
    "P30": ComponentSpec("P30", latency=(0.025, 0.035), fwhm=(0.008, 0.014), amplitude=(1.5, 4.0)),
    "N45": ComponentSpec("N45", latency=(0.040, 0.055), fwhm=(0.012, 0.020),
                         amplitude=(-5.0, -2.0)),
    "P60": ComponentSpec("P60", latency=(0.055, 0.070), fwhm=(0.012, 0.020), amplitude=(2.0, 5.0)),
    "N100": ComponentSpec("N100", latency=(0.090, 0.130), fwhm=(0.030, 0.050),
                          amplitude=(-8.0, -3.0)),
    "P180": ComponentSpec("P180", latency=(0.160, 0.220), fwhm=(0.050, 0.080),
                          amplitude=(3.0, 7.0)),
}

# CCEP components (spec §5.12): N1 15-40 ms, FWHM 5-15 ms; N2 80-250 ms, FWHM 40-150 ms;
# delayed responses 100 ms-1 s [LIT-ccep-19]. Amplitudes in µV.
CCEP_N1 = ComponentSpec("N1", latency=(0.015, 0.040), fwhm=(0.005, 0.015),
                        amplitude=(-500.0, -50.0))
CCEP_N2 = ComponentSpec("N2", latency=(0.080, 0.250), fwhm=(0.040, 0.150),
                        amplitude=(-300.0, -30.0))
CCEP_DELAYED = ComponentSpec("delayed", latency=(0.100, 1.000), fwhm=(0.050, 0.200),
                             amplitude=(-200.0, -30.0))
