# SPDX-License-Identifier: Apache-2.0
"""Presets: every setting a user would otherwise choose, for one recording modality (spec §6).

Presets are data: one JSON file per preset in this folder, read with the standard library
into the frozen :class:`Preset` dataclass. Each value is justified by the literature keys of
``citation`` or marked as a choice of the specification (``choices``: field -> reason).
This module imports nothing from the package but the leaves ``errors`` and ``_plugins``
(spec §4.2), so every module may use it. ``PRESETS`` adds the presets of the entry-point
group ``evoparam.presets`` on first access (plugins, spec §10.8).

>>> from evoparam import presets
>>> presets.get("ccep").fit_window
(0.009, 0.5)
>>> p = presets.derive("ccep", fit_window=(0.012, 0.5))
>>> p.parent, p.overrides["fit_window"]
('ccep', (0.012, 0.5))
"""

import copy
import dataclasses
import json
from importlib import resources
from typing import Any, cast

from evoparam import _plugins
from evoparam.errors import SettingsError
from evoparam.utils._optional import require

__all__ = [
    "Preset",
    "get",
    "list",
    "derive",
    "targeted",
    "register",
    "to_json",
    "from_json",
    "to_yaml",
    "from_yaml",
    "PRESETS",
    "fit_fields",
    "methods_note",
]

_list = list  # the public function list() shadows the builtin in this module


@dataclasses.dataclass(frozen=True)
class Preset:
    """One preset (spec §4.3.4, §6.1). Tuples for sequences, so that a preset is immutable.

    The behaviours of spec §6.2 that the preset pipeline
    (:func:`evoparam.reports.run_pipeline`, ``evoparam fit``) applies: ``reference_sensitivity``
    (``ccep_seeg``: :func:`evoparam.prep.reference_sensitivity` is run and its kappa report
    attached), ``trial_covariates`` (``rodent_ep``, ``primate_ep``: ``("state",)``, the
    covariates of the trial fits) and ``train_baseline`` (``erna``: the baseline is the last
    ``train_baseline`` seconds before each train, :func:`evoparam.prep.train_baseline`).
    """

    name: str
    modality: str
    species: tuple[str, ...]
    epoch: tuple[float, ...]
    baseline: tuple[float, ...] | None
    artifact_window: tuple[float, ...] | None
    artifact_method: str | None
    reference: str | None
    fit_window: tuple[float, ...]
    shapes: tuple[str, ...]
    background: str
    max_components: int
    min_fwhm: float
    max_fwhm: float
    polarity: str
    existence_method: str
    existence_alpha: float
    component_labels: dict[str, tuple[float, ...]]
    trial_fit: bool
    citation: tuple[str, ...]
    version: str
    validation: str
    label_polarity: dict[str, int] = dataclasses.field(default_factory=dict)
    trial_mode: str | None = None
    selection: str = "bic"
    # the fall of BIC a component needs (spec §5.5.4), or "calibrated": the threshold of the
    # engine calibrated on each fit's noise model (evoparam.models._calibration)
    delta: float | str = 10.0
    uncertainty: str = "jacobian"
    choices: dict[str, str] = dataclasses.field(default_factory=dict)
    notes: dict[str, str] = dataclasses.field(default_factory=dict)
    fit_extras: dict[str, Any] = dataclasses.field(default_factory=dict)
    reference_sensitivity: bool = False
    trial_covariates: tuple[str, ...] = ()
    train_baseline: float | None = None
    parent: str | None = None
    overrides: dict[str, Any] = dataclasses.field(default_factory=dict)
    # name -> {"window": [start, end] (s), "polarity": "negative" or "positive", "k": depth in
    # units of the fit's noise SD}: the preset's decision that the response has this peak, read
    # on the fitted curve (:func:`evoparam.models.curve_detection`, ccep 1.2: the N1)
    detections: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check and normalise the fields."""
        for name in ("species", "shapes", "citation", "trial_covariates"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("epoch", "baseline", "artifact_window", "fit_window"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(float(v) for v in value))
        object.__setattr__(
            self,
            "component_labels",
            {k: tuple(float(x) for x in v) for k, v in self.component_labels.items()},
        )
        names = ("label_polarity", "choices", "notes", "fit_extras", "overrides", "detections")
        for name in names:
            object.__setattr__(self, name, dict(getattr(self, name)))
        if not self.fit_window[1] > self.fit_window[0]:
            raise SettingsError(f"preset {self.name!r}: fit_window must increase.")
        if not 0 < self.min_fwhm < self.max_fwhm:
            raise SettingsError(f"preset {self.name!r}: need 0 < min_fwhm < max_fwhm.")
        if self.train_baseline is not None:
            object.__setattr__(self, "train_baseline", float(self.train_baseline))
            if not self.train_baseline > 0:
                raise SettingsError(
                    f"preset {self.name!r}: train_baseline must be a positive duration (s)."
                )
        if isinstance(self.delta, str) and self.delta != "calibrated":
            raise SettingsError(f"preset {self.name!r}: delta must be a number or 'calibrated'.")
        if self.polarity not in ("both", "negative", "positive"):
            raise SettingsError(
                f"preset {self.name!r}: polarity must be both, negative or positive."
            )

    @property
    def fields(self) -> tuple[str, ...]:
        """Names of the fields of the preset."""
        return tuple(f.name for f in dataclasses.fields(self))

    def to_dict(self) -> dict[str, Any]:
        """Return the preset as a dict of plain values."""
        out = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, tuple):
                value = _list(value)
            elif isinstance(value, dict):
                value = {k: _list(v) if isinstance(v, tuple) else v for k, v in value.items()}
            out[f.name] = value
        return out


def _load_builtin() -> dict[str, "Preset"]:
    out = {}
    folder = resources.files("evoparam.presets")
    for entry in sorted(folder.iterdir(), key=lambda e: e.name):
        if entry.name.endswith(".json"):
            preset = from_json(entry.read_text("utf-8"))
            out[preset.name] = preset
    return out


def get(name: "str | Preset") -> "Preset":
    """Return the preset ``name`` (a :class:`Preset` is returned as it is).

    Raises
    ------
    SettingsError
        Unknown name; the message lists the presets.
    """
    if isinstance(name, Preset):
        return name
    if name not in PRESETS:
        raise SettingsError(f"Unknown preset {name!r}; known: {sorted(PRESETS)}.")
    return cast("Preset", PRESETS[name])


def list() -> Any:  # noqa: A001  (the name of the spec's API)
    """Table of the presets: name, modality, species, citation keys, version, validation."""
    require("pandas", extra="data", caller="evoparam.presets.list")
    import pandas as pd

    rows = [
        {
            "name": p.name,
            "modality": p.modality,
            "species": ", ".join(p.species),
            "citation": ", ".join(p.citation),
            "version": p.version,
            "validation": p.validation,
        }
        for p in PRESETS.values()
    ]
    return pd.DataFrame(
        rows, columns=["name", "modality", "species", "citation", "version", "validation"]
    )


def derive(base: "str | Preset", **overrides: Any) -> "Preset":
    """Return a new frozen preset: ``base`` with ``overrides``.

    The parent is recorded and never changed.

    Raises
    ------
    SettingsError
        An override that is not a preset field; the message lists the fields.
    """
    parent = get(base)
    known = {f.name for f in dataclasses.fields(Preset)} - {"parent", "overrides"}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise SettingsError(f"Unknown preset field(s) {unknown}; fields: {sorted(known)}.")
    values = copy.deepcopy(parent.to_dict())
    values.update(copy.deepcopy(overrides))
    values["parent"] = parent.name
    record = dict(parent.overrides)
    record.update(overrides)
    values["overrides"] = record
    if "name" not in overrides:
        values["name"] = parent.name
    return Preset(**values)


def targeted(
    base: "str | Preset",
    label: str,
    window: tuple[float, float],
    polarity: str,
    *,
    margin: float = 0.05,
    name: str | None = None,
    n_components: int = 1,
    background: str = "linear",
) -> "Preset":
    """Return a preset that measures one expected component.

    One component of the given polarity, fitted in ``window`` widened by ``margin`` on each side,
    on a linear background (``selection="fixed"``). Every waveform gets a value; whether the
    component is there is the existence test's answer, not the model selection's.

    On ERP CORE (40 participants, parent waves at the published channels, measurement
    window +/- 50 ms) the amplitudes measured so are as reliable as or more reliable than
    the classical mean amplitude (split-half, Spearman-Brown): P3 0.89 (classical 0.80),
    MMN 0.79 (0.66), N170 0.95 (0.97), but N400 0.76 (0.85); see CHANGELOG.

    Parameters
    ----------
    base : str or Preset
    label : str
        Component label, for example ``"P3"``.
    window : (float, float)
        Measurement window (s), for example the published one.
    polarity : {"negative", "positive"}
    margin : float
        Widening of the window on each side (s).
    name : str, optional
        Default ``"<base>_<label>"``.
    n_components : int
        Components of that polarity fitted in the window (``label``, ``label_b``, ...); with
        more than one, the component is their sum (a P3 made of overlapping parts). The
        ERP CORE validation (REQ-274) uses 3 on difference waves.
    background : str
        Background under the component (``"none"`` for baseline-corrected difference
        waves).
    """
    if polarity not in ("negative", "positive"):
        raise SettingsError("targeted: polarity must be 'negative' or 'positive'.")
    if int(n_components) < 1:
        raise SettingsError("targeted: n_components must be at least 1.")
    parent = get(base)
    lo, hi = float(window[0]) - margin, float(window[1]) + margin
    extras = dict(parent.fit_extras)
    extras["n_components"] = int(n_components)
    return derive(
        parent,
        name=name or f"{parent.name}_{label}",
        fit_window=(lo, hi),
        polarity=polarity,
        selection="fixed",
        max_components=int(n_components),
        background=background,
        component_labels={label: (lo, hi)},
        label_polarity={label: -1 if polarity == "negative" else 1},
        fit_extras=extras,
    )


def register(preset: "Preset", *, overwrite: bool = False) -> None:
    """Add a preset to the registry (a built-in name only with ``overwrite=True``)."""
    if not isinstance(preset, Preset):
        raise SettingsError("register expects a Preset (see derive or from_json).")
    if preset.name in PRESETS and not overwrite:
        raise SettingsError(f"Preset {preset.name!r} exists; pass overwrite=True to replace it.")
    PRESETS[preset.name] = preset


def to_json(preset: "str | Preset") -> str:
    """JSON text of a preset (the format of the files of this folder)."""
    return json.dumps(get(preset).to_dict(), indent=2)


def from_json(text: str) -> "Preset":
    """Preset from JSON text."""
    values = json.loads(text)
    known = {f.name for f in dataclasses.fields(Preset)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise SettingsError(f"Unknown preset field(s) {unknown}.")
    return Preset(**values)


def _yaml() -> Any:
    try:
        import yaml
    except ImportError as err:
        raise ImportError(
            "YAML presets need PyYAML: pip install PyYAML (JSON works without "
            "it: presets.to_json / presets.from_json)."
        ) from err
    return yaml


def to_yaml(preset: "str | Preset") -> str:
    """YAML text of a preset (needs PyYAML)."""
    return cast("str", _yaml().safe_dump(get(preset).to_dict(), sort_keys=False))


def from_yaml(text: str) -> "Preset":
    """Preset from YAML text (needs PyYAML)."""
    return Preset(**_yaml().safe_load(text))


def fit_fields(preset: "str | Preset") -> dict[str, Any]:
    """Return the settings of the fitting engine that a preset fixes (``FitSettings`` names)."""
    p = get(preset)
    windows = {
        label: (lo, hi, int(p.label_polarity.get(label, 0)))
        for label, (lo, hi) in p.component_labels.items()
    }
    out = {
        "fit_window": p.fit_window,
        "baseline": p.baseline,
        "shapes": p.shapes,
        "background": p.background,
        "max_components": p.max_components,
        "min_fwhm": p.min_fwhm,
        "max_fwhm": p.max_fwhm,
        "polarity": p.polarity,
        "selection": p.selection,
        "delta": p.delta,
        # the presets compare shapes on the quick fits of the forward search (spec §5.5.3
        # step 4), as when they were validated
        "shape_choice": "bic",
        "resolution_check": False,
        "uncertainty": p.uncertainty,
        "windows": windows,
    }
    out.update(p.fit_extras)
    return out


def methods_note(preset: "str | Preset") -> str:
    """Return the preset's part of a methods text.

    Name, version, parent and overrides, the validation status (spec REQ-279) and the preset's
    caveat for the methods (``notes["methods"]``, for example the sensory confound of ``tep``,
    REQ-275).
    """
    p = get(preset)
    text = f"preset '{p.name}' v{p.version}"
    if p.parent:
        changed = ", ".join(f"{k}={v}" for k, v in p.overrides.items())
        text += f" (derived from '{p.parent}' with {changed})"
    if p.validation == "simulation_only":
        text += (
            "; this preset is validated on simulations only (no public dataset validates it yet)"
        )
    else:
        text += f"; validation: {p.validation}"
    if p.notes.get("methods"):
        text += f"; {p.notes['methods']}"
    return text


def _plugin_preset(name: str, obj: object, origin: str) -> "Preset":
    """Check a preset of the entry-point group ``evoparam.presets`` (spec §10.8)."""
    if not isinstance(obj, Preset):
        raise SettingsError(
            f"plugin preset {name!r} of {origin} is a {type(obj).__name__}, not a Preset."
        )
    if obj.name != name:
        raise SettingsError(
            f"plugin preset {name!r} of {origin}: its name is {obj.name!r}; "
            f"the entry-point name must be the same."
        )
    return obj


PRESETS = _plugins.LazyRegistry("evoparam.presets", _load_builtin(), _plugin_preset)
