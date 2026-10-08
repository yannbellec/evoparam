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

import evoparam._plugins as _plugins
from evoparam.errors import SettingsError

__all__ = ["Preset", "get", "list", "derive", "register", "to_json", "from_json", "to_yaml",
           "from_yaml", "PRESETS", "fit_fields", "methods_note"]

_list = list  # the public function list() shadows the builtin in this module


@dataclasses.dataclass(frozen=True)
class Preset:
    """One preset (spec §4.3.4, §6.1). Tuples for sequences, so that a preset is immutable."""

    name: str
    modality: str
    species: tuple
    epoch: tuple
    baseline: tuple | None
    artifact_window: tuple | None
    artifact_method: str | None
    reference: str | None
    fit_window: tuple
    shapes: tuple
    background: str
    max_components: int
    min_fwhm: float
    max_fwhm: float
    polarity: str
    existence_method: str
    existence_alpha: float
    component_labels: dict
    trial_fit: bool
    citation: tuple
    version: str
    validation: str
    label_polarity: dict = dataclasses.field(default_factory=dict)
    trial_mode: str | None = None
    selection: str = "bic"
    delta: float = 10.0
    uncertainty: str = "jacobian"
    choices: dict = dataclasses.field(default_factory=dict)
    notes: dict = dataclasses.field(default_factory=dict)
    fit_extras: dict = dataclasses.field(default_factory=dict)
    parent: str | None = None
    overrides: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        for name in ("species", "shapes", "citation"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("epoch", "baseline", "artifact_window", "fit_window"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(float(v) for v in value))
        object.__setattr__(self, "component_labels",
                           {k: tuple(float(x) for x in v) for k, v in
                            self.component_labels.items()})
        for name in ("label_polarity", "choices", "notes", "fit_extras", "overrides"):
            object.__setattr__(self, name, dict(getattr(self, name)))
        if not self.fit_window[1] > self.fit_window[0]:
            raise SettingsError(f"preset {self.name!r}: fit_window must increase.")
        if not 0 < self.min_fwhm < self.max_fwhm:
            raise SettingsError(f"preset {self.name!r}: need 0 < min_fwhm < max_fwhm.")
        if self.polarity not in ("both", "negative", "positive"):
            raise SettingsError(f"preset {self.name!r}: polarity must be both, negative or "
                                f"positive.")

    @property
    def fields(self):
        return tuple(f.name for f in dataclasses.fields(self))

    def to_dict(self):
        out = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, tuple):
                value = _list(value)
            elif isinstance(value, dict):
                value = {k: _list(v) if isinstance(v, tuple) else v for k, v in value.items()}
            out[f.name] = value
        return out


def _load_builtin():
    out = {}
    folder = resources.files("evoparam.presets")
    for entry in sorted(folder.iterdir(), key=lambda e: e.name):
        if entry.name.endswith(".json"):
            preset = from_json(entry.read_text("utf-8"))
            out[preset.name] = preset
    return out


def get(name):
    """The preset ``name`` (a :class:`Preset` is returned as it is).

    Raises
    ------
    SettingsError
        Unknown name; the message lists the presets.
    """

    if isinstance(name, Preset):
        return name
    if name not in PRESETS:
        raise SettingsError(f"Unknown preset {name!r}; known: {sorted(PRESETS)}.")
    return PRESETS[name]


def list():  # noqa: A001 (the name of the spec's API)
    """Table of the presets: name, modality, species, citation keys, version, validation."""

    import pandas as pd

    rows = [{"name": p.name, "modality": p.modality, "species": ", ".join(p.species),
             "citation": ", ".join(p.citation), "version": p.version,
             "validation": p.validation} for p in PRESETS.values()]
    return pd.DataFrame(rows, columns=["name", "modality", "species", "citation", "version",
                                       "validation"])


def derive(base, **overrides):
    """A new frozen preset: ``base`` with ``overrides``; the parent is recorded and never
    changed.

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


def register(preset, *, overwrite=False):
    """Add a preset to the registry (a built-in name only with ``overwrite=True``)."""

    if not isinstance(preset, Preset):
        raise SettingsError("register expects a Preset (see derive or from_json).")
    if preset.name in PRESETS and not overwrite:
        raise SettingsError(f"Preset {preset.name!r} exists; pass overwrite=True to replace it.")
    PRESETS[preset.name] = preset


def to_json(preset):
    """JSON text of a preset (the format of the files of this folder)."""

    return json.dumps(get(preset).to_dict(), indent=2)


def from_json(text):
    """Preset from JSON text."""

    values = json.loads(text)
    known = {f.name for f in dataclasses.fields(Preset)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise SettingsError(f"Unknown preset field(s) {unknown}.")
    return Preset(**values)


def _yaml():
    try:
        import yaml
    except ImportError as err:
        raise ImportError("YAML presets need PyYAML: pip install PyYAML (JSON works without "
                          "it: presets.to_json / presets.from_json).") from err
    return yaml


def to_yaml(preset):
    """YAML text of a preset (needs PyYAML)."""

    return _yaml().safe_dump(get(preset).to_dict(), sort_keys=False)


def from_yaml(text):
    """Preset from YAML text (needs PyYAML)."""

    return Preset(**_yaml().safe_load(text))


def fit_fields(preset):
    """The settings of the fitting engine that a preset fixes (``FitSettings`` names)."""

    p = get(preset)
    windows = {label: (lo, hi, int(p.label_polarity.get(label, 0)))
               for label, (lo, hi) in p.component_labels.items()}
    out = {"fit_window": p.fit_window, "baseline": p.baseline, "shapes": p.shapes,
           "background": p.background, "max_components": p.max_components,
           "min_fwhm": p.min_fwhm, "max_fwhm": p.max_fwhm, "polarity": p.polarity,
           "selection": p.selection, "delta": p.delta, "uncertainty": p.uncertainty,
           "windows": windows}
    out.update(p.fit_extras)
    return out


def methods_note(preset):
    """The preset's part of a methods text: name, version, parent and overrides, and the
    validation status (spec REQ-279)."""

    p = get(preset)
    text = f"preset '{p.name}' v{p.version}"
    if p.parent:
        changed = ", ".join(f"{k}={v}" for k, v in p.overrides.items())
        text += f" (derived from '{p.parent}' with {changed})"
    if p.validation == "simulation_only":
        text += ("; this preset is validated on simulations only (no public dataset "
                 "validates it yet)")
    else:
        text += f"; validation: {p.validation}"
    return text


def _plugin_preset(name, obj, origin):
    """Check a preset of the entry-point group ``evoparam.presets`` (spec §10.8)."""

    if not isinstance(obj, Preset):
        raise SettingsError(f"plugin preset {name!r} of {origin} is a {type(obj).__name__}, "
                            f"not a Preset.")
    if obj.name != name:
        raise SettingsError(f"plugin preset {name!r} of {origin}: its name is {obj.name!r}; "
                            f"the entry-point name must be the same.")
    return obj


PRESETS = _plugins.LazyRegistry("evoparam.presets", _load_builtin(), _plugin_preset)
