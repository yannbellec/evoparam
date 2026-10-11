# SPDX-License-Identifier: Apache-2.0
"""Harmonise stimulation events into the standard metadata columns (spec §10.2, REQ-313).

In ``evoparam.data`` so that every reader uses it (``read_bids``, ``read_nwb``,
``TrialSet.from_mne``); ``evoparam.io`` re-exports it.

BIDS describes electrical stimulation only through example ``events.tsv``
columns and a free-text sidecar field, so deposits encode it differently. This
module maps the source columns to the standard columns of the specification
(Table 4.1), parses units, and reports what was mapped, inferred or missing.
"""

import copy
import math
import re
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np

from evoparam.errors import DataError

__all__ = ["harmonize_stim_events", "HarmonizationReport", "validate_stimulation_events",
           "ValidationReport", "STIM_FIELDS", "POLARITIES", "normalize_site"]

# Target column -> recognised source columns, first match wins (case-insensitive).
# The target name itself comes last: an already harmonised table keeps its values.
SOURCES = {
    "stim_site": ("electrical_stimulation_site", "stim_site", "site", "electrodes_involved_onset"),
    "stim_current_mA": ("electrical_stimulation_current", "stim_current", "current",
                        "stim_current_mA"),
    "stim_pulse_width_us": ("electrical_stimulation_pulsewidth",
                            "electrical_stimulation_pulse_width", "pulse_width", "pulsewidth",
                            "stim_pulse_width_us"),
    "stim_polarity": ("electrical_stimulation_type", "stim_type", "polarity", "stim_polarity"),
    "stim_frequency_hz": ("electrical_stimulation_frequency", "frequency", "stim_frequency_hz"),
    "stim_n_pulses": ("n_pulses", "pulse_count", "stim_n_pulses"),
    "condition": ("trial_type", "condition"),
}

# Fields checked by the validator, in report order.
STIM_FIELDS = ("stim_site", "stim_current_mA", "stim_pulse_width_us", "stim_polarity",
               "stim_frequency_hz", "stim_n_pulses")

POLARITIES = ("monophasic_cathodic", "monophasic_anodic", "biphasic_cathodic_first",
              "biphasic_anodic_first", "alternating")

# Unit name (lower case) -> factor to the target unit.
_CURRENT = {"a": 1e3, "amp": 1e3, "ampere": 1e3, "amperes": 1e3, "ma": 1.0, "milliampere": 1.0,
            "milliamperes": 1.0, "ua": 1e-3, "µa": 1e-3, "μa": 1e-3, "microampere": 1e-3,
            "microamperes": 1e-3}
_WIDTH = {"s": 1e6, "sec": 1e6, "second": 1e6, "seconds": 1e6, "ms": 1e3, "msec": 1e3,
          "millisecond": 1e3, "milliseconds": 1e3, "us": 1.0, "µs": 1.0, "μs": 1.0, "usec": 1.0,
          "microsecond": 1.0, "microseconds": 1.0, "ns": 1e-3}
_FREQUENCY = {"hz": 1.0, "khz": 1e3}

# Quantity target -> (unit table, target unit, unit of unit-less values below the threshold,
# threshold, unit of the other unit-less values).
_QUANTITIES = {
    "stim_current_mA": (_CURRENT, "mA", "A", 0.1, "mA"),
    "stim_pulse_width_us": (_WIDTH, "us", "s", 0.01, "us"),
    "stim_frequency_hz": (_FREQUENCY, "Hz", None, None, "Hz"),
}

UNITS = {"onset": "s", "duration": "s", "stim_current_mA": "mA", "stim_pulse_width_us": "us",
         "stim_frequency_hz": "Hz", "stim_charge_uC": "uC"}

_MISSING = {"", "n/a", "na", "nan", "none", "null"}
_NUMBER = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([^\d\s].*?)?\s*$")
_PAIR = re.compile(r"^[A-Za-z]+\d+-[A-Za-z]+\d+$")


###################################################################################################
## Reports

@dataclass
class HarmonizationReport:
    """What :func:`harmonize_stim_events` did.

    Attributes
    ----------
    mapping : dict
        Target column -> source column used (None when no source was found).
    units : dict
        Target column -> how the unit was found: ``"sidecar:<unit>"``, ``"inline"``,
        ``"inferred:<unit>"``, ``"native"`` or ``"mixed"``.
    warnings : list of str
        One line per issue, starting with a code: ``unit_inferred``, ``unit_unknown``,
        ``unparsable``, ``site_from_trial_type``, ``polarity_unknown``,
        ``n_pulses_not_integer``.
    missing : list of str
        Target columns without a source column.
    n_rows : int
        Rows in the events table.
    user_mapping : dict
        The ``mapping`` argument, as given.
    """

    mapping: dict = field(default_factory=dict)
    units: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    n_rows: int = 0
    user_mapping: dict = field(default_factory=dict)

    def to_dict(self):
        """Return a JSON-serialisable dictionary."""

        return copy.deepcopy(asdict(self))


@dataclass
class ValidationReport:
    """What :func:`validate_stimulation_events` found in one events table.

    Attributes
    ----------
    missing : list of str
        Standard stimulation fields absent from every stimulation row.
    warnings : list of str
        Partially filled fields and harmonisation warnings.
    n_rows : int
        Stimulation rows checked.
    n_sites : int
        Distinct stimulation sites.
    sites : list of str
        The distinct sites, sorted.
    """

    missing: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    n_rows: int = 0
    n_sites: int = 0
    sites: list = field(default_factory=list)

    @property
    def complete(self):
        """True when no standard field is missing."""

        return not self.missing

    def to_dict(self):
        """Return a JSON-serialisable dictionary."""

        return copy.deepcopy(asdict(self))


###################################################################################################
## Helpers

def _pandas(caller):
    try:
        import pandas
    except ImportError as err:
        raise ImportError(f"{caller} needs pandas: pip install \"evoparam[data]\".") from err
    return pandas


def _is_missing(value):
    if value is None or type(value).__name__ in ("NAType", "NaTType"):
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in _MISSING:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _find_column(columns, candidates):
    lower = {}
    for column in columns:
        lower.setdefault(str(column).lower(), column)
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _sidecar_unit(sidecar, column):
    """``Units`` of a column in the events.json sidecar (keys matched case-insensitively)."""

    for key, value in (sidecar or {}).items():
        if str(key).lower() == str(column).lower() and isinstance(value, dict):
            for sub, unit in value.items():
                if str(sub).lower() == "units" and isinstance(unit, str) and unit.strip():
                    return unit.strip()
    return None


def _unit_factor(unit, table):
    if unit is None:
        return None
    return table.get(unit.strip().lower().replace("μ", "µ"), table.get(unit.strip().lower()))


def normalize_site(value):
    """Contact pair in the form ``"A-B"``; separators ``-``, ``_``, ``,`` and spaces become ``-``.

    Missing values give ``"n/a"``.
    """

    if _is_missing(value):
        return "n/a"
    parts = [part for part in re.split(r"[-_,\s]+", str(value).strip()) if part]
    return "-".join(parts) if parts else "n/a"


def _parse_quantity(values, column, target, sidecar, report):
    """Numbers in the target unit, from numbers or strings such as "4.0 mA" or "1ms"."""

    table, target_unit, small_unit, threshold, large_unit = _QUANTITIES[target]
    native = column.lower() == target.lower()
    side_unit = None if native else _sidecar_unit(sidecar, column)
    side_factor = _unit_factor(side_unit, table)
    if side_unit is not None and side_factor is None:
        report.warnings.append(f"unit_unknown: {column} has Units {side_unit!r} in the sidecar, "
                               f"which is not a unit of {target}; values left as missing")
        report.units[target] = f"sidecar:{side_unit}"
        return np.full(len(values), np.nan)

    out = np.full(len(values), np.nan)
    unitless = []                                     # positions of numbers without a unit
    inline = False
    for pos, value in enumerate(values):
        if _is_missing(value):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            number, unit = float(value), None
        else:
            match = _NUMBER.match(str(value))
            if not match:
                report.warnings.append(f"unparsable: {column} value {value!r} is not a number "
                                       "with a unit; left as missing")
                continue
            number, unit = float(match.group(1)), match.group(2)
        if unit:
            factor = _unit_factor(unit, table)
            if factor is None:
                report.warnings.append(f"unit_unknown: {column} value {value!r} has an unknown "
                                       f"unit; left as missing")
                continue
            out[pos] = number * factor
            inline = True
        elif native:
            out[pos] = number
        elif side_factor is not None:
            out[pos] = number * side_factor
        else:
            unitless.append(pos)
            out[pos] = number

    if unitless:
        numbers = np.abs(out[unitless])
        if small_unit is not None and np.all(numbers < threshold):
            unit = small_unit
        else:
            unit = large_unit
        out[unitless] = out[unitless] * table[unit.lower()]
        if unit != target_unit or small_unit is not None:
            rule = (f"values < {threshold} are read as {small_unit}, others as {large_unit}"
                    if small_unit is not None else f"values are read as {large_unit}")
            report.warnings.append(f"unit_inferred: {column} has no unit ({rule}); read as "
                                   f"{unit} and converted to {target_unit}")
        report.units[target] = f"inferred:{unit}"
        if inline:
            report.units[target] = "mixed"
    elif native:
        report.units[target] = "native"
    elif side_unit is not None:
        report.units[target] = f"sidecar:{side_unit}"
    elif inline:
        report.units[target] = "inline"
    return out


def _polarity_kind(text):
    text = text.lower()
    if "alternat" in text:
        return "alternating"
    if "biphasic" in text or "bi-phasic" in text:
        return "biphasic"
    if "monophasic" in text or "mono-phasic" in text:
        return "monophasic"
    return None


def _polarity_order(text):
    """'cathodic' or 'anodic' (the one stated first), or None."""

    text = text.lower()
    cathodic, anodic = text.find("cathod"), text.find("anod")
    if cathodic >= 0 and (anodic < 0 or cathodic < anodic):
        return "cathodic"
    if anodic >= 0:
        return "anodic"
    return None


def _polarity(value, parameters):
    """One of POLARITIES, or None when the value does not state the phase order."""

    text = str(value).strip()
    if text.lower() in POLARITIES:
        return text.lower()
    kind = _polarity_kind(text) or (_polarity_kind(parameters) if parameters else None)
    if kind == "alternating":
        return kind
    order = _polarity_order(text) or (_polarity_order(parameters) if parameters else None)
    if kind is None or order is None:
        return None
    return f"monophasic_{order}" if kind == "monophasic" else f"biphasic_{order}_first"


###################################################################################################
## Public functions

def harmonize_stim_events(events, sidecar=None, *, mapping=None):
    """Map stimulation columns of an events table to the standard metadata columns.

    Recognised source columns (first match wins, case-insensitive; spec §10.2):

    ==========================  =============================================================
    target                      sources
    ==========================  =============================================================
    ``stim_site``               electrical_stimulation_site, stim_site, site,
                                electrodes_involved_onset (rows whose trial_type contains
                                "stim" only)
    ``stim_current_mA``         electrical_stimulation_current, stim_current, current
    ``stim_pulse_width_us``     electrical_stimulation_pulsewidth,
                                electrical_stimulation_pulse_width, pulse_width, pulsewidth
    ``stim_polarity``           electrical_stimulation_type, stim_type, polarity
    ``stim_frequency_hz``       electrical_stimulation_frequency, frequency
    ``stim_n_pulses``           n_pulses, pulse_count
    ``condition``               trial_type
    ==========================  =============================================================

    Units come from the sidecar (``events.json[column]["Units"]``), else from the
    value itself ("4.0 mA", "1000 µs", "1ms"), else they are inferred: currents
    below 0.1 are read as A and pulse widths below 0.01 as s (warning
    ``unit_inferred``). ``n/a`` gives NaN. Sites are written ``"A-B"``; without a
    site column, ``trial_type`` values that look like a contact pair (``"PT01-PT02"``)
    fill ``stim_site`` (warning ``site_from_trial_type``). Polarity is mapped to the
    vocabulary of Table 4.1 when the phase order is stated in the value or in the
    sidecar field ``ElectricalStimulationParameters``, else ``"unknown"``, with the raw
    value kept in ``evoparam_stim_type_raw``. ``stim_charge_uC`` is derived from
    current and pulse width.

    Parameters
    ----------
    events : pandas.DataFrame
        An events table (for example a BIDS ``*_events.tsv``).
    sidecar : dict, optional
        The ``events.json`` sidecar; may also hold ``ElectricalStimulationParameters``.
    mapping : dict, optional
        ``{source column: target column}``, overriding the table above.

    Returns
    -------
    harmonized : pandas.DataFrame
        A copy of ``events`` with the standard columns added (original columns kept
        under their own names, ``trial_type`` unchanged). Units of the new columns are
        in ``harmonized.attrs["units"]``.
    report : HarmonizationReport

    Raises
    ------
    DataError
        If ``events`` is not a DataFrame, or ``mapping`` names an unknown column or target.
    """

    pd = _pandas("evoparam.io.harmonize_stim_events")
    if not isinstance(events, pd.DataFrame):
        raise DataError("harmonize_stim_events: events must be a pandas DataFrame.")
    sidecar = sidecar or {}
    mapping = dict(mapping or {})
    out = events.copy().reset_index(drop=True)
    n_rows = len(out)
    report = HarmonizationReport(n_rows=n_rows, user_mapping=dict(mapping))

    chosen = {}
    for source, target in mapping.items():
        if target not in SOURCES:
            raise DataError(f"mapping: unknown target column {target!r}; "
                            f"expected one of {sorted(SOURCES)}.")
        if source not in events.columns:
            raise DataError(f"mapping: column {source!r} is not in the events table.")
        chosen[target] = source
    for target, candidates in SOURCES.items():
        if target not in chosen:
            chosen[target] = _find_column(events.columns, candidates)
    report.mapping = {target: chosen[target] for target in SOURCES}
    report.missing = [target for target in SOURCES if chosen[target] is None]

    trial_type = out["trial_type"] if "trial_type" in out.columns else None
    parameters = sidecar.get("ElectricalStimulationParameters")
    parameters = parameters if isinstance(parameters, str) else None

    # Site
    column = chosen["stim_site"]
    if column is not None:
        values = out[column].tolist()
        if column.lower() == "electrodes_involved_onset" and "stim_site" not in mapping.values():
            is_stim = ([False] * n_rows if trial_type is None else
                       [not _is_missing(t) and "stim" in str(t).lower() for t in trial_type])
            values = [value if keep else None for value, keep in zip(values, is_stim)]
        sites = [normalize_site(value) for value in values]
    else:
        sites = ["n/a"] * n_rows
        if trial_type is not None:
            pairs = [not _is_missing(t) and bool(_PAIR.match(str(t).strip())) for t in trial_type]
            if any(pairs):
                sites = [str(t).strip() if pair else "n/a" for t, pair in zip(trial_type, pairs)]
                report.mapping["stim_site"] = "trial_type"
                report.missing.remove("stim_site")
                report.warnings.append(f"site_from_trial_type: no site column; {sum(pairs)} "
                                       "trial_type values that look like a contact pair "
                                       "were used as stim_site")
    out["stim_site"] = pd.Series(sites)

    # Quantities
    for target in ("stim_current_mA", "stim_pulse_width_us", "stim_frequency_hz"):
        column = chosen[target]
        if column is None:
            out[target] = np.full(n_rows, np.nan)
        else:
            out[target] = _parse_quantity(out[column].tolist(), column, target, sidecar, report)

    # Pulse count: nullable integers (missing when not stated)
    column = chosen["stim_n_pulses"]
    counts = [None] * n_rows
    if column is not None:
        counts = _parse_count(out[column].tolist(), column, report)
    out["stim_n_pulses"] = pd.array(counts, dtype="Int64")

    # Polarity
    column = chosen["stim_polarity"]
    raw = ["n/a"] * n_rows if column is None else [
        "n/a" if _is_missing(value) else str(value).strip() for value in out[column].tolist()]
    polarity, unknown = [], set()
    for value in raw:
        if value == "n/a":
            polarity.append("n/a")
            continue
        mapped = _polarity(value, parameters)
        if mapped is None:
            unknown.add(value)
        polarity.append(mapped or "unknown")
    out["stim_polarity"] = pd.Series(polarity)
    if unknown:
        out["evoparam_stim_type_raw"] = pd.Series(raw)
        report.warnings.append("polarity_unknown: the phase order is not stated for "
                               f"{sorted(unknown)}; stim_polarity set to 'unknown', raw values "
                               "kept in evoparam_stim_type_raw")

    # Charge per phase (µC) = current (mA) × pulse width (µs) / 1000
    out["stim_charge_uC"] = out["stim_current_mA"] * out["stim_pulse_width_us"] / 1000.0

    # Condition
    column = chosen["condition"]
    values = out[column].tolist() if column else [None] * n_rows
    out["condition"] = pd.Series(["n/a" if _is_missing(value) else str(value) for value in values])

    units = dict(out.attrs.get("units", {}))
    units.update({key: value for key, value in UNITS.items() if key in out.columns})
    out.attrs["units"] = units
    return out, report


def _parse_count(values, column, report):
    out = []
    for value in values:
        if _is_missing(value):
            out.append(None)
            continue
        try:
            number = float(value) if isinstance(value, (int, float)) else float(str(value).strip())
        except ValueError:
            report.warnings.append(f"unparsable: {column} value {value!r} is not a count; "
                                   "left as missing")
            out.append(None)
            continue
        if number != int(number):
            report.warnings.append(f"n_pulses_not_integer: {column} value {value!r} is not an "
                                   "integer; left as missing")
            out.append(None)
            continue
        out.append(int(number))
    return out


def _stimulation_rows(events):
    """Rows that describe stimulations: trial_type 'electrical_stimulation', else containing
    'stim', else every row."""

    if "trial_type" not in events.columns:
        return np.ones(len(events), dtype=bool)
    types = [str(value).strip().lower() for value in events["trial_type"].tolist()]
    exact = np.array([value == "electrical_stimulation" for value in types], dtype=bool)
    if exact.any():
        return exact
    loose = np.array(["stim" in value for value in types], dtype=bool)
    return loose if loose.any() else np.ones(len(events), dtype=bool)


def validate_stimulation_events(events, sidecar=None):
    """Check which standard stimulation fields an events table provides.

    The table is harmonised first (:func:`harmonize_stim_events`) unless it already
    holds every standard column. Missing fields are reported, never raised.

    Parameters
    ----------
    events : pandas.DataFrame
        One events table, raw or harmonised.
    sidecar : dict, optional
        Its ``events.json`` sidecar.

    Returns
    -------
    report : ValidationReport
        ``missing`` lists the fields of :data:`STIM_FIELDS` absent from every
        stimulation row; ``warnings`` lists partially filled fields and the
        harmonisation warnings; ``sites`` the distinct stimulation sites.
    """

    pd = _pandas("evoparam.io.validate_stimulation_events")
    if not isinstance(events, pd.DataFrame):
        raise DataError("validate_stimulation_events: events must be a pandas DataFrame.")
    messages = []
    if not all(name in events.columns for name in STIM_FIELDS):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            events, harmonization = harmonize_stim_events(events, sidecar)
        messages.extend(harmonization.warnings)
    rows = events.iloc[np.flatnonzero(_stimulation_rows(events))]
    n_rows = len(rows)

    missing = []
    for name in STIM_FIELDS:
        values = rows[name].tolist()
        if name == "stim_polarity":
            present = [value not in ("n/a", "unknown") and not _is_missing(value)
                       for value in values]
        elif name == "stim_site":
            present = [value != "n/a" and not _is_missing(value) for value in values]
        else:
            present = [not _is_missing(value) for value in values]
        n_present = int(np.sum(present))
        if n_rows == 0 or n_present == 0:
            missing.append(name)
        elif n_present < n_rows:
            messages.append(f"partial: {name} is missing in {n_rows - n_present} of "
                            f"{n_rows} stimulation rows")
    sites = sorted({value for value in rows["stim_site"].tolist() if value != "n/a"})
    return ValidationReport(missing=missing, warnings=messages, n_rows=n_rows,
                            n_sites=len(sites), sites=sites)
