# SPDX-License-Identifier: Apache-2.0
"""Stimulation-metadata validator over a BIDS tree (spec §10.2, REQ-314).

For every run with an ``*_events.tsv``, each stimulation field of Table 4.1, the
``ElectricalStimulation`` flag of the recording sidecar and the channel-table fields
(``reference``, ``type``, coordinates) get one status:

* ``present``: stated for every stimulation event (or every channel);
* ``inferred``: filled, but by a rule rather than by the deposit (a unit guessed from the
  magnitude, a site taken from ``trial_type``, a reference from the recording sidecar);
* ``missing``: not stated anywhere;
* ``inconsistent``: stated for some events or channels only, or with values that cannot be
  read (unknown unit, text in a number column, polarity without phase order, coordinates
  without units, ``ElectricalStimulation`` false while the events describe stimulations).
"""

import glob
import os
import re
import warnings

import numpy as np

from evoparam._typing import Any, StrPath
from evoparam.data._harmonize import _is_missing, _stimulation_rows
from evoparam.errors import ReaderError
from evoparam.utils._optional import require

from ._harmonize import STIM_FIELDS, harmonize_stim_events

__all__ = ["validate_stim_metadata", "format_stim_report", "VALIDATOR_FIELDS", "STATUSES"]

VALIDATOR_FIELDS = STIM_FIELDS + ("ElectricalStimulation", "reference", "type", "coordinates")
STATUSES = ("present", "inferred", "missing", "inconsistent")
_DATATYPES = ("ieeg", "eeg")
_ENTITY = re.compile(r"(sub|ses|task|acq|run)-([a-zA-Z0-9]+)")
_KNOWN_TYPES = {
    "ecog",
    "seeg",
    "dbs",
    "eeg",
    "meg",
    "megmag",
    "meggradaxial",
    "meggradplanar",
    "emg",
    "eog",
    "ecg",
    "misc",
    "trig",
    "resp",
    "temp",
    "gsr",
    "ppg",
    "audio",
    "sysclock",
    "ref",
    "vehicle",
    "eyegaze",
    "pupil",
    "hrv",
}


def _events_files(root: Any, subject: Any) -> list[Any]:
    subjects = (
        [f"sub-{subject.removeprefix('sub-')}"]
        if subject
        else sorted(
            os.path.basename(p) for p in glob.glob(os.path.join(root, "sub-*")) if os.path.isdir(p)
        )
    )
    files = []
    for sub in subjects:
        for datatype in _DATATYPES:
            for pattern in (
                os.path.join(root, sub, datatype, "*_events.tsv"),
                os.path.join(root, sub, "ses-*", datatype, "*_events.tsv"),
            ):
                files += [(path, datatype) for path in glob.glob(pattern)]
    return sorted(set(files))


def _field_status(
    name: Any, values: Any, n_rows: Any, units: Any, mapping: Any, messages: Any
) -> Any:
    """(status, message) of one stimulation field over the stimulation rows."""
    if name == "stim_polarity":
        filled = [not _is_missing(v) and v != "n/a" for v in values]
        unknown = sum(v == "unknown" for v in values)
    elif name == "stim_site":
        filled = [not _is_missing(v) and v != "n/a" for v in values]
        unknown = 0
    else:
        filled = [not _is_missing(v) for v in values]
        unknown = 0
    n_filled = int(np.sum(filled))
    column = mapping.get(name)
    problems = [
        m
        for m in messages
        if column is not None
        and m.split(":", 1)[0] in ("unparsable", "unit_unknown", "n_pulses_not_integer")
        and f" {column} " in f" {m.split(':', 1)[1]} "
    ]
    if n_rows == 0:
        return "missing", "no stimulation event in the run"
    if n_filled == 0 and not problems:
        source = "no source column" if column is None else f"column {column!r} is empty"
        return "missing", f"{name}: {source}"
    if problems or unknown or n_filled < n_rows:
        parts = []
        if n_filled < n_rows:
            parts.append(f"stated for {n_filled} of {n_rows} stimulation events")
        if unknown:
            parts.append(f"{unknown} value(s) without phase order")
        parts += [p.split(";")[0] for p in problems]
        return "inconsistent", f"{name} (from {column!r}): " + "; ".join(parts)
    unit = units.get(name, "")
    if unit.startswith("inferred:") or (name == "stim_site" and column == "trial_type"):
        how = (
            f"unit {unit.split(':', 1)[1]} guessed from the magnitude"
            if unit.startswith("inferred:")
            else "site read from trial_type"
        )
        return "inferred", f"{name} (from {column!r}): {how}"
    return "present", f"{name} from {column!r}" + (f" ({unit})" if unit else "")


def _read_tsv(path: Any) -> Any:
    pd = require("pandas", extra="data", caller="evoparam.io.validate_stim_metadata")
    return pd.read_csv(path, sep="\t", na_values=["n/a"], keep_default_na=False)


def _channel_rows(folder: Any, stem_parts: Any, recording: Any, n_events: Any) -> Any:
    """Statuses of reference, type and coordinates for one run."""
    stem = "_".join(
        f"{k}-{stem_parts[k]}" for k in ("sub", "ses", "task", "acq", "run") if stem_parts.get(k)
    )
    rows = []
    channels_path = os.path.join(folder, stem + "_channels.tsv")
    channels = _read_tsv(channels_path) if os.path.isfile(channels_path) else None
    side_ref = next(
        (
            recording[k]
            for k in ("iEEGReference", "EEGReference")
            if isinstance(recording.get(k), str) and recording[k].strip()
        ),
        None,
    )

    def column_values(table: Any, column: Any) -> Any:
        if table is None or column not in table.columns:
            return None
        return ["" if _is_missing(v) else str(v).strip() for v in table[column].tolist()]

    # reference
    values = column_values(channels, "reference")
    if values is not None and all(values):
        rows.append(("reference", "present", f"channels.tsv reference for {len(values)} channels"))
    elif values is not None and any(values):
        rows.append(
            (
                "reference",
                "inconsistent",
                f"channels.tsv reference stated for "
                f"{sum(map(bool, values))} of {len(values)} "
                f"channels",
            )
        )
    elif side_ref is not None:
        rows.append(
            (
                "reference",
                "inferred",
                f"from the recording sidecar: {side_ref!r} (not per channel)",
            )
        )
    else:
        rows.append(
            (
                "reference",
                "missing",
                "no reference column in channels.tsv and no "
                "iEEGReference/EEGReference in the sidecar",
            )
        )
    # type
    values = column_values(channels, "type")
    if values is None:
        rows.append(
            (
                "type",
                "missing",
                "no channels.tsv" if channels is None else "channels.tsv has no type column",
            )
        )
    else:
        bad = [v for v in values if v.lower() not in _KNOWN_TYPES]
        if not bad:
            rows.append(("type", "present", f"{len(values)} channels typed"))
        elif len(bad) == len(values):
            rows.append(("type", "missing", "no channel has a BIDS type"))
        else:
            rows.append(
                (
                    "type",
                    "inconsistent",
                    f"{len(bad)} of {len(values)} channels "
                    f"without a BIDS type ({sorted(set(bad))[:5]})",
                )
            )
    # coordinates (electrodes.tsv of the subject/session/acquisition, units in coordsystem)
    keys = "_".join(f"{k}-{stem_parts[k]}" for k in ("sub", "ses", "acq") if stem_parts.get(k))
    electrodes_path = os.path.join(folder, keys + "_electrodes.tsv")
    if not os.path.isfile(electrodes_path):
        rows.append(("coordinates", "missing", "no electrodes.tsv"))
    else:
        import json

        electrodes = _read_tsv(electrodes_path)
        system_path = os.path.join(folder, keys + "_coordsystem.json")
        system = {}
        if os.path.isfile(system_path):
            with open(system_path, encoding="utf-8") as fobj:
                system = json.load(fobj)
        unit = next(
            (system[k] for k in ("iEEGCoordinateUnits", "EEGCoordinateUnits") if k in system), None
        )
        if not {"name", "x", "y", "z"} <= set(electrodes.columns):
            rows.append(("coordinates", "inconsistent", "electrodes.tsv lacks x, y or z"))
        else:
            xyz = (
                electrodes[["x", "y", "z"]]
                .apply(lambda c: c.map(lambda v: np.nan if _is_missing(v) else float(v)))
                .to_numpy()
            )
            n_ok = int(np.isfinite(xyz).all(axis=1).sum())
            names = set(map(str, electrodes["name"]))
            listed = set(column_values(channels, "name") or [])
            n_without = len(listed - names) if listed else 0
            if unit is None or str(unit).lower() not in ("m", "cm", "mm"):
                rows.append(
                    (
                        "coordinates",
                        "inconsistent",
                        f"electrodes.tsv has coordinates but coordsystem.json gives no "
                        f"metric unit ({unit!r})",
                    )
                )
            elif n_ok == 0:
                rows.append(("coordinates", "missing", "electrodes.tsv coordinates are all n/a"))
            elif n_ok < len(electrodes) or n_without:
                rows.append(
                    (
                        "coordinates",
                        "inconsistent",
                        f"coordinates for {n_ok} of {len(electrodes)} electrodes"
                        + (f"; {n_without} channels not in electrodes.tsv" if n_without else ""),
                    )
                )
            else:
                space = system.get(
                    "iEEGCoordinateSystem", system.get("EEGCoordinateSystem", "unknown space")
                )
                rows.append(("coordinates", "present", f"{n_ok} electrodes in {unit}, {space}"))
    return [(field, status, n_events, message) for field, status, message in rows]


def _sidecars(
    root: Any, folder: Any, parts: Any, datatype: Any
) -> tuple[dict[Any, Any], dict[Any, Any]]:
    """events.json by inheritance (dataset root, task, run) and the recording sidecar."""
    import json

    def load(path: Any) -> Any:
        if not os.path.isfile(path):
            return {}
        with open(path, encoding="utf-8") as fobj:
            content = json.load(fobj)
        return content if isinstance(content, dict) else {}

    stem = "_".join(
        f"{k}-{parts[k]}" for k in ("sub", "ses", "task", "acq", "run") if parts.get(k)
    )
    sidecar = {}
    for path in (
        os.path.join(root, "events.json"),
        os.path.join(root, f"task-{parts.get('task')}_events.json"),
        os.path.join(folder, stem + "_events.json"),
    ):
        sidecar.update(load(path))
    recording = {}
    for path in (
        os.path.join(root, f"task-{parts.get('task')}_{datatype}.json"),
        os.path.join(folder, f"{stem}_{datatype}.json"),
    ):
        recording.update(load(path))
    if "ElectricalStimulationParameters" in recording:
        sidecar["ElectricalStimulationParameters"] = recording["ElectricalStimulationParameters"]
    return sidecar, recording


def validate_stim_metadata(
    bids_root: StrPath, *, subject: str | None = None, strict: bool = False
) -> Any:
    """Which stimulation metadata a local BIDS dataset states, run by run (REQ-314).

    Parameters
    ----------
    bids_root : str or path
        Root of a BIDS dataset on disk (``sub-*/[ses-*/]{ieeg,eeg}/*_events.tsv``).
    subject : str, optional
        One subject (with or without ``sub-``); default every subject.
    strict : bool
        Raise :class:`~evoparam.errors.ReaderError` when any field is ``missing``.

    Returns
    -------
    pandas.DataFrame
        One row per run and field: ``subject``, ``session``, ``task``, ``acquisition``,
        ``run``, ``datatype``, ``events_file``, ``field`` (``stim_site``,
        ``stim_current_mA``, ``stim_pulse_width_us``, ``stim_polarity``,
        ``stim_frequency_hz``, ``stim_n_pulses``, ``ElectricalStimulation``, ``reference``,
        ``type``, ``coordinates``), ``status`` (``present``, ``inferred``, ``missing``,
        ``inconsistent``), ``n_events`` (stimulation events of the run) and ``message``.
    """
    pd = require("pandas", extra="data", caller="evoparam.io.validate_stim_metadata")
    root = os.fspath(bids_root)
    if not os.path.isdir(root):
        raise ReaderError(f"validate_stim_metadata: {root} is not a local BIDS folder.")
    files = _events_files(root, subject)
    if not files:
        raise ReaderError(
            f"No *_events.tsv under {root}"
            + (f" for sub-{subject.removeprefix('sub-')}" if subject else "")
            + "."
        )
    rows = []
    for path, datatype in files:
        folder = os.path.dirname(path)
        parts = dict(_ENTITY.findall(os.path.basename(path)))
        sidecar, recording = _sidecars(root, folder, parts, datatype)
        events = _read_tsv(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            harmonized, report = harmonize_stim_events(events, sidecar)
        stim = harmonized.iloc[np.flatnonzero(_stimulation_rows(harmonized))]
        n_events = len(stim)
        found = []
        for name in STIM_FIELDS:
            status, message = _field_status(
                name, stim[name].tolist(), n_events, report.units, report.mapping, report.warnings
            )
            found.append((name, status, n_events, message))
        flag = recording.get("ElectricalStimulation")
        if flag is None:
            found.append(
                (
                    "ElectricalStimulation",
                    "missing",
                    n_events,
                    f"no ElectricalStimulation in the {datatype}.json sidecar",
                )
            )
        elif (
            str(flag).lower() in ("false", "0")
            and n_events
            and report.mapping.get("stim_site") is not None
        ):
            found.append(
                (
                    "ElectricalStimulation",
                    "inconsistent",
                    n_events,
                    "ElectricalStimulation is false but the events describe stimulations",
                )
            )
        else:
            found.append(
                ("ElectricalStimulation", "present", n_events, f"ElectricalStimulation = {flag}")
            )
        found += _channel_rows(folder, parts, recording, n_events)
        for name, status, n, message in found:
            rows.append(
                {
                    "subject": parts.get("sub"),
                    "session": parts.get("ses"),
                    "task": parts.get("task"),
                    "acquisition": parts.get("acq"),
                    "run": parts.get("run"),
                    "datatype": datatype,
                    "events_file": os.path.relpath(path, root).replace(os.sep, "/"),
                    "field": name,
                    "status": status,
                    "n_events": int(n),
                    "message": message,
                }
            )
    table = pd.DataFrame(rows)
    if strict and (table["status"] == "missing").any():
        missing = table[table["status"] == "missing"]
        lines = [f"{r.events_file}: {r.field}" for r in missing.itertuples()]
        raise ReaderError(
            f"{len(lines)} stimulation metadata field(s) missing:\n  "
            + "\n  ".join(lines[:50])
            + ("\n  ..." if len(lines) > 50 else "")
        )
    return table


def format_stim_report(table: Any) -> Any:
    """Plain-text report of :func:`validate_stim_metadata`.

    One block per run, one line per field (``status  field  message``), and the counts of each
    status.
    """
    lines = []
    for events_file, run in table.groupby("events_file", sort=True):
        lines.append(f"{events_file}  ({int(run['n_events'].iloc[0])} stimulation events)")
        for row in run.itertuples():
            lines.append(f"  {row.status:<12} {row.field:<22} {row.message}")
    counts = table["status"].value_counts()
    lines.append("summary: " + ", ".join(f"{counts.get(s, 0)} {s}" for s in STATUSES))
    return "\n".join(lines)
