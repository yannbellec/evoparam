# SPDX-License-Identifier: Apache-2.0
"""Read BIDS datasets: stimulation events (events.tsv) and epochs around them.

:func:`read_bids` reads one run (or every run that matches) of a BIDS dataset,
locally or remotely by HTTP byte ranges (OpenNeuro by default), from BrainVision or
MEF 3.0 recordings (and, locally, EDF), harmonises the stimulation events into the
standard metadata columns and returns a TrialSet.
"""

import contextlib
import csv
import glob
import io as _io
import json
import logging
import os
import re
import time as _time

import numpy as np

from evoparam.data import Provenance, TrialSet
from evoparam.errors import DataError, EventError, ReaderError
from evoparam.errors import warn as _warn
from evoparam.io._fetch import TransferLog, fetch_bytes, is_url, sha256_file
from evoparam.io._harmonize import (
    HarmonizationReport,
    ValidationReport,
    harmonize_stim_events,
    normalize_site,
    validate_stimulation_events,
)
from evoparam.io._validate import format_stim_report, validate_stim_metadata
from evoparam.io.brainvision import (
    _n_samples,
    _open_data,
    _read_windows,
    _warn_units,
    read_brainvision_header,
)
from evoparam.io.epochs import epoch_windows, overlap_flags
from evoparam.io.fetch import OPENNEURO_URL
from evoparam.utils._optional import require

__all__ = [
    "read_bids",
    "read_bids_events",
    "select_stimulation_events",
    "harmonize_stim_events",
    "HarmonizationReport",
    "validate_stimulation_events",
    "ValidationReport",
    "validate_stim_metadata",
    "format_stim_report",
]

logger = logging.getLogger(__name__)

###################################################################################################
###################################################################################################


def _read_bytes(path):
    if is_url(path):
        return fetch_bytes(path)
    with open(path, "rb") as file:
        return file.read()


def read_bids_events(events_tsv):
    """Read a BIDS events.tsv file.

    Parameters
    ----------
    events_tsv : str
        Path or URL of the events.tsv file.

    Returns
    -------
    events : list of dict
        One dict per row, with column names as keys. 'onset' and 'duration' are converted
        to float; the BIDS value 'n/a' becomes None.
    """
    lines = _read_bytes(events_tsv).decode("utf-8").splitlines()

    events = []
    for raw_row in csv.DictReader(lines, delimiter="\t"):
        row = {key: (None if val == "n/a" else val) for key, val in raw_row.items()}
        for key in ("onset", "duration"):
            if row.get(key) is not None:
                row[key] = float(row[key])
        events.append(row)

    return events


def select_stimulation_events(
    events, *, site=None, trial_type="electrical_stimulation", current=None
):
    """Select stimulation events, optionally for one stimulation site or current.

    Parameters
    ----------
    events : list of dict or pandas.DataFrame
        Events, as returned by `read_bids_events` (list) or a table (harmonised or not).
    site : str, optional
        Stimulation site, for example 'PT01-PT02'. Compared with the 'stim_site' column
        when present (separators normalised), else with 'electrical_stimulation_site'.
        If None, keeps all sites.
    trial_type : str, optional, default: 'electrical_stimulation'
        Value of the 'trial_type' column that marks a stimulation. If None, keeps all types.
    current : float, optional
        Current in mA, compared with the 'stim_current_mA' column (relative tolerance
        1e-6). Needs a harmonised table, or 'stim_current_mA' keys in the dicts.

    Returns
    -------
    selected : list of dict or pandas.DataFrame
        The selected events, in their original order, in the type of `events`
        (a DataFrame keeps its index).
    """

    def keep(get):
        if trial_type is not None and get("trial_type") != trial_type:
            return False
        if site is not None:
            value = get("stim_site")
            if value is None:
                value = get("electrical_stimulation_site")
            if value is None or normalize_site(value) != normalize_site(site):
                return False
        if current is not None:
            value = get("stim_current_mA")
            try:
                if value is None or not np.isclose(float(value), current, rtol=1e-6, atol=0):
                    return False
            except (TypeError, ValueError):
                return False
        return True

    if isinstance(events, list):
        return [event for event in events if keep(event.get)]

    columns = set(events.columns)
    rows = [
        index
        for index, row in zip(events.index, events.to_dict("records"), strict=False)
        if keep(lambda key, row=row: row.get(key) if key in columns else None)
    ]
    return events.loc[rows]


###################################################################################################
## BIDS file access

_ENTITY = re.compile(r"(sub|ses|task|acq|run)-([a-zA-Z0-9]+)")
_ORDER = ("sub", "ses", "task", "acq", "run")

# BIDS channel types -> types of the channel table (spec Table 4.2).
_CH_TYPES = {
    "ecog": "ecog",
    "seeg": "seeg",
    "dbs": "dbs",
    "eeg": "eeg",
    "meg": "meg",
    "megmag": "meg",
    "meggradaxial": "meg",
    "meggradplanar": "meg",
    "emg": "emg",
}

# Coordinate units -> factor to mm.
_TO_MM = {"m": 1e3, "cm": 10.0, "mm": 1.0}


def _strip(label, prefix):
    if label is None:
        return None
    label = str(label)
    return label[len(prefix) :] if label.startswith(prefix) else label


def _stem(entities):
    return "_".join(f"{key}-{entities[key]}" for key in _ORDER if entities.get(key))


class _Tree:
    """Files of a BIDS dataset, on disk or behind an http(s) base address."""

    def __init__(self, root, base, log):
        self.root, self.base, self.log = root, base, log

    @property
    def remote(self):
        return self.base is not None

    def location(self, rel):
        return self.base + rel if self.remote else os.path.join(self.root, *rel.split("/"))

    def read(self, rel, optional=False):
        """Content of a file (bytes), or None for a missing optional file."""
        if self.remote:
            return fetch_bytes(self.location(rel), optional=optional, log=self.log)
        path = self.location(rel)
        if not os.path.isfile(path):
            if optional:
                return None
            raise ReaderError(f"BIDS file not found: {path}")
        with open(path, "rb") as file:
            return file.read()

    def json(self, rel):
        content = self.read(rel, optional=True)
        if content is None:
            return None
        try:
            return json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ReaderError(f"{self.location(rel)} is not valid JSON: {exc}") from exc


def _remote_base(root, remote):
    root = str(root)
    if is_url(root):
        return root.rstrip("/") + "/"
    if remote is False or remote is None:
        return None
    if remote is True or remote == "openneuro":
        return OPENNEURO_URL + root.strip("/") + "/"
    if is_url(remote):
        return remote.rstrip("/") + "/" + root.strip("/") + "/"
    raise ReaderError(
        f"remote must be False, True, 'openneuro' or an http(s) base address, got {remote!r}."
    )


def _find_runs(tree, wanted, datatype):
    """Return the runs to read: list of (folder, entities, data file name)."""
    folder = (
        "sub-{}/".format(wanted["sub"])
        + ("ses-{}/".format(wanted["ses"]) if wanted.get("ses") else "")
        + datatype
        + "/"
    )
    if tree.remote:
        if not wanted.get("task"):
            raise ReaderError(
                "Remote reads need task= (and run= when the dataset has runs): "
                "a remote folder cannot be listed."
            )
        # BrainVision, else a MEF 3.0 session (the Mayo Clinic datasets)
        name = f"{_stem(wanted)}_{datatype}.vhdr"
        if tree.read(folder + name, optional=True) is None:
            name = f"{_stem(wanted)}_{datatype}.mefd"
        return [(folder, dict(wanted), name)]

    pattern = os.path.join(
        tree.root,
        "sub-" + wanted["sub"],
        "ses-" + wanted["ses"] if wanted.get("ses") else "",
        datatype,
        "sub-{}_*_{}.*".format(wanted["sub"], datatype),
    )
    candidates = glob.glob(pattern)
    if not wanted.get("ses"):
        candidates += glob.glob(
            os.path.join(
                tree.root,
                "sub-" + wanted["sub"],
                "ses-*",
                datatype,
                "sub-{}_*_{}.*".format(wanted["sub"], datatype),
            )
        )
    runs = []
    for path in sorted(set(candidates)):
        name = os.path.basename(path)
        if not name.endswith((".vhdr", ".edf", ".mefd")):
            continue
        entities = dict(_ENTITY.findall(name))
        if any(wanted.get(key) and entities.get(key) != wanted[key] for key in _ORDER):
            continue
        rel = os.path.relpath(os.path.dirname(path), tree.root).replace(os.sep, "/") + "/"
        runs.append((rel, entities, name))
    if not runs:
        raise ReaderError(
            f"No {datatype} recording matches "
            f"{ ({k: v for k, v in wanted.items() if v}) } under {tree.root}."
        )
    # One data file per run: prefer BrainVision when several formats exist (then EDF, MEF).
    by_stem = {}
    for rel, entities, name in runs:
        key = (rel, _stem(entities))
        if key not in by_stem or name.endswith(".vhdr"):
            by_stem[key] = (rel, entities, name)
    return [by_stem[key] for key in sorted(by_stem)]


def _read_table(content):
    pd = require("pandas", extra="data", caller="evoparam.io.read_bids")
    return pd.read_csv(_io.BytesIO(content), sep="\t", na_values=["n/a"], keep_default_na=False)


def _events_sidecar(tree, folder, entities, datatype):
    """Return events.json by the inheritance principle (run level, then dataset root).

    It is completed with the ElectricalStimulationParameters field of the recording sidecar.
    """
    stem = _stem(entities)
    sidecar = {}
    for rel in (
        "events.json",
        "task-{}_events.json".format(entities.get("task")),
        folder + stem + "_events.json",
    ):
        content = tree.json(rel)
        if isinstance(content, dict):
            sidecar.update(content)
    recording = tree.json(folder + stem + f"_{datatype}.json") or {}
    if "ElectricalStimulationParameters" in recording:
        sidecar["ElectricalStimulationParameters"] = recording["ElectricalStimulationParameters"]
    return sidecar, recording


def _channel_table(tree, folder, entities, names, content=None):
    """Channel table (Table 4.2) from channels.tsv and electrodes.tsv, or None."""
    pd = require("pandas", extra="data", caller="evoparam.io.read_bids")
    if content is None:
        content = tree.read(folder + _stem(entities) + "_channels.tsv", optional=True)
    if content is None:
        return None
    table = _read_table(content)
    if "name" not in table.columns:
        raise DataError("channels.tsv has no name column.")
    listed = [str(name) for name in table["name"]]
    if set(listed) != set(names):
        extra, absent = sorted(set(listed) - set(names)), sorted(set(names) - set(listed))
        raise DataError(
            f"channels.tsv does not match the data file: only in channels.tsv: {extra[:10]}; "
            f"only in the data: {absent[:10]}."
        )
    table = table.set_index(table["name"].astype(str)).loc[names].reset_index(drop=True)

    def text(column, default=""):
        if column not in table.columns:
            return [default] * len(table)
        return ["" if isinstance(v, float) and np.isnan(v) else str(v) for v in table[column]]

    types = [_CH_TYPES.get(value.lower(), "misc") if value else "misc" for value in text("type")]
    out = pd.DataFrame(
        {
            "name": list(names),
            "type": types,
            "group": text("group"),
            "reference": [_reference(value) for value in text("reference")],
            "bad": [value.lower() == "bad" for value in text("status")],
            "units": text("units"),
        }
    )
    return _add_coordinates(tree, folder, entities, out)


def _reference(value):
    value = value.strip()
    if not value or value.lower() == "n/a":
        return ""
    if value.lower() in ("average", "car", "common average", "common average reference"):
        return "car"
    if re.match(r"^[A-Za-z]+\d+-[A-Za-z]+\d+$", value):
        return "bipolar:" + value
    return "monopolar:" + value


def _add_coordinates(tree, folder, entities, table):
    """x, y, z (mm) and coord_space from electrodes.tsv and coordsystem.json, when present."""
    keys = {key: entities.get(key) for key in ("sub", "ses", "acq")}
    stem = _stem(keys)
    content = tree.read(folder + stem + "_electrodes.tsv", optional=True)
    if content is None:
        return table
    system = tree.json(folder + stem + "_coordsystem.json") or {}
    unit = next(
        (system[key] for key in ("iEEGCoordinateUnits", "EEGCoordinateUnits") if key in system),
        None,
    )
    space = next(
        (system[key] for key in ("iEEGCoordinateSystem", "EEGCoordinateSystem") if key in system),
        "",
    )
    factor = _TO_MM.get(str(unit).lower()) if unit else None
    electrodes = _read_table(content)
    if factor is None or not {"name", "x", "y", "z"} <= set(electrodes.columns):
        return table
    lookup = {
        str(name): row
        for name, row in zip(
            electrodes["name"], electrodes[["x", "y", "z"]].to_numpy(float), strict=False
        )
    }
    coords = np.array([lookup.get(name, (np.nan,) * 3) for name in table["name"]], dtype=float)
    for axis, column in enumerate("xyz"):
        table[column] = coords[:, axis] * factor
    table["coord_space"] = str(space)
    return table


def _pick_channels(names, picks, ch_info):
    """Return the indices of the picked channels: names, indices, or types ('ecog', ...)."""
    from evoparam.io.brainvision import _pick

    if picks is None:
        return list(range(len(names)))
    items = [picks] if isinstance(picks, str) else list(picks)
    if (
        ch_info is not None
        and items
        and all(isinstance(item, str) and item not in names for item in items)
    ):
        types = {item.lower() for item in items}
        if types <= set(ch_info["type"]):
            index = [ind for ind, kind in enumerate(ch_info["type"]) if kind in types]
            if index:
                return index
    return _pick(names, items)


@contextlib.contextmanager
def _verbosity(verbose):
    if verbose is None:
        yield
        return
    level = {True: logging.INFO, False: logging.WARNING}.get(verbose, verbose)
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    package = logging.getLogger("evoparam")
    previous = package.level
    package.setLevel(level)
    try:
        yield
    finally:
        package.setLevel(previous)


###################################################################################################
## Events


def _auto_rows(events, datatype, report):
    """Rows kept by events='auto': stimulations for iEEG, every row otherwise."""
    if datatype != "ieeg":
        return np.ones(len(events), dtype=bool)
    types = [str(value).strip().lower() for value in events.get("trial_type", [])]
    keep = np.array([value == "electrical_stimulation" for value in types], dtype=bool)
    if not keep.any() and report.mapping.get("stim_site") == "trial_type":
        keep = np.array([site != "n/a" for site in events["stim_site"]], dtype=bool)
    return keep


def _apply_select(events, select):
    if select is None:
        return events
    if isinstance(select, str):
        return events.query(select)
    if isinstance(select, dict):
        keep = np.ones(len(events), dtype=bool)
        for column, raw_value in select.items():
            value = raw_value
            if column not in events.columns:
                raise EventError(f"select: no column {column!r} in the events table.")
            values = events[column]
            if column == "stim_site":
                value = (
                    [normalize_site(v) for v in value]
                    if isinstance(value, (list, tuple))
                    else normalize_site(value)
                )
            if isinstance(value, (list, tuple, set)):
                keep &= values.isin(list(value)).to_numpy()
            elif isinstance(value, float):
                keep &= np.isclose(values.to_numpy(dtype=float), value, rtol=1e-6, atol=0)
            else:
                keep &= (values == value).to_numpy()
        return events[keep]
    raise EventError("select must be a dict, a query string or None.")


def _metadata(events, entities, overlaps, pd):
    """Return the standard metadata columns (Table 4.1) for the kept events."""
    table = events.reset_index(drop=True).copy()
    for column, key in (("subject", "sub"), ("session", "ses"), ("run", "run"), ("task", "task")):
        table[column] = pd.Series([entities.get(key) or "n/a"] * len(table))
    table["overlaps_previous"] = np.asarray(overlaps, dtype=bool)
    table["trial"] = table["trial"].astype(np.int64)
    if "stim_n_pulses" in table.columns:
        counts = table["stim_n_pulses"]
        if counts.isna().any():
            table = table.drop(columns="stim_n_pulses")
        else:
            table["stim_n_pulses"] = counts.astype(np.int64)
    for column in table.columns:
        if str(table[column].dtype) not in ("bool", "int64", "float64", "object", "str"):
            if table[column].dtype.kind in "iu":
                table[column] = table[column].astype(np.int64)
            elif table[column].dtype.kind == "f":
                table[column] = table[column].astype(np.float64)
    table.attrs["units"] = dict(events.attrs.get("units", {}))
    return table


###################################################################################################
## Reader


def _auto_chunk(trial_bytes):
    """Cache chunk for remote epochs: a power of two of at least one trial, 64 KiB to 8 MiB."""
    size = 1 << 16
    while size < trial_bytes and size < (8 << 20):
        size <<= 1
    return size


def _read_run(
    tree,
    folder,
    entities,
    data_name,
    datatype,
    epoch,
    events,
    select,
    picks,
    cache_dir,
    pd,
    stim_mapping=None,
):
    stem = _stem(entities)
    tmin, tmax = epoch
    sidecar, recording = _events_sidecar(tree, folder, entities, datatype)

    # Events table
    if isinstance(events, str) and events == "auto":
        table = _read_table(tree.read(folder + stem + "_events.tsv"))
        events_source = tree.location(folder + stem + "_events.tsv")
    elif isinstance(events, str):
        table = _read_table(_read_bytes(events))
        events_source = events
    elif isinstance(events, pd.DataFrame):
        table, events_source = events, "DataFrame"
    else:
        raise EventError("events must be 'auto', a path to an events.tsv or a DataFrame.")
    if "onset" not in table.columns:
        raise EventError("The events table has no onset column.")
    table = table.reset_index(drop=True)
    table["trial"] = np.arange(len(table), dtype=np.int64)
    harmonized, report = harmonize_stim_events(table, sidecar, mapping=stim_mapping)
    for line in report.warnings:
        logger.info("%s: %s", stem, line)

    if isinstance(events, str) and events == "auto":
        harmonized = harmonized[_auto_rows(harmonized, datatype, report)]
    harmonized = _apply_select(harmonized, select)
    harmonized = harmonized[np.isfinite(harmonized["onset"].to_numpy(dtype=float))]
    if len(harmonized) == 0:
        raise EventError(f"No event of {stem} matches the selection.")
    onsets = harmonized["onset"].to_numpy(dtype=float)

    if data_name.endswith(".edf"):
        return _read_edf_run(
            tree,
            folder,
            entities,
            data_name,
            harmonized,
            onsets,
            report,
            epoch,
            picks,
            events_source,
            pd,
        )
    if data_name.endswith(".mefd"):
        return _read_mef_run(
            tree,
            folder,
            entities,
            data_name,
            datatype,
            harmonized,
            onsets,
            report,
            epoch,
            picks,
            events_source,
            recording,
            pd,
        )

    # Recording
    header = read_brainvision_header(tree.location(folder + data_name), log=tree.log)
    names = header["ch_names"]
    ch_info = _channel_table(tree, folder, entities, names)
    ch_index = _pick_channels(names, picks, ch_info)
    _warn_units(header, ch_index)
    sfreq = header["sfreq"]
    frame = header["n_channels"] * np.dtype(header["dtype"]).itemsize
    n_window = int(np.round(tmax * sfreq)) - int(np.round(tmin * sfreq))
    with _open_data(
        header,
        tree.remote,
        chunk_size=_auto_chunk(n_window * frame),
        cache_dir=cache_dir,
        log=tree.log,
    ) as reader:
        n_samples = _n_samples(header, reader)
        times, starts, kept = epoch_windows(onsets, sfreq, tmin, tmax, n_samples)
        if not kept.any():
            raise EventError(
                f"No epoch of ({tmin}, {tmax}) s around the selected events of {stem} fits "
                "inside the recording."
            )
        order = np.argsort(starts[kept], kind="stable")  # read in file order
        data = np.empty((int(kept.sum()), len(ch_index), times.size))
        data[order] = _read_windows(
            reader, header, starts[kept][order], times.size, ch_index, n_samples
        )
        if tree.remote:
            source = {
                "sha256": reader.sha256(),
                "ranges": reader.ranges,
                "data_bytes_read": reader.bytes_read,
                "full_download": reader.full_download,
            }
        else:
            source = {"sha256": sha256_file(header["data_file"])}

    source.update({"path": header["data_file"], "format": "brainvision", "events": events_source})
    return _run_result(
        data,
        times,
        sfreq,
        names,
        ch_index,
        ch_info,
        harmonized,
        onsets,
        kept,
        entities,
        epoch,
        report,
        recording,
        source,
        datatype,
        pd,
    )


def _run_result(
    data,
    times,
    sfreq,
    names,
    ch_index,
    ch_info,
    harmonized,
    onsets,
    kept,
    entities,
    epoch,
    report,
    recording,
    source,
    datatype,
    pd,
    masks=None,
):
    """Return the part of one run: metadata of the kept events, channel table, provenance."""
    tmin, tmax = epoch
    kept_events = harmonized[kept]
    overlaps = overlap_flags(onsets[kept], tmax - tmin)
    metadata = _metadata(kept_events, entities, overlaps, pd)
    ch_names = [names[ind] for ind in ch_index]
    if ch_info is not None:
        ch_info = ch_info.iloc[ch_index].reset_index(drop=True)
        contacts = {part for site in metadata["stim_site"] for part in str(site).split("-")}
        ch_info["stim_contact"] = [name in contacts for name in ch_names]

    info = {
        "kept": kept.tolist(),
        "harmonization": report.to_dict(),
        "ElectricalStimulation": recording.get("ElectricalStimulation"),
    }
    return {
        "data": data,
        "times": times,
        "sfreq": sfreq,
        "ch_names": ch_names,
        "metadata": metadata,
        "ch_info": ch_info,
        "event_times": onsets[kept],
        "source": source,
        "info": info,
        "masks": masks,
        "stimulation": bool(datatype == "ieeg" and (metadata["stim_site"] != "n/a").any()),
    }


def _read_mef_run(
    tree,
    folder,
    entities,
    data_name,
    datatype,
    harmonized,
    onsets,
    report,
    epoch,
    picks,
    events_source,
    recording,
    pd,
):
    """Epochs of a MEF 3.0 session.

    The channels come from channels.tsv (or the session), then the metadata of the picked channels
    and the blocks under the epochs are read.
    """
    from evoparam.io import mef

    stem = _stem(entities)
    tmin, tmax = epoch
    location = tree.location(folder + data_name)
    content = tree.read(folder + stem + "_channels.tsv", optional=True)
    listed = mef._names_from_tsv(content) if content is not None else None
    names = mef._channel_names(location, remote=tree.remote, hint=listed, log=tree.log)
    ch_info = _channel_table(tree, folder, entities, names, content=content)
    ch_index = _pick_channels(names, picks, ch_info)
    header = mef.read_mef_header(
        location, channels=[names[ind] for ind in ch_index], remote=tree.remote, log=tree.log
    )
    if header["sfreq"] is None:
        raise DataError(
            f"{location}: the picked channels have different sampling frequencies; pick "
            "channels of one frequency."
        )
    columns = list(range(len(ch_index)))
    mef._warn_units(header, columns)
    sfreq = header["sfreq"]
    times, starts, kept = epoch_windows(onsets, sfreq, tmin, tmax, header["n_samples"])
    if not kept.any():
        raise EventError(
            f"No epoch of ({tmin}, {tmax}) s around the selected events of {stem} fits "
            "inside the recording."
        )
    data, record = mef._read_grid(header, columns, starts[kept], times.size, log=tree.log)
    nan = np.isnan(data)
    source = {
        "path": header["path"],
        "format": "mef",
        "sha256": record["sha256"],
        "blocks_read": record["blocks_read"],
        "events": events_source,
    }
    if tree.remote:
        source["data_bytes_read"] = record["data_bytes_read"]
    return _run_result(
        data,
        times,
        sfreq,
        names,
        ch_index,
        ch_info,
        harmonized,
        onsets,
        kept,
        entities,
        epoch,
        report,
        recording,
        source,
        datatype,
        pd,
        masks=nan if nan.any() else None,
    )


def _read_edf_run(
    tree, folder, entities, data_name, harmonized, onsets, report, epoch, picks, events_source, pd
):
    from evoparam.io.readers import read_edf

    if tree.remote:
        raise ReaderError("Remote reads support BrainVision and MEF 3.0 files only.")
    path = tree.location(folder + data_name)
    data, sfreq, names, t0 = read_edf(path, picks=picks)
    tmin, tmax = epoch
    times, starts, kept = epoch_windows(onsets, sfreq, tmin, tmax, data.shape[1], t0=t0)
    if not kept.any():
        raise EventError(f"No epoch of ({tmin}, {tmax}) s fits inside {path}.")
    index = starts[kept][:, None] + np.arange(times.size)[None, :]
    trials = np.moveaxis(data[:, index], 0, 1)
    overlaps = overlap_flags(onsets[kept], tmax - tmin)
    metadata = _metadata(harmonized[kept], entities, overlaps, pd)
    source = {"path": path, "format": "edf", "sha256": sha256_file(path), "events": events_source}
    info = {"kept": kept.tolist(), "harmonization": report.to_dict()}
    return {
        "data": trials,
        "times": times,
        "sfreq": sfreq,
        "ch_names": names,
        "metadata": metadata,
        "ch_info": None,
        "event_times": onsets[kept],
        "source": source,
        "info": info,
        "stimulation": bool((metadata["stim_site"] != "n/a").any()),
    }


def _dataset_record(root, base, tree):
    """Dataset identifier and DOI: from the registry, else from dataset_description.json."""
    from evoparam.io.registry import match_dataset

    entry = match_dataset(base if base is not None else os.path.abspath(str(root)))
    if entry is not None:
        return entry["id"], entry["doi"]
    description = tree.json("dataset_description.json") if not tree.remote else None
    if isinstance(description, dict):
        return None, description.get("DatasetDOI")
    return None, None


def read_bids(  # noqa: PLR0913  (public API, signature fixed by spec §5.1)
    root,
    subject,
    *,
    session=None,
    task=None,
    run=None,
    acquisition=None,
    datatype="ieeg",
    epoch=(-0.5, 1.0),
    events="auto",
    select=None,
    picks=None,
    remote=False,
    cache_dir=None,
    verbose=None,
    stim_mapping=None,
):
    """Read epochs around the events of a BIDS dataset, locally or remotely.

    Recordings in BrainVision (``.vhdr``), MEF 3.0 (``.mefd`` session folders: the Mayo
    Clinic datasets ds003708, ds004457, ds004696 and ds004977) and, locally, EDF are read.
    For MEF, the channels and their order come from ``channels.tsv``; only the metadata
    of the picked channels and the blocks under the epochs are read (see
    :func:`~evoparam.io.read_mef`); samples without data are NaN and marked in ``masks``.

    Parameters
    ----------
    root : str or path-like
        Dataset folder; with ``remote``, the OpenNeuro identifier (for example
        ``"ds004080"``) or an http(s) address of the dataset root.
    subject : str
        Subject label, with or without the ``sub-`` prefix.
    session, task, run, acquisition : str, optional
        BIDS entities. Locally, every run that matches is read and the trials are
        concatenated (same channels and sampling frequency required). Remote reads need
        ``task`` (and ``run`` / ``session`` when the dataset has them), because a remote
        folder cannot be listed.
    datatype : str
        BIDS datatype folder: ``"ieeg"``, ``"eeg"``, ...
    epoch : tuple of float
        Epoch window ``(tmin, tmax)`` in seconds around each onset; tmin included,
        tmax excluded.
    events : "auto", str or pandas.DataFrame
        ``"auto"``: read ``*_events.tsv``, harmonise it (:func:`harmonize_stim_events`) and
        keep the ``trial_type == "electrical_stimulation"`` rows for iEEG (all rows for
        other datatypes). A path gives another events.tsv; a DataFrame is used as given
        (harmonised, every row kept).
    select : dict or str, optional
        Applied to the harmonised events: a dict ``{column: value or list of values}``
        (for example ``{"stim_site": "PT01-PT02"}``) or a pandas query string.
    picks : list of str or int, or str, optional
        Channels by name or index, or channel types of channels.tsv (``"ecog"``,
        ``["ecog", "seeg"]``). Default: all channels.
    remote : bool or str
        False: ``root`` is a local folder. True or ``"openneuro"``: files are read from
        ``https://s3.amazonaws.com/openneuro.org/<root>/`` by HTTP byte ranges, so only
        the epochs are transferred. An http(s) address: base address under which
        ``root`` lives. Remotely, a run is read from ``*_<datatype>.vhdr`` when it
        exists, else from the ``*_<datatype>.mefd`` session.
    cache_dir : str, optional
        Where a remote data file is saved when the server ignores byte ranges.
    verbose : bool, str or None
        Logging level of the package during the call (None: unchanged).
    stim_mapping : dict, optional
        ``{source column: standard column}`` passed to :func:`harmonize_stim_events` as
        ``mapping``: overrides the recognised columns (spec §10.2); the applied mapping is
        in the harmonisation report of the provenance.

    Returns
    -------
    trials : TrialSet
        Data in µV. ``metadata`` holds the kept events with the standard columns of
        Table 4.1 (``trial`` = row of the event in events.tsv, ``subject``, ``session``,
        ``run``, ``task``, ``condition``, ``stim_site``, ``stim_current_mA``,
        ``stim_pulse_width_us``, ...), every original column, and ``overlaps_previous``.
        ``ch_info`` comes from channels.tsv (and electrodes.tsv). ``provenance.source``
        records the data file, ``sha256`` of the bytes read (the whole data file for
        local reads; the byte ranges for remote reads, with ``ranges``), ``bytes_read``
        (bytes transferred, sidecars included), the BIDS entities and the dataset DOI
        when the dataset is in the registry.

    Raises
    ------
    ReaderError
        Missing or unreadable files.
    EventError
        No event matches the selection, or no epoch fits inside the recording.
    DataError
        channels.tsv does not match the data file; runs with different channels or
        sampling frequencies.
    """
    start = _time.perf_counter()
    pd = require("pandas", extra="data", caller="evoparam.io.read_bids")
    tmin, tmax = float(epoch[0]), float(epoch[1])
    wanted = {
        "sub": _strip(subject, "sub-"),
        "ses": _strip(session, "ses-"),
        "task": _strip(task, "task-"),
        "acq": _strip(acquisition, "acq-"),
        "run": _strip(run, "run-"),
    }
    log = TransferLog()
    base = _remote_base(root, remote)
    tree = _Tree(None if base else os.fspath(root), base, log)

    with _verbosity(verbose):
        if not tree.remote and not os.path.isdir(tree.root):
            raise ReaderError(f"BIDS root not found: {tree.root}")
        parts = []
        for folder, entities, data_name in _find_runs(tree, wanted, datatype):
            parts.append(
                (
                    entities,
                    _read_run(
                        tree,
                        folder,
                        entities,
                        data_name,
                        datatype,
                        (tmin, tmax),
                        events,
                        select,
                        picks,
                        cache_dir,
                        pd,
                        stim_mapping,
                    ),
                )
            )
    dataset, doi = _dataset_record(root, base, tree)

    if len(parts) == 1:
        entities, part = parts[0]
        data, metadata, event_times = part["data"], part["metadata"], part["event_times"]
        masks = part.get("masks")
        source = dict(part["source"])
    else:
        entities, part = parts[0]
        for _other_entities, other in parts[1:]:
            if not np.isclose(other["sfreq"], part["sfreq"], rtol=1e-9):
                raise DataError(
                    "Runs have different sampling frequencies ({} and {} Hz); read "
                    "them separately.".format(part["sfreq"], other["sfreq"])
                )
            if other["ch_names"] != part["ch_names"]:
                raise DataError(
                    "Runs have different channels; read them separately or give picks."
                )
        data = np.concatenate([p["data"] for _, p in parts])
        masks = None
        if any(p.get("masks") is not None for _, p in parts):
            masks = np.concatenate(
                [
                    np.zeros(p["data"].shape, dtype=bool) if p.get("masks") is None else p["masks"]
                    for _, p in parts
                ]
            )
        metadata = pd.concat([p["metadata"] for _, p in parts], ignore_index=True)
        metadata.attrs["units"] = dict(part["metadata"].attrs.get("units", {}))
        event_times = np.concatenate([p["event_times"] for _, p in parts])
        source = {
            "path": [p["source"]["path"] for _, p in parts],
            "format": "bids",
            "runs": [p["source"] for _, p in parts],
        }

    # Epochs whose samples are all exactly zero: data censored in the deposit (ds004977
    # keeps one stimulation site per run and writes zeros elsewhere), never real signal
    flat = np.flatnonzero(np.all(np.nan_to_num(data) == 0, axis=(1, 2)))
    if flat.size:
        _warn(
            f"{flat.size} of {data.shape[0]} epochs hold only zeros "
            "(data censored in the deposit, or a "
            f"stimulation site outside the published span?); trials {flat[:10].tolist()}.",
            UserWarning,
            stacklevel=2,
        )
        source["zero_epochs"] = flat.tolist()
    source.update(
        {
            "reader": "evoparam.io.read_bids",
            "bytes_read": log.bytes_read,
            "root": base or os.path.abspath(os.fspath(root)),
            "dataset": dataset,
            "doi": doi,
            "bids": {
                "subject": wanted["sub"],
                "session": wanted["ses"],
                "task": wanted["task"],
                "run": wanted["run"],
                "acquisition": wanted["acq"],
                "datatype": datatype,
            },
        }
    )
    settings = {
        "subject": wanted["sub"],
        "session": wanted["ses"],
        "task": wanted["task"],
        "run": wanted["run"],
        "acquisition": wanted["acq"],
        "datatype": datatype,
        "epoch": [tmin, tmax],
        "events": events if isinstance(events, str) else "DataFrame",
        "select": select,
        "picks": None
        if picks is None
        else (
            [picks]
            if isinstance(picks, str)
            else [p if isinstance(p, str) else int(p) for p in picks]
        ),
        "remote": remote if isinstance(remote, (bool, str)) else bool(remote),
    }
    provenance = Provenance(source=source).with_step(
        "evoparam.io.read_bids",
        settings,
        duration_s=_time.perf_counter() - start,
        runs=[p["info"] for _, p in parts],
    )
    return TrialSet(
        data=data,
        times=part["times"],
        sfreq=part["sfreq"],
        ch_names=part["ch_names"],
        units="uV",
        metadata=metadata,
        ch_info=part["ch_info"],
        masks=masks,
        event_times=event_times,
        modality="spes" if any(p["stimulation"] for _, p in parts) else None,
        provenance=provenance,
    )
