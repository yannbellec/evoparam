"""NWB reading and writing (spec §10.3, REQ-317, REQ-318; extra ``nwb``).

``read_nwb`` epochs one ``ElectricalSeries`` (or any ``TimeSeries`` whose data are
``(n_times, n_channels)``) around event times, read lazily through h5py (local files, or
http(s) addresses read by byte ranges with ``remfile``: a DANDI asset is read without
downloading it). ``write_nwb`` adds evoparam's tables to a processing module.
"""

import json
import os

import numpy as np

from evoparam.errors import DataError, ReaderError
from evoparam.utils._optional import require

__all__ = ["read_nwb", "write_nwb", "nwb_events", "EVENT_SOURCES"]

EVENT_SOURCES = ("events_table", "trials", "intervals", "stim_table", "icephys")
NWB_UNITS = {"amplitude": "A", "pulse_width": "s", "frequency": "Hz"}
NWB_MAPPING = {"amplitude": "stim_current_mA", "pulse_width": "stim_pulse_width_us",
               "bipolar_pair": "stim_site", "frequency": "stim_frequency_hz"}
ICEPHYS_FRACTION = 0.1


def _open(path):
    pynwb = require("pynwb", extra="nwb", caller="evoparam.io.read_nwb")
    path = os.fspath(path)
    if path.startswith(("http://", "https://")):
        h5py = require("h5py", extra="nwb", caller="evoparam.io.read_nwb")
        remfile = require("remfile", extra="nwb", caller="evoparam.io.read_nwb (remote)")
        handle = h5py.File(remfile.File(path), "r")
        io = pynwb.NWBHDF5IO(file=handle, mode="r", load_namespaces=True)
    else:
        if not os.path.exists(path):
            raise ReaderError(f"NWB file not found: {path}")
        io = pynwb.NWBHDF5IO(path, mode="r", load_namespaces=True)
    return io


def _all_objects(nwbfile):
    """(path, object) of every data interface of the file."""

    out = []
    for name, obj in nwbfile.acquisition.items():
        out.append((f"acquisition/{name}", obj))
    for mod_name, module in nwbfile.processing.items():
        for name, obj in module.data_interfaces.items():
            out.append((f"processing/{mod_name}/{name}", obj))
            for sub_name, sub in getattr(obj, "electrical_series", {}).items():
                out.append((f"processing/{mod_name}/{name}/{sub_name}", sub))
    for name, obj in nwbfile.acquisition.items():
        for sub_name, sub in getattr(obj, "electrical_series", {}).items():
            out.append((f"acquisition/{name}/{sub_name}", sub))
    return out


def _series(nwbfile, name):
    from pynwb.base import TimeSeries

    candidates = [(p, o) for p, o in _all_objects(nwbfile) if isinstance(o, TimeSeries)
                  and getattr(o.data, "ndim", len(getattr(o.data, "shape", ()))) >= 1]
    if name:
        for path, obj in candidates:
            if obj.name == name or path.endswith(name):
                return obj
        raise DataError(f"No TimeSeries named {name!r}; found {[p for p, _ in candidates]}.")
    electrical = [(p, o) for p, o in candidates if type(o).__name__ == "ElectricalSeries"]
    if not electrical:
        raise DataError(f"No ElectricalSeries in the file; name one of "
                        f"{[p for p, _ in candidates]} with electrical_series=.")
    return electrical[0][1]


def _table_frame(table):
    frame = table.to_dataframe()
    keep = [c for c in frame.columns if frame[c].map(lambda v: np.isscalar(v) or v is None).all()]
    return frame[keep].reset_index(drop=False)


def nwb_events(nwbfile, events=None):
    """(onsets in s, events table, source name) by the order of spec §10.3."""

    import pandas as pd

    if isinstance(events, pd.DataFrame):
        column = "onset" if "onset" in events else "start_time"
        return events[column].to_numpy(float), events.reset_index(drop=True), "table"
    if events is not None and not isinstance(events, str):
        onsets = np.asarray(events, float)
        return onsets, pd.DataFrame({"onset": onsets}), "array"
    # (1) ndx-events EventsTable
    for path, obj in _all_objects(nwbfile):
        if type(obj).__name__ in ("EventsTable", "AnnotatedEventsTable", "Events") and \
                (events is None or obj.name == events):
            if hasattr(obj, "to_dataframe"):
                frame = _table_frame(obj)
                column = "timestamp" if "timestamp" in frame else "onset"
                return frame[column].to_numpy(float), frame, "events_table"
            stamps = np.asarray(obj.timestamps[:], float)
            return stamps, pd.DataFrame({"onset": stamps}), "events_table"
    # (2) trials
    if nwbfile.trials is not None and (events in (None, "trials")):
        frame = _table_frame(nwbfile.trials)
        column = "stim_onset" if "stim_onset" in frame else "start_time"
        return frame[column].to_numpy(float), frame, "trials"
    # (3) TimeIntervals under /intervals
    intervals = dict(nwbfile.intervals) if nwbfile.intervals is not None else {}
    intervals.pop("trials", None)
    if events is not None and events in intervals:
        frame = _table_frame(intervals[events])
        return frame["start_time"].to_numpy(float), frame, "intervals"
    if events is None and intervals:
        name = sorted(intervals)[0]
        frame = _table_frame(intervals[name])
        return frame["start_time"].to_numpy(float), frame, "intervals"
    # (4) StimTable of ndx-electrical-stim
    for path, obj in _all_objects(nwbfile):
        if type(obj).__name__ == "StimTable":
            frame = _table_frame(obj)
            return frame["start_time"].to_numpy(float), frame, "stim_table"
    # (5) intracellular stimulus series
    stimuli = list(nwbfile.stimulus.values()) if nwbfile.stimulus else []
    onsets = []
    for series in stimuli:
        if type(series).__name__ not in ("CurrentClampStimulusSeries",
                                         "VoltageClampStimulusSeries"):
            continue
        data = np.asarray(series.data[:], float)
        level = ICEPHYS_FRACTION * np.max(np.abs(data))
        above = np.flatnonzero(np.abs(data) > level)
        if above.size:
            rate = series.rate
            start = series.starting_time or 0.0
            onsets.append(start + above[0] / rate)
    if onsets:
        onsets = np.sort(np.asarray(onsets, float))
        return onsets, pd.DataFrame({"onset": onsets}), "icephys"
    raise DataError("No event source in the file (EventsTable, trials, intervals, StimTable, "
                    "intracellular stimuli); pass events= (a table name, onsets or a table).")


def _harmonise(frame):
    from ._harmonize import harmonize_stim_events

    sidecar = {c: {"Units": u} for c, u in NWB_UNITS.items() if c in frame}
    mapping = {c: t for c, t in NWB_MAPPING.items() if c in frame}
    table, report = harmonize_stim_events(frame.rename(columns={"start_time": "onset"})
                                          if "onset" not in frame else frame,
                                          sidecar, mapping=mapping)
    return table, report


def _electrodes(nwbfile, series, picks):
    import pandas as pd

    region = getattr(series, "electrodes", None)
    if region is not None:
        table = region.table.to_dataframe()
        rows = list(region.data[:])
        info = table.iloc[rows].reset_index(drop=False)
        info = info.rename(columns={"id": "electrode_id"})
        if "label" in info and info["label"].notna().all():
            names = [str(v) for v in info["label"]]
        else:
            names = [f"e{int(v)}" for v in info["electrode_id"]]
        if "group_name" in info:
            info["group"] = info["group_name"]
    else:
        n = series.data.shape[1] if len(series.data.shape) > 1 else 1
        names = [f"ch{i}" for i in range(n)]
        info = pd.DataFrame({"name": names})
    info = info[[c for c in info.columns if info[c].map(
        lambda v: np.isscalar(v) or v is None).all()]]
    info.insert(0, "name", names) if "name" not in info else None
    index = list(range(len(names)))
    if picks is not None:
        index = [names.index(p) if isinstance(p, str) else int(p) for p in picks]
    return [names[i] for i in index], info.iloc[index].reset_index(drop=True), index


def read_nwb(path, *, electrical_series=None, events=None, tmin=-0.1, tmax=0.5, picks=None,
             stim_columns="auto", select=None, units="uV"):
    """Epochs of an NWB ``ElectricalSeries`` around events (spec §10.3, REQ-317).

    Parameters
    ----------
    path : str
        Local NWB file, or an http(s) address (read by byte ranges; DANDI assets).
    electrical_series : str, optional
        Name (or path end) of the series; default the first ``ElectricalSeries``.
    events : str, array or DataFrame, optional
        A table name, onsets (s) or an events table (``onset`` or ``start_time``, the other
        columns become metadata; for events stored in another file, such as the stimulus
        tables of an Allen session). Default: the first source found in the order
        EventsTable (ndx-events), ``trials`` (``stim_onset`` or ``start_time``),
        ``/intervals``, StimTable (ndx-electrical-stim), intracellular stimulus onsets.
    tmin, tmax : float
        Epoch window (s); ``tmin`` included, ``tmax`` excluded.
    select : str, optional
        pandas query on the events table (for example ``"stimulus_name == 'flashes'"``).
    stim_columns : "auto" or None
        Harmonise stimulation columns (NWB ``amplitude`` in A, ``pulse_width`` in s) into
        the standard metadata columns; None keeps the table as it is.

    Returns
    -------
    TrialSet
        Data in µV (the series' ``conversion`` and ``offset`` applied); events outside the
        recording are dropped.
    """

    from evoparam.data import Provenance, TrialSet

    io = _open(path)
    try:
        nwbfile = io.read()
        series = _series(nwbfile, electrical_series)
        onsets, frame, source = nwb_events(nwbfile, events)
        if select:
            keep = frame.query(select).index.to_numpy()
            onsets, frame = onsets[keep], frame.loc[keep].reset_index(drop=True)
        names, info, index = _electrodes(nwbfile, series, picks)
        data = series.data
        n_total = data.shape[0]
        stamps = None
        if series.timestamps is not None:
            stamps = np.asarray(series.timestamps[:], float)
            sfreq = float(1.0 / np.median(np.diff(stamps)))
            sample_of = lambda t: int(np.searchsorted(stamps, t))
        else:
            sfreq = float(series.rate)
            start = float(series.starting_time or 0.0)
            sample_of = lambda t: int(np.ceil(np.round((t - start) * sfreq, 6)))
        first = int(np.ceil(np.round(tmin * sfreq, 6)))
        n_times = int(np.ceil(np.round(tmax * sfreq, 6))) - first
        times = (first + np.arange(n_times)) / sfreq
        scale = float(series.conversion or 1.0)
        offset = float(getattr(series, "offset", 0.0) or 0.0)
        unit = (series.unit or "volts").lower()
        to_uv = 1e6 if unit in ("volts", "v", "volt") else 1.0
        epochs, kept = [], []
        contiguous = index == list(range(min(index), max(index) + 1))
        in_gap = 0
        for k, onset in enumerate(onsets):
            start_sample = sample_of(onset) + first
            stop_sample = start_sample + n_times
            if start_sample < 0 or stop_sample > n_total:
                continue
            if stamps is not None:
                # with timestamps, the recording may have gaps: the epoch must be there
                # and contiguous (an event inside a gap has no data)
                at = start_sample - first
                if abs(stamps[at] - onset) > 1.5 / sfreq or \
                        abs(stamps[stop_sample - 1] - stamps[start_sample]
                            - (n_times - 1) / sfreq) > 2.0 / sfreq:
                    in_gap += 1
                    continue
            if data.ndim == 1:
                block = np.asarray(data[start_sample:stop_sample], float)[:, None]
            elif contiguous:
                block = np.asarray(data[start_sample:stop_sample, min(index):max(index) + 1],
                                   float)
            else:
                block = np.asarray(data[start_sample:stop_sample], float)[:, index]
            epochs.append((block * scale + offset) * to_uv)
            kept.append(k)
        if in_gap:
            import warnings

            warnings.warn(f"{in_gap} event(s) fall in gaps of the recording (no samples) "
                          f"and were dropped.", UserWarning, stacklevel=2)
        if not epochs:
            raise DataError(f"No epoch fits inside the recording ({in_gap} event(s) in "
                            f"gaps of the timestamps).")
        metadata = frame.iloc[kept].reset_index(drop=True)
        report = None
        if stim_columns == "auto":
            metadata, report = _harmonise(metadata)
        source_info = {"path": os.fspath(path), "format": "nwb", "series": series.name,
                       "events": source, "reader": "evoparam.io.read_nwb",
                       "unit": unit, "conversion": scale}
        provenance = Provenance(source=source_info).with_step(
            "evoparam.io.read_nwb",
            {"electrical_series": series.name, "tmin": tmin, "tmax": tmax,
             "picks": names, "events_source": source, "select": select,
             "n_events": int(len(onsets)), "n_kept": int(len(kept)), "n_in_gap": in_gap},
            harmonization=getattr(report, "to_dict", lambda: None)() if report else None)
        return TrialSet(data=np.stack(epochs).transpose(0, 2, 1), times=times, sfreq=sfreq,
                        ch_names=names, units=units, metadata=metadata, ch_info=info,
                        event_times=onsets[kept], provenance=provenance)
    finally:
        io.close()


def write_nwb(result, nwbfile_or_path, *, module="evoparam", units="uV"):
    """Add a result's tables to ``processing/<module>`` of an NWB file (spec §10.3).

    ``result``: GroupFit (``evoparam_components``), ResponseTest (``evoparam_existence``),
    trial fits (``evoparam_trials``) or a list of them. The module description holds the
    settings and provenance as JSON (REQ-336). Amplitudes are converted to volts. A path
    is opened in append mode; an ``NWBFile`` object is modified in memory.
    """

    pynwb = require("pynwb", extra="nwb", caller="evoparam.io.write_nwb")
    from hdmf.common import DynamicTable

    from evoparam.data import GroupFit, ResponseTest
    from evoparam.data._jsonutil import to_jsonable

    items = list(result) if isinstance(result, (list, tuple)) else [result]
    io = None
    if isinstance(nwbfile_or_path, (str, os.PathLike)):
        io = pynwb.NWBHDF5IO(os.fspath(nwbfile_or_path), mode="a", load_namespaces=True)
        nwbfile = io.read()
    else:
        nwbfile = nwbfile_or_path
    to_volt = {"uV": 1e-6, "mV": 1e-3, "V": 1.0}.get(units, 1.0)
    description = {"settings": [to_jsonable(getattr(i, "settings", {})) for i in items],
                   "provenance": [getattr(getattr(i, "provenance", None), "to_dict",
                                          lambda: None)() for i in items]}
    if module in nwbfile.processing:
        raise DataError(f"The file already has a processing module {module!r}; pass "
                        f"another module= name.")
    mod = nwbfile.create_processing_module(module, json.dumps(description, allow_nan=False))

    def table(name, desc, columns, rows):
        tab = DynamicTable(name=name, description=desc)
        for column, column_desc in columns:
            tab.add_column(column, column_desc)
        for row in rows:
            tab.add_row(**{c: row[c] for c, _ in columns})
        return tab

    for item in items:
        if isinstance(item, GroupFit):
            rows = []
            for ch, cond, pos in item.index[["channel", "condition", "position"]].itertuples(
                    index=False):
                fit = item.fits[int(pos)]
                comps = fit.components or [None]
                for k, comp in enumerate(comps):
                    nan = float("nan")
                    rows.append({
                        "channel": ch, "condition": cond, "component":
                        (comp.label or f"c{k}") if comp else "", "shape": comp.shape if comp
                        else "", "status": fit.status,
                        "latency": comp.latency if comp else nan,
                        "latency_se": (comp.se or {}).get("latency", nan) if comp else nan,
                        "amplitude": comp.amplitude * to_volt if comp else nan,
                        "amplitude_se": (comp.se or {}).get("amplitude", nan) * to_volt
                        if comp else nan,
                        "polarity": int(comp.polarity) if comp else 0,
                        "fwhm": comp.fwhm if comp else nan,
                        "onset": comp.onset if comp else nan,
                        "offset": comp.offset if comp else nan,
                        "r2": fit.gof.r2 if fit.gof is not None else nan})
            columns = [("channel", "channel name"), ("condition", "condition"),
                       ("component", "component label"), ("shape", "shape function"),
                       ("status", "fit status"), ("latency", "latency (s)"),
                       ("latency_se", "latency standard error (s)"),
                       ("amplitude", "amplitude (V)"),
                       ("amplitude_se", "amplitude standard error (V)"),
                       ("polarity", "sign of the extremum"), ("fwhm", "width (s)"),
                       ("onset", "onset (s)"), ("offset", "offset (s)"),
                       ("r2", "goodness of fit")]
            mod.add(table("evoparam_components", "Fitted components", columns, rows))
        elif isinstance(item, ResponseTest):
            names = item.settings.get("channels") or [f"ch{i}" for i in
                                                      range(len(item.statistic))]
            rows = [{"channel": n, "method": item.method,
                     "statistic": float(item.statistic[i]), "p_value": float(item.p_value[i]),
                     "significant": bool(item.significant[i]), "onset": float(item.onset[i]),
                     "offset": float(item.offset[i]), "duration": float(item.duration[i]),
                     "snr_db": float(item.snr_db[i])} for i, n in enumerate(names)]
            columns = [("channel", "channel name"), ("method", "test method"),
                       ("statistic", "test statistic"), ("p_value", "corrected p-value"),
                       ("significant", "decision"), ("onset", "response onset (s)"),
                       ("offset", "response offset (s)"), ("duration", "duration (s)"),
                       ("snr_db", "signal-to-noise ratio (dB)")]
            mod.add(table("evoparam_existence", "Existence tests", columns, rows))
        elif hasattr(item, "table") and hasattr(item, "fits"):
            frame = item.table
            rows = [{"trial": int(r["trial"]), "channel": str(r["channel"]),
                     "component": str(r["label"] or r["component"]), "shift": float(r["shift"]),
                     "scale": float(r["scale"]), "latency": float(r["latency"]),
                     "amplitude": float(r["amplitude"]) * to_volt,
                     "converged": bool(r["converged"])} for r in frame.to_dict("records")]
            columns = [("trial", "trial index"), ("channel", "channel name"),
                       ("component", "component"), ("shift", "latency shift (s)"),
                       ("scale", "amplitude scale"), ("latency", "latency (s)"),
                       ("amplitude", "amplitude (V)"), ("converged", "fit converged")]
            mod.add(table("evoparam_trials", "Trial-by-trial parameters", columns, rows))
        else:
            raise DataError(f"write_nwb: cannot write a {type(item).__name__}.")
    if io is not None:
        io.write(nwbfile)
        io.close()
    return nwbfile
