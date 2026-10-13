# SPDX-License-Identifier: Apache-2.0
"""NWB reading and writing (spec §10.3, REQ-317, REQ-318; extra ``nwb``).

``read_nwb`` epochs one ``ElectricalSeries`` (or any ``TimeSeries`` whose data are
``(n_times, n_channels)``) around event times, read lazily through h5py (local files, or
http(s) addresses read by byte ranges with ``remfile``: a DANDI asset is read without
downloading it). ``write_nwb`` adds evoparam's tables to a processing module and
``read_nwb_results`` reads the results back.
"""

import json
import os
from typing import TYPE_CHECKING

import numpy as np

from evoparam._typing import Any, NDArray
from evoparam.errors import DataError, ReaderError, warn
from evoparam.utils._optional import require

if TYPE_CHECKING:
    from evoparam.data.containers import TrialSet

__all__ = ["read_nwb", "write_nwb", "read_nwb_results", "nwb_events", "EVENT_SOURCES"]

EVENT_SOURCES = ("events_table", "trials", "intervals", "stim_table", "icephys")
NWB_UNITS = {"amplitude": "A", "pulse_width": "s", "frequency": "Hz"}
NWB_MAPPING = {
    "amplitude": "stim_current_mA",
    "pulse_width": "stim_pulse_width_us",
    "bipolar_pair": "stim_site",
    "frequency": "stim_frequency_hz",
}
ICEPHYS_FRACTION = 0.1


def _open(path: Any) -> Any:
    """Open a local or remote NWB file for reading; return the pynwb IO object."""
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
        # opened here with h5py's default chunk cache (1 MiB per dataset): given a path,
        # hdmf opens the file with a larger cache, which added about 35 MB to a read of
        # epochs scattered over a 1 GB series (REQ-317); NWBHDF5IO.close closes this file
        h5py = require("h5py", extra="nwb", caller="evoparam.io.read_nwb")
        handle = h5py.File(path, "r")
        io = pynwb.NWBHDF5IO(file=handle, mode="r", load_namespaces=True)
    return io


def _all_objects(nwbfile: Any) -> list[Any]:
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


def _series(nwbfile: Any, name: Any) -> Any:
    """Return the TimeSeries named ``name``, else the first ElectricalSeries."""
    require("pynwb", extra="nwb", caller="evoparam.io.nwb._series")
    from pynwb.base import TimeSeries

    candidates = [
        (p, o)
        for p, o in _all_objects(nwbfile)
        if isinstance(o, TimeSeries)
        and getattr(o.data, "ndim", len(getattr(o.data, "shape", ()))) >= 1
    ]
    if name:
        for path, obj in candidates:
            if obj.name == name or path.endswith(name):
                return obj
        raise DataError(f"No TimeSeries named {name!r}; found {[p for p, _ in candidates]}.")
    electrical = [(p, o) for p, o in candidates if type(o).__name__ == "ElectricalSeries"]
    if not electrical:
        raise DataError(
            f"No ElectricalSeries in the file; name one of "
            f"{[p for p, _ in candidates]} with electrical_series=."
        )
    return electrical[0][1]


def _table_frame(table: Any) -> Any:
    """Return a table as a DataFrame of its scalar columns, with its ids."""
    frame = table.to_dataframe()
    keep = [c for c in frame.columns if frame[c].map(lambda v: np.isscalar(v) or v is None).all()]
    return frame[keep].reset_index(drop=False)


def _given_events(events: Any, pd: Any) -> Any:
    """Return the events given as a table or as onsets (None: a name, or no events given)."""
    if isinstance(events, pd.DataFrame):
        column = "onset" if "onset" in events else "start_time"
        return events[column].to_numpy(float), events.reset_index(drop=True), "table"
    if events is not None and not isinstance(events, str):
        onsets: Any
        onsets = np.asarray(events, float)
        return onsets, pd.DataFrame({"onset": onsets}), "array"
    return None


def _events_table_source(nwbfile: Any, events: Any, pd: Any) -> Any:
    """Return the events of an ndx-events EventsTable (None: none in the file)."""
    for _path, obj in _all_objects(nwbfile):
        if type(obj).__name__ in ("EventsTable", "AnnotatedEventsTable", "Events") and (
            events is None or obj.name == events
        ):
            if hasattr(obj, "to_dataframe"):
                frame = _table_frame(obj)
                column = "timestamp" if "timestamp" in frame else "onset"
                return frame[column].to_numpy(float), frame, "events_table"
            stamps = np.asarray(obj.timestamps[:], float)
            return stamps, pd.DataFrame({"onset": stamps}), "events_table"
    return None


def _trials_source(nwbfile: Any, events: Any, _pd: Any) -> Any:
    """Return the events of the trials table (None: no trials table, or another name)."""
    if nwbfile.trials is not None and (events in (None, "trials")):
        frame = _table_frame(nwbfile.trials)
        column = "stim_onset" if "stim_onset" in frame else "start_time"
        return frame[column].to_numpy(float), frame, "trials"
    return None


def _intervals_source(nwbfile: Any, events: Any, _pd: Any) -> Any:
    """Return the events of a TimeIntervals table under /intervals (None: none)."""
    intervals = dict(nwbfile.intervals) if nwbfile.intervals is not None else {}
    intervals.pop("trials", None)
    if events is not None and events in intervals:
        frame = _table_frame(intervals[events])
        return frame["start_time"].to_numpy(float), frame, "intervals"
    if events is None and intervals:
        name = sorted(intervals)[0]
        frame = _table_frame(intervals[name])
        return frame["start_time"].to_numpy(float), frame, "intervals"
    return None


def _stim_table_source(nwbfile: Any, _events: Any, _pd: Any) -> Any:
    """Return the events of a StimTable of ndx-electrical-stim (None: none)."""
    for _path, obj in _all_objects(nwbfile):
        if type(obj).__name__ == "StimTable":
            frame = _table_frame(obj)
            return frame["start_time"].to_numpy(float), frame, "stim_table"
    return None


def _icephys_source(nwbfile: Any, _events: Any, pd: Any) -> Any:
    """Return the onsets of the intracellular stimulus series (None: none)."""
    stimuli = list(nwbfile.stimulus.values()) if nwbfile.stimulus else []
    onsets: Any = []
    for series in stimuli:
        if type(series).__name__ not in (
            "CurrentClampStimulusSeries",
            "VoltageClampStimulusSeries",
        ):
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
    return None


def nwb_events(nwbfile: Any, events: Any = None) -> Any:
    """Return (onsets in s, events table, source name) by the order of spec §10.3."""
    require("pandas", extra="data", caller="evoparam.io.nwb.nwb_events")
    import pandas as pd

    found = _given_events(events, pd)
    if found is not None:
        return found
    # (1) ndx-events EventsTable, (2) trials, (3) TimeIntervals under /intervals,
    # (4) StimTable of ndx-electrical-stim, (5) intracellular stimulus series
    for source in (
        _events_table_source,
        _trials_source,
        _intervals_source,
        _stim_table_source,
        _icephys_source,
    ):
        found = source(nwbfile, events, pd)
        if found is not None:
            return found
    raise DataError(
        "No event source in the file (EventsTable, trials, intervals, StimTable, "
        "intracellular stimuli); pass events= (a table name, onsets or a table)."
    )


def _harmonise(frame: Any) -> tuple[Any, Any]:
    """Harmonise the stimulation columns of an NWB events table."""
    from ._harmonize import harmonize_stim_events

    sidecar = {c: {"Units": u} for c, u in NWB_UNITS.items() if c in frame}
    mapping = {c: t for c, t in NWB_MAPPING.items() if c in frame}
    table, report = harmonize_stim_events(
        frame.rename(columns={"start_time": "onset"}) if "onset" not in frame else frame,
        sidecar,
        mapping=mapping,
    )
    return table, report


def _electrodes(_nwbfile: Any, series: Any, picks: Any) -> tuple[list[str], Any, list[int]]:
    """Return the channel names, the channel table and the column index of the picks."""
    require("pandas", extra="data", caller="evoparam.io.nwb._electrodes")
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
    info = info[
        [c for c in info.columns if info[c].map(lambda v: np.isscalar(v) or v is None).all()]
    ]
    info.insert(0, "name", names) if "name" not in info else None
    index = list(range(len(names)))
    if picks is not None:
        index = [names.index(p) if isinstance(p, str) else int(p) for p in picks]
    return [names[i] for i in index], info.iloc[index].reset_index(drop=True), index


class _Clock:
    """Sample times of a series: timestamps, or a rate and a starting time."""

    def __init__(self, series: Any) -> None:
        """Read the timing of ``series``."""
        self.stamps = None
        self.start = 0.0
        if series.timestamps is not None:
            self.stamps = np.asarray(series.timestamps[:], float)
            self.sfreq = float(1.0 / np.median(np.diff(self.stamps)))
        else:
            self.sfreq = float(series.rate)
            self.start = float(series.starting_time or 0.0)

    def sample_of(self, t: float) -> int:
        """Return the sample at time ``t`` (s)."""
        if self.stamps is not None:
            return int(np.searchsorted(self.stamps, t))
        return int(np.ceil(np.round((t - self.start) * self.sfreq, 6)))

    def in_gap(self, onset: float, start_sample: int, window: tuple[int, int]) -> bool:
        """Return True when the epoch of ``onset`` is not contiguous in the timestamps.

        With timestamps, the recording may have gaps: the epoch must be there and contiguous
        (an event inside a gap has no data).
        """
        stamps, sfreq = self.stamps, self.sfreq
        if stamps is None:
            return False
        first, n_times = window
        stop_sample = start_sample + n_times
        at = start_sample - first
        return bool(
            abs(stamps[at] - onset) > 1.5 / sfreq
            or abs(stamps[stop_sample - 1] - stamps[start_sample] - (n_times - 1) / sfreq)
            > 2.0 / sfreq
        )

    def epochs(
        self, onsets: Any, window: tuple[int, int], n_total: int
    ) -> tuple[list[int], list[int], int]:
        """Return the events whose epoch is in the recording, and their first samples.

        The third value is the number of events that fall in gaps of the timestamps.
        """
        first, n_times = window
        kept, starts = [], []
        in_gap = 0
        for k, onset in enumerate(onsets):
            start_sample = self.sample_of(onset) + first
            stop_sample = start_sample + n_times
            if start_sample < 0 or stop_sample > n_total:
                continue
            if self.in_gap(onset, start_sample, window):
                in_gap += 1
                continue
            kept.append(k)
            starts.append(start_sample)
        return kept, starts, in_gap


def _read_epochs(
    data: Any, starts: list[int], index: list[int], n_times: int, contiguous: bool
) -> NDArray[Any]:
    """Return the epochs ``[start, start + n_times)`` of the channels ``index``.

    ``contiguous``: the channels are a range of columns, read as one slice.
    """
    out = np.empty((len(starts), len(index), n_times))
    for i, start_sample in enumerate(starts):
        stop_sample = start_sample + n_times
        if data.ndim == 1:
            out[i, 0] = data[start_sample:stop_sample]
        elif contiguous:
            out[i] = data[start_sample:stop_sample, min(index) : max(index) + 1].T
        else:
            out[i] = np.asarray(data[start_sample:stop_sample])[:, index].T
    return out


def read_nwb(  # noqa: PLR0913  (spec §5.1 fixes 7 parameters, plus select and units; DECISIONS.md §7.2)
    path: str,
    *,
    electrical_series: str | None = None,
    events: Any = None,
    tmin: float = -0.1,
    tmax: float = 0.5,
    picks: Any = None,
    stim_columns: Any = "auto",
    select: str | None = None,
    units: str = "uV",
) -> "TrialSet":
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
        clock = _Clock(series)
        sfreq = clock.sfreq
        first = int(np.ceil(np.round(tmin * sfreq, 6)))
        n_times = int(np.ceil(np.round(tmax * sfreq, 6))) - first
        times = (first + np.arange(n_times)) / sfreq
        scale = float(series.conversion or 1.0)
        offset = float(getattr(series, "offset", 0.0) or 0.0)
        unit = (series.unit or "volts").lower()
        to_uv = 1e6 if unit in ("volts", "v", "volt") else 1.0
        # first the events whose epoch is in the recording, then one array filled in
        # place: the memory used is the epochs and one block (REQ-317)
        contiguous = index == list(range(min(index), max(index) + 1))
        kept, starts, in_gap = clock.epochs(onsets, (first, n_times), data.shape[0])
        if in_gap:
            warn(
                f"{in_gap} event(s) fall in gaps of the recording (no samples) and were dropped.",
                UserWarning,
                stacklevel=2,
            )
        if not kept:
            raise DataError(
                f"No epoch fits inside the recording ({in_gap} event(s) in "
                f"gaps of the timestamps)."
            )
        out = _read_epochs(data, starts, index, n_times, contiguous)
        out *= scale * to_uv
        if offset:
            out += offset * to_uv
        metadata = frame.iloc[kept].reset_index(drop=True)
        report = None
        if stim_columns == "auto":
            metadata, report = _harmonise(metadata)
        source_info = {
            "path": os.fspath(path),
            "format": "nwb",
            "series": series.name,
            "events": source,
            "reader": "evoparam.io.read_nwb",
            "unit": unit,
            "conversion": scale,
        }
        provenance = Provenance(source=source_info).with_step(
            "evoparam.io.read_nwb",
            {
                "electrical_series": series.name,
                "tmin": tmin,
                "tmax": tmax,
                "picks": names,
                "events_source": source,
                "select": select,
                "n_events": int(len(onsets)),
                "n_kept": int(len(kept)),
                "n_in_gap": in_gap,
            },
            harmonization=getattr(report, "to_dict", lambda: None)() if report else None,
        )
        return TrialSet(
            data=out,
            times=times,
            sfreq=sfreq,
            ch_names=names,
            units=units,
            metadata=metadata,
            ch_info=info,
            event_times=onsets[kept],
            provenance=provenance,
        )
    finally:
        io.close()


def _electrode_rows(nwbfile: Any) -> Any:
    """Map each channel name to its row of ``/general/extracellular_ephys/electrodes``.

    With the naming of :func:`read_nwb` (``label`` column, else ``e<id>``).
    """
    table = getattr(nwbfile, "electrodes", None)
    if table is None or len(table) == 0:
        return {}
    frame = table.to_dataframe()
    if "label" in frame and frame["label"].notna().all():
        names = [str(v) for v in frame["label"]]
    else:
        names = [f"e{int(v)}" for v in frame.index]
    return {name: row for row, name in enumerate(names)}


def _pack(result: Any) -> str:
    """Return the result as the bytes of a zipped ``.evoparam`` container, base64 text."""
    import base64
    import tempfile

    from evoparam.data.serialize import save_object

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "result.evoparam.zip")
        if (
            hasattr(result, "fits")
            and hasattr(result, "table")
            and type(result).__name__ == "TrialFitSet"
        ):
            raise DataError("internal: trial fit sets are packed fit by fit.")
        save_object(result, path)
        with open(path, "rb") as fobj:
            return base64.b64encode(fobj.read()).decode("ascii")


def _unpack(text: Any, verify: bool = True) -> Any:
    """Return the result held in a base64 text of a zipped container."""
    import base64
    import tempfile

    from evoparam.data.serialize import load_object

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "result.evoparam.zip")
        with open(path, "wb") as fobj:
            fobj.write(base64.b64decode(text.encode("ascii")))
        return load_object(path, verify=verify)


def _stim_site(trials: Any, condition: Any) -> Any:
    """Return the stimulation site of a condition ("" for none)."""
    from evoparam.io.derivatives import _stim_fields

    return _stim_fields(trials, condition, ())["stim_site"] or ""


def _component_rows(group: Any, test: Any, trials: Any, to_volt: Any) -> Any:
    """Return the rows of ``evoparam_components``: one per channel, condition and component."""
    present = {}
    if test is not None:
        names = test.settings.get("channels") or []
        conditions = test.settings.get("conditions") or [""] * len(names)
        for name, cond, sig in zip(names, conditions, np.ravel(test.significant), strict=False):
            present[(name, cond)] = "true" if sig else "false"
    nan = float("nan")
    rows = []
    for ch, cond, pos in group.index[["channel", "condition", "position"]].itertuples(index=False):
        fit = group.fits[int(pos)]
        n_trials = (fit.settings or {}).get("n_trials")
        ci_level = (fit.settings or {}).get("ci_level", group.settings.get("ci_level", nan))
        for k, comp in enumerate(fit.components or [None]):
            rows.append(
                {
                    "channel": str(ch),
                    "stim_site": _stim_site(trials, cond),
                    "condition": str(cond or ""),
                    "component": (comp.label or f"c{k}") if comp else "",
                    "shape": comp.shape if comp else "",
                    "status": fit.status,
                    "present": present.get((ch, cond), present.get((ch, ""), "n/a")),
                    "latency": comp.latency if comp else nan,
                    "latency_se": (comp.se or {}).get("latency", nan) if comp else nan,
                    "amplitude": comp.amplitude * to_volt if comp else nan,
                    "amplitude_se": (comp.se or {}).get("amplitude", nan) * to_volt
                    if comp
                    else nan,
                    "polarity": int(comp.polarity) if comp else 0,
                    "resolvable": "true" if comp is None or comp.resolvable else "false",
                    "fwhm": comp.fwhm if comp else nan,
                    "onset": comp.onset if comp else nan,
                    "offset": comp.offset if comp else nan,
                    "area": comp.area * to_volt if comp else nan,
                    "r2": fit.gof.r2 if fit.gof is not None else nan,
                    "n_trials": int(n_trials) if n_trials is not None else -1,
                    "ci_level": float(ci_level) if ci_level is not None else nan,
                }
            )
    return rows


COMPONENT_TABLE = [
    ("channel", "channel name"),
    ("stim_site", "stimulation site (empty: none)"),
    ("condition", "condition"),
    ("component", "component label"),
    ("shape", "shape function"),
    ("status", "fit status: ok, no_response, failed, skipped"),
    ("present", "existence decision: true, false or n/a"),
    ("latency", "latency (s)"),
    ("latency_se", "latency standard error (s)"),
    ("amplitude", "amplitude (V)"),
    ("amplitude_se", "amplitude standard error (V)"),
    ("polarity", "sign of the extremum"),
    ("resolvable", "false: trades off with a neighbour, not reliable alone"),
    ("fwhm", "full width at half maximum (s)"),
    ("onset", "onset (s)"),
    ("offset", "offset (s)"),
    ("area", "area (V*s)"),
    ("r2", "goodness of fit"),
    ("n_trials", "trials averaged (-1: unknown)"),
    ("ci_level", "level of the intervals"),
]
EXISTENCE_TABLE = [
    ("channel", "channel name"),
    ("stim_site", "stimulation site (empty: none)"),
    ("condition", "condition"),
    ("method", "test method"),
    ("statistic", "test statistic"),
    ("p_value", "corrected p-value"),
    ("significant", "decision"),
    ("onset", "response onset (s)"),
    ("offset", "response offset (s)"),
    ("duration", "duration (s)"),
    ("snr_db", "signal-to-noise ratio (dB)"),
]
TRIAL_TABLE = [
    ("trial_index", "trial index in the source run"),
    ("channel", "channel name"),
    ("component", "component"),
    ("shift", "latency shift (s)"),
    ("scale", "amplitude scale"),
    ("latency", "latency (s)"),
    ("amplitude", "amplitude (V)"),
    ("converged", "fit converged"),
]


class _NwbWriter:
    """Writer of the evoparam tables into one processing module of an NWB file."""

    def __init__(self, nwbfile: Any, mod: Any, ndx: Any, context: dict[str, Any]) -> None:
        """Write into the module ``mod`` of ``nwbfile``; ``ndx``: the extension module or None.

        ``context`` holds ``items`` (the results), ``trials``, ``to_volt``, ``model``, the
        ``pynwb`` module and the ``DynamicTable`` and ``DynamicTableRegion`` classes of hdmf.
        """
        self.nwbfile = nwbfile
        self.mod = mod
        self.ndx = ndx
        self.electrodes = _electrode_rows(nwbfile)
        self.items = context["items"]
        self.trials = context["trials"]
        self.to_volt = context["to_volt"]
        self.model = context["model"]
        self.pynwb = context["pynwb"]
        self.dynamic_table = context["DynamicTable"]
        self.dynamic_table_region = context["DynamicTableRegion"]
        self.stored: list[Any] = []

    def table(self, name: Any, desc: Any, columns: Any, rows: Any, kind: Any) -> Any:
        """Return a table of ``rows``, with the electrode and trial references it can have."""
        nwbfile, electrodes = self.nwbfile, self.electrodes
        cls = self.dynamic_table if self.ndx is None else self.ndx.table_class(kind)
        tab = cls(name=name, description=desc)
        names = [r["channel"] for r in rows]
        region = bool(rows) and bool(electrodes) and all(n in electrodes for n in names)
        for column, column_desc in columns:
            if column not in tab.colnames:
                tab.add_column(column, column_desc, index=column in ("params", "param_names"))
        if region and "electrode" not in tab.colnames:
            tab.add_column("electrode", "electrode of the channel", table=nwbfile.electrodes)
        if (
            kind == "trials"
            and "trial" not in tab.colnames
            and rows
            and nwbfile.trials is not None
            and max(r["trial_index"] for r in rows) < len(nwbfile.trials)
        ):
            tab.add_column("trial", "trial of the file's trials table", table=nwbfile.trials)
        for row in rows:
            values = {c: row[c] for c, _ in columns}
            if "electrode" in tab.colnames:
                values["electrode"] = electrodes[row["channel"]]
            if "trial" in tab.colnames:
                values["trial"] = int(row["trial_index"])
            tab.add_row(**values)
        return tab

    def add(self, item: Any) -> None:
        """Add the table(s) of one result."""
        from evoparam.data import GroupFit, ResponseTest

        if isinstance(item, GroupFit):
            self.add_group(item)
        elif isinstance(item, ResponseTest):
            self.add_test(item)
        elif hasattr(item, "table") and hasattr(item, "fits"):
            self.add_trial_fits(item)
        else:
            raise DataError(f"write_nwb: cannot write a {type(item).__name__}.")

    def add_group(self, item: Any) -> None:
        """Add ``evoparam_components`` (and ``evoparam_model``) of a group fit."""
        from evoparam.data import ResponseTest

        test = next((i for i in self.items if isinstance(i, ResponseTest)), None)
        rows = _component_rows(item, test, self.trials, self.to_volt)
        columns = list(COMPONENT_TABLE)
        if self.ndx is not None:
            columns += _add_shape_params(item, rows)
        self.mod.add(
            self.table(
                "evoparam_components",
                "Fitted components (evoparam)",
                columns,
                rows,
                "components",
            )
        )
        self.stored.append(("GroupFit", item))
        if self.model:
            series = _model_series(
                item,
                self.nwbfile,
                self.electrodes,
                self.to_volt,
                self.pynwb,
                self.dynamic_table_region,
            )
            if series is not None:
                self.mod.add(series)

    def add_test(self, item: Any) -> None:
        """Add ``evoparam_existence`` of a response test."""
        names = item.settings.get("channels") or [f"ch{i}" for i in range(len(item.statistic))]
        conditions = item.settings.get("conditions") or [""] * len(names)
        rows = [
            {
                "channel": str(n),
                "stim_site": _stim_site(self.trials, c),
                "condition": str(c or ""),
                "method": item.method,
                "statistic": float(item.statistic[i]),
                "p_value": float(item.p_value[i]),
                "significant": bool(item.significant[i]),
                "onset": float(item.onset[i]),
                "offset": float(item.offset[i]),
                "duration": float(item.duration[i]),
                "snr_db": float(item.snr_db[i]),
            }
            for i, (n, c) in enumerate(zip(names, conditions, strict=False))
        ]
        self.mod.add(
            self.table(
                "evoparam_existence",
                "Existence tests (evoparam)",
                list(EXISTENCE_TABLE),
                rows,
                "existence",
            )
        )
        self.stored.append(("ResponseTest", item))

    def add_trial_fits(self, item: Any) -> None:
        """Add ``evoparam_trials`` of trial fits."""
        rows = [
            {
                "trial_index": int(r["trial"]),
                "channel": str(r["channel"]),
                "component": str(r["label"] or r["component"]),
                "shift": float(r["shift"]),
                "scale": float(r["scale"]),
                "latency": float(r["latency"]),
                "amplitude": float(r["amplitude"]) * self.to_volt,
                "converged": bool(r["converged"]),
            }
            for r in item.table.to_dict("records")
        ]
        self.mod.add(
            self.table(
                "evoparam_trials",
                "Trial-by-trial parameters (evoparam)",
                list(TRIAL_TABLE),
                rows,
                "trials",
            )
        )
        self.stored += [("TrialFit", fit) for fit in item.fits]

    def add_results(self) -> None:
        """Add ``evoparam_results``: each stored result as a zipped container."""
        from evoparam.version import __version__

        results = self.dynamic_table(
            name="evoparam_results",
            description="Each result as a zipped .evoparam container (base64), "
            "read back exactly by evoparam.io.read_nwb_results",
        )
        results.add_column("result_type", "class of the result")
        results.add_column("package_version", "evoparam version that wrote it")
        results.add_column("container", "zipped .evoparam container, base64")
        for kind, obj in self.stored:
            results.add_row(result_type=kind, package_version=__version__, container=_pack(obj))
        self.mod.add(results)


def _add_shape_params(group: Any, rows: list[Any]) -> list[tuple[str, str]]:
    """Add the shape parameters of each component to its row; return the two new columns."""
    params = []
    for _ch, _cond, pos in group.index[["channel", "condition", "position"]].itertuples(
        index=False
    ):
        for comp in group.fits[int(pos)].components or [None]:
            params.append(comp)
    for row, comp in zip(rows, params, strict=False):
        row["param_names"] = list(comp.param_names) if comp else []
        row["params"] = [float(v) for v in comp.params] if comp else []
    return [
        ("param_names", "names of the shape parameters (ragged)"),
        ("params", "shape parameters (ragged; times in s)"),
    ]


def _module_description(items: list[Any]) -> dict[str, Any]:
    """Return the settings, provenance and methods paragraph of each result."""
    from evoparam.data._jsonutil import to_jsonable

    return {
        "settings": [to_jsonable(getattr(i, "settings", {})) for i in items],
        "provenance": [
            getattr(getattr(i, "provenance", None), "to_dict", lambda: None)() for i in items
        ],
        # the methods paragraph of each result (spec §11.8, REQ-390)
        "methods": [
            i.methods_text() if callable(getattr(i, "methods_text", None)) else None for i in items
        ],
    }


def write_nwb(
    result: Any,
    nwbfile_or_path: Any,
    *,
    module: str = "evoparam",
    extension: bool = False,
    units: str = "uV",
    trials: "TrialSet | None" = None,
    model: bool = True,
) -> Any:
    """Add a result's tables to ``processing/<module>`` of an NWB file (spec §10.3).

    Parameters
    ----------
    result : GroupFit, ResponseTest, trial fits, or a list of them
        ``evoparam_components`` (GroupFit), ``evoparam_existence`` (ResponseTest) and
        ``evoparam_trials`` (trial fits). A TrialSet in the list is used as ``trials``.
    nwbfile_or_path : NWBFile or str
        A path is opened in append mode; an ``NWBFile`` object is modified in memory.
    module : str
        Name of the processing module.
    extension : bool
        Write the same content with the ``ndx-evoparam`` namespace
        (:mod:`evoparam.io.ndx_evoparam`): typed tables with the shape parameters, and a
        ``ResponseFitSettings`` record.
    units : str
        Amplitude unit of the results (amplitudes are written in V).
    trials : TrialSet, optional
        The fitted trials: their metadata give ``stim_site``.
    model : bool
        Add ``evoparam_model``, an ``ElectricalSeries`` of the fitted curves (in V), when
        every channel is a row of the file's electrodes table.

    Every table has an ``electrode`` column (``DynamicTableRegion`` into the electrodes
    table) when the channels are electrodes of the file; ``evoparam_trials`` has ``trial``
    (``DynamicTableRegion`` into ``trials``) when the file has a trials table covering the
    indices. The module description holds the settings, provenance (REQ-336) and methods
    paragraph (REQ-390) of each result as JSON; the table ``evoparam_results`` holds each
    result as a zipped ``.evoparam`` container, so that :func:`read_nwb_results` gives back
    the exact objects.
    """
    pynwb = require("pynwb", extra="nwb", caller="evoparam.io.write_nwb")
    require("hdmf", extra="nwb", caller="evoparam.io.nwb.write_nwb")
    from hdmf.common import DynamicTable, DynamicTableRegion

    from evoparam.data import TrialSet
    from evoparam.version import __version__

    items = list(result) if isinstance(result, (list, tuple)) else [result]
    trials = (
        trials if trials is not None else next((r for r in items if isinstance(r, TrialSet)), None)
    )
    items = [r for r in items if not isinstance(r, TrialSet)]
    ndx: Any = None
    if extension:
        # loaded before the file is opened: NWBHDF5IO copies the type map it starts with
        from evoparam.io import ndx_evoparam as ndx

        ndx.load()
    io = None
    if isinstance(nwbfile_or_path, (str, os.PathLike)):
        io = pynwb.NWBHDF5IO(os.fspath(nwbfile_or_path), mode="a", load_namespaces=True)
        nwbfile = io.read()
    else:
        nwbfile = nwbfile_or_path
    to_volt = {"uV": 1e-6, "mV": 1e-3, "V": 1.0}.get(units)
    if to_volt is None:
        raise DataError(f"write_nwb: units must be uV, mV or V, not {units!r}.")
    description = _module_description(items)
    if module in nwbfile.processing:
        raise DataError(
            f"The file already has a processing module {module!r}; pass another module= name."
        )
    mod = nwbfile.create_processing_module(module, json.dumps(description, allow_nan=False))
    writer = _NwbWriter(
        nwbfile,
        mod,
        ndx,
        {
            "items": items,
            "trials": trials,
            "to_volt": to_volt,
            "model": model,
            "pynwb": pynwb,
            "DynamicTable": DynamicTable,
            "DynamicTableRegion": DynamicTableRegion,
        },
    )
    for item in items:
        writer.add(item)
    writer.add_results()
    if ndx is not None:
        nwbfile.add_lab_meta_data(
            ndx.settings_record(items, description, __version__, name=f"{module}_settings")
        )
    if io is not None:
        io.write(nwbfile)
        io.close()
    return nwbfile


def _model_series(
    group: Any, nwbfile: Any, electrodes: Any, to_volt: Any, pynwb: Any, dynamic_table_region: Any
) -> Any:
    """``evoparam_model``.

    The fitted curves (fit window, zero elsewhere) of every channel x condition, in V, at the rate
    of the fits.
    """
    import warnings

    channels = [str(c) for c in group.index["channel"]]
    fits = [group.fits[int(p)] for p in group.index["position"]]
    if (
        not electrodes
        or not all(c in electrodes for c in channels)
        or not any(f.model is not None and len(f.model) > 1 for f in fits)
    ):
        return None
    ref = next(f for f in fits if f.model is not None and len(f.model) > 1)
    rate = (len(ref.model) - 1) / (ref.window[1] - ref.window[0])
    start = min(f.window[0] for f in fits)
    stop = max(f.window[1] for f in fits)
    times = start + np.arange(int(round((stop - start) * rate)) + 1) / rate
    data = np.zeros((times.size, len(fits)))
    for i, fit in enumerate(fits):
        if fit.model is not None and len(fit.model) > 1:
            grid = np.linspace(fit.window[0], fit.window[1], len(fit.model))
            inside = (times >= fit.window[0] - 1e-12) & (times <= fit.window[1] + 1e-12)
            data[inside, i] = np.interp(times[inside], grid, fit.model) * to_volt
    region = dynamic_table_region(
        name="electrodes",
        data=[electrodes[c] for c in channels],
        description="channel of each column (one column per channel x condition)",
        table=nwbfile.electrodes,
    )
    conditions = [str(c) for c in group.index["condition"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pynwb.ecephys.ElectricalSeries(
            name="evoparam_model",
            data=data,
            electrodes=region,
            rate=float(rate),
            starting_time=float(times[0]),
            conversion=1.0,
            description="evoparam fitted curves in V, time relative to the stimulus; columns: "
            + ", ".join(
                f"{c}|{d}" if d else c for c, d in zip(channels, conditions, strict=False)
            ),
        )


def _verify_components(mod: Any, found: dict[str, list[Any]]) -> None:
    """Check that ``evoparam_components`` still holds the values of the stored GroupFit."""
    group = found["GroupFit"][0]
    test = found["ResponseTest"][0] if found["ResponseTest"] else None
    stored = mod["evoparam_components"].to_dataframe()
    unit = 1.0
    expected = _component_rows(group, test, None, 1.0)
    if len(expected) != len(stored):
        raise DataError(
            "evoparam_components does not match the stored GroupFit (rows "
            "differ): the table was edited."
        )
    amplitudes = stored["amplitude"].to_numpy(float)
    reference = np.array([r["amplitude"] for r in expected], float)
    finite = np.isfinite(reference)
    if finite.any():
        ratio = amplitudes[finite] / np.where(reference[finite] == 0, 1, reference[finite])
        unit = float(np.median(ratio))
    for column in ("latency", "fwhm", "onset", "offset", "r2"):
        a = stored[column].to_numpy(float)
        b = np.array([r[column] for r in expected], float)
        if not np.allclose(a, b, rtol=1e-12, atol=0, equal_nan=True):
            raise DataError(
                f"evoparam_components column {column!r} does not match "
                f"the stored GroupFit: the table was edited."
            )
    if not np.allclose(amplitudes, reference * unit, rtol=1e-9, equal_nan=True):
        raise DataError(
            "evoparam_components column 'amplitude' does not match the "
            "stored GroupFit: the table was edited."
        )
    if list(stored["component"]) != [r["component"] for r in expected]:
        raise DataError("evoparam_components labels do not match the stored GroupFit.")


def read_nwb_results(
    path: Any, *, module: str = "evoparam", kind: str = "components", verify: bool = True
) -> Any:
    """Read back what :func:`write_nwb` wrote (REQ-318).

    Parameters
    ----------
    path : str or NWBFile
    module : str
        Name of the processing module.
    kind : str
        ``"components"`` (the GroupFit), ``"existence"`` (the ResponseTest), ``"trials"``
        (the TrialFits) or ``"all"`` (a dict of the three).
    verify : bool
        Check the containers' checksums and that ``evoparam_components`` still holds the
        values of the GroupFit (raises DataError when it was edited).

    Returns
    -------
    The result(s), equal to those written.
    """
    require("pynwb", extra="nwb", caller="evoparam.io.read_nwb_results")
    kinds = {"components": "GroupFit", "existence": "ResponseTest", "trials": "TrialFit"}
    if kind not in tuple(kinds) + ("all",):
        raise DataError(f"kind must be one of {tuple(kinds) + ('all',)}.")
    io = None
    if isinstance(path, (str, os.PathLike)):
        io = _open(path)
        nwbfile = io.read()
    else:
        nwbfile = path
    try:
        if module not in nwbfile.processing:
            raise ReaderError(f"No processing module {module!r} in the NWB file.")
        mod = nwbfile.processing[module]
        if "evoparam_results" not in mod.data_interfaces:
            raise ReaderError(
                "The module has no evoparam_results table (written by evoparam "
                "before 0.1 of the NWB round trip?)."
            )
        frame = mod["evoparam_results"].to_dataframe()
        found: dict[str, list[Any]]
        found = {name: [] for name in kinds.values()}
        for kind_name, text in zip(frame["result_type"], frame["container"], strict=False):
            found.setdefault(kind_name, []).append(_unpack(text, verify=verify))
        if verify and found["GroupFit"] and "evoparam_components" in mod.data_interfaces:
            _verify_components(mod, found)
        out = {
            "components": found["GroupFit"][0] if found["GroupFit"] else None,
            "existence": found["ResponseTest"][0] if found["ResponseTest"] else None,
            "trials": found["TrialFit"],
        }
    finally:
        if io is not None:
            io.close()
    if kind == "all":
        return out
    if not out[kind]:
        raise ReaderError(f"No {kind} result in module {module!r}.")
    return out[kind]
