"""Conversions between evoparam containers and MNE-Python objects (spec §10.1, REQ-310).

MNE stores voltages in V; evoparam in µV (``units``). The conversion keeps the channel
types (``ch_info.type``), the bad channels (``ch_info.bad`` <-> ``info["bads"]``), the
channel positions (``ch_info`` x, y, z in mm <-> ``info["chs"][i]["loc"][:3]`` in m, with
the coordinate frame), the metadata (``epochs.metadata`` plus the event names in
``condition``) and the event times. The ``mne`` package is imported lazily (extra ``mne``).
"""

import numpy as np

from evoparam.errors import DataError

__all__ = ["trialset_from_mne", "trialset_to_mne", "evoked_from_mne", "evoked_to_mne"]

VOLTS = {"eeg": "eeg", "ecog": "ecog", "seeg": "seeg", "dbs": "dbs", "emg": "emg",
         "eog": "eog", "ecg": "ecg", "misc": "misc"}
FACTORS = {"uV": 1e6, "mV": 1e3, "V": 1.0}
HEAD_FRAME = 4  # mne.io.constants.FIFF.FIFFV_COORD_HEAD


def _mne():
    try:
        import mne
    except ImportError as err:
        raise ImportError("MNE conversions need the extra 'mne': pip install "
                          "\"evoparam[mne]\".") from err
    return mne


def _channel_table(info, names):
    import pandas as pd

    mne = _mne()
    types = info.get_channel_types()
    for kind in types:
        if kind not in VOLTS:
            raise DataError(f"Channel type {kind!r} is not a voltage; pick the voltage "
                            f"channels first (obj.copy().pick(...)).")
    coords = np.array([ch["loc"][:3] for ch in info["chs"]], dtype=float) * 1e3
    frames = {int(ch.get("coord_frame", 0)) for ch in info["chs"]}
    space = "head" if frames == {HEAD_FRAME} else (
        mne.transforms._frame_to_str.get(frames.pop(), "") if len(frames) == 1 else "")
    finite = np.isfinite(coords).all(axis=1) & np.any(coords != 0, axis=1)
    return pd.DataFrame({
        "name": names, "type": types,
        "x": np.where(finite, coords[:, 0], np.nan), "y": np.where(finite, coords[:, 1], np.nan),
        "z": np.where(finite, coords[:, 2], np.nan),
        "coord_space": [space if f else "" for f in finite],
        "bad": [name in info["bads"] for name in names]})


def trialset_from_mne(cls, epochs, *, picks=None, units="uV", metadata=True,
                      stim_columns="auto"):
    """``TrialSet`` from ``mne.Epochs`` (``TrialSet.from_mne``)."""

    import pandas as pd

    from .provenance import Provenance

    mne = _mne()
    if not isinstance(epochs, mne.BaseEpochs):
        raise DataError(f"from_mne expects mne.Epochs, got {type(epochs).__name__}.")
    if picks is not None:
        epochs = epochs.copy().pick(picks)
    factor = FACTORS.get(units)
    if factor is None:
        raise DataError(f"units must be one of {sorted(FACTORS)}.")
    names = list(epochs.ch_names)
    sfreq = float(epochs.info["sfreq"])
    inverse = {code: name for name, code in epochs.event_id.items()}
    conditions = [inverse.get(int(code), str(code)) for code in epochs.events[:, 2]]
    table = pd.DataFrame({"condition": conditions})
    if metadata and epochs.metadata is not None:
        extra = epochs.metadata.reset_index(drop=True)
        table = pd.concat([extra, table], axis=1) if "condition" not in extra else extra
    # MNE event samples count from the start of the acquisition: they already include
    # first_samp (mne.find_events, Epochs.events), so the time of an event is
    # events[:, 0] / sfreq. Adding first_samp / sfreq again (the formula of spec §10.1)
    # would shift every event by the time between acquisition start and the first kept
    # sample, and to_mne would not give the events back (CHANGELOG, step S17).
    event_times = epochs.events[:, 0] / sfreq
    source = {"format": "mne", "object": type(epochs).__name__, "factor_to_units": factor,
              "condition_from_event_id": not (metadata and epochs.metadata is not None and
                                              "condition" in epochs.metadata),
              "first_samp": int(getattr(epochs, "_raw_first_samp", 0) or 0)}
    if stim_columns is not None and metadata and epochs.metadata is not None:
        table, source = _harmonize_metadata(table, stim_columns, source)
    provenance = Provenance(source=source).with_step("evoparam.data.TrialSet.from_mne",
                                                     {"units": units, "picks": picks})
    return cls(data=epochs.get_data(copy=True) * factor, times=np.array(epochs.times),
               sfreq=sfreq, ch_names=names, units=units, metadata=table,
               ch_info=_channel_table(epochs.info, names), event_times=event_times,
               provenance=provenance)


def _harmonize_metadata(table, stim_columns, source):
    """Standard stimulation columns (spec Table 4.1, REQ-313) from the metadata of the epochs:
    ``"auto"`` maps the recognised columns, a dict is a ``{source: target}`` mapping."""

    import warnings

    from ._harmonize import SOURCES, _find_column, harmonize_stim_events

    if isinstance(stim_columns, dict):
        mapping = dict(stim_columns)
    elif stim_columns == "auto":
        mapping = None
        if not any(_find_column(table.columns, cands) for target, cands in SOURCES.items()
                   if target != "condition"):
            return table, source
    else:
        raise DataError("stim_columns must be 'auto', a dict {column: target} or None.")
    before = list(table.columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out, report = harmonize_stim_events(table, None, mapping=mapping)
    if "condition" in table:
        out["condition"] = table["condition"].to_numpy()     # the event names, unchanged
    source = dict(source, harmonization=report.to_dict(),
                  added_columns=[c for c in out.columns if c not in before])
    return out, source


def _info(ch_names, sfreq, ch_info, ch_types):
    mne = _mne()
    if ch_types is None:
        ch_types = list(ch_info["type"]) if ch_info is not None and "type" in ch_info \
            else "eeg"
    info = mne.create_info(list(ch_names), float(sfreq), ch_types)
    if ch_info is not None:
        if "bad" in ch_info:
            info["bads"] = [n for n, b in zip(ch_names, ch_info["bad"]) if bool(b)]
        if {"x", "y", "z"} <= set(ch_info.columns):
            frame = HEAD_FRAME if "coord_space" in ch_info and \
                (ch_info["coord_space"] == "head").any() else None
            with info._unlock():
                for i, row in enumerate(ch_info[["x", "y", "z"]].to_numpy(float)):
                    if np.all(np.isfinite(row)):
                        info["chs"][i]["loc"][:3] = row / 1e3
                        if frame is not None:
                            info["chs"][i]["coord_frame"] = frame
    return info


def trialset_to_mne(ts, *, ch_types=None):
    """``mne.EpochsArray`` from a ``TrialSet`` (``TrialSet.to_mne``)."""

    mne = _mne()
    factor = FACTORS.get(ts.units)
    if factor is None:
        raise DataError(f"to_mne needs voltage units, not {ts.units!r}.")
    info = _info(ts.ch_names, ts.sfreq, ts.ch_info, ch_types)
    meta = ts.metadata
    conditions = list(meta["condition"].astype(str)) if meta is not None and \
        "condition" in meta else ["1"] * ts.n_trials
    names = list(dict.fromkeys(conditions))
    event_id = {name: k + 1 for k, name in enumerate(names)}
    samples = np.round(np.asarray(ts.event_times if ts.event_times is not None else
                                  np.arange(ts.n_trials), float) * ts.sfreq).astype(int)
    if len(set(samples)) < len(samples):
        samples = np.arange(ts.n_trials) * ts.n_times
    events = np.column_stack([samples, np.zeros(ts.n_trials, int),
                              [event_id[c] for c in conditions]])
    metadata = None if meta is None else meta.reset_index(drop=True)
    added = ts.provenance.source.get("added_columns") or []
    if metadata is not None and added:
        # the standard columns that from_mne added (REQ-313) are not given back to MNE
        metadata = metadata.drop(columns=[c for c in added if c in metadata])
    if metadata is not None and ts.provenance.source.get("condition_from_event_id"):
        # the column that from_mne added from the event names goes back into event_id
        metadata = metadata.drop(columns=["condition"])
        metadata = metadata if len(metadata.columns) else None
    return mne.EpochsArray(np.nan_to_num(ts.data) / factor, info, events=events,
                           tmin=float(ts.times[0]), event_id=event_id, metadata=metadata,
                           verbose="error")


def evoked_from_mne(cls, evoked, *, picks=None, units="uV"):
    """``Evoked`` from ``mne.Evoked`` (``Evoked.from_mne``)."""

    from .provenance import Provenance

    mne = _mne()
    if not isinstance(evoked, mne.Evoked):
        raise DataError(f"from_mne expects mne.Evoked, got {type(evoked).__name__}.")
    if picks is not None:
        evoked = evoked.copy().pick(picks)
    factor = FACTORS.get(units)
    if factor is None:
        raise DataError(f"units must be one of {sorted(FACTORS)}.")
    names = list(evoked.ch_names)
    provenance = Provenance(source={"format": "mne", "object": "Evoked",
                                    "comment": evoked.comment, "nave": int(evoked.nave)}) \
        .with_step("evoparam.data.Evoked.from_mne", {"units": units, "picks": picks})
    return cls(data=np.array(evoked.data) * factor, times=np.array(evoked.times),
               sfreq=float(evoked.info["sfreq"]), ch_names=names, units=units,
               ch_info=_channel_table(evoked.info, names), provenance=provenance)


def evoked_to_mne(ev, *, ch_types=None, comment=""):
    """``mne.EvokedArray`` from an ``Evoked`` (``Evoked.to_mne``)."""

    mne = _mne()
    factor = FACTORS.get(ev.units)
    if factor is None:
        raise DataError(f"to_mne needs voltage units, not {ev.units!r}.")
    info = _info(ev.ch_names, ev.sfreq, getattr(ev, "ch_info", None), ch_types)
    return mne.EvokedArray(np.nan_to_num(ev.data) / factor, info, tmin=float(ev.times[0]),
                           comment=comment, verbose="error")
