# SPDX-License-Identifier: Apache-2.0
"""Neo reader (spec §10.3, REQ-319; extra ``neo``)."""

import os

import numpy as np

from evoparam.errors import DataError
from evoparam.utils._optional import require

__all__ = ["read_neo"]


def _block(source, io_class):
    neo = require("neo", extra="neo", caller="evoparam.io.read_neo")
    if isinstance(source, neo.Block):
        return source
    path = os.fspath(source)
    if io_class:
        reader = getattr(neo.io, io_class)(path)
    else:
        reader = neo.io.get_io(path)
    return reader.read_block()


def _microvolts(signal):
    """Samples of an AnalogSignal rescaled to µV (dimensionless signals unchanged)."""

    quantities = require("quantities", extra="neo", caller="evoparam.io.read_neo")
    try:
        return np.asarray(signal.rescale(quantities.uV).magnitude, float), "uV"
    except ValueError:
        return np.asarray(signal.magnitude, float), "au"


def read_neo(source, *, segment=None, signal=0, events=None, tmin=-0.1, tmax=0.5,
             io_class=None):
    """Epochs of a Neo ``AnalogSignal`` around events (spec §10.3).

    Parameters
    ----------
    source : neo.Block or str
        A Block, or a file opened with ``neo.io.get_io`` (or the IO class ``io_class``).
    segment : int, optional
        Read one segment; default every segment, as runs (``metadata["run"]``, the segment
        index as text).
    signal : int or str
        AnalogSignal of each segment, by index or name.
    events : str, optional
        Name of the ``Event`` whose times are the onsets (labels become ``condition``);
        default the first Event, else the starts of the first ``Epoch``.
    tmin, tmax : float
        Epoch window (s).

    Returns
    -------
    TrialSet
        Data in µV (units through ``quantities``).
    """

    import pandas as pd

    from evoparam.data import Provenance, TrialSet

    block = _block(source, io_class)
    segments = list(enumerate(block.segments))
    if segment is not None:
        segments = [segments[segment]]
    epochs, rows, sfreq, names, units = [], [], None, None, None
    for run, seg in segments:
        signals = seg.analogsignals
        if not signals:
            continue
        sig = signals[signal] if isinstance(signal, int) else \
            next(s for s in signals if s.name == signal)
        values, units = _microvolts(sig)
        rate = float(sig.sampling_rate.rescale("Hz").magnitude)
        start = float(sig.t_start.rescale("s").magnitude)
        if sfreq is not None and abs(rate - sfreq) > 1e-6 * sfreq:
            raise DataError("Segments with different sampling rates.")
        sfreq = rate
        n_ch = values.shape[1]
        this_names = [str(n) for n in sig.array_annotations.get("channel_names",
                                                                [f"ch{i}" for i in range(n_ch)])]
        names = names or this_names
        if seg.events and events is not False:
            event = seg.events[0] if events is None else \
                next(e for e in seg.events if e.name == events)
            times = np.asarray(event.times.rescale("s").magnitude, float)
            labels = [str(l) for l in event.labels] if len(event.labels) else [""] * len(times)
        elif seg.epochs:
            epoch = seg.epochs[0]
            times = np.asarray(epoch.times.rescale("s").magnitude, float)
            labels = [str(l) for l in epoch.labels] if len(epoch.labels) else [""] * len(times)
        else:
            raise DataError(f"Segment {run} has no Event or Epoch.")
        first = int(np.ceil(np.round(tmin * rate, 6)))
        n_times = int(np.ceil(np.round(tmax * rate, 6))) - first
        for t, label in zip(times, labels):
            i0 = int(np.ceil(np.round((t - start) * rate, 6))) + first
            if i0 < 0 or i0 + n_times > values.shape[0]:
                continue
            epochs.append(values[i0:i0 + n_times].T)
            rows.append({"onset": t, "condition": label, "run": str(run)})
    if not epochs:
        raise DataError("No epoch fits inside the signals.")
    first = int(np.ceil(np.round(tmin * sfreq, 6)))
    times_axis = (first + np.arange(epochs[0].shape[1])) / sfreq
    provenance = Provenance(source={"format": "neo", "reader": "evoparam.io.read_neo",
                                    "path": None if hasattr(source, "segments")
                                    else os.fspath(source)}).with_step(
        "evoparam.io.read_neo", {"segment": segment, "signal": signal, "events": events,
                                 "tmin": tmin, "tmax": tmax, "io_class": io_class})
    meta = pd.DataFrame(rows)
    return TrialSet(data=np.stack(epochs), times=times_axis, sfreq=sfreq, ch_names=names,
                    units=units, metadata=meta, event_times=meta["onset"].to_numpy(float),
                    provenance=provenance)
