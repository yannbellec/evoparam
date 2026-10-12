# SPDX-License-Identifier: Apache-2.0
"""Lab Streaming Layer adapters of the online mode (extra ``stream``: ``pylsl`` >= 1.16).

For research use only: not a medical device. Every outlet of pulse estimates carries the
metadata ``intended_use = "research only; not a medical device"``.
"""

import json
import time

import numpy as np

from evoparam.errors import DataError
from evoparam.utils._optional import require
from evoparam.version import __version__

from ._fitter import INTENDED_USE, STATUSES

__all__ = ["LSLInlet", "LSLOutlet", "FIELDS"]

PYLSL_FLOOR = "1.16"
# Seconds an LSLInlet waits for its stream to be found.
RESOLVE_TIMEOUT = 5.0

# Channels of the outlet (float32), the fields of PulseEstimate with the strings as indices
# into the metadata; the LSL timestamp of each sample is also the pulse time (in double
# precision, which float32 lacks for clock values).
FIELDS = (
    "pulse_time",
    "channel_index",
    "component_index",
    "amplitude",
    "latency",
    "scale",
    "shift",
    "existence_z",
    "present",
    "r2",
    "compute_ms",
    "status_code",
)


def _pylsl(caller):
    return require("pylsl", extra="stream", min_version=PYLSL_FLOOR, caller=caller)


def _channel_labels(info, n_channels):
    """Channel labels from the ``desc/channels/channel/label`` entries of a stream."""

    labels = []
    channel = info.desc().child("channels").child("channel")
    while not channel.empty() and len(labels) < n_channels:
        labels.append(channel.child_value("label"))
        channel = channel.next_sibling("channel")
    if len(labels) != n_channels or not all(labels):
        labels = [f"ch{i}" for i in range(n_channels)]
    return labels


class LSLInlet:
    """Read a stream of samples from LSL and feed a :class:`StreamFitter`.

    Parameters
    ----------
    name : str, optional
        Stream name; by default the first stream of type ``stype``.
    stype : str
        Stream type searched when ``name`` is None.
    max_buflen : int
        Seconds of data the inlet keeps when it is not read.

    The stream is searched for :data:`RESOLVE_TIMEOUT` seconds.

    Attributes
    ----------
    sfreq : float
        Nominal sampling rate of the stream.
    ch_names : list of str
        Channel labels from the stream description (``ch0``, ``ch1``, ... when absent).
    """

    def __init__(self, name=None, stype="EEG", *, max_buflen=360):
        timeout = RESOLVE_TIMEOUT
        pylsl = _pylsl("evoparam.stream.LSLInlet")
        prop, value = ("name", name) if name is not None else ("type", stype)
        found = pylsl.resolve_byprop(prop, value, 1, timeout)
        if not found:
            raise DataError(f"No LSL stream with {prop} {value!r} found within {timeout} s.")
        self._inlet = pylsl.StreamInlet(found[0], max_buflen=int(max_buflen))
        info = self._inlet.info(timeout)
        # Samples pushed from now on are received (an inlet otherwise subscribes on its
        # first pull)
        self._inlet.open_stream(timeout)
        self.sfreq = float(info.nominal_srate())
        self.n_channels = int(info.channel_count())
        self.ch_names = _channel_labels(info, self.n_channels)
        self.name = info.name()

    def pull(self, timeout=0.0, max_samples=1024):
        """Samples available now (waiting up to ``timeout`` s): ``(samples [n, n_channels],
        timestamps [n])``."""

        chunk, stamps = self._inlet.pull_chunk(timeout=timeout, max_samples=max_samples)
        if not stamps:
            return np.zeros((0, self.n_channels)), np.zeros(0)
        return np.asarray(chunk, float).reshape(len(stamps), self.n_channels), np.asarray(
            stamps, float
        )

    def feed(self, fitter, *, duration=None, max_pulses=None, timeout=0.05, callback=None):
        """Pull samples and push them into ``fitter`` (with their LSL timestamps) until
        ``duration`` seconds have passed or ``max_pulses`` pulses have been estimated.

        ``callback(estimates)`` is called with the estimates of every chunk that has some.
        Returns every estimate.
        """

        if duration is None and max_pulses is None:
            raise DataError("feed needs duration or max_pulses (it would never return).")
        out, start, pulses = [], time.monotonic(), 0
        while True:
            samples, stamps = self.pull(timeout=timeout)
            if len(stamps):
                estimates = fitter.push(samples, stamps)
                if estimates:
                    pulses += len({e.pulse_time for e in estimates})
                    out.extend(estimates)
                    if callback is not None:
                        callback(estimates)
            if max_pulses is not None and pulses >= max_pulses:
                return out
            if duration is not None and time.monotonic() - start >= duration:
                return out

    def close(self):
        self._inlet.close_stream()


class LSLOutlet:
    """Publish :class:`PulseEstimate` rows as an irregular-rate LSL stream.

    One LSL sample per estimate, float32 channels :data:`FIELDS` (the fields of
    :class:`PulseEstimate`), stamped with the pulse time. ``channel_index`` and
    ``component_index`` point into the ``channels`` and ``components`` lists of the JSON
    metadata (``desc/evoparam``), which also holds the status codes, the fitter settings,
    the package version and ``intended_use``; the stream description also carries
    ``intended_use`` as its own entry.

    Parameters
    ----------
    fitter : StreamFitter
        Gives the channels and the component labels.
    name : str
        Stream name.
    source_id : str
    """

    def __init__(self, fitter, name="evoparam-pulses", source_id="evoparam-stream"):
        pylsl = _pylsl("evoparam.stream.LSLOutlet")
        self.channels = list(fitter.model.ch_names)
        self.components = [list(labels) for labels in fitter.model.labels]
        self.metadata = {
            "intended_use": INTENDED_USE,
            "fields": list(FIELDS),
            "channels": self.channels,
            "components": self.components,
            "status_codes": {status: i for i, status in enumerate(STATUSES)},
            "settings": fitter.settings,
            "evoparam_version": __version__,
        }
        info = pylsl.StreamInfo(
            name, "EvokedParameters", len(FIELDS), pylsl.IRREGULAR_RATE, "float32", source_id
        )
        desc = info.desc()
        desc.append_child_value("intended_use", INTENDED_USE)
        desc.append_child_value("evoparam", json.dumps(self.metadata))
        fields = desc.append_child("channels")
        for field in FIELDS:
            fields.append_child("channel").append_child_value("label", field)
        self._outlet = pylsl.StreamOutlet(info)
        self._channel = {name: i for i, name in enumerate(self.channels)}
        self._component = [
            {label: k for k, label in enumerate(labels)} for labels in self.components
        ]
        self._status = {status: i for i, status in enumerate(STATUSES)}

    def row(self, estimate):
        """The float values of one estimate, in the order of :data:`FIELDS`."""

        c = self._channel[estimate.channel]
        return [
            estimate.pulse_time,
            float(c),
            float(self._component[c][estimate.component]),
            estimate.amplitude,
            estimate.latency,
            estimate.scale,
            estimate.shift,
            estimate.existence_z,
            float(estimate.present),
            estimate.r2,
            estimate.compute_ms,
            float(self._status[estimate.status]),
        ]

    def push(self, estimates):
        """Publish estimates (one LSL sample each, stamped with its pulse time)."""

        for estimate in estimates:
            self._outlet.push_sample(self.row(estimate), estimate.pulse_time)

    def have_consumers(self):
        return bool(self._outlet.have_consumers())
