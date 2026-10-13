# SPDX-License-Identifier: Apache-2.0
"""
Neo: epochs from a Neo ``Block``
================================

Neo represents electrophysiology from about fifty file formats (Blackrock, Plexon,
Intan, Spike2, ...) as ``Block`` / ``Segment`` / ``AnalogSignal`` / ``Event``.
:func:`evoparam.io.read_neo` epochs an ``AnalogSignal`` around a named ``Event`` (or the
starts of an ``Epoch``), converts its units to µV through ``quantities`` and keeps the
segments as runs.

Data: a Neo block built in memory (two segments, a signal in mV, stimulus events), as a
file reader of Neo would return it.
"""  # noqa: D205, D400  (sphinx-gallery: the docstring opens with an RST title)

# %%
# A Neo block
# -----------

import numpy as np
import quantities as pq
from neo import AnalogSignal, Block, Event, Segment

import evoparam
from evoparam.io import read_neo

rng = np.random.default_rng(0)
sfreq = 1000.0
block = Block(name="slice recording")
for run in range(2):
    n = int(30 * sfreq)
    data = rng.normal(0, 0.01, (n, 3))  # mV
    onsets = np.arange(2.0, 28.0, 1.0)  # s
    for t in onsets:
        i = int(t * sfreq)
        data[i + 8 : i + 30, :] += -0.08 * np.hanning(22)[:, None]  # an evoked response
    segment = Segment(name=f"run {run}")
    segment.analogsignals.append(
        AnalogSignal(
            data,
            units="mV",
            sampling_rate=sfreq * pq.Hz,
            name="field",
            array_annotations={"channel_names": np.array(["F1", "F2", "F3"])},
        )
    )
    segment.events.append(
        Event(onsets * pq.s, labels=np.array(["shock"] * len(onsets)), name="stimulus")
    )
    block.segments.append(segment)

# %%
# Epochs
# ------

trials = read_neo(block, events="stimulus", tmin=-0.1, tmax=0.35)
print(trials)
print(trials.metadata.groupby("run").size().to_dict(), "epochs per run; units:", trials.units)

# %%
# Fit
# ---

group = evoparam.fit(trials, "rodent_ep", existence=False, random_state=0)
print(group.to_dataframe()[["channel", "latency", "amplitude", "fwhm"]].round(4))

# %%
# Known limit (under investigation)
# ---------------------------------
# The simulated response is one negative Hanning bump (peak -80 µV at 19 ms). On some
# channels the fit describes it with two components of opposite sign, the same
# model-selection problem as in the ECAP tutorials, being fixed (CHANGELOG).

# %%
# Methods paragraph
# -----------------

print(evoparam.reports.methods_text(group))
