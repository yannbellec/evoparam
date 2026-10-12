# SPDX-License-Identifier: Apache-2.0
"""
NWB: epochs from an ElectricalSeries, results in the file
=========================================================

:func:`evoparam.io.read_nwb` epochs an ``ElectricalSeries`` around events found in the file
(an ``EventsTable``, the ``trials`` table, ``TimeIntervals``, a ``StimTable``, or the
onsets of intracellular stimuli), reading only the epochs; :func:`evoparam.io.write_nwb`
adds the results to a processing module (tables linked to the electrodes and trials, the
fitted curves as an ``ElectricalSeries``, optionally the ``ndx-evoparam`` extension types)
and :func:`evoparam.io.read_nwb_results` reads them back exactly.

Data: a simulated laminar recording written to NWB with pynwb (a DANDI asset is read the
same way, by its https address).
"""  # noqa: D205, D400  (sphinx-gallery: the docstring opens with an RST title)

# %%
# An NWB file
# -----------

import os
import tempfile
from datetime import datetime, timezone

import numpy as np
from pynwb import NWBHDF5IO, NWBFile
from pynwb.ecephys import ElectricalSeries

import evoparam
from evoparam import sim
from evoparam.io import read_nwb, read_nwb_results, write_nwb

trials, _ = sim.simulate_animal_ep(n_trials=40, n_channels=8, seed=0)
sfreq = trials.sfreq
n_epoch = trials.n_times
gap = int(0.3 * sfreq)
continuous = np.zeros((trials.n_trials * (n_epoch + gap), trials.n_channels))
onsets = []
for k in range(trials.n_trials):
    start = k * (n_epoch + gap)
    continuous[start : start + n_epoch] = trials.data[k].T * 1e-6  # V
    onsets.append((start - trials.times[0] * sfreq) / sfreq)

nwb = NWBFile(
    session_description="simulated laminar LFP",
    identifier="evoparam-tutorial",
    session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
)
device = nwb.create_device(name="probe")
shank = nwb.create_electrode_group(
    name="shank0", description="laminar probe", location="VISp", device=device
)
for i in range(trials.n_channels):
    nwb.add_electrode(group=shank, location="VISp", x=0.0, y=0.0, z=40.0 * i)
region = nwb.create_electrode_table_region(list(range(trials.n_channels)), "probe")
nwb.add_acquisition(
    ElectricalSeries(name="lfp", data=continuous, electrodes=region, rate=sfreq, starting_time=0.0)
)
nwb.add_trial_column("stim_onset", "time of the stimulus (s)")
for t in onsets:
    nwb.add_trial(start_time=t - 0.1, stop_time=t + 0.5, stim_onset=t)
path = os.path.join(tempfile.mkdtemp(), "session.nwb")
with NWBHDF5IO(path, "w") as io:
    io.write(nwb)

# %%
# Epochs from the trials table
# ----------------------------

epochs = read_nwb(path, tmin=-0.1, tmax=0.5)  # onsets: the stim_onset column of trials
print(epochs, epochs.provenance.source["events"])

# %%
# Fit and write the results into the file
# ---------------------------------------

group = evoparam.fit(epochs, "rodent_ep", existence=False, random_state=0)
write_nwb([group, epochs], path, extension=True)
with NWBHDF5IO(path, "r", load_namespaces=True) as io:
    module = io.read().processing["evoparam"]
    print(list(module.data_interfaces))
    print(type(module["evoparam_components"]).__name__)

# %%
# Read back
# ---------

print("same result:", read_nwb_results(path) == group)

# %%
# Methods paragraph
# -----------------

print(evoparam.reports.methods_text(group))
