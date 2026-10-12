# SPDX-License-Identifier: Apache-2.0
"""
Evoked resonant neural activity: the ``erna`` preset
====================================================

During deep brain stimulation of the subthalamic nucleus at 130 Hz, every pulse evokes a
damped oscillation of 200 to 450 Hz (ERNA). The ``erna`` preset removes the stimulation
artifact with a template of one pulse period, then fits one damped oscillation per pulse
(frequency, decay, first-peak latency).

Data: a simulated burst of 10 pulses on 4 DBS contacts (no public ERNA dataset exists; the
preset is validated on simulations, REQ-276).
"""  # noqa: D205, D400  (sphinx-gallery: the docstring opens with an RST title)

# %%
# Simulate ERNA bursts
# --------------------

import numpy as np

import evoparam
from evoparam import presets, sim
from evoparam.data import TrialSet

ts, truth = sim.simulate_erna(
    10, n_trials=20, seed=0, snr=4, artifact=False, tmax=0.09, onset_delay=0.002
)
print(ts)

# %%
# One epoch per pulse, from the last five pulses of each burst
# -------------------------------------------------------------
# The ERNA grows over the first pulses of a burst; its frequency is read once it is stable.

period = 1 / 130.0
n = int(round(0.0105 * ts.sfreq))
epochs = []
for k in range(5, 10):
    start = np.searchsorted(ts.times, k * period - 0.002)
    epochs.append(ts.data[:, :, start : start + n])
start = np.searchsorted(ts.times, 9 * period - 0.002)
times = ts.times[start : start + n] - 9 * period
pulses = TrialSet(
    data=np.concatenate(epochs), times=times, sfreq=ts.sfreq, ch_names=ts.ch_names, modality="erna"
)

# %%
# Fit with the preset
# -------------------

print(presets.methods_note("erna"))
group = evoparam.fit(pulses, "erna", existence=False, random_state=0)
table = group.to_dataframe()
osc = table[table["shape"] == "damped_oscillation"]
freq = [fit.components[0].params[2] for fit in group.fits if fit.components]
print(osc[["channel", "latency", "amplitude"]].round(4))
truth_table = truth.to_dataframe()
print("fitted frequency (Hz):", np.round(freq, 1))
print(
    "simulated frequency (Hz):",
    np.round(truth_table.groupby("channel")["param_freq"].first().to_numpy(), 1),
)

# %%
# Figure
# ------

import matplotlib

matplotlib.use("Agg")
from evoparam import plots

fig = plots.plot_group(group, pulses.average())

# %%
# Methods paragraph
# -----------------

print(evoparam.reports.methods_text(group))
