# SPDX-License-Identifier: Apache-2.0
"""
MNE-Python: fit MNE objects, get MNE objects back
=================================================

evoparam reads ``mne.Epochs`` and ``mne.Evoked`` directly (``evoparam.fit``), converts both
ways without loss (``TrialSet.from_mne`` / ``to_mne``: units, bad channels, positions,
metadata, event times), and gives its results back as MNE objects for display: the fitted
curves as an ``Evoked`` and the components as ``Annotations``.

Data: simulated CCEP epochs converted to ``mne.EpochsArray`` (the MNE ``sample`` example is
in ``examples/mne/plot_fit_mne_sample.py``).
"""  # noqa: D205, D400

# %%
# MNE epochs
# ----------

import mne
import numpy as np

import evoparam
from evoparam import sim
from evoparam.data import TrialSet

mne.set_log_level("ERROR")
trials, _ = sim.simulate_ccep(n_trials=20, n_channels=6, seed=0, artifact=False)
epochs = trials.to_mne()
print(epochs)

# %%
# Round trip
# ----------

back = TrialSet.from_mne(epochs)
print("max difference (µV):", float(np.max(np.abs(back.data - trials.data))))
print("stimulation columns harmonised:", [c for c in back.metadata if c.startswith("stim_")])

# %%
# Fit the MNE object directly
# ---------------------------

group = evoparam.fit(epochs, "ccep", existence=False, random_state=0)
print(group.to_dataframe()[["channel", "label", "latency", "amplitude"]].round(4).head())

evokeds = [epochs.average()]
evokeds[0].comment = "all"
per_condition = evoparam.fit(evokeds, "ccep", existence=False, random_state=0)
print(set(per_condition.index["condition"]))

# %%
# Results as MNE objects
# ----------------------

model = group.to_evoked_model()
annotations = group.to_annotations()
print(model.comment, len(annotations), annotations.description[:3])

import matplotlib  # noqa: E402

matplotlib.use("Agg")
fig = model.plot(show=False)

# %%
# Methods paragraph
# -----------------

print(evoparam.reports.methods_text(group))
