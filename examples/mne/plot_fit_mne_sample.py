"""
Fit the auditory N1 and P2 of the MNE ``sample`` dataset
========================================================

The MNE ``sample`` dataset holds the averaged responses to left-ear tones. evoparam fits the
``mne.Evoked`` directly: each EEG channel becomes a set of components (the N1 near 100 ms
and the P2 near 200 ms, labelled from their latency windows) with latency, amplitude, width
and their uncertainty. The fitted curves come back as an ``mne.Evoked`` and the components
as ``mne.Annotations``, so MNE's own plots show them.

Data: ``mne.datasets.sample`` (about 1.6 GB, downloaded by MNE on first use).
"""

# %%
# The averaged response
# ---------------------

import numpy as np

import evoparam
from evoparam import presets

import mne

mne.set_log_level("ERROR")
path = mne.datasets.sample.data_path() / "MEG" / "sample" / "sample_audvis-ave.fif"
evoked = mne.read_evokeds(path, condition="Left Auditory", baseline=(None, 0))
evoked = evoked.pick("eeg").interpolate_bads(reset_bads=True)
print(evoked)

# %%
# A preset for auditory ERPs
# --------------------------
# The ``erp`` preset, with the fit window inside the epoch (-200 to 500 ms) and the
# windows of the auditory N1 and P2.

preset = presets.derive("erp", name="erp_auditory", fit_window=(0.04, 0.45),
                        baseline=(-0.2, 0.0),
                        component_labels={"N1": (0.07, 0.14), "P2": (0.14, 0.27)},
                        label_polarity={"N1": -1, "P2": 1})

# %%
# Fit the MNE object
# ------------------

group = evoparam.fit(evoked, preset, existence=False, random_state=0)
table = group.to_dataframe()
summary = table[table["label"].isin(["N1", "P2"])].groupby("label").agg(
    channels=("channel", "nunique"), latency_ms=("latency", lambda v: 1e3 * v.median()),
    amplitude_uV=("amplitude", "median"), fwhm_ms=("fwhm", lambda v: 1e3 * v.median()))
print(summary.round(1))

# %%
# The model as MNE objects
# ------------------------

model = group.to_evoked_model()
annotations = group.to_annotations()
print(model.comment, "-", len(annotations), "components as annotations")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

best = int(np.argmax([fit.gof.r2 if fit.gof else 0 for fit in group.fits]))
channel = group.index["channel"].iloc[best]
fig, ax = plt.subplots()
ax.plot(evoked.times * 1e3, evoked.copy().pick([channel]).data[0] * 1e6, label="data")
ax.plot(model.times * 1e3, model.data[best], label="evoparam model")
ax.set(xlabel="time (ms)", ylabel="µV", title=f"{channel}, left-ear tones")
ax.legend()

# %%
# Methods paragraph
# -----------------

print(evoparam.reports.methods_text(group))
