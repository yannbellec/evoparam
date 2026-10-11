# SPDX-License-Identifier: Apache-2.0
"""
Somatosensory evoked potentials: the ``sep`` preset
===================================================

Median-nerve SEPs have an N20 about 20 ms after the stimulus, followed by P25 and N33.
The ``sep`` preset fits 10 to 60 ms after a short interpolation of the stimulus artifact,
with Gaussian and skewed Gaussian components, and labels the N20 from its latency window.

Data: a simulated average (``evoparam`` has no public SEP dataset yet; the preset is
validated on simulations, REQ-277).
"""

# %%
# Simulate a median-nerve SEP
# ---------------------------
# 5 kHz, -50 to 100 ms; N20 (FWHM 4 ms), P25 and N33, white noise at SNR 8.

import numpy as np

import evoparam
from evoparam import presets
from evoparam.data import Evoked

sfreq = 5000.0
times = np.arange(-0.05, 0.1, 1 / sfreq)
rng = np.random.default_rng(0)


def bump(latency, fwhm):
    return np.exp(-0.5 * ((times - latency) / (fwhm / 2.3548)) ** 2)


waveform = -2.0 * bump(0.0201, 0.004) + 1.6 * bump(0.0251, 0.006) - 1.2 * bump(0.033, 0.008)
data = waveform + rng.normal(0, 0.25, times.size)
evoked = Evoked(data=data[None], times=times, sfreq=sfreq, ch_names=["C3'"], modality="sep")

# %%
# The preset
# ----------

preset = presets.get("sep")
print(preset.fit_window, preset.shapes, preset.component_labels)
print(presets.methods_note("sep"))

# %%
# Fit and read the N20
# --------------------

group = evoparam.fit(evoked, "sep", existence=False, random_state=0)
table = group.to_dataframe()
print(table[["component", "label", "latency", "amplitude", "fwhm"]].round(4))
n20 = table[table["label"] == "N20"].iloc[0]
print(f"N20 at {n20['latency'] * 1e3:.2f} ms (simulated 20.1 ms), "
      f"amplitude {n20['amplitude']:.2f} µV")

# %%
# Figure
# ------

import matplotlib

matplotlib.use("Agg")
from evoparam import plots

fig = plots.plot_fit(group.fits[0], evoked)

# %%
# Methods paragraph
# -----------------

print(evoparam.reports.methods_text(group))
