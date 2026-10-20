# SPDX-License-Identifier: Apache-2.0
r"""
01: Model description
=====================

A first look at evoparam: build one evoked response whose parts we know, fit it, and
check that the fit gives those parts back.

evoparam describes an evoked response (the average of the signal of one recording
contact over many stimulations, time-locked to the stimulus) as

.. math::

   y(t) = b(t) + \sum_{k=1}^{K} A_k \, s_k(t; \theta_k) + \varepsilon(t)

- :math:`y(t)` is the averaged signal of one contact, in microvolts (µV), at time
  :math:`t` after the stimulus.
- :math:`b(t)` is the **background**: the slow part that is not a response, for example
  the recovery of the amplifier after a stimulation pulse. It plays the role that the
  aperiodic (1/f) part plays in a power spectrum.
- Each :math:`s_k` is a **component**: a bump of a known shape (here a Gaussian bell) with
  a height of 1, a peak time (**latency**, ms) and a width (**FWHM**, full width at half
  maximum, ms). :math:`A_k` is its **amplitude** in µV, negative for a downward bump (N1,
  N2). Components play the role that the periodic peaks play in a power spectrum.
- :math:`\varepsilon(t)` is the **noise**: what is left, measured on the pre-stimulus
  baseline.

The number of components :math:`K` is not given by the user: components are added one at
a time and kept only when the BIC (Bayesian information criterion, a score that rewards a
better fit and penalises extra parameters) drops by at least 10.
"""  # noqa: D205, D400  (sphinx-gallery: the docstring opens with an RST title)

# %%
# Build a response whose parts we know
# ------------------------------------
# One contact, sampled at 2048 Hz, from 0.5 s before to 1 s after a stimulation pulse.
# Known parts: an N1 (peak -150 µV at 25 ms, 10 ms wide), an N2 (-80 µV at 150 ms,
# 80 ms wide), an exponential recovery of the amplifier after the pulse (60 µV at the
# pulse, time constant 30 ms) and pink (1/f) noise of 8 µV standard deviation.

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evoparam import ResponseModel  # noqa: E402
from evoparam.shapes import get_shape  # noqa: E402
from evoparam.sim import make_noise, shape_params  # noqa: E402

sfreq = 2048.0
times = np.arange(-0.5, 1.0, 1 / sfreq)
gaussian = get_shape("gaussian")

truth = {"N1": (0.025, 0.010, -150.0), "N2": (0.150, 0.080, -80.0)}  # latency s, FWHM s, µV
components = {
    label: amp * gaussian.evaluate(times, shape_params("gaussian", lat, fwhm))
    for label, (lat, fwhm, amp) in truth.items()
}
background = np.where(times > 0, 60.0 * np.exp(-times / 0.030), 0.0)
noise = make_noise(times.size, sfreq, "pink", sd=8.0, seed=3)
y = background + sum(components.values()) + noise

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(times * 1000, y, color="0.2", lw=0.8, label="signal = sum of the parts")
ax.plot(times * 1000, background, color="tab:orange", label="background (amplifier recovery)")
for (label, part), color in zip(components.items(), ("tab:blue", "tab:green"), strict=True):
    ax.plot(times * 1000, part, color=color, label=f"component {label}")
ax.set(xlim=(-100, 500), xlabel="Time after the pulse (ms)", ylabel="µV")
ax.legend(frameon=False, fontsize=8)

# %%
# Fit it
# ------
# A :class:`~evoparam.ResponseModel` holds the settings; the ``ccep`` preset sets them for
# cortico-cortical evoked potentials (fit window 9 to 500 ms, exponential background, up
# to 6 components). ``fit_waveform`` fits one curve and returns a ``ResponseFit``.

model = ResponseModel("ccep", random_state=0)
fit = model.fit_waveform(times, y)
print(fit.status)

# %%
# What the fit found, next to the truth
# -------------------------------------
# One row per component found. ``latency`` and ``fwhm`` are in seconds, ``amplitude``
# in µV; the ``_ci_low`` / ``_ci_high`` columns are 95 % intervals.

table = fit.to_dataframe()
print(
    table[["label", "latency", "amplitude", "fwhm", "latency_ci_low", "latency_ci_high"]].round(4)
)
print("truth:", truth)
print("background found:", {k: round(v, 4) for k, v in fit.background.items() if k != "name"})

# %%
# The fit, part by part
# ---------------------
# Data in grey, full model in black, background and each component in colour, and the
# residual (data minus model) below: it should look like the baseline noise.

from evoparam import plots  # noqa: E402

fig = plots.plot_fit(fit)

# %%
# Remove the background, keep the components
# ------------------------------------------
# As a power spectrum is flattened by removing its aperiodic part, the response is
# isolated by removing the fitted background: what is left is the components and the noise.

inside = (times >= fit.window[0]) & (times <= fit.window[1])
t_fit = times[inside]
found = [c.amplitude * get_shape(c.shape).evaluate(t_fit, c.params) for c in fit.components]
background_fit = fit.model - sum(found)

fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
top.plot(t_fit * 1000, y[inside], color="0.6", lw=0.8, label="data")
top.plot(t_fit * 1000, background_fit, color="tab:orange", label="background found")
top.plot(t_fit * 1000, background[inside], color="tab:orange", ls=":", label="true background")
top.set(ylabel="µV")
top.legend(frameon=False, fontsize=8)
bottom.plot(
    t_fit * 1000, y[inside] - background_fit, color="0.6", lw=0.8, label="data minus background"
)
bottom.plot(t_fit * 1000, sum(found), color="k", label="components found")
bottom.plot(
    t_fit * 1000,
    sum(components.values())[inside],
    color="tab:blue",
    ls=":",
    label="true components",
)
bottom.set(xlabel="Time after the pulse (ms)", ylabel="µV")
bottom.legend(frameon=False, fontsize=8)

# %%
# Same response, other noise
# --------------------------
# The numbers above come from one draw of the noise. With five other draws the N1 latency
# stays within about 0.3 ms of the truth, while its amplitude moves by up to about 17 µV,
# and one draw adds a small extra bump near the end of the window: that spread is what the
# noise allows, and the 95 % intervals say how large it is.

import pandas as pd  # noqa: E402

rows = []
for seed in range(5):
    y_other = (
        background
        + sum(components.values())
        + make_noise(times.size, sfreq, "pink", sd=8.0, seed=seed)
    )
    for c in model.fit_waveform(times, y_other).components:
        rows.append(
            {
                "noise draw": seed,
                "label": c.label,
                "latency_ms": 1000 * c.latency,
                "amplitude_uV": c.amplitude,
                "fwhm_ms": 1000 * c.fwhm,
            }
        )
print(pd.DataFrame(rows).round(1).to_string(index=False))

# %%
# The methods paragraph
# ---------------------
# Every fit can write the paragraph that describes how it was obtained, for a paper.

print(fit.methods_text())
