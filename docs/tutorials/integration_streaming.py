# SPDX-License-Identifier: Apache-2.0
"""
Online mode: streaming estimates
================================

For research use only: evoparam is not a medical device and is not validated for clinical
decisions. The online mode exposes numbers; it controls nothing.

``evoparam.stream`` (spec §10.6) fits the template of an offline fit to each new pulse of a
stream: the threshold detector finds the pulse, and once its window is complete the frozen
template is shifted and scaled on every channel (amplitude, latency, per-pulse existence
score). A running existence test per channel keeps its error rate at 5 % however often it is
looked at. With Lab Streaming Layer (extra ``stream``), the same fitter reads an
``LSLInlet`` and publishes its estimates with an ``LSLOutlet``.

Data: a simulated stream, one pulse per second, an N1 on half of the eight channels. The
standalone script ``tutorials/92_online_mapping_with_lsl.py`` runs the same steps.
"""  # noqa: D205, D400  (sphinx-gallery: the docstring opens with an RST title)

# %%
# Imports
# -------

import warnings

import numpy as np

import evoparam
from evoparam.data import TrialSet
from evoparam.models import fit
from evoparam.stream import StreamFitter

# %%
# Settings
# --------

SFREQ = 2048.0
N_CHANNELS = 8  # responses on the first half
N1 = (-40.0, 0.03, 0.02)  # amplitude (uV), latency (s), FWHM (s)
ARTIFACT = (-0.002, 0.008)  # artifact window, interpolated (s)


def gauss(times, latency, fwhm):
    """Return a Gaussian bump of the given latency and FWHM (s)."""
    return np.exp(-0.5 * ((times - latency) / (fwhm / 2.3548)) ** 2)


# %%
# 1. Offline templates: 40 trials of a first block
# ------------------------------------------------

rng = np.random.default_rng(0)
times = np.arange(-1024, 2048) / SFREQ
trials = rng.normal(0, 10, (40, N_CHANNELS, len(times)))
trials[:, : N_CHANNELS // 2] += N1[0] * gauss(times, N1[1], N1[2])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    group = fit(
        TrialSet(
            data=trials, times=times, sfreq=SFREQ, ch_names=[f"c{i}" for i in range(N_CHANNELS)]
        ),
        "ccep",
    )
template = group.fits[0]
print(f"template: {template.components[0].label} at {template.components[0].latency * 1e3:.1f} ms")

# %%
# 2. The stream: one pulse per second, pushed in chunks of 32 samples
# -------------------------------------------------------------------

fitter = StreamFitter(
    template,
    sfreq=SFREQ,
    ch_names=[f"c{i}" for i in range(N_CHANNELS)],
    artifact_window=ARTIFACT,
    existence_max_pulses=30,
    random_state=0,
)
estimates = []
for _ in range(30):
    block = rng.normal(0, 10, (int(SFREQ), N_CHANNELS))
    onset = int(0.45 * SFREQ)
    block[onset : onset + 3] += np.array([3000.0, -2000.0, 500.0])[:, None]
    rel = (np.arange(len(block)) - onset) / SFREQ
    block[:, : N_CHANNELS // 2] += N1[0] * gauss(rel, N1[1], N1[2])[:, None]
    for first in range(0, len(block), 32):
        estimates += fitter.push(block[first : first + 32])

print("last pulse:")
for est in estimates[-N_CHANNELS:]:
    print(
        f"{est.channel}: amplitude {est.amplitude:7.1f} uV, latency "
        f"{est.latency * 1e3:5.1f} ms, z {est.existence_z:5.1f}, {est.status}"
    )

# %%
# 3. Running existence test and compute time
# ------------------------------------------
#
# The test keeps a 5 % false-positive rate per channel over all its looks (one after every
# pulse here), so a silent channel is now and then declared responsive: with four silent
# channels, about one run in five has one such channel.

result = fitter.existence()
for name, significant, n in zip(
    result["channels"], result["significant"], result["n"], strict=False
):
    print(f"{name}: {'response' if significant else 'no response'} after {n} pulses")
print("compute time per pulse (ms):", {k: round(v, 2) for k, v in fitter.latency_stats.items()})

# %%
# 4. With Lab Streaming Layer (needs pylsl and liblsl)
# ----------------------------------------------------
#
# The same loop, reading from a Lab Streaming Layer stream::
#
#     from evoparam.stream import LSLInlet, LSLOutlet
#     inlet = LSLInlet(stype="EEG")
#     fitter = StreamFitter(template, sfreq=inlet.sfreq, ch_names=inlet.ch_names,
#                           artifact_window=ARTIFACT)
#     outlet = LSLOutlet(fitter)          # metadata: intended_use = "research only; not a
#                                         # medical device"
#     inlet.feed(fitter, duration=60.0, callback=outlet.push)

# %%
# Methods paragraph (the offline fit that gives the template)
# ------------------------------------------------------------

print(evoparam.reports.methods_text(group))
