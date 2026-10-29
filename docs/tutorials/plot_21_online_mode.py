# SPDX-License-Identifier: Apache-2.0
"""
21: Online mode
===============

Fit a frozen template to each new pulse as it arrives, with a running existence test.
"""  # noqa: D205, D400  (sphinx-gallery: the docstring opens with an RST title)

###################################################################################################
# Introduction
# ------------
#
# For research use only: evoparam is not a medical device and controls nothing. The online
# mode exposes numbers as a stream comes in; any control loop is the user's own.
#
# ``evoparam.stream`` turns an offline fit into a live estimator. You fit a first block the
# usual way to get a **template** (the bumps). Then, for every new pulse in the stream, a
# threshold detector finds the pulse, and once its window is complete the frozen template
# is **shifted and scaled** on each channel: you get an amplitude, a latency and a per-pulse
# existence score, in a few milliseconds, without re-fitting from scratch. A running
# existence test keeps its 5 % false-positive rate per channel however often you read it.
#
# We simulate a stream (no hardware): one pulse per second, an N1 on half of eight
# channels, fed in small chunks as a device would deliver it.

###################################################################################################

# Import what we need
import warnings

import numpy as np

from evoparam.data import TrialSet
from evoparam.models import fit
from evoparam.stream import StreamFitter

SFREQ = 2048.0
N_CHANNELS = 8  # the first four respond, the last four are silent
N1 = (-40.0, 0.03, 0.02)  # amplitude (uV), latency (s), FWHM (s)
ARTIFACT = (-0.002, 0.008)  # artifact window, interpolated (s)


def gauss(times, latency, fwhm):
    """Return a Gaussian bump of the given latency and width (s)."""
    return np.exp(-0.5 * ((times - latency) / (fwhm / 2.3548)) ** 2)


###################################################################################################
# 1. The offline template
# ------------------------
#
# First, a normal fit on an offline block of 40 trials. Its bumps become the template the
# stream will track. This is the only fit done; the stream only shifts and scales it.

###################################################################################################

rng = np.random.default_rng(0)
times = np.arange(-1024, 2048) / SFREQ
block = rng.normal(0.0, 10.0, (40, N_CHANNELS, times.size))
block[:, : N_CHANNELS // 2] += N1[0] * gauss(times, N1[1], N1[2])

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    offline = fit(
        TrialSet(data=block, times=times, sfreq=SFREQ,
                 ch_names=[f"c{channel}" for channel in range(N_CHANNELS)]),
        "ccep",
    )  # fmt: skip
template = offline.fits[0]
print(f"template: {template.components[0].label} at {template.components[0].latency * 1e3:.1f} ms")

###################################################################################################
# 2. The stream
# -------------
#
# ``StreamFitter`` holds the template and the detector. We feed it 30 pulses, one per
# second, in chunks of 32 samples (``push`` returns the finished estimates). Each pulse has
# a big artifact and an N1 on the first four channels.

###################################################################################################

fitter = StreamFitter(
    template, sfreq=SFREQ, ch_names=[f"c{channel}" for channel in range(N_CHANNELS)],
    artifact_window=ARTIFACT, existence_max_pulses=30, random_state=0,
)  # fmt: skip

estimates = []
for _ in range(30):
    chunk = rng.normal(0.0, 10.0, (int(SFREQ), N_CHANNELS))
    onset = int(0.45 * SFREQ)
    chunk[onset : onset + 3] += np.array([3000.0, -2000.0, 500.0, 400.0,
                                          0.0, 0.0, 0.0, 0.0])[None, :]  # artifact
    relative = (np.arange(len(chunk)) - onset) / SFREQ
    chunk[:, : N_CHANNELS // 2] += N1[0] * gauss(relative, N1[1], N1[2])[:, None]
    for first in range(0, len(chunk), 32):
        estimates += fitter.push(chunk[first : first + 32])

print(f"\n{len(estimates)} per-channel estimates over 30 pulses; the last pulse:")
for estimate in estimates[-N_CHANNELS:]:
    print(f"  {estimate.channel}: amplitude {estimate.amplitude:7.1f} uV, "
          f"latency {estimate.latency * 1e3:5.1f} ms, "
          f"z {estimate.existence_z:5.1f}, {estimate.status}")

###################################################################################################
#
# The responding channels come back near the template's amplitude and latency with a high
# existence z; the silent channels stay near zero. Each estimate is one pulse on one
# channel, delivered as soon as its window closes.

###################################################################################################
# 3. The running existence test and the timing budget
# ----------------------------------------------------
#
# ``existence`` is the verdict per channel after all the pulses seen so far. It corrects for
# being looked at after every pulse, so the 5 % false-positive rate holds however often you
# read it. ``latency_stats`` is the compute time per pulse: the online budget.

###################################################################################################

result = fitter.existence()
triples = zip(result["channels"], result["significant"], result["n"], strict=False)
for name, significant, n in triples:
    print(f"  {name}: {'response' if significant else 'no response'} after {n} pulses")
budget = {key: round(value, 2) for key, value in fitter.latency_stats.items()}
print("\ncompute time per pulse (ms):", budget)

###################################################################################################
#
# The four responding channels are found; among the four silent ones, a correction holds
# the false-positive rate at 5 % per channel across all the looks, so now and then one
# silent channel is declared responsive (with four of them, roughly one run in five). That
# is the test working as specified, not a bug: read a single channel's verdict with that
# rate in mind.
#
# .. note::
#
#     With Lab Streaming Layer (the optional ``stream`` extra), the same fitter reads an
#     ``LSLInlet`` and publishes with an ``LSLOutlet``, so it drops into an LSL setup
#     unchanged:
#
#     .. code-block:: python
#
#         from evoparam.stream import LSLInlet, LSLOutlet
#         inlet = LSLInlet(stype="EEG")
#         fitter = StreamFitter(template, sfreq=inlet.sfreq, ch_names=inlet.ch_names,
#                               artifact_window=ARTIFACT)
#         outlet = LSLOutlet(fitter)   # tagged "research only; not a medical device"
#         inlet.feed(fitter, duration=60.0, callback=outlet.push)

###################################################################################################
# Conclusion
# ----------
#
# The online mode reuses an offline fit as a template and tracks it pulse by pulse: one
# amplitude, latency and existence score per channel, in milliseconds, with a running test
# that stays honest under repeated looks. It targets soft real time (a latency budget), not
# hard deadlines, and it reports numbers only; closing a loop on them is the user's
# responsibility.
