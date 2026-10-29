# SPDX-License-Identifier: Apache-2.0
"""
13: Loading your own data
=========================

Turn a numpy array, an MNE object or a file into a ``TrialSet`` that evoparam can fit.
"""  # noqa: D205, D400  (sphinx-gallery: the docstring opens with an RST title)

###################################################################################################
# Introduction
# ------------
#
# The first twelve tutorials used made-up data from ``evoparam.sim``, so that we always
# knew the truth. Your own recording comes as one of three things:
#
# 1. a **numpy array** you built yourself (``from_array``);
# 2. an **MNE** object (``Epochs``, ``Evoked``), ``read_mne``;
# 3. a **file** on disk (BrainVision, EDF, FIF, NWB, BIDS folder), ``read``.
#
# All three give the same object, a ``TrialSet``: trials x channels x time, with the
# sampling rate, the channel names, the units and the time of the stimulus. Every later
# step (clean, fit, test) works on that object, whatever it came from.

###################################################################################################

# Import what we need
import numpy as np

from evoparam import models, reports
from evoparam.io import from_array, read_mne

###################################################################################################
# 1. From a numpy array
# ---------------------
#
# This is the lowest level and the one that always works. You give the data as an array
# shaped ``(n_trials, n_channels, n_times)`` (one trial, one channel and one curve are
# also allowed), the sampling rate in Hz, and the time of the first sample. Here we build
# 20 trials of 2 channels: channel ``RESP`` has a small negative bump at 25 ms (an N1),
# channel ``FLAT`` has only noise.

###################################################################################################

sfreq = 1024.0
times = np.arange(-0.2, 0.5, 1.0 / sfreq)  # seconds, stimulus at t = 0
rng = np.random.default_rng(0)

bump = -60.0 * np.exp(-0.5 * ((times - 0.025) / 0.004) ** 2)  # N1: -60 uV, 25 ms, narrow
data = rng.normal(0.0, 10.0, size=(20, 2, times.size))  # 20 trials, 2 channels, noise
data[:, 0, :] += bump  # add the N1 to the first channel only

ts = from_array(data, sfreq=sfreq, times=times, ch_names=["RESP", "FLAT"], units="uV")
print(ts)

###################################################################################################
#
# ``times`` sets the sampling rate and the stimulus time together: ``t = 0`` is the pulse,
# negative times are the baseline. If you pass ``times``, do not pass ``sfreq`` as well;
# evoparam reads the rate from the spacing of the samples. ``units="uV"`` tells evoparam
# the amplitudes are microvolts, so the reports carry the right unit.
#
# The object is ready to fit. We use ``models.fit`` with no preset: it picks a generic
# setting from the sampling rate. The N1 of ``RESP`` comes back near 25 ms; ``FLAT`` gets
# no bump.

###################################################################################################

group = models.fit(ts, window=(0.005, 0.3), random_state=0)
table = group.to_dataframe().dropna(subset=["label"])
print(table[["channel", "label", "latency", "amplitude", "fwhm"]].round(4).to_string(index=False))

###################################################################################################
# 2. From an MNE object
# ---------------------
#
# If your data are already in MNE (``Epochs`` or ``Evoked``), ``read_mne`` converts them
# without going through a file. evoparam reads the sampling rate, the channel names, the
# times (including the baseline) and the units from the object. Here we make a small
# ``EpochsArray`` from the same array, then read it back.

###################################################################################################

import mne  # noqa: E402  (optional dependency, imported where it is used)

info = mne.create_info(["RESP", "FLAT"], sfreq=sfreq, ch_types="eeg")
epochs = mne.EpochsArray(data * 1e-6, info, tmin=float(times[0]), verbose="error")  # MNE: volts
ts_mne = read_mne(epochs)
print(ts_mne)

###################################################################################################
#
# MNE stores EEG in volts; evoparam keeps microvolts, so the numbers match what we started
# with. ``read_mne`` also accepts an ``Evoked`` (one averaged curve) and a
# ``units="uV"`` keyword if your object is already in microvolts.

###################################################################################################
# 3. From a file
# --------------
#
# ``read`` is the single door for files. It guesses the format from the name, or you can
# name it with ``format=``:
#
# .. code-block:: python
#
#     from evoparam.io import read
#
#     ts = read("session.vhdr")           # BrainVision
#     ts = read("recording.edf")          # EDF / EDF+
#     ts = read("epochs-epo.fif")         # MNE FIF
#     ts = read("file.nwb")               # NWB
#     ts = read("trials.npz")             # a saved evoparam array
#
# A continuous file (not yet cut into trials) has no stimulus times. Find them from the
# signal with ``detect_events`` (tutorial 14), or pass the event times you already have.
# For a whole **BIDS** dataset, ``read_bids`` reads one subject, session, task and run,
# and can read a few pulses straight from the server without downloading the file
# (``remote=True``), as shown in tutorial 08.

###################################################################################################
# 4. Saving the result
# --------------------
#
# After a fit, ``write_bids_derivatives`` writes the parameters next to the data as BIDS
# derivatives (TSV tables plus a JSON with every setting), so the result is self-describing
# and can be re-read. The reporting tutorial (12) covers the methods text, the JSON and
# the HTML report. Here we only show the call; it returns the list of files written.
#
# .. code-block:: python
#
#     from evoparam.io import write_bids_derivatives
#
#     result = reports.run_pipeline(ts, "ccep", random_state=0)
#     files = write_bids_derivatives(result, subject="01", task="SPESclin", out="derivatives/")

###################################################################################################
# Conclusion
# ----------
#
# Whatever the source, the first step is always to get a ``TrialSet``: ``from_array`` for
# a numpy array, ``read_mne`` for an MNE object, ``read`` or ``read_bids`` for a file or a
# dataset. Set the units and the stimulus time right at this step; everything downstream
# trusts them. From here, the next tutorials clean the trials, test for a response, and
# fit the bumps exactly as they did on the made-up data.
_ = reports  # noqa: B018  (named in the text; kept imported for the reader)
