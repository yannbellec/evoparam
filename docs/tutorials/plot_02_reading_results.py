# SPDX-License-Identifier: Apache-2.0
"""
02: Reading the results
=======================

Get the numbers out of a fit, one by one or all in a table.
"""  # noqa: D205, D400  (sphinx-gallery: the docstring opens with an RST title)

###################################################################################################
# Introduction
# ------------
#
# In the first tutorial, ``report`` did everything at once: fit, summary and figure.
#
# Here we look at what a fit gives back, and how to use it: read one number, check that the
# fit is good, keep all the numbers in a table, and choose the part of the curve to describe.

###################################################################################################

# Import what we need
from evoparam import fit_response, gen_response

###################################################################################################
# Fitting without drawing
# -----------------------
#
# ``fit_response`` does only the fit: no figure, no summary. It is the one to use when you
# have many curves, or when you only want the numbers.
#
# We use the same curve as in the first tutorial: a sharp bump at 25 ms and a broad one at
# 150 ms.

###################################################################################################

# Make the example curve
times, signal = gen_response([[25, -150, 10], [150, -80, 80]], seed=3)

# Fit the model, without drawing
fit = fit_response(times, signal, preset="ccep")

###################################################################################################
#
# ``fit`` now holds everything evoparam found. ``print(fit)`` still shows the summary at any
# time.

###################################################################################################

print(fit)

###################################################################################################
# Reading the bumps
# -----------------
#
# The bumps are in ``fit.components``, a list in order of time. Each bump has the three
# numbers of the first tutorial:
#
# - ``latency``: when the bump peaks;
# - ``amplitude``: its height above the background, in µV;
# - ``fwhm``: its width at half its height ("full width at half maximum").
#
# .. note::
#
#     Inside evoparam, times are in **seconds** (``latency`` = 0.025 means 25 ms). To read
#     them in milliseconds, add ``_ms`` to the name: ``latency_ms``, ``fwhm_ms``,
#     ``onset_ms``...

###################################################################################################

# Go through the bumps, one line each
for bump in fit.components:
    print(bump.label, round(bump.latency_ms, 1), "ms,", round(bump.amplitude, 1), "µV")

###################################################################################################
#
# To keep one number, take the bump by its place in the list: ``[0]`` is the first one.

###################################################################################################

# The first bump, and its latency in ms
n1 = fit.components[0]
print(round(n1.latency_ms, 1))

###################################################################################################
#
# Each bump also has a few more numbers, all measured on the bump alone:
#
# - ``onset_ms`` and ``offset_ms``: when the bump starts and ends;
# - ``area``: the surface of the bump, the coloured area of the first tutorial
#   (µV × seconds);
# - ``rise_time_ms`` and ``decay_time_ms``: how long it takes to reach its peak, and to
#   come back.

###################################################################################################

# When the first bump starts and ends, in ms
print(round(n1.onset_ms, 1), round(n1.offset_ms, 1))

###################################################################################################
# Reading the background
# ----------------------
#
# The background is in ``fit.background``. Here it is a curve that fades away, and the
# number to read is its **time constant**: how long it takes to lose about two thirds of its
# height. The smaller it is, the faster it fades.

###################################################################################################

# The time constant of the background, in ms
print(round(fit.background_ms["tau"], 1))

###################################################################################################
# Is the fit good?
# ----------------
#
# Before using the numbers, check that the model really follows the curve. Two numbers in
# ``fit.gof`` ("goodness of fit") say it:
#
# - ``r2``: how much of the curve the model explains, from 0 (nothing) to 1 (everything);
# - ``rmse``: the typical distance between the curve and the model, in µV. It should be
#   close to the level of the noise.

###################################################################################################

print(round(fit.gof.r2, 2), round(fit.gof.rmse, 1))

###################################################################################################
#
# .. note::
#
#     Here R² is about 0.94: the model explains 94 % of the curve, and what is left is the
#     noise we put in. A low R² means the model missed something: look at the figure
#     (``report``) before trusting the numbers.

###################################################################################################
# All the numbers in a table
# --------------------------
#
# ``to_dataframe`` puts every number of every bump in a table, one row per bump. This table
# can be sorted, plotted or saved like any spreadsheet (it needs the ``pandas`` package,
# already there in most notebooks).

###################################################################################################

# One row per bump; show a few columns
table = fit.to_dataframe()
table[["label", "latency", "amplitude", "fwhm", "area"]]

###################################################################################################
#
# To keep it, save it as a file that any spreadsheet program opens:
#
# .. code-block:: python
#
#     table.to_csv("my_results.csv")

###################################################################################################
# Choosing the part of the curve to describe
# ------------------------------------------
#
# The preset decides which part of the curve is described: for ``"ccep"``, from 9 to 500 ms
# after the stimulus. To look at another part, give a ``window``, in seconds.
#
# Here we keep only the first 100 ms: the broad bump at 150 ms is left out, and only the
# sharp one is described.

###################################################################################################

# Describe only 9 to 100 ms after the stimulus
early = fit_response(times, signal, preset="ccep", window=(0.009, 0.1))
print(early)

###################################################################################################
#
# .. note::
#
#     The preset matters too: it holds the settings made for one kind of recording.
#     ``"ccep"`` is for responses to electrical pulses in the brain; ``"erp"`` is for
#     responses recorded on the scalp (EEG), ``"sep"`` for touch, ``"tep"`` for magnetic
#     pulses (TMS), and there are a few more.

###################################################################################################
# Conclusion
# ----------
#
# A fit gives back its bumps (``fit.components``), its background (``fit.background``) and
# how good it is (``fit.gof``), and ``to_dataframe`` puts everything in one table. With
# ``window`` and ``preset``, you choose what is described.
