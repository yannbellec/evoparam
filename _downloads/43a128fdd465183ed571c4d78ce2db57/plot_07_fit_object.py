# SPDX-License-Identifier: Apache-2.0
"""
07: Exploring the fit object
============================

What else a fit keeps: its settings, its curves, the classical measures, its history, and
how to save it.
"""  # noqa: D205, D400  (sphinx-gallery: the docstring opens with an RST title)

###################################################################################################
# Introduction
# ------------
#
# Tutorial 02 read the bumps, the background, the scores and the table of a fit. A fit
# keeps more than that. Here we look at the rest of what it holds:
#
# - ``fit.settings``: every setting that was used;
# - ``fit.model``, ``fit.residual`` and ``fit.window``: the fitted curve and what is left;
# - ``fit.classical``: the usual measures, taken directly on the curve;
# - ``fit.provenance``: how the fit was made;
# - ``fit.save`` and ``load``: keep a fit on disk and open it again.

###################################################################################################

# Import what we need
import os
import tempfile

import numpy as np

from evoparam import fit_response, gen_response
from evoparam.data import ResponseFit

###################################################################################################

# The curve of tutorial 01, and its fit
times, signal = gen_response([[25, -150, 10], [150, -80, 80]], seed=3)
fit = fit_response(times, signal, preset="ccep", delta=20, random_state=0)

###################################################################################################
# The settings
# ------------
#
# ``fit.settings`` is a dictionary with every setting of the fit: those of the preset and
# those you gave. ``_changed_from_preset`` lists the ones you changed, so that they can be
# found later.

###################################################################################################

print(len(fit.settings), "settings")
print("fit window:", fit.settings["fit_window"], "s")
print("delta:", fit.settings["delta"])
print("changed from the preset:", fit.settings["_changed_from_preset"])

###################################################################################################
#
# Here we gave ``delta=20``, so it is listed as changed. The methods text (tutorial 12)
# reads the same list, so a changed setting is always written in the methods.

###################################################################################################
# The curves
# ----------
#
# ``fit.window`` is the part of the curve that was fitted (9 to 500 ms for ``"ccep"``).
# ``fit.model`` is the fitted curve on that window, and ``fit.residual`` what is left
# (data minus model). They have one value per time point of the window.

###################################################################################################

in_window = (times >= fit.window[0]) & (times <= fit.window[1])
print("window:", fit.window, "s,", in_window.sum(), "time points")
print("model:", fit.model.shape, " residual:", fit.residual.shape)
print(f"largest leftover: {np.abs(fit.residual).max():.1f} µV")

###################################################################################################
#
# With these, you can draw the fit yourself, or compute your own measure of the leftover.

###################################################################################################
# The classical measures
# ----------------------
#
# Most papers measure a response directly on the curve: the lowest point in a window
# (peak amplitude and latency), the mean over a window, the area. ``fit.classical`` gives
# these, measured the usual way, so that evoparam's numbers can be compared with them.

###################################################################################################

classical = fit.classical
print(f"lowest point: {1000 * classical['peak_latency']:.1f} ms, ", end="")
print(f"{classical['peak_amplitude']:.0f} µV")
print(
    f"N1 of the model: {fit.components[0].latency_ms:.1f} ms, {fit.components[0].amplitude:.0f} µV"
)

###################################################################################################
#
# The lowest point of the curve is at 25.9 ms and −128 µV; the model's N1 is at 25.0 ms
# and −152 µV. They differ for two reasons:
#
# - the lowest point is measured from zero, and the model's amplitude from the
#   background, which is still above zero at 25 ms (it fades from 60 µV);
# - the lowest point moves with the noise of the samples around it, while the model's
#   latency uses the whole bump.
#
# ``classical["definition"]`` says exactly how each classical measure was taken.

###################################################################################################

print(classical["definition"]["peak_amplitude"])

###################################################################################################
# The history
# -----------
#
# ``fit.provenance`` records how the fit was made: the version of evoparam, the date, the
# function and its settings, the random seed, and the computer (Python and package
# versions). This is what allows a result to be made again (tutorial 12).

###################################################################################################

step = fit.provenance.steps[0]
print(step["function"], "| evoparam", fit.provenance.package_version)
print("seed:", step["random_state"], "| python", fit.provenance.environment["python"])

###################################################################################################
# Saving and opening a fit
# ------------------------
#
# ``fit.save`` writes the whole fit (bumps, background, scores, settings, history) in a
# ``.evoparam`` folder, or a single ``.zip`` file. ``ResponseFit.load`` opens it again,
# and checks that it was not changed.

###################################################################################################

folder = tempfile.mkdtemp()
path = os.path.join(folder, "my_fit.zip")
fit.save(path)

again = ResponseFit.load(path)
print(again.components[0].label, f"{again.components[0].latency_ms:.1f} ms")

###################################################################################################
#
# ``fit.to_dict()`` gives the same content as a plain dictionary, for example to store it
# in your own format.

###################################################################################################
# Conclusion
# ----------
#
# Besides the bumps and the scores, a fit keeps its settings (and which ones you changed),
# its fitted curve and leftover, the classical measures to compare with, and the history
# of how it was made. It can be saved and opened again with all of it.
