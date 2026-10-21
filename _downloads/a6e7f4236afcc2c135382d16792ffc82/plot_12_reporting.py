# SPDX-License-Identifier: Apache-2.0
"""
12: Reporting and referencing
=============================

Write what was done, share the results, and cite the tools.
"""  # noqa: D205, D400  (sphinx-gallery: the docstring opens with an RST title)

###################################################################################################
# Introduction
# ------------
#
# A result is only useful if others can tell how it was made, and make it again. evoparam
# keeps a record of every step (tutorial 07). This last tutorial turns that record into:
#
# 1. the **methods paragraph** of a paper;
# 2. a **report** to share (HTML, PDF, Markdown) and a full **JSON** file;
# 3. the **citations** of evoparam and of the methods it used;
# 4. a way to **make the result again** later.

###################################################################################################

# Import what we need
import os
import tempfile

import evoparam
from evoparam import fit_response, gen_response, reports

###################################################################################################

# The curve of tutorial 01, fitted with one setting changed from the preset
times, signal = gen_response([[25, -150, 10], [150, -80, 80]], seed=3)
fit = fit_response(times, signal, preset="ccep", delta=20, random_state=0)

###################################################################################################
# 1. The methods paragraph
# ------------------------
#
# ``reports.methods_text`` writes, in plain English, every step that was recorded: the
# preset and its version, the model, how bumps were chosen, the windows of the names, how
# the uncertainty was computed, every setting changed from the preset, the random seed,
# and the citations.

###################################################################################################

print(reports.methods_text(fit))

###################################################################################################
#
# The setting we changed (``delta=20``) is written in it: "kept when the BIC decreased by
# at least 20". This is why it is safer to give settings to evoparam than to change the
# results by hand afterwards: what is given is recorded.
#
# With real data, run through ``reports.run_pipeline`` (tutorial 08), the same function
# also describes the reading, the artifact removal, the re-referencing, the baseline and
# the test of a response.
#
# .. note::
#
#     Adapt the wording to your paper, but keep the numbers: they are the ones that were
#     used. ``cite=False`` leaves out the citations, if you add them yourself.

###################################################################################################
# 2. Reports to share
# -------------------
#
# Three report formats hold the methods text, a table of the results, a summary of the
# quality of the fits, and the steps of the record:
#
# - ``reports.html(result, path)``: one HTML file, with the figures inside it, that opens
#   in any browser without internet;
# - ``reports.pdf(result, path)``: the same as a PDF;
# - ``reports.markdown(result)``: the same as Markdown text.
#
# ``reports.to_json(result, path)`` writes **everything** (every number, setting and
# step) in one JSON file, which any language can read.

###################################################################################################

folder = tempfile.mkdtemp()

page = os.path.join(folder, "my_fit.html")
reports.html(fit, page)
print(f"HTML report: {os.path.getsize(page) / 1000:.0f} kB")

everything = os.path.join(folder, "my_fit.json")
reports.to_json(fit, everything)
print(f"JSON file: {os.path.getsize(everything) / 1000:.0f} kB")

###################################################################################################
#
# With many electrodes (a group fit, or the result of ``run_pipeline``), the reports show
# one figure per electrode, up to ``max_channels``. To store results next to a BIDS
# dataset, use ``evoparam.io.write_bids_derivatives``.

###################################################################################################
# 3. Citations
# ------------
#
# ``evoparam.cite()`` prints how to cite evoparam (with its version); ``bibtex=True``
# gives an entry for a reference manager. The methods text also cites the published
# methods that the steps used (for example the false-discovery-rate correction of the
# test of a response).

###################################################################################################

text = evoparam.cite()

###################################################################################################

entry = evoparam.cite(bibtex=True)

###################################################################################################
#
# Also cite the data: for the recording of tutorials 08 and 11, van Blooijs et al. (2023),
# *Nature Neuroscience*, and its OpenNeuro identifier (ds004080).

###################################################################################################
# 4. Making the result again
# --------------------------
#
# A saved fit (``fit.save``, tutorial 07) holds its full record: the version of evoparam,
# every setting, the seed, and where the data came from. ``evoparam.reproduce(path)``
# runs the fit again from that record and compares every number with the saved ones.
# When the data were not read from a file (for example a curve made in Python, as here),
# give them with ``data=``.
#
# The record also lists the computer's versions of Python and of the main packages, so
# that a difference in a later run can be traced to its cause.

###################################################################################################

saved = os.path.join(folder, "my_fit.zip")
fit.save(saved)
print("saved with evoparam", fit.provenance.package_version)

###################################################################################################
# Conclusion
# ----------
#
# evoparam writes the methods paragraph from what was actually done, makes reports to
# share and a complete JSON file, says how to cite it, and keeps what is needed to make
# a result again. Report every setting you changed, each number with its 95 % range
# (tutorial 06), and cite both the tools and the data.
#
# This is the end of the tutorials: from what the model is (01) to how to report it (12).
