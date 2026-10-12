# SPDX-License-Identifier: Apache-2.0
"""
Reporting your results
======================

What a paper needs from an evoked-response analysis: the settings (preset and version, and
every setting changed), the package version, the goodness of fit, the existence test, the
artifact method and the montage. evoparam writes all of it from the result's provenance:

- ``methods_text()``: the methods paragraph, with the citations;
- ``to_dataframe()``: the tidy table of every component, with units;
- BIDS derivatives (``write_bids_derivatives``) or NWB (``write_nwb``) to share;
- ``save()`` and ``evoparam.reproduce``: the result regenerated from its provenance;
- ``evoparam.reports.citation()``: how to cite evoparam (``CITATION.cff``).

Data: simulated CCEPs through the ``ccep`` preset chain.
"""

# %%
# A result
# --------

import os
import tempfile

import evoparam
from evoparam import reports, sim

trials, _ = sim.simulate_ccep(n_trials=12, n_channels=10, seed=0)
result = reports.run_pipeline(trials, "ccep", random_state=0, trial_fits=False)

# %%
# The methods paragraph
# ---------------------
# Built from the provenance only: what was run, with which settings, in which order.

print(reports.methods_text(result.results))

# %%
# The tidy table, with units
# --------------------------

table = result.group.to_dataframe()  # one row per component
print(table[["channel", "label", "latency", "amplitude", "fwhm", "existence_p"]].round(4).head())
print({k: v for k, v in table.attrs["units"].items() if k in ("latency", "amplitude")})
fits = result.group.to_dataframe("fit")  # one row per channel: goodness of fit
print(fits[["channel", "status", "n_components", "r2", "bic"]].round(3))

# %%
# Save, then reproduce
# --------------------

folder = tempfile.mkdtemp()
path = result.group.save(os.path.join(folder, "group"))
report = evoparam.reproduce(path)
print("identical:", report.identical, "max relative difference:", report.max_rel_diff)

# %%
# A report to share
# -----------------

markdown = reports.markdown(result.group)
print(markdown[:400])

# %%
# How to cite
# -----------

print(reports.citation())

# %%
# The methods paragraph, to paste in the paper
# --------------------------------------------

print(reports.methods_text(result.results))
