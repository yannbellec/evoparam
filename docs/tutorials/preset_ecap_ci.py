# SPDX-License-Identifier: Apache-2.0
"""
Cochlear-implant ECAPs: the ``ecap_ci`` preset
======================================================

The electrically evoked compound action potential of the auditory nerve (N1 about 0.3 ms
after the pulse, then P2) is recorded on the implant contacts. The ``ecap_ci`` preset
removes the artifact with the forward-masking paradigm (A - B + C - D: probe alone,
masker and probe, masker alone, no pulse) and fits N1 and P2.

Data: simulated cochlear implant recording, forward masking, 30 kHz.
"""  # noqa: D205, D400  (sphinx-gallery: the docstring opens with an RST title)

# %%
# Simulate forward-masking ECAPs
# -------------------------------

from evoparam import presets, reports, sim

ts, truth = sim.simulate_ecap(40, kind="cochlear", paradigm="forward_masking", seed=0)
print(ts)
print(ts.metadata["condition"].value_counts())
print(presets.methods_note("ecap_ci"))

# %%
# The preset's chain
# ------------------

result = reports.run_pipeline(ts, "ecap_ci", random_state=0, trial_fits=False, existence=False)
table = result.group.to_dataframe()
print(table[["channel", "label", "latency", "amplitude"]].round(5).head(8))

# %%
# Against the simulation
# ----------------------
# The forward-masking combination A - B + C - D keeps the response to the probe (condition
# A of the simulation).

truth_a = truth.to_dataframe()
truth_a = truth_a[truth_a["condition"] == "A"]
scores = sim.score_fits(truth_a, result.group)
print(scores[["recall", "precision", "latency_mae", "amplitude_rel_mae"]].round(4))

# %%
# Known limit (under investigation)
# ---------------------------------
# On some contacts the fit returns two large components of opposite sign at almost the same
# latency instead of the N1 and P2 (a "cancelling pair"; above, the rows whose labels are
# C1/C3 with amplitudes larger than the simulated ones). It happens on clean simulated
# waveforms too, so it comes from the model selection, not from the artifact removal; it
# is being fixed (CHANGELOG). Until then, check that N1 and P2 are labelled and have
# plausible widths before using the amplitudes.

# %%
# Figure
# ------

import matplotlib  # noqa: E402  (sphinx-gallery section)

matplotlib.use("Agg")
from evoparam import plots  # noqa: E402  (sphinx-gallery section)

fig = plots.plot_fit(result.group.fits[0], result.trials.average())

# %%
# Methods paragraph
# -----------------

print(reports.methods_text(result.results))
