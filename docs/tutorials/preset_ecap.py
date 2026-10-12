# SPDX-License-Identifier: Apache-2.0
"""
Spinal evoked compound action potentials: the ``ecap`` preset
=============================================================

During spinal cord stimulation, the ECAP of the dorsal columns (N1 then P2, within 3 ms)
follows a stimulation artifact thousands of times larger. The ``ecap`` preset cancels the
artifact by alternating the pulse polarity (the neural response keeps its sign, the
artifact flips) and fits a biexponential N1 and a Gaussian P2.

Data: simulated epidural recording, 8 contacts, alternating biphasic pulses at 30 kHz.
"""  # noqa: D205, D400  (sphinx-gallery: the docstring opens with an RST title)

# %%
# Simulate alternating-polarity ECAPs
# -----------------------------------

from evoparam import presets, reports, sim

ts, truth = sim.simulate_ecap(40, kind="spinal", paradigm="alternating", seed=0)
print(ts)
print(ts.metadata["stim_polarity"].value_counts())
print(presets.methods_note("ecap"))

# %%
# The preset's chain
# ------------------

result = reports.run_pipeline(ts, "ecap", random_state=0, trial_fits=False, existence=False)
table = result.group.to_dataframe()
print(table[["channel", "label", "latency", "amplitude"]].round(5).head(8))

# %%
# Against the simulation
# ----------------------

scores = sim.score_fits(truth, result.group)
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

import matplotlib

matplotlib.use("Agg")
from evoparam import plots

fig = plots.plot_fit(result.group.fits[0], result.trials.average())

# %%
# Methods paragraph
# -----------------

print(reports.methods_text(result.results))
