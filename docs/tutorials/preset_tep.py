# SPDX-License-Identifier: Apache-2.0
"""
TMS-evoked potentials: the ``tep`` preset
=========================================

TMS-EEG responses (N15, P30, N45, P60, N100, P180) sit under a pulse artifact, a decay
artifact and muscle activity. The ``tep`` preset interpolates the pulse window, projects
out the decay and muscle subspace, re-references to the average and fits 12 to 300 ms.

Data: simulated TMS-EEG over the left motor cortex (C3), 32 channels.
"""

# %%
# Simulate TMS-EEG
# ----------------

import evoparam
from evoparam import presets, reports, sim

ts, truth = sim.simulate_tep(n_trials=40, n_channels=32, stim_site="C3", seed=0)
print(ts)
print(presets.get("tep").artifact_method, presets.get("tep").artifact_window)

# %%
# The preset's chain
# ------------------
# Artifact (interpolation, then subspace projection), average reference, baseline, fit,
# existence test.

result = reports.run_pipeline(ts, "tep", random_state=0, trial_fits=False)
table = result.group.to_dataframe()
print(table.groupby("label")["latency"].median().round(4))
print("channels with a response:", int(result.test.significant.sum()), "of",
      len(result.test.significant))

# %%
# Under the coil: fitted and simulated components at C3
# -----------------------------------------------------
# Far from the coil the simulated components are a fraction of a microvolt, below what
# 40 trials resolve. Under the coil, in this simulation, a residual of the decay artifact
# remains around 17 ms after interpolation and subspace projection (component C1; preset
# version 1.0 described it as a pair of opposite components near 20 ms, which version 1.1
# refuses), and the N15 is not separated from it. On ds002094 (REQ-275, not met) the
# preset labels N45 and N100 on the channel of maximal response for 8 of 20 subjects:
# compare with the artifact figure below before trusting early TEPs.

fitted = table[table["channel"] == "C3"][["label", "latency", "amplitude"]]
simulated = truth.to_dataframe()
simulated = simulated[simulated["channel"] == "C3"][["label", "latency", "amplitude"]]
print("fitted:\n", fitted.round(4).to_string(index=False))
print("simulated:\n", simulated.round(4).to_string(index=False))

# %%
# Figures
# -------

import matplotlib

matplotlib.use("Agg")
from evoparam import plots

fig = plots.plot_artifact(result.artifact[-1], ts)
fig = plots.plot_existence(result.test)

# %%
# Methods paragraph
# -----------------

print(reports.methods_text(result.results))
