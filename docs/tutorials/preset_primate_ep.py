# SPDX-License-Identifier: Apache-2.0
"""
Macaque visual evoked LFP: the ``primate_ep`` preset
====================================================

Laminar local field potentials of the visual cortex: a fast negative component in the
middle layers whose polarity reverses at depth, then slower components. The ``primate_ep``
preset fits 5 to 300 ms with gamma and skewed Gaussian shapes after a template artifact
removal; the trials of each behavioural state are averaged separately.

Data: simulated macaque recording, 16 contacts 40 µm apart, awake and NREM states
(no public macaque dataset is in the registry yet; the preset is validated on simulations).
"""  # noqa: D205, D400  (sphinx-gallery: the docstring opens with an RST title)

# %%
# Simulate a laminar recording
# ----------------------------

import numpy as np

import evoparam
from evoparam import presets, sim

ts, truth = sim.simulate_animal_ep(n_trials=80, species="macaque", n_channels=16, seed=0)
print(ts)
print(ts.metadata["state"].value_counts())
print(presets.methods_note("primate_ep"))

# %%
# Fit each state separately
# -------------------------

group = evoparam.fit(ts, "primate_ep", by="state", existence=False, random_state=0)
table = group.to_dataframe()
first = table[table["component"] == 0]
summary = first.groupby("condition").agg(
    latency_ms=("latency", lambda v: 1e3 * v.median()), amplitude=("amplitude", "median")
)
print(summary.round(2))

# %%
# Polarity reversal with depth
# ----------------------------
# The sign of the first component along the probe (awake trials).

awake = first[first["condition"] == "awake"].set_index("channel")["amplitude"]
signs = np.sign(awake.reindex(ts.ch_names).to_numpy())
print("".join("+" if s > 0 else "-" if s < 0 else "." for s in signs))

# %%
# Figure
# ------

import matplotlib

matplotlib.use("Agg")
from evoparam import plots

fig = plots.plot_group(group, kind="grid", max_channels=8)

# %%
# Methods paragraph
# -----------------

print(evoparam.reports.methods_text(group))
