# SPDX-License-Identifier: Apache-2.0
"""
P300 for brain-computer interfaces: the ``p300_bci`` preset
===========================================================

In a P300 speller, the attended (target) flashes evoke a P3 that the non-target flashes do
not. The ``p300_bci`` preset fits 200 to 700 ms; trial-by-trial amplitudes of the P3 then
serve as features for a classifier (see the scikit-learn tutorial).

Data: simulated scalp EEG, targets (P3 present) and non-targets, 8 channels.
"""

# %%
# Simulate targets and non-targets
# --------------------------------

import numpy as np
import pandas as pd

import evoparam
from evoparam import presets, sim
from evoparam.data import TrialSet

targets, _ = sim.simulate_erp(60, n_channels=8, components=("P1", "N1", "P3"), seed=0, snr=3)
others, _ = sim.simulate_erp(60, n_channels=8, components=("P1", "N1"), seed=1, snr=3)
metadata = pd.concat(
    [targets.metadata.assign(condition="target"), others.metadata.assign(condition="non_target")],
    ignore_index=True,
)
ts = TrialSet(
    data=np.concatenate([targets.data, others.data]),
    times=targets.times,
    sfreq=targets.sfreq,
    ch_names=targets.ch_names,
    metadata=metadata,
)
print(presets.methods_note("p300_bci"))

# %%
# Fit each condition
# ------------------

group = evoparam.fit(ts, "p300_bci", by="condition", existence=False, random_state=0)
table = group.to_dataframe()
p3 = table[table["label"] == "P3"]
print(p3.groupby("condition")["amplitude"].describe()[["count", "mean"]].round(2))
print("channels with a P3, per condition:", p3.groupby("condition")["channel"].nunique().to_dict())

# %%
# Figure
# ------

import matplotlib

matplotlib.use("Agg")
from evoparam import plots

target = [
    group.fits[int(p)]
    for c, p in zip(group.index["condition"], group.index["position"])
    if c == "target"
]
fig = plots.plot_fit(
    target[0],
    ts.select(trials=list(np.flatnonzero(ts.metadata["condition"] == "target"))).average(),
)

# %%
# Methods paragraph
# -----------------

print(evoparam.reports.methods_text(group))
