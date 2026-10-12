# SPDX-License-Identifier: Apache-2.0
"""
CCEPs in stereo-EEG: the ``ccep_seeg`` preset
=============================================

In sEEG the contacts lie along depth electrodes, in grey and white matter; the
``ccep_seeg`` preset re-references bipolarly (neighbours on a shaft) and otherwise follows
``ccep``. Because a fixed common average spreads the stimulation response over every
contact, the adjusted common average (CARLA: only the contacts whose responses do not
correlate across trials) is shown as the alternative. The response shapes are shared across
the contacts of a shaft (one latency and width per component, free amplitudes), which makes
the weak contacts' fits stable. Responses of both polarities are fitted (in sEEG the sign
of an N1 depends on where the contact lies relative to the generator).

Data: OpenNeuro ds004977 (Mayo Clinic, CC0), subject 1, run 01; tables checked against the
registry, the recording (4800 Hz) read by byte ranges. The deposit keeps the signal around
one stimulation site per run (ROC11-ROC12 here) and writes zeros elsewhere: reading another
site gives epochs of zeros, and ``read_bids`` warns.
"""

# %%
# Tables and the epochs of one site
# ---------------------------------

import os

import numpy as np
import pandas as pd

from evoparam import reports
from evoparam.io import fetch_openneuro_file, read_bids
from evoparam.prep import rereference

DATASET = "ds004977"
STEM = "sub-1/ses-ieeg01/ieeg/sub-1_ses-ieeg01_task-ccep_run-01"
cache = os.path.join(os.path.expanduser("~"), ".cache", "evoparam", "openneuro")
events = pd.read_csv(fetch_openneuro_file(DATASET, STEM + "_events.tsv", folder=cache), sep="\t")
channels = pd.read_csv(
    fetch_openneuro_file(DATASET, STEM + "_channels.tsv", folder=cache), sep="\t"
)
site = "ROC11-ROC12"
stim = events[events["electrical_stimulation_site"] == site]
print(
    len(stim),
    "pulses on",
    site,
    "; polarities:",
    stim["electrical_stimulation_type"].value_counts().to_dict(),
)
# the stimulated shaft (ROC) and a neighbouring one (RB)
shaft = [
    n
    for n, t in zip(channels["name"], channels["type"], strict=False)
    if t == "SEEG" and n.rstrip("0123456789") in ("ROC", "RB") and n not in site.split("-")
]
trials = read_bids(
    DATASET,
    "1",
    session="ieeg01",
    task="ccep",
    run="01",
    remote=True,
    select={"stim_site": site},
    picks=shaft,
    epoch=(-1.0, 2.0),
)
print(trials)

# %%
# The preset's chain (bipolar reference)
# --------------------------------------

result = reports.run_pipeline(trials, "ccep_seeg", random_state=0, trial_fits=False)
table = result.group.to_dataframe()
responding = np.array(result.test.settings["channels"])[np.ravel(result.test.significant)]
print("responding bipolar channels:", list(responding))
print(
    table[table["label"] == "N1"][["channel", "latency", "amplitude", "polarity"]]
    .round(4)
    .to_string(index=False)
)

# %%
# Adjusted common average instead
# -------------------------------
# CARLA keeps out of the average the contacts whose responses correlate across trials.

from evoparam.prep import car_adaptive_select

selection = car_adaptive_select(trials.data, trials.times, random_state=0)
members = [trials.ch_names[i] for i in selection["members"]]
left_out = sorted(set(trials.ch_names) - set(members))
print(f"{len(members)} contacts in the adjusted average; left out (responsive): {left_out}")
carla = rereference(trials, "car_adaptive", random_state=0)

# %%
# Shapes shared along the shaft
# -----------------------------

from evoparam.spatial import fit_shared

shared = fit_shared(result.trials, channels=[str(c) for c in responding], preset="ccep_seeg")
print(
    shared.to_dataframe()[["channel", "component", "latency", "amplitude"]]
    .head(12)
    .round(4)
    .to_string(index=False)
)

# %%
# Known limit (under investigation)
# ---------------------------------
# On this run the shared fit returns a component before the stimulus (outside the fit
# window) and a pair of opposite components 3 ms apart that cancel each other: the same
# model-selection problem as in the ECAP tutorials, being fixed (CHANGELOG). Use the
# per-channel fits above until then.

# %%
# Methods paragraph
# -----------------

print(reports.methods_text(result.results))
