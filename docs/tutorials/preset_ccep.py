# SPDX-License-Identifier: Apache-2.0
"""
Cortico-cortical evoked potentials in ECoG: the ``ccep`` preset
================================================================

Single-pulse electrical stimulation of a pair of ECoG contacts evokes, on the other
contacts, an early N1 (10 to 50 ms) and a later N2 (50 to 300 ms). The ``ccep`` preset
removes the stimulation artifact (template and tail), re-references with an adaptive
common average, fits 9 to 500 ms and labels N1 and N2; an existence test tells which
contacts respond.

Data: OpenNeuro ds003708 (Mayo Clinic, CC0; Huang et al.), subject 01, run 01. The events
and channel tables are downloaded with :func:`evoparam.io.fetch_openneuro_file` and checked
against the SHA-256 of the dataset registry; the MEF 3.0 recording (about 600 MB) is read
by byte ranges, only around the pulses of one stimulation site.
"""

# %%
# The run's tables, checked against the registry
# ----------------------------------------------

import os

import numpy as np
import pandas as pd

import evoparam
from evoparam import reports
from evoparam.io import fetch_openneuro_file, read_bids

DATASET = "ds003708"
STEM = "sub-01/ses-ieeg01/ieeg/sub-01_ses-ieeg01_task-ccep_run-01"
cache = os.path.join(os.path.expanduser("~"), ".cache", "evoparam", "openneuro")
events = pd.read_csv(fetch_openneuro_file(DATASET, STEM + "_events.tsv", folder=cache),
                     sep="\t")
channels = pd.read_csv(fetch_openneuro_file(DATASET, STEM + "_channels.tsv", folder=cache),
                       sep="\t")
stim = events[events["trial_type"] == "electrical_stimulation"]
print(stim["electrical_stimulation_site"].value_counts().head())

# %%
# Epochs around the pulses of one site, read remotely
# ---------------------------------------------------
# The ECoG grid contacts only, without the two stimulated ones.

site = "LTG1-LTG2"
grid = [n for n, t in zip(channels["name"], channels["type"])
        if t == "ECOG" and n.startswith("LTG") and n not in site.split("-")][:16]
trials = read_bids(DATASET, "01", session="ieeg01", task="ccep", run="01", remote=True,
                   select={"stim_site": site}, picks=grid, epoch=(-1.0, 2.0))
print(trials)
print("bytes transferred:", trials.provenance.source["bytes_read"])

# %%
# The preset's chain
# ------------------

result = reports.run_pipeline(trials, "ccep", random_state=0, trial_fits=False)
table = result.group.to_dataframe()
n1 = table[table["label"] == "N1"][["channel", "latency", "amplitude", "fwhm"]]
print(n1.round(4).to_string(index=False))

# %%
# Existence map
# -------------
# Corrected p-value and decision of each contact.

existence = pd.DataFrame({"channel": result.test.settings["channels"],
                          "p_value": np.ravel(result.test.p_value),
                          "response": np.ravel(result.test.significant)})
print(existence.round(4).to_string(index=False))

# %%
# Figures
# -------

import matplotlib

matplotlib.use("Agg")
from evoparam import plots

fig = plots.plot_group(result.group, result.trials.average(), max_channels=16)
fig = plots.plot_existence(result.test)

# %%
# Methods paragraph
# -----------------

print(reports.methods_text(result.results))
