"""
BIDS: read a deposit, check its stimulation metadata, write derivatives
=======================================================================

1. :func:`evoparam.io.validate_stim_metadata` tells, run by run, which stimulation fields a
   BIDS dataset states (present, inferred, missing, inconsistent);
2. :func:`evoparam.io.read_bids` reads epochs (here remotely, by byte ranges) with the
   stimulation columns of every deposit harmonised;
3. :func:`evoparam.io.write_bids_derivatives` writes the results as BIDS derivatives that
   pass the BIDS validator, and :func:`evoparam.io.read_bids_derivatives` reads them back.

Data: OpenNeuro ds003708 (Mayo Clinic, CC0), subject 01, run 01; the run's tables are
downloaded with checksums into a local skeleton of the dataset.
"""

# %%
# A local skeleton of the run
# ---------------------------

import os
import tempfile

import numpy as np

from evoparam import reports
from evoparam.io import (fetch_openneuro_file, format_stim_report, read_bids,
                         read_bids_derivatives, validate_stim_metadata,
                         write_bids_derivatives)

DATASET = "ds003708"
STEM = "sub-01/ses-ieeg01/ieeg/sub-01_ses-ieeg01_task-ccep_run-01"
work = tempfile.mkdtemp(prefix="evoparam-bids-")
for name in ("dataset_description.json", STEM + "_events.tsv", STEM + "_channels.tsv",
             STEM + "_ieeg.json"):
    fetch_openneuro_file(DATASET, name, folder=work)
root = os.path.join(work, DATASET)

# %%
# Which stimulation metadata does the deposit state?
# --------------------------------------------------

table = validate_stim_metadata(root)
print(format_stim_report(table))

# %%
# Epochs of one site, read remotely
# ---------------------------------

trials = read_bids(DATASET, "01", session="ieeg01", task="ccep", run="01", remote=True,
                   select={"stim_site": "LTG1-LTG2"},
                   picks=["LTG3", "LTG4", "LTG9", "LTG10", "LTG17", "LTG18"],
                   epoch=(-1.0, 2.0))
print(trials.metadata[["stim_site", "stim_current_mA", "stim_polarity",
                       "electrical_stimulation_current"]].head(3))

# %%
# Results as BIDS derivatives
# ---------------------------

result = reports.run_pipeline(trials, "ccep", random_state=0, trial_fits=False)
out = os.path.join(root, "derivatives", "evoparam")
files = write_bids_derivatives(result.results, root, subject="01", session="ieeg01",
                               task="ccep", run="01", out=out, trials=result.trials,
                               overwrite=True)
for path in files:
    print(os.path.relpath(path, root))
with open(next(f for f in files if f.endswith("_responses.tsv")), encoding="utf-8") as fobj:
    print(fobj.readline().strip().split("\t"))

# %%
# Read back
# ---------

group = read_bids_derivatives(out, subject="01", session="ieeg01", task="ccep", run="01")
print("same result:", group == result.group)

# %%
# Methods paragraph
# -----------------

print(reports.methods_text(result.results))
