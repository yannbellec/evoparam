# SPDX-License-Identifier: Apache-2.0
"""
20: The command line
=====================

Run the whole pipeline from a shell, with no Python, and read the JSON summary it prints.
"""  # noqa: D205, D400  (sphinx-gallery: the docstring opens with an RST title)

###################################################################################################
# Introduction
# ------------
#
# Everything the tutorials did in Python is also a shell command. The ``evoparam`` console
# script has thirteen sub-commands, for a pipeline you can put in a batch script, a cluster
# job or a Makefile without writing any Python:
#
# - ``simulate`` writes a synthetic BIDS dataset with a known truth;
# - ``detect-events`` finds the pulses in a continuous file;
# - ``clean`` removes the artifact; ``test`` runs the existence test;
# - ``fit`` does the whole chain (read, clean, reference, fit, test) and writes BIDS
#   derivatives;
# - ``report`` makes an HTML, PDF or Markdown report;
# - ``validate-bids``, ``reproduce``, ``migrate``, ``info``, ``presets`` and ``cite``.
#
# Every command takes ``--json`` (one JSON summary on stdout), ``--random-state`` and
# ``--config`` (a JSON or TOML file of options). The **exit code** says what happened:
# 0 success, 2 usage error, 3 partial (some channel failed), 4 a missing optional package,
# 5 a bad input, 6 a reproduction mismatch. We drive it here with :mod:`subprocess`, exactly
# as a shell would.

###################################################################################################

# Import what we need
import json
import subprocess
import sys
import tempfile
from pathlib import Path

work = Path(tempfile.mkdtemp(prefix="evoparam-cli-"))


def evoparam(*args):
    """Run one ``evoparam`` command with ``--json`` and return its parsed summary."""
    proc = subprocess.run(
        [sys.executable, "-m", "evoparam", *map(str, args), "--json"],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    summary = json.loads(proc.stdout.splitlines()[-1])
    cmd = " ".join(map(str, args))
    print(f"$ evoparam {cmd}  ->  {summary['status']} (exit {proc.returncode})")
    return summary


###################################################################################################
# 1. Make a dataset
# -----------------
#
# ``simulate`` writes a small BIDS dataset with ground truth, so the rest of the tutorial
# needs no data of your own. In real use you point ``fit`` at your own BIDS root instead.

###################################################################################################

evoparam("simulate", "--modality", "ccep", "--n-trials", "10", "--n-channels", "4",
         "--seed", "0", "--out", work / "bids")  # fmt: skip

###################################################################################################
# 2. Check the stimulation metadata, then fit
# -------------------------------------------
#
# ``validate-bids`` reports, run by run, which stimulation fields are present, inferred,
# missing or inconsistent. Then ``fit`` runs the whole chain and writes the derivatives.

###################################################################################################

summary = evoparam("validate-bids", work / "bids")
print("  metadata:", summary["result"]["counts"])

summary = evoparam("fit", work / "bids", "--sub", "01", "--task", "ccep", "--preset", "ccep",
                    "--random-state", "0",
                    "--out", work / "bids/derivatives/evoparam")  # fmt: skip
print("  units:", summary["n_units"])
print("  wrote:", len(summary["outputs"]), "files")

###################################################################################################
# 3. Report, and the JSON summary
# -------------------------------
#
# ``report`` turns the derivatives into a Markdown (or HTML, or PDF) report. The ``--json``
# object is the same on every command: ``status``, ``exit_code``, ``outputs``, ``n_units``
# (ok, no_response, failed, skipped), ``warnings`` and ``duration_s``. A script reads that
# object to decide what to do next, and the exit code to stop on an error.

###################################################################################################

summary = evoparam("report", work / "bids/derivatives/evoparam", "--format", "md",
                   "--out", work / "report")  # fmt: skip
print("\nfull JSON summary of the last command:")
keys = ("command", "status", "exit_code", "n_units")
print(json.dumps({key: summary[key] for key in keys}, indent=2))

###################################################################################################
#
# The same four lines, as a shell script, are:
#
# .. code-block:: bash
#
#     evoparam simulate --modality ccep --n-trials 10 --n-channels 4 --seed 0 --out bids/
#     evoparam validate-bids bids/
#     evoparam fit bids/ --sub 01 --task ccep --preset ccep --out bids/derivatives/evoparam
#     evoparam report bids/derivatives/evoparam --format html
#
# ``evoparam fit bids/ ...`` writes exactly the same files as the Python
# ``write_bids_derivatives(fit(read_bids(...)))``, so you can prototype in Python and run in
# a shell, or the other way round.

###################################################################################################
# Conclusion
# ----------
#
# The command line is the whole package without Python: ``simulate`` for test data,
# ``fit`` for the pipeline, ``report`` for the write-up, and the smaller commands for
# events, cleaning, the existence test, validation, reproduction and citation. ``--json``
# plus the exit codes make it safe to script over many subjects on a cluster.
