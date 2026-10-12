# SPDX-License-Identifier: Apache-2.0
"""
The ``evoparam`` command
========================

Everything above is also a command: ``evoparam simulate``, ``fit`` (a BIDS dataset or one
recording), ``test``, ``clean``, ``report``, ``validate-bids``, ``reproduce``, ``migrate``,
``info``, ``presets`` and ``cite``. Every command accepts ``--json`` (one JSON summary on
stdout), ``--random-state`` and ``--config`` (a JSON or TOML file of options); the exit code
tells what happened (0 success, 3 partial, 4 missing package, 5 input error, 6
reproduction mismatch).

This tutorial runs the commands with :mod:`subprocess`, as a shell script would.
"""  # noqa: D205, D400  (sphinx-gallery: the docstring opens with an RST title)

# %%
# A simulated BIDS dataset
# ------------------------

import json
import subprocess
import sys
import tempfile

work = tempfile.mkdtemp(prefix="evoparam-cli-")


def evoparam(*args):
    """Run one ``evoparam`` command and print and return its JSON summary."""
    proc = subprocess.run(
        [sys.executable, "-m", "evoparam", *map(str, args), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    summary = json.loads(proc.stdout.splitlines()[-1])
    print(
        "$ evoparam",
        " ".join(map(str, args)),
        "->",
        summary["status"],
        f"(exit {proc.returncode})",
    )
    return summary


evoparam(
    "simulate",
    "--modality",
    "ccep",
    "--n-trials",
    "10",
    "--n-channels",
    "4",
    "--seed",
    "0",
    "--out",
    f"{work}/bids",
)

# %%
# Check the stimulation metadata, then fit
# ----------------------------------------

summary = evoparam("validate-bids", f"{work}/bids")
print(summary["result"]["counts"])
summary = evoparam(
    "fit",
    f"{work}/bids",
    "--sub",
    "01",
    "--task",
    "ccep",
    "--preset",
    "ccep",
    "--random-state",
    "0",
    "--out",
    f"{work}/bids/derivatives/evoparam",
)
print(summary["n_units"], len(summary["outputs"]), "files")

# %%
# Options from a TOML file
# ------------------------

with open(f"{work}/options.toml", "w", encoding="utf-8") as fobj:
    fobj.write("random-state = 0\n[fit]\nno-existence = true\n")
evoparam(
    "fit",
    f"{work}/bids",
    "--sub",
    "01",
    "--task",
    "ccep",
    "--config",
    f"{work}/options.toml",
    "--out",
    f"{work}/fit-without-test",
)

# %%
# Report, reproduce
# -----------------

container = next(p for p in summary["outputs"] if p.endswith("_responses.evoparam"))
evoparam("report", container, "--format", "md", "--out", work)
result = evoparam("reproduce", container)["result"]
print("identical:", result["identical"])

# %%
# Methods paragraph
# -----------------

from evoparam import load_object, reports

print(reports.methods_text(load_object(container)))
