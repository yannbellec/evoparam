# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Yann V. Bellec. See LICENSE in this folder.
"""Task files of the evoparam benchmark, read and written with NumPy only.

An adapter runs in the environment of its tool, where evoparam need not be installed: this
module reads a task input (``<cell>.json`` descriptor and its ``.npz`` arrays, schema
``benchmark-task/<id>/1``) and writes a task output (TSV with the columns of the task, in
the order of the schema, ``n/a`` for a missing value). evoparam validates the output when
the adapter returns.
"""

import json
import math
import os

import numpy as np

# Output columns (required, optional) of the tasks the adapters answer, in the order of
# the schemas.
COLUMNS = {
    "T1": (["site", "channel", "responsive", "p_value"], ["score"]),
    "T2": (["site", "channel", "latency", "amplitude"], []),
}


def read_task(path):
    """The descriptor (dict) and the arrays (dict of ndarray) of a task input."""

    with open(path, encoding="utf-8") as fobj:
        descriptor = json.load(fobj)
    npz = os.path.join(os.path.dirname(os.path.abspath(path)), descriptor["data_file"])
    with np.load(npz, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name], dtype=float) for name in archive.files}
    return descriptor, arrays


def times(descriptor, n_times):
    """Sample times (s) of the arrays of a task input."""

    return descriptor["tmin"] + np.arange(n_times) / float(descriptor["sfreq"])


def scored_channels(descriptor):
    """Channels to answer: every channel but the stimulated ones."""

    stim = set(descriptor.get("stim_channels", ()))
    return [(i, c) for i, c in enumerate(descriptor["ch_names"]) if c not in stim]


def _field(value):
    if value is None:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "n/a" if not math.isfinite(value) else repr(float(value))
    return str(value)


def write_tsv(path, task, rows):
    """Write the rows (dicts) of a task output; an optional column absent from every row
    is left out."""

    required, optional = COLUMNS[task]
    columns = required + [c for c in optional if any(c in row for row in rows)]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fobj:
        fobj.write("\t".join(columns) + "\n")
        for row in rows:
            fobj.write("\t".join(_field(row.get(c)) for c in columns) + "\n")
    return path
