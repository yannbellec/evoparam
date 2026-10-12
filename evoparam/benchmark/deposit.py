# SPDX-License-Identifier: Apache-2.0
"""Local preparation of the deposit of the simulated suites (spec §8.2, REQ-355, REQ-356).

The deposit itself (files with a DOI) is a public step and is not made by this module:
nothing is uploaded anywhere. :func:`prepare_deposit` writes, into a local folder, what
the deposit will hold so that methods in other languages can be evaluated without the
package:

- ``bids/``: one BIDS dataset, one subject per public cell (``sub-<cell>``, the trials laid
  end to end as a BrainVision recording with its events), the truth table in
  ``derivatives/truth`` (:func:`evoparam.sim.bids.write_bids`);
- ``nwb/<cell>.nwb``: the same trials as an NWB ``ElectricalSeries`` (stored in µV with a
  conversion of 1e-6 to volts) with a ``trials`` table (``start_time``, ``stop_time``,
  ``stim_onset`` and the stimulation site), readable by :func:`evoparam.io.read_nwb`;
- ``inputs/<task>/``: the task inputs (JSON descriptor and ``.npz``) and ``truth/``: the
  ground truth of every public cell (:meth:`GroundTruth.to_json`, without arrays);
- ``manifest.json``: the generator (package and version), the suite (name, version,
  SHA-256), every public cell with its generator, settings and **seed**, every file with
  its SHA-256, and the hidden split: its cells (settings without seeds) and the
  **SHA-256 commitment** of the hidden seeds, checked against the hidden seeds document
  when it is given (the seeds themselves are never written). ``doi`` is null until the
  deposit is made.
"""

import json
import os
import shutil
import time
from datetime import datetime, timezone

import numpy as np

from evoparam.errors import DataError
from evoparam.version import __version__

from ._canon import clean, sha256_file
from .suite import hidden_commitment, load_hidden_seeds, load_suite, make_inputs, simulate_cell
from .tasks import write_input

__all__ = ["prepare_deposit", "write_trials_nwb", "DEPOSIT_SCHEMA", "DEPOSIT_STATUS"]

DEPOSIT_SCHEMA = "evoparam-benchmark-deposit/1"
DEPOSIT_STATUS = (
    "prepared locally; not deposited: the deposit (a public step with a DOI) "
    "waits for the owner's HIIE decision (nothing public before it)"
)


def _label(cell_id):
    return "".join(ch for ch in cell_id if ch.isalnum())


def write_trials_nwb(ts, path, *, identifier, description):
    """Write a TrialSet as an NWB file: the trials end to end in one ``ElectricalSeries``
    (µV, conversion 1e-6) and a ``trials`` table with ``stim_onset`` at each trial's time
    zero. Returns the path.
    """
    from evoparam.utils._optional import require

    pynwb = require("pynwb", extra="nwb", caller="evoparam.benchmark.write_trials_nwb")
    from pynwb.ecephys import ElectricalSeries

    n_trials, n_channels, n_times = ts.data.shape
    nwb = pynwb.NWBFile(
        session_description=description,
        identifier=identifier,
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    device = nwb.create_device(name="simulation")
    group = nwb.create_electrode_group(
        name="simulated", description="simulated contacts", location="simulated", device=device
    )
    nwb.add_electrode_column(name="label", description="channel name")
    for name in ts.ch_names:
        nwb.add_electrode(group=group, location="simulated", label=str(name))
    region = nwb.create_electrode_table_region(list(range(n_channels)), "all channels")
    data = np.nan_to_num(ts.data).transpose(0, 2, 1).reshape(n_trials * n_times, n_channels)
    nwb.add_acquisition(
        ElectricalSeries(
            name="trials",
            data=data,
            electrodes=region,
            rate=float(ts.sfreq),
            starting_time=0.0,
            conversion=1e-6,
            description="simulated trials laid end to end",
        )
    )
    nwb.add_trial_column("stim_onset", "time zero of the trial (s)")
    site = None
    if ts.metadata is not None and "stim_site" in ts.metadata.columns:
        nwb.add_trial_column("stim_site", "stimulation site")
        site = ts.metadata["stim_site"].astype(str).tolist()
    duration = n_times / ts.sfreq
    for k in range(n_trials):
        start = k * duration
        extra = {"stim_site": site[k]} if site is not None else {}
        nwb.add_trial(
            start_time=start,
            stop_time=start + duration,
            stim_onset=start - float(ts.times[0]),
            **extra,
        )
    with pynwb.NWBHDF5IO(path, "w") as io:
        io.write(nwb)
    return path


def prepare_deposit(
    suite="core",
    out="deposit",
    *,
    formats=("bids", "nwb"),
    tasks=None,
    hidden_seeds=None,
    overwrite=False,
    log=None,
):
    """Write the public split of a suite and its manifest into a local folder.

    Parameters
    ----------
    suite : str or Suite
    out : str
        Folder to write (refused when it exists and is not empty, unless ``overwrite``).
    formats : sequence of {"bids", "nwb"}
    tasks : sequence of str, optional
        Default: every task of the suite.
    hidden_seeds : dict or str, optional
        The hidden seeds document: checked against the suite's commitment (its content is
        not written).
    overwrite : bool
    log : callable, optional

    Returns
    -------
    str
        Path of ``manifest.json``.
    """
    log = log or (lambda message: None)
    if not hasattr(suite, "cells"):
        suite = load_suite(suite)
    unknown = [f for f in formats if f not in ("bids", "nwb")]
    if unknown:
        raise DataError(f"unknown formats {unknown}; use 'bids' and/or 'nwb'.")
    if os.path.isdir(out) and os.listdir(out):
        if not overwrite:
            raise DataError(f"{out} exists and is not empty (overwrite=True to replace it).")
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)
    commitment = suite.hidden_split["commitment_sha256"]
    verified = None
    if hidden_seeds is not None:
        verified = hidden_commitment(load_hidden_seeds(hidden_seeds)) == commitment
        if not verified:
            raise DataError(
                "the hidden seeds document does not match the commitment of the "
                f"suite ({commitment})."
            )
    tasks = list(tasks) if tasks else suite.task_ids()
    start = time.perf_counter()
    cells_out = []
    for cell in suite.cells(split="public"):
        cell_tasks = [t for t in tasks if t in cell.tasks]
        if not cell_tasks:
            continue
        ts, truth = simulate_cell(cell)
        files = []
        if "bids" in formats:
            from evoparam.sim.bids import write_bids

            vhdr = write_bids(
                ts,
                truth,
                os.path.join(out, "bids"),
                subject=_label(cell.id),
                task=_label(cell.group),
            )
            files.append(os.path.relpath(vhdr, out))
        if "nwb" in formats:
            path = os.path.join(out, "nwb", f"{cell.id}.nwb")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            write_trials_nwb(
                ts,
                path,
                identifier=f"evoparam-benchmark-{suite.name}-{cell.id}",
                description=f"evoparam benchmark suite {suite.name} "
                f"{suite.version}, cell {cell.id}",
            )
            files.append(os.path.relpath(path, out))
        truth_path = os.path.join(out, "truth", f"{cell.id}.json")
        os.makedirs(os.path.dirname(truth_path), exist_ok=True)
        with open(truth_path, "w", encoding="utf-8") as fobj:
            fobj.write(truth.to_json(arrays=False))
        files.append(os.path.relpath(truth_path, out))
        for task, task_input in make_inputs(cell, ts, truth, tasks=cell_tasks).items():
            path = write_input(task_input, os.path.join(out, "inputs", task))
            files += [
                os.path.relpath(path, out),
                os.path.relpath(
                    os.path.join(os.path.dirname(path), task_input.descriptor["data_file"]), out
                ),
            ]
        cells_out.append(
            {
                "id": cell.id,
                "group": cell.group,
                "generator": cell.generator,
                "seed": cell.seed,
                "settings": cell.settings,
                "tasks": cell_tasks,
                "files": files,
            }
        )
        log(f"{cell.id}: written")
    shutil.copyfile(suite.path, os.path.join(out, f"suite-{suite.name}.json"))
    hashes = {}
    for folder, _, names in os.walk(out):
        for name in names:
            path = os.path.join(folder, name)
            rel = os.path.relpath(path, out).replace(os.sep, "/")
            hashes[rel] = sha256_file(path)
    for cell in cells_out:
        cell["files"] = {
            rel.replace(os.sep, "/"): hashes[rel.replace(os.sep, "/")] for rel in cell["files"]
        }
    manifest = {
        "schema": DEPOSIT_SCHEMA,
        "status": DEPOSIT_STATUS,
        "doi": None,
        "licence": None,
        "suite": {
            "name": suite.name,
            "version": suite.version,
            "sha256": suite.sha256,
            "file": f"suite-{suite.name}.json",
        },
        "generator": {
            "package": "evoparam",
            "version": __version__,
            "functions": sorted({f"evoparam.sim.{c['generator']}" for c in cells_out}),
        },
        "formats": list(formats),
        "tasks": tasks,
        "public_split": {"n_cells": len(cells_out), "cells": cells_out},
        "hidden_split": {
            "n_cells": suite.hidden_split["n_hidden_cells"],
            "commitment_sha256": commitment,
            "commitment": suite.hidden_split["commitment"],
            "verified_against_seeds": verified,
            "cells": [
                {
                    "id": c.id,
                    "group": c.group,
                    "generator": c.generator,
                    "settings": c.settings,
                    "tasks": list(c.tasks),
                }
                for c in suite.cells(split="hidden")
            ],
        },
        "files": hashes,
        "duration_s": time.perf_counter() - start,
    }
    path = os.path.join(out, "manifest.json")
    with open(path, "w", encoding="utf-8") as fobj:
        json.dump(clean(manifest), fobj, indent=1, sort_keys=True, ensure_ascii=False)
        fobj.write("\n")
    return path
