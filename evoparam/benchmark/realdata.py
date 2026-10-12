# SPDX-License-Identifier: Apache-2.0
"""Real-data suites: index files and loaders, no copies of data (spec §8.2).

Each index (``realdata/<name>.json``, schema ``evoparam-benchmark-realdata/1``) names a
dataset of the registry of :mod:`evoparam.io` (id, version, DOI, licence), the tasks it
serves, how its trials are read, the entries (subject, session, task, run) chosen by a rule
fixed before any result, and how the truth is made. The data are read from their public
repository when a loader runs (OpenNeuro by HTTP byte ranges, ERP CORE from OSF); the
repository holds none of them. CC0 data may be redistributed as epoched NWB by the deposit;
CC BY and CC BY-SA data are referenced only.

Truth on real data:

- T1 / T2 (``ds004080``, ``ds003708``, ``ds004977``): the consensus of at least two
  published rules of the registry of §5.6 (``truth.rules``, with the unstated fields set as
  in the baselines); contacts where exactly one rule detects a response are not scored; the
  T2 truth is the N1 peak of ``vanblooijs2023`` on the consensus responders. No human
  scoring is available in these datasets' derivatives.
- T6 (ERP CORE): the reference definitions of ``erpcore.json`` computed on the data (the
  ``vs_definition`` metrics; there is no noise-free truth).
"""

import json
from importlib import resources

import numpy as np

from evoparam.errors import DataError

from .suite import ERPCORE
from .tasks import TaskInput, make_descriptor

__all__ = [
    "list_indices",
    "load_index",
    "load_bids_trials",
    "real_inputs",
    "consensus_truth",
    "REALDATA_SCHEMA",
]

REALDATA_SCHEMA = "evoparam-benchmark-realdata/1"


def list_indices():
    """Names of the real-data indices shipped with the package."""

    folder = resources.files("evoparam.benchmark").joinpath("realdata")
    return sorted(p.name[:-5] for p in folder.iterdir() if p.name.endswith(".json"))


def load_index(name):
    """A real-data index by name."""

    if name not in list_indices():
        raise DataError(f"Unknown real-data index {name!r}; indices: {list_indices()}.")
    text = (
        resources.files("evoparam.benchmark")
        .joinpath("realdata", f"{name}.json")
        .read_text("utf-8")
    )
    doc = json.loads(text)
    if doc.get("schema") != REALDATA_SCHEMA:
        raise DataError(f"realdata/{name}.json: schema must be {REALDATA_SCHEMA}.")
    return doc


def load_bids_trials(index, entry, *, cache_dir=None):
    """The trials of one entry of a BIDS index, split by stimulation site.

    Returns
    -------
    list of (str, TrialSet)
        ``(site, trials)`` for the sites of the index's ``site_rule`` (at least
        ``min_trials`` trials, the first ``max_sites`` in the order of the events), or one
        ``("", trials)`` pair when the events have no site.
    """

    from evoparam.io import read_bids

    if isinstance(index, str):
        index = load_index(index)
    if index["reader"] != "bids":
        raise DataError(f"index {index['dataset']} is not read from BIDS.")
    ts = read_bids(
        index["dataset"],
        entry["subject"],
        session=entry.get("session"),
        task=entry.get("task"),
        run=entry.get("run"),
        datatype=index["datatype"],
        epoch=tuple(index["epoch"]),
        picks=None,
        remote=index.get("remote", "openneuro"),
        cache_dir=cache_dir,
    )
    if index.get("picks"):
        # channel types of channels.tsv; a run need not have every type of the index
        types = {str(t).lower() for t in index["picks"]}
        info = ts.ch_info
        if info is None or "type" not in info:
            raise DataError(f"{index['dataset']} {entry}: no channel types to pick {types}.")
        keep = [i for i, t in enumerate(info["type"].astype(str).str.lower()) if t in types]
        if not keep:
            raise DataError(f"{index['dataset']} {entry}: no channel of type {sorted(types)}.")
        ts = ts.select(picks=keep)
    meta = ts.metadata
    if meta is None or "stim_site" not in meta.columns:
        return [("", ts)]
    rule = index["site_rule"]
    sites = meta["stim_site"].astype(str)
    out = []
    for site in dict.fromkeys(sites):
        rows = np.flatnonzero(sites.to_numpy() == site)
        if len(rows) < rule["min_trials"]:
            continue
        keep = [i for i, c in enumerate(ts.ch_names) if c not in site.split("-")]
        out.append((site, ts.select(trials=rows, picks=keep)))
        if len(out) >= rule["max_sites"]:
            break
    return out


def consensus_truth(ts, index):
    """T1 / T2 truth of real trials: the consensus of the index's published rules.

    Returns
    -------
    dict
        ``responders``, ``excluded`` (exactly one rule detects), ``n1`` (latency and
        amplitude of the ``vanblooijs2023`` peak on the responders) and the decisions of
        every rule (``decisions``).
    """

    from evoparam.existence import test_response

    from .baselines import RULE_OVERRIDES

    spec = index["truth"]
    decisions, peaks = {}, None
    for rule in spec["rules"]:
        result = test_response(
            ts,
            method="threshold",
            rule=rule,
            rule_overrides=dict(RULE_OVERRIDES[rule]),
            by=(),
            random_state=0,
        )
        decisions[rule] = np.asarray(result.significant, bool)
        if rule == "vanblooijs2023":
            peaks = (np.asarray(result.onset, float), result.settings["rule_spec"])
    votes = np.sum([decisions[r] for r in spec["rules"]], axis=0)
    responders = [c for c, v in zip(ts.ch_names, votes) if v >= spec["min_agree"]]
    excluded = [c for c, v in zip(ts.ch_names, votes) if 0 < v < spec["min_agree"]]
    n1 = {}
    if peaks is not None:
        latency, rule_spec = peaks
        avg = np.nanmean(ts.data, axis=0)
        base = (ts.times >= rule_spec["baseline"][0]) & (ts.times <= rule_spec["baseline"][1])
        for i, c in enumerate(ts.ch_names):
            if c in responders and np.isfinite(latency[i]):
                k = int(np.argmin(np.abs(ts.times - latency[i])))
                n1[c] = {
                    "latency": float(ts.times[k]),
                    "amplitude": float(avg[i, k] - avg[i, base].mean()),
                }
    return {
        "responders": responders,
        "excluded": excluded,
        "n1": n1,
        "decisions": {r: v.tolist() for r, v in decisions.items()},
    }


def real_inputs(name, entry_index=0, *, task="T1", cache_dir=None):
    """Task inputs (with their truth) of one entry of a real-data index.

    T1 / T2: one input per stimulation site of the entry (stimulated contacts removed),
    truth from :func:`consensus_truth`. T6 (``erpcore``): one input per entry (component and
    participant), definitions of ``erpcore.json``.

    Returns
    -------
    list of TaskInput
        ``split`` is ``"real"``; the suite field names the index.
    """

    index = load_index(name)
    if task not in index["tasks"]:
        raise DataError(f"index {name} does not serve task {task} ({index['tasks']}).")
    entry = index["entries"][entry_index]
    suite = {"name": f"real-{name}", "version": "1.0.0"}
    out = []
    if task in ("T1", "T2"):
        for site, ts in load_bids_trials(index, entry, cache_dir=cache_dir):
            truth = consensus_truth(ts, index)
            truth["stim_channels"] = []
            cell = f"{name}-{entry_index:03d}-{''.join(ch for ch in site if ch.isalnum())}"
            data = np.nan_to_num(ts.data)
            fields = {
                "suite": suite,
                "cell": cell,
                "split": "real",
                "sfreq": float(ts.sfreq),
                "tmin": float(ts.times[0]),
                "units": ts.units,
                "ch_names": list(ts.ch_names),
                "n_trials": int(ts.n_trials),
                "modality": ts.modality or "",
                "site": site,
                "stim_channels": [],
                "hints": {"window": [0.009, 0.1], "baseline": [-1.0, -0.1], "preset": "ccep"},
                "source": {
                    "dataset": index["dataset"],
                    "version": index["version"],
                    "doi": index["doi"],
                    **entry,
                },
            }
            out.append(
                TaskInput(
                    descriptor=make_descriptor(task, {"data": data}, **fields),
                    arrays={"data": data},
                    truth=truth,
                )
            )
        return out
    if task == "T6":
        return [_erpcore_input(index, entry, entry_index, suite)]
    raise DataError(f"no loader for task {task} of index {name}.")  # pragma: no cover


def _erpcore_input(index, entry, entry_index, suite):
    from evoparam.io import fetch_erpcore
    from evoparam.utils._optional import require

    mne = require("mne", extra="mne", caller="evoparam.benchmark.realdata (ERP CORE)")
    component, participant = entry["component"], int(entry["participant"])
    definition = ERPCORE["components"][component]
    epochs = mne.read_epochs_eeglab(fetch_erpcore(component, participant), verbose="error")
    labels = list(epochs.event_id)
    first = [k for k in labels if k.split("(")[0].startswith("B1")]
    if not first:
        raise DataError(f"ERP CORE {component} {participant}: no epoch of bin 1.")
    picked = epochs[first].pick([definition["channel"]])
    data = picked.get_data() * 1e6
    times = picked.times
    cell = f"erpcore-{component}-{participant:02d}"
    arrays = {"data": data, "average": data.mean(axis=0)}
    fields = {
        "suite": suite,
        "cell": cell,
        "split": "real",
        "sfreq": float(picked.info["sfreq"]),
        "tmin": float(times[0]),
        "units": "uV",
        "ch_names": [definition["channel"]],
        "n_trials": int(len(data)),
        "modality": "erp",
        "definitions": [
            {
                "condition": component,
                "channel": definition["channel"],
                "window": list(definition["window"]),
                "polarity": definition["polarity"],
                "measures": list(ERPCORE["measures"]),
                "source": ERPCORE["source"],
            }
        ],
        "hints": {"preset": "erp"},
        "source": {
            "dataset": index["dataset"],
            "doi": index["doi"],
            "component": component,
            "participant": participant,
        },
    }
    return TaskInput(
        descriptor=make_descriptor("T6", arrays, **fields),
        arrays=arrays,
        truth={"signal_average": arrays["average"]},
    )
