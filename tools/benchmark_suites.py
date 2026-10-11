"""Write the frozen suite files of the benchmark (spec §8.2): evoparam/benchmark/suites/
core.json and full.json.

The suites are frozen: this script documents how they were made and is run again only for
a new suite version (a new version number, a changelog entry). Each group of cells has a
generator of evoparam.sim, shared settings and a grid of cell settings; 20 % of the cells of
every group are hidden (spec §8.2). Public seeds are written in the suite files; hidden
seeds are not: they live in a separate document (``--hidden-seeds``) whose SHA-256 is
written in the suite files as the commitment of REQ-356.

Usage::

    python tools/benchmark_suites.py --hidden-seeds PATH            # write the suites
    python tools/benchmark_suites.py --hidden-seeds PATH --new      # draw the hidden seeds
                                                                     # first (once; refuses
                                                                     # to overwrite PATH)

The hidden seeds document must stay outside the repository until the benchmark paper is
accepted.
"""

import argparse
import copy
import json
import os
import pathlib
import secrets
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evoparam.benchmark._canon import canonical_bytes, sha256_bytes  # noqa: E402

VERSION = "1.0.0"
OUT = ROOT / "evoparam" / "benchmark" / "suites"
HIDDEN_SCHEMA = "evoparam-benchmark-hidden-seeds/1"

# Groups: generator, tasks, shared settings and the grid of cell settings (in order). The
# last two settings of every group form the hidden cells of the core suite; the first eight
# are public (20 % hidden, spec §8.2).
CCEP_DETECTION = {
    "generator": "simulate_ccep", "tasks": ["T1", "T2"],
    "description": "Single-pulse stimulation, one site, 24 sEEG contacts, about 30 % "
                   "responding with an N1 (80 % also with an N2); no stimulation artifact "
                   "(the artifact is task T4); pink noise of SD 20 uV; per-trial SNR of the "
                   "components from 0.5 to 4.",
    "settings": {"n_channels": 24, "artifact": False, "tmin": -1.0, "tmax": 1.0,
                 "noise": "pink", "noise_sd": 20.0, "p_response": 0.3,
                 "polarity": "negative", "sfreq": 2048.0,
                 "_": {"hints": {"window": [0.009, 0.1], "baseline": [-1.0, -0.1],
                                 "preset": "ccep"}}},
    "grid": [{"snr": s, "n_trials": n} for n in (10, 30) for s in (0.5, 1.0, 2.0, 4.0)]
    + [{"snr": 1.0, "n_trials": 10}, {"snr": 2.0, "n_trials": 30}],
}
EVOKED = {
    "generator": None, "tasks": ["T3"],
    "description": "Averaged waveforms: scalp ERPs (4 channels, P1, N1 and P3, 60 trials, "
                   "500 Hz) and CCEPs (12 contacts, N1 and N2 on half of them, 10 trials, "
                   "2048 Hz, no artifact); the input is the average.",
    "cells": [("simulate_erp", {"snr": s, "noise": n}) for s in (1.0, 2.0)
              for n in ("pink", "white")]
    + [("simulate_ccep", {"snr": s, "n_trials": n}) for s in (1.0, 2.0) for n in (10, 30)]
    + [("simulate_erp", {"snr": 1.5, "noise": "pink"}),
       ("simulate_ccep", {"snr": 1.5, "n_trials": 10})],
}
ERP_EVOKED = {"n_trials": 60, "n_channels": 4, "components": ["P1", "N1", "P3"],
              "sfreq": 500.0, "tmin": -0.2, "tmax": 0.8, "noise_sd": 10.0,
              "_": {"hints": {"window": [0.05, 0.7], "baseline": [-0.2, 0.0], "preset": "erp"}}}
CCEP_EVOKED = {"n_channels": 12, "artifact": False, "tmin": -1.0, "tmax": 1.0, "noise": "pink",
               "noise_sd": 20.0, "p_response": 0.5, "polarity": "negative", "sfreq": 2048.0,
               "_": {"hints": {"window": [0.009, 0.5], "baseline": [-0.5, -0.02],
                               "preset": "ccep"}}}
ARTIFACT = {
    "generator": "simulate_ccep", "tasks": ["T4"],
    "description": "Single-pulse stimulation with its artifact (dipole potential of the "
                   "pulse, anti-aliasing filter, double-exponential tail), 30 pulses, 12 "
                   "contacts, half of them responding (drawn amplitudes), pink noise of SD "
                   "20 uV; currents 2 to 8 mA and tails of 2 or 5 % of the pulse.",
    "settings": {"n_trials": 30, "n_channels": 12, "artifact": True, "tmin": -0.5,
                 "tmax": 0.5, "noise": "pink", "noise_sd": 20.0, "p_response": 0.5,
                 "polarity": "negative", "sfreq": 2048.0,
                 "_": {"artifact_window": [-0.002, 0.008],
                       "hints": {"window": [0.009, 0.1], "baseline": [-0.5, -0.05],
                                 "preset": "ccep"}}},
    "grid": [{"stim_current_mA": c, "tail_fraction": f} for f in (0.02, 0.05)
             for c in (2.0, 4.0, 6.0, 8.0)]
    + [{"stim_current_mA": 3.0, "tail_fraction": 0.02},
       {"stim_current_mA": 5.0, "tail_fraction": 0.05}],
}
SINGLE_TRIAL = {
    "generator": None, "tasks": ["T5"],
    "description": "Trials with latency jitter and amplitude variability (CV 30 %): a scalp "
                   "P3 (2 channels, 40 trials, jitter SD 10 or 30 ms) and a CCEP N1 (6 "
                   "contacts all responding, 30 trials, jitter SD 1 or 3 ms); per-trial SNR "
                   "1 or 3. The targets are the component of every channel that holds it.",
    "cells": [("simulate_erp", {"jitter_sd": j, "snr": s}) for j in (0.01, 0.03)
              for s in (1.0, 3.0)]
    + [("simulate_ccep", {"jitter_sd": j, "snr": s}) for j in (0.001, 0.003)
       for s in (1.0, 3.0)]
    + [("simulate_erp", {"jitter_sd": 0.02, "snr": 2.0}),
       ("simulate_ccep", {"jitter_sd": 0.002, "snr": 2.0})],
}
ERP_TRIAL = {"n_trials": 40, "n_channels": 2, "components": ["P3"], "amplitude_cv": 0.3,
             "sfreq": 500.0, "tmin": -0.2, "tmax": 0.8, "noise": "pink", "noise_sd": 10.0,
             "_": {"target_label": "P3", "target_window": [0.25, 0.7],
                   "hints": {"window": [0.05, 0.7], "baseline": [-0.2, 0.0],
                             "preset": "erp"}}}
CCEP_TRIAL = {"n_trials": 30, "n_channels": 6, "artifact": False, "tmin": -0.5, "tmax": 0.5,
              "amplitude_cv": 0.3, "noise": "pink", "noise_sd": 20.0, "p_response": 1.0,
              "p_n2": 0.0, "polarity": "negative", "sfreq": 2048.0,
              "_": {"target_label": "N1", "target_window": [0.009, 0.06],
                    "hints": {"window": [0.009, 0.5], "baseline": [-0.5, -0.02],
                              "preset": "ccep"}}}
ERP_SCORING = {
    "generator": "simulate_erp", "tasks": ["T6"],
    "description": "ERP CORE-like scalp components on their canonical channel (P3 Pz, N170 "
                   "PO8, N400 CPz, MMN FCz, ERN FCz), 60 trials at 256 Hz, latency jitter "
                   "SD 10 ms, amplitude CV 20 %, pink noise of SD 10 uV, per-trial SNR 1 or "
                   "2; scored with the ERP CORE measurement windows.",
    "settings": {"n_trials": 60, "n_channels": 1, "sfreq": 256.0, "tmin": -0.2, "tmax": 0.8,
                 "jitter_sd": 0.01, "amplitude_cv": 0.2, "noise": "pink", "noise_sd": 10.0},
    "grid": [{"component": c, "snr": 1.0} for c in ("P3", "N170", "N400", "MMN", "ERN")]
    + [{"component": c, "snr": 2.0} for c in ("P3", "N170", "N400")]
    + [{"component": "MMN", "snr": 2.0}, {"component": "ERN", "snr": 2.0}],
}
CHANNELS = {"P3": "Pz", "N170": "PO8", "N400": "CPz", "MMN": "FCz", "ERN": "FCz"}
BURST_AMP = 10.33   # adds 20 % to the signal power over a 100 ms burst in AR(1) noise of SD 10
TF = {
    "generator": "simulate_bursts", "tasks": ["T7"],
    "description": "Induced bursts (random phase per trial, Hann envelope) in AR(1) noise "
                   "(rho 0.9, SD 10 uV), 30 trials, one channel at 1000 Hz; cells without "
                   "burst estimate the type-I error.",
    "settings": {"n_trials": 30, "n_channels": 1, "sfreq": 1000.0, "tmin": -0.6, "tmax": 0.8,
                 "noise": "ar1", "noise_sd": 10.0,
                 "_": {"hints": {"window": [0.0, 0.8], "baseline": [-0.5, -0.1]}}},
    "grid": [{"bursts": []}, {"bursts": []},
             {"bursts": [{"freq": 80.0, "centre": 0.25, "duration": 0.1,
                          "amplitude": BURST_AMP}]},
             {"bursts": [{"freq": 80.0, "centre": 0.25, "duration": 0.1, "amplitude": 15.0}]},
             {"bursts": [{"freq": 30.0, "centre": 0.4, "duration": 0.2,
                          "amplitude": BURST_AMP}]},
             {"bursts": [{"freq": 30.0, "centre": 0.4, "duration": 0.2, "amplitude": 15.0}]},
             {"bursts": [{"freq": 150.0, "centre": 0.2, "duration": 0.08, "amplitude": 15.0}]},
             {"bursts": [{"freq": 80.0, "centre": 0.15, "duration": 0.1, "amplitude": 15.0},
                         {"freq": 20.0, "centre": 0.5, "duration": 0.2, "amplitude": 15.0}]},
             {"bursts": []},
             {"bursts": [{"freq": 60.0, "centre": 0.3, "duration": 0.12, "amplitude": 12.0}]}],
}


def _cell_settings(group, entry):
    """(generator, settings) of one entry of a group's grid."""

    if group["generator"] is not None:
        settings = copy.deepcopy(entry)
        if group is ERP_SCORING:
            comp = settings.pop("component")
            settings.update(components=[comp], ch_names=[CHANNELS[comp]],
                            _={"component": comp,
                               "hints": {"window": [-0.1, 0.7], "baseline": [-0.2, 0.0],
                                         "preset": "erp"}})
        return group["generator"], settings
    generator, overrides = entry
    base = {(EVOKED["tasks"][0], "simulate_erp"): ERP_EVOKED,
            (EVOKED["tasks"][0], "simulate_ccep"): CCEP_EVOKED,
            (SINGLE_TRIAL["tasks"][0], "simulate_erp"): ERP_TRIAL,
            (SINGLE_TRIAL["tasks"][0], "simulate_ccep"): CCEP_TRIAL}[(group["tasks"][0],
                                                                      generator)]
    settings = copy.deepcopy(base)
    settings.update(copy.deepcopy(overrides))
    return generator, settings


GROUPS = {"ccep_detection": CCEP_DETECTION, "evoked": EVOKED, "artifact": ARTIFACT,
          "single_trial": SINGLE_TRIAL, "erp_scoring": ERP_SCORING, "tf": TF}


def build(name, replicates, hidden_per_group, seed_base):
    """The suite document and the ids of its hidden cells."""

    groups, hidden_ids = {}, []
    for g_index, (gname, group) in enumerate(GROUPS.items(), start=1):
        entries = group["grid"] if group["generator"] is not None else group["cells"]
        public, hidden = entries[:8], entries[8:]
        cells = []
        k = 0
        for _ in range(replicates):
            for entry in public:
                k += 1
                generator, settings = _cell_settings(group, entry)
                cells.append({"id": f"{gname}-p{k:03d}", "split": "public",
                              "seed": seed_base + 1000 * g_index + k,
                              "generator": generator, "settings": settings})
        for h in range(hidden_per_group):
            entry = hidden[h] if h < len(hidden) else public[h % len(public)]
            generator, settings = _cell_settings(group, entry)
            cid = f"{gname}-h{h + 1:03d}"
            hidden_ids.append(cid)
            cells.append({"id": cid, "split": "hidden", "generator": generator,
                          "settings": settings})
        if group["generator"] is not None:
            for cell in cells:
                del cell["generator"]          # the group's generator
        groups[gname] = {"generator": group["generator"] or "mixed", "tasks": group["tasks"],
                         "description": group["description"],
                         "settings": copy.deepcopy(group.get("settings", {})),
                         "cells": cells}
    tasks = {}
    for gname, group in groups.items():
        for task in group["tasks"]:
            tasks.setdefault(task, []).append(gname)
    doc = {"schema": "evoparam-benchmark-suite/1", "name": name, "version": VERSION,
           "description": f"Simulated suite '{name}' of the evoparam benchmark (spec §8.2): "
                          "the data are generated on the fly by evoparam.sim from the "
                          "settings and seeds below.",
           "groups": groups, "tasks": tasks}
    return doc, hidden_ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hidden-seeds", required=True,
                        help="path of the hidden seeds document (outside the repository)")
    parser.add_argument("--new", action="store_true",
                        help="draw the hidden seeds and write the document (refuses to "
                             "overwrite)")
    args = parser.parse_args(argv)
    core, core_hidden = build("core", replicates=1, hidden_per_group=2, seed_base=8_000_000)
    full, full_hidden = build("full", replicates=5, hidden_per_group=10, seed_base=9_000_000)
    if args.new:
        if os.path.exists(args.hidden_seeds):
            parser.error(f"{args.hidden_seeds} exists; the hidden seeds are drawn once.")
        document = {"schema": HIDDEN_SCHEMA, "nonce": secrets.token_hex(32),
                    "seeds": {"core": {c: secrets.randbelow(2 ** 63) for c in core_hidden},
                              "full": {c: secrets.randbelow(2 ** 63) for c in full_hidden}}}
        os.makedirs(os.path.dirname(os.path.abspath(args.hidden_seeds)), exist_ok=True)
        with open(args.hidden_seeds, "x", encoding="utf-8") as fobj:
            json.dump(document, fobj, indent=1, sort_keys=True)
            fobj.write("\n")
    with open(args.hidden_seeds, encoding="utf-8") as fobj:
        document = json.load(fobj)
    assert document["schema"] == HIDDEN_SCHEMA
    assert sorted(document["seeds"]["core"]) == sorted(core_hidden)
    assert sorted(document["seeds"]["full"]) == sorted(full_hidden)
    digest = sha256_bytes(canonical_bytes(document))
    for doc, hidden in ((core, core_hidden), (full, full_hidden)):
        n_cells = sum(len(g["cells"]) for g in doc["groups"].values())
        doc["hidden_split"] = {
            "fraction": round(len(hidden) / n_cells, 6), "n_hidden_cells": len(hidden),
            "n_cells": n_cells, "commitment_sha256": digest,
            "commitment": "SHA-256 of the canonical JSON (keys sorted, no whitespace, UTF-8) "
                          "of the hidden seeds document (schema "
                          f"{HIDDEN_SCHEMA}: a random 256-bit nonce and the seeds of the "
                          "hidden cells of the core and full suites)",
            "reveal": "the hidden seeds document is published at the acceptance of the "
                      "benchmark paper; evoparam.benchmark checks it against this hash"}
        path = OUT / f"{doc['name']}.json"
        OUT.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fobj:
            json.dump(doc, fobj, indent=1, ensure_ascii=False)
            fobj.write("\n")
        print(f"{path}: {n_cells} cells, {len(hidden)} hidden, commitment {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
