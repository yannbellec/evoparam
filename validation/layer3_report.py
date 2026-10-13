# SPDX-License-Identifier: Apache-2.0
"""Layer-3 comparison records (spec §7.3, REQ-374).

Every layer-3 comparison reports the comparator's number, evoparam's number, the difference
with its interval, and the exact versions of the data and of the comparator definition. This
script builds those records from the layer-3 result files and writes them to
``validation/layer3/comparisons.json``:

- ``n1_ds004080.json`` (per-channel decisions and latencies of the run in
  ``validation/layer4/<version>/report.json``): for each reading of the published N1 rule
  (registry rules ``vanblooijs2023`` and ``vanblooijs2023_code``), the fraction of channels
  with an N1 and the N1 latency on the channels where both find one;
- ``erpcore_difference.json``, ``erpcore_parent.json``: per component, the 50 % area latency
  of the published measurement and of the fitted component;
- ``carla_ds004977.json``: per stimulation site, the fraction of channels left out of the
  adjusted common average by the authors' CARLA code and by ``car_adaptive``.

Differences are evoparam minus comparator. Intervals are 95 % percentile bootstrap intervals
(2 000 resamples, seed 0) over the units named in each record (participants or channels),
resampled in pairs. The comparator definition is identified by its registry rule id (or the
definition written out in full) and the SHA-256 of its canonical JSON; the data by dataset
id, version and DOI. A version that was not recorded when the result was produced is
``null`` with a note, never guessed.

Usage::

    python validation/layer3_report.py [--check]

``--check`` exits 1 when the committed file differs from the one the result files give.
"""

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
LAYER3 = HERE / "layer3"
OUT = LAYER3 / "comparisons.json"
SCHEMA = "evoparam-layer3-comparisons/1"
REQUIRED = ("id", "quantity", "unit", "comparator", "comparator_value", "evoparam_value",
            "difference", "difference_ci", "interval", "n", "data", "comparator_definition")
N_BOOT, SEED, LEVEL = 2000, 0, 0.95
DIGITS = 6


def digest(obj):
    """SHA-256 of the canonical JSON of ``obj`` (sorted keys, no spaces)."""

    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=float)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _round(x):
    return None if x is None or not np.isfinite(x) else round(float(x), DIGITS)


def bootstrap_ci(statistic, *arrays, n_boot=N_BOOT, seed=SEED, level=LEVEL):
    """Percentile interval of ``statistic(*arrays)`` over resamples of the rows (the arrays
    are resampled together, so paired values stay paired)."""

    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    for b in range(n_boot):
        rows = rng.integers(0, n, n)
        values[b] = statistic(*(a[rows] for a in arrays))
    lo, hi = np.percentile(values, [50 * (1 - level), 50 * (1 + level)])
    return [float(lo), float(hi)]


def record(id, quantity, unit, comparator, comparator_value, evoparam_value, difference, ci,
           interval, n, data, definition):
    return {"id": id, "quantity": quantity, "unit": unit, "comparator": comparator,
            "comparator_value": _round(comparator_value),
            "evoparam_value": _round(evoparam_value), "difference": _round(difference),
            "difference_ci": [_round(ci[0]), _round(ci[1])], "interval": interval, "n": int(n),
            "data": data, "comparator_definition": definition}


def _paired_rate(id, quantity, comparator, ours, theirs, unit_name, data, definition):
    ours, theirs = np.asarray(ours, float), np.asarray(theirs, float)
    diff = lambda a, b: a.mean() - b.mean()                       # noqa: E731
    return record(id, quantity, "fraction", comparator, theirs.mean(), ours.mean(),
                  diff(ours, theirs), bootstrap_ci(diff, ours, theirs),
                  f"95 % percentile bootstrap over {unit_name}, paired, {N_BOOT} resamples, "
                  f"seed {SEED}", len(ours), data, definition)


def _paired_median(id, quantity, comparator, ours, theirs, unit_name, data, definition,
                   scale=1e3):
    ours, theirs = np.asarray(ours, float) * scale, np.asarray(theirs, float) * scale
    diff = lambda a, b: np.median(a - b)                          # noqa: E731
    return record(id, quantity, "ms", comparator, np.median(theirs), np.median(ours),
                  diff(ours, theirs), bootstrap_ci(diff, ours, theirs),
                  f"95 % percentile bootstrap over {unit_name}, paired, {N_BOOT} resamples, "
                  f"seed {SEED}", len(ours), data, definition)


def dataset(id, **extra):
    """Dataset id, version and DOI from ``evoparam/io/registry.json``."""

    registry = json.loads((ROOT / "evoparam" / "io" / "registry.json").read_text("utf-8"))
    entry = next(d for d in registry["datasets"] if d["id"] == id)
    return {"dataset": id, "version": entry["version"], "doi": entry["doi"], **extra}


def rule_definition(rule_id, **overrides):
    """A registry rule: its id, the overrides of its fields, and the SHA-256 of the entry of
    ``evoparam/existence/rules.json`` with the overrides applied."""

    rules = json.loads((ROOT / "evoparam" / "existence" / "rules.json").read_text("utf-8"))
    entry = dict(rules["rules"][rule_id], **overrides)
    return {"registry_rule": rule_id, "registry": "evoparam/existence/rules.json",
            "overrides": overrides, "version": digest(entry)}


###################################################################################################
## The comparisons of each file

def n1_comparisons(n1, layer4):
    """N1 on ds004080 against both readings of the published rule."""

    import importlib.util

    spec = importlib.util.spec_from_file_location("_robustness", HERE / "robustness.py")
    robustness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(robustness)
    runs = [f"sub-{r['subject']} run-{r['run']} {r['pair']}" for r in layer4["ds004080"]]
    if runs != n1["runs"]:
        raise ValueError("the layer-4 report and n1_ds004080.json name different runs")
    subjects = (HERE / "ds004080_subjects.txt").read_text("utf-8").splitlines()[0]
    data = dataset("ds004080", runs=runs, subject_list=subjects.lstrip("# "))
    model = np.concatenate([np.asarray(r["native"]["decision"], bool)
                            for r in layer4["ds004080"]])
    model_lat = np.concatenate([np.asarray(r["native"]["latency"], float)
                                for r in layer4["ds004080"]])
    out = []
    for reading, rule_id in robustness.READINGS.items():
        rule = np.concatenate([np.asarray(r["rule"][reading]["decision"], bool)
                               for r in layer4["ds004080"]])
        rule_lat = np.concatenate([np.asarray(r["rule"][reading]["latency"], float)
                                   for r in layer4["ds004080"]])
        check = robustness.n1_metrics(rule, rule_lat, model, model_lat)
        stored = n1["metrics"][reading]
        for key in ("n_channels", "n_rule", "n_model", "n_both"):
            if check[key] != stored[key]:
                raise ValueError(f"n1_ds004080.json {reading} {key}: {stored[key]} in the file, "
                                 f"{check[key]} from the layer-4 report")
        definition = rule_definition(rule_id, baseline=[-2.0, -0.1])
        name = f"published N1 rule, registry '{rule_id}' ({reading} reading)"
        out.append(_paired_rate(f"n1_ds004080/{reading}/n1_fraction",
                                "fraction of channels with an N1", name, model, rule,
                                f"the {len(rule)} channels", data, definition))
        both = rule & model
        out.append(_paired_median(f"n1_ds004080/{reading}/n1_latency",
                                  "N1 latency, median over the channels where both find one",
                                  name, model_lat[both], rule_lat[both],
                                  f"the {int(both.sum())} channels", data, definition))
    return out


def erpcore_comparisons(document, wave):
    """ERP CORE: model 50 % area latency against the published measurement."""

    import importlib.util

    spec = importlib.util.spec_from_file_location("_erpcore", HERE / "erpcore.py")
    erpcore = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(erpcore)
    files = document.get("data", {}).get("files_sha256")
    data = dataset("erp-core", files_sha256=files)
    if files is None:
        data["note"] = ("the OSF deposit has no version number and the SHA-256 of the files "
                        "read were not recorded by this run; validation/erpcore.py records "
                        "them from now on")
    out = []
    for component in sorted(erpcore.DESIGNS):
        rows = [r for r in document["rows"] if r["component"] == component and
                not r.get("excluded") and r.get("area_latency") is not None and
                r.get("model_latency") is not None and np.isfinite(r["area_latency"]) and
                np.isfinite(r["model_latency"])]
        if not rows:
            continue
        design = {k: list(v) if isinstance(v, tuple) else v
                  for k, v in erpcore.DESIGNS[component].items()}
        definition = {"id": f"erpcore-{component}-area50-{wave}",
                      "definition": dict(design, fraction=0.5, wave=wave,
                                         measure="fractional area latency of the "
                                                 "component's polarity"),
                      "source": "validation/erpcore.py (DESIGNS, area_latency)"}
        definition["version"] = digest(definition["definition"])
        out.append(_paired_median(
            f"erpcore_{wave}/{component}/area_latency",
            "50 % area latency, median over participants",
            f"ERP CORE published measurement ({wave} wave)",
            [r["model_latency"] for r in rows], [r["area_latency"] for r in rows],
            f"the {len(rows)} participants with both latencies", data, definition))
    return out


def carla_comparisons(document):
    """ds004977: channels left out by CARLA (the authors' code) and by ``car_adaptive``."""

    commit = document.get("comparator_commit")
    definition = {"id": "CARLA_JNM: CARLA(tt, V, srate, true), sensitive optimum, "
                        "rng('default')",
                  "repository": "https://github.com/hharveygit/CARLA_JNM",
                  "commit": commit}
    if commit is None:
        definition["note"] = ("the commit of CARLA_JNM run on 2026-10-08 was not recorded; "
                              "validation/carla_ds004977.py records it from now on")
    definition["version"] = commit
    out = []
    for site in document["sites"]:
        if "evoparam_used" not in site:
            continue
        channels = site["channels"]
        theirs = np.array([c not in site["carla_used"] for c in channels])
        ours = np.array([c not in site["evoparam_used"] for c in channels])
        data = dataset("ds004977", subject=site["subject"], run=site["run"],
                       site=site["site"])
        out.append(_paired_rate(f"carla_ds004977/{site['site']}/left_out_fraction",
                                "fraction of channels left out of the common average",
                                "CARLA, the authors' code (Huang et al. 2024)", ours, theirs,
                                f"the {len(channels)} channels", data, definition))
    return out


###################################################################################################

def build():
    read = lambda name: json.loads((LAYER3 / name).read_text("utf-8"))  # noqa: E731
    n1 = read("n1_ds004080.json")
    layer4 = json.loads((HERE / "layer4" / n1["version"] / "report.json").read_text("utf-8"))
    comparisons = (n1_comparisons(n1, layer4)
                   + erpcore_comparisons(read("erpcore_difference.json"), "difference")
                   + erpcore_comparisons(read("erpcore_parent.json"), "parent")
                   + carla_comparisons(read("carla_ds004977.json")))
    return {"schema": SCHEMA,
            "description": "Layer-3 comparisons (spec §7.3, REQ-374), generated by "
                           "validation/layer3_report.py from the layer-3 result files; "
                           "difference = evoparam - comparator.",
            "sources": ["n1_ds004080.json", f"../layer4/{n1['version']}/report.json",
                        "erpcore_difference.json", "erpcore_parent.json",
                        "carla_ds004977.json"],
            "comparisons": comparisons}


def render():
    return json.dumps(build(), indent=1, ensure_ascii=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        current = OUT.read_text("utf-8") if OUT.exists() else ""
        if current != text:
            print(f"{OUT} is out of date: run python validation/layer3_report.py")
            return 1
        print(f"{OUT} is up to date")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}: {len(json.loads(text)['comparisons'])} comparisons")
    return 0


if __name__ == "__main__":
    sys.exit(main())
