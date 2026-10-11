# SPDX-License-Identifier: Apache-2.0
"""Regenerate the figures of the evoparam papers from a tagged version (spec §11.8, REQ-393).

``make figures`` runs this script. Every figure is drawn from files of the repository only
(the validation results under ``validation/`` and seeded simulations of
:mod:`evoparam.sim`), written as SVG with fixed metadata and a fixed hash salt so that the
same tag and the same matplotlib version give byte-identical files, and listed with its
SHA-256 in ``figures/manifest.json``. ``--check`` regenerates the figures in a temporary
folder and compares their hashes with the manifest.

Figures:

- ``fig_ccep_fit.svg``: a simulated CCEP (``evoparam.sim.simulate_ccep``, seed 0), the model
  fitted by the ``ccep`` preset and its components;
- ``fig_n1_ds004080.svg``: agreement of the N1 of the ``ccep`` preset with the two readings
  of the published rule on ds004080 (``validation/layer3/n1_ds004080.json``);
- ``fig_erpcore.svg``: ERP CORE components labelled within 10 ms of the reference latency
  (``validation/layer3/erpcore_*.json``).

Usage::

    python tools/make_figures.py [--out figures] [--check]
"""

import argparse
import hashlib
import json
import pathlib
import sys
import tempfile
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "figures"
LAYER3 = ROOT / "validation" / "layer3"


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"svg.hashsalt": "evoparam", "svg.fonttype": "path",
                                "font.family": "DejaVu Sans", "figure.dpi": 100})
    import matplotlib.pyplot as plt

    return matplotlib, plt


def _save(fig, path):
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": None})
    import matplotlib.pyplot as plt

    plt.close(fig)


def fig_ccep_fit(path):
    _, plt = _matplotlib()
    import evoparam
    from evoparam import sim

    trials, _ = sim.simulate_ccep(n_trials=20, n_channels=3, seed=0, artifact=False,
                                  p_response=1.0, layout="grid")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = evoparam.fit(trials, preset="ccep", random_state=0, existence=False)
    fit = group.fits[0]
    average = trials.data[:, 0].mean(axis=0)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(trials.times * 1000, average, color="0.6", lw=1, label="average of 20 trials")
    if fit.model is not None:
        # the model is defined on the fit window
        start = int(np.searchsorted(trials.times, fit.window[0] - 1e-9))
        times = trials.times[start:start + len(fit.model)]
        ax.plot(times * 1000, fit.model, color="#0072B2", lw=1.5, label="model")
    for comp in fit.components:
        ax.axvline(comp.latency * 1000, color="#D55E00", lw=0.8, ls="--")
        ax.text(comp.latency * 1000, comp.amplitude, f" {comp.label or ''}", fontsize=8)
    ax.set_xlim(-100, 500)
    ax.set_xlabel("time after the pulse (ms)")
    ax.set_ylabel(f"amplitude ({trials.units})")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Simulated CCEP fitted with the ccep preset", fontsize=10)
    _save(fig, path)


def fig_n1_ds004080(path):
    _, plt = _matplotlib()
    doc = json.loads((LAYER3 / "n1_ds004080.json").read_text(encoding="utf-8"))
    readings = sorted(doc["metrics"])
    names = ("kappa", "recall", "within_2ms")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    width = 0.8 / len(readings)
    for k, reading in enumerate(readings):
        values = [doc["metrics"][reading][n] for n in names]
        ax.bar([i + k * width for i in range(len(names))], values, width,
               label=f"published rule, {reading} reading")
    ax.set_xticks([i + width * (len(readings) - 1) / 2 for i in range(len(names))])
    ax.set_xticklabels(["Cohen's kappa", "recall", "latency within 2 ms"])
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"N1 on ds004080 ({len(doc['runs'])} runs), evoparam {doc['version']}",
                 fontsize=10)
    _save(fig, path)


def fig_erpcore(path):
    _, plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for k, name in enumerate(("erpcore_parent", "erpcore_difference")):
        doc = json.loads((LAYER3 / f"{name}.json").read_text(encoding="utf-8"))
        rows = doc["summary"]
        ax.bar([i + k * 0.4 for i in range(len(rows))], [r["within_10ms"] for r in rows], 0.4,
               label=name.split("_")[1] + " waves")
        ax.set_xticks([i + 0.2 for i in range(len(rows))])
        ax.set_xticklabels([r["component"] for r in rows], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share within 10 ms of the reference")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("ERP CORE components (40 participants)", fontsize=10)
    _save(fig, path)


FIGURES = {"fig_ccep_fit.svg": fig_ccep_fit, "fig_n1_ds004080.svg": fig_n1_ds004080,
           "fig_erpcore.svg": fig_erpcore}


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def make(out):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    matplotlib, _ = _matplotlib()
    from evoparam.version import __version__

    for name, draw in FIGURES.items():
        draw(out / name)
    return {"schema": "evoparam-figures/1", "evoparam": __version__,
            "matplotlib": matplotlib.__version__,
            "figures": {name: sha256(out / name) for name in FIGURES}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--check", action="store_true",
                        help="regenerate in a temporary folder and compare with the manifest")
    args = parser.parse_args(argv)
    manifest_path = pathlib.Path(args.out) / "manifest.json"
    if args.check:
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as folder:
            now = make(folder)
        if now["matplotlib"] != recorded["matplotlib"]:
            print(f"matplotlib {now['matplotlib']} here, {recorded['matplotlib']} in the "
                  "manifest: hashes are comparable only with the same version")
        bad = [n for n in FIGURES if now["figures"][n] != recorded["figures"].get(n)]
        for name in bad:
            print(f"{name}: sha256 {now['figures'][name]} != {recorded['figures'].get(name)}")
        print(f"{len(FIGURES) - len(bad)} of {len(FIGURES)} figures identical")
        return 1 if bad else 0
    manifest = make(args.out)
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(FIGURES)} figures and {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
