# SPDX-License-Identifier: Apache-2.0
"""Layer 2 of the scientific validation: recovery on simulations with a known truth (spec §7.2).

Two parts, one JSON report and one Markdown summary in ``validation/layer2/<version>/``:

1. **Grid.** Each cell simulates trials of one modality (the time scale, sampling rate and
   shapes of its preset), with ``K`` components of a given width (samples) and separation
   (FWHM), a signal-to-noise ratio (peak / noise SD of the average), a trial count, a latency
   jitter (fraction of the FWHM) and a noise family; the average is fitted with the preset
   and scored against the truth with :func:`evoparam.sim.score_fits` (recall, precision,
   F1, latency error in samples, amplitude bias, width error, false components at
   ``K = 0``). The full factorial of §7.2 has 276 480 cells; the designs run here are
   ``"weekly"`` (every factor varied one at a time around a base cell, for every modality,
   plus the SNR x separation map of F1 for two components, which shows where the method
   fails) and ``"smoke"`` (a few cells, for the test suite).
2. **REQ tests.** The tests of §5 whose thresholds are defined on simulations are run with
   pytest; each one is reported passed or failed with its duration.

Run ``python validation/layer2.py --design weekly --replicates 50 --jobs -1``.
The artifact factor of §7.2 is not part of the grid (artifact removal is validated by its
own REQs, REQ-060 to REQ-069, which part 2 runs).
"""

import argparse
import itertools
import json
import os
import pathlib
import subprocess
import sys
import time
import warnings

import numpy as np

MODALITIES = {  # generator name -> (preset, sampling rate, epoch)
    "erp": ("erp", 500.0, (-0.2, 0.8)),
    "ccep": ("ccep", 2048.0, (-0.5, 0.6)),
    "tep": ("tep", 5000.0, (-0.3, 0.35)),
    "erna": ("erna", 4096.0, (-0.002, 0.009)),
    "ecap": ("ecap", 30000.0, (-0.0005, 0.0035)),
    "animal": ("rodent_ep", 1250.0, (-0.1, 0.35)),
}
FACTORS = {
    "snr": (1, 2, 3, 5, 10, 20),
    "n_components": (0, 1, 2, 3),
    "separation": (0.3, 0.5, 1.0, 2.0),
    "width": (3, 6, 12, 25),
    "n_trials": (5, 10, 20, 40, 100),
    "jitter": (0.0, 0.1, 0.25, 0.5),
    "noise": ("white", "ar1:0.5", "ar1:0.9", "pink:1", "pink:2"),
}
BASE = {"snr": 5, "n_components": 2, "separation": 2.0, "width": 12, "n_trials": 20,
        "jitter": 0.0, "noise": "white"}

# Simulation-based acceptance tests of spec §5 (files and the REQs they hold).
REQ_TESTS = {
    "evoparam/tests/models/test_engine_sim.py": "REQ-104 to REQ-112",
    "evoparam/tests/existence/test_existence_null.py": "REQ-130, REQ-131",
    "evoparam/tests/existence/test_existence_power.py": "REQ-132, REQ-133",
    "evoparam/tests/trials/test_trials_sim.py": "REQ-150 to REQ-155",
    "evoparam/tests/hierarchical/test_hier_eb.py": "REQ-170 to REQ-172, REQ-175",
    "evoparam/tests/tf/test_tf_sim.py": "REQ-212 to REQ-215",
    "evoparam/tests/presets/test_presets_sim.py": "REQ-276, REQ-277",
}


def cells(design):
    """The (modality, settings) cells of a design."""

    out = []
    if design == "smoke":
        for modality in ("erp", "ccep"):
            out.append((modality, dict(BASE)))
            out.append((modality, dict(BASE, n_components=0)))
        return out
    for modality in MODALITIES:
        seen = set()
        for factor, levels in FACTORS.items():
            for level in levels:
                cell = dict(BASE, **{factor: level})
                key = tuple(sorted(cell.items()))
                if key not in seen:
                    seen.add(key)
                    out.append((modality, cell))
    for modality in ("erp", "ccep"):
        for snr, sep in itertools.product(FACTORS["snr"], FACTORS["separation"]):
            cell = dict(BASE, snr=snr, separation=sep)
            out.append((modality, cell))
    return out


def _noise(kind, n_trials, n_times, sfreq, rng):
    from evoparam.sim.noise import make_noise

    name, _, value = kind.partition(":")
    kwargs = {}
    if name == "ar1":
        kwargs["rho"] = float(value)
    elif name == "pink":
        kwargs["chi"] = float(value)
    return make_noise(n_times, sfreq, name, size=(n_trials,), seed=rng, **kwargs)


def simulate_cell(modality, cell, seed):
    """Trials (TrialSet) and the truth table of one replicate of a cell."""

    import pandas as pd

    from evoparam import presets
    from evoparam.data import TrialSet
    from evoparam.shapes import get_shape
    from evoparam.sim.components import shape_params

    name, sfreq, epoch = MODALITIES[modality]
    p = presets.get(name)
    rng = np.random.default_rng(seed)
    times = np.arange(int(round(epoch[0] * sfreq)), int(round(epoch[1] * sfreq))) / sfreq
    shape = p.shapes[0]
    fwhm = float(np.clip(cell["width"] / sfreq, p.min_fwhm * 1.05, p.max_fwhm * 0.95))
    start, stop = p.fit_window
    first = start + 0.15 * (stop - start)
    k = cell["n_components"]
    latencies = first + np.arange(k) * cell["separation"] * fwhm
    latencies = latencies[latencies < stop - 0.1 * (stop - start)]
    noise = _noise(cell["noise"], cell["n_trials"], times.size, sfreq, rng)
    sd_average = float(np.std(noise.mean(axis=0)[times < 0]))
    amplitude = cell["snr"] * sd_average
    data = noise.copy()
    rows = []
    form = {"freq": 300.0} if shape in ("damped_oscillation", "gabor") else {}
    model = get_shape(shape)
    for j, latency in enumerate(latencies):
        sign = -1.0 if j % 2 == 0 else 1.0
        shifts = rng.normal(0, cell["jitter"] * fwhm, cell["n_trials"])
        for i, shift in enumerate(shifts):
            params = shape_params(shape, latency + shift, fwhm, **form)
            data[i] += sign * amplitude * model.evaluate(times, params)
        rows.append({"channel": "c0", "latency": latency, "amplitude": sign * amplitude,
                     "fwhm": fwhm})
    truth = pd.DataFrame(rows, columns=["channel", "latency", "amplitude", "fwhm"])
    truth.attrs["sfreq"] = sfreq
    ts = TrialSet(data=data[:, None], times=times, sfreq=sfreq, ch_names=["c0"])
    return ts, truth


def run_cell(modality, cell, replicates, seed0):
    """Scores of one cell, averaged over replicates (and the time it took)."""

    import pandas as pd

    from evoparam.models import fit
    from evoparam.sim import score_fits

    start = time.perf_counter()
    name = MODALITIES[modality][0]
    scores = []
    for r in range(replicates):
        ts, truth = simulate_cell(modality, cell, seed0 + r)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                group = fit(ts, name)
            except Exception as err:  # a failed fit is a result of the grid
                scores.append({"error": type(err).__name__})
                continue
        if len(truth):
            scores.append(score_fits(truth, group).iloc[0].to_dict())
        else:
            scores.append({"false_per_waveform":
                           float(len(group.fits[0].components))})
    table = pd.DataFrame(scores)
    out = {"modality": modality, **cell, "replicates": replicates,
           "duration_s": time.perf_counter() - start,
           "errors": int(table["error"].notna().sum()) if "error" in table else 0}
    for column in ("recall", "precision", "f1", "latency_mae_samples", "amplitude_rel_bias",
                   "fwhm_rel_mae", "false_per_waveform"):
        if column in table:
            values = pd.to_numeric(table[column], errors="coerce")
            out[column] = float(values.mean()) if values.notna().any() else None
    return out


def run_grid(design, replicates, jobs):
    todo = cells(design)
    args = [(m, c, replicates, 1000 * i) for i, (m, c) in enumerate(todo)]
    if jobs == 1:
        return [run_cell(*a) for a in args]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=None if jobs < 0 else jobs) as pool:
        return list(pool.map(run_cell, *zip(*args)))


def run_req_tests(root):
    """Run the simulation-based REQ test files; one result per test."""

    import xml.etree.ElementTree as ET

    results = []
    report = pathlib.Path(root) / "validation" / "layer2" / "_junit.xml"
    report.parent.mkdir(parents=True, exist_ok=True)
    files = [f for f in REQ_TESTS if (pathlib.Path(root) / f).exists()]
    env = dict(os.environ, EVOPARAM_SLOW_TESTS="1")
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    f"--junitxml={report}", *files], cwd=root, env=env, check=False)
    for case in ET.parse(report).getroot().iter("testcase"):
        failed = case.find("failure") is not None or case.find("error") is not None
        skipped = case.find("skipped") is not None
        path = case.get("classname", "").replace(".", "/") + ".py"
        results.append({"test": f"{case.get('classname')}::{case.get('name')}",
                        "reqs": REQ_TESTS.get(path, ""),
                        "status": "skipped" if skipped else ("failed" if failed else "passed"),
                        "duration_s": float(case.get("time", 0))})
    report.unlink()
    return results


def summary_markdown(report):
    lines = [f"# Layer-2 validation, evoparam {report['version']}", "",
             f"Design `{report['design']}`, {report['replicates']} replicates per cell, "
             f"{len(report['grid'])} cells, {report['duration_s'] / 60:.1f} min.", "",
             "## REQ tests on simulations", "", "| test | REQs | status | s |", "|---|---|---|---|"]
    for row in report["req_tests"]:
        lines.append(f"| {row['test']} | {row['reqs']} | {row['status']} | "
                     f"{row['duration_s']:.1f} |")
    lines += ["", "## Grid", "",
              "| modality | snr | K | sep | width | trials | jitter | noise | F1 | "
              "latency MAE (samples) | amp. bias | false/waveform |",
              "|" + "---|" * 12]
    fmt = lambda v, f="{:.2f}": "" if v is None else f.format(v)
    for row in report["grid"]:
        lines.append(f"| {row['modality']} | {row['snr']} | {row['n_components']} | "
                     f"{row['separation']} | {row['width']} | {row['n_trials']} | "
                     f"{row['jitter']} | {row['noise']} | {fmt(row.get('f1'))} | "
                     f"{fmt(row.get('latency_mae_samples'))} | "
                     f"{fmt(row.get('amplitude_rel_bias'))} | "
                     f"{fmt(row.get('false_per_waveform'))} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    from evoparam.version import __version__

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--design", default="weekly", choices=("weekly", "smoke"))
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--no-tests", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    root = pathlib.Path(__file__).resolve().parents[1]
    start = time.perf_counter()
    grid = run_grid(args.design, args.replicates, args.jobs)
    tests = [] if args.no_tests else run_req_tests(root)
    report = {"version": __version__, "design": args.design, "replicates": args.replicates,
              "grid": grid, "req_tests": tests, "duration_s": time.perf_counter() - start,
              "factors": FACTORS, "base": BASE}
    out = pathlib.Path(args.out or root / "validation" / "layer2" / __version__)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    (out / "report.md").write_text(summary_markdown(report))
    failed = [t for t in tests if t["status"] == "failed"]
    print(f"{len(grid)} cells, {len(tests)} REQ tests ({len(failed)} failed), "
          f"{report['duration_s'] / 60:.1f} min -> {out}")
    return report


if __name__ == "__main__":
    main()
