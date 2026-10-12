# SPDX-License-Identifier: Apache-2.0
"""Layer 4 of the scientific validation: robustness and sensitivity (spec §7.4, REQ-375).

The five analyses of §7.4, run by the release pipeline of every minor version
(``.github/workflows/release.yml``, ``RELEASING.md``) and attached to the release notes:

1. **montage**: every ds004080 run of layer 3 (the example run and the REQ-371 runs) and
   every ds004977 site is cleaned (template subtraction of the preset), then the existence
   test (§5.6) is run under the bipolar, common-average and adjusted common-average
   references with :func:`evoparam.prep.reference_sensitivity`; Cohen's kappa of the
   decisions and the latency differences are reported for every pair of references.
2. **preset**: each *(choice)* field of §6 (``Preset.choices``) is multiplied by 0.5 and by
   1.5, one at a time (intervals: both ends; integers rounded, at least 1), and the preset's
   simulation is fitted again (20 seeds; the ``ccep`` artifact window on simulated SPES
   with artifacts); a change of recall larger than 10 % of the unperturbed recall, or of
   the median latency error larger than 2 ms, is reported as a sensitivity and listed in
   ``docs/preset_sensitivity.md``.
3. **sampling**: the ds004080 layer-3 runs decimated to 1024 and 512 Hz
   (:func:`evoparam.prep.resample` on the raw trials, then the whole ``ccep`` pipeline; a
   run recorded at or below the rate is kept as recorded, see :func:`decimates`);
   N1 recall against the published rule at the native rate must stay within 5 percentage
   points of the native recall down to 1024 Hz.
4. **trials**: 5, 7 and 10 trials drawn per site (5 draws each, seeded), cleaned and
   re-referenced as a recording of that size would be, existence test again; the fraction
   of decisions that change and the kappa with the all-trial decision are reported per
   trial count (``evoparam.stats.trials_needed`` gives the same curve for one channel).
5. **platform**: the fit of the bundled fixture against the reference values of
   ``evoparam/tests/reference/fit_fixture_ccep.json`` (REQ-295, rtol 1e-9); this machine's
   platform is recorded; the other operating systems are the cells of the CI matrix.

The native-rate N1 agreement of ds004080 is also the layer-3 N1 metric file
``validation/layer3/n1_ds004080.json`` read by the release gate of REQ-376
(``tools/layer3_gate.py``).

Data: the epochs cached by the network tests under the evoparam data folder
(``n1-validation/ds004080``, ``reference-validation/ds004977``); a missing file is fetched by
those tests (``EVOPARAM_NETWORK_TESTS=1``), not here. Nothing is downloaded by this script.

Run ``python validation/robustness.py --jobs 2`` (writes ``validation/layer4/<version>/``).
"""

import argparse
import json
import os
import pathlib
import platform
import sys
import time
import warnings

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
ANALYSES = ("montage", "preset", "sampling", "trials", "platform")
SCHEMES = ("bipolar", "car", "car_adaptive")
RATES = (1024.0, 512.0)
TRIAL_COUNTS = (5, 7, 10)
N_DRAWS = 5
PRESET_FACTORS = (0.5, 1.5)
PRESET_SEEDS = 20
# fields compared with their own unperturbed case (the artifact simulation; label recall)
OWN_BASE = ("artifact_window", "component_labels")
# criteria of §7.4
SAMPLING_RECALL_POINTS = 5.0          # N1 recall within 5 points down to 1024 Hz
PRESET_RECALL_REL = 0.10              # > 10 % change in recall
PRESET_LATENCY_MS = 2.0               # > 2 ms change of the median latency
EXAMPLE = ("ccepAgeUMCU01", "021448", "PT01-PT02")
DS004977 = [("1", "01", "ROC11-ROC12"), ("2", "12", "RZ4-RZ5"), ("3", "03", "LA2-LA3"),
            ("4", "02", "RO3-RO4")]
READINGS = {"paper": "vanblooijs2023", "code": "vanblooijs2023_code"}


def data_folder():
    from evoparam.io._fetch import data_folder as folder

    return folder()


###################################################################################################
## Data

def layer3_runs():
    """(subject, run, pair) of layer 3 on ds004080: the example run of spec §1, then the
    REQ-371 runs of ``validation/ds004080_subjects.txt`` with their first pair of >= 10
    trials (the cached events give the pair)."""

    import pandas as pd

    out = [EXAMPLE]
    cache = pathlib.Path(data_folder()) / "n1-validation" / "ds004080"
    with open(HERE / "ds004080_subjects.txt", encoding="utf-8") as fobj:
        rows = [line.strip().split("\t") for line in fobj if not line.startswith("#")]
    for subject, stem in rows:
        events = cache / f"{stem}_events.tsv"
        if not events.exists():
            continue
        table = pd.read_csv(events, sep="\t")
        sites = [s for s in table["electrical_stimulation_site"] if isinstance(s, str)]
        counts = pd.Series(sites).value_counts()
        pair = next(s for s in dict.fromkeys(sites) if counts[s] >= 10)
        out.append((subject[len("sub-"):], stem.split("_run-")[1], pair))
    return out


def load_ds004080(subject, run, pair):
    import pandas as pd

    from evoparam.data import TrialSet

    path = pathlib.Path(data_folder()) / "n1-validation" / "ds004080" / \
        f"sub-{subject}_run-{run}_{pair}.npz"
    with np.load(path, allow_pickle=False) as seg:
        names = [str(n) for n in seg["names"]]
        return TrialSet(data=seg["data"].astype(float), times=seg["times"],
                        sfreq=float(seg["sfreq"]), ch_names=names,
                        metadata=pd.DataFrame({"stim_site": [pair] * len(seg["data"])}))


def load_ds004977(subject, run, site):
    import pandas as pd

    from evoparam.data import TrialSet

    path = pathlib.Path(data_folder()) / "reference-validation" / "ds004977" / \
        f"sub-{subject}_run-{run}_{site}.npz"
    with np.load(path, allow_pickle=False) as saved:
        names = [str(name) for name in saved["names"]]
        bad = saved["bad"].astype(bool)
        stim = set(site.split("-"))
        keep = ~bad & np.array([n not in stim for n in names]) & \
            np.isfinite(saved["data"]).all(axis=(0, 2))
        return TrialSet(data=saved["data"][:, keep].astype(float), times=saved["times"],
                        sfreq=float(saved["sfreq"]),
                        ch_names=[n for n, k in zip(names, keep, strict=True) if k],
                        metadata=pd.DataFrame({"stim_site": saved["sites"].astype(str)}))


def available(kind):
    folder = pathlib.Path(data_folder())
    if kind == "ds004080":
        return [r for r in layer3_runs() if (folder / "n1-validation" / "ds004080" /
                                            f"sub-{r[0]}_run-{r[1]}_{r[2]}.npz").exists()]
    return [r for r in DS004977 if (folder / "reference-validation" / "ds004977" /
                                    f"sub-{r[0]}_run-{r[1]}_{r[2]}.npz").exists()]


###################################################################################################
## The ccep pipeline and the published rule (as the layer-3 tests run them)

def published_rule(ts, reading):
    """The registry rule of one reading of the published N1 rule (owner decision D8), as
    ``evoparam/tests/models/test_engine_ccep.py`` applies it: decision, latency (s)."""

    from evoparam.existence.rules import apply_rule, resolve_rule

    t = ts.times
    x = ts.data - np.median(ts.data[..., (t > -1) & (t < -0.1)], axis=-1, keepdims=True)
    avg = x.mean(axis=0)
    base = (t > -2) & (t < -0.1)
    avg = avg - np.median(avg[:, base], axis=1, keepdims=True)
    spec = resolve_rule(READINGS[reading], baseline=(-2.0, -0.1))
    out = apply_rule(avg, t + 1.0 / ts.sfreq, ts.sfreq, spec, "uV")
    return np.asarray(out["significant"], bool), np.asarray(out["onset"], float) - 1 / ts.sfreq


def clean(ts, tmin=-1.0, tmax=2.0):
    from evoparam.artifact import remove_artifact

    sub = ts.select(tmin=tmin, tmax=tmax)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return remove_artifact(sub, method="template", window=(-0.002, 0.008),
                               baseline=(-0.5, -0.02)).cleaned


def model_n1(ts):
    """The ccep preset pipeline of layer 3 and the preset's N1 (a negative component in
    9-50 ms; latency of the largest): decision and latency per channel."""

    from evoparam.models import ResponseModel
    from evoparam.prep import baseline_correct, rereference

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sub = rereference(clean(ts), "car_adaptive", random_state=0)
        evoked = baseline_correct(sub, (-0.5, -0.02), check=False).average()
        fits = ResponseModel("ccep").fit_group(evoked).fits
    det, lat = [], []
    for fit in fits:
        n1 = [c for c in fit.components if 0.009 <= c.latency <= 0.05 and c.amplitude < 0]
        det.append(bool(n1))
        lat.append(max(n1, key=lambda c: abs(c.amplitude)).latency if n1 else np.nan)
    return np.array(det), np.array(lat)


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if not len(a):
        return float("nan")
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def n1_metrics(rule, rule_lat, model, model_lat):
    """Kappa, recall (rule N1s the model finds), median |latency difference| (ms) and the
    fraction within 2 ms on the channels where both find an N1."""

    rule, model = np.asarray(rule, bool), np.asarray(model, bool)
    both = rule & model
    diff = np.abs(np.asarray(model_lat)[both] - np.asarray(rule_lat)[both]) * 1000
    return {"kappa": kappa(rule, model),
            "recall": float(both.sum() / rule.sum()) if rule.sum() else float("nan"),
            "median_abs_latency_diff_ms": float(np.median(diff)) if diff.size else float("nan"),
            "within_2ms": float(np.mean(diff <= 2.0)) if diff.size else float("nan"),
            "n_channels": int(rule.size), "n_rule": int(rule.sum()),
            "n_model": int(model.sum()), "n_both": int(both.sum())}


###################################################################################################
## Per-run work (one process per run)

def _existence(ts, random_state=0):
    from evoparam.existence import existence_detector

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        det, lat, _ = existence_detector(ts, random_state=random_state)
    return np.asarray(det, bool).ravel(), np.asarray(lat, float).ravel()


def montage_table(ts, baseline):
    from evoparam.prep import reference_sensitivity

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = reference_sensitivity(ts, schemes=SCHEMES, detector="existence",
                                      test_kwargs={"baseline": baseline, "random_state": 0},
                                      reference_kwargs={"random_state": 0})
    return [{k: (None if isinstance(v, float) and not np.isfinite(v) else
                 (v.item() if hasattr(v, "item") else v)) for k, v in row.items()}
            for row in table.to_dict("records")]


def decimates(sfreq, rate):
    """Whether the sampling analysis resamples a run recorded at ``sfreq`` to ``rate``.

    §7.4 (3) *decimates* ds004080 to 1024 and 512 Hz: only a run recorded above the rate is
    resampled. A run recorded at or below it (the example run of spec §1 is at 512 Hz) is
    analysed as recorded. Upsampling it would not give a 1024 Hz recording: the result has
    no power between the old and the new Nyquist frequency. On that run the ``ccep`` fit then
    finds N1s on 54 channels instead of 14 (CHANGELOG, REQ-375).
    """

    return sfreq > rate * (1 + 1e-6)


def run_ds004080(subject, run, pair, analyses):
    """Everything §7.4 asks of one ds004080 run."""

    from evoparam.prep import rereference, resample

    start = time.perf_counter()
    ts = load_ds004080(subject, run, pair)
    out = {"subject": subject, "run": run, "pair": pair, "sfreq": ts.sfreq,
           "n_trials": ts.n_trials, "n_channels": ts.n_channels}
    rules = {reading: published_rule(ts, reading) for reading in READINGS}
    out["rule"] = {r: {"decision": d.tolist(), "latency": np.where(d, lat, np.nan).tolist()}
                   for r, (d, lat) in rules.items()}
    if "sampling" in analyses or "layer3" in analyses:
        det, lat = model_n1(ts)
        out["native"] = {"decision": det.tolist(), "latency": lat.tolist()}
    if "sampling" in analyses:
        out["sampling"] = {}
        for rate in RATES:
            if not decimates(ts.sfreq, rate):
                # §7.4 decimates: a run recorded at or below the rate is kept as recorded
                out["sampling"][str(int(rate))] = dict(out["native"], resampled=False)
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                low = resample(ts, rate)
            d, lat = model_n1(low)
            out["sampling"][str(int(rate))] = {"decision": d.tolist(), "latency": lat.tolist(),
                                               "resampled": True}
    if "montage" in analyses:
        out["montage"] = montage_table(clean(ts), (-0.5, -0.05))
    if "trials" in analyses:
        cleaned = clean(ts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full, _ = _existence(rereference(cleaned, "car_adaptive", random_state=0))
        rng = np.random.default_rng(375)
        rows = []
        for n in TRIAL_COUNTS:
            if n > cleaned.n_trials:
                continue
            for draw in range(N_DRAWS):
                idx = np.sort(rng.choice(cleaned.n_trials, n, replace=False))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sub = rereference(cleaned.select(trials=idx), "car_adaptive",
                                      random_state=0)
                det, _ = _existence(sub)
                rows.append({"n_trials": n, "draw": draw, "kappa": kappa(full, det),
                             "changed": float(np.mean(det != full)),
                             "n_detected": int(det.sum()), "n_detected_all": int(full.sum())})
        out["trials"] = rows
    out["duration_s"] = time.perf_counter() - start
    return out


def run_ds004977(subject, run, site):
    start = time.perf_counter()
    ts = load_ds004977(subject, run, site)
    from evoparam.artifact import remove_artifact

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cleaned = remove_artifact(ts, method="template", window=(-0.002, 0.008),
                                  baseline=(-0.2, -0.02)).cleaned
    table = montage_table(cleaned, (-0.19, -0.02))
    return {"subject": subject, "run": run, "site": site, "sfreq": ts.sfreq,
            "n_trials": ts.n_trials, "n_channels": ts.n_channels, "montage": table,
            "duration_s": time.perf_counter() - start}


###################################################################################################
## Preset sensitivity (simulations)

SIM = {  # preset -> (sampling rate, epoch) of its simulation (layer 2 for the shared ones)
    "erp": (500.0, (-0.2, 0.8)), "ccep": (2048.0, (-0.5, 0.6)),
    "ccep_seeg": (2048.0, (-0.5, 0.6)), "sep": (5000.0, (-0.05, 0.1)),
    "tep": (5000.0, (-0.3, 0.35)), "erna": (4096.0, (-0.002, 0.009)),
    "ecap": (30000.0, (-0.0005, 0.0035)), "ecap_ci": (50000.0, (-0.0002, 0.002)),
    "rodent_ep": (1250.0, (-0.1, 0.35)), "primate_ep": (1250.0, (-0.1, 0.35)),
    "p300_bci": (500.0, (-0.2, 0.8)),
}


def perturb(value, factor):
    """``value`` times ``factor``: numbers, intervals (both ends), dicts of intervals."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return max(1, int(round(value * factor)))
    if isinstance(value, float):
        return value * factor
    if isinstance(value, (list, tuple)):
        return [perturb(v, factor) for v in value]
    if isinstance(value, dict):
        return {k: perturb(v, factor) for k, v in value.items()}
    return value


def simulate_preset(name, seed, epoch=None):
    """Trials of one replicate for a preset (the layer-2 base cell: two components, SNR 5,
    width 12 samples, separation 2 FWHM, 20 trials, white noise) and the truth table."""

    import pandas as pd

    from evoparam import presets
    from evoparam.data import TrialSet
    from evoparam.shapes import get_shape
    from evoparam.sim.components import shape_params

    p = presets.get(name)
    sfreq, default_epoch = SIM[name]
    epoch = epoch or default_epoch
    rng = np.random.default_rng(seed)
    times = np.arange(int(round(epoch[0] * sfreq)), int(round(epoch[1] * sfreq))) / sfreq
    shape = p.shapes[0]
    fwhm = float(np.clip(12 / sfreq, p.min_fwhm * 1.05, p.max_fwhm * 0.95))
    start, stop = p.fit_window
    latencies = start + 0.15 * (stop - start) + np.arange(2) * 2.0 * fwhm
    latencies = latencies[latencies < stop - 0.1 * (stop - start)]
    noise = rng.standard_normal((20, times.size))
    amplitude = 5 * float(np.std(noise.mean(axis=0)[times < 0])) if (times < 0).any() \
        else 5 / np.sqrt(20)
    data = noise.copy()
    rows = []
    form = {"freq": 300.0} if shape in ("damped_oscillation", "gabor") else {}
    model = get_shape(shape)
    for j, latency in enumerate(latencies):
        sign = -1.0 if j % 2 == 0 else 1.0
        params = shape_params(shape, latency, fwhm, **form)
        data += sign * amplitude * model.evaluate(times, params)
        rows.append({"channel": "c0", "latency": latency, "amplitude": sign * amplitude,
                     "fwhm": fwhm})
    truth = pd.DataFrame(rows, columns=["channel", "latency", "amplitude", "fwhm"])
    ts = TrialSet(data=data[:, None], times=times, sfreq=sfreq, ch_names=["c0"])
    return ts, truth


def _score(truth, group):
    from evoparam.sim import score_fits

    row = score_fits(truth, group).iloc[0]
    return float(row["recall"]), float(row["latency_mae"]) * 1000


def preset_case(name, field, factor, seeds=PRESET_SEEDS):
    """Recall and median latency error (ms) of a preset with one choice field scaled (the
    preset is rebuilt with :func:`dataclasses.replace` and passed to the fit)."""

    import dataclasses

    from evoparam import presets
    from evoparam.models import fit

    p = presets.get(name)
    value = None if field is None else perturb(getattr(p, field), factor)
    q = dataclasses.replace(p, **{field: value}) if field is not None else p
    recalls, errors, failures, errors_seen = [], [], 0, []
    for seed in range(seeds):
        try:
            if field == "artifact_window":
                recall, error = _artifact_case(q, seed)
            else:
                sim_epoch = SIM[name][1]
                if field == "epoch":
                    epoch = tuple(q.epoch)
                else:   # the simulated epoch holds the (perturbed) fit window
                    epoch = (sim_epoch[0], max(sim_epoch[1], q.fit_window[1] * 1.05))
                ts, truth = simulate_preset(name, seed, epoch=epoch)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    group = fit(ts, q, random_state=seed)
                recall, error = _score(truth, group)
                if field == "component_labels":
                    recall = _label_recall(truth, group, value)
        except Exception as err:  # noqa: BLE001 - a failed fit is a result of the analysis
            failures += 1
            errors_seen.append(f"{type(err).__name__}: {err}")
            recalls.append(0.0)
            continue
        recalls.append(recall)
        if np.isfinite(error):
            errors.append(error)
    return {"preset": name, "field": field, "factor": factor,
            "value": value, "recall": float(np.mean(recalls)),
            "median_latency_error_ms": float(np.median(errors)) if errors else float("nan"),
            "failures": failures, "failure_messages": sorted(set(errors_seen))[:3]}


def _label_recall(truth, group, labels):
    """Fraction of true components whose label window (first label holding the true latency)
    is the label of a fitted component within FWHM/2 of it."""

    fit = group.fits[0]
    found = 0
    for row in truth.itertuples():
        names = [k for k, (a, b) in labels.items() if a <= row.latency <= b]
        found += any(abs(c.latency - row.latency) <= row.fwhm / 2 and c.label in names
                     for c in fit.components)
    return found / len(truth) if len(truth) else float("nan")


def _artifact_case(p, seed):
    """Simulation with artifacts, the preset's artifact method with ``window``, then the
    preset fit; N1 recall and latency error (ms) against the truth. ``ccep``/``ccep_seeg``:
    SPES with template subtraction; ``ecap``: alternating-polarity ECAPs with the paradigm
    subtraction of the preset (which does not read the window: the analysis then shows no
    change, and would show one if a later version used it)."""

    from evoparam.artifact import remove_artifact
    from evoparam.models import fit
    from evoparam.sim import score_fits, simulate_ccep, simulate_ecap

    name, window = p.name, tuple(p.artifact_window)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name.startswith("ccep"):
            ts, truth = simulate_ccep(n_trials=20, n_channels=12, p_response=1.0, seed=seed,
                                      layout="grid" if name == "ccep" else "seeg")
            cleaned = remove_artifact(ts, method="template", window=window,
                                      baseline=tuple(p.baseline)).cleaned
        else:
            ts, truth = simulate_ecap(40, kind="spinal" if name == "ecap" else "cochlear",
                                      paradigm="alternating", seed=seed)
            cleaned = remove_artifact(ts, method="paradigm", scheme="alternating",
                                      window=window, baseline=tuple(p.baseline)).cleaned
        group = fit(cleaned, p, random_state=seed)
    table = truth.to_dataframe()
    row = score_fits(table[table["label"] == "N1"], group).iloc[0]
    return float(row["recall"]), float(row["latency_mae"]) * 1000


def preset_sensitivity(jobs=1, presets_=None, seeds=PRESET_SEEDS):
    from evoparam import presets

    cases = []
    for name in presets_ or list(presets.list()["name"]):
        p = presets.get(name)
        if not p.choices:
            continue
        cases.append((name, None, 1.0))
        for field in OWN_BASE:      # their own base (artifact simulation, label recall)
            if field in p.choices:
                cases.append((name, field, 1.0))
        for field in p.choices:
            for factor in PRESET_FACTORS:
                cases.append((name, field, factor))
    rows = _map(preset_case, cases, jobs, extra=(seeds,))
    base = {(r["preset"], r["field"]): r for r in rows
            if r["factor"] == 1.0 and r["field"] in (None,) + OWN_BASE}
    for r in rows:
        b = base[(r["preset"], r["field"] if r["field"] in OWN_BASE else None)]
        r["recall_change"] = r["recall"] - b["recall"]
        r["recall_change_rel"] = (r["recall_change"] / b["recall"]) if b["recall"] else \
            float("inf") if r["recall_change"] else 0.0
        r["latency_change_ms"] = r["median_latency_error_ms"] - b["median_latency_error_ms"]
        r["sensitive"] = bool(r["field"] is not None and r["factor"] != 1.0 and (
            abs(r["recall_change_rel"]) > PRESET_RECALL_REL or
            not np.isfinite(r["latency_change_ms"]) or
            abs(r["latency_change_ms"]) > PRESET_LATENCY_MS))
    return rows


###################################################################################################
## Platform (REQ-295)

def platform_check():
    sys.path.insert(0, str(ROOT))
    from evoparam.tests.reference import compare_fixture

    out = compare_fixture()
    out.update({"platform": platform.platform(), "machine": platform.machine(),
                "python": platform.python_version(),
                "other_platforms": "macOS and Windows (x86-64, arm64) are the cells of the "
                                   "tests matrix of .github/workflows/ci.yml "
                                   "(tests/test_determinism.py); not run on this machine",
                "bayesian_paths": "within Monte Carlo error: compared by the same CI cells"})
    return out


###################################################################################################
## Summaries

def _map(func, cases, jobs, extra=()):
    if jobs == 1:
        return [func(*c, *extra) for c in cases]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=jobs) as pool:
        columns = zip(*[tuple(c) + tuple(extra) for c in cases], strict=True)
        return list(pool.map(func, *columns))


def summarise_ds004080(runs):
    """Pooled N1 metrics at the native rate (layer 3) and per rate (sampling analysis)."""

    out = {}
    for reading in READINGS:
        rule = np.concatenate([r["rule"][reading]["decision"] for r in runs])
        rule_lat = np.concatenate([r["rule"][reading]["latency"] for r in runs])
        rows = {}
        if all("native" in r for r in runs):
            rows["native"] = n1_metrics(rule, rule_lat,
                                        np.concatenate([r["native"]["decision"] for r in runs]),
                                        np.concatenate([r["native"]["latency"] for r in runs]))
        for rate in RATES:
            key = str(int(rate))
            if all(key in r.get("sampling", {}) for r in runs):
                rows[key] = n1_metrics(
                    rule, rule_lat,
                    np.concatenate([r["sampling"][key]["decision"] for r in runs]),
                    np.concatenate([r["sampling"][key]["latency"] for r in runs]))
        out[reading] = rows
    return out


def sampling_verdict(pooled):
    out = {}
    for reading, rows in pooled.items():
        if "native" not in rows or "1024" not in rows:
            continue
        drop = (rows["native"]["recall"] - rows["1024"]["recall"]) * 100
        out[reading] = {"recall_native": rows["native"]["recall"],
                        "recall_1024": rows["1024"]["recall"],
                        "recall_512": rows.get("512", {}).get("recall"),
                        "drop_points_1024": drop,
                        "met": bool(drop <= SAMPLING_RECALL_POINTS)}
    return out


def trials_summary(runs):
    rows = [row for r in runs for row in r.get("trials", [])]
    out = []
    for n in TRIAL_COUNTS:
        sel = [row for row in rows if row["n_trials"] == n]
        if sel:
            kap = [row["kappa"] for row in sel if np.isfinite(row["kappa"])]
            out.append({"n_trials": n, "draws": len(sel),
                        "changed_mean": float(np.mean([row["changed"] for row in sel])),
                        "kappa_median": float(np.median(kap)) if kap else float("nan")})
    return out


def montage_summary(results):
    out = {}
    for pair in [("bipolar", "car"), ("bipolar", "car_adaptive"), ("car", "car_adaptive")]:
        rows = [row for r in results for row in r.get("montage", [])
                if (row["scheme_a"], row["scheme_b"]) == pair]
        kap = [row["kappa"] for row in rows if row["kappa"] is not None]
        lat = [row["median_latency_diff_ms"] for row in rows
               if row["median_latency_diff_ms"] is not None]
        out["-".join(pair)] = {"n": len(rows),
                               "kappa_median": float(np.median(kap)) if kap else None,
                               "kappa_min": float(np.min(kap)) if kap else None,
                               "median_latency_diff_ms_median":
                                   float(np.median(lat)) if lat else None}
    return out


def layer3_file(pooled, runs, version):
    """The N1 metrics of layer 3 at the native rate (``validation/layer3/n1_ds004080.json``),
    the input of the REQ-376 gate."""

    return {"dataset": "ds004080", "version": version,
            "data": {"dataset": "ds004080", "version": "1.2.4"},
            "comparators": {reading: {"registry_rule": rule, "overrides": {"baseline":
                                                                         [-2.0, -0.1]}}
                            for reading, rule in READINGS.items()},
            "runs": [f"sub-{r['subject']} run-{r['run']} {r['pair']}" for r in runs],
            "pipeline": "ccep preset (template artifact removal, car_adaptive, baseline, "
                        "average, fit_group); N1: negative component in 9-50 ms",
            "metrics": {reading: rows["native"] for reading, rows in pooled.items()
                        if "native" in rows}}


def _f(value, spec=".3f"):
    missing = value is None or (isinstance(value, float) and not np.isfinite(value))
    return "" if missing else format(value, spec)


def _rounded(value):
    """A perturbed setting for display: floats to 6 significant digits."""

    if isinstance(value, float):
        return float(f"{value:.6g}")
    if isinstance(value, (list, tuple)):
        return [_rounded(v) for v in value]
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    return value


def _sampling_notes(report):
    """Lines of the sampling section: when it was recomputed alone, and the runs kept as
    recorded."""

    lines = []
    again = report["sampling"].get("regenerated")
    if again:
        lines.append(f"\nThis section was recomputed alone on {again['date']} "
                     f"({again['duration_min']} min; {again['machine']}).")
    for rate in (str(int(x)) for x in RATES):
        names = [f"{r['subject']} run {r['run']} ({r['sfreq']:g} Hz)"
                 for r in report.get("ds004080", [])
                 if r.get("sampling", {}).get(rate, {}).get("resampled") is False]
        if names:
            lines.append(f"\nAt {rate} Hz, recorded at or below the rate and not "
                         f"resampled: {', '.join(names)}.")
    return lines


def markdown(report):
    lines = [f"# Layer-4 robustness, evoparam {report['version']}", "",
             f"{report['date']}; {report['machine']}; {report['duration_s'] / 60:.1f} min "
             f"with {report['jobs']} processes. Analyses: {', '.join(report['analyses'])}.", ""]
    if "montage" in report:
        lines += ["## 1. Montage (existence test under three references)", "",
                  "| pair | runs/sites | median kappa | lowest kappa | median latency diff (ms) |",
                  "|---|---|---|---|---|"]
        for pair, row in report["montage"]["summary"].items():
            lines.append(f"| {pair} | {row['n']} | {_f(row['kappa_median'], '.2f')} | "
                         f"{_f(row['kappa_min'], '.2f')} | "
                         f"{_f(row['median_latency_diff_ms_median'], '.2f')} |")
        lines.append("")
    if "preset" in report:
        lines += ["## 2. Preset choice fields (x0.5, x1.5)", "",
                  "| preset | field | factor | recall | change | median latency error (ms) "
                  "| change (ms) | sensitive |", "|---|---|---|---|---|---|---|---|"]
        for r in report["preset"]:
            lines.append(f"| {r['preset']} | {r['field'] or '(as set)'} | {r['factor']} | "
                         f"{_f(r['recall'])} | {_f(r['recall_change'], '+.3f')} | "
                         f"{_f(r['median_latency_error_ms'])} | "
                         f"{_f(r['latency_change_ms'], '+.3f')} | "
                         f"{'yes' if r['sensitive'] else ''} |")
        lines.append("")
    if "sampling" in report:
        lines += ["## 3. Sampling rate (ds004080, N1 against the published rule)", "",
                  "| reading | rate | kappa | recall | median abs latency diff (ms) |",
                  "|---|---|---|---|---|"]
        for reading, rows in report["sampling"]["pooled"].items():
            for rate, m in rows.items():
                lines.append(f"| {reading} | {rate} | {_f(m['kappa'])} | {_f(m['recall'])} | "
                             f"{_f(m['median_abs_latency_diff_ms'], '.2f')} |")
        lines += _sampling_notes(report)
        for reading, v in report["sampling"]["verdict"].items():
            lines.append(f"\n{reading}: recall drop at 1024 Hz {v['drop_points_1024']:+.2f} "
                         f"points (criterion <= {SAMPLING_RECALL_POINTS}): "
                         f"{'met' if v['met'] else 'NOT met'}.")
        lines.append("")
    if "trials" in report:
        lines += ["## 4. Trial count (existence decisions against all trials)", "",
                  "| trials | draws | decisions changed | median kappa |", "|---|---|---|---|"]
        for r in report["trials"]["summary"]:
            lines.append(f"| {r['n_trials']} | {r['draws']} | {_f(r['changed_mean'])} | "
                         f"{_f(r['kappa_median'])} |")
        lines.append("")
    if "platform" in report:
        p = report["platform"]
        lines += ["## 5. Platform", "",
                  f"{p['platform']}: fixture fit within rtol {p['rtol']:g} of the reference: "
                  f"{'yes' if p['match'] else 'no'} (largest relative difference "
                  f"{_f(p['max_rel_diff'], '.3g')}). {p['other_platforms']}.", ""]
    return "\n".join(lines) + "\n"


def preset_doc(rows, version):
    lines = ["# Sensitivity of the preset choice fields", "",
             "Generated by `validation/robustness.py` (spec §7.4, analysis 2) for evoparam "
             f"{version}; do not edit by hand. Each field that spec §6 marks *(choice)* "
             "(`Preset.choices`) is multiplied by 0.5 and by 1.5 (intervals: both ends), one at "
             "a time, and the preset's simulation is fitted again "
             f"({PRESET_SEEDS} seeds). A change of recall larger than 10 % of the unperturbed "
             "recall, or of the median latency error larger than 2 ms, is a sensitivity.", "",
             "| preset | field | factor | value | recall | change | latency change (ms) | "
             "sensitivity |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["field"] is None or r["factor"] == 1.0:
            continue
        lines.append(f"| `{r['preset']}` | `{r['field']}` | {r['factor']} | "
                     f"{json.dumps(_rounded(r['value']))} | {_f(r['recall'])} | "
                     f"{_f(r['recall_change'], '+.3f')} | "
                     f"{_f(r['latency_change_ms'], '+.3f') or 'n/a'} | "
                     f"{'**yes**' if r['sensitive'] else 'no'} |")
    return "\n".join(lines) + "\n"


def _clean_json(obj):
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, np.generic):
        return _clean_json(obj.item())
    return obj


def main(argv=None):
    from evoparam.version import __version__

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--analyses", default=",".join(ANALYSES),
                        help="comma-separated subset of " + ", ".join(ANALYSES))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=None,
                        help="first N ds004080 runs only (default: all cached)")
    parser.add_argument("--seeds", type=int, default=PRESET_SEEDS)
    parser.add_argument("--presets", default="", help="comma-separated presets (analysis 2)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--layer3", default=str(HERE / "layer3" / "n1_ds004080.json"),
                        help="where to write the native-rate N1 metrics ('' to skip)")
    args = parser.parse_args(argv)
    analyses = [a for a in args.analyses.split(",") if a]
    unknown = set(analyses) - set(ANALYSES)
    if unknown:
        parser.error(f"unknown analyses {sorted(unknown)}")
    start = time.perf_counter()
    report = {"version": __version__, "analyses": analyses, "jobs": args.jobs,
              "date": time.strftime("%Y-%m-%d"),
              "machine": f"{platform.platform()}, {os.cpu_count()} CPUs"}
    need = {"montage", "sampling", "trials"} & set(analyses)
    if need:
        runs = available("ds004080")[:args.runs]
        report["data"] = {"ds004080": [list(r) for r in runs]}
        results = _map(run_ds004080, runs, args.jobs, extra=(tuple(analyses) + ("layer3",),))
        report["ds004080"] = results
        pooled = summarise_ds004080(results)
        if "sampling" in analyses:
            report["sampling"] = {"pooled": pooled, "verdict": sampling_verdict(pooled)}
        if "trials" in analyses:
            report["trials"] = {"summary": trials_summary(results)}
        montage = list(results)
        if "montage" in analyses:
            sites = available("ds004977")
            report["data"]["ds004977"] = [list(s) for s in sites]
            report["ds004977"] = _map(run_ds004977, sites, args.jobs)
            montage += report["ds004977"]
            report["montage"] = {"summary": montage_summary(montage)}
        if args.layer3 and all("native" in r for r in results) and results:
            path = pathlib.Path(args.layer3)
            path.write_text(json.dumps(_clean_json(layer3_file(pooled, results, __version__)),
                                       indent=1) + "\n", encoding="utf-8")
    if "preset" in analyses:
        chosen = [n for n in args.presets.split(",") if n] or None
        report["preset"] = preset_sensitivity(args.jobs, presets_=chosen, seeds=args.seeds)
        if chosen is None:
            (ROOT / "docs" / "preset_sensitivity.md").write_text(
                preset_doc(report["preset"], __version__), encoding="utf-8")
    if "platform" in analyses:
        report["platform"] = platform_check()
    report["duration_s"] = time.perf_counter() - start
    out = pathlib.Path(args.out or HERE / "layer4" / __version__)
    out.mkdir(parents=True, exist_ok=True)
    report = _clean_json(report)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    (out / "report.md").write_text(markdown(report), encoding="utf-8")
    print(f"layer 4 written to {out} ({report['duration_s'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
