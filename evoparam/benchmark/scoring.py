"""Scoring of task outputs against the truth, and the signed result (spec §8.1, §8.4).

``score(task, predictions, suite)`` regenerates the inputs and the truth of every cell of
the suite from its seeds, reads and validates the method's output for each cell (the first
violation is named, REQ-351), computes the metrics of the task pooled over the cells and
for each cell, and returns the result document. The document is *signed*: it carries the
SHA-256 of every input and output, the suite (name, version, SHA-256 of the suite file),
the package version, and ``signature.digest``, the SHA-256 of its own canonical JSON
without the signature. Nothing in it depends on the clock or on the machine's paths, so the
same inputs and outputs give the same bytes (REQ-352).

Metrics (units: seconds for latencies, data units for amplitudes):

- **T1** (true responders: the contacts given an N1 by the simulation, or on real data the
  consensus of the published rules; the stimulated contacts, and on real data the contacts
  where the rules disagree, are not scored): ``auc`` (Mann-Whitney, from the ``score`` column, else
  ``1 - p_value``, else the decision; ``auc_source`` says which), ``f1``, ``precision``,
  ``recall``, ``specificity`` at the method's own decision, and the calibration of the
  p-values on the true non-responders: ``fpr_at_0.01``, ``fpr_at_0.05``, ``fpr_at_0.1``
  (fraction of null p-values at or below the level) and ``ks_uniform`` (largest distance
  between their empirical distribution and the uniform one).
- **T2** (true responders): ``reported_fraction`` (latency given), ``latency_mae``,
  ``latency_bias``, ``amplitude_mae``, ``amplitude_bias``, ``amplitude_rel_mae``,
  ``coverage_latency`` and ``coverage_amplitude`` (fraction of reported intervals holding
  the truth; null without intervals).
- **T3**: the matching of :func:`evoparam.sim.score_fits` (Hungarian matching on latency
  within half the true FWHM): ``f1``, ``recall``, ``precision``, ``latency_mae``,
  ``amplitude_rel_mae``, ``fwhm_rel_mae``, ``false_per_waveform`` and ``false_rate_empty``
  (false components per waveform without true component).
- **T4** (stimulated contacts excluded): ``artifact_residual_rms`` (RMS of cleaned minus
  artifact-free epochs from the start of the artifact window to 50 ms, every trial and
  contact), ``artifact_residual_rel`` (the same over the RMS of the artifact there),
  ``response_rmse_50ms`` (RMS over the responding contacts of the trial average of that
  difference on 0-50 ms) and ``n1_latency_error`` (mean absolute shift of the most negative
  sample of the average on 9-50 ms, cleaned against artifact-free).
- **T5** (target channels): ``reported_fraction``, ``latency_mae``, ``latency_bias``,
  ``amplitude_mae``, ``amplitude_rel_mae``, ``latency_r`` and ``amplitude_r`` (Pearson
  correlation with the truth over the trials of a channel, averaged over channels),
  ``jitter_sd_error`` and ``jitter_sd_rel_error`` (absolute error of the SD of the
  estimated latencies against the SD of the true ones).
- **T6** (per measure ``m``): ``m.mae_vs_definition`` and ``m.max_abs_vs_definition``
  (against the reference definition computed by :mod:`evoparam.models.classical` on the
  input average), ``m.mae_vs_truth`` and ``m.bias_vs_truth`` (against the same definition
  on the noise-free average), ``m.sme_mean`` (mean reported SME) and ``m.sme_ratio`` (RMS
  error against the truth over the RMS of the reported SME; 1 when the SME is right).
- **T7**: ``recovery_rate`` (bursts whose matched positive zone has its frequency centroid
  within 10 % and its time centroid within 20 ms, the tolerances of REQ-213),
  ``freq_centroid_rel_error`` and ``time_centroid_error`` (of the matched zones),
  ``type_i_error`` (fraction of channels without burst where a zone is reported) and
  ``extra_zones_per_channel``.
"""

import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

from evoparam.errors import DataError
from evoparam.utils._optional import require
from evoparam.version import __version__

from ._canon import canonical_bytes, clean, sha256_bytes, sha256_file
from .suite import ERPCORE, load_suite, make_inputs
from .tasks import TASKS, BenchmarkValidationError, output_path, read_output, schema_id

__all__ = ["score", "score_outputs", "write_result", "read_result", "verify_result",
           "reference_scores", "RESULT_SCHEMA", "HEADLINE"]

RESULT_SCHEMA = "evoparam-benchmark-result/1"
# The metrics shown first in reports and on the leaderboard, per task.
HEADLINE = {
    "T1": ["auc", "f1", "precision", "recall", "fpr_at_0.05", "ks_uniform"],
    "T2": ["reported_fraction", "latency_mae", "latency_bias", "amplitude_rel_mae",
           "coverage_latency"],
    "T3": ["f1", "recall", "precision", "latency_mae", "amplitude_rel_mae",
           "false_rate_empty"],
    "T4": ["artifact_residual_rel", "artifact_residual_rms", "response_rmse_50ms",
           "n1_latency_error"],
    "T5": ["latency_mae", "latency_r", "amplitude_rel_mae", "amplitude_r",
           "jitter_sd_rel_error"],
    "T6": ["mean_amplitude.mae_vs_truth", "peak_latency.mae_vs_truth",
           "fractional_area_latency.mae_vs_truth", "mean_amplitude.mae_vs_definition",
           "mean_amplitude.sme_ratio"],
    "T7": ["recovery_rate", "freq_centroid_rel_error", "time_centroid_error",
           "type_i_error"],
}
T7_FREQ_TOL, T7_TIME_TOL = 0.10, 0.020


def _mean(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else None


def _ratio(a, b):
    return float(a / b) if b else None


def _rows_by(rows, keys, cell, allowed=None, what="channel"):
    out = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if allowed is not None and key not in allowed:
            raise BenchmarkValidationError(f"cell {cell}: row {dict(zip(keys, key))} is not a "
                                           f"{what} of the input")
        if key in out:
            raise BenchmarkValidationError(f"cell {cell}: two rows for "
                                           f"{dict(zip(keys, key))}")
        out[key] = row
    return out


def auc_score(labels, scores):
    """Area under the ROC curve (Mann-Whitney U, ties counted one half)."""

    labels = np.asarray(labels, bool)
    scores = np.asarray(scores, float)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


###################################################################################################
## T1, T2

def _t1(items):
    rows_all = []
    for ti, rows in items:
        d = ti.descriptor
        excluded = set(ti.truth["stim_channels"]) | set(ti.truth.get("excluded", ()))
        scored = [c for c in d["ch_names"] if c not in excluded]
        allowed = {(c,) for c in d["ch_names"]}
        by = _rows_by(rows, ("channel",), ti.cell, allowed)
        for row in rows:
            if row["site"] != d["site"]:
                raise BenchmarkValidationError(f"cell {ti.cell}: site {row['site']!r} is not "
                                               f"the input's site {d['site']!r}")
        missing = [c for c in scored if (c,) not in by]
        if missing:
            raise BenchmarkValidationError(f"cell {ti.cell}: no row for channel {missing[0]!r} "
                                           "(T1 needs one row per channel)")
        responders = set(ti.truth["responders"])
        for c in scored:
            row = by[(c,)]
            rows_all.append((ti.cell, c in responders, row))
    if all(r.get("score") is not None for _, _, r in rows_all):
        source = "score"
        values = [r["score"] for _, _, r in rows_all]
    elif all(r.get("p_value") is not None for _, _, r in rows_all):
        source = "p_value"
        values = [1.0 - r["p_value"] for _, _, r in rows_all]
    else:
        source = "decision"
        values = [float(r["responsive"]) for _, _, r in rows_all]
    labels = np.array([lab for _, lab, _ in rows_all], bool)
    decision = np.array([r["responsive"] == 1 for _, _, r in rows_all], bool)
    values = np.asarray(values, float)
    cells = [c for c, _, _ in rows_all]

    def counts(mask):
        lab, dec = labels[mask], decision[mask]
        tp, fp = int(np.sum(lab & dec)), int(np.sum(~lab & dec))
        fn, tn = int(np.sum(lab & ~dec)), int(np.sum(~lab & ~dec))
        return {"n_channels": int(mask.sum()), "n_responders": int(lab.sum()),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "auc": auc_score(lab, values[mask]),
                "f1": _ratio(2 * tp, 2 * tp + fp + fn), "precision": _ratio(tp, tp + fp),
                "recall": _ratio(tp, tp + fn), "specificity": _ratio(tn, tn + fp)}

    metrics = counts(np.ones(len(labels), bool))
    metrics["auc_source"] = source
    null_p = np.array([r["p_value"] for _, lab, r in rows_all
                       if not lab and r.get("p_value") is not None], float)
    metrics["n_null_p_values"] = int(null_p.size)
    for level in (0.01, 0.05, 0.1):
        metrics[f"fpr_at_{level:g}"] = float(np.mean(null_p <= level)) if null_p.size \
            else None
    if null_p.size:
        p = np.sort(null_p)
        n = p.size
        ecdf_hi, ecdf_lo = np.arange(1, n + 1) / n, np.arange(0, n) / n
        metrics["ks_uniform"] = float(max(np.max(ecdf_hi - p), np.max(p - ecdf_lo)))
    else:
        metrics["ks_uniform"] = None
    per_cell = {}
    for cell in dict.fromkeys(cells):
        mask = np.array([c == cell for c in cells])
        per_cell[cell] = {k: v for k, v in counts(mask).items()
                          if k in ("n_channels", "n_responders", "auc", "f1", "tp", "fp",
                                   "fn", "tn")}
    return metrics, per_cell


def _t2(items):
    pairs, per_cell = [], {}
    for ti, rows in items:
        d = ti.descriptor
        by = _rows_by(rows, ("channel",), ti.cell, {(c,) for c in d["ch_names"]})
        for row in rows:
            if row["site"] != d["site"]:
                raise BenchmarkValidationError(f"cell {ti.cell}: site {row['site']!r} is not "
                                               f"the input's site {d['site']!r}")
        cell_pairs = []
        for channel, n1 in sorted(ti.truth["n1"].items()):
            row = by.get((channel,))
            cell_pairs.append((n1, row))
        pairs += cell_pairs
        per_cell[ti.cell] = _t2_metrics(cell_pairs, short=True)
    return _t2_metrics(pairs), per_cell


def _covered(row, name, value):
    low, high = row.get(f"{name}_ci_low"), row.get(f"{name}_ci_high")
    if low is None or high is None:
        return None
    return float(low <= value <= high)


def _t2_metrics(pairs, short=False):
    reported = [(t, r) for t, r in pairs if r is not None and r.get("latency") is not None]
    lat = np.array([r["latency"] - t["latency"] for t, r in reported], float)
    with_amp = [(t, r) for t, r in reported if r.get("amplitude") is not None]
    amp = np.array([r["amplitude"] - t["amplitude"] for t, r in with_amp], float)
    rel = np.array([abs(r["amplitude"] - t["amplitude"]) / abs(t["amplitude"])
                    for t, r in with_amp if t["amplitude"] != 0], float)
    out = {"n_true": len(pairs), "n_reported": len(reported),
           "reported_fraction": _ratio(len(reported), len(pairs)),
           "latency_mae": float(np.mean(np.abs(lat))) if lat.size else None,
           "latency_bias": float(np.mean(lat)) if lat.size else None}
    if short:
        return out
    out.update(amplitude_mae=float(np.mean(np.abs(amp))) if amp.size else None,
               amplitude_bias=float(np.mean(amp)) if amp.size else None,
               amplitude_rel_mae=float(np.mean(rel)) if rel.size else None,
               coverage_latency=_mean([_covered(r, "latency", t["latency"])
                                       for t, r in reported]),
               coverage_amplitude=_mean([_covered(r, "amplitude", t["amplitude"])
                                         for t, r in with_amp]))
    return out


###################################################################################################
## T3

def _t3(items):
    from evoparam.sim import score_fits

    pd = require("pandas", extra="data", caller="evoparam.benchmark.score")
    truth_parts, fit_rows, per_cell = [], [], {}
    for ti, rows in items:
        names = ti.descriptor["ch_names"]
        _rows_by(rows, ("channel", "component"), ti.cell,
                 {(c, k) for c in names for k in range(10_000)}, what="channel")
        truth = ti.truth["components"].copy()
        truth["channel"] = [f"{ti.cell}|{c}" for c in truth["channel"]]
        truth_parts.append(truth)
        cell_rows = []
        for row in rows:
            amp = row["amplitude"]
            cell_rows.append({"channel": f"{ti.cell}|{row['channel']}",
                              "component": row["component"], "latency": row["latency"],
                              "amplitude": amp, "fwhm": np.nan if row["fwhm"] is None
                              else row["fwhm"],
                              "polarity": float(np.sign(amp)) if amp is not None else np.nan,
                              **{k: (np.nan if row.get(k) is None else row[k])
                                 for k in ("latency_ci_low", "latency_ci_high",
                                           "amplitude_ci_low", "amplitude_ci_high")}})
        reported = {r["channel"] for r in cell_rows}
        for c in names:
            if f"{ti.cell}|{c}" not in reported:
                cell_rows.append({"channel": f"{ti.cell}|{c}", "component": -1,
                                  "latency": np.nan, "amplitude": np.nan, "fwhm": np.nan,
                                  "polarity": np.nan})
        fit_rows += cell_rows
        one = score_fits(truth, pd.DataFrame(cell_rows), tolerance="fwhm/2").iloc[0]
        per_cell[ti.cell] = {k: _num(one[k]) for k in ("n_true", "n_fit", "n_matched", "f1")}
    truth = pd.concat(truth_parts, ignore_index=True)
    table = score_fits(truth, pd.DataFrame(fit_rows), tolerance="fwhm/2").iloc[0]
    keys = ("n_waveforms", "n_true", "n_fit", "n_matched", "n_false", "recall", "precision",
            "f1", "latency_mae", "latency_bias", "amplitude_rel_bias", "amplitude_rel_mae",
            "fwhm_rel_mae", "coverage_latency", "coverage_amplitude", "false_per_waveform",
            "false_rate_empty")
    return {k: _num(table[k]) for k in keys}, per_cell


def _num(value):
    value = float(value)
    return value if np.isfinite(value) else None


###################################################################################################
## T4

def _t4(items):
    sums = {"art": 0.0, "art_n": 0, "artifact": 0.0, "resp": 0.0, "resp_n": 0}
    shifts, per_cell = [], {}
    for ti, cleaned in items:
        d = ti.descriptor
        data_shape = ti.arrays["data"].shape
        if cleaned.shape != data_shape:
            raise BenchmarkValidationError(f"cell {ti.cell}: cleaned data of shape "
                                           f"{list(cleaned.shape)}, the input has "
                                           f"{list(data_shape)}")
        times = ti.times
        keep = [i for i, c in enumerate(d["ch_names"]) if c not in ti.truth["stim_channels"]]
        resp = [d["ch_names"].index(c) for c in ti.truth["responders"]]
        clean = ti.truth["clean"]
        residual = cleaned - clean
        win = (times >= d["artifact_window"][0]) & (times <= 0.05)
        part = residual[:, keep][:, :, win]
        art = ti.truth["artifact"][:, keep][:, :, win]
        cell_art, cell_n = float(np.sum(part ** 2)), part.size
        sums["art"] += cell_art
        sums["art_n"] += cell_n
        sums["artifact"] += float(np.sum(art ** 2))
        first = (times >= 0.0) & (times <= 0.05)
        cell_resp, cell_resp_n, cell_shift = 0.0, 0, []
        if resp:
            avg = residual[:, resp].mean(axis=0)[:, first]
            cell_resp, cell_resp_n = float(np.sum(avg ** 2)), avg.size
            n1 = (times >= 0.009) & (times <= 0.05)
            t_n1 = times[n1]
            got = cleaned[:, resp].mean(axis=0)[:, n1]
            ref = clean[:, resp].mean(axis=0)[:, n1]
            cell_shift = list(np.abs(t_n1[np.argmin(got, axis=1)]
                                     - t_n1[np.argmin(ref, axis=1)]))
        sums["resp"] += cell_resp
        sums["resp_n"] += cell_resp_n
        shifts += cell_shift
        per_cell[ti.cell] = {
            "artifact_residual_rms": float(np.sqrt(cell_art / cell_n)),
            "artifact_residual_rel": float(np.sqrt(cell_art / np.sum(art ** 2))),
            "response_rmse_50ms": float(np.sqrt(cell_resp / cell_resp_n)) if cell_resp_n
            else None,
            "n1_latency_error": float(np.mean(cell_shift)) if cell_shift else None}
    metrics = {
        "artifact_residual_rms": float(np.sqrt(sums["art"] / sums["art_n"])),
        "artifact_rms": float(np.sqrt(sums["artifact"] / sums["art_n"])),
        "artifact_residual_rel": float(np.sqrt(sums["art"] / sums["artifact"])),
        "response_rmse_50ms": float(np.sqrt(sums["resp"] / sums["resp_n"]))
        if sums["resp_n"] else None,
        "n1_latency_error": float(np.mean(shifts)) if shifts else None,
        "n_responding_channels": len(shifts)}
    return metrics, per_cell


###################################################################################################
## T5

def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _t5(items):
    acc = {"lat": [], "amp": [], "rel": [], "lat_r": [], "amp_r": [], "jit": [], "jit_rel": [],
           "n_true": 0}
    per_cell = {}
    for ti, rows in items:
        d = ti.descriptor
        targets = {t["channel"] for t in d["targets"]}
        allowed = {(c, k) for c in targets for k in range(d["n_trials"])}
        by = _rows_by(rows, ("channel", "trial"), ti.cell, allowed, what="target channel "
                                                                          "and trial")
        truth = ti.truth["trials"]
        cell_lat = []
        for channel in sorted(targets):
            sub = truth[truth["channel"] == channel]
            est_lat, true_lat, est_amp, true_amp = [], [], [], []
            for r in sub.itertuples():
                acc["n_true"] += 1
                row = by.get((channel, int(r.trial)))
                if row is None or row["latency"] is None:
                    continue
                est_lat.append(row["latency"])
                true_lat.append(r.latency)
                acc["lat"].append(row["latency"] - r.latency)
                cell_lat.append(abs(row["latency"] - r.latency))
                if row["amplitude"] is not None:
                    est_amp.append(row["amplitude"])
                    true_amp.append(r.amplitude)
                    acc["amp"].append(abs(row["amplitude"] - r.amplitude))
                    if r.amplitude != 0:
                        acc["rel"].append(abs(row["amplitude"] - r.amplitude) / abs(r.amplitude))
            acc["lat_r"].append(_pearson(est_lat, true_lat))
            acc["amp_r"].append(_pearson(est_amp, true_amp))
            if len(est_lat) >= 2:
                sd_est, sd_true = np.std(est_lat, ddof=1), np.std(true_lat, ddof=1)
                acc["jit"].append(abs(sd_est - sd_true))
                if sd_true > 0:
                    acc["jit_rel"].append(abs(sd_est - sd_true) / sd_true)
        per_cell[ti.cell] = {"latency_mae": float(np.mean(cell_lat)) if cell_lat else None,
                             "n_reported": len(cell_lat)}
    lat = np.asarray(acc["lat"], float)
    metrics = {"n_true": acc["n_true"], "n_reported": int(lat.size),
               "reported_fraction": _ratio(lat.size, acc["n_true"]),
               "latency_mae": float(np.mean(np.abs(lat))) if lat.size else None,
               "latency_bias": float(np.mean(lat)) if lat.size else None,
               "amplitude_mae": _mean(acc["amp"]), "amplitude_rel_mae": _mean(acc["rel"]),
               "latency_r": _mean(acc["lat_r"]), "amplitude_r": _mean(acc["amp_r"]),
               "jitter_sd_error": _mean(acc["jit"]),
               "jitter_sd_rel_error": _mean(acc["jit_rel"])}
    return metrics, per_cell


###################################################################################################
## T6

def reference_scores(times, y, definition):
    """The scores of a T6 definition on one waveform, computed with the reference
    implementation (:func:`evoparam.models.classical.classical_measures`).

    Returns dict measure -> value (s for latencies, data units otherwise).
    """

    from evoparam.models.classical import classical_measures

    polarity = definition["polarity"]
    scores = classical_measures(times, y, window=tuple(definition["window"]),
                                polarity=polarity,
                                measures=("peak", "mean", "fractional_area_latency"),
                                fraction=ERPCORE["fraction"], area=polarity,
                                n_neighbors=ERPCORE["n_neighbors"])
    return {"mean_amplitude": float(scores["mean_amplitude"]),
            "peak_amplitude": float(scores["peak_amplitude"]),
            "peak_latency": float(scores["peak_latency"]),
            "fractional_area_latency": float(scores["fractional_area_latency"])}


def _t6(items):
    acc = {m: {"def": [], "truth": [], "sme": [], "n": 0, "reported": 0}
           for m in ERPCORE["measures"]}
    per_cell = {}
    for ti, rows in items:
        d = ti.descriptor
        allowed = {(df["channel"], df["condition"], m) for df in d["definitions"]
                   for m in df["measures"]}
        by = _rows_by(rows, ("channel", "condition", "measure"), ti.cell, allowed,
                      what="definition")
        times = ti.times
        cell_err = []
        for df in d["definitions"]:
            index = d["ch_names"].index(df["channel"])
            on_data = reference_scores(times, ti.arrays["average"][index], df)
            on_truth = reference_scores(times, ti.truth["signal_average"][index], df)
            for measure in df["measures"]:
                acc[measure]["n"] += 1
                row = by.get((df["channel"], df["condition"], measure))
                if row is None or row["value"] is None:
                    continue
                acc[measure]["reported"] += 1
                acc[measure]["def"].append(row["value"] - on_data[measure])
                acc[measure]["truth"].append(row["value"] - on_truth[measure])
                cell_err.append(abs(row["value"] - on_truth[measure]))
                acc[measure]["sme"].append(row["sme"])
        per_cell[ti.cell] = {"n_reported": len(cell_err)}
    metrics = {}
    for measure, a in acc.items():
        dev, err = np.asarray(a["def"], float), np.asarray(a["truth"], float)
        sme = np.asarray([s for s in a["sme"] if s is not None], float)
        rms_sme = float(np.sqrt(np.mean(sme ** 2))) if sme.size else 0.0
        metrics[measure] = {
            "n": a["n"], "reported_fraction": _ratio(a["reported"], a["n"]),
            "mae_vs_definition": float(np.mean(np.abs(dev))) if dev.size else None,
            "max_abs_vs_definition": float(np.max(np.abs(dev))) if dev.size else None,
            "mae_vs_truth": float(np.mean(np.abs(err))) if err.size else None,
            "bias_vs_truth": float(np.mean(err)) if err.size else None,
            "sme_mean": float(np.mean(sme)) if sme.size else None,
            "sme_ratio": (float(np.sqrt(np.mean(err ** 2))) / rms_sme
                          if err.size and rms_sme > 0 else None)}
    return metrics, per_cell


###################################################################################################
## T7

def _t7(items):
    n_bursts = recovered = null_channels = null_hits = 0
    freq_err, time_err, extra = [], [], []
    per_cell = {}
    for ti, rows in items:
        names = ti.descriptor["ch_names"]
        _rows_by(rows, ("channel", "zone"), ti.cell,
                 {(c, k) for c in names for k in range(10_000)}, what="channel")
        cell_rec, cell_bursts = 0, 0
        for channel in names:
            bursts = [b for b in ti.truth["bursts"] if b["channel"] == channel]
            zones = [r for r in rows if r["channel"] == channel]
            if not bursts:
                null_channels += 1
                null_hits += int(bool(zones))
                continue
            positive = [z for z in zones if z["sign"] == 1]
            n_bursts += len(bursts)
            cell_bursts += len(bursts)
            matched = 0
            if positive:
                cost = np.array([[abs(z["freq_centroid"] / b["freq"] - 1.0) / T7_FREQ_TOL
                                  + abs(z["time_centroid"] - b["centre"]) / T7_TIME_TOL
                                  for z in positive] for b in bursts])
                for i, j in zip(*linear_sum_assignment(cost)):
                    b, z = bursts[i], positive[j]
                    f_err = abs(z["freq_centroid"] / b["freq"] - 1.0)
                    t_err = abs(z["time_centroid"] - b["centre"])
                    freq_err.append(f_err)
                    time_err.append(t_err)
                    matched += 1
                    if f_err <= T7_FREQ_TOL and t_err <= T7_TIME_TOL:
                        recovered += 1
                        cell_rec += 1
            extra.append(len(zones) - matched)
        per_cell[ti.cell] = {"n_bursts": cell_bursts, "recovered": cell_rec,
                             "n_zones": len(rows)}
    metrics = {"n_bursts": n_bursts, "recovered": recovered,
               "recovery_rate": _ratio(recovered, n_bursts),
               "freq_centroid_rel_error": _mean(freq_err),
               "time_centroid_error": _mean(time_err),
               "n_null_channels": null_channels,
               "type_i_error": _ratio(null_hits, null_channels),
               "extra_zones_per_channel": _mean(extra)}
    return metrics, per_cell


SCORERS = {"T1": _t1, "T2": _t2, "T3": _t3, "T4": _t4, "T5": _t5, "T6": _t6, "T7": _t7}


###################################################################################################
## Results

def _flatten(metrics, prefix=""):
    out = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            out.update(_flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def score_outputs(task, inputs, outputs, *, suite, method=None, missing=(),
                  output_hashes=None):
    """Score outputs already read and validated.

    Parameters
    ----------
    task : str
    inputs : list of TaskInput
        With their truth, in suite order.
    outputs : dict
        Cell id -> rows (or array for T4), for the cells that have an output.
    suite : Suite
    method : dict, optional
        ``name``, ``version``, ``citation`` and ``settings`` of the method.
    missing : sequence of str
        Cells without output (recorded, not scored).
    output_hashes : dict, optional
        Cell id -> SHA-256 of the output file.

    Returns
    -------
    dict
        The signed result document.
    """

    present = [ti for ti in inputs if ti.cell in outputs]
    if not present:
        raise DataError(f"{task}: no output to score.")
    metrics, per_cell = SCORERS[task]([(ti, outputs[ti.cell]) for ti in present])
    cells = []
    for ti in inputs:
        entry = {"cell": ti.cell, "split": ti.descriptor["split"],
                 "input_sha256": ti.sha256()}
        if ti.cell in outputs:
            entry["output_sha256"] = (output_hashes or {}).get(ti.cell)
            entry["metrics"] = per_cell.get(ti.cell, {})
        cells.append(entry)
    method = dict(method or {})
    method.setdefault("name", "unnamed")
    method.setdefault("version", "")
    method.setdefault("citation", "")
    method.setdefault("settings", {})
    document = {
        "schema": RESULT_SCHEMA, "task": task, "task_name": TASKS[task],
        "task_schema": schema_id(task),
        "suite": {"name": suite.name, "version": suite.version, "sha256": suite.sha256,
                  "hidden_seeds": suite.hidden_seeds is not None},
        "package": {"name": "evoparam", "version": __version__},
        "method": method,
        "metrics": metrics, "headline": {k: _flatten(metrics).get(k) for k in HEADLINE[task]},
        "cells": cells, "missing_cells": list(missing),
        "inputs_sha256": sha256_bytes(canonical_bytes([[c["cell"], c["input_sha256"]]
                                                       for c in cells])),
        "outputs_sha256": sha256_bytes(canonical_bytes([[c["cell"], c.get("output_sha256")]
                                                        for c in cells])),
    }
    document = clean(document)
    document["signature"] = {"algorithm": "sha256",
                             "covers": "canonical JSON of this document without 'signature' "
                                       "(keys sorted, no whitespace, UTF-8, NaN as null)",
                             "digest": sha256_bytes(canonical_bytes(document))}
    return document


def score(task, predictions, suite="core", *, method=None, hidden_seeds=None, split="public",
          allow_missing=False, inputs=None):
    """Score a method's outputs for one task on a suite.

    Parameters
    ----------
    task : str
        ``"T1"`` ... ``"T7"``.
    predictions : str
        Folder holding ``<cell>.tsv`` (``<cell>.npz`` for T4) for every cell of the task.
    suite : str or Suite
        A shipped suite name, a suite file, or a loaded suite.
    method : dict, optional
        ``name``, ``version``, ``citation``, ``settings`` recorded in the result.
    hidden_seeds : dict or str, optional
        The hidden seeds document; with it and ``split="all"`` the hidden cells are scored
        too.
    split : {"public", "hidden", "all"}
    allow_missing : bool
        Score the cells that have an output and list the others in ``missing_cells``;
        default: a missing output raises.
    inputs : list of TaskInput, optional
        The inputs with their truth (regenerated from the suite when not given).

    Returns
    -------
    dict
        The signed result (see the module docstring).

    Raises
    ------
    BenchmarkValidationError
        An output does not validate: the message names the file and its first violation.
    DataError
        A missing output (without ``allow_missing``).
    """

    if task not in TASKS:
        raise DataError(f"Unknown task {task!r}; tasks: {list(TASKS)}.")
    if not hasattr(suite, "cells"):
        suite = load_suite(suite, hidden_seeds=hidden_seeds)
    elif hidden_seeds is not None:
        suite = suite.with_hidden_seeds(hidden_seeds)
    if inputs is None:
        inputs = [make_inputs(cell, tasks=[task])[task]
                  for cell in suite.cells(task, split=split)]
    outputs, hashes, missing = {}, {}, []
    for ti in inputs:
        path = output_path(predictions, task, ti.cell)
        if not os.path.isfile(path):
            if not allow_missing:
                raise DataError(f"{path}: no output for cell {ti.cell} of task {task}.")
            missing.append(ti.cell)
            continue
        outputs[ti.cell] = read_output(task, path)
        hashes[ti.cell] = sha256_file(path)
    return score_outputs(task, inputs, outputs, suite=suite, method=method, missing=missing,
                         output_hashes=hashes)


def write_result(document, path):
    """Write a result document: sorted keys, one-space indent, UTF-8, final new line (the
    bytes depend only on the document)."""

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fobj:
        fobj.write(json.dumps(clean(document), sort_keys=True, indent=1, ensure_ascii=False,
                              allow_nan=False))
        fobj.write("\n")
    return path


def read_result(path):
    """Read a result document and check its signature (raises DataError when it does not
    match)."""

    with open(path, encoding="utf-8") as fobj:
        document = json.load(fobj)
    if document.get("schema") != RESULT_SCHEMA:
        raise DataError(f"{path}: not a benchmark result ({RESULT_SCHEMA}).")
    if not verify_result(document):
        raise DataError(f"{path}: the signature does not match the content (edited by hand?).")
    return document


def verify_result(document):
    """True when ``signature.digest`` is the SHA-256 of the document without it."""

    body = {k: v for k, v in document.items() if k != "signature"}
    return document.get("signature", {}).get("digest") == sha256_bytes(canonical_bytes(body))
