# SPDX-License-Identifier: Apache-2.0
"""Baseline methods of the benchmark (spec §8.3), run on the task inputs.

1. The published threshold rules of the registry of §5.6 (``vanblooijs2023``,
   ``hays2021``, ``keller2014``, ``silverstein2020``, ``ftract``, ``prime2020``; plus
   ``vanblooijs2023_code``, the reading of the published code), applied to the letter by
   :func:`evoparam.existence.test_response` with ``method="threshold"``. A field the paper
   does not state is set by the benchmark and recorded in the method's settings:
   ``vanblooijs2023`` baseline -1.0 to -0.1 s (the default baseline search epoch of the
   ER-detect tool, as the registry hints); ``hays2021`` and ``silverstein2020`` baseline
   -0.5 to -0.05 s; ``ftract`` ``k = 5`` (the middle of the published 3 to 7) without the
   optional peak-delay cut; ``prime2020`` ``k = 3`` and baseline -0.5 to -0.05 s.
   ``vanblooijs2023_code`` keeps its published baseline (-2.0 to -0.1 s), cut to the
   epoch (-1.0 s). T1 decision: the rule's; score: the rule's statistic; no p-value. T2,
   on the channels the rule detects: the latency of the rule's peak (for ``prime2020``,
   whose statistic is an RMS, the most negative sample of the average in its window), and
   the value of the average there minus its baseline mean.
2. ``projection``: the cross-projection test (§5.6 method 2) on 9-500 ms, baseline -0.5 to
   -0.05 s; T1 p-value: uncorrected; decision: Benjamini-Hochberg at 0.05 across the
   channels; T2 on the detected channels: the most negative sample of the average on
   9-50 ms (the method gives a duration, not a latency).
3. ``classical``: local peak (3 samples on each side) in the label windows of the preset
   (T3), in the target window of every trial (T5), and the reference definitions of
   ERP CORE with bootstrap SME (T6) (§5.5.6).
4. ``template``: single-trial alignment of the channel's fitted component curve (Woody
   style, ``fit_trials(mode="template")``, §5.7) (T5).
5. ``evoparam-preset``: the defaults of the preset named by the input's hint (``ccep`` or
   ``erp``): the existence test of the preset and ``evoparam.fit`` (T1, T2), ``evoparam.fit``
   of the average (T3), the preset's artifact removal (T4), ``fit_trials`` with the
   preset's trial model (T5), the fitted component (T6), and ``tf_fit`` (T7).
6. ``interpolate`` (linear interpolation of the artifact window) and ``none`` (the input
   unchanged), the two reference points of T4; ``tf-cluster`` (``tf_fit`` with
   cluster-mass permutations) for T7.

Every baseline is deterministic (``random_state=0``).
"""

import warnings
from dataclasses import dataclass, field

import numpy as np

from evoparam.errors import DataError
from evoparam.version import __version__

__all__ = ["Baseline", "BASELINES", "RULES", "get_baseline", "baselines_for"]

RULES = ("vanblooijs2023", "hays2021", "keller2014", "silverstein2020", "ftract", "prime2020")
RULE_OVERRIDES = {
    "vanblooijs2023": {"baseline": (-1.0, -0.1)},
    "vanblooijs2023_code": {},
    "hays2021": {"baseline": (-0.5, -0.05)},
    "keller2014": {},
    "silverstein2020": {"baseline": (-0.5, -0.05)},
    "ftract": {"k": 5.0},
    "prime2020": {"k": 3.0, "baseline": (-0.5, -0.05)},
}
CITATIONS = {
    "vanblooijs2023": "van Blooijs et al. 2023, Nat Neurosci (LIT-ccep-7, LIT-ccep-8)",
    "vanblooijs2023_code": "van Blooijs et al. 2023, Nat Neurosci, published code "
    "(LIT-ccep-7, LIT-ccep-30)",
    "hays2021": "Hays et al. 2021 (LIT-ccep-5)",
    "keller2014": "Keller et al. 2014, Brain (LIT-ccep-3)",
    "silverstein2020": "Silverstein et al. 2020 (LIT-ccep-6)",
    "ftract": "Lemarechal et al. 2022, Brain; Trebaul et al. 2018 (LIT-ccep-14, LIT-ccep-31)",
    "prime2020": "Prime et al. 2020 (LIT-ccep-47)",
    "projection": "cross-projection test, re-implemented from the published description "
    "(LIT-ccep-15, LIT-fitting-1)",
    "classical": "textbook ERP measures (LIT-erpmeas-1, LIT-erpmeas-3, LIT-erpmeas-41)",
    "template": "Woody-style template alignment (LIT-erpmeas-7), evoparam.trials",
    "evoparam-preset": "evoparam (this package), preset defaults",
    "interpolate": "linear interpolation of the artifact window (spec §5.3)",
    "none": "no artifact removal (reference point)",
    "tf-cluster": "evoparam.tf with cluster-mass permutations (spec §5.10)",
}


@dataclass(frozen=True)
class Baseline:
    """A baseline method: its name, the tasks it answers, its settings and its function.

    ``run(inputs)`` takes the task inputs of one cell (task id -> TaskInput) and returns
    task id -> output (rows, or the cleaned array for T4).
    """

    name: str
    tasks: tuple
    func: object = field(repr=False, compare=False)
    settings: dict = field(default_factory=dict, compare=False)
    description: str = ""

    def run(self, inputs):
        """Return the outputs of the cell for the tasks of ``inputs`` this baseline answers."""
        wanted = {t: ti for t, ti in inputs.items() if t in self.tasks}
        if not wanted:
            return {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = self.func(wanted, self.settings)
        return {t: out[t] for t in wanted}

    def info(self):
        """Return the ``method`` record of a result document."""
        settings = {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.settings.items()}
        return {
            "name": self.name,
            "version": __version__,
            "kind": "baseline",
            "citation": CITATIONS.get(self.name, ""),
            "description": self.description,
            "settings": settings,
        }


###################################################################################################
## Helpers


def _preset(ti, settings=None):
    from evoparam import presets

    name = (settings or {}).get("preset") or ti.descriptor.get("hints", {}).get("preset")
    if not name:
        raise DataError(f"cell {ti.cell}: no preset hint; give settings['preset'].")
    return presets.get(name)


def _baseline_corrected(ts, preset):
    from evoparam.prep import baseline_correct

    return (
        baseline_correct(ts, tuple(preset.baseline), check=False)
        if preset.baseline is not None
        else ts
    )


def _num(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _peak_rows(ti, sig, latency, avg, base_mean):
    """T2 rows.

    The latency and baseline-relative value of the average at the peak, for the detected channels.
    """
    times = ti.times
    rows = []
    for index, channel in enumerate(ti.descriptor["ch_names"]):
        lat = latency[index]
        if not sig[index] or lat is None or not np.isfinite(lat):
            continue
        k = int(np.argmin(np.abs(times - lat)))
        rows.append(
            {
                "site": ti.descriptor["site"],
                "channel": channel,
                "latency": float(times[k]),
                "amplitude": float(avg[index, k] - base_mean[index]),
            }
        )
    return rows


def _most_negative(times, avg, window):
    inside = (times >= window[0]) & (times <= window[1])
    t = times[inside]
    return t[np.argmin(avg[:, inside], axis=1)]


def _t1_rows(ti, significant, p_value, score):
    return [
        {
            "site": ti.descriptor["site"],
            "channel": channel,
            "responsive": int(bool(significant[i])),
            "p_value": _num(p_value[i]),
            "score": _num(score[i]),
        }
        for i, channel in enumerate(ti.descriptor["ch_names"])
    ]


###################################################################################################
## T1 / T2: published rules, projection, evoparam


def _rule(inputs, settings):
    from evoparam.existence import test_response

    ti = inputs.get("T1") or inputs.get("T2")
    ts = ti.to_trialset()
    result = test_response(
        ts,
        method="threshold",
        rule=settings["rule"],
        rule_overrides=dict(settings.get("overrides", {})),
        by=(),
        random_state=0,
    )
    spec = result.settings["rule_spec"]
    times = ti.times
    avg = ts.data.mean(axis=0)
    base = (times >= spec["baseline"][0]) & (times <= spec["baseline"][1])
    base_mean = avg[:, base].mean(axis=1)
    sig = np.asarray(result.significant, bool)
    latency = np.asarray(result.onset, float)
    if spec["statistic"] == "rms_z":
        latency = _most_negative(times, avg, spec["window"])
    out = {}
    if "T1" in inputs:
        out["T1"] = _t1_rows(inputs["T1"], sig, [None] * len(sig), result.statistic)
    if "T2" in inputs:
        out["T2"] = _peak_rows(inputs["T2"], sig, latency, avg, base_mean)
    return out


def _projection(inputs, settings):
    from evoparam.existence import test_response

    ti = inputs.get("T1") or inputs.get("T2")
    ts = ti.to_trialset()
    window, baseline = tuple(settings["window"]), tuple(settings["baseline"])
    result = test_response(
        ts,
        method="projection",
        window=window,
        baseline=baseline,
        by=(),
        alpha=0.05,
        correction="fdr_bh",
        random_state=0,
    )
    times = ti.times
    avg = ts.data.mean(axis=0)
    base = (times >= baseline[0]) & (times <= baseline[1])
    sig = np.asarray(result.significant, bool)
    p_raw = np.asarray(result.p_value_raw, float)
    out = {}
    if "T1" in inputs:
        out["T1"] = _t1_rows(inputs["T1"], sig, p_raw, 1.0 - p_raw)
    if "T2" in inputs:
        latency = _most_negative(times, avg, settings["n1_window"])
        out["T2"] = _peak_rows(inputs["T2"], sig, latency, avg, avg[:, base].mean(axis=1))
    return out


def _evoparam_detection(inputs, settings):
    from evoparam.existence import fit, test_response

    ti = inputs.get("T1") or inputs.get("T2")
    preset = _preset(ti, settings)
    ts = _baseline_corrected(ti.to_trialset(), preset)
    out = {}
    if "T1" in inputs:
        test = test_response(
            ts,
            method=preset.existence_method,
            window=tuple(preset.fit_window),
            baseline=tuple(preset.baseline),
            alpha=preset.existence_alpha,
            by=(),
            n_boot=0,
            random_state=0,
        )
        out["T1"] = _t1_rows(
            ti,
            np.asarray(test.significant, bool),
            np.asarray(test.p_value_raw, float),
            np.asarray(test.statistic, float),
        )
    if "T2" in inputs:
        group = fit(ts, preset.name, random_state=0)
        table = group.to_dataframe()
        rows = []
        for channel in ti.descriptor["ch_names"]:
            sub = table[
                (table["channel"] == channel)
                & (table["label"] == "N1")
                & (table["status"] == "ok")
            ]
            if not len(sub):
                continue
            r = sub.iloc[0]
            rows.append(
                {
                    "site": ti.descriptor["site"],
                    "channel": channel,
                    "latency": _num(r["latency"]),
                    "amplitude": _num(r["amplitude"]),
                    **{
                        k: _num(r.get(k))
                        for k in (
                            "latency_ci_low",
                            "latency_ci_high",
                            "amplitude_ci_low",
                            "amplitude_ci_high",
                        )
                    },
                }
            )
        out["T2"] = rows
    return out


###################################################################################################
## T3


def _evoparam_t3(inputs, settings):
    from evoparam.existence import fit

    ti = inputs["T3"]
    preset = _preset(ti, settings)
    group = fit(ti.to_evoked(), preset.name, random_state=0)
    table = group.to_dataframe()
    table = table[(table["status"] == "ok") & np.isfinite(table["latency"].astype(float))]
    rows = []
    for channel in ti.descriptor["ch_names"]:
        sub = table[table["channel"] == channel].sort_values("latency")
        for k, r in enumerate(sub.itertuples(index=False)):
            row = r._asdict()
            rows.append(
                {
                    "channel": channel,
                    "component": k,
                    "latency": _num(row["latency"]),
                    "amplitude": _num(row["amplitude"]),
                    "fwhm": _num(row["fwhm"]),
                    **{
                        c: _num(row.get(c))
                        for c in (
                            "latency_ci_low",
                            "latency_ci_high",
                            "amplitude_ci_low",
                            "amplitude_ci_high",
                        )
                    },
                }
            )
    return {"T3": rows}


POLARITY = {1: "positive", -1: "negative", 0: "absolute"}


def _local_peak(times, y, window, polarity):
    """Return (latency, amplitude) of the local peak in the window.

    None when there is no local peak (the simple-extremum fallback is not a component).
    """
    from evoparam.models.classical import classical_measures

    window = (max(window[0], times[0]), min(window[1], times[-1]))
    if window[1] - window[0] < 7 / (len(times) / (times[-1] - times[0])):
        return None
    scores = classical_measures(
        times, y, window=window, polarity=polarity, measures=("peak",), n_neighbors=3
    )
    if "fallback" in scores["definition"]["peak_latency"]:
        return None
    return float(scores["peak_latency"]), float(scores["peak_amplitude"])


def _classical_t3(inputs, settings):
    ti = inputs["T3"]
    preset = _preset(ti, settings)
    windows = {}
    for label, window in preset.component_labels.items():
        key = (tuple(window), preset.label_polarity.get(label, 0))
        windows.setdefault(key, label)
    times = ti.times
    rows = []
    for index, channel in enumerate(ti.descriptor["ch_names"]):
        y = ti.arrays["evoked"][index]
        found = []
        for (window, pol), _label in windows.items():
            peak = _local_peak(times, y, window, POLARITY[pol])
            if peak is not None and peak[0] not in [f[0] for f in found]:
                found.append(peak)
        for k, (lat, amp) in enumerate(sorted(found)):
            rows.append(
                {
                    "channel": channel,
                    "component": k,
                    "latency": lat,
                    "amplitude": amp,
                    "fwhm": None,
                }
            )
    return {"T3": rows}


###################################################################################################
## T4


def _evoparam_t4(inputs, settings):
    from evoparam.artifact import remove_artifact
    from evoparam.reports.pipeline import _artifact_baseline, artifact_steps

    ti = inputs["T4"]
    preset = _preset(ti, settings)
    ts = ti.to_trialset()
    window = tuple(ti.descriptor["artifact_window"])
    for step in artifact_steps(preset.name):
        step = dict(step, window=window)
        ts = remove_artifact(
            ts, baseline=_artifact_baseline(ts, preset, window), **step, random_state=0
        ).cleaned
    return {"T4": np.asarray(ts.data, float)}


def _interpolate_t4(inputs, _settings):
    from evoparam.artifact import remove_artifact

    ti = inputs["T4"]
    cleaned = remove_artifact(
        ti.to_trialset(),
        method="interpolate",
        window=tuple(ti.descriptor["artifact_window"]),
        interpolation="linear",
    ).cleaned
    return {"T4": np.asarray(cleaned.data, float)}


def _none_t4(inputs, _settings):
    return {"T4": np.array(inputs["T4"].arrays["data"], float)}


###################################################################################################
## T5


def _target_component(sub, target):
    """Return the fitted component of a channel answering a target.

    The one with the target's label, else the largest one whose latency is in the target window.
    """
    lab = sub[sub["label"] == target["label"]]
    if len(lab):
        return lab.iloc[0]["component"]
    window = target["window"]
    inside = sub[(sub["latency"] >= window[0]) & (sub["latency"] <= window[1])]
    if not len(inside):
        return None
    return inside.iloc[int(np.argmax(np.abs(inside["amplitude"].to_numpy(float))))]["component"]


def _trials_t5(inputs, settings):
    from evoparam.existence import fit
    from evoparam.trials import fit_trials

    ti = inputs["T5"]
    preset = _preset(ti, settings)
    ts = _baseline_corrected(ti.to_trialset(), preset)
    group = fit(ts, preset.name, random_state=0)
    mode = settings.get("mode") or preset.trial_mode or "shift_scale"
    trial_fits = fit_trials(ts, group, mode=mode, random_state=0)
    trials = trial_fits.to_dataframe()
    comps = group.to_dataframe()
    comps = comps[comps["status"] == "ok"]
    rows = []
    for target in ti.descriptor["targets"]:
        channel = target["channel"]
        k = _target_component(comps[comps["channel"] == channel], target)
        sub = (
            trials[(trials["channel"] == channel) & (trials["component"] == k)]
            if k is not None
            else trials.iloc[0:0]
        )
        by_trial = {int(r.trial): r for r in sub.itertuples()}
        for trial in range(ti.descriptor["n_trials"]):
            r = by_trial.get(trial)
            rows.append(
                {
                    "channel": channel,
                    "trial": trial,
                    "latency": _num(r.latency) if r is not None else None,
                    "amplitude": _num(r.amplitude) if r is not None else None,
                }
            )
    return {"T5": rows}


def _classical_t5(inputs, settings):
    ti = inputs["T5"]
    preset = _preset(ti, settings)
    ts = _baseline_corrected(ti.to_trialset(), preset)
    times = ti.times
    rows = []
    for target in ti.descriptor["targets"]:
        index = ti.descriptor["ch_names"].index(target["channel"])
        pol = POLARITY[preset.label_polarity.get(target["label"], 0)]
        for trial in range(ti.descriptor["n_trials"]):
            peak = _local_peak(times, ts.data[trial, index], target["window"], pol)
            rows.append(
                {
                    "channel": target["channel"],
                    "trial": trial,
                    "latency": None if peak is None else peak[0],
                    "amplitude": None if peak is None else peak[1],
                }
            )
    return {"T5": rows}


###################################################################################################
## T6


def _classical_t6(inputs, settings):
    from evoparam.models.classical import classical_measures

    from .suite import ERPCORE

    ti = inputs["T6"]
    times = ti.times
    rows = []
    for df in ti.descriptor["definitions"]:
        index = ti.descriptor["ch_names"].index(df["channel"])
        scores = classical_measures(
            times,
            ti.arrays["average"][index],
            window=tuple(df["window"]),
            polarity=df["polarity"],
            measures=("peak", "mean", "fractional_area_latency"),
            fraction=ERPCORE["fraction"],
            area=df["polarity"],
            n_neighbors=ERPCORE["n_neighbors"],
            trials=ti.arrays["data"][:, index],
            n_bootstrap=settings.get("n_bootstrap", 1000),
            random_state=0,
        )
        for measure in df["measures"]:
            rows.append(
                {
                    "channel": df["channel"],
                    "condition": df["condition"],
                    "measure": measure,
                    "value": _num(scores[measure]),
                    "sme": _num(scores["sme"].get(measure)),
                }
            )
    return {"T6": rows}


def _area_latency(times, curve, window, polarity, fraction=0.5):
    inside = (times >= window[0]) & (times <= window[1])
    t, y = times[inside], curve[inside]
    y = np.clip(y if polarity == "positive" else -y, 0, None)
    area = np.concatenate([[0.0], np.cumsum((y[1:] + y[:-1]) / 2 * np.diff(t))])
    if area[-1] <= 0:
        return None
    return float(np.interp(fraction * area[-1], area, t))


def _evoparam_t6(inputs, settings):
    from evoparam.existence import fit
    from evoparam.models import evaluate_fit

    ti = inputs["T6"]
    preset = _preset(ti, settings)
    times = ti.times
    ts = _baseline_corrected(ti.to_trialset(), preset)
    group = fit(ts, preset.name, random_state=0)
    rows = []
    for df in ti.descriptor["definitions"]:
        channel_fit = group.get(df["channel"])
        sign = 1 if df["polarity"] == "positive" else -1
        window = tuple(df["window"])
        best, best_k = None, None
        for k, comp in enumerate(channel_fit.components if channel_fit.status == "ok" else []):
            if comp.polarity != sign or not window[0] <= comp.latency <= window[1]:
                continue
            if best is None or abs(comp.amplitude) > abs(best.amplitude):
                best, best_k = comp, k
        values, sme = {}, {}
        if best is not None:
            curve = evaluate_fit(channel_fit, times, components=[best_k], background=False)
            inside = (times >= window[0]) & (times <= window[1])
            values = {
                "peak_latency": best.latency,
                "peak_amplitude": best.amplitude,
                "mean_amplitude": float(np.mean(curve[inside])),
                "fractional_area_latency": _area_latency(times, curve, window, df["polarity"]),
            }
            if best.ci is not None:
                for measure, key in (("peak_latency", "latency"), ("peak_amplitude", "amplitude")):
                    if key in best.ci:
                        low, high = best.ci[key]
                        sme[measure] = (high - low) / (2 * 1.959963984540054)
        for measure in df["measures"]:
            rows.append(
                {
                    "channel": df["channel"],
                    "condition": df["condition"],
                    "measure": measure,
                    "value": _num(values.get(measure)),
                    "sme": _num(sme.get(measure)),
                }
            )
    return {"T6": rows}


###################################################################################################
## T7


def _tf_t7(inputs, settings):
    from evoparam.tf import tf_fit

    ti = inputs["T7"]
    ts = ti.to_trialset()
    hints = ti.descriptor.get("hints", {})
    rows = []
    for channel in ti.descriptor["ch_names"]:
        tf = tf_fit(
            ts,
            picks=[channel],
            window=tuple(hints.get("window", (0.0, 1.0))),
            baseline=tuple(hints.get("baseline", (-0.5, -0.1))),
            freqs=tuple(settings.get("freqs", (4.0, 250.0, 40))),
            stats=settings.get("stats", "fdr"),
            n_permutations=settings.get("n_permutations", 1000),
            by=(),
            random_state=0,
        )
        zones = tf.zones if tf.zones is not None else []
        for k, z in enumerate(zones.itertuples() if len(zones) else []):
            rows.append(
                {
                    "channel": channel,
                    "zone": k,
                    "sign": int(z.sign),
                    "freq_centroid": float(z.freq_centroid),
                    "time_centroid": float(z.time_centroid),
                    "f_low": _num(z.f_low),
                    "f_high": _num(z.f_high),
                    "t_start": _num(z.t_start),
                    "t_end": _num(z.t_end),
                    "p_value": _num(min(max(float(z.p_value), 0.0), 1.0)),
                }
            )
    return {"T7": rows}


###################################################################################################
## Registry


def _evoparam_preset(inputs, settings):
    out = {}
    detection = {t: inputs[t] for t in ("T1", "T2") if t in inputs}
    if detection:
        out.update(_evoparam_detection(detection, settings))
    for task, func in (
        ("T3", _evoparam_t3),
        ("T4", _evoparam_t4),
        ("T5", _trials_t5),
        ("T6", _evoparam_t6),
        ("T7", _tf_t7),
    ):
        if task in inputs:
            out.update(func({task: inputs[task]}, settings))
    return out


def _classical(inputs, settings):
    out = {}
    for task, func in (("T3", _classical_t3), ("T5", _classical_t5), ("T6", _classical_t6)):
        if task in inputs:
            out.update(func({task: inputs[task]}, settings))
    return out


def _make():
    out = {}
    for name in RULES + ("vanblooijs2023_code",):
        out[name] = Baseline(
            name=name,
            tasks=("T1", "T2"),
            func=_rule,
            settings={"rule": name, "overrides": RULE_OVERRIDES[name]},
            description=f"published threshold rule '{name}' of the "
            "registry of spec §5.6, applied to the letter",
        )
    out["projection"] = Baseline(
        name="projection",
        tasks=("T1", "T2"),
        func=_projection,
        settings={
            "window": (0.009, 0.5),
            "baseline": (-0.5, -0.05),
            "n1_window": (0.009, 0.05),
            "alpha": 0.05,
            "correction": "fdr_bh",
        },
        description="cross-projection response test (spec §5.6 method 2)",
    )
    out["evoparam-preset"] = Baseline(
        name="evoparam-preset",
        tasks=("T1", "T2", "T3", "T4", "T5", "T6", "T7"),
        func=_evoparam_preset,
        settings={"preset": None},
        description="evoparam with the defaults of the preset named by the input",
    )
    out["classical"] = Baseline(
        name="classical",
        tasks=("T3", "T5", "T6"),
        func=_classical,
        settings={"n_bootstrap": 1000},
        description="classical measures (spec §5.5.6)",
    )
    out["template"] = Baseline(
        name="template",
        tasks=("T5",),
        func=_trials_t5,
        settings={"mode": "template"},
        description="template (Woody-style) single-trial alignment (spec §5.7)",
    )
    out["interpolate"] = Baseline(
        name="interpolate",
        tasks=("T4",),
        func=_interpolate_t4,
        description="linear interpolation of the artifact window",
    )
    out["none"] = Baseline(
        name="none", tasks=("T4",), func=_none_t4, description="no artifact removal"
    )
    out["tf-cluster"] = Baseline(
        name="tf-cluster",
        tasks=("T7",),
        func=_tf_t7,
        settings={"stats": "cluster", "n_permutations": 1000},
        description="evoparam.tf with cluster-mass permutations",
    )
    return out


BASELINES = _make()


def get_baseline(name):
    """Return a baseline by name."""
    if name not in BASELINES:
        raise DataError(f"Unknown baseline {name!r}; baselines: {sorted(BASELINES)}.")
    return BASELINES[name]


def baselines_for(task):
    """Names of the baselines that answer ``task``."""
    return [name for name, b in BASELINES.items() if task in b.tasks]
