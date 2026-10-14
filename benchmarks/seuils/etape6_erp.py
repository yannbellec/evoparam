"""Step-6 benchmarks on ERP CORE (40 subjects x 7 components).

- ``measure``: the targeted measurement of the paradigm's component (``presets.targeted``),
  for each window margin (0 to 60 ms) and number of parts (1 to 4). On semi-simulated waves
  (one component of known shape, latency and width, with a neighbouring component of opposite
  polarity in half of them, added to the subject's plus-minus noise) and on the real waves
  (agreement with the published 50 % area latency; agreement between the odd and even trials).
  The same fits give the latency definitions (raw peak, raw 50 % area, model peak, model 50 %
  area) and the onset/offset at each fraction of the peak.
- ``erp_delta``, ``erp_reject_cancelling``: the ``erp`` preset itself on the stimulus-locked
  waves (semi-simulated with 0 to 3 components, plus-minus noise, real).
- ``trials``: the trial-by-trial fit (``fit_trials``, shift and scale) with each maximum shift,
  on single trials whose latency jitter is known (the subject's own trial residuals plus a
  component with a known shift per trial) and on the real single trials.
"""

import functools
import glob
import os
import pickle
import sys
import warnings
import zlib
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "validation"))
import erpcore  # noqa: E402
import etape6_common as C  # noqa: E402

N_SYNTH = 10
# FWHM range (s) of the injected component, per component
FWHM = {
    "P3": (0.10, 0.30),
    "N170": (0.02, 0.05),
    "N400": (0.10, 0.30),
    "MMN": (0.04, 0.12),
    "N2pc": (0.04, 0.10),
    "LRP": (0.05, 0.15),
    "ERN": (0.03, 0.08),
}
MARGINS = [0.0, 0.010, 0.020, 0.025, 0.030, 0.040, 0.050, 0.060]
PARTS = [1, 2, 3, 4]
PROPOSED = (0.025, 3)
FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
SHIFT_FACTORS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
JITTERS = [0.0, 0.005, 0.010, 0.020]
DELTAS = [
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    6.0,
    8.0,
    10.0,
    12.0,
    15.0,
    20.0,
    25.0,
    30.0,
    40.0,
    60.0,
    "calibrated",
]
STIMULUS_LOCKED = ("P3", "N170", "N400", "MMN", "N2pc")
TRIAL_COMPONENTS = ("P3", "N170", "N400", "MMN")


def keys():
    files = glob.glob(os.path.join(C.WORK, "erp", "*.npz"))
    return sorted(os.path.basename(p)[:-4] for p in files)


def subject_of(key):
    return key.split("_")[1]


@functools.lru_cache(maxsize=8)
def load(key):
    z = np.load(os.path.join(C.WORK, "erp", f"{key}.npz"))
    return {k: z[k] for k in z.files}


def _rng(*parts):
    return np.random.default_rng(zlib.crc32("/".join(map(str, parts)).encode()))


def targeted_preset(comp, times, margin, parts):
    from evoparam import presets

    d = erpcore.DESIGNS[comp]
    lo, hi = d["window"]
    pol = "positive" if d["polarity"] > 0 else "negative"
    p = presets.targeted(
        "erp", comp, d["window"], pol, margin=margin, n_components=parts, background="none"
    )
    fw = (max(lo - margin, times[0] + 0.01), min(hi + margin, times[-1] - 0.01))
    if lo < presets.get("erp").fit_window[0]:
        return presets.derive(p, name=p.name, fit_window=fw, baseline=(times[0], min(0.0, fw[0])))
    return presets.derive(p, name=p.name, fit_window=fw)


def fit_targeted(comp, times, wave, noise, margin, parts):
    from evoparam.models import ResponseModel, engine

    preset = targeted_preset(comp, times, margin, parts)
    model = ResponseModel(preset, random_state=0)
    ref = noise if noise is not None and np.all(np.isfinite(noise)) else None
    if ref is not None and not engine.use_plus_minus(model.settings(), times, wave):
        ref = None
    r = C.fit(times, wave, preset, noise_ref=ref, curve=True)
    r["parts"] = [c for c in r["components"] if c["label"] and c["label"].startswith(comp)]
    return r


def area_latency(times, wave, window, polarity, fraction=0.5):
    return erpcore.area_latency(times, wave, window, polarity, fraction)


def raw_peak(times, wave, window, polarity):
    inside = (times >= window[0]) & (times <= window[1])
    return float(times[inside][np.argmax(polarity * wave[inside])])


def bounds(times, curve, window, polarity, fraction, margin=0.025):
    """Onset and offset of a curve at ``fraction`` of its peak in the (widened) window; None
    when the curve is flat or the crossing is cut by the epoch edge."""
    y = polarity * np.asarray(curve, float)
    inside = np.flatnonzero((times >= window[0] - margin) & (times <= window[1] + margin))
    i = inside[np.argmax(y[inside])]
    if y[i] <= 0:
        return None
    level = fraction * y[i]
    a = i
    while a > 0 and y[a - 1] >= level:
        a -= 1
    b = i
    while b < len(y) - 1 and y[b + 1] >= level:
        b += 1
    if a == 0 or b == len(y) - 1:
        return None
    return float(times[a]), float(times[b])


def readings(comp, times, wave, r):
    """Latency by four definitions, and onset/offset of the model at each fraction."""
    d = erpcore.DESIGNS[comp]
    win, pol = d["window"], d["polarity"]
    out = {
        "raw_peak": raw_peak(times, wave, win, pol),
        "raw_area": area_latency(times, wave, win, pol),
        "model_peak": np.nan,
        "model_area": np.nan,
    }
    parts = r.get("parts") or []
    if parts and r.get("curve") is not None:
        out["model_peak"] = max(parts, key=lambda c: abs(c["area"]))["latency"]
        out["model_area"] = area_latency(times, r["curve"], win, pol)
        for f in FRACTIONS:
            b = bounds(times, r["curve"], win, pol, f)
            out[f"on_{f}"], out[f"off_{f}"] = b if b else (np.nan, np.nan)
    return out


@functools.lru_cache(maxsize=8)
def synthetic(key):
    """N_SYNTH semi-simulated waves of one subject and component (identical for every setting)."""
    comp = key.split("_")[0]
    d = load(key)
    t = d["times"]
    noise = d["ref"]
    if not np.all(np.isfinite(noise)):
        return []
    design = erpcore.DESIGNS[comp]
    lo, hi = design["window"]
    pol = design["polarity"]
    rng = _rng("erp", key)
    sd = float(np.std(noise[(t >= lo - 0.1) & (t <= hi + 0.1)]))
    out = []
    for _ in range(N_SYNTH):
        c = C.draw_component(rng, (lo, hi), FWHM[comp], pol)
        amp = sd * float(np.exp(rng.uniform(np.log(2.0), np.log(15.0))))
        curve = amp * pol * C.curve_of(c)(t)
        y = noise + curve
        neighbour = rng.random() < 0.5
        if neighbour:
            side = rng.choice([-1, 1])
            nb = C.draw_component(rng, (0, 0), FWHM[comp], -pol)
            gap = rng.uniform(0.04, 0.12)
            nb["lat_target"] = (lo - gap) if side < 0 else (hi + gap)
            y = y + rng.uniform(0.5, 1.5) * amp * (-pol) * C.curve_of(nb)(t)
        tf = np.arange(t[0], t[-1], 1e-4)
        truth_curve = pol * C.curve_of(c)(tf)
        truth = {
            "peak": C.true_features(c)["latency"],
            "area": area_latency(tf, truth_curve, (lo, hi), pol),
        }
        for f in FRACTIONS:
            b = bounds(tf, truth_curve, (lo, hi), pol, f)
            truth[f"on_{f}"], truth[f"off_{f}"] = b if b else (np.nan, np.nan)
        out.append({"y": y, "truth": truth, "neighbour": neighbour})
    return out


def task_measure(task):
    key, margin, parts = task
    comp = key.split("_")[0]
    d = load(key)
    t = d["times"]
    recs = []
    ref = d["ref"] if np.all(np.isfinite(d["ref"])) else None
    for i, s in enumerate(synthetic(key)):
        r = fit_targeted(comp, t, s["y"], ref, margin, parts)
        recs.append(
            {
                "kind": "synth",
                "i": i,
                "neighbour": s["neighbour"],
                **{f"truth_{k}": v for k, v in s["truth"].items()},
                **readings(comp, t, s["y"], r),
            }
        )
    for kind in ("real", "odd", "even"):
        wave = d["wave"] if kind == "real" else d[kind]
        r = fit_targeted(comp, t, wave, ref, margin, parts)
        recs.append(
            {
                "kind": kind,
                "i": 0,
                **readings(comp, t, wave, r),
                "published": area_latency(
                    t,
                    d["wave"],
                    erpcore.DESIGNS[comp]["window"],
                    erpcore.DESIGNS[comp]["polarity"],
                ),
            }
        )
    for r in recs:
        r.update(
            {
                "key": key,
                "component": comp,
                "subject": subject_of(key),
                "margin": margin,
                "parts": parts,
            }
        )
    return recs


# ---------------------------------------------------------------------------------------------
# erp preset on stimulus-locked waves: delta and reject_cancelling


@functools.lru_cache(maxsize=8)
def synthetic_full(key):
    """Semi-simulated waves for the full erp preset: 0 to 3 components in 50 to 700 ms."""
    d = load(key)
    t = d["times"]
    noise = d["ref"]
    if not np.all(np.isfinite(noise)):
        return []
    rng = _rng("erp-full", key)
    win = (t >= 0.05) & (t <= 0.7)
    sd = float(np.std(noise[win]))
    out = []
    slots = (
        ((0.08, 0.16), (0.02, 0.06)),
        ((0.18, 0.32), (0.04, 0.12)),
        ((0.35, 0.60), (0.10, 0.30)),
    )
    for _ in range(N_SYNTH):
        k = int(rng.integers(0, 4))
        which = sorted(rng.choice(len(slots), size=k, replace=False))
        y = noise.copy()
        truth = []
        for j in which:
            sign = int(rng.choice([-1, 1]))
            c = C.draw_component(rng, slots[j][0], slots[j][1], sign)
            amp = sd * float(np.exp(rng.uniform(np.log(2.0), np.log(15.0))))
            y += sign * amp * C.curve_of(c)(t)
            truth.append({**C.true_features(c), "sign": sign, "amplitude": amp})
        out.append({"y": y, "truth": truth})
    return out


FULL_SWEEPS = {
    "erp_delta": (DELTAS, 7, lambda v: {"delta": v}),
    "erp_reject_cancelling": ([False, True], 1, lambda v: {"reject_cancelling": v}),
}


def task_full(task):
    sweep, k, key = task
    values, _, make = FULL_SWEEPS[sweep]
    ov = make(values[k])
    d = load(key)
    t = d["times"]
    recs = []
    for i, s in enumerate(synthetic_full(key)):
        r = C.fit(t, s["y"], "erp", ov)
        sc = C.score_wave(s["truth"], [c for c in r["components"] if 0.05 <= c["latency"] <= 0.7])
        recs.append({"kind": "synth", "i": i, **sc})
    if np.all(np.isfinite(d["ref"])):
        r = C.fit(t, d["ref"], "erp", ov)
        recs.append({"kind": "null", "i": 0, "n_found": len(r["components"])})
    r = C.fit(t, d["wave"], "erp", ov)
    recs.append(
        {"kind": "real", "i": 0, "n_found": len(r["components"]), "r2": r.get("r2", np.nan)}
    )
    for r in recs:
        r.update({"sweep": sweep, "value_index": k, "key": key, "subject": subject_of(key)})
    return recs


# ---------------------------------------------------------------------------------------------
# Trial by trial


def _trialset(times, trials):
    from evoparam.data import TrialSet

    return TrialSet(
        data=np.asarray(trials, float)[:, None, :],
        times=times,
        sfreq=1 / np.median(np.diff(times)),
        ch_names=["ch"],
    )


def _trial_fit(ts, preset, comp_index=None, max_shift="auto"):
    """fit_trials with the targeted preset of the component (one part): the channel fit of the
    trial average holds the paradigm's component, which is then shifted and scaled per trial."""
    from evoparam.trials import fit_trials

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kw = {"components": [comp_index]} if comp_index is not None else {}
        return fit_trials(ts, preset=preset, by=(), random_state=0, max_shift=max_shift, **kw)


def _nearest(template, latency, sign):
    best, k_best = np.inf, None
    for k, c in enumerate(template.components):
        if np.sign(c.amplitude) == sign and abs(c.latency - latency) < best:
            best, k_best = abs(c.latency - latency), k
    return k_best


def task_trials(key):
    comp = key.split("_")[0]
    d = load(key)
    if "trials" not in d or len(d["trials"]) < 20:
        return []
    t = d["times"]
    trials = d["trials"].astype(float)
    design = erpcore.DESIGNS[comp]
    lo, hi = design["window"]
    pol = design["polarity"]
    recs = []
    preset = targeted_preset(comp, t, 0.025, 1)
    # real trials: the component of the channel fit nearest to the published latency
    try:
        base = _trial_fit(_trialset(t, trials), preset)
        tmpl = base.fits[0].template
        k = _nearest(tmpl, area_latency(t, trials.mean(0), (lo, hi), pol), pol)
    except Exception:  # noqa: BLE001
        k = None
    if k is not None:
        w = tmpl.components[k].fwhm
        for f in SHIFT_FACTORS:
            try:
                u = _trial_fit(_trialset(t, trials), preset, k, f * w).fits[0]
                sh = np.asarray(u.shift, float).reshape(len(u.shift), -1)[:, 0]
                recs.append(
                    {
                        "kind": "real",
                        "factor": f,
                        "jitter": float(np.std(sh)) * 1000,
                        "null": float(next(iter(u.jitter_sd_null.values()), np.nan)) * 1000,
                        "at_bound": float(np.mean(np.abs(sh) >= 0.98 * f * w)) * 100,
                        "fwhm": w * 1000,
                    }
                )
            except Exception:  # noqa: BLE001
                pass
    # semi-simulated: residual noise of each trial + a component with a known shift
    resid = trials - trials.mean(0)
    rng = _rng("trials", key)
    c = C.draw_component(rng, (lo, hi), FWHM[comp], pol, skew_prob=0.0)
    amp = float(np.abs(np.max(pol * trials.mean(0)[(t >= lo) & (t <= hi)]))) or float(
        np.std(resid)
    )
    for jit in JITTERS:
        shifts = rng.normal(0.0, jit, len(trials)) if jit > 0 else np.zeros(len(trials))
        y = np.stack(
            [
                resid[i]
                + pol * amp * C.curve_of({**c, "lat_target": c["lat_target"] + shifts[i]})(t)
                for i in range(len(trials))
            ]
        )
        ts = _trialset(t, y)
        try:
            tmpl = _trial_fit(ts, preset).fits[0].template
            k = _nearest(tmpl, c["lat_target"], pol)
        except Exception:  # noqa: BLE001
            k = None
        if k is None:
            continue
        w = tmpl.components[k].fwhm
        for f in SHIFT_FACTORS:
            try:
                u = _trial_fit(ts, preset, k, f * w).fits[0]
                sh = np.asarray(u.shift, float).reshape(len(u.shift), -1)[:, 0]
                err = (sh - (shifts - shifts.mean())) * 1000
                recs.append(
                    {
                        "kind": "synth",
                        "jitter_true": jit * 1000,
                        "factor": f,
                        "jitter": float(np.std(sh)) * 1000,
                        "rmse": float(np.sqrt(np.mean((err - err.mean()) ** 2))),
                        "at_bound": float(np.mean(np.abs(sh) >= 0.98 * f * w)) * 100,
                        "fwhm": w * 1000,
                    }
                )
            except Exception:  # noqa: BLE001
                pass
    for r in recs:
        r.update(
            {"key": key, "component": comp, "subject": subject_of(key), "n_trials": len(trials)}
        )
    return recs


def main(parts):
    out = os.path.join(C.WORK, "results")
    os.makedirs(out, exist_ok=True)
    ks = keys()
    with Pool(9) as pool:
        if "measure" in parts and not os.path.exists(os.path.join(out, "erp_measure.pkl")):
            tasks = [(k, m, p) for k in ks for m in MARGINS for p in PARTS]
            recs = [
                r for chunk in pool.imap_unordered(task_measure, tasks, chunksize=4) for r in chunk
            ]
            pickle.dump({"records": recs}, open(os.path.join(out, "erp_measure.pkl"), "wb"))
            print("erp measure", len(recs), flush=True)
        for sweep in FULL_SWEEPS:
            if sweep in parts and not os.path.exists(os.path.join(out, f"{sweep}.pkl")):
                values = FULL_SWEEPS[sweep][0]
                tasks = [
                    (sweep, i, k)
                    for k in ks
                    if k.split("_")[0] in STIMULUS_LOCKED
                    for i in range(len(values))
                ]
                recs = [
                    r
                    for chunk in pool.imap_unordered(task_full, tasks, chunksize=4)
                    for r in chunk
                ]
                pickle.dump(
                    {"values": values, "proposed": FULL_SWEEPS[sweep][1], "records": recs},
                    open(os.path.join(out, f"{sweep}.pkl"), "wb"),
                )
                print(sweep, len(recs), flush=True)
        if "trials" in parts and not os.path.exists(os.path.join(out, "erp_trials.pkl")):
            tk = [k for k in ks if k.split("_")[0] in TRIAL_COMPONENTS]
            recs = [r for chunk in pool.imap_unordered(task_trials, tk) for r in chunk]
            pickle.dump({"records": recs}, open(os.path.join(out, "erp_trials.pkl"), "wb"))
            print("erp trials", len(recs), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["measure", "erp_delta", "erp_reject_cancelling", "trials"])
