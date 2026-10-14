"""Shared pieces of the step-6 threshold benchmarks (version 2).

Step 6 of the pipeline describes the response: it fits components (bumps) and a background
to an average and reads latency, amplitude, width, onset and offset from them. Each benchmark
here varies ONE setting of that step, all others at the preset value, and measures:

- **known truth**: components of known shape, latency, amplitude and width added to real noise
  (the plus-minus average of the same subject's trials: the trials averaged with alternating
  signs, which cancels the response and keeps the noise of the average; Schimmel 1967);
- **real noise alone**: the plus-minus average itself; every component found there is false;
- **real data**: the trial average, where the truth is unknown, to show the effect in practice.

Statistics: the subject is the unit. Each metric is first averaged within a subject, then
summarised over subjects with a 95 % bootstrap confidence interval (10 000 resamples of the
subjects). Each value is compared with the proposed one by a paired Wilcoxon signed-rank test
over subjects, Holm-corrected across the values of a figure. The subjects are split once
into two halves (seed SPLIT_SEED): the best value is chosen on the development half, and every
number shown is computed on the confirmation half only.
"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
from scipy import special, stats

SPLIT_SEED = 2026
N_BOOT = 10_000
CACHE = os.path.expanduser("~/.cache/evoparam")
WORK = os.path.join(CACHE, "seuils")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "figures", "v2", "etape-6")
DATASETS = {
    "ds004080": "OpenNeuro ds004080 v1.2.4 (CCEP, UMC Utrecht)",
    "erpcore": "ERP CORE (Kappenman et al. 2021), OSF 9f5w7, the published epoched files",
    "ds002094": "OpenNeuro ds002094 (TMS-EEG), published TESA cleaning as in the TEP validation",
}


# ------------------------------------------------------------------------------------------
# Subject split


def split(subjects, seed=SPLIT_SEED):
    """Return (development, confirmation): a fixed random half of the sorted subjects each."""
    subjects = sorted(subjects)
    order = np.random.default_rng(seed).permutation(len(subjects))
    half = len(subjects) // 2
    dev = sorted(subjects[i] for i in order[:half])
    conf = sorted(subjects[i] for i in order[half:])
    return dev, conf


# ------------------------------------------------------------------------------------------
# Shapes of the injected components (written here, independent of the package's own code)


def gaussian(t, mu, sigma):
    return np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def skewed(t, xi, omega, alpha):
    """Skew-normal bump, peak scaled to 1."""
    z = (t - xi) / omega
    y = np.exp(-0.5 * z**2) * (1 + special.erf(alpha * z / np.sqrt(2)))
    peak = y.max()
    return y / peak if peak > 0 else y


def shape_features(t, y, fraction=0.1):
    """Peak time, FWHM, onset and offset (first and last time at ``fraction`` of the peak),
    and 50 % area latency of a positive unit bump sampled on a fine grid."""
    i = int(np.argmax(y))
    peak = y[i]

    def edges(level):
        above = np.flatnonzero(y >= level * peak)
        return t[above[0]], t[above[-1]]

    half = edges(0.5)
    on, off = edges(fraction)
    area = np.cumsum(y)
    return {
        "latency": t[i],
        "fwhm": half[1] - half[0],
        "onset": on,
        "offset": off,
        "area_latency": t[np.searchsorted(area, 0.5 * area[-1])],
    }


def draw_component(rng, lat_range, fwhm_range, sign, skew_prob=0.5):
    """Random component: shape, parameters and its true features (times in s)."""
    fwhm = rng.uniform(*fwhm_range)
    lat = rng.uniform(*lat_range)
    if rng.random() < skew_prob:
        alpha = rng.choice([-1, 1]) * rng.uniform(1.0, 4.0)
        kind = "skewed"
    else:
        alpha = 0.0
        kind = "gaussian"
    return {"kind": kind, "lat_target": lat, "fwhm_target": fwhm, "alpha": alpha, "sign": sign}


def _unit_skew(alpha):
    """Peak position and FWHM of the skew-normal with xi = 0, omega = 1, and its peak value."""
    z = np.arange(-8, 8, 1e-4)
    y = np.exp(-0.5 * z**2) * (1 + special.erf(alpha * z / np.sqrt(2)))
    f = shape_features(z, y)
    return f["latency"], f["fwhm"], y.max()


def curve_of(comp):
    """Unit-peak function of time: peak at the target latency, FWHM equal to the target."""
    lat, fwhm = comp["lat_target"], comp["fwhm_target"]
    if comp["kind"] == "gaussian":
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        return lambda t: gaussian(np.asarray(t, float), lat, sigma)
    z_peak, w_unit, top = _unit_skew(comp["alpha"])
    omega = fwhm / w_unit
    xi = lat - z_peak * omega
    alpha = comp["alpha"]

    def f(t):
        z = (np.asarray(t, float) - xi) / omega
        return np.exp(-0.5 * z**2) * (1 + special.erf(alpha * z / np.sqrt(2))) / top

    return f


def true_features(comp, fraction=0.1):
    tf = np.arange(comp["lat_target"] - 1.0, comp["lat_target"] + 1.0, 1e-5)
    return shape_features(tf, curve_of(comp)(tf), fraction)


# ------------------------------------------------------------------------------------------
# Matching fitted components to the truth


def match(truth, found, min_tol=0.005):
    """Greedy one-to-one match by latency, same sign, tolerance max(5 ms, FWHM/2) of the true
    component. Returns the list of (truth index, found index)."""
    pairs = []
    for i, tr in enumerate(truth):
        tol = max(min_tol, 0.5 * tr["fwhm"])
        for j, fo in enumerate(found):
            if (
                np.sign(fo["amplitude"]) == tr["sign"]
                and abs(fo["latency"] - tr["latency"]) <= tol
            ):
                pairs.append((abs(fo["latency"] - tr["latency"]), i, j))
    pairs.sort()
    used_t, used_f, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_t and j not in used_f:
            used_t.add(i)
            used_f.add(j)
            out.append((i, j))
    return out


def score_wave(truth, found, window=None):
    """Per-wave counts and errors against the known truth."""
    if window is not None:
        found = [c for c in found if window[0] <= c["latency"] <= window[1]]
    pairs = match(truth, found)
    rec = {
        "n_true": len(truth),
        "n_found": len(found),
        "tp": len(pairs),
        "fp": len(found) - len(pairs),
        "fn": len(truth) - len(pairs),
        "lat_err": [],
        "amp_err": [],
        "fwhm_err": [],
        "onset_err": [],
        "offset_err": [],
    }
    for i, j in pairs:
        tr, fo = truth[i], found[j]
        rec["lat_err"].append(abs(fo["latency"] - tr["latency"]) * 1000)
        rec["amp_err"].append((abs(fo["amplitude"]) - tr["amplitude"]) / tr["amplitude"] * 100)
        rec["fwhm_err"].append((fo["fwhm"] - tr["fwhm"]) * 1000)
        rec["onset_err"].append(abs(fo["onset"] - tr["onset"]) * 1000)
        rec["offset_err"].append(abs(fo["offset"] - tr["offset"]) * 1000)
    return rec


# ------------------------------------------------------------------------------------------
# Fitting


def components(result):
    return [
        {
            "shape": c.shape,
            "latency": float(c.latency),
            "amplitude": float(c.amplitude),
            "fwhm": float(c.fwhm),
            "onset": float(c.onset),
            "offset": float(c.offset),
            "label": c.label,
            "area": float(c.area),
        }
        for c in result.components
    ]


def fit(times, y, preset, overrides=None, baseline=None, window=None, noise_ref=None, curve=False):
    """Fit one waveform; returns a dict (never raises)."""
    from evoparam.models import ResponseModel, evaluate_fit

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = ResponseModel(preset, random_state=0, **(overrides or {}))
            res = model.fit_waveform(
                times, y, window=window, baseline=baseline, noise_ref=noise_ref
            )
        except Exception as err:  # noqa: BLE001  (a failed fit is a recorded outcome)
            return {"error": repr(err), "components": [], "seconds": time.perf_counter() - t0}
    out = {
        "components": components(res),
        "status": res.status,
        "seconds": time.perf_counter() - t0,
        "r2": float(getattr(res.gof, "r2", np.nan)) if res.gof is not None else np.nan,
    }
    if curve:
        try:
            out["curve"] = np.asarray(evaluate_fit(res, times), float)
        except Exception:  # noqa: BLE001
            out["curve"] = None
    return out


# ------------------------------------------------------------------------------------------
# Statistics over subjects


def boot_ci(values, n_boot=N_BOOT, seed=0, level=0.95):
    """Mean over subjects and its percentile bootstrap CI (subjects resampled)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(v.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def holm(pvalues):
    p = np.asarray(pvalues, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    order = np.argsort(q)
    m = q.size
    adj = np.empty(m)
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, (m - rank) * q[k])
        adj[k] = min(1.0, running)
    out[ok] = adj
    return out


def paired_vs(matrix, ref):
    """``matrix``: subjects x values (NaN allowed). For each value: mean paired difference to
    column ``ref`` with its bootstrap CI, Wilcoxon p (two-sided), Holm-corrected p."""
    m = np.asarray(matrix, float)
    rows = []
    for k in range(m.shape[1]):
        d = m[:, k] - m[:, ref]
        d = d[np.isfinite(d)]
        if k == ref or d.size < 2 or np.allclose(d, 0):
            p = np.nan if k == ref else 1.0
        else:
            p = float(stats.wilcoxon(d, zero_method="zsplit").pvalue)
        mean, lo, hi = boot_ci(d) if d.size else (np.nan,) * 3
        rows.append({"diff": mean, "diff_lo": lo, "diff_hi": hi, "p": p, "n": int(d.size)})
    ph = holm([r["p"] for r in rows])
    for r, q in zip(rows, ph, strict=True):
        r["p_holm"] = float(q) if np.isfinite(q) else np.nan
    return rows


def environment():
    import platform

    import evoparam

    return {
        "evoparam": evoparam.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "split_seed": SPLIT_SEED,
        "n_boot": N_BOOT,
        "datasets": DATASETS,
    }
