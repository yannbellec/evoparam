"""Step-6 threshold figures ("Describe the response") on real data, version 1.

CCEP: ds004080 (6 patients, 7 runs) and ds004977 (4 patients), trial averages, the 10 most
responsive channels of each run and 10 drawn at random (seed 0). ERP CORE: the published
difference waves of the 7 components, 40 participants. TEP: ds002094, 20 subjects, the
published (TESA) cleaning, channel of largest response from 38 to 140 ms.
"""

import os
import sys
import warnings
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/evoparam/validation"))
import common  # noqa: E402

OUT = os.path.expanduser("~/evoparam-figures-seuils/etape-6")
os.makedirs(OUT, exist_ok=True)
CAPTIONS = {}
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 150,
    }
)
PROPOSED = "#c0392b"
OTHER = "#7f8c8d"
POOL = None


def save(fig, name, caption):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name + ".png"), metadata={"Software": None})
    plt.close(fig)
    CAPTIONS[name] = caption
    print("saved", name, flush=True)


def run(jobs):
    return POOL.map(common.fit_job, jobs, chunksize=4)


def sem(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def colours(labels, proposed):
    return [PROPOSED if lab == proposed else OTHER for lab in labels]


# ------------------------------------------------------------------------------------------
# CCEP sweeps
WAVES = common.ccep_waves()
N_CCEP = len({w["subject"] for w in WAVES})
STRONG = np.array([w["strong"] for w in WAVES])


def ccep_sweep(overrides_list, curve=False):
    jobs = []
    for s, ov in enumerate(overrides_list):
        for i, w in enumerate(WAVES):
            jobs.append(
                {
                    "key": (s, i),
                    "times": w["times"],
                    "wave": w["wave"],
                    "preset": "ccep",
                    "overrides": ov,
                    "baseline": w["baseline"],
                    "curve": curve,
                }
            )
    res = run(jobs)
    out = [[None] * len(WAVES) for _ in overrides_list]
    for r in res:
        s, i = r["key"]
        out[s][i] = r
    return out


def n_comp(r):
    return len(r.get("components", [])) if r and "error" not in r else np.nan


def has(r, label):
    return bool(r) and any(c["label"] == label for c in r.get("components", []))


def r2(r):
    return r.get("r2", np.nan) if r else np.nan


def bars(ax, labels, values, errors, proposed, ylabel, title):
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=errors, color=colours(labels, proposed), capsize=3)
    ax.set_xticks(x, labels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)


def example(ax, results, labels, proposed, index, title):
    w = WAVES[index]
    t = w["times"] * 1000
    keep = (t > 9) & (t < 400)
    ax.plot(t[keep], w["wave"][keep], color="black", lw=1, label="signal")
    styles = ["-", "--", ":", "-.", (0, (1, 1)), (0, (5, 1))]
    for k, (lab, rs) in enumerate(zip(labels, results, strict=True)):
        r = rs[index]
        if r and "curve" in r:
            ax.plot(
                t[keep],
                r["curve"][keep],
                ls=styles[k % len(styles)],
                lw=1.4,
                color=PROPOSED if lab == proposed else None,
                label=f"{lab} ({n_comp(r)} components)",
            )
    ax.set_xlim(0, 400)
    ax.set_xlabel("time after the pulse (ms)")
    ax.set_ylabel("µV")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, frameon=False)


def summary_panels(results, labels, proposed, name, title, caption, extra=None):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for sel, tag, ls in ((STRONG, "responsive channels", "-"), (~STRONG, "random channels", "--")):
        pass
    nb_s = [
        np.nanmean([n_comp(r) for r, s in zip(rs, STRONG, strict=True) if s]) for rs in results
    ]
    nb_se = [sem([n_comp(r) for r, s in zip(rs, STRONG, strict=True) if s]) for rs in results]
    nb_r = [
        np.nanmean([n_comp(r) for r, s in zip(rs, STRONG, strict=True) if not s]) for rs in results
    ]
    x = np.arange(len(labels))
    axes[0].bar(
        x - 0.2,
        nb_s,
        0.4,
        yerr=nb_se,
        color=colours(labels, proposed),
        capsize=3,
        label="10 most responsive channels per recording",
    )
    axes[0].bar(
        x + 0.2,
        nb_r,
        0.4,
        color="none",
        edgecolor=colours(labels, proposed),
        hatch="//",
        label="10 random channels per recording",
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("mean number of components")
    axes[0].set_title("Components kept by the model", fontsize=10)
    axes[0].legend(fontsize=7, frameon=False)
    vals = [[r2(r) for r, s in zip(rs, STRONG, strict=True) if s] for rs in results]
    bp = axes[1].boxplot(
        [np.array(v)[np.isfinite(v)] for v in vals],
        tick_labels=labels,
        showfliers=False,
        patch_artist=True,
    )
    for patch, col in zip(bp["boxes"], colours(labels, proposed), strict=True):
        patch.set_facecolor(col)
        patch.set_alpha(0.5)
    axes[1].set_ylabel("R² (variance explained)")
    axes[1].set_title("Goodness of fit (responsive channels)", fontsize=10)
    best = int(
        np.argmax(
            [
                np.nanmax(np.abs(w["wave"][(w["times"] > 0.01) & (w["times"] < 0.1)]))
                / (np.std(w["wave"][w["times"] < -0.02]) + 1e-9)
                if w["strong"]
                else 0
                for w in WAVES
            ]
        )
    )
    example(axes[2], results, labels, proposed, best, "Example: one CCEP channel")
    if extra:
        extra(fig, axes)
    fig.suptitle(title, fontsize=12)
    save(fig, name, caption)


def fig_shapes():
    labels = ["Gaussian\nonly", "skewed\nonly", "both"]
    ovs = [{"shapes": ("gaussian",)}, {"shapes": ("skewed_gaussian",)}, {}]
    res = ccep_sweep(ovs, curve=True)
    summary_panels(
        res,
        labels,
        "both",
        "6-01-shapes",
        "Allowed component shapes (CCEP)",
        f"Three sets of allowed shapes on {len(WAVES)} CCEP channels from {N_CCEP} patients (ds004080, ds004977). "
        "Shows the number of components kept, the variance explained (R²) and an example.",
    )


def fig_background():
    labels = ["none", "constant", "linear", "exponential"]
    ovs = [{"background": "none"}, {"background": "constant"}, {"background": "linear"}, {}]
    res = ccep_sweep(ovs, curve=True)
    summary_panels(
        res,
        labels,
        "exponential",
        "6-02-background",
        "Background model under the components (CCEP)",
        f"Four background models on {len(WAVES)} CCEP channels from {N_CCEP} patients. "
        "Without a suitable background, the slow drift is fitted with extra components.",
    )


def fig_widths():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    mc = [2, 3, 4, 6, 10]
    res = ccep_sweep([{"max_components": m} for m in mc])
    lab = [str(m) for m in mc]
    bars(
        axes[0],
        lab,
        [np.nanmean([n_comp(r) for r, s in zip(rs, STRONG, strict=True) if s]) for rs in res],
        [sem([n_comp(r) for r, s in zip(rs, STRONG, strict=True) if s]) for rs in res],
        "6",
        "mean number of components",
        "Maximum number of components",
    )
    axes[0].set_xlabel("maximum allowed")
    hit = [
        np.mean([n_comp(r) >= m for r, s in zip(rs, STRONG, strict=True) if s]) * 100
        for rs, m in zip(res, mc, strict=True)
    ]
    axes[1].plot(lab, hit, "o-", color="black")
    axes[1].set_ylabel("% of channels that reach the maximum")
    axes[1].set_xlabel("maximum allowed")
    axes[1].set_title("Does the maximum constrain the fit?", fontsize=10)
    mf = [0.05, 0.1, 0.25, 0.5]
    res2 = ccep_sweep([{"max_fwhm": m} for m in mf])
    lab2 = [f"{m * 1000:.0f} ms" for m in mf]
    widths = [
        [
            c["fwhm"] * 1000
            for r, s in zip(rs, STRONG, strict=True)
            if s
            for c in r.get("components", [])
        ]
        for rs in res2
    ]
    nb = [np.nanmean([n_comp(r) for r, s in zip(rs, STRONG, strict=True) if s]) for rs in res2]
    bp = axes[2].boxplot(widths, tick_labels=lab2, showfliers=False, patch_artist=True)
    for patch, col in zip(bp["boxes"], colours(lab2, "250 ms"), strict=True):
        patch.set_facecolor(col)
        patch.set_alpha(0.5)
    for k, v in enumerate(nb):
        axes[2].text(
            k + 1, axes[2].get_ylim()[1] * 0.95, f"{v:.1f} comp.", ha="center", fontsize=7
        )
    axes[2].set_ylabel("component FWHM (ms)")
    axes[2].set_xlabel("maximum FWHM allowed")
    axes[2].set_title("Maximum FWHM allowed", fontsize=10)
    fig.suptitle("Bounds: maximum number of components and maximum width (CCEP)", fontsize=12)
    save(
        fig,
        "6-03-width-and-max-components",
        f"Maximum number of components (2 to 10) and maximum FWHM (50 to 500 ms) on {len(WAVES)} CCEP channels "
        f"from {N_CCEP} patients. Shows whether the bound constrains the model and which widths it uses.",
    )


def fig_delta():
    deltas = [2.0, 5.0, 10.0, 20.0, 40.0, "calibrated"]
    labels = ["2", "5", "10", "20", "40", "calibrated"]
    res = ccep_sweep([{"delta": d} for d in deltas], curve=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x = np.arange(len(labels))
    for sel, tag, style in (
        (STRONG, "responsive channels", "o-"),
        (~STRONG, "random channels", "s--"),
    ):
        m = [np.nanmean([n_comp(r) for r, s in zip(rs, sel, strict=True) if s]) for rs in res]
        e = [sem([n_comp(r) for r, s in zip(rs, sel, strict=True) if s]) for rs in res]
        axes[0].errorbar(x, m, yerr=e, fmt=style, capsize=3, label=tag, color="black")
        n1 = [
            np.mean([has(r, "N1") for r, s in zip(rs, sel, strict=True) if s]) * 100 for rs in res
        ]
        axes[1].plot(x, n1, style, label=tag, color="black")
    for ax in axes[:2]:
        ax.set_xticks(x, labels)
        ax.axvline(2, color=PROPOSED, alpha=0.3, lw=8)
        ax.set_xlabel("|ΔBIC| threshold to keep one more component")
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("mean number of components")
    axes[0].set_title("Components kept", fontsize=10)
    axes[1].set_ylabel("% of channels with an N1")
    axes[1].set_title("Channels with an N1", fontsize=10)
    best = int(
        np.argmax(
            [
                n_comp(r) - n_comp(res[4][i]) if WAVES[i]["strong"] else -99
                for i, r in enumerate(res[0])
            ]
        )
    )
    example(
        axes[2],
        [res[0], res[2], res[4]],
        ["ΔBIC 2", "ΔBIC 10", "ΔBIC 40"],
        "ΔBIC 10",
        best,
        "Example: the channel most affected by the threshold",
    )
    fig.suptitle("Rule to keep one more component: ΔBIC ≤ −threshold (CCEP)", fontsize=12)
    save(
        fig,
        "6-04-delta-bic",
        f"ΔBIC threshold from 2 to 40, and calibrated on simulation, on {len(WAVES)} CCEP channels from {N_CCEP} patients. "
        "Shows the components kept and the share of channels with an N1, on responsive and random channels.",
    )


def fig_candidate():
    th = [1.0, 1.5, 2.0, 3.0, 4.0]
    labels = [f"{t:g} σ" for t in th]
    res = ccep_sweep([{"detection_threshold": t} for t in th])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x = np.arange(len(th))
    for sel, tag, style in (
        (STRONG, "responsive channels", "o-"),
        (~STRONG, "random channels", "s--"),
    ):
        axes[0].errorbar(
            x,
            [np.nanmean([n_comp(r) for r, s in zip(rs, sel, strict=True) if s]) for rs in res],
            yerr=[sem([n_comp(r) for r, s in zip(rs, sel, strict=True) if s]) for rs in res],
            fmt=style,
            color="black",
            capsize=3,
            label=tag,
        )
        axes[1].plot(
            x,
            [
                np.mean([has(r, "N1") for r, s in zip(rs, sel, strict=True) if s]) * 100
                for rs in res
            ],
            style,
            color="black",
            label=tag,
        )
    axes[2].bar(
        x,
        [np.nanmedian([r.get("seconds", np.nan) for r in rs]) * 1000 for rs in res],
        color=colours(labels, "2 σ"),
    )
    for ax, yl, tt in zip(
        axes,
        ["mean number of components", "% of channels with an N1", "median fit time (ms)"],
        ["Components kept", "Channels with an N1", "Computation time"],
        strict=True,
    ):
        ax.set_xticks(x, labels)
        ax.set_ylabel(yl)
        ax.set_title(tt, fontsize=10)
        ax.set_xlabel("candidate threshold (residual peak, in noise σ)")
    axes[0].legend(fontsize=7, frameon=False)
    for ax in axes[:2]:
        ax.axvline(2, color=PROPOSED, alpha=0.3, lw=8)
    fig.suptitle("Starting-candidate threshold (CCEP)", fontsize=12)
    save(
        fig,
        "6-05-candidate-threshold",
        f"Starting-candidate threshold from 1 to 4 σ on {len(WAVES)} CCEP channels from {N_CCEP} patients. "
        "Shows the components kept, the N1s found and the computation time.",
    )


def gaps(r):
    comps = r.get("components", []) if r else []
    lats = sorted(c["latency"] for c in comps)
    return [(b - a) * 1000 for a, b in zip(lats, lats[1:])]


def fig_merge():
    oc = [0.0, 0.25, 0.5, 1.0, 2.0]
    labels = [f"{o:g}" for o in oc]
    res = ccep_sweep([{"overlap_c": o} for o in oc])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bars(
        axes[0],
        labels,
        [np.nanmean([n_comp(r) for r, s in zip(rs, STRONG, strict=True) if s]) for rs in res],
        [sem([n_comp(r) for r, s in zip(rs, STRONG, strict=True) if s]) for rs in res],
        "0.5",
        "mean number of components",
        "Components kept (responsive channels)",
    )
    axes[0].set_xlabel("merge factor c: merge if gap < c × (σ₁ + σ₂)")
    g = [[x for r, s in zip(rs, STRONG, strict=True) if s for x in gaps(r)] for rs in res]
    bp = axes[1].boxplot(g, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch, col in zip(bp["boxes"], colours(labels, "0.5"), strict=True):
        patch.set_facecolor(col)
        patch.set_alpha(0.5)
    axes[1].set_ylabel("gap between neighbouring components (ms)")
    axes[1].set_xlabel("factor c")
    axes[1].set_title("Are close components kept apart?", fontsize=10)
    fig.suptitle("Merging two close components (CCEP)", fontsize=12)
    save(
        fig,
        "6-06-merge",
        f"Merge factor c from 0 to 2 on {len(WAVES)} CCEP channels from {N_CCEP} patients. "
        "Shows the number of components and the gap between neighbours; a larger c merges more.",
    )


def cancelling_pairs(r):
    comps = r.get("components", []) if r else []
    n = 0
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            a, b = comps[i], comps[j]
            if np.sign(a["amplitude"]) != np.sign(b["amplitude"]) and abs(
                a["latency"] - b["latency"]
            ) < 0.5 * (a["fwhm"] + b["fwhm"]):
                n += 1
    return n


# ------------------------------------------------------------------------------------------
# ERP CORE
ERP = np.load(os.path.join(HERE, "erp_waves.npz"))


def T(key):
    return ERP[key.split("_")[0] + "_times"]


import erpcore  # noqa: E402

ERP_KEYS = sorted(
    {k.rsplit("_", 1)[0] for k in ERP.files if k.endswith("_wave")},
    key=lambda k: (k.split("_")[0], int(k.split("_")[1])),
)
COMPS = list(erpcore.DESIGNS)


def erp_preset(
    comp, margin=erpcore.MARGIN, n_components=erpcore.N_COMPONENTS, onset_fraction=None
):
    from evoparam import presets

    d = erpcore.DESIGNS[comp]
    lo, hi = d["window"]
    pol = "positive" if d["polarity"] > 0 else "negative"
    p = presets.targeted(
        "erp", comp, d["window"], pol, margin=margin, n_components=n_components, background="none"
    )
    t = T(comp)
    fw = (max(lo - margin, t[0] + 0.01), min(hi + margin, t[-1] - 0.01))
    if lo < presets.get("erp").fit_window[0]:
        p = presets.derive(p, name=p.name, fit_window=fw, baseline=(t[0], min(0.0, fw[0])))
    else:
        p = presets.derive(p, name=p.name, fit_window=fw)
    return p


def erp_job(key, margin=erpcore.MARGIN, n_components=erpcore.N_COMPONENTS, overrides=None):
    comp = key.split("_")[0]
    return {
        "key": key,
        "times": T(key),
        "wave": ERP[key + "_wave"],
        "curve": True,
        "preset": erp_preset(comp, margin, n_components),
        "overrides": overrides or {},
        "noise_ref": ERP[key + "_ref"] if key + "_ref" in ERP.files else None,
        "comp": comp,
    }


def erp_fit_job(job):
    from evoparam.models import engine, ResponseModel

    if job["noise_ref"] is not None:
        model = ResponseModel(job["preset"], **job["overrides"])
        if not engine.use_plus_minus(model.settings(), job["times"], job["wave"]):
            job = dict(job, noise_ref=None)
    r = common.fit_job(job)
    comp = job["comp"]
    parts = [c for c in r.get("components", []) if c["label"] and c["label"].startswith(comp)]
    r["parts"] = parts
    return r


def erp_measures(r, key):
    comp = key.split("_")[0]
    d = erpcore.DESIGNS[comp]
    wave = ERP[key + "_wave"]
    tt = T(key)
    ref = erpcore.area_latency(tt, wave, d["window"], d["polarity"])
    inside = (tt >= d["window"][0]) & (tt <= d["window"][1])
    raw_peak = float(tt[inside][np.argmax(d["polarity"] * wave[inside])])
    out = {"ref": ref, "raw_peak": raw_peak, "model_peak": np.nan, "model_area": np.nan}
    parts = r.get("parts") or []
    if parts:
        big = max(parts, key=lambda c: abs(c["area"]))
        out["model_peak"] = big["latency"]
        # the sum of the labelled parts: their curves are not returned one by one, so the
        # 50 % area is taken on the whole model curve, which in the targeted fit (no
        # background, components of the polarity only) is that sum
        if "curve" in r:
            out["model_area"] = erpcore.area_latency(tt, r["curve"], d["window"], d["polarity"])
    return out


def fig_latency():
    res = POOL.map(erp_fit_job, [erp_job(k) for k in ERP_KEYS], chunksize=2)
    rows = {k: erp_measures(r, k) for k, r in zip(ERP_KEYS, res, strict=True)}
    defs = [
        ("raw_peak", "raw signal peak"),
        ("model_peak", "model component peak"),
        ("model_area", "50 % area of the model"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
    width = 0.27
    for j, (field, lab) in enumerate(defs):
        data = [
            [(rows[k][field] - rows[k]["ref"]) * 1000 for k in ERP_KEYS if k.startswith(c + "_")]
            for c in COMPS
        ]
        data = [np.array(v)[np.isfinite(v)] for v in data]
        pos = np.arange(len(COMPS)) + (j - 1) * width
        bp = axes[0].boxplot(
            data, positions=pos, widths=width * 0.9, showfliers=False, patch_artist=True
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(["#bdc3c7", "#5d6d7e", PROPOSED][j])
            patch.set_alpha(0.7)
        axes[0].plot([], [], "s", color=["#bdc3c7", "#5d6d7e", PROPOSED][j], label=lab)
    axes[0].axhspan(-10, 10, color="0.93", zorder=0)
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set_xticks(np.arange(len(COMPS)), COMPS)
    axes[0].set_ylabel("difference from the published latency (ms)\n(50 % area of the raw signal)")
    axes[0].set_title("Difference per ERP component (grey band: ± 10 ms)", fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)
    pct = [
        [
            np.mean(
                [
                    abs(rows[k][f] - rows[k]["ref"]) <= 0.010
                    for k in ERP_KEYS
                    if k.startswith(c + "_") and np.isfinite(rows[k][f])
                ]
            )
            * 100
            for c in COMPS
        ]
        for f, _ in defs
    ]
    for j, (_, lab) in enumerate(defs):
        axes[1].plot(COMPS, pct[j], "o-", color=["#bdc3c7", "#5d6d7e", PROPOSED][j], label=lab)
    axes[1].set_ylabel("% of subjects within ± 10 ms of the published value")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Agreement with the published measure", fontsize=10)
    fig.suptitle(
        f"Latency definition (ERP CORE, {len(ERP_KEYS)} waves, 40 subjects × 7 components)",
        fontsize=12,
    )
    save(
        fig,
        "6-08-latency-definition",
        f"Three latency definitions (raw peak, model peak, 50 % area of the model) on {len(ERP_KEYS)} "
        "ERP CORE waves (40 subjects, 7 components). Shows the difference from the published latency and the share within ± 10 ms.",
    )
    return res


def bounds(r, key, fraction):
    """Start and end of the modelled component (sum of the parts) at a fraction of its peak."""
    if "curve" not in r or not r.get("parts"):
        return None
    d = erpcore.DESIGNS[key.split("_")[0]]
    t = T(key)
    y = d["polarity"] * np.asarray(r["curve"], float)
    lo, hi = d["window"]
    inside = np.flatnonzero((t >= lo - 0.025) & (t <= hi + 0.025))
    i = inside[np.argmax(y[inside])]
    level = fraction * y[i]
    if y[i] <= 0:
        return None
    a = i
    while a > 0 and y[a - 1] >= level:
        a -= 1
    b = i
    while b < len(y) - 1 and y[b + 1] >= level:
        b += 1
    if a == 0 or b == len(y) - 1:  # cut by the epoch edge
        return None
    return t[a], t[b]


def span(r, key, fraction):
    b = bounds(r, key, fraction)
    return (b[1] - b[0]) * 1000 if b else np.nan


def fig_onset():
    fr = [0.05, 0.1, 0.2, 0.5]
    labs = [f"{int(f * 100)} %" for f in fr]
    res = [
        POOL.map(
            erp_fit_job,
            [erp_job(k, overrides={"onset_fraction": f}) for k in ERP_KEYS],
            chunksize=2,
        )
        for f in fr
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for j, (rs, lab) in enumerate(zip(res, labs, strict=True)):
        dur = [
            [span(r, k, fr[j]) for k, r in zip(ERP_KEYS, rs, strict=True) if k.startswith(c + "_")]
            for c in COMPS
        ]
        axes[0].plot(
            COMPS,
            [np.nanmedian(d) if np.isfinite(d).any() else np.nan for d in dur],
            "o-",
            color=PROPOSED if lab == "10 %" else None,
            lw=2.5 if lab == "10 %" else 1.2,
            label=lab,
        )
    axes[0].set_ylabel("median duration of the modelled component (ms)")
    axes[0].set_title("Duration (offset − onset) by threshold", fontsize=10)
    axes[0].legend(title="threshold (% of peak)", fontsize=8, frameon=False)
    # one example: the P3 of the participant with the largest P3
    keys = [k for k in ERP_KEYS if k.startswith("P3_")]
    k = max(keys, key=lambda k: np.max(ERP[k + "_wave"]))
    i = ERP_KEYS.index(k)
    t = T(k) * 1000
    axes[1].plot(t, ERP[k + "_wave"], color="black", lw=1, label="signal")
    r = res[1][i]
    if "curve" in r:
        axes[1].plot(t, r["curve"], color=PROPOSED, lw=1.5, label="model")
    ymax = axes[1].get_ylim()[1]
    for f, lab, rs in zip(fr, labs, res, strict=True):
        b = bounds(rs[i], k, f)
        if b:
            axes[1].hlines(
                ymax * (0.95 - 0.07 * fr.index(f)),
                b[0] * 1000,
                b[1] * 1000,
                color=PROPOSED if lab == "10 %" else "0.4",
                lw=3,
            )
            axes[1].text(
                b[1] * 1000 + 5, ymax * (0.95 - 0.07 * fr.index(f)), lab, va="center", fontsize=7
            )
    axes[1].set_xlim(-100, 900)
    axes[1].set_xlabel("time (ms)")
    axes[1].set_ylabel("µV")
    axes[1].set_title(
        f"Example: P3, subject {k.split('_')[1]} (bars = onset → offset)", fontsize=10
    )
    axes[1].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Onset and offset of a component: at what % of its peak? (ERP CORE)", fontsize=12)
    save(
        fig,
        "6-09-onset-offset",
        f"Onset/offset threshold of 5, 10, 20 and 50 % of the peak on {len(ERP_KEYS)} ERP CORE waves "
        "(40 subjects, 7 components). Shows the reported duration and a P3 example.",
    )


def fig_targeted():
    margins = [0.0, 0.010, 0.025, 0.050]
    ncs = [1, 3]
    settings = [(m, n) for n in ncs for m in margins]
    labs = [f"±{m * 1000:.0f} ms\n{n} part{'s' if n > 1 else ''}" for m, n in settings]
    pct = np.zeros((len(COMPS), len(settings)))
    diffs_default = None
    for s, (m, n) in enumerate(settings):
        rs = POOL.map(
            erp_fit_job, [erp_job(k, margin=m, n_components=n) for k in ERP_KEYS], chunksize=2
        )
        rows = {k: erp_measures(r, k) for k, r in zip(ERP_KEYS, rs, strict=True)}
        for c, comp in enumerate(COMPS):
            ks = [k for k in ERP_KEYS if k.startswith(comp + "_")]
            pct[c, s] = (
                np.mean(
                    [
                        np.isfinite(rows[k]["model_area"])
                        and abs(rows[k]["model_area"] - rows[k]["ref"]) <= 0.010
                        for k in ks
                    ]
                )
                * 100
            )
        if (m, n) == (0.025, 3):
            diffs_default = {
                comp: np.array(
                    [
                        abs(rows[k]["model_area"] - rows[k]["ref"]) * 1000
                        for k in ERP_KEYS
                        if k.startswith(comp + "_")
                    ]
                )
                for comp in COMPS
            }
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    im = axes[0].imshow(pct, cmap="Greys", vmin=0, vmax=100, aspect="auto")
    for c in range(len(COMPS)):
        for s in range(len(settings)):
            axes[0].text(
                s,
                c,
                f"{pct[c, s]:.0f}",
                ha="center",
                va="center",
                color="white" if pct[c, s] > 60 else "black",
                fontsize=8,
            )
    axes[0].set_xticks(range(len(settings)), labs, fontsize=7)
    axes[0].set_yticks(range(len(COMPS)), COMPS)
    k = settings.index((0.025, 3))
    axes[0].add_patch(
        plt.Rectangle((k - 0.5, -0.5), 1, len(COMPS), fill=False, ec=PROPOSED, lw=2.5)
    )
    axes[0].set_title("% of subjects within ± 10 ms of the published latency", fontsize=10)
    fig.colorbar(im, ax=axes[0], fraction=0.03)
    tol = np.arange(1, 41)
    for comp in COMPS:
        d = diffs_default[comp]
        axes[1].plot(tol, [np.mean(np.nan_to_num(d, nan=1e9) <= x) * 100 for x in tol], label=comp)
    axes[1].axvline(10, color=PROPOSED, lw=2, alpha=0.6)
    axes[1].set_xlabel("tolerance (ms)")
    axes[1].set_ylabel("% of subjects within tolerance")
    axes[1].set_title("Choice of tolerance (proposed setting)", fontsize=10)
    axes[1].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle(
        "Targeted ERP measure: window margin, number of parts, tolerance (ERP CORE)", fontsize=12
    )
    save(
        fig,
        "6-10-targeted-measure",
        f"Margin around the published window (0 to 50 ms) and number of parts (1 or 3) on {len(ERP_KEYS)} "
        "ERP CORE waves (40 subjects). Shows agreement with the published latency, then the effect of the tolerance (± 10 ms proposed).",
    )


def fig_cancelling():
    labels = ["no (keep)", "yes (reject)"]
    res = ccep_sweep([{"reject_cancelling": False}, {"reject_cancelling": True}])
    erp_res = [
        POOL.map(
            erp_fit_job,
            [
                dict(
                    erp_job(k),
                    preset="erp",
                    overrides={"reject_cancelling": v},
                    window=None,
                    baseline=None,
                )
                for k in ERP_KEYS
                if not k.startswith(("LRP_", "ERN_"))
            ],
            chunksize=2,
        )
        for v in (False, True)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, rs, tag, prop in (
        (axes[0], res, f"CCEP, {len(WAVES)} channels, {N_CCEP} patients", "no (keep)"),
        (axes[1], erp_res, "ERP CORE, stimulus-locked waves, 40 subjects", "yes (reject)"),
    ):
        nb = [np.nanmean([n_comp(r) for r in x]) for x in rs]
        pairs = [np.mean([cancelling_pairs(r) > 0 for r in x]) * 100 for x in rs]
        xx = np.arange(2)
        ax.bar(xx - 0.2, nb, 0.4, color=colours(labels, prop), label="components per wave")
        ax2 = ax.twinx()
        ax2.bar(
            xx + 0.2,
            pairs,
            0.4,
            color="none",
            edgecolor="black",
            hatch="//",
            label="% with an opposite pair",
        )
        ax.set_xticks(xx, labels)
        ax.set_ylabel("mean number of components")
        ax2.set_ylabel("% of waves with an overlapping + / − pair")
        ax.set_title(tag, fontsize=10)
        ax.set_xlabel("reject cancelling pairs?")
    fig.suptitle("Cancelling pairs of opposite-sign components", fontsize=12)
    save(
        fig,
        "6-07-cancelling-pairs",
        f"Keeping or rejecting close + / − component pairs that cancel, on {len(WAVES)} CCEP channels "
        "and on ERP CORE waves (full erp preset). Solid bars: components per wave; hatched: % of waves with such a pair.",
    )


# ------------------------------------------------------------------------------------------
# TEP
def tep_data():
    import glob

    out = []
    for path in sorted(glob.glob(f"{common.CACHE}/tep-validation/ds002094/sub-*_avgB_v1.npz")):
        z = np.load(path)
        times, data, names = z["times"], z["data"], [str(n) for n in z["names"]]
        inside = (times >= 0.038) & (times <= 0.140)
        ch = int(np.argmax(np.sqrt(np.nanmean(data[:, inside] ** 2, axis=1))))
        out.append(
            {
                "subject": os.path.basename(path).split("_")[0],
                "times": times,
                "wave": data[ch].astype(float),
                "channel": names[ch],
            }
        )
    return out


def fig_tep():
    subs = tep_data()
    res = POOL.map(
        common.fit_job,
        [
            {
                "key": s["subject"],
                "times": s["times"],
                "wave": s["wave"],
                "preset": "tep",
                "curve": True,
            }
            for s in subs
        ],
    )
    options = {
        "proposed: N45 38–55, N100 80–140": ((0.038, 0.055), (0.080, 0.140)),
        "narrow: N45 40–50, N100 90–120": ((0.040, 0.050), (0.090, 0.120)),
        "wide: N45 30–60, N100 70–160": ((0.030, 0.060), (0.070, 0.160)),
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1.5, 1.3, 1]})
    t = subs[0]["times"] * 1000
    keep = (t > -50) & (t < 300)
    ga = np.mean([s["wave"] for s in subs], axis=0)
    sd = np.std([s["wave"] for s in subs], axis=0) / np.sqrt(len(subs))
    axes[0].plot(t[keep], ga[keep], color="black")
    axes[0].fill_between(t[keep], (ga - sd)[keep], (ga + sd)[keep], color="0.8")
    for (lo, hi), col in zip(
        options["proposed: N45 38–55, N100 80–140"], [PROPOSED, PROPOSED], strict=True
    ):
        axes[0].axvspan(lo * 1000, hi * 1000, color=col, alpha=0.15)
    for name, ls in (
        ("narrow: N45 40–50, N100 90–120", ":"),
        ("wide: N45 30–60, N100 70–160", "--"),
    ):
        for lo, hi in options[name]:
            for edge in (lo, hi):
                axes[0].axvline(edge * 1000, ls=ls, color="0.4", lw=0.8)
    axes[0].set_xlabel("time after the TMS pulse (ms)")
    axes[0].set_ylabel("µV")
    axes[0].set_title(
        f"Mean of {len(subs)} subjects (most responsive channel)\nred = proposed, dotted = narrow, dashed = wide",
        fontsize=9,
    )
    neg = [
        [c["latency"] * 1000 for c in r.get("components", []) if c["amplitude"] < 0] for r in res
    ]
    for i, lats in enumerate(neg):
        axes[1].plot(lats, [i] * len(lats), "o", color="black", ms=4)
    for lo, hi in options["proposed: N45 38–55, N100 80–140"]:
        axes[1].axvspan(lo * 1000, hi * 1000, color=PROPOSED, alpha=0.15)
    axes[1].set_yticks(range(len(subs)), [s["subject"] for s in subs], fontsize=6)
    axes[1].set_xlim(0, 300)
    axes[1].set_xlabel("latency of the negative model components (ms)")
    axes[1].set_title("Negative components found, per subject", fontsize=10)
    names = list(options)
    n45 = [
        np.mean(
            [
                any(options[n][0][0] * 1000 <= x <= options[n][0][1] * 1000 for x in lats)
                for lats in neg
            ]
        )
        * 100
        for n in names
    ]
    n100 = [
        np.mean(
            [
                any(options[n][1][0] * 1000 <= x <= options[n][1][1] * 1000 for x in lats)
                for lats in neg
            ]
        )
        * 100
        for n in names
    ]
    x = np.arange(len(names))
    axes[2].bar(x - 0.2, n45, 0.4, color="0.6", label="N45")
    axes[2].bar(x + 0.2, n100, 0.4, color="0.2", label="N100")
    axes[2].set_xticks(x, ["proposed", "narrow", "wide"])
    axes[2].set_ylabel("% of subjects with a negative component in the window")
    axes[2].set_ylim(0, 105)
    axes[2].legend(fontsize=8, frameon=False)
    axes[2].set_title("Effect of the window", fontsize=10)
    fig.suptitle("TEP component windows N45 and N100 (ds002094, published cleaning)", fontsize=12)
    save(
        fig,
        "6-11-tep-windows",
        f"Three sets of N45 / N100 windows on {len(subs)} TMS-EEG subjects (ds002094, TESA cleaning). "
        "Shows the subject mean, each subject's negative components and the share of subjects with a component in each window.",
    )


# ------------------------------------------------------------------------------------------
# Trial by trial (CCEP trials)
FACTORS = [0.1, 0.25, 0.5, 1.0, 2.0]


def trial_job(w):
    from evoparam.data import TrialSet
    from evoparam.trials import fit_trials

    folder = "n1-validation/ds004080" if "UMCU" in w["run"] else "reference-validation/ds004977"
    z = np.load(f"{common.CACHE}/{folder}/{w['run']}.npz")
    names = [str(n) for n in z["names"]]
    i = names.index(w["channel"])
    ts = TrialSet(
        data=z["data"][:, i : i + 1].astype(float),
        times=z["times"],
        sfreq=float(z["sfreq"]),
        ch_names=[w["channel"]],
    )
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            base = fit_trials(ts, preset="ccep", by=(), random_state=0, components=[0])
            if not base.fits or not base.fits[0].template.components:
                return []
            group = base.fits[0].template
            width = group.components[0].fwhm
        except Exception:  # noqa: BLE001
            return []
        for f in FACTORS:
            try:
                tf = fit_trials(
                    ts, preset="ccep", by=(), random_state=0, components=[0], max_shift=f * width
                )
                u = tf.fits[0]
                shift = np.asarray(u.shift, float).reshape(len(u.shift), -1)[:, 0]
                bound = (
                    float(tf.settings.get("max_shift", [f * width])[0])
                    if isinstance(tf.settings.get("max_shift"), list)
                    else f * width
                )
                gof = np.asarray(u.gof, float).ravel()
                out.append(
                    {
                        "factor": f,
                        "fwhm": width * 1000,
                        "jitter": float(np.std(shift)) * 1000,
                        "null": float(list(u.jitter_sd_null.values())[0]) * 1000
                        if u.jitter_sd_null
                        else np.nan,
                        "bound": float(np.mean(np.abs(shift) >= 0.98 * f * width)) * 100,
                        "r2": float(np.nanmedian(gof)),
                        "shifts": (shift * 1000).tolist(),
                    }
                )
            except Exception as err:  # noqa: BLE001
                out.append({"factor": f, "error": repr(err)})
    return out


def fig_trials():
    chans = [w for w in WAVES if w["strong"]]
    res = POOL.map(trial_job, chans, chunksize=1)
    res = [r for r in res if r and all("error" not in x for x in r)]
    print("trial channels", len(res), flush=True)
    labels = [f"{f:g}" for f in FACTORS]
    x = np.arange(len(FACTORS))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    jit = np.array([[r[k]["jitter"] for k in range(len(FACTORS))] for r in res])
    nul = np.array([[r[k]["null"] for k in range(len(FACTORS))] for r in res])
    bnd = np.array([[r[k]["bound"] for k in range(len(FACTORS))] for r in res])
    r2v = np.array([[r[k]["r2"] for k in range(len(FACTORS))] for r in res])
    axes[0].errorbar(
        x,
        np.nanmedian(jit, 0),
        yerr=[
            np.nanmedian(jit, 0) - np.nanpercentile(jit, 25, 0),
            np.nanpercentile(jit, 75, 0) - np.nanmedian(jit, 0),
        ],
        fmt="o-",
        color="black",
        capsize=3,
        label="measured shifts (real trials)",
    )
    axes[0].plot(
        x, np.nanmedian(nul, 0), "s--", color="0.5", label="shifts produced by noise alone"
    )
    axes[0].set_ylabel("SD of the trial shifts (ms)")
    axes[0].set_title("Trial-to-trial variability found", fontsize=10)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.93))
    axes[1].bar(x, np.nanmean(bnd, 0), color=colours(labels, "0.25"))
    axes[1].set_ylabel("% of trials at the bound")
    axes[1].set_title("Does the shift hit the bound?", fontsize=10)
    axes[2].bar(x, np.nanmedian(r2v, 0), color=colours(labels, "0.25"))
    axes[2].set_ylim(max(0, np.nanmin(np.nanmedian(r2v, 0)) - 0.1), 1)
    axes[2].set_ylabel("median single-trial R²")
    axes[2].set_title("Single-trial goodness of fit", fontsize=10)
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlabel("maximum shift allowed (× FWHM)")
    for ax in axes[:1]:
        ax.axvline(1, color=PROPOSED, alpha=0.25, lw=8)
        ax.axvline(2, color="#e67e22", alpha=0.25, lw=8)
    axes[0].text(
        1, axes[0].get_ylim()[1] * 0.97, "intracranial", ha="center", fontsize=7, color=PROPOSED
    )
    axes[0].text(
        2, axes[0].get_ylim()[1] * 0.90, "scalp", ha="center", fontsize=7, color="#e67e22"
    )
    fig.suptitle(
        f"Trial by trial: maximum shift of the first component (CCEP, {len(res)} channels)",
        fontsize=12,
    )
    save(
        fig,
        "6-12-trial-by-trial",
        f"Maximum single-trial shift from 0.1 to 2 × the component FWHM on {len(res)} responsive CCEP channels "
        f"({N_CCEP} patients, 10 to 12 trials each). Shows the variability found, against noise alone, "
        "and the share of trials at the bound.",
    )


def main():
    global POOL
    POOL = Pool(9)
    which = sys.argv[1:] or [
        "shapes",
        "background",
        "widths",
        "delta",
        "candidate",
        "merge",
        "cancelling",
        "latency",
        "onset",
        "targeted",
        "tep",
        "trials",
    ]
    for name in which:
        globals()["fig_" + name]()
    import json

    path = os.path.join(OUT, "_captions.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(CAPTIONS)
    json.dump(old, open(path, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
