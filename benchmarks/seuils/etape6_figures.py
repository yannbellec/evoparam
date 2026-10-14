"""Statistics, figures and CSV tables of the step-6 benchmarks (version 2).

Every number comes from the confirmation half of the subjects; the development half is used
only to pick the best value of each setting (marked by a triangle). Subject means, 95 %
bootstrap CIs over subjects, paired Wilcoxon tests against the proposed value (Holm-corrected
across the values of a panel; * p_Holm < 0.05).
"""

import json
import os
import pickle
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "validation"))
import etape6_common as C  # noqa: E402

RES = os.path.join(C.WORK, "results")
RED, BLUE, GREY = "#c0392b", "#2471a3", "#7f8c8d"
plt.rcParams.update(
    {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 150}
)
TABLES = []
CAPTIONS = {}


def save(fig, name, caption):
    os.makedirs(C.OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT, name + ".png"), metadata={"Software": None})
    plt.close(fig)
    CAPTIONS[name] = caption
    print("saved", name, flush=True)


def label(v):
    if isinstance(v, tuple):
        return "+".join({"gaussian": "G", "skewed_gaussian": "S"}[x] for x in v)
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


# ------------------------------------------------------------------------------------------
# Generic summary of one metric over values


def summarise(per_subject, values, proposed, figure, metric, unit=""):
    """per_subject: DataFrame index subject, columns value index -> confirmation statistics.
    Adds the rows to TABLES and returns them."""
    m = per_subject.reindex(columns=range(len(values))).to_numpy(float)
    tests = C.paired_vs(m, proposed)
    rows = []
    for k, v in enumerate(values):
        mean, lo, hi = C.boot_ci(m[:, k])
        rows.append(
            {
                "figure": figure,
                "metric": metric,
                "unit": unit,
                "value": label(v),
                "proposed": k == proposed,
                "mean": mean,
                "ci_low": lo,
                "ci_high": hi,
                "n_subjects": int(np.isfinite(m[:, k]).sum()),
                "diff_vs_proposed": tests[k]["diff"],
                "diff_ci_low": tests[k]["diff_lo"],
                "diff_ci_high": tests[k]["diff_hi"],
                "p_wilcoxon": tests[k]["p"],
                "p_holm": tests[k]["p_holm"],
            }
        )
    TABLES.extend(rows)
    return rows


def panel(ax, rows, values, proposed, best, title, ylabel, xlabel):
    x = np.arange(len(values))
    mean = np.array([r["mean"] for r in rows])
    lo = np.array([r["ci_low"] for r in rows])
    hi = np.array([r["ci_high"] for r in rows])
    ax.fill_between(x, lo, hi, color="0.88")
    ax.plot(x, mean, "-o", color="black", ms=3.5, lw=1)
    ax.plot(x[proposed], mean[proposed], "o", color=RED, ms=7, zorder=5)
    if best is not None:
        ax.plot(x[best], mean[best], "^", color=BLUE, ms=8, zorder=6, mfc="none", mew=1.5)
    top = np.nanmax(hi) if np.isfinite(hi).any() else 1
    for k, r in enumerate(rows):
        if k != proposed and np.isfinite(r["p_holm"]) and r["p_holm"] < 0.05:
            ax.text(k, hi[k] + 0.02 * abs(top), "*", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x, [label(v) for v in values], rotation=45 if len(values) > 6 else 0, fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8)


def legend_note(fig):
    fig.text(
        0.01,
        0.005,
        "Red: proposed value. Blue triangle: best value on the development half (F1 on known truth). "
        "Grey band: 95 % bootstrap CI over subjects. *: differs from the proposed value (paired Wilcoxon, "
        "Holm-corrected, p < 0.05). Confirmation half only.",
        fontsize=6.5,
        color="0.3",
    )


# ------------------------------------------------------------------------------------------
# CCEP sweeps


CCEP_TEXT = {
    "shapes": (
        "6-01-shapes",
        "Allowed component shapes (G: Gaussian, S: skewed Gaussian)",
        "allowed shapes",
    ),
    "background": ("6-02-background", "Background model under the components", "background"),
    "max_components": (
        "6-03a-max-components",
        "Maximum number of components",
        "maximum number of components",
    ),
    "max_fwhm": ("6-03b-max-fwhm", "Maximum component width (FWHM)", "maximum FWHM (s)"),
    "delta": (
        "6-04-delta-bic",
        "Rule to keep one more component: ΔBIC ≤ −threshold",
        "|ΔBIC| threshold",
    ),
    "detection_threshold": (
        "6-05-candidate-threshold",
        "Starting-candidate threshold",
        "threshold (noise σ)",
    ),
    "overlap_c": ("6-06-merge", "Merge factor c: merge if gap < c × (σ₁ + σ₂)", "factor c"),
    "reject_cancelling": (
        "6-07a-cancelling-pairs-ccep",
        "Reject cancelling pairs of opposite-sign components",
        "reject cancelling pairs",
    ),
}


def ccep_per_subject(df):
    syn = df[df.kind == "synth"]
    g = syn.groupby(["subject", "value_index"])
    tp, fp, fn, nt = g.tp.sum(), g.fp.sum(), g.fn.sum(), g.n_true.sum()
    out = {
        "recall": tp / nt * 100,
        "false_synth": g.fp.mean(),
        "f1": 2 * tp / (2 * tp + fp + fn) * 100,
        "lat_mae": g.lat_err.apply(
            lambda s: np.mean([x for v in s for x in v]) if sum(map(len, s)) else np.nan
        ),
        "amp_err": g.amp_err.apply(
            lambda s: np.median([abs(x) for v in s for x in v]) if sum(map(len, s)) else np.nan
        ),
    }
    nul = df[df.kind == "null"].groupby(["subject", "value_index"])
    out["false_null"] = nul.n_found.mean()
    out["n1_null"] = nul.n1.mean() * 100
    real = df[df.kind == "real"].copy()
    real["agree"] = real.n1_odd == real.n1_even
    real["lat_diff"] = (real.lat_odd - real.lat_even).abs() * 1000
    rg = real.groupby(["subject", "value_index"])
    out["n_real"] = rg.n_found.mean()
    out["n1_real"] = rg.n1.mean() * 100
    out["split_agree"] = rg.agree.mean() * 100
    out["split_lat"] = rg.lat_diff.median()
    out["seconds"] = df.groupby(["subject", "value_index"]).seconds.median() * 1000
    return {k: v.unstack("value_index") for k, v in out.items()}


def ccep_figure(sweep):
    path = os.path.join(RES, f"ccep_{sweep}.pkl")
    if not os.path.exists(path):
        return
    blob = pickle.load(open(path, "rb"))
    values, proposed = blob["values"], blob["proposed"]
    df = pd.DataFrame(blob["records"])
    subs = sorted(df.subject.unique())
    dev, conf = C.split(subs)
    ps = ccep_per_subject(df)
    best = int(np.nanargmax(ps["f1"].loc[dev].mean().reindex(range(len(values))).to_numpy()))
    name, title, xlabel = CCEP_TEXT[sweep]
    metrics = [
        ("recall", "Known truth: components recovered", "recall (%)"),
        ("false_synth", "Known truth: false components", "false components per wave"),
        ("lat_mae", "Known truth: latency error", "mean |latency error| (ms)"),
        ("false_null", "Real noise alone: false components", "components per wave (all false)"),
        ("n_real", "Real data: components kept", "components per channel"),
        ("split_agree", "Real data: odd vs even trials agree on N1", "agreement (% of channels)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (metric, ttl, yl) in zip(axes.ravel(), metrics, strict=True):
        rows = summarise(ps[metric].loc[conf], values, proposed, name, metric)
        panel(ax, rows, values, proposed, best, ttl, yl, xlabel)
    summarise(ps["f1"].loc[conf], values, proposed, name, "f1")
    summarise(ps["n1_null"].loc[conf], values, proposed, name, "n1_null")
    summarise(ps["n1_real"].loc[conf], values, proposed, name, "n1_real")
    summarise(ps["seconds"].loc[conf], values, proposed, name, "fit_ms")
    for k in range(len(values)):
        TABLES.append(
            {
                "figure": name,
                "metric": "f1_development",
                "value": label(values[k]),
                "proposed": k == proposed,
                "mean": float(ps["f1"].loc[dev][k].mean()),
                "n_subjects": len(dev),
                "chosen_on_development": k == best,
            }
        )
    n_syn = int((df.kind == "synth").sum() / len(values))
    fig.suptitle(
        f"{title} — CCEP, ds004080, {len(subs)} patients (confirmation half: {len(conf)})",
        fontsize=11,
    )
    legend_note(fig)
    save(
        fig,
        name,
        f"{title}. CCEP, ds004080, {len(subs)} patients; numbers on the {len(conf)} confirmation patients. "
        f"Known truth: {n_syn} semi-simulated waves per value (60 per patient, components added to the patient's "
        "plus-minus noise). Real noise: the plus-minus average of every channel. Real data: every good channel; "
        "split-half agreement of the N1 decision between odd and even trials.",
    )


# ------------------------------------------------------------------------------------------
# ERP CORE


def erp_measure_figures():
    path = os.path.join(RES, "erp_measure.pkl")
    if not os.path.exists(path):
        return
    import erpcore

    df = pd.DataFrame(pickle.load(open(path, "rb"))["records"])
    subs = sorted(df.subject.unique(), key=int)
    dev, conf = C.split(subs)
    comps = list(erpcore.DESIGNS)
    prop = df[(df.margin == 0.025) & (df.parts == 3)]

    # --- 6-08 latency definitions
    defs = [
        ("raw_peak", "peak", "raw peak"),
        ("raw_area", "area", "raw 50 % area"),
        ("model_peak", "peak", "model peak"),
        ("model_area", "area", "model 50 % area"),
    ]
    syn = prop[prop.kind == "synth"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    width = 0.2
    for j, (col, truth, lab) in enumerate(defs):
        err = (syn[col] - syn[f"truth_{truth}"]).abs() * 1000
        per = syn.assign(err=err).groupby(["component", "subject"]).err.mean()
        means, los, his = [], [], []
        for c in comps:
            v = (
                per.loc[c].reindex(conf).to_numpy()
                if c in per.index.get_level_values(0)
                else np.array([])
            )
            m, lo, hi = C.boot_ci(v)
            means.append(m), los.append(m - lo), his.append(hi - m)
            TABLES.append(
                {
                    "figure": "6-08-latency-definition",
                    "metric": "synthetic_abs_error_ms",
                    "component": c,
                    "value": lab,
                    "mean": m,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_subjects": int(np.isfinite(v).sum()),
                }
            )
        axes[0].bar(
            np.arange(len(comps)) + (j - 1.5) * width,
            means,
            width,
            yerr=[los, his],
            capsize=2,
            color=["#d5d8dc", "#aab7b8", "#5d6d7e", RED][j],
            label=lab,
        )
    axes[0].set_xticks(range(len(comps)), comps)
    axes[0].set_ylabel("|error| against the true latency of the same definition (ms)")
    axes[0].set_title("Known truth: error of each definition", fontsize=9)
    axes[0].legend(fontsize=7, frameon=False)
    odd = prop[prop.kind == "odd"].set_index("key")
    even = prop[prop.kind == "even"].set_index("key")
    real = prop[prop.kind == "real"].set_index("key")
    for j, (col, _, lab) in enumerate(defs):
        d = (odd[col] - even[col].reindex(odd.index)).abs() * 1000
        per = (
            pd.DataFrame({"d": d, "component": odd.component, "subject": odd.subject})
            .groupby(["component", "subject"])
            .d.mean()
        )
        means, los, his = [], [], []
        for c in comps:
            v = per.loc[c].reindex(conf).to_numpy()
            m, lo, hi = C.boot_ci(v)
            means.append(m), los.append(m - lo), his.append(hi - m)
            TABLES.append(
                {
                    "figure": "6-08-latency-definition",
                    "metric": "split_half_abs_difference_ms",
                    "component": c,
                    "value": lab,
                    "mean": m,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_subjects": int(np.isfinite(v).sum()),
                }
            )
        axes[1].bar(
            np.arange(len(comps)) + (j - 1.5) * width,
            means,
            width,
            yerr=[los, his],
            capsize=2,
            color=["#d5d8dc", "#aab7b8", "#5d6d7e", RED][j],
            label=lab,
        )
        within = ((real[col] - real["published"]).abs() <= 0.010).astype(float) * 100
        per = (
            pd.DataFrame({"w": within, "component": real.component, "subject": real.subject})
            .groupby(["component", "subject"])
            .w.mean()
        )
        pct = []
        for c in comps:
            v = per.loc[c].reindex(conf).to_numpy()
            m, lo, hi = C.boot_ci(v)
            pct.append(m)
            TABLES.append(
                {
                    "figure": "6-08-latency-definition",
                    "metric": "within_10ms_of_published_pct",
                    "component": c,
                    "value": lab,
                    "mean": m,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_subjects": int(np.isfinite(v).sum()),
                }
            )
        axes[2].plot(comps, pct, "o-", color=["#d5d8dc", "#aab7b8", "#5d6d7e", RED][j], label=lab)
    axes[1].set_xticks(range(len(comps)), comps)
    axes[1].set_ylabel("|odd − even trials| (ms)")
    axes[1].set_title("Real data: split-half reliability (lower is better)", fontsize=9)
    axes[2].set_ylim(0, 105)
    axes[2].set_ylabel("% of subjects within ± 10 ms")
    axes[2].set_title(
        "Real data: agreement with the published latency\n(the published value is the raw 50 % area)",
        fontsize=9,
    )
    axes[2].legend(fontsize=7, frameon=False)
    fig.suptitle(
        f"Latency definition — ERP CORE, {len(subs)} subjects × 7 components (confirmation half: {len(conf)})",
        fontsize=11,
    )
    fig.text(
        0.01,
        0.005,
        "Bars: mean over confirmation subjects, 95 % bootstrap CI. Fits: targeted measure, ± 25 ms, 3 parts.",
        fontsize=6.5,
        color="0.3",
    )
    save(
        fig,
        "6-08-latency-definition",
        "Four latency definitions on ERP CORE. Known truth: one component of known shape (half skewed) added to the "
        "subject's plus-minus noise, with a neighbour of opposite polarity in half of the waves, 10 per subject and "
        "component; error against the true latency of the same definition. Real data: odd vs even trials, and the "
        "published 50 % area latency.",
    )

    # --- 6-09 onset / offset fraction
    from etape6_erp import FRACTIONS

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, which, ttl in (
        (axes[0], "on", "Known truth: onset error"),
        (axes[1], "off", "Known truth: offset error"),
    ):
        mat = {}
        for f in FRACTIONS:
            e = (syn[f"{which}_{f}"] - syn[f"truth_{which}_{f}"]).abs() * 1000
            mat[f] = syn.assign(e=e).groupby("subject").e.mean()
        per = pd.DataFrame(mat).reindex(conf)
        per.columns = range(len(FRACTIONS))
        rows = summarise(
            per, FRACTIONS, 1, "6-09-onset-offset", f"synthetic_{which}set_abs_error_ms", "ms"
        )
        panel(ax, rows, FRACTIONS, 1, None, ttl, "|error| (ms)", "fraction of the peak")
    mat = {}
    for f in FRACTIONS:
        d = (odd[f"on_{f}"] - even[f"on_{f}"].reindex(odd.index)).abs() * 1000
        mat[f] = pd.DataFrame({"d": d, "subject": odd.subject}).groupby("subject").d.mean()
    per = pd.DataFrame(mat).reindex(conf)
    per.columns = range(len(FRACTIONS))
    rows = summarise(
        per, FRACTIONS, 1, "6-09-onset-offset", "split_half_onset_abs_difference_ms", "ms"
    )
    panel(
        axes[2],
        rows,
        FRACTIONS,
        1,
        None,
        "Real data: onset, odd vs even trials",
        "|odd − even| (ms)",
        "fraction of the peak",
    )
    fig.suptitle(
        f"Onset and offset: at what fraction of the peak? — ERP CORE, {len(subs)} subjects (confirmation half: {len(conf)})",
        fontsize=11,
    )
    legend_note(fig)
    save(
        fig,
        "6-09-onset-offset",
        "Onset and offset read at 5 to 50 % of the peak of the modelled component (targeted measure). Known truth: "
        "error against the true onset/offset at the same fraction; real data: odd vs even trials. Subject means "
        "over the 7 components.",
    )

    # --- 6-10 targeted measure: margin x parts, and tolerance
    from etape6_erp import MARGINS, PARTS

    grid_syn = np.full((len(PARTS), len(MARGINS)), np.nan)
    grid_real = np.full((len(PARTS), len(MARGINS)), np.nan)
    for a, p in enumerate(PARTS):
        for b, m in enumerate(MARGINS):
            sub = df[(df.margin == m) & (df.parts == p)]
            s = sub[sub.kind == "synth"]
            w = ((s.model_area - s.truth_area).abs() <= 0.010).astype(float) * 100
            v = s.assign(w=w).groupby("subject").w.mean().reindex(conf).to_numpy()
            grid_syn[a, b] = C.boot_ci(v)[0]
            r = sub[sub.kind == "real"]
            w = ((r.model_area - r.published).abs() <= 0.010).astype(float) * 100
            v2 = r.assign(w=w).groupby("subject").w.mean().reindex(conf).to_numpy()
            grid_real[a, b] = C.boot_ci(v2)[0]
            for metric, vv in (
                ("synthetic_within_10ms_pct", v),
                ("published_within_10ms_pct", v2),
            ):
                mean, lo, hi = C.boot_ci(vv)
                TABLES.append(
                    {
                        "figure": "6-10-targeted-measure",
                        "metric": metric,
                        "value": f"margin {m * 1000:g} ms, {p} parts",
                        "proposed": (m, p) == (0.025, 3),
                        "mean": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n_subjects": int(np.isfinite(vv).sum()),
                    }
                )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, grid, ttl in (
        (axes[0], grid_syn, "Known truth: % within ± 10 ms of the true latency"),
        (axes[1], grid_real, "Real data: % within ± 10 ms of the published latency"),
    ):
        im = ax.imshow(grid, cmap="Greys", vmin=50, vmax=100, aspect="auto")
        for a in range(len(PARTS)):
            for b in range(len(MARGINS)):
                ax.text(
                    b,
                    a,
                    f"{grid[a, b]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if grid[a, b] > 80 else "black",
                )
        ax.set_xticks(range(len(MARGINS)), [f"{m * 1000:g}" for m in MARGINS])
        ax.set_yticks(range(len(PARTS)), PARTS)
        ax.set_xlabel("window margin (ms)")
        ax.set_ylabel("number of parts")
        ax.set_title(ttl, fontsize=9)
        ax.add_patch(
            plt.Rectangle(
                (MARGINS.index(0.025) - 0.5, PARTS.index(3) - 0.5), 1, 1, fill=False, ec=RED, lw=2
            )
        )
        fig.colorbar(im, ax=ax, fraction=0.04)
    e_syn = (syn.model_area - syn.truth_area).abs() * 1000
    e_split = (odd.model_area - even.model_area.reindex(odd.index)).abs() * 1000
    keep_syn = syn.subject.isin(conf)
    keep_split = odd.subject.isin(conf)
    tol = np.arange(0, 41)
    axes[2].plot(
        tol,
        [np.mean(e_syn[keep_syn].dropna() <= x) * 100 for x in tol],
        color="black",
        label="known truth: |error|",
    )
    axes[2].plot(
        tol,
        [np.mean(e_split[keep_split].dropna() <= x) * 100 for x in tol],
        "--",
        color="black",
        label="real data: |odd − even|",
    )
    q95 = float(np.nanpercentile(e_syn[keep_syn], 95))
    axes[2].axvline(10, color=RED, lw=2, alpha=0.7, label="proposed tolerance 10 ms")
    axes[2].axvline(
        q95, color=BLUE, lw=1, ls=":", label=f"95th percentile of the error: {q95:.1f} ms"
    )
    axes[2].set_xlabel("tolerance (ms)")
    axes[2].set_ylabel("% of measurements within the tolerance")
    axes[2].set_title("Tolerance (proposed measure)", fontsize=9)
    axes[2].legend(fontsize=7, frameon=False, loc="lower right")
    TABLES.append(
        {
            "figure": "6-10-targeted-measure",
            "metric": "synthetic_error_95th_percentile_ms",
            "value": "margin 25 ms, 3 parts",
            "mean": q95,
        }
    )
    fig.suptitle(
        f"Targeted ERP measure — ERP CORE, {len(subs)} subjects × 7 components (confirmation half: {len(conf)})",
        fontsize=11,
    )
    fig.text(
        0.01,
        0.005,
        "Cells: mean over confirmation subjects of the % of their waves within ± 10 ms. Red box: proposed (± 25 ms, 3 parts).",
        fontsize=6.5,
        color="0.3",
    )
    save(
        fig,
        "6-10-targeted-measure",
        "Targeted measure of the paradigm's component: window margin 0 to 60 ms and 1 to 4 parts. Known truth "
        "(component added to the subject's plus-minus noise) and real data (published 50 % area latency); right: "
        "the tolerance against the distribution of the measurement error.",
    )


def erp_full_figure(sweep, name, title, xlabel):
    path = os.path.join(RES, f"{sweep}.pkl")
    if not os.path.exists(path):
        return
    blob = pickle.load(open(path, "rb"))
    values, proposed = blob["values"], blob["proposed"]
    df = pd.DataFrame(blob["records"])
    subs = sorted(df.subject.unique(), key=int)
    dev, conf = C.split(subs)
    syn = df[df.kind == "synth"]
    g = syn.groupby(["subject", "value_index"])
    tp, fp, fn, nt = g.tp.sum(), g.fp.sum(), g.fn.sum(), g.n_true.sum()
    ps = {
        "recall": (tp / nt * 100).unstack(),
        "false_synth": g.fp.mean().unstack(),
        "f1": (2 * tp / (2 * tp + fp + fn) * 100).unstack(),
        "lat_mae": g.lat_err.apply(
            lambda s: np.mean([x for v in s for x in v]) if sum(map(len, s)) else np.nan
        ).unstack(),
        "false_null": df[df.kind == "null"]
        .groupby(["subject", "value_index"])
        .n_found.mean()
        .unstack(),
        "n_real": df[df.kind == "real"]
        .groupby(["subject", "value_index"])
        .n_found.mean()
        .unstack(),
    }
    best = int(np.nanargmax(ps["f1"].loc[dev].mean().reindex(range(len(values))).to_numpy()))
    metrics = [
        ("recall", "Known truth: components recovered", "recall (%)"),
        ("false_synth", "Known truth: false components", "false components per wave"),
        ("lat_mae", "Known truth: latency error", "mean |latency error| (ms)"),
        ("false_null", "Real noise alone: false components", "components per wave (all false)"),
        ("n_real", "Real data: components kept", "components per wave"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (metric, ttl, yl) in zip(axes.ravel(), metrics, strict=False):
        rows = summarise(ps[metric].loc[conf], values, proposed, name, metric)
        panel(ax, rows, values, proposed, best, ttl, yl, xlabel)
    summarise(ps["f1"].loc[conf], values, proposed, name, "f1")
    axes[1, 2].axis("off")
    fig.suptitle(
        f"{title} — ERP CORE stimulus-locked waves, {len(subs)} subjects (confirmation half: {len(conf)})",
        fontsize=11,
    )
    legend_note(fig)
    save(
        fig,
        name,
        f"{title}, with the full erp preset on the 5 stimulus-locked ERP CORE difference waves. Known truth: "
        "0 to 3 components added to the subject's plus-minus noise (10 per wave); real noise: the plus-minus noise; real "
        "data: the published difference waves.",
    )


def erp_trials_figure():
    path = os.path.join(RES, "erp_trials.pkl")
    if not os.path.exists(path):
        return
    from etape6_erp import JITTERS, SHIFT_FACTORS

    df = pd.DataFrame(pickle.load(open(path, "rb"))["records"])
    subs = sorted(df.subject.unique(), key=int)
    dev, conf = C.split(subs)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    syn = df[df.kind == "synth"]
    proposed = SHIFT_FACTORS.index(0.5)
    for j, jit in enumerate([x * 1000 for x in JITTERS]):
        s = syn[syn.jitter_true == jit]
        per = s.groupby(["subject", "factor"]).rmse.mean().unstack().reindex(columns=SHIFT_FACTORS)
        per.columns = range(len(SHIFT_FACTORS))
        rows = summarise(
            per.reindex(conf),
            SHIFT_FACTORS,
            proposed,
            "6-12-trial-by-trial",
            f"synthetic_shift_rmse_ms_jitter_{jit:g}ms",
            "ms",
        )
        m = [r["mean"] for r in rows]
        axes[0].errorbar(
            np.arange(len(SHIFT_FACTORS)) + 0.05 * j,
            m,
            yerr=[
                [r["mean"] - r["ci_low"] for r in rows],
                [r["ci_high"] - r["mean"] for r in rows],
            ],
            fmt="o-",
            ms=3,
            capsize=2,
            label=f"true jitter SD {jit:g} ms",
        )
        per = (
            s.groupby(["subject", "factor"]).jitter.mean().unstack().reindex(columns=SHIFT_FACTORS)
        )
        per.columns = range(len(SHIFT_FACTORS))
        rows = summarise(
            per.reindex(conf),
            SHIFT_FACTORS,
            proposed,
            "6-12-trial-by-trial",
            f"synthetic_found_jitter_ms_jitter_{jit:g}ms",
            "ms",
        )
        axes[1].plot(
            np.arange(len(SHIFT_FACTORS)),
            [r["mean"] for r in rows],
            "o-",
            ms=3,
            label=f"true {jit:g} ms",
        )
        axes[1].axhline(jit, color=axes[1].lines[-1].get_color(), ls=":", lw=0.8)
    real = df[df.kind == "real"]
    per = (
        real.groupby(["subject", "factor"])
        .at_bound.mean()
        .unstack()
        .reindex(columns=SHIFT_FACTORS)
    )
    per.columns = range(len(SHIFT_FACTORS))
    rows = summarise(
        per.reindex(conf),
        SHIFT_FACTORS,
        proposed,
        "6-12-trial-by-trial",
        "real_trials_at_bound_pct",
        "%",
    )
    panel(
        axes[2],
        rows,
        SHIFT_FACTORS,
        proposed,
        None,
        "Real trials: share of trials at the bound",
        "% of trials",
        "maximum shift (× FWHM)",
    )
    for ax in axes[:2]:
        ax.set_xticks(range(len(SHIFT_FACTORS)), [f"{f:g}" for f in SHIFT_FACTORS])
        ax.set_xlabel("maximum shift (× FWHM)")
        ax.axvline(proposed, color=RED, alpha=0.25, lw=8)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("RMS error of the single-trial shifts (ms)")
    axes[0].set_title("Known truth: error of the shifts", fontsize=9)
    axes[1].set_ylabel("SD of the shifts found (ms); dotted: true SD")
    axes[1].set_title("Known truth: jitter found vs true jitter", fontsize=9)
    n_tr = int(df.groupby("key").n_trials.first().median())
    fig.suptitle(
        f"Trial by trial: maximum single-trial shift — ERP CORE P3, N170, N400, MMN, {len(subs)} subjects "
        f"(median {n_tr} trials; confirmation half: {len(conf)})",
        fontsize=11,
    )
    legend_note(fig)
    save(
        fig,
        "6-12-trial-by-trial",
        "Shift-and-scale fit of single trials with a maximum shift of 0.1 to 2 × FWHM (proposed for scalp data: 0.5). "
        "Known truth: the subject's single-trial residuals plus a component with a known shift per trial (jitter SD 0 "
        "to 20 ms); real data: the trials of the condition carrying the component.",
    )


# ------------------------------------------------------------------------------------------
# TEP windows


def tep_figure():
    import glob

    files = sorted(glob.glob(os.path.join(C.WORK, "tep", "*.npz")))
    if not files:
        return
    rows = []
    for f in files:
        z = np.load(f)
        r = C.fit(z["times"], z["wave"], "tep")
        for c in r["components"]:
            if c["amplitude"] < 0:
                rows.append({"subject": os.path.basename(f)[:-4], "latency": c["latency"] * 1000})
    df = pd.DataFrame(rows)
    subs = sorted(os.path.basename(f)[:-4] for f in files)
    dev, conf = C.split(subs)
    ranges = {"N45": (25, 70), "N100": (70, 200)}
    windows = {
        "proposed": {"N45": (38, 55), "N100": (80, 140)},
        "narrow": {"N45": (40, 50), "N100": (90, 120)},
        "wide": {"N45": (30, 60), "N100": (70, 160)},
    }
    derived = {}
    for comp, (lo, hi) in ranges.items():
        lat = (
            df[df.subject.isin(dev) & df.latency.between(lo, hi)].groupby("subject").latency.min()
        )
        derived[comp] = (
            (float(np.percentile(lat, 5)), float(np.percentile(lat, 95)))
            if len(lat) >= 3
            else (np.nan, np.nan)
        )
    windows["from development half (5–95 %)"] = derived
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for i, s in enumerate(subs):
        lats = df[df.subject == s].latency
        axes[0].plot(lats, [i] * len(lats), "o", ms=3.5, color="black" if s in conf else "0.6")
    for comp in ranges:
        lo, hi = windows["proposed"][comp]
        axes[0].axvspan(lo, hi, color=RED, alpha=0.12)
        if np.isfinite(derived[comp][0]):
            for e in derived[comp]:
                axes[0].axvline(e, color=BLUE, ls=":", lw=1)
    axes[0].set_yticks(
        range(len(subs)), [s + (" (dev)" if s in dev else "") for s in subs], fontsize=6
    )
    axes[0].set_xlim(0, 300)
    axes[0].set_xlabel("latency of the negative model components (ms)")
    axes[0].set_title(
        "Negative components per subject (grey: development half)\nred: proposed windows; blue: from the development half",
        fontsize=9,
    )
    names = list(windows)
    x = np.arange(len(names))
    for j, comp in enumerate(ranges):
        means, los, his = [], [], []
        for n in names:
            lo, hi = windows[n][comp]
            v = [
                float(any(lo <= l <= hi for l in df[df.subject == s].latency)) * 100 for s in conf
            ]
            m, a, b = C.boot_ci(v)
            means.append(m), los.append(m - a), his.append(b - m)
            TABLES.append(
                {
                    "figure": "6-11-tep-windows",
                    "metric": f"{comp}_subjects_with_component_pct",
                    "value": f"{n}: {lo:.0f}-{hi:.0f} ms",
                    "mean": m,
                    "ci_low": a,
                    "ci_high": b,
                    "n_subjects": len(conf),
                }
            )
        axes[1].bar(
            x + (j - 0.5) * 0.38,
            means,
            0.38,
            yerr=[los, his],
            capsize=3,
            color=["0.6", "0.2"][j],
            label=comp,
        )
    axes[1].set_xticks(x, [n.replace(" (", "\n(") for n in names], fontsize=7)
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("% of subjects with a negative\ncomponent in the window")
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].set_title("Coverage of each window (confirmation half, 95 % CI)", fontsize=9)
    fig.suptitle(
        f"TEP windows N45 and N100 — ds002094, {len(subs)} subjects, published cleaning",
        fontsize=11,
    )
    save(
        fig,
        "6-11-tep-windows",
        "Windows of the TEP components. The negative components of the tep preset fit on each subject; a data-driven "
        "window is taken on the development half (5th to 95th percentile of the earliest negative component in 25-70 "
        "and 70-200 ms) and every window is scored on the confirmation half.",
    )


def main():
    warnings.simplefilter("ignore")
    for sweep in CCEP_TEXT:
        ccep_figure(sweep)
    erp_full_figure(
        "erp_delta",
        "6-04b-delta-bic-erp",
        "Rule to keep one more component: ΔBIC ≤ −threshold",
        "|ΔBIC| threshold",
    )
    erp_full_figure(
        "erp_reject_cancelling",
        "6-07b-cancelling-pairs-erp",
        "Reject cancelling pairs of opposite-sign components",
        "reject cancelling pairs",
    )
    erp_measure_figures()
    erp_trials_figure()
    tep_figure()
    os.makedirs(C.OUT, exist_ok=True)
    pd.DataFrame(TABLES).to_csv(os.path.join(C.OUT, "etape6_numbers.csv"), index=False)
    json.dump(
        {"captions": CAPTIONS, "environment": C.environment()},
        open(os.path.join(C.OUT, "etape6_meta.json"), "w"),
        indent=1,
        ensure_ascii=False,
    )


if __name__ == "__main__":
    main()
