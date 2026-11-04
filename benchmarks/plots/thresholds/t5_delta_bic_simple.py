"""T5, simple version: what changes when evoparam asks for more evidence before keeping a bump.

Same data as t5_delta_bic.py (t5_delta.csv). Per patient: true N1 / N2 found per 100
responding contacts, false N1 / N2 per 100 empty contacts. Mean over patients, light
95 % bootstrap band over patients. The x axis is ordinal (tested values, evenly spaced).
"""
import os

import numpy as np
import pandas as pd

import common as c

CSV = os.path.join(c.HERE, "t5_delta.csv")
NOW = 10.0


def figure():
    plt = c.style()
    from matplotlib import patheffects

    halo = [patheffects.withStroke(linewidth=3.0, foreground="white", alpha=0.9)]
    d = pd.read_csv(CSV)
    deltas = np.sort(d.delta.unique())
    x = np.arange(len(deltas))
    i_now = int(np.flatnonzero(deltas == NOW)[0])
    rows = (("found", c.BLUE, "true {W} found\n(per 100 contacts with a known {W})", 100),
            ("invented", c.ORANGE, "false {W}\n(per 100 noise-only contacts)", 5))
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0), sharex=True, gridspec_kw=dict(height_ratios=(1.6, 1)))
    for j, w in enumerate(("n1", "n2")):
        W = w.upper()
        for i, (kind, color, ylab, top) in enumerate(rows):
            ax = axes[i, j]
            ax.axvspan(i_now - 0.3, i_now + 0.3, color="0.92", lw=0, zorder=0)
            tab = d.pivot_table(index="subject", columns="delta", values=f"{w}_{kind}")[deltas]
            mean = tab.mean().to_numpy()
            ci = np.array([c.boot_ci(tab[v]) for v in deltas])
            ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.18, lw=0, zorder=1)
            ax.plot(x, mean, color=color, lw=1.6, marker="o", ms=4, mec="white", mew=0.8, zorder=3, path_effects=halo)
            for xi, m in zip(x, mean):
                ax.annotate(f"{m:.0f}" if m >= 10 else f"{m:.1f}", (xi, m), xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=7, color=color, path_effects=halo)
            ax.set_ylim(0, top)
            ax.set_ylabel(ylab.format(W=W), fontsize=7.5)
            ax.minorticks_off()
            if i == 1:
                ax.text(0.98, 0.95, "y axis 0-5", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color="0.35")
            if i == 0:
                ax.set_title(W, loc="left", fontsize=9, fontweight="bold")
                ax.text(i_now, top * 0.97, "current\ndefault", ha="center", va="top", fontsize=6.5, color="0.35")
        axes[1, j].set_xticks(x, [f"{v:g}" for v in deltas])
        axes[1, j].set_xlim(-0.4, len(x) - 0.6)
        axes[1, j].set_xlabel("evidence needed to keep a bump (\u0394BIC)\n\u2190 less          more \u2192")
    fig.suptitle("\u0394BIC 10 \u2192 40: true N2 found 45 \u2192 32 %, false N2 2.1 \u2192 0.5 %, true N1 found 63 \u2192 56 % (51 patients)",
                 x=0.01, ha="left", fontsize=9)
    fig.text(0.01, -0.02, "N1, N2: early (9-50 ms) and late (50-300 ms) negative peaks of the evoked response. Bump: a peak the fitted model keeps. \u0394BIC: drop of the Bayesian information criterion\nneeded to keep a new bump. Known N1/N2 were added to real noise on half of the contacts; the other half are noise only. "
             "Line: mean of 51 patients. Band: 95 % confidence interval over patients.", fontsize=7, color="0.15", va="top")
    fig.tight_layout(w_pad=3)
    c.save(fig, "T5_delta_bic_simple")


if __name__ == "__main__":
    figure()
