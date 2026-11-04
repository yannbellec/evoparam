"""T8, simple version: onset error when the onset is read at a given % of the peak.

Same data as t8_onset_fraction.py (t8_onset_fraction.csv). Per patient: median |fitted onset -
true onset| (ms) over found N1 / N2. Mean over patients, light 95 % bootstrap band.
"""
import os

import numpy as np
import pandas as pd

import common as c

CSV = os.path.join(c.HERE, "t8_onset_fraction.csv")
NOW = 0.10


def figure():
    plt = c.style()
    from matplotlib import patheffects

    halo = [patheffects.withStroke(linewidth=3.0, foreground="white", alpha=0.9)]
    d = pd.read_csv(CSV)
    fr = np.sort(d.fraction.unique())
    x = 100 * fr
    i_now = int(np.flatnonzero(np.isclose(fr, NOW))[0])
    means = {}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for ax, w, color in zip(axes, ("n1", "n2"), (c.BLUE, c.GREEN)):
        W = w.upper()
        tab = d[d.wave == w].pivot_table(index="subject", columns="fraction", values="onset_err_ms")[fr]
        mean = tab.mean().to_numpy()
        means[w] = (mean[i_now], mean[-1])
        ci = np.array([c.boot_ci(tab[v]) for v in fr])
        top = 1.35 * ci[:, 1].max()
        ax.axvspan(x[i_now] - 1.5, x[i_now] + 1.5, color="0.92", lw=0, zorder=0)
        ax.text(x[i_now], top * 0.97, "current\ndefault", ha="center", va="top", fontsize=6.5, color="0.35")
        ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.18, lw=0, zorder=1)
        ax.plot(x, mean, color=color, lw=1.6, marker="o", ms=4, mec="white", mew=0.8, zorder=3, path_effects=halo)
        for xi, m in zip(x, mean):
            ax.annotate((f"{m:.2f} ms" if w == "n1" else f"{m:.1f} ms"), (xi, m), xytext=(0, 7), textcoords="offset points", ha="center",
                        fontsize=7, color=color, path_effects=halo)
        ax.set_ylim(0, top)
        ax.set_xticks(x, [f"{100 * v:g} %" for v in fr])
        ax.set_xlim(0, 55)
        ax.minorticks_off()
        ax.set_xlabel(f"onset read where the {W} reaches this % of its peak\n← lower on the {W} slope          higher on the slope →")
        ax.set_ylabel(f"{W} onset error (ms)")
        ax.set_title(W, loc="left", fontsize=9, fontweight="bold")
    fig.suptitle("Reading the onset at 50 % instead of 10 % of the peak: error N1 {:.2f} → {:.2f} ms, N2 {:.1f} → {:.1f} ms (51 patients)".format(*means["n1"], *means["n2"]),
                 x=0.01, ha="left", fontsize=9)
    fig.text(0.01, -0.02, "Onset: first time the fitted N1 or N2 reaches the given % of its peak. Error: |fitted onset - true onset| in ms, both read at the same %.\n"
             "Truth: known N1 and N2 added to real recorded noise. Per patient: median over contacts. "
             "Line: mean of 51 patients. Band: 95 % confidence interval over patients.", fontsize=7, color="0.15", va="top")
    fig.tight_layout(w_pad=3)
    c.save(fig, "T8_onset_fraction_simple")


if __name__ == "__main__":
    figure()
