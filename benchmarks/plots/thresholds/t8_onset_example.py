"""T8 with an example: what "onset at 10 % / 50 % of the peak" means, then the error over patients.

(a) One contact of the first patient: a known N1 added to real pre-pulse noise (common.inject),
    the trial average, the true N1 and the N1 fitted by evoparam (ccep). Onsets read on both
    curves at 10 % and at 50 % of their own peak; the gap is the onset error.
(b) Mean over 51 patients of the per-patient median |onset error| (t8_onset_fraction.csv).
"""
import os

import numpy as np
import pandas as pd

import common as c

CSV = os.path.join(c.HERE, "t8_onset_fraction.csv")
CONTACT = 26  # moderate example: N1 of -128 µV, 11 ms wide
GRID = np.arange(0.0, 0.08, 0.00005)


def onset(curve, f):
    a = np.abs(curve)
    i = int(np.argmax(a))
    below = np.flatnonzero(a[: i + 1] < f * a[i])
    return GRID[below[-1] + 1]


def example():
    from evoparam import fit_response
    from evoparam.shapes import get_shape

    rng = np.random.default_rng(20261102)
    s = next(c.subjects().itertuples())
    sim, truth = c.inject(c.trials(s), rng)
    avg = c.average(sim)[CONTACT]
    row = truth.iloc[CONTACT]
    fit = fit_response(sim.times, avg, preset="ccep", random_state=0)
    b = [b for b in fit.components if b.label == "N1"][0]
    curve = get_shape(b.shape).evaluate(GRID, np.asarray(b.params))
    fitted = curve / np.abs(curve).max() * b.amplitude * np.sign(curve[np.argmax(np.abs(curve))])
    fitted = -np.abs(fitted) if b.amplitude < 0 else fitted
    sd = row.n1_fwhm / 2.3548
    true = row.n1_amplitude * np.exp(-0.5 * ((GRID - row.n1_latency) / sd) ** 2)
    return sim.times, avg, true, fitted


def figure():
    plt = c.style()
    from matplotlib import patheffects

    halo = [patheffects.withStroke(linewidth=3.0, foreground="white", alpha=0.9)]
    t, avg, true, fitted = example()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw=dict(width_ratios=(1.5, 1)))
    m = (t >= 0) & (t <= 0.08)
    ax.plot(1e3 * t[m], avg[m], color="0.6", lw=0.8, zorder=1)
    ax.plot(1e3 * GRID, true, color="black", lw=1.4, ls=(0, (2.0, 1.4)), zorder=2)
    ax.plot(1e3 * GRID, fitted, color=c.BLUE, lw=1.6, zorder=3, path_effects=halo)
    ax.text(52, 12, "simulated N1 + real noise (average of trials)", fontsize=6.5, color="0.45")
    ax.text(33, true.min() * 0.97, "true N1 (what was added)", fontsize=6.5, color="black", va="center")
    ax.text(34, true.min() * 0.6, "fitted N1 (model fit\nto the grey trace)", fontsize=7, color=c.BLUE, va="center")
    for f in (0.1, 0.5):
        ot, of = onset(true, f), onset(fitted, f)
        yt, yf = f * true.min(), f * fitted.min()
        ax.axhline(yt, color="0.7", lw=0.6, ls=":", zorder=0)
        ax.text(79, yt - 2, f"{100 * f:g} % of the true peak", ha="right", va="top", fontsize=6.5, color="0.4")
        ax.plot([1e3 * ot], [yt], "o", color="black", ms=4, mec="white", mew=0.8, zorder=4)
        ax.plot([1e3 * of], [yf], "o", color=c.BLUE, ms=4, mec="white", mew=0.8, zorder=4)
        ax.annotate(f"at {100 * f:g} %: error {1e3 * abs(of - ot):.1f} ms", (1e3 * min(ot, of), (yt + yf) / 2), xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=6.5, color="0.15", path_effects=halo)
    ax.axhline(0, color="0.3", lw=0.5)
    ax.plot([47], [-95], "o", color="black", ms=4, mec="white", mew=0.8)
    ax.text(49, -95, "true onset", va="center", fontsize=6.5)
    ax.plot([47], [-107], "o", color=c.BLUE, ms=4, mec="white", mew=0.8)
    ax.text(49, -107, "fitted onset", va="center", fontsize=6.5, color=c.BLUE)
    ax.text(79, 1.5, "0 \u00b5V", ha="right", fontsize=6, color="0.4")
    ax.set(xlim=(0, 80), ylim=(-140, 25), xlabel="time after the pulse (ms)", ylabel="voltage (\u00b5V)")
    ax.tick_params(top=False, right=False, which="both")
    ax.set_title("a. One contact: where does the N1 start?", loc="left", fontsize=8.5)
    d = pd.read_csv(CSV)
    for k, (w, color) in enumerate((("n1", c.BLUE), ("n2", c.GREEN))):
        tab = d[d.wave == w].pivot_table(index="subject", columns="fraction", values="onset_err_ms")
        for j, f in enumerate((0.1, 0.5)):
            v = tab[f].mean()
            lo, hi = c.boot_ci(tab[f])
            xpos = 3 * k + j
            bx.bar(xpos, v, color="white" if f == 0.1 else color, edgecolor=color, hatch="////" if f == 0.1 else None, lw=1.0, width=0.8)
            bx.errorbar(xpos, v, yerr=[[v - lo], [hi - v]], color="0.2", lw=0.8, capsize=2)
            bx.text(xpos, hi + 0.12, f"{v:.1f} ms", ha="center", fontsize=7)
            bx.text(xpos, -0.25, f"{100 * f:g} %" + ("\n(used now)" if f == 0.1 else ""), ha="center", va="top", fontsize=7)
        better = int((tab[0.5] < tab[0.1]).sum())
        bx.text(3 * k + 0.5, 7.9, f"50 % better in\n{better} of {len(tab)} patients", ha="center", va="top", fontsize=6.5, color=color)
        bx.text(3 * k + 0.5, -1.6, w.upper(), ha="center", va="top", fontsize=8, fontweight="bold", color=color)
    bx.set_xticks([])
    bx.set_xlim(-0.7, 4.7)
    bx.set_ylim(0, 8)
    bx.set_ylabel("onset error (ms)")
    bx.spines[["top", "right"]].set_visible(False)
    bx.tick_params(top=False, right=False, which="both")
    bx.set_title("b. Onset error, 51 patients (mean, 95 % CI)", loc="left", fontsize=8.5)
    fig.suptitle("The onset is closer to the truth when read at 50 % of the peak than at 10 %", x=0.01, ha="left", fontsize=9)
    fig.text(0.01, -0.03, "N1, N2: early (9-50 ms) and late (50-300 ms) negative peaks after the pulse. Onset: first time the curve reaches "
             "the given % of its own peak.\nError: |fitted onset - true onset| in ms. b: simulated N1 and N2 added to real recorded noise; "
             "median over contacts per patient, then mean of 51 patients; bars: 95 % confidence interval.",
             fontsize=7, color="0.15", va="top")
    fig.tight_layout(w_pad=2)
    c.save(fig, "T8_onset_example")


if __name__ == "__main__":
    figure()
