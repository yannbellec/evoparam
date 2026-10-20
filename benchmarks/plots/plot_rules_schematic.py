"""Schematics of the two readings of the published N1 rule (van Blooijs et al. 2023), on one made-up average."""
import os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from scipy.ndimage import gaussian_filter1d
os.chdir(os.path.dirname(os.path.abspath(__file__)))

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "savefig.dpi": 300, "figure.dpi": 150})
OUT = os.path.join("..", "..", "figures", "rules-schematic")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GREY = "#1b6cae", "#c4632a", "0.55"
K = 3.4

# One made-up trial average (uV): smooth noise before the pulse, a stimulation artifact, an N1 at 22 ms
# (-120 uV), a small dip at 60 ms (10 uV deep) and an N2 bump.
fs = 2048
t = np.arange(-2.0, 0.15, 1 / fs)
rng = np.random.default_rng(3)
x = gaussian_filter1d(rng.normal(0, 1, t.size), 12)
x = 18 * x / x[t < -0.1].std()
g = lambda mu, w, a: a * np.exp(-0.5 * ((t - mu) / w) ** 2)
x += g(0.022, 0.005, -120) + g(0.060, 0.003, -12) + g(0.075, 0.012, 25) + g(0.115, 0.025, -45)
art = (t >= 0) & (t < 0.009)
x[art] = 900 * np.exp(-t[art] / 0.002) * np.cos(2 * np.pi * 300 * t[art])


def strict_minima(y):
    return np.flatnonzero((y[1:-1] < y[:-2]) & (y[1:-1] < y[2:])) + 1


def draw(name, title, base, level_kind, floor, select, color, steps):
    b = (t > base[0]) & (t < base[1])
    level = np.median(x[b]) if level_kind == "median" else x[b].mean()
    sd_raw = x[b].std()
    sd = max(sd_raw, floor) if floor else sd_raw
    thr = level - K * sd
    win = (t >= 0.009) & (t <= 0.100)
    idx = strict_minima(x)
    idx = idx[win[idx]]
    if select:  # keep only dips at least `select` uV below the higher of their two side maxima
        keep = []
        for i in idx:
            left = x[max(0, i - 200):i].max()
            right = x[i:i + 200].max()
            if min(left, right) - x[i] >= select:
                keep.append(i)
        rejected, idx = np.setdiff1d(idx, keep), np.array(keep, int)
    else:
        rejected = np.array([], int)
    best = idx[np.argmin(x[idx])]
    found = x[best] < thr

    fig = plt.figure(figsize=(7.2, 4.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.2], height_ratios=[3, 1.25], wspace=0.05, hspace=0.35)
    a0, a1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    tx = fig.add_subplot(gs[1, :]); tx.axis("off")
    ylim = (-175 if not floor else -200, 90)
    pre = t < 0
    a0.plot(t[pre], x[pre], color="0.15", lw=0.6)
    a0.axvspan(*base, color=color, alpha=0.12, lw=0)
    a0.axhline(level, color=color, lw=0.9)
    for s in (+1, -1):
        a0.axhline(level + s * sd_raw, color=color, lw=0.6, ls=(0, (2, 2)))
    a0.text(np.mean(base), 70, f"baseline\n{base[0]:g} to {base[1]:g} s", ha="center", va="top", fontsize=6.5, color=color)
    a0.annotate("", xy=(base[0] + 0.05, level + sd_raw), xytext=(base[0] + 0.05, level - sd_raw),
                arrowprops=dict(arrowstyle="<->", color=color, lw=0.8))
    a0.text(base[0] + 0.08, level + sd_raw + 4, f"±1 SD (SD = {sd_raw:.0f} µV)", fontsize=6.5, color=color, va="bottom", bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.6))
    if floor:
        a0.axhline(level - floor, color=color, lw=1.0, ls="--")
        a0.text(np.mean(base), level - floor - 4, f"SD floor = {floor:g} µV (used instead)", ha="center", va="top", fontsize=6.5, color=color, bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.6))
    a0.set(xlim=(t[0] if base[0] < -1.5 else -1.05, 0), ylim=ylim, xlabel="Time (s)", ylabel="Trial average (µV)")
    a0.set_title("1. Measure the noise before the pulse", loc="left", fontsize=8)

    r = (t >= -0.005) & (t <= 0.14)
    ms = t * 1000
    a1.plot(ms[r & ~art], np.where(art, np.nan, x)[r & ~art], color="0.15", lw=0.9)
    a1.plot(ms[art], np.clip(x[art], *ylim), color=GREY, lw=0.6)
    a1.axvspan(0, 9, color=GREY, alpha=0.25, lw=0)
    a1.text(4.5, -25, "pulse\nartifact\n(ignored)", ha="center", va="top", fontsize=6, color="0.3", bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.6))
    a1.axvspan(9, 100, color=color, alpha=0.07, lw=0)
    a1.text(54, 80, "N1 search window, 9 to 100 ms", ha="center", va="top", fontsize=6.5, color=color)
    a1.axvline(50, color="0.4", lw=0.5, ls=":")
    a1.text(50.8, 60, "evoparam's N1\nwindow ends at 50 ms", fontsize=5.5, color="0.4", va="top")
    a1.axhline(level, color=color, lw=0.9)
    a1.axhline(thr, color=color, lw=1.2, ls="--")
    a1.text(139, thr + 3, f"threshold = level − {K} × SD = {thr:.0f} µV", ha="right", va="bottom", fontsize=6.5, color=color, bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.6))
    a1.plot(ms[idx], x[idx], "o", ms=3.5, mfc="white", mec="0.2", mew=0.7)
    if rejected.size:
        a1.plot(ms[rejected], x[rejected], "x", ms=4.5, color=ORANGE, mew=1)
        i = rejected[0]
        a1.annotate(f"dip < {select:g} µV deep:\nnot counted", (ms[i], x[i]), xytext=(ms[i] + 4, x[i] - 45),
                    fontsize=6, color=ORANGE, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.5))
    a1.plot(ms[best], x[best], "o", ms=6, color=BLUE if found else ORANGE, mec="white", mew=0.8, zorder=5)
    a1.annotate("", xy=(ms[best] - 3, x[best]), xytext=(ms[best] - 3, level),
                arrowprops=dict(arrowstyle="<->", color="0.2", lw=0.8))
    verdict = "N1: YES" if found else "N1: NO"
    a1.annotate(f"deepest trough, {level - x[best]:.0f} µV deep\n(arrow: depth below the level)\n{verdict}",
                (ms[best], x[best]), xytext=(48, -118), fontsize=7, va="center", color=BLUE if found else ORANGE,
                fontweight="bold", arrowprops=dict(arrowstyle="-", color="0.4", lw=0.5))
    a1.set(xlim=(-5, 140), ylim=ylim, xlabel="Time after the pulse (ms)")
    a1.tick_params(labelleft=False)
    a1.set_title("2. Find the deepest trough after the pulse, compare it to k × SD", loc="left", fontsize=8)

    tx.text(0, 1, steps(sd_raw, sd, level, thr, x[best], found), va="top", fontsize=7, linespacing=1.45, family="DejaVu Sans")
    fig.suptitle(title, x=0.125, ha="left", fontsize=10, y=0.985)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    return sd_raw, sd, thr, x[best], found


paper = draw(
    "rule_as_written", "Rule as written in the paper (van Blooijs et al. 2023)", (-1.0, -0.1), "mean", None, None, BLUE,
    lambda r, s, l, th, v, f: (
        f"• Noise: SD of the average between −1 and −0.1 s = {s:.0f} µV.   Reference level = MEAN of that period.\n"
        f"• Trough: the lowest point between 9 and 100 ms that is lower than both of its neighbours (any depth counts; ○ = candidate troughs).\n"
        f"• Decision: N1 if the trough is deeper than k × SD below the level:  {K} × {s:.0f} = {K*s:.0f} µV.\n"
        f"• Here: trough {l - v:.0f} µV deep > {K*s:.0f} µV  →  N1 {'YES' if f else 'NO'}."))
code = draw(
    "published_code", "Published code (mnl_ccepAge, ccep_detect_n1peak_ECoG.m)", (-2.0, -0.1), "median", 50.0, 20.0, ORANGE,
    lambda r, s, l, th, v, f: (
        f"• Noise: SD of the average between −2 and −0.1 s = {r:.0f} µV, raised to a FLOOR of 50 µV → SD used = {s:.0f} µV.   "
        f"Reference level = MEDIAN.\n"
        f"• Trough: only dips at least 20 µV deep count (× = too shallow; first sample after 9 ms skipped; > 3000 µV refused as saturation).\n"
        f"• Decision: N1 if the trough is deeper than k × SD:  {K} × {s:.0f} = {K*s:.0f} µV (never less than 170 µV).\n"
        f"• Same average, here: trough {l - v:.0f} µV deep < {K*s:.0f} µV  →  N1 {'YES' if f else 'NO'}."))
print("paper", paper, "code", code)
