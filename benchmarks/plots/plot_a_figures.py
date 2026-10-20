"""Plot A: N1 detection threshold k (x SD of baseline) -> % of contacts with an N1, ds004080."""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from cmcrameri import cm
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # inputs and CSVs live next to the script

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "savefig.dpi": 300, "figure.dpi": 150})
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.join(ROOT, "figures", "plot-A")
os.makedirs(OUT, exist_ok=True)
C = {"paper": "#1b6cae", "code": "#c4632a"}
LAB = {"paper": "Published rule as written (|x| > k SD, peak in 9-50 ms)",
       "code": "Published code (SD floor 50 µV, 20 µV peak selectivity)"}
K0 = 3.4
rng = np.random.default_rng(0)
d = pd.read_csv("plotA_rule_threshold.csv")
n_sub = d.subject.nunique()


def boot_ci(x, n=5000):
    x = np.asarray(x)
    m = rng.choice(x, (n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


# A1: threshold sweep, every patient + mean and 95% bootstrap CI over patients
fig, ax = plt.subplots(figsize=(3.6, 2.7))
for s, g in d[d.reading == "paper"].groupby("subject"):
    ax.plot(g.k, g.pct, color="0.82", lw=0.5, zorder=1)
for r in ("paper", "code"):
    g = d[d.reading == r].groupby("k").pct
    ks = np.array(sorted(d.k.unique()))
    ci = np.array([boot_ci(d[(d.reading == r) & (d.k == k)].pct) for k in ks])
    ax.fill_between(ks, ci[:, 0], ci[:, 1], color=C[r], alpha=0.22, lw=0, zorder=2)
    ax.plot(ks, g.mean().values, color=C[r], lw=1.6, zorder=3)
    y0 = g.mean().loc[K0]
    ax.plot([K0], [y0], "o", ms=4.5, color=C[r], mec="white", mew=0.8, zorder=4)
    ax.annotate(f"{y0:.1f} %", (K0, y0), xytext=(6, 4 if r == "paper" else -9),
                textcoords="offset points", fontsize=7, color="0.15")
ax.axvline(K0, color="0.3", ls="--", lw=0.7, zorder=0)
ax.text(K0 + 0.08, 63, "published k = 3.4", fontsize=7, color="0.3")
ax.text(6.3, 17.5, "rule as written", color=C["paper"], fontsize=7)
ax.text(6.3, 1.2, "published code", color=C["code"], fontsize=7)
ax.text(1.15, 66, "grey: one patient\n(rule as written)", fontsize=6, color="0.45", va="top")
ax.set(xlabel="Detection threshold k (× SD of the pre-stimulus baseline)",
       ylabel="Contacts with an N1 (%)", xlim=(1, 10), ylim=(0, 70))
ax.set_title(f"N1 detection vs threshold, ds004080 ({n_sub} patients)", loc="left")
save(fig, "A1_threshold_sweep")

# A2: ridgeline of the per-patient % at a few thresholds (rule as written)
sel = [2.0, 2.6, 3.0, 3.4, 4.0, 5.0, 6.0]
from scipy.stats import gaussian_kde
fig, ax = plt.subplots(figsize=(3.4, 3.3))
grid = np.linspace(0, 80, 400)
for i, k in enumerate(sel[::-1]):
    x = d[(d.reading == "paper") & (np.isclose(d.k, k))].pct.values
    dens = gaussian_kde(x, bw_method=0.35)(grid)
    dens = dens / dens.max() * 0.9
    base = i
    col = cm.batlow(0.15 + 0.7 * (len(sel) - 1 - i) / (len(sel) - 1))
    ax.fill_between(grid, base, base + dens, color=col, alpha=0.9, lw=0, zorder=10 - i)
    ax.plot(grid, base + dens, color="white", lw=0.8, zorder=10 - i)
    ax.plot([np.median(x)] * 2, [base, base + 0.9 * np.interp(np.median(x), grid, dens / 0.9)],
            color="0.1", lw=0.8, zorder=11)
    ax.text(-1.5, base + 0.15, f"k = {k}" + ("  (published)" if k == K0 else ""), ha="right",
            fontsize=7, fontweight="bold" if k == K0 else "normal")
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", which="both", left=False, right=False)
ax.set(xlabel="Contacts with an N1, per patient (%)", xlim=(0, 80))
ax.set_title(f"Distribution over {n_sub} patients (rule as written)\nblack tick: median",
             loc="left")
save(fig, "A2_ridgeline_per_patient")

# A3: paired estimation plot (DABEST) of k = 2.6 and k = 4.2 against the published 3.4
import dabest
w = (d[d.reading == "paper"].pivot(index="subject", columns="k", values="pct")
     .rename(columns=lambda k: f"k = {k}")[["k = 3.4", "k = 2.6", "k = 4.2"]].reset_index())
long = w.melt(id_vars="subject", var_name="threshold", value_name="Contacts with an N1 (%)")
db = dabest.load(long, x="threshold", y="Contacts with an N1 (%)",
                 idx=("k = 3.4", "k = 2.6", "k = 4.2"),
                 paired="baseline", id_col="subject", resamples=5000, random_seed=0)
fig = db.mean_diff.plot(raw_marker_size=1.5, fig_size=(5.2, 4.2), delta_text=False,
                        custom_palette={"k = 3.4": C["paper"], "k = 2.6": "#555555",
                                        "k = 4.2": "#555555"})
fig.suptitle(f"Changing k from 3.4: paired difference per patient ({n_sub})", x=0.02,
             ha="left", fontsize=9)
save(fig, "A3_paired_estimation")
res = db.mean_diff.results[["control", "test", "difference", "bca_low", "bca_high", "pvalue_permutation"]]
res.to_csv(f"{OUT}/A3_paired_differences.csv", index=False)
print(res.round(3).to_string())

# A4: example patient, stacked averages (Miller et al. style), N1 marked at k = 3.4
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "evoparam", "tests", "models"))
import test_engine_ccep as t
from evoparam.existence.rules import apply_rule, resolve_rule
ex = "sub-ccepAgeUMCU01"
row = pd.read_csv("ds004080_subjects.tsv", sep="\t", dtype=str).set_index("subject").loc[ex]
ts = t._pair_trials(ex[4:], row.run, row.pair)
avg = t._rule_average(ts)
tt = ts.times
det = {}
for k in (2.6, 3.4):
    spec = resolve_rule("vanblooijs2023", baseline=(-1.0, -0.1), peak="strict", k=k, window=(t.N1_START, t.N1_END))
    o = apply_rule(avg, tt, ts.sfreq, spec, "uV")
    det[k] = (np.asarray(o["significant"], bool), o["onset"])
amp = np.abs(avg[:, (tt > 0.009) & (tt < 0.1)]).max(1)
hit = np.flatnonzero(det[2.6][0])
rest = [i for i in np.argsort(amp) if i not in hit][: max(0, 16 - len(hit))]
order = np.array(sorted(list(hit) + list(rest), key=lambda i: ts.ch_names[i]))
fig, ax = plt.subplots(figsize=(3.4, 5.0))
win = (tt > -0.05) & (tt < 0.25)
masked = avg.copy()
masked[:, (tt > -0.002) & (tt < 0.009)] = np.nan  # stimulation artifact, not drawn
step = 1000
for j, i in enumerate(order):
    y = -j * step
    ax.plot(tt[win] * 1000, masked[i, win] + y, color="black", lw=0.7)
    ax.text(-58, y, ts.ch_names[i], ha="right", va="center", fontsize=6)
    for k, mk in ((3.4, dict(marker="o", ms=4, color=C["paper"])),
                  (2.6, dict(marker="o", ms=4, mfc="none", mec="#555555", mew=0.8))):
        s, on = det[k]
        if s[i] and (k == 3.4 or not det[3.4][0][i]):
            v = np.interp(on[i], tt, avg[i])
            ax.plot(on[i] * 1000, v + y, ls="none", **mk, zorder=5)
ax.axvspan(9, 50, color="0.93", zorder=0, lw=0)
ax.axvspan(-2, 9, color="#f3d9c9", zorder=0, lw=0)
ax.axvline(0, color="0.5", lw=0.6)
ax.plot([230, 230], [-len(order) * step, -len(order) * step + 500], color="black", lw=1)
ax.text(222, -len(order) * step + 250, "500 µV", fontsize=6, va="center", ha="right")
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", which="both", left=False, right=False)
ax.set(xlabel="Time from stimulation (ms)", xlim=(-50, 250))
ax.set_title(f"{ex[4:]}, stimulation {row.pair}: averaged CCEPs\n"
             "filled: N1 at k = 3.4; open: only at k = 2.6\ngrey: N1 window 9-50 ms; pink: artifact (-2 to 9 ms, not drawn)",
             loc="left", fontsize=7.5)
save(fig, "A4_example_patient_traces")
print("saved", sorted(os.listdir(OUT)))
