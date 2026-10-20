"""How k is decided: sensitivity and false positives of the published N1 rule vs k (known N1s in real noise)."""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
os.chdir(os.path.dirname(os.path.abspath(__file__)))

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "savefig.dpi": 300, "figure.dpi": 150})
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.join(ROOT, "figures", "plot-K")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE = "#1b6cae", "#c4632a"
K0 = 3.4
rng = np.random.default_rng(0)
d = pd.read_csv("plotK_calibration.csv")
c = pd.read_csv("plotK_contacts.csv.gz")
n_sub = d.subject.nunique()
ks = np.array(sorted(d.k.unique()))


def boot_ci(x, n=5000):
    x = np.asarray(x)
    return np.percentile(rng.choice(x, (n, len(x))).mean(1), [2.5, 97.5])


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


m = d.groupby("k")[["sensitivity", "false_positive"]].mean()
j = m.sensitivity - m.false_positive
k_best = float(j.idxmax())

# K1: % of true N1s found and % of invented N1s, vs k; mean and 95% bootstrap CI over patients
fig, ax = plt.subplots(figsize=(3.8, 2.8))
for col, color in (("sensitivity", BLUE), ("false_positive", ORANGE)):
    ci = np.array([boot_ci(d[d.k == k][col]) for k in ks])
    ax.fill_between(ks, ci[:, 0], ci[:, 1], color=color, alpha=0.22, lw=0)
    ax.plot(ks, m[col].values, color=color, lw=1.6)
    for k, dy in ((K0, 5), (k_best, 5)):
        y = m[col].loc[k]
        ax.plot([k], [y], "o", ms=4, color=color, mec="white", mew=0.8, zorder=4)
        ax.annotate(f"{y:.1f} %", (k, y), xytext=(5, dy), textcoords="offset points", fontsize=6.5, color="0.15")
ax.axvline(K0, color="0.3", ls="--", lw=0.7)
ax.axvline(k_best, color="0.3", ls=":", lw=0.7)
ax.text(K0 + 0.1, 96, "published k = 3.4", fontsize=6.5, color="0.3", va="top")
ax.text(k_best - 0.1, 96, f"best compromise\nk = {k_best:.1f}", fontsize=6.5, color="0.3", va="top", ha="right")
ax.text(7.0, 33, "true N1s found", color=BLUE, fontsize=7)
ax.text(7.0, 5, "N1s invented on\nnoise-only contacts", color=ORANGE, fontsize=7)
ax.set(xlabel="Detection threshold k (× SD of the pre-stimulus baseline)", ylabel="Contacts (%)",
       xlim=(1, 10), ylim=(0, 100))
ax.set_title(f"Choosing k: known N1s in real noise ({n_sub} patients)", loc="left")
save(fig, "K1_sensitivity_false_positives")

# K2: chance to find an N1 vs its size relative to the noise (peak / baseline SD), at a few k
c = c[c.injected]
edges = np.array([0, 1, 2, 3, 4, 6, 10, 20, 1e4])
labels = ["<1", "1-2", "2-3", "3-4", "4-6", "6-10", "10-20", ">20"]
c = c.assign(bin=pd.cut(c.snr, edges, labels=labels))
fig, ax = plt.subplots(figsize=(3.8, 2.8))
for k, color, ls in ((2.0, "0.6", "-"), (k_best, "0.3", ":"), (K0, BLUE, "-"), (5.0, ORANGE, "-")):
    per = c[c.k == k].groupby(["bin", "subject"], observed=True).found.mean().mul(100).reset_index()
    mean = per.groupby("bin", observed=True).found.mean()
    ci = np.array([boot_ci(per[per.bin == b].found) for b in mean.index])
    x = np.arange(len(mean))
    ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.15, lw=0)
    ax.plot(x, mean.values, color=color, ls=ls, lw=1.4, marker="o", ms=2.5, label=f"k = {k:g}")
ax.set_xticks(np.arange(len(labels)), labels)
ax.set(xlabel="N1 size: peak / SD of the pre-stimulus baseline", ylabel="True N1s found (%)", ylim=(0, 102))
ax.legend(frameon=False, fontsize=6.5, loc="lower right")
ax.set_title("Which N1s each k can see", loc="left")
save(fig, "K2_found_by_size")
print(m.loc[[K0, k_best]].round(1), "best", k_best)
