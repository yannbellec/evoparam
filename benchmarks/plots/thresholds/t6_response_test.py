"""T6: "is there a response?" test (ccep: sign-flip permutation, max_t, α = 0.05, FDR).

Known truth: N1 and N2 injected into the real pre-pulse noise of half of the contacts
(common.inject). test_response (window 9-500 ms, baseline -500..-20 ms, 1000 sign-flip
permutations) gives one raw p-value per contact; the correction across contacts is then
applied for every α: Benjamini-Hochberg FDR (ccep), Holm, or none. Per patient: % of
injected contacts declared responding (power) and % of noise-only contacts declared
responding (false alarms). Mean over patients, 95 % bootstrap CI.
"""
import os

import numpy as np
import pandas as pd

import common as c
from evoparam.existence import test_response

ALPHAS = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
CSV = os.path.join(c.HERE, "t6_test.csv")


def bh(p, a):
    p = np.asarray(p, float)
    order = np.argsort(p)
    ok = p[order] <= a * np.arange(1, len(p) + 1) / len(p)
    keep = np.zeros(len(p), bool)
    if ok.any():
        keep[order[: np.flatnonzero(ok).max() + 1]] = True
    return keep


def holm(p, a):
    p = np.asarray(p, float)
    order = np.argsort(p)
    ok = p[order] <= a / (len(p) - np.arange(len(p)))
    keep = np.zeros(len(p), bool)
    stop = np.flatnonzero(~ok)
    keep[order[: stop[0] if len(stop) else len(p)]] = True
    return keep


def compute():
    rng = np.random.default_rng(20261031)
    rows = []
    for s in c.subjects().itertuples():
        sim, truth = c.inject(c.trials(s), rng)
        r = test_response(sim, window=(0.009, 0.5), baseline=c.BASELINE, by=(), random_state=0)
        p = np.asarray(r.p_value_raw, float).ravel()
        inj = truth.injected.to_numpy()
        for a in ALPHAS:
            for name, dec in (("fdr", bh(p, a)), ("holm", holm(p, a)), ("none", p <= a)):
                rows.append(dict(subject=s.subject, alpha=a, correction=name,
                                 power=100 * dec[inj].mean(), false_alarm=100 * dec[~inj].mean()))
        print("T6", s.subject, flush=True)
    pd.DataFrame(rows).to_csv(CSV, index=False)


def figure():
    plt = c.style()
    d = pd.read_csv(CSV)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    styles = {"fdr": ("-", "FDR (ccep)"), "holm": ("--", "Holm"), "none": (":", "no correction")}
    for ax, col, color, lab in ((axes[0], "power", c.BLUE, "a. Responding contacts found"),
                                (axes[1], "false_alarm", c.ORANGE, "b. Noise-only contacts flagged")):
        for corr, (ls, txt) in styles.items():
            tab = d[d.correction == corr].pivot_table(index="subject", columns="alpha", values=col)
            ci = np.array([c.boot_ci(tab[x]) for x in tab.columns])
            if corr == "fdr":
                ax.fill_between(tab.columns, ci[:, 0], ci[:, 1], color=color, alpha=0.22, lw=0)
            ax.plot(tab.columns, tab.mean().values, color=color, ls=ls, lw=1.5, label=txt)
        c.log_ticks(ax, [0.001, 0.005, 0.01, 0.05, 0.1, 0.2])
        ax.axvline(0.05, color="0.3", ls=":", lw=0.7)
        ax.set(xlabel="α (significance level)", ylabel="Contacts (%)")
        ax.set_title(lab, loc="left")
        ax.legend(frameon=False, fontsize=6.5)
    x = np.array(sorted(d.alpha.unique()))
    axes[1].plot(x, 100 * x, color="0.5", lw=0.6)
    axes[1].text(0.15, 100 * 0.15 * 1.05, "y = α", fontsize=6.5, color="0.5", ha="right", va="bottom")
    fig.suptitle(f"Response test: known N1 and N2 in real noise ({d.subject.nunique()} patients)", x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    c.save(fig, "T6_response_test")


if __name__ == "__main__":
    if not os.path.exists(CSV):
        compute()
    figure()
