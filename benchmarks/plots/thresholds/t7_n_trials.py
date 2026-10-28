"""T7: number of trials needed (spec: at least 5 trials for the response test).

Known truth: N1 and N2 injected into the real pre-pulse noise of half of the contacts
(common.inject). For n = 3..10 trials drawn at random (2 draws per n), the average of the n
trials of every contact is fitted with the ccep preset. Per patient: % of injected N1 / N2
found, % invented on noise-only contacts, and median |latency error| (ms) of the found
N1 / N2. Mean over patients, 95 % bootstrap CI.
"""
import os

import numpy as np
import pandas as pd

import common as c

NS = (3, 4, 5, 7, 10)
CSV = os.path.join(c.HERE, "t7_n_trials.csv")


def compute():
    rng = np.random.default_rng(20261101)
    rows = []
    for s in c.subjects().itertuples():
        sim, truth = c.inject(c.trials(s), rng)
        m = (sim.times > c.BASELINE[0]) & (sim.times < c.BASELINE[1])
        x = sim.data - np.median(sim.data[..., m], axis=-1, keepdims=True)
        for n in NS:
            for draw in range(1 if n >= len(x) else 2):
                pick = rng.choice(len(x), min(n, len(x)), replace=False)
                fits = c.fit_contacts(sim.times, x[pick].mean(0), sim.ch_names)
                rows.append(dict(subject=s.subject, n_trials=n, draw=draw, **c.score(fits, truth)))
        print("T7", s.subject, flush=True)
        pd.DataFrame(rows).to_csv(CSV, index=False)


def figure():
    plt = c.style()
    d = pd.read_csv(CSV).groupby(["subject", "n_trials"]).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    for ax, w, lab in ((axes[0], "n1", "a. N1"), (axes[1], "n2", "b. N2")):
        for col, color, txt in ((f"{w}_found", c.BLUE, "true found"), (f"{w}_invented", c.ORANGE, "invented on noise")):
            tab = d.pivot_table(index="subject", columns="n_trials", values=col)
            c.band(ax, tab.columns.values, tab.values, color, label=txt)
        ax.axvline(5, color="0.3", ls=":", lw=0.7)
        ax.set(xlabel="Number of trials averaged", ylabel="Contacts (%)", ylim=(0, 100))
        ax.set_title(lab, loc="left")
    axes[0].legend(frameon=False, fontsize=6.5)
    for w, color, txt in (("n1", c.BLUE, "N1"), ("n2", c.GREEN, "N2")):
        tab = d.pivot_table(index="subject", columns="n_trials", values=f"{w}_lat_err_ms")
        c.band(axes[2], tab.columns.values, tab.values, color, label=txt)
    axes[2].axvline(5, color="0.3", ls=":", lw=0.7)
    axes[2].set_yscale("log")
    axes[2].set(xlabel="Number of trials averaged", ylabel="|Latency error| (ms, median)")
    axes[2].set_title("c. Latency error", loc="left")
    axes[2].legend(frameon=False, fontsize=6.5)
    fig.suptitle(f"Known N1 and N2 in real noise ({d.subject.nunique()} patients)", x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    c.save(fig, "T7_n_trials")


if __name__ == "__main__":
    if not os.path.exists(CSV) or "--compute" in os.sys.argv:
        compute()
    figure()
