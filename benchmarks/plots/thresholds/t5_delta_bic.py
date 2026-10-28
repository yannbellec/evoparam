"""T5: evidence needed to keep a bump (ccep: ΔBIC ≥ 10, ``delta = 10``).

A new bump is kept when it lowers BIC = n_eff·ln(RSS_w/n_eff) + p·ln(n_eff) by at least Δ.
Known truth: N1 and N2 injected into the real pre-pulse noise of half of the contacts
(common.inject); every contact's trial average fitted with Δ = 2..40. Per patient: % of
injected N1 / N2 found, % invented on noise-only contacts, and mean number of bumps on
noise-only contacts. Mean over patients, 95 % bootstrap CI.
"""
import os

import numpy as np
import pandas as pd

import common as c

DELTAS = (2.0, 5.0, 10.0, 20.0, 40.0)
CSV = os.path.join(c.HERE, "t5_delta.csv")


def compute():
    rng = np.random.default_rng(20261030)
    rows = []
    for s in c.subjects().itertuples():
        sim, truth = c.inject(c.trials(s), rng)
        avg = c.average(sim)
        for d in DELTAS:
            fits = c.fit_contacts(sim.times, avg, sim.ch_names, delta=d)
            noise = fits.merge(truth, on="contact").query("~injected")
            rows.append(dict(subject=s.subject, delta=d, bumps_on_noise=noise.n_bumps.mean(), **c.score(fits, truth)))
        print("T5", s.subject, flush=True)
        pd.DataFrame(rows).to_csv(CSV, index=False)


def figure():
    plt = c.style()
    d = pd.read_csv(CSV)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    for ax, w, lab in ((axes[0], "n1", "a. N1"), (axes[1], "n2", "b. N2")):
        for col, color, txt in ((f"{w}_found", c.BLUE, "true found"), (f"{w}_invented", c.ORANGE, "invented on noise")):
            tab = d.pivot_table(index="subject", columns="delta", values=col)
            c.band(ax, tab.columns.values, tab.values, color, label=txt)
        c.log_ticks(ax, DELTAS)
        ax.axvline(10, color="0.3", ls=":", lw=0.7)
        ax.set(xlabel="ΔBIC needed to keep a bump", ylabel="Contacts (%)", ylim=(0, 100))
        ax.set_title(lab, loc="left")
    axes[0].legend(frameon=False, fontsize=6.5)
    tab = d.pivot_table(index="subject", columns="delta", values="bumps_on_noise")
    c.band(axes[2], tab.columns.values, tab.values, c.ORANGE)
    c.log_ticks(axes[2], DELTAS)
    axes[2].axvline(10, color="0.3", ls=":", lw=0.7)
    axes[2].set(xlabel="ΔBIC needed to keep a bump", ylabel="Bumps per noise-only contact")
    axes[2].set_title("c. Bumps fitted on pure noise", loc="left")
    fig.suptitle(f"Known N1 and N2 in real noise ({d.subject.nunique()} patients)", x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    c.save(fig, "T5_delta_bic")


if __name__ == "__main__":
    if not os.path.exists(CSV) or "--compute" in os.sys.argv:
        compute()
    figure()
