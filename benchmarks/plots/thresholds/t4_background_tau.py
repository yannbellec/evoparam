"""T4: largest time constant of the background (ccep: background_tau_max = 50 ms).

The background b(t) = c0 + c1·exp(-t/τ) models the slow tail left by the pulse.
(a) Real data: every contact's trial average fitted with τ allowed up to 500 ms; histogram
    of the fitted τ (ms), all contacts of all patients.
(b, c) Known truth: N1 and N2 injected into real pre-pulse noise of half of the contacts
    (common.inject), plus on every contact an exponential tail c1·exp(-t/τ), τ drawn from
    the fitted τ of (a), c1 such that the tail is 20-500 µV at 9 ms (log-uniform, random
    sign). Fitted with τ_max = 10..500 ms. Per patient: % of injected N1 (b) / N2 (c) found
    and % invented on noise-only contacts; mean over patients, 95 % bootstrap CI.
"""
import os

import numpy as np
import pandas as pd

import common as c

TAU_MAX = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
CSV_A, CSV_B = os.path.join(c.HERE, "t4a_real_tau.csv"), os.path.join(c.HERE, "t4b_tau_max.csv")


def compute_a():
    rows = []
    for s in c.subjects().itertuples():
        ts = c.trials(s)
        f = c.fit_contacts(ts.times, c.average(ts), ts.ch_names, background_tau_max=0.5)
        rows.append(f.assign(subject=s.subject)[["subject", "contact", "tau_ms", "n_bumps"]])
        print("T4a", s.subject, flush=True)
    pd.concat(rows).to_csv(CSV_A, index=False)


def compute_b():
    taus = pd.read_csv(CSV_A).tau_ms.dropna().to_numpy() / 1000
    rng = np.random.default_rng(20261029)
    rows = []
    for s in c.subjects().itertuples():
        sim, truth = c.inject(c.trials(s), rng)
        n_ch = len(sim.ch_names)
        tau = rng.choice(taus, n_ch)
        amp = np.exp(rng.uniform(np.log(20), np.log(500), n_ch)) * rng.choice([-1, 1], n_ch)
        t = np.clip(sim.times, 0, None)
        tail = (amp * np.exp(0.009 / tau))[:, None] * np.exp(-t[None] / tau[:, None]) * (sim.times >= 0)[None]
        avg = c.average(sim) + tail
        for tm in TAU_MAX:
            fits = c.fit_contacts(sim.times, avg, sim.ch_names, background_tau_max=float(tm))
            rows.append(dict(subject=s.subject, tau_max_ms=1000 * tm, **c.score(fits, truth)))
        print("T4b", s.subject, flush=True)
        pd.DataFrame(rows).to_csv(CSV_B, index=False)


def figure():
    plt = c.style()
    a, b = pd.read_csv(CSV_A), pd.read_csv(CSV_B)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    tau = a.tau_ms.dropna()
    bins = np.logspace(np.log10(max(tau.min(), 0.4)), np.log10(500), 40)
    axes[0].hist(tau, bins=bins, color=c.BLUE, alpha=0.7, edgecolor="white", lw=0.4)
    c.log_ticks(axes[0], [1, 5, 10, 50, 100, 500])
    axes[0].axvline(50, color="0.3", ls=":", lw=0.7)
    share = 100 * (tau > 50).mean()
    axes[0].text(55, axes[0].get_ylim()[1] * 0.95, f"ccep: 50 ms\n{share:.0f} % of contacts\nabove", fontsize=6.5, color="0.3", va="top")
    axes[0].set(xlabel="Fitted τ of the background (ms)", ylabel="Contacts")
    axes[0].set_title(f"a. Real tails ({a.subject.nunique()} patients)", loc="left")
    for ax, w, lab in ((axes[1], "n1", "b. N1"), (axes[2], "n2", "c. N2")):
        for col, color, txt in ((f"{w}_found", c.BLUE, "true found"), (f"{w}_invented", c.ORANGE, "invented on noise")):
            tab = b.pivot_table(index="subject", columns="tau_max_ms", values=col)
            c.band(ax, tab.columns.values, tab.values, color, label=txt)
        c.log_ticks(ax, 1000 * TAU_MAX)
        ax.axvline(50, color="0.3", ls=":", lw=0.7)
        ax.set(xlabel="Largest τ allowed (ms)", ylabel="Contacts (%)", ylim=(0, 100))
        ax.set_title(f"{lab}: known truth", loc="left")
    axes[1].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    c.save(fig, "T4_background_tau")


if __name__ == "__main__":
    if not os.path.exists(CSV_A):
        compute_a()
    if not os.path.exists(CSV_B):
        compute_b()
    figure()
