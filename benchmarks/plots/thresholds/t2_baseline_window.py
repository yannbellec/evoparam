"""T2: baseline window used to measure the noise (ccep: -500 to -20 ms).

SD_c(W) = standard deviation of the trial average of contact c over the window W (µV;
per-trial median of W removed before averaging).
(a) Window end e (start fixed at -500 ms): ratio SD_c(-500..e) / SD_c(-500..-100), median
    over the contacts of a patient. A rise near 0 ms shows the pulse leaking into the
    baseline (amplifier and filter response).
(b) Window length L (window -20-L..-20 ms): % of contacts whose N1 decision (published rule
    as written, k = 3.4, N1 in 9..50 ms, noise SD from W) equals the decision with the
    longest window (-1000..-20 ms). Mean over patients.
"""
import os

import numpy as np
import pandas as pd

import common as c
from evoparam.existence.rules import apply_rule, resolve_rule

ENDS = np.round(np.arange(-0.100, 0.0001, 0.002), 3)
LENGTHS = np.array([0.05, 0.1, 0.2, 0.3, 0.48, 0.7, 0.98])
CSV_A, CSV_B = os.path.join(c.HERE, "t2a_baseline_end.csv"), os.path.join(c.HERE, "t2b_baseline_length.csv")


def decide(ts, base):
    spec = resolve_rule("vanblooijs2023", baseline=base, peak="strict", k=c.K0, window=(c.t.N1_START, c.t.N1_END))
    return np.asarray(apply_rule(c.average(ts, base), ts.times, ts.sfreq, spec, "uV")["significant"], bool)


def compute():
    rows_a, rows_b = [], []
    for s in c.subjects().itertuples():
        ts = c.trials(s)
        ref_w = (-0.5, -0.1)
        avg = c.average(ts, ref_w)
        ref = avg[:, (ts.times > ref_w[0]) & (ts.times < ref_w[1])].std(1)
        for e in ENDS:
            m = (ts.times > -0.5) & (ts.times < e)
            rows_a.append(dict(subject=s.subject, end_ms=1000 * e, ratio=float(np.median(avg[:, m].std(1) / ref))))
        full = decide(ts, (-1.0, -0.02))
        for L in LENGTHS:
            d = decide(ts, (-0.02 - L, -0.02))
            rows_b.append(dict(subject=s.subject, length_ms=1000 * L, pct_same=100 * (d == full).mean(), pct_n1=100 * d.mean()))
        print("T2", s.subject, flush=True)
    pd.DataFrame(rows_a).to_csv(CSV_A, index=False)
    pd.DataFrame(rows_b).to_csv(CSV_B, index=False)


def figure():
    plt = c.style()
    a, b = pd.read_csv(CSV_A), pd.read_csv(CSV_B)
    n = a.subject.nunique()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.8))
    w = a.pivot_table(index="subject", columns="end_ms", values="ratio")
    c.band(ax1, w.columns.values, w.values, c.BLUE)
    ax1.axvline(-20, color="0.3", ls=":", lw=0.7)
    ax1.axhline(1, color="0.3", ls="--", lw=0.7)
    ax1.text(-21, 1.15, "ccep: -20 ms", fontsize=6.5, color="0.3", ha="right")
    ax1.set(xlabel="End of the baseline window (ms before the pulse)", ylabel="Noise SD / SD of -500..-100 ms")
    ax1.set_title(f"a. Where the baseline must stop ({n} patients)", loc="left")
    w = b.pivot_table(index="subject", columns="length_ms", values="pct_same")
    c.band(ax2, w.columns.values, w.values, c.BLUE)
    ax2.axvline(480, color="0.3", ls=":", lw=0.7)
    ax2.text(495, 55, "ccep: 480 ms", fontsize=6.5, color="0.3")
    ax2.set(xlabel="Baseline length (ms, ending at -20 ms)", ylabel="Same N1 decision as 980 ms (%)", ylim=(None, 102))
    ax2.set_title("b. How long the baseline must be", loc="left")
    c.save(fig, "T2_baseline_window")


if __name__ == "__main__":
    if not os.path.exists(CSV_B):
        compute()
    figure()
