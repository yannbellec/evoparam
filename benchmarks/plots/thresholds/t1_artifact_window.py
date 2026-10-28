"""T1: end of the masked stimulation-artifact window (ccep: 9 ms).

(a) Size of the averaged signal after the pulse, in units of baseline noise:
    z_c(t) = |x̄_c(t)| / SD_c, with x̄_c the trial average of contact c (per-trial median of
    -500..-20 ms removed) and SD_c the SD of x̄_c over -500..-20 ms. The median over the
    contacts of a patient follows the artifact (present on every contact) and not the
    responses (present on a minority of contacts). One value per patient and time point.
(b) N1 detection with the published rule as written (deepest local minimum below
    -3.4 SD, baseline -1..-0.1 s) when the search window starts at s ms (window s..50 ms):
    % of contacts with an N1, and % whose N1 sits in the first 1 ms of the window (an
    edge trough: the end of the artifact taken for an N1). Mean over patients.
"""
import os

import numpy as np
import pandas as pd

import common as c
from evoparam.existence.rules import apply_rule, resolve_rule

STARTS = np.round(np.arange(0.004, 0.0161, 0.001), 3)
CSV_A, CSV_B = os.path.join(c.HERE, "t1a_artifact_decay.csv"), os.path.join(c.HERE, "t1b_window_start.csv")


def compute():
    rows_a, rows_b = [], []
    for s in c.subjects().itertuples():
        ts = c.trials(s)
        avg = c.average(ts)
        m = (ts.times > c.BASELINE[0]) & (ts.times < c.BASELINE[1])
        sd = avg[:, m].std(axis=1)
        post = (ts.times >= 0) & (ts.times <= 0.03)
        z = np.median(np.abs(avg[:, post]) / sd[:, None], axis=0)
        rows_a += [dict(subject=s.subject, t_ms=1000 * tt, median_z=zz) for tt, zz in zip(ts.times[post], z)]
        x = ts.data - np.median(ts.data[..., (ts.times > -1) & (ts.times < -0.1)], axis=-1, keepdims=True)
        ravg = x.mean(0)
        for st in STARTS:
            spec = resolve_rule("vanblooijs2023", baseline=(-1.0, -0.1), peak="strict", k=c.K0, window=(st, c.t.N1_END))
            out = apply_rule(ravg, ts.times, ts.sfreq, spec, "uV")
            sig = np.asarray(out["significant"], bool)
            lat = np.asarray(out["onset"], float)
            edge = sig & (lat < st + 0.001)
            rows_b.append(dict(subject=s.subject, start_ms=1000 * st, pct_n1=100 * sig.mean(), pct_edge=100 * edge.mean()))
        print("T1", s.subject, flush=True)
    pd.DataFrame(rows_a).to_csv(CSV_A, index=False)
    pd.DataFrame(rows_b).to_csv(CSV_B, index=False)


def figure():
    plt = c.style()
    a, b = pd.read_csv(CSV_A), pd.read_csv(CSV_B)
    n = a.subject.nunique()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.8))
    wa = a.assign(t_ms=a.t_ms.round(2)).pivot_table(index="subject", columns="t_ms", values="median_z")
    c.band(ax1, wa.columns.values, wa.values, c.BLUE)
    ax1.set_yscale("log")
    ax1.axhline(c.K0, color="0.3", ls="--", lw=0.7)
    ax1.axhline(0.674, color="0.3", ls="-.", lw=0.6)
    ax1.text(29.5, 0.674 * 1.12, "pure noise (median |z| = 0.67)", fontsize=6.5, color="0.3", ha="right")
    ax1.axvline(9, color="0.3", ls=":", lw=0.7)
    ax1.text(9.4, ax1.get_ylim()[1] * 0.6, "end of masked\nwindow = 9 ms", fontsize=6.5, color="0.3", va="top")
    ax1.text(29.5, c.K0 * 1.15, "3.4 × noise SD", fontsize=6.5, color="0.3", ha="right")
    ax1.set(xlabel="Time after the pulse (ms)", ylabel="Median |average| / noise SD", xlim=(0, 30), ylim=(0.4, None))
    ax1.set_title(f"a. Artifact decay ({n} patients)", loc="left")
    for col, color, lab in (("pct_n1", c.BLUE, "contacts with an N1"), ("pct_edge", c.ORANGE, "N1 at the window edge")):
        w = b.pivot_table(index="subject", columns="start_ms", values=col)
        c.band(ax2, w.columns.values, w.values, color, label=lab)
    ax2.axvline(9, color="0.3", ls=":", lw=0.7)
    ax2.set(xlabel="Start of the N1 search window (ms)", ylabel="Contacts (%)", xlim=(4, 16), ylim=(0, None))
    ax2.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax2.set_title("b. N1 found at the end of the artifact", loc="left")
    c.save(fig, "T1_artifact_window")


if __name__ == "__main__":
    if not os.path.exists(CSV_B):
        compute()
    figure()
