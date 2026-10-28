"""T3: re-referencing (ccep: adaptive common average, car_adaptive).

Known N1 and N2 injected into the real pre-pulse noise of half of the contacts of every
patient (common.inject). Each reference is applied to all trials, then the trial average
of every contact is fitted with the ccep preset. Per patient: % of injected N1/N2 found,
% invented on noise-only contacts, and amplitude kept = median(found / true amplitude).
- monopolar: no re-reference;
- car: minus the mean of all contacts (the injected responses leak into every contact);
- car_adaptive: minus the mean of the least responsive contacts only.
"""
import os

import numpy as np
import pandas as pd

import common as c
from evoparam.prep import rereference

SCHEMES = ("monopolar", "car", "car_adaptive")
CSV = os.path.join(c.HERE, "t3_reference.csv")


def compute():
    rng = np.random.default_rng(20261028)
    rows = []
    for s in c.subjects().itertuples():
        sim, truth = c.inject(c.trials(s), rng)
        for sch in SCHEMES:
            r = rereference(sim, sch, random_state=0)
            fits = c.fit_contacts(r.times, c.average(r), r.ch_names)
            rows.append(dict(subject=s.subject, scheme=sch, **c.score(fits, truth)))
        print("T3", s.subject, flush=True)
        pd.DataFrame(rows).to_csv(CSV, index=False)


def figure():
    plt = c.style()
    d = pd.read_csv(CSV)
    n = d.subject.nunique()
    labels = ["none", "CAR", "adaptive\nCAR"]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6))
    for i, w in enumerate(("n1", "n2")):
        W = w.upper()
        for j, (col, color, ylab) in enumerate(((f"{w}_found", c.BLUE, f"True {W}s found (%)"),
                                               (f"{w}_invented", c.ORANGE, f"{W}s invented on noise (%)"),
                                               (f"{w}_amp_kept", c.GREEN, f"{W} amplitude found / true (%)"))):
            tab = d.pivot_table(index="subject", columns="scheme", values=col)[list(SCHEMES)]
            c.paired(axes[i, j], tab.values, labels, color, ylab, f"{'abcdef'[3 * i + j]}. {W}")
    axes[0, 2].axhline(100, color="0.3", ls="--", lw=0.7)
    axes[1, 2].axhline(100, color="0.3", ls="--", lw=0.7)
    fig.suptitle(f"Reference: known N1 and N2 in real noise ({n} patients)", x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    c.save(fig, "T3_reference")


if __name__ == "__main__":
    if not os.path.exists(CSV) or "--compute" in os.sys.argv:
        compute()
    figure()
