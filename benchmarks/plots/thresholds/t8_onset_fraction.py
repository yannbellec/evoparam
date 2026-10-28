"""T8: onset and offset of a bump (ccep: where the bump reaches 10 % of its peak).

Known truth: N1 and N2 injected into the real pre-pulse noise of half of the contacts
(common.inject), Gaussian shape, so the true onset at a fraction f of the peak is
latency - FWHM/2.3548 · sqrt(2 ln(1/f)). Each contact's average is fitted once with the
ccep preset; the fitted bump curve is evaluated on a 0.05 ms grid and its onset read at
f = 5..50 %. Per patient: median |fitted onset - true onset| (ms) over found N1 / N2.
Mean over patients, 95 % bootstrap CI.
"""
import os

import numpy as np
import pandas as pd

import common as c
from evoparam import fit_response
from evoparam.shapes import get_shape

FRACTIONS = np.array([0.05, 0.1, 0.2, 0.3, 0.5])
CSV = os.path.join(c.HERE, "t8_onset_fraction.csv")
GRID = np.arange(-0.05, 0.5, 0.00005)


def onset_at(curve, f):
    a = np.abs(curve)
    i = int(np.argmax(a))
    above = np.flatnonzero(a[: i + 1] < f * a[i])
    return GRID[above[-1] + 1] if len(above) else np.nan


def compute():
    rng = np.random.default_rng(20261102)
    rows = []
    for s in c.subjects().itertuples():
        sim, truth = c.inject(c.trials(s), rng)
        avg = c.average(sim)
        err = {(w, f): [] for w in ("n1", "n2") for f in FRACTIONS}
        for k, row in truth[truth.injected].iterrows():
            fit = fit_response(sim.times, avg[k], preset="ccep", random_state=0)
            for w in ("n1", "n2"):
                b = [b for b in fit.components if b.label == w.upper()]
                if not b:
                    continue
                curve = get_shape(b[0].shape).evaluate(GRID, np.asarray(b[0].params))
                sigma = row[f"{w}_fwhm"] / 2.3548
                for f in FRACTIONS:
                    true = row[f"{w}_latency"] - sigma * np.sqrt(2 * np.log(1 / f))
                    err[(w, f)].append(abs(onset_at(curve, f) - true))
        for (w, f), e in err.items():
            rows.append(dict(subject=s.subject, wave=w, fraction=f, onset_err_ms=1000 * np.nanmedian(e) if e else np.nan))
        print("T8", s.subject, flush=True)
        pd.DataFrame(rows).to_csv(CSV, index=False)


def figure():
    plt = c.style()
    d = pd.read_csv(CSV)
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    for w, color, txt in (("n1", c.BLUE, "N1"), ("n2", c.GREEN, "N2")):
        tab = d[d.wave == w].pivot_table(index="subject", columns="fraction", values="onset_err_ms")
        c.band(ax, 100 * tab.columns.values, tab.values, color, label=txt)
    ax.axvline(10, color="0.3", ls=":", lw=0.7)
    ax.set_yscale("log")
    ax.set(xlabel="Onset read at this % of the peak", ylabel="|Onset error| (ms, median)")
    ax.set_title(f"Onset: known N1 and N2 in real noise ({d.subject.nunique()} patients)", loc="left")
    ax.legend(frameon=False, fontsize=6.5)
    c.save(fig, "T8_onset_fraction")


if __name__ == "__main__":
    if not os.path.exists(CSV) or "--compute" in os.sys.argv:
        compute()
    figure()
