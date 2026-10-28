"""Shared helpers for the threshold figures: data access, style, bootstrap, saving."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "evoparam", "tests", "models"))
import test_engine_ccep as t  # noqa: E402  (ds004080 cache of the N1 validation)

FIG = os.path.join(ROOT, "figures", "thresholds")
BASELINE = (-0.5, -0.02)  # ccep preset baseline (s)
K0 = 3.4
BLUE, ORANGE, GREEN, PINK = "#0072B2", "#D55E00", "#009E73", "#CC79A7"


def subjects():
    s = pd.read_csv(os.path.join(HERE, "..", "ds004080_subjects.tsv"), sep="\t", dtype=str)
    return s[s.status == "ok"].reset_index(drop=True)


def trials(row):
    """All trials of the patient's pair: (n_trials, n_channels, n_times) in µV, -2..3 s."""
    return t._pair_trials(row.subject[4:], row.run, row.pair)


def average(ts, baseline=BASELINE):
    """Per-trial median baseline removed, then mean over trials (µV)."""
    m = (ts.times > baseline[0]) & (ts.times < baseline[1])
    x = ts.data - np.median(ts.data[..., m], axis=-1, keepdims=True)
    return x.mean(axis=0)


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "savefig.dpi": 300, "figure.dpi": 150})
    return plt


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    return np.percentile(rng.choice(x, (n, len(x))).mean(1), [2.5, 97.5])


def band(ax, x, table, color, label=None, lw=1.6):
    """Thin grey line per patient, mean over patients, 95 % bootstrap band over patients.

    ``table``: rows = patients, columns = x values.
    """
    arr = np.asarray(table, float)
    for row in arr:
        ax.plot(x, row, color="0.8", lw=0.4, zorder=1)
    ci = np.array([boot_ci(arr[:, j]) for j in range(arr.shape[1])])
    ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.22, lw=0, zorder=2)
    ax.plot(x, np.nanmean(arr, 0), color=color, lw=lw, label=label, zorder=3)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight",
                    metadata={"Software": None} if ext == "png" else None)
    import matplotlib.pyplot as plt
    plt.close(fig)


N1_PRIOR = dict(lat=(0.015, 0.045), fwhm=(0.005, 0.015), amp=(20.0, 500.0))
N2_PRIOR = dict(lat=(0.080, 0.200), fwhm=(0.040, 0.100), amp=(20.0, 300.0))


def inject(ts, rng, fraction=0.5, n2=True, jitter=0.002):
    """Known responses in real noise (as plot K).

    The -2..0 s part of every trial (before the pulse: no response, no artifact) becomes a
    -1..+1 s epoch. A random ``fraction`` of the contacts receive a negative Gaussian N1
    (latency 15-45 ms, FWHM 5-15 ms, peak 20-500 µV log-uniform) and, if ``n2``, a negative
    Gaussian N2 (80-200 ms, FWHM 40-100 ms, 20-300 µV); ±``jitter`` s latency jitter per
    trial. Returns the new TrialSet and the truth table (one row per contact).
    """
    from evoparam.data import TrialSet

    keep = ts.times < 0
    x = np.asarray(ts.data)[..., keep] * 1.0
    times = ts.times[keep] + 1.0
    n_tr, n_ch, _ = x.shape
    inj = np.zeros(n_ch, bool)
    inj[rng.permutation(n_ch)[: int(round(n_ch * fraction))]] = True
    truth = pd.DataFrame({"contact": ts.ch_names, "injected": inj})
    for name, prior in (("n1", N1_PRIOR), ("n2", N2_PRIOR)) if n2 else (("n1", N1_PRIOR),):
        lat = rng.uniform(*prior["lat"], n_ch)
        fwhm = rng.uniform(*prior["fwhm"], n_ch)
        amp = np.exp(rng.uniform(np.log(prior["amp"][0]), np.log(prior["amp"][1]), n_ch))
        for ch in np.flatnonzero(inj):
            jit = rng.normal(0, jitter, n_tr)[:, None]
            x[:, ch] -= amp[ch] * np.exp(-0.5 * ((times[None] - lat[ch] - jit) / (fwhm[ch] / 2.3548)) ** 2)
        truth[f"{name}_latency"], truth[f"{name}_fwhm"] = lat, fwhm
        truth[f"{name}_amplitude"] = np.where(inj, -amp, 0.0)
    new = TrialSet(data=x, times=times, sfreq=ts.sfreq, ch_names=list(ts.ch_names),
                   metadata=pd.DataFrame({"stim_site": ["sim"] * n_tr}))
    return new, truth


def fit_contacts(times, avg, names, **overrides):
    """Fit every contact's average with the ccep preset; one row per contact.

    N1 / N2 = the evoparam labels (ccep: negative bumps in 9-50 ms and 50-300 ms); NaN when
    the label is absent. ``tau_ms`` = fitted background time constant.
    """
    from evoparam import fit_response

    rows = []
    for name, y in zip(names, avg):
        f = fit_response(times, y, preset="ccep", random_state=0, **overrides)
        row = {"contact": name, "n_bumps": len(f.components), "tau_ms": 1000 * f.background.get("tau", np.nan)}
        for lab in ("N1", "N2"):
            b = [b for b in f.components if b.label == lab]
            row[f"{lab.lower()}_fit_amplitude"] = b[0].amplitude if b else np.nan
            row[f"{lab.lower()}_fit_latency"] = b[0].latency if b else np.nan
            row[f"{lab.lower()}_fit_onset"] = b[0].onset if b else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def score(fits, truth):
    """Per-patient scores: % of injected N1/N2 found, % invented on noise-only contacts,
    found amplitude / true amplitude (median, %), |latency error| (median, ms)."""
    d = fits.merge(truth, on="contact")
    out = {}
    for w in ("n1", "n2"):
        found = d[f"{w}_fit_amplitude"].notna()
        inj = d.injected
        out[f"{w}_found"] = 100 * found[inj].mean()
        out[f"{w}_invented"] = 100 * found[~inj].mean()
        ok = inj & found
        out[f"{w}_amp_kept"] = 100 * float(np.median(d[f"{w}_fit_amplitude"][ok] / d[f"{w}_amplitude"][ok])) if ok.any() else np.nan
        out[f"{w}_lat_err_ms"] = 1000 * float(np.median(np.abs(d[f"{w}_fit_latency"][ok] - d[f"{w}_latency"][ok]))) if ok.any() else np.nan
    return out


def paired(ax, table, labels, color, ylabel, title):
    """Slopegraph: one grey line per patient across settings, mean ± 95 % bootstrap CI."""
    arr = np.asarray(table, float)
    x = np.arange(len(labels))
    for row in arr:
        ax.plot(x, row, color="0.8", lw=0.4, zorder=1)
    ci = np.array([boot_ci(arr[:, j]) for j in range(len(labels))])
    mean = np.nanmean(arr, 0)
    ax.errorbar(x, mean, yerr=[mean - ci[:, 0], ci[:, 1] - mean], color=color, lw=1.4, capsize=2, marker="o",
                ms=4, mec="white", mew=0.8, zorder=3)
    for xi, m in zip(x, mean):
        ax.annotate(f"{m:.1f}" if abs(m) < 10 else f"{m:.0f}", (xi, m), xytext=(5, 0), textcoords="offset points", fontsize=6.5, va="center")
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.4, len(labels) - 0.6)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")


def log_ticks(ax, values):
    """Log x axis labelled with the plain tested values."""
    from matplotlib.ticker import FixedLocator, NullLocator
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(values))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{v:g}" for v in values])
