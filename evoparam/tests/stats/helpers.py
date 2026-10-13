# SPDX-License-Identifier: Apache-2.0
"""Simulated data for the statistics tests."""

from typing import Any

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from evoparam.data import TrialSet
from evoparam.shapes import get_shape
from evoparam.sim import simulate_erp

pd = pytest.importorskip("pandas")

# Window (s) of the mean amplitude, and the size above which the unbalanced ANOVA
# estimator replaces REML.
WINDOW_START = 0.08
WINDOW_END = 0.12
LARGE_SET = 3000

# A negative component at 100 ms, given in data units (snr=None).
N1 = {"label": "N1", "latency": 0.1, "fwhm": 0.04, "amplitude": -5.0}
WINDOW = (0.092, 0.108)


def nominal_mean(truth: Any, times: Any, window: Any, index: Any = 0) -> Any:
    """Mean over the window of the nominal component (amplitude factor of mean 1)."""

    comp = truth.components[index]
    curve = comp["amplitude"] * get_shape(comp["shape"]).evaluate(
        times, np.asarray(comp["params"], float)
    )
    inside = (times >= window[0]) & (times <= window[1])
    return float(curve[inside].mean())


def replicates(n_trials: Any, n_replicates: Any, seed: Any, **kwargs: Any) -> Any:
    """``n_replicates`` independent simulations of ``n_trials`` trials of one channel, as
    the channels of one TrialSet (cropped around the N1), and the true window mean."""

    ts, truth = simulate_erp(
        n_trials * n_replicates,
        components=[N1],
        sfreq=250.0,
        tmin=-0.1,
        tmax=0.15,
        seed=seed,
        keep_arrays=False,
        **kwargs,
    )
    true = nominal_mean(truth, ts.times, WINDOW)
    inside = (ts.times >= WINDOW_START) & (ts.times <= WINDOW_END)
    data = ts.data[:, 0][:, inside].reshape(n_replicates, n_trials, -1).transpose(1, 0, 2)
    big = TrialSet(
        data=np.ascontiguousarray(data),
        times=ts.times[inside],
        sfreq=ts.sfreq,
        ch_names=[f"r{i}" for i in range(n_replicates)],
    )
    return big, true


def subjects_trialset(
    n_subjects: Any,
    n_trials: Any,
    *,
    sd_subject: Any,
    sd_trial: Any,
    seed: Any,
    n_times: Any = 41,
    unbalanced: Any = False,
) -> Any:
    """Trials of several subjects: a constant per-subject level plus white noise, one channel.

    With ``unbalanced``, subject k keeps ``n_trials - (k % 4) * 5`` trials.
    """

    rng = np.random.default_rng(seed)
    times = np.linspace(-0.1, 0.3, n_times)
    counts = [n_trials - (k % 4) * 5 if unbalanced else n_trials for k in range(n_subjects)]
    subject = np.repeat(np.arange(n_subjects), counts)
    level = rng.normal(0, sd_subject, n_subjects)[subject]
    data = level[:, None, None] + rng.normal(0, sd_trial, (subject.size, 1, n_times))
    meta = pd.DataFrame({"subject": [f"s{k:02d}" for k in subject]})
    return TrialSet(data=data, times=times, sfreq=100.0, ch_names=["Cz"], metadata=meta)


def independent_split_half(ts: Any, score: Any) -> Any:
    """Odd/even split half from the published formulas, one subject at a time."""

    odd, even = [], []
    for subject in dict.fromkeys(ts.metadata["subject"]):
        rows = np.flatnonzero(ts.metadata["subject"].to_numpy() == subject)
        for store, part in ((odd, rows[0::2]), (even, rows[1::2])):
            avg = ts.data[part, 0].mean(axis=0)
            store.append(score(avg))
    r = np.corrcoef(odd, even)[0, 1]
    return 2 * r / (1 + r)


def independent_dependability(ts: Any, window: Any) -> Any:
    """Dependability from variance components estimated independently: the ANOVA
    estimator for a balanced design (equal to REML when it is positive), else a dense REML
    maximised over the variance ratio (the unbalanced ANOVA estimator above 3000 trials)."""

    inside = (ts.times >= window[0]) & (ts.times <= window[1])
    y = ts.data[:, 0][:, inside].mean(axis=1)
    subjects = ts.metadata["subject"].to_numpy()
    labels = list(dict.fromkeys(subjects))
    groups = [y[subjects == s] for s in labels]
    sizes = np.array([len(g) for g in groups])
    if len(y) > LARGE_SET:
        # large unbalanced sets (real data): the ANOVA estimator of unbalanced one-way
        # designs (Searle, Casella and McCulloch 1992, n0 = (N - sum n_i^2 / N) / (k - 1)),
        # close to REML when the design is near balance
        k, n_obs = len(groups), len(y)
        means = np.array([g.mean() for g in groups])
        msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (n_obs - k)
        msb = (sizes * (means - y.mean()) ** 2).sum() / (k - 1)
        n0 = (n_obs - (sizes**2).sum() / n_obs) / (k - 1)
        var_u, var_e = (msb - msw) / n0, msw
    elif np.all(sizes == sizes[0]):
        n, k = sizes[0], len(groups)
        means = np.array([g.mean() for g in groups])
        msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (k * (n - 1))
        msb = n * ((means - y.mean()) ** 2).sum() / (k - 1)
        var_u, var_e = (msb - msw) / n, msw
    else:
        zmat = (subjects[:, None] == np.array(labels)[None, :]).astype(float)
        xmat = np.ones((len(y), 1))

        def neg(log_ratio):
            hmat = np.eye(len(y)) + np.exp(log_ratio) * zmat @ zmat.T
            hmat_inv = np.linalg.inv(hmat)
            beta = np.linalg.solve(xmat.T @ hmat_inv @ xmat, xmat.T @ hmat_inv @ y)
            r = y - xmat @ beta
            s2 = r @ hmat_inv @ r / (len(y) - 1)
            return (
                (len(y) - 1) * np.log(s2)
                + np.linalg.slogdet(hmat)[1]
                + np.linalg.slogdet(xmat.T @ hmat_inv @ xmat)[1]
            )

        best = minimize_scalar(neg, bounds=(-10, 5), method="bounded", options={"xatol": 1e-10})
        hmat = np.eye(len(y)) + np.exp(best.x) * zmat @ zmat.T
        hmat_inv = np.linalg.inv(hmat)
        beta = (xmat.T @ hmat_inv @ y) / (xmat.T @ hmat_inv @ xmat)
        r = y - xmat @ beta.ravel()
        var_e = r @ hmat_inv @ r / (len(y) - 1)
        var_u = np.exp(best.x) * var_e
    return var_u / (var_u + var_e / sizes.mean()), var_u, var_e
