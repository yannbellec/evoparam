# SPDX-License-Identifier: Apache-2.0
"""Jackknife scoring over participants (REQ-231)."""

from typing import Any

import numpy as np
import pytest
from scipy import stats as sps

from evoparam.data import Evoked
from evoparam.errors import DataError
from evoparam.sim import simulate_erp
from evoparam.stats import jackknife_f, jackknife_latency, jackknife_t, score_waveforms

pd = pytest.importorskip("pandas")

###################################################################################################
###################################################################################################

# Smulders (2010), Table 1: hypothetical Age (between, 5 participants per group) x Sleep
# deprivation (within) design; the retrieved latencies (ms) are the participants' own.
SMULDERS_LATENCIES = {
    "young": [[479, 503, 659], [471, 580, 593], [490, 476, 654], [440, 434, 620], [270, 457, 374]],
    "middle": [
        [616, 459, 512],
        [535, 377, 510],
        [589, 451, 505],
        [514, 463, 523],
        [546, 600, 600],
    ],
    "old": [[596, 578, 707], [556, 694, 694], [594, 626, 667], [603, 590, 748], [501, 662, 634]],
}
# Printed subaverage (jackknife) scores of the young group without sleep deprivation.
SMULDERS_SUBAVERAGES = [417.75, 419.75, 415.0, 427.5, 470.0]
# Printed F statistics (identical for the retrieved scores and the corrected subaverages).
SMULDERS_F = {"age": 9.2, "sleep": 9.8, "interaction": 4.9}
CONDITIONS = ["none", "mild", "heavy"]


def _constant_evokeds(rows: Any, group: Any) -> Any:
    """One Evoked per participant whose 'mean amplitude' on each channel is a latency."""

    times = np.array([0.0, 0.001])
    return [
        Evoked(
            data=np.repeat(np.asarray(row, float)[:, None], 2, axis=1),
            times=times,
            sfreq=1000.0,
            ch_names=CONDITIONS,
            metadata=pd.DataFrame({"subject": [f"{group}{k}"]}),
        )
        for k, row in enumerate(rows)
    ]


def _mixed_anova(y: Any, group: Any) -> Any:
    """F of a one-between (group) x one-within (columns) design, equal group sizes."""

    levels = np.unique(group)
    a, n, b = len(levels), int(np.sum(group == levels[0])), y.shape[1]
    gm = y.mean()
    gmean = np.array([y[group == g].mean() for g in levels])
    cmean = y.mean(axis=0)
    cell = np.array([y[group == g].mean(axis=0) for g in levels])
    sm = y.mean(axis=1)
    ss_a = n * b * np.sum((gmean - gm) ** 2)
    ss_s = b * np.sum((sm - gmean[np.searchsorted(levels, group)]) ** 2)
    ss_b = a * n * np.sum((cmean - gm) ** 2)
    ss_ab = n * np.sum((cell - gmean[:, None] - cmean[None, :] + gm) ** 2)
    ss_e = np.sum((y - gm) ** 2) - ss_a - ss_s - ss_b - ss_ab
    df_s, df_e = a * (n - 1), a * (n - 1) * (b - 1)
    return {
        "age": (ss_a / (a - 1)) / (ss_s / df_s),
        "sleep": (ss_b / (b - 1)) / (ss_e / df_e),
        "interaction": (ss_ab / ((a - 1) * (b - 1))) / (ss_e / df_e),
    }


def test_req231_smulders_worked_example() -> None:

    jack, retrieved, group = [], [], []
    for name, rows in SMULDERS_LATENCIES.items():
        table = jackknife_latency(
            _constant_evokeds(rows, name), scorer="mean_amplitude", window=(0.0, 0.001)
        )
        wide = table.pivot(index="subject", columns="channel")
        jack.append(wide["jackknife"][CONDITIONS].to_numpy())
        retrieved.append(wide["retrieved"][CONDITIONS].to_numpy())
        group += [name] * len(rows)
        if name == "young":
            assert np.allclose(wide["jackknife"]["none"], SMULDERS_SUBAVERAGES, atol=0.005)
    jack, retrieved, group = np.vstack(jack), np.vstack(retrieved), np.array(group)
    own = np.vstack([np.asarray(rows, float) for rows in SMULDERS_LATENCIES.values()])
    assert np.allclose(retrieved, own, rtol=0, atol=1e-9)
    f_jack = {k: jackknife_f(v, 5) for k, v in _mixed_anova(jack, group).items()}
    f_retrieved = _mixed_anova(retrieved, group)
    for key, printed in SMULDERS_F.items():
        assert round(f_jack[key], 1) == printed, (key, f_jack[key])
        assert f_retrieved[key] == pytest.approx(f_jack[key], rel=1e-9)


# Kiesel, Miller, Jolicoeur & Brisson (2008), read in full on 2026-10-13 (25 pages, authors'
# copy archived by the Wayback Machine on 2024-06-01; the university URL now answers 404).
# It prints the corrections only as Equation 1 (p. 252) and reports no t or F value:
# Tables 1-8 are Monte Carlo means, SDs, power and Type I error rates over data sets A and B,
# whose participant data are not published. Its one worked number is the linear
# interpolation of a criterion crossing (p. 254): -.98 at 62.50 ms, -1.02 at 66.41 ms,
# criterion -1.0, "64.46 ms".
KIESEL_INTERPOLATION = {
    "times_ms": (62.50, 66.41),
    "values": (-0.98, -1.02),
    "criterion": -1.0,
    "printed_ms": 64.46,
}


def test_req231_kiesel_worked_example() -> None:
    """Kiesel et al. (2008), read in full on 2026-10-13 (the 25 pages of the authors' copy,
    Wayback Machine snapshot of 2024-06-01, digest 756ZYZ3MIITWVGFS5VBH4NIYRDYSBBCM): the
    paper prints the corrections only as Equation 1 (p. 252), t_c = t / (n - 1) and
    F_c = F / (n - 1)^2, and no t or F value (Tables 1-8 are Monte Carlo summaries; its one
    worked number is the interpolation of p. 254, tested below). Its worked content is
    therefore Equation 1 applied to jackknifed scores: on 12 participants x 3 conditions,
    the corrected t and F of the subaverage scores equal the ordinary t and F of the
    retrieved individual scores (the identity behind Equation 1), to 1e-9."""

    rng = np.random.default_rng(2008)
    rows = 0.1 + rng.normal(0, 0.01, (12, 3)) + np.array([0.0, 0.004, 0.009])
    table = jackknife_latency(_constant_evokeds(rows, "p"), scorer="mean_amplitude",
                              window=(0.0, 0.001))
    wide = table.pivot(index="subject", columns="channel")
    jack, own = wide["jackknife"][CONDITIONS].to_numpy(), wide["retrieved"][CONDITIONS].to_numpy()
    order = [int(name[1:]) for name in wide.index]   # pivot sorts the subject names
    assert np.allclose(own, rows[order], rtol=0, atol=1e-12)
    n = len(rows)
    t_jack = sps.ttest_rel(jack[:, 2], jack[:, 0]).statistic
    t_own = sps.ttest_rel(own[:, 2], own[:, 0]).statistic
    assert jackknife_t(t_jack, n) == pytest.approx(t_own, rel=1e-9)
    assert jackknife_f(_repeated_f(jack), n) == pytest.approx(_repeated_f(own), rel=1e-9)


def _repeated_f(y: Any) -> float:
    """F of a one-way repeated-measures design (rows: participants, columns: conditions)."""

    n, k = y.shape
    gm = y.mean()
    ss_c = n * np.sum((y.mean(axis=0) - gm) ** 2)
    ss_s = k * np.sum((y.mean(axis=1) - gm) ** 2)
    ss_e = np.sum((y - gm) ** 2) - ss_c - ss_s
    return float((ss_c / (k - 1)) / (ss_e / ((n - 1) * (k - 1))))


def test_req231_kiesel_interpolation_example() -> None:
    """The worked number of Kiesel et al. (2008, p. 254): the criterion crossing,
    linearly interpolated between the printed samples, is 64.46 ms to the printed
    precision (62.50 + (-1.0 + .98) / (-1.02 + .98) x 3.91 = 64.455). It is scored here
    as the 50 % fractional peak latency of a waveform whose negative peak is -2.0."""

    from evoparam.models import classical_measures

    ex = KIESEL_INTERPOLATION
    times = np.array([0.050, ex["times_ms"][0] / 1e3, ex["times_ms"][1] / 1e3, 0.070, 0.075])
    y = np.array([0.0, *ex["values"], 2 * ex["criterion"], -1.0])
    out = classical_measures(
        times,
        y,
        window=(0.05, 0.075),
        polarity="negative",
        measures=("fractional_peak_latency",),
        fraction=0.5,
        n_neighbors=1,
    )
    ms = 1e3 * out["fractional_peak_latency"]
    assert ms == pytest.approx(64.455, abs=1e-9)
    assert abs(ms - ex["printed_ms"]) <= 0.005 + 1e-9


def test_req231_noise_free_retrieval() -> None:
    """Retrieved latencies equal the participants' own on noise-free averages (1e-9 s)."""

    shifts = np.linspace(-0.012, 0.015, 9)
    evokeds = []
    for k, d in enumerate(shifts):
        ts, _ = simulate_erp(
            1,
            components=[
                {"label": "N1", "latency": 0.1 + d, "fwhm": 0.04, "amplitude": -4.0 - 0.2 * k}
            ],
            snr=None,
            noise="none",
            sfreq=1000.0,
            tmin=-0.2,
            tmax=0.5,
            seed=k,
        )
        evokeds.append(ts.average())

    def centroid(ev):
        """Area-weighted mean time over the whole epoch (linear across participants of
        equal area: the latency measure that jackknifing retrieves exactly)."""

        y = ev.data / ev.data.sum(axis=1, keepdims=True)
        return (y * ev.times).sum(axis=1)

    direct = np.array([centroid(ev)[0] for ev in evokeds])
    assert np.allclose(direct, 0.1 + shifts, atol=1e-9)
    normalised = [
        Evoked(data=ev.data / -ev.data.sum(), times=ev.times, sfreq=ev.sfreq, ch_names=ev.ch_names)
        for ev in evokeds
    ]
    table = jackknife_latency(normalised, scorer=centroid)
    assert np.allclose(table["retrieved"], direct, rtol=0, atol=1e-9)
    # the default scorer (50 % fractional area latency) is not linear across participants:
    # retrieval is then close, not exact
    table = jackknife_latency(evokeds, window=(0.0, 0.3))
    own = [
        score_waveforms(ev.times, ev.data[0], "fractional_area_latency", window=(0.0, 0.3))
        for ev in evokeds
    ]
    error = np.abs(table["retrieved"].to_numpy() - np.array(own))
    assert 1e-6 < error.max() < 3e-3


def test_corrected_statistics_on_retrieved_scores() -> None:
    """t on the retrieved scores equals the corrected jackknife t, and SE matches."""

    rng = np.random.default_rng(3)
    a, b = [], []
    for k in range(12):
        lat = 0.1 + rng.normal(0, 0.008)
        for store, extra in ((a, 0.0), (b, 0.01)):
            ts, _ = simulate_erp(
                20,
                components=[
                    {"label": "N1", "latency": lat + extra, "fwhm": 0.04, "amplitude": -5.0}
                ],
                snr=None,
                noise="white",
                noise_sd=4.0,
                seed=100 * k + len(store),
            )
            store.append(ts.average())
    ja = jackknife_latency(a, window=(0.0, 0.3), scorer="fractional_area_latency")
    jb = jackknife_latency(b, window=(0.0, 0.3), scorer="fractional_area_latency")
    t_jack = sps.ttest_rel(jb["jackknife"], ja["jackknife"]).statistic
    t_retrieved = sps.ttest_rel(jb["retrieved"], ja["retrieved"]).statistic
    assert t_retrieved == pytest.approx(jackknife_t(t_jack, 12), rel=1e-9)
    assert jackknife_f(t_jack**2, 12) == pytest.approx(t_retrieved**2, rel=1e-9)
    summary = pd.DataFrame(ja.attrs["summary"])
    j = ja["jackknife"].to_numpy()
    assert summary["jackknife_se"].iloc[0] == pytest.approx(
        np.sqrt(11 / 12 * np.sum((j - j.mean()) ** 2))
    )
    assert t_retrieved > 2  # the 10 ms delay is detected
    assert ja.attrs["units"]["retrieved"] == "s"
    assert ja.attrs["provenance"]["steps"][-1]["function"] == "evoparam.stats.jackknife_latency"


def test_edge_cases() -> None:

    ts, _ = simulate_erp(5, components=["N1"], snr=3.0, noise="white", seed=0)
    ev = ts.average()
    with pytest.raises(DataError, match="at least 3"):
        jackknife_latency([ev, ev])
    other = ev.select(picks=[0], tmin=0.0)
    with pytest.raises(DataError, match="same channels"):
        jackknife_latency([ev, ev, other])
    with pytest.raises(DataError):
        jackknife_latency([ev, ev, ts])
