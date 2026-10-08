"""Jackknife scoring over participants (REQ-231)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from evoparam.data import Evoked
from evoparam.errors import DataError
from evoparam.sim import simulate_erp
from evoparam.stats import jackknife_f, jackknife_latency, jackknife_t, score_waveforms

###################################################################################################
###################################################################################################

# Smulders (2010), Table 1: hypothetical Age (between, 5 participants per group) x Sleep
# deprivation (within) design; the retrieved latencies (ms) are the participants' own.
SMULDERS_LATENCIES = {
    "young": [[479, 503, 659], [471, 580, 593], [490, 476, 654], [440, 434, 620],
              [270, 457, 374]],
    "middle": [[616, 459, 512], [535, 377, 510], [589, 451, 505], [514, 463, 523],
               [546, 600, 600]],
    "old": [[596, 578, 707], [556, 694, 694], [594, 626, 667], [603, 590, 748],
            [501, 662, 634]],
}
# Printed subaverage (jackknife) scores of the young group without sleep deprivation.
SMULDERS_SUBAVERAGES = [417.75, 419.75, 415.0, 427.5, 470.0]
# Printed F statistics (identical for the retrieved scores and the corrected subaverages).
SMULDERS_F = {"age": 9.2, "sleep": 9.8, "interaction": 4.9}
CONDITIONS = ["none", "mild", "heavy"]


def _constant_evokeds(rows, group):
    """One Evoked per participant whose 'mean amplitude' on each channel is a latency."""

    times = np.array([0.0, 0.001])
    return [Evoked(data=np.repeat(np.asarray(row, float)[:, None], 2, axis=1), times=times,
                   sfreq=1000., ch_names=CONDITIONS,
                   metadata=pd.DataFrame({"subject": [f"{group}{k}"]}))
            for k, row in enumerate(rows)]


def _mixed_anova(y, group):
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
    return {"age": (ss_a / (a - 1)) / (ss_s / df_s), "sleep": (ss_b / (b - 1)) / (ss_e / df_e),
            "interaction": (ss_ab / ((a - 1) * (b - 1))) / (ss_e / df_e)}


def test_req231_smulders_worked_example():

    jack, retrieved, group = [], [], []
    for name, rows in SMULDERS_LATENCIES.items():
        table = jackknife_latency(_constant_evokeds(rows, name), scorer="mean_amplitude",
                                  window=(0.0, 0.001))
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


@pytest.mark.xfail(strict=False, reason="REQ-231, Kiesel et al. (2008) part: the worked "
                   "example of the paper could not be read in the build environment (the "
                   "publisher copy is not reachable), so its printed values are not frozen "
                   "here; the corrections it uses (t / (n - 1), F / (n - 1)^2) are tested "
                   "on the Smulders example and on simulations")
def test_req231_kiesel_worked_example():

    printed = None            # the worked example of [LIT-erpmeas-2], to be transcribed
    assert printed is not None


def test_req231_noise_free_retrieval():
    """Retrieved latencies equal the participants' own on noise-free averages (1e-9 s)."""

    shifts = np.linspace(-0.012, 0.015, 9)
    evokeds = []
    for k, d in enumerate(shifts):
        ts, _ = simulate_erp(1, components=[{"label": "N1", "latency": 0.1 + d, "fwhm": 0.04,
                                             "amplitude": -4.0 - 0.2 * k}],
                             snr=None, noise="none", sfreq=1000., tmin=-0.2, tmax=0.5, seed=k)
        evokeds.append(ts.average())

    def centroid(ev):
        """Area-weighted mean time over the whole epoch (linear across participants of
        equal area: the latency measure that jackknifing retrieves exactly)."""

        y = ev.data / ev.data.sum(axis=1, keepdims=True)
        return (y * ev.times).sum(axis=1)

    direct = np.array([centroid(ev)[0] for ev in evokeds])
    assert np.allclose(direct, 0.1 + shifts, atol=1e-9)
    normalised = [Evoked(data=ev.data / -ev.data.sum(), times=ev.times, sfreq=ev.sfreq,
                         ch_names=ev.ch_names) for ev in evokeds]
    table = jackknife_latency(normalised, scorer=centroid)
    assert np.allclose(table["retrieved"], direct, rtol=0, atol=1e-9)
    # the default scorer (50 % fractional area latency) is not linear across participants:
    # retrieval is then close, not exact
    table = jackknife_latency(evokeds, window=(0.0, 0.3))
    own = [score_waveforms(ev.times, ev.data[0], "fractional_area_latency", window=(0.0, 0.3))
           for ev in evokeds]
    error = np.abs(table["retrieved"].to_numpy() - np.array(own))
    assert 1e-6 < error.max() < 3e-3


def test_corrected_statistics_on_retrieved_scores():
    """t on the retrieved scores equals the corrected jackknife t, and SE matches."""

    rng = np.random.default_rng(3)
    a, b = [], []
    for k in range(12):
        lat = 0.1 + rng.normal(0, 0.008)
        for store, extra in ((a, 0.0), (b, 0.01)):
            ts, _ = simulate_erp(20, components=[{"label": "N1", "latency": lat + extra,
                                                  "fwhm": 0.04, "amplitude": -5.0}],
                                 snr=None, noise="white", noise_sd=4., seed=100 * k + len(store))
            store.append(ts.average())
    ja = jackknife_latency(a, window=(0.0, 0.3), scorer="fractional_area_latency")
    jb = jackknife_latency(b, window=(0.0, 0.3), scorer="fractional_area_latency")
    t_jack = sps.ttest_rel(jb["jackknife"], ja["jackknife"]).statistic
    t_retrieved = sps.ttest_rel(jb["retrieved"], ja["retrieved"]).statistic
    assert t_retrieved == pytest.approx(jackknife_t(t_jack, 12), rel=1e-9)
    assert jackknife_f(t_jack ** 2, 12) == pytest.approx(t_retrieved ** 2, rel=1e-9)
    summary = pd.DataFrame(ja.attrs["summary"])
    j = ja["jackknife"].to_numpy()
    assert summary["jackknife_se"].iloc[0] == pytest.approx(
        np.sqrt(11 / 12 * np.sum((j - j.mean()) ** 2)))
    assert t_retrieved > 2                             # the 10 ms delay is detected
    assert ja.attrs["units"]["retrieved"] == "s"
    assert ja.attrs["provenance"]["steps"][-1]["function"] == "evoparam.stats.jackknife_latency"


def test_edge_cases():

    ts, _ = simulate_erp(5, components=["N1"], snr=3., noise="white", seed=0)
    ev = ts.average()
    with pytest.raises(DataError, match="at least 3"):
        jackknife_latency([ev, ev])
    other = ev.select(picks=[0], tmin=0.0)
    with pytest.raises(DataError, match="same channels"):
        jackknife_latency([ev, ev, other])
    with pytest.raises(DataError):
        jackknife_latency([ev, ev, ts])
