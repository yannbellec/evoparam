# SPDX-License-Identifier: Apache-2.0
"""Bootstrap intervals (REQ-230), the vectorised scores and the bootstrap interface."""

import copy

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from evoparam.data import TrialSet
from evoparam.errors import DataError, SettingsError
from evoparam.models import classical_measures
from evoparam.sim import simulate_erp
from evoparam.stats import SCORES, bootstrap, bootstrap_interval, score_waveforms

from .helpers import N1, WINDOW, replicates

###################################################################################################
###################################################################################################


@pytest.mark.parametrize("method", ["bca", "percentile"])
@pytest.mark.parametrize("distribution", ["normal", "lognormal"])
def test_req230_coverage(method, distribution):
    """93-97 % coverage of 95 % intervals of a mean amplitude over 30 trials.

    Normal: white noise around a fixed component. Log-normal: the trial amplitude factor
    is log-normal (coefficient of variation 0.5) with little noise. 10 000 simulated
    channels; binomial SE of the coverage 0.002.
    """

    kwargs = (
        dict(noise="white", snr=1.0)
        if distribution == "normal"
        else dict(noise="white", snr=None, noise_sd=0.5, amplitude_cv=0.5)
    )
    ts, true = replicates(30, 10000, seed=31 if distribution == "normal" else 32, **kwargs)
    table = bootstrap(
        ts, "mean_amplitude", window=WINDOW, n_boot=1000, method=method, by=(), random_state=5
    )
    covered = (table["ci_low"] <= true) & (true <= table["ci_high"])
    assert 0.93 <= covered.mean() <= 0.97, covered.mean()
    assert set(table["method"]) == {method}
    # the textbook interval (no small-sample expansion) is narrower
    plain = bootstrap(
        ts,
        "mean_amplitude",
        window=WINDOW,
        n_boot=1000,
        method=method,
        by=(),
        random_state=5,
        expand=False,
    )
    assert np.all(plain["ci_high"] - plain["ci_low"] < table["ci_high"] - table["ci_low"])
    assert ((plain["ci_low"] <= true) & (true <= plain["ci_high"])).mean() < covered.mean()


def test_scores_match_classical_measures():

    rng = np.random.default_rng(0)
    times = np.linspace(-0.05, 0.3, 176)
    waves = np.cumsum(rng.normal(size=(12, 176)), axis=1)
    waves[:3] = -np.exp(-0.5 * ((times - 0.1) / 0.02) ** 2) + 0.1 * rng.normal(size=(3, 176))
    names = {
        "mean_amplitude": "mean",
        "peak_amplitude": "peak",
        "peak_latency": "peak",
        "area": "area",
        "fractional_area_latency": "fractional_area_latency",
        "fractional_peak_latency": "fractional_peak_latency",
    }
    for polarity in ("negative", "positive", "absolute"):
        for area in ("total", "positive", "negative", "rectified"):
            for base in ("zero", "fraction_of_peak"):
                kw = dict(
                    window=(0.02, 0.25),
                    polarity=polarity,
                    area=area,
                    area_baseline=base,
                    fraction=0.3,
                )
                for score in SCORES:
                    got = score_waveforms(times, waves, score, **kw)
                    for k, y in enumerate(waves):
                        ref = classical_measures(times, y, measures=(names[score],), **kw)
                        assert got[k] == pytest.approx(
                            ref[score], rel=1e-12, abs=1e-12, nan_ok=True
                        ), (score, polarity, area, base, k)


def test_callable_and_named_score_agree():

    ts, _ = simulate_erp(25, n_channels=2, components=[N1], snr=2.0, noise="white", seed=3)

    def peak(sub):
        avg = sub.average()
        return np.array(
            [
                classical_measures(avg.times, y, window=(0.05, 0.15))["fractional_area_latency"]
                for y in avg.data
            ]
        )

    named = bootstrap(
        ts, "fractional_area_latency", window=(0.05, 0.15), n_boot=300, random_state=1
    )
    called = bootstrap(ts, peak, n_boot=300, random_state=1)
    for col in ("estimate", "ci_low", "ci_high", "se", "bias", "z0", "acceleration"):
        assert np.allclose(named[col], called[col], rtol=0, atol=1e-12, equal_nan=True), col
    assert named["channel"].tolist() == list(ts.ch_names)
    assert called["statistic"].tolist() == ["0", "1"]
    assert named.attrs["units"]["estimate"] == "s"


def test_bca_matches_scipy():
    """Same BCa definition as scipy.stats.bootstrap (independent implementation)."""

    rng = np.random.default_rng(7)
    x = rng.lognormal(0, 0.8, 40)
    times = np.array([0.0, 0.01])
    ts = TrialSet(
        data=np.repeat(x[:, None, None], 2, axis=2), times=times, sfreq=100.0, ch_names=["a"]
    )
    ours = bootstrap(
        ts, "mean_amplitude", window=(0, 0.01), n_boot=40000, by=(), random_state=0, expand=False
    )
    ref = sps.bootstrap((x,), np.mean, n_resamples=40000, method="BCa", random_state=1)
    assert ours["ci_low"].iloc[0] == pytest.approx(ref.confidence_interval.low, rel=0.01)
    assert ours["ci_high"].iloc[0] == pytest.approx(ref.confidence_interval.high, rel=0.01)
    assert ours["acceleration"].iloc[0] > 0  # right-skewed: positive acceleration


def test_strata_keep_their_size():

    ts, _ = simulate_erp(24, components=[N1], snr=2.0, noise="white", seed=2)
    meta = ts.metadata.copy()
    meta["stim_site"] = ["A-B"] * 6 + ["C-D"] * 18
    ts = TrialSet(
        data=ts.data, times=ts.times, sfreq=ts.sfreq, ch_names=ts.ch_names, metadata=meta
    )
    seen = []

    def share(sub):
        value = float(np.mean(sub.metadata["stim_site"] == "A-B"))
        seen.append(value)
        return value

    table = bootstrap(ts, share, n_boot=50, method="percentile", random_state=0)
    assert set(seen[1:51]) == {0.25}  # 6 of 24 in every resample
    table = bootstrap(ts, share, n_boot=50, method="percentile", by=(), random_state=0)
    assert len(set(seen)) > 1
    assert table.attrs["settings"]["n_strata"] == 1


def test_bca_falls_back_with_few_values():

    ts, _ = simulate_erp(6, components=[N1], snr=1.0, noise="white", sfreq=100.0, seed=4)
    with pytest.warns(UserWarning, match="fell back to the percentile"):
        table = bootstrap(ts, "peak_latency", window=(0.08, 0.12), n_boot=200, random_state=0)
    assert table["method"].tolist() == ["percentile"]
    assert table["ci_low"].iloc[0] <= table["ci_high"].iloc[0]


def test_interval_details_and_errors():

    out = bootstrap_interval(1.0, [1.0] * 50, method="bca")
    assert out["method"] == "percentile" and "distinct" in out["reason"]
    out = bootstrap_interval(np.nan, np.arange(50.0), method="percentile")
    assert np.isnan(out["low"])
    with pytest.raises(SettingsError):
        bootstrap_interval(0.0, np.arange(50.0), method="studentized")
    ts, _ = simulate_erp(10, components=[N1], snr=1.0, noise="white", seed=0)
    with pytest.raises(DataError):
        bootstrap(ts.average(), "mean_amplitude")
    with pytest.raises(SettingsError):
        bootstrap(ts, "median_amplitude")
    with pytest.raises(SettingsError):
        bootstrap(ts, "mean_amplitude", n_boot=1)
    before = copy.deepcopy(ts)
    table = bootstrap(ts, "mean_amplitude", n_boot=20, random_state=0)
    assert ts == before  # input unchanged
    steps = table.attrs["provenance"]["steps"]
    assert steps[-1]["function"] == "evoparam.stats.bootstrap"
    assert steps[-1]["settings"]["n_boot"] == 20
    assert isinstance(table, pd.DataFrame)
