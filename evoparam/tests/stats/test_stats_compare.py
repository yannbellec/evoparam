# SPDX-License-Identifier: Apache-2.0
"""Condition contrasts by permutation (REQ-232)."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from evoparam.errors import DataError, SettingsError
from evoparam.sim import simulate_erp
from evoparam.stats import compare_conditions, hedges_g, score_waveforms

N_SIM = 4000

###################################################################################################
###################################################################################################


def _rate(table: Any) -> Any:
    return float(np.mean(table["p_value_raw"] <= 0.05))


def test_req232_type_one_error_trial_level() -> None:
    """Trials of one channel, two conditions assigned at random (unpaired)."""

    ts, _ = simulate_erp(
        20 * N_SIM,
        components=["N1"],
        snr=1.0,
        noise="pink",
        sfreq=250.0,
        tmin=-0.1,
        tmax=0.3,
        seed=11,
        keep_arrays=False,
    )
    score = score_waveforms(ts.times, ts.data[:, 0], "mean_amplitude", window=(0.1, 0.2))
    rng = np.random.default_rng(1)
    table = pd.DataFrame(
        {
            "sim": np.repeat(np.arange(N_SIM), 20),
            "trial": np.arange(20 * N_SIM),
            "condition": np.where(rng.random(20 * N_SIM) < 0.5, "a", "b"),
            "amplitude": score,
        }
    )
    out = compare_conditions(
        table,
        parameter="amplitude",
        condition="condition",
        unit="trial",
        paired=False,
        n_permutations=999,
        correction="none",
        by=["sim"],
        random_state=2,
    )
    assert len(out) == N_SIM
    assert 0.04 <= _rate(out) <= 0.06, _rate(out)


def test_req232_type_one_error_channel_level() -> None:
    """Channel latencies (CCEP N1) in two conditions with no true difference (paired)."""

    rng = np.random.default_rng(3)
    n_ch = 16
    latency = rng.normal(0.025, 0.004, (N_SIM, n_ch))  # channel effects
    values = latency[..., None] + rng.normal(0, 0.003, (N_SIM, n_ch, 2))
    table = pd.DataFrame(
        {
            "sim": np.repeat(np.arange(N_SIM), n_ch * 2),
            "channel": np.tile(np.repeat([f"c{i}" for i in range(n_ch)], 2), N_SIM),
            "condition": np.tile(["1mA", "3mA"], N_SIM * n_ch),
            "latency": values.ravel(),
        }
    )
    out = compare_conditions(
        table,
        parameter="latency",
        condition="condition",
        unit="channel",
        n_permutations=999,
        correction="none",
        by=["sim"],
        random_state=4,
    )
    assert 0.04 <= _rate(out) <= 0.06, _rate(out)
    assert out["difference_ms"].notna().all()


@pytest.mark.parametrize("paired", [True, False])
def test_req232_type_one_error_subject_level(paired: Any) -> None:
    """Subject means (several trials each) compared within subjects (paired) or between
    two groups of subjects (unpaired), skewed (log-normal) values."""

    rng = np.random.default_rng(5 if paired else 6)
    n_sub, n_trials = 14, 5
    subject_level = rng.lognormal(0, 0.5, (N_SIM, n_sub))
    rows = n_sub * 2 * n_trials if paired else n_sub * n_trials
    if paired:
        cond = np.tile(np.repeat(["a", "b"], n_trials), N_SIM * n_sub)
        subj = np.tile(np.repeat(np.arange(n_sub), 2 * n_trials), N_SIM)
    else:
        cond = np.tile(
            np.repeat(np.where(np.arange(n_sub) < n_sub // 2, "a", "b"), n_trials), N_SIM
        )
        subj = np.tile(np.repeat(np.arange(n_sub), n_trials), N_SIM)
    sim = np.repeat(np.arange(N_SIM), rows)
    value = subject_level[sim, subj] * rng.lognormal(0, 0.3, sim.size)
    table = pd.DataFrame(
        {
            "sim": sim,
            "subject": subj,
            "trial": np.arange(sim.size),
            "condition": cond,
            "amplitude": value,
        }
    )
    out = compare_conditions(
        table,
        parameter="amplitude",
        condition="condition",
        unit="subject",
        paired=paired,
        n_permutations=999,
        correction="none",
        by=["sim"],
        random_state=7,
    )
    assert 0.04 <= _rate(out) <= 0.06, _rate(out)
    assert set(out["n_a"]) == {n_sub if paired else n_sub // 2}


def test_req232_refuses_trials_across_subjects() -> None:

    rng = np.random.default_rng(0)
    table = pd.DataFrame(
        {
            "subject": np.repeat(["s1", "s2"], 20),
            "trial": np.arange(40),
            "condition": np.tile(["a", "b"], 20),
            "latency": rng.normal(size=40),
        }
    )
    with pytest.raises(SettingsError, match="not exchangeable"):
        compare_conditions(
            table, parameter="latency", condition="condition", unit="trial", paired=False
        )
    out = compare_conditions(
        table,
        parameter="latency",
        condition="condition",
        unit="trial",
        paired=False,
        by=["subject"],
        n_permutations=200,
        random_state=0,
    )
    assert out["subject"].tolist() == ["s1", "s2"]
    ccep = pd.DataFrame(
        {
            "subject": np.repeat(["s1", "s2"], 8),
            "channel": np.tile(np.repeat(["A", "B", "C", "D"], 2), 2),
            "condition": np.tile(["a", "b"], 8),
            "latency": rng.normal(size=16),
        }
    )
    with pytest.raises(SettingsError, match="unit='subject'"):
        compare_conditions(ccep, parameter="latency", condition="condition", unit="channel")


def test_effect_detected_with_effect_sizes() -> None:

    rng = np.random.default_rng(1)
    base = rng.normal(0.025, 0.004, 12)
    table = pd.DataFrame(
        {
            "channel": np.repeat([f"c{i}" for i in range(12)], 2),
            "condition": np.tile(["slow", "fast"], 12),
            "latency": np.ravel(np.c_[base + 0.004, base]) + rng.normal(0, 0.001, 24),
        }
    )
    out = compare_conditions(
        table, parameter="latency", condition="condition", unit="channel", random_state=0
    )
    row = out.iloc[0]
    assert row["exact"]
    assert row["n_permutations"] == 2**12
    assert row["p_value"] < 0.01
    assert row["difference_ms"] == pytest.approx(4, abs=1)
    assert row["ci_low_ms"] < row["difference_ms"] < row["ci_high_ms"]
    d = table["latency"].to_numpy()[0::2] - table["latency"].to_numpy()[1::2]
    assert row["hedges_g"] == pytest.approx(d.mean() / d.std(ddof=1) * (1 - 3 / (4 * 11 - 1)))
    assert row["hedges_g"] == pytest.approx(hedges_g(d, paired=True))
    assert out.attrs["units"]["difference"] == "s"


def test_status_rows_are_excluded_and_counted() -> None:

    rng = np.random.default_rng(2)
    table = pd.DataFrame(
        {
            "subject": np.repeat([f"s{i}" for i in range(10)], 2),
            "condition": np.tile(["a", "b"], 10),
            "latency": rng.normal(size=20),
            "status": ["ok"] * 16 + ["no_response", "ok", "failed", "ok"],
        }
    )
    table.loc[3, "latency"] = np.nan
    out = compare_conditions(
        table, parameter="latency", condition="condition", n_permutations=500, random_state=0
    )
    assert out["n_excluded"].iloc[0] == 2
    assert out["n_missing"].iloc[0] == 1
    assert out["n_a"].iloc[0] == 7  # s1 (missing b), s8, s9 dropped


def test_several_conditions_and_errors() -> None:

    rng = np.random.default_rng(4)
    table = pd.DataFrame(
        {
            "subject": np.repeat(np.arange(8).astype(str), 3),
            "condition": np.tile(["1", "2", "3"], 8),
            "latency": rng.normal(size=24),
        }
    )
    out = compare_conditions(
        table,
        parameter="latency",
        condition="condition",
        n_permutations=300,
        random_state=0,
        correction="holm",
    )
    assert list(zip(out["condition_a"], out["condition_b"], strict=False)) == [
        ("1", "2"),
        ("1", "3"),
        ("2", "3"),
    ]
    assert np.all(out["p_value"] >= out["p_value_raw"])
    assert out["difference_ms"].notna().all()  # a latency: also reported in ms
    with pytest.raises(SettingsError):
        compare_conditions(table, parameter="latency", condition="condition", unit="session")
    with pytest.raises(SettingsError, match="paired=True"):
        compare_conditions(table, parameter="latency", condition="condition", paired=False)
    with pytest.raises(DataError):
        compare_conditions(
            table[table["condition"] == "1"], parameter="latency", condition="condition"
        )
