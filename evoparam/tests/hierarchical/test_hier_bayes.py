# SPDX-License-Identifier: Apache-2.0
"""Full Bayesian engines (REQ-172 to REQ-174, REQ-176). The reference runs take minutes:
they run with EVOPARAM_SLOW_TESTS=1 (weekly in CI, spec §5.8); a short run of each engine
is part of the regular suite."""

import os
import time
from typing import Any

import numpy as np
import pytest

from evoparam.hierarchical import compare_hierarchical, fit_hierarchical
from evoparam.tests.hierarchical.helpers import simulate_group

# Fewest channels with a posterior latency interval in the short run (3 subjects x 8
# channels, 80 % responding).
MIN_CHANNELS_WITH_CI = 5
# Latency models compared by LOO: ex-Gaussian and normal.
N_MODELS = 2
# NUTS diagnostics of REQ-173: largest R-hat, smallest effective sample size, largest
# fraction of divergent transitions.
MAX_R_HAT = 1.01
MIN_ESS = 400
MAX_DIVERGENT_FRACTION = 0.01
# Fewest variance components whose posterior interval holds the truth (REQ-172).
MIN_COVERAGE = 0.9
# Longest NUTS time per subject (s, REQ-176).
MAX_SECONDS_PER_SUBJECT = 300

SLOW = os.environ.get("EVOPARAM_SLOW_TESTS") == "1"
slow = pytest.mark.skipif(not SLOW, reason="set EVOPARAM_SLOW_TESTS=1 for the NUTS reference runs")
ENGINES = ["numpyro", "pymc"]


def _engine_available(engine: Any) -> None:
    pytest.importorskip("jax" if engine == "numpyro" else "pytensor")
    pytest.importorskip(engine)


def reference_dataset(seed: Any = 7) -> Any:
    """The simulated reference dataset: 10 subjects x 20 channels x 10 trials, half
    responding, per-trial SNR 2, 2 ms jitter."""

    return simulate_group(
        n_subjects=10, n_channels=20, n_trials=10, p_respond=0.5, snr=2.0, seed=seed
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_short_run_of_each_engine(engine: Any) -> None:

    _engine_available(engine)
    sets, truth = simulate_group(
        n_subjects=3, n_channels=8, n_trials=8, p_respond=0.8, snr=3.0, seed=11
    )
    fit = fit_hierarchical(
        sets,
        components=["N1"],
        engine=engine,
        n_chains=2,
        n_warmup=150,
        n_draws=150,
        random_state=0,
        formula="latency ~ age",
    )
    assert fit.engine == engine
    d = fit.diagnostics["bayes"]["N1"]
    assert {"r_hat_max", "ess_min", "divergences", "ppc_p_latency_spread"} <= set(d)
    ch = fit.level_tables["channel"]
    used = ch["latency_ci_low"].notna()
    assert used.sum() >= MIN_CHANNELS_WITH_CI
    assert np.all(ch.loc[used, "latency_ci_low"] <= ch.loc[used, "latency_ci_high"])
    assert ch["p_response"].between(0, 1).all()
    assert set(fit.variance_components["method"]) == {f"NUTS ({engine})"}
    assert "age" in set(fit.effects["term"])


def test_compare_needs_bayesian_fits() -> None:

    from evoparam.errors import SettingsError

    sets, _ = simulate_group(
        n_subjects=2, n_channels=6, n_trials=8, p_respond=1.0, snr=3.0, seed=12
    )
    with pytest.raises(SettingsError, match="Bayesian"):
        compare_hierarchical([fit_hierarchical(sets, components=["N1"])])


def test_loo_comparison_of_two_latency_models() -> None:

    _engine_available("numpyro")
    sets, _ = simulate_group(
        n_subjects=3, n_channels=8, n_trials=8, p_respond=0.8, snr=3.0, seed=13
    )
    fits = [
        fit_hierarchical(
            sets,
            components=["N1"],
            engine="numpyro",
            n_chains=2,
            n_warmup=150,
            n_draws=150,
            random_state=0,
            latency_dist=dist,
        )
        for dist in ("exgaussian", "normal")
    ]
    table = compare_hierarchical(fits)
    assert len(table) == N_MODELS


@pytest.fixture(scope="module")
def reference_fits() -> Any:
    sets, truth = reference_dataset()
    out = {}
    for engine in ENGINES:
        _engine_available(engine)
        out[engine] = fit_hierarchical(
            sets, components=["N1"], engine=engine, n_jobs=-1, random_state=0
        )
    return out, truth


@slow
@pytest.mark.parametrize("engine", ENGINES)
def test_req173_nuts_diagnostics(reference_fits: Any, engine: Any) -> None:

    fits, _ = reference_fits
    d = fits[engine].diagnostics["bayes"]["N1"]
    assert d["r_hat_max"] <= MAX_R_HAT
    assert d["ess_min"] >= MIN_ESS
    assert d["divergent_fraction"] < MAX_DIVERGENT_FRACTION


@slow
def test_req174_engines_agree(reference_fits: Any) -> None:

    fits, _ = reference_fits
    a, b = fits["numpyro"], fits["pymc"]
    pa, pb = (f.diagnostics["bayes"]["N1"]["posterior"] for f in (a, b))
    va, vb = (f.variance_components.set_index(["parameter", "level"]) for f in (a, b))
    for key in va.index:
        assert abs(va.loc[key, "sd"] - vb.loc[key, "sd"]) <= va.loc[key, "sd_se"], key
    assert set(pa) == set(pb)
    for column in ("latency", "amplitude"):
        ca = a.level_tables["channel"].set_index(["subject", "channel"])
        cb = b.level_tables["channel"].set_index(["subject", "channel"])
        ok = ca[column].notna()
        diff = (ca.loc[ok, column] - cb.loc[ok, column]).abs()
        assert np.all(diff <= ca.loc[ok, f"{column}_se"]), column
    sa = a.level_tables["subject"].set_index("subject")
    sb = b.level_tables["subject"].set_index("subject")
    assert np.all((sa["latency"] - sb["latency"]).abs() <= sa["latency_se"])


def _truth_sds(truth: Any) -> Any:
    resp = truth[truth["respond"]]
    within = resp["latency"] - resp.groupby("subject")["latency"].transform("mean")
    n_sub = resp["subject"].nunique()
    return {
        ("latency", "channel"): np.sqrt(np.sum(within**2) / (len(resp) - n_sub)),
        ("latency", "trial"): 0.002,
    }


@slow
def test_req172_variance_components_in_the_posterior_interval() -> None:

    _engine_available("numpyro")
    covered = []
    for seed in range(20, 30):
        sets, truth = reference_dataset(seed)
        fit = fit_hierarchical(
            sets, components=["N1"], engine="numpyro", n_jobs=-1, random_state=0
        )
        vc = fit.variance_components.set_index(["parameter", "level"])
        for key, value in _truth_sds(truth).items():
            covered.append(vc.loc[key, "ci_low"] <= value <= vc.loc[key, "ci_high"])
    assert np.mean(covered) >= MIN_COVERAGE


@slow
@pytest.mark.parametrize("engine", ENGINES)
def test_req176_nuts_time_per_subject(engine: Any) -> None:

    _engine_available(engine)
    sets, _ = simulate_group(
        n_subjects=3, n_channels=100, n_trials=10, p_respond=0.5, snr=3.0, n2=0.8, seed=31
    )
    start = time.perf_counter()
    fit = fit_hierarchical(sets, components=["N1", "N2"], engine=engine, n_jobs=-1, random_state=0)
    per_subject = (time.perf_counter() - start) / 3
    assert set(fit.diagnostics["bayes"]) == {"N1", "N2"}
    assert per_subject <= MAX_SECONDS_PER_SUBJECT
