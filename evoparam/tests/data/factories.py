# SPDX-License-Identifier: Apache-2.0
"""Small, deterministic containers and results for the data-model tests."""

from typing import Any

import numpy as np
import pytest

from evoparam.data import (
    ArtifactResult,
    Component,
    GoodnessOfFit,
    GroupFit,
    HierarchicalFit,
    Provenance,
    ResponseFit,
    ResponseTest,
    ShapeBasis,
    TFFit,
    TrialFit,
    TrialSet,
)

pd = pytest.importorskip("pandas")

SFREQ = 1000.0


def make_trialset(
    n_trials: Any = 6,
    n_channels: Any = 3,
    seed: Any = 0,
    with_tables: Any = True,
    with_masks: Any = False,
) -> Any:
    rng = np.random.default_rng(seed)
    times = np.arange(-100, 300) / SFREQ
    data = rng.normal(0.0, 5.0, (n_trials, n_channels, times.size))
    data += -40.0 * np.exp(-0.5 * ((times - 0.025) / 0.004) ** 2)
    ch_names = [f"CH{i:02d}" for i in range(n_channels)]
    metadata = ch_info = masks = None
    if with_tables:
        metadata = pd.DataFrame(
            {
                "trial": np.arange(n_trials, dtype=np.int64),
                "condition": ["low", "high"] * (n_trials // 2) + ["low"] * (n_trials % 2),
                "stim_site": ["PT01-PT02"] * n_trials,
                "stim_current_mA": np.linspace(1.0, 6.0, n_trials),
                "rejected": np.zeros(n_trials, dtype=bool),
                "reject_reason": [""] * n_trials,
            }
        )
        ch_info = pd.DataFrame(
            {
                "name": ch_names,
                "type": ["ecog"] * n_channels,
                "x": np.arange(n_channels, dtype=float),
                "bad": [False] * n_channels,
            }
        )
    if with_masks:
        masks = np.zeros(data.shape, dtype=bool)
        masks[:, :, 98:106] = True
        data[masks] = np.nan
    return TrialSet(
        data=data,
        times=times,
        sfreq=SFREQ,
        ch_names=ch_names,
        metadata=metadata,
        ch_info=ch_info,
        masks=masks,
        event_times=np.arange(n_trials) * 5.0 + 12.5,
        species="human",
        modality="spes",
        provenance=Provenance(source={"path": "synthetic", "format": "array"}, random_state=seed),
    )


def make_component(latency: Any = 0.025, label: Any = "N1") -> Any:
    return Component(
        shape="gaussian",
        params=np.array([latency, -40.0, 0.004]),
        param_names=("latency", "amplitude", "width"),
        latency=latency,
        amplitude=-40.0,
        polarity=-1,
        fwhm=0.0094,
        onset=0.0165,
        offset=0.0335,
        area=-0.4,
        rise_time=0.0085,
        decay_time=0.0085,
        symmetry=0.0,
        sharpness=1.7,
        label=label,
        se={"latency": 0.0004},
        ci={"latency": (0.0242, 0.0258)},
    )


def make_gof() -> Any:
    return GoodnessOfFit(
        r2=0.91,
        adj_r2=0.9,
        rmse=4.8,
        mae=3.9,
        nrmse=0.06,
        aic=812.0,
        aicc=812.3,
        bic=830.1,
        n_params=4,
        n_samples=300,
        n_eff=120.5,
        residual_lag1_autocorr=0.41,
        baseline_sd=5.1,
    )


def make_response_test(_n_channels: Any = 3) -> Any:
    return ResponseTest(
        method="baseline_permutation",
        statistic=np.array([5.0, 0.4, 3.1]),
        p_value=np.array([0.001, 0.8, 0.02]),
        p_value_raw=np.array([0.0005, 0.8, 0.01]),
        significant=np.array([True, False, True]),
        duration=np.array([0.12, np.nan, 0.08]),
        onset=np.array([0.011, np.nan, 0.013]),
        offset=np.array([0.131, np.nan, 0.093]),
        snr_db=np.array([9.0, -2.0, np.inf]),
        rule=None,
        alpha=0.05,
        correction="fdr_bh",
        n_permutations=1000,
        window=(0.009, 0.5),
        settings={"tail": "both"},
    )


def make_response_fit(status: Any = "ok", with_curves: Any = True) -> Any:
    components = (
        [make_component(0.180, "N2"), make_component(0.025, "N1")] if status == "ok" else []
    )
    n = 300
    return ResponseFit(
        components=components,
        background={
            "mode": "exponential",
            "params": np.array([1.0, 0.05]),
            "param_names": ("amplitude", "tau"),
        },
        gof=make_gof() if status == "ok" else None,
        status=status,
        message="" if status == "ok" else "no significant response",
        window=(0.009, 0.3),
        baseline=(-0.1, -0.002),
        settings={
            "min_fwhm": 0.003,
            "shapes": ["gaussian"],
            "max_iter": None,
            "threshold": float("nan"),
        },
        classical={"peak_latency": 0.025, "mean_amplitude": float("nan")},
        model=np.linspace(0, 1, n) if with_curves else None,
        residual=np.zeros(n) if with_curves else None,
        existence=make_response_test() if status == "ok" else None,
        warnings=["edge"] if status == "ok" else [],
    )


def make_group_fit() -> Any:
    fits = [make_response_fit(), make_response_fit("no_response"), make_response_fit()]
    return GroupFit.from_fits(
        fits,
        ["CH00", "CH01", "CH02"],
        conditions=["3mA"] * 3,
        settings={"preset": {"name": "ccep", "version": "1"}},
    )


def make_trial_fit() -> Any:
    n_trials, n_comp = 5, 2
    table = pd.DataFrame(
        {
            "trial": np.repeat(np.arange(n_trials), n_comp),
            "component": np.tile(["N1", "N2"], n_trials),
            "latency": np.linspace(0.02, 0.2, n_trials * n_comp),
        }
    )
    return TrialFit(
        channel="CH00",
        template=make_response_fit(),
        shift=np.zeros(n_trials),
        scale=np.ones(n_trials),
        params=np.zeros((n_trials, n_comp, 3)),
        latency=np.full((n_trials, n_comp), 0.025),
        amplitude=np.full((n_trials, n_comp), -40.0),
        fwhm=np.full((n_trials, n_comp), 0.009),
        gof=np.full(n_trials, 0.8),
        converged=np.ones(n_trials, bool),
        jitter_sd={"N1": 0.0012},
        jitter_sd_null={"N1": 0.0004},
        amplitude_jitter_corrected={"N1": -42.0},
        trusted=np.ones(n_trials, bool),
        table=table,
        covariates=pd.DataFrame({"trend": np.arange(n_trials, dtype=float)}),
        realigned=make_trialset(n_trials=5),
    )


def make_hierarchical_fit() -> Any:
    summary = pd.DataFrame(
        {
            "level": ["channel"],
            "unit": ["CH00"],
            "component": ["N1"],
            "parameter": ["latency"],
            "mean": [0.025],
            "sd": [0.001],
            "hdi_low": [0.023],
            "hdi_high": [0.027],
            "ess_bulk": [np.nan],
            "r_hat": [np.nan],
        }
    )
    return HierarchicalFit(
        levels=("trial", "channel"),
        summary=summary,
        level_tables={"channel": summary.copy()},
        p_response=pd.DataFrame({"channel": ["CH00"], "p": [0.97]}),
        variance_components=pd.DataFrame({"level": ["trial"], "var": [1e-6]}),
        effects=None,
        engine="eb",
        diagnostics={"runtime_s": 1.5},
    )


def make_artifact_result() -> Any:
    cleaned = make_trialset(with_masks=True)
    return ArtifactResult(
        cleaned=cleaned,
        method="interpolate",
        window=(-0.002, 0.006),
        template=None,
        quality=np.linspace(0.5, 1.0, cleaned.n_trials),
        quality_components=pd.DataFrame({"residual_ratio": np.ones(cleaned.n_trials)}),
        settings={"buffer": 0.002},
    )


def make_shape_basis() -> Any:
    times = np.arange(-10, 290) / SFREQ
    curves = np.vstack([np.exp(-0.5 * ((times - c) / 0.01) ** 2) for c in (0.03, 0.15)])
    return ShapeBasis(
        curves=curves,
        times=times,
        weights=np.abs(np.eye(3, 2)),
        r2=np.array([0.8, 0.5, 0.1]),
        assignment=pd.DataFrame(
            {
                "group": ["A", "B", "C"],
                "shape_id": [0, 1, -1],
                "weight": [1.0, 1.0, 0.0],
                "r2": [0.8, 0.5, 0.1],
            }
        ),
        stability=np.array([0.95, 0.7]),
        n_shapes=2,
        method="nmf",
    )


def make_tf_fit() -> Any:
    shape = (3, 4, 5)
    zones = pd.DataFrame(
        {
            "zone_id": [0],
            "t_start": [0.01],
            "t_end": [0.1],
            "f_low": [80.0],
            "f_high": [150.0],
            "sign": [1],
            "peak_t": [0.03],
            "peak_f": [110.0],
            "peak_value": [3.2],
            "area": [0.5],
            "p_value": [0.01],
        }
    )
    return TFFit(
        freqs=np.array([10.0, 30.0, 80.0, 150.0]),
        times=np.linspace(0, 0.4, 5),
        maps={"total": np.ones(shape), "residual": np.zeros(shape)},
        tmaps=np.arange(60, dtype=float).reshape(shape),
        significant=np.zeros(shape, bool),
        zones=zones,
        zone_metrics=pd.DataFrame({"channel": ["A"], "zone_0": [0.4]}),
        method="nmf",
    )


def all_results() -> Any:
    """One instance of every savable type."""

    trials = make_trialset()
    return {
        "TrialSet": trials,
        "TrialSet_masked": make_trialset(with_masks=True),
        "TrialSet_bare": make_trialset(with_tables=False),
        "Evoked": trials.average(),
        "Component": make_component(),
        "GoodnessOfFit": make_gof(),
        "ResponseTest": make_response_test(),
        "ResponseFit": make_response_fit(),
        "ResponseFit_failed": make_response_fit("failed"),
        "GroupFit": make_group_fit(),
        "TrialFit": make_trial_fit(),
        "HierarchicalFit": make_hierarchical_fit(),
        "ArtifactResult": make_artifact_result(),
        "ShapeBasis": make_shape_basis(),
        "TFFit": make_tf_fit(),
    }


# The six public functions of the chain of spec REQ-294, in order, by the name each records
# (``evoparam.fit`` records the method that makes the GroupFit, ResponseModel.fit_group,
# and its existence test, test_response).
CHAIN_STEPS = (
    "evoparam.io.read",
    "evoparam.events.detect_events",
    "evoparam.artifact.remove_artifact",
    "evoparam.models.ResponseModel.fit_group",
    "evoparam.existence.test_response",
    "evoparam.trials.fit_trials",
)
# Steps that the chain needs between those six, each the record of a container created by
# a public function (REQ-294): the epochs cut around the detected events, and the average
# of the trials that the group fit is made from.
CHAIN_INTERMEDIATE = ("evoparam.io.epoch", "evoparam.data.TrialSet.average")


def bundled_recording() -> Any:
    """The bundled fixture of ``evoparam.selftest`` (``simulate_ccep``, seed 0, 3 channels,
    10 trials, a response on every channel), with the stimulation artifacts that the
    detection needs, laid end to end as one continuous recording: a TrialSet of one trial
    whose times start at 0."""

    from evoparam.sim import simulate_ccep

    trials, _ = simulate_ccep(n_trials=10, n_channels=3, p_response=1.0, seed=0)
    data = np.concatenate(list(trials.data), axis=-1)[np.newaxis]
    return TrialSet(
        data=data,
        times=np.arange(data.shape[-1]) / trials.sfreq,
        sfreq=trials.sfreq,
        ch_names=list(trials.ch_names),
        units=trials.units,
    )


def core_chain(folder: Any) -> Any:
    """The chain of REQ-294 on the bundled fixture: read -> detect_events -> (epoch) ->
    remove_artifact -> fit (with its existence test, test_response) -> fit_trials.

    The recording is saved as a container in ``folder`` and read back with
    ``evoparam.io.read``; returns every intermediate result."""

    import os

    import evoparam
    from evoparam.data import save_object

    recording = bundled_recording()
    path = save_object(recording, os.path.join(str(folder), "recording.evoparam"))
    raw = evoparam.io.read(path)
    events = evoparam.detect_events(raw)
    trials = evoparam.io.epoch(
        raw.data[0], raw.sfreq, events, -0.4, 0.9, t0=float(raw.times[0]), ch_names=raw.ch_names
    )
    clean = evoparam.artifact.remove_artifact(trials, method="template", window=(-0.002, 0.008))
    group = evoparam.fit(clean.cleaned, preset="ccep", random_state=0)
    single = evoparam.trials.fit_trials(clean.cleaned, group, random_state=0)
    return {
        "recording": recording,
        "read": raw,
        "events": events,
        "trials": trials,
        "clean": clean,
        "group": group,
        "single": single,
    }
