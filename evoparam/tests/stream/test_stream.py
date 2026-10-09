"""Online mode: agreement with fit_trials, chunk independence, latency (REQ-324, REQ-347) and
the running existence test (REQ-347, REQ-348).

The latency budget itself is timed only with EVOPARAM_PERF_TESTS=1 (CI machines are about
twice as slow as the reference machine); ``test_latency_stats`` always runs.
"""

import dataclasses
import json
import math
import os
import pathlib
import sys
import warnings

import numpy as np
import pytest

import evoparam.stream
from evoparam.cli import main
from evoparam.data import TrialSet
from evoparam.errors import DataError, SettingsError
from evoparam.events import OnlinePulseDetector, detect_pulses
from evoparam.existence import SequentialSignFlip, alpha_spending, sign_flip_test
from evoparam.stream import INTENDED_USE, STATUSES, PulseEstimate, StreamFitter, replay
from evoparam.trials import FixedShapeModel, fit_trials

from .helpers import N1, SFREQ, TIMES, channel_fits, gauss, no_pulses, simulate

BENCHMARKS = pathlib.Path(__file__).parents[3] / "benchmarks"
FIELDS = ("pulse_time", "channel", "component", "amplitude", "latency", "scale", "shift",
          "existence_z", "present", "r2")


@pytest.fixture(scope="module")
def pulses():
    """200 simulated pulses on two channels, their channel fits and the offline trial fits."""

    ts = simulate(n_trials=200, n_channels=2, seed=0)
    group = channel_fits(ts)
    assert all(fit.status == "ok" and len(fit.components) == 1 for fit in group.fits)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        offline = fit_trials(ts, group, mode="shift_scale", max_shift=0.005, r2_min=0.0)
    return ts, group, offline


@pytest.fixture(scope="module")
def mixed():
    """60 pulses on 8 channels, a response on the first 4, one channel template for all."""

    ts = simulate(n_trials=60, n_channels=8, responsive=range(4), seed=5)
    template = channel_fits(simulate(n_trials=40, n_channels=1, seed=6)).fits[0]
    return ts, template


def _fitter(template, ts, **kwargs):
    kwargs.setdefault("detector", no_pulses)
    return StreamFitter(template, sfreq=ts.sfreq, ch_names=ts.ch_names, **kwargs)


def _table(estimates, channel):
    rows = [e for e in estimates if e.channel == channel]
    return {field: np.array([getattr(e, field) for e in rows]) for field in FIELDS}


###################################################################################################
## REQ-324 / REQ-347: agreement with fit_trials(mode="shift_scale")

def test_req347_agreement_with_fit_trials(pulses):

    ts, group, offline = pulses
    fitter = _fitter(group, ts, existence_max_pulses=ts.n_trials)
    estimates = replay(ts, fitter, chunk_size=32)
    assert len(estimates) == ts.n_trials * 2
    assert {e.status for e in estimates} <= {"ok", "late"}
    for fit in offline.fits:
        online = _table(estimates, fit.channel)
        rel = np.abs(online["amplitude"] / fit.amplitude[:, 0] - 1)
        lag = np.abs(online["latency"] - fit.latency[:, 0]) * SFREQ
        assert rel.max() <= 0.01, rel.max()
        assert lag.max() <= 1.0, lag.max()
        assert np.allclose(online["pulse_time"], (np.arange(ts.n_trials) * ts.n_times + 1024)
                           / SFREQ)


@pytest.fixture(scope="module", params=["common jitter", "independent jitter"])
def two_components(request):
    """200 pulses with two components (N1 -40 uV at 30 ms, N2 +30 uV at 120 ms, jitter SD
    1.5 ms, the same for both or drawn for each, amplitude CV 20 %), the channel fit and
    the offline trial fits."""

    rng = np.random.default_rng(0)
    data = rng.normal(0, 10, (200, 1, len(TIMES)))
    for i in range(200):
        jit = rng.normal(0, 0.0015)
        jit2 = jit if request.param == "common jitter" else rng.normal(0, 0.0015)
        data[i, 0] += -40 * (1 + 0.2 * rng.normal()) * gauss(TIMES, 0.03 + jit, 0.02)
        data[i, 0] += 30 * (1 + 0.2 * rng.normal()) * gauss(TIMES, 0.12 + jit2, 0.05)
    ts = TrialSet(data=data, times=TIMES, sfreq=SFREQ, ch_names=["c0"])
    group = channel_fits(ts)
    assert len(group.fits[0].components) == 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        offline = fit_trials(ts, group, mode="shift_scale", max_shift=0.005, r2_min=0.0).fits[0]
    return ts, group, offline


def _two_component_agreement(two_components, **kwargs):
    ts, group, offline = two_components
    estimates = replay(ts, _fitter(group, ts, existence_max_pulses=200, **kwargs), chunk_size=32)
    amplitude = np.array([e.amplitude for e in estimates]).reshape(200, 2)
    latency = np.array([e.latency for e in estimates]).reshape(200, 2)
    return (np.abs(amplitude / offline.amplitude - 1).max(),
            np.abs(latency - offline.latency).max() * SFREQ)


def test_req347_agreement_two_components(two_components):
    """The default mode (a shift and a scale per component) on a two-component template."""

    rel, lag = _two_component_agreement(two_components)
    assert rel <= 0.01, rel
    assert lag <= 1.0, lag


@pytest.mark.xfail(strict=True, reason=(
    "REQ-347 with a two-component template and the single shift and scale of spec §10.6 "
    "(mode='common'): fit_trials(mode='shift_scale') gives each component its own shift and "
    "scale. Measured on 200 pulses (N1 -40 uV at 30 ms, N2 +30 uV at 120 ms, jitter SD "
    "1.5 ms, amplitude CV 20 %): 1.5 % (common jitter) and 0.5 % (independent jitter) of the "
    "pulses within 1 % and one sample on both components; largest differences 92 % and "
    "8.8 samples (common), 117 % and 12.7 samples (independent)."))
def test_req347_agreement_two_components_common_mode(two_components):

    rel, lag = _two_component_agreement(two_components, mode="common")
    assert rel <= 0.01, rel
    assert lag <= 1.0, lag


def test_modes_agree_for_a_single_component(pulses):

    ts, group, _ = pulses
    runs = [[(e.amplitude, e.latency, e.scale, e.shift, e.r2)
             for e in replay(ts, _fitter(group, ts, existence_max_pulses=ts.n_trials,
                                            mode=mode), chunk_size=256)]
            for mode in ("shift_scale", "scale", "common")]
    assert runs[0] == runs[1] == runs[2]


def test_req347_chunk_sizes_give_identical_estimates(pulses):

    ts, group, _ = pulses
    runs = []
    for chunk in (1, 32, 1024):
        fitter = _fitter(group, ts, existence_max_pulses=ts.n_trials, random_state=0)
        runs.append([tuple(getattr(e, f) for f in FIELDS)
                     for e in replay(ts, fitter, chunk_size=chunk)])
    assert len(runs[0]) == ts.n_trials * 2
    for other in runs[1:]:
        # bit for bit (NaN compared as equal: the first pulse has no baseline yet)
        assert len(other) == len(runs[0])
        for a, b in zip(runs[0], other):
            assert all(x == y or (isinstance(x, float) and math.isnan(x) and math.isnan(y))
                       for x, y in zip(a, b))


def test_req347_running_existence_matches_offline_sign_flip(mixed):

    ts, template = mixed
    fitter = _fitter(template, ts, existence_max_pulses=ts.n_trials, existence_looks=None,
                     random_state=3)
    replay(ts, fitter, chunk_size=32)
    online = fitter.existence(final=True)
    # Offline: the same per-pulse values from the epochs, tested at once
    model = FixedShapeModel(template, sfreq=SFREQ, ch_names=ts.ch_names)
    res = model.fit_trialset(ts)
    values = res["projection"] - res["baseline_projection"]
    assert np.allclose(values, fitter.existence_stats.test.values, rtol=1e-12, atol=1e-9)
    offline = sign_flip_test(values, alpha=0.05, n_permutations=1000, random_state=3)
    assert np.array_equal(online["significant"], offline["significant"])
    assert np.allclose(online["p_value"], offline["p_value"])
    # the decisions are not trivial: responses found, silent channels not
    assert online["significant"][:4].all() and not online["significant"][4:].any()


def test_running_existence_detects_responses_with_looks_after_every_pulse(mixed):

    ts, template = mixed
    fitter = _fitter(template, ts, existence_max_pulses=ts.n_trials, random_state=0)
    estimates = replay(ts, fitter, chunk_size=64)
    result = fitter.existence()
    assert result["channels"] == ts.ch_names
    assert result["significant"][:4].all()
    assert result["alpha_spent"].max() <= 0.05 + 1e-12
    # per-pulse score: high on responsive channels, centred on silent ones
    z = {c: np.array([e.existence_z for e in estimates if e.channel == c])[2:]
         for c in ts.ch_names}
    assert min(np.median(z[c]) for c in ts.ch_names[:4]) > 3
    assert max(abs(np.median(z[c])) for c in ts.ch_names[4:]) < 1
    present = [e.present for e in estimates if e.channel == "c0"][2:]
    assert np.mean(present) > 0.8


###################################################################################################
## REQ-348: false-positive rate of the sequential test over 50 looks

def test_req348_sequential_fpr_on_null_values():

    rng = np.random.default_rng(1)
    hits = naive = total = 0
    for rep in range(20):
        test = SequentialSignFlip(200, alpha=0.05, max_n=50, n_permutations=500,
                                  random_state=rep)
        crossed_naive = np.zeros(200, bool)
        for _ in range(50):
            test.update(rng.standard_t(3, size=200))           # heavy tails, symmetric
            out = test.look()
            crossed_naive |= np.nan_to_num(out["p_value"], nan=1.0) <= 0.05
        hits += out["significant"].sum()
        naive += crossed_naive.sum()
        total += 200
    assert test.n_looks == 50
    assert hits / total <= 0.05 + 0.01, hits / total
    # the same looks without the spending rule inflate the error several times
    assert naive / total > 0.15


def test_req348_sequential_fpr_on_null_streams(mixed):

    _, template = mixed
    names = [f"c{i}" for i in range(64)]
    n = int(SFREQ)
    hits = total = 0
    for seed in range(16):
        rng = np.random.default_rng(100 + seed)
        fitter = StreamFitter(template, sfreq=SFREQ, ch_names=names, detector=no_pulses,
                              existence_max_pulses=50, random_state=seed)
        for k in range(50):
            fitter.push_pulse((k * n + n // 2) / SFREQ)
        for _ in range(50):
            fitter.push(rng.normal(0, 10, (n, 64)))
        fitter.push(rng.normal(0, 10, (n, 64)))
        result = fitter.existence()
        assert fitter.existence_stats.test.n_looks >= 50 and fitter.n_pulses == 50
        hits += result["significant"].sum()
        total += 64
    assert hits / total <= 0.05 + 0.01, hits / total


def test_alpha_spending_and_single_look():

    t = np.linspace(0.02, 1, 50)
    spent = alpha_spending(t, 0.05)
    assert np.all(np.diff(spent) >= 0) and spent[-1] == 0.05 and spent[0] < 1e-6
    # one look at full information is the fixed-sample test
    rng = np.random.default_rng(0)
    values = rng.normal(0.3, 1, (25, 40))
    test = SequentialSignFlip(40, max_n=25, n_permutations=999, random_state=2)
    for row in values:
        test.update(row)
    online = test.look()
    offline = sign_flip_test(values, n_permutations=999, random_state=2)
    assert np.array_equal(online["significant"], offline["significant"])
    assert 0 < online["significant"].sum() < 40


###################################################################################################
## REQ-324 / REQ-347: latency

def test_latency_stats():

    sys.path.insert(0, str(BENCHMARKS))
    from stream_latency import run

    summary, fitter = run(n_pulses=12, chunk_size=32)
    stats = fitter.latency_stats
    assert set(stats) == {"p50", "p95", "p99", "max"}
    assert 0 < stats["p50"] <= stats["p95"] <= stats["p99"] <= stats["max"]
    assert summary["pulses_estimated"] == 12 and summary["estimates"] == 12 * 64
    assert fitter.n_pulses == 12


@pytest.mark.skipif(os.environ.get("EVOPARAM_PERF_TESTS") != "1",
                    reason="set EVOPARAM_PERF_TESTS=1 to time the online mode")
@pytest.mark.parametrize("n_components", [1, 2])
def test_req347_latency_budget(n_components):
    """The default mode, with one component and with two (a shift and a scale each)."""

    sys.path.insert(0, str(BENCHMARKS))
    from stream_latency import run

    summary, fitter = run(n_pulses=1000, chunk_size=32, n_components=n_components)
    assert summary["pulses_estimated"] == 1000
    assert summary["n_components"] == n_components
    assert summary["push_p99_ms"] < 20.0, summary
    assert fitter.latency_stats["p99"] < 20.0


###################################################################################################
## Detector, triggers, statuses, errors

def _continuous(n_pulses=6, seed=0):
    rng = np.random.default_rng(seed)
    n = int(SFREQ)
    data = rng.normal(0, 10, (4, n_pulses * n + n))
    onsets = n + n * np.arange(n_pulses) + 300
    for onset in onsets:
        data[:, onset:onset + 3] += np.array([3000.0, -2000.0, 500.0])
    return data, onsets


def test_online_detector_matches_offline_threshold_and_ignores_chunks():

    data, onsets = _continuous()
    offline = detect_pulses(data, SFREQ, method="threshold", refine=False, saturation=False,
                            vote=False)
    assert np.array_equal(offline["onset_sample"].to_numpy(), onsets)
    for chunk in (1, 7, 1000):
        detector = OnlinePulseDetector(SFREQ, 4)
        found = []
        for first in range(0, data.shape[1], chunk):
            found += detector.update(data[:, first:first + chunk].T)
        assert found == list(onsets), chunk


def test_threshold_detector_in_the_fitter(mixed):

    _, template = mixed
    data, onsets = _continuous(n_pulses=4)
    fitter = StreamFitter(template, sfreq=SFREQ, ch_names=["c0", "c1", "c2", "c3"],
                          artifact_window=(-0.002, 0.008), detector_channel="c1")
    estimates = replay(data, fitter, chunk_size=50)
    times = sorted({e.pulse_time for e in estimates})
    assert np.allclose(times, onsets / SFREQ)


def test_push_pulse_and_timestamps(mixed):

    ts, template = mixed
    fitter = _fitter(template, ts)
    t0 = 1000.25
    data = ts.data[:3, :, :].transpose(0, 2, 1).reshape(-1, ts.n_channels)
    stamps = t0 + np.arange(len(data)) / SFREQ
    fitter.push(data[:500], stamps[:500])
    fitter.push_pulse(stamps[1024])                       # onset of the first epoch
    out = fitter.push(data[500:], stamps[500:])
    assert {e.pulse_time for e in out} == {stamps[1024]}
    assert all(isinstance(e, PulseEstimate) for e in out)
    with pytest.raises(dataclasses.FrozenInstanceError):
        out[0].amplitude = 0.0
    fitter.reset()
    assert fitter.n_samples == 0 and fitter.n_pulses == 0
    assert math.isnan(fitter.latency_stats["p99"])


def test_statuses(mixed):

    ts, template = mixed
    data = ts.data[:3].copy()
    data[1, 2, 1100] = np.nan                             # inside pulse 1's window, channel 2
    data[2, 5, 1200] = 5000.0                             # saturated sample, pulse 2
    stream = data.transpose(0, 2, 1).reshape(-1, ts.n_channels)
    fitter = _fitter(template, ts, saturation=4000.0)
    for k in range(3):
        fitter.push_pulse((k * ts.n_times + 1024) / SFREQ)
    out = fitter.push(stream)
    status = {(round(e.pulse_time * SFREQ) // ts.n_times, e.channel): e.status for e in out}
    assert status[(1, "c2")] == "artifact" and status[(2, "c5")] == "saturated"
    assert set(status.values()) <= set(STATUSES)
    assert sum(s == "ok" for s in status.values()) >= len(status) - 3
    assert np.isfinite([e.amplitude for e in out]).all()


def test_artifact_window_is_interpolated(mixed):

    ts, template = mixed
    clean = ts.data[:20].copy()
    spiked = clean.copy()
    spiked[:, :, 1024:1035] += 5000.0                     # 0 to 5 ms: inside the artifact window
    results = []
    for data in (clean, spiked):
        fitter = StreamFitter(template, sfreq=SFREQ, ch_names=ts.ch_names, detector=no_pulses,
                              window=(0.0, 0.5), artifact_window=(-0.002, 0.008))
        stream = data.transpose(0, 2, 1).reshape(-1, ts.n_channels)
        for k in range(len(data)):
            fitter.push_pulse((k * ts.n_times + 1024) / SFREQ)
        results.append(np.array([e.amplitude for e in fitter.push(stream)]))
    assert np.allclose(results[0], results[1], rtol=1e-9, atol=1e-9)


def test_errors(mixed):

    ts, template = mixed
    with pytest.raises(SettingsError):
        _fitter(template, ts, backend="cupy")
    with pytest.raises(SettingsError):
        _fitter(template, ts, detector="template")
    with pytest.raises(SettingsError):
        _fitter(template, ts, detector="threshold", detector_channel="nope")
    with pytest.raises(SettingsError):
        _fitter(template, ts, buffer_seconds=0.5)
    with pytest.raises(SettingsError):
        _fitter(template, ts, mode="free")
    fitter = _fitter(template, ts)
    with pytest.raises(DataError):
        fitter.push(np.zeros((10, 3)))
    with pytest.raises(DataError):
        fitter.push(np.zeros((10, 8)), timestamps=np.zeros(3))
    with pytest.raises(DataError):
        StreamFitter("not a fit", sfreq=SFREQ, ch_names=["a"])


###################################################################################################
## REQ-325 / REQ-015: intended use

def test_intended_use_statements(capsys):

    assert INTENDED_USE == "research only; not a medical device"
    assert "not a medical device" in evoparam.stream.__doc__
    assert "clinical decisions" in evoparam.stream.__doc__
    assert main(["info", "--json"]) == 0
    info = json.loads(capsys.readouterr().out.splitlines()[-1])["result"]
    assert "not a medical device" in info["intended_use"]
    fitter = StreamFitter(channel_fits(simulate(40, 1, seed=6)).fits[0], sfreq=SFREQ,
                          ch_names=["a"])
    assert fitter.settings["intended_use"] == INTENDED_USE


def test_simulation_helpers_are_consistent():

    # the epochs have a sample at t = 0 and the response where the template expects it
    assert TIMES[1024] == 0.0
    assert np.argmax(np.abs(N1[0] * gauss(TIMES, N1[1], N1[2]))) == 1024 + round(N1[1] * SFREQ)
