"""REQ-013 (determinism): identical inputs, settings and ``random_state`` give identical
outputs (REQ-119, REQ-250, §9.4): the same numbers, bit for bit, and the same bytes for the
saved files under a fixed clock."""

import hashlib
import os
import warnings

import numpy as np
import pytest

import evoparam
import evoparam.stats
from evoparam import sim
from evoparam.data import save_object

pytest.importorskip("pandas")


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1767225600")


def _trials():
    trials, _ = sim.simulate_ccep(n_trials=10, n_channels=3, p_response=1.0, seed=4)
    return trials


CHAIN = {
    "simulate": lambda ts: sim.simulate_ccep(n_trials=10, n_channels=3, p_response=1.0,
                                             seed=4)[0],
    "fit": lambda ts: evoparam.fit(ts, preset="ccep", random_state=0),
    "test_response": lambda ts: evoparam.test_response(ts, window=(0.009, 0.5),
                                                       baseline=(-0.5, -0.05), by=(),
                                                       random_state=0),
    "remove_artifact": lambda ts: evoparam.artifact.remove_artifact(
        ts, method="template", window=(-0.002, 0.006), random_state=0),
    "fit_trials": lambda ts: evoparam.trials.fit_trials(
        ts, evoparam.fit(ts, preset="ccep", random_state=0), random_state=0).to_dataframe(),
    "bootstrap": lambda ts: evoparam.stats.bootstrap(ts, "peak_latency", n_boot=50,
                                                     window=(0.009, 0.1), random_state=0),
}


def _digest(obj, folder):
    if hasattr(obj, "to_csv"):                     # a table
        return hashlib.sha256(obj.to_csv(index=False).encode()).hexdigest()
    path = save_object(obj, os.path.join(folder, "obj.evoparam"))
    digest = hashlib.sha256()
    for root, _, files in sorted(os.walk(path)):
        for name in sorted(files):
            with open(os.path.join(root, name), "rb") as fobj:
                digest.update(name.encode() + fobj.read())
    return digest.hexdigest()


@pytest.mark.parametrize("step", sorted(CHAIN))
def test_req013_identical_outputs(step, tmp_path):
    ts = _trials()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = CHAIN[step](ts)
        second = CHAIN[step](_trials())
    os.makedirs(tmp_path / "a")
    os.makedirs(tmp_path / "b")
    assert _digest(first, str(tmp_path / "a")) == _digest(second, str(tmp_path / "b"))


def test_req013_parallel_fit_equals_serial():
    ts = _trials()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        serial = evoparam.fit(ts, preset="ccep", random_state=0, n_jobs=1).to_dataframe()
        parallel = evoparam.fit(ts, preset="ccep", random_state=0, n_jobs=2).to_dataframe()
    assert serial.equals(parallel)


def test_req013_different_seeds_differ_where_randomness_is_used():
    ts = _trials()
    kwargs = dict(window=(0.009, 0.5), baseline=(-0.5, -0.05), by=(), n_permutations=200)
    a = evoparam.test_response(ts, random_state=0, **kwargs)
    b = evoparam.test_response(ts, random_state=1, **kwargs)
    assert not np.array_equal(a.settings["duration_ci"], b.settings["duration_ci"]) or \
        not np.array_equal(a.p_value_raw, b.p_value_raw)
