"""Tests for Provenance (spec §4.3.3; REQ-294 steps, REQ-335 environment keys, REQ-337 data hash)."""

import json

import numpy as np

from evoparam import __version__
from evoparam.data import Provenance, capture_environment, compute_data_hash
from evoparam.tests.data.factories import make_trialset

ENVIRONMENT_KEYS = {"python", "platform", "cpu", "numpy", "scipy", "optional", "backend",
                    "jax_enable_x64", "n_threads", "git_commit", "lock_sha256", "executable"}


def test_defaults_and_json_round_trip():
    prov = Provenance(random_state=7)
    assert prov.package_version == __version__
    assert prov.created.endswith("+00:00")
    again = Provenance.from_json(prov.to_json())
    assert again == prov
    json.loads(prov.to_json(), parse_constant=lambda name: (_ for _ in ()).throw(ValueError(name)))


def test_with_step_copies_and_converts():
    prov = Provenance()
    new = prov.with_step("evoparam.x.f", {"window": (0.0, 0.1), "level": float("nan")},
                         duration_s=0.5, preset={"name": "ccep", "version": "1"})
    assert prov.steps == []
    step = new.steps[-1]
    assert step["function"] == "evoparam.x.f" and step["version"] == __version__
    assert step["settings"] == {"window": [0.0, 0.1], "level": None}
    assert step["preset"] == {"name": "ccep", "version": "1"}


def test_environment_keys():
    env = capture_environment()
    assert set(env) == ENVIRONMENT_KEYS
    assert "pandas" in env["optional"]
    env["python"] = "changed"
    assert capture_environment()["python"] != "changed"


def test_data_hash_changes_with_one_sample():
    trials = make_trialset(with_tables=False)
    reference = compute_data_hash(trials.data, trials.times, trials.ch_names)
    assert reference == trials.provenance.data_hash
    data = trials.data.copy()
    data[2, 1, 50] += 1e-9
    assert compute_data_hash(data, trials.times, trials.ch_names) != reference
    assert compute_data_hash(trials.data, trials.times, ["X", "CH01", "CH02"]) != reference
    assert compute_data_hash(np.array(trials.data), trials.times, trials.ch_names) == reference


def test_every_method_appends_one_step():
    trials = make_trialset()
    n_steps = len(trials.provenance.steps)
    for derived in (trials.select(trials=[0, 1]), trials.crop(0.0, 0.1),
                    trials.convert_units("mV"), trials.apply(np.negative), trials.average()):
        assert len(derived.provenance.steps) == n_steps + 1
        json.dumps(derived.provenance.to_dict(), allow_nan=False)


def test_fixed_clock_makes_records_reproducible(monkeypatch):
    """REQ-345: with SOURCE_DATE_EPOCH the creation time is that instant and durations are
    recorded as 0, so two runs write the same bytes; without it the real clock is used."""

    import time

    from evoparam.data.provenance import elapsed, fixed_clock

    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1767225600")
    assert fixed_clock() == 1767225600
    first = Provenance(environment={}).with_step("f", {"a": 1}, duration_s=3.2)
    second = Provenance(environment={}).with_step("f", {"a": 1}, duration_s=0.7)
    assert first.created == "2026-01-01T00:00:00.000+00:00"
    assert first.steps[-1]["duration_s"] == 0.0
    assert first.to_json() == second.to_json()
    assert elapsed(time.perf_counter() - 5.0) == 0.0
    monkeypatch.delenv("SOURCE_DATE_EPOCH")
    assert fixed_clock() is None
    assert Provenance().with_step("f", duration_s=3.2).steps[-1]["duration_s"] == 3.2
    assert elapsed(time.perf_counter() - 5.0) >= 5.0
