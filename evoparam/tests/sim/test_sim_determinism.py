"""REQ-250: every generator is deterministic for a given seed."""

import copy
import hashlib
import json
import pathlib

import numpy as np
import pytest

from evoparam import sim
from evoparam.errors import SettingsError
from evoparam.tests.sim._helpers import SMALL, real_baselines

REFERENCE = pathlib.Path(__file__).with_name("reference_sim.json")


def _same(first, second):
    """Byte-identical data, masks and truth."""

    ts_a, truth_a = first
    ts_b, truth_b = second
    assert ts_a.data.tobytes() == ts_b.data.tobytes()
    assert (ts_a.masks is None) == (ts_b.masks is None)
    if ts_a.masks is not None:
        assert np.array_equal(ts_a.masks, ts_b.masks)
    assert truth_a.signal.tobytes() == truth_b.signal.tobytes()
    assert truth_a.components == truth_b.components
    for name in ("shift", "scale", "present"):
        assert getattr(truth_a, name).tobytes() == getattr(truth_b, name).tobytes()
    assert truth_a.noise == truth_b.noise
    assert truth_a.artifact == truth_b.artifact
    assert truth_a.seed == truth_b.seed
    if ts_a.metadata is not None:
        assert ts_a.metadata.equals(ts_b.metadata)


@pytest.mark.parametrize("name", sorted(sim.GENERATORS))
def test_same_int_seed_same_bytes(name):
    gen = sim.GENERATORS[name]
    first = gen(seed=11, **SMALL[name])
    second = gen(seed=11, **SMALL[name])
    _same(first, second)
    assert first[1].seed == 11
    assert first[0].provenance.random_state == 11
    assert first[0].provenance.steps[-1]["function"] == f"evoparam.sim.{gen.__name__}"


@pytest.mark.parametrize("name", sorted(sim.GENERATORS))
def test_different_seeds_differ(name):
    gen = sim.GENERATORS[name]
    ts_a, _ = gen(seed=1, **SMALL[name])
    ts_b, _ = gen(seed=2, **SMALL[name])
    assert not np.array_equal(ts_a.data, ts_b.data)


@pytest.mark.parametrize("name", sorted(sim.GENERATORS))
def test_recorded_seed_regenerates(name):
    """seed=None and a Generator record the seed actually used, which regenerates the data."""

    gen = sim.GENERATORS[name]
    for seed in (None, np.random.default_rng(5)):
        first = gen(seed=seed, **SMALL[name])
        again = gen(seed=first[1].seed, **SMALL[name])
        _same(first, again)
    from_generator = gen(seed=np.random.default_rng(5), **SMALL[name])
    _same(from_generator, gen(seed=np.random.default_rng(5), **SMALL[name]))


def test_noise_model_does_not_change_components():
    """Independent random streams: the components drawn do not depend on the noise model."""

    _, white = sim.simulate_ccep(seed=4, noise="white", **SMALL["ccep"])
    _, pink = sim.simulate_ccep(seed=4, noise="pink+line", **SMALL["ccep"])
    params = [c["params"] for c in white.components]
    assert params == [c["params"] for c in pink.components]
    assert white.settings["responders"] == pink.settings["responders"]


def test_make_noise_and_inject_are_deterministic():
    for kind in ("white", "ar1", "pink", "line", "pink+line"):
        a = sim.make_noise(300, 250., kind, size=(2, 3), seed=9)
        b = sim.make_noise(300, 250., kind, size=(2, 3), seed=9)
        assert a.tobytes() == b.tobytes()
    real = real_baselines()
    a, truth_a = sim.inject(real, ["N1"], snr=3., seed=3)
    b, truth_b = sim.inject(real, ["N1"], snr=3., seed=3)
    assert a.data.tobytes() == b.data.tobytes()
    assert truth_a.components == truth_b.components
    noise_a = sim.make_noise(400, 500., "real", size=(3, 2), source=real, seed=1)
    noise_b = sim.make_noise(400, 500., "real", size=(3, 2), source=real, seed=1)
    assert noise_a.tobytes() == noise_b.tobytes()


def test_truth_compares_equal_to_its_copy():
    _, truth = sim.simulate_erp(seed=0, **SMALL["erp"])
    assert truth == copy.deepcopy(truth)
    assert truth == truth.copy()


@pytest.mark.parametrize("name", sorted(sim.GENERATORS))
def test_reference_fingerprints(name):
    """Values committed from a reference run (seed 2024): the same on every platform of the
    CI within 1e-9 relative (REQ-295); byte-identical when numpy and scipy are the versions
    that produced them."""

    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    expected = reference["generators"][name]
    ts, truth = sim.GENERATORS[name](seed=2024, **SMALL[name])
    assert list(ts.data.shape) == expected["shape"]
    assert truth.n_components == expected["n_components"]
    flat = ts.data.ravel()
    scale = np.max(np.abs(ts.data))
    np.testing.assert_allclose(flat[expected["index"]], expected["values"], rtol=1e-9,
                               atol=1e-9 * scale)
    np.testing.assert_allclose(np.sum(np.abs(ts.data)), expected["sum_abs"], rtol=1e-9)
    np.testing.assert_allclose([c["features"]["latency"] for c in truth.components][:8],
                               expected["latencies"], rtol=1e-9, atol=1e-12)
    made_with = reference["made_with"]
    import scipy
    if (np.__version__, scipy.__version__) == (made_with["numpy"], made_with["scipy"]):
        assert hashlib.sha256(ts.data.tobytes()).hexdigest() == expected["sha256"]


def test_bad_seed():
    for seed in (-1, 1.5, "1", True):
        with pytest.raises(SettingsError):
            sim.simulate_erp(seed=seed, n_trials=2)
