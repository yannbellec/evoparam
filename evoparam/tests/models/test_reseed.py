"""Reseeding the greedy path: no cancelling pair where the data have two components (noise
from the baseline: the cases were found with it)."""

import warnings

import numpy as np
import pytest

from evoparam import fit, sim
from evoparam.models import evaluate_fit


# contacts where the greedy path ended in a cancelling pair (seed 0)
REPAIRED = {"cochlear": (1, 3), "spinal": (2,)}


@pytest.mark.parametrize("kind", ["cochlear", "spinal"])
def test_ecap_components_are_found_not_a_cancelling_pair(kind):

    preset = "ecap_ci" if kind == "cochlear" else "ecap"
    ts, truth = sim.simulate_ecap(40, kind=kind, paradigm=None, artifact=False, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = fit(ts, preset, random_state=0, existence=False, noise_source="baseline")
        plain = fit(ts, preset, random_state=0, existence=False, reseed=False,
                    noise_source="baseline")
    table = truth.to_dataframe()
    signal = np.asarray(truth.signal).mean(axis=0)
    average = ts.average()
    better = 0
    for ch in REPAIRED[kind]:
        fitted = group.fits[ch]
        true = table[table["channel"] == ts.ch_names[ch]].sort_values("latency")
        comps = sorted(fitted.components, key=lambda c: c.latency)
        assert len(comps) == len(true)
        for comp, (_, row) in zip(comps, true.iterrows()):
            assert np.sign(comp.amplitude) == np.sign(row["amplitude"])
            assert abs(comp.latency - row["latency"]) < 0.5 * row["fwhm"]
        w = (average.times >= fitted.window[0]) & (average.times <= fitted.window[1])
        rss = lambda f: float(np.sum((average.data[ch][w] - evaluate_fit(f, average.times)[w])
                                     ** 2))
        noise = float(np.sum((average.data[ch][w] - signal[ch][w]) ** 2))
        assert rss(fitted) < 2 * noise                  # as good as the true curve
        better += rss(fitted) < rss(plain.fits[ch]) - 1e-9
    assert better == len(REPAIRED[kind])                # each was repaired by reseeding


def test_a_cancelling_pair_is_flagged_unresolvable():
    """Without reseeding the greedy path leaves a cancelling pair on contact E01 of the
    simulated cochlear ECAP: both components are flagged, in the table too."""

    ts, _ = sim.simulate_ecap(40, kind="cochlear", paradigm=None, artifact=False, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = fit(ts, "ecap_ci", random_state=0, existence=False, reseed=False,
                    noise_source="baseline")
        fixed = fit(ts, "ecap_ci", random_state=0, existence=False,
                    noise_source="baseline")
    pair = plain.fits[1]
    assert [c.resolvable for c in pair.components] == [False, False]
    assert any(w.startswith("unresolvable:") for w in pair.warnings)
    assert all(c.resolvable for c in fixed.fits[1].components)
    table = plain.to_dataframe()
    assert str(table["resolvable"].dtype) == "boolean"
    assert not table[table["channel"] == plain.index["channel"].iloc[1]]["resolvable"].any()
