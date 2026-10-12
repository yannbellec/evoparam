# SPDX-License-Identifier: Apache-2.0
"""Published zone on OpenNeuro ds004080 (REQ-216); runs only with EVOPARAM_NETWORK_TESTS=1."""

import os
import warnings

import numpy as np
import pytest

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("EVOPARAM_NETWORK_TESTS") != "1",
        reason="set EVOPARAM_NETWORK_TESTS=1 to read OpenNeuro",
    ),
]

ZONE = "published:brinyark2026_delayed_high_gamma"


def _relations():
    from scipy.stats import spearmanr

    from evoparam.tests.existence.test_existence_ccep import RUNS, _trials
    from evoparam.tf import tf_fit

    out = {}
    for run in RUNS:
        ts = _trials(run)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tf = tf_fit(ts, zones=ZONE, window=(0.0, 0.6), by=(), random_state=0)
        metrics = tf.zone_metrics.set_index("channel").reindex(ts.ch_names)
        # N1 amplitude: most negative deflection of the trial average, 9 to 100 ms
        early = (ts.times >= 0.009) & (ts.times <= 0.1)
        n1 = -ts.data.mean(axis=0)[:, early].min(axis=1)
        for column in ("fraction_increase", "value"):
            values = metrics[column].to_numpy()
            ok = np.isfinite(values)
            out.setdefault(column, []).append(spearmanr(n1[ok], values[ok])[0])
    return out


@pytest.fixture(scope="module")
def relations():
    return _relations()


def test_req216_published_zone_follows_the_n1(relations):
    """The published summary statistic (the share of significant increases in the zone)
    rises with the N1 amplitude across recording sites in at least half of the subjects;
    the sign is positive."""

    rho = np.array(relations["fraction_increase"])
    assert len(rho) >= 5
    assert np.mean(np.abs(rho) >= 0.3) >= 0.5
    assert np.median(rho) > 0


def test_mean_zone_power_relation_is_weaker(relations):
    """Documented: the mean normalised power over the whole zone relates to the N1 in fewer
    subjects (seven or eight trials per site: the mean dB over a large zone is noisy)."""

    rho = np.array(relations["value"])
    assert np.mean(np.abs(rho) >= 0.3) <= np.mean(np.abs(relations["fraction_increase"]) >= 0.3)
