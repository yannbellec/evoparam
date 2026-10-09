"""Trial fits on OpenNeuro ds004080 (REQ-156) and the train-position effect (REQ-158).

The ds004080 tests run only with EVOPARAM_NETWORK_TESTS=1. REQ-156 uses every trial of one
stimulation pair (10 per site, both polarities) of the six subjects of decision D14 (the first
five subjects of validation/ds004080_subjects.txt other than UMCU01, plus UMCU01), read and
cached as for REQ-113 (``_pair_trials`` of test_engine_ccep.py), with the ccep preset pipeline
of REQ-113 (template artifact removal, adjusted common average, baseline) and
``evoparam.fit`` (channel fits with their existence test), and ``fit_trials`` with its defaults
(trial model "anchored": the trial's N1 moves, the channel's other components are fitted in
amplitude at their channel latency).
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import pytest

NETWORK = os.environ.get("EVOPARAM_NETWORK_TESTS") == "1"


def network(test):
    test = pytest.mark.skipif(not NETWORK, reason="set EVOPARAM_NETWORK_TESTS=1 to read "
                              "OpenNeuro")(test)
    return pytest.mark.network(test)


# (subject, run, stimulation pair): decision D14; the pair is the first of the run with at
# least 10 trials (the rule of validation/ds004080_subjects.txt), as in test_engine_ccep.py
PAIRS_156 = [("ccepAgeUMCU01", "061233", "PT01-PT02"),
             ("ccepAgeUMCU02", "021804", "FC01-FC02"),
             ("ccepAgeUMCU03", "031411", "C01-C02"),
             ("ccepAgeUMCU04", "031148", "TPa1-TPa2"),
             ("ccepAgeUMCU05", "031133", "P01-P02"),
             ("ccepAgeUMCU06", "030854", "C01-C02")]


def _prepared(subject, run, site):
    """The pair's trials through the ccep preset pipeline, and evoparam.fit of them."""

    import evoparam
    from evoparam.artifact import remove_artifact
    from evoparam.prep import baseline_correct, rereference
    from evoparam.tests.models.test_engine_ccep import _pair_trials

    ts = _pair_trials(subject, run, site).select(tmin=-1.0, tmax=2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = remove_artifact(ts, method="template", window=(-0.002, 0.008),
                             baseline=(-0.5, -0.02)).cleaned
        ts = rereference(ts, "car_adaptive", random_state=0)
        ts = baseline_correct(ts, (-0.5, -0.02), check=False)
        group = evoparam.fit(ts, "ccep", random_state=0)
    return ts, group


def _n1_channels(group):
    """Channels whose fit has an N1, and the subset with a significant existence test."""

    names = list(group.index["channel"])
    n1 = [n for n, f in zip(names, group.fits)
          if f.status == "ok" and any(c.label == "N1" for c in f.components)]
    significant = {n for n, f in zip(names, group.fits)
                   if f.existence is not None and bool(np.all(f.existence.significant))}
    return n1, significant


@pytest.fixture(scope="module")
def results():
    from evoparam.trials import fit_trials

    out = []
    for subject, run, site in PAIRS_156:
        ts, group = _prepared(subject, run, site)
        assert ts.n_trials == 10
        n1, significant = _n1_channels(group)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_trials(ts, group, components=["N1"])           # warm-up (imports, caches)
            start = time.perf_counter()
            tf = fit_trials(ts, group, components=["N1"])
            duration = time.perf_counter() - start
        assert sorted(f.channel for f in tf.fits) == sorted(n1)
        out.append(dict(subject=subject, duration=duration, fits=tf.fits, n1=n1,
                        significant=significant))
    return out


@network
def test_req156_trusted_fraction(results):

    fractions = [f.trusted.mean() for r in results for f in r["fits"]
                 if f.channel in r["significant"]]
    assert len({r["subject"] for r in results if r["fits"]}) >= 5
    assert np.median(fractions) >= 0.8


@network
def test_req156_speed(results):
    """fit_trials on all N1 channels of each subject (10 trials per site) in <= 2 s."""

    assert len({r["subject"] for r in results if r["fits"]}) >= 5
    assert max(r["duration"] for r in results) <= 2.0


###################################################################################################
## REQ-158: amplitude slope across a train

def _trend_table(ts, group):
    from evoparam.trials import fit_trials

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tf = fit_trials(ts, group, components=["N1"], covariates=("trend",))
    return pd.DataFrame(tf.settings["covariate_tables"]["trend"])


def _sign_agreement(table, alpha=0.05):
    """Among the sites (channel x stimulation pair) whose slope is significant, the fraction
    whose amplitude declines across the train."""

    significant = table[table["p_value"] <= alpha]
    return len(significant), float(np.mean(significant["slope"] < 0)) if len(significant) \
        else np.nan


def test_req158_direction_on_simulated_1hz_trains():
    """The index_in_train effect (relative amplitude slope per pulse, t test) recovers a
    declining N1 across simulated 1 Hz trains of 30 single pulses (gain 1 - 0.01 x index, the
    adaptation of simulate_ccep) with sign agreement >= 80 % of the significant sites."""

    from evoparam.models import fit as fit_models
    from evoparam.prep import baseline_correct
    from evoparam.sim import simulate_ccep

    counts, agree = 0, []
    for seed in (10, 11, 12):
        ts, _ = simulate_ccep(30, n_channels=20, artifact=False, seed=seed, p_response=0.5,
                              train_structure={"n_pulses": 30, "rate_hz": 1.0,
                                               "adaptation": -0.01}, tmin=-0.5, tmax=0.6)
        assert np.all(ts.metadata["stim_frequency_hz"] == 1.0)
        ts = baseline_correct(ts, (-0.5, -0.02), check=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            group = fit_models(ts, "ccep", baseline=(-0.5, -0.02))
        table = _trend_table(ts, group)
        n, fraction = _sign_agreement(table)
        counts += n
        agree += [fraction] * n
    assert counts >= 10
    assert np.mean(agree) >= 0.8


@pytest.mark.xfail(strict=True, run=False, reason=(
    "REQ-158 as written needs 'the reference dataset used in spec 7' with repeated single "
    "pulses at 1 Hz and a reported declining N1. None exists: the 7.3 datasets with CCEPs "
    "(ds004080, ds003708, ds004977) use about 0.2 Hz; Localize-MI (1 Hz) keeps epochs from "
    "-250 to +10 ms only, before any N1 (Mikulan et al. 2020, doi:10.1038/s41597-020-0467-x);"
    " the 1 Hz "
    "study cited for train effects (Cornblath et al. 2023, Epilepsia, 30 pulses at 1 Hz) "
    "reports infrequent trends of both signs (0.84 % of CCEPs per patient), not a decline. "
    "Owner decision needed; see CHANGELOG (trials, F17)."))
def test_req158_as_written():

    raise AssertionError("no 1 Hz reference dataset with an N1 in spec 7")
