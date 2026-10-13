# SPDX-License-Identifier: Apache-2.0
"""Agreement with the field on OpenNeuro ds004080 (REQ-135); runs only with
EVOPARAM_NETWORK_TESTS=1.

REQ-135 as written runs on the subjects chosen by the rule of spec §7.3 (the REQ-371 list,
validation/ds004080_subjects.txt, owner decision D14): every trial of each run's first
stimulation pair of at least 10 trials, cached as for REQ-113 (``_pair_trials`` of
test_engine_ccep.py). The method and the registry rules see the same trials (template
artifact removal, baseline -1 to -0.1 s, as ``compare_detectors`` gives them one TrialSet).
The regression floor and the development measures use the 40 s segments cached by the
events validation (``RUNS``: the example subject and the five development subjects).
"""

import os
import re
import warnings
from typing import Any

import numpy as np
import pytest

from evoparam.artifact import remove_artifact
from evoparam.data import TrialSet
from evoparam.existence import test_response as run_test
from evoparam.io import epoch_data
from evoparam.prep import baseline_correct

pd = pytest.importorskip("pandas")

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("EVOPARAM_NETWORK_TESTS") != "1",
        reason="set EVOPARAM_NETWORK_TESTS=1 to read OpenNeuro",
    ),
]

RUNS = [
    "sub-ccepAgeUMCU01/ses-1/ieeg/sub-ccepAgeUMCU01_ses-1_task-SPESclin_run-021448",
    "sub-ccepAgeUMCU07/ses-1/ieeg/sub-ccepAgeUMCU07_ses-1_task-SPESclin_run-021752",
    "sub-ccepAgeUMCU19/ses-1/ieeg/sub-ccepAgeUMCU19_ses-1_task-SPESclin_run-040955",
    "sub-ccepAgeUMCU28/ses-1/ieeg/sub-ccepAgeUMCU28_ses-1_task-SPESclin_run-021305",
    "sub-ccepAgeUMCU40/ses-1/ieeg/sub-ccepAgeUMCU40_ses-1_task-SPESclin_run-031553",
    "sub-ccepAgeUMCU49/ses-1/ieeg/sub-ccepAgeUMCU49_ses-1_task-SPESclin_run-031049",
]
BASELINE = (-1.0, -0.1)


def _trials(run: Any) -> Any:
    from evoparam.tests.events.test_ds004080_events import _segment

    data, sfreq, names, t0, onsets, sites = _segment(run)
    site = pd.Series(sites).mode()[0]
    onsets = onsets[np.asarray(sites) == site]
    keep = [
        i
        for i, n in enumerate(names)
        if re.match(r"^[A-Za-z]+\d+$", n)
        and not n.upper().startswith(("MKR", "ECG"))
        and n not in site.split("-")
    ]
    times, epochs, _ = epoch_data(data[keep], sfreq, onsets, -1.0, 1.0, t0=t0)
    ts = TrialSet(
        data=epochs,
        times=times,
        sfreq=sfreq,
        ch_names=[names[i] for i in keep],
        metadata=pd.DataFrame({"stim_site": [site] * len(epochs)}),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = remove_artifact(
            ts, method="template", window=(-0.002, 0.008), baseline=BASELINE
        ).cleaned
    return baseline_correct(ts, BASELINE, check=False)


def _kappa_with(rule: Any, **extra: Any) -> Any:
    assert len({r.split("/")[0] for r in RUNS}) >= 5
    a_all, b_all = [], []
    for run in RUNS:
        ts = _trials(run)
        a = run_test(
            ts,
            window=(0.009, 0.1),
            baseline=BASELINE,
            polarity="negative",
            n_permutations=1000,
            n_boot=0,
            by=(),
            random_state=0,
        ).significant
        b = run_test(
            ts,
            method="threshold",
            rule=rule,
            rule_overrides={"baseline": BASELINE, **extra},
            by=(),
        ).significant
        a_all += list(a)
        b_all += list(b)
    a, b = np.array(a_all), np.array(b_all)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe)


def _prepared(ts: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = remove_artifact(
            ts, method="template", window=(-0.002, 0.008), baseline=BASELINE
        ).cleaned
    return baseline_correct(ts, BASELINE, check=False)


@pytest.fixture(scope="module")
def req371_decisions() -> Any:
    """Per REQ-371 run: the default method's decision over 9-100 ms (negative: N1
    presence), with its default correction across channels (fdr_bh) and with none; the
    registry row (its unspecified baseline given as -1 to -0.1 s, the ER-detect default
    the registry names) and the published code (its own baseline, -2 to -0.1 s)."""

    from evoparam.tests.models.test_engine_ccep import _pair_trials, req371_pairs

    out = []
    for subject, run, site in req371_pairs():
        ts = _prepared(_pair_trials(subject, run, site))
        row = {"subject": subject}
        for correction in ("fdr_bh", "none"):
            row[correction] = run_test(
                ts,
                window=(0.009, 0.1),
                baseline=BASELINE,
                polarity="negative",
                correction=correction,
                n_permutations=1000,
                n_boot=0,
                by=(),
                random_state=0,
            ).significant
        row["vanblooijs2023"] = run_test(
            ts,
            method="threshold",
            rule="vanblooijs2023",
            rule_overrides={"baseline": BASELINE},
            by=(),
        ).significant
        row["vanblooijs2023_code"] = run_test(
            ts, method="threshold", rule="vanblooijs2023_code", by=()
        ).significant
        out.append(row)
    return out


def _pooled(rows: Any, method: Any, rule: Any) -> Any:
    from evoparam.tests.models.test_engine_ccep import _kappa

    assert len({r["subject"] for r in rows}) >= 5
    a = np.concatenate([r[method] for r in rows]).astype(bool)
    b = np.concatenate([r[rule] for r in rows]).astype(bool)
    return _kappa(a, b)


# Measured on the REQ-371 runs (six subjects, 515 channels, 10 trials each): kappa 0.586
# with the registry row, 0.496 with the published code (per run 0.27 to 0.72 and 0.13 to
# 0.75). On the development segments (below, RUNS) the same test gave 0.687 and 0.399.
@pytest.mark.xfail(
    strict=False,
    reason="REQ-135 is judged on the registry row (D8 revised, DECISIONS §5.8); the "
    "code's reading is reported. REQ-135 is not met under either reading of the "
    "published rule: on the REQ-371 runs kappa 0.586 with the registry row, 0.496 with the "
    "published code; under both readings at once no decision can reach 0.7 "
    "(test_req135_cannot_hold_under_both_readings: at most 0.592). With the per-channel "
    "decision of the rule (correction='none') the registry row reaches 0.807 and the code "
    "falls to 0.321 (CHANGELOG, 'N1 case: the two readings')",
)
@pytest.mark.parametrize("rule", ["vanblooijs2023", "vanblooijs2023_code"])
def test_req135_permutation_agrees_with_vanblooijs2023(req371_decisions: Any, rule: Any) -> None:
    """REQ-135: on ds004080 (>= 5 subjects: the REQ-371 list), the default method agrees
    with vanblooijs2023 with kappa >= 0.7 on N1 presence over 9-100 ms, under the two
    readings of the rule."""

    assert _pooled(req371_decisions, "fdr_bh", rule) >= 0.7


def test_req135_cannot_hold_under_both_readings(req371_decisions: Any) -> None:
    """Not a REQ (owner decision D8, measured): for every possible decision of the 515
    channels, the smaller of its kappas with the two readings is at most 0.592 (exact
    search, :func:`joint_kappa_bound`): REQ-135 cannot hold under both readings at once on
    the REQ-371 runs, whatever the method. The two readings agree with each other at
    kappa 0.31 there (125 and 28 N1s)."""

    from evoparam.tests.models.test_engine_ccep import joint_kappa_bound

    paper = np.concatenate([r["vanblooijs2023"] for r in req371_decisions])
    code = np.concatenate([r["vanblooijs2023_code"] for r in req371_decisions])
    assert joint_kappa_bound(paper, code) < 0.7


def test_req135_per_channel_decision_registry_row_regression(req371_decisions: Any) -> None:
    """Not the REQ as written (the ccep preset keeps the spec's FDR across channels, §6):
    the default method with the rule's own per-channel decision (correction='none', each
    channel tested at alpha over 9-100 ms, as the rule decides each channel alone) agrees
    with the registry row with kappa 0.807 on the REQ-371 runs and 0.808 on the five
    development runs (a principled choice, no value tuned); with the code it falls to
    0.321."""

    assert _pooled(req371_decisions, "none", "vanblooijs2023") >= 0.7


def test_req135_regression_with_the_code_floor() -> None:
    """Not the REQ (an intermediate reading, kept as a regression floor, on the development
    segments): the registry row with the published code's 50 uV SD floor (0.763)."""

    assert _kappa_with("vanblooijs2023", sd_floor_uv=50.0) >= 0.7
