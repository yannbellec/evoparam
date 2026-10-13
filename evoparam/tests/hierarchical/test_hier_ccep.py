# SPDX-License-Identifier: Apache-2.0
"""Age and N1 latency on OpenNeuro ds004080 (REQ-178); runs only with
EVOPARAM_NETWORK_TESTS=1."""

import dataclasses
import io
import os
from typing import Any

import pytest

# Fewest subjects of the real CCEP dataset left in the subject table.
MIN_SUBJECTS = 5

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("EVOPARAM_NETWORK_TESTS") != "1",
        reason="set EVOPARAM_NETWORK_TESTS=1 to read OpenNeuro",
    ),
]

PARTICIPANTS = "https://s3.amazonaws.com/openneuro.org/ds004080/participants.tsv"


def _ages() -> Any:
    import pandas as pd

    from evoparam.io._fetch import fetch_bytes

    table = pd.read_csv(io.BytesIO(fetch_bytes(PARTICIPANTS)), sep="\t")
    return dict(zip(table["participant_id"], table["age"], strict=False))


def _subjects() -> Any:
    """One run per subject of the cached validation runs (the first one), one site."""

    from evoparam.tests.events.test_ds004080_events import RUNS
    from evoparam.tests.existence.test_existence_ccep import _trials

    ages = _ages()
    out, seen = [], set()
    for run in RUNS:
        subject = run.split("/")[0]
        if subject in seen:
            continue
        seen.add(subject)
        ts = _trials(run)
        meta = ts.metadata.assign(subject=subject, age=float(ages[subject]))
        out.append(dataclasses.replace(ts, metadata=meta, modality="spes"))
    return out


@pytest.fixture(scope="module")
def age_fit() -> Any:
    from evoparam.hierarchical import fit_hierarchical

    return fit_hierarchical(_subjects(), components=["N1"], formula="latency ~ age", n_jobs=-1)


def test_req178_n1_latency_decreases_with_age(age_fit: Any) -> None:

    effects = age_fit.effects
    row = effects[effects["term"] == "age"].iloc[0]
    assert age_fit.level_tables["subject"]["subject"].nunique() >= MIN_SUBJECTS
    # the sign reported by the dataset's authors: shorter N1 latencies with age
    assert row["estimate"] < 0
