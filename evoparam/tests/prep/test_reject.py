# SPDX-License-Identifier: Apache-2.0
"""Trial rejection (REQ-083)."""

from dataclasses import replace

import numpy as np
import pytest

from evoparam.artifact import apply_mask
from evoparam.prep import reject_trials

from .helpers import ccep, epochs

pd = pytest.importorskip("pandas")

# Response window (s), trials simulated, and the rejection rates required.
RESPONSE_START = 0.01
RESPONSE_END = 0.5
N_TRIALS = 200
MIN_REJECTED = 0.95
MAX_FALSE_REJECTED = 0.02

###################################################################################################
###################################################################################################


def test_req083_flags_corrupted_trials_without_deleting() -> None:

    rng = np.random.default_rng(3)
    ts = epochs(n_trials=200, seed=4)
    data = ts.data + ccep(ts.times)[None, None]
    corrupted = rng.choice(200, 30, replace=False)
    big, flat, masked = corrupted[:10], corrupted[10:20], corrupted[20:]
    data[big] *= 10  # 10x the baseline amplitude
    data[flat] = 0.0  # flat
    mask = np.zeros(data.shape, dtype=bool)
    resp = np.flatnonzero((ts.times >= RESPONSE_START) & (ts.times <= RESPONSE_END))
    mask[np.ix_(masked, np.arange(ts.n_channels), resp[: int(0.3 * len(resp))])] = True
    out = reject_trials(apply_mask(replace(ts, data=data), mask))

    assert out.n_trials == N_TRIALS
    rejected = out.metadata["rejected"].to_numpy()
    assert rejected[corrupted].mean() >= MIN_REJECTED
    clean = np.setdiff1d(np.arange(200), corrupted)
    assert rejected[clean].mean() <= MAX_FALSE_REJECTED
    assert set(out.metadata.loc[flat, "reject_reason"]) == {"flat"}
    assert out.select(query="not rejected").n_trials == 200 - rejected.sum()


def test_quality_threshold_and_cap() -> None:

    meta = pd.DataFrame({"quality": [0.9] * 18 + [0.2, 0.3]})
    ts = epochs(n_trials=20, metadata=meta)
    out = reject_trials(ts)
    assert out.metadata["rejected"].tolist() == [False] * 18 + [True, True]
    assert out.metadata["reject_reason"].iloc[-1] == "quality"

    # A broken channel: most trials extreme on it -> capped, with a warning
    data = ts.data.copy()
    data[:8, 0] *= 50  # 40 % of the trials
    with pytest.warns(UserWarning, match="the channel, not the trials"):
        out = reject_trials(replace(ts, data=data, metadata=None))
    assert out.provenance.steps[-1]["settings"]["capped_channels"] == ["A1"]
