# SPDX-License-Identifier: Apache-2.0
"""Paradigm-based cancellation (REQ-067)."""

from typing import Any

import numpy as np
import pytest

from evoparam.artifact import paradigm_subtract, remove_artifact
from evoparam.data import TrialSet
from evoparam.errors import ArtifactError

from .helpers import artifact_shape, response, times_axis

pd = pytest.importorskip("pandas")

###################################################################################################
###################################################################################################


def alternating(n_pairs: Any = 10, extra: Any = 0, seed: Any = 0) -> Any:
    rng = np.random.default_rng(seed)
    sfreq = 2048.0
    times = times_axis(sfreq)
    art = artifact_shape(times, sfreq)
    resp = response(times, 0.012, -80.0)
    n = 2 * n_pairs + extra
    signs = np.array([-1, 1] * n_pairs + [-1] * extra)
    data = np.empty((n, 2, len(times)))
    for k in range(n):
        data[k] = resp + signs[k] * art + 0.0 * rng.normal(size=(2, len(times)))
    meta = pd.DataFrame(
        {
            "stim_site": ["A-B"] * n,
            "stim_polarity": np.where(signs < 0, "monophasic_cathodic", "monophasic_anodic"),
        }
    )
    ts = TrialSet(data=data, times=times, sfreq=sfreq, ch_names=["x", "y"], metadata=meta)
    return ts, resp, art


def test_req067_alternating_polarity_cancels() -> None:

    ts, resp, art = alternating()
    out = paradigm_subtract(ts, scheme="alternating")
    assert out.n_trials == 10
    assert out.metadata["paired"].all()
    assert (out.metadata["n_combined"] == 2).all()
    # The non-inverting response is unchanged to 1e-9, the inverting artifact cancels
    assert np.max(np.abs(out.data - resp)) <= 1e-9
    residual = np.abs(out.data - resp).max() / np.abs(art).max()
    assert residual <= 0.01


def test_alternating_with_imperfect_inversion_and_unpaired_trials() -> None:

    ts, resp, art = alternating(extra=1)
    data = ts.data.copy()
    data[1::2][:10] += 0.005 * art  # the anodic artifact is 0.5 % larger
    from dataclasses import replace

    out = paradigm_subtract(replace(ts, data=data), scheme="alternating")
    paired = out.data[out.metadata["paired"].to_numpy()]
    assert np.abs(paired - resp).max() / np.abs(art).max() <= 0.01
    assert out.n_trials == 11
    assert out.metadata["paired"].tolist()[-1] is False or not out.metadata["paired"].iloc[-1]


def test_explicit_pair_ids() -> None:

    ts, resp, _ = alternating(n_pairs=3)
    meta = ts.metadata.copy()
    meta["pair_id"] = [0, 1, 0, 1, 2, 2]
    meta.loc[[0, 1, 2, 3], "stim_polarity"] = [
        "monophasic_cathodic",
        "monophasic_cathodic",
        "monophasic_anodic",
        "monophasic_anodic",
    ]
    from dataclasses import replace

    out = paradigm_subtract(replace(ts, metadata=meta), scheme="alternating", pairing="pair_id")
    assert out.n_trials == 3


def test_req067_forward_masking_identity() -> None:

    rng = np.random.default_rng(1)
    times = times_axis(1000.0)
    conds = {name: rng.normal(size=(1, len(times))) for name in "ABCD"}
    order = list("ABCD") * 3
    data = np.array([conds[name] for name in order])
    meta = pd.DataFrame({"paradigm_condition": order})
    ts = TrialSet(data=data, times=times, sfreq=1000.0, ch_names=["x"], metadata=meta)
    out = paradigm_subtract(ts, scheme="forward_masking")
    expected = conds["A"] - conds["B"] + (conds["C"] - conds["D"])
    assert out.n_trials == 3
    assert np.allclose(out.data, expected[None], atol=1e-12)


def test_paradigm_errors() -> None:

    ts, _, _ = alternating()
    with pytest.raises(ArtifactError):
        paradigm_subtract(ts, scheme="forward_masking")
    with pytest.raises(ArtifactError, match="scheme"):
        remove_artifact(ts, method="paradigm")
    result = remove_artifact(ts, method="paradigm", scheme="alternating")
    assert result.cleaned.n_trials == 10
    assert len(result.quality) == 10
