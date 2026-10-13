# SPDX-License-Identifier: Apache-2.0
"""Quality score (REQ-066)."""

from dataclasses import replace
from typing import Any

import numpy as np

from evoparam.artifact import apply_mask, artifact_quality, remove_artifact

from .helpers import artifact_shape, make_overlay

# Start (s) of the residual tail, the clipped span (s), and the AUC of the quality score.
TAIL_START = 0.008
CLIP_START = -0.002
CLIP_END = 0.02
MIN_AUC = 0.9

###################################################################################################
###################################################################################################


def auc(scores_good: Any, scores_bad: Any) -> Any:
    """Probability that a clean trial scores higher than a corrupted one."""

    good, bad = np.asarray(scores_good)[:, None], np.asarray(scores_bad)[None]
    return float(np.mean((good > bad) + 0.5 * (good == bad)))


def test_req066_quality_separates_corrupted_trials() -> None:

    noisy, truth = make_overlay(n_trials=100, n_channels=8, onset=0.02, seed=8)
    data = noisy.data.copy()
    meta = noisy.metadata.copy()
    times, sfreq = noisy.times, noisy.sfreq
    # Two stimulation sites with different artifacts
    site_b = np.arange(50, 100)
    other = artifact_shape(times, sfreq, -2500.0, -300.0, 0.004)
    data[site_b] += (other - artifact_shape(times, sfreq, 3000.0, 500.0, 0.002))[None, None]
    meta.loc[site_b, "stim_site"] = "C3-C4"

    corrupted = np.r_[0:4, 50:52, 4:6, 52:53, 6:8]
    # Residual artifact: a long extra tail that no template contains
    after = times > TAIL_START
    data[0:4, :, after] += 200 * np.exp(-(times[after] - 0.008) / 0.01)
    data[50:52, :, after] += 200 * np.exp(-(times[after] - 0.008) / 0.01)
    # Saturation: clipped samples, masked
    sat = np.zeros(data.shape, dtype=bool)
    clip = (times >= CLIP_START) & (times <= CLIP_END)
    sat[4:6, :, clip] = True
    sat[52, :, clip] = True
    data[sat] = 9000.0
    # Wrong site label: the template of the other site is subtracted
    meta.loc[6:7, "stim_site"] = "C3-C4"

    ts = apply_mask(replace(noisy, data=data, metadata=meta), sat)
    result = remove_artifact(ts, method="template", window=(-0.002, 0.008))
    good = np.setdiff1d(np.arange(100), corrupted)
    assert auc(result.quality[good], result.quality[corrupted]) >= MIN_AUC
    assert np.allclose(result.cleaned.metadata["quality"], result.quality)
    assert list(result.quality_components.columns) == [
        "residual_ratio",
        "edge_discontinuity",
        "saturation_fraction",
        "template_fit_r2",
        "quality",
    ]


def test_quality_range_and_components() -> None:

    noisy, _ = make_overlay()
    table = artifact_quality(noisy, window=(-0.002, 0.008))
    assert ((table["quality"] >= 0) & (table["quality"] <= 1)).all()
    assert table["template_fit_r2"].isna().all()
    assert (table["saturation_fraction"] == 0).all()
