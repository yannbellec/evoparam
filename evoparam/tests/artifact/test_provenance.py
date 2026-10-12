# SPDX-License-Identifier: Apache-2.0
"""Settings and provenance re-create the result (REQ-069)."""

import warnings

import numpy as np
import pytest

from evoparam.artifact import remove_artifact

from .helpers import make_overlay

###################################################################################################
###################################################################################################

CASES = [
    ("interpolate", {"interpolation": "pchip"}),
    ("template", {"template": "local"}),
    ("learned", {}),
    ("tail", {"tail": "subtract"}),
    ("subspace", {}),
    ("polynomial", {}),
    ("matching_pursuit", {}),
    ("none", {}),
]


@pytest.mark.parametrize(("method", "kwargs"), CASES)
def test_req069_settings_recreate_the_result(method, kwargs):

    noisy, _ = make_overlay(n_channels=12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = remove_artifact(noisy, method=method, **kwargs)
        settings = {k: v for k, v in result.settings.items() if k != "_info"}
        again = remove_artifact(noisy, **settings)
    assert np.max(np.abs(again.cleaned.data - result.cleaned.data)) <= 1e-12
    assert result.method == method
    step = result.cleaned.provenance.steps[-1]
    assert step["function"] == "evoparam.artifact.remove_artifact"
    assert step["settings"]["method"] == method
    assert (
        step["settings"]["window"] == list(result.window)
        or tuple(step["settings"]["window"]) == result.window
    )
    if result.cleaned.masks is not None:
        assert result.cleaned.masks.shape == noisy.data.shape


def test_result_saves_and_loads(tmp_path):

    noisy, _ = make_overlay()
    result = remove_artifact(noisy, method="template")
    path = result.cleaned.save(tmp_path / "cleaned.evoparam")
    from evoparam.data import TrialSet

    loaded = TrialSet.load(path)
    assert np.array_equal(loaded.data, result.cleaned.data)
    assert np.allclose(loaded.metadata["quality"], result.quality)
