# SPDX-License-Identifier: Apache-2.0
"""The scikit-learn contract of ResponseModel, TrialFitter and ArtifactRemover (REQ-320)."""

import pickle

import numpy as np
import pytest

from evoparam.artifact import ArtifactRemover
from evoparam.models import ResponseModel
from evoparam.trials import TrialFitter
from evoparam.utils._estimator import NotFittedError

SFREQ = 1000.0


def _array(seed=0, n=20, channels=2):
    rng = np.random.default_rng(seed)
    times = np.arange(-0.1, 0.4, 1 / SFREQ)
    data = rng.normal(0, 2, (n, channels, times.size))
    data[:, 0] += -30 * np.exp(-0.5 * ((times - 0.08) / 0.01) ** 2)
    return data


def _estimators():
    return [
        ResponseModel(sfreq=SFREQ, tmin=-0.1, fit_window=(0.01, 0.39), max_components=3),
        TrialFitter(sfreq=SFREQ, tmin=-0.1, fit_window=(0.01, 0.39), random_state=0),
        ArtifactRemover("interpolate", sfreq=SFREQ, tmin=-0.1, window=(0.0, 0.002)),
    ]


def _input(est, data):
    return data.mean(axis=0) if isinstance(est, ResponseModel) else data


@pytest.mark.parametrize("est", _estimators(), ids=lambda e: type(e).__name__)
def test_req320_contract(est):

    data = _array()
    params = est.get_params()
    assert est.set_params(**params) is est and est.get_params() == params
    assert not est.__sklearn_is_fitted__()
    with pytest.raises(NotFittedError):
        est.transform(_input(est, data))
    assert isinstance(NotFittedError("x"), (ValueError, AttributeError))
    assert est.fit(_input(est, data)) is est
    assert est.__sklearn_is_fitted__()
    assert all(not k.endswith("_") for k in params)
    again = pickle.loads(pickle.dumps(est))
    assert np.allclose(
        np.nan_to_num(again.transform(_input(est, data))),
        np.nan_to_num(est.transform(_input(est, data))),
    )
    # no learned state on a fresh copy built from the parameters
    fresh = type(est)(**est.get_params())
    assert not fresh.__sklearn_is_fitted__()


def test_same_random_state_same_fit():

    data = _array()
    a = TrialFitter(sfreq=SFREQ, tmin=-0.1, fit_window=(0.01, 0.39), random_state=0).fit(data)
    b = TrialFitter(sfreq=SFREQ, tmin=-0.1, fit_window=(0.01, 0.39), random_state=0).fit(data)
    assert a.group_fit_.to_dataframe().equals(b.group_fit_.to_dataframe())
    features = a.transform(data)
    assert features.shape == (20, 2 * a.n_components_ * 3)
    assert len(a.get_feature_names_out()) == features.shape[1]
    assert np.nanmax(np.abs(features[:, 2] - 0.08)) < 0.01  # channel 0 latency ~ 80 ms


def test_pipeline_and_nested_parameters():

    sklearn = pytest.importorskip("sklearn")
    from sklearn.base import clone
    from sklearn.pipeline import make_pipeline

    pipe = make_pipeline(
        ArtifactRemover("interpolate", sfreq=SFREQ, tmin=-0.1, window=(0.0, 0.002)),
        TrialFitter(sfreq=SFREQ, tmin=-0.1, fit_window=(0.01, 0.39)),
    )
    pipe.set_params(trialfitter__max_components=3)  # the call of REQ-320
    assert pipe.get_params()["trialfitter__max_components"] == 3
    pipe.set_params(trialfitter__max_components=2)
    copy = clone(pipe)
    assert not copy.steps[1][1].__sklearn_is_fitted__()
    features = pipe.fit_transform(_array())
    assert features.shape == (20, 2 * 2 * 3)
    del sklearn
