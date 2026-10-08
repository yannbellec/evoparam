"""Learned artifact basis (REQ-061)."""

import numpy as np

from evoparam.artifact import remove_artifact

from .helpers import make_overlay
from .test_template import WINDOW, residual_db

###################################################################################################
###################################################################################################

def test_req061_learned_reduces_window_residual_by_20_db():

    noisy, truth = make_overlay(onset=0.012)
    result = remove_artifact(noisy, method="learned", window=WINDOW)
    assert residual_db(noisy, truth, result.cleaned) >= 20
    n_window = np.sum((noisy.times >= WINDOW[0]) & (noisy.times <= WINDOW[1]))
    assert result.basis.shape[1] == n_window
    assert 1 <= result.basis.shape[0] <= 8
    # Orthonormal rows
    assert np.allclose(result.basis @ result.basis.T, np.eye(result.basis.shape[0]), atol=1e-8)


def test_learned_fixed_number_of_components():

    noisy, _ = make_overlay()
    result = remove_artifact(noisy, method="learned", window=WINDOW, n_components=2)
    assert result.basis.shape[0] == 2
