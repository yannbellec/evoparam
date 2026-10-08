"""Tests for evoparam.param.epochs."""

import numpy as np
from pytest import raises

from evoparam.param import fit_epochs
from evoparam.core.sim import simulate_erp

###################################################################################################
###################################################################################################

def test_fit_epochs():

    times, erp = simulate_erp([-0.5, 1.0], [0.05, -2, 0.01, 0.2, 1.5, 0.05], nlv=0.001, fs=1000)
    rng = np.random.default_rng(1)
    epochs = erp[None, None, :] + rng.normal(0, 0.05, (20, 3, len(times)))

    model = fit_epochs(times, epochs, [0.01, 0.6], [-0.5, -0.05], max_n_peaks=4)
    assert len(model) == 3

    centers = np.sort(model.get_evoparam(0).gaussian_params_[:, 0])
    assert np.any(np.abs(centers - 0.05) < 0.01)
    assert np.any(np.abs(centers - 0.2) < 0.02)

    with raises(NotImplementedError):
        fit_epochs(times, epochs, [0.01, 0.6], [-0.5, -0.05], average=False)
    with raises(ValueError):
        fit_epochs(times, epochs[0], [0.01, 0.6], [-0.5, -0.05])
