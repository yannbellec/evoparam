"""Tests for evoparam.io.epochs."""

import numpy as np

from evoparam.io.epochs import epoch_data

###################################################################################################
###################################################################################################

def test_epoch_data():

    sfreq = 100.
    data = np.vstack([np.arange(1000), -np.arange(1000)]).astype(float)

    times, epochs, kept = epoch_data(data, sfreq, [1.0, 5.0, 9.95], -0.1, 0.2)
    assert np.allclose(times, np.arange(-10, 21) / sfreq)
    assert epochs.shape == (2, 2, 31)
    assert list(kept) == [True, True, False]
    assert epochs[0, 0, 10] == 100 and epochs[1, 1, 10] == -500

    # With an offset clock, and 1d data
    times, epochs, kept = epoch_data(data[0], sfreq, [11.0], -0.1, 0.2, t0=10.)
    assert epochs.shape == (1, 31) and epochs[0, 10] == 100
