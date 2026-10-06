"""Fit the peak model on epoched data."""

import numpy as np

from evoparam.objs import evoparamGroup

###################################################################################################
###################################################################################################

def fit_epochs(times, epochs, time_range, baseline, average=True, **settings):
    """Fit the peak model on each channel of epoched data.

    Parameters
    ----------
    times : 1d array
        Time of each sample relative to the stimulation, in seconds.
    epochs : 3d array
        Epochs, as [n_epochs, n_channels, n_times].
    time_range : list of [float, float]
        Time range to fit, in seconds (for example [0.010, 0.500] for a CCEP).
    baseline : list of [float, float]
        Time range used to estimate the noise level, in seconds (for example [-0.5, -0.05]).
    average : bool, optional, default: True
        If True, fits the average across epochs, one model per channel.
        Trial-by-trial fitting is planned, but not available yet.
    **settings
        Settings passed to `evoparamGroup` (for example max_n_peaks, peak_width_limits).

    Returns
    -------
    model : evoparamGroup
        Fitted models, one per channel, in channel order.
    """

    if not average:
        raise NotImplementedError('Trial-by-trial fitting is not available yet.')

    epochs = np.asarray(epochs, dtype=float)
    if epochs.ndim != 3:
        raise ValueError('epochs must be a 3d array, as [n_epochs, n_channels, n_times].')

    settings.setdefault('verbose', False)
    model = evoparamGroup(**settings)
    model.fit(np.asarray(times, dtype=float), epochs.mean(axis=0), time_range, baseline)

    return model
