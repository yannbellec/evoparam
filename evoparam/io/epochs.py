"""Cut a continuous recording into epochs around events."""

import numpy as np

###################################################################################################
###################################################################################################

def epoch_data(data, sfreq, onsets, tmin, tmax, t0=0.):
    """Cut epochs from continuous data, around event onsets.

    Parameters
    ----------
    data : 1d or 2d array
        Continuous data, as [n_times] or [n_channels, n_times].
    sfreq : float
        Sampling frequency, in Hz.
    onsets : 1d array
        Event onsets, in seconds, on the same clock as `t0`.
    tmin, tmax : float
        Epoch window relative to each onset, in seconds (for example -0.5 and 1.0).
    t0 : float, optional, default: 0
        Time of the first sample of `data`, in seconds.

    Returns
    -------
    times : 1d array
        Time of each epoch sample relative to the onset, in seconds.
    epochs : 2d or 3d array
        Epochs, as [n_events, n_times] or [n_events, n_channels, n_times].
    kept : 1d array of bool
        Which onsets gave an epoch; epochs that do not fit inside `data` are dropped.
    """

    data = np.asarray(data)
    first, last = int(np.round(tmin * sfreq)), int(np.round(tmax * sfreq))
    times = np.arange(first, last + 1) / sfreq

    centers = np.round((np.asarray(onsets, dtype=float) - t0) * sfreq).astype(int)
    kept = (centers + first >= 0) & (centers + last < data.shape[-1])

    inds = centers[kept][:, None] + np.arange(first, last + 1)[None, :]
    epochs = data[..., inds]
    if data.ndim == 2:
        epochs = np.moveaxis(epochs, 0, 1)

    return times, epochs, kept
