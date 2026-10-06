"""Remove the stimulation artifact by replacing it with a straight line."""

import numpy as np

###################################################################################################
###################################################################################################

def interpolate_artifact(times, data, window=(-0.002, 0.010), buffer=3):
    """Replace a time window by a straight line between the samples around it.

    Parameters
    ----------
    times : 1d array
        Time of each sample, in seconds (for epochs, relative to the stimulation).
    data : array
        Signal values; time must be the last axis (for example [n_epochs, n_channels, n_times]).
    window : tuple of (float, float), optional, default: (-0.002, 0.010)
        Start and end of the window to replace, in seconds, inclusive.
    buffer : int, optional, default: 3
        Number of samples on each side of the window that are averaged (median) to anchor
        the line. A larger buffer is more robust to noise.

    Returns
    -------
    cleaned : array
        Copy of `data`, with the window replaced.

    Raises
    ------
    ValueError
        If the window is empty, or if there are not `buffer` samples on each side of it.

    Notes
    -----
    This is the simplest artifact removal method. It also removes any brain signal inside
    the window, so the window must end before the earliest response of interest
    (about 10 ms for the CCEP N1).
    """

    times = np.asarray(times)
    inds = np.flatnonzero((times >= window[0]) & (times <= window[1]))
    if len(inds) == 0:
        raise ValueError('The interpolation window contains no samples.')

    first, last = inds[0], inds[-1]
    if first - buffer < 0 or last + 1 + buffer > len(times):
        raise ValueError('Not enough samples around the window for the requested buffer.')

    cleaned = np.array(data, dtype=float, copy=True)
    x_left = np.median(times[first - buffer:first])
    x_right = np.median(times[last + 1:last + 1 + buffer])
    y_left = np.median(cleaned[..., first - buffer:first], axis=-1)
    y_right = np.median(cleaned[..., last + 1:last + 1 + buffer], axis=-1)

    slope = (y_right - y_left) / (x_right - x_left)
    cleaned[..., inds] = y_left[..., None] + slope[..., None] * (times[inds] - x_left)

    return cleaned
