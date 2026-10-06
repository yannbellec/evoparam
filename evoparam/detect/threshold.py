"""Find stimulation onsets from the stimulation artifact, with a robust threshold."""

import numpy as np

###################################################################################################
###################################################################################################

def detect_stim_onsets(data, sfreq, threshold=20., min_interval=0.5, t0=0.):
    """Detect stimulation onsets from the sharp artifact that each pulse leaves in the signal.

    Parameters
    ----------
    data : 1d or 2d array
        Continuous data, as [n_times] or [n_channels, n_times].
    sfreq : float
        Sampling frequency, in Hz.
    threshold : float, optional, default: 20
        Detection threshold, in robust standard deviations (median absolute deviation
        scaled to match a standard deviation) of the artifact score.
    min_interval : float, optional, default: 0.5
        Minimum time between two onsets, in seconds. A crossing closer than this to the
        previous onset is treated as part of the same artifact.
    t0 : float, optional, default: 0
        Time of the first sample of `data`, in seconds.

    Returns
    -------
    onsets : 1d array
        Detected onsets, in seconds.

    Notes
    -----
    The artifact score is the absolute sample-to-sample change of the signal, taken as the
    median across channels, so that a few noisy channels cannot create or hide a detection.
    The onset is the first sample whose score crosses the threshold.

    This is the baseline method of the module: a starting point to compare better methods against.
    """

    data = np.atleast_2d(np.asarray(data, dtype=float))
    score = np.median(np.abs(np.diff(data, axis=-1)), axis=0)

    center = np.median(score)
    spread = 1.4826 * np.median(np.abs(score - center))
    if spread == 0:
        spread = np.std(score) or 1.
    zscore = (score - center) / spread

    crossings = np.flatnonzero((zscore[1:] > threshold) & (zscore[:-1] <= threshold)) + 1
    if zscore[0] > threshold:
        crossings = np.r_[0, crossings]

    onsets, last = [], -np.inf
    for ind in crossings:
        if ind - last >= min_interval * sfreq:
            onsets.append(ind)
            last = ind

    # Score index i measures the change from sample i to i + 1: the artifact starts at i + 1
    return t0 + (np.array(onsets, dtype=float) + 1) / sfreq


def match_onsets(detected, reference, tolerance=0.01):
    """Compare detected onsets to reference onsets.

    Parameters
    ----------
    detected, reference : 1d array
        Onsets, in seconds.
    tolerance : float, optional, default: 0.01
        Largest time difference, in seconds, for a detection to count as a match.

    Returns
    -------
    results : dict
        'hits' (number of reference onsets found), 'misses', 'false_alarms', and
        'errors' (1d array, detected minus reference time for each hit, in seconds).
    """

    detected, reference = np.sort(np.asarray(detected)), np.sort(np.asarray(reference))
    used, errors = np.zeros(len(detected), dtype=bool), []

    for ref in reference:
        if len(detected) == 0:
            break
        diffs = np.abs(detected - ref)
        diffs[used] = np.inf
        ind = np.argmin(diffs)
        if diffs[ind] <= tolerance:
            used[ind] = True
            errors.append(detected[ind] - ref)

    return {'hits' : len(errors), 'misses' : len(reference) - len(errors),
            'false_alarms' : int(np.sum(~used)), 'errors' : np.array(errors)}
