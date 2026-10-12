# SPDX-License-Identifier: Apache-2.0
"""Baseline normalisation of time-frequency power (spec §5.10, method 2)."""

import warnings

import numpy as np

from evoparam.errors import SettingsError

__all__ = ["BASELINE_MODES", "BASELINE_LEVELS", "apply_mode", "normalize"]

BASELINE_MODES = ("db", "percent", "zscore", "logratio", "ratio")
BASELINE_LEVELS = ("evoked", "single_trial", "full_epoch")


def apply_mode(power, ref, sd, mode):
    """Normalised power for a reference level ``ref`` (and spread ``sd`` for ``zscore``)."""

    with np.errstate(divide="ignore", invalid="ignore"):
        ref = np.where(ref > 0, ref, np.nan)
        if mode == "db":
            return 10.0 * np.log10(power / ref)
        if mode == "logratio":
            return np.log10(power / ref)
        if mode == "percent":
            return 100.0 * (power - ref) / ref
        if mode == "ratio":
            return power / ref
        if mode == "zscore":
            sd = np.where(sd > 0, sd, np.nan)
            return (power - ref) / sd
    raise SettingsError(f"baseline_mode must be one of {BASELINE_MODES}, not {mode!r}.")


def _nanmean(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _nanstd(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(a, axis=axis, ddof=1)


def normalize(power, ref_times, mode, level, trials=None):
    """Normalise per-trial power (n_trials, n_freqs, n_times).

    Parameters
    ----------
    power : ndarray
        Power per trial; NaN marks invalid pixels.
    ref_times : bool array (n_times,)
        The baseline samples (ignored for ``full_epoch``, which uses every valid sample).
    mode, level : str
        See :data:`BASELINE_MODES` and :data:`BASELINE_LEVELS`.
    trials : index array, optional
        Restrict to these trials (split halves); the evoked reference is then the subset's.

    Returns
    -------
    per_trial : ndarray (n_trials, n_freqs, n_times)
        Each trial's normalised power (with the common reference for ``evoked``).
    reported : ndarray (n_freqs, n_times)
        The map that is reported: for ``single_trial`` and ``full_epoch``, the trial mean of
        the power ratios to each trial's own reference, then the mode (``db``:
        ``10 log10`` of the mean ratio; Grandchamp and Delorme 2011); the mean of the
        per-trial values for ``zscore``; for ``evoked``, the normalised trial-average power.
    test : ndarray (n_trials, n_freqs, n_times)
        The quantity tested against zero: the cube root of each trial's power minus its
        mean over the reference samples. Three choices keep the tests at their level.
        (1) Testing against the baseline level of the same quantity, not against zero,
        removes the bias of single-trial normalisation (the mean of the log of a power
        ratio is not zero under the null; Ciuparu and Muresan 2016). (2) The cube root
        (Wilson-Hilferty) makes the power of a narrow-band signal, an exponential
        (chi-square) variable, nearly symmetric: the t-test on dB values, which are strongly
        left-skewed, gave four times too many positive false alarms on null data, and the
        sign-flip permutation needs symmetry. (3) No division by the trial's own baseline:
        that noisy divisor, correlated with the subtracted baseline level, biased the
        difference upwards (Jensen) and the cluster test gave 17 % family-wise errors on
        null data. The test is therefore the same whatever ``mode`` and ``level``; only the
        reported map changes.
    """

    if mode not in BASELINE_MODES:
        raise SettingsError(f"baseline_mode must be one of {BASELINE_MODES}, not {mode!r}.")
    if level not in BASELINE_LEVELS:
        raise SettingsError(f"baseline_level must be one of {BASELINE_LEVELS}, not {level!r}.")
    if level == "full_epoch" and mode == "zscore":
        raise SettingsError(
            "baseline_level='full_epoch' with baseline_mode='zscore' is "
            "undefined (the full epoch contains the response); use 'db', "
            "'percent', 'logratio' or 'ratio'."
        )
    if trials is not None:
        power = power[trials]
    ref_sel = np.ones(power.shape[-1], bool) if level == "full_epoch" else np.asarray(ref_times)
    if level == "evoked":
        average = _nanmean(power, 0)
        ref = _nanmean(average[:, ref_sel], -1)[:, None]
        sd = _nanstd(average[:, ref_sel], -1)[:, None]
        per_trial = apply_mode(power, ref[None], sd[None], mode)
        reported = apply_mode(average, ref, sd, mode)
    else:
        ref = _nanmean(power[..., ref_sel], -1)[..., None]
        sd = _nanstd(power[..., ref_sel], -1)[..., None]
        per_trial = apply_mode(power, ref, sd, mode)
        if mode == "zscore":
            reported = _nanmean(per_trial, 0)
        else:
            # average the power ratios, then take the log: the mean of per-trial dB values
            # is about -2.5 dB on noise (the mean of a log is below the log of the mean)
            ratio = _nanmean(apply_mode(power, ref, None, "ratio"), 0)
            reported = apply_mode(ratio, np.ones_like(ratio), None, mode)
    root = np.cbrt(power)
    test = root - _nanmean(root[..., ref_sel], -1)[..., None]
    return per_trial, reported, test


def reference_level(average_power, ref_times, mode):
    """Reference (and spread) of a trial-average power map, for normalising other maps
    (the footprint) on the total's scale."""

    ref = _nanmean(average_power[:, ref_times], -1)[:, None]
    sd = _nanstd(average_power[:, ref_times], -1)[:, None]
    return ref, sd
