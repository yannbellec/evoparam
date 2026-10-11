# SPDX-License-Identifier: Apache-2.0
"""Averaging over trials: mean, trimmed mean, median, quality-weighted mean."""

import time as _time
import warnings
from dataclasses import replace

import numpy as np

from evoparam.data import Evoked
from evoparam.errors import DataError, SettingsError

from ._common import check_trialset, masked_values

__all__ = ["average", "AVERAGE_METHODS"]

AVERAGE_METHODS = ("mean", "trimmed", "median", "weighted")


def average(ts, *, by=None, method="mean", trim=0.2, use_rejected=False):
    """Average the trials into an :class:`Evoked`.

    - ``"mean"`` / ``"trimmed"`` (20 % trimmed mean): :meth:`TrialSet.average`.
    - ``"median"``: median over trials, sample by sample.
    - ``"weighted"``: mean weighted by ``metadata["quality"]``.

    Rejected trials (``metadata["rejected"]``) are left out unless ``use_rejected``; masked
    samples are left out sample by sample. ``sem`` is computed from the untrimmed trials
    (standard deviation over sqrt(n)); ``n_trials`` is per channel (the smallest count over
    time for that channel).

    Parameters
    ----------
    ts : TrialSet
    by : str or list of str, optional
        Metadata columns to group by (one Evoked per group).
    method : str
        One of :data:`AVERAGE_METHODS`.
    trim : float
        Proportion trimmed at each end for ``"trimmed"`` (only 0.2 is supported, the value
        of :meth:`TrialSet.average`).
    use_rejected : bool

    Returns
    -------
    Evoked or list of Evoked
    """

    start = _time.perf_counter()
    check_trialset(ts)
    if method not in AVERAGE_METHODS:
        raise SettingsError(f"method must be one of {AVERAGE_METHODS}, not {method!r}.")
    if method == "trimmed" and trim != 0.2:
        raise SettingsError("trim: only the 20 % trimmed mean is available.")
    work = ts
    if use_rejected and ts.metadata is not None and "rejected" in ts.metadata.columns:
        meta = ts.metadata.copy()
        meta["rejected"] = False
        work = replace(ts, metadata=meta)
    if method in ("mean", "trimmed"):
        return work.average(by=by, robust=method == "trimmed")

    if by is not None:
        columns = [by] if isinstance(by, str) else list(by)
        out = []
        for _, rows in work.metadata.groupby(columns, sort=True, dropna=False).indices.items():
            out.append(average(work.select(trials=list(rows)), method=method,
                               use_rejected=use_rejected))
        return out

    keep = np.arange(work.n_trials)
    if work.metadata is not None and "rejected" in work.metadata.columns:
        keep = np.flatnonzero(~work.metadata["rejected"].to_numpy(dtype=bool))
    if keep.size == 0:
        raise DataError("average: every trial is marked rejected.")
    values = masked_values(work)[keep]
    count = np.sum(~np.isnan(values), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if method == "median":
            mean = np.nanmedian(values, axis=0)
        else:
            if work.metadata is None or "quality" not in work.metadata.columns:
                raise DataError("method 'weighted' needs metadata['quality'].")
            w = work.metadata["quality"].to_numpy(dtype=float)[keep][:, None, None]
            w = np.where(np.isnan(values), 0.0, np.clip(w, 0, None))
            mean = np.nansum(w * np.nan_to_num(values), axis=0) / np.sum(w, axis=0)
        sem = np.nanstd(values, axis=0, ddof=1) / np.sqrt(count)
    sem[count < 2] = np.nan
    masks = count == 0
    provenance = work.provenance.with_step(
        "evoparam.prep.average", {"method": method, "use_rejected": use_rejected,
                                  "n_used": int(keep.size)},
        duration_s=_time.perf_counter() - start)
    provenance.data_hash = None
    return Evoked(data=np.where(masks, np.nan, mean), times=work.times, sfreq=work.sfreq,
                  ch_names=work.ch_names, units=work.units, n_trials=count.min(axis=1),
                  sem=sem, masks=masks if masks.any() else None, ch_info=work.ch_info,
                  species=work.species, modality=work.modality, provenance=provenance)
