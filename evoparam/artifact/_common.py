# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the artifact-removal methods."""

import time as _time
from dataclasses import replace

import numpy as np

from evoparam.data import TrialSet
from evoparam.errors import ArtifactError, DataError
from evoparam.utils._interp import window_indices

__all__ = []

# Default grouping of pulses that share one artifact template.
GROUP_BY = ("stim_site", "stim_current_mA", "stim_polarity")


def check_trialset(ts):
    if not isinstance(ts, TrialSet):
        raise DataError(f"Expected a TrialSet, got {type(ts).__name__}.")
    return ts


def window_slice(ts, window, buffer=0):
    """Indices of the samples of ``ts`` inside ``window`` (s, inclusive)."""

    return window_indices(ts.times, window, buffer)


def groups(ts, group_by):
    """Trial groups sharing the values of the ``group_by`` metadata columns.

    Columns missing from the metadata are ignored; without metadata (or with no usable
    column) every trial is in one group. NaN values form their own group.

    Returns
    -------
    list of (key, indices)
        ``key`` is a tuple of the group values (empty when there is one group).
    """

    meta = ts.metadata
    columns = [col for col in (group_by or ()) if meta is not None and col in meta.columns]
    if not columns:
        return [((), np.arange(ts.n_trials))]
    keys = meta[columns].astype(object).where(meta[columns].notna(), "nan")
    out = {}
    for row, key in enumerate(map(tuple, keys.to_numpy())):
        out.setdefault(key, []).append(row)
    return [(key, np.array(rows)) for key, rows in out.items()]


def base_masks(ts):
    """Existing masks (for example saturation), or all False."""

    return np.zeros(ts.data.shape, dtype=bool) if ts.masks is None else ts.masks.copy()


def finish(ts, data, masks, function, settings, start, **changes):
    """New TrialSet with cleaned data, masks and one provenance step."""

    masks = masks if masks is not None and masks.any() else None
    if masks is None and ts.masks is not None and ts.masks.any():
        masks = ts.masks.copy()
    provenance = ts.provenance.with_step(function, settings,
                                         duration_s=_time.perf_counter() - start)
    provenance.data_hash = None
    return replace(ts, data=data, masks=masks, provenance=provenance, **changes)


def apply_mask(ts, mask):
    """Return a copy of ``ts`` whose masks also include ``mask``.

    Parameters
    ----------
    ts : TrialSet
    mask : array of bool
        Shape of ``ts.data``, or broadcastable to it (for example [n_channels, n_times]
        or [n_times]).

    Returns
    -------
    TrialSet
        Same data; ``masks`` is the union of the old masks and ``mask``.
    """

    start = _time.perf_counter()
    check_trialset(ts)
    try:
        mask = np.broadcast_to(np.asarray(mask, dtype=bool), ts.data.shape)
    except ValueError:
        raise DataError(f"mask of shape {np.shape(mask)} does not broadcast to "
                        f"{ts.data.shape}.") from None
    masks = base_masks(ts) | mask
    return finish(ts, ts.data.copy(), masks, "evoparam.artifact.apply_mask",
                  {"n_masked": int(mask.sum())}, start)


def baseline_rms(data, inds):
    """Root mean square per trial (over channels and samples) in a baseline window."""

    part = data[:, :, inds]
    center = np.nanmean(part, axis=-1, keepdims=True)
    return np.sqrt(np.nanmean((part - center) ** 2, axis=(1, 2)))


def require_trials(n, needed, method, what="trials"):
    if n < needed:
        raise ArtifactError(f"method {method!r} needs at least {needed} {what}, got {n}.")
