# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the artifact-removal methods."""

import time as _time
from dataclasses import replace
from typing import cast

import numpy as np

from evoparam._typing import Any, BoolArray, NDArray
from evoparam.data import TrialSet
from evoparam.data._table import is_missing
from evoparam.errors import ArtifactError, DataError
from evoparam.utils._interp import window_indices

__all__ = []

# Default grouping of pulses that share one artifact template.
GROUP_BY = ("stim_site", "stim_current_mA", "stim_polarity")


def check_trialset(ts: Any) -> "TrialSet":
    """Return ``ts`` after checking that it is a TrialSet."""
    if not isinstance(ts, TrialSet):
        raise DataError(f"Expected a TrialSet, got {type(ts).__name__}.")
    return ts


def window_slice(ts: Any, window: Any, buffer: int = 0) -> NDArray[Any]:
    """Return the indices of the samples of ``ts`` inside ``window`` (s, inclusive)."""
    return window_indices(ts.times, window, buffer)


def groups(ts: Any, group_by: Any) -> Any:
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
    # the values of each column (a DataFrame, or a Table without pandas), missing as "nan"
    values = [np.asarray(meta[col], dtype=object) for col in columns]
    out: dict[Any, Any]
    out = {}
    for row in range(ts.n_trials):
        key = tuple("nan" if is_missing(column[row]) else column[row] for column in values)
        out.setdefault(key, []).append(row)
    return [(key, np.array(rows)) for key, rows in out.items()]


def base_masks(ts: Any) -> Any:
    """Existing masks (for example saturation), or all False."""
    return np.zeros(ts.data.shape, dtype=bool) if ts.masks is None else ts.masks.copy()


def finish(
    ts: Any, data: Any, masks: Any, function: Any, settings: Any, start: Any, **changes: Any
) -> Any:
    """Return a new TrialSet with cleaned data, masks and one provenance step."""
    masks = masks if masks is not None and masks.any() else None
    if masks is None and ts.masks is not None and ts.masks.any():
        masks = ts.masks.copy()
    provenance = ts.provenance.with_step(
        function, settings, duration_s=_time.perf_counter() - start
    )
    provenance.data_hash = None
    return replace(ts, data=data, masks=masks, provenance=provenance, **changes)


def apply_mask(ts: "TrialSet", mask: BoolArray) -> "TrialSet":
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
        raise DataError(
            f"mask of shape {np.shape(mask)} does not broadcast to {ts.data.shape}."
        ) from None
    masks = base_masks(ts) | mask
    return cast(
        "TrialSet",
        finish(
            ts,
            ts.data.copy(),
            masks,
            "evoparam.artifact.apply_mask",
            {"n_masked": int(mask.sum())},
            start,
        ),
    )


def baseline_rms(data: Any, inds: Any) -> NDArray[Any]:
    """Root mean square per trial (over channels and samples) in a baseline window."""
    part = data[:, :, inds]
    center = np.nanmean(part, axis=-1, keepdims=True)
    spread: NDArray[Any] = np.sqrt(np.nanmean((part - center) ** 2, axis=(1, 2)))
    return spread


def require_trials(n: Any, needed: Any, method: Any, what: str = "trials") -> None:
    """Raise ArtifactError when ``method`` gets fewer than ``needed`` ``what`` (``n``)."""
    if n < needed:
        raise ArtifactError(f"method {method!r} needs at least {needed} {what}, got {n}.")
