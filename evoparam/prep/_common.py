# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the preparation functions."""

import time as _time
from dataclasses import replace

import numpy as np

from evoparam.data import TrialSet
from evoparam.errors import DataError

__all__ = []


def check_trialset(ts):
    if not isinstance(ts, TrialSet):
        raise DataError(f"Expected a TrialSet, got {type(ts).__name__}.")
    return ts


def window_mask(times, window, name="window"):
    """Boolean mask of the samples inside ``window`` (inclusive); DataError if empty."""

    inside = (times >= window[0]) & (times <= window[1])
    if not inside.any():
        raise DataError(
            f"{name} {tuple(window)} s contains no sample "
            f"(epoch {times[0]:.4f} to {times[-1]:.4f} s)."
        )
    return inside


def masked_values(ts):
    """Data with masked samples set to NaN (a copy)."""

    data = ts.data.copy()
    if ts.masks is not None:
        data[ts.masks] = np.nan
    return data


def new(ts, function, settings, start, **changes):
    """Copy of ``ts`` with changes and one provenance step."""

    provenance = ts.provenance.with_step(
        function, settings, duration_s=_time.perf_counter() - start
    )
    if "data" in changes:
        provenance.data_hash = None
    return replace(ts, provenance=provenance, **changes)


def ch_table(ts):
    """The channel table, or a minimal one built from the names."""

    import pandas as pd

    if ts.ch_info is not None:
        return ts.ch_info.copy()
    return pd.DataFrame({"name": list(ts.ch_names)})


def stim_contacts(ts):
    """Per trial, the set of stimulated contact names from ``metadata["stim_site"]``."""

    out = [set() for _ in range(ts.n_trials)]
    if ts.metadata is None or "stim_site" not in ts.metadata.columns:
        return out
    for row, site in enumerate(ts.metadata["stim_site"].fillna("").astype(str)):
        out[row] = {part.strip() for part in site.split("-") if part.strip()}
    return out


def bad_channels(ts):
    """Set of channel names marked bad in ``ch_info``."""

    if ts.ch_info is None or "bad" not in ts.ch_info.columns:
        return set()
    info = ts.ch_info
    return set(info.loc[info["bad"].fillna(False).astype(bool), "name"])
