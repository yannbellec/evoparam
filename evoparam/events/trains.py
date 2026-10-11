# SPDX-License-Identifier: Apache-2.0
"""Group pulses into trains, summarise them, and turn an event table into trial metadata."""

import numpy as np

from evoparam.errors import EventError
from evoparam.utils._optional import require

__all__ = ["group_trains", "trains", "events_to_metadata"]

# Largest gap inside a train, as a multiple of the expected (or median) interval.
GAP_FACTOR = 3.0

# Default largest gap inside a train, in seconds, when there is a single pulse.
DEFAULT_MAX_GAP = 2.0


def _times(events):
    if "onset_time" not in events.columns:
        raise EventError("The event table needs an 'onset_time' column.")
    return np.asarray(events["onset_time"], dtype=float)


def group_trains(events, *, max_gap=None, expected_rate=None, site_column="stim_site_guess"):
    """Assign each pulse to a train.

    Pulses are sorted by time. A new train starts when the stimulation site changes
    (when ``site_column`` exists and both sites are known: in SPES protocols each site
    receives its pulses in a block), or at a gap longer than ``max_gap``. When both
    sites are known and the gap was not set (no ``max_gap``, no ``expected_rate``), a
    pause inside a site does not start a new train: protocols such as Mayo's pause for
    tens of seconds within a site.

    Parameters
    ----------
    events : pandas.DataFrame
        Event table with an ``onset_time`` column (s).
    max_gap : float, optional
        Largest gap inside a train, in seconds. Default: ``3 / expected_rate`` when the
        rate is known, else 3 times the median interval between pulses (2 s for a single
        pulse). The median interval is the rate the recording itself shows; a fixed 2 s
        would split a 0.2 Hz SPES protocol into single pulses.
    expected_rate : float, optional
        Pulse rate inside a train, in Hz.
    site_column : str or None, default: "stim_site_guess"
        Column holding the stimulation site; None to split on gaps only.

    Returns
    -------
    events : pandas.DataFrame
        Copy sorted by time, with ``train_id`` and ``index_in_train`` (int64, from 0).
    """

    require("pandas", extra="data", caller="evoparam.events.group_trains")
    out = events.copy()
    explicit = max_gap is not None or bool(expected_rate)
    if max_gap is None:
        if expected_rate:
            max_gap = GAP_FACTOR / expected_rate
        elif len(out) > 1:
            max_gap = GAP_FACTOR * float(np.median(np.diff(np.sort(_times(out)))))
        else:
            max_gap = DEFAULT_MAX_GAP
    order = np.argsort(_times(out), kind="stable")
    out = out.iloc[order].reset_index(drop=True)
    times = _times(out)

    new = np.zeros(len(out), dtype=bool)
    if len(out):
        new[0] = True
        gap = np.diff(times) > max_gap
        if site_column is not None and site_column in out.columns:
            sites = out[site_column].fillna("").astype(str).to_numpy()
            known = (sites[1:] != "") & (sites[:-1] != "")
            change = known & (sites[1:] != sites[:-1])
            # With both sites known, a pause inside a site is not a new train unless the
            # gap was set explicitly (max_gap or expected_rate)
            new[1:] = change | (gap & (explicit | ~known))
        else:
            new[1:] = gap
    train_id = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    index_in_train = np.arange(len(out)) - starts[train_id] if len(out) else np.array([], int)
    out["train_id"] = train_id.astype("int64")
    out["index_in_train"] = np.asarray(index_in_train, dtype="int64")
    return out


def trains(events, *, site_column="stim_site_guess", current_column="stim_current_mA"):
    """One row per train.

    Parameters
    ----------
    events : pandas.DataFrame
        Event table with ``onset_time`` and ``train_id`` columns.
    site_column, current_column : str
        Columns used for the site (most frequent value) and the current (median),
        when present.

    Returns
    -------
    table : pandas.DataFrame
        ``train_id``, ``site``, ``n_pulses``, ``onset`` (s, first pulse),
        ``mean_interval`` (s, NaN for a single pulse), ``current`` (mA, NaN if unknown).
    """

    pd = require("pandas", extra="data", caller="evoparam.events.trains")
    if "train_id" not in events.columns:
        raise EventError("The event table has no 'train_id' column; run group_trains first.")
    rows = []
    for train_id, group in events.groupby("train_id", sort=True):
        times = np.sort(_times(group))
        site = ""
        if site_column in group.columns:
            values = group[site_column].fillna("").astype(str)
            values = values[values != ""]
            site = values.mode().iloc[0] if len(values) else ""
        current = np.nan
        if current_column in group.columns:
            current = float(np.nanmedian(pd.to_numeric(group[current_column], errors="coerce")))
        rows.append({"train_id": int(train_id), "site": site, "n_pulses": len(group),
                     "onset": float(times[0]),
                     "mean_interval": float(np.mean(np.diff(times))) if len(times) > 1 else np.nan,
                     "current": current})
    columns = ["train_id", "site", "n_pulses", "onset", "mean_interval", "current"]
    return pd.DataFrame(rows, columns=columns)


def events_to_metadata(events, *, site_column="stim_site_guess"):
    """Turn an event table into trial metadata (standard columns of the TrialSet).

    Parameters
    ----------
    events : pandas.DataFrame
        Event table, as returned by :func:`detect_pulses`.
    site_column : str, default: "stim_site_guess"
        Column copied into ``stim_site``.

    Returns
    -------
    metadata : pandas.DataFrame
        One row per pulse: ``trial``, ``stim_site``, ``block`` (the train), ``onset``
        (s), ``detector``, and ``stim_current_mA`` when the table has it. Pass it, with
        ``onset``, to :func:`evoparam.io.epoch`.
    """

    pd = require("pandas", extra="data", caller="evoparam.events.events_to_metadata")
    n = len(events)
    meta = pd.DataFrame({"trial": np.arange(n, dtype="int64")})
    sites = events[site_column] if site_column in events.columns else [""] * n
    meta["stim_site"] = pd.Series(sites, dtype=object).fillna("").astype(str).to_numpy()
    if "train_id" in events.columns:
        meta["block"] = events["train_id"].to_numpy(dtype="int64")
    meta["onset"] = _times(events)
    if "detector" in events.columns:
        meta["detector"] = events["detector"].astype(str).to_numpy()
    if "stim_current_mA" in events.columns:
        meta["stim_current_mA"] = pd.to_numeric(events["stim_current_mA"],
                                                errors="coerce").to_numpy(dtype=float)
    return meta
