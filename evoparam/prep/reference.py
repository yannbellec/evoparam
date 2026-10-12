# SPDX-License-Identifier: Apache-2.0
"""Re-referencing: monopolar, bipolar, common average, adjusted common average, Laplacian."""

import time as _time
import warnings

import numpy as np

from evoparam.errors import DataError, SettingsError
from evoparam.errors import warn as _warn
from evoparam.utils._names import contact_group, contact_number
from evoparam.utils._optional import require

from ._common import bad_channels, ch_table, check_trialset, masked_values, new, stim_contacts

__all__ = ["rereference", "car_adaptive_select", "SCHEMES"]

SCHEMES = ("monopolar", "bipolar", "car", "car_adaptive", "laplacian")

# Smallest share of the channels kept in the adjusted common average.
MIN_CA_FRACTION = 0.1

# Fewest channels for the adjusted common average (below: plain common average).
MIN_ADAPTIVE_CHANNELS = 8


###################################################################################################
## Bipolar


def bipolar_pairs(ts, groups="group"):
    """Consecutive contacts of each electrode: (anode, cathode) index pairs, and gaps.

    Groups come from ``ch_info[groups]`` (a column name), from a dict name -> group, or,
    when the column is missing, from the names (:func:`contact_group`). Contacts are ordered
    by their trailing number; a pair is formed only between consecutive existing numbers,
    and the missing numbers are reported as gaps.
    """
    names = list(ts.ch_names)
    if isinstance(groups, dict):
        group_of = [str(groups.get(name, contact_group(name))) for name in names]
    elif groups is not None and ts.ch_info is not None and groups in ts.ch_info.columns:
        group_of = ts.ch_info[groups].fillna("").astype(str).tolist()
        group_of = [g if g else contact_group(n) for g, n in zip(group_of, names, strict=False)]
    else:
        group_of = [contact_group(name) for name in names]
    numbers = [contact_number(name) for name in names]

    pairs, gaps = [], []
    for group in dict.fromkeys(group_of):
        members = [i for i, g in enumerate(group_of) if g == group and numbers[i] is not None]
        members.sort(key=lambda i: numbers[i])
        for a, b in zip(members[:-1], members[1:], strict=False):
            if numbers[b] == numbers[a] + 1:
                pairs.append((a, b))
            else:
                gaps.append((names[a], names[b]))
    if not pairs:
        raise DataError(
            "bipolar: no pair of consecutive contacts found; give the electrode "
            "groups in ch_info['group'] or with groups=."
        )
    return pairs, gaps


def _bipolar(ts, groups):
    pd = require("pandas", extra="data", caller="evoparam.prep.rereference")
    pairs, gaps = bipolar_pairs(ts, groups)
    if gaps:
        _warn(
            f"bipolar: missing contacts between {gaps}; no pair across a gap.",
            UserWarning,
            stacklevel=3,
        )
    a, b = np.array(pairs).T
    data = ts.data[:, a] - ts.data[:, b]
    masks = None if ts.masks is None else ts.masks[:, a] | ts.masks[:, b]
    names = [f"{ts.ch_names[i]}-{ts.ch_names[j]}" for i, j in pairs]
    old = ch_table(ts)
    rows = []
    for i, j in pairs:
        row = {"name": f"{ts.ch_names[i]}-{ts.ch_names[j]}", "reference": "bipolar"}
        for col in old.columns:
            if col in ("name", "reference"):
                continue
            first, second = old[col].iloc[i], old[col].iloc[j]
            if col in ("x", "y", "z"):
                row[col] = float(np.mean([first, second]))
            elif col in ("bad", "stim_contact"):
                row[col] = bool(first) or bool(second)
            else:
                row[col] = first
        rows.append(row)
    info = pd.DataFrame(
        rows, columns=list(dict.fromkeys(["name", "reference"] + list(old.columns)))
    )
    return data, masks, names, info, {"pairs": [list(map(int, p)) for p in pairs], "gaps": gaps}


###################################################################################################
## Common averages


def _excluded(ts, exclude):
    """Per trial, a boolean matrix [n_channels] of channels left out of a common average."""
    names = list(ts.ch_names)
    out = np.zeros((ts.n_trials, ts.n_channels), dtype=bool)
    fixed = set(exclude or ()) | bad_channels(ts)
    if ts.ch_info is not None and "stim_contact" in ts.ch_info.columns:
        fixed |= set(ts.ch_info.loc[ts.ch_info["stim_contact"].fillna(False).astype(bool), "name"])
    for row, stim in enumerate(stim_contacts(ts)):
        out[row] = [name in fixed or name in stim for name in names]
    return out


def _common_average(ts, members):
    """Mean over ``members`` (bool [n_trials, n_channels]) of the unmasked samples."""
    values = masked_values(ts)
    values[~np.broadcast_to(members[:, :, None], values.shape)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=1)


def _cross_trial_covariance(data, window):
    """Mean covariance between pairs of different trials, per channel, in the window."""
    seg = data[:, :, window]
    seg = seg - seg.mean(axis=-1, keepdims=True)
    n_trials, _, n_times = seg.shape
    if n_trials < 2:
        return np.mean(seg[0] ** 2, axis=-1)
    total = seg.sum(axis=0)
    pairs = np.sum(total**2, axis=-1) - np.sum(seg**2, axis=(0, 2))
    return pairs / (n_trials * (n_trials - 1) * n_times)


def _zeta_curve(average, order):
    """zeta(n) for n = 2..N on one (trial-averaged) signal set, from its covariance.

    For the n least responsive channels U (``order[:n]``) and their average m, the
    published statistic (Huang et al. 2024) correlates each channel U_i, *not*
    re-referenced, with every re-referenced channel R_j = U_j - m (j != i), takes the mean
    Fisher z of each U_i, and keeps the most negative one. With C the covariance (over time)
    of the ordered channels: cov(U_i, R_j) = C_ij - s_i and var(R_j) = C_jj - 2 s_j + v,
    where s_i = (1/n) sum_k C_ik and v = (1/n^2) sum_kl C_kl, so every n costs O(n^2).
    """
    seg = average[order]
    seg = seg - seg.mean(axis=1, keepdims=True)
    cov = seg @ seg.T / max(seg.shape[1] - 1, 1)
    out = np.empty(len(order) - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        for n in range(2, len(order) + 1):
            c = cov[:n, :n]
            row = c.sum(axis=1) / n
            var_m = c.sum() / n**2
            var_r = np.diag(c) - 2 * row + var_m
            corr = (c - row[:, None]) / np.sqrt(np.outer(np.diag(c), var_r))
            np.fill_diagonal(corr, np.nan)
            z = np.arctanh(np.clip(corr, -0.999999, 0.999999))
            out[n - 2] = np.nanmin(np.nanmean(z, axis=1))
    return out


def _line_notch(data, sfreq, line_freq):
    """Band-stop at the line frequency and its harmonics up to 180 Hz.

    4th order Butterworth, +/- 2 Hz, zero phase, as the published code does before ranking.
    """
    from scipy.signal import butter, filtfilt

    if not line_freq:
        return data
    out = np.array(data, dtype=float, copy=True)
    for f0 in np.arange(line_freq, 180.0 + 1e-9, line_freq):
        if f0 + 2 >= sfreq / 2:
            break
        b, a = butter(2, [f0 - 2, f0 + 2], btype="bandstop", fs=sfreq)
        out = filtfilt(b, a, out, axis=-1)
    return out


def car_adaptive_select(
    data,
    times,
    *,
    response_window=(0.01, 0.3),
    n_boot=100,
    random_state=None,
    candidates=None,
    sensitive=True,
    line_freq=60.0,
):
    """Choose the channels of an adjusted common average (least anticorrelation).

    Re-implemented from the published method (Huang et al., 2024, J Neurosci Methods,
    "CARLA"), and checked against the authors' code (spec REQ-372): line noise is removed
    (``line_freq`` and harmonics to 180 Hz; None to skip); channels are ranked by increasing
    mean covariance between pairs of trials in the response window (their evoked activity);
    for ``n = 2 ... N``, ``zeta_n`` is the most negative mean Fisher-z correlation between
    one of the ``n`` least responsive channels (not re-referenced) and the others after
    re-referencing to their average, computed on ``n_boot`` bootstrap averages of the trials
    and averaged over them. Adding a responsive channel imprints its response, negatively,
    on all the others, so ``zeta`` drops. ``sensitive=True`` (the authors' choice on real
    data): starting at 10 % of the channels, the first local maximum of ``zeta`` followed by
    a significant fall (95th percentile of all bootstrap differences trough - peak below 0)
    before the next higher value; else the global maximum.

    Parameters
    ----------
    data : array [n_trials, n_channels, n_times]
    times : 1d array
    response_window : (float, float)
    n_boot : int
    random_state : int or Generator, optional
    candidates : 1d array of int, optional
        Channels allowed in the average (default: all).
    sensitive : bool
    line_freq : float or None
        Line frequency (Hz); the published code notches 60, 120 and 180 Hz.

    Returns
    -------
    dict
        ``members`` (channel indices of the average), ``n_star``, ``order`` (ranking),
        ``zeta`` (array over n = 2..N), ``ci`` (95th percentile of the fall at n*, or None),
        ``responsiveness`` (per channel).
    """
    window = (times >= response_window[0]) & (times <= response_window[1])
    if not window.any():
        raise DataError("response_window contains no sample.")
    sfreq = 1.0 / np.median(np.diff(times))
    candidates = np.arange(data.shape[1]) if candidates is None else np.asarray(candidates)
    clean = _line_notch(data[:, candidates], sfreq, line_freq)[:, :, window]
    resp = _cross_trial_covariance(clean, np.ones(clean.shape[-1], bool))
    local_order = np.argsort(resp, kind="stable")
    order = candidates[local_order]
    n_total = len(order)
    ns = np.arange(2, n_total + 1)
    rng = np.random.default_rng(random_state)
    n_trials = clean.shape[0]
    boots = None
    if n_boot and n_trials >= 3:
        boots = np.empty((n_boot, len(ns)))
        for b in range(n_boot):
            pick = rng.integers(0, n_trials, n_trials)
            boots[b] = _zeta_curve(clean[pick].mean(axis=0), local_order)
        zeta = boots.mean(axis=0)
    else:
        zeta = _zeta_curve(clean.mean(axis=0), local_order)
    n_min = max(2, int(np.ceil(MIN_CA_FRACTION * n_total)))
    chosen = int(ns[n_min - 2 + int(np.argmax(zeta[n_min - 2 :]))])
    ci = None
    if sensitive and boots is not None:
        values = zeta.copy()
        pos = n_min - 2
        while pos < len(ns):
            if pos == len(ns) - 1:
                chosen = int(ns[pos])
                break
            if values[pos + 1] > values[pos]:
                pos += 1
                continue
            values[:pos] = np.nan
            later = np.flatnonzero(values > values[pos])
            if not later.size:
                chosen = int(ns[pos])
                break
            nxt = int(later[0])
            trough = pos + int(np.nanargmin(values[pos:nxt]))
            diffs = (boots[:, trough][None, :] - boots[:, pos][:, None]).ravel()
            upper = float(np.percentile(diffs, 95))
            if upper < 0:
                chosen, ci = int(ns[pos]), (-np.inf, upper)
                break
            pos = nxt
    resp_all = np.full(data.shape[1], np.nan)
    resp_all[candidates] = resp
    return {
        "members": np.sort(order[:chosen]),
        "n_star": int(chosen),
        "order": order,
        "zeta": zeta,
        "ci": ci,
        "responsiveness": resp_all,
    }


###################################################################################################
## Laplacian


def _laplacian(ts, radius):
    info = ts.ch_info
    if info is None or not {"x", "y", "z"} <= set(info.columns):
        raise DataError("laplacian needs channel coordinates (ch_info x, y, z).")
    pos = info[["x", "y", "z"]].to_numpy(dtype=float)
    if np.isnan(pos).all():
        raise DataError("laplacian needs channel coordinates (ch_info x, y, z).")
    dist = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    neighbours = (dist <= radius) & (dist > 0)
    lonely = [ts.ch_names[i] for i in np.flatnonzero(~neighbours.any(axis=1))]
    if lonely:
        _warn(
            f"laplacian: no neighbour within {radius} for {lonely}; these channels "
            "keep their recording reference.",
            UserWarning,
            stacklevel=3,
        )
    weights = neighbours / np.maximum(neighbours.sum(axis=1, keepdims=True), 1)
    values = masked_values(ts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = np.einsum("ij,kjt->kit", weights, np.nan_to_num(values))
    data = ts.data - ref
    labels = ["monopolar" if name in lonely else "laplacian" for name in ts.ch_names]
    return data, labels, {"radius": radius, "lonely": lonely}


###################################################################################################
## Entry point


def rereference(
    ts,
    scheme="bipolar",
    *,
    exclude=None,
    groups="group",
    response_window=(0.01, 0.3),
    n_boot=100,
    radius=12.0,
    random_state=None,
):
    """Change the reference of every channel and record it in ``ch_info["reference"]``.

    - ``"monopolar"``: keep the recording reference.
    - ``"bipolar"``: consecutive contacts of each electrode, ``A1 - A2`` named ``"A1-A2"``
      (the naming and order of ``mne.set_bipolar_reference``); masks are combined.
    - ``"car"``: minus the mean of all channels that are not bad, not stimulating contacts
      of that trial (``metadata["stim_site"]``, ``ch_info["stim_contact"]``), not in
      ``exclude`` and not masked at that sample.
    - ``"car_adaptive"``: minus the mean of the least responsive channels chosen by
      :func:`car_adaptive_select` (falls back to ``"car"`` below 8 channels).
    - ``"laplacian"``: minus the mean of the neighbours within ``radius`` (coordinate
      units of ``ch_info``).

    Parameters
    ----------
    ts : TrialSet
    scheme : str
        One of :data:`SCHEMES`.
    exclude : list of str, optional
        Channels never used in a common average.
    groups : str or dict
        Electrode groups for ``"bipolar"`` (``ch_info`` column or dict name -> group).
    response_window, n_boot, random_state
        For ``"car_adaptive"``.
    radius : float
        For ``"laplacian"``.

    Returns
    -------
    TrialSet
        With the new data and ``ch_info["reference"]``; the provenance step records the
        scheme and, for ``"car_adaptive"``, ``n_star``, ``zeta`` and the channels used.
    """
    start = _time.perf_counter()
    check_trialset(ts)
    require("pandas", extra="data", caller="evoparam.prep.rereference")
    if scheme not in SCHEMES:
        raise SettingsError(f"scheme must be one of {SCHEMES}, not {scheme!r}.")
    settings = {
        "scheme": scheme,
        "exclude": list(exclude or []),
        "groups": groups if not isinstance(groups, dict) else "dict",
        "response_window": list(response_window),
        "n_boot": n_boot,
        "radius": radius,
        "random_state": random_state,
    }
    info = ch_table(ts)
    changes = {}

    if scheme == "monopolar":
        if "reference" not in info.columns:
            info["reference"] = "monopolar"
        info["reference"] = info["reference"].fillna("monopolar")
        changes = {"data": ts.data.copy()}
    elif scheme == "bipolar":
        data, masks, names, info, extra = _bipolar(ts, groups)
        settings.update(extra)
        changes = {"data": data, "masks": masks, "ch_names": names}
    elif scheme == "laplacian":
        data, labels, extra = _laplacian(ts, radius)
        info["reference"] = labels
        settings.update(extra)
        changes = {"data": data}
    else:
        excluded = _excluded(ts, exclude)
        members = ~excluded
        used = scheme
        if scheme == "car_adaptive":
            candidates = np.flatnonzero(~excluded.any(axis=0))
            if len(candidates) < MIN_ADAPTIVE_CHANNELS:
                _warn(
                    f"car_adaptive: only {len(candidates)} usable channels; the "
                    "plain common average is used.",
                    UserWarning,
                    stacklevel=2,
                )
                used = "car"
            else:
                chosen = car_adaptive_select(
                    ts.data,
                    ts.times,
                    response_window=response_window,
                    n_boot=n_boot,
                    random_state=random_state,
                    candidates=candidates,
                )
                in_ca = np.zeros(ts.n_channels, dtype=bool)
                in_ca[chosen["members"]] = True
                members = members & in_ca[None]
                settings.update(
                    n_star=chosen["n_star"],
                    zeta=chosen["zeta"].tolist(),
                    ci=chosen["ci"],
                    members=[ts.ch_names[i] for i in chosen["members"]],
                )
        ref = _common_average(ts, members)
        if np.isnan(ref).all(axis=-1).any():
            raise DataError("common average: a trial has no usable channel.")
        # Samples masked on every channel (for example a blanked artifact window) stay
        # masked; their reference is set to 0
        ref = np.nan_to_num(ref, nan=0.0)
        changes = {"data": ts.data - ref[:, None]}
        info["reference"] = used
        settings["used"] = used

    changes["ch_info"] = info.reset_index(drop=True)
    return new(ts, "evoparam.prep.rereference", settings, start, **changes)
