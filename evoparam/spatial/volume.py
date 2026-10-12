# SPDX-License-Identifier: Apache-2.0
"""Local versus volume-conducted responses (spec §5.9, method 4).

Three published screens, each a flag on every (stimulation site, channel):

1. polarity reversal between adjacent contacts of the same shaft, the dipolar signature of
   a local source; its absence (with adjacent contacts available) is the flag
   ``no_reversal``;
2. ``decay``: the channel's waveform is a scaled copy of the waveform next to the
   stimulation site (zero-lag correlation >= 0.9 once corrected for the attenuation by the
   channel's noise, estimated on the baseline of the average, for channels whose response
   carries at least a quarter of the window's variance; same sign) whose amplitude follows
   the monotone decay with distance fitted over such copies (``log|a| = c - p log r`` with
   ``p > 0.5``; within a factor 2 of the fit), the signature of passive spread;
3. ``reference``: the response loses more than half of its RMS, or its shape (correlation
   < 0.5), under another reference scheme (spec §5.4).

``volume_conduction_risk`` is ``high`` with two flags or more, ``medium`` with one and
``low`` with none; nothing is removed.
"""

import re
import time as _time
import warnings

import numpy as np

from evoparam.data import GroupFit, TrialSet
from evoparam.errors import DataError, SettingsError
from evoparam.models.references import SCHEMES, rereference
from evoparam.utils._optional import require

__all__ = ["screen_volume_conduction", "SCREEN_COLUMNS", "RISKS"]

RISKS = ("low", "medium", "high")
SCREEN_COLUMNS = (
    "stim_site",
    "channel",
    "evaluated",
    "response",
    "reversal",
    "no_reversal",
    "copy_corr",
    "distance_mm",
    "decay",
    "reference_ratio",
    "reference_corr",
    "reference",
    "n_screens",
    "screens",
    "volume_conduction_risk",
)
REVERSAL_CORR = -0.5
REVERSAL_RMS = 0.2
COPY_CORR = 0.9
MIN_RELIABILITY = 0.25
DECAY_MIN_EXPONENT = 0.5
DECAY_FACTOR = 2.0
REFERENCE_RATIO = 0.5
REFERENCE_CORR = 0.5
N_SOURCE = 2


def _contacts(site):
    return [part.strip() for part in re.split(r"[-–]", str(site)) if part.strip()]


def _shaft_number(name):
    match = re.match(r"^(.*?)(\d+)$", str(name))
    return (match.group(1), int(match.group(2))) if match else (str(name), None)


def _corr(a, b):
    sa, sb = np.std(a), np.std(b)
    if sa == 0 or sb == 0:
        return np.nan
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _rms(a):
    return float(np.sqrt(np.mean(a**2)))


def _counterpart(name, names, scheme):
    """Name of the channel that carries ``name`` after re-referencing."""
    if scheme != "bipolar":
        return name if name in names else None
    first = [n for n in names if n.split("-")[0] == name]
    if first:
        return first[0]
    last = [n for n in names if n.split("-")[-1] == name]
    return last[0] if last else None


def _site_table(ts, site, rows, fits, window, baseline, schemes, decay_model, info, pd):
    sub = ts.select(trials=list(rows)) if rows is not None else ts
    names = list(sub.ch_names)
    win = (sub.times >= window[0]) & (sub.times <= window[1])
    avg = np.nanmean(sub.data, axis=0)
    if baseline is not None:
        base = (sub.times >= baseline[0]) & (sub.times <= baseline[1])
        if base.any():
            avg = avg - np.nanmean(avg[:, base], axis=1, keepdims=True)
    wave = np.nan_to_num(avg[:, win])
    # Noise of the average from its baseline, to correct correlations for attenuation
    pre = (
        (sub.times >= baseline[0]) & (sub.times <= baseline[1])
        if baseline is not None
        else sub.times < min(0.0, window[0])
    )
    noise_var = np.nanvar(avg[:, pre], axis=1) if pre.sum() > 2 else np.zeros(len(names))
    reliability = np.clip(1 - noise_var / np.maximum(np.var(wave, axis=1), 1e-300), 0, 1)
    stim = set(_contacts(site))
    bad = set()
    if info is not None and "bad" in info.columns:
        bad = {str(n) for n, b in zip(info["name"], info["bad"]) if bool(b)}
    evaluated = np.array([n not in stim and n not in bad for n in names])

    # Screen 1: polarity reversal with an adjacent contact of the same shaft
    groups = {}
    if info is not None and "group" in info.columns:
        groups = {str(n): str(g) for n, g in zip(info["name"], info["group"]) if str(g)}
    position = {}
    for n in names:
        prefix, number = _shaft_number(n)
        position[n] = (groups.get(n, prefix), number)
    reversal = np.full(len(names), np.nan)
    for c, n in enumerate(names):
        if not evaluated[c] or position[n][1] is None:
            continue
        shaft, number = position[n]
        adjacent = [
            j
            for j, m in enumerate(names)
            if evaluated[j]
            and position[m][0] == shaft
            and position[m][1] is not None
            and abs(position[m][1] - number) == 1
        ]
        if not adjacent:
            continue
        own = _rms(wave[c])
        reversal[c] = float(
            any(
                _corr(wave[c], wave[j]) <= REVERSAL_CORR and _rms(wave[j]) >= REVERSAL_RMS * own
                for j in adjacent
            )
        )

    # Screen 2: scaled copy of the waveform next to the site, decaying with distance
    copy_corr = np.full(len(names), np.nan)
    distance = np.full(len(names), np.nan)
    decay = np.zeros(len(names), bool)
    coords = None
    if info is not None and {"name", "x", "y", "z"} <= set(info.columns):
        table = info.set_index(info["name"].astype(str))
        coords = table.reindex(names)[["x", "y", "z"]].to_numpy(float)
    if (
        coords is not None
        and all(s in names for s in stim)
        and stim
        and np.isfinite(coords[[names.index(s) for s in stim]]).all()
    ):
        centre = coords[[names.index(s) for s in stim]].mean(axis=0)
        distance = np.linalg.norm(coords - centre, axis=1)
        candidates = np.flatnonzero(evaluated & np.isfinite(distance))
        nearest = candidates[np.argsort(distance[candidates])][:N_SOURCE]
        if len(nearest):
            # Signs aligned on the nearest channel before averaging
            ref = wave[nearest[0]]
            source = np.mean(
                [wave[j] * (1 if _corr(wave[j], ref) >= 0 else -1) for j in nearest], axis=0
            )
            norm = np.linalg.norm(source)
            if norm > 0:
                unit = source / norm
                amp = wave @ unit
                for c in np.flatnonzero(evaluated & (reliability >= MIN_RELIABILITY)):
                    # Correlation corrected for the attenuation by the channel's noise
                    copy_corr[c] = min(_corr(wave[c], source) / np.sqrt(reliability[c]), 1.0)
                copies = (
                    evaluated
                    & (np.nan_to_num(copy_corr) >= COPY_CORR)
                    & (amp > 0)
                    & (distance > 0)
                )
                decay = copies.copy()
                if decay_model == "published":
                    decay[:] = False
                    if copies.sum() >= 3:
                        x, y = np.log(distance[copies]), np.log(amp[copies])
                        slope, intercept = np.polyfit(x, y, 1)
                        if -slope > DECAY_MIN_EXPONENT:
                            fitted = intercept + slope * np.log(np.where(copies, distance, 1))
                            resid = np.log(np.where(copies, amp, 1)) - fitted
                            decay = copies & (np.abs(resid) <= np.log(DECAY_FACTOR))

    # Screen 3: sensitivity to the reference scheme
    ratio = np.full(len(names), np.nan)
    corr = np.full(len(names), np.nan)
    for scheme in schemes:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                other = rereference(sub, scheme, random_state=0)
            except (DataError, SettingsError, ValueError):
                continue
        o_avg = np.nanmean(other.data, axis=0)
        if baseline is not None:
            base = (other.times >= baseline[0]) & (other.times <= baseline[1])
            if base.any():
                o_avg = o_avg - np.nanmean(o_avg[:, base], axis=1, keepdims=True)
        o_names = list(other.ch_names)
        for c, n in enumerate(names):
            if not evaluated[c]:
                continue
            match = _counterpart(n, o_names, scheme)
            if match is None:
                continue
            w = np.nan_to_num(o_avg[o_names.index(match), win])
            own = _rms(wave[c])
            r = _rms(w) / own if own > 0 else np.nan
            k = abs(_corr(w, wave[c])) if np.std(w) > 0 else 0.0
            ratio[c] = r if np.isnan(ratio[c]) else min(ratio[c], r)
            corr[c] = k if np.isnan(corr[c]) else min(corr[c], k)
    sensitive = (np.nan_to_num(ratio, nan=1.0) < REFERENCE_RATIO) | (
        np.nan_to_num(corr, nan=1.0) < REFERENCE_CORR
    )

    out = []
    for c, n in enumerate(names):
        fit = fits.get(n)
        flags = []
        if evaluated[c]:
            if reversal[c] == 0:
                flags.append("no_reversal")
            if decay[c]:
                flags.append("decay")
            if sensitive[c]:
                flags.append("reference")
        risk = RISKS[min(len(flags), 2)] if evaluated[c] else ""
        out.append(
            {
                "stim_site": site,
                "channel": n,
                "evaluated": bool(evaluated[c]),
                "response": bool(fit is not None and fit.status == "ok"),
                "reversal": pd.NA if np.isnan(reversal[c]) else bool(reversal[c]),
                "no_reversal": bool(reversal[c] == 0),
                "copy_corr": copy_corr[c],
                "distance_mm": distance[c],
                "decay": bool(decay[c] and evaluated[c]),
                "reference_ratio": ratio[c],
                "reference_corr": corr[c],
                "reference": bool(sensitive[c] and evaluated[c]),
                "n_screens": len(flags),
                "screens": ",".join(flags),
                "volume_conduction_risk": risk,
            }
        )
    return out


def screen_volume_conduction(
    ts,
    group_fit=None,
    *,
    reference_schemes=("bipolar", "car_adaptive"),
    decay_model="published",
    window=None,
    baseline=None,
):
    """Flag channels whose response may be volume-conducted rather than local.

    For every stimulation site of ``ts`` (``metadata["stim_site"]``; the whole TrialSet
    when there is none) and every channel that is neither a stimulating contact of that
    site nor bad, the three screens of the module docstring are computed on the trial
    average in the response window (``window``, default the fit window of ``group_fit``,
    else 9 to 500 ms; ``baseline`` subtracted when given, default the fit's baseline).
    ``group_fit`` (conditions = stimulation sites, as from ``evoparam.fit(...,
    by="stim_site")``) gives the ``response`` column. Shafts come from ``ch_info["group"]``
    or the channel name without its trailing number; the site's position is the mean of
    its contacts' coordinates.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`SCREEN_COLUMNS`; ``screens`` lists the triggered flags and
        ``volume_conduction_risk`` is ``"low"``, ``"medium"`` or ``"high"`` (empty for the
        channels not evaluated).
    """
    pd = require("pandas", extra="data", caller="evoparam.spatial.screen_volume_conduction")
    start = _time.perf_counter()
    if not isinstance(ts, TrialSet):
        raise DataError("screen_volume_conduction expects a TrialSet.")
    if decay_model not in ("published", "none"):
        raise SettingsError("decay_model must be 'published' or 'none'.")
    unknown = set(reference_schemes) - set(SCHEMES)
    if unknown:
        raise SettingsError(f"Unknown reference schemes {sorted(unknown)}.")
    if group_fit is not None and not isinstance(group_fit, GroupFit):
        raise DataError("group_fit must be a GroupFit.")
    settings0 = group_fit.settings if group_fit is not None else {}
    if window is None:
        window = tuple(settings0.get("fit_window") or (0.009, 0.5))
    if baseline is None and settings0.get("baseline"):
        baseline = tuple(settings0["baseline"])
    meta = ts.metadata
    if meta is not None and "stim_site" in meta.columns:
        sites = meta["stim_site"].astype(str).to_numpy()
        groups = [(s, np.flatnonzero(sites == s)) for s in dict.fromkeys(sites)]
    else:
        groups = [("", None)]
    rows = []
    for site, idx in groups:
        fits = {}
        if group_fit is not None:
            for ch, cond, pos in group_fit.index[["channel", "condition", "position"]].itertuples(
                index=False
            ):
                if str(cond) in (site, ""):
                    fits[str(ch)] = group_fit.fits[int(pos)]
        rows += _site_table(
            ts, site, idx, fits, window, baseline, reference_schemes, decay_model, ts.ch_info, pd
        )
    table = pd.DataFrame(rows, columns=list(SCREEN_COLUMNS))
    table["reversal"] = table["reversal"].astype("boolean")
    table.attrs["settings"] = {
        "reference_schemes": list(reference_schemes),
        "decay_model": decay_model,
        "window": list(window),
        "baseline": None if baseline is None else list(baseline),
    }
    table.attrs["duration_s"] = _time.perf_counter() - start
    return table
