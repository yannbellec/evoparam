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
from typing import NamedTuple

import numpy as np

from evoparam._typing import Any, FloatArray
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
_MIN_NOISE_SAMPLES = 3  # baseline samples for a noise variance
_MIN_DECAY_COPIES = 3  # copies for a power-law fit of the decay with distance


def _contacts(site: Any) -> list[Any]:
    """Return the contact names of a bipolar site name (``"A1-A2"``: ``A1`` and ``A2``)."""
    return [part.strip() for part in re.split(r"[-–]", str(site)) if part.strip()]


def _shaft_number(name: Any) -> Any:
    """Return the shaft prefix and the contact number of a channel name (None without one)."""
    match = re.match(r"^(.*?)(\d+)$", str(name))
    return (match.group(1), int(match.group(2))) if match else (str(name), None)


def _corr(a: Any, b: Any) -> float:
    """Return the Pearson correlation of two waveforms (NaN when one is flat)."""
    sa, sb = np.std(a), np.std(b)
    if sa == 0 or sb == 0:
        return np.nan
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _rms(a: Any) -> float:
    """Return the root mean square of a waveform."""
    return float(np.sqrt(np.mean(a**2)))


def _counterpart(name: Any, names: Any, scheme: Any) -> Any:
    """Name of the channel that carries ``name`` after re-referencing."""
    if scheme != "bipolar":
        return name if name in names else None
    first = [n for n in names if n.split("-")[0] == name]
    if first:
        return first[0]
    last = [n for n in names if n.split("-")[-1] == name]
    return last[0] if last else None


class _Settings(NamedTuple):
    """The settings of the screens: response window, baseline, reference schemes, decay."""

    window: Any
    baseline: Any
    schemes: Any
    decay_model: Any


class _Screens(NamedTuple):
    """The outcome of the three screens, one value per channel."""

    reversal: FloatArray
    copy_corr: FloatArray
    distance: FloatArray
    decay: Any
    ratio: FloatArray
    corr: FloatArray
    sensitive: Any


def _baselined(ts: Any, baseline: Any) -> Any:
    """Return the trial average of ``ts`` [n_channels, n_times], minus its baseline mean."""
    avg = np.nanmean(ts.data, axis=0)
    if baseline is not None:
        base = (ts.times >= baseline[0]) & (ts.times <= baseline[1])
        if base.any():
            avg = avg - np.nanmean(avg[:, base], axis=1, keepdims=True)
    return avg


def _reliability(times: Any, avg: Any, wave: Any, settings: _Settings) -> Any:
    """Return the share of each channel's variance in the window that is not baseline noise.

    It corrects the correlations for the attenuation by the noise of the average.
    """
    window, baseline = settings.window, settings.baseline
    pre = (
        (times >= baseline[0]) & (times <= baseline[1])
        if baseline is not None
        else times < min(0.0, window[0])
    )
    noise_var = (
        np.nanvar(avg[:, pre], axis=1) if pre.sum() >= _MIN_NOISE_SAMPLES else np.zeros(len(avg))
    )
    return np.clip(1 - noise_var / np.maximum(np.var(wave, axis=1), 1e-300), 0, 1)


def _shaft_positions(names: Any, info: Any) -> dict[Any, tuple[Any, Any]]:
    """Return the shaft and the contact number of each channel (number None when unknown)."""
    groups = {}
    if info is not None and "group" in info.columns:
        groups = {
            str(n): str(g) for n, g in zip(info["name"], info["group"], strict=False) if str(g)
        }
    position = {}
    for n in names:
        prefix, number = _shaft_number(n)
        position[n] = (groups.get(n, prefix), number)
    return position


def _reversal(names: Any, wave: Any, evaluated: Any, info: Any) -> FloatArray:
    """Screen 1: polarity reversal with an adjacent contact of the same shaft.

    1 or 0 for each evaluated channel with an evaluated neighbour, NaN otherwise.
    """
    position = _shaft_positions(names, info)
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
    return reversal


def _site_distance(names: Any, stim: Any, info: Any) -> Any:
    """Return each channel's distance (mm) to the mean position of the site's contacts.

    None when a contact is not a channel or has no finite coordinates.
    """
    if info is None or not {"name", "x", "y", "z"} <= set(info.columns):
        return None
    table = info.set_index(info["name"].astype(str))
    coords = table.reindex(names)[["x", "y", "z"]].to_numpy(float)
    if not (
        all(s in names for s in stim)
        and stim
        and np.isfinite(coords[[names.index(s) for s in stim]]).all()
    ):
        return None
    centre = coords[[names.index(s) for s in stim]].mean(axis=0)
    return np.linalg.norm(coords - centre, axis=1)


def _copy_screen(
    wave: Any, evaluated: Any, reliability: Any, distance: Any, decay_model: Any
) -> tuple[FloatArray, Any]:
    """Screen 2: scaled copy of the waveform next to the site, decaying with distance.

    Returns the correlation of each channel with the source waveform and the decay flags.
    """
    copy_corr = np.full(len(wave), np.nan)
    decay = np.zeros(len(wave), bool)
    candidates = np.flatnonzero(evaluated & np.isfinite(distance))
    nearest = candidates[np.argsort(distance[candidates])][:N_SOURCE]
    if not len(nearest):
        return copy_corr, decay
    # Signs aligned on the nearest channel before averaging
    ref = wave[nearest[0]]
    source = np.mean([wave[j] * (1 if _corr(wave[j], ref) >= 0 else -1) for j in nearest], axis=0)
    norm = np.linalg.norm(source)
    if not norm > 0:
        return copy_corr, decay
    unit = source / norm
    amp = wave @ unit
    for c in np.flatnonzero(evaluated & (reliability >= MIN_RELIABILITY)):
        # Correlation corrected for the attenuation by the channel's noise
        copy_corr[c] = min(_corr(wave[c], source) / np.sqrt(reliability[c]), 1.0)
    copies = evaluated & (np.nan_to_num(copy_corr) >= COPY_CORR) & (amp > 0) & (distance > 0)
    if decay_model != "published":
        return copy_corr, copies.copy()
    return copy_corr, _published_decay(copies, distance, amp)


def _published_decay(copies: Any, distance: Any, amp: Any) -> Any:
    """Return the copies whose amplitude follows the power-law decay fitted on all copies."""
    decay = np.zeros_like(copies)
    if copies.sum() >= _MIN_DECAY_COPIES:
        x, y = np.log(distance[copies]), np.log(amp[copies])
        slope, intercept = np.polyfit(x, y, 1)
        if -slope > DECAY_MIN_EXPONENT:
            fitted = intercept + slope * np.log(np.where(copies, distance, 1))
            resid = np.log(np.where(copies, amp, 1)) - fitted
            decay = copies & (np.abs(resid) <= np.log(DECAY_FACTOR))
    return decay


def _reference_screen(
    sub: Any, wave: Any, evaluated: Any, settings: _Settings
) -> tuple[FloatArray, FloatArray]:
    """Screen 3: sensitivity to the reference scheme.

    Returns, over the schemes, the smallest RMS ratio and the smallest absolute correlation
    between each channel's waveform and its counterpart under another reference.
    """
    names = list(sub.ch_names)
    win = (sub.times >= settings.window[0]) & (sub.times <= settings.window[1])
    ratio = np.full(len(names), np.nan)
    corr = np.full(len(names), np.nan)
    for scheme in settings.schemes:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                other = rereference(sub, scheme, random_state=0)
            except (DataError, SettingsError, ValueError):
                continue
        o_avg = _baselined(other, settings.baseline)
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
    return ratio, corr


def _site_table(
    ts: Any, site: Any, rows: Any, fits: Any, settings: _Settings, info: Any, pd: Any
) -> Any:
    """Return the screening rows of one stimulation site, one per channel."""
    sub = ts.select(trials=list(rows)) if rows is not None else ts
    names = list(sub.ch_names)
    win = (sub.times >= settings.window[0]) & (sub.times <= settings.window[1])
    avg = _baselined(sub, settings.baseline)
    wave = np.nan_to_num(avg[:, win])
    reliability = _reliability(sub.times, avg, wave, settings)
    stim = set(_contacts(site))
    bad = set()
    if info is not None and "bad" in info.columns:
        bad = {str(n) for n, b in zip(info["name"], info["bad"], strict=False) if bool(b)}
    evaluated = np.array([n not in stim and n not in bad for n in names])

    reversal = _reversal(names, wave, evaluated, info)
    distance = _site_distance(names, stim, info)
    if distance is None:
        distance = np.full(len(names), np.nan)
        copy_corr, decay = np.full(len(names), np.nan), np.zeros(len(names), bool)
    else:
        copy_corr, decay = _copy_screen(
            wave, evaluated, reliability, distance, settings.decay_model
        )
    ratio, corr = _reference_screen(sub, wave, evaluated, settings)
    sensitive = (np.nan_to_num(ratio, nan=1.0) < REFERENCE_RATIO) | (
        np.nan_to_num(corr, nan=1.0) < REFERENCE_CORR
    )
    screens = _Screens(reversal, copy_corr, distance, decay, ratio, corr, sensitive)
    return [
        _channel_row(site, n, fits.get(n), bool(evaluated[c]), [x[c] for x in screens], pd)
        for c, n in enumerate(names)
    ]


def _channel_row(
    site: Any, name: Any, fit: Any, evaluated: bool, values: list[Any], pd: Any
) -> dict[str, Any]:
    """Return the screening row of one channel from its values of :class:`_Screens`."""
    reversal, copy_corr, distance, decay, ratio, corr, sensitive = values
    flags = []
    if evaluated:
        if reversal == 0:
            flags.append("no_reversal")
        if decay:
            flags.append("decay")
        if sensitive:
            flags.append("reference")
    risk = RISKS[min(len(flags), 2)] if evaluated else ""
    return {
        "stim_site": site,
        "channel": name,
        "evaluated": bool(evaluated),
        "response": bool(fit is not None and fit.status == "ok"),
        "reversal": pd.NA if np.isnan(reversal) else bool(reversal),
        "no_reversal": bool(reversal == 0),
        "copy_corr": copy_corr,
        "distance_mm": distance,
        "decay": bool(decay and evaluated),
        "reference_ratio": ratio,
        "reference_corr": corr,
        "reference": bool(sensitive and evaluated),
        "n_screens": len(flags),
        "screens": ",".join(flags),
        "volume_conduction_risk": risk,
    }


def screen_volume_conduction(
    ts: Any,
    group_fit: Any = None,
    *,
    reference_schemes: Any = ("bipolar", "car_adaptive"),
    decay_model: str = "published",
    window: Any = None,
    baseline: Any = None,
) -> Any:
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
        groups: list[tuple[str, Any]] = [
            (s, np.flatnonzero(sites == s)) for s in dict.fromkeys(sites)
        ]
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
        settings = _Settings(window, baseline, reference_schemes, decay_model)
        rows += _site_table(ts, site, idx, fits, settings, ts.ch_info, pd)
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
