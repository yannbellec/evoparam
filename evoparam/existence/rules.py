# SPDX-License-Identifier: Apache-2.0
"""Registry of published detection rules, applied to the letter (spec §5.6, method 4)."""

import copy
import json
from importlib import resources

import numpy as np

from evoparam._typing import Any, ArrayLike, IntArray, NDArray
from evoparam.errors import SettingsError

__all__ = ["load_rules", "resolve_rule", "apply_rule", "selective_maxima", "RULE_FIELDS"]

RULE_FIELDS = (
    "statistic",
    "k",
    "window",
    "baseline",
    "polarity",
    "min_duration",
    "sd_floor_uv",
    "peak",
    "peak_selectivity_uv",
    "peak_search_end",
    "exclude_first_sample",
    "saturation_uv",
)
STATISTICS = (
    "negative_peak_over_sd",
    "negative_peak_z",
    "abs_peak_z",
    "abs_peak_z_bandpass",
    "rms_z",
)


def load_rules() -> Any:
    """Return the registry (``rules.json``), as a dict name -> rule."""
    text = resources.files("evoparam.existence").joinpath("rules.json").read_text("utf-8")
    return json.loads(text)["rules"]


def resolve_rule(rule: Any, **overrides: Any) -> Any:
    """Return a complete rule: a registry name or a dict (``"custom"``), overrides applied.

    Raises
    ------
    SettingsError
        Unknown rule, or a field the paper does not state (``unspecified``) and the user did
        not supply; the message names the field, the paper and a hint.
    """
    registry = load_rules()
    if isinstance(rule, dict):
        spec: dict[str, Any] = {
            "statistic": "abs_peak_z",
            "polarity": "both",
            "min_duration": 0.0,
            "sd_floor_uv": None,
            "unspecified": [],
            "source": "custom",
        }
        spec.update(rule)
        name = spec.get("name", "custom")
    elif rule in registry:
        spec, name = copy.deepcopy(registry[rule]), rule
    else:
        raise SettingsError(f"Unknown rule {rule!r}; registered: {sorted(registry)}, or a dict.")
    for key, value in overrides.items():
        if value is not None:
            spec[key] = value
    missing = [f for f in ("k", "window", "baseline") if spec.get(f) is None]
    if missing:
        hint = spec.get("hint", "")
        raise SettingsError(
            f"Rule {name!r} ({spec.get('source', 'custom')}): the field(s) {missing} are not "
            f"stated by the paper and must be supplied (for example "
            f"{missing[0]}=...){'; ' + hint if hint else ''}."
        )
    if spec["statistic"] not in STATISTICS:
        raise SettingsError(f"Rule statistic must be one of {STATISTICS}.")
    if spec.get("peak", "extremum") not in ("extremum", "local", "strict"):
        raise SettingsError("Rule peak must be 'extremum', 'local' or 'strict'.")
    spec["name"] = name
    spec["window"] = tuple(float(v) for v in spec["window"])
    spec["baseline"] = tuple(float(v) for v in spec["baseline"])
    return spec


def selective_maxima(x: ArrayLike, sel: float) -> IntArray:
    """Local maxima that rise at least ``sel`` above the data around them.

    The noise-tolerant peak finder used by the published code of van Blooijs et al. 2023
    (``peakfinder`` by N. C. Yoder, 2015, with the end points included), re-implemented from
    its description: the signal is reduced to its alternating turning points (the first of
    repeated values counts, the end points are candidates), and a maximum is kept when it
    exceeds the lowest point since the previous kept maximum by more than ``sel`` and the
    signal then falls by more than ``sel`` before rising above it.

    Parameters
    ----------
    x : 1d array
    sel : float
        Selectivity, in the unit of ``x`` (0: every turning point that is a maximum).

    Returns
    -------
    index : 1d array of int
        Positions of the maxima in ``x``, in increasing order.
    """
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return np.zeros(0, int)
    slope = np.diff(x)
    slope[slope == 0] = -np.finfo(float).eps
    turns = np.flatnonzero(slope[:-1] * slope[1:] < 0) + 1
    idx = [0] + turns.tolist() + [n - 1]
    val = [x[i] for i in idx]
    floor = min(val)
    if len(val) <= 2:
        best = int(np.argmax(val))
        return np.array([idx[best]] if val[best] > floor + sel else [], int)
    # the end points were added without checking that the extrema still alternate
    first, second = np.sign(val[1] - val[0]), np.sign(val[2] - val[1])
    if first == second:
        drop = 1 if first <= 0 else 0
        del idx[drop], val[drop]
    found, left, best, best_pos = [], floor, floor, None
    peak_open = True
    # walk the alternating sequence: even steps are candidate maxima, odd steps valleys
    i = 0 if val[0] >= val[1] else 1
    m = len(val)
    while i < m:
        if not peak_open:
            best, peak_open = floor, True
        if val[i] > best and val[i] > left + sel:
            best, best_pos = val[i], i
        if i == m - 1:
            break
        valley = val[i + 1]
        if peak_open and best > valley + sel:
            found.append(best_pos)
            left, peak_open = valley, False
        elif valley < left:
            left = valley
        i += 2
    if val[-1] > best and val[-1] > left + sel:
        found.append(m - 1)
    elif peak_open and best > floor and best_pos is not None:
        found.append(best_pos)
    return np.array([idx[j] for j in found], int)


def _local_negative_peak(
    z: Any, times: Any, spec: Any, units: Any, sd: Any
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return the published N1 peak (van Blooijs et al. 2023).

    The largest local minimum of the normalised average between the first sample after
    ``window[0]`` (excluded as a peak when ``exclude_first_sample``) and the first sample after
    ``window[1]``, the minima being searched up to ``peak_search_end``. Returns (statistic, peak
    time) per channel.

    As in the published code, the decision uses the magnitude of the chosen minimum and is
    refused only when every minimum is positive, or when the peak exceeds
    ``saturation_uv`` (data in microvolts).
    """
    start = int(np.argmax(times > spec["window"][0]))
    end = (
        int(np.argmax(times > spec["window"][1]))
        if times[-1] > spec["window"][1]
        else len(times) - 1
    )
    last = spec.get("peak_search_end") or spec["window"][1]
    stop = int(np.argmax(times > last)) if times[-1] > last else len(times) - 1
    in_uv = units == "uV"
    stat = np.full(len(z), -np.inf)
    peak = np.full(len(z), np.nan)
    for c, row in enumerate(z):
        seg = row[start : stop + 1]
        if not np.all(np.isfinite(seg)):
            continue
        sel = (spec.get("peak_selectivity_uv") or 0.0) / sd[c] if in_uv else 0.0
        pos = selective_maxima(-seg, sel) + start
        if spec.get("exclude_first_sample", False):
            pos = pos[pos != start]
        pos = pos[pos <= end]
        if not len(pos):
            continue
        values = row[pos]
        if np.all(values > 0):
            continue
        j = int(np.argmax(np.abs(values)))
        if in_uv and spec.get("saturation_uv") and abs(values[j]) * sd[c] > spec["saturation_uv"]:
            continue
        stat[c] = abs(values[j])
        peak[c] = times[pos[j]]
    return stat, peak


def _strict_negative_peak(z: Any, times: Any, win: Any) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return the deepest negative peak of the window.

    A peak is a sample of the window strictly lower than both of its neighbours (read
    outside the window when the sample is at its edge; the first and last samples of the
    epoch are never peaks) and below the baseline level. Returns (statistic = minus the
    peak's z, peak time) per channel; NaN where the window holds no negative peak.
    """
    z = np.asarray(z, float)
    peak_mask = np.zeros(z.shape, bool)
    peak_mask[:, 1:-1] = (z[:, 1:-1] < z[:, :-2]) & (z[:, 1:-1] < z[:, 2:])
    peak_mask &= win[None, :] & (z < 0)
    values = np.where(peak_mask, z, np.inf)
    idx = np.argmin(values, axis=1)
    found = peak_mask[np.arange(len(z)), idx]
    stat = np.where(found, -values[np.arange(len(z)), idx], np.nan)
    peak = np.where(found, times[idx], np.nan)
    return stat, peak


def _bandpass(avg: Any, sfreq: float, band: Any) -> Any:
    from scipy.signal import butter, sosfiltfilt

    high = min(band[1], 0.45 * sfreq)
    sos = butter(4, [band[0], high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, avg, axis=-1)


def apply_rule(avg: Any, times: Any, sfreq: float, spec: Any, units: str = "uV") -> dict[str, Any]:
    """Apply a complete rule to trial averages [n_channels, n_times].

    Returns dict of arrays [n_channels]: statistic (the rule's quantity, in units of k),
    significant (statistic > k, strictly), onset and offset (both the peak time; NaN for the
    RMS rule); and ``curve`` [n_channels, n_times] (the baseline z-score of the average in
    the window), ``curve_times`` and ``null_band`` [n_channels, n_times, 2] (the level k).

    With ``peak="local"`` (for vanblooijs2023, the reading of the published code) the peak is
    a local minimum found with a selectivity of ``peak_selectivity_uv``, never the window's
    first sample, measured from the baseline median, so that a monotone recovery tail or a
    slow drift through the window edges is not a peak (see :func:`selective_maxima`). The
    default ``peak="extremum"`` takes the window's most negative sample. With
    ``peak="strict"`` the peak is the deepest strict local minimum of the window (lower
    than both neighbours, read outside the window at its edges), measured from the
    baseline mean, with no selectivity and no floor; a window with no negative local
    minimum has no peak (statistic NaN, not significant).
    The SD is the population SD (the published code's ``std`` divides by ``n - 1``: a
    relative difference of ``1/(2n)``, 1e-4 for a 1.9 s baseline at 2 kHz).
    """
    avg = np.asarray(avg, float)
    if spec["statistic"] == "abs_peak_z_bandpass":
        avg = _bandpass(avg, sfreq, spec.get("band", (1.0, 45.0)))
    win = (times >= spec["window"][0]) & (times <= spec["window"][1])
    base = (times >= spec["baseline"][0]) & (times <= spec["baseline"][1])
    if not win.any() or not base.any():
        raise SettingsError(f"Rule {spec['name']!r}: window or baseline outside the epoch.")
    local = spec.get("peak") == "local" and spec["statistic"] in (
        "negative_peak_over_sd",
        "negative_peak_z",
    )
    if local:
        level = np.median(avg[:, base], axis=1, keepdims=True)
    else:
        level = avg[:, base].mean(axis=1, keepdims=True)
    sd = avg[:, base].std(axis=1)
    if spec.get("sd_floor_uv") and units == "uV":
        sd = np.maximum(sd, spec["sd_floor_uv"])
    sd = np.where(sd > 0, sd, np.nan)[:, None]
    z = (avg - level) / sd
    seg = z[:, win]
    tw = times[win]
    stat_kind = spec["statistic"]
    k = float(spec["k"])
    negative = stat_kind in ("negative_peak_over_sd", "negative_peak_z")
    # The rule's quantity over time and the level it must cross (spec §5.13,
    # plot_existence): the baseline z-score of the average, -k for the negative-peak rules,
    # +/-k for the absolute ones, no band for the RMS rule (one value for the window)
    band = np.full(seg.shape + (2,), np.nan)
    if stat_kind != "rms_z":
        band[..., 0] = -k
        if not negative:
            band[..., 1] = k
    curve = {"curve": seg, "curve_times": tw, "null_band": band}
    if spec.get("peak") == "strict" and negative:
        stat, peak = _strict_negative_peak(z, times, win)
        significant = np.nan_to_num(stat, nan=-np.inf) > k
        return {
            "statistic": stat,
            "significant": significant,
            "onset": peak,
            "offset": peak,
            **curve,
        }
    if local:
        stat, peak = _local_negative_peak(z, times, spec, units, sd[:, 0])
        significant = stat > spec["k"]
        stat = np.where(np.isfinite(stat), stat, np.nan)
        return {
            "statistic": stat,
            "significant": significant,
            "onset": peak,
            "offset": peak,
            **curve,
        }
    if stat_kind == "rms_z":
        stat = np.sqrt(np.mean(seg**2, axis=1))
        peak = np.full(len(avg), np.nan)
    else:
        values = -seg if stat_kind in ("negative_peak_over_sd", "negative_peak_z") else np.abs(seg)
        idx = np.nanargmax(values, axis=1)
        stat = values[np.arange(len(avg)), idx]
        peak = tw[idx]
    significant = stat > spec["k"]
    if spec.get("max_peak_delay") is not None:
        significant &= np.nan_to_num(peak, nan=np.inf) <= spec["max_peak_delay"]
    return {"statistic": stat, "significant": significant, "onset": peak, "offset": peak, **curve}
