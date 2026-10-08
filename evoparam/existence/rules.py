"""Registry of published detection rules, applied to the letter (spec §5.6, method 4)."""

import copy
import json
from importlib import resources

import numpy as np

from evoparam.errors import SettingsError

__all__ = ["load_rules", "resolve_rule", "apply_rule", "RULE_FIELDS"]

RULE_FIELDS = ("statistic", "k", "window", "baseline", "polarity", "min_duration", "sd_floor_uv")
STATISTICS = ("negative_peak_over_sd", "negative_peak_z", "abs_peak_z", "abs_peak_z_bandpass",
              "rms_z")


def load_rules():
    """The registry (``rules.json``), as a dict name -> rule."""

    text = resources.files("evoparam.existence").joinpath("rules.json").read_text("utf-8")
    return json.loads(text)["rules"]


def resolve_rule(rule, **overrides):
    """A complete rule: a registry name or a dict (``"custom"``), with the overrides applied.

    Raises
    ------
    SettingsError
        Unknown rule, or a field the paper does not state (``unspecified``) and the user did
        not supply; the message names the field, the paper and a hint.
    """

    registry = load_rules()
    if isinstance(rule, dict):
        spec = {"statistic": "abs_peak_z", "polarity": "both", "min_duration": 0.0,
                "sd_floor_uv": None, "unspecified": [], "source": "custom"}
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
            f"{missing[0]}=...){'; ' + hint if hint else ''}.")
    if spec["statistic"] not in STATISTICS:
        raise SettingsError(f"Rule statistic must be one of {STATISTICS}.")
    spec["name"] = name
    spec["window"] = tuple(float(v) for v in spec["window"])
    spec["baseline"] = tuple(float(v) for v in spec["baseline"])
    return spec


def _bandpass(avg, sfreq, band):
    from scipy.signal import butter, sosfiltfilt

    high = min(band[1], 0.45 * sfreq)
    sos = butter(4, [band[0], high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, avg, axis=-1)


def apply_rule(avg, times, sfreq, spec, units="uV"):
    """Apply a complete rule to trial averages [n_channels, n_times].

    Returns dict of arrays [n_channels]: statistic (the rule's quantity, in units of k),
    significant (statistic > k, strictly), onset and offset (both the peak time; NaN for the
    RMS rule).
    """

    avg = np.asarray(avg, float)
    if spec["statistic"] == "abs_peak_z_bandpass":
        avg = _bandpass(avg, sfreq, spec.get("band", (1.0, 45.0)))
    win = (times >= spec["window"][0]) & (times <= spec["window"][1])
    base = (times >= spec["baseline"][0]) & (times <= spec["baseline"][1])
    if not win.any() or not base.any():
        raise SettingsError(f"Rule {spec['name']!r}: window or baseline outside the epoch.")
    level = avg[:, base].mean(axis=1, keepdims=True)
    sd = avg[:, base].std(axis=1)
    if spec.get("sd_floor_uv") and units == "uV":
        sd = np.maximum(sd, spec["sd_floor_uv"])
    sd = np.where(sd > 0, sd, np.nan)[:, None]
    z = (avg - level) / sd
    seg = z[:, win]
    tw = times[win]
    stat_kind = spec["statistic"]
    if stat_kind == "rms_z":
        stat = np.sqrt(np.mean(seg ** 2, axis=1))
        peak = np.full(len(avg), np.nan)
    else:
        if stat_kind in ("negative_peak_over_sd", "negative_peak_z"):
            values = -seg
        else:
            values = np.abs(seg)
        idx = np.nanargmax(values, axis=1)
        stat = values[np.arange(len(avg)), idx]
        peak = tw[idx]
    significant = stat > spec["k"]
    if spec.get("max_peak_delay") is not None:
        significant &= np.nan_to_num(peak, nan=np.inf) <= spec["max_peak_delay"]
    return {"statistic": stat, "significant": significant, "onset": peak, "offset": peak}
