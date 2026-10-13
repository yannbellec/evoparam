# SPDX-License-Identifier: Apache-2.0
"""Scorers: match fitted components to the ground truth and summarise the errors.

Components are matched per waveform (channel, and condition when the fits have
one) by the Hungarian algorithm on the latency difference, a pair being allowed
only when the difference is within a tolerance (default half the FWHM of the
true component). The match table then gives recall, precision, F1, latency,
amplitude and width errors, and the coverage of reported intervals. The tests
and the benchmark (spec §8) use these functions.
"""

import re
from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from evoparam._typing import Any
from evoparam.data import GroupFit, ResponseFit
from evoparam.errors import SettingsError
from evoparam.utils._optional import require

from ._truth import GroundTruth

__all__ = ["match_components", "score_fits", "MATCH_COLUMNS", "SCORE_COLUMNS"]

# Columns of match_components() and score_fits(), with their units.
MATCH_COLUMNS = {
    "channel": "",
    "condition": "",
    "kind": "",
    "truth_component": "",
    "fit_component": "",
    "label": "",
    "role": "",
    "latency_true": "s",
    "latency_fit": "s",
    "latency_error": "s",
    "amplitude_true": "data",
    "amplitude_fit": "data",
    "amplitude_error": "data",
    "amplitude_rel_error": "",
    "fwhm_true": "s",
    "fwhm_fit": "s",
    "fwhm_error": "s",
    "fwhm_rel_error": "",
    "polarity_true": "",
    "polarity_fit": "",
    "latency_covered": "",
    "amplitude_covered": "",
    "fwhm_covered": "",
}
SCORE_COLUMNS = {
    "n_waveforms": "",
    "n_true": "",
    "n_fit": "",
    "n_matched": "",
    "n_false": "",
    "recall": "",
    "precision": "",
    "f1": "",
    "latency_mae": "s",
    "latency_bias": "s",
    "latency_mae_samples": "",
    "amplitude_rel_bias": "",
    "amplitude_rel_mae": "",
    "fwhm_rel_mae": "",
    "coverage_latency": "",
    "coverage_amplitude": "",
    "coverage_fwhm": "",
    "false_per_waveform": "",
    "false_rate_empty": "",
}
_COVERED = ("latency", "amplitude", "fwhm")
# Cost of a forbidden pair: larger than any sum of allowed latency differences (s).
_FORBIDDEN = 1e9
# rounding slack on the latency tolerance of a match (s)
_TOLERANCE_SLACK = 1e-12


###################################################################################################
## Inputs


def _tolerance(tolerance: Any, fwhm: Any) -> Any:
    """Per-component tolerance (s) from "fwhm/k", "k*fwhm", a number or None."""
    fwhm = np.asarray(fwhm, dtype=float)
    if tolerance is None:
        return np.full(fwhm.shape, np.inf)
    if isinstance(tolerance, str):
        text = tolerance.replace(" ", "").lower()
        match = re.fullmatch(r"fwhm/([0-9.]+)", text)
        if match:
            return np.nan_to_num(fwhm / float(match.group(1)), nan=0.0)
        match = re.fullmatch(r"([0-9.]+)\*fwhm", text)
        if match:
            return np.nan_to_num(fwhm * float(match.group(1)), nan=0.0)
        raise SettingsError(
            f"tolerance {tolerance!r}: use 'fwhm/<k>', '<k>*fwhm', a number of seconds or None."
        )
    value = float(tolerance)
    if not value >= 0:
        raise SettingsError(f"tolerance must be >= 0 s, got {tolerance}.")
    return np.full(fwhm.shape, value)


def _truth_rows(truth: Any, level: Any, window: Any) -> tuple[Any, Any, Any, Any]:
    """Return the truth table, its sampling rate, the window and the truth's waveforms.

    The window defaults to the epoch of a GroundTruth; the waveforms are those of a
    GroundTruth (every channel and condition), None for a table.
    """
    if isinstance(truth, GroundTruth):
        if window is None:
            window = (float(truth.times[0]), float(truth.times[-1]))
        conditions = list(dict.fromkeys(truth.conditions))
        waveforms = [(ch, cond) for ch in truth.ch_names for cond in conditions]
        return truth.to_dataframe(level), truth.sfreq, window, waveforms
    if type(truth).__name__ == "DataFrame":
        return truth.copy(), truth.attrs.get("sfreq"), window, None
    raise SettingsError(f"truth must be a GroundTruth or a DataFrame, got {type(truth).__name__}.")


def _complete_truth_columns(table: Any) -> None:
    """Check the required columns of a truth table and add the optional ones it lacks."""
    for column in ("channel", "latency", "amplitude", "fwhm"):
        if column not in table.columns:
            raise SettingsError(f"truth table has no column '{column}'.")
    if "condition" not in table.columns:
        table["condition"] = ""
    if "component" not in table.columns:
        table["component"] = np.arange(len(table))
    if "role" not in table.columns:
        table["role"] = "target"
    if "label" not in table.columns:
        table["label"] = ""
    if "polarity" not in table.columns:
        table["polarity"] = np.sign(table["amplitude"]).astype(int)


def _truth_table(
    truth: Any, level: Any, roles: Any, window: Any
) -> tuple[Any, Any, list[tuple[Any, Any]], Any]:
    """Truth rows, the waveforms they define, and the sampling rate when known."""
    pd = require("pandas", extra="data", caller="evoparam.sim.match_components")
    table, sfreq, window, waveforms = _truth_rows(truth, level, window)
    _complete_truth_columns(table)
    table = table[np.isfinite(table["latency"].to_numpy(dtype=float))]
    if roles is not None:
        roles = (roles,) if isinstance(roles, str) else tuple(roles)
        table = table[table["role"].isin(roles)]
    if window is not None:
        latency = table["latency"].to_numpy(dtype=float)
        table = table[(latency >= window[0]) & (latency <= window[1])]
    table = table.reset_index(drop=True)
    table["condition"] = table["condition"].fillna("").astype(str)
    table["channel"] = table["channel"].astype(str)
    if waveforms is None:
        waveforms = list(dict.fromkeys(zip(table["channel"], table["condition"], strict=False)))
    return pd, table, waveforms, sfreq


def _component_rows(fit: Any, channel: Any, condition: Any) -> list[Any]:
    """Return one row per fitted component of ``fit``, with its interval columns."""
    rows = []
    for position, comp in enumerate(fit.components):
        row = {
            "channel": str(channel),
            "condition": str(condition),
            "component": position,
            "label": comp.label or "",
            "latency": comp.latency,
            "amplitude": comp.amplitude,
            "fwhm": comp.fwhm,
            "polarity": comp.polarity,
        }
        for name in _COVERED:
            if comp.ci is not None and name in comp.ci:
                row[f"{name}_ci_low"], row[f"{name}_ci_high"] = comp.ci[name]
        rows.append(row)
    return rows


def _fit_table(
    pd: Any, fit: Any, channel: Any, condition: Any, truth_waveforms: Any
) -> tuple[Any, list[tuple[Any, Any]]]:
    """Fitted components (with interval columns when available) and the fitted waveforms."""
    if isinstance(fit, GroupFit):
        rows, waveforms = [], []
        for ch, cond, pos in fit.index[["channel", "condition", "position"]].itertuples(
            index=False
        ):
            waveforms.append((str(ch), str(cond)))
            rows.extend(_component_rows(fit.fits[int(pos)], ch, cond))
        table = pd.DataFrame(rows)
    elif isinstance(fit, ResponseFit):
        if channel is None:
            channels = {ch for ch, _ in truth_waveforms}
            if len(channels) != 1:
                raise SettingsError(
                    "match_components: name the channel of a ResponseFit "
                    "(channel=...) when the truth has several channels."
                )
            channel = channels.pop()
        condition = "" if condition is None else condition
        table = pd.DataFrame(_component_rows(fit, channel, condition))
        waveforms = [(str(channel), str(condition))]
    elif type(fit).__name__ == "DataFrame":
        table = fit.copy()
        if "condition" not in table.columns:
            table["condition"] = ""
        table["condition"] = table["condition"].fillna("").astype(str)
        table["channel"] = table["channel"].astype(str)
        waveforms = list(dict.fromkeys(zip(table["channel"], table["condition"], strict=False)))
    else:
        raise SettingsError(
            f"fit must be a GroupFit, a ResponseFit or a DataFrame, got {type(fit).__name__}."
        )
    columns = [
        "channel",
        "condition",
        "component",
        "label",
        "latency",
        "amplitude",
        "fwhm",
        "polarity",
    ]
    for column in columns:
        if column not in table.columns:
            table[column] = {"label": "", "component": -1}.get(column, np.nan)
    if len(table):
        table = table[np.isfinite(table["latency"].to_numpy(dtype=float))]
    if "component" in table.columns and len(table):
        table = table[table["component"].astype(int) >= 0]
    return table.reset_index(drop=True), waveforms


###################################################################################################
## Matching


def _covered(row: Any, name: Any, true_value: Any) -> float:
    """Return 1.0 when the interval of ``name`` holds the true value, 0.0 when not, else NaN."""
    low, high = row.get(f"{name}_ci_low"), row.get(f"{name}_ci_high")
    if low is None or high is None or not (np.isfinite(low) and np.isfinite(high)):
        return np.nan
    return float(low <= true_value <= high)


def _pair(true_row: Any, fit_row: Any, key: Any) -> dict[str, Any]:
    """Return the match-table row of a true and a fitted component (either may be None)."""
    out = {"channel": key[0], "condition": key[1]}
    if true_row is not None:
        out.update(
            truth_component=int(true_row["component"]),
            label=true_row["label"],
            role=true_row["role"],
            latency_true=float(true_row["latency"]),
            amplitude_true=float(true_row["amplitude"]),
            fwhm_true=float(true_row["fwhm"]),
            polarity_true=int(true_row["polarity"]),
        )
    if fit_row is not None:
        out.update(
            fit_component=int(fit_row["component"]),
            latency_fit=float(fit_row["latency"]),
            amplitude_fit=float(fit_row["amplitude"]),
            fwhm_fit=float(fit_row["fwhm"]),
            polarity_fit=int(fit_row["polarity"]) if np.isfinite(fit_row["polarity"]) else 0,
        )
        if not out.get("label"):
            out["label"] = fit_row["label"]
    if true_row is not None and fit_row is not None:
        out["kind"] = "match"
        for name in ("latency", "amplitude", "fwhm"):
            error = out[f"{name}_fit"] - out[f"{name}_true"]
            out[f"{name}_error"] = error
            if name != "latency":
                true_value = out[f"{name}_true"]
                out[f"{name}_rel_error"] = error / true_value if true_value != 0 else np.nan
            out[f"{name}_covered"] = _covered(fit_row, name, out[f"{name}_true"])
    else:
        out["kind"] = "miss" if true_row is not None else "false"
    return out


def _waveform_key(channel_name: Any, cond: Any, by_condition: bool) -> tuple[Any, Any]:
    """Return the waveform key of a channel and condition (the condition only ``by_condition``)."""
    return (channel_name, cond if by_condition else "")


def _with_waveform_condition(table: Any, by_condition: bool) -> Any:
    """Return ``table`` with its condition column replaced by that of the waveform key."""
    return table.assign(
        condition=[
            _waveform_key(c, d, by_condition)[1]
            for c, d in zip(table["channel"], table["condition"], strict=False)
        ]
    )


def _check_levels(true_table: Any, fit_table: Any, by_condition: bool, level: str) -> bool:
    """Check that the fits can be compared with the truth; return whether trials are matched."""
    if not by_condition and len(set(true_table["condition"])) > 1:
        raise SettingsError(
            "match_components: the truth has several conditions but the fits "
            "have none; select the truth rows of one condition (or fit per "
            "condition)."
        )
    trial_level = level == "trial" and "trial" in true_table.columns
    if trial_level and "trial" not in fit_table.columns:
        raise SettingsError("match_components: level='trial' needs a 'trial' column in the fits.")
    return trial_level


def _match_waveform(
    t_rows: Any, f_rows: Any, limits: Any, match_polarity: bool, wave: tuple[Any, Any]
) -> list[dict[str, Any]]:
    """Return the matched pairs, then the missed true and the false fitted components.

    The pairs minimise the summed latency difference (Hungarian assignment) among the
    pairs within the tolerance ``limits`` (and of the same polarity with
    ``match_polarity``).
    """
    t_lat = t_rows["latency"].to_numpy(dtype=float)
    f_lat = f_rows["latency"].to_numpy(dtype=float)
    matched_t, matched_f = set(), set()
    group_rows = []
    if len(t_rows) and len(f_rows):
        diff = np.abs(t_lat[:, None] - f_lat[None, :])
        allowed = diff <= limits[:, None] + _TOLERANCE_SLACK
        if match_polarity:
            allowed &= (
                np.sign(t_rows["amplitude"].to_numpy(dtype=float))[:, None]
                == np.sign(f_rows["amplitude"].to_numpy(dtype=float))[None, :]
            )
        cost = np.where(allowed, diff, _FORBIDDEN)
        for i, j in zip(*linear_sum_assignment(cost), strict=False):
            if allowed[i, j]:
                matched_t.add(i)
                matched_f.add(j)
                group_rows.append(_pair(t_rows.iloc[i], f_rows.iloc[j].to_dict(), wave))
    for i in range(len(t_rows)):
        if i not in matched_t:
            group_rows.append(_pair(t_rows.iloc[i], None, wave))
    for j in range(len(f_rows)):
        if j not in matched_f:
            group_rows.append(_pair(None, f_rows.iloc[j].to_dict(), wave))
    return group_rows


def match_components(  # noqa: PLR0913  (public keywords; DECISIONS.md §7.2)
    truth: Any,
    fit: Any,
    tolerance: str | float | None = "fwhm/2",
    *,
    level: str = "component",
    roles: Any = None,
    window: tuple[float, float] | None = None,
    match_polarity: bool = False,
    channel: str | None = None,
    condition: str | None = None,
) -> Any:
    """Match fitted components to true components, one waveform at a time.

    Parameters
    ----------
    truth : GroundTruth or pandas.DataFrame
        The truth, or a table with at least ``channel``, ``latency``, ``amplitude``
        and ``fwhm`` (the schema of :meth:`GroundTruth.to_dataframe`).
    fit : GroupFit, ResponseFit or pandas.DataFrame
        Fitted components; a table in the schema of ``GroupFit.to_dataframe()``
        (rows with ``component == -1`` or a NaN latency mark a waveform without
        component). Intervals are read from ``Component.ci`` (keys ``latency``,
        ``amplitude``, ``fwhm``) or from ``<name>_ci_low`` / ``<name>_ci_high`` columns.
    tolerance : str, float or None
        Largest latency difference of a match: ``"fwhm/2"`` (half the FWHM of the
        true component, default), ``"fwhm/<k>"``, ``"<k>*fwhm"``, seconds, or None.
    level : {"component", "trial"}
        Level of the truth table (see :meth:`GroundTruth.to_dataframe`); fits of
        averages are compared with ``"component"``.
    roles : str or sequence of str, optional
        Keep only true components with these roles (default: all).
    window : (float, float), optional
        Keep only true components whose latency lies in this window (s); default
        the epoch of a GroundTruth.
    match_polarity : bool
        Allow only pairs of the same polarity.
    channel, condition : str, optional
        Waveform of a single ResponseFit.

    Returns
    -------
    pandas.DataFrame
        One row per matched pair (``kind == "match"``), per missed true component
        (``"miss"``) and per false fitted component (``"false"``), with the columns of
        :data:`MATCH_COLUMNS` (errors are fit minus truth; relative errors divide by
        the true value, so a fitted amplitude larger in magnitude than the truth gives a
        positive relative error whatever the polarity; ``*_covered`` is 1 when the
        interval contains the truth, 0 when not, NaN without interval).
        ``table.attrs["waveforms"]`` lists the (channel, condition) waveforms compared
        and ``attrs["sfreq"]`` the sampling rate when known.

    Examples
    --------
    >>> import pandas as pd
    >>> truth = pd.DataFrame({"channel": ["c"], "latency": [0.1], "amplitude": [-5.0],
    ...                       "fwhm": [0.02]})
    >>> fit = pd.DataFrame({"channel": ["c", "c"], "component": [0, 1],
    ...                     "latency": [0.104, 0.3], "amplitude": [-4.0, 1.0],
    ...                     "fwhm": [0.022, 0.05], "polarity": [-1, 1]})
    >>> list(match_components(truth, fit)["kind"])
    ['match', 'false']
    """
    if level not in ("component", "trial"):
        raise SettingsError(f"level must be 'component' or 'trial', got {level!r}.")
    pd, true_table, truth_waveforms, sfreq = _truth_table(truth, level, roles, window)
    fit_table, fit_waveforms = _fit_table(pd, fit, channel, condition, truth_waveforms)
    by_condition = any(cond for _, cond in fit_waveforms)
    trial_level = _check_levels(true_table, fit_table, by_condition, level)
    waveforms = list(
        dict.fromkeys(
            [_waveform_key(*w, by_condition) for w in truth_waveforms]
            + [_waveform_key(*w, by_condition) for w in fit_waveforms]
        )
    )
    group_cols = ["channel", "condition"] + (["trial"] if trial_level else [])
    true_table = _with_waveform_condition(true_table, by_condition)
    fit_table = _with_waveform_condition(fit_table, by_condition)
    true_groups = (
        dict(tuple(true_table.groupby(group_cols, sort=False))) if len(true_table) else {}
    )
    fit_groups = dict(tuple(fit_table.groupby(group_cols, sort=False))) if len(fit_table) else {}
    rows = []
    for key in dict.fromkeys(list(true_groups) + list(fit_groups)):
        t_rows = true_groups.get(key, true_table.iloc[0:0])
        f_rows = fit_groups.get(key, fit_table.iloc[0:0])
        limits = _tolerance(tolerance, t_rows["fwhm"].to_numpy(dtype=float))
        group_rows = _match_waveform(t_rows, f_rows, limits, match_polarity, (key[0], key[1]))
        if trial_level:
            for row in group_rows:
                row["trial"] = int(key[2])
        rows.extend(group_rows)
    columns = list(MATCH_COLUMNS) + (["trial"] if trial_level else [])
    table = pd.DataFrame(rows, columns=columns)
    for column in ("truth_component", "fit_component", "polarity_true", "polarity_fit"):
        table[column] = table[column].astype("Int64")
    table.attrs["units"] = dict(MATCH_COLUMNS)
    table.attrs["waveforms"] = waveforms
    table.attrs["sfreq"] = sfreq
    table.attrs["true_waveforms"] = sorted(
        {(c, d) for c, d in zip(true_table["channel"], true_table["condition"], strict=False)}
    )
    return table


###################################################################################################
## Summary


def _summary(group: Any, waveforms: Any, true_waveforms: Any, sfreq: float) -> dict[str, Any]:
    """Return the scores of one group of the match table (see :func:`score_fits`)."""
    kinds = group["kind"].to_numpy()
    matched = group[kinds == "match"]
    n_matched = int(len(matched))
    n_true = int(np.sum(kinds != "false"))
    n_fit = int(np.sum(kinds != "miss"))
    n_false = n_fit - n_matched
    empty = [w for w in waveforms if w not in true_waveforms]
    false_empty = int(
        np.sum(
            [
                (c, d) in empty
                for c, d, k in zip(group["channel"], group["condition"], kinds, strict=False)
                if k == "false"
            ]
        )
    )

    def mean(values: Any) -> Any:
        """Return the mean of the finite values (NaN when there are none)."""
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else np.nan

    latency_error = matched["latency_error"].to_numpy(dtype=float)
    out = {
        "n_waveforms": len(waveforms),
        "n_true": n_true,
        "n_fit": n_fit,
        "n_matched": n_matched,
        "n_false": n_false,
        "recall": n_matched / n_true if n_true else np.nan,
        "precision": n_matched / n_fit if n_fit else np.nan,
        "f1": 2.0 * n_matched / (n_true + n_fit) if (n_true + n_fit) else np.nan,
        "latency_mae": mean(np.abs(latency_error)),
        "latency_bias": mean(latency_error),
        "latency_mae_samples": mean(np.abs(latency_error)) * sfreq if sfreq else np.nan,
        "amplitude_rel_bias": mean(matched["amplitude_rel_error"]),
        "amplitude_rel_mae": mean(np.abs(matched["amplitude_rel_error"].to_numpy(float))),
        "fwhm_rel_mae": mean(np.abs(matched["fwhm_rel_error"].to_numpy(float))),
        "false_per_waveform": n_false / len(waveforms) if waveforms else np.nan,
        "false_rate_empty": false_empty / len(empty) if empty else np.nan,
    }
    for name in _COVERED:
        out[f"coverage_{name}"] = mean(matched[f"{name}_covered"])
    return out


def score_fits(
    truth: Any,
    fits: Any,
    *,
    tolerance: str = "fwhm/2",
    by: str | Sequence[str] | None = None,
    level: str = "component",
    roles: Any = None,
    window: Any = None,
    match_polarity: bool = False,
) -> Any:
    """Recall, precision, F1, parameter errors and interval coverage of fits against the truth.

    Parameters
    ----------
    truth : GroundTruth or pandas.DataFrame
    fits : GroupFit, ResponseFit or pandas.DataFrame
    tolerance, level, roles, window, match_polarity
        As in :func:`match_components`.
    by : str or list of str, optional
        Columns of the match table to group by (for example ``"channel"`` or
        ``"label"``); one overall row when None.

    Returns
    -------
    pandas.DataFrame
        Columns of :data:`SCORE_COLUMNS`: counts, ``recall``, ``precision``, ``f1``
        (``2 n_matched / (n_true + n_fit)``), ``latency_mae`` and ``latency_bias`` (s),
        ``latency_mae_samples``, ``amplitude_rel_bias`` and ``amplitude_rel_mae``,
        ``fwhm_rel_mae``, the coverage of the latency, amplitude and FWHM intervals
        (fraction of matched components whose interval holds the truth),
        ``false_per_waveform`` and ``false_rate_empty`` (false components per
        waveform without a true component, the false-component rate at K = 0).
    """
    table = match_components(
        truth,
        fits,
        tolerance,
        level=level,
        roles=roles,
        window=window,
        match_polarity=match_polarity,
    )
    pd = require("pandas", extra="data", caller="evoparam.sim.score_fits")
    waveforms = table.attrs["waveforms"]
    true_waveforms = set(table.attrs["true_waveforms"])
    sfreq = table.attrs["sfreq"]
    if by is None:
        rows = [_summary(table, waveforms, true_waveforms, sfreq)]
        out = pd.DataFrame(rows, columns=list(SCORE_COLUMNS))
    else:
        by = [by] if isinstance(by, str) else list(by)
        rows = []
        for raw_key, group in table.groupby(by, sort=False, dropna=False):
            key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            subset = waveforms
            if by == ["channel"]:
                subset = [w for w in waveforms if w[0] == key[0]]
            row = dict(zip(by, key, strict=False))
            row.update(_summary(group, subset, true_waveforms, sfreq))
            rows.append(row)
        out = pd.DataFrame(rows, columns=by + list(SCORE_COLUMNS))
    out.attrs["units"] = dict(SCORE_COLUMNS)
    return out
