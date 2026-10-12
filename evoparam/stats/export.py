# SPDX-License-Identifier: Apache-2.0
"""Tidy exports for mixed models, model formulas and group summaries (spec §5.11, 6 and 7).

``to_long`` turns any result into one row per (trial x) channel x component x parameter,
with the parameter's standard error, interval and weight ``1 / se^2`` next to every
identifying column; ``column_dictionary`` describes every column with its unit (the JSON
sidecar of a BIDS derivative table). ``mixed_formula`` writes the model formula for
``statsmodels``, ``lme4`` or ``brms``; ``fit_mixed`` fits the linear mixed model of
``statsmodels`` (extra ``stats``). ``group_summary`` summarises parameters per group and
counts the fit status of every channel, so that non-responses are never averaged away.
"""

import json
import keyword
import re
import time as _time

import numpy as np

from evoparam.data import GroupFit, HierarchicalFit, Provenance, ResponseFit, TFFit, TrialFit
from evoparam.data.results import COMPONENT_COLUMNS, EXISTENCE_COLUMNS, FIT_STATUSES
from evoparam.errors import DataError, SettingsError
from evoparam.utils._optional import require

from .bootstrap import bootstrap_interval

__all__ = [
    "to_long",
    "column_dictionary",
    "mixed_formula",
    "fit_mixed",
    "group_summary",
    "PARAMETERS",
    "LONG_COLUMNS",
]

# Component parameters exported, with their units ("data": the amplitude unit).
PARAMETERS = {
    "latency": "s",
    "amplitude": "data",
    "fwhm": "s",
    "onset": "s",
    "offset": "s",
    "area": "data·s",
    "rise_time": "s",
    "decay_time": "s",
    "symmetry": "",
    "sharpness": "",
}

# Trial-level and other parameters that may appear in TrialFit, HierarchicalFit and TFFit
# tables, with their units.
OTHER_PARAMETERS = {
    "shift": "s",
    "scale": "",
    "r2": "",
    "gof": "",
    "jitter_sd": "s",
    "duration": "s",
    "peak_value": "",
    "zone_metric": "",
    "metric": "",
    "power": "dB",
    "p_response": "",
}

# The value columns of the long format (identifying columns come before them).
LONG_COLUMNS = {
    "parameter": "",
    "value": "unit",
    "unit": "",
    "se": "unit",
    "ci_low": "unit",
    "ci_high": "unit",
    "weight": "unit^-2",
}

# Descriptions of the columns the package writes (the sidecar of REQ-235 / REQ-342).
DESCRIPTIONS = {
    "channel": "Channel name.",
    "condition": "Condition label of the averaged trials ('' when there is none).",
    "component": "Position of the component among the channel's components, by latency "
    "(-1: the channel has no component).",
    "label": "Component label assigned from latency windows (for example 'N1'); '' when none.",
    "shape": "Shape function of the fitted component.",
    "polarity": "Sign of the component's extremum (+1 or -1; 0 without component).",
    "status": "Outcome of the fit of this channel.",
    "message": "Reason when the status is not 'ok'.",
    "existence_p": "Corrected p-value of the existence test of the channel (NaN when not run).",
    "existence_significant": "Whether the existence test was significant (empty when not run).",
    "parameter": "Name of the parameter in the 'value' column.",
    "value": "Value of the parameter.",
    "unit": "Unit of 'value', 'se', 'ci_low' and 'ci_high'.",
    "se": "Standard error of the value (NaN when not estimated).",
    "ci_low": "Lower bound of the confidence (or credible) interval of the value.",
    "ci_high": "Upper bound of the confidence (or credible) interval of the value.",
    "weight": "Inverse-variance weight 1 / se^2 (NaN without a standard error).",
    "trial": "Trial index in the source run (0-based).",
    "subject": "Participant label.",
    "session": "Session label.",
    "run": "Run label.",
    "task": "Task label.",
    "stim_site": "Stimulated contact pair ('n/a' for sensory paradigms).",
    "level": "Level of the hierarchical model.",
    "unit_id": "Unit of that level.",
    "trusted": "Whether the trial fit is trusted (converged, good fit, shift not at a bound).",
    "converged": "Whether the trial fit converged.",
    "latency": "Time of the extremum of the component.",
    "amplitude": "Signed value of the component at its extremum.",
    "fwhm": "Full width at half maximum of the component.",
    "onset": "First time the component reaches the onset fraction of its amplitude.",
    "offset": "Last time the component is above the onset fraction of its amplitude.",
    "area": "Signed integral of the component between onset and offset.",
    "rise_time": "Onset to latency.",
    "decay_time": "Latency to offset.",
    "symmetry": "(decay_time - rise_time) / (decay_time + rise_time).",
    "sharpness": "Curvature of the unit-amplitude shape at the peak times fwhm^2.",
    "shift": "Latency shift of the whole response in the trial.",
    "scale": "Amplitude scale of the whole response in the trial.",
}

STATUS_LEVELS = {
    "ok": "fitted",
    "no_response": "no component improved the criterion",
    "failed": "the fit raised an error (see message)",
    "skipped": "not fitted (bad or stimulating channel, no valid sample)",
}

_ID_ORDER = (
    "subject",
    "session",
    "run",
    "task",
    "stim_site",
    "condition",
    "channel",
    "trial",
    "level",
    "unit_id",
    "component",
    "label",
    "shape",
    "polarity",
    "status",
    "message",
    "existence_p",
    "existence_significant",
)


###################################################################################################
## Long format


def _jsonable_cell(value):
    """Return the JSON text of a cell holding a list, dict, tuple, set or array."""
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, (list, tuple, dict, set)):
        return json.dumps(list(value) if isinstance(value, set) else value, default=str)
    return value


def _flatten(pd, table):
    for column in table.columns:
        if table[column].dtype == object:
            table[column] = table[column].map(_jsonable_cell)
    return table


def _resolve_unit(unit, data_units):
    if unit == "data":
        return data_units
    if unit == "data·s":
        return f"{data_units}·s"
    return unit


def _from_group(pd, result):
    rows = []
    for ch, cond, pos in result.index[["channel", "condition", "position"]].itertuples(
        index=False
    ):
        fit = result.fits[int(pos)]
        exist_p, exist_sig = np.nan, pd.NA
        if fit.existence is not None:
            exist_p = float(np.ravel(fit.existence.p_value)[0])
            exist_sig = bool(np.ravel(fit.existence.significant)[0])
        base = {
            "channel": ch,
            "condition": cond,
            "status": fit.status,
            "message": fit.message,
            "existence_p": exist_p,
            "existence_significant": exist_sig,
        }
        if not fit.components:
            for name in PARAMETERS:
                rows.append(
                    {
                        **base,
                        "component": -1,
                        "label": "",
                        "shape": "",
                        "polarity": 0,
                        "parameter": name,
                        "value": np.nan,
                        "se": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                    }
                )
            continue
        for position, comp in enumerate(fit.components):
            ids = {
                **base,
                "component": position,
                "label": comp.label if comp.label is not None else "",
                "shape": comp.shape,
                "polarity": comp.polarity,
            }
            for name in PARAMETERS:
                se = (comp.se or {}).get(name, np.nan)
                low, high = (comp.ci or {}).get(name, (np.nan, np.nan))
                rows.append(
                    {
                        **ids,
                        "parameter": name,
                        "value": float(getattr(comp, name)),
                        "se": float(se),
                        "ci_low": float(low),
                        "ci_high": float(high),
                    }
                )
    return pd.DataFrame(rows), dict(PARAMETERS)


def _melt(pd, table, parameters, units):
    """Melt the given numeric columns into (parameter, value[, se, ci]) rows."""
    se_cols = {p: c for p in parameters for c in (f"{p}_se", f"se_{p}") if c in table.columns}
    lo_cols = {p: f"{p}_ci_low" for p in parameters if f"{p}_ci_low" in table.columns}
    hi_cols = {p: f"{p}_ci_high" for p in parameters if f"{p}_ci_high" in table.columns}
    extra = set(se_cols.values()) | set(lo_cols.values()) | set(hi_cols.values())
    ids = [c for c in table.columns if c not in parameters and c not in extra]
    parts = []
    for p in parameters:
        part = table[ids].copy()
        part["parameter"] = p
        part["value"] = pd.to_numeric(table[p], errors="coerce").astype(float)
        part["se"] = table[se_cols[p]].astype(float) if p in se_cols else np.nan
        part["ci_low"] = table[lo_cols[p]].astype(float) if p in lo_cols else np.nan
        part["ci_high"] = table[hi_cols[p]].astype(float) if p in hi_cols else np.nan
        parts.append(part)
    if not parts:
        raise DataError("to_long: the table has no parameter column to export.")
    return pd.concat(parts, ignore_index=True), {p: units.get(p, "") for p in parameters}


def _known_parameters(table):
    known = {**PARAMETERS, **OTHER_PARAMETERS}
    return [
        c
        for c in table.columns
        if c in known and np.issubdtype(np.asarray(table[c]).dtype, np.number)
    ]


def _from_table(pd, table):
    """Return a trial table (one row per trial x component, with the metadata columns)."""
    params = _known_parameters(table)
    units = dict(getattr(table, "attrs", {}).get("units", {}))
    return _melt(pd, table, params, {**PARAMETERS, **OTHER_PARAMETERS, **units})


def _from_trials(pd, result):
    table = result.table
    if table is not None and hasattr(table, "columns") and len(table):
        return _from_table(pd, table)
    # no table: one row per trial and component from the arrays
    labels = [c.label or str(k) for k, c in enumerate(result.template.components)]
    n_trials, n_comp = (
        result.latency.shape[:2] if result.latency.ndim == 2 else (result.latency.shape[0], 0)
    )
    rows = []
    for i in range(n_trials):
        for k in range(n_comp):
            rows.append(
                {
                    "channel": result.channel,
                    "trial": i,
                    "component": k,
                    "label": labels[k] if k < len(labels) else "",
                    "trusted": bool(result.trusted[i]),
                    "latency": result.latency[i, k],
                    "amplitude": result.amplitude[i, k],
                    "fwhm": result.fwhm[i, k],
                    "shift": result.shift[i],
                    "scale": result.scale[i],
                }
            )
    table = pd.DataFrame(rows)
    return _melt(
        pd,
        table,
        ["latency", "amplitude", "fwhm", "shift", "scale"],
        {**PARAMETERS, **OTHER_PARAMETERS},
    )


def _from_hierarchical(pd, result):
    summary = result.summary.copy()
    rename = {
        "mean": "value",
        "sd": "se",
        "hdi_low": "ci_low",
        "hdi_high": "ci_high",
        "unit": "unit_id",
    }
    summary = summary.rename(columns=rename)
    for col in ("value", "se", "ci_low", "ci_high"):
        if col not in summary.columns:
            summary[col] = np.nan
    if "parameter" not in summary.columns:
        raise DataError("to_long: HierarchicalFit.summary needs a 'parameter' column.")
    units = {p: {**PARAMETERS, **OTHER_PARAMETERS}.get(p, "") for p in summary["parameter"]}
    return summary, units


def _ordered(table):
    ids = [c for c in _ID_ORDER if c in table.columns]
    rest = [c for c in table.columns if c not in ids and c not in LONG_COLUMNS]
    return table[ids + rest + [c for c in LONG_COLUMNS if c in table.columns]]


def to_long(result, *, wide=False, units=None, metadata=None):
    """Tidy table of every parameter of a result, ready for mixed-effects modelling.

    Parameters
    ----------
    result : GroupFit, ResponseFit, TrialFit, TrialFitSet, HierarchicalFit, TFFit or DataFrame
        The result to export (a DataFrame: a component table, or a table that is already
        long with ``parameter`` and ``value`` columns).
    wide : bool
        One column per parameter (``<parameter>``, ``<parameter>_se``,
        ``<parameter>_weight``) instead of one row per parameter.
    units : str, optional
        Amplitude unit of the data (replaces "data" in the units; default "data").
    metadata : dict, optional
        Constant columns added to every row (for example ``{"subject": "01"}``), so that
        tables of several recordings can be stacked.

    Returns
    -------
    DataFrame
        Long: identifying columns (channel, condition, component, label, shape, polarity,
        status, ... and every metadata column), then ``parameter``, ``value``, ``unit``,
        ``se``, ``ci_low``, ``ci_high``, ``weight`` (``1 / se^2``). One value per cell: list
        or dictionary cells are written as JSON text. Channels without a component keep one
        row per parameter with NaN values and their status. ``attrs["sidecar"]`` holds the
        column dictionary (:func:`column_dictionary`); ``attrs["units"]`` the unit of each
        column.

    Examples
    --------
    >>> import evoparam
    >>> from evoparam.sim import simulate_erp
    >>> ts, _ = simulate_erp(20, components=["N1"], snr=5., noise="white", seed=0)
    >>> fit = evoparam.fit(ts, preset="erp", window=(0.0, 0.4), baseline=(-0.2, 0.0))
    >>> long = to_long(fit, units="uV")
    >>> sorted(set(long["parameter"]))[:3]
    ['amplitude', 'area', 'decay_time']
    """
    start = _time.perf_counter()
    pd = require("pandas", extra="data", caller="evoparam.stats.to_long")
    data_units = units or "data"
    if isinstance(result, ResponseFit):
        result = GroupFit.from_fits([result], [""])
    if isinstance(result, GroupFit):
        table, param_units = _from_group(pd, result)
        kind = "GroupFit"
    elif isinstance(result, TrialFit):
        table, param_units = _from_trials(pd, result)
        kind = "TrialFit"
    elif isinstance(result, HierarchicalFit):
        table, param_units = _from_hierarchical(pd, result)
        kind = "HierarchicalFit"
    elif isinstance(result, TFFit):
        zm = result.zone_metrics
        params = [
            c
            for c in zm.columns
            if np.issubdtype(np.asarray(zm[c]).dtype, np.number)
            and c not in ("zone_id", "trial", "component")
        ]
        table, param_units = _melt(pd, zm, params, dict(getattr(zm, "attrs", {}).get("units", {})))
        kind = "TFFit"
    elif hasattr(result, "fits") and hasattr(getattr(result, "table", None), "columns"):
        # evoparam.trials.TrialFitSet (not imported: stats sits beside trials, §4.2)
        table, param_units = _from_table(pd, result.table)
        kind = "TrialFitSet"
    elif hasattr(result, "columns"):
        if {"parameter", "value"} <= set(result.columns):
            table = result.copy()
            for col in ("se", "ci_low", "ci_high"):
                if col not in table.columns:
                    table[col] = np.nan
            known = {
                **PARAMETERS,
                **OTHER_PARAMETERS,
                **dict(getattr(result, "attrs", {}).get("units", {})),
            }
            param_units = {p: known.get(p, "") for p in table["parameter"].unique()}
        else:
            params = _known_parameters(result)
            known = {
                **PARAMETERS,
                **OTHER_PARAMETERS,
                **dict(getattr(result, "attrs", {}).get("units", {})),
            }
            table, param_units = _melt(pd, result, params, known)
        kind = "DataFrame"
    else:
        raise DataError(f"to_long: cannot export a {type(result).__name__}.")

    for key, value in (metadata or {}).items():
        table[key] = value
    table = table.copy()
    if "unit" in table.columns and kind == "DataFrame":
        table["unit"] = table["unit"].astype(str)
    else:
        table["unit"] = [
            _resolve_unit(param_units.get(p, ""), data_units) for p in table["parameter"]
        ]
    se = table["se"].to_numpy(float)
    table["weight"] = np.divide(
        1.0, se**2, out=np.full(se.shape, np.nan), where=np.isfinite(se) & (se > 0)
    )
    table = _flatten(pd, _ordered(table.reset_index(drop=True)))
    if wide:
        table = _widen(pd, table)
    source_units = (
        dict(getattr(result, "attrs", {}).get("units", {})) if hasattr(result, "attrs") else {}
    )
    table.attrs["units"] = _column_units(table, data_units, source_units, param_units)
    table.attrs["sidecar"] = column_dictionary(table, units=data_units)
    provenance = getattr(result, "provenance", None)
    provenance = provenance if isinstance(provenance, Provenance) else Provenance()
    table.attrs["provenance"] = provenance.with_step(
        "evoparam.stats.to_long",
        {
            "wide": bool(wide),
            "units": data_units,
            "source": kind,
            "metadata": sorted(metadata or {}),
        },
        duration_s=_time.perf_counter() - start,
    ).to_dict()
    return table


def _widen(pd, table):
    ids = [c for c in table.columns if c not in LONG_COLUMNS]
    key = table[ids].map(repr).agg("\x1f".join, axis=1) if ids else pd.Series([""] * len(table))
    order = list(dict.fromkeys(key))
    base = table.loc[~key.duplicated(), ids].reset_index(drop=True)
    position = {k: i for i, k in enumerate(order)}
    rows = key.map(position).to_numpy()
    for p in dict.fromkeys(table["parameter"]):
        sel = (table["parameter"] == p).to_numpy()
        for suffix, col in (
            ("", "value"),
            ("_se", "se"),
            ("_ci_low", "ci_low"),
            ("_ci_high", "ci_high"),
            ("_weight", "weight"),
        ):
            values = np.full(len(base), np.nan)
            values[rows[sel]] = table.loc[sel, col].to_numpy(float)
            base[f"{p}{suffix}"] = values
        units = table.loc[sel, "unit"]
        base.attrs.setdefault("param_units", {})[p] = str(units.iloc[0]) if len(units) else ""
    return base


def _column_units(table, data_units, source_units, param_units):
    out = {}
    wide_units = table.attrs.get("param_units", {})
    for col in table.columns:
        if col in ("value", "se", "ci_low", "ci_high"):
            out[col] = "see 'unit'"
        elif col == "weight":
            out[col] = "1 / ('unit')^2"
        elif col in wide_units:
            out[col] = wide_units[col]
        elif any(col == f"{p}{s}" for p in wide_units for s in ("_se", "_ci_low", "_ci_high")):
            p = col.rsplit("_", 2 if col.endswith(("_ci_low", "_ci_high")) else 1)[0]
            out[col] = wide_units.get(p, "")
        elif col.endswith("_weight") and col[:-7] in wide_units:
            u = wide_units[col[:-7]]
            out[col] = f"1/{u}^2" if u else ""
        else:
            unit = source_units.get(
                col, COMPONENT_COLUMNS.get(col, EXISTENCE_COLUMNS.get(col, ""))
            )
            out[col] = _resolve_unit(unit, data_units)
    return out


def column_dictionary(table, *, units=None):
    """JSON sidecar describing every column of a table.

    ``Description``, ``Units`` (and ``Levels`` for ``status``), in the BIDS tabular-file
    convention.

    Columns the package does not know are described as copied from the input, with the
    unit recorded in ``table.attrs["units"]`` when there is one. ``Units`` is ``"n/a"``
    for labels, counts and dimensionless numbers.

    Examples
    --------
    >>> import pandas as pd
    >>> column_dictionary(pd.DataFrame({"channel": ["Cz"], "latency": [0.1]}))["latency"]
    {'Description': 'Time of the extremum of the component.', 'Units': 's'}
    """
    data_units = units or "data"
    recorded = dict(getattr(table, "attrs", {}).get("units", {}))
    wide_units = dict(getattr(table, "attrs", {}).get("param_units", {}))
    out = {}
    for col in table.columns:
        name = str(col)
        base = name
        for suffix in ("_ci_low", "_ci_high", "_weight", "_se"):
            if name.endswith(suffix) and name[: -len(suffix)] in {**PARAMETERS, **wide_units}:
                base = name[: -len(suffix)]
                break
        if name in DESCRIPTIONS:
            text = DESCRIPTIONS[name]
        elif base != name:
            what = {
                "_se": "Standard error of",
                "_weight": "Inverse-variance weight of",
                "_ci_low": "Lower interval bound of",
                "_ci_high": "Upper interval bound of",
            }[name[len(base) :]]
            text = f"{what} '{base}'."
        else:
            text = f"Column '{name}' of the input table."
        unit = recorded.get(name)
        if unit is None:
            unit = {**PARAMETERS, **OTHER_PARAMETERS, **COMPONENT_COLUMNS}.get(base, "")
            unit = wide_units.get(base, _resolve_unit(unit, data_units))
            if name.endswith("_weight") and unit:
                unit = f"1/{unit}^2"
        entry = {"Description": text, "Units": unit if unit else "n/a"}
        if name == "status":
            entry["Levels"] = {s: STATUS_LEVELS[s] for s in FIT_STATUSES}
        out[name] = entry
    return out


###################################################################################################
## Formulas

_R_NAME = re.compile(r"^(?:[A-Za-z]|\.(?![0-9]))[A-Za-z0-9._]*$")
_PY_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_R_RESERVED = {
    "if",
    "else",
    "repeat",
    "while",
    "function",
    "for",
    "next",
    "break",
    "TRUE",
    "FALSE",
    "NULL",
    "Inf",
    "NaN",
    "NA",
    "in",
}


def _r(name):
    return name if _R_NAME.match(name) and name not in _R_RESERVED else f"`{name}`"


def _py(name):
    plain = _PY_NAME.match(name) and not keyword.iskeyword(name) and name not in ("C", "Q", "I")
    return name if plain else f'Q("{name}")'


def mixed_formula(table, *, response, fixed, groups, engine="lme4", se=None):
    """Model formula of a linear mixed model with nested random intercepts.

    Parameters
    ----------
    table : DataFrame
        The table the model will be fitted on (columns are checked; non-numeric fixed
        effects are written as factors for ``statsmodels``).
    response : str
        Response column (``"value"`` for :func:`to_long` output).
    fixed : sequence of str
        Fixed-effect columns (an empty sequence: intercept only).
    groups : sequence of str
        Grouping columns, outermost first: ``("subject", "channel")`` gives random
        intercepts for subjects and for channels within subjects.
    engine : {"lme4", "brms", "statsmodels"}
        ``lme4``: ``y ~ x + (1 | subject) + (1 | subject:channel)``. ``brms``: the same,
        with ``y | se(se, sigma = TRUE)`` when ``se`` names a standard-error column (the
        measurement-error model). ``statsmodels``: the fixed part only, ``y ~ C(x)``; the
        groups go to :func:`fit_mixed` (``groups=`` the first, the others as variance
        components).
    se : str, optional
        Standard-error column for the brms measurement-error term.

    Returns
    -------
    str

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"value": [1.], "condition": ["a"], "subject": ["01"],
    ...                    "channel": ["C3"]})
    >>> mixed_formula(df, response="value", fixed=["condition"],
    ...               groups=["subject", "channel"])
    'value ~ condition + (1 | subject) + (1 | subject:channel)'
    """
    engines = ("lme4", "brms", "statsmodels")
    if engine not in engines:
        raise SettingsError(f"engine must be one of {engines}, not {engine!r}.")
    fixed = [fixed] if isinstance(fixed, str) else list(fixed)
    groups = [groups] if isinstance(groups, str) else list(groups)
    columns = [response] + fixed + groups + ([se] if se is not None else [])
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise SettingsError(f"mixed_formula: missing columns {missing}.")
    if not groups:
        raise SettingsError("mixed_formula: give at least one grouping column.")
    if engine == "statsmodels":
        terms = []
        for name in fixed:
            numeric = (
                np.issubdtype(np.asarray(table[name]).dtype, np.number)
                and table[name].dtype != bool
            )
            terms.append(_py(name) if numeric else f"C({_py(name)})")
        return f"{_py(response)} ~ {' + '.join(terms) if terms else '1'}"
    lhs = _r(response)
    if engine == "brms" and se is not None:
        lhs = f"{lhs} | se({_r(se)}, sigma = TRUE)"
    rhs = " + ".join(_r(name) for name in fixed) if fixed else "1"
    random = [f"(1 | {':'.join(_r(g) for g in groups[: k + 1])})" for k in range(len(groups))]
    return f"{lhs} ~ {rhs} + {' + '.join(random)}"


def fit_mixed(table, formula, groups, *, nested=(), re_formula=None, reml=True, **fit_kw):
    """Fit a linear mixed model with ``statsmodels`` (extra ``stats``).

    Parameters
    ----------
    table : DataFrame
    formula : str
        Fixed part, for example from ``mixed_formula(..., engine="statsmodels")``.
    groups : str
        Top-level grouping column (random intercept per group).
    nested : sequence of str
        Columns nested in ``groups`` with their own random intercepts (variance
        components ``0 + C(column)``).
    re_formula : str, optional
        Random-effects formula of the top level (default: intercept).
    reml : bool
    **fit_kw
        Passed to ``MixedLM.fit``.

    Returns
    -------
    statsmodels MixedLMResults
    """
    require("statsmodels", extra="stats", min_version="0.14", caller="evoparam.stats.fit_mixed")
    import statsmodels.formula.api as smf

    if groups not in table.columns:
        raise SettingsError(f"fit_mixed: no column '{groups}'.")
    nested = [nested] if isinstance(nested, str) else list(nested)
    vc = {name: f"0 + C({_py(name)})" for name in nested} or None
    model = smf.mixedlm(
        formula,
        _complete_rows(table, formula, [groups] + nested, re_formula),
        groups=groups,
        re_formula=re_formula,
        vc_formula=vc,
        missing="none",
    )
    return model.fit(reml=reml, **fit_kw)


def _complete_rows(table, formula, columns, re_formula=None):
    """Return the rows of ``table`` with every variable of the model present.

    What ``missing="drop"`` of statsmodels does. statsmodels 0.15 drops them itself only for plain
    names: a quoted one (``Q("N1 latency")``, ``Q("stim site")``) is not found.
    """
    import patsy  # a dependency of statsmodels

    data = table.loc[table[columns].notna().all(axis=1)]
    _, design = patsy.dmatrices(formula, data, NA_action="drop", return_type="dataframe")
    data = data.loc[design.index]
    if re_formula:
        design = patsy.dmatrix(re_formula, data, NA_action="drop", return_type="dataframe")
        data = data.loc[design.index]
    return data


###################################################################################################
## Group summaries

_UNIT_KEYS = ("subject", "session", "run", "stim_site", "condition", "channel")


def group_summary(
    table, *, by, parameters=None, ch_info=None, n_boot=1000, ci_level=0.95, random_state=None
):
    """Summary of each parameter per group, with the count of every fit status.

    Parameters
    ----------
    table : DataFrame
        A component table (``GroupFit.to_dataframe()``, one row per component) or a long
        table (:func:`to_long`). Statistics use the rows whose ``status`` is ``"ok"``.
    by : sequence of str
        Grouping columns (``"channel"``, ``"region"``, ``"subject"``, ``"condition"``,
        ``"label"``, ...). Columns of ``ch_info`` (for example ``region``) are joined on the
        channel name when the table lacks them.
    parameters : sequence of str, optional
        Parameters to summarise (default: every component parameter in the table).
    ch_info : DataFrame, optional
        Channel table (``name`` column), for grouping by its columns.
    n_boot : int
        Resamples of the bootstrap interval of each mean.
    ci_level : float
    random_state : int, Generator or None

    Returns
    -------
    DataFrame
        One row per group and parameter: the ``by`` columns, ``parameter``, ``n`` (values
        used), ``mean``, ``median``, ``sd``, ``iqr``, ``ci_low`` / ``ci_high`` (percentile
        bootstrap of the mean), and per group ``n_ok``, ``n_no_response``, ``n_failed``,
        ``n_skipped`` and ``n_channels``: the fits (channel x condition, and subject,
        session, run, stimulation site when present) of the group by status, so that
        ``n_ok + n_no_response + n_failed + n_skipped == n_channels``.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"channel": ["A", "B", "C"], "region": ["x", "x", "y"],
    ...                    "latency": [0.02, 0.03, float("nan")],
    ...                    "status": ["ok", "ok", "no_response"]})
    >>> out = group_summary(df, by=["region"], parameters=["latency"], random_state=0)
    >>> out[["region", "n", "n_ok", "n_no_response", "n_channels"]].values.tolist()
    [['x', 2, 2, 0, 2], ['y', 0, 0, 1, 1]]
    """
    start = _time.perf_counter()
    pd = require("pandas", extra="data", caller="evoparam.stats.group_summary")
    by = [by] if isinstance(by, str) else list(by)
    data = table.copy()
    absent = [c for c in by if c not in data.columns]
    if absent and ch_info is not None:
        info = ch_info.rename(columns={"name": "channel"})
        join = [c for c in absent if c in info.columns]
        if "channel" in data.columns and join:
            data = data.merge(info[["channel"] + join], on="channel", how="left")
    absent = [c for c in by if c not in data.columns]
    if absent:
        raise SettingsError(
            f"group_summary: missing columns {absent} (give ch_info for "
            "channel columns such as 'region')."
        )
    if "status" not in data.columns:
        data["status"] = "ok"
    unknown = set(data["status"].astype(str)) - set(FIT_STATUSES)
    if unknown:
        raise DataError(f"group_summary: unknown status values {sorted(unknown)}.")
    long = {"parameter", "value"} <= set(data.columns)
    if long:
        params = list(dict.fromkeys(data["parameter"])) if parameters is None else list(parameters)
    else:
        params = (
            [p for p in PARAMETERS if p in data.columns]
            if parameters is None
            else list(parameters)
        )
        missing = [p for p in params if p not in data.columns]
        if missing:
            raise SettingsError(f"group_summary: missing parameter columns {missing}.")
    rng = np.random.default_rng(random_state)
    unit_cols = [c for c in _UNIT_KEYS if c in data.columns]
    if not unit_cols:
        data["_row"] = np.arange(len(data))
        unit_cols = ["_row"]
    groups = (
        [((), data)]
        if not by
        else [
            ((key if isinstance(key, tuple) else (key,)), part)
            for key, part in data.groupby(by, sort=False, dropna=False)
        ]
    )
    rows = []
    for key, part in groups:
        units = part.drop_duplicates(unit_cols)
        status = part.groupby(unit_cols, sort=False, dropna=False)["status"].agg(
            lambda s: s.astype(str).iloc[0]
        )
        counts = {f"n_{s}": int(np.sum(status.to_numpy() == s)) for s in FIT_STATUSES}
        ok = part[part["status"].astype(str) == "ok"]
        for p in params:
            if long:
                values = pd.to_numeric(
                    ok.loc[ok["parameter"] == p, "value"], errors="coerce"
                ).to_numpy(float)
            else:
                values = pd.to_numeric(ok[p], errors="coerce").to_numpy(float)
            values = values[np.isfinite(values)]
            row = dict(zip(by, key, strict=False))
            row.update(parameter=p, n=int(values.size))
            if values.size:
                q1, med, q3 = np.percentile(values, [25, 50, 75])
                row.update(
                    mean=float(values.mean()),
                    median=float(med),
                    sd=float(values.std(ddof=1)) if values.size > 1 else np.nan,
                    iqr=float(q3 - q1),
                )
                if values.size > 1:
                    boot = values[rng.integers(0, values.size, (n_boot, values.size))].mean(axis=1)
                    ci = bootstrap_interval(
                        values.mean(),
                        boot,
                        method="percentile",
                        ci_level=ci_level,
                        expand_n=values.size,
                    )
                    row.update(ci_low=ci["low"], ci_high=ci["high"])
                else:
                    row.update(ci_low=np.nan, ci_high=np.nan)
            else:
                row.update(
                    mean=np.nan,
                    median=np.nan,
                    sd=np.nan,
                    iqr=np.nan,
                    ci_low=np.nan,
                    ci_high=np.nan,
                )
            row.update(counts, n_channels=int(len(units)))
            rows.append(row)
    columns = by + [
        "parameter",
        "n",
        "mean",
        "median",
        "sd",
        "iqr",
        "ci_low",
        "ci_high",
        "n_ok",
        "n_no_response",
        "n_failed",
        "n_skipped",
        "n_channels",
    ]
    out = pd.DataFrame(rows, columns=columns)
    settings = {
        "by": by,
        "parameters": params,
        "n_boot": int(n_boot),
        "ci_level": ci_level,
        "units": unit_cols,
        "random_state": random_state
        if isinstance(random_state, (int, type(None)))
        else "generator",
    }
    out.attrs["settings"] = settings
    out.attrs["provenance"] = (
        Provenance()
        .with_step(
            "evoparam.stats.group_summary", settings, duration_s=_time.perf_counter() - start
        )
        .to_dict()
    )
    return out
