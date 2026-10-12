# SPDX-License-Identifier: Apache-2.0
"""Condition contrasts by permutation at the level of the exchangeable unit (spec §5.11).

A contrast compares one parameter between two conditions. The observations are first
averaged to one value per unit (``trial``, ``channel`` or ``subject``) and condition;
the null distribution then exchanges the condition labels at that level only: random sign
flips of each unit's difference (paired: every unit seen in both conditions; every
pattern when there are few units) or random reassignments of the units to the two
conditions (unpaired). The statistic is the paired t or the Welch t, so that unequal
variances do not distort the permutation test. Units above the named one must not vary
inside a contrast: trials of different subjects are not exchangeable (a subject effect
would pass for a condition effect), so such a request raises SettingsError.
"""

import itertools
import time as _time

import numpy as np
from scipy.stats import norm

from evoparam.data import Provenance
from evoparam.errors import DataError, SettingsError
from evoparam.existence.corrections import CORRECTIONS, correct
from evoparam.utils._optional import require

from .bootstrap import _quantile_z

__all__ = ["compare_conditions", "hedges_g", "UNITS", "COMPARE_COLUMNS"]

UNITS = ("trial", "channel", "subject")

# Levels that contain each unit: a contrast must stay inside one of each.
_ABOVE = {"trial": ("channel", "subject"), "channel": ("subject",), "subject": ()}

# Parameters measured in seconds (reported in ms as well) when the table has no unit record.
_TIME_NAMES = (
    "latency",
    "onset",
    "offset",
    "fwhm",
    "rise_time",
    "decay_time",
    "duration",
    "shift",
)

COMPARE_COLUMNS = {
    "parameter": "",
    "condition_a": "",
    "condition_b": "",
    "unit": "",
    "paired": "",
    "n_a": "",
    "n_b": "",
    "mean_a": "param",
    "mean_b": "param",
    "difference": "param",
    "ci_low": "param",
    "ci_high": "param",
    "difference_ms": "ms",
    "ci_low_ms": "ms",
    "ci_high_ms": "ms",
    "statistic": "",
    "p_value_raw": "",
    "p_value": "",
    "hedges_g": "",
    "n_permutations": "",
    "exact": "",
    "n_excluded": "",
    "n_missing": "",
}


###################################################################################################
## Statistics


def _paired_t(d):
    """Paired t of each row of ``d`` [..., n] (0 when the differences have no spread)."""
    n = d.shape[-1]
    mean = d.mean(axis=-1)
    sd = d.std(axis=-1, ddof=1)
    return np.divide(mean, sd / np.sqrt(n), out=np.zeros_like(mean), where=sd > 0)


def _welch_t(a, b):
    """Welch t of each row pair of ``a`` [..., n_a] and ``b`` [..., n_b]."""
    va = a.var(axis=-1, ddof=1) / a.shape[-1]
    vb = b.var(axis=-1, ddof=1) / b.shape[-1]
    diff = a.mean(axis=-1) - b.mean(axis=-1)
    se = np.sqrt(va + vb)
    return np.divide(diff, se, out=np.zeros_like(diff), where=se > 0)


def hedges_g(a, b=None, *, paired=False):
    """Hedges' g: the standardised mean difference with the small-sample correction.

    Unpaired: ``(mean a - mean b) / pooled SD`` times ``J = 1 - 3 / (4 df - 1)``,
    ``df = n_a + n_b - 2``. Paired (``b`` None and ``a`` the differences, or ``a`` and
    ``b`` matched): ``mean(d) / SD(d)`` times ``J`` with ``df = n - 1``.

    Examples
    --------
    >>> round(hedges_g([2., 3., 4.], [1., 2., 3.]), 3)
    0.8
    """
    a = np.asarray(a, float)
    if paired or b is None:
        d = a if b is None else a - np.asarray(b, float)
        df = d.size - 1
        sd = d.std(ddof=1) if d.size > 1 else np.nan
        g = d.mean() / sd if sd > 0 else np.nan
    else:
        b = np.asarray(b, float)
        df = a.size + b.size - 2
        pooled = (
            np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / df)
            if df > 0
            else np.nan
        )
        g = (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan
    if not df > 0:
        return float("nan")
    return float(g * (1 - 3 / (4 * df - 1)))


def _paired_test(d, n_permutations, rng):
    n = d.size
    observed = _paired_t(d)
    if 2**n <= n_permutations:
        grid = (np.arange(2**n)[:, None] >> np.arange(n)) & 1
        signs, exact = 1.0 - 2.0 * grid, True
    else:
        signs, exact = rng.choice([-1.0, 1.0], size=(n_permutations, n)), False
    null = _paired_t(signs * d)
    count = np.sum(np.abs(null) >= abs(observed) - 1e-12)
    p = count / len(null) if exact else (1 + count) / (1 + len(null))
    return float(observed), float(p), len(null), exact


def _unpaired_test(a, b, n_permutations, rng):
    n_a, n_b = a.size, b.size
    pooled = np.concatenate([a, b])
    observed = _welch_t(a, b)
    n_split = _n_choose(n_a + n_b, n_a)
    if n_split <= n_permutations:
        rows = [
            np.r_[list(c), [i for i in range(n_a + n_b) if i not in c]]
            for c in itertools.combinations(range(n_a + n_b), n_a)
        ]
        order, exact = np.array(rows), True
    else:
        order = rng.permuted(np.tile(np.arange(n_a + n_b), (n_permutations, 1)), axis=1)
        exact = False
    shuffled = pooled[order]
    null = _welch_t(shuffled[:, :n_a], shuffled[:, n_a:])
    count = np.sum(np.abs(null) >= abs(observed) - 1e-12)
    p = count / len(null) if exact else (1 + count) / (1 + len(null))
    return float(observed), float(p), len(null), exact


def _percentile(boot, ci_level, n):
    """Return the expanded percentile interval (see :func:`~evoparam.stats.bootstrap_interval`)."""
    low, high = np.quantile(boot, norm.cdf(_quantile_z(ci_level, n)))
    return {"low": float(low), "high": float(high)}


def _n_choose(n, k):
    from math import comb

    return comb(n, k)


###################################################################################################
## Public function


def _is_time(table, parameter):
    units = getattr(table, "attrs", {}).get("units", {})
    if parameter in units:
        return units[parameter] == "s"
    return parameter in _TIME_NAMES or parameter.endswith("_latency")


def compare_conditions(
    table,
    *,
    parameter,
    condition,
    unit="subject",
    paired=True,
    n_permutations=5000,
    correction="fdr_bh",
    random_state=None,
    by=None,
    conditions=None,
    ci_level=0.95,
    n_boot=2000,
):
    """Contrast a parameter between conditions by permutation at the named unit level.

    Parameters
    ----------
    table : DataFrame
        Tidy table: one row per observation, with the column ``parameter``, the column
        ``condition`` and a column named after ``unit`` (``"trial"`` may be absent: each
        row is then a trial). Rows whose ``status`` is not ``"ok"`` are left out and
        counted (``n_excluded``), as are rows with a missing value (``n_missing``).
    parameter : str
        Column to compare (for example ``"latency"``).
    condition : str
        Column holding the condition labels.
    unit : {"trial", "channel", "subject"}
        Exchangeable unit. Observations are averaged to one value per unit and condition,
        and condition labels are exchanged between or within units only.
    paired : bool
        Each unit is observed in both conditions (sign flips of the within-unit
        differences; units missing a condition are dropped) or in one (reassignment of
        units to conditions).
    n_permutations : int
        Random permutations; every permutation is used when there are no more than this.
    correction : {"fdr_bh", "holm", "bonferroni", "none"}
        Multiple-comparison correction across every contrast of the call (as in §5.6).
    random_state : int, Generator or None
    by : sequence of str, optional
        Columns defining separate contrasts (for example ``("channel", "label")``).
    conditions : sequence, optional
        Condition labels to compare, in order (default every label, in order of first
        appearance); every pair is contrasted.
    ci_level : float
        Coverage of the bootstrap interval of the difference (over units).
    n_boot : int
        Resamples of that interval.

    Returns
    -------
    DataFrame
        One row per contrast (``by`` columns first): ``parameter``, ``condition_a``,
        ``condition_b``, ``unit``, ``paired``, ``n_a``, ``n_b``, ``mean_a``, ``mean_b``,
        ``difference`` (a - b) with ``ci_low`` / ``ci_high``, the same in ms
        (``difference_ms``, ...) when the parameter is a time, ``statistic`` (paired or
        Welch t), ``p_value_raw``, ``p_value`` (corrected), ``hedges_g``,
        ``n_permutations`` (used), ``exact``, ``n_excluded``, ``n_missing``.

    Raises
    ------
    SettingsError
        A level above ``unit`` varies inside a contrast (``unit="trial"`` across subjects
        or channels): add it to ``by`` or compare at that level.

    Examples
    --------
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"subject": np.repeat(np.arange(12), 2),
    ...                    "condition": ["a", "b"] * 12,
    ...                    "latency": 0.1 + rng.normal(0, 0.005, 24) + np.tile([0.01, 0], 12)})
    >>> out = compare_conditions(df, parameter="latency", condition="condition",
    ...                          n_permutations=2000, random_state=0)
    >>> bool(out["p_value"].iloc[0] < 0.05), round(float(out["difference_ms"].iloc[0]))
    (True, 8)
    """
    start = _time.perf_counter()
    pd = require("pandas", extra="data", caller="evoparam.stats.compare_conditions")
    if unit not in UNITS:
        raise SettingsError(f"unit must be one of {UNITS}, not {unit!r}.")
    if correction not in CORRECTIONS:
        raise SettingsError(f"correction must be one of {CORRECTIONS}.")
    by = [] if by is None else ([by] if isinstance(by, str) else list(by))
    needed = [parameter, condition] + by + ([unit] if unit != "trial" else [])
    missing = [c for c in needed if c not in table.columns]
    if missing:
        raise SettingsError(f"compare_conditions: missing columns {missing}.")
    if unit in by:
        raise SettingsError("the unit column cannot also be in by.")
    rng = np.random.default_rng(random_state)
    is_time = _is_time(table, parameter)

    data = table.copy()
    excluded = np.zeros(len(data), bool)
    if "status" in data.columns:
        excluded = (data["status"].astype(str) != "ok").to_numpy()
    values = pd.to_numeric(data[parameter], errors="coerce").to_numpy(float)
    missing_rows = ~excluded & ~np.isfinite(values)
    data["_value"] = values
    data["_excluded"] = excluded
    data["_missing"] = missing_rows
    if unit == "trial" and "trial" not in data.columns:
        data["trial"] = np.arange(len(data))
    data["_cond"] = data[condition].astype(str)
    labels = [
        str(c) for c in (conditions if conditions is not None else dict.fromkeys(data["_cond"]))
    ]
    if len(labels) < 2:
        raise DataError(f"compare_conditions: column '{condition}' has fewer than 2 conditions.")

    if by:
        group_code, group_keys = pd.MultiIndex.from_frame(data[by].astype(object)).factorize()
        group_keys = [tuple(k) for k in group_keys]
    else:
        group_code, group_keys = np.zeros(len(data), np.int64), [()]
    n_groups = len(group_keys)
    data["_group"] = group_code
    n_excluded = np.bincount(
        group_code, weights=data["_excluded"].to_numpy(float), minlength=n_groups
    ).astype(int)
    n_missing = np.bincount(
        group_code, weights=data["_missing"].to_numpy(float), minlength=n_groups
    ).astype(int)
    use = data[~data["_excluded"] & ~data["_missing"]]
    for level in _ABOVE[unit]:
        if level in use.columns and level not in by:
            counts = use.groupby("_group")[level].nunique()
            if (counts > 1).any():
                key = group_keys[int(counts.index[np.argmax(counts.to_numpy() > 1)])]
                where = f" in group {dict(zip(by, key, strict=False))}" if by else ""
                raise SettingsError(
                    f"unit='{unit}' but several {level}s{where}: {unit}s of different "
                    f"{level}s are not exchangeable. Compare at unit='{level}', or add "
                    f"'{level}' to by to test each {level} separately."
                )
    # one value per group x unit x condition, then each group as plain arrays
    cells = (
        use.groupby(["_group", unit, "_cond"], sort=False, dropna=False)["_value"]
        .mean()
        .reset_index()
    )
    cell_group = cells["_group"].to_numpy()
    cell_unit = pd.factorize(cells[unit])[0]
    cell_cond = cells["_cond"].to_numpy()
    cell_value = cells["_value"].to_numpy(float)
    order = np.argsort(cell_group, kind="stable")
    bounds = np.searchsorted(cell_group[order], np.arange(n_groups + 1))
    rows = []
    for gi, key in enumerate(group_keys):
        idx = order[bounds[gi] : bounds[gi + 1]]
        units_g, cond_g, value_g = cell_unit[idx], cell_cond[idx], cell_value[idx]
        for cond_a, cond_b in itertools.combinations(labels, 2):
            row = dict(zip(by, key, strict=False))
            row.update(
                parameter=parameter,
                condition_a=cond_a,
                condition_b=cond_b,
                unit=unit,
                paired=bool(paired),
                n_excluded=int(n_excluded[gi]),
                n_missing=int(n_missing[gi]),
            )
            in_a, in_b = cond_g == cond_a, cond_g == cond_b
            ua, va, ub, vb = units_g[in_a], value_g[in_a], units_g[in_b], value_g[in_b]
            if paired:
                _, ia, ib = np.intersect1d(ua, ub, return_indices=True)
                a, b = va[ia], vb[ib]
            else:
                shared = np.intersect1d(ua, ub)
                if shared.size:
                    raise SettingsError(
                        f"paired=False but {shared.size} {unit}s are observed in both "
                        f"'{cond_a}' and '{cond_b}'; use paired=True."
                    )
                a, b = va, vb
            row.update(
                n_a=int(a.size),
                n_b=int(b.size),
                mean_a=float(a.mean()) if a.size else np.nan,
                mean_b=float(b.mean()) if b.size else np.nan,
            )
            enough = a.size >= 2 if paired else (a.size >= 2 and b.size >= 2)
            if not enough:
                row.update(
                    difference=row["mean_a"] - row["mean_b"],
                    ci_low=np.nan,
                    ci_high=np.nan,
                    statistic=np.nan,
                    p_value_raw=np.nan,
                    hedges_g=np.nan,
                    n_permutations=0,
                    exact=False,
                )
                rows.append(row)
                continue
            if paired:
                d = a - b
                stat, p, n_perm, exact = _paired_test(d, int(n_permutations), rng)
                boot = d[rng.integers(0, d.size, (n_boot, d.size))].mean(axis=1)
                ci = _percentile(boot, ci_level, d.size)
                g = hedges_g(d, paired=True)
                diff = float(d.mean())
            else:
                stat, p, n_perm, exact = _unpaired_test(a, b, int(n_permutations), rng)
                boot = a[rng.integers(0, a.size, (n_boot, a.size))].mean(axis=1) - b[
                    rng.integers(0, b.size, (n_boot, b.size))
                ].mean(axis=1)
                diff = float(a.mean() - b.mean())
                ci = _percentile(boot, ci_level, min(a.size, b.size))
                g = hedges_g(a, b)
            row.update(
                difference=diff,
                ci_low=ci["low"],
                ci_high=ci["high"],
                statistic=stat,
                p_value_raw=p,
                hedges_g=g,
                n_permutations=int(n_perm),
                exact=bool(exact),
            )
            rows.append(row)

    out = pd.DataFrame(rows, columns=by + list(COMPARE_COLUMNS))
    out["p_value"] = correct(out["p_value_raw"].to_numpy(float), correction)
    scale = 1000.0 if is_time else np.nan
    for name in ("difference", "ci_low", "ci_high"):
        out[f"{name}_ms"] = out[name] * scale
    units = dict(getattr(table, "attrs", {}).get("units", {}))
    param_unit = units.get(parameter, "s" if is_time else "")
    out.attrs["units"] = {
        k: (param_unit if v == "param" else v) for k, v in COMPARE_COLUMNS.items()
    }
    settings = {
        "parameter": parameter,
        "condition": condition,
        "unit": unit,
        "paired": bool(paired),
        "n_permutations": int(n_permutations),
        "correction": correction,
        "by": by,
        "conditions": labels,
        "ci_level": ci_level,
        "n_boot": int(n_boot),
        "random_state": random_state
        if isinstance(random_state, (int, type(None)))
        else "generator",
    }
    out.attrs["settings"] = settings
    out.attrs["provenance"] = (
        Provenance()
        .with_step(
            "evoparam.stats.compare_conditions", settings, duration_s=_time.perf_counter() - start
        )
        .to_dict()
    )
    return out
