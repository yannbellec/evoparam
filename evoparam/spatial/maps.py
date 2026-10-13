# SPDX-License-Identifier: Apache-2.0
"""Parameter maps and the (stimulation site x channel) connectivity export.

Spec §5.9, methods 3 and 6.
"""

import re
import warnings
from typing import TYPE_CHECKING

import numpy as np

from evoparam._typing import Any
from evoparam.data import GroupFit
from evoparam.data.results import COMPONENT_FEATURES
from evoparam.errors import DataError, SettingsError
from evoparam.errors import warn as _warn
from evoparam.utils._optional import require

if TYPE_CHECKING:
    from evoparam.data.results import HierarchicalFit, ResponseTest

__all__ = [
    "parameter_map",
    "connectivity_matrix",
    "connectivity_to_long",
    "connectivity_from_long",
    "connectivity_array",
    "plot_parameter_map",
    "MAP_COLUMNS",
    "MARKERS",
]

# Columns of parameter_map(), one row per (stimulation site, channel).
MAP_COLUMNS = (
    ("stim_site", "channel", "status", "component", "label", "shape")
    + COMPONENT_FEATURES
    + (
        "r2",
        "value",
        "existence_p",
        "existence_significant",
        "p_response",
        "shape_id",
        "shape_weight",
        "x",
        "y",
        "z",
        "coord_space",
        "marker",
    )
)
# Marker of each row: a stimulating contact of that site, a bad channel, a channel whose
# window was masked (not fitted), a fitted response, no response, or a failed fit.
MARKERS = ("stim_contact", "bad", "masked", "response", "no_response", "failed")
PARAMETERS = COMPONENT_FEATURES + ("r2", "existence_p", "p_response", "shape_weight")
LONG_COLUMNS = ("stim_site", "channel", "parameter", "value")


def _contacts(site: Any) -> set[Any]:
    return {part.strip() for part in re.split(r"[-–]", str(site)) if part.strip()}


def _pick(fit: Any, component: Any) -> Any:
    """Index of the reported component: the largest absolute amplitude, a position, or a label."""
    comps = fit.components
    if not comps:
        return None
    if component is None:
        return int(np.argmax([abs(c.amplitude) for c in comps]))
    if isinstance(component, str):
        found = [k for k, c in enumerate(comps) if c.label == component]
        return found[0] if found else None
    return int(component) if 0 <= int(component) < len(comps) else None


def _ch_info(ch_info: Any) -> Any:
    if ch_info is None:
        return None
    return getattr(ch_info, "ch_info", ch_info)


def _existence_lookup(existence: Any) -> tuple[dict[Any, Any], dict[Any, Any]]:
    """(channel, group) -> (p, significant) from a ResponseTest.

    (channel, site) -> P from a table ``p_response`` (hierarchical fits).
    """
    probs: dict[Any, Any]
    tests: dict[Any, Any]
    tests, probs = {}, {}
    if existence is None:
        return tests, probs
    p_table = getattr(existence, "p_response", None)
    if p_table is not None and hasattr(p_table, "columns"):
        site_col = next(
            (c for c in ("stim_site", "condition", "site") if c in p_table.columns), None
        )
        value_col = next(
            (c for c in ("p_response", "probability", "p") if c in p_table.columns), None
        )
        if value_col is not None and "channel" in p_table.columns:
            for _, row in p_table.iterrows():
                site = str(row[site_col]) if site_col else ""
                key = (str(row["channel"]), site)
                probs[key] = max(probs.get(key, -np.inf), float(row[value_col]))
        return tests, probs
    settings = getattr(existence, "settings", {}) or {}
    channels = settings.get("channels")
    if channels is None:
        raise DataError("existence: a ResponseTest with settings['channels'] is needed.")
    groups = settings.get("groups") or [""]
    p = np.asarray(existence.p_value, float)
    sig = np.asarray(existence.significant, bool)
    if p.ndim == 1:
        p, sig = p[:, None], sig[:, None]
    for i, ch in enumerate(channels):
        for j, group in enumerate(groups):
            tests[(str(ch), str(group))] = (float(p[i, j]), bool(sig[i, j]))
    return tests, probs


def _basis_lookup(basis: Any) -> dict[Any, Any]:
    """(site, channel) -> (shape_id, weight) from ShapeBasis objects with channel groups."""
    out: dict[Any, Any]
    out = {}
    if basis is None:
        return out
    items = basis.items() if isinstance(basis, dict) else [(None, basis)]
    for raw_site, b in items:
        site = str(raw_site if raw_site is not None else b.settings.get("stim_site") or "")
        for _, row in b.assignment.iterrows():
            out[(site, str(row["group"]))] = (int(row["shape_id"]), float(row["weight"]))
    return out


def parameter_map(  # noqa: C901, PLR0912, PLR0915  (public API, signature fixed by spec §5.9)
    group_fit: "GroupFit",
    *,
    parameter: str = "amplitude",
    stim_site: Any = None,
    existence: "ResponseTest | HierarchicalFit | None" = None,
    ch_info: Any = None,
    component: Any = None,
    basis: Any = None,
) -> Any:
    """Per (stimulation site, channel) table of a component's parameters with coordinates.

    The stimulation site of a fit is its condition (``evoparam.fit(..., by="stim_site")``)
    or ``stim_site`` when the GroupFit has no conditions; ``stim_site`` also selects one
    site. The reported component is the one of largest absolute amplitude (``component``:
    a position or a label to choose another). ``value`` repeats the column ``parameter``.

    Parameters
    ----------
    group_fit : GroupFit
    parameter : str
        One of :data:`PARAMETERS`.
    existence : ResponseTest or HierarchicalFit, optional
        Adds ``existence_p`` and ``existence_significant`` (ResponseTest, matched on
        channel and group) or ``p_response`` (a ``p_response`` table with ``channel``,
        ``stim_site`` and ``p_response`` columns).
    ch_info : DataFrame, TrialSet or Evoked, optional
        Channel table with ``x, y, z`` (mm) and ``coord_space``; ``bad`` and
        ``stim_contact`` give the markers. Without coordinates the table has NaN
        coordinates and a warning is issued.
    basis : ShapeBasis or dict of site -> ShapeBasis, optional
        Divergent shared shapes (groups = channels) for ``shape_id`` and ``shape_weight``.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`MAP_COLUMNS`; ``marker`` is one of :data:`MARKERS`.
    """
    pd = require("pandas", extra="data", caller="evoparam.spatial.parameter_map")
    if not isinstance(group_fit, GroupFit):
        raise DataError("parameter_map expects a GroupFit.")
    if parameter not in PARAMETERS:
        raise SettingsError(f"parameter must be one of {PARAMETERS}, not {parameter!r}.")
    info = _ch_info(ch_info)
    coords = {}
    has_coords = info is not None and {"name", "x", "y", "z"} <= set(info.columns)
    if has_coords:
        for _, row in info.iterrows():
            coords[str(row["name"])] = (
                float(row["x"]),
                float(row["y"]),
                float(row["z"]),
                str(row.get("coord_space", "") or ""),
            )
    if not has_coords or not any(np.isfinite(v[:3]).all() for v in coords.values()):
        _warn(
            "parameter_map: no channel coordinates (ch_info x, y, z); the map is a table only.",
            UserWarning,
            stacklevel=2,
        )
    bad, stim_flag = set(), set()
    if info is not None and "name" in info.columns:
        names = info["name"].astype(str)
        if "bad" in info.columns:
            bad = {n for n, v in zip(names, info["bad"], strict=False) if bool(v)}
        if "stim_contact" in info.columns:
            stim_flag = {n for n, v in zip(names, info["stim_contact"], strict=False) if bool(v)}
    tests, probs = _existence_lookup(existence)
    shapes = _basis_lookup(basis)

    rows = []
    for ch, cond, pos in group_fit.index[["channel", "condition", "position"]].itertuples(
        index=False
    ):
        site = str(cond) if str(cond) else (str(stim_site) if stim_site is not None else "")
        if stim_site is not None and site != str(stim_site):
            continue
        fit = group_fit.fits[int(pos)]
        k = _pick(fit, component) if fit.status == "ok" else None
        row = {
            "stim_site": site,
            "channel": str(ch),
            "status": fit.status,
            "component": -1 if k is None else k,
            "label": "",
            "shape": "",
        }
        row.update(dict.fromkeys(COMPONENT_FEATURES, np.nan))
        row["polarity"] = 0
        if k is not None:
            comp = fit.components[k]
            row.update({"label": comp.label or "", "shape": comp.shape})
            row.update({name: getattr(comp, name) for name in COMPONENT_FEATURES})
        row["r2"] = fit.gof.r2 if fit.gof is not None else np.nan
        p, sig = tests.get((str(ch), str(cond)), tests.get((str(ch), ""), (np.nan, None)))
        row["existence_p"], row["existence_significant"] = p, sig
        row["p_response"] = probs.get((str(ch), site), probs.get((str(ch), ""), np.nan))
        shape_id, weight = shapes.get((site, str(ch)), (-1, np.nan))
        row["shape_id"], row["shape_weight"] = shape_id, weight
        x, y, z, space = coords.get(str(ch), (np.nan, np.nan, np.nan, ""))
        row.update({"x": x, "y": y, "z": z, "coord_space": space})
        if str(ch) in _contacts(site) or (not site and str(ch) in stim_flag):
            marker = "stim_contact"
        elif str(ch) in bad:
            marker = "bad"
        elif fit.status in ("skipped", "failed") and "mask" in fit.message.lower():
            marker = "masked"
        elif fit.status == "ok":
            marker = "response"
        elif fit.status == "no_response":
            marker = "no_response"
        else:
            marker = "failed"
        row["marker"] = marker
        rows.append(row)
    table = pd.DataFrame(rows, columns=[c for c in MAP_COLUMNS if c != "value"])
    table["component"] = table["component"].astype("int64")
    table["polarity"] = table["polarity"].astype("int64")
    table["shape_id"] = table["shape_id"].astype("int64")
    table["existence_significant"] = table["existence_significant"].astype("boolean")
    table.insert(list(MAP_COLUMNS).index("value"), "value", table[parameter].astype(float))
    table.attrs["parameter"] = parameter
    return table


###################################################################################################
## Connectivity export


def connectivity_matrix(
    group_fit: Any,
    parameter: str = "amplitude",
    *,
    long: bool = False,
    component: Any = None,
    existence: Any = None,
) -> Any:
    """(stimulation site x channel) matrix of one parameter, wide or long.

    Wide: rows are the stimulation sites (index ``stim_site``), columns the channels, in
    the order of the GroupFit; NaN where a pair has no fitted component. Long: columns
    :data:`LONG_COLUMNS` plus ``status``. The reported component is chosen as in
    :func:`parameter_map`. ``connectivity_array`` gives the plain array (sites x channels)
    for network tools outside the package.

    >>> import numpy as np
    >>> from evoparam.data import GroupFit, ResponseFit
    >>> fit = ResponseFit(components=[], background={}, gof=None, status="no_response",
    ...                   message="none", window=(0., 1.), baseline=(-1., 0.), settings={})
    >>> wide = connectivity_matrix(GroupFit.from_fits([fit], ["c1"], ["A1-A2"]))
    >>> wide.shape
    (1, 1)
    """
    pd = require("pandas", extra="data", caller="evoparam.spatial.connectivity_matrix")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        table = parameter_map(
            group_fit, parameter=parameter, component=component, existence=existence
        )
    if long:
        out = table[["stim_site", "channel"]].copy()
        out["parameter"] = parameter
        out["value"] = table["value"].to_numpy(float)
        out["status"] = table["status"].to_numpy()
        return out.reset_index(drop=True)
    sites = list(dict.fromkeys(table["stim_site"]))
    channels = list(dict.fromkeys(table["channel"]))
    wide = pd.DataFrame(
        np.nan, index=pd.Index(sites, name="stim_site"), columns=pd.Index(channels, name="channel")
    )
    for site, ch, value in table[["stim_site", "channel", "value"]].itertuples(index=False):
        wide.loc[site, ch] = value
    wide.attrs["parameter"] = parameter
    return wide


def connectivity_to_long(wide: Any, parameter: Any = None) -> Any:
    """Long form of a wide connectivity matrix (every cell, NaN included).

    >>> import pandas as pd
    >>> wide = pd.DataFrame([[1.0, 2.0]], index=pd.Index(["A1-A2"], name="stim_site"),
    ...                     columns=pd.Index(["c1", "c2"], name="channel"))
    >>> connectivity_to_long(wide, "amplitude")["value"].tolist()
    [1.0, 2.0]
    """
    pd = require("pandas", extra="data", caller="evoparam.spatial.connectivity_to_long")
    parameter = parameter or wide.attrs.get("parameter", "")
    rows = [
        {
            "stim_site": str(site),
            "channel": str(ch),
            "parameter": parameter,
            "value": float(wide.loc[site, ch]),
        }
        for site in wide.index
        for ch in wide.columns
    ]
    return pd.DataFrame(rows, columns=list(LONG_COLUMNS))


def connectivity_from_long(long_table: Any) -> Any:
    """Wide matrix from the long form (site order and channel order of first appearance)."""
    pd = require("pandas", extra="data", caller="evoparam.spatial.connectivity_from_long")
    missing = {"stim_site", "channel", "value"} - set(long_table.columns)
    if missing:
        raise DataError(f"connectivity_from_long: missing columns {sorted(missing)}.")
    sites = list(dict.fromkeys(long_table["stim_site"].astype(str)))
    channels = list(dict.fromkeys(long_table["channel"].astype(str)))
    wide = pd.DataFrame(
        np.nan, index=pd.Index(sites, name="stim_site"), columns=pd.Index(channels, name="channel")
    )
    for site, ch, value in long_table[["stim_site", "channel", "value"]].itertuples(index=False):
        wide.loc[str(site), str(ch)] = float(value)
    params = long_table["parameter"].unique() if "parameter" in long_table.columns else []
    if len(params) == 1:
        wide.attrs["parameter"] = str(params[0])
    return wide


def connectivity_array(wide: Any) -> tuple[Any, list[str], list[str]]:
    """``(array [n_sites, n_channels], sites, channels)`` of a wide matrix.

    The layout of a directed (site -> channel) connectivity array for network tools
    (mne-connectivity and others).
    """
    return wide.to_numpy(float), [str(s) for s in wide.index], [str(c) for c in wide.columns]


###################################################################################################
## Figure

_STYLE = {
    "stim_contact": ("*", "black"),
    "bad": ("x", "red"),
    "masked": ("s", "grey"),
    "no_response": ("o", "lightgrey"),
    "failed": ("x", "grey"),
}


# coord_space values (BIDS, lower case) in a template space whose glass brain nilearn draws
MNI_SPACES = ("mni", "icbm", "fsaverage", "talairach")


def _is_mni(table: Any) -> bool:
    spaces = (
        {str(v).lower() for v in table["coord_space"].dropna()} - {"", "nan", "none"}
        if "coord_space" in table.columns
        else set()
    )
    return bool(spaces) and all(space.startswith(MNI_SPACES) for space in spaces)


def _have_nilearn() -> bool:
    import importlib.util

    return importlib.util.find_spec("nilearn") is not None


def plot_parameter_map(
    table: Any,
    *,
    ax: Any = None,
    projection: tuple[str, str] = ("x", "y"),
    backend: str = "auto",
    stim_site: str | None = None,
    display_mode: str = "ortho",
) -> Any:
    """Plot a glass-brain or electrode-grid figure of a :func:`parameter_map` table.

    Spec §5.9, method 3; needs the extra ``plot``. Responses are coloured by ``value``; the
    stimulating contacts, bad, masked, non-responding and failed channels are drawn as distinct
    markers.

    Parameters
    ----------
    table : DataFrame
        Output of :func:`parameter_map` (columns ``x, y, z`` in mm, ``coord_space``,
        ``marker``, ``value``).
    ax : matplotlib Axes, optional
        Axes of the electrode-grid figure (matplotlib backend).
    projection : (str, str)
        Coordinates of the electrode-grid figure's axes.
    backend : {"auto", "nilearn", "matplotlib"}
        ``"nilearn"``: a glass brain (``nilearn.plotting.plot_markers``, then
        ``add_markers`` for the other markers) in the template space of the coordinates,
        which must be MNI-like (``coord_space`` starting with MNI, ICBM, fsaverage or
        Talairach); no template is shipped, nilearn draws its own outline.
        ``"matplotlib"``: the electrode grid in the coordinates as given (subject space
        is honoured). ``"auto"``: nilearn when it is installed and the coordinates are
        MNI-like, else matplotlib.
    stim_site : str, optional
        Draw the rows of this stimulation site only (default: every row).
    display_mode : str
        nilearn display mode (``"ortho"``, ``"lzr"``, ...).

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = require("matplotlib.pyplot", extra="plot", caller="evoparam.spatial.plot_parameter_map")
    if backend not in ("auto", "nilearn", "matplotlib"):
        raise SettingsError("backend must be 'auto', 'nilearn' or 'matplotlib'.")
    if stim_site is not None:
        table = table[table["stim_site"].astype(str) == str(stim_site)]
        if not len(table):
            raise DataError(f"plot_parameter_map: no row of stimulation site {stim_site!r}.")
    if not np.isfinite(table[list(projection)].to_numpy(float)).any():
        raise DataError("plot_parameter_map: the table has no coordinates.")
    mni = _is_mni(table)
    if backend == "nilearn" and not mni:
        raise SettingsError(
            "plot_parameter_map(backend='nilearn'): a glass brain needs "
            "MNI-like coordinates (coord_space); these are in subject space, "
            "use backend='matplotlib' (electrode grid)."
        )
    use_nilearn = backend == "nilearn" or (
        backend == "auto" and mni and _have_nilearn() and ax is None
    )
    if use_nilearn:
        return _glass_brain(table, display_mode)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    a, b = projection
    resp = table[table["marker"] == "response"]
    scatter = ax.scatter(
        resp[a], resp[b], c=resp["value"], cmap="coolwarm", s=40, label="response"
    )
    for marker, (symbol, colour) in _STYLE.items():
        sub = table[table["marker"] == marker]
        if len(sub):
            ax.scatter(sub[a], sub[b], marker=symbol, c=colour, s=30, label=marker)
    space = sorted({str(v) for v in table.get("coord_space", []) if str(v) not in ("", "nan")})
    ax.set_xlabel(f"{a} (mm{', ' + space[0] if len(space) == 1 else ''})")
    ax.set_ylabel(f"{b} (mm)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.figure.colorbar(scatter, ax=ax, label=table.attrs.get("parameter", "value"))
    ax.legend(loc="best", fontsize="small")
    return ax.figure


def _glass_brain(table: Any, display_mode: Any) -> Any:
    """Return the nilearn glass brain of a parameter-map table (responses coloured by value)."""
    plotting = require(
        "nilearn.plotting", extra="brain", caller="evoparam.spatial.plot_parameter_map"
    )
    require("matplotlib", extra="plot", caller="evoparam.spatial.maps._glass_brain")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    xyz = ["x", "y", "z"]
    figure = plt.figure(figsize=(8, 3.2))
    resp = table[(table["marker"] == "response") & np.isfinite(table["value"].to_numpy(float))]
    resp = resp[np.isfinite(resp[xyz].to_numpy(float)).all(axis=1)]
    if len(resp):
        values = resp["value"].to_numpy(float)
        limit = float(np.max(np.abs(values))) or 1.0
        display = plotting.plot_markers(
            values,
            resp[xyz].to_numpy(float),
            node_cmap="coolwarm",
            node_vmin=-limit,
            node_vmax=limit,
            node_size=30,
            display_mode=display_mode,
            figure=figure,
            colorbar=True,
            title=str(table.attrs.get("parameter", "value")),
        )
    else:
        display = plotting.plot_glass_brain(None, display_mode=display_mode, figure=figure)
    for marker, (_, colour) in _STYLE.items():
        sub = table[table["marker"] == marker]
        coords = sub[xyz].to_numpy(float)
        coords = coords[np.isfinite(coords).all(axis=1)]
        if len(coords):
            display.add_markers(coords, marker_color=colour, marker_size=20)
    figure.legend(
        handles=[
            Line2D([], [], ls="", marker="o", color=c, label=m)
            for m, (_, c) in _STYLE.items()
            if (table["marker"] == m).any()
        ],
        loc="lower left",
        fontsize="small",
        frameon=False,
    )
    return figure
