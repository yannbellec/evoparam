# SPDX-License-Identifier: Apache-2.0
"""Time-frequency zones: data-driven parts of the significant map, and published fixed zones.

Data-driven zones (spec §5.10, method 5): the positive and the negative parts of the significant
group map are factorised separately, by non-negative matrix factorisation (``nmf``: frequency
profile x time course, Lee and Seung 1999) or by a non-negative canonical-polyadic decomposition
over (frequency x time x map row) (``cp``; Morup et al. 2006). The rank is the largest one whose
factors are reproduced (mean Tucker congruence >= 0.8) across 20 split halves, up to ``max_zones``
(the search stops after two unstable ranks in a row): halves of the trials, of the group map
(``nmf``) or of each map row (``cp``, which can also split the rows, the sample mode of its tensor,
or compare restarts). Each significant pixel goes to the zone that explains most of it there;
touching zones with no dip between them are merged.
"""

import json
from importlib import resources
from typing import cast

import numpy as np
from scipy.optimize import linear_sum_assignment

from evoparam._typing import Any, NDArray
from evoparam.errors import SettingsError
from evoparam.utils._optional import require

__all__ = [
    "nmf",
    "cp_nonneg",
    "cp_split_values",
    "CP_STABILITY",
    "select_rank",
    "zone_regions",
    "zone_table",
    "load_zones",
    "resolve_published",
    "list_published_zones",
    "ZONE_COLUMNS",
    "STABILITY_MIN",
    "N_SPLITS",
    "zone_mask",
    "volume_conduction_weight",
]

STABILITY_MIN = 0.8
N_SPLITS = 20
# Two touching zones are one when the pass between them is at least this fraction of the
# smaller peak.
DIP_FRACTION = 0.5
EPS = 1e-12

# Columns of the zone table and their units.
ZONE_COLUMNS = {
    "zone_id": "",
    "sign": "",
    "t_start": "s",
    "t_end": "s",
    "f_low": "Hz",
    "f_high": "Hz",
    "time_centroid": "s",
    "time_extent": "s",
    "freq_centroid": "Hz",
    "freq_extent": "octave",
    "peak_t": "s",
    "peak_f": "Hz",
    "peak_value": "normalised power",
    "mean_value": "normalised power",
    "onset": "s",
    "offset": "s",
    "area": "s x octave",
    "energy_fraction": "",
    "stability": "",
    "p_value": "",
    "source": "",
}


def nmf(
    x: Any, rank: Any, rng: np.random.Generator, n_iter: int = 400, tol: float = 1e-6
) -> tuple[Any, Any, Any]:
    """Non-negative factorisation ``x ~ w @ h`` (Frobenius, multiplicative updates).

    Returns (w (m, rank), h (rank, n), relative error).
    """
    m, n = x.shape
    scale = np.sqrt(max(x.mean(), EPS) / rank)
    w = rng.uniform(0.1, 1.0, (m, rank)) * scale
    h = rng.uniform(0.1, 1.0, (rank, n)) * scale
    norm = np.linalg.norm(x) + EPS
    previous: Any
    previous = np.inf
    for it in range(n_iter):
        h *= (w.T @ x) / (w.T @ w @ h + EPS)
        w *= (x @ h.T) / (w @ (h @ h.T) + EPS)
        if it % 20 == 19:
            err = np.linalg.norm(x - w @ h) / norm
            if previous - err < tol:
                break
            previous = err
    err = np.linalg.norm(x - w @ h) / norm
    # scale so that each frequency profile has unit maximum
    peak = w.max(axis=0) + EPS
    return w / peak, h * peak[:, None], err


def cp_nonneg(
    x: Any, rank: Any, rng: np.random.Generator, n_iter: int = 400, tol: float = 1e-6
) -> tuple[list[Any], Any]:
    """Non-negative CP decomposition of a 3-way array ``x`` (I, J, K) by multiplicative updates.

    Returns ([a (I, r), b (J, r), c (K, r)], relative error).
    """
    dims = x.shape
    scale = (max(x.mean(), EPS) / rank) ** (1 / 3)
    factors = [rng.uniform(0.1, 1.0, (d, rank)) * scale for d in dims]
    norm = np.linalg.norm(x) + EPS
    unfold = [np.moveaxis(x, mode, 0).reshape(dims[mode], -1) for mode in range(3)]
    previous: Any
    previous = np.inf
    for it in range(n_iter):
        for mode in range(3):
            others = [factors[k] for k in range(3) if k != mode]
            kr = (others[0][:, None, :] * others[1][None, :, :]).reshape(-1, rank)
            gram = (others[0].T @ others[0]) * (others[1].T @ others[1])
            factors[mode] *= (unfold[mode] @ kr) / (factors[mode] @ gram + EPS)
        if it % 20 == 19:
            err = np.linalg.norm(x - np.einsum("ir,jr,kr->ijk", *factors)) / norm
            if previous - err < tol:
                break
            previous = err
    err = np.linalg.norm(x - np.einsum("ir,jr,kr->ijk", *factors)) / norm
    return factors, err


def _parts(w: Any, h: Any) -> NDArray[Any]:
    """Each zone's (frequency x time) part, flattened."""
    return np.stack([np.outer(w[:, k], h[k]).ravel() for k in range(w.shape[1])])


def congruence(parts_a: Any, parts_b: Any) -> float:
    """Mean Tucker congruence (cosine) of matched parts (Hungarian matching)."""
    na = parts_a / (np.linalg.norm(parts_a, axis=1, keepdims=True) + EPS)
    nb = parts_b / (np.linalg.norm(parts_b, axis=1, keepdims=True) + EPS)
    sim = na @ nb.T
    rows, cols = linear_sum_assignment(-sim)
    return float(sim[rows, cols].mean())


def _best_nmf(x: Any, rank: Any, rng: np.random.Generator, restarts: int = 3) -> Any:
    best = None
    for _ in range(restarts):
        w, h, err = nmf(x, rank, rng)
        if best is None or err < best[2]:
            best = (w, h, err)
    return best


def _cp_parts(tensor: Any, rank: Any, rng: np.random.Generator) -> NDArray[Any]:
    (a, b, _c), _ = cp_nonneg(tensor, rank, rng)
    return _parts(a, b.T)


CP_STABILITY = ("split_half", "rows", "restarts")


def _cp_congruence(first: Any, second: Any, rank: Any, rng: np.random.Generator) -> float:
    if not (np.any(first > 0) and np.any(second > 0)):
        return 0.0
    return congruence(_cp_parts(first, rank, rng), _cp_parts(second, rank, rng))


def cp_split_values(
    tensor: Any,
    halves: Any,
    rank: Any,
    rng: np.random.Generator,
    row_halves: Any = None,
    split: str = "trials",
) -> Any:
    """Split-half congruences of a rank-``rank`` CP (frequency x time x row).

    ``split="trials"`` (spec §5.10): the tensors of two halves of each row's trials,
    ``row_halves`` (n_splits, 2, n_freqs, n_times, n_rows; the same random halves as for
    NMF), are decomposed and their matched (frequency x time) parts compared. Without
    ``row_halves``, or with ``split="rows"``: the rows that carry the map (the sample mode
    of ``tensor``) are split at random into two halves N_SPLITS times (split-half analysis
    of PARAFAC models, Bro 1997); with fewer than two such rows, the trial halves of the
    group map (``halves``), each a one-row tensor; without them, None.
    """
    if split == "trials" and row_halves is not None:
        return [_cp_congruence(pair[0], pair[1], rank, rng) for pair in row_halves]
    active = np.flatnonzero(tensor.reshape(-1, tensor.shape[-1]).any(axis=0))
    if active.size >= 2:
        values = []
        for _ in range(N_SPLITS):
            order = rng.permutation(active)
            first, second = order[: active.size // 2], order[active.size // 2 :]
            values.append(_cp_congruence(tensor[..., first], tensor[..., second], rank, rng))
        return values
    if halves is None:
        return None
    return [_cp_congruence(pair[0][..., None], pair[1][..., None], rank, rng) for pair in halves]


def select_rank(
    x: Any,
    halves: Any,
    max_zones: Any,
    rng: np.random.Generator,
    method: str = "nmf",
    tensor: Any = None,
    cp_stability: str = "split_half",
    row_halves: Any = None,
) -> Any:
    """Rank and stability per rank.

    ``x`` (n_freqs, n_times) is the non-negative part of the group map; ``halves`` an
    array (n_splits, 2, n_freqs, n_times) of the same part computed on two halves of the
    trials (``nmf``). For ``cp``, ``tensor`` (n_freqs, n_times, n_rows) is compared over
    split halves (:func:`cp_split_values`): of each row's trials (``row_halves``,
    ``cp_stability="split_half"``, the spec's rule), of the rows (``"rows"``), or, as in
    the first version, across random restarts of the whole tensor (``"restarts"``; also
    used when no split is possible).
    """
    if cp_stability not in CP_STABILITY:
        raise SettingsError(f"cp_stability must be one of {CP_STABILITY}, not {cp_stability!r}.")
    stability: dict[Any, Any]
    stability = {}
    chosen, failures = 1, 0
    if not np.any(x > 0):
        return 0, stability
    for rank in range(1, max_zones + 1):
        if method == "nmf":
            if np.count_nonzero(x) < rank:
                break
            values: Any
            values = []
            for split in halves:
                if not (np.any(split[0] > 0) and np.any(split[1] > 0)):
                    values.append(0.0)
                    continue
                wa, ha, _ = nmf(split[0], rank, rng)
                wb, hb, _ = nmf(split[1], rank, rng)
                values.append(congruence(_parts(wa, ha), _parts(wb, hb)))
        else:
            values = (
                None
                if cp_stability == "restarts"
                else cp_split_values(
                    tensor,
                    halves,
                    rank,
                    rng,
                    row_halves=row_halves,
                    split="rows" if cp_stability == "rows" else "trials",
                )
            )
            if values is None:
                runs = [_cp_parts(tensor, rank, rng) for _ in range(N_SPLITS)]
                values = [congruence(runs[0], other) for other in runs[1:]]
        stability[rank] = float(np.mean(values))
        if stability[rank] >= STABILITY_MIN:
            chosen, failures = rank, 0
        else:
            # one rank can be unstable while the next is not (two parts of similar weight:
            # a single factor picks either), so stop only after two failures in a row
            failures += 1
            if failures >= 2:
                break
    return chosen, stability


def decompose(
    x: Any, rank: Any, rng: np.random.Generator, method: str = "nmf", tensor: Any = None
) -> Any:
    """Zone parts (rank, n_freqs, n_times) of a non-negative map, and row loadings for cp."""
    if method == "nmf":
        w, h, _ = _best_nmf(x, rank, rng)
        return np.stack([np.outer(w[:, k], h[k]) for k in range(rank)]), None
    best: Any = None
    for _ in range(3):
        factors, err = cp_nonneg(tensor, rank, rng)
        if best is None or err < best[1]:
            best = (factors, err)
    a, b, c = best[0]
    parts = np.stack([np.outer(a[:, k], b[:, k]) * c[:, k].mean() for k in range(rank)])
    return parts, c


def zone_regions(parts_by_sign: Any, group: Any) -> list[Any]:
    """Assign each significant pixel to the zone of its sign that explains most of it.

    ``parts_by_sign``: {+1: parts (r, F, T), -1: parts}. Returns a list of
    (sign, part, region bool (F, T)).
    """
    out: list[Any]
    out = []
    for sign, parts in parts_by_sign.items():
        if parts is None or not len(parts):
            continue
        support = (sign * group) > 0
        owner = np.argmax(parts, axis=0)
        found = []
        for k, part in enumerate(parts):
            region = support & (owner == k) & (part > 0)
            if region.any():
                found.append((part, region))
        out.extend(
            (sign, part, region) for part, region in _merge_without_dip(found, sign * group)
        )
    return out


def _saddle(region_a: Any, region_b: Any, values: Any) -> float:
    """Highest pass between two regions.

    Max over neighbouring pixel pairs (one in each) of the lower of the two values; -inf when they
    do not touch.
    """
    best = -np.inf
    for axis in (0, 1):
        for shift in (1, -1):
            nb = np.roll(region_b, shift, axis=axis)
            vb = np.roll(values, shift, axis=axis)
            edge = np.ones(region_b.shape, bool)
            if axis == 0:
                edge[0 if shift == 1 else -1, :] = False
            else:
                edge[:, 0 if shift == 1 else -1] = False
            pair = region_a & nb & edge
            if pair.any():
                best = max(best, float(np.max(np.minimum(values[pair], vb[pair]))))
    return best


def _merge_without_dip(found: Any, values: Any) -> list[Any]:
    """Merge zones that touch without a dip between them.

    The pass between the two is at least half the smaller of their peaks (NMF splits one smooth
    blob into stable pieces when the blob is not separable in time and frequency).
    """
    found = list(found)
    merged = True
    while merged and len(found) > 1:
        merged = False
        for a in range(len(found)):
            for b in range(a + 1, len(found)):
                (pa, ra), (pb, rb) = found[a], found[b]
                peak = min(values[ra].max(), values[rb].max())
                if _saddle(ra, rb, values) >= DIP_FRACTION * peak:
                    found[a] = (pa + pb, ra | rb)
                    del found[b]
                    merged = True
                    break
            if merged:
                break
    return cast("list[Any]", found)


def zone_table(zones: Any, group: Any, freqs: Any, times: Any, stability: Any) -> Any:
    """Tidy zone table from (sign, part, region) triples and the signed group map."""
    require("pandas", extra="data", caller="evoparam.tf.zones.zone_table")
    import pandas as pd

    rows = []
    logf = np.log2(freqs)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    dlogf = float(np.median(np.diff(logf))) if len(freqs) > 1 else 0.0
    total = np.nansum(np.abs(group))
    order = sorted(range(len(zones)), key=lambda i: -np.nansum(np.abs(group[zones[i][2]])))
    for zone_id, i in enumerate(order):
        sign, part, region = zones[i]
        weight = np.where(region, part, 0.0)
        wsum = weight.sum() + EPS
        wt = weight.sum(0)
        wf = weight.sum(1)
        t_c = float(np.sum(wt * times) / wsum)
        t_sd = float(np.sqrt(np.sum(wt * (times - t_c) ** 2) / wsum))
        lf_c = float(np.sum(wf * logf) / wsum)
        lf_sd = float(np.sqrt(np.sum(wf * (logf - lf_c) ** 2) / wsum))
        values = np.where(region, group, np.nan)
        fi, ti = np.unravel_index(np.nanargmax(sign * values), values.shape)
        f_rows, t_cols = np.nonzero(region)
        row_c = int(np.argmin(np.abs(logf - lf_c)))
        at_centre = np.flatnonzero(region[row_c])
        if not at_centre.size:
            at_centre = t_cols
        rows.append(
            {
                "zone_id": zone_id,
                "sign": int(sign),
                "t_start": float(times[t_cols.min()]),
                "t_end": float(times[t_cols.max()]),
                "f_low": float(freqs[f_rows.min()]),
                "f_high": float(freqs[f_rows.max()]),
                "time_centroid": t_c,
                "time_extent": t_sd,
                "freq_centroid": float(2**lf_c),
                "freq_extent": lf_sd,
                "peak_t": float(times[ti]),
                "peak_f": float(freqs[fi]),
                "peak_value": float(group[fi, ti]),
                "mean_value": float(np.nanmean(values)),
                "onset": float(times[np.min(at_centre)]),
                "offset": float(times[np.max(at_centre)]),
                "area": float(region.sum() * dt * dlogf),
                "energy_fraction": float(np.nansum(np.abs(group[region])) / total)
                if total
                else np.nan,
                "stability": float(stability.get(int(sign), np.nan)),
                "p_value": np.nan,
                "source": "data-driven",
            }
        )
    table = pd.DataFrame(rows, columns=list(ZONE_COLUMNS))
    table.attrs["units"] = dict(ZONE_COLUMNS)
    return table


###################################################################################################
## Published fixed zones (compatibility layer)


def load_zones() -> dict[Any, Any]:
    """Return the registry (``zones.json``).

    Name -> zone definition, with aliases (``alias_of``) and shared method blocks (``settings_of``:
    source, reference, note, field status and settings common to the zones of one publication)
    resolved.
    """
    text = resources.files("evoparam.tf").joinpath("zones.json").read_text("utf-8")
    raw = json.loads(text)
    zones = {}
    for name, raw_zone in raw["zones"].items():
        zone = dict(raw_zone)
        if "alias_of" in zone:
            target = zone["alias_of"]
            zone = {**raw["zones"][target], "alias_of": target}
        if "settings_of" in zone:
            zone = {**raw[zone["settings_of"]], **zone}
        zones[name] = zone
    return zones


def _interval(values: Any, bounds: Any, closure: Any) -> Any:
    lo, hi = bounds
    upper = values < hi if closure == "[)" else values <= hi
    return (values >= lo) & upper


def zone_mask(zone: Any, freqs: Any, times: Any) -> Any:
    """Return the pixels (n_freqs, n_times) of a registry zone, with its interval closures.

    ``intervals``: ``"[]"`` closed, the default, or ``"[)"`` half-open.
    """
    closure = zone.get("intervals") or {}
    in_f = _interval(np.asarray(freqs, float), zone["freq"], closure.get("freq", "[]"))
    in_t = _interval(np.asarray(times, float), zone["time"], closure.get("time", "[]"))
    return in_f[:, None] & in_t[None, :]


def volume_conduction_weight(
    distance_mm: Any, d_min: float = 10.0, d_max: float = 20.0
) -> NDArray[Any]:
    """Weight of a stimulation-response pair against volume conduction ([LIT-tf-4], §2.5).

    0 within ``d_min`` mm of the stimulated pair, 1 beyond ``d_max`` mm, and
    ``(d^2 - d_min^2) / (d_max^2 - d_min^2)`` in between; multiplies a zone's significance
    fraction.

    Examples
    --------
    >>> float(volume_conduction_weight(15.0))
    0.4166666666666667
    """
    d = np.asarray(distance_mm, float)
    w = np.clip((d**2 - d_min**2) / (d_max**2 - d_min**2), 0.0, 1.0)
    return np.where(np.isfinite(d), w, np.nan)


def list_published_zones() -> Any:
    """Table of the published zones, with their verification status and source."""
    require("pandas", extra="data", caller="evoparam.tf.zones.list_published_zones")
    import pandas as pd

    rows = []
    for name, zone in load_zones().items():
        rows.append(
            {
                "name": name,
                "t_start": (zone.get("time") or [np.nan] * 2)[0],
                "t_end": (zone.get("time") or [np.nan] * 2)[1],
                "f_low": (zone.get("freq") or [np.nan] * 2)[0],
                "f_high": (zone.get("freq") or [np.nan] * 2)[1],
                "intervals": "x".join(
                    (zone.get("intervals") or {}).get(k, "[]") for k in ("time", "freq")
                ),
                "label": zone.get("label", ""),
                "alias_of": zone.get("alias_of", ""),
                "verified": bool(zone.get("verified")),
                "source": zone.get("source", ""),
                "reference": zone.get("reference", ""),
                "note": zone.get("note", ""),
            }
        )
    return pd.DataFrame(rows)


def resolve_published(name: Any, allow_unverified: bool = False) -> dict[Any, Any]:
    """Return the registry entry ``name``.

    Raises
    ------
    SettingsError
        Unknown zone; an ``unverified`` zone without ``allow_unverified=True`` (naming the
        zone and its source); a zone whose bounds are not entered in the registry.
    """
    registry = load_zones()
    if name not in registry:
        raise SettingsError(f"Unknown published zone {name!r}; registered: {sorted(registry)}.")
    zone = dict(registry[name])
    if not zone.get("verified") and not allow_unverified:
        raise SettingsError(
            f"Published zone {name!r} is unverified (source: {zone.get('source', '?')}; "
            f"{zone.get('reference', 'no page reference')}): check its bounds against the "
            f"source, then run with allow_unverified=True."
        )
    if not zone.get("time") or not zone.get("freq"):
        raise SettingsError(
            f"Published zone {name!r} has no bounds in the registry yet "
            f"({zone.get('note', '')}); enter them from {zone.get('source', 'the source')} "
            f"or pass a custom zone."
        )
    zone["name"] = name
    return zone
