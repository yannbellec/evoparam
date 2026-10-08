"""Time-frequency zones: data-driven parts of the significant map, and published fixed zones.

Data-driven zones (spec §5.10, method 5): the positive and the negative parts of the
significant group map are factorised separately, by non-negative matrix factorisation
(``nmf``: frequency profile x time course, Lee and Seung 1999) or by a non-negative
canonical-polyadic decomposition over (frequency x time x map row) (``cp``; Morup et al.
2006). The rank is the largest one whose factors are reproduced (mean Tucker congruence
>= 0.8) across 20 split halves of the trials (``nmf``) or 20 random restarts (``cp``), up to
``max_zones`` (the search stops after two unstable ranks in a row). Each significant
pixel goes to the zone that explains most of it there; touching zones with no dip between
them are merged.
"""

import json
from importlib import resources

import numpy as np
from scipy.optimize import linear_sum_assignment

from evoparam.errors import SettingsError

__all__ = ["nmf", "cp_nonneg", "select_rank", "zone_regions", "zone_table", "load_zones",
           "resolve_published", "list_published_zones", "ZONE_COLUMNS", "STABILITY_MIN",
           "N_SPLITS"]

STABILITY_MIN = 0.8
N_SPLITS = 20
# Two touching zones are one when the pass between them is at least this fraction of the
# smaller peak.
DIP_FRACTION = 0.5
EPS = 1e-12

# Columns of the zone table and their units.
ZONE_COLUMNS = {
    "zone_id": "", "sign": "", "t_start": "s", "t_end": "s", "f_low": "Hz", "f_high": "Hz",
    "time_centroid": "s", "time_extent": "s", "freq_centroid": "Hz", "freq_extent": "octave",
    "peak_t": "s", "peak_f": "Hz", "peak_value": "normalised power",
    "mean_value": "normalised power", "onset": "s", "offset": "s", "area": "s x octave",
    "energy_fraction": "", "stability": "", "p_value": "", "source": "",
}


def nmf(x, rank, rng, n_iter=400, tol=1e-6):
    """Non-negative factorisation ``x ~ w @ h`` (Frobenius, multiplicative updates).
    Returns (w (m, rank), h (rank, n), relative error)."""

    m, n = x.shape
    scale = np.sqrt(max(x.mean(), EPS) / rank)
    w = rng.uniform(0.1, 1.0, (m, rank)) * scale
    h = rng.uniform(0.1, 1.0, (rank, n)) * scale
    norm = np.linalg.norm(x) + EPS
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


def cp_nonneg(x, rank, rng, n_iter=400, tol=1e-6):
    """Non-negative CP decomposition of a 3-way array ``x`` (I, J, K) by multiplicative
    updates. Returns ([a (I, r), b (J, r), c (K, r)], relative error)."""

    dims = x.shape
    scale = (max(x.mean(), EPS) / rank) ** (1 / 3)
    factors = [rng.uniform(0.1, 1.0, (d, rank)) * scale for d in dims]
    norm = np.linalg.norm(x) + EPS
    unfold = [np.moveaxis(x, mode, 0).reshape(dims[mode], -1) for mode in range(3)]
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


def _parts(w, h):
    """Each zone's (frequency x time) part, flattened."""

    return np.stack([np.outer(w[:, k], h[k]).ravel() for k in range(w.shape[1])])


def congruence(parts_a, parts_b):
    """Mean Tucker congruence (cosine) of matched parts (Hungarian matching)."""

    na = parts_a / (np.linalg.norm(parts_a, axis=1, keepdims=True) + EPS)
    nb = parts_b / (np.linalg.norm(parts_b, axis=1, keepdims=True) + EPS)
    sim = na @ nb.T
    rows, cols = linear_sum_assignment(-sim)
    return float(sim[rows, cols].mean())


def _best_nmf(x, rank, rng, restarts=3):
    best = None
    for _ in range(restarts):
        w, h, err = nmf(x, rank, rng)
        if best is None or err < best[2]:
            best = (w, h, err)
    return best


def select_rank(x, halves, max_zones, rng, method="nmf", tensor=None):
    """Rank and stability per rank.

    ``x`` (n_freqs, n_times) is the non-negative part of the group map; ``halves`` an
    array (n_splits, 2, n_freqs, n_times) of the same part computed on two halves of the
    trials (``nmf``). For ``cp``, ``tensor`` (n_freqs, n_times, n_rows) is decomposed from
    random restarts and their factors compared.
    """

    stability = {}
    chosen, failures = 1, 0
    if not np.any(x > 0):
        return 0, stability
    for rank in range(1, max_zones + 1):
        if method == "nmf":
            if np.count_nonzero(x) < rank:
                break
            values = []
            for split in halves:
                if not (np.any(split[0] > 0) and np.any(split[1] > 0)):
                    values.append(0.0)
                    continue
                wa, ha, _ = nmf(split[0], rank, rng)
                wb, hb, _ = nmf(split[1], rank, rng)
                values.append(congruence(_parts(wa, ha), _parts(wb, hb)))
        else:
            runs = []
            for _ in range(N_SPLITS):
                (a, b, _c), _ = cp_nonneg(tensor, rank, rng)
                runs.append(_parts(a, b.T))
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


def decompose(x, rank, rng, method="nmf", tensor=None):
    """Zone parts (rank, n_freqs, n_times) of a non-negative map, and row loadings for cp."""

    if method == "nmf":
        w, h, _ = _best_nmf(x, rank, rng)
        return np.stack([np.outer(w[:, k], h[k]) for k in range(rank)]), None
    best = None
    for _ in range(3):
        factors, err = cp_nonneg(tensor, rank, rng)
        if best is None or err < best[1]:
            best = (factors, err)
    a, b, c = best[0]
    parts = np.stack([np.outer(a[:, k], b[:, k]) * c[:, k].mean() for k in range(rank)])
    return parts, c


def zone_regions(parts_by_sign, group):
    """Assign each significant pixel to the zone of its sign that explains most of it.

    ``parts_by_sign``: {+1: parts (r, F, T), -1: parts}. Returns a list of
    (sign, part, region bool (F, T)).
    """

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
        out.extend((sign, part, region) for part, region in _merge_without_dip(found,
                                                                               sign * group))
    return out


def _saddle(region_a, region_b, values):
    """Highest pass between two regions: max over neighbouring pixel pairs (one in each)
    of the lower of the two values; -inf when they do not touch."""

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


def _merge_without_dip(found, values):
    """Merge zones that touch without a dip between them: the pass between the two is at
    least half the smaller of their peaks (NMF splits one smooth blob into stable pieces
    when the blob is not separable in time and frequency)."""

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
    return found


def zone_table(zones, group, freqs, times, stability):
    """Tidy zone table from (sign, part, region) triples and the signed group map."""

    import pandas as pd

    rows = []
    logf = np.log2(freqs)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    dlogf = float(np.median(np.diff(logf))) if len(freqs) > 1 else 0.0
    total = np.nansum(np.abs(group))
    order = sorted(range(len(zones)),
                   key=lambda i: -np.nansum(np.abs(group[zones[i][2]])))
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
        rows.append({
            "zone_id": zone_id, "sign": int(sign),
            "t_start": float(times[t_cols.min()]), "t_end": float(times[t_cols.max()]),
            "f_low": float(freqs[f_rows.min()]), "f_high": float(freqs[f_rows.max()]),
            "time_centroid": t_c, "time_extent": t_sd,
            "freq_centroid": float(2 ** lf_c), "freq_extent": lf_sd,
            "peak_t": float(times[ti]), "peak_f": float(freqs[fi]),
            "peak_value": float(group[fi, ti]), "mean_value": float(np.nanmean(values)),
            "onset": float(times[at_centre.min()]), "offset": float(times[at_centre.max()]),
            "area": float(region.sum() * dt * dlogf),
            "energy_fraction": float(np.nansum(np.abs(group[region])) / total) if total else
            np.nan,
            "stability": float(stability.get(int(sign), np.nan)),
            "p_value": np.nan, "source": "data-driven",
        })
    table = pd.DataFrame(rows, columns=list(ZONE_COLUMNS))
    table.attrs["units"] = dict(ZONE_COLUMNS)
    return table


###################################################################################################
## Published fixed zones (compatibility layer)

def load_zones():
    """The registry (``zones.json``): name -> zone definition."""

    text = resources.files("evoparam.tf").joinpath("zones.json").read_text("utf-8")
    return json.loads(text)["zones"]


def list_published_zones():
    """Table of the published zones, with their verification status and source."""

    import pandas as pd

    rows = []
    for name, zone in load_zones().items():
        rows.append({"name": name, "t_start": (zone.get("time") or [np.nan] * 2)[0],
                     "t_end": (zone.get("time") or [np.nan] * 2)[1],
                     "f_low": (zone.get("freq") or [np.nan] * 2)[0],
                     "f_high": (zone.get("freq") or [np.nan] * 2)[1],
                     "verified": bool(zone.get("verified")), "source": zone.get("source", ""),
                     "reference": zone.get("reference", ""), "note": zone.get("note", "")})
    return pd.DataFrame(rows)


def resolve_published(name, allow_unverified=False):
    """The registry entry ``name``.

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
            f"source, then run with allow_unverified=True.")
    if not zone.get("time") or not zone.get("freq"):
        raise SettingsError(
            f"Published zone {name!r} has no bounds in the registry yet "
            f"({zone.get('note', '')}); enter them from {zone.get('source', 'the source')} "
            f"or pass a custom zone.")
    zone["name"] = name
    return zone
