# SPDX-License-Identifier: Apache-2.0
"""Paradigm-based cancellation: alternating polarity and forward masking."""

import time as _time
from dataclasses import replace

import numpy as np

from evoparam.errors import ArtifactError, SettingsError
from evoparam.utils._optional import require

from ._common import check_trialset

__all__ = ["paradigm_subtract", "SCHEMES"]

SCHEMES = ("alternating", "forward_masking")
CONDITIONS = ("A", "B", "C", "D")


def _polarity_sign(value):
    """+1 for anodic(-first), -1 for cathodic(-first), 0 when unknown."""

    text = str(value).lower()
    if "cathod" in text or text in ("-1", "-1.0", "negative"):
        return -1
    if "anod" in text or text in ("1", "1.0", "+1", "positive"):
        return 1
    return 0


def _site_groups(meta):
    if meta is not None and "stim_site" in meta.columns:
        sites = meta["stim_site"].fillna("").astype(str).to_numpy()
    else:
        sites = np.array([""] * (0 if meta is None else len(meta)))
    return {site: np.flatnonzero(sites == site) for site in dict.fromkeys(sites)}


def _pairs_alternating(meta, n_trials, pairing):
    if meta is None or "stim_polarity" not in meta.columns:
        raise ArtifactError("scheme 'alternating' needs metadata['stim_polarity'].")
    signs = np.array([_polarity_sign(v) for v in meta["stim_polarity"]])
    pairs, used = [], np.zeros(n_trials, dtype=bool)
    if pairing == "pair_id" or (
        pairing == "auto" and "pair_id" in meta.columns and meta["pair_id"].notna().all()
    ):
        if "pair_id" not in meta.columns:
            raise ArtifactError("pairing='pair_id' needs metadata['pair_id'].")
        for _, rows in meta.groupby("pair_id", sort=False).indices.items():
            rows = np.asarray(rows)
            pos, neg = rows[signs[rows] > 0], rows[signs[rows] < 0]
            if len(pos) == 1 and len(neg) == 1:
                pairs.append((neg[0], pos[0]))
                used[[neg[0], pos[0]]] = True
    elif pairing == "auto":
        for rows in _site_groups(meta).values():
            pending = None
            for row in rows:
                if signs[row] == 0:
                    continue
                if pending is not None and signs[pending] == -signs[row]:
                    pair = (pending, row) if signs[pending] < 0 else (row, pending)
                    pairs.append(pair)
                    used[list(pair)] = True
                    pending = None
                else:
                    pending = row
    else:
        raise SettingsError("pairing must be 'auto' or 'pair_id'.")
    return pairs, np.flatnonzero(~used)


def _sets_forward_masking(meta, pairing):
    if meta is None or "paradigm_condition" not in meta.columns:
        raise ArtifactError(
            "scheme 'forward_masking' needs metadata['paradigm_condition'] "
            "with the conditions A, B, C and D."
        )
    cond = meta["paradigm_condition"].astype(str).str.upper().to_numpy()
    if pairing == "pair_id" or (pairing == "auto" and "pair_id" in meta.columns):
        keys = meta["pair_id"].to_numpy()
    else:
        # k-th occurrence of each condition forms the k-th set
        keys = np.zeros(len(cond), dtype=int)
        for name in CONDITIONS:
            rows = np.flatnonzero(cond == name)
            keys[rows] = np.arange(len(rows))
    sets, used = [], np.zeros(len(cond), dtype=bool)
    for key in dict.fromkeys(keys):
        rows = np.flatnonzero(keys == key)
        found = {name: rows[cond[rows] == name] for name in CONDITIONS}
        if all(len(found[name]) == 1 for name in CONDITIONS):
            sets.append(tuple(found[name][0] for name in CONDITIONS))
            used[[found[name][0] for name in CONDITIONS]] = True
    return sets, np.flatnonzero(~used)


def paradigm_subtract(ts, *, scheme, pairing="auto"):
    """Cancel the artifact with the stimulation design.

    - ``"alternating"``: trials of opposite ``stim_polarity`` (cathodic and anodic) are
      averaged in pairs, ``x = (x_cathodic + x_anodic) / 2``: the artifact inverts with
      the polarity, the neural response does not. Pairs are formed in trial order within
      each ``stim_site`` (``pairing="auto"``) or by ``metadata["pair_id"]``.
    - ``"forward_masking"``: ``x = A - B + (C - D)`` for each set of the four conditions in
      ``metadata["paradigm_condition"]`` (sets by ``pair_id``, or by order of occurrence).

    Unpaired trials are kept unchanged and flagged ``paired = False``.

    Returns
    -------
    TrialSet
        One trial per pair or set, then the unpaired trials; ``metadata`` gains ``paired``
        (bool) and ``n_combined`` (int).
    """

    pd = require("pandas", extra="data", caller="evoparam.artifact.paradigm_subtract")
    start = _time.perf_counter()
    check_trialset(ts)
    if scheme not in SCHEMES:
        raise SettingsError(f"scheme must be one of {SCHEMES}, not {scheme!r}.")
    meta = ts.metadata
    if scheme == "alternating":
        groups, unpaired = _pairs_alternating(meta, ts.n_trials, pairing)
        combined = [0.5 * (ts.data[a] + ts.data[b]) for a, b in groups]
    else:
        groups, unpaired = _sets_forward_masking(meta, pairing)
        combined = [ts.data[a] - ts.data[b] + (ts.data[c] - ts.data[d]) for a, b, c, d in groups]
    if not groups:
        raise ArtifactError(f"scheme {scheme!r}: no complete pair or set was found.")

    order = [g[0] for g in groups] + list(unpaired)
    data = (
        np.concatenate([np.array(combined), ts.data[unpaired]])
        if len(unpaired)
        else np.array(combined)
    )
    masks = None
    if ts.masks is not None:
        masks = np.array(
            [np.any([ts.masks[r] for r in g], axis=0) for g in groups]
            + [ts.masks[r] for r in unpaired]
        )
    if meta is not None:
        new_meta = meta.iloc[order].reset_index(drop=True).copy()
    else:
        new_meta = pd.DataFrame(index=range(len(order)))
    new_meta["paired"] = [True] * len(groups) + [False] * len(unpaired)
    new_meta["n_combined"] = [len(g) for g in groups] + [1] * len(unpaired)
    event_times = None if ts.event_times is None else ts.event_times[order]
    provenance = ts.provenance.with_step(
        "evoparam.artifact.paradigm_subtract",
        {"scheme": scheme, "pairing": pairing},
        duration_s=_time.perf_counter() - start,
    )
    provenance.data_hash = None
    return replace(
        ts,
        data=data,
        masks=masks,
        metadata=new_meta,
        event_times=event_times,
        provenance=provenance,
    )
