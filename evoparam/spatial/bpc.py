# SPDX-License-Identifier: Apache-2.0
"""Shared response shapes across stimulation sites or channels (spec §5.9, method 1).

Re-implemented from the published description of the basis-profile-curve method (Miller,
Müller and Hermes, 2021, PLOS Computational Biology); the reference implementation is GPL
and is neither imported nor copied. For one recording channel and many stimulation sites
(or one site and many channels), the trials ``V_{m,i}`` of each group ``m`` are projected
into one another, ``P = <V_k / ||V_k||, V_l>``; the projections between two groups give a
t-value, the positive part of the matrix of t-values is clustered by non-negative matrix
factorisation, and each cluster's basis curve is the first right singular vector of its
trials. Each trial is then ``V = alpha B_q + residual``.
"""

import itertools
import re
import time as _time

import numpy as np

from evoparam.data import ShapeBasis, TrialSet
from evoparam.errors import DataError, SettingsError
from evoparam.errors import warn as _warn
from evoparam.models.component_shapes import Template
from evoparam.utils._optional import require
from evoparam.version import __version__

__all__ = [
    "shared_shapes",
    "convergent",
    "divergent",
    "as_template",
    "adjusted_rand_index",
    "BASIS_COLUMNS",
]

# Columns of ShapeBasis.assignment, one row per group (stimulation site or channel).
BASIS_COLUMNS = ("group", "shape_id", "weight", "r2", "n_trials", "used", "nmf_loading")

MIN_TRIALS = 3
MIN_GROUPS = 3
SPLIT_MAX = 0.75  # largest between / within ratio of the mean t-values of two clusters
TOLERANCE = 0.5  # inclusion: at least half the equal-contribution loading
NMF_TOL = 1e-5
NMF_MAX_ITER = 1000


###################################################################################################
## Clustering tools


def adjusted_rand_index(a, b):
    """Return the adjusted Rand index of two labelings (1 = identical partitions).

    >>> adjusted_rand_index([0, 0, 1, 1], [1, 1, 0, 0])
    1.0
    """
    a, b = np.asarray(a), np.asarray(b)
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    table = np.zeros((ia.max() + 1, ib.max() + 1))
    np.add.at(table, (ia, ib), 1)

    def comb(x):
        return x * (x - 1) / 2.0

    index = comb(table).sum()
    rows, cols = comb(table.sum(axis=1)).sum(), comb(table.sum(axis=0)).sum()
    total = comb(float(len(a)))
    expected = rows * cols / total if total else 0.0
    maximum = 0.5 * (rows + cols)
    if maximum == expected:
        return 1.0
    return float((index - expected) / (maximum - expected))


def _nmf(x, q, rng, n_runs=1):
    """Factorise ``x ~ W H`` with W, H >= 0 and the rows of H of unit norm.

    For ``n_runs`` random starts at once (multiplicative updates; the norms of H are folded into W
    at each step, as in the published description). Stops a run when its relative change of error
    is below 1e-5. Returns W [runs, n, q], H [runs, q, m] and the errors [runs].
    """
    n, m = x.shape
    w = rng.uniform(0, 1, (n_runs, n, q))
    h = rng.uniform(0, 1, (n_runs, q, m))
    eps = 1e-12
    previous = np.full(n_runs, np.inf)
    active = np.ones(n_runs, bool)
    error = np.full(n_runs, np.inf)
    for step in range(NMF_MAX_ITER):
        wa, ha = w[active], h[active]
        wt = np.transpose(wa, (0, 2, 1))
        ha = ha * (wt @ x) / (wt @ wa @ ha + eps)
        norms = np.linalg.norm(ha, axis=2)
        norms[norms == 0] = 1.0
        ha = ha / norms[:, :, None]
        wa = wa * norms[:, None, :]
        ht = np.transpose(ha, (0, 2, 1))
        wa = wa * (x @ ht) / (wa @ ha @ ht + eps)
        w[active], h[active] = wa, ha
        err = np.sum((x[None] - wa @ ha) ** 2, axis=(1, 2))
        index = np.flatnonzero(active)
        error[index] = err
        if step > 10:
            done = np.abs(previous[index] - err) <= NMF_TOL * np.maximum(err, eps)
            active[index[done]] = False
            if not active.any():
                break
        previous[index] = err
    return w, h, error


def _assign(h):
    """Winner-take-all on the columns of H.

    A group whose largest loading is below half the equal-contribution loading 1/sqrt(n) belongs to
    no cluster (-1).
    """
    n = h.shape[1]
    best = np.argmax(h, axis=0)
    loading = h[best, np.arange(n)]
    return np.where(loading >= TOLERANCE / np.sqrt(n), best, -1), loading


def _degeneracy(h):
    if len(h) < 2:
        return 0.0
    unit = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-300)
    overlap = unit @ unit.T
    return float(np.max(overlap[np.triu_indices(len(h), 1)]))


def _kmeans(x, q, rng, n_iter=100):
    """k-means++ on the rows of x; returns labels."""
    centres = [x[rng.integers(len(x))]]
    for _ in range(1, q):
        d2 = np.min([np.sum((x - c) ** 2, axis=1) for c in centres], axis=0)
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(len(x), 1 / len(x))
        centres.append(x[rng.choice(len(x), p=p)])
    centres = np.array(centres)
    labels = np.zeros(len(x), int)
    for _ in range(n_iter):
        d = np.sum((x[:, None, :] - centres[None]) ** 2, axis=2)
        new = np.argmin(d, axis=1)
        for k in range(q):
            if np.any(new == k):
                centres[k] = x[new == k].mean(axis=0)
        if np.array_equal(new, labels):
            break
        labels = new
    return labels


def _jaccard_stability(labels, others, q):
    """Return the stability of each cluster of ``labels``.

    Mean over the other runs of the best Jaccard index with one of their clusters.
    """
    out = np.zeros(q)
    for k in range(q):
        mine = labels == k
        if not mine.any():
            continue
        scores = []
        for other in others:
            best = 0.0
            for j in np.unique(other[other >= 0]):
                theirs = other == j
                best = max(best, np.sum(mine & theirs) / np.sum(mine | theirs))
            scores.append(best)
        out[k] = float(np.mean(scores)) if scores else 1.0
    return out


def _split_ratio(xi, labels, q):
    """Return the largest ratio of the between-cluster to the within-cluster mean t-values.

    Over pairs of clusters, the mean t-value between them divided by the geometric mean of the mean
    t-values within them: about 1 when one cluster has been split in two, well below 1 for distinct
    shapes (even partly correlated ones).
    """
    within = [
        xi[np.ix_(labels == k, labels == k)].mean() if np.any(labels == k) else 0.0
        for k in range(q)
    ]
    ratio = 0.0
    for a, b in itertools.combinations(range(q), 2):
        if within[a] > 0 and within[b] > 0:
            between = 0.5 * (
                xi[np.ix_(labels == a, labels == b)].mean()
                + xi[np.ix_(labels == b, labels == a)].mean()
            )
            ratio = max(ratio, between / np.sqrt(within[a] * within[b]))
    return float(ratio)


def _cluster_nmf(xi, q, n_restarts, rng):
    ws, hs, errors = _nmf(xi, q, rng, n_restarts)
    labels = [_assign(h)[0] for h in hs]
    best = int(np.argmin(errors))
    return labels[best], labels, hs[best], float(errors[best]), _degeneracy(hs[best]), ws[best]


def _cluster_kpca(xi, q, n_restarts, rng):
    """Cluster with a kernel PCA (cosine kernel of the groups' t-value profiles) and k-means.

    The k-means runs in the first ``q`` components; groups without any positive t-value belong to
    no cluster.
    """
    profiles = 0.5 * (xi + xi.T)
    norms = np.linalg.norm(profiles, axis=1)
    keep = norms > 1e-12
    unit = profiles[keep] / norms[keep, None]
    kernel = unit @ unit.T
    n = len(kernel)
    centring = np.eye(n) - 1.0 / n
    vals, vecs = np.linalg.eigh(centring @ kernel @ centring)
    order = np.argsort(vals)[::-1][: max(q, 1)]
    coords = vecs[:, order] * np.sqrt(np.clip(vals[order], 0, None))
    labels_all = []
    for _ in range(n_restarts):
        lab = np.full(len(xi), -1)
        lab[keep] = _kmeans(coords, q, rng) if q > 1 else 0
        labels_all.append(lab)
    # The run with the smallest within-cluster spread is kept
    spread = []
    for lab in labels_all:
        sub = lab[keep]
        spread.append(
            sum(
                np.sum((coords[sub == k] - coords[sub == k].mean(axis=0)) ** 2)
                for k in range(q)
                if np.any(sub == k)
            )
        )
    best = int(np.argmin(spread))
    h = np.zeros((q, len(xi)))
    for k in range(q):
        h[k, labels_all[best] == k] = 1.0
    return labels_all[best], labels_all, h, float(spread[best]), 0.0, None


###################################################################################################
## Projections


def _t_matrix(trials, owner, n_groups):
    """t-value of the projections between every pair of groups (self-projections left out).

    ``P[k, l] = <V_k / ||V_k||, V_l>``: the projection of trial l into the unit-norm trial k.
    """
    norms = np.linalg.norm(trials, axis=1)
    unit = trials / np.maximum(norms, 1e-300)[:, None]
    proj = unit @ trials.T
    xi = np.zeros((n_groups, n_groups))
    for a in range(n_groups):
        rows = np.flatnonzero(owner == a)
        for b in range(n_groups):
            cols = np.flatnonzero(owner == b)
            block = proj[np.ix_(rows, cols)]
            if a == b:
                block = block[~np.eye(len(rows), dtype=bool)]
            values = np.ravel(block)
            if len(values) < 2:
                continue
            sd = np.std(values, ddof=1)
            xi[a, b] = np.mean(values) / (sd / np.sqrt(len(values))) if sd > 0 else 0.0
    return xi


def _cross_t(trials, owner, n_groups, others):
    """Return the t-value of the projections of the trials ``others`` [n, n_times].

    ``others`` is one group that does not define the bases; its trials are projected into the
    unit-norm trials of each defining group.
    """
    norms = np.linalg.norm(trials, axis=1)
    unit = trials / np.maximum(norms, 1e-300)[:, None]
    proj = unit @ others.T
    out = np.zeros(n_groups)
    for a in range(n_groups):
        values = np.ravel(proj[owner == a])
        sd = np.std(values, ddof=1) if len(values) > 1 else 0.0
        out[a] = np.mean(values) / (sd / np.sqrt(len(values))) if sd > 0 else 0.0
    return out


def _basis_curve(trials):
    """Return the first right singular vector of the trials.

    It is signed so that the mean projection is positive.
    """
    _, _, vt = np.linalg.svd(trials, full_matrices=False)
    curve = vt[0]
    if np.mean(trials @ curve) < 0:
        curve = -curve
    return curve


###################################################################################################
## Groups


def _contacts(site):
    return [part.strip() for part in re.split(r"[-–]", str(site)) if part.strip()]


def _usable_trials(ts, rows, channel, win):
    """Trials of one channel in the window, without trials containing NaN or rejected."""
    data = ts.data[rows][:, channel][:, win]
    ok = np.isfinite(data).all(axis=1)
    if ts.masks is not None:
        ok &= ~ts.masks[rows][:, channel][:, win].any(axis=1)
    return data[ok], rows[ok]


def _groups(ts, by, channel, stim_site, win):
    """(label, trials [n, n_times], trial indices) per group."""
    meta = ts.metadata
    rows_all = np.arange(ts.n_trials)
    if meta is not None and "rejected" in meta.columns:
        rows_all = rows_all[~meta["rejected"].fillna(False).to_numpy(bool)]
    names = list(ts.ch_names)
    if by == "stim_site":
        if meta is None or "stim_site" not in meta.columns:
            raise DataError("shared_shapes(by='stim_site') needs metadata['stim_site'].")
        if channel is None:
            if len(names) != 1:
                raise SettingsError(
                    "shared_shapes(by='stim_site'): give channel= (the recording channel)."
                )
            channel = names[0]
        ch = names.index(channel) if isinstance(channel, str) else int(channel)
        sites = meta["stim_site"].astype(str).to_numpy()
        out = []
        for site in dict.fromkeys(sites[rows_all]):
            if names[ch] in _contacts(site) or site in ("", "n/a", "nan"):
                continue
            trials, rows = _usable_trials(ts, rows_all[sites[rows_all] == site], ch, win)
            if len(trials):
                out.append((site, trials, rows))
        return out, {"channel": names[ch], "stim_site": None}
    if by == "channel":
        rows = rows_all
        if meta is not None and "stim_site" in meta.columns:
            sites = meta["stim_site"].astype(str).to_numpy()
            if stim_site is None:
                present = list(dict.fromkeys(sites[rows_all]))
                if len(present) > 1:
                    raise SettingsError(
                        "shared_shapes(by='channel'): give stim_site= (the "
                        "trials hold several sites)."
                    )
                stim_site = present[0] if present else None
            if stim_site is not None:
                rows = rows_all[sites[rows_all] == str(stim_site)]
        if len(rows) == 0:
            raise DataError(f"No trial of stimulation site {stim_site!r}.")
        excluded = set(_contacts(stim_site)) if stim_site is not None else set()
        info = ts.ch_info
        bad = set()
        if info is not None and "bad" in info.columns:
            bad = {
                n for n, b in zip(info["name"].astype(str), info["bad"], strict=False) if bool(b)
            }
        out = []
        for ch, name in enumerate(names):
            if name in excluded or name in bad:
                continue
            trials, used = _usable_trials(ts, rows, ch, win)
            if len(trials):
                out.append((name, trials, used))
        return out, {"channel": None, "stim_site": None if stim_site is None else str(stim_site)}
    raise SettingsError(f"by must be 'stim_site' or 'channel', not {by!r}.")


###################################################################################################
## Main entry point


def shared_shapes(  # noqa: C901, PLR0912, PLR0913, PLR0915  (public API, signature fixed by spec §5.9)
    ts,
    *,
    by="stim_site",
    channel=None,
    stim_site=None,
    window=(0.009, 0.5),
    max_shapes=6,
    method="nmf",
    n_restarts=20,
    stability_min=0.8,
    random_state=None,
    n_jobs=1,
):
    """Find the few response shapes shared across stimulation sites (or channels).

    ``by="stim_site"``: the groups are the stimulation sites recorded on one ``channel``
    (sites that include the channel are left out): the channel's input repertoire.
    ``by="channel"``: the groups are the channels for one ``stim_site`` (its contacts and
    bad channels are left out). Trials with NaN or masked samples in the window, and
    trials with ``metadata["rejected"]``, are left out.

    1. Projections between all trials, ``P[k, l] = <V_k / ||V_k||, V_l>`` on ``window``;
       per pair of groups, the t-value of their projections (self-projections left out).
    2. The positive part of the t-matrix, scaled to a maximum of 1, is clustered by
       non-negative matrix factorisation (``method="nmf"``) or by kernel PCA with a cosine
       kernel and k-means (``"kpca"``). A group belongs to the cluster of its largest
       loading, or to none when that loading is below half the equal-contribution loading.
    3. The number of shapes is the largest ``q <= max_shapes`` whose clustering uses all
       ``q`` clusters, splits no cluster (for every pair of clusters, the mean t-value
       between them is below 0.75 times the geometric mean of the mean t-values within
       them) and is stable: mean adjusted Rand index of the assignments of ``n_restarts``
       random restarts ``>= stability_min``. The overlap of the rows of H (the degeneracy
       index of the published method) is reported in the selection table.
    4. Each cluster's curve is the first right singular vector of its trials (groups with
       at least 3 trials); every group is then projected: ``alpha = <B_q, V>`` per trial,
       its mean is the group's weight and ``R² = 1 - sum ||V - alpha B_q||² / sum ||V||²``.

    Groups with fewer than 3 trials are projected but do not define bases (they take the
    shape with the largest R²). When no ``q >= 2`` is stable although some are
    non-degenerate, ``n_shapes = 1`` with the warning ``"unstable_basis"``.

    Parameters
    ----------
    ts : TrialSet
    by : {"stim_site", "channel"}
    channel : str or int, optional
        Recording channel for ``by="stim_site"`` (optional with one channel).
    stim_site : str, optional
        Stimulation site for ``by="channel"`` (optional when the trials hold one site).
    window : tuple of float
        ``(t0, tau)`` in s.
    max_shapes, method, n_restarts, stability_min, random_state
        See above. ``n_jobs`` is accepted for the API; the restarts are fast and run in
        one thread.

    Returns
    -------
    ShapeBasis
        ``weights`` [n_groups, n_shapes] are the non-negative clustering loadings (spec
        §4.3.4: the columns of the NMF factor H, rows of unit norm; 0/1 cluster indicators
        for ``"kpca"``; for a group with fewer than 3 trials, the non-negative least-squares
        loadings of its t-values on the factor W); ``assignment`` has the columns of
        :data:`BASIS_COLUMNS`: ``weight`` is the group-level weight ``alpha_m``, the mean
        projection of the group's trials on its shape (data units times sqrt(samples),
        signed), and ``nmf_loading`` the group's loading on that shape; ``settings``
        holds the groups, the t-matrix and the selection table.

    Raises
    ------
    DataError
        Fewer than 3 groups with at least 3 trials.
    """
    pd = require("pandas", extra="data", caller="evoparam.spatial.shared_shapes")
    start = _time.perf_counter()
    if not isinstance(ts, TrialSet):
        raise DataError("shared_shapes expects a TrialSet.")
    if method not in ("nmf", "kpca"):
        raise SettingsError(f"method must be 'nmf' or 'kpca', not {method!r}.")
    if max_shapes < 1 or n_restarts < 2:
        raise SettingsError("max_shapes must be >= 1 and n_restarts >= 2.")
    if not 0 < stability_min <= 1:
        raise SettingsError("stability_min must be in (0, 1].")
    win = (ts.times >= window[0]) & (ts.times <= window[1])
    if win.sum() < 4:
        raise DataError(f"The window {window} holds fewer than 4 samples.")
    groups, where = _groups(ts, by, channel, stim_site, win)
    defining = [i for i, (_, trials, _) in enumerate(groups) if len(trials) >= MIN_TRIALS]
    if len(defining) < MIN_GROUPS:
        raise DataError(
            f"shared_shapes needs at least {MIN_GROUPS} groups with "
            f"{MIN_TRIALS} trials or more; found {len(defining)}."
        )
    rng = np.random.default_rng(random_state)

    trials = np.concatenate([groups[i][1] for i in defining])
    owner = np.concatenate([np.full(len(groups[i][1]), j) for j, i in enumerate(defining)])
    xi_raw = _t_matrix(trials, owner, len(defining))
    xi = np.clip(xi_raw, 0, None)
    if xi.max() > 0:
        xi = xi / xi.max()

    cluster = _cluster_nmf if method == "nmf" else _cluster_kpca
    selection, chosen = [], None
    for q in range(1, min(max_shapes, len(defining) - 1) + 1):
        labels, runs, h, err, overlap, w = cluster(xi, q, n_restarts, rng)
        pairs = list(itertools.combinations(range(len(runs)), 2))
        ari = float(np.mean([adjusted_rand_index(runs[a], runs[b]) for a, b in pairs]))
        used = len(set(labels[labels >= 0]))
        split = _split_ratio(xi, labels, q)
        ok = split < SPLIT_MAX and ari >= stability_min and used == q
        selection.append(
            {
                "n_shapes": q,
                "error": err,
                "split_ratio": split,
                "h_overlap": overlap,
                "stability": ari,
                "clusters_used": used,
                "accepted": bool(ok),
            }
        )
        if ok or q == 1:
            stab = _jaccard_stability(labels, runs, q)
            chosen = (q, labels, h, stab, w) if ok or chosen is None else chosen
    unstable = all(not row["accepted"] for row in selection[1:]) and any(
        row["split_ratio"] < SPLIT_MAX and row["clusters_used"] == row["n_shapes"]
        for row in selection[1:]
    )
    q, labels, h, stability, w_nmf = chosen
    warns = []
    if q == 1 and unstable:
        _warn(
            "unstable_basis: no factorisation with two shapes or more was stable "
            f"(adjusted Rand index >= {stability_min}); one shape is returned.",
            UserWarning,
            stacklevel=2,
        )
        warns.append("unstable_basis")
    if q == 1:
        labels = np.where(labels >= 0, 0, -1)
        if not np.any(labels == 0):
            labels[:] = 0

    # Basis curves from the trials of each cluster's defining groups
    curves = []
    for k in range(q):
        members = [defining[j] for j in np.flatnonzero(labels == k)]
        curves.append(_basis_curve(np.concatenate([groups[i][1] for i in members])))
    curves = np.array(curves)

    # Projection of every group on every curve: the group-level weight alpha_m (mean
    # projection, data units) and R² of its shape
    n_groups = len(groups)
    alphas = np.zeros((n_groups, q))
    r2_all = np.zeros((n_groups, q))
    for g, (_, v, _) in enumerate(groups):
        alpha = v @ curves.T  # trials x shapes
        alphas[g] = alpha.mean(axis=0)
        total = np.sum(v**2)
        for k in range(q):
            resid = v - np.outer(alpha[:, k], curves[k])
            r2_all[g, k] = 1 - np.sum(resid**2) / total if total > 0 else np.nan
    # ShapeBasis.weights (spec §4.3.4, §5.9): the clustering loadings, group x shape,
    # non-negative (the columns of H); a group that does not define the bases (fewer than
    # 3 trials) gets the non-negative least-squares loadings of its t-values against the
    # defining groups on the factor W (x ~ W H), or, for kpca, its shape's indicator
    from scipy.optimize import nnls

    position = {i: j for j, i in enumerate(defining)}
    weights = np.zeros((n_groups, q))
    scale = float(np.max(xi_raw)) if np.max(xi_raw) > 0 else 1.0
    shape_ids = np.full(n_groups, -1)
    for g, (_, v, _) in enumerate(groups):
        if g in position:
            weights[g] = h[:, position[g]]
            shape_ids[g] = int(labels[position[g]])
            continue
        shape_ids[g] = int(np.nanargmax(r2_all[g])) if np.nanmax(r2_all[g]) > 0 else -1
        if w_nmf is not None:
            column = np.clip(_cross_t(trials, owner, len(defining), v), 0, None) / scale
            weights[g] = nnls(w_nmf, column)[0]
        elif shape_ids[g] >= 0:
            weights[g, shape_ids[g]] = 1.0
    rows, r2 = [], np.full(n_groups, np.nan)
    for g, (label, v, _) in enumerate(groups):
        shape_id = int(shape_ids[g])
        loading = float(weights[g, shape_id]) if shape_id >= 0 else float(np.max(weights[g]))
        if shape_id >= 0:
            r2[g] = r2_all[g, shape_id]
        rows.append(
            {
                "group": str(label),
                "shape_id": shape_id,
                "weight": float(alphas[g, shape_id]) if shape_id >= 0 else np.nan,
                "r2": float(r2[g]),
                "n_trials": int(len(v)),
                "used": g in position,
                "nmf_loading": loading,
            }
        )
    assignment = pd.DataFrame(rows, columns=list(BASIS_COLUMNS))
    assignment["shape_id"] = assignment["shape_id"].astype("int64")
    assignment["n_trials"] = assignment["n_trials"].astype("int64")

    settings = {
        "by": by,
        **where,
        "window": [float(window[0]), float(window[1])],
        "max_shapes": int(max_shapes),
        "method": method,
        "n_restarts": int(n_restarts),
        "stability_min": float(stability_min),
        "random_state": random_state,
        "groups": [str(g[0]) for g in groups],
        "defining_groups": [str(groups[i][0]) for i in defining],
        "t_matrix": xi_raw.tolist(),
        "selection": selection,
        "warnings": warns,
        "package_version": __version__,
    }
    provenance = ts.provenance.with_step(
        "evoparam.spatial.shared_shapes",
        {k: v for k, v in settings.items() if k not in ("t_matrix", "selection")},
        duration_s=_time.perf_counter() - start,
    )
    return ShapeBasis(
        curves=curves,
        times=ts.times[win],
        weights=weights,
        r2=r2,
        assignment=assignment,
        stability=stability,
        n_shapes=q,
        method=method,
        settings=settings,
        provenance=provenance,
    )


def convergent(ts, channel, **kwargs):
    """Input repertoire of one channel.

    :func:`shared_shapes` with the stimulation sites as groups (many sites into one channel).
    """
    return shared_shapes(ts, by="stim_site", channel=channel, **kwargs)


def divergent(ts, stim_site, **kwargs):
    """Output repertoire of one stimulation site.

    :func:`shared_shapes` with the channels as groups (one site into all channels).
    """
    return shared_shapes(ts, by="channel", stim_site=stim_site, **kwargs)


def as_template(basis, k, *, max_shift=None, stretch=False):
    """Return basis curve ``k`` as a ``template`` component shape (spec §5.5.1).

    For the fitting engine and the trial fits; the ends are tapered to zero (10 % cosine taper) so
    that the template starts and ends at rest.

    (``ShapeBasis.as_template`` of the specification lives here: the data containers may not
    import the shapes, spec §4.2.)
    """
    if not 0 <= int(k) < basis.n_shapes:
        raise SettingsError(f"k must be in [0, {basis.n_shapes - 1}], got {k}.")
    curve = np.array(basis.curves[int(k)], float)
    n = len(curve)
    edge = max(2, int(0.1 * n))
    taper = np.ones(n)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(edge) / edge))
    taper[:edge], taper[n - edge :] = ramp, ramp[::-1]
    values = curve * taper
    values[0] = values[-1] = 0.0
    return Template(
        template_times=np.array(basis.times, float),
        template_values=values,
        max_shift=max_shift,
        stretch=stretch,
    )
