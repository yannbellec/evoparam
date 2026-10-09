"""Time-frequency parameterisation of evoked responses (spec §5.10).

``tf_fit`` computes, for every channel (and condition), the power of the trials in a
time-frequency plane, normalises it to the baseline, tests every pixel, and describes the
responsive regions as zones: found in the data (``zones="auto"``) or published fixed zones
(``zones="published:<name>"``). With a channel fit (``group_fit``) the power is split into
the part produced by the fitted evoked components (``evoked_footprint``), the rest
(``residual``) and their cross term, exactly.
"""

import time as _time
import warnings

import numpy as np

from evoparam.data import TFFit, TrialSet
from evoparam.errors import DataError, SettingsError, warn as _warn
from evoparam.models import evaluate_fit
from evoparam.version import __version__

from .normalize import BASELINE_LEVELS, BASELINE_MODES, apply_mode, normalize, reference_level
from .stats import (CHANNEL_CORRECTIONS, STATS, baseline_normal_map, row_levels, t_map, test_map,
                    within_map_p)
from .transforms import TRANSFORMS, apply_kernels, log_freqs, make_kernels
from .zones import (N_SPLITS, ZONE_COLUMNS, decompose, list_published_zones, resolve_published,
                    select_rank, zone_mask, zone_regions, zone_table)

__all__ = ["tf_fit", "tf_zones", "tf_compare", "list_published_zones", "methods_text"]

NYQUIST_FACTOR = 2.5
MIN_CLUSTER_TRIALS = 10
ZONE_METHODS = ("nmf", "cp")
# Ramp (s) that brings the fitted background to zero at the edges of the fit window, so
# that cutting it does not put a step (broadband power) into the footprint.
BACKGROUND_TAPER = 0.005

# Unit of a normalised power value, per baseline mode.
MODE_UNITS = {"db": "dB", "percent": "%", "zscore": "z", "logratio": "log10 ratio",
              "ratio": "ratio"}

METRIC_COLUMNS = {"zone_id": "", "zone": "", "channel": "", "condition": "", "trial": "",
                  "value": "normalised power", "t": "", "p_value": "",
                  "fraction_significant": "", "fraction_increase": "", "fraction_decrease": "",
                  "modality": ""}


def _freq_grid(freqs, sfreq):
    if isinstance(freqs, tuple) and len(freqs) == 3:
        grid = log_freqs(*freqs)
    else:
        grid = np.asarray(freqs, float)
    if grid.ndim != 1 or grid.size == 0 or np.any(grid <= 0):
        raise SettingsError("freqs must be positive: an array or (fmin, fmax, n).")
    keep = grid <= sfreq / NYQUIST_FACTOR
    if not keep.all():
        _warn(f"{int((~keep).sum())} frequencies above sfreq/{NYQUIST_FACTOR:g} = "
              f"{sfreq / NYQUIST_FACTOR:g} Hz were dropped.", stacklevel=3)
    if not keep.any():
        raise SettingsError(f"No frequency at or below sfreq/{NYQUIST_FACTOR:g}.")
    return grid[keep]


def _picks(ts, picks):
    if picks is None:
        return list(range(ts.n_channels))
    out = []
    for p in np.atleast_1d(picks):
        if isinstance(p, (int, np.integer)):
            out.append(int(p))
        elif p in ts.ch_names:
            out.append(ts.ch_names.index(p))
        else:
            raise SettingsError(f"Unknown channel {p!r} in picks.")
    return out


def _labels(ts, by):
    meta = ts.metadata
    columns = [c for c in (by or ()) if meta is not None and c in meta.columns]
    if columns:
        return meta[columns].astype(str).agg("/".join, axis=1).to_numpy(), columns
    return np.array([""] * ts.n_trials), columns


def _invalid_pixels(masks, half_support, n_times, out_idx):
    """True where a pixel's kernel reaches an epoch edge (cone of influence) or a masked
    sample (artifact mask widened by the kernel's temporal support at each frequency).
    Shape (n_trials or 1, n_freqs, n_out)."""

    n_freqs = len(half_support)
    t = np.arange(n_times)[out_idx]
    if masks is None:
        out = np.zeros((1, n_freqs, len(t)), bool)
        for j, h in enumerate(half_support):
            out[0, j] = (t < h) | (t > n_times - 1 - h)
        return out
    cs = np.concatenate([np.zeros((masks.shape[0], 1)), np.cumsum(masks, axis=1)], axis=1)
    out = np.zeros((masks.shape[0], n_freqs, len(t)), bool)
    for j, h in enumerate(half_support):
        lo = np.clip(t - h, 0, n_times)
        hi = np.clip(t + h + 1, 0, n_times)
        out[:, j] = (cs[:, hi] - cs[:, lo]) > 0
        out[:, j] |= ((t < h) | (t > n_times - 1 - h))[None]
    return out


def _footprint_curve(fit, times):
    """The fitted model ŷ(t): components everywhere, background inside the fit window with
    short cosine ramps at its edges."""

    curve = evaluate_fit(fit, times, background=False)
    if fit.background:
        bg = evaluate_fit(fit, times) - curve
        start, stop = fit.window
        inside = (times >= start) & (times <= stop)
        ramp = np.ones_like(times)
        up = (times - start) / BACKGROUND_TAPER
        down = (stop - times) / BACKGROUND_TAPER
        ramp = np.minimum(ramp, 0.5 - 0.5 * np.cos(np.pi * np.clip(up, 0, 1)))
        ramp = np.minimum(ramp, 0.5 - 0.5 * np.cos(np.pi * np.clip(down, 0, 1)))
        curve = curve + np.where(inside, bg * ramp, 0.0)
    return curve


def _filled(x, masks):
    """Samples with masked and missing values set to zero (their pixels are invalid)."""

    x = np.nan_to_num(np.asarray(x, float))
    return x if masks is None else np.where(masks, 0.0, x)


def _channel_fit(group_fit, name, label):
    if label:
        try:
            return group_fit.get(name, condition=label)
        except KeyError:
            pass
    return group_fit.get(name)


def _power(kernels, x, out_idx):
    """Complex coefficients on the output samples (n_tapers, ..., n_freqs, n_out)."""

    return apply_kernels(kernels, x, crop=slice(int(out_idx[0]), int(out_idx[-1]) + 1))


def _row_power(ts, trial_idx, ch, kernels, out_idx):
    """Trials of one row with masked samples zeroed, their power (invalid pixels NaN) and
    the pixels with enough valid trials."""

    masks = None if ts.masks is None else ts.masks[trial_idx, ch]
    invalid = _invalid_pixels(masks, kernels.half_support, ts.n_times, out_idx)
    filled = _filled(ts.data[trial_idx, ch], masks)
    power = np.where(invalid, np.nan, np.mean(np.abs(_power(kernels, filled, out_idx)) ** 2,
                                              axis=0))
    enough = np.sum(np.isfinite(power), axis=0) >= max(3, len(trial_idx) / 2)
    return filled, np.where(enough[None], power, np.nan), enough


def _row_test(stats, test, power, ref_times, tested, alpha, n_permutations, rng):
    """t (or z) map and pixel p-values of one row (see :func:`evoparam.tf.stats.test_map`
    and :func:`evoparam.tf.stats.baseline_normal_map`)."""

    if stats == "baseline_normal":
        return baseline_normal_map(power, ref_times, tested)
    t, _, p = test_map(test, tested, method=stats, alpha=alpha, n_permutations=n_permutations,
                       rng=rng)
    return t, p


def _row_p(padj):
    """Each row's p-value: its smallest adjusted pixel p-value (NaN when none is tested)."""

    flat = padj.reshape(padj.shape[0], -1)
    out = np.full(padj.shape[0], np.nan)
    tested = np.isfinite(flat).any(axis=1)
    out[tested] = np.nanmin(flat[tested], axis=1)
    return out


def _region_stats(per_trial, test, tmap, reported, sig):
    """Zone metric from one row's values on the zone's valid pixels (``per_trial`` and
    ``test`` (n_trials, n_cells), the others (n_cells,)): signed mean, trial-level t and p,
    fractions of significant pixels (any sign, increases, decreases), per-trial values."""

    from scipy import stats as sstats

    if not reported.size:
        return (np.nan,) * 6 + (np.full(per_trial.shape[0], np.nan),)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        value = float(np.nanmean(reported))
        trial_values = np.nanmean(per_trial, axis=1)
        d = np.nanmean(test, axis=1)
    d = d[np.isfinite(d)]
    if d.size >= 3 and np.std(d) > 0:
        t = float(np.mean(d) / (np.std(d, ddof=1) / np.sqrt(d.size)))
        p = float(2 * sstats.t.sf(abs(t), d.size - 1))
    else:
        t, p = np.nan, np.nan
    s = np.asarray(sig, bool)
    up = np.nan_to_num(tmap) > 0
    return (value, t, p, float(s.mean()), float((s & up).mean()), float((s & ~up).mean()),
            trial_values)


def _region_values(per_trial, test, sig, tmap, reported, region, valid):
    """Zone metric of one map row on ``region & valid`` (see :func:`_region_stats`)."""

    cells = region & valid
    return _region_stats(per_trial[:, cells], test[:, cells], tmap[cells], reported[cells],
                         sig[cells])


def tf_fit(ts, *, transform="morse", freqs=(2.0, 250.0, 40), gamma=3.0, p2=30.0,
           n_cycles=7.0, time_bandwidth=4.0, stft_window=0.2, window=(0.0, 1.0),
           baseline=(-0.5, -0.1), baseline_mode="db", baseline_level="single_trial",
           group_fit=None, stats="fdr", alpha=0.05, n_permutations=1000,
           channel_correction="fdr_bh", zones="auto", max_zones=8, zone_method="nmf",
           allow_unverified=False, by_trial=False, picks=None, by=("stim_site",), n_jobs=1,
           random_state=None, verbose=None):
    """Time-frequency maps, pixel statistics and zones of a TrialSet.

    Parameters
    ----------
    ts : TrialSet
    transform : {"morse", "morlet", "multitaper", "stft"}
        Generalised Morse wavelets (``gamma``, ``beta = p2 / gamma``; default 3 and 10),
        complex Morlet (``n_cycles``), DPSS multitaper (window ``n_cycles / f``,
        ``time_bandwidth``) or a sliding Hann window of ``stft_window`` s.
    freqs : ndarray or (fmin, fmax, n)
        Frequencies (Hz); a tuple gives a log grid. Frequencies above ``sfreq / 2.5`` are
        dropped with a warning.
    window, baseline : (float, float)
        Test window and baseline (s). Pixels whose kernel reaches an epoch edge or a
        masked sample are invalid: never filled, left out of statistics and zones.
    baseline_mode : {"db", "percent", "zscore", "logratio", "ratio"}
    baseline_level : {"single_trial", "evoked", "full_epoch"}
        Normalise each trial by its own baseline (default; less sensitive to noisy trials),
        the trial-average power by its baseline, or each trial by its own full-epoch mean.
    group_fit : GroupFit, optional
        Channel fits (``evoparam.models.fit``); enables the footprint / residual split.
    stats : {"fdr", "cluster", "tfce", "baseline_normal", "none"}
        Benjamini-Hochberg over pixels, cluster-mass or TFCE sign-flip permutation, or the
        published test of the eleven-zone work ([LIT-tf-4] §2.4: trial-average dB power
        against a normal fitted to the baseline of each frequency, two-sided,
        Benjamini-Hochberg over the window's pixels).
    channel_correction : {"fdr_bh", "holm", "bonferroni", "none"}
        Correction across map rows (channels x conditions), as in §5.6. Each row is a
        family of pixel tests whose p-value is its smallest adjusted pixel p-value (the
        Simes p-value for ``fdr``, the family-wise permutation p-value for ``cluster`` and
        ``tfce``). ``fdr_bh`` (default): Benjamini-Hochberg over the rows selects ``R`` of
        ``m``, whose pixels are then tested at ``R * alpha / m`` (Benjamini and Bogomolov
        2014); ``holm``: step-down, each rejected row at its Holm level; ``bonferroni``:
        every row at ``alpha / m``; ``none``: every row at ``alpha``. The levels are in
        ``settings["row_alpha"]`` and the rows' p-values in ``settings["row_p_value"]``.
    zones : "auto", "none" or "published:<name>"
        Data-driven zones of the significant group map, no zones, or a registry zone
        (:func:`list_published_zones`); a published zone also sets the settings of its
        source given in the registry (transform, baseline, and for the zones of [LIT-tf-4]
        the statistic, the test window and the correction across channels).
    zone_method : {"nmf", "cp"}
    allow_unverified : bool
        Allow an ``unverified`` registry zone.
    by_trial : bool
        Zone metric per trial as well (rows with ``trial >= 0``).
    by : sequence of str
        Metadata columns that split the trials into conditions.
    n_jobs, verbose
        Accepted for the common signature of the package; the computation runs in one
        process (100 channels x 50 trials x 2 048 samples x 40 frequencies take under 60 s).

    Returns
    -------
    TFFit
        ``maps``: ``total`` (trial-average power) and, with ``group_fit``,
        ``evoked_footprint``, ``residual`` and ``cross`` (power; ``footprint + residual +
        cross == total`` exactly); ``total_norm``, ``residual_norm``, ``footprint_norm``
        (baseline-normalised); ``residual_t`` and ``residual_significant``. ``tmaps`` and
        ``significant`` are those of the total. Every map has shape (n_rows, n_freqs,
        n_times); the rows are listed in ``settings["rows"]`` as (channel, condition).
    """

    start_clock = _time.perf_counter()
    if not isinstance(ts, TrialSet):
        raise DataError("tf_fit expects a TrialSet.")
    published = None
    if isinstance(zones, str) and zones.startswith("published:"):
        published = resolve_published(zones.split(":", 1)[1], allow_unverified)
        pub = published.get("settings", {})
        transform = pub.get("transform", transform)
        gamma = pub.get("gamma", gamma)
        p2 = pub.get("p2", p2)
        baseline = tuple(pub.get("baseline", baseline))
        baseline_mode = pub.get("baseline_mode", baseline_mode)
        baseline_level = pub.get("baseline_level", baseline_level)
        stats = pub.get("stats", stats)
        window = tuple(pub.get("window", window))
        channel_correction = pub.get("channel_correction", channel_correction)
    elif zones not in ("auto", "none"):
        raise SettingsError(f"zones must be 'auto', 'none' or 'published:<name>', not {zones!r}.")
    for name, value, allowed in (("transform", transform, TRANSFORMS),
                                 ("baseline_mode", baseline_mode, BASELINE_MODES),
                                 ("baseline_level", baseline_level, BASELINE_LEVELS),
                                 ("stats", stats, STATS),
                                 ("channel_correction", channel_correction, CHANNEL_CORRECTIONS),
                                 ("zone_method", zone_method, ZONE_METHODS)):
        if value not in allowed:
            raise SettingsError(f"{name} must be one of {allowed}, not {value!r}.")
    if baseline_level == "full_epoch" and baseline_mode == "zscore":
        raise SettingsError("baseline_level='full_epoch' with baseline_mode='zscore' is "
                            "undefined; use 'db', 'percent', 'logratio' or 'ratio'.")
    times = ts.times
    window = (max(float(window[0]), float(times[0])), min(float(window[1]), float(times[-1])))
    baseline = (float(baseline[0]), float(baseline[1]))
    in_window = (times >= window[0]) & (times <= window[1])
    in_base = (times >= baseline[0]) & (times <= baseline[1])
    if not in_window.any():
        raise SettingsError(f"window {window} is outside the epoch.")
    if baseline_level != "full_epoch" and not in_base.any():
        raise SettingsError(f"baseline {baseline} is outside the epoch.")
    first = min(window[0], baseline[0]) if baseline_level != "full_epoch" else times[0]
    out_idx = np.flatnonzero((times >= first) & (times <= window[1]))
    out_times = times[out_idx]
    ref_times = in_base[out_idx] if baseline_level != "full_epoch" else \
        np.ones(len(out_idx), bool)
    test_window = in_window[out_idx]

    grid = _freq_grid(freqs, ts.sfreq)
    picks = _picks(ts, picks)
    if group_fit is not None:
        missing = [ts.ch_names[c] for c in picks if ts.ch_names[c] not in group_fit.ch_names]
        if missing:
            raise SettingsError(f"group_fit has no fit for channels {missing}: it was made on "
                                f"a different channel set.")
    labels, by_columns = _labels(ts, by)
    groups = list(dict.fromkeys(labels))
    if stats == "cluster":
        smallest = min(int(np.sum(labels == g)) for g in groups)
        if smallest < MIN_CLUSTER_TRIALS:
            _warn(f"Only {smallest} trials in a condition: the sign-flip permutation "
                  f"has few distinct patterns (coarse p-values).", stacklevel=2)
    kernels = make_kernels(transform, grid, ts.sfreq, ts.n_times, gamma=gamma, p2=p2,
                           n_cycles=n_cycles, time_bandwidth=time_bandwidth,
                           stft_window=stft_window)
    rng = np.random.default_rng(random_state)
    n_rows = len(groups) * len(picks)
    want_zones = zones == "auto"

    shape = (n_rows, len(grid), len(out_idx))
    maps = {key: np.full(shape, np.nan) for key in ("total", "total_norm")}
    if group_fit is not None:
        for key in ("evoked_footprint", "residual", "cross", "footprint_norm", "residual_norm",
                    "residual_t", "residual_significant"):
            maps[key] = np.full(shape, np.nan)
    tmaps = np.full(shape, np.nan)
    halves = np.zeros((N_SPLITS, 2, len(grid), len(out_idx))) if want_zones else None
    rows, row_trials = [], []
    published_region = None
    if published is not None:
        published_region = zone_mask(published, grid, out_times)
    metric_rows = []
    # pixel p-values adjusted within each row (the row's own test), then the rows'
    # levels from the correction across channels (§5.6), then the significant pixels
    padj = np.full(shape, np.nan)
    padj_residual = np.full(shape, np.nan) if group_fit is not None else None
    stash = []                      # published zone: each row's values inside the zone

    r = -1
    for label in groups:
        trial_idx = np.flatnonzero(labels == label)
        for ch in picks:
            r += 1
            name = ts.ch_names[ch]
            rows.append((name, label))
            row_trials.append(trial_idx)
            filled, power, enough = _row_power(ts, trial_idx, ch, kernels, out_idx)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                maps["total"][r] = np.nanmean(power, axis=0)
            per_trial, reported, test = normalize(power, ref_times, baseline_mode,
                                                  baseline_level)
            maps["total_norm"][r] = reported
            t, p = _row_test(stats, test, power, ref_times, test_window & enough, alpha,
                             n_permutations, rng)
            tmaps[r], padj[r] = t, within_map_p(p, stats)
            if group_fit is not None:
                fit = _channel_fit(group_fit, name, label)
                yhat = _footprint_curve(fit, times)
                c_fp = _power(kernels, yhat[None], out_idx)
                c_res = _power(kernels, filled - yhat[None], out_idx)
                p_fp = np.mean(np.abs(c_fp) ** 2, axis=0)
                p_res = np.where(np.isfinite(power), np.mean(np.abs(c_res) ** 2, axis=0),
                                 np.nan)
                cross = np.where(np.isfinite(power),
                                 np.mean(2 * np.real(c_fp * np.conj(c_res)), axis=0), np.nan)
                p_fp = np.where(np.isfinite(power), p_fp, np.nan)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    maps["evoked_footprint"][r] = np.nanmean(p_fp, axis=0)
                    maps["residual"][r] = np.nanmean(p_res, axis=0)
                    maps["cross"][r] = np.nanmean(cross, axis=0)
                ref, sd = reference_level(maps["total"][r], ref_times, baseline_mode)
                maps["footprint_norm"][r] = apply_mode(maps["evoked_footprint"][r], ref, sd,
                                                       baseline_mode)
                _, res_reported, res_test = normalize(p_res, ref_times, baseline_mode,
                                                      baseline_level)
                maps["residual_norm"][r] = res_reported
                rt, rp = _row_test(stats, res_test, p_res, ref_times, test_window & enough,
                                   alpha, n_permutations, rng)
                maps["residual_t"][r] = rt
                padj_residual[r] = within_map_p(rp, stats)
            if published_region is not None:
                cells = published_region & enough
                stash.append((cells, per_trial[:, cells], test[:, cells], t[cells],
                              reported[cells]))

    row_p = _row_p(padj)
    row_alpha = row_levels(row_p, alpha, channel_correction)
    significant = (padj <= row_alpha[:, None, None]) & (row_alpha[:, None, None] > 0)
    if group_fit is not None:
        residual_alpha = row_levels(_row_p(padj_residual), alpha, channel_correction)
        maps["residual_significant"] = ((padj_residual <= residual_alpha[:, None, None])
                                        & (residual_alpha[:, None, None] > 0)).astype(float)

    # split halves of the rows with significant pixels, kept per row (float32) inside the
    # box of the significant pixels for the CP rank (the NMF uses their mean, ``halves``)
    union = significant.any(axis=0)
    box = None
    if want_zones and union.any():
        f_idx, t_idx = np.flatnonzero(union.any(axis=1)), np.flatnonzero(union.any(axis=0))
        box = (slice(f_idx[0], f_idx[-1] + 1), slice(t_idx[0], t_idx[-1] + 1))
    row_halves, halves_rows = [], []
    r = -1
    for label in groups:
        trial_idx = np.flatnonzero(labels == label)
        for ch in picks:
            r += 1
            name = ts.ch_names[ch]
            if want_zones and significant[r].any():
                # second pass on the rows with significant pixels: split halves of the trials
                _, power, _ = _row_power(ts, trial_idx, ch, kernels, out_idx)
                signed = np.where(significant[r], 1.0, 0.0)
                own = np.zeros((N_SPLITS, 2) + signed[box].shape, np.float32)
                n_tr = len(trial_idx)
                for s in range(N_SPLITS):
                    order = rng.permutation(n_tr)
                    for h, part in enumerate((order[:n_tr // 2], order[n_tr // 2:])):
                        _, half_map, _ = normalize(power, ref_times, baseline_mode,
                                                   baseline_level, trials=part)
                        contribution = np.nan_to_num(half_map) * signed
                        halves[s, h] += contribution
                        own[s, h] = contribution[box]
                row_halves.append(own)
                halves_rows.append(r)
            if published_region is not None:
                cells, *inside = stash[r]
                *values, trial_values = _region_stats(*inside, significant[r][cells])
                metric_rows.append(_metric_row(0, published["name"], name, label, -1, *values,
                                               modality=ts.modality))
                if by_trial:
                    for i, v in zip(trial_idx, trial_values):
                        metric_rows.append(_metric_row(0, published["name"], name, label,
                                                       int(i), v, modality=ts.modality))

    if want_zones:
        halves /= max(n_rows, 1)
    import pandas as pd

    settings = {
        "transform": transform, "transform_params": kernels.params,
        "freqs": grid.tolist(), "window": list(window), "baseline": list(baseline),
        "baseline_mode": baseline_mode, "baseline_level": baseline_level, "stats": stats,
        "alpha": alpha, "n_permutations": n_permutations,
        "channel_correction": channel_correction,
        "row_p_value": [None if not np.isfinite(v) else float(v) for v in row_p],
        "row_alpha": [float(v) for v in row_alpha],
        "zones": zones, "max_zones": max_zones,
        "zone_method": zone_method, "by": list(by_columns), "by_trial": by_trial,
        "rows": [list(row) for row in rows], "n_trials": [len(t) for t in row_trials],
        "half_support_s": (kernels.half_support / ts.sfreq).tolist(),
        "footprint": group_fit is not None, "random_state": random_state,
        "modality": ts.modality, "package_version": __version__,
    }
    if published is not None:
        settings["published_zone"] = {k: published.get(k) for k in (
            "name", "time", "freq", "intervals", "source", "reference", "verified")}
    if want_zones:
        maps["zone_halves"] = halves.reshape(N_SPLITS * 2, len(grid), len(out_idx))
        if row_halves:
            # (2 N_SPLITS, n_freqs_box, n_times_box, n_rows_significant)
            maps["zone_halves_rows"] = np.stack(row_halves, axis=-1).reshape(
                (N_SPLITS * 2,) + row_halves[0].shape[2:] + (len(row_halves),))
            settings["zone_halves_rows"] = {
                "rows": [int(v) for v in halves_rows],
                "freq_slice": [int(box[0].start), int(box[0].stop)],
                "time_slice": [int(box[1].start), int(box[1].stop)]}
    method = zone_method if want_zones else ("published" if published is not None else "none")
    result = TFFit(freqs=grid, times=out_times, maps=maps, tmaps=tmaps, significant=significant,
                   zones=None, zone_metrics=None, method=method, settings=settings,
                   provenance=ts.provenance.copy())
    if want_zones:
        table, regions = _zones_of(result, zone_method, max_zones, rng)
        if len(table):
            # second pass: the zone metrics need the trials
            metric_rows = _zone_metrics(ts, result, regions, kernels, out_idx, ref_times,
                                        groups, labels, picks, by_trial)
            p_by_zone = {}
            for row in metric_rows:
                if row["trial"] < 0 and np.isfinite(row["p_value"]):
                    p_by_zone.setdefault(row["zone_id"], []).append(row["p_value"])
            table["p_value"] = [min(1.0, min(p_by_zone.get(z, [np.nan])) * n_rows)
                                for z in table["zone_id"]]
        result.zones = table
    elif published is not None:
        zone = published
        result.zones = pd.DataFrame([{
            "zone_id": 0, "sign": 0, "t_start": zone["time"][0], "t_end": zone["time"][1],
            "f_low": zone["freq"][0], "f_high": zone["freq"][1], "source": zone["source"]}])
    else:
        result.zones = pd.DataFrame(columns=list(ZONE_COLUMNS))
    result.zone_metrics = pd.DataFrame(metric_rows, columns=list(METRIC_COLUMNS))
    unit = MODE_UNITS[baseline_mode]
    result.zone_metrics.attrs["units"] = {**METRIC_COLUMNS, "value": unit}
    if hasattr(result.zones, "attrs"):
        result.zones.attrs["units"] = {**ZONE_COLUMNS, "peak_value": unit, "mean_value": unit}
    result.provenance = ts.provenance.with_step(
        "evoparam.tf.tf_fit", {k: v for k, v in settings.items()
                                if k not in ("rows", "freqs", "row_p_value", "row_alpha")},
        duration_s=_time.perf_counter() - start_clock)
    return result


def _metric_row(zone_id, zone, channel, condition, trial, value, t=np.nan, p=np.nan,
                frac=np.nan, frac_up=np.nan, frac_down=np.nan, modality=None):
    return {"zone_id": zone_id, "zone": zone, "channel": channel, "condition": condition,
            "trial": trial, "value": value, "t": t, "p_value": p,
            "fraction_significant": frac, "fraction_increase": frac_up,
            "fraction_decrease": frac_down, "modality": modality}


def _group_map(tf):
    """Signed group map: mean over rows of the significant normalised pixels."""

    signed = np.where(tf.significant, np.nan_to_num(tf.maps["total_norm"]), 0.0)
    return signed.mean(axis=0)


def _zones_of(tf, method, max_zones, rng, cp_stability="split_half"):
    """Zone table and regions from a TFFit's group map and split halves."""

    rng = np.random.default_rng(rng)
    group = _group_map(tf)
    halves = tf.maps.get("zone_halves")
    halves = None if halves is None else halves.reshape(-1, 2, *group.shape)
    row_halves = tf.maps.get("zone_halves_rows") if method == "cp" else None
    if row_halves is not None:
        row_halves = np.asarray(row_halves, float).reshape(-1, 2, *row_halves.shape[1:])
    signed_rows = np.where(tf.significant, np.nan_to_num(tf.maps["total_norm"]), 0.0)
    parts_by_sign, stability = {}, {}
    for sign in (1, -1):
        x = np.maximum(sign * group, 0.0)
        tensor = np.moveaxis(np.maximum(sign * signed_rows, 0.0), 0, -1) \
            if method == "cp" else None
        split = None if halves is None else np.maximum(sign * halves, 0.0)
        trial_split = None if row_halves is None else np.maximum(sign * row_halves, 0.0)
        rank, stab = select_rank(x, split, max_zones, rng, method=method, tensor=tensor,
                                 cp_stability=cp_stability, row_halves=trial_split)
        if rank == 0:
            continue
        parts, _ = decompose(x, rank, rng, method=method, tensor=tensor)
        parts_by_sign[sign] = parts
        stability[sign] = stab.get(rank, np.nan)
    regions = zone_regions(parts_by_sign, group)
    table = zone_table(regions, group, tf.freqs, tf.times, stability)
    # zone_table orders zones by energy; keep the regions in the same order
    order = sorted(range(len(regions)), key=lambda i: -np.nansum(np.abs(group[regions[i][2]])))
    return table, [regions[i][2] for i in order]


def _zone_metrics(ts, tf, regions, kernels, out_idx, ref_times, groups, labels, picks,
                  by_trial):
    s = tf.settings
    rows = []
    r = -1
    for label in groups:
        trial_idx = np.flatnonzero(labels == label)
        for ch in picks:
            r += 1
            name = ts.ch_names[ch]
            masks = None if ts.masks is None else ts.masks[trial_idx, ch]
            invalid = _invalid_pixels(masks, kernels.half_support, ts.n_times, out_idx)
            coef = _power(kernels, _filled(ts.data[trial_idx, ch], masks), out_idx)
            power = np.where(invalid, np.nan, np.mean(np.abs(coef) ** 2, axis=0))
            enough = np.isfinite(tf.maps["total_norm"][r])
            power = np.where(enough[None], power, np.nan)
            per_trial, reported, test = normalize(power, ref_times, s["baseline_mode"],
                                                  s["baseline_level"])
            for z, region in enumerate(regions):
                *values, trial_values = _region_values(per_trial, test, tf.significant[r],
                                                       tf.tmaps[r], reported, region, enough)
                rows.append(_metric_row(z, f"zone{z}", name, label, -1, *values,
                                        modality=ts.modality))
                if by_trial:
                    for i, v in zip(trial_idx, trial_values):
                        rows.append(_metric_row(z, f"zone{z}", name, label, int(i), v,
                                                modality=ts.modality))
    return rows


def tf_zones(tf, *, method="nmf", max_zones=8, random_state=None, cp_stability="split_half"):
    """Data-driven zones of a TFFit (made with ``zones="auto"``), with another method or
    bound. Returns the zone table (without p-values, which need the trials).

    ``cp_stability`` (``method="cp"``): the rank is chosen on split halves of each row's
    trials (``"split_half"``, spec §5.10; halves of the rows when the TFFit holds no trial
    halves), on halves of the rows (``"rows"``) or, as in the first version, across random
    restarts of the whole tensor (``"restarts"``).
    """

    if method not in ZONE_METHODS:
        raise SettingsError(f"method must be one of {ZONE_METHODS}, not {method!r}.")
    if method == "nmf" and "zone_halves" not in tf.maps:
        raise SettingsError("The TFFit has no split halves: run tf_fit with zones='auto'.")
    table, _ = _zones_of(tf, method, max_zones, random_state, cp_stability=cp_stability)
    return table


def tf_compare(tf_a, tf_b, *, stats="cluster", n_permutations=1000, alpha=0.05,
               random_state=None):
    """Contrast of two TFFits (conditions or modalities), with the map rows as units.

    Rows present in both (same channel and condition order) are paired (sign flips of the
    differences); otherwise the rows of the two fits are compared as independent groups
    (permutation of the group labels, Welch ``t``). Returns one row per cluster of pixels
    (``cluster``) or per significant pixel set (``fdr``): sign, bounds, mass, p-value.
    """

    import pandas as pd
    from scipy import ndimage, stats as sstats

    if tf_a.freqs.shape != tf_b.freqs.shape or tf_a.times.shape != tf_b.times.shape or \
            not np.allclose(tf_a.freqs, tf_b.freqs) or not np.allclose(tf_a.times, tf_b.times):
        raise SettingsError("tf_compare needs the same frequencies and times in both fits.")
    if stats not in ("cluster", "fdr"):
        raise SettingsError("stats must be 'cluster' or 'fdr'.")
    a = tf_a.maps["total_norm"]
    b = tf_b.maps["total_norm"]
    paired = tf_a.settings.get("rows") == tf_b.settings.get("rows")
    rng = np.random.default_rng(random_state)

    def statistic(xa, xb):
        if paired:
            t, _ = t_map(xa - xb)
            return t
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ma, mb = np.nanmean(xa, 0), np.nanmean(xb, 0)
            va, vb = np.nanvar(xa, 0, ddof=1), np.nanvar(xb, 0, ddof=1)
            na, nb = np.sum(np.isfinite(xa), 0), np.sum(np.isfinite(xb), 0)
            return (ma - mb) / np.sqrt(va / na + vb / nb)

    t = statistic(a, b)
    n_units = a.shape[0] if paired else a.shape[0] + b.shape[0]
    df = max(n_units - (1 if paired else 2), 1)
    if stats == "fdr":
        from .stats import fdr_significant

        p = 2 * sstats.t.sf(np.abs(t), df)
        p[~np.isfinite(t)] = np.nan
        sig = fdr_significant(p, alpha)
        rows = []
        for sign in (1, -1):
            mask = sig & (np.nan_to_num(sign * t) > 0)
            if mask.any():
                fi, ti = np.nonzero(mask)
                rows.append({"sign": sign, "t_start": tf_a.times[ti.min()],
                             "t_end": tf_a.times[ti.max()], "f_low": tf_a.freqs[fi.min()],
                             "f_high": tf_a.freqs[fi.max()], "mass": float(np.abs(t[mask]).sum()),
                             "n_pixels": int(mask.sum()), "p_value": float(np.nanmin(p[mask]))})
        return pd.DataFrame(rows, columns=["sign", "t_start", "t_end", "f_low", "f_high", "mass",
                                           "n_pixels", "p_value"])
    threshold = sstats.t.ppf(1 - alpha / 2, df)

    def clusters(tmap):
        out = []
        for sign in (1, -1):
            lab, n = ndimage.label(np.nan_to_num(sign * tmap) > threshold)
            for k in range(1, n + 1):
                region = lab == k
                out.append((sign, region, float(np.abs(tmap[region]).sum())))
        return out

    null = np.zeros(n_permutations)
    both = np.concatenate([a, b]) if not paired else None
    for i in range(n_permutations):
        if paired:
            flips = rng.choice((-1.0, 1.0), a.shape[0])[:, None, None]
            perm_t, _ = t_map(flips * (a - b))
        else:
            order = rng.permutation(len(both))
            perm_t = statistic(both[order[:len(a)]], both[order[len(a):]])
        found = clusters(perm_t)
        null[i] = max([m for _, _, m in found] or [0.0])
    rows = []
    for sign, region, mass in clusters(t):
        fi, ti = np.nonzero(region)
        rows.append({"sign": sign, "t_start": tf_a.times[ti.min()], "t_end": tf_a.times[ti.max()],
                     "f_low": tf_a.freqs[fi.min()], "f_high": tf_a.freqs[fi.max()], "mass": mass,
                     "n_pixels": int(region.sum()),
                     "p_value": float((1 + np.sum(null >= mass)) / (1 + n_permutations))})
    table = pd.DataFrame(rows, columns=["sign", "t_start", "t_end", "f_low", "f_high", "mass",
                                        "n_pixels", "p_value"])
    return table.sort_values("p_value", ignore_index=True)


def methods_text(tf):
    """Methods paragraph of a TFFit (spec REQ-219)."""

    return tf.methods_text()
