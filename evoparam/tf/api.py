# SPDX-License-Identifier: Apache-2.0
"""Time-frequency parameterisation of evoked responses (spec §5.10).

``tf_fit`` computes, for every channel (and condition), the power of the trials in a
time-frequency plane, normalises it to the baseline, tests every pixel, and describes the
responsive regions as zones: found in the data (``zones="auto"``) or published fixed zones
(``zones="published:<name>"``). With a channel fit (``group_fit``) the power is split into
the part produced by the fitted evoked components (``evoked_footprint``), the rest
(``residual``) and their cross term, exactly.
"""

import logging
import time as _time
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from evoparam._typing import Any, FloatArray, NDArray
from evoparam.data import TFFit, TrialSet
from evoparam.errors import DataError, SettingsError
from evoparam.errors import warn as _warn
from evoparam.models import evaluate_fit
from evoparam.utils._logging import progress, verbose
from evoparam.utils._optional import require
from evoparam.version import __version__

from .normalize import BASELINE_LEVELS, BASELINE_MODES, apply_mode, normalize, reference_level
from .stats import (
    CHANNEL_CORRECTIONS,
    STATS,
    baseline_normal_map,
    row_levels,
    t_map,
    test_map,
    within_map_p,
)
from .transforms import TRANSFORMS, apply_kernels, log_freqs, make_kernels
from .zones import (
    N_SPLITS,
    ZONE_COLUMNS,
    decompose,
    list_published_zones,
    resolve_published,
    select_rank,
    zone_mask,
    zone_regions,
    zone_table,
)

if TYPE_CHECKING:
    from evoparam.data.results import GroupFit

logger = logging.getLogger(__name__)

__all__ = ["tf_fit", "tf_zones", "tf_compare", "list_published_zones", "methods_text"]

NYQUIST_FACTOR = 2.5
MIN_CLUSTER_TRIALS = 10
ZONE_METHODS = ("nmf", "cp")
# Ramp (s) that brings the fitted background to zero at the edges of the fit window, so
# that cutting it does not put a step (broadband power) into the footprint.
BACKGROUND_TAPER = 0.005
_GRID_SPEC = 3  # freqs given as (fmin, fmax, n)
_MIN_TRIALS = 3  # valid trials for a pixel, and for the t of a zone metric
_N_METRICS = 6  # value, t, p and the three fractions of a zone metric row

# Unit of a normalised power value, per baseline mode.
MODE_UNITS = {
    "db": "dB",
    "percent": "%",
    "zscore": "z",
    "logratio": "log10 ratio",
    "ratio": "ratio",
}

METRIC_COLUMNS = {
    "zone_id": "",
    "zone": "",
    "channel": "",
    "condition": "",
    "trial": "",
    "value": "normalised power",
    "t": "",
    "p_value": "",
    "fraction_significant": "",
    "fraction_increase": "",
    "fraction_decrease": "",
    "modality": "",
}


def _freq_grid(freqs: Any, sfreq: float) -> Any:
    """Return the frequency grid, without the frequencies above ``sfreq / NYQUIST_FACTOR``."""
    if isinstance(freqs, tuple) and len(freqs) == _GRID_SPEC:
        grid = log_freqs(*freqs)
    else:
        grid = np.asarray(freqs, float)
    if grid.ndim != 1 or grid.size == 0 or np.any(grid <= 0):
        raise SettingsError("freqs must be positive: an array or (fmin, fmax, n).")
    keep = grid <= sfreq / NYQUIST_FACTOR
    if not keep.all():
        _warn(
            f"{int((~keep).sum())} frequencies above sfreq/{NYQUIST_FACTOR:g} = "
            f"{sfreq / NYQUIST_FACTOR:g} Hz were dropped.",
            stacklevel=3,
        )
    if not keep.any():
        raise SettingsError(f"No frequency at or below sfreq/{NYQUIST_FACTOR:g}.")
    return grid[keep]


def _picks(ts: Any, picks: Any) -> Any:
    """Return the indices of the channels picked (names or positions; None: every channel)."""
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


def _labels(ts: Any, by: Any) -> Any:
    """Return each trial's condition label (the ``by`` columns joined) and the columns used."""
    meta = ts.metadata
    columns = [c for c in (by or ()) if meta is not None and c in meta.columns]
    if columns:
        return meta[columns].astype(str).agg("/".join, axis=1).to_numpy(), columns
    return np.array([""] * ts.n_trials), columns


def _invalid_pixels(masks: Any, half_support: Any, n_times: int, out_idx: Any) -> Any:
    """Return True where a pixel's kernel reaches an epoch edge or a masked sample.

    An epoch edge is the cone of influence; the artifact mask is widened by the kernel's temporal
    support at each frequency. Shape (n_trials or 1, n_freqs, n_out).
    """
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


def _footprint_curve(fit: Any, times: Any) -> Any:
    """Return the fitted model ŷ(t).

    Components everywhere, background inside the fit window with short cosine ramps at its edges.
    """
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


def _filled(x: Any, masks: Any) -> Any:
    """Return the samples with masked and missing values set to zero (their pixels are invalid)."""
    x = np.nan_to_num(np.asarray(x, float))
    return x if masks is None else np.where(masks, 0.0, x)


def _channel_fit(group_fit: Any, name: Any, label: Any) -> Any:
    """Return the fit of a channel for a condition (else its only fit)."""
    if label:
        try:
            return group_fit.get(name, condition=label)
        except KeyError:
            pass
    return group_fit.get(name)


def _power(kernels: Any, x: Any, out_idx: Any) -> NDArray[Any]:
    """Complex coefficients on the output samples (n_tapers, ..., n_freqs, n_out)."""
    return apply_kernels(kernels, x, crop=slice(int(out_idx[0]), int(out_idx[-1]) + 1))


def _row_power(
    ts: Any, trial_idx: Any, ch: Any, kernels: Any, out_idx: Any
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Return the trials of one row with masked samples zeroed, and their power.

    Also the power with invalid pixels NaN and the pixels with enough valid trials.
    """
    masks = None if ts.masks is None else ts.masks[trial_idx, ch]
    invalid = _invalid_pixels(masks, kernels.half_support, ts.n_times, out_idx)
    filled = _filled(ts.data[trial_idx, ch], masks)
    power = np.where(
        invalid, np.nan, np.mean(np.abs(_power(kernels, filled, out_idx)) ** 2, axis=0)
    )
    enough = np.sum(np.isfinite(power), axis=0) >= max(_MIN_TRIALS, len(trial_idx) / 2)
    return filled, np.where(enough[None], power, np.nan), enough


def _row_test(
    stats: Any,
    test: Any,
    power: Any,
    ref_times: Any,
    tested: Any,
    alpha: Any,
    n_permutations: Any,
    rng: np.random.Generator,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return the T (or z) map and the pixel p-values of one row.

    See :func:`evoparam.tf.stats.test_map` and :func:`evoparam.tf.stats.baseline_normal_map`.
    """
    if stats == "baseline_normal":
        return baseline_normal_map(power, ref_times, tested)
    t, _, p = test_map(
        test, tested, method=stats, alpha=alpha, n_permutations=n_permutations, rng=rng
    )
    return t, p


def _row_p(padj: Any) -> NDArray[Any]:
    """Each row's p-value: its smallest adjusted pixel p-value (NaN when none is tested)."""
    flat = padj.reshape(padj.shape[0], -1)
    out = np.full(padj.shape[0], np.nan)
    tested = np.isfinite(flat).any(axis=1)
    out[tested] = np.nanmin(flat[tested], axis=1)
    return out


def _region_stats(per_trial: Any, test: Any, tmap: Any, reported: Any, sig: Any) -> Any:
    """Return the zone metrics from one row's values on the zone's valid pixels.

    ``per_trial`` and ``test`` are (n_trials, n_cells), the others (n_cells,): signed mean,
    trial-level t and p, fractions of significant pixels (any sign, increases, decreases),
    per-trial values.
    """
    from scipy import stats as sstats

    if not reported.size:
        return (np.nan,) * _N_METRICS + (np.full(per_trial.shape[0], np.nan),)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        value = float(np.nanmean(reported))
        trial_values = np.nanmean(per_trial, axis=1)
        d = np.nanmean(test, axis=1)
    d = d[np.isfinite(d)]
    if d.size >= _MIN_TRIALS and np.std(d) > 0:
        t = float(np.mean(d) / (np.std(d, ddof=1) / np.sqrt(d.size)))
        p = float(2 * sstats.t.sf(abs(t), d.size - 1))
    else:
        t, p = np.nan, np.nan
    s = np.asarray(sig, bool)
    up = np.nan_to_num(tmap) > 0
    return (
        value,
        t,
        p,
        float(s.mean()),
        float((s & up).mean()),
        float((s & ~up).mean()),
        trial_values,
    )


def _region_values(
    per_trial: Any, test: Any, sig: Any, tmap: Any, reported: Any, region: Any, valid: Any
) -> Any:
    """Zone metric of one map row on ``region & valid`` (see :func:`_region_stats`)."""
    cells = region & valid
    return _region_stats(
        per_trial[:, cells], test[:, cells], tmap[cells], reported[cells], sig[cells]
    )


@verbose
def tf_fit(  # noqa: PLR0913  (spec §5.10 fixes 25 parameters, plus stft_window and channel_correction; DECISIONS.md §7.2)
    ts: "TrialSet",
    *,
    transform: str = "morse",
    freqs: Any = (2.0, 250.0, 40),
    gamma: float = 3.0,
    p2: float = 30.0,
    n_cycles: float = 7.0,
    time_bandwidth: float = 4.0,
    stft_window: float = 0.2,
    window: tuple[float, float] = (0.0, 1.0),
    baseline: tuple[float, float] = (-0.5, -0.1),
    baseline_mode: str = "db",
    baseline_level: str = "single_trial",
    group_fit: "GroupFit | None" = None,
    stats: str = "fdr",
    alpha: float = 0.05,
    n_permutations: int = 1000,
    channel_correction: str = "fdr_bh",
    zones: Any = "auto",
    max_zones: int = 8,
    zone_method: str = "nmf",
    allow_unverified: bool = False,
    by_trial: bool = False,
    picks: Any = None,
    by: Sequence[str] = ("stim_site",),
    n_jobs: int = 1,  # noqa: ARG001  (spec §5.10 signature; documented: the computation runs in one process)
    random_state: Any = None,
    verbose: Any = None,  # noqa: ARG001  (spec §5.10 signature; read by the @verbose decorator)
) -> "TFFit":
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
    n_jobs
        Accepted for the common signature of the package; the computation runs in one
        process (100 channels x 50 trials x 2 048 samples x 40 frequencies take under 60 s).
    random_state : int, numpy Generator or None
        Seed of the permutations and of the zone decomposition. None draws a seed from the
        operating system and records it in ``settings`` and the provenance, so that
        :func:`evoparam.reproduce` regenerates the result (REQ-007).
    verbose : bool, str or None
        Level of the ``evoparam`` logger for the call (spec §9.8): one INFO record at the
        end; ``"progress"`` adds a progress bar over the conditions (extra ``progress``).

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
    opts: dict[str, Any] = {
        "transform": transform,
        "gamma": gamma,
        "p2": p2,
        "baseline": baseline,
        "baseline_mode": baseline_mode,
        "baseline_level": baseline_level,
        "stats": stats,
        "window": window,
        "channel_correction": channel_correction,
    }
    published = _published(zones, allow_unverified, opts)
    _check_options(opts, zone_method)
    axis = _time_axis(ts.times, opts["window"], opts["baseline"], opts["baseline_level"])

    grid = _freq_grid(freqs, ts.sfreq)
    picks = _picks(ts, picks)
    _check_group_fit(ts, picks, group_fit)
    labels, by_columns = _labels(ts, by)
    layout = _Layout(list(dict.fromkeys(labels)), labels, picks)
    if opts["stats"] == "cluster":
        _warn_few_trials(layout)
    kernels = make_kernels(
        opts["transform"],
        grid,
        ts.sfreq,
        ts.n_times,
        gamma=opts["gamma"],
        p2=opts["p2"],
        n_cycles=n_cycles,
        time_bandwidth=time_bandwidth,
        stft_window=stft_window,
    )
    if random_state is None:
        # a seed drawn from the operating system and recorded in the settings, so that the
        # result can be regenerated from its provenance (REQ-007, REQ-333)
        random_state = int(np.random.default_rng().integers(2**31 - 1))
    setup = _Setup(
        ts,
        kernels,
        axis,
        opts["baseline_mode"],
        opts["baseline_level"],
        (opts["stats"], alpha, n_permutations),
        np.random.default_rng(random_state),
    )
    n_rows = len(layout.groups) * len(picks)
    want_zones = zones == "auto"
    shape = (n_rows, len(grid), len(axis.out_idx))
    published_region = None if published is None else zone_mask(published, grid, axis.out_times)
    first = _first_pass(setup, layout, shape, group_fit, published_region)
    row_p, row_alpha, significant = _significance(first, alpha, opts["channel_correction"])
    metric_rows = []
    halves = None
    if want_zones:
        halves = _split_halves(setup, layout, significant, shape)
    elif published is not None:
        metric_rows = _published_metrics(
            setup, layout, (published, first.stash, significant), by_trial=by_trial
        )
    require("pandas", extra="data", caller="evoparam.tf.api.tf_fit")
    import pandas as pd

    settings = {
        "transform": opts["transform"],
        "transform_params": kernels.params,
        "freqs": grid.tolist(),
        "window": list(axis.window),
        "baseline": list(axis.baseline),
        "baseline_mode": opts["baseline_mode"],
        "baseline_level": opts["baseline_level"],
        "stats": opts["stats"],
        "alpha": alpha,
        "n_permutations": n_permutations,
        "channel_correction": opts["channel_correction"],
        "row_p_value": [None if not np.isfinite(v) else float(v) for v in row_p],
        "row_alpha": [float(v) for v in row_alpha],
        "zones": zones,
        "max_zones": max_zones,
        "zone_method": zone_method,
        "by": list(by_columns),
        "by_trial": by_trial,
        "rows": [list(row) for row in first.rows],
        "n_trials": [len(t) for t in first.row_trials],
        "half_support_s": (kernels.half_support / ts.sfreq).tolist(),
        "footprint": group_fit is not None,
        "random_state": random_state,
        "modality": ts.modality,
        "package_version": __version__,
    }
    if published is not None:
        settings["published_zone"] = {
            k: published.get(k)
            for k in ("name", "time", "freq", "intervals", "source", "reference", "verified")
        }
    if halves is not None:
        _store_halves(first.maps, settings, halves)
    method = zone_method if want_zones else ("published" if published is not None else "none")
    result = TFFit(
        freqs=grid,
        times=axis.out_times,
        maps=first.maps,
        tmaps=first.tmaps,
        significant=significant,
        zones=None,
        zone_metrics=None,
        method=method,
        settings=settings,
        provenance=ts.provenance.copy(),
    )
    if want_zones:
        table, regions = _zones_of(result, zone_method, max_zones, setup.rng)
        if len(table):
            # second pass: the zone metrics need the trials
            metric_rows = _zone_metrics(setup, result, regions, layout, by_trial=by_trial)
            table["p_value"] = _zone_p_values(table, metric_rows, n_rows)
        result.zones = table
    elif published is not None:
        result.zones = pd.DataFrame([_published_zone_row(published)])
    else:
        result.zones = pd.DataFrame(columns=list(ZONE_COLUMNS))
    result.zone_metrics = pd.DataFrame(metric_rows, columns=list(METRIC_COLUMNS))
    unit = MODE_UNITS[opts["baseline_mode"]]
    result.zone_metrics.attrs["units"] = {**METRIC_COLUMNS, "value": unit}
    if hasattr(result.zones, "attrs"):
        result.zones.attrs["units"] = {**ZONE_COLUMNS, "peak_value": unit, "mean_value": unit}
    # REQ-007: the step holds every keyword argument that changes the result, by its name,
    # so that evoparam.reproduce calls tf_fit again with them (the frequencies as the grid
    # used, the channels by name); a group fit is recorded as used ("footprint")
    step = {k: v for k, v in settings.items() if k not in ("rows", "row_p_value", "row_alpha")}
    step.update(
        gamma=opts["gamma"],
        p2=opts["p2"],
        n_cycles=n_cycles,
        time_bandwidth=time_bandwidth,
        stft_window=stft_window,
        allow_unverified=allow_unverified,
        picks=[ts.ch_names[c] for c in picks],
    )
    result.provenance = ts.provenance.with_step(
        "evoparam.tf.tf_fit", step, duration_s=_time.perf_counter() - start_clock
    )
    logger.info(
        "tf_fit: %d row(s) (channel x condition), %d zone(s), %.2f s",
        n_rows,
        len(result.zones),
        _time.perf_counter() - start_clock,
    )
    return result


###################################################################################################
## tf_fit, step by step


class _Axis(NamedTuple):
    """The time axis of the maps: windows (s), output samples, baseline and test pixels."""

    window: tuple[float, float]
    baseline: tuple[float, float]
    out_idx: Any
    out_times: Any
    ref_times: Any
    test_window: Any


class _Layout(NamedTuple):
    """The map rows: every condition (label of the trials) times every channel picked."""

    groups: list[Any]
    labels: Any
    picks: list[int]


def _each_row(layout: _Layout, desc: str | None = None) -> Iterator[tuple[int, Any, Any, int]]:
    """Yield the position, condition, trial indices and channel of every map row.

    ``desc``: show a progress bar over the conditions (under ``verbose="progress"``).
    """
    groups = layout.groups
    r = -1
    for label in groups if desc is None else progress(groups, total=len(groups), desc=desc):
        trial_idx = np.flatnonzero(layout.labels == label)
        for ch in layout.picks:
            r += 1
            yield r, label, trial_idx, ch


@dataclass
class _Setup:
    """What every map row is computed with; ``tests``: the statistic, alpha, permutations."""

    ts: Any
    kernels: Any
    axis: _Axis
    baseline_mode: str
    baseline_level: str
    tests: tuple[str, float, int]
    rng: np.random.Generator

    def power(self, trial_idx: Any, ch: int) -> tuple[Any, Any, Any]:
        """Return the trials of one row, their power and the pixels with enough trials."""
        return _row_power(self.ts, trial_idx, ch, self.kernels, self.axis.out_idx)

    def normalize(self, power: Any, trials: Any = None) -> tuple[Any, Any, Any]:
        """Normalise the power of one row to its baseline (see :func:`normalize`)."""
        return normalize(
            power, self.axis.ref_times, self.baseline_mode, self.baseline_level, trials=trials
        )

    def test(self, test: Any, power: Any, enough: Any) -> tuple[Any, Any]:
        """Return the T map and the pixel p-values of one row (see :func:`_row_test`)."""
        stats, alpha, n_permutations = self.tests
        tested = self.axis.test_window & enough
        return _row_test(
            stats, test, power, self.axis.ref_times, tested, alpha, n_permutations, self.rng
        )

    def adjusted(self, p: Any) -> Any:
        """Return the pixel p-values adjusted within the row (see :func:`within_map_p`)."""
        return within_map_p(p, self.tests[0])


@dataclass
class _FirstPass:
    """The maps of every row, before the correction across rows."""

    maps: dict[str, Any]
    tmaps: Any
    padj: Any  # pixel p-values adjusted within each row
    padj_residual: Any
    stash: list[Any]  # published zone: each row's values inside the zone
    rows: list[tuple[Any, Any]] = field(default_factory=list)
    row_trials: list[Any] = field(default_factory=list)


def _published(zones: Any, allow_unverified: bool, opts: dict[str, Any]) -> Any:
    """Return the registry entry of a published zone (None otherwise).

    A published zone sets, in ``opts``, the settings of its source given in the registry.
    """
    if isinstance(zones, str) and zones.startswith("published:"):
        published = resolve_published(zones.split(":", 1)[1], allow_unverified)
        pub = published.get("settings", {})
        for key in _PUBLISHED_SETTINGS:
            opts[key] = pub.get(key, opts[key])
        opts["baseline"] = tuple(pub.get("baseline", opts["baseline"]))
        opts["window"] = tuple(pub.get("window", opts["window"]))
        return published
    if zones not in ("auto", "none"):
        raise SettingsError(f"zones must be 'auto', 'none' or 'published:<name>', not {zones!r}.")
    return None


# the settings of tf_fit that a published zone may set (besides its baseline and window)
_PUBLISHED_SETTINGS = (
    "transform",
    "gamma",
    "p2",
    "baseline_mode",
    "baseline_level",
    "stats",
    "channel_correction",
)


def _check_options(opts: dict[str, Any], zone_method: str) -> None:
    """Raise SettingsError when an option of :func:`tf_fit` is not one of its values."""
    for name, value, allowed in (
        ("transform", opts["transform"], TRANSFORMS),
        ("baseline_mode", opts["baseline_mode"], BASELINE_MODES),
        ("baseline_level", opts["baseline_level"], BASELINE_LEVELS),
        ("stats", opts["stats"], STATS),
        ("channel_correction", opts["channel_correction"], CHANNEL_CORRECTIONS),
        ("zone_method", zone_method, ZONE_METHODS),
    ):
        if value not in allowed:
            raise SettingsError(f"{name} must be one of {allowed}, not {value!r}.")
    if opts["baseline_level"] == "full_epoch" and opts["baseline_mode"] == "zscore":
        raise SettingsError(
            "baseline_level='full_epoch' with baseline_mode='zscore' is "
            "undefined; use 'db', 'percent', 'logratio' or 'ratio'."
        )


def _time_axis(times: Any, window: Any, baseline: Any, baseline_level: str) -> _Axis:
    """Return the time axis of the maps: from the baseline (or the epoch start) to the window end.

    The window is clipped to the epoch.
    """
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
    ref_times = in_base[out_idx] if baseline_level != "full_epoch" else np.ones(len(out_idx), bool)
    return _Axis(window, baseline, out_idx, times[out_idx], ref_times, in_window[out_idx])


def _check_group_fit(ts: Any, picks: list[int], group_fit: Any) -> None:
    """Raise SettingsError when ``group_fit`` has no fit for a channel picked."""
    if group_fit is not None:
        missing = [ts.ch_names[c] for c in picks if ts.ch_names[c] not in group_fit.ch_names]
        if missing:
            raise SettingsError(
                f"group_fit has no fit for channels {missing}: it was made on "
                f"a different channel set."
            )


def _warn_few_trials(layout: _Layout) -> None:
    """Warn when a condition has too few trials for a sign-flip permutation."""
    smallest = min(int(np.sum(layout.labels == g)) for g in layout.groups)
    if smallest < MIN_CLUSTER_TRIALS:
        _warn(
            f"Only {smallest} trials in a condition: the sign-flip permutation "
            f"has few distinct patterns (coarse p-values).",
            stacklevel=3,
        )


def _first_pass(
    setup: _Setup, layout: _Layout, shape: tuple[int, int, int], group_fit: Any, region: Any
) -> _FirstPass:
    """Return the power, normalised and T maps of every row, and the pixel p-values.

    With ``group_fit``, also the footprint / residual split; with a published ``region``,
    each row's values inside it.
    """
    maps: dict[str, Any] = {key: np.full(shape, np.nan) for key in ("total", "total_norm")}
    if group_fit is not None:
        for key in (
            "evoked_footprint",
            "residual",
            "cross",
            "footprint_norm",
            "residual_norm",
            "residual_t",
            "residual_significant",
        ):
            maps[key] = np.full(shape, np.nan)
    out = _FirstPass(
        maps=maps,
        tmaps=np.full(shape, np.nan),
        padj=np.full(shape, np.nan),
        padj_residual=np.full(shape, np.nan) if group_fit is not None else None,
        stash=[],
    )
    for r, label, trial_idx, ch in _each_row(layout, desc="tf_fit"):
        name = setup.ts.ch_names[ch]
        out.rows.append((name, label))
        out.row_trials.append(trial_idx)
        filled, power, enough = setup.power(trial_idx, ch)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            maps["total"][r] = np.nanmean(power, axis=0)
        per_trial, reported, test = setup.normalize(power)
        maps["total_norm"][r] = reported
        t, p = setup.test(test, power, enough)
        out.tmaps[r], out.padj[r] = t, setup.adjusted(p)
        if group_fit is not None:
            fit = _channel_fit(group_fit, name, label)
            out.padj_residual[r] = _split_row(setup, maps, r, fit, (filled, power, enough))
        if region is not None:
            cells = region & enough
            out.stash.append(
                (cells, per_trial[:, cells], test[:, cells], t[cells], reported[cells])
            )
    return out


def _split_row(
    setup: _Setup, maps: dict[str, Any], r: int, fit: Any, row: tuple[Any, Any, Any]
) -> Any:
    """Split the power of row ``r`` into the fitted footprint, the residual and the cross term.

    ``row``: the trials, their power and the pixels with enough trials. Returns the residual's
    pixel p-values adjusted within the row.
    """
    filled, power, enough = row
    out_idx = setup.axis.out_idx
    yhat = _footprint_curve(fit, setup.ts.times)
    c_fp = _power(setup.kernels, yhat[None], out_idx)
    c_res = _power(setup.kernels, filled - yhat[None], out_idx)
    p_fp = np.mean(np.abs(c_fp) ** 2, axis=0)
    p_res = np.where(np.isfinite(power), np.mean(np.abs(c_res) ** 2, axis=0), np.nan)
    cross = np.where(
        np.isfinite(power), np.mean(2 * np.real(c_fp * np.conj(c_res)), axis=0), np.nan
    )
    p_fp = np.where(np.isfinite(power), p_fp, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        maps["evoked_footprint"][r] = np.nanmean(p_fp, axis=0)
        maps["residual"][r] = np.nanmean(p_res, axis=0)
        maps["cross"][r] = np.nanmean(cross, axis=0)
    ref, sd = reference_level(maps["total"][r], setup.axis.ref_times, setup.baseline_mode)
    maps["footprint_norm"][r] = apply_mode(
        maps["evoked_footprint"][r], ref, sd, setup.baseline_mode
    )
    _, res_reported, res_test = setup.normalize(p_res)
    maps["residual_norm"][r] = res_reported
    rt, rp = setup.test(res_test, p_res, enough)
    maps["residual_t"][r] = rt
    return setup.adjusted(rp)


def _significance(first: _FirstPass, alpha: float, correction: str) -> tuple[Any, Any, Any]:
    """Return the rows' p-values, their levels and the significant pixels.

    The levels come from the correction across channels (§5.6). With the footprint split,
    the residual's significant pixels are added to the maps.
    """
    row_p = _row_p(first.padj)
    row_alpha = row_levels(row_p, alpha, correction)
    significant = (first.padj <= row_alpha[:, None, None]) & (row_alpha[:, None, None] > 0)
    if first.padj_residual is not None:
        residual_alpha = row_levels(_row_p(first.padj_residual), alpha, correction)
        first.maps["residual_significant"] = (
            (first.padj_residual <= residual_alpha[:, None, None])
            & (residual_alpha[:, None, None] > 0)
        ).astype(float)
    return row_p, row_alpha, significant


def _split_halves(
    setup: _Setup, layout: _Layout, significant: Any, shape: tuple[int, int, int]
) -> tuple[Any, list[Any], list[int], Any]:
    """Return the split halves of the rows with significant pixels (second pass on the trials).

    Their mean over the rows (for the NMF rank), each row's halves (float32) inside the box of
    the significant pixels (for the CP rank), the rows kept and the box.
    """
    halves = np.zeros((N_SPLITS, 2, shape[1], shape[2]))
    union: Any = significant.any(axis=0)
    box: Any = None
    if union.any():
        f_idx, t_idx = np.flatnonzero(union.any(axis=1)), np.flatnonzero(union.any(axis=0))
        box = (slice(f_idx[0], f_idx[-1] + 1), slice(t_idx[0], t_idx[-1] + 1))
    row_halves, halves_rows = [], []
    for r, _, trial_idx, ch in _each_row(layout):
        if not significant[r].any():
            continue
        _, power, _ = setup.power(trial_idx, ch)
        signed = np.where(significant[r], 1.0, 0.0)
        own = np.zeros((N_SPLITS, 2) + signed[box].shape, np.float32)
        n_tr = len(trial_idx)
        for s in range(N_SPLITS):
            order = setup.rng.permutation(n_tr)
            for h, part in enumerate((order[: n_tr // 2], order[n_tr // 2 :])):
                _, half_map, _ = setup.normalize(power, trials=part)
                contribution = np.nan_to_num(half_map) * signed
                halves[s, h] += contribution
                own[s, h] = contribution[box]
        row_halves.append(own)
        halves_rows.append(r)
    halves /= max(shape[0], 1)
    return halves, row_halves, halves_rows, box


def _store_halves(maps: dict[str, Any], settings: dict[str, Any], split: Any) -> None:
    """Keep the split halves in the maps, and the rows and box of each row's halves."""
    halves, row_halves, halves_rows, box = split
    maps["zone_halves"] = halves.reshape(N_SPLITS * 2, *halves.shape[2:])
    if row_halves:
        # (2 N_SPLITS, n_freqs_box, n_times_box, n_rows_significant)
        maps["zone_halves_rows"] = np.stack(row_halves, axis=-1).reshape(
            (N_SPLITS * 2,) + row_halves[0].shape[2:] + (len(row_halves),)
        )
        settings["zone_halves_rows"] = {
            "rows": [int(v) for v in halves_rows],
            "freq_slice": [int(box[0].start), int(box[0].stop)],
            "time_slice": [int(box[1].start), int(box[1].stop)],
        }


def _published_metrics(
    setup: _Setup, layout: _Layout, zone: tuple[Any, list[Any], Any], *, by_trial: bool
) -> list[dict[str, Any]]:
    """Return the metric rows of a published zone.

    ``zone``: the registry entry, each row's values inside the zone and the significant pixels.
    """
    published, stash, significant = zone
    rows = []
    for r, label, trial_idx, ch in _each_row(layout):
        cells, in_trials, in_test, in_t, in_reported = stash[r]
        *values, trial_values = _region_stats(
            in_trials, in_test, in_t, in_reported, significant[r][cells]
        )
        rows += _metric_rows(
            (0, published["name"]),
            (setup.ts.ch_names[ch], label),
            values,
            (trial_idx, trial_values) if by_trial else None,
            setup.ts.modality,
        )
    return rows


def _zone_p_values(table: Any, metric_rows: list[dict[str, Any]], n_rows: int) -> list[float]:
    """Return each zone's p-value: its rows' smallest p-value, Bonferroni-corrected over rows."""
    p_by_zone: dict[Any, Any]
    p_by_zone = {}
    for row in metric_rows:
        if row["trial"] < 0 and np.isfinite(row["p_value"]):
            p_by_zone.setdefault(row["zone_id"], []).append(row["p_value"])
    return [min(1.0, min(p_by_zone.get(z, [np.nan])) * n_rows) for z in table["zone_id"]]


def _published_zone_row(zone: dict[str, Any]) -> dict[str, Any]:
    """Return the zone table row of a published zone."""
    return {
        "zone_id": 0,
        "sign": 0,
        "t_start": zone["time"][0],
        "t_end": zone["time"][1],
        "f_low": zone["freq"][0],
        "f_high": zone["freq"][1],
        "source": zone["source"],
    }


def _metric_row(
    zone: tuple[Any, Any], row: tuple[Any, Any], trial: Any, values: Any, modality: Any
) -> dict[str, Any]:
    """Return one row of the zone metrics.

    ``zone``: (id, name); ``row``: (channel, condition); ``values``: the value, then
    optionally t, p and the fractions of significant, increasing and decreasing pixels.
    """
    value, t, p, frac, frac_up, frac_down = list(values) + [np.nan] * (_N_METRICS - len(values))
    return {
        "zone_id": zone[0],
        "zone": zone[1],
        "channel": row[0],
        "condition": row[1],
        "trial": trial,
        "value": value,
        "t": t,
        "p_value": p,
        "fraction_significant": frac,
        "fraction_increase": frac_up,
        "fraction_decrease": frac_down,
        "modality": modality,
    }


def _metric_rows(
    zone: tuple[Any, Any], row: tuple[Any, Any], values: Any, trials: Any, modality: Any
) -> list[dict[str, Any]]:
    """Return a row's metric row for a zone, then one per trial when ``trials`` is given.

    ``trials``: (trial indices, per-trial values) or None.
    """
    out = [_metric_row(zone, row, -1, values, modality)]
    if trials is not None:
        out += [
            _metric_row(zone, row, int(i), (v,), modality) for i, v in zip(*trials, strict=False)
        ]
    return out


def _group_map(tf: Any) -> FloatArray:
    """Signed group map: mean over rows of the significant normalised pixels."""
    signed = np.where(tf.significant, np.nan_to_num(tf.maps["total_norm"]), 0.0)
    group: FloatArray = signed.mean(axis=0)
    return group


def _zones_of(
    tf: Any,
    method: Any,
    max_zones: Any,
    rng: np.random.Generator,
    cp_stability: str = "split_half",
) -> tuple[Any, list[Any]]:
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
        tensor = (
            np.moveaxis(np.maximum(sign * signed_rows, 0.0), 0, -1) if method == "cp" else None
        )
        split = None if halves is None else np.maximum(sign * halves, 0.0)
        trial_split = None if row_halves is None else np.maximum(sign * row_halves, 0.0)
        rank, stab = select_rank(
            x,
            split,
            max_zones,
            rng,
            method=method,
            tensor=tensor,
            cp_stability=cp_stability,
            row_halves=trial_split,
        )
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


def _zone_metrics(
    setup: _Setup, tf: Any, regions: Any, layout: _Layout, *, by_trial: bool
) -> list[Any]:
    """Return the metric rows of the data-driven zones, from the trials of every row."""
    s = tf.settings
    ts, kernels, out_idx = setup.ts, setup.kernels, setup.axis.out_idx
    rows = []
    for r, label, trial_idx, ch in _each_row(layout):
        masks = None if ts.masks is None else ts.masks[trial_idx, ch]
        invalid = _invalid_pixels(masks, kernels.half_support, ts.n_times, out_idx)
        coef = _power(kernels, _filled(ts.data[trial_idx, ch], masks), out_idx)
        power = np.where(invalid, np.nan, np.mean(np.abs(coef) ** 2, axis=0))
        enough = np.isfinite(tf.maps["total_norm"][r])
        power = np.where(enough[None], power, np.nan)
        per_trial, reported, test = normalize(
            power, setup.axis.ref_times, s["baseline_mode"], s["baseline_level"]
        )
        for z, region in enumerate(regions):
            *values, trial_values = _region_values(
                per_trial, test, tf.significant[r], tf.tmaps[r], reported, region, enough
            )
            rows += _metric_rows(
                (z, f"zone{z}"),
                (ts.ch_names[ch], label),
                values,
                (trial_idx, trial_values) if by_trial else None,
                ts.modality,
            )
    return rows


def tf_zones(
    tf: Any,
    *,
    method: str = "nmf",
    max_zones: int = 8,
    random_state: Any = None,
    cp_stability: str = "split_half",
) -> Any:
    """Data-driven zones of a TFFit (made with ``zones="auto"``), with another method or bound.

    Returns the zone table (without p-values, which need the trials).

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


def tf_compare(
    tf_a: Any,
    tf_b: Any,
    *,
    stats: str = "cluster",
    n_permutations: int = 1000,
    alpha: float = 0.05,
    random_state: Any = None,
) -> Any:
    """Contrast of two TFFits (conditions or modalities), with the map rows as units.

    Rows present in both (same channel and condition order) are paired (sign flips of the
    differences); otherwise the rows of the two fits are compared as independent groups
    (permutation of the group labels, Welch ``t``). Returns one row per cluster of pixels
    (``cluster``) or per significant pixel set (``fdr``): sign, bounds, mass, p-value.
    """
    require("pandas", extra="data", caller="evoparam.tf.api.tf_compare")
    import pandas as pd
    from scipy import stats as sstats

    if (
        tf_a.freqs.shape != tf_b.freqs.shape
        or tf_a.times.shape != tf_b.times.shape
        or not np.allclose(tf_a.freqs, tf_b.freqs)
        or not np.allclose(tf_a.times, tf_b.times)
    ):
        raise SettingsError("tf_compare needs the same frequencies and times in both fits.")
    if stats not in ("cluster", "fdr"):
        raise SettingsError("stats must be 'cluster' or 'fdr'.")
    a = tf_a.maps["total_norm"]
    b = tf_b.maps["total_norm"]
    paired = tf_a.settings.get("rows") == tf_b.settings.get("rows")
    rng = np.random.default_rng(random_state)
    t = _contrast_t(a, b, paired=paired)
    n_units = a.shape[0] if paired else a.shape[0] + b.shape[0]
    df = max(n_units - (1 if paired else 2), 1)
    if stats == "fdr":
        rows = _fdr_regions(tf_a, t, df, alpha)
        return pd.DataFrame(rows, columns=list(_COMPARE_COLUMNS))
    threshold = sstats.t.ppf(1 - alpha / 2, df)
    null = np.zeros(n_permutations)
    both: Any = np.concatenate([a, b]) if not paired else None
    for i in range(n_permutations):
        if paired:
            flips = rng.choice((-1.0, 1.0), a.shape[0])[:, None, None]
            perm_t, _ = t_map(flips * (a - b))
        else:
            order = rng.permutation(len(both))
            perm_t = _contrast_t(both[order[: len(a)]], both[order[len(a) :]], paired=paired)
        found = _clusters(perm_t, threshold)
        null[i] = max([m for _, _, m in found] or [0.0])
    rows = []
    for sign, region, mass in _clusters(t, threshold):
        p_value = float((1 + np.sum(null >= mass)) / (1 + n_permutations))
        rows.append(_compare_row(tf_a, sign, region, mass, p_value))
    table = pd.DataFrame(rows, columns=list(_COMPARE_COLUMNS))
    return table.sort_values("p_value", ignore_index=True)


_COMPARE_COLUMNS = ("sign", "t_start", "t_end", "f_low", "f_high", "mass", "n_pixels", "p_value")


def _contrast_t(xa: Any, xb: Any, *, paired: bool) -> Any:
    """Return the t map of the difference: paired, or Welch between independent rows."""
    if paired:
        t, _ = t_map(xa - xb)
        return t
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ma, mb = np.nanmean(xa, 0), np.nanmean(xb, 0)
        va, vb = np.nanvar(xa, 0, ddof=1), np.nanvar(xb, 0, ddof=1)
        na, nb = np.sum(np.isfinite(xa), 0), np.sum(np.isfinite(xb), 0)
        return (ma - mb) / np.sqrt(va / na + vb / nb)


def _compare_row(tf: Any, sign: int, region: Any, mass: float, p_value: float) -> dict[str, Any]:
    """Return the row of one region of the contrast: sign, bounds, mass and p-value."""
    fi, ti = np.nonzero(region)
    return {
        "sign": sign,
        "t_start": tf.times[ti.min()],
        "t_end": tf.times[ti.max()],
        "f_low": tf.freqs[fi.min()],
        "f_high": tf.freqs[fi.max()],
        "mass": mass,
        "n_pixels": int(region.sum()),
        "p_value": p_value,
    }


def _fdr_regions(tf: Any, t: Any, df: int, alpha: float) -> list[dict[str, Any]]:
    """Return the rows of the pixels significant after Benjamini-Hochberg, one per sign."""
    from scipy import stats as sstats

    from .stats import fdr_significant

    p = 2 * sstats.t.sf(np.abs(t), df)
    p[~np.isfinite(t)] = np.nan
    sig = fdr_significant(p, alpha)
    rows = []
    for sign in (1, -1):
        mask = sig & (np.nan_to_num(sign * t) > 0)
        if mask.any():
            mass = float(np.abs(t[mask]).sum())
            rows.append(_compare_row(tf, sign, mask, mass, float(np.nanmin(p[mask]))))
    return rows


def _clusters(tmap: Any, threshold: float) -> list[tuple[int, Any, float]]:
    """Return the sign, pixels and mass of every cluster of the t map beyond ``threshold``."""
    from scipy import ndimage

    out = []
    for sign in (1, -1):
        lab, n = ndimage.label(np.nan_to_num(sign * tmap) > threshold)
        for k in range(1, n + 1):
            region = lab == k
            out.append((sign, region, float(np.abs(tmap[region]).sum())))
    return out


def methods_text(tf: Any) -> Any:
    """Methods paragraph of a TFFit (spec REQ-219)."""
    return tf.methods_text()
