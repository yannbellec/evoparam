"""Does the result depend on the montage? Detections and latencies under several references."""

import itertools
import time as _time
import warnings

import numpy as np

from evoparam.errors import SettingsError
from evoparam.utils._optional import require

from ._common import check_trialset, masked_values
from .reference import rereference

__all__ = ["reference_sensitivity", "threshold_detector", "register_detector", "DETECTORS"]

# Detectors by name. "existence" (the existence test of spec §5.6, the default) is registered
# by the package root when evoparam is imported: the existence test is built on the fitting
# engine, which itself uses this module, so prep cannot import it (spec §4.2).
DETECTORS = {}


def register_detector(name, detector):
    """Register a detector for :func:`reference_sensitivity` under ``name``.

    ``detector(ts, **test_kwargs)`` returns ``(detected, latency)`` (arrays [n_channels])
    or ``(detected, latency, groups)`` with arrays [n_channels, n_groups] and the group
    names (for example one group per stimulation site).
    """

    if not callable(detector):
        raise SettingsError("detector must be callable.")
    DETECTORS[str(name)] = detector


def threshold_detector(ts, *, window=(0.009, 0.1), baseline=(-0.5, -0.05), threshold=3.4,
                       sd_floor_uv=50.0):
    """Fixed-threshold response detector (``detector="threshold"``; the interim default of
    build step S6, replaced by the existence test).

    The rule of the Utrecht SPES studies and of ER-detect's default: on the trial average,
    a channel responds when its most negative value in ``window`` is below
    ``-threshold`` times the standard deviation of the baseline (floored at
    ``sd_floor_uv`` microvolts for data in uV). The latency is the time of that minimum.

    Returns
    -------
    detected : 1d bool array [n_channels]
    latency : 1d array [n_channels], seconds (NaN when not detected)
    """

    values = masked_values(ts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        avg = np.nanmean(values, axis=0)
    base = (ts.times >= baseline[0]) & (ts.times <= baseline[1])
    win = (ts.times >= window[0]) & (ts.times <= window[1])
    sd = np.nanstd(avg[:, base], axis=1)
    if ts.units == "uV":
        sd = np.maximum(sd, sd_floor_uv)
    level = avg[:, base].mean(axis=1)
    seg = avg[:, win] - level[:, None]
    low = np.nanmin(seg, axis=1)
    detected = low < -threshold * sd
    latency = np.where(detected, ts.times[win][np.nanargmin(seg, axis=1)], np.nan)
    return detected, latency


def _key(name):
    """Comparison key of a channel: a bipolar ``"A1-A2"`` is compared with contact A1."""

    return str(name).split("-")[0]


def reference_sensitivity(ts, *, schemes=("bipolar", "car", "car_adaptive"),
                          detector="existence", test_kwargs=None, reference_kwargs=None):
    """Detections and latencies under several references, and their agreement.

    Each scheme is applied with :func:`rereference`, then the existence test of spec §5.6
    is run (``detector="existence"``: :func:`evoparam.existence.existence_detector`, the
    sign-flip permutation test of each channel and stimulation site, latency of the largest
    deflection of the trial average within the significant interval). For every pair of
    schemes, Cohen's kappa of the detections and the latency differences are computed on
    the channels (and sites) both schemes share (a bipolar channel ``"A1-A2"`` is compared
    with contact ``A1``).

    Parameters
    ----------
    ts : TrialSet
    schemes : tuple of str
    detector : str or callable, default: "existence"
        A registered name (``"existence"``, ``"threshold"``: :func:`threshold_detector`), or
        ``detector(ts, **test_kwargs) -> (detected, latency[, groups])`` (see
        :func:`register_detector`).
    test_kwargs : dict, optional
    reference_kwargs : dict, optional
        Passed to :func:`rereference` (for example ``random_state``, ``n_boot``).

    Returns
    -------
    pandas.DataFrame
        One row per pair: ``scheme_a``, ``scheme_b``, ``kappa``, ``n_channels``,
        ``n_detected_a``, ``n_detected_b``, ``n_both``, ``median_latency_diff_ms`` (b minus a,
        channels detected by both), ``max_abs_latency_diff_ms``. ``attrs["channels"]`` holds
        the per-channel table (one column of detections and one of latencies per scheme) and
        ``attrs["duration_s"]`` the run time, ``attrs["provenance"]`` the provenance of the
        trials with this step (for the methods text).
    """

    pd = require("pandas", extra="data", caller="evoparam.prep.reference_sensitivity")
    start = _time.perf_counter()
    check_trialset(ts)
    if detector is None:
        detector = "existence"
    if isinstance(detector, str):
        if detector not in DETECTORS:
            raise SettingsError(f"unknown detector {detector!r}; registered: {sorted(DETECTORS)} "
                                "('existence' is registered when evoparam is imported).")
        detector = DETECTORS[detector]
    detector_name = getattr(detector, "__name__", type(detector).__name__)
    test_kwargs = test_kwargs or {}
    reference_kwargs = reference_kwargs or {}
    per_scheme = {}
    for scheme in schemes:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ref = rereference(ts, scheme, **reference_kwargs)
        out = detector(ref, **test_kwargs)
        detected, latency = np.asarray(out[0]), np.asarray(out[1], dtype=float)
        keys = [_key(n) for n in ref.ch_names]
        if detected.ndim == 2:
            groups = list(out[2]) if len(out) > 2 else [str(g) for g in range(detected.shape[1])]
            keys = [f"{key}@{group}" if group else key for group in groups for key in keys]
            detected, latency = detected.T.ravel(), latency.T.ravel()
        per_scheme[scheme] = pd.DataFrame({"key": keys, "detected": detected.astype(bool),
                                           "latency": latency}
                                          ).drop_duplicates("key").set_index("key")

    rows = []
    for a, b in itertools.combinations(schemes, 2):
        joined = per_scheme[a].join(per_scheme[b], lsuffix="_a", rsuffix="_b", how="inner")
        da, db = joined["detected_a"].to_numpy(bool), joined["detected_b"].to_numpy(bool)
        n = len(joined)
        observed = np.mean(da == db) if n else np.nan
        expected = (da.mean() * db.mean() + (1 - da.mean()) * (1 - db.mean())) if n else np.nan
        kappa = (observed - expected) / (1 - expected) if n and expected < 1 else np.nan
        both = da & db
        diff = (joined["latency_b"] - joined["latency_a"]).to_numpy()[both] * 1000
        rows.append({"scheme_a": a, "scheme_b": b, "kappa": kappa, "n_channels": n,
                     "n_detected_a": int(da.sum()), "n_detected_b": int(db.sum()),
                     "n_both": int(both.sum()),
                     "median_latency_diff_ms": float(np.median(diff)) if len(diff) else np.nan,
                     "max_abs_latency_diff_ms": float(np.max(np.abs(diff))) if len(diff)
                     else np.nan})
    table = pd.DataFrame(rows)
    channels = pd.concat({scheme: frame for scheme, frame in per_scheme.items()}, axis=1)
    table.attrs["channels"] = channels
    table.attrs["detector"] = detector_name
    table.attrs["duration_s"] = _time.perf_counter() - start
    # the step, for the methods text of a pipeline that attaches this report (spec §6.2)
    table.attrs["provenance"] = ts.provenance.with_step(
        "evoparam.prep.reference_sensitivity",
        {"schemes": list(schemes), "detector": getattr(detector, "__name__", "custom"),
         "test_kwargs": {k: v for k, v in test_kwargs.items()
                         if isinstance(v, (int, float, str, bool, list, tuple))}},
        duration_s=table.attrs["duration_s"]).to_dict()
    return table


DETECTORS["threshold"] = threshold_detector
