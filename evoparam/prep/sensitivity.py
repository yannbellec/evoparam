"""Does the result depend on the montage? Detections and latencies under several references."""

import itertools
import time as _time
import warnings

import numpy as np

from evoparam.utils._optional import require

from ._common import check_trialset, masked_values
from .reference import rereference

__all__ = ["reference_sensitivity", "threshold_detector"]


def threshold_detector(ts, *, window=(0.009, 0.1), baseline=(-0.5, -0.05), threshold=3.4,
                       sd_floor_uv=50.0):
    """Interim response detector, until the existence test of build step S8.

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
                          detector=None, test_kwargs=None, reference_kwargs=None):
    """Detections and latencies under several references, and their agreement.

    Each scheme is applied with :func:`rereference`, then ``detector`` (default
    :func:`threshold_detector`; the existence test of build step S8 will replace it) is run.
    For every pair of schemes, Cohen's kappa of the detections and the latency differences
    are computed on the channels both schemes share (a bipolar channel ``"A1-A2"`` is
    compared with contact ``A1``).

    Parameters
    ----------
    ts : TrialSet
    schemes : tuple of str
    detector : callable, optional
        ``detector(ts, **test_kwargs) -> (detected bool [n_channels], latency s [n_channels])``.
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
    detector = detector or threshold_detector
    test_kwargs = test_kwargs or {}
    reference_kwargs = reference_kwargs or {}
    per_scheme = {}
    for scheme in schemes:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ref = rereference(ts, scheme, **reference_kwargs)
        detected, latency = detector(ref, **test_kwargs)
        per_scheme[scheme] = pd.DataFrame({"key": [_key(n) for n in ref.ch_names],
                                           "detected": detected, "latency": latency}
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
    table.attrs["duration_s"] = _time.perf_counter() - start
    # the step, for the methods text of a pipeline that attaches this report (spec §6.2)
    table.attrs["provenance"] = ts.provenance.with_step(
        "evoparam.prep.reference_sensitivity",
        {"schemes": list(schemes), "detector": getattr(detector, "__name__", "custom"),
         "test_kwargs": {k: v for k, v in test_kwargs.items()
                         if isinstance(v, (int, float, str, bool, list, tuple))}},
        duration_s=table.attrs["duration_s"]).to_dict()
    return table
