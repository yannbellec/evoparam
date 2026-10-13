# SPDX-License-Identifier: Apache-2.0
"""Find stimulation pulses in the raw signal: robust threshold, then template confirmation."""

import logging
import time as _time

import numpy as np

from evoparam._typing import Any, ArrayLike, FloatArray, NDArray
from evoparam.data._table import new_table
from evoparam.errors import DataError, EventError, SettingsError
from evoparam.utils._logging import verbose

from . import _spec
from ._score import (
    SCORE_CHANNELS,
    as_continuous,
    crossings,
    difference_score,
    far_from,
    pick_index,
    robust_z,
    score_index,
    shifted_snippets,
    snippet_lengths,
)
from .saturation import detect_saturation
from .site import identify_stim_site, vote_sites
from .trains import group_trains

logger = logging.getLogger(__name__)

__all__ = ["detect_pulses", "match_onsets", "estimate_delay", "detect_events", "EVENT_COLUMNS"]

METHODS = ("threshold", "template", "threshold+template")

# Detection procedures: the default one (below), and the specification's methods 1 and 2 as
# written (evoparam.events._spec).
PROCEDURES = ("evoparam", "spec")

# Fixed columns of the event table (spec §4.3.4), in order.
EVENT_COLUMNS = (
    "onset_sample",
    "onset_time",
    "peak_abs",
    "channel_argmax",
    "saturated",
    "n_saturated_channels",
    "stim_site_guess",
    "site_confidence",
    "train_id",
    "index_in_train",
    "detector",
    "threshold",
)

# Crossings closer than this belong to the same artifact (s).
MERGE_WINDOW = 0.020

# Window after a crossing in which the artifact peak is searched (s).
PEAK_WINDOW = 0.003

# Window before the peak in which the onset is searched (s), and the fraction of the
# peak score that marks the onset: low enough for the first sample of the artifact,
# high enough to skip the small switching transient some stimulators emit 1-2 ms
# earlier. 0.25 gave the most stable trigger-to-onset delay across the 25 ds004080 runs
# of the validation (0.1 and 0.5 were tried).
ONSET_WINDOW = 0.002
ONSET_FRACTION = 0.25

# A recovered pulse must still reach this fraction of the strength gate: below it, on
# low-resolution recordings, background fluctuations correlate with short templates.
RECOVERY_FRACTION = 0.1

# Lower edge of the grey zone, as a fraction of the strength gate.
GREY_FRACTION = 0.4


def _group_crossings(cross: Any, score: Any, sfreq: float, merge_s: Any = MERGE_WINDOW) -> Any:
    """One candidate per artifact: the crossing group's largest peak.

    Crossings closer than ``merge_s`` seconds form a group. Returns (peak_index, peak_score,
    first_crossing) arrays, one entry per group.
    """
    if len(cross) == 0:
        return np.array([], dtype=int), np.array([]), np.array([], dtype=int)
    merge = max(1, int(round(merge_s * sfreq)))
    width = max(1, int(round(PEAK_WINDOW * sfreq)))
    starts = np.r_[True, np.diff(cross) >= merge]
    group = np.cumsum(starts) - 1
    peaks, values, firsts = [], [], []
    for g in range(group[-1] + 1):
        members = cross[group == g]
        lo, hi = members[0], min(len(score), members[-1] + width + 1)
        ind = lo + int(np.argmax(score[lo:hi]))
        peaks.append(ind)
        values.append(score[ind])
        firsts.append(members[0])
    return np.array(peaks, dtype=int), np.array(values), np.array(firsts, dtype=int)


def _onsets_from_peaks(peaks: Any, firsts: Any, score: Any, sfreq: float) -> NDArray[Any]:
    """First sample of each artifact.

    The onset is the earliest score value that reaches ONSET_FRACTION of the peak,
    searched from ONSET_WINDOW before the peak (or from the group's first crossing, if
    earlier, so that the rising edge of a long artifact is never missed). Score index i
    is the change from sample i to i + 1, so the artifact starts at sample i + 1.
    """
    back = max(1, int(round(ONSET_WINDOW * sfreq)))
    onsets = np.empty(len(peaks), dtype=int)
    for row, (peak, first) in enumerate(zip(peaks, firsts, strict=False)):
        lo = max(0, min(peak - back, first - 1))
        above = np.flatnonzero(score[lo : peak + 1] >= ONSET_FRACTION * score[peak])
        onsets[row] = lo + above[0] + 1
    return onsets


def _suppress(positions: Any, strength: Any, min_samples: Any) -> NDArray[Any]:
    """Keep a position unless a stronger one lies within ``min_samples``.

    Two passes, so that a position suppressed only by a suppressed neighbour comes back.
    """
    order = np.argsort(positions, kind="stable")
    pos, val = positions[order], strength[order]
    lo = np.searchsorted(pos, pos - min_samples + 1)
    hi = np.searchsorted(pos, pos + min_samples)
    keep = np.ones(len(pos), dtype=bool)
    for _ in range(2):
        new = np.zeros(len(pos), dtype=bool)
        for ind in range(len(pos)):
            rivals = np.flatnonzero(keep[lo[ind] : hi[ind]]) + lo[ind]
            rivals = rivals[rivals != ind]
            new[ind] = len(rivals) == 0 or val[ind] >= val[rivals].max()
        if np.array_equal(new, keep):
            break
        keep = new
    out = np.zeros(len(positions), dtype=bool)
    out[order[keep]] = True
    return out


def _cluster(
    stacks: Any, rho_min: Any, n_members: Any
) -> tuple[NDArray[Any], NDArray[Any], FloatArray]:
    """Leader clustering of unit-norm snippets, tolerant to small onset shifts.

    ``stacks`` holds, for each candidate, its snippet at several shifts (middle row: no
    shift). A candidate joins the cluster whose template it matches best, over the
    shifts, when the correlation reaches ``rho_min``; otherwise it starts a new cluster
    with its unshifted snippet. A template is the normalised mean of the best-aligned
    snippets of the first ``n_members`` members of its cluster.

    Returns (labels, templates, best_rho).
    """
    middle = stacks.shape[1] // 2
    labels = np.full(len(stacks), -1)
    best = np.zeros(len(stacks))
    counts: list[Any]
    sums: list[Any]
    templates: list[Any]
    sums, counts, templates = [], [], []
    for row, stack in enumerate(stacks):
        if not np.any(stack[middle]):
            continue
        if templates:
            rho = stack @ np.array(templates).T  # shifts x clusters
            shift, k = np.unravel_index(int(np.argmax(rho)), rho.shape)
            if rho[shift, k] >= rho_min:
                labels[row], best[row] = k, rho[shift, k]
                if counts[k] < n_members:
                    sums[k] = sums[k] + stack[shift]
                    counts[k] += 1
                    templates[k] = sums[k] / np.linalg.norm(sums[k])
                continue
        labels[row], best[row] = len(templates), 1.0
        sums.append(stack[middle].copy())
        counts.append(1)
        templates.append(stack[middle].copy())
    return labels, np.array(templates), best


def _match(stacks: Any, templates: Any) -> Any:
    """Best correlation over shifts and templates: (rho, template index) per candidate."""
    if len(stacks) == 0 or len(templates) == 0:
        return np.zeros(len(stacks)), np.zeros(len(stacks), dtype=int)
    rho = np.einsum("nsf,kf->nsk", stacks, templates).max(axis=1)
    return rho.max(axis=1), rho.argmax(axis=1)


@verbose
def detect_pulses(  # noqa: C901, PLR0912, PLR0913, PLR0915  (public API, signature fixed by spec §5.2)
    data: FloatArray,
    sfreq: float,
    *,
    method: str = "threshold+template",
    threshold_sd: float = 20.0,
    min_interval: float | None = None,
    expected_rate: float | None = None,
    t0: float = 0.0,
    picks: ArrayLike | None = None,
    template: ArrayLike | None = None,
    rho_min: float = 0.8,
    n_template: int = 5,
    min_relative: float = 0.25,
    recovery_sd: float | None = None,
    refine: bool = True,
    ch_names: list[str] | None = None,
    ch_info: Any = None,
    saturation: bool = True,
    vote: bool = True,
    score_channels: int | None = SCORE_CHANNELS,
    procedure: str = "evoparam",
    return_scores: bool = False,
    site_method: str = "evidence",
    verbose: bool | str | None = None,  # noqa: ARG001  (read by the @verbose decorator)
) -> Any:
    """Find stimulation pulses from the artifact they leave in the signal.

    1. **Score.** ``d[n] = median_c |x_c[n + 1] - x_c[n]|`` over the picked channels
       (at most ``score_channels`` of them, evenly spaced), turned into a robust z-score
       (median and scaled MAD). The median over channels ignores transients that touch
       only a few channels.
    2. **Candidates.** Crossings of ``threshold_sd`` closer than 20 ms (or half of
       ``min_interval``, if shorter) are one artifact;
       its peak is the largest score within 3 ms of the crossings, and its onset is the
       first sample, within 2 ms before the peak, whose score reaches 25 % of the peak
       (``refine``). Without ``refine`` the onset is the first sample after the crossing.
    3. **Strength gate and repetition** (``"threshold+template"``). Each candidate's
       snippet (first difference of every picked channel, -1 ms to +3 ms, unit norm) is
       clustered by normalised cross-correlation ``rho`` (tolerance +/-0.5 ms on the
       onset; a cluster template is the mean of its first ``n_template`` members). A
       candidate whose peak reaches ``min_relative`` times the 90th percentile of the
       candidates' peaks is a pulse: stimulation artifacts reach hundreds to thousands
       of robust SD, movement transients and electrode pops rarely more than a few tens.
       A candidate between 0.4 and 1 times that gate is a pulse only if it repeats (its
       cluster has another member, ``rho >= rho_min``): a protocol delivers many pulses
       with the same artifact, a transient is unique. Weaker candidates are rejected
       (step 4 may recover them). Alternating polarities and several stimulation sites
       form several clusters. A user ``template`` also confirms the candidates it
       matches.
    4. **Recovery.** Crossings of the lower level ``recovery_sd`` that match a template
       with ``rho >= rho_min``, and reach a tenth of the strength gate, are added
       (``detector = "template"``): weaker pulses, for example at a lower current.
    5. **Refractory period.** Within ``min_interval`` the strongest pulse is kept, so a
       small switching click shortly before each pulse never replaces the pulse.

    Parameters
    ----------
    data : 1d or 2d array
        Continuous data, as [n_times] or [n_channels, n_times].
    sfreq : float
        Sampling frequency, in Hz.
    method : {"threshold+template", "threshold", "template"}
        ``"threshold"``: steps 1, 2 and 5 only (every crossing is a pulse). ``"template"``:
        only candidates that match a template (the user ``template``, or the clusters of
        the candidates that pass the strength gate).
    threshold_sd : float, default: 20
        Detection threshold, in robust standard deviations of the score.
    min_interval : float, optional
        Shortest time between two pulses, in seconds. Default: ``0.5 / expected_rate``
        when the rate is given, else 0.5 s.
    expected_rate : float, optional
        Pulse rate inside a train, in Hz. Sets ``min_interval``, the train gap, and the
        ``interval_flag`` column (intervals shorter than ``0.5 / rate`` or longer than
        ``2 / rate``).
    t0 : float, default: 0
        Time of the first sample, in seconds.
    picks : array-like, optional
        Channels used for the score and the snippets (indices, names or boolean mask).
    template : 1d array, optional
        A known artifact snippet, in the layout of the snippets (picked channels x
        samples, flattened); its correlation is computed on unit-norm versions.
    rho_min : float, default: 0.8
        Smallest normalised cross-correlation for a template match.
    n_template : int, default: 5
        Members averaged into a cluster template.
    min_relative : float, default: 0.25
        Strength gate, as a fraction of the 90th percentile of the candidates' peaks.
        The gate is relative: in a recording without any pulse, the strongest
        transients pass it, so check the ``rho`` and ``z_peak`` columns.
    recovery_sd : float, optional
        Lower level for the recovery pass. Default: ``threshold_sd / 4``.
    refine : bool, default: True
        Place the onset at the first artifact sample (step 2). If False, the onset is the
        first sample after the first threshold crossing (the skeleton's rule).
    ch_names : list of str, optional
        Channel names: needed for ``channel_argmax`` names and for the stimulation-site
        guess (:func:`identify_stim_site`); without them both use channel indices and the
        site is not guessed.
    ch_info : pandas.DataFrame, optional
        Channel table passed to :func:`identify_stim_site`.
    score_channels : int or None, default: 16
        Largest number of channels in the score's median (None: all picked channels).
        Snippets, saturation and the site guess always use every picked channel.
    vote : bool, default: True
        Replace each pulse's site guess by the vote of its block of consecutive pulses
        (:func:`vote_sites`); the per-pulse guess is kept in ``stim_site_pulse``.
    site_method : {"evidence", "spec"}, default: "evidence"
        Rule of :func:`identify_stim_site`; ``"spec"`` is spec §5.2 method 5 as reviewed.
    saturation : bool, default: True
        Fill ``saturated`` and ``n_saturated_channels`` with :func:`detect_saturation`.
    procedure : {"evoparam", "spec"}, default: "evoparam"
        ``"spec"``: steps 1 to 4 are replaced by methods 1 and 2 of the specification as
        written (:mod:`evoparam.events._spec`: the median over every picked channel; the
        first crossing after the refractory period; the onset at the largest change within
        +/-2 ms minus one sample, the last pre-artifact sample; a template from the first
        ``n_template`` detections scanned by normalised cross-correlation over the whole
        recording, which rejects threshold detections that do not match it and adds the
        pulses the threshold missed). ``min_relative``, ``recovery_sd`` and
        ``score_channels`` are not used. On the 25 runs of OpenNeuro ds004080 it is less
        accurate than the default (CHANGELOG).
    return_scores : bool, default: False
        Also return the z-score and the per-event correlation.
    verbose : bool, str or None
        Level of the ``evoparam`` logger for the call (spec §9.8): None or False inherit,
        True or ``"info"`` give one INFO record with the number of pulses found,
        ``"debug"`` also the threshold and the score of the detection.

    Returns
    -------
    events : pandas.DataFrame
        The event table: the fixed columns ``EVENT_COLUMNS`` (spec §4.3.4), then
        ``z_peak``, ``rho``, ``template_class`` and ``interval_flag``.
    z, rho : 1d arrays
        Only with ``return_scores``: the robust z-score of every sample step, and the
        correlation of every event with its template.

    Raises
    ------
    EventError
        No pulse found; the message gives the largest z-score and a threshold that would
        detect it.
    SettingsError
        Unknown method or invalid setting.
    """
    clock = _time.perf_counter()
    if method not in METHODS:
        raise SettingsError(f"method must be one of {METHODS}, not {method!r}.")
    if not threshold_sd > 0:
        raise SettingsError("threshold_sd must be positive.")
    if not 0 <= min_relative < 1:
        raise SettingsError("min_relative must be in [0, 1).")
    if not 0 < rho_min <= 1:
        raise SettingsError("rho_min must be in (0, 1].")
    if procedure not in PROCEDURES:
        raise SettingsError(f"procedure must be one of {PROCEDURES}, not {procedure!r}.")
    if expected_rate is not None and not expected_rate > 0:
        raise SettingsError("expected_rate must be positive.")
    data = as_continuous(data)
    if not sfreq > 0:
        raise DataError("sfreq must be positive.")
    n_channels, n_times = data.shape
    names = list(ch_names) if ch_names is not None else [str(ind) for ind in range(n_channels)]
    if len(names) != n_channels:
        raise DataError("ch_names must have one name per channel.")
    index = pick_index(n_channels, picks, names)
    if min_interval is None:
        min_interval = 0.5 / expected_rate if expected_rate else 0.5
    if recovery_sd is None:
        recovery_sd = threshold_sd / 4
    min_samples = max(1, int(round(min_interval * sfreq)))
    # Crossings of one artifact are merged, but never across two pulses of a fast train
    merge_s = min(MERGE_WINDOW, min_interval / 2)

    if procedure == "spec":
        score, z, onsets, strength, detector, rho, labels = _spec_candidates(
            data,
            sfreq,
            index,
            method,
            threshold_sd,
            min_samples,
            refine,
            template,
            rho_min,
            n_template,
        )
    else:
        score = difference_score(data, score_index(index, score_channels))
        z = robust_z(score)

        # Candidates at the detection threshold
        peaks, strength, firsts = _group_crossings(crossings(z, threshold_sd), z, sfreq, merge_s)
        onsets = _onsets_from_peaks(peaks, firsts, score, sfreq) if refine else firsts + 1

        n_pre, n_post, max_shift = snippet_lengths(sfreq)
        detector = np.array(["threshold"] * len(onsets), dtype=object)
        rho = np.ones(len(onsets))
        labels = np.full(len(onsets), -1)

        user = None
        if template is not None:
            user = np.asarray(template, dtype=float).ravel()
            if user.size != len(index) * (n_pre + n_post):
                raise SettingsError(
                    f"template must have {len(index) * (n_pre + n_post)} values "
                    f"({len(index)} channels x {n_pre + n_post} samples)."
                )
            user = user / np.linalg.norm(user)

        if method != "threshold" and len(onsets):
            # Strength gate. Strong candidates (>= min_relative of the reference) are pulses;
            # candidates in the grey zone (>= GREY_FRACTION of the gate) are pulses when they
            # repeat, that is when another candidate carries the same artifact.
            reference = np.percentile(strength, 90)
            relative = strength / reference
            strong = relative >= min_relative
            eligible = relative >= GREY_FRACTION * min_relative
            stacks = shifted_snippets(data, index, onsets, n_pre, n_post, max_shift)
            labels = np.full(len(onsets), -1)
            rho = np.zeros(len(onsets))
            labels[eligible], clusters, rho[eligible] = _cluster(
                stacks[eligible], rho_min, int(n_template)
            )
            sizes = np.bincount(labels[eligible], minlength=len(clusters))
            repeated = eligible & (sizes[np.maximum(labels, 0)] >= 2)
            confirmed = strong | repeated
            templates: Any
            templates = clusters.reshape(len(clusters), -1)
            if user is not None:
                user_rho, _ = _match(stacks, user[np.newaxis])
                hit = user_rho >= rho_min
                labels[hit & ~confirmed] = len(templates)
                confirmed |= hit
                rho = np.where(hit, np.maximum(rho, user_rho), rho)
                templates = np.vstack([templates, user])
            # Only the templates of confirmed pulses are used for recovery
            good = np.unique(labels[confirmed & (labels >= 0)])
            relabel = np.full(len(templates), -1)
            relabel[good] = np.arange(len(good))
            labels = np.where(labels >= 0, relabel[np.maximum(labels, 0)], -1)
            templates = templates[good]
            if method == "template":
                confirmed[:] = False
            peaks, strength, onsets = peaks[confirmed], strength[confirmed], onsets[confirmed]
            detector, rho, labels = detector[confirmed], rho[confirmed], labels[confirmed]

            # Recovery: weaker artifacts that match a template of the confirmed pulses
            if len(templates):
                low_peaks, low_strength, low_firsts = _group_crossings(
                    crossings(z, recovery_sd), z, sfreq, merge_s
                )
                low_onsets = (
                    _onsets_from_peaks(low_peaks, low_firsts, score, sfreq)
                    if refine
                    else low_firsts + 1
                )
                new = far_from(low_onsets, np.sort(onsets), min_samples)
                low_peaks, low_strength = low_peaks[new], low_strength[new]
                low_onsets = low_onsets[new]
                low_rho, low_label = _match(
                    shifted_snippets(data, index, low_onsets, n_pre, n_post, max_shift), templates
                )
                match = (low_rho >= rho_min) & (
                    low_strength >= RECOVERY_FRACTION * min_relative * reference
                )
                peaks = np.r_[peaks, low_peaks[match]]
                strength = np.r_[strength, low_strength[match]]
                onsets = np.r_[onsets, low_onsets[match]]
                detector = np.r_[detector, ["template"] * int(match.sum())]
                rho = np.r_[rho, low_rho[match]]
                labels = np.r_[labels, low_label[match]]

    # Refractory period: the strongest pulse wins
    keep = _suppress(onsets, strength, min_samples)
    order = np.argsort(onsets[keep], kind="stable")
    onsets = onsets[keep][order]
    strength, detector = strength[keep][order], detector[keep][order]
    rho, labels = rho[keep][order], labels[keep][order]

    if len(onsets) == 0:
        top = float(np.max(z)) if len(z) else 0.0
        raise EventError(
            f"No stimulation pulse found: the largest score is {top:.1f} robust SD "
            f"(threshold {threshold_sd}); try threshold_sd={max(top / 2, 1):.1f}, "
            "or check that the picked channels record the stimulation."
        )

    # Artifact size and the channel that carries it
    width = max(1, int(round(PEAK_WINDOW * sfreq)))
    peak_abs = np.zeros(len(onsets))
    argmax = []
    for row, onset in enumerate(onsets):
        lo, hi = max(0, onset - 1), min(n_times, onset + width + 1)
        dev = np.abs(data[index, lo:hi] - data[index, lo : lo + 1])
        per_channel = np.nanmax(dev, axis=1)
        best = int(np.argmax(per_channel))
        peak_abs[row] = per_channel[best]
        argmax.append(names[index[best]])

    # Template classes renumbered from 0 in order of first appearance (-1: no template)
    classes = np.full(len(labels), -1)
    seen: dict[Any, Any]
    seen = {}
    for row, label in enumerate(labels):
        if label >= 0:
            classes[row] = seen.setdefault(int(label), len(seen))

    events = new_table(
        {
            "onset_sample": onsets.astype("int64"),
            "onset_time": t0 + onsets / sfreq,
            "peak_abs": peak_abs,
            "channel_argmax": argmax,
            "saturated": np.zeros(len(onsets), dtype=bool),
            "n_saturated_channels": np.zeros(len(onsets), dtype="int64"),
            "stim_site_guess": [""] * len(onsets),
            "site_confidence": np.zeros(len(onsets)),
            "train_id": np.zeros(len(onsets), dtype="int64"),
            "index_in_train": np.zeros(len(onsets), dtype="int64"),
            "detector": detector.astype(str),
            "threshold": np.full(len(onsets), float(threshold_sd)),
            "z_peak": strength,
            "rho": rho,
            "template_class": classes.astype("int64"),
        }
    )

    if saturation:
        _, sat = detect_saturation(data, sfreq, events=events, ch_names=names, picks=index, t0=t0)
        if len(sat):
            # the number of distinct channels saturated at each pulse
            pairs = set(
                zip(
                    np.asarray(sat["event_index"]).tolist(),
                    np.asarray(sat["channel"]),
                    strict=True,
                )
            )
            counts = np.zeros(len(events), dtype="int64")
            for event, _ in pairs:
                counts[event] += 1
            events["n_saturated_channels"] = counts
            events["saturated"] = counts > 0

    if ch_names is not None:
        located = identify_stim_site(
            data,
            sfreq,
            events,
            ch_names=names,
            ch_info=ch_info,
            t0=t0,
            picks=index,
            method=site_method,
        )
        events["stim_site_guess"] = located["stim_site_guess"]
        events["site_confidence"] = located["site_confidence"]
        if vote and len(events) > 1:
            voted = vote_sites(events)
            events["stim_site_guess"] = voted["stim_site_guess"]
            events["site_confidence"] = voted["site_confidence"]
            events["stim_site_pulse"] = voted["stim_site_pulse"]

    events = group_trains(events, expected_rate=expected_rate)

    flags = [""] * len(events)
    if expected_rate:
        intervals = np.diff(events["onset_time"].to_numpy())
        for row, interval in enumerate(intervals, start=1):
            if interval < 0.5 / expected_rate:
                flags[row] = "short"
            elif interval > 2.0 / expected_rate:
                flags[row] = "long"
    events["interval_flag"] = flags

    columns = list(EVENT_COLUMNS) + ["z_peak", "rho", "template_class", "interval_flag"]
    if "stim_site_pulse" in events.columns:
        columns.append("stim_site_pulse")
    events = events[columns]
    # The detection step, carried into the TrialSet by evoparam.io.epoch (onsets = this
    # table) so that the methods text describes it (spec §5.13)
    from evoparam.data import Provenance

    settings = {
        "method": method,
        "threshold_sd": float(threshold_sd),
        "min_interval": float(min_interval),
        "expected_rate": None if expected_rate is None else float(expected_rate),
        "rho_min": float(rho_min),
        "n_template": int(n_template),
        "min_relative": float(min_relative),
        "recovery_sd": float(recovery_sd),
        "refine": bool(refine),
        "saturation": bool(saturation),
        "site": ch_names is not None,
        "vote": bool(vote and ch_names is not None),
        "site_method": site_method,
        "user_template": template is not None,
        "score_channels": score_channels,
    }
    events.attrs["provenance"] = (
        Provenance(source={"reader": "evoparam.events.detect_pulses"})
        .with_step("evoparam.events.detect_pulses", settings, n_events=int(len(events)))
        .to_dict()
    )
    if logger.isEnabledFor(logging.DEBUG):
        finite = np.asarray(z, float)[np.isfinite(z)]
        logger.debug(
            "detect_pulses: largest z-score %.1f, threshold %.1f SD, %d channel(s)",
            float(finite.max()) if finite.size else float("nan"),
            float(threshold_sd),
            n_channels,
        )
    logger.info(
        "detect_pulses (%s): %d pulse(s) in %.1f s of signal, %.2f s",
        method,
        len(events),
        n_times / sfreq,
        _time.perf_counter() - clock,
    )
    if return_scores:
        return events, z, events["rho"].to_numpy()
    return events


def _spec_candidates(
    data: Any,
    sfreq: float,
    index: Any,
    method: Any,
    threshold_sd: Any,
    min_samples: Any,
    refine: Any,
    template: Any,
    rho_min: Any,
    n_template: Any,
) -> Any:
    """Methods 1 and 2 of the specification as written (``procedure="spec"``)."""
    score = _spec.spec_score(data, index)
    z = _spec.spec_z(score)
    onsets, strength = _spec.spec_threshold(score, z, sfreq, threshold_sd, min_samples, refine)
    detector = np.array(["threshold"] * len(onsets), dtype=object)
    rho = np.ones(len(onsets))
    labels = np.full(len(onsets), -1)
    if method == "threshold":
        return score, z, onsets, strength, detector, rho, labels
    if template is not None:
        n_pre = max(1, int(round(0.001 * sfreq)))
        shape = np.asarray(template, dtype=float).reshape(len(index), -1)
        norms = np.linalg.norm(shape, axis=1, keepdims=True)
        tpl = np.divide(shape, norms, out=np.zeros_like(shape), where=norms > 0)
    else:
        tpl, n_pre = _spec.spec_template(data, index, onsets, sfreq, n_template)
    if tpl is None:
        empty = np.array([], dtype=int)
        return (score, z, empty, np.array([]), np.array([], dtype=object), np.array([]), empty)
    scan = _spec.spec_rho(data, index, tpl)
    tolerance = max(1, int(round(0.0005 * sfreq)))

    def rho_near(onset: Any) -> Any:
        lo, hi = max(0, onset - n_pre - tolerance), min(len(scan), onset - n_pre + tolerance + 1)
        return float(np.max(scan[lo:hi])) if hi > lo else 0.0

    matched = np.array([rho_near(onset) for onset in onsets])
    keep = matched >= rho_min
    if method == "template":
        keep[:] = False
    onsets, strength, detector = onsets[keep], strength[keep], detector[keep]
    rho, labels = matched[keep], np.zeros(int(keep.sum()), dtype=int)
    half = max(1, int(round(0.002 * sfreq)))
    peaks = _spec.spec_template_peaks(scan, rho_min, min_samples)
    found = peaks + n_pre
    new = (
        far_from(found, np.sort(onsets), min_samples)
        if len(onsets)
        else np.ones(len(found), dtype=bool)
    )
    found = found[new]
    found_strength = (
        np.array([float(np.max(z[max(0, f - half) : f + half + 1])) for f in found])
        if len(found)
        else np.array([])
    )
    onsets = np.r_[onsets, found].astype(int)
    strength = np.r_[strength, found_strength]
    detector = np.r_[detector, ["template"] * len(found)]
    rho = np.r_[rho, scan[peaks[new]]]
    labels = np.r_[labels, np.zeros(len(found), dtype=int)]
    return score, z, onsets, strength, detector, rho, labels


def match_onsets(
    detected: ArrayLike, reference: ArrayLike, tolerance: float = 0.01
) -> dict[str, Any]:
    """Compare detected onsets to reference onsets.

    Each reference onset is matched to the nearest unused detection within
    ``tolerance``, in time order.

    Parameters
    ----------
    detected, reference : 1d array
        Onsets, in seconds.
    tolerance : float, default: 0.01
        Largest time difference, in seconds, for a detection to count as a match.

    Returns
    -------
    results : dict
        ``hits`` (reference onsets found), ``misses``, ``false_alarms``, ``errors``
        (1d array, detected minus reference time for each hit, s), ``recall`` and
        ``precision``.
    """
    detected, reference = (
        np.sort(np.asarray(detected, float)),
        np.sort(np.asarray(reference, float)),
    )
    used, errors = np.zeros(len(detected), dtype=bool), []
    for ref in reference:
        if len(detected) == 0:
            break
        diffs = np.abs(detected - ref)
        diffs[used] = np.inf
        ind = int(np.argmin(diffs))
        if diffs[ind] <= tolerance:
            used[ind] = True
            errors.append(detected[ind] - ref)
    hits = len(errors)
    return {
        "hits": hits,
        "misses": len(reference) - hits,
        "false_alarms": int(np.sum(~used)),
        "errors": np.array(errors),
        "recall": hits / len(reference) if len(reference) else np.nan,
        "precision": hits / len(detected) if len(detected) else np.nan,
    }


def estimate_delay(detected: Any, reference: Any, tolerance: float = 0.01) -> tuple[float, float]:
    """Constant delay between the artifact onsets and the event times of a system.

    The event time written by a stimulator or an acquisition system (the trigger) and
    the first sample of the artifact in the recorded signal differ by a delay that is
    constant for one system: trigger latency, and the group delay of the amplifier's
    anti-aliasing filter. On OpenNeuro ds004080 it is about 1 ms (2 samples at 2048 Hz,
    1 sample at 512 Hz). Estimate it on recordings where both exist, then subtract it
    from detections made where the events file is missing.

    Parameters
    ----------
    detected, reference : 1d array or list of 1d arrays
        Onsets, in seconds; lists give several recordings, matched separately.
    tolerance : float, default: 0.01
        Matching tolerance, in seconds (see :func:`match_onsets`).

    Returns
    -------
    delay : float
        Median of detected minus reference time over all matched pairs, in seconds
        (NaN when nothing matches).
    spread : float
        Median absolute deviation of the errors around the delay, in seconds.
    """
    if len(detected) and np.ndim(detected[0]) == 0:
        detected, reference = [detected], [reference]
    errors = np.concatenate(
        [
            match_onsets(det, ref, tolerance)["errors"]
            for det, ref in zip(detected, reference, strict=False)
        ]
        or [np.array([])]
    )
    if len(errors) == 0:
        return np.nan, np.nan
    delay = float(np.median(errors))
    return delay, float(np.median(np.abs(errors - delay)))


def detect_events(
    source: Any,
    *,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
    t0: float | None = None,
    **kwargs: Any,
) -> Any:
    """Detect stimulation pulses in a continuous recording of any supported type.

    Parameters
    ----------
    source : ndarray, TrialSet or mne.io.Raw
        - ``ndarray`` [n_channels, n_times]: ``sfreq`` is required;
        - ``TrialSet`` holding one continuous trial (as returned by
          :func:`evoparam.io.read` without ``onsets``): its ``times`` give ``t0``;
        - MNE ``Raw``: data, sampling rate and channel names are taken from it, and
          times are seconds from the start of the data.
    sfreq : float, optional
        Sampling frequency (arrays only).
    ch_names : list of str, optional
        Channel names (arrays only; enables the site guess).
    t0 : float, optional
        Time of the first sample; overrides the value taken from ``source``.
    **kwargs
        Passed to :func:`detect_pulses`.

    Returns
    -------
    events : pandas.DataFrame
        The event table of :func:`detect_pulses`. Its provenance
        (``events.attrs["provenance"]``) is the record of ``source`` (a TrialSet) followed
        by one ``evoparam.events.detect_events`` step (REQ-294), which
        :func:`evoparam.io.epoch` carries into the trials.
    """
    start = _time.perf_counter()
    from evoparam.data import Provenance, TrialSet

    if isinstance(source, TrialSet):
        if source.n_trials != 1:
            raise DataError("detect_events needs continuous data: a TrialSet with one trial.")
        data: Any
        rate: Any
        data, rate = source.data[0], source.sfreq
        names: Any
        names = list(source.ch_names) if ch_names is None else ch_names
        first: Any
        first = float(source.times[0])
        kwargs.setdefault("ch_info", source.ch_info)
    elif (type(source).__module__ or "").split(".")[0] == "mne":
        data, rate, names, first = source.get_data(), source.info["sfreq"], source.ch_names, 0.0
    else:
        if sfreq is None:
            raise DataError("detect_events: sfreq is required for an array.")
        data, rate, names, first = source, sfreq, ch_names, 0.0
    events = detect_pulses(data, rate, ch_names=names, t0=first if t0 is None else t0, **kwargs)
    # one step for this call, after the record of the source: the settings of the detection
    # (the keyword arguments of detect_pulses, resolved)
    settings = dict(events.attrs["provenance"]["steps"][-1]["settings"])
    base = (
        source.provenance
        if isinstance(source, TrialSet)
        else Provenance(
            source={
                "format": "mne" if (type(source).__module__ or "").startswith("mne") else "array",
                "reader": "evoparam.events.detect_events",
            }
        )
    )
    duration = _time.perf_counter() - start
    events.attrs["provenance"] = base.with_step(
        "evoparam.events.detect_events", settings, duration_s=duration, n_events=int(len(events))
    ).to_dict()
    events.attrs["duration_s"] = duration
    return events
