# SPDX-License-Identifier: Apache-2.0
"""Find stimulation pulses in the raw signal: robust threshold, then template confirmation."""

import dataclasses
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
    quantile_score,
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


def artifact_size(
    data: Any, index: Any, onsets: Any, sfreq: float, *, argmax: bool = False
) -> Any:
    """Return the largest deflection after each onset, in data units.

    The deflection is taken on the picked channels within ``PEAK_WINDOW`` after each onset,
    from the sample before it. With ``argmax``, also returns the position (in ``index``) of
    the channel that carries it.
    """
    n_times = data.shape[-1]
    width = max(1, int(round(PEAK_WINDOW * sfreq)))
    size = np.zeros(len(onsets))
    best = np.zeros(len(onsets), dtype=int)
    for row, onset in enumerate(np.asarray(onsets, dtype=int)):
        lo, hi = max(0, onset - 1), min(n_times, onset + width + 1)
        dev = np.abs(data[index, lo:hi] - data[index, lo : lo + 1])
        per_channel = np.nanmax(dev, axis=1)
        best[row] = int(np.argmax(per_channel))
        size[row] = per_channel[best[row]]
    return (size, best) if argmax else size


@verbose
def detect_pulses(  # noqa: PLR0913  (spec §5.2 fixes 14 parameters, plus 9 keywords; DECISIONS.md §7.2)
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
    _check_settings(method, threshold_sd, min_relative, rho_min, procedure, expected_rate)
    data = as_continuous(data)
    if not sfreq > 0:
        raise DataError("sfreq must be positive.")
    n_channels, n_times = data.shape
    names, index = _channels(n_channels, ch_names, picks)
    rec = _Recording(data, sfreq, index, names, t0)
    opts = _Detection.resolve(
        sfreq,
        method,
        threshold_sd,
        (min_interval, expected_rate),
        (rho_min, n_template, min_relative, recovery_sd),
        refine,
    )
    score_used = "median"
    if procedure == "spec":
        score, z, cands = _spec_candidates(rec, opts, template)
    else:
        score = difference_score(data, score_index(index, score_channels))
        z = robust_z(score)
        cands, (score, z), score_used = _evoparam_candidates(rec, (score, z), opts, template)

    # Refractory period: the strongest pulse wins
    keep = _suppress(cands.onsets, cands.strength, opts.min_samples)
    cands = cands.select(keep)
    cands = cands.select(np.argsort(cands.onsets, kind="stable"))
    if len(cands.onsets) == 0:
        top = float(np.max(z)) if len(z) else 0.0
        raise EventError(
            f"No stimulation pulse found: the largest score is {top:.1f} robust SD "
            f"(threshold {threshold_sd}); try threshold_sd={max(top / 2, 1):.1f}, "
            "or check that the picked channels record the stimulation."
        )

    events = _event_table(rec, cands, threshold_sd)
    if saturation:
        _add_saturation(events, rec)
    if ch_names is not None:
        _add_sites(events, rec, ch_info, site_method, vote)
    events = group_trains(events, expected_rate=expected_rate)
    events["interval_flag"] = _interval_flags(events, expected_rate)
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
        "min_interval": float(opts.min_interval),
        "expected_rate": None if expected_rate is None else float(expected_rate),
        "rho_min": float(rho_min),
        "n_template": int(n_template),
        "min_relative": float(min_relative),
        "recovery_sd": float(opts.recovery_sd),
        "refine": bool(refine),
        "saturation": bool(saturation),
        "site": ch_names is not None,
        "vote": bool(vote and ch_names is not None),
        "site_method": site_method,
        "user_template": template is not None,
        "score_channels": score_channels,
        "score": score_used,
    }
    events.attrs["provenance"] = (
        Provenance(source={"reader": "evoparam.events.detect_pulses"})
        .with_step("evoparam.events.detect_pulses", settings, n_events=int(len(events)))
        .to_dict()
    )
    _log_detection(z, threshold_sd, n_channels)
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


# A grey-zone candidate is a pulse when its cluster has at least this many members.
_MIN_REPEATS = 2
# The strength gate is relative to this percentile of the candidates' peaks.
_REFERENCE_PERCENTILE = 90
# Fallback to the 90th percentile score: only with more than this many picked channels, and
# at this quantile of the non-flat channels.
_MIN_FALLBACK_CHANNELS = 2
_FALLBACK_QUANTILE = 0.9
# Interval flags: shorter than this fraction of the expected period, or longer than this
# multiple of it.
_SHORT_INTERVAL = 0.5
_LONG_INTERVAL = 2.0


@dataclasses.dataclass(frozen=True)
class _Recording:
    """The continuous data, its sampling rate, the picked channels and the channel names."""

    data: NDArray[Any]
    sfreq: float
    index: Any
    names: list[str]
    t0: float


@dataclasses.dataclass(frozen=True)
class _Detection:
    """The resolved settings of one detection."""

    method: str
    threshold_sd: float
    min_interval: float
    rho_min: float
    n_template: int
    min_relative: float
    recovery_sd: float
    refine: bool
    min_samples: int
    merge_s: float

    @classmethod
    def resolve(
        cls,
        sfreq: float,
        method: str,
        threshold_sd: float,
        rate: Any,
        template: Any,
        refine: bool,
    ) -> "_Detection":
        """Fill the defaults of the settings of :func:`detect_pulses`.

        ``rate`` is ``(min_interval, expected_rate)`` and ``template`` is
        ``(rho_min, n_template, min_relative, recovery_sd)``.
        """
        min_interval, expected_rate = rate
        rho_min, n_template, min_relative, recovery_sd = template
        if min_interval is None:
            min_interval = 0.5 / expected_rate if expected_rate else 0.5
        if recovery_sd is None:
            recovery_sd = threshold_sd / 4
        return cls(
            method=method,
            threshold_sd=threshold_sd,
            min_interval=min_interval,
            rho_min=rho_min,
            n_template=n_template,
            min_relative=min_relative,
            recovery_sd=recovery_sd,
            refine=refine,
            min_samples=max(1, int(round(min_interval * sfreq))),
            # Crossings of one artifact are merged, but never across two pulses of a fast
            # train
            merge_s=min(MERGE_WINDOW, min_interval / 2),
        )


@dataclasses.dataclass(frozen=True)
class _Candidates:
    """Candidate pulses: onset samples, strengths, detectors, correlations and clusters."""

    onsets: NDArray[Any]
    strength: NDArray[Any]
    detector: NDArray[Any]
    rho: NDArray[Any]
    labels: NDArray[Any]

    def select(self, keep: Any) -> "_Candidates":
        """Return the candidates ``keep`` (a mask or indices)."""
        return _Candidates(
            self.onsets[keep],
            self.strength[keep],
            self.detector[keep],
            self.rho[keep],
            self.labels[keep],
        )


def _check_settings(
    method: str,
    threshold_sd: float,
    min_relative: float,
    rho_min: float,
    procedure: str,
    expected_rate: Any,
) -> None:
    """Raise SettingsError for an invalid setting of :func:`detect_pulses`."""
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


def _channels(n_channels: int, ch_names: Any, picks: Any) -> tuple[list[str], Any]:
    """Return the channel names (indices without ``ch_names``) and the picked indices."""
    names = list(ch_names) if ch_names is not None else [str(ind) for ind in range(n_channels)]
    if len(names) != n_channels:
        raise DataError("ch_names must have one name per channel.")
    return names, pick_index(n_channels, picks, names)


def _threshold_candidates(
    rec: _Recording, scores: Any, level: float, opts: _Detection
) -> tuple[Any, Any]:
    """Return the strength and onset of each artifact crossing ``level`` (step 2)."""
    score, z = scores
    peaks, strength, firsts = _group_crossings(crossings(z, level), z, rec.sfreq, opts.merge_s)
    onsets = _onsets_from_peaks(peaks, firsts, score, rec.sfreq) if opts.refine else firsts + 1
    return strength, onsets


def _user_template(template: Any, n_channels: int, snip: Any) -> Any:
    """Return the user's template with unit norm, or None; checks its size."""
    if template is None:
        return None
    n_pre, n_post, _ = snip
    user = np.asarray(template, dtype=float).ravel()
    if user.size != n_channels * (n_pre + n_post):
        raise SettingsError(
            f"template must have {n_channels * (n_pre + n_post)} values "
            f"({n_channels} channels x {n_pre + n_post} samples)."
        )
    return user / np.linalg.norm(user)


def _evoparam_candidates(
    rec: _Recording, scores: Any, opts: _Detection, template: Any
) -> tuple[_Candidates, Any, str]:
    """Run steps 2 to 4 of :func:`detect_pulses`: candidates, strength gate, recovery.

    Returns the candidates, the ``(score, z)`` used and the name of that score.
    """
    strength, onsets = _threshold_candidates(rec, scores, opts.threshold_sd, opts)
    score_used = "median"
    if not len(strength) and len(rec.index) > _MIN_FALLBACK_CHANNELS:
        # No pulse on the median score: the artifact may reach only a minority of the
        # channels (ds004977, stimulating contacts switched to the stimulator: z 9 to 15
        # on the median, 47 to 322 on the 90th percentile of the non-flat channels, 10 at
        # most between pulses). The 90th percentile is then tried before giving up.
        score_used = "p90"
        score = quantile_score(rec.data, rec.index, _FALLBACK_QUANTILE)
        scores = (score, robust_z(score))
        strength, onsets = _threshold_candidates(rec, scores, opts.threshold_sd, opts)
    snip = snippet_lengths(rec.sfreq)
    cands = _Candidates(
        onsets,
        strength,
        np.array(["threshold"] * len(onsets), dtype=object),
        np.ones(len(onsets)),
        np.full(len(onsets), -1),
    )
    user = _user_template(template, len(rec.index), snip)
    if opts.method == "threshold" or not len(onsets):
        return cands, scores, score_used
    cands, templates, reference = _strength_gate(rec, cands, opts, user, snip)
    if len(templates):
        # Recovery: weaker artifacts that match a template of the confirmed pulses
        cands = _recover(rec, scores, cands, (templates, reference), opts, snip)
    return cands, scores, score_used


def _strength_gate(
    rec: _Recording, cands: _Candidates, opts: _Detection, user: Any, snip: Any
) -> tuple[_Candidates, Any, Any]:
    """Step 3: the candidates that pass the strength gate or repeat.

    Strong candidates (>= ``min_relative`` of the reference) are pulses; candidates in the
    grey zone (>= ``GREY_FRACTION`` of the gate) are pulses when they repeat, that is when
    another candidate carries the same artifact. Returns the pulses, the templates of their
    clusters and the reference strength.
    """
    n_pre, n_post, max_shift = snip
    onsets, strength = cands.onsets, cands.strength
    reference = np.percentile(strength, _REFERENCE_PERCENTILE)
    # the gate holds on both scales: the robust z of the first difference and the
    # size of the deflection (µV). A channel that saturates for an instant (an
    # electrode pop) has a steep edge, so a z near the pulses', but a deflection tens
    # of times smaller than the stimulation artifact (ds004696 sub-07: 0.1 to 2 mV
    # against 35 mV). The size of reference is taken among the candidates that pass on
    # z, since pops can outnumber the pulses (47 candidates for 5 pulses there).
    relative = strength / reference
    size = artifact_size(rec.data, rec.index, onsets, rec.sfreq)
    by_z = relative >= opts.min_relative
    size_reference = np.percentile(size[by_z] if by_z.any() else size, _REFERENCE_PERCENTILE)
    if size_reference > 0:
        relative = np.minimum(relative, size / size_reference)
    strong = relative >= opts.min_relative
    eligible = relative >= GREY_FRACTION * opts.min_relative
    stacks = shifted_snippets(rec.data, rec.index, onsets, n_pre, n_post, max_shift)
    labels = np.full(len(onsets), -1)
    rho = np.zeros(len(onsets))
    labels[eligible], clusters, rho[eligible] = _cluster(
        stacks[eligible], opts.rho_min, int(opts.n_template)
    )
    sizes = np.bincount(labels[eligible], minlength=len(clusters))
    repeated = eligible & (sizes[np.maximum(labels, 0)] >= _MIN_REPEATS)
    confirmed = strong | repeated
    templates: Any
    templates = clusters.reshape(len(clusters), -1)
    if user is not None:
        user_rho, _ = _match(stacks, user[np.newaxis])
        hit = user_rho >= opts.rho_min
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
    if opts.method == "template":
        confirmed[:] = False
    kept = _Candidates(onsets, strength, cands.detector, rho, labels).select(confirmed)
    return kept, templates, reference


def _recover(
    rec: _Recording,
    scores: Any,
    cands: _Candidates,
    gate: Any,
    opts: _Detection,
    snip: Any,
) -> _Candidates:
    """Step 4: add the crossings of ``recovery_sd`` that match a template (``gate[0]``).

    They must reach ``RECOVERY_FRACTION`` of the strength gate (``gate[1]`` is the reference).
    """
    templates, reference = gate
    n_pre, n_post, max_shift = snip
    low_strength, low_onsets = _threshold_candidates(rec, scores, opts.recovery_sd, opts)
    new = far_from(low_onsets, np.sort(cands.onsets), opts.min_samples)
    low_strength = low_strength[new]
    low_onsets = low_onsets[new]
    low_rho, low_label = _match(
        shifted_snippets(rec.data, rec.index, low_onsets, n_pre, n_post, max_shift), templates
    )
    match = (low_rho >= opts.rho_min) & (
        low_strength >= RECOVERY_FRACTION * opts.min_relative * reference
    )
    return _Candidates(
        np.r_[cands.onsets, low_onsets[match]],
        np.r_[cands.strength, low_strength[match]],
        np.r_[cands.detector, ["template"] * int(match.sum())],
        np.r_[cands.rho, low_rho[match]],
        np.r_[cands.labels, low_label[match]],
    )


def _event_table(rec: _Recording, cands: _Candidates, threshold_sd: float) -> Any:
    """Return the event table of the pulses, before saturation, sites and trains."""
    index, sfreq = rec.index, rec.sfreq
    onsets, labels = cands.onsets, cands.labels
    # Artifact size and the channel that carries it
    peak_abs, best = artifact_size(rec.data, index, onsets, sfreq, argmax=True)
    argmax = [rec.names[index[b]] for b in best]

    # Template classes renumbered from 0 in order of first appearance (-1: no template)
    classes = np.full(len(labels), -1)
    seen: dict[Any, Any]
    seen = {}
    for row, label in enumerate(labels):
        if label >= 0:
            classes[row] = seen.setdefault(int(label), len(seen))

    return new_table(
        {
            "onset_sample": onsets.astype("int64"),
            "onset_time": rec.t0 + onsets / sfreq,
            "peak_abs": peak_abs,
            "channel_argmax": argmax,
            "saturated": np.zeros(len(onsets), dtype=bool),
            "n_saturated_channels": np.zeros(len(onsets), dtype="int64"),
            "stim_site_guess": [""] * len(onsets),
            "site_confidence": np.zeros(len(onsets)),
            "train_id": np.zeros(len(onsets), dtype="int64"),
            "index_in_train": np.zeros(len(onsets), dtype="int64"),
            "detector": cands.detector.astype(str),
            "threshold": np.full(len(onsets), float(threshold_sd)),
            "z_peak": cands.strength,
            "rho": cands.rho,
            "template_class": classes.astype("int64"),
        }
    )


def _add_saturation(events: Any, rec: _Recording) -> None:
    """Fill ``saturated`` and ``n_saturated_channels`` (:func:`detect_saturation`)."""
    _, sat = detect_saturation(
        rec.data, rec.sfreq, events=events, ch_names=rec.names, picks=rec.index, t0=rec.t0
    )
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


def _add_sites(events: Any, rec: _Recording, ch_info: Any, site_method: str, vote: bool) -> None:
    """Fill the stimulation-site guess (:func:`identify_stim_site`, then the vote)."""
    located = identify_stim_site(
        rec.data,
        rec.sfreq,
        events,
        ch_names=rec.names,
        ch_info=ch_info,
        t0=rec.t0,
        picks=rec.index,
        method=site_method,
    )
    events["stim_site_guess"] = located["stim_site_guess"]
    events["site_confidence"] = located["site_confidence"]
    if vote and len(events) > 1:
        voted = vote_sites(events)
        events["stim_site_guess"] = voted["stim_site_guess"]
        events["site_confidence"] = voted["site_confidence"]
        events["stim_site_pulse"] = voted["stim_site_pulse"]


def _interval_flags(events: Any, expected_rate: Any) -> list[str]:
    """Return ``"short"`` or ``"long"`` for each interval far from the expected period."""
    flags = [""] * len(events)
    if expected_rate:
        intervals = np.diff(events["onset_time"].to_numpy())
        for row, interval in enumerate(intervals, start=1):
            if interval < _SHORT_INTERVAL / expected_rate:
                flags[row] = "short"
            elif interval > _LONG_INTERVAL / expected_rate:
                flags[row] = "long"
    return flags


def _log_detection(z: Any, threshold_sd: float, n_channels: int) -> None:
    """Log the largest z-score and the threshold at DEBUG level."""
    if logger.isEnabledFor(logging.DEBUG):
        finite = np.asarray(z, float)[np.isfinite(z)]
        logger.debug(
            "detect_pulses: largest z-score %.1f, threshold %.1f SD, %d channel(s)",
            float(finite.max()) if finite.size else float("nan"),
            float(threshold_sd),
            n_channels,
        )


def _spec_candidates(
    rec: _Recording, opts: _Detection, template: Any
) -> tuple[Any, Any, _Candidates]:
    """Methods 1 and 2 of the specification as written (``procedure="spec"``).

    Returns the score, its z-score and the candidates.
    """
    data, sfreq, index = rec.data, rec.sfreq, rec.index
    min_samples = opts.min_samples
    score = _spec.spec_score(data, index)
    z = _spec.spec_z(score)
    onsets, strength = _spec.spec_threshold(
        score, z, sfreq, opts.threshold_sd, min_samples, opts.refine
    )
    detector = np.array(["threshold"] * len(onsets), dtype=object)
    cands = _Candidates(onsets, strength, detector, np.ones(len(onsets)), np.full(len(onsets), -1))
    if opts.method == "threshold":
        return score, z, cands
    tpl, n_pre = _spec_template(rec, cands.onsets, opts, template)
    if tpl is None:
        empty = np.array([], dtype=int)
        none = _Candidates(empty, np.array([]), np.array([], dtype=object), np.array([]), empty)
        return score, z, none
    scan = _spec.spec_rho(data, index, tpl)
    tolerance = max(1, int(round(_SPEC_TOLERANCE * sfreq)))

    def rho_near(onset: Any) -> Any:
        """Return the largest correlation within the tolerance of ``onset``."""
        lo, hi = max(0, onset - n_pre - tolerance), min(len(scan), onset - n_pre + tolerance + 1)
        return float(np.max(scan[lo:hi])) if hi > lo else 0.0

    matched = np.array([rho_near(onset) for onset in onsets])
    keep = matched >= opts.rho_min
    if opts.method == "template":
        keep[:] = False
    onsets, strength, detector = onsets[keep], strength[keep], detector[keep]
    rho, labels = matched[keep], np.zeros(int(keep.sum()), dtype=int)
    half = max(1, int(round(_SPEC_HALF_WINDOW * sfreq)))
    peaks = _spec.spec_template_peaks(scan, opts.rho_min, min_samples)
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
    return (
        score,
        z,
        _Candidates(
            np.r_[onsets, found].astype(int),
            np.r_[strength, found_strength],
            np.r_[detector, ["template"] * len(found)],
            np.r_[rho, scan[peaks[new]]],
            np.r_[labels, np.zeros(len(found), dtype=int)],
        ),
    )


def _spec_template(rec: _Recording, onsets: Any, opts: _Detection, template: Any) -> Any:
    """Return the template of spec method 2 (the user's, unit norm per channel) and ``n_pre``."""
    if template is not None:
        n_pre = max(1, int(round(_SPEC_PRE * rec.sfreq)))
        shape = np.asarray(template, dtype=float).reshape(len(rec.index), -1)
        norms = np.linalg.norm(shape, axis=1, keepdims=True)
        tpl = np.divide(shape, norms, out=np.zeros_like(shape), where=norms > 0)
        return tpl, n_pre
    return _spec.spec_template(rec.data, rec.index, onsets, rec.sfreq, opts.n_template)


# Spec method 2: the snippet starts 1 ms before the onset; a detection matches the template
# scan within 0.5 ms; the strength of a template detection is the largest z within 2 ms.
_SPEC_PRE = 0.001
_SPEC_TOLERANCE = 0.0005
_SPEC_HALF_WINDOW = 0.002


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
