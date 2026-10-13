# SPDX-License-Identifier: Apache-2.0
"""Identify the stimulated contact pair from the stimulation artifact."""

import re
from typing import cast

import numpy as np

from evoparam._typing import Any, ArrayLike, FloatArray
from evoparam.data._table import as_text
from evoparam.errors import DataError, SettingsError
from evoparam.errors import warn as _warn
from evoparam.utils._names import contact_group, contact_number

from ._score import MAD_TO_SD, as_continuous, pick_index

__all__ = ["identify_stim_site", "vote_sites", "contact_group", "SITE_METHODS"]

# "evidence": held and leading contacts (the default, measured better on ds004080:
# docs/spec/DECISIONS.md, CHANGELOG); "spec": the rule of spec §5.2 method 5 as reviewed.
SITE_METHODS = ("evidence", "spec")

# Shortest reference lag before the onset (s): used when pulses are too close for the
# full lead window, and to decide that a train is too fast to identify the site.
REFERENCE_LAG = 0.002

# Lead window (s): the reference sample is taken this long before the onset, and the
# samples between it and the onset are where a stimulating contact may move first. The
# onset of detect_pulses is the start of the artifact shared by most channels; on the
# Utrecht SPES recordings (OpenNeuro ds004080) the stimulating contacts move up to 3 ms
# earlier, so 5 ms leaves a margin.
LEAD_WINDOW = 0.005

# Hold window (s, after the onset): once the common artifact is over (about 5 ms), a
# stimulating contact that the amplifier holds, or that stays at the rail, is flat here.
HOLD_WINDOW = (0.005, 0.100)

# Pre-pulse window used to measure each channel's noise level (s, relative to the onset).
NOISE_WINDOW = (-0.100, -0.005)

# A channel without noise of its own (held, or at the rail) gets this fraction of the
# median noise of the channels, so that its normalised artifact stays finite.
NOISE_FLOOR = 0.1

# Evidence that a channel carries a stimulating contact, besides the artifact size:
# - it moves: its artifact is at least MOVED_FRACTION of the largest one (in data units);
# - it leads: before the onset, it already moves by LEAD_MIN_SD times its noise and by
#   LEAD_MIN_FRACTION of its whole artifact;
# - it is held: at least HOLD_MIN_FRACTION of its steps in the hold window are smaller
#   than HOLD_TOLERANCE times the typical step of the channels before the pulse.
MOVED_FRACTION = 0.05
LEAD_MIN_SD = 10.0
LEAD_MIN_FRACTION = 0.3
HOLD_MIN_FRACTION = 0.5
HOLD_TOLERANCE = 0.1

# Two more kinds of evidence, for recordings where the stimulating contacts behave
# differently (measured on the Mayo Clinic CCEP sets of OpenNeuro, CHANGELOG):
# - it reaches the rail (counted as held, the same amplifier limit): within ±RAIL_WINDOW of
#   the onset, a channel that moves reaches (to RAIL_TOLERANCE) the largest absolute value
#   of the candidate channels over the whole recording, the input range of the amplifier:
#   the stimulating contacts receive the stimulation voltage itself (ds003708, ds004457,
#   ds004696: 34.8 mV). Only when at least two channels reach it at that pulse;
# - a silent pair: when the stimulating contacts are switched from the amplifier to the
#   stimulator, they carry no signal at all (ds004977: 3 to 5 uV of noise against 260 to
#   340 uV on the other contacts) and the largest artifact is on the contacts next to them.
#   Two adjacent contacts of one electrode whose noise is below NOISE_FLOOR times the median
#   noise and whose artifact is below MOVED_FRACTION of the median artifact of the other
#   contacts of the electrode (themselves not silent), with the largest normalised artifact
#   on the contact just outside the pair, are the site when exactly one pair qualifies.
RAIL_WINDOW = 0.005
RAIL_TOLERANCE = 1e-3

# Channel types that can carry a stimulating contact (channel table ``type`` column).
NEURAL_TYPES = {"ecog", "seeg", "dbs", "eeg", "lfp", "ieeg", "meg"}

# Names of channels that never are stimulating contacts: markers, triggers, status,
# physiological and auxiliary channels.
NON_NEURAL = re.compile(
    r"^(mkr|trig|sti|status|event|marker|dc\d|ecg|ekg|emg|eog|resp|"
    r"spo2|pulse|photic|aux|misc|ref$|gnd)",
    re.IGNORECASE,
)


def _robust_sd(values: Any) -> FloatArray:
    """Scaled MAD of each row (NaN-aware only when needed, as it is slower)."""
    median = np.nanmedian if np.isnan(values).any() else np.median
    return cast(
        "FloatArray",
        MAD_TO_SD * median(np.abs(values - median(values, axis=1, keepdims=True)), axis=1),
    )


def _first_sign(trace: Any) -> Any:
    """Sign of the first deflection: the first sample beyond half the largest one."""
    peak = np.nanmax(np.abs(trace))
    beyond: Any
    beyond = np.flatnonzero(np.abs(trace) >= 0.5 * peak) if peak > 0 else []
    return np.sign(trace[beyond[0]]) if len(beyond) else 0.0


def _silent_pair(
    names: Any, group: Any, raw_noise: Any, floor: Any, amplitude: Any, size: Any
) -> Any:
    """Positions of the one silent pair of adjacent contacts (comment above RAIL_WINDOW)."""
    quiet = np.flatnonzero((raw_noise > 0) & (raw_noise < NOISE_FLOOR * floor))
    if len(quiet) < 2:
        return None
    number: list[Any] = [contact_number(names[k]) for k in range(len(names))]  # None: no number
    top = int(np.argmax(size))
    found = []
    for a in quiet:
        for b in quiet:
            if not (
                a < b
                and group[a] == group[b]
                and number[a] is not None
                and number[b] is not None
                and abs(number[a] - number[b]) == 1
            ):
                continue
            others = [k for k in range(len(names)) if group[k] == group[a] and k not in (a, b)]
            if not others or np.median(raw_noise[others]) < NOISE_FLOOR * floor:
                continue
            if max(amplitude[a], amplitude[b]) >= MOVED_FRACTION * np.median(amplitude[others]):
                continue
            low, high = sorted((number[a], number[b]))
            if group[top] == group[a] and number[top] in (low - 1, high + 1):
                found.append((a, b))
    return found[0] if len(found) == 1 else None


def identify_stim_site(  # noqa: C901, PLR0912, PLR0913, PLR0915  (public API, signature fixed by spec §5.2)
    data: FloatArray,
    sfreq: float,
    events: Any,
    *,
    ch_names: list[str],
    ch_info: Any = None,
    window: float = 0.001,
    t0: float = 0.0,  # noqa: ARG001  (documented: kept for symmetry with the other functions)
    picks: ArrayLike | None = None,
    exclude: Any = NON_NEURAL,
    lead: float = LEAD_WINDOW,
    hold: tuple[float, ...] = HOLD_WINDOW,
    method: str = "evidence",
) -> Any:
    """Guess the stimulated contact pair of each pulse from the artifact.

    ``method="spec"`` applies the rule of spec §5.2 method 5 as reviewed, to the letter:
    ``a_c = max_{|t| <= window} |x_c(t) - x_c(-2 ms)|`` (data units) is ranked, the guess
    is the two largest channels ordered by channel index and ``site_confidence`` is
    ``(a_1 - a_3) / a_1``, multiplied by 1 when the two first deflections have opposite
    signs and, when ``ch_info`` has a ``group`` column, the two contacts are on the same
    shaft or grid, and by 0.5 otherwise. The default ``method="evidence"`` is described
    below; on ds004080 it agrees with ``events.tsv`` on 0.980 of the pulses against 0.61
    for the reviewed rule (docs/spec/DECISIONS.md).

    Each pulse is examined on its own. For each channel ``c``:

    - the artifact is ``a_c = max |x_c(t) - x_c(-lead)|`` for ``-lead <= t <= window``,
      and its size ``s_c = a_c / noise_c``, where ``noise_c`` is the scaled MAD over -100
      to -5 ms (a channel without noise gets a tenth of the median noise);
    - it *leads* when, before the onset, it already moves by 10 times its noise and by
      30 % of its artifact. The onset found by :func:`detect_pulses` is where the artifact
      shared by most channels starts (the common reference spreads it to every contact);
      a stimulating contact can move before it, when the stimulator connects to it;
    - it is *held* when, in the ``hold`` window, at least half of its steps are below a
      tenth of the channels' typical pre-pulse step: the amplifier holds a stimulating
      contact, or leaves it at the rail, while the other contacts recover;
      it also counts as held when, within 5 ms of the onset, its absolute value reaches
      (to 0.1 %) the largest absolute value of the candidate channels over the whole
      recording, the amplifier's input range, which the stimulating contacts reach (when
      at least two channels reach it at that pulse);
    - its evidence ``e_c`` (0, 1 or 2) counts how many of the two it shows, among the
      channels that move (artifact at least 5 % of the largest one).

    When the stimulating contacts are disconnected from the amplifier during stimulation
    they carry no signal: two adjacent contacts of one electrode with almost no noise
    (below a tenth of the median noise) and no artifact (below 5 % of the median artifact
    of the other contacts of that electrode), next to the contact with the largest
    artifact, are the site when exactly one such pair exists (score 1).

    Channels are ranked by evidence, then by size; ``stim_site_guess`` is ``"A-B"``, the
    two first channels ordered by channel index. Without held or leading channels this
    is the largest-artifact rule of the specification. ``site_confidence`` is the margin
    of the weaker contact of the pair (rank 2) over the best other channel (rank 3):
    ``(e_2 - e_3) / e_2`` when the evidence separates them, else ``(s_2 - s_3) / s_2``
    (the specification's ``(a_1 - a_3) / a_1`` stays high when one contact dominates and
    the second is a toss-up). It is halved when the first deflections of the pair do not
    have opposite signs or when the two contacts are not on the same electrode.

    On OpenNeuro ds004080 (25 runs, 199 pulses, without the channel table), this agrees
    with ``events.tsv`` on 99.5 % of the pulses after :func:`vote_sites`; on the Mayo sets
    (ds003708, ds004457, ds004696, ds004977: 15 runs, 162 detected pulses) on 98 %
    (CHANGELOG).

    Parameters
    ----------
    data : 1d or 2d array
        Continuous data, as [n_channels, n_times].
    sfreq : float
        Sampling frequency, in Hz.
    events : pandas.DataFrame
        Event table with an ``onset_sample`` column.
    ch_names : list of str
        Channel names.
    ch_info : pandas.DataFrame, optional
        Channel table; its ``group`` column (electrode, shaft or grid) is used for the
        electrode check, and channels with ``bad`` True are ignored. Without it, the
        group is guessed from the names (:func:`contact_group`).
    window : float, default: 0.001
        Part of the artifact after the onset used for its size, in seconds (at least one
        sample).
    t0 : float, default: 0
        Time of the first sample, in seconds (kept for symmetry with the other functions).
    picks : array-like, optional
        Channels that may be stimulation contacts (indices, names or boolean mask).
    exclude : re.Pattern or None
        Channel names that are never stimulating contacts (default: markers, triggers,
        ECG, EMG, EOG and other auxiliary channels). With a ``ch_info`` that has a
        ``type`` column, only neural types are kept instead.
    lead : float, default: 0.005
        Lead window, in seconds: the reference sample is taken this long before the onset
        (shortened to a quarter of the interval between pulses, but not below 2 ms).
    hold : tuple of float, default: (0.005, 0.1)
        Hold window, in seconds after the onset (cut before the next pulse).
    method : {"evidence", "spec"}, default: "evidence"
        ``"spec"``: the reviewed rule of spec §5.2 method 5 (``lead`` and ``hold`` are not
        used; the reference sample is 2 ms before the onset).

    Returns
    -------
    events : pandas.DataFrame
        Copy of ``events`` with ``stim_site_guess`` (str, "" when not identifiable),
        ``site_confidence`` (0-1) and ``polarity_check`` (bool) columns.
    """
    if method not in SITE_METHODS:
        raise SettingsError(f"method must be one of {SITE_METHODS}, got {method!r}.")
    data = as_continuous(data)
    n_channels, n_times = data.shape
    names = list(ch_names)
    if len(names) != n_channels:
        raise DataError("ch_names must have one name per channel.")
    index = pick_index(n_channels, picks, names)

    # the channel table, read column by column (a DataFrame, or a Table without pandas)
    info_names = [] if ch_info is None else list(np.asarray(ch_info["name"], dtype=object))
    if ch_info is not None and "type" in ch_info.columns:
        kinds = [str(kind).lower() for kind in np.asarray(ch_info["type"], dtype=object)]
        types = dict(zip(info_names, kinds, strict=False))
        index = np.array([ind for ind in index if types.get(names[ind], "") in NEURAL_TYPES])
    elif exclude is not None:
        index = np.array([ind for ind in index if not exclude.match(names[ind])])
    if ch_info is not None and "bad" in ch_info.columns:
        flagged = np.asarray(ch_info["bad"]).astype(bool)
        bad = {name for name, is_bad in zip(info_names, flagged, strict=False) if is_bad}
        index = np.array([ind for ind in index if names[ind] not in bad])
    if ch_info is not None and "group" in ch_info.columns:
        group_names = [str(name) for name in np.asarray(ch_info["group"], dtype=object)]
        groups = dict(zip(info_names, group_names, strict=False))
        group = [groups.get(name, contact_group(name)) for name in names]
    elif method == "spec":
        # the reviewed rule checks the shaft only "when ch_info.group is available"
        group = None
    else:
        group = [contact_group(name) for name in names]

    out = events.copy()
    onsets = np.asarray(events["onset_sample"], dtype=int)
    n_events = len(onsets)
    guesses = [""] * n_events
    confidence = np.zeros(n_events)
    polarity = np.zeros(n_events, dtype=bool)

    half = max(1, int(round(window * sfreq)))
    shortest = max(half + 1, int(round(REFERENCE_LAG * sfreq)))
    interval = float(np.median(np.diff(np.sort(onsets)))) if n_events > 1 else np.inf
    lag = max(shortest, int(round(min(lead * sfreq, 0.25 * interval))))
    noise_lo, noise_hi = (int(round(edge * sfreq)) for edge in NOISE_WINDOW)
    hold_lo, hold_hi = (int(round(edge * sfreq)) for edge in hold)
    ordered = np.sort(onsets)
    rail_span = int(round(RAIL_WINDOW * sfreq))
    names_index = [names[ind] for ind in index]
    # (method="spec" without ch_info.group has no groups; only the evidence rule uses these)
    group_index = None if group is None else [group[ind] for ind in index]
    # Largest absolute value of the candidate channels (one row at a time, no full copy)
    rail = max(
        (float(np.nan_to_num(np.nanmax(np.abs(data[ind])), nan=0.0)) for ind in index), default=0.0
    )

    if n_events > 1 and interval < 2 * shortest + 2 * half:
        _warn(
            "identify_stim_site: pulses are closer than the identification window "
            "(high-frequency train); the site is not identified.",
            stacklevel=2,
        )
        index = np.array([], dtype=int)

    if len(index) >= 2 and method == "spec":
        for row, onset in enumerate(onsets):
            found = _spec_site(data, index, onset, shortest, half, group)
            if found is None:
                continue
            pair, confidence[row], polarity[row] = found
            guesses[row] = f"{names[pair[0]]}-{names[pair[1]]}"
    elif len(index) >= 2:
        for row, onset in enumerate(onsets):
            lag_row = min(lag, onset)
            if lag_row < shortest or onset + half + 1 > n_times:
                continue
            # Only the samples around this pulse are copied, never the whole recording
            reference = data[index, onset - lag_row]
            deviation = data[index, onset - lag_row : onset + half + 1] - reference[:, None]
            amplitude = np.nan_to_num(np.nanmax(np.abs(deviation), axis=1), nan=0.0)
            if not amplitude.max() > 0:
                continue
            early = np.nan_to_num(np.nanmax(np.abs(deviation[:, 1:lag_row]), axis=1), nan=0.0)

            lo, hi = max(0, onset + noise_lo), max(0, onset + noise_hi)
            known = hi - lo >= 8
            if known:
                base = data[index, lo:hi]
                noise = _robust_sd(base)
                floor = np.median(noise[noise > 0]) if np.any(noise > 0) else 1.0
                raw_noise = np.nan_to_num(noise, nan=0.0)
                noise = np.maximum(raw_noise, NOISE_FLOOR * floor)
                median = np.nanmedian if np.isnan(base).any() else np.median
                typical_step = float(np.nanmedian(median(np.abs(np.diff(base, axis=1)), axis=1)))
            else:
                noise = np.ones(len(index))
                typical_step = np.nan
                raw_noise, floor = noise, 1.0
            size = amplitude / noise

            # Evidence needs the noise level, so not for a pulse at the very start
            moved = (amplitude >= MOVED_FRACTION * amplitude.max()) & known
            leads = (
                moved & (early >= LEAD_MIN_SD * noise) & (early >= LEAD_MIN_FRACTION * amplitude)
            )
            held = np.zeros(len(index), dtype=bool)
            start = onset + hold_lo
            later = ordered[np.searchsorted(ordered, onset, side="right") :]
            stop = min(onset + hold_hi, n_times, later[0] - lag if len(later) else n_times)
            if stop - start >= 4 and known:
                steps = np.abs(np.diff(data[index, start:stop], axis=1))
                flat = np.mean(steps <= HOLD_TOLERANCE * typical_step, axis=1)
                held = moved & (flat >= HOLD_MIN_FRACTION)
            if rail > 0:
                # Reaching the amplifier's range is the same evidence as being held there
                near = data[index, max(0, onset - rail_span) : onset + rail_span + 1]
                peak = np.nan_to_num(np.nanmax(np.abs(near), axis=1), nan=0.0)
                at_range = moved & (peak >= (1 - RAIL_TOLERANCE) * rail)
                # A range is reached by both contacts of the pair; one channel alone is
                # just the largest artifact of a recording that never saturates
                if at_range.sum() >= 2:
                    held = held | at_range
            evidence = leads.astype(int) + held.astype(int)

            silent = (
                _silent_pair(names_index, group_index, raw_noise, floor, amplitude, size)
                if known
                else None
            )
            order = np.lexsort((-size, -evidence))
            if silent is not None:
                rest = [k for k in order if k not in silent]
                order = np.array(list(silent) + rest)
            first, second = order[0], order[1]
            if len(order) > 2:
                third = order[2]
                e_2, e_3 = evidence[second], evidence[third]
                if e_2 > e_3:
                    score = (e_2 - e_3) / e_2
                elif size[second] > 0:
                    score = (size[second] - size[third]) / size[second]
                else:
                    score = 0.0
            else:
                score = 1.0
            if silent is not None:
                score = 1.0

            opposite = _first_sign(deviation[first]) * _first_sign(deviation[second]) < 0
            groups_of = cast("list[Any]", group)  # set whenever method is not "spec"
            same_group = groups_of[index[first]] == groups_of[index[second]]

            pair = sorted((index[first], index[second]))
            guesses[row] = f"{names[pair[0]]}-{names[pair[1]]}"
            confidence[row] = score * (1.0 if (opposite and same_group) else 0.5)
            polarity[row] = bool(opposite)

    out["stim_site_guess"] = guesses
    out["site_confidence"] = confidence
    out["polarity_check"] = polarity
    return out


def _spec_site(data: Any, index: Any, onset: Any, lag: Any, half: Any, group: Any) -> Any:
    """Spec §5.2 method 5 as reviewed, for one pulse: (pair, confidence, opposite)."""
    n_times = data.shape[1]
    if onset - lag < 0 or onset - half < 0 or onset + half + 1 > n_times:
        return None
    reference = data[index, onset - lag]
    deviation = data[index, onset - half : onset + half + 1] - reference[:, None]
    amplitude = np.nan_to_num(np.nanmax(np.abs(deviation), axis=1), nan=0.0)
    if not amplitude.max() > 0:
        return None
    order = np.argsort(-amplitude, kind="stable")
    first, second = order[0], order[1]
    a_1 = amplitude[first]
    a_3 = amplitude[order[2]] if len(order) > 2 else 0.0
    opposite = bool(_first_sign(deviation[first]) * _first_sign(deviation[second]) < 0)
    same_group = group is None or group[index[first]] == group[index[second]]
    score = (a_1 - a_3) / a_1 * (1.0 if (opposite and same_group) else 0.5)
    return sorted((index[first], index[second])), float(score), opposite


def _contacts(site: Any) -> list[str]:
    return [part for part in str(site).split("-") if part] if site else []


def vote_sites(
    events: Any, *, site_column: str = "stim_site_guess", max_gap: float | None = None
) -> Any:
    """Make the site guesses of consecutive pulses agree, by a vote per block.

    SPES protocols stimulate one pair for several pulses in a row. Pulses are split
    into blocks: a block ends at a gap longer than ``max_gap`` or when two consecutive
    guesses share no contact. In each block, every contact gets one vote per pulse
    whose guess contains it, weighted by that pulse's ``site_confidence``; the two
    contacts with the most votes form the block's site. A single pulse whose partner
    contact is wrong is corrected by its neighbours.

    Parameters
    ----------
    events : pandas.DataFrame
        Event table with ``onset_time``, ``site_column`` and ``site_confidence``.
    site_column : str, default: "stim_site_guess"
        Column of per-pulse guesses ("A-B").
    max_gap : float, optional
        Largest gap inside a block, in seconds. Default: 3 times the median interval
        between pulses.

    Returns
    -------
    events : pandas.DataFrame
        Copy with ``site_column`` replaced by the block vote, the per-pulse guess kept
        in ``stim_site_pulse``, ``site_confidence`` replaced by the fraction of the
        block's pulses whose own guess contains both voted contacts (times the mean
        per-pulse confidence), and ``site_block`` (int64).
    """
    # in time order (stable), then back in the order of ``events`` at the end
    order = np.argsort(np.asarray(events["onset_time"], dtype=float), kind="stable")
    out = events.iloc[order].copy()
    times = np.asarray(out["onset_time"], dtype=float)
    guesses = as_text(out[site_column]).tolist()
    conf = np.asarray(out["site_confidence"], dtype=float)
    if max_gap is None:
        max_gap = 3.0 * float(np.median(np.diff(times))) if len(times) > 1 else np.inf

    block = np.zeros(len(out), dtype=int)
    for row in range(1, len(out)):
        shared = set(_contacts(guesses[row])) & set(_contacts(guesses[row - 1]))
        new = times[row] - times[row - 1] > max_gap or not shared
        block[row] = block[row - 1] + int(new)

    voted, voted_conf = list(guesses), conf.copy()
    for b in np.unique(block):
        rows: Any = np.flatnonzero(block == b)
        votes: dict[Any, Any] = {}
        for row in rows:
            for contact in _contacts(guesses[row]):
                votes[contact] = votes.get(contact, 0.0) + max(conf[row], 1e-6)
        if len(votes) < 2:
            continue
        top = sorted(votes, key=lambda contact: (-votes[contact], contact))[:2]
        # Orientation: the one the per-pulse guesses use most, else alphabetical
        forward, backward = f"{top[0]}-{top[1]}", f"{top[1]}-{top[0]}"
        counts = {name: sum(guesses[row] == name for row in rows) for name in (forward, backward)}
        if counts[forward] == counts[backward]:
            site = "-".join(sorted(top))
        else:
            site = max(counts, key=lambda name: counts[name])
        agree = np.mean([set(top) <= set(_contacts(guesses[row])) for row in rows])
        for row in rows:
            voted[row] = site
            voted_conf[row] = float(agree * np.mean(conf[rows]))

    out["stim_site_pulse"] = guesses
    out[site_column] = voted
    out["site_confidence"] = voted_conf
    out["site_block"] = block.astype("int64")
    return out.iloc[np.argsort(order, kind="stable")]
