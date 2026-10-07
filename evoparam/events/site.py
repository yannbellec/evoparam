"""Identify the stimulated contact pair from the stimulation artifact."""

import re
import warnings

import numpy as np

from evoparam.errors import DataError
from evoparam.utils._optional import require

from ._score import MAD_TO_SD, as_continuous, pick_index

__all__ = ["identify_stim_site", "vote_sites", "contact_group"]

# Reference sample for the artifact size, before the onset (s).
REFERENCE_LAG = 0.002

# Pre-pulse window used to measure each channel's noise level (s, relative to the onset).
NOISE_WINDOW = (-0.100, -0.005)

# Channel types that can carry a stimulating contact (channel table ``type`` column).
NEURAL_TYPES = {"ecog", "seeg", "dbs", "eeg", "lfp", "ieeg", "meg"}

# Names of channels that never are stimulating contacts: markers, triggers, status,
# physiological and auxiliary channels.
NON_NEURAL = re.compile(r"^(mkr|trig|sti|status|event|marker|dc\d|ecg|ekg|emg|eog|resp|"
                        r"spo2|pulse|photic|aux|misc|ref$|gnd)", re.IGNORECASE)


def contact_group(name):
    """Electrode group guessed from a contact name: the name without its trailing number.

    For example ``"PT01"`` -> ``"PT"``, ``"LA'12"`` -> ``"LA'"``, ``"Grid_3"`` -> ``"Grid"``.
    """

    match = re.match(r"^(.*?)[\s_\-]*\d+$", str(name).strip())
    return match.group(1) if match and match.group(1) else str(name)


def identify_stim_site(data, sfreq, events, *, ch_names, ch_info=None, window=0.001, t0=0.0,
                       picks=None, exclude=NON_NEURAL):
    """Guess the stimulated contact pair of each pulse from the artifact.

    For each pulse and channel ``c``, the artifact size is
    ``a_c = max_{|t| <= window} |x_c(t) - x_c(-2 ms)|``, divided by the channel's
    noise level (scaled MAD over -100 to -5 ms before the pulse) so that channels
    with different gains compare fairly. For a bipolar stimulation the two largest
    sizes are expected on the two stimulating contacts, with first deflections of
    opposite signs, on the same electrode.

    ``stim_site_guess`` is ``"A-B"`` (the two top channels, ordered by channel index).
    ``site_confidence = (a_1 - a_3) / a_1`` (the margin of the pair over the third
    channel), halved when the polarity check or the electrode check fails.

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
        Half-width of the window around the onset, in seconds (at least one sample).
    t0 : float, default: 0
        Time of the first sample, in seconds (kept for symmetry with the other functions).
    picks : array-like, optional
        Channels that may be stimulation contacts (indices, names or boolean mask).
    exclude : re.Pattern or None
        Channel names that are never stimulating contacts (default: markers, triggers,
        ECG, EMG, EOG and other auxiliary channels). With a ``ch_info`` that has a
        ``type`` column, only neural types are kept instead.

    Returns
    -------
    events : pandas.DataFrame
        Copy of ``events`` with ``stim_site_guess`` (str, "" when not identifiable),
        ``site_confidence`` (0-1) and ``polarity_check`` (bool) columns.
    """

    require("pandas", extra="data", caller="evoparam.events.identify_stim_site")
    data = as_continuous(data)
    n_channels, n_times = data.shape
    names = list(ch_names)
    if len(names) != n_channels:
        raise DataError("ch_names must have one name per channel.")
    index = pick_index(n_channels, picks, names)

    if ch_info is not None and "type" in ch_info.columns:
        types = dict(zip(ch_info["name"], ch_info["type"].astype(str).str.lower()))
        index = np.array([ind for ind in index if types.get(names[ind], "") in NEURAL_TYPES])
    elif exclude is not None:
        index = np.array([ind for ind in index if not exclude.match(names[ind])])
    if ch_info is not None and "bad" in ch_info.columns:
        bad = set(ch_info.loc[ch_info["bad"].astype(bool), "name"])
        index = np.array([ind for ind in index if names[ind] not in bad])
    if ch_info is not None and "group" in ch_info.columns:
        groups = dict(zip(ch_info["name"], ch_info["group"].astype(str)))
        group = [groups.get(name, contact_group(name)) for name in names]
    else:
        group = [contact_group(name) for name in names]

    out = events.copy()
    onsets = np.asarray(events["onset_sample"], dtype=int)
    n_events = len(onsets)
    guesses = [""] * n_events
    confidence = np.zeros(n_events)
    polarity = np.zeros(n_events, dtype=bool)

    half = max(1, int(round(window * sfreq)))
    lag = max(half + 1, int(round(REFERENCE_LAG * sfreq)))
    noise_lo, noise_hi = (int(round(edge * sfreq)) for edge in NOISE_WINDOW)

    if n_events > 1 and np.median(np.diff(np.sort(onsets))) < 2 * lag + 2 * half:
        warnings.warn("identify_stim_site: pulses are closer than the identification window "
                      "(high-frequency train); the site is not identified.", stacklevel=2)
        index = np.array([], dtype=int)

    if len(index) >= 2:
        for row, onset in enumerate(onsets):
            if onset - lag < 0 or onset + half + 1 > n_times:
                continue
            # Only the samples around this pulse are copied, never the whole recording
            reference = data[index, onset - lag]
            deviation = data[index, onset - half:onset + half + 1] - reference[:, None]
            size = np.nanmax(np.abs(deviation), axis=1)
            lo, hi = max(0, onset + noise_lo), max(0, onset + noise_hi)
            if hi - lo >= 8:
                base = data[index, lo:hi]
                median = np.nanmedian if np.isnan(base).any() else np.median
                noise = MAD_TO_SD * median(
                    np.abs(base - median(base, axis=1, keepdims=True)), axis=1)
                floor = np.median(noise[noise > 0]) if np.any(noise > 0) else 1.0
                size = size / np.maximum(noise, 1e-3 * floor)
            size = np.nan_to_num(size, nan=0.0)
            order = np.argsort(-size, kind="stable")
            first, second = order[0], order[1]
            if not size[first] > 0:
                continue
            third = size[order[2]] if len(order) > 2 else 0.0

            # Sign of the first deflection: first sample beyond half the artifact size
            signs = []
            for ch in (first, second):
                trace = deviation[ch]
                beyond = np.flatnonzero(np.abs(trace) >= 0.5 * np.nanmax(np.abs(trace)))
                signs.append(np.sign(trace[beyond[0]]) if len(beyond) else 0.0)
            opposite = signs[0] * signs[1] < 0
            same_group = group[index[first]] == group[index[second]]

            pair = sorted((index[first], index[second]))
            guesses[row] = f"{names[pair[0]]}-{names[pair[1]]}"
            score = (size[first] - third) / size[first]
            confidence[row] = score * (1.0 if (opposite and same_group) else 0.5)
            polarity[row] = bool(opposite)

    out["stim_site_guess"] = guesses
    out["site_confidence"] = confidence
    out["polarity_check"] = polarity
    return out


def _contacts(site):
    return [part for part in str(site).split("-") if part] if site else []


def vote_sites(events, *, site_column="stim_site_guess", max_gap=None):
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

    require("pandas", extra="data", caller="evoparam.events.vote_sites")
    out = events.sort_values("onset_time", kind="stable").copy()
    times = out["onset_time"].to_numpy(dtype=float)
    guesses = out[site_column].fillna("").astype(str).to_list()
    conf = out["site_confidence"].to_numpy(dtype=float)
    if max_gap is None:
        max_gap = 3.0 * float(np.median(np.diff(times))) if len(times) > 1 else np.inf

    block = np.zeros(len(out), dtype=int)
    for row in range(1, len(out)):
        shared = set(_contacts(guesses[row])) & set(_contacts(guesses[row - 1]))
        new = times[row] - times[row - 1] > max_gap or not shared
        block[row] = block[row - 1] + int(new)

    voted, voted_conf = list(guesses), conf.copy()
    for b in np.unique(block):
        rows = np.flatnonzero(block == b)
        votes = {}
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
            site = max(counts, key=counts.get)
        agree = np.mean([set(top) <= set(_contacts(guesses[row])) for row in rows])
        for row in rows:
            voted[row] = site
            voted_conf[row] = float(agree * np.mean(conf[rows]))

    out["stim_site_pulse"] = guesses
    out[site_column] = voted
    out["site_confidence"] = voted_conf
    out["site_block"] = block.astype("int64")
    return out.loc[events.index]
