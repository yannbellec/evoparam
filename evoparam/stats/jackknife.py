"""Jackknife scoring of latencies over participants (spec §5.11, method 2).

A latency measured on one participant's noisy average is unreliable; measured on the
grand average it is reliable but leaves no variance for a test. The jackknife scores the
``n`` leave-one-out grand averages ``J_i`` instead (Miller, Patterson and Ulrich 1998).
Their spread is ``(n - 1)`` times too small, so a t statistic computed on them is divided
by ``n - 1`` and an F statistic by ``(n - 1)^2`` (Ulrich and Miller 2001; Kiesel et al.
2008). Smulders (2010) showed that ``o_i = n * mean(J) - (n - 1) * J_i`` retrieves
estimates of the participants' own latencies, on which ordinary statistics give the
corrected values directly: the retrieved values have the mean of the ``J_i``, and their
deviations from it are ``-(n - 1)`` times those of the ``J_i``.
"""

import time as _time

import numpy as np

from evoparam.data import Evoked, Provenance
from evoparam.errors import DataError, SettingsError
from evoparam.utils._optional import require

from ._scorers import TIME_SCORES, default_window, resolve_scorer

__all__ = ["jackknife_latency", "jackknife_t", "jackknife_f", "JACKKNIFE_COLUMNS"]

MIN_SUBJECTS = 3

JACKKNIFE_COLUMNS = {"subject": "", "channel": "", "jackknife": "score", "retrieved": "score",
                     "n_subjects": ""}


def jackknife_t(t, n):
    """Corrected t statistic of jackknifed scores: ``t / (n - 1)``.

    Examples
    --------
    >>> jackknife_t(12.0, 13)
    1.0
    """

    if n < 2:
        raise SettingsError("n must be at least 2.")
    return float(t) / (n - 1)


def jackknife_f(f, n):
    """Corrected F statistic of jackknifed scores: ``F / (n - 1)^2``.

    Examples
    --------
    >>> jackknife_f(144.0, 13)
    1.0
    """

    if n < 2:
        raise SettingsError("n must be at least 2.")
    return float(f) / (n - 1) ** 2


def _subject_label(evoked, position):
    meta = evoked.metadata
    if meta is not None and "subject" in getattr(meta, "columns", ()) and len(meta):
        return str(meta["subject"].iloc[0])
    return str(position)


def jackknife_latency(evokeds, *, scorer="fractional_area_latency", retrieve=True, picks=None,
                      window=None, **scorer_kw):
    """Score the leave-one-out grand averages of participants, and retrieve each one's score.

    Parameters
    ----------
    evokeds : sequence of Evoked
        One average per participant, on the same channels and time axis.
    scorer : str or callable
        A classical score (``"fractional_area_latency"``, ``"fractional_peak_latency"``,
        ``"peak_latency"``, ``"peak_amplitude"``, ``"mean_amplitude"``, ``"area"``) with
        ``window`` and the options ``scorer_kw`` of
        :func:`evoparam.models.classical_measures` (``polarity``, ``fraction``, ...), or
        ``scorer(evoked, **scorer_kw) -> float | ndarray`` (one value per channel) called on
        each leave-one-out grand average.
    retrieve : bool
        Add the retrieved individual scores ``o_i = n * mean(J) - (n - 1) * J_i``
        (Smulders 2010).
    picks : sequence of str, optional
        Channels to score (default all).
    window : (float, float), optional
        Scoring window (s) of a named score; default the post-stimulus epoch.

    Returns
    -------
    DataFrame
        One row per participant and channel: ``subject`` (the ``subject`` metadata of each
        Evoked, else its position), ``channel``, ``jackknife`` (``J_i``), ``retrieved``
        (``o_i``; NaN without ``retrieve``), ``n_subjects``. ``attrs["summary"]`` (columns as lists) holds,
        per channel, the score of the grand average, the mean of the ``J_i`` and the
        jackknife standard error ``sqrt((n - 1) / n * sum((J_i - mean J)^2))``; ``attrs``
        also holds ``units``, ``settings`` and ``provenance``. Run ordinary t tests or
        ANOVAs on ``retrieved``, or apply :func:`jackknife_t` / :func:`jackknife_f` to
        those run on ``jackknife``.

    Raises
    ------
    DataError
        Fewer than 3 participants, or averages on different channels or time axes.

    Examples
    --------
    >>> from evoparam.sim import simulate_erp
    >>> evokeds = [simulate_erp(20, components=["N1"], noise="white", snr=4., seed=s)[0]
    ...            .average() for s in range(6)]
    >>> table = jackknife_latency(evokeds, window=(0.05, 0.2))
    >>> table.shape
    (6, 5)
    """

    start = _time.perf_counter()
    pd = require("pandas", extra="data", caller="evoparam.stats.jackknife_latency")
    evokeds = list(evokeds)
    n = len(evokeds)
    if n < MIN_SUBJECTS:
        raise DataError(f"jackknife_latency needs at least {MIN_SUBJECTS} participants, "
                        f"got {n}.")
    for ev in evokeds:
        if not isinstance(ev, Evoked):
            raise DataError("jackknife_latency expects one Evoked per participant "
                            "(use TrialSet.average()).")
    first = evokeds[0]
    for ev in evokeds[1:]:
        if list(ev.ch_names) != list(first.ch_names) or ev.times.shape != first.times.shape \
                or not np.allclose(ev.times, first.times, rtol=0, atol=1e-9):
            raise DataError("jackknife_latency: every Evoked must have the same channels "
                            "and time axis.")
    names = list(first.ch_names)
    index = list(range(len(names))) if picks is None else \
        [names.index(p) if isinstance(p, str) else int(p) for p in picks]
    stack = np.stack([np.where(ev.masks, np.nan, ev.data) if ev.masks is not None else ev.data
                      for ev in evokeds])[:, index]
    finite = np.isfinite(stack)
    total, count = np.where(finite, stack, 0.0).sum(axis=0), finite.sum(axis=0)
    loo_count = count[None] - finite
    loo = np.divide(total[None] - np.where(finite, stack, 0.0), loo_count,
                    out=np.full(stack.shape, np.nan), where=loo_count > 0)
    grand = np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)

    if isinstance(scorer, str):
        window = default_window(first.times) if window is None else window
        func, name, _, window = resolve_scorer(scorer, first.times, window, **scorer_kw)
        jack = func(loo)                                       # (n, n_channels)
        full = func(grand)
        unit = "s" if name in TIME_SCORES else ("data·s" if name == "area" else first.units)
    elif callable(scorer):
        name, unit = getattr(scorer, "__name__", "custom"), "score"
        picked = [names[i] for i in index]

        def score(data):
            ev = Evoked(data=data, times=first.times, sfreq=first.sfreq, ch_names=picked,
                        units=first.units, n_trials=n - 1)
            return np.broadcast_to(np.asarray(scorer(ev, **scorer_kw), float),
                                   (len(picked),))

        jack = np.stack([score(loo[i]) for i in range(n)])
        full = score(grand)
    else:
        raise SettingsError("scorer must be a score name or a callable.")

    mean_j = np.nanmean(jack, axis=0)
    retrieved = n * mean_j[None] - (n - 1) * jack if retrieve else np.full(jack.shape, np.nan)
    se = np.sqrt((n - 1) / n * np.nansum((jack - mean_j[None]) ** 2, axis=0))
    subjects = [_subject_label(ev, i) for i, ev in enumerate(evokeds)]
    channels = [names[i] for i in index]
    table = pd.DataFrame({"subject": np.repeat(subjects, len(channels)),
                          "channel": channels * n, "jackknife": jack.ravel(),
                          "retrieved": retrieved.ravel(), "n_subjects": n})
    summary = pd.DataFrame({"channel": channels, "grand_average": full, "mean_jackknife": mean_j,
                            "jackknife_se": se})
    settings = {"scorer": name, "retrieve": bool(retrieve), "n_subjects": n,
                "window": None if window is None else list(window),
                "channels": channels, **{k: v for k, v in scorer_kw.items()
                                         if isinstance(v, (int, float, str, bool))}}
    table.attrs["summary"] = summary.to_dict("list")
    table.attrs["units"] = {k: (unit if v == "score" else v) for k, v in JACKKNIFE_COLUMNS.items()}
    table.attrs["settings"] = settings
    table.attrs["provenance"] = Provenance().with_step(
        "evoparam.stats.jackknife_latency", settings,
        duration_s=_time.perf_counter() - start).to_dict()
    return table
