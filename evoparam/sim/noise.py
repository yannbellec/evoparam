"""Noise models: white, AR(1), power-law ("pink"), real baselines and line noise.

All synthetic noise is generated here with numpy and scipy (no external
package). Real noise is cut from the baselines of a :class:`TrialSet` read from a
public dataset; every segment is recorded (dataset, version, subject, run,
sample offsets) and datasets whose licence is not CC0, CC BY or CC BY-SA are
refused (spec §5.12, REQ-256).
"""

import re
from fractions import Fraction

import numpy as np
from scipy import fft as sp_fft
from scipy import signal as sp_signal

from evoparam.data import TrialSet
from evoparam.data.containers import UNITS
from evoparam.errors import SettingsError

__all__ = ["make_noise", "NOISE_KINDS", "ALLOWED_LICENSES", "normalize_license",
           "check_license", "resolve_rng"]

NOISE_KINDS = ("white", "ar1", "pink", "real", "line")
BROADBAND = ("white", "ar1", "pink", "real")

# Licences under which real baselines may be injected (spec REQ-256).
ALLOWED_LICENSES = ("CC0", "CC-BY", "CC-BY-SA")

# Attempts at finding a segment without masked or missing samples.
_MAX_TRIES = 100


def resolve_rng(seed):
    """A :class:`numpy.random.Generator` from an int, a Generator or None."""

    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None or isinstance(seed, (int, np.integer)):
        return np.random.default_rng(seed)
    raise SettingsError(f"seed must be an int, a numpy Generator or None, got {seed!r}.")


def parse_kind(kind):
    """Split ``"pink+line"`` into (broadband kind or None, with_line)."""

    if kind is None or kind == "none":
        return None, False
    parts = [part.strip() for part in str(kind).split("+")]
    unknown = [part for part in parts if part not in NOISE_KINDS]
    if unknown or len(set(parts)) != len(parts):
        raise SettingsError(f"noise kind {kind!r}: use one of {NOISE_KINDS}, optionally "
                            "combined with '+line' (for example 'pink+line').")
    broadband = [part for part in parts if part in BROADBAND]
    if len(broadband) > 1:
        raise SettingsError(f"noise kind {kind!r}: at most one of {BROADBAND}.")
    return (broadband[0] if broadband else None), "line" in parts


###################################################################################################
## Synthetic noise

def _white(rng, shape):
    return rng.standard_normal(shape)


def _ar1(rng, shape, rho):
    """Stationary AR(1) with unit variance: ``x_t = rho x_(t-1) + sqrt(1 - rho²) e_t``."""

    if not -1.0 < rho < 1.0:
        raise SettingsError(f"ar1 noise: rho must be in (-1, 1), got {rho}.")
    innovations = rng.standard_normal(shape)
    first = rng.standard_normal(shape[:-1])
    zi = (rho * first)[..., None]
    out, _ = sp_signal.lfilter([np.sqrt(1.0 - rho ** 2)], [1.0, -rho], innovations, axis=-1,
                               zi=zi)
    return out


def _powerlaw(rng, shape, sfreq, chi):
    """Noise whose power falls as ``1/f^chi``, with unit variance.

    White noise is filtered in the frequency domain by ``|H(f)| = f^(-chi/2)``
    (zero at DC) over a segment twice as long as requested, of which the first
    half is kept, so that the segment is not periodic.
    """

    if not (np.isfinite(chi) and chi >= 0):
        raise SettingsError(f"pink noise: chi must be a number >= 0, got {chi}.")
    n_times = shape[-1]
    n_fft = sp_fft.next_fast_len(2 * n_times, real=True)
    freqs = sp_fft.rfftfreq(n_fft, 1.0 / sfreq)
    gain = np.zeros(freqs.size)
    gain[1:] = freqs[1:] ** (-chi / 2.0)
    # Variance of irfft(gain · rfft(white)) for unit white noise: sum of |gain|² over the
    # full two-sided spectrum divided by n_fft.
    weights = np.full(freqs.size, 2.0)
    weights[0] = 1.0
    if n_fft % 2 == 0:
        weights[-1] = 1.0
    variance = np.sum(weights * gain ** 2) / n_fft
    white = rng.standard_normal(shape[:-1] + (n_fft,))
    out = sp_fft.irfft(sp_fft.rfft(white, axis=-1) * gain, n=n_fft, axis=-1)[..., :n_times]
    return out / np.sqrt(variance)


def _line(rng, shape, sfreq, line_freq, n_harmonics, amplitude):
    """Line noise: ``sum_h (amplitude / h) sin(2π h f t + φ_h)``, random phases."""

    if not (line_freq > 0 and n_harmonics >= 1):
        raise SettingsError("line noise: line_freq must be positive and line_harmonics >= 1.")
    t = np.arange(shape[-1]) / sfreq
    out = np.zeros(shape)
    phases = rng.uniform(0.0, 2.0 * np.pi, shape[:-1] + (int(n_harmonics),))
    for h in range(1, int(n_harmonics) + 1):
        if h * line_freq >= sfreq / 2:
            break
        out += (amplitude / h) * np.sin(2 * np.pi * h * line_freq * t + phases[..., h - 1, None])
    return out


###################################################################################################
## Licences and real noise

def normalize_license(text):
    """Canonical form of a Creative Commons licence name, or None.

    ``"CC0 1.0"`` → ``"CC0"``, ``"CC BY-SA 4.0"`` → ``"CC-BY-SA"``, ``"cc-by-4.0"`` →
    ``"CC-BY"``. Version numbers are dropped.
    """

    if text is None:
        return None
    words = re.sub(r"[_\-\s]+", " ", str(text).upper())
    words = re.sub(r"\b\d+(\.\d+)*\b", " ", words).split()
    if not words:
        return None
    return "-".join(words)


def check_license(dataset):
    """Registry entry of ``dataset`` if its licence allows injection; else SettingsError.

    Parameters
    ----------
    dataset : str
        Identifier in the dataset registry (:func:`evoparam.io.dataset_info`).

    Returns
    -------
    dict
        ``{"dataset", "version", "doi", "license"}``.

    Raises
    ------
    SettingsError
        The dataset is not in the registry, or its licence is unknown or not one
        of :data:`ALLOWED_LICENSES`.
    """

    from evoparam.errors import ReaderError
    from evoparam.io.registry import dataset_info
    try:
        entry = dataset_info(dataset)
    except ReaderError as exc:
        raise SettingsError(f"real noise: dataset {dataset!r} is not in the registry, so its "
                            "licence cannot be checked.") from exc
    licence = normalize_license(entry.get("license"))
    if licence not in ALLOWED_LICENSES:
        raise SettingsError(f"real noise: the licence of {entry['id']} in the registry is "
                            f"{entry.get('license')!r}; only {ALLOWED_LICENSES} may be "
                            "injected.")
    return {"dataset": entry["id"], "version": entry.get("version"), "doi": entry.get("doi"),
            "license": entry.get("license")}


def _source_value(source, trial, column, bids_key):
    """Subject or run of one source trial: metadata column, else the BIDS entities."""

    meta = source.metadata
    if meta is not None and column in meta.columns:
        value = meta[column].iloc[trial]
        if value is not None and str(value) not in ("", "n/a", "nan"):
            return str(value)
    bids = source.provenance.source.get("bids") or {}
    value = bids.get(bids_key) if isinstance(bids, dict) else None
    return None if value in (None, "") else str(value)


def _real(rng, shape, sfreq, source, window, dataset, units):
    """Segments of the baselines of ``source``; returns (noise, info)."""

    if not isinstance(source, TrialSet):
        raise SettingsError("real noise needs source=<TrialSet of real recordings>.")
    dataset = dataset or source.provenance.source.get("dataset")
    if not dataset:
        raise SettingsError("real noise: the source does not name its dataset "
                            "(provenance.source['dataset']); pass dataset=<registry id>.")
    record = check_license(dataset)
    if source.units != units:
        if UNITS.get(source.units) is None or UNITS.get(units) is None:
            raise SettingsError(f"real noise: cannot convert {source.units} to {units}.")
    factor = 1.0 if source.units == units else UNITS[source.units] / UNITS[units]

    tmin, tmax = (source.times[0], 0.0) if window is None else window
    tmin = source.times[0] if tmin is None else float(tmin)
    tmax = source.times[-1] + 0.5 / source.sfreq if tmax is None else float(tmax)
    keep = (source.times >= tmin - 1e-9) & (source.times < tmax - 1e-9)
    first = int(np.flatnonzero(keep)[0]) if keep.any() else 0
    values = source.data[:, :, keep] * factor
    masks = None if source.masks is None else source.masks[:, :, keep]
    resampling = None
    if abs(source.sfreq - sfreq) > 1e-9 * sfreq:
        ratio = Fraction(sfreq / source.sfreq).limit_denominator(1000)
        values = sp_signal.resample_poly(values, ratio.numerator, ratio.denominator, axis=-1)
        if masks is not None:
            index = np.floor(np.arange(values.shape[-1]) * source.sfreq / sfreq).astype(int)
            masks = masks[..., np.minimum(index, masks.shape[-1] - 1)]
        resampling = {"method": "scipy.signal.resample_poly", "from_sfreq": source.sfreq,
                      "to_sfreq": float(sfreq), "up": ratio.numerator,
                      "down": ratio.denominator}
    n_times = shape[-1]
    available = values.shape[-1]
    if available < n_times:
        raise SettingsError(f"real noise: the source window {tmin, tmax} s holds {available} "
                            f"samples at {sfreq} Hz; {n_times} are needed per segment.")
    size = shape[:-1]
    n_channels = size[-1] if len(size) >= 2 else 1
    n_segments = int(np.prod(size[:-1])) if len(size) >= 2 else int(np.prod(size))
    channel_map = np.arange(n_channels) % source.n_channels
    out = np.empty((n_segments, n_channels, n_times))
    segments = []
    for index in range(n_segments):
        for _ in range(_MAX_TRIES):
            trial = int(rng.integers(source.n_trials))
            start = int(rng.integers(available - n_times + 1))
            piece = values[trial][channel_map, start:start + n_times]
            bad = np.isnan(piece).any()
            if masks is not None:
                bad = bad or masks[trial][channel_map, start:start + n_times].any()
            if not bad:
                break
        else:
            raise SettingsError("real noise: no segment without masked or missing samples "
                                f"was found in {_MAX_TRIES} attempts.")
        out[index] = piece - piece.mean(axis=-1, keepdims=True)
        scale = source.sfreq / sfreq
        offset = first + int(round(start * scale))
        stop = first + int(round((start + n_times) * scale))
        absolute = None
        if source.event_times is not None:
            onset_sample = source.event_times[trial] * source.sfreq
            epoch_start = int(round(onset_sample + source.times[0] * source.sfreq))
            absolute = [epoch_start + offset, epoch_start + stop]
        segments.append({
            "segment": index, "dataset": record["dataset"], "version": record["version"],
            "doi": record["doi"], "license": record["license"],
            "subject": _source_value(source, trial, "subject", "subject"),
            "session": _source_value(source, trial, "session", "session"),
            "run": _source_value(source, trial, "run", "run"),
            "source_trial": trial,
            "source_channels": [source.ch_names[c] for c in channel_map],
            "epoch_samples": [offset, stop],
            "recording_samples": absolute,
            "source_sfreq": source.sfreq,
        })
    info = {"dataset": record, "window": [tmin, tmax], "segments": segments,
            "resampling": resampling, "source_units": source.units,
            "source_path": source.provenance.source.get("path")}
    return out.reshape(shape), info


###################################################################################################
## Public function

def make_noise(n_times, sfreq, kind="pink", *, size=(), sd=None, chi=1.0, rho=0.5,
               line_freq=50.0, line_harmonics=3, line_amplitude=None, source=None,
               window=None, dataset=None, units="uV", seed=None, return_info=False):
    """Generate noise of one of the models of spec §5.12.

    Parameters
    ----------
    n_times : int
        Samples per realisation.
    sfreq : float
        Sampling frequency (Hz).
    kind : str
        ``"white"``, ``"ar1"`` (lag-1 autocorrelation ``rho``), ``"pink"`` (power
        spectrum ``1/f^chi``), ``"real"`` (segments of real baselines from
        ``source``), ``"line"`` (``line_freq`` and its harmonics), or one broadband
        kind plus ``"+line"``, for example ``"pink+line"``.
    size : tuple of int
        Leading shape; the output has shape ``size + (n_times,)``. For real noise
        the last entry of ``size`` is the number of channels (when ``size`` has two
        or more entries): the channels of one realisation come from the same source
        trial and time, which keeps the correlation between channels.
    sd : float, optional
        Standard deviation of the broadband part (data units). Default 1 for the
        synthetic kinds; real noise keeps its own scale unless ``sd`` is given, in
        which case it is rescaled to that SD over all segments.
    chi : float
        Spectral exponent of pink noise (``chi >= 0``; 1 is pink, 2 is brown).
    rho : float
        Lag-1 autocorrelation of AR(1) noise, in (-1, 1).
    line_freq : float
        Line frequency (Hz), usually 50 or 60.
    line_harmonics : int
        Number of line components (fundamental included); harmonic ``h`` has
        amplitude ``line_amplitude / h``. Harmonics above Nyquist are left out.
    line_amplitude : float, optional
        Amplitude of the fundamental (data units). Default: ``sd`` (or 1).
    source : TrialSet, optional
        Real recordings for ``kind="real"``. Its ``provenance.source["dataset"]``
        (or ``dataset``) must be a registry dataset under CC0, CC BY or CC BY-SA.
    window : (float, float), optional
        Part of each source epoch from which segments are cut (s). Default: the
        pre-stimulus part, from the start of the epoch to 0.
    dataset : str, optional
        Registry identifier of the source, when its provenance does not name it.
    units : str
        Units of the output; real noise is converted from the source units.
    seed : int, numpy.random.Generator or None
        Random seed.
    return_info : bool
        Also return a dictionary with the model, its parameters and, for real
        noise, the record of every segment.

    Returns
    -------
    noise : ndarray, shape ``size + (n_times,)``
    info : dict
        Only with ``return_info=True``.

    Raises
    ------
    SettingsError
        Unknown kind or invalid parameter; real noise from an unlisted dataset, a
        dataset whose licence is not allowed, or a source window too short.

    Notes
    -----
    Pink noise is white noise filtered in the frequency domain by
    ``|H(f)| = f^(-chi/2)`` (``scipy.fft``) and scaled to the requested SD with the
    exact variance of the filter. Real noise with a sampling rate different from
    ``sfreq`` is resampled with ``scipy.signal.resample_poly``, recorded in the
    returned information.

    Examples
    --------
    >>> noise = make_noise(1000, 500., "ar1", rho=0.9, size=(2, 3), seed=0)
    >>> noise.shape
    (2, 3, 1000)
    """

    n_times = int(n_times)
    if n_times < 2:
        raise SettingsError(f"n_times must be at least 2, got {n_times}.")
    if not (np.isfinite(sfreq) and sfreq > 0):
        raise SettingsError(f"sfreq must be a positive number, got {sfreq}.")
    size = (size,) if isinstance(size, (int, np.integer)) else tuple(int(s) for s in size)
    shape = size + (n_times,)
    broadband, with_line = parse_kind(kind)
    if broadband is None and not with_line:
        raise SettingsError("make_noise: kind must name a noise model.")
    rng = resolve_rng(seed)
    info = {"kind": str(kind), "sfreq": float(sfreq)}
    out = np.zeros(shape)
    level = 1.0 if sd is None else float(sd)
    if sd is not None and not (np.isfinite(level) and level >= 0):
        raise SettingsError(f"sd must be a number >= 0, got {sd}.")
    if broadband == "white":
        out = level * _white(rng, shape)
        info["sd"] = level
    elif broadband == "ar1":
        out = level * _ar1(rng, shape, float(rho))
        info.update(sd=level, rho=float(rho))
    elif broadband == "pink":
        out = level * _powerlaw(rng, shape, float(sfreq), float(chi))
        info.update(sd=level, chi=float(chi))
    elif broadband == "real":
        out, real_info = _real(rng, shape, float(sfreq), source, window, dataset, units)
        if sd is not None:
            current = float(np.std(out))
            out = out * (level / current) if current > 0 else out
            real_info["rescaled_to_sd"] = level
        info.update(real_info)
    if with_line:
        if line_amplitude is not None:
            amplitude = float(line_amplitude)
        elif broadband == "real" and sd is None:
            amplitude = float(np.std(out))
        else:
            amplitude = level
        out = out + _line(rng, shape, float(sfreq), float(line_freq), int(line_harmonics),
                          amplitude)
        info["line"] = {"freq": float(line_freq), "harmonics": int(line_harmonics),
                        "amplitude": amplitude}
    return (out, info) if return_info else out
