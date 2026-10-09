"""StreamFitter: samples in, one estimate per pulse, channel and component out (spec §10.6).

This module buffers samples, finds the pulses and hands each complete window to the
estimators of the analysis modules (REQ-349: it computes no number of its own):
:class:`evoparam.trials.FixedShapeModel` (shift and scale of the frozen template),
:class:`evoparam.existence.OnlineExistence` (per-pulse score and running test) and
:class:`evoparam.events.OnlinePulseDetector`.
"""

import dataclasses
import heapq
import statistics
import time

import numpy as np

from evoparam.data import TrialSet
from evoparam.errors import DataError, SettingsError, warn as _warn
from evoparam.events import OnlinePulseDetector
from evoparam.existence import OnlineExistence
from evoparam.trials import FixedShapeModel

__all__ = ["StreamFitter", "PulseEstimate", "replay", "INTENDED_USE", "STATUSES"]

INTENDED_USE = "research only; not a medical device"
STATUSES = ("ok", "artifact", "late", "saturated")
BACKENDS = ("numpy",)
# Compute time per pulse above which its estimates are "late" (REQ-324: 20 ms).
BUDGET_MS = 20.0
# Sign flips of the running existence test (the default of test_response).
N_PERMUTATIONS = 1000


@dataclasses.dataclass(frozen=True, slots=True)
class PulseEstimate:
    """Estimate of one component on one channel for one pulse.

    ``amplitude`` is the template amplitude times ``scale`` (data units), ``latency`` the
    template latency plus ``shift`` (s), ``existence_z`` the projection of the window on the
    template against the baseline projections of the stream (SD units), ``present`` is
    ``existence_z > existence_threshold_sd``, ``r2`` the fit over the samples around the
    components, ``compute_ms`` the compute time of the pulse; ``status`` is ``"ok"``,
    ``"artifact"`` (non-finite samples in the window, left out of the fit), ``"saturated"``
    (a window sample at or beyond the saturation level) or ``"late"`` (over the time budget).
    """

    pulse_time: float
    channel: str
    component: str
    amplitude: float
    latency: float
    scale: float
    shift: float
    existence_z: float
    present: bool
    r2: float
    compute_ms: float
    status: str


def _quantile(values, q):
    """Quantile of the compute times (linear interpolation between order statistics)."""

    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1]


class StreamFitter:
    """Online estimation of evoked parameters, pulse by pulse.

    Parameters
    ----------
    template : ResponseFit or GroupFit
        Channel fits of an offline analysis; their components are frozen.
    sfreq : float
        Sampling frequency of the stream (Hz).
    ch_names : list of str
        Channels of the stream, in the order of the columns of ``push``.
    window : (float, float), optional
        Response window relative to the pulse (s); default the window of the fits.
    artifact_window : (float, float), optional
        Interpolated with the core ``"interpolate"`` method (spec §5.3) before the fit and
        the projections.
    detector : "threshold" or callable
        ``"threshold"``: :class:`evoparam.events.OnlinePulseDetector` (its defaults) on
        ``detector_channel``; a callable ``f(samples, first_sample) -> onset samples``
        receives every chunk ([n_samples, n_detector_channels]) and the stream index of its
        first sample (``lambda samples, first: []`` leaves the pulses to :meth:`push_pulse`).
        Pulses registered by :meth:`push_pulse` are estimated whatever the detector.
    detector_channel : str, optional
        Channel the detector looks at (default all channels).
    max_shift : float
        Largest shift of the template (s).
    mode : {"shift_scale", "scale", "common"}
        Free parameters of a template with several components
        (:class:`evoparam.trials.FixedShapeModel`): a shift and a scale per component
        (default), one shift and a scale per component, or the single shift and scale of
        spec §10.6 (``"common"``). The same model for a single component.
    existence_threshold_sd : float
        ``present`` is ``existence_z > existence_threshold_sd``.
    buffer_seconds : float
        Length of the ring buffer.
    backend : {"numpy"}
    saturation : float, optional
        Absolute level at or beyond which a window sample is saturated (status
        ``"saturated"``; the specification names the status but not the level).
    existence_alpha, existence_max_pulses, existence_looks
        Settings of the running existence test (:class:`evoparam.existence.OnlineExistence`,
        1 000 sign flips): family-wise error over all looks, planned number of pulses (the
        information fraction of the alpha-spending rule), ``"pulse"`` (a look after every
        pulse) or None (looks only through :meth:`existence`).
    random_state : int, optional
        Seed of the sign flips.

    Estimates whose compute time exceeds the 20 ms budget of REQ-324 have the status
    ``"late"``.

    Notes
    -----
    ``push`` takes samples as ``[n_samples, n_channels]`` (the layout of LSL chunks). The
    estimates of a pulse are returned by the ``push`` call that delivers the last sample of
    its window; they do not depend on the chunk sizes. For research use only (not a
    medical device).
    """

    def __init__(self, template, *, sfreq, ch_names, window=None, artifact_window=None,
                 detector="threshold", detector_channel=None, max_shift=0.005,
                 existence_threshold_sd=3.0, buffer_seconds=5.0, backend="numpy",
                 saturation=None, existence_alpha=0.05, existence_max_pulses=100,
                 existence_looks="pulse", random_state=None, mode="shift_scale"):
        if backend not in BACKENDS:
            raise SettingsError(f"backend must be one of {BACKENDS}, not {backend!r}.")
        self.sfreq = float(sfreq)
        self.ch_names = [str(name) for name in ch_names]
        self.model = FixedShapeModel(template, sfreq=sfreq, ch_names=self.ch_names,
                                     window=window, artifact_window=artifact_window,
                                     max_shift=max_shift, mode=mode)
        self.settings = {"window": list(self.model.window), "max_shift": float(max_shift),
                         "mode": mode,
                         "artifact_window": None if artifact_window is None
                         else [float(v) for v in artifact_window],
                         "baseline_offsets": list(self.model.baseline_offsets),
                         "existence_threshold_sd": float(existence_threshold_sd),
                         "existence_alpha": float(existence_alpha),
                         "existence_max_pulses": int(existence_max_pulses),
                         "existence_looks": existence_looks, "n_permutations": N_PERMUTATIONS,
                         "budget_ms": BUDGET_MS, "saturation": saturation,
                         "buffer_seconds": float(buffer_seconds), "backend": backend,
                         "random_state": random_state, "intended_use": INTENDED_USE}
        self.saturation = None if saturation is None else float(saturation)
        self.existence_stats = OnlineExistence(
            len(self.model.ch_names), threshold_sd=existence_threshold_sd,
            alpha=existence_alpha, max_n=existence_max_pulses, n_permutations=N_PERMUTATIONS,
            looks=existence_looks, random_state=random_state)

        i0, i1 = self.model.segment_offsets
        b0, _ = self.model.baseline_offsets
        self._first = min(i0, b0)
        self._last = i1
        self._span = self._last - self._first + 1
        self._n_buffer = int(round(buffer_seconds * self.sfreq))
        if self._n_buffer < 2 * self._span:
            raise SettingsError(f"buffer_seconds must hold at least twice the window and "
                                f"baseline ({2 * self._span / self.sfreq:.3f} s).")
        self._window_rows = np.arange(i0, i1 + 1)
        self._baseline_rows = np.arange(b0, b0 + len(self.model.tw))
        self._picks = self.model.picks
        self._all_picked = list(self._picks) == list(range(len(self.ch_names)))

        if detector_channel is None:
            det_names = self.ch_names
        elif detector_channel in self.ch_names:
            det_names = [detector_channel]
        else:
            raise SettingsError(f"detector_channel: unknown channel {detector_channel!r}.")
        self._detector_picks = np.array([self.ch_names.index(name) for name in det_names])
        if detector == "threshold":
            self.detector = OnlinePulseDetector(self.sfreq, len(det_names))
        elif callable(detector):
            self.detector = detector
        else:
            raise SettingsError("detector must be 'threshold' or a callable.")
        self.reset()

    ###############################################################################################
    ## State

    def reset(self):
        """Forget the stream: buffer, pending pulses, detector, existence test and timings."""

        self._buffer = np.zeros((len(self.ch_names), self._n_buffer))
        self._stamps = np.zeros(self._n_buffer)
        self._has_stamps = False
        self.n_samples = 0
        self._pending = []
        self._compute_ms = []
        self.n_pulses = 0
        if isinstance(self.detector, OnlinePulseDetector):
            self.detector.reset()
        self.existence_stats.reset()

    @property
    def latency_stats(self):
        """Compute time per pulse (ms): ``p50``, ``p95``, ``p99`` and ``max`` (``n_pulses``
        pulses so far)."""

        values = sorted(self._compute_ms)
        if not values:
            return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan"),
                    "max": float("nan")}
        return {"p50": _quantile(values, 50), "p95": _quantile(values, 95),
                "p99": _quantile(values, 99), "max": values[-1]}

    def existence(self, final=False):
        """Running existence test, per channel (a look of the sequential test).

        ``final=True`` declares the end of the stream and spends what remains of the error
        rate. Returns a dict with ``channels`` and arrays ``significant``, ``statistic``,
        ``p_value`` (fixed-sample, for information), ``alpha_spent`` and ``n``.
        """

        out = dict(self.existence_stats.look(final=final))
        out["channels"] = list(self.model.ch_names)
        return out

    ###############################################################################################
    ## Input

    def _time_of(self, sample):
        """Stream-clock time of a sample (its timestamp when the stream has them)."""

        if not self._has_stamps:
            return sample / self.sfreq
        newest = self.n_samples - 1
        oldest = max(0, self.n_samples - self._n_buffer)
        if oldest <= sample <= newest:
            return float(self._stamps[sample % self._n_buffer])
        anchor = min(max(sample, oldest), newest)
        return float(self._stamps[anchor % self._n_buffer]) + (sample - anchor) / self.sfreq

    def push_pulse(self, onset_time):
        """Register a pulse at ``onset_time`` (s, stream clock) from an external trigger."""

        if self._has_stamps and self.n_samples:
            oldest = max(0, self.n_samples - self._n_buffer)
            stamps = self._stamps[np.arange(oldest, self.n_samples) % self._n_buffer]
            if onset_time <= stamps[-1]:
                sample = oldest + int(np.searchsorted(stamps, onset_time - 0.5 / self.sfreq))
            else:
                sample = self.n_samples - 1 + int(round((onset_time - stamps[-1]) * self.sfreq))
        else:
            sample = int(round(onset_time * self.sfreq))
        heapq.heappush(self._pending, (sample, float(onset_time)))

    def push(self, samples, timestamps=None):
        """Add samples ``[n_samples, n_channels]`` (one sample may be 1-d).

        Returns the :class:`PulseEstimate` of every pulse whose window is complete.
        """

        samples = np.asarray(samples, float)
        if samples.ndim == 1:
            samples = samples[None]
        if samples.ndim != 2 or samples.shape[1] != len(self.ch_names):
            raise DataError(f"samples must be [n_samples, {len(self.ch_names)}], got "
                            f"{list(samples.shape)}.")
        if timestamps is not None:
            timestamps = np.asarray(timestamps, float).ravel()
            if len(timestamps) != len(samples):
                raise DataError("timestamps must have one value per sample.")
            self._has_stamps = True
        out = []
        step = self._n_buffer - self._span
        for start in range(0, len(samples), step):
            chunk = samples[start:start + step]
            stamps = None if timestamps is None else timestamps[start:start + step]
            out.extend(self._push_chunk(chunk, stamps))
        return out

    def _push_chunk(self, chunk, stamps):
        first = self.n_samples
        rows = np.arange(first, first + len(chunk)) % self._n_buffer
        self._buffer[:, rows] = chunk.T
        if stamps is not None:
            self._stamps[rows] = stamps
        self.n_samples += len(chunk)
        onsets = self.detector.update(chunk[:, self._detector_picks]) \
            if isinstance(self.detector, OnlinePulseDetector) \
            else self.detector(chunk[:, self._detector_picks], first)
        for onset in onsets:
            heapq.heappush(self._pending, (int(onset), self._time_of(int(onset))))
        out = []
        while self._pending and self._pending[0][0] + self._last < self.n_samples:
            onset, pulse_time = heapq.heappop(self._pending)
            out.extend(self._estimate(onset, pulse_time))
        return out

    ###############################################################################################
    ## Estimation

    def _segment(self, first, length):
        """Samples ``first`` ... ``first + length - 1`` of the modelled channels."""

        start = first % self._n_buffer
        stop = start + length
        if stop <= self._n_buffer:
            return self._buffer[:, start:stop] if self._all_picked else \
                self._buffer[self._picks, start:stop]
        return self._buffer[self._picks[:, None], np.arange(start, stop) % self._n_buffer]

    def _estimate(self, onset, pulse_time):
        start = time.perf_counter()
        oldest = max(0, self.n_samples - self._n_buffer)
        if onset + self._window_rows[0] < oldest:
            _warn(f"Pulse at sample {onset}: its window is no longer (or not yet) in the "
                  "buffer; skipped.", UserWarning, stacklevel=3)
            return []
        window = self._segment(onset + self._window_rows[0], len(self._window_rows))
        baseline = None
        if onset + self._baseline_rows[0] >= oldest:
            baseline = self._segment(onset + self._baseline_rows[0], len(self._baseline_rows))
        res = self.model.fit(window, baseline)
        z, present = self.existence_stats.update(res["projection"], res["baseline_projection"])
        saturated = None if self.saturation is None else \
            (np.abs(window) >= self.saturation).any(axis=1)
        compute_ms = (time.perf_counter() - start) * 1e3
        self._compute_ms.append(compute_ms)
        self.n_pulses += 1
        late = compute_ms > BUDGET_MS
        finite = res["finite"].tolist()
        sat = saturated.tolist() if saturated is not None else [False] * len(finite)
        amp, lat = res["amplitude"].tolist(), res["latency"].tolist()
        scale, shift, r2 = res["scale"].tolist(), res["shift"].tolist(), res["r2"].tolist()
        z, present = z.tolist(), present.tolist()
        out = []
        for c, name in enumerate(self.model.ch_names):
            status = "saturated" if sat[c] else ("artifact" if not finite[c] else
                                                 ("late" if late else "ok"))
            for k, label in enumerate(self.model.labels[c]):
                out.append(PulseEstimate(pulse_time, name, label, amp[c][k], lat[c][k],
                                         scale[c][k], shift[c][k], z[c], bool(present[c]),
                                         r2[c], compute_ms, status))
        return out


def replay(trialset_or_raw, fitter, chunk_size=32):
    """Push recorded data through ``fitter`` in chunks, as a stream would deliver it.

    Parameters
    ----------
    trialset_or_raw : TrialSet or array [n_channels, n_times]
        A TrialSet is replayed epoch after epoch (epochs joined end to end), each epoch's
        onset (t = 0) pushed as an external trigger; use a fitter whose detector finds
        nothing (``detector=lambda samples, first: []``), since the joins between epochs are
        steps a threshold detector may take for pulses.
        An array is a continuous recording at the fitter's sampling frequency, in its
        channel order; its pulses are found by the fitter's detector (or registered
        beforehand with :meth:`StreamFitter.push_pulse`).
    fitter : StreamFitter
    chunk_size : int
        Samples per ``push``.

    Returns
    -------
    list of PulseEstimate
    """

    if int(chunk_size) < 1:
        raise SettingsError("chunk_size must be at least 1.")
    start = fitter.n_samples
    if isinstance(trialset_or_raw, TrialSet):
        ts = trialset_or_raw
        if abs(ts.sfreq - fitter.sfreq) > 1e-9 * fitter.sfreq:
            raise DataError("The TrialSet and the fitter have different sampling frequencies.")
        missing = [name for name in fitter.ch_names if name not in ts.ch_names]
        if missing:
            raise DataError(f"The TrialSet lacks the fitter's channels {missing}.")
        index = [ts.ch_names.index(name) for name in fitter.ch_names]
        zero = int(np.argmin(np.abs(ts.times)))
        data = ts.data[:, index]
        if ts.masks is not None:
            data = np.where(ts.masks[:, index], np.nan, data)
        stream = np.swapaxes(data, 1, 2).reshape(-1, len(index))
        pulse_samples = start + np.arange(ts.n_trials) * ts.n_times + zero
    else:
        raw = np.asarray(trialset_or_raw, float)
        if raw.ndim != 2 or raw.shape[0] != len(fitter.ch_names):
            raise DataError(f"A raw array must be [{len(fitter.ch_names)}, n_times].")
        stream = raw.T
        pulse_samples = []
    for sample in pulse_samples:
        fitter.push_pulse(int(sample) / fitter.sfreq)
    out = []
    for first in range(0, len(stream), int(chunk_size)):
        out.extend(fitter.push(stream[first:first + int(chunk_size)]))
    return out
