# SPDX-License-Identifier: Apache-2.0
"""Pulse detection on a stream: the threshold step of :func:`detect_pulses`, causal.

The score is that of the offline detector (median across channels of the absolute
sample-to-sample change, at most 16 evenly spaced channels), scaled by its median and MAD
over a calibration period at the start of the stream. A pulse is the first sample whose
score exceeds ``threshold_sd`` (the offline ``"threshold"`` method without onset
refinement); further crossings within ``min_interval`` belong to the same pulse. The
result does not depend on how the stream is cut into chunks.
"""

import numpy as np

from evoparam._typing import Any, NDArray
from evoparam.errors import DataError, SettingsError

from ._score import MAD_TO_SD, score_index

__all__ = ["OnlinePulseDetector"]


# The samples of an update are a matrix [n_samples, n_channels].
_MATRIX_NDIM = 2


class OnlinePulseDetector:
    """Causal threshold detector of stimulation pulses.

    Parameters
    ----------
    sfreq : float
    n_channels : int
        Channels of the samples passed to :meth:`update`.
    threshold_sd : float
        Threshold in robust standard deviations of the score (as in :func:`detect_pulses`).
    min_interval : float
        Shortest interval between two pulses (s), as in :func:`detect_pulses`.
    calibration : float
        Length of the calibration period (s) whose score gives the median and MAD. Pulses
        inside it are reported when it ends.
    """

    def __init__(
        self,
        sfreq: float,
        n_channels: int,
        *,
        threshold_sd: float = 20.0,
        min_interval: float = 0.5,
        calibration: float = 1.0,
    ) -> None:
        """Check the settings; the threshold is set after ``calibration`` seconds of samples."""
        if not sfreq > 0:
            raise SettingsError("sfreq must be positive.")
        if not threshold_sd > 0:
            raise SettingsError("threshold_sd must be positive.")
        if not min_interval > 0:
            raise SettingsError("min_interval must be positive.")
        self.sfreq: float = float(sfreq)
        self.n_channels: int = int(n_channels)
        self.threshold_sd: float = float(threshold_sd)
        self.min_samples: int = max(1, int(round(min_interval * self.sfreq)))
        self.n_calibration: int = max(3, int(round(calibration * self.sfreq)))
        self._index = score_index(np.arange(self.n_channels))
        self.reset()

    def reset(self) -> None:
        """Forget the stream (the calibration is redone)."""
        self.threshold: float | None = None
        self.center: float = 0.0
        self.spread: float = 1.0
        self._previous = None
        self._pending: list[Any]
        self._pending = []
        self._next_allowed = 0
        self._n_seen = 0

    def _score(self, samples: Any) -> NDArray[Any]:
        """Return the score of a block of samples: the median absolute step over channels."""
        block = samples[:, self._index]
        if self._previous is None:
            diff = np.vstack([np.zeros((1, block.shape[1])), np.diff(block, axis=0)])
        else:
            diff = np.diff(np.vstack([self._previous, block]), axis=0)
        self._previous = block[-1:].copy()
        diff = np.nan_to_num(np.abs(diff), nan=0.0)
        if diff.shape[1] == 1:
            first: NDArray[Any] = diff[:, 0]
            return first
        return np.partition(diff, diff.shape[1] // 2, axis=1)[:, diff.shape[1] // 2]

    def _calibrate(self, score: Any) -> None:
        """Set the centre and the robust spread of the score from the calibration samples."""
        self.center = float(np.median(score))
        spread = MAD_TO_SD * float(np.median(np.abs(score - self.center)))
        if not spread > 0:
            spread = float(np.std(score))
        self.spread = spread if spread > 0 else 1.0
        self.threshold = self.center + self.threshold_sd * self.spread

    def _crossings(self, score: Any, first: Any) -> list[Any]:
        """Return the samples above the threshold, outside the refractory period."""
        out = []
        for i in np.flatnonzero(score > self.threshold):
            sample = first + int(i)
            if sample >= self._next_allowed:
                out.append(sample)
                self._next_allowed = sample + self.min_samples
        return out

    def update(self, samples: Any) -> list[Any]:
        """Add samples [n_samples, n_channels].

        Returns the onset samples (stream indices) of the pulses found.
        """
        samples = np.asarray(samples, float)
        if samples.ndim != _MATRIX_NDIM or samples.shape[1] != self.n_channels:
            raise DataError(f"samples must be [n_samples, {self.n_channels}].")
        if not len(samples):
            return []
        first = self._n_seen
        self._n_seen += len(samples)
        score = self._score(samples)
        if self.threshold is not None:
            return self._crossings(score, first)
        self._pending.append(score)
        held = sum(len(s) for s in self._pending)
        if held < self.n_calibration:
            return []
        score = np.concatenate(self._pending)
        self._pending = []
        self._calibrate(score[: self.n_calibration])
        return self._crossings(score, self._n_seen - len(score))
