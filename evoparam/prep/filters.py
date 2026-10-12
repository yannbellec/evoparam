# SPDX-License-Identifier: Apache-2.0
"""Zero-phase filters (thin wrappers over scipy.signal)."""

import time as _time

import numpy as np

from evoparam._typing import Any
from evoparam.errors import SettingsError
from evoparam.errors import warn as _warn

from ._common import check_trialset, new

__all__ = ["highpass", "lowpass", "notch"]


def _warn_masks(ts: Any, name: Any) -> None:
    if ts.masks is not None and ts.masks.any():
        _warn(
            f"{name}: the data have masked (replaced or blanked) samples; filtering "
            "can ring around them. Recommended order: artifact, filter, baseline, "
            "reference.",
            UserWarning,
            stacklevel=3,
        )


def _apply(ts: Any, sos: Any, name: Any, settings: Any, start: Any) -> Any:
    from scipy.signal import sosfiltfilt

    _warn_masks(ts, name)
    data = sosfiltfilt(sos, ts.data, axis=-1)
    return new(ts, f"evoparam.prep.{name}", settings, start, data=data)


def highpass(ts: Any, cutoff: Any, order: int = 4) -> Any:
    """Zero-phase Butterworth high-pass (``sosfiltfilt``)."""
    from scipy.signal import butter

    start = _time.perf_counter()
    check_trialset(ts)
    if not 0 < cutoff < ts.sfreq / 2:
        raise SettingsError("cutoff must be between 0 and the Nyquist frequency.")
    sos = butter(order, cutoff, btype="highpass", fs=ts.sfreq, output="sos")
    return _apply(ts, sos, "highpass", {"cutoff": cutoff, "order": order}, start)


def lowpass(ts: Any, cutoff: Any, order: int = 4) -> Any:
    """Zero-phase Butterworth low-pass (``sosfiltfilt``)."""
    from scipy.signal import butter

    start = _time.perf_counter()
    check_trialset(ts)
    if not 0 < cutoff < ts.sfreq / 2:
        raise SettingsError("cutoff must be between 0 and the Nyquist frequency.")
    sos = butter(order, cutoff, btype="lowpass", fs=ts.sfreq, output="sos")
    return _apply(ts, sos, "lowpass", {"cutoff": cutoff, "order": order}, start)


def notch(ts: Any, freq: float = 60.0, q: float = 30.0, harmonics: int = 3) -> Any:
    """Zero-phase notch filters at ``freq`` and its harmonics below the Nyquist frequency."""
    from scipy.signal import iirnotch, tf2sos

    start = _time.perf_counter()
    check_trialset(ts)
    if not 0 < freq < ts.sfreq / 2:
        raise SettingsError("freq must be between 0 and the Nyquist frequency.")
    sections = []
    used = []
    for h in range(1, harmonics + 1):
        f = h * freq
        if f >= ts.sfreq / 2:
            break
        b, a = iirnotch(f, q, fs=ts.sfreq)
        sections.append(tf2sos(b, a))
        used.append(f)
    sos = np.vstack(sections)
    return _apply(
        ts,
        sos,
        "notch",
        {"freq": freq, "q": q, "harmonics": harmonics, "frequencies": used},
        start,
    )
