# SPDX-License-Identifier: Apache-2.0
"""Resampling with an anti-aliasing filter."""

import time as _time
from fractions import Fraction
from typing import TYPE_CHECKING, cast

import numpy as np

from evoparam._typing import Any
from evoparam.errors import SettingsError

from ._common import check_trialset, new

if TYPE_CHECKING:
    from evoparam.data.containers import TrialSet

__all__ = ["resample"]


def resample(ts: Any, sfreq: float) -> "TrialSet":
    """Change the sampling frequency (polyphase filtering, anti-aliasing FIR).

    ``scipy.signal.resample_poly`` with the ratio ``sfreq / ts.sfreq`` as a fraction
    (denominator at most 1000). The new time axis keeps the first time of the epoch and
    the onset at 0 when 0 falls on the new grid; ``event_times`` are unchanged (they are
    times, not samples); masks are carried to the nearest new sample.

    Returns
    -------
    TrialSet
    """
    from scipy.signal import resample_poly

    start = _time.perf_counter()
    check_trialset(ts)
    if not sfreq > 0:
        raise SettingsError("sfreq must be positive.")
    ratio = Fraction(float(sfreq) / ts.sfreq).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    new_sfreq = ts.sfreq * up / down
    data = resample_poly(ts.data, up, down, axis=-1, padtype="line")
    n_new = data.shape[-1]
    # Keep the sample nearest to t = 0 at exactly 0 when the grid allows it
    first = ts.times[0]
    offset = round(-first * new_sfreq)
    times = (
        (np.arange(n_new) - offset) / new_sfreq
        if abs(-first * new_sfreq - offset) < 1e-6
        else first + np.arange(n_new) / new_sfreq
    )
    masks = None
    if ts.masks is not None:
        nearest = np.clip(
            np.round((times - ts.times[0]) * ts.sfreq).astype(int), 0, ts.n_times - 1
        )
        masks = ts.masks[:, :, nearest]
    return cast(
        "TrialSet",
        new(
            ts,
            "evoparam.prep.resample",
            {"sfreq": float(sfreq), "up": up, "down": down, "from_sfreq": ts.sfreq},
            start,
            data=data,
            times=times,
            sfreq=new_sfreq,
            masks=masks,
        ),
    )
