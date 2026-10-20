# SPDX-License-Identifier: Apache-2.0
"""One example response in one line, for a first try of the package (tutorial 01).

The counterpart of a one-line spectrum generator: the components are given in
milliseconds and microvolts, the way a response is described in words, and the times are
returned in seconds, the unit of the rest of the package.
"""

import numpy as np

from evoparam._typing import Any, FloatArray
from evoparam.errors import SettingsError
from evoparam.shapes import get_shape

from .components import shape_params
from .noise import make_noise

# Numbers given for each component: latency (ms), amplitude (µV), width (ms).
N_VALUES = 3
# Dimensions of the table of components (one row per component).
TABLE_NDIM = 2


def gen_response(
    components: Any,
    background: Any = (60.0, 30.0),
    noise: float = 8.0,
    *,
    tmin: float = -500.0,
    tmax: float = 1000.0,
    sfreq: float = 2048.0,
    seed: Any = None,
) -> tuple[FloatArray, FloatArray]:
    """Make one example evoked response: bumps, a slow background and noise.

    Parameters
    ----------
    components : list of [latency, amplitude, width]
        One entry per bump (Gaussian): the time of its peak after the stimulus (ms), its
        height (µV, negative for a downward bump) and its full width at half maximum (ms).
    background : [height, time_constant] or None
        A slow decay that starts at the stimulus, ``height × exp(−t / time_constant)``
        (µV, ms), like the recovery of an amplifier after a pulse; None for none.
    noise : float
        Standard deviation of the noise (µV); the noise is pink (power ``1/f``). 0 for none.
    tmin, tmax : float
        First and last time (ms), the stimulus being at 0.
    sfreq : float
        Samples per second (Hz).
    seed : int, numpy.random.Generator or None
        Seed of the noise, for the same signal at every run.

    Returns
    -------
    times : ndarray
        Time of each sample, in seconds.
    signal : ndarray
        The response, in µV.

    Examples
    --------
    >>> times, signal = gen_response([[25, -150, 10], [150, -80, 80]], seed=0)
    >>> times.shape == signal.shape
    True
    """
    comps = np.asarray(components, dtype=float)
    if comps.size == 0:
        comps = np.zeros((0, N_VALUES))
    if comps.ndim != TABLE_NDIM or comps.shape[1] != N_VALUES:
        raise SettingsError("components: a list of [latency (ms), amplitude (µV), width (ms)].")
    if np.any(comps[:, 2] <= 0):
        raise SettingsError("components: every width (ms) must be positive.")
    if not tmin < 0 < tmax:
        raise SettingsError("tmin and tmax (ms) must surround the stimulus at 0.")
    if sfreq <= 0 or noise < 0:
        raise SettingsError("sfreq must be positive and noise must not be negative.")
    times = np.arange(round(tmin * sfreq / 1000), round(tmax * sfreq / 1000) + 1) / sfreq
    gaussian = get_shape("gaussian")
    signal = np.zeros(times.size)
    for latency, amplitude, width in comps:
        params = shape_params("gaussian", latency / 1000, width / 1000)
        signal += amplitude * gaussian.evaluate(times, params)
    if background is not None:
        height, tau = (float(v) for v in background)
        if tau <= 0:
            raise SettingsError("background: the time constant (ms) must be positive.")
        after = times >= 0
        signal[after] += height * np.exp(-times[after] / (tau / 1000))
    if noise > 0:
        signal += make_noise(times.size, sfreq, "pink", sd=noise, seed=seed)
    return times, signal
