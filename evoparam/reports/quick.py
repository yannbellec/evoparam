# SPDX-License-Identifier: Apache-2.0
"""Fit one waveform and show the result in one call, for a first try of the package.

The counterpart of a spectral model's ``report``: fit, summary and figure together. The
summary is the fit itself (its text form is shown when it is the last expression of a
notebook cell, or by ``print(fit)``); the package prints nothing (REQ-304).
"""

import numpy as np

from evoparam._typing import Any
from evoparam.data import Evoked
from evoparam.errors import SettingsError
from evoparam.models import ResponseModel

# Fewest samples of a waveform: two give the sampling rate.
MIN_SAMPLES = 2


def report(
    times: Any,
    signal: Any,
    preset: Any = "ccep",
    *,
    units: str = "uV",
    plot: bool = True,
    random_state: Any = 0,
    **settings: Any,
) -> Any:
    """Fit one evoked response and draw the fit; returns the :class:`ResponseFit`.

    Parameters
    ----------
    times : array
        Time of each sample, in seconds (the stimulus at 0).
    signal : array
        The response, one value per time.
    preset : str
        The preset whose settings are used (``"ccep"``, ``"erp"``, ...).
    units : str
        Unit of ``signal``.
    plot : bool
        Draw the data, the model, its parts and the residual (needs matplotlib).
    random_state : int or None
        Seed of the fit (0: the same result at every run).
    **settings
        Any setting of :class:`~evoparam.ResponseModel`, replacing the preset's.

    Returns
    -------
    ResponseFit
        The fit; ``print(fit)`` shows its summary.

    Examples
    --------
    >>> from evoparam.sim import gen_response
    >>> times, signal = gen_response([[25, -150, 10]], seed=0)
    >>> fit = report(times, signal, plot=False)
    >>> fit.components[0].label
    'N1'
    """
    if len(times) != len(signal) or len(times) < MIN_SAMPLES:
        raise SettingsError("times and signal must have the same length (at least 2).")
    times = np.asarray(times, dtype=float)
    evoked = Evoked(
        data=np.asarray(signal, dtype=float)[None, :],
        times=times,
        sfreq=1.0 / (times[1] - times[0]),
        ch_names=["signal"],
        units=units,
    )
    model = ResponseModel(preset, random_state=random_state, **settings)
    fit = model.fit_group(evoked).get("signal")
    if plot:
        from evoparam.plots import plot_fit

        fig = plot_fit(fit, evoked, channel="signal")
        # show the fit window and a little before it, where the response is
        start, end = (1000 * v for v in fit.window)
        fig.axes[0].set_xlim(min(start, 0.0) - 0.1 * (end - start), end + 0.1 * (end - start))
        fig.axes[0].set_ylabel(f"amplitude ({units.replace('u', 'µ', 1)})")
    return fit
