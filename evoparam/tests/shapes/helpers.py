"""Shared fixtures of the shape tests: time axis, template and parameter samplers."""

import numpy as np

from evoparam import shapes

SFREQ = 2000.0
# Fit window of 320 ms at 2 kHz, starting 20 ms before the event.
TIMES = -0.02 + np.arange(640) / SFREQ


def make_template(stretch=False):
    """Biphasic template at rest at both ends (sampled at 1 kHz over 200 ms)."""

    t = np.arange(201) / 1000.0
    taper = np.sin(np.pi * t / 0.2) ** 2
    values = taper * (np.exp(-((t - 0.05) / 0.012) ** 2) - 0.6 * np.exp(-((t - 0.1) / 0.02) ** 2))
    return shapes.Template(t, values, stretch=stretch)


def sample_params(shape, times, n_sets, rng):
    """Uniform parameter sets inside the default bounds, with the shape's constraints."""

    lower, upper = shape.default_bounds(times)
    params = lower + (upper - lower) * rng.random((n_sets, shape.n_params))
    if shape.name == "biexponential":
        # tau_decay >= 1.5 tau_rise
        low = np.maximum(lower[2], 1.5 * params[:, 1])
        params[:, 2] = low + (upper[2] - low) * rng.random(n_sets)
    return params


def sample_background_params(background, times, n_sets, rng, data_range=None):
    """Uniform parameter sets inside the default bounds, with tau2 >= 3 tau1."""

    lower, upper = background.default_bounds(times, data_range)
    params = lower + (upper - lower) * rng.random((n_sets, background.n_params))
    if background.name == "double_exponential":
        low = np.maximum(lower[4], 3.0 * params[:, 3])
        params[:, 4] = low + (upper[4] - low) * rng.random(n_sets)
    return params
