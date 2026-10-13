# SPDX-License-Identifier: Apache-2.0
# Modified from the upstream project and commit named in NOTICE (Apache-2.0); changes: signal and
#   summary figures of plts/signals.py and plts/templates.py rewritten: each returns the Figure,
#   accepts ax=, never calls show(), uses the Okabe and Ito palette and draws numbers computed by
#   evoparam.stats
"""Waveform and summary figures kept from the inherited plotting helpers (spec §4.5).

Spec §4.5 keeps the signal plots (``plts/signals.py``) and the summary plots
(``plts/templates.py``) inside :mod:`evoparam.plots` with a lazy ``matplotlib`` import. They
follow the conventions of the module: each function returns the ``Figure``, accepts ``ax=``,
never calls ``show()`` and uses the Okabe and Ito palette. The numbers they draw come from
:mod:`evoparam.stats.waveforms` (REQ-349): the average and its band from
:func:`~evoparam.stats.waveform_band`, the jitter of the scatter plots from
:func:`~evoparam.stats.strip_jitter` (``numpy.random.default_rng(random_state)``, spec §9.4,
not the global state).
"""

from collections.abc import Sequence
from itertools import cycle

import numpy as np

from evoparam._typing import Any, ArrayLike, FloatArray, RandomState
from evoparam.errors import SettingsError
from evoparam.stats import strip_jitter, waveform_band

from ._plots import PALETTE, _figure_ax

__all__ = []


def _style(ax: Any, xlabel: str = "Time (s)", ylabel: str = "Amplitude") -> None:
    """Label the axes, draw a light grid and the legend (when something is labelled)."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right")


def _rows(values: Any) -> Any:
    """List of 1-D rows from a 1-D array, a 2-D array or a list of 1-D arrays."""
    if isinstance(values, np.ndarray):
        return [values] if values.ndim == 1 else list(values)
    return [np.asarray(row) for row in values]


def plot_signals(
    times: Any,
    signals: Any,
    *,
    colors: str | Sequence[str] | None = None,
    labels: str | Sequence[str] | None = None,
    ax: Any = None,
    linewidth: float = 2.0,
    ylabel: str = "Amplitude",
) -> Any:
    """Plot one or several waveforms.

    Parameters
    ----------
    times : 1-D array, 2-D array or list of 1-D arrays
        Times in seconds; one array shared by every waveform, or one per waveform.
    signals : 1-D array, 2-D array or list of 1-D arrays
        Waveforms, one per row.
    colors, labels : str or list of str, optional
        Line colours (default: the palette) and legend labels.
    ax : matplotlib Axes, optional
    linewidth : float
    ylabel : str
        Axis label, with the unit when known.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _figure_ax(ax)
    rows = _rows(signals)
    if isinstance(times, np.ndarray) and times.ndim == 1:
        time_rows = [times] * len(rows)
    else:
        time_rows = _rows(times)
    if len(time_rows) != len(rows):
        raise SettingsError("times must be one array or one array per waveform.")
    color_cycle = cycle(
        colors if isinstance(colors, (list, tuple)) else [colors] if colors else PALETTE
    )
    label_cycle = cycle(labels if isinstance(labels, (list, tuple)) else [labels])
    for time, row, color, label in zip(time_rows, rows, color_cycle, label_cycle, strict=False):
        ax.plot(time, row, color=color, label=label, linewidth=linewidth)
    _style(ax, ylabel=ylabel)
    return fig


def plot_signals_shading(
    times: Any,
    signals: Any,
    shades: Any,
    *,
    shade_colors: str | Sequence[str] = "#E69F00",
    add_center: bool = False,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot waveforms with one or several shaded time windows.

    Parameters
    ----------
    times, signals
        As :func:`plot_signals`.
    shades : [start, end] or list of [start, end]
        Windows in seconds.
    shade_colors : str or list of str
    add_center : bool
        Also draw a vertical line at the centre of each window.
    ax : matplotlib Axes, optional
    **kwargs
        Passed to :func:`plot_signals` (``colors``, ``labels``, ``linewidth``, ``ylabel``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _figure_ax(ax)
    plot_signals(times, signals, ax=ax, **kwargs)
    shades = [shades] if np.ndim(shades) == 1 else list(shades)
    colors = cycle(shade_colors if isinstance(shade_colors, (list, tuple)) else [shade_colors])
    for (start, end), color in zip(shades, colors, strict=False):
        ax.axvspan(start, end, color=color, alpha=0.2, lw=0)
        if add_center:
            ax.axvline((start + end) / 2.0, color=color, lw=1.0)
    _style(ax, ylabel=kwargs.get("ylabel", "Amplitude"))
    return fig


def plot_signals_yshade(  # noqa: PLR0913  (public keywords; DECISIONS.md §7.2)
    times: FloatArray,
    signals: FloatArray,
    *,
    shade: str = "std",
    average: str = "mean",
    scale: float = 1.0,
    color: str | None = None,
    label: str | None = None,
    alpha: float = 0.25,
    ax: Any = None,
    ylabel: str = "Amplitude",
) -> Any:
    """Plot the average waveform with a shaded band above and below it.

    Parameters
    ----------
    times : 1-D array
    signals : 1-D or 2-D array
        Waveforms (trials × times). A 1-D waveform needs ``shade`` given as an array.
    shade : {"std", "sem"}, 1-D array or callable
        Half-width of the band (callable: applied to the 2-D array along axis 0).
    average : {"mean", "median"} or callable
        Average of a 2-D array along axis 0.
    scale : float
        Factor applied to the half-width.
    color, label : str, optional
    alpha : float
    ax : matplotlib Axes, optional
    ylabel : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    mean, half = waveform_band(signals, spread=shade, average=average, scale=scale)
    color = color or PALETTE[0]
    fig, ax = _figure_ax(ax)
    ax.plot(times, mean, linewidth=2.0, color=color, label=label)
    ax.fill_between(times, mean - half, mean + half, alpha=alpha, color=color, lw=0)
    _style(ax, ylabel=ylabel)
    return fig


def _alpha(n_points: Any) -> float:
    """Marker transparency that decreases with the number of points."""
    for limit, value in ((1000, 0.10), (500, 0.25), (100, 0.40)):
        if n_points > limit:
            return value
    return 0.50


def plot_scatter_1(
    data: ArrayLike,
    label: str | None = None,
    title: str | None = None,
    *,
    x_val: float = 0,
    ax: Any = None,
    random_state: RandomState = 0,
) -> Any:
    """Strip plot of one set of values (horizontal jitter only for legibility).

    Parameters
    ----------
    data : 1-D array
    label : str, optional
        Axis and tick label.
    title : str, optional
    x_val : float
        Horizontal position.
    ax : matplotlib Axes, optional
    random_state : int, Generator or None
        Seed of the horizontal jitter.

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = np.asarray(data, dtype=float)
    positions = strip_jitter(data.shape, x=x_val, random_state=random_state)
    return _strip(data, positions, label, title, x_val, ax)


def _strip(data: Any, positions: Any, label: Any, title: Any, x_val: Any, ax: Any) -> Any:
    """Scatter the values at their jittered positions around ``x_val``."""
    fig, ax = _figure_ax(ax, figsize=(3.0, 3.5))
    ax.scatter(positions, data, s=36, alpha=_alpha(data.size), color=PALETTE[0])
    if label:
        ax.set_ylabel(label)
        ax.set(xticks=[x_val], xticklabels=[label])
    if title:
        ax.set_title(title)
    ax.set_xlim([x_val - 0.5, x_val + 0.5])
    return fig


def plot_scatter_2(
    data_0: Any,
    label_0: Any,
    data_1: Any,
    label_1: Any,
    title: Any = None,
    *,
    ax: Any = None,
    random_state: int = 0,
) -> Any:
    """Strip plots of two sets of values, each on its own vertical axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    data_0 = np.asarray(data_0, dtype=float)
    data_1 = np.asarray(data_1, dtype=float)
    # one stream for both strips (the second continues the first), as with one Generator
    both = strip_jitter(data_0.size + data_1.size, random_state=random_state)
    fig, ax = _figure_ax(ax, figsize=(4.0, 3.5))
    other = ax.twinx()
    _strip(data_0, both[: data_0.size].reshape(data_0.shape), label_0, None, 0, ax)
    _strip(data_1, 1 + both[data_0.size :].reshape(data_1.shape), label_1, None, 1, other)
    for collection in other.collections:
        collection.set_color(PALETTE[1])
    if title:
        ax.set_title(title)
    ax.set(xlim=[-0.5, 1.5], xticks=[0, 1], xticklabels=[label_0, label_1])
    return fig


def plot_hist(
    data: Any,
    label: Any,
    title: Any = None,
    *,
    n_bins: int = 25,
    x_lims: Any = None,
    ax: Any = None,
) -> Any:
    """Histogram of the finite values of ``data``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = np.asarray(data, dtype=float)
    fig, ax = _figure_ax(ax)
    ax.hist(data[np.isfinite(data)], n_bins, range=x_lims, alpha=0.8, color=PALETTE[0])
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    if x_lims:
        ax.set_xlim(x_lims)
    if title:
        ax.set_title(title)
    return fig
