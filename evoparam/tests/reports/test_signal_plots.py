# SPDX-License-Identifier: Apache-2.0
"""The waveform and summary figures of :mod:`evoparam.plots` (spec §4.5, REQ-344, REQ-349).

Each function returns the Figure, draws on the ``ax`` it is given, uses the Okabe and Ito
palette, and draws the numbers computed by :mod:`evoparam.stats.waveforms`: the lines hold
the input waveforms, the band is :func:`~evoparam.stats.waveform_band`, the jitter of the
strip plots is :func:`~evoparam.stats.strip_jitter`.
"""

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

pytest.importorskip("matplotlib")

from matplotlib.colors import to_hex  # noqa: E402  (after pytest.importorskip)
from matplotlib.figure import Figure  # noqa: E402  (after pytest.importorskip)

from evoparam import plots  # noqa: E402  (after pytest.importorskip)
from evoparam.errors import SettingsError  # noqa: E402  (after pytest.importorskip)
from evoparam.plots._plots import PALETTE  # noqa: E402  (after pytest.importorskip)
from evoparam.stats import strip_jitter, waveform_band  # noqa: E402  (after pytest.importorskip)

TIMES = np.linspace(-0.1, 0.4, 51)
SIGNALS = np.stack([np.sin(2 * np.pi * f * TIMES) for f in (2.0, 3.0, 5.0)])


def _close(fig: Any) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_signals_draws_each_waveform() -> None:
    fig = plots.plot_signals(TIMES, SIGNALS, labels=["a", "b", "c"], ylabel="Amplitude (uV)")
    assert isinstance(fig, Figure)
    (ax,) = fig.axes
    lines = ax.get_lines()
    assert len(lines) == len(SIGNALS)
    for line, row, color in zip(lines, SIGNALS, PALETTE, strict=False):
        assert_array_equal(line.get_xdata(), TIMES)
        assert_array_equal(line.get_ydata(), row)
        assert to_hex(line.get_color()) == to_hex(color)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["a", "b", "c"]
    assert ax.get_ylabel() == "Amplitude (uV)"
    assert ax.get_xlabel() == "Time (s)"
    _close(fig)


def test_plot_signals_with_one_time_axis_per_waveform_and_one_colour() -> None:
    times = [TIMES, TIMES[:20]]
    signals = [SIGNALS[0], SIGNALS[1][:20]]
    fig = plots.plot_signals(times, signals, colors="#000000", labels="x")
    lines = fig.axes[0].get_lines()
    assert [len(line.get_xdata()) for line in lines] == [TIMES.size, 20]
    assert {to_hex(line.get_color()) for line in lines} == {"#000000"}
    with pytest.raises(SettingsError, match="one array per waveform"):
        plots.plot_signals([TIMES], signals)
    # a 1-D waveform, on an axis given by the caller
    import matplotlib.pyplot as plt

    other, ax = plt.subplots()
    assert plots.plot_signals(TIMES, SIGNALS[0], ax=ax, colors=["#123456"]) is other
    assert to_hex(ax.get_lines()[0].get_color()) == "#123456"
    assert ax.get_legend() is None  # no label, no legend
    _close(fig)
    _close(other)


def test_plot_signals_shading() -> None:
    fig = plots.plot_signals_shading(TIMES, SIGNALS, [0.0, 0.1], add_center=True)
    ax = fig.axes[0]
    assert len(ax.get_lines()) == len(SIGNALS) + 1  # the waveforms and the centre line
    assert_allclose(ax.get_lines()[-1].get_xdata(), [0.05, 0.05])
    assert len(ax.patches) == 1
    _close(fig)
    windows = [[0.0, 0.1], [0.2, 0.3]]
    fig = plots.plot_signals_shading(TIMES, SIGNALS, windows, shade_colors=["#111111", "#222222"])
    ax = fig.axes[0]
    assert len(ax.patches) == len(windows)
    assert len(ax.get_lines()) == len(SIGNALS)
    assert [to_hex(p.get_facecolor()) for p in ax.patches] == ["#111111", "#222222"]
    _close(fig)


@pytest.mark.parametrize(("shade", "average"), [("std", "mean"), ("sem", "median")])
def test_plot_signals_yshade_draws_the_waveform_band(shade: str, average: str) -> None:
    fig = plots.plot_signals_yshade(TIMES, SIGNALS, shade=shade, average=average, scale=2.0)
    ax = fig.axes[0]
    mean, half = waveform_band(SIGNALS, spread=shade, average=average, scale=2.0)
    assert_allclose(ax.get_lines()[0].get_ydata(), mean)
    (band,) = ax.collections
    vertices = band.get_paths()[0].vertices
    assert_allclose(vertices[:, 1].max(), (mean + half).max())
    assert_allclose(vertices[:, 1].min(), (mean - half).min())
    assert to_hex(ax.get_lines()[0].get_color()) == to_hex(PALETTE[0])
    _close(fig)


def test_plot_signals_yshade_of_one_waveform_needs_the_band() -> None:
    half = np.full(TIMES.size, 0.5)
    fig = plots.plot_signals_yshade(TIMES, SIGNALS[0], shade=half, color="#000000", label="x")
    assert_allclose(fig.axes[0].get_lines()[0].get_ydata(), SIGNALS[0])
    _close(fig)
    with pytest.raises(SettingsError, match="2-D"):
        plots.plot_signals_yshade(TIMES, SIGNALS[0])


@pytest.mark.parametrize(("n_points", "alpha"), [(50, 0.5), (200, 0.4), (600, 0.25), (2000, 0.1)])
def test_plot_scatter_1_draws_the_jitter_of_stats(n_points: int, alpha: float) -> None:
    data = np.random.default_rng(3).normal(size=n_points)
    fig = plots.plot_scatter_1(data, label="latency (s)", title="N1", x_val=2, random_state=7)
    ax = fig.axes[0]
    offsets = ax.collections[0].get_offsets()
    assert_allclose(offsets[:, 0], strip_jitter(data.shape, x=2, random_state=7))
    assert_allclose(offsets[:, 1], data)
    assert ax.collections[0].get_alpha() == alpha
    assert ax.get_title() == "N1"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["latency (s)"]
    assert ax.get_xlim() == (1.5, 2.5)
    _close(fig)


def test_plot_scatter_1_without_labels() -> None:
    fig = plots.plot_scatter_1([1.0, 2.0, 3.0])
    ax = fig.axes[0]
    assert ax.get_title() == ""
    assert ax.get_ylabel() == ""
    _close(fig)


def test_plot_scatter_2_continues_one_jitter_stream() -> None:
    first, second = np.arange(4.0), np.arange(3.0) + 10.0
    fig = plots.plot_scatter_2(first, "a", second, "b", title="two", random_state=5)
    ax, other = fig.axes
    jitter = strip_jitter(first.size + second.size, random_state=5)
    assert_allclose(ax.collections[0].get_offsets()[:, 0], jitter[: first.size])
    assert_allclose(other.collections[0].get_offsets()[:, 0], 1 + jitter[first.size :])
    assert_allclose(other.collections[0].get_offsets()[:, 1], second)
    assert to_hex(other.collections[0].get_facecolor()[0]) == to_hex(PALETTE[1])
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"]
    assert ax.get_title() == "two"
    _close(fig)
    fig = plots.plot_scatter_2(first, "a", second, "b")
    assert fig.axes[0].get_title() == ""
    _close(fig)


def test_plot_hist_counts_the_finite_values() -> None:
    data = np.array([0.1, 0.2, 0.2, np.nan, np.inf, 0.9])
    fig = plots.plot_hist(data, "latency (s)", title="h", n_bins=4, x_lims=(0.0, 1.0))
    ax = fig.axes[0]
    heights = [patch.get_height() for patch in ax.patches]
    assert_array_equal(heights, np.histogram(data[np.isfinite(data)], 4, range=(0.0, 1.0))[0])
    assert ax.get_xlim() == (0.0, 1.0)
    assert ax.get_title() == "h"
    assert ax.get_ylabel() == "Count"
    _close(fig)
    fig = plots.plot_hist(data, "x")
    assert sum(patch.get_height() for patch in fig.axes[0].patches) == np.isfinite(data).sum()
    _close(fig)
