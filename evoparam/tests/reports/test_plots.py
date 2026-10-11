# SPDX-License-Identifier: Apache-2.0
"""Every plot function runs headless on its result type and returns a Figure (REQ-344)."""

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from evoparam import plots  # noqa: E402
from evoparam.data import ShapeBasis  # noqa: E402


def _is_figure(fig):
    from matplotlib.figure import Figure

    return isinstance(fig, Figure)


def test_plot_fit_with_and_without_data(results):

    assert _is_figure(plots.plot_fit(results["fit"]))
    assert _is_figure(plots.plot_fit(results["fit"], results["ts"], channel="c0"))


@pytest.mark.parametrize("kind", ["grid", "shaft"])
def test_plot_group(results, kind):

    assert _is_figure(plots.plot_group(results["group"], results["ts"], kind=kind))


def test_plot_group_brain_needs_coordinates(results):

    from evoparam.errors import SettingsError

    with pytest.raises(SettingsError, match="coords"):
        plots.plot_group(results["group"], kind="brain")
    coords = np.random.default_rng(0).normal(size=(3, 2))
    assert _is_figure(plots.plot_group(results["group"], kind="brain", coords=coords))


def test_plot_trials_existence_tf_artifact_compare(results):

    assert _is_figure(plots.plot_trials(results["trials"], results["ts"], channel="c0"))
    assert _is_figure(plots.plot_trials(results["trials"]))
    assert _is_figure(plots.plot_existence(results["test"]))
    assert _is_figure(plots.plot_tf(results["tf"]))
    assert _is_figure(plots.plot_artifact(results["artifact"], results["ts"]))
    assert _is_figure(plots.plot_compare([results["fit"], results["group"].fits[1]],
                                         results["ts"], channel="c0"))


def test_plot_shapes(results):

    basis = results["basis"] or ShapeBasis(
        curves=np.ones((1, 10)), times=np.linspace(0, 0.1, 10), weights=np.ones((2, 1)),
        r2=np.ones(2), assignment=None, stability=np.ones(1), n_shapes=1, method="nmf")
    assert _is_figure(plots.plot_shapes(basis))


def test_axes_are_reused(results):

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    assert plots.plot_fit(results["fit"], ax=ax) is fig
    plt.close("all")


def test_req344_every_result_type_has_a_figure(results):
    """One figure per result type of spec §4.3 (the fitted ones above, the others from the
    data-model factories), under the Agg backend."""

    import matplotlib
    import matplotlib.pyplot as plt

    from evoparam.tests.data.factories import all_results

    made = all_results()
    figures = {
        "ResponseFit": plots.plot_fit(results["fit"]),
        "GroupFit": plots.plot_group(results["group"]),
        "TrialFit": plots.plot_trials(results["trials"]),
        "ResponseTest": plots.plot_existence(made["ResponseTest"]),
        "TFFit": plots.plot_tf(results["tf"]),
        "ShapeBasis": plots.plot_shapes(made["ShapeBasis"]),
        "ArtifactResult": plots.plot_artifact(made["ArtifactResult"]),
        "HierarchicalFit": plots.plot_hierarchical(made["HierarchicalFit"]),
    }
    assert matplotlib.get_backend().lower() == "agg"
    for name, fig in figures.items():
        assert _is_figure(fig), name
        fig.canvas.draw()
    fig, axes = plt.subplots(1, 2)
    assert plots.plot_hierarchical(made["HierarchicalFit"], axes=axes) is fig
    plt.close("all")


def test_plot_existence_shows_the_statistic_over_time_with_its_null_band(results):
    """Spec §5.13: plot_existence(test) draws the statistic over time with the null band."""

    import dataclasses

    from matplotlib.collections import PolyCollection

    test = results["test"]
    assert test.curve is not None and test.null_band is not None
    fig = plots.plot_existence(test)
    panels = [a for a in fig.axes if a.lines]
    channels = test.settings["channels"]
    assert len(panels) == len(channels)
    for c, a in enumerate(panels):
        curve = np.asarray(test.curve)[c]
        curve = curve[0] if curve.ndim > 1 else curve          # the first group (site)
        drawn = [line for line in a.lines if len(line.get_xdata()) == curve.size]
        assert any(np.allclose(line.get_ydata(), curve, equal_nan=True) for line in drawn)
        assert np.allclose(drawn[0].get_xdata(), test.curve_times * 1e3)
        assert any(isinstance(item, PolyCollection) for item in a.collections)   # the band
    # one axes: every channel on it; a test without a time course: one bar per channel
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    assert plots.plot_existence(test, ax=ax) is ax.figure and len(ax.lines) >= len(channels)
    old = dataclasses.replace(test, curve=None, null_band=None, curve_times=None)
    fig = plots.plot_existence(old)
    assert len(fig.axes[0].patches) == len(channels)
