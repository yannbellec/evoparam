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
