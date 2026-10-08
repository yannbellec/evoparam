"""Figures of every result type (spec §5.13, extra ``plot``).

Every function returns the matplotlib ``Figure``, accepts ``ax=`` (or ``axes=`` when it
draws several panels), never calls ``show()`` and uses a colour-blind-safe palette (Okabe
and Ito). The functions draw what the results hold; they compute no new scientific quantity
(spec REQ-349): a fitted curve is evaluated with :func:`evoparam.models.evaluate_fit`.
"""

from ._plots import (PALETTE, plot_artifact, plot_compare, plot_existence, plot_fit,
                     plot_group, plot_hierarchical, plot_shapes, plot_tf, plot_trials)

__all__ = ["plot_fit", "plot_group", "plot_trials", "plot_existence", "plot_tf", "plot_shapes",
           "plot_artifact", "plot_compare", "plot_hierarchical", "PALETTE"]
