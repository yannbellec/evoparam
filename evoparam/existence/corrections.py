# SPDX-License-Identifier: Apache-2.0
"""Multiple-comparison corrections of p-values."""

import numpy as np

from evoparam._typing import Any
from evoparam.errors import SettingsError

__all__ = ["correct", "CORRECTIONS"]

CORRECTIONS = ("fdr_bh", "holm", "bonferroni", "none")


def correct(p: Any, method: str = "fdr_bh") -> Any:
    """Adjust p-values (NaN entries are left out and stay NaN).

    - ``fdr_bh``: Benjamini-Hochberg, ``p_(i) * m / i`` made monotone from the largest.
    - ``holm``: Holm-Bonferroni step-down.
    - ``bonferroni``: ``p * m``.
    - ``none``: unchanged.
    """
    if method not in CORRECTIONS:
        raise SettingsError(f"correction must be one of {CORRECTIONS}, not {method!r}.")
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    flat = p.ravel()
    ok = np.isfinite(flat)
    values = flat[ok]
    m = len(values)
    if m == 0 or method == "none":
        out.ravel()[ok] = values
        return out
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    if method == "bonferroni":
        adj = np.minimum(values * m, 1.0)
    elif method == "holm":
        steps = np.minimum(1.0, np.maximum.accumulate(ranked * (m - np.arange(m))))
        adj = np.empty(m)
        adj[order] = steps
    else:
        steps = ranked * m / np.arange(1, m + 1)
        steps = np.minimum(1.0, np.minimum.accumulate(steps[::-1])[::-1])
        adj = np.empty(m)
        adj[order] = steps
    flat_out = out.ravel()
    flat_out[ok] = adj
    return flat_out.reshape(p.shape)
