"""Component names by latency window and polarity (spec §5.5.7). Labels are cosmetic."""

import dataclasses

__all__ = ["label_components"]


def _label_fit(fit, windows, unassigned):
    """Labels for one ResponseFit; windows: name -> (start, end, polarity) with polarity
    -1, +1 or 0 (either)."""

    assigned = {}
    warnings = []
    for name, (start, end, polarity) in windows.items():
        inside = [k for k, comp in enumerate(fit.components)
                  if start <= comp.latency <= end and (polarity == 0 or comp.polarity == polarity)]
        inside.sort(key=lambda k: -abs(fit.components[k].amplitude))
        for rank, k in enumerate(inside):
            if k in assigned:
                continue
            assigned[k] = name if rank == 0 else f"{name}_b"
            if rank:
                warnings.append(f"multiple_in_window:{name}")
    comps = [dataclasses.replace(comp, label=assigned.get(k, unassigned.format(k=k + 1)))
             for k, comp in enumerate(fit.components)]
    return dataclasses.replace(fit, components=comps,
                               warnings=list(fit.warnings) + sorted(set(warnings)))


def label_components(fit, windows, *, unassigned="C{k}"):
    """Name the components of a ResponseFit or GroupFit.

    Parameters
    ----------
    fit : ResponseFit or GroupFit
    windows : dict or str
        Name -> (start, end, polarity) in seconds (polarity -1, +1 or 0 for either), or the
        name of a preset whose windows are used (for example ``"ccep"``: N1 9-50 ms and N2
        50-300 ms, both negative).
    unassigned : str
        Name of a component outside every window (``{k}`` is its position from 1).

    Returns
    -------
    Same type as ``fit``. When two components fall in one window, the larger ``|A|`` gets the
    name and the other ``"<name>_b"`` with the warning ``"multiple_in_window:<name>"``.
    """

    from evoparam.data import GroupFit

    from ._settings import FIT_PRESETS

    if isinstance(windows, str):
        windows = FIT_PRESETS[windows]["windows"]
    if isinstance(fit, GroupFit):
        return dataclasses.replace(fit, fits=[_label_fit(f, windows, unassigned)
                                              for f in fit.fits])
    return _label_fit(fit, windows, unassigned)
