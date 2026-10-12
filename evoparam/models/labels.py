# SPDX-License-Identifier: Apache-2.0
"""Component names by latency window and polarity (spec §5.5.7). Labels are cosmetic."""

import dataclasses
import re

__all__ = ["label_components", "sign_name", "POLARITY_RULES"]

POLARITY_RULES = ("sign", "none")


def sign_name(name, polarity):
    """Return the name of a component in a window named by sign, ``"P1/N1"``.

    ``"P1"`` for a positive component and ``"N1"`` for a negative one (spec §6.1, rodent and
    primate presets: "P1/N1 by depth": in a laminar recording the early field potential reverses
    polarity with depth, so each contact names its component by its own sign). Other names are
    returned unchanged.

    >>> sign_name("P1/N1", -1), sign_name("P1/N1", 1), sign_name("N2", -1)
    ('N1', 'P1', 'N2')
    """
    parts = str(name).split("/")
    if len(parts) != 2 or polarity == 0:
        return name
    signs = [{"P": 1, "N": -1}.get(part[:1]) for part in parts]
    if sorted(s for s in signs if s is not None) != [-1, 1] or not all(
        re.fullmatch(r"[PN]\w*", part) for part in parts
    ):
        return name
    return parts[signs.index(1 if polarity > 0 else -1)]


def _label_fit(fit, windows, unassigned, polarity_rule="sign"):
    """Return the labels for one ResponseFit.

    ``windows``: name -> (start, end, polarity) with polarity -1, +1 or 0 (either).
    """
    assigned = {}
    warnings = []
    for name, (start, end, polarity) in windows.items():
        inside = [
            k
            for k, comp in enumerate(fit.components)
            if start <= comp.latency <= end and polarity in (0, comp.polarity)
        ]
        inside.sort(key=lambda k: -abs(fit.components[k].amplitude))
        for rank, k in enumerate(inside):
            if k in assigned:
                continue
            label = (
                sign_name(name, fit.components[k].polarity) if polarity_rule == "sign" else name
            )
            assigned[k] = label if rank == 0 else f"{label}_b"
            if rank:
                warnings.append(f"multiple_in_window:{name}")
    comps = [
        dataclasses.replace(comp, label=assigned.get(k, unassigned.format(k=k + 1)))
        for k, comp in enumerate(fit.components)
    ]
    return dataclasses.replace(
        fit, components=comps, warnings=list(fit.warnings) + sorted(set(warnings))
    )


def label_components(fit, windows, *, polarity_rule="sign", unassigned="C{k}"):
    """Name the components of a ResponseFit or GroupFit.

    Parameters
    ----------
    fit : ResponseFit or GroupFit
    windows : dict or str
        Name -> (start, end, polarity) in seconds (polarity -1, +1 or 0 for either), or the
        name of a preset whose windows are used (for example ``"ccep"``: N1 9-50 ms and N2
        50-300 ms, both negative).
    polarity_rule : {"sign", "none"}
        ``"sign"`` (default): a window named by sign, ``"P1/N1"`` (rodent and primate
        presets, "P1/N1 by depth"), names each component by its own sign, ``"P1"`` when
        positive and ``"N1"`` when negative (:func:`sign_name`); ``"none"`` keeps the window
        name.
    unassigned : str
        Name of a component outside every window (``{k}`` is its position from 1).

    Returns
    -------
    Same type as ``fit``. When two components fall in one window, the larger ``|A|`` gets the
    name and the other ``"<name>_b"`` with the warning ``"multiple_in_window:<name>"``.
    """
    from evoparam.data import GroupFit
    from evoparam.errors import SettingsError

    from ._settings import FIT_PRESETS

    if polarity_rule not in POLARITY_RULES:
        raise SettingsError(f"polarity_rule must be one of {POLARITY_RULES}.")
    if isinstance(windows, str):
        windows = FIT_PRESETS[windows]["windows"]
    if isinstance(fit, GroupFit):
        return dataclasses.replace(
            fit, fits=[_label_fit(f, windows, unassigned, polarity_rule) for f in fit.fits]
        )
    return _label_fit(fit, windows, unassigned, polarity_rule)
