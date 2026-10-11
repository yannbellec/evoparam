# SPDX-License-Identifier: Apache-2.0
"""Named latency windows that label fitted components (spec §4.4, §4.5, §5.5.7).

:class:`ComponentWindows` replaces the inherited frequency-band object: the windows name
components in time (seconds), with an optional polarity, and are read by
:func:`label_components`. A ``ComponentWindows`` is a read-only mapping
``name -> (start, end, polarity)``, so it can be passed wherever a plain dictionary of
windows is accepted (``FitSettings.windows``, ``ResponseModel(windows=...)``).
"""

from collections.abc import Mapping

from evoparam.errors import SettingsError

from .labels import label_components

__all__ = ["ComponentWindows", "label_components"]

_POLARITIES = {-1, 0, 1}


def _check(name, definition):
    if not isinstance(name, str) or not name:
        raise SettingsError(f"window name must be a non-empty string, got {name!r}.")
    if len(definition) == 2:
        start, end = definition
        polarity = 0
    elif len(definition) == 3:
        start, end, polarity = definition
    else:
        raise SettingsError(f"window {name!r} must be (start, end) or (start, end, polarity) "
                            f"in seconds.")
    start, end = float(start), float(end)
    if not start < end:
        raise SettingsError(f"window {name!r}: start ({start} s) must be before end ({end} s).")
    if polarity not in _POLARITIES:
        raise SettingsError(f"window {name!r}: polarity must be -1, +1 or 0 (either), "
                            f"got {polarity!r}.")
    return start, end, int(polarity)


class ComponentWindows(Mapping):
    """Latency windows that name components.

    Parameters
    ----------
    windows : mapping, optional
        ``name -> (start, end)`` or ``name -> (start, end, polarity)``; times in seconds,
        polarity ``-1`` (negative), ``+1`` (positive) or ``0`` (either, the default).

    Examples
    --------
    >>> windows = ComponentWindows({"N1": (0.009, 0.050, -1), "N2": (0.050, 0.300, -1)})
    >>> windows["N1"]
    (0.009, 0.05, -1)
    >>> windows.labels
    ['N1', 'N2']
    """

    def __init__(self, windows=None):
        self._windows = {}
        for name, definition in dict(windows or {}).items():
            self._windows[name] = _check(name, definition)

    @classmethod
    def from_preset(cls, name):
        """The labelling windows of a preset (spec §6.1), for example ``"ccep"``."""

        from ._settings import FIT_PRESETS

        if name not in FIT_PRESETS:
            raise SettingsError(f"unknown preset {name!r}.")
        return cls(FIT_PRESETS[name]["windows"] or {})

    def __getitem__(self, name):
        try:
            return self._windows[name]
        except KeyError:
            raise KeyError(f"no window named {name!r}; windows: {self.labels}") from None

    def __iter__(self):
        return iter(self._windows)

    def __len__(self):
        return len(self._windows)

    def __repr__(self):
        inside = ", ".join(f"{k}: {v}" for k, v in self._windows.items())
        return f"ComponentWindows({{{inside}}})"

    @property
    def labels(self):
        """Window names, in insertion order."""

        return list(self._windows)

    @property
    def definitions(self):
        """``(start, end, polarity)`` of every window, in the order of :attr:`labels`."""

        return list(self._windows.values())

    def add(self, name, definition):
        """A copy with one more window."""

        if name in self._windows:
            raise SettingsError(f"window {name!r} already exists.")
        return ComponentWindows({**self._windows, name: definition})

    def remove(self, name):
        """A copy without the window ``name``."""

        self[name]
        return ComponentWindows({k: v for k, v in self._windows.items() if k != name})

    def to_dict(self):
        """Plain dictionary ``name -> [start, end, polarity]`` (JSON-serialisable)."""

        return {k: list(v) for k, v in self._windows.items()}
