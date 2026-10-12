# SPDX-License-Identifier: Apache-2.0
"""Plugins through ``importlib.metadata`` entry points (spec §10.8, REQ-328).

Third-party distributions extend three registries::

    [project.entry-points."evoparam.shapes"]     # a ComponentShape (class or instance)
    [project.entry-points."evoparam.presets"]    # a Preset instance
    [project.entry-points."evoparam.readers"]    # callable(path, **kw) -> TrialSet, with
                                                 # an attribute ``extensions = (".tev", ...)``

Loading is lazy: :class:`LazyRegistry` (``SHAPES``, ``PRESETS``, the readers of
``evoparam.io.read``) loads its group on first access. ``EVOPARAM_DISABLE_PLUGINS=1`` turns
plugins off. Each registry checks its objects (the shape checks of ``register_shape``, the
``Preset`` type, the reader signature); a name already taken by a built-in or by another
plugin raises :class:`~evoparam.errors.SettingsError` naming both. Every plugin object
handed out by a registry is recorded in :data:`USED` (``{"group:name": "dist==version"}``),
which goes into ``provenance.environment["plugins"]`` so that ``reproduce`` can report a
missing plugin.

This module imports only the standard library and :mod:`evoparam.errors`, so every module
may import it.
"""

import os
from importlib import metadata

from evoparam.errors import SettingsError

__all__ = ["GROUPS", "LazyRegistry", "load_group", "loaded", "used", "enabled", "USED", "SKIP"]

GROUPS = ("evoparam.shapes", "evoparam.presets", "evoparam.readers")

# "group:name" -> "dist==version" of every plugin object handed out by a registry
USED = {}
# group -> {name: "dist==version"} of the loaded plugins
_LOADED = {}

# returned by a registry's check for an object that belongs to another registry of the same
# group (a background in the group evoparam.shapes)
SKIP = object()


def enabled():
    """Return False when ``EVOPARAM_DISABLE_PLUGINS`` is set to a true value."""
    return os.environ.get("EVOPARAM_DISABLE_PLUGINS", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    )


def _entry_points(group):
    return list(metadata.entry_points(group=group))


def _origin(entry):
    dist = getattr(entry, "dist", None)
    if dist is None:
        return f"unknown=={entry.value}"
    return f"{dist.metadata['Name']}=={dist.version}"


def load_group(group, builtin, check):
    """``{name: (object, "dist==version")}`` of the plugins of ``group``.

    ``builtin`` holds the built-in names; ``check(name, obj, origin)`` validates an object
    and returns what the registry stores (it raises SettingsError on a bad object), or
    :data:`SKIP` for an object of another registry of the group.
    """
    if group not in GROUPS:
        raise SettingsError(f"unknown plugin group {group!r}; groups: {GROUPS}.")
    out = {}
    if not enabled():
        _LOADED[group] = {}
        return out
    for entry in _entry_points(group):
        origin = _origin(entry)
        name = entry.name
        if name in builtin:
            raise SettingsError(
                f"plugin {name!r} of {origin} ({group}) has the name of a "
                f"built-in of evoparam; rename it in its entry point."
            )
        if name in out:
            raise SettingsError(
                f"plugin name {name!r} ({group}) is used by both {out[name][1]} "
                f"and {origin}; uninstall one of them."
            )
        try:
            obj = entry.load()
        except Exception as err:  # a broken plugin is named, never silently skipped
            raise SettingsError(
                f"plugin {name!r} of {origin} ({group}) cannot be loaded: "
                f"{type(err).__name__}: {err}"
            ) from err
        out[name] = (check(name, obj, origin), origin)
    _LOADED.setdefault(group, {}).update({name: origin for name, (_, origin) in out.items()})
    out = {name: item for name, item in out.items() if item[0] is not SKIP}
    return out


def loaded():
    """``{group: {name: "dist==version"}}`` of the groups loaded so far."""
    return {group: dict(names) for group, names in _LOADED.items()}


def used():
    """``{"group:name": "dist==version"}`` of the plugin objects handed out so far."""
    return dict(USED)


class LazyRegistry(dict):
    """A dict that adds the plugins of ``group`` on first access.

    ``check(name, obj, origin)`` validates each plugin object; the built-in names are the keys
    present when the registry is created, plus ``builtin``. Objects registered later by the user
    (``register_shape``, ``presets.register``) are kept as they are.
    """

    def __init__(self, group, items, check, *, builtin=()):
        super().__init__(items)
        self._group, self._check = group, check
        self._builtin = frozenset(items) | frozenset(builtin)
        self._plugins = None

    @property
    def builtin(self):
        return self._builtin

    def _load(self):
        if self._plugins is None:
            self._plugins = {}  # set first: a check may read the registry
            found = load_group(self._group, self._builtin, self._check)
            for name, (obj, origin) in found.items():
                super().__setitem__(name, obj)
                self._plugins[name] = origin

    def plugins(self):
        """``{name: "dist==version"}`` of the plugins of this registry."""
        self._load()
        return dict(self._plugins)

    def reload(self):
        """Forget the loaded plugins and load the entry points again (tests, new installs)."""
        for name in self._plugins or {}:
            super().pop(name, None)
        self._plugins = None
        self._load()

    def _mark(self, key):
        if self._plugins and key in self._plugins:
            USED[f"{self._group}:{key}"] = self._plugins[key]

    def _builtin_entry(self, key):
        # a built-in name is answered without loading the plugins: a plugin module may use
        # the built-ins while it is being imported (a preset derived from "ccep")
        try:
            return key in self._builtin and super().__contains__(key)
        except TypeError:  # unhashable key
            return False

    def __getitem__(self, key):
        if not self._builtin_entry(key):
            self._load()
        value = super().__getitem__(key)
        self._mark(key)
        return value

    def get(self, key, default=None):
        if self._builtin_entry(key):
            return super().__getitem__(key)
        self._load()
        if super().__contains__(key):
            return self[key]
        return default

    def __contains__(self, key):
        if self._builtin_entry(key):
            return True
        self._load()
        return super().__contains__(key)

    def __iter__(self):
        self._load()
        return super().__iter__()

    def __len__(self):
        self._load()
        return super().__len__()

    def keys(self):
        self._load()
        return super().keys()

    def values(self):
        self._load()
        return super().values()

    def items(self):
        self._load()
        return super().items()

    def __repr__(self):
        self._load()
        return super().__repr__()
