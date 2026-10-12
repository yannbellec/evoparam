# SPDX-License-Identifier: Apache-2.0
"""Regenerate a saved result from its provenance and compare (spec §9.12).

REQ-333, REQ-334, REQ-335.

:func:`reproduce` reloads a ``.evoparam`` container, re-runs the recorded
``provenance.steps`` in order from the recorded source (or from ``data``), with the
recorded settings (the result's own step with the settings of ``settings.json``), and
compares every array, number and table of the new result with the saved one.
:func:`migrate` rewrites a container under the current schema.
"""

import dataclasses
import inspect
import math
import os
import re
import tempfile
import zipfile
from importlib import import_module

import numpy as np

from evoparam.data import TrialSet, load_object, save_object
from evoparam.data._jsonutil import loads, to_jsonable
from evoparam.data.provenance import capture_environment
from evoparam.data.serialize import SCHEMA
from evoparam.errors import DataError, EvoparamError, EvoparamVersionError
from evoparam.errors import warn as _warn
from evoparam.version import __version__

__all__ = ["ReproduceReport", "reproduce", "migrate"]

_FIT_STEP = "evoparam.models.ResponseModel.fit_group"
_TEST_STEP = "evoparam.existence.test_response"
_NOT_FIT_KWARGS = (
    "preset",
    "preset_note",
    "package_version",
    "_overrides",
    "_changed_from_preset",
)
# Fields that change between two runs of the same computation.
_VOLATILE = {"provenance", "duration_s"}


@dataclasses.dataclass
class ReproduceReport:
    """Outcome of :func:`reproduce`.

    ``identical`` is True when every compared value agrees within the tolerances;
    ``max_rel_diff`` is the largest relative difference found; ``steps`` holds, per
    recorded step, ``function``, ``status`` (``replayed``, ``skipped``, ``not_replayable``,
    ``failed``) and ``message``; ``differences`` lists the fields that disagree;
    ``environment_differences`` maps each environment key that changed to
    ``[recorded, current]``; ``checksums_ok`` tells whether the container's files match
    its manifest; ``missing_plugins`` maps each plugin the result used (recorded in
    ``provenance.environment["plugins"]``, spec §10.8) that is not available now to
    ``{"recorded": "dist==version", "installed": version or None, "message"}``.
    """

    path: str
    identical: bool
    max_rel_diff: float
    steps: list
    differences: list
    environment_differences: dict
    checksums_ok: bool
    settings_differences: list = dataclasses.field(default_factory=list)
    missing_plugins: dict = dataclasses.field(default_factory=dict)

    def to_dict(self):
        out = dataclasses.asdict(self)
        out["max_rel_diff"] = None if not math.isfinite(self.max_rel_diff) else self.max_rel_diff
        return out


###################################################################################################
## Reading the container


def _read_settings_json(path):
    """``settings.json`` of a container (directory or zip), as written on disk."""
    if os.path.isdir(path):
        with open(os.path.join(path, "settings.json"), encoding="utf-8") as fobj:
            return loads(fobj.read())
    with zipfile.ZipFile(path) as archive:
        return loads(archive.read("settings.json").decode("utf-8"))


def _resolve_path(path):
    path = os.fspath(path)
    if not os.path.exists(path) and os.path.exists(path + ".evoparam"):
        path += ".evoparam"
    return path


###################################################################################################
## Replaying the steps


def _callable(function):
    """Return the public function (or container method name) of a dotted step name."""
    parts = function.split(".")
    for cut in range(len(parts) - 1, 1, -1):
        try:
            module = import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        obj = module
        for name in parts[cut:]:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return None


def _accepted(func, settings):
    """Return the settings that ``func`` accepts as keyword arguments."""
    params = inspect.signature(func).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return dict(settings)
    out = {}
    for key, raw_value in settings.items():
        value = raw_value
        if key in params:
            p = params[key]
            if isinstance(value, list) and isinstance(p.default, tuple):
                value = tuple(value)
            out[key] = value
    return out


def _check_preset(settings):
    """EvoparamVersionError when the recorded preset version no longer exists."""
    from evoparam import presets

    note = settings.get("preset_note") or ""
    match = re.match(r"preset '([^']+)' v([^\s;(]+)", note)
    if not match:
        return
    name, version = match.group(1), match.group(2)
    if name not in presets.PRESETS:
        raise EvoparamVersionError(
            f"The recorded preset {name!r} (v{version}) does not exist in this evoparam any more."
        )
    current = presets.get(name).version
    if current != version:
        raise EvoparamVersionError(
            f"The result was made with preset {name!r} v{version}; this evoparam has v{current}."
        )


def _fit_group(trials, settings):
    from evoparam.models import ResponseModel
    from evoparam.models._settings import FitSettings

    _check_preset(settings)
    names = set(FitSettings.__dataclass_fields__) - {"preset", "preset_note"}
    kwargs = {
        k: (tuple(v) if isinstance(v, list) and k in ("fit_window", "baseline", "shapes") else v)
        for k, v in settings.items()
        if k in names
    }
    if isinstance(kwargs.get("windows"), dict):
        kwargs["windows"] = {k: tuple(v) for k, v in kwargs["windows"].items()}
    accepted = _accepted(ResponseModel, kwargs)
    preset = settings.get("preset")
    model = ResponseModel(preset.get("name") if isinstance(preset, dict) else preset, **accepted)
    # every setting is passed again; the regenerated result records the keys the user
    # had set (REQ-331 ``_overrides``), not all of them
    if settings.get("_overrides") is not None:
        model._overrides_record = list(settings["_overrides"])
    # Settings that the constructor does not take (e.g. freq_min) are read by
    # ResponseModel.settings() from the model's attributes.
    for key in set(kwargs) - set(accepted):
        setattr(model, key, kwargs[key])
    return model.fit_group(trials)


def _replay_source(step, source, data):
    function = step["function"]
    settings = step.get("settings") or {}
    if data is not None:
        return data, "replayed", "data given by the caller"
    if function.startswith("evoparam.sim."):
        func = _callable(function)
        ts, _ = func(**_accepted(func, settings))
        return ts, "replayed", ""
    if function == "evoparam.io.read_bids":
        from evoparam.io import read_bids

        root = (source or {}).get("root")
        if not root:
            return None, "not_replayable", "the source has no BIDS root"
        kwargs = _accepted(read_bids, {k: v for k, v in settings.items() if k != "subject"})
        if kwargs.get("events") == "DataFrame":
            return None, "not_replayable", "the events were given as a DataFrame"
        return read_bids(root, settings["subject"], **kwargs), "replayed", ""
    path = (source or {}).get("path")
    if function == "evoparam.io.read" and path and "onsets" not in settings:
        from evoparam.io import read

        kwargs = {k: v for k, v in settings.items() if k in ("tmin", "tmax", "picks")}
        return read(path, **kwargs), "replayed", ""
    return None, "not_replayable", ("the source cannot be read again from the record; pass data=")


def _replay(obj, final_settings, data):
    """Re-run the steps of ``obj.provenance``; returns (new object, step records)."""
    steps = list(obj.provenance.steps)
    records = []
    state = None
    trials_before_average = None
    group = None
    for i, step in enumerate(steps):
        function = step["function"]
        settings = dict(step.get("settings") or {})
        last = i == len(steps) - 1 or (
            function == _FIT_STEP and all(s["function"] == _TEST_STEP for s in steps[i + 1 :])
        )
        if function == _FIT_STEP and last and final_settings is not None:
            settings = dict(final_settings)
        try:
            if state is None:
                state, status, message = _replay_source(step, obj.provenance.source, data)
                records.append({"function": function, "status": status, "message": message})
                if state is None:
                    return None, records
                if data is not None and not function.startswith(("evoparam.io.", "evoparam.sim.")):
                    # the caller's data stand for the source; this step is replayed below
                    records[-1]["status"] = "skipped"
                else:
                    continue
            if function == "evoparam.data.TrialSet.average":
                if i + 1 < len(steps) and steps[i + 1]["function"] == _FIT_STEP:
                    trials_before_average = state
                    records.append(
                        {
                            "function": function,
                            "status": "replayed",
                            "message": "inside the group fit",
                        }
                    )
                    continue
                state = state.average(**_accepted(state.average, settings))
            elif function == _FIT_STEP:
                trials_before_average = trials_before_average or state
                group = _fit_group(
                    trials_before_average
                    if isinstance(trials_before_average, TrialSet)
                    else state,
                    settings,
                )
                state = group
            elif function == _TEST_STEP and group is not None:
                from evoparam.existence import attach_test, test_response

                kwargs = _accepted(test_response, settings)
                spec = settings.get("rule_spec") or {}
                if settings.get("method") == "threshold" and spec.get("name"):
                    kwargs["rule"] = spec["name"]
                for key in ("window", "baseline"):
                    if key in kwargs:
                        kwargs[key] = tuple(kwargs[key])
                kwargs["by"] = tuple(kwargs.get("by") or ())
                test = test_response(trials_before_average, **kwargs)
                state = group = attach_test(group, test)
            elif function.startswith("evoparam.data."):
                method = getattr(state, function.rsplit(".", 1)[1])
                state = method(**_accepted(method, settings))
            else:
                func = _callable(function)
                if func is None:
                    records.append(
                        {
                            "function": function,
                            "status": "not_replayable",
                            "message": "unknown function",
                        }
                    )
                    return None, records
                out = func(state, **_accepted(func, settings))
                # the result itself when it is the saved object (an ArtifactResult), else
                # the cleaned data that the next step works on
                state = (
                    out
                    if i == len(steps) - 1 and type(out) is type(obj)
                    else getattr(out, "cleaned", out)
                )
            records.append({"function": function, "status": "replayed", "message": ""})
        except EvoparamVersionError:
            raise
        except Exception as exc:  # noqa: BLE001  (reported with status failed, never hidden)
            records.append(
                {
                    "function": function,
                    "status": "failed",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
            return None, records
    return state, records


###################################################################################################
## Comparing


def _rel(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    both_nan = np.isnan(a) & np.isnan(b)
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = np.abs(a - b)
        scale = np.maximum(np.abs(a), np.abs(b))
        rel = np.where(scale > 0, diff / scale, 0.0)
    rel = np.where(both_nan, 0.0, rel)
    rel = np.where(np.isnan(rel), np.inf, rel)
    return float(rel.max()) if rel.size else 0.0, diff, both_nan


def _compare(new, old, path, rtol, atol, out):
    """Append (path, relative difference or None) of every disagreement.

    Return the largest relative difference among the numbers compared.
    """
    if dataclasses.is_dataclass(old) and not isinstance(old, type):
        if type(new) is not type(old):
            out.append((path, None))
            return math.inf
        worst = 0.0
        for fld in dataclasses.fields(old):
            if fld.name in _VOLATILE:
                continue
            worst = max(
                worst,
                _compare(
                    getattr(new, fld.name),
                    getattr(old, fld.name),
                    f"{path}.{fld.name}" if path else fld.name,
                    rtol,
                    atol,
                    out,
                ),
            )
        return worst
    if type(old).__name__ == "DataFrame":
        if (
            type(new).__name__ != "DataFrame"
            or list(new.columns) != list(old.columns)
            or len(new) != len(old)
        ):
            out.append((path, None))
            return math.inf
        worst = 0.0
        for column in old.columns:
            worst = max(
                worst,
                _compare(
                    new[column].to_numpy(),
                    old[column].to_numpy(),
                    f"{path}[{column}]",
                    rtol,
                    atol,
                    out,
                ),
            )
        return worst
    if isinstance(old, dict):
        if not isinstance(new, dict) or set(new) != set(old):
            out.append((path, None))
            return math.inf
        worst = 0.0
        for key in old:
            if key in _VOLATILE:
                continue
            worst = max(worst, _compare(new[key], old[key], f"{path}.{key}", rtol, atol, out))
        return worst
    if isinstance(old, (list, tuple)) and not (
        old and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in old)
    ):
        if not isinstance(new, (list, tuple)) or len(new) != len(old):
            out.append((path, None))
            return math.inf
        worst = 0.0
        for i, (a, b) in enumerate(zip(new, old, strict=False)):
            worst = max(worst, _compare(a, b, f"{path}[{i}]", rtol, atol, out))
        return worst
    numeric_old = isinstance(
        old, (float, int, np.number, np.ndarray, list, tuple)
    ) and not isinstance(old, bool)
    if numeric_old:
        try:
            a = np.asarray(new)
            b = np.asarray(old)
        except Exception:  # noqa: BLE001 (reported as a difference)
            out.append((path, None))
            return math.inf
        if a.shape != b.shape:
            out.append((path, None))
            return math.inf
        if b.dtype.kind in "fiucb" and a.dtype.kind in "fiucb":
            rel, diff, both_nan = _rel(a, b)
            ok = np.all(
                both_nan
                | (
                    diff
                    <= atol + rtol * np.maximum(np.abs(a.astype(float)), np.abs(b.astype(float)))
                )
            )
            if not ok:
                out.append((path, rel))
            return rel
        if not np.array_equal(a, b):
            out.append((path, None))
            return math.inf
        return 0.0
    if new != old and not (
        isinstance(old, float) and isinstance(new, float) and math.isnan(old) and math.isnan(new)
    ):
        out.append((path, None))
        return math.inf
    return 0.0


def _missing_plugins(recorded):
    """Plugins a result used that are not installed now, or not at the recorded version."""
    from importlib import metadata

    from evoparam import presets
    from evoparam.io.readers import READERS
    from evoparam.shapes import BACKGROUNDS, SHAPES

    out = {}
    available = set()
    for registry in (SHAPES, BACKGROUNDS, presets.PRESETS, READERS):
        try:
            available |= set(registry)
        except EvoparamError as err:  # a broken plugin set: report it, go on
            out["<plugins>"] = {"recorded": None, "installed": None, "message": str(err)}
    for name, origin in sorted((recorded or {}).items()):
        dist, _, version = str(origin).partition("==")
        try:
            installed = metadata.version(dist)
        except metadata.PackageNotFoundError:
            installed = None
        if installed is None or name not in available:
            out[name] = {
                "recorded": origin,
                "installed": installed,
                "message": f"plugin {name!r} ({origin}) is not installed; "
                f"pip install '{dist}=={version}'",
            }
        elif version and installed != version:
            out[name] = {
                "recorded": origin,
                "installed": installed,
                "message": f"plugin {name!r}: {dist} {installed} is installed, the "
                f"result was made with {version}",
            }
    return out


def _environment_differences(recorded):
    current = capture_environment()
    out = {}
    for key in sorted(set(recorded) | set(current)):
        if key in ("optional", "plugins"):  # plugins: reported by _missing_plugins
            continue
        if recorded.get(key) != current.get(key):
            out[key] = [recorded.get(key), current.get(key)]
    rec_opt, cur_opt = recorded.get("optional") or {}, current.get("optional") or {}
    for key in sorted(set(rec_opt) | set(cur_opt)):
        if rec_opt.get(key) != cur_opt.get(key):
            out[f"optional.{key}"] = [rec_opt.get(key), cur_opt.get(key)]
    return out


###################################################################################################
## Public functions


def reproduce(result_path, *, data=None, rtol=1e-9, atol=0.0, check_environment=True):
    """Regenerate a saved result from its provenance and compare it with the saved one.

    Parameters
    ----------
    result_path : str or path-like
        ``.evoparam`` container (directory or zip) of a result.
    data : TrialSet, optional
        The input data, when the recorded source cannot be read again (arrays given to
        ``from_array``, files no longer at their path).
    rtol, atol : float
        Tolerances of the comparison of every number.
    check_environment : bool
        Report (and warn about) the differences between the recorded software environment
        and the current one.

    Returns
    -------
    ReproduceReport

    Raises
    ------
    EvoparamVersionError
        A recorded step used a preset version that no longer exists (the message names the
        plugin to install when the preset came from a plugin that is missing).
    """
    path = _resolve_path(result_path)
    try:
        saved = load_object(path, verify=True)
        checksums_ok = True
    except DataError as err:
        if "checksum" not in str(err) and "missing" not in str(err):
            raise
        saved = load_object(path, verify=False)
        checksums_ok = False
    final_settings = None
    settings_differences = []
    try:
        on_disk = _read_settings_json(path)
    except (OSError, KeyError, ValueError):
        on_disk = None
    if isinstance(on_disk, dict) and isinstance(getattr(saved, "settings", None), dict):
        final_settings = on_disk
        settings_differences = sorted(
            key
            for key in set(on_disk) | set(saved.settings)
            if key != "package_version"
            and on_disk.get(key) != to_jsonable(saved.settings.get(key))
        )
    env_diff = {}
    if check_environment:
        env_diff = _environment_differences(saved.provenance.environment or {})
        if saved.provenance.package_version != __version__:
            env_diff["package_version"] = [saved.provenance.package_version, __version__]
        if env_diff:
            _warn(
                "evoparam.reproduce: the software environment differs from the "
                f"recorded one ({', '.join(sorted(env_diff))}).",
                UserWarning,
                stacklevel=2,
            )
    missing_plugins = _missing_plugins((saved.provenance.environment or {}).get("plugins"))
    if missing_plugins:
        _warn(
            "evoparam.reproduce: "
            + "; ".join(item["message"] for item in missing_plugins.values()),
            UserWarning,
            stacklevel=2,
        )
    try:
        new, records = _replay(saved, final_settings, data)
    except EvoparamVersionError as err:
        if not missing_plugins:
            raise
        # the preset came from a plugin that is not installed: say which one
        raise EvoparamVersionError(
            f"{err} " + "; ".join(item["message"] for item in missing_plugins.values())
        ) from err
    if settings_differences:
        for record in records:
            if record["function"] == _FIT_STEP:
                record["message"] = (
                    "settings.json differs from the recorded settings: "
                    + ", ".join(settings_differences)
                )
    differences = []
    if new is None:
        worst = math.inf
        differences.append(("<not regenerated>", None))
    else:
        worst = _compare(new, saved, "", rtol, atol, differences)
    identical = new is not None and not differences
    return ReproduceReport(
        path=os.path.abspath(path),
        identical=identical,
        max_rel_diff=worst,
        steps=records,
        differences=[{"field": p, "rel_diff": d} for p, d in differences],
        environment_differences=env_diff,
        checksums_ok=checksums_ok,
        settings_differences=settings_differences,
        missing_plugins=missing_plugins,
    )


def migrate(path):
    """Rewrite a container in place under the current schema (``evoparam-result/<major>``).

    The migrations applied are recorded in its provenance (REQ-334).

    Returns
    -------
    dict
        ``{"path", "from_schema", "to_schema", "migrated"}``.
    """
    path = _resolve_path(path)
    if os.path.isdir(path):
        with open(os.path.join(path, "manifest.json"), encoding="utf-8") as fobj:
            manifest = loads(fobj.read())
    else:
        with zipfile.ZipFile(path) as archive:
            manifest = loads(archive.read("manifest.json").decode("utf-8"))
    obj = load_object(path)
    from_schema = manifest.get("schema")
    if from_schema == SCHEMA:
        return {"path": path, "from_schema": from_schema, "to_schema": SCHEMA, "migrated": False}
    parent = os.path.dirname(os.path.abspath(path))
    with tempfile.TemporaryDirectory(prefix=".evoparam-migrate-", dir=parent) as tmp:
        # Write and re-read the new container next to the old one, then swap them, so that
        # an interrupted migration leaves the original in place.
        target = os.path.join(tmp, "new" + os.path.splitext(path)[1])
        written = save_object(obj, target)
        load_object(written)
        os.replace(path, os.path.join(tmp, "old"))
        os.replace(written, path)
    return {"path": path, "from_schema": from_schema, "to_schema": SCHEMA, "migrated": True}
