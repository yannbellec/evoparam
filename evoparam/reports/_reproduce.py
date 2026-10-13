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
from typing import TYPE_CHECKING

import numpy as np

from evoparam._typing import Any, NDArray, StrPath
from evoparam.data import TrialSet, load_object, save_object
from evoparam.data._jsonutil import loads, to_jsonable
from evoparam.data.provenance import capture_environment
from evoparam.data.serialize import SCHEMA
from evoparam.errors import DataError, EvoparamError, EvoparamVersionError
from evoparam.errors import warn as _warn
from evoparam.utils._numeric import divide
from evoparam.version import __version__

if TYPE_CHECKING:
    from evoparam.data.results import GroupFit

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
    steps: list[Any]
    differences: list[Any]
    environment_differences: dict[str, Any]
    checksums_ok: bool
    settings_differences: list[Any] = dataclasses.field(default_factory=list)
    missing_plugins: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-ready dict (an infinite difference as None)."""
        out = dataclasses.asdict(self)
        out["max_rel_diff"] = None if not math.isfinite(self.max_rel_diff) else self.max_rel_diff
        return out


###################################################################################################
## Reading the container


def _read_settings_json(path: Any) -> Any:
    """``settings.json`` of a container (directory or zip), as written on disk."""
    if os.path.isdir(path):
        with open(os.path.join(path, "settings.json"), encoding="utf-8") as fobj:
            return loads(fobj.read())
    with zipfile.ZipFile(path) as archive:
        return loads(archive.read("settings.json").decode("utf-8"))


def _resolve_path(path: Any) -> Any:
    """Return the container path (``.evoparam`` appended when only that exists)."""
    path = os.fspath(path)
    if not os.path.exists(path) and os.path.exists(path + ".evoparam"):
        path += ".evoparam"
    return path


###################################################################################################
## Replaying the steps


def _callable(function: Any) -> Any:
    """Return the public function (or container method name) of a dotted step name."""
    parts = function.split(".")
    for cut in range(len(parts) - 1, 1, -1):
        try:
            module = import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        obj: Any
        obj = module
        for name in parts[cut:]:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return None


def _accepted(func: Any, settings: Any) -> dict[Any, Any]:
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


def _check_preset(settings: Any) -> None:
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


def _fit_group(trials: Any, settings: Any) -> "GroupFit":
    """Fit ``trials`` again with the recorded group-fit settings."""
    from evoparam.models import ResponseModel
    from evoparam.models._settings import FitSettings

    _check_preset(settings)
    names = set(FitSettings.__dataclass_fields__) - {"preset", "preset_note"}
    kwargs: dict[str, Any] = {
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


def _fit_trials(obj: Any, trials: Any, group: Any, step: Any) -> Any:
    """Replay a fit_trials step; the TrialFit of ``obj``'s channel when ``obj`` is one."""
    from evoparam.trials import fit_trials

    settings = dict(step.get("settings") or {})
    kwargs = _accepted(fit_trials, settings)
    for key in ("by", "covariates", "prestim_band", "prestim_window"):
        if isinstance(kwargs.get(key), list):
            kwargs[key] = tuple(kwargs[key])
    kwargs["random_state"] = step.get("random_state")
    out = fit_trials(trials, group, **kwargs)
    if type(obj).__name__ == "TrialFit":
        return out.get(obj.channel, condition=obj.settings.get("condition") or None)
    return out


def _replay_test(settings: Any, trials: Any, group: Any) -> "GroupFit":
    """Replay the existence test of a group fit and attach it."""
    from evoparam.existence import attach_test, test_response

    kwargs = _accepted(test_response, settings)
    spec = settings.get("rule_spec") or {}
    if settings.get("method") == "threshold" and spec.get("name"):
        kwargs["rule"] = spec["name"]
    for key in ("window", "baseline"):
        if key in kwargs:
            kwargs[key] = tuple(kwargs[key])
    kwargs["by"] = tuple(kwargs.get("by") or ())
    tested: GroupFit = attach_test(group, test_response(trials, **kwargs))
    return tested


_CHAIN = ("evoparam.events.detect_events", "evoparam.io.epoch", "evoparam.trials.fit_trials")


def _replay_chain(
    obj: Any,
    step: Any,
    state: Any,
    chain: dict[str, Any],
    *,
    trials: Any,
    group: Any,
    final_settings: Any,
) -> Any:
    """Replay a step of the chain of REQ-294 that is not a call on the data.

    The steps are the events, the epochs and the trial fits.

    ``chain["recording"]`` keeps the continuous recording that the events were detected in,
    for the epochs. ``final_settings`` (the last step only): a value of the result's
    settings file that differs from the saved result replaces the recorded keyword
    argument of the trial fit, as for a group fit.
    """
    function = step["function"]
    settings = dict(step.get("settings") or {})
    if function == "evoparam.events.detect_events":
        from evoparam.events import detect_events, detect_pulses

        chain["recording"] = state
        return detect_events(state, **_accepted(detect_pulses, settings))
    if function == "evoparam.io.epoch":
        from evoparam.io import epoch

        recording = chain.get("recording")
        if recording is None:
            raise DataError("the epochs need the recording of a detect_events step.")
        return epoch(
            recording.data[0],
            recording.sfreq,
            state,
            settings["tmin"],
            settings["tmax"],
            t0=settings.get("t0", 0.0),
            ch_names=list(recording.ch_names),
            units=settings.get("units", recording.units),
            ch_info=recording.ch_info,
        )
    if final_settings is not None:
        saved = getattr(obj, "settings", None) or {}
        changed = {
            k: v
            for k, v in final_settings.items()
            if k in settings and v != to_jsonable(saved.get(k))
        }
        step = dict(step, settings={**settings, **changed})
    return _fit_trials(obj, trials or state, group, step)


def _replay_bids(settings: Any, source: Any) -> Any:
    """Read the recorded BIDS source again: (TrialSet or None, status, message)."""
    from evoparam.io import read_bids

    root = (source or {}).get("root")
    if not root:
        return None, "not_replayable", "the source has no BIDS root"
    kwargs = _accepted(read_bids, {k: v for k, v in settings.items() if k != "subject"})
    if kwargs.get("events") == "DataFrame":
        return None, "not_replayable", "the events were given as a DataFrame"
    return read_bids(root, settings["subject"], **kwargs), "replayed", ""


def _replay_source(step: Any, source: Any, data: Any) -> Any:
    """Return the data of the first step: (TrialSet or None, status, message)."""
    function = step["function"]
    settings = step.get("settings") or {}
    if data is not None:
        return data, "replayed", "data given by the caller"
    if function.startswith("evoparam.sim."):
        func = _callable(function)
        ts, _ = func(**_accepted(func, settings))
        return ts, "replayed", ""
    if function == "evoparam.io.read_bids":
        return _replay_bids(settings, source)
    path = (source or {}).get("path")
    if function == "evoparam.io.read" and path and "onsets" not in settings:
        from evoparam.io import read

        kwargs = {k: v for k, v in settings.items() if k in ("tmin", "tmax", "picks")}
        return read(path, **kwargs), "replayed", ""
    return None, "not_replayable", ("the source cannot be read again from the record; pass data=")


@dataclasses.dataclass
class _Replay:
    """The state of a replay: the saved object, its steps and what the steps produced.

    ``state`` is the current output; ``trials`` the trials averaged inside the group fit;
    ``group`` the group fit; ``chain`` the continuous TrialSet that events were detected in.
    """

    obj: Any
    steps: list[Any]
    state: Any = None
    trials: Any = None
    group: Any = None
    chain: dict[str, Any] = dataclasses.field(default_factory=dict)


def _replay(obj: Any, final_settings: Any, data: Any) -> Any:
    """Re-run the steps of ``obj.provenance``; returns (new object, step records)."""
    steps = list(obj.provenance.steps)
    records = []
    run = _Replay(obj, steps)
    for i, step in enumerate(steps):
        function = step["function"]
        settings = dict(step.get("settings") or {})
        last = i == len(steps) - 1 or (
            function == _FIT_STEP and all(s["function"] == _TEST_STEP for s in steps[i + 1 :])
        )
        if function == _FIT_STEP and last and final_settings is not None:
            settings = dict(final_settings)
        try:
            if run.state is None:
                run.state, status, message = _replay_source(step, obj.provenance.source, data)
                records.append({"function": function, "status": status, "message": message})
                if run.state is None:
                    return None, records
                if data is not None and not function.startswith(("evoparam.io.", "evoparam.sim.")):
                    # the caller's data stand for the source; this step is replayed below
                    records[-1]["status"] = "skipped"
                else:
                    continue
            records.append(_replay_step(run, i, settings, final_settings if last else None))
            if records[-1]["status"] == "not_replayable":
                return None, records
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
    return run.state, records


def _replay_step(run: _Replay, i: int, settings: Any, final_settings: Any) -> dict[str, Any]:
    """Re-run step ``i`` (after the source) on ``run.state``; return the step record.

    ``final_settings`` (the settings of ``settings.json``) is given for the result's own step.
    """
    steps = run.steps
    function = steps[i]["function"]
    if function == "evoparam.data.TrialSet.average":
        if i + 1 < len(steps) and steps[i + 1]["function"] == _FIT_STEP:
            run.trials = run.state
            return {"function": function, "status": "replayed", "message": "inside the group fit"}
        run.state = run.state.average(**_accepted(run.state.average, settings))
    elif function == _FIT_STEP:
        run.trials = run.trials or run.state
        run.group = _fit_group(
            run.trials if isinstance(run.trials, TrialSet) else run.state, settings
        )
        run.state = run.group
    elif function == _TEST_STEP and run.group is not None:
        run.state = run.group = _replay_test(settings, run.trials, run.group)
    elif function in _CHAIN:
        run.state = _replay_chain(
            run.obj,
            steps[i],
            run.state,
            run.chain,
            trials=run.trials,
            group=run.group,
            final_settings=final_settings,
        )
    elif function.startswith("evoparam.data."):
        method = getattr(run.state, function.rsplit(".", 1)[1])
        run.state = method(**_accepted(method, settings))
    else:
        func = _callable(function)
        if func is None:
            return {
                "function": function,
                "status": "not_replayable",
                "message": "unknown function",
            }
        out = func(run.state, **_accepted(func, settings))
        # the result itself when it is the saved object (an ArtifactResult), else
        # the cleaned data that the next step works on
        run.state = (
            out
            if i == len(steps) - 1 and type(out) is type(run.obj)
            else getattr(out, "cleaned", out)
        )
    return {"function": function, "status": "replayed", "message": ""}


###################################################################################################
## Comparing


def _difference(a: NDArray[Any], b: NDArray[Any]) -> NDArray[Any]:
    """Return ``a - b``, NaN where both are the same infinity (``inf - inf`` not computed)."""
    a, b = np.broadcast_arrays(a, b)
    out = np.full(a.shape, np.nan)
    np.subtract(a, b, out=out, where=~(np.isinf(a) & (a == b)))
    return out


def _rel(a: Any, b: Any) -> tuple[float, NDArray[Any], NDArray[Any]]:
    """Return the largest relative difference, the absolute differences and the both-NaN mask."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    both_nan = np.isnan(a) & np.isnan(b)
    diff = np.abs(_difference(a, b))
    scale = np.maximum(np.abs(a), np.abs(b))
    rel = np.where(scale > 0, divide(diff, scale), 0.0)
    rel = np.where(both_nan, 0.0, rel)
    rel = np.where(np.isnan(rel), np.inf, rel)
    return float(rel.max()) if rel.size else 0.0, diff, both_nan


def _compare(new: Any, old: Any, path: Any, rtol: Any, atol: Any, out: Any) -> Any:
    """Append (path, relative difference or None) of every disagreement.

    Return the largest relative difference among the numbers compared.
    """
    for applies, compare in _COMPARERS:
        if applies(old):
            return compare(new, old, path, rtol, atol, out)
    if new != old and not (
        isinstance(old, float) and isinstance(new, float) and math.isnan(old) and math.isnan(new)
    ):
        return _mismatch(out, path)
    return 0.0


def _mismatch(out: Any, path: Any) -> float:
    """Record a difference that has no relative size (type, shape, keys or value)."""
    out.append((path, None))
    return math.inf


def _compare_dataclass(new: Any, old: Any, path: Any, rtol: Any, atol: Any, out: Any) -> Any:
    """Compare two dataclass instances field by field (volatile fields left out)."""
    if type(new) is not type(old):
        return _mismatch(out, path)
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


def _compare_frame(new: Any, old: Any, path: Any, rtol: Any, atol: Any, out: Any) -> Any:
    """Compare two DataFrames column by column (same columns and length needed)."""
    if (
        type(new).__name__ != "DataFrame"
        or list(new.columns) != list(old.columns)
        or len(new) != len(old)
    ):
        return _mismatch(out, path)
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


def _compare_dict(new: Any, old: Any, path: Any, rtol: Any, atol: Any, out: Any) -> Any:
    """Compare two dicts key by key (same keys needed; volatile keys left out)."""
    if not isinstance(new, dict) or set(new) != set(old):
        return _mismatch(out, path)
    worst = 0.0
    for key in old:
        if key in _VOLATILE:
            continue
        worst = max(worst, _compare(new[key], old[key], f"{path}.{key}", rtol, atol, out))
    return worst


def _compare_sequence(new: Any, old: Any, path: Any, rtol: Any, atol: Any, out: Any) -> Any:
    """Compare two lists or tuples (not of numbers only) item by item."""
    if not isinstance(new, (list, tuple)) or len(new) != len(old):
        return _mismatch(out, path)
    worst = 0.0
    for i, (a, b) in enumerate(zip(new, old, strict=False)):
        worst = max(worst, _compare(a, b, f"{path}[{i}]", rtol, atol, out))
    return worst


def _compare_numbers(new: Any, old: Any, path: Any, rtol: Any, atol: Any, out: Any) -> Any:
    """Compare numbers or arrays within the tolerances; other arrays must be equal."""
    try:
        a = np.asarray(new)
        b = np.asarray(old)
    except Exception:  # noqa: BLE001  (reported as a difference)
        return _mismatch(out, path)
    if a.shape != b.shape:
        return _mismatch(out, path)
    if b.dtype.kind in "fiucb" and a.dtype.kind in "fiucb":
        rel, diff, both_nan = _rel(a, b)
        ok = np.all(
            both_nan
            | (diff <= atol + rtol * np.maximum(np.abs(a.astype(float)), np.abs(b.astype(float))))
        )
        if not ok:
            out.append((path, rel))
        return rel
    if not np.array_equal(a, b):
        return _mismatch(out, path)
    return 0.0


# (applies to the saved value, comparison), tried in order by :func:`_compare`.
_COMPARERS: tuple[tuple[Any, Any], ...] = (
    (lambda old: dataclasses.is_dataclass(old) and not isinstance(old, type), _compare_dataclass),
    (lambda old: type(old).__name__ == "DataFrame", _compare_frame),
    (lambda old: isinstance(old, dict), _compare_dict),
    (
        lambda old: (
            isinstance(old, (list, tuple))
            and not (
                old and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in old)
            )
        ),
        _compare_sequence,
    ),
    (
        lambda old: (
            isinstance(old, (float, int, np.number, np.ndarray, list, tuple))
            and not isinstance(old, bool)
        ),
        _compare_numbers,
    ),
)


def _missing_plugins(recorded: Any) -> dict[Any, Any]:
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


def _environment_differences(recorded: Any) -> dict[Any, Any]:
    """Return {key: [recorded, current]} of the environment entries that differ."""
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


def _settings_on_disk(path: Any, saved: Any) -> tuple[Any, list[Any]]:
    """Return the settings of ``settings.json`` (None without one) and the keys that differ.

    The differing keys are those whose value differs from the result's recorded settings.
    """
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
    return final_settings, settings_differences


def _environment_report(saved: Any) -> dict[Any, Any]:
    """Differences between the recorded and current environments, with a warning if any."""
    env_diff = _environment_differences(saved.provenance.environment or {})
    if saved.provenance.package_version != __version__:
        env_diff["package_version"] = [saved.provenance.package_version, __version__]
    if env_diff:
        _warn(
            "evoparam.reproduce: the software environment differs from the "
            f"recorded one ({', '.join(sorted(env_diff))}).",
            UserWarning,
            stacklevel=3,
        )
    return env_diff


def _name_settings_step(records: Any, saved: Any, settings_differences: Any) -> None:
    """Name the differing ``settings.json`` keys in the record of the step that uses them.

    That is the group fit for a GroupFit (its existence test follows it), else the step
    that made the result (the last one).
    """
    named = (
        _FIT_STEP
        if any(record["function"] == _FIT_STEP for record in records)
        and type(saved).__name__ == "GroupFit"
        else records[-1]["function"]
    )
    for record in records:
        if record["function"] == named:
            record["message"] = "settings.json differs from the recorded settings: " + ", ".join(
                settings_differences
            )


def reproduce(
    result_path: StrPath,
    *,
    data: "TrialSet | None" = None,
    rtol: float = 1e-9,
    atol: float = 0.0,
    check_environment: bool = True,
) -> "ReproduceReport":
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
    final_settings, settings_differences = _settings_on_disk(path, saved)
    env_diff = _environment_report(saved) if check_environment else {}
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
    if settings_differences and records:
        _name_settings_step(records, saved, settings_differences)
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


def migrate(path: Any) -> dict[str, Any]:
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
