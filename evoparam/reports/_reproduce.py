"""``evoparam.reproduce``: re-run the recorded processing steps of a saved result and compare
(spec REQ-333)."""

import dataclasses
import inspect
import warnings

import numpy as np

from evoparam.errors import DataError, EvoparamVersionError
from evoparam.version import __version__

__all__ = ["reproduce", "ReproduceReport", "replay_steps"]


@dataclasses.dataclass
class ReproduceReport:
    """Outcome of :func:`reproduce`."""

    identical: bool
    max_rel_diff: float
    steps: list
    differences: list
    environment: dict
    result: object = None

    def to_dict(self):
        return {"identical": self.identical, "max_rel_diff": self.max_rel_diff,
                "steps": self.steps, "differences": self.differences,
                "environment": self.environment}


def _accepts(function, settings):
    params = inspect.signature(function).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return dict(settings)
    return {k: v for k, v in settings.items() if k in params}


def _tuples(settings):
    out = {}
    for key, value in settings.items():
        if isinstance(value, list) and len(value) == 2 and all(
                isinstance(v, (int, float)) for v in value):
            out[key] = tuple(value)
        else:
            out[key] = value
    return out


def _read_step(step, source, data):
    from evoparam.io import read_bids

    if data is not None:
        return data
    settings = dict(step["settings"])
    root = source.get("root")
    if not root:
        raise DataError("The recorded source has no BIDS root; pass data= to reproduce.")
    subject = settings.pop("subject")
    remote = settings.pop("remote", False)
    return read_bids(root, subject, remote=remote, **_tuples(settings))


def _artifact(ts, settings):
    from evoparam.artifact import remove_artifact

    return remove_artifact(ts, **_accepts(remove_artifact, _tuples(settings))).cleaned


def _baseline(ts, settings):
    from evoparam.prep import baseline_correct

    return baseline_correct(ts, **_accepts(baseline_correct, _tuples(settings)))


def _reference(ts, settings):
    from evoparam.prep import rereference

    return rereference(ts, **_accepts(rereference, _tuples(settings)))


def _fit_group(ts, settings):
    from evoparam.models import ResponseModel

    values = {k: v for k, v in _tuples(settings).items()
              if k not in ("package_version", "preset_note")}
    windows = values.get("windows")
    if windows:
        values["windows"] = {k: tuple(v) for k, v in windows.items()}
    preset = values.pop("preset", None)
    model = ResponseModel(preset, **_accepts(ResponseModel.__init__, values))
    return model.fit_group(ts)


REPLAY = {
    "evoparam.artifact.remove_artifact": _artifact,
    "evoparam.prep.baseline_correct": _baseline,
    "evoparam.prep.rereference": _reference,
    "evoparam.models.ResponseModel.fit_group": _fit_group,
}
# Steps that another replayed step redoes (the average is made by fit_group).
IMPLIED = {"evoparam.data.TrialSet.average", "evoparam.data.TrialSet.select"}


def replay_steps(steps, source, data=None):
    """Run the recorded steps in order; returns (result, per-step status)."""

    status = []
    current = None
    for step in steps:
        name = step["function"]
        if name.endswith("read_bids") or name.startswith("evoparam.io.read"):
            current = _read_step(step, source, data)
            status.append({"function": name, "status": "replayed"})
        elif name in REPLAY:
            if current is None:
                current = data
            if current is None:
                raise DataError(f"Step {name} has no input: pass data= to reproduce.")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                current = REPLAY[name](current, step.get("settings", {}))
            status.append({"function": name, "status": "replayed"})
        elif name in IMPLIED:
            status.append({"function": name, "status": "implied"})
        else:
            status.append({"function": name, "status": "not replayable"})
    return current, status


def _compare(a, b, rtol, atol):
    """Largest relative difference of the numeric columns of the two results' tables."""

    ta, tb = a.to_dataframe(), b.to_dataframe()
    differences = []
    if list(ta.columns) != list(tb.columns) or len(ta) != len(tb):
        return np.inf, [f"tables differ in shape: {ta.shape} vs {tb.shape}"]
    worst = 0.0
    for column in ta.columns:
        x, y = ta[column].to_numpy(), tb[column].to_numpy()
        if x.dtype.kind in "fiu" and y.dtype.kind in "fiu":
            x, y = x.astype(float), y.astype(float)
            both_nan = np.isnan(x) & np.isnan(y)
            scale = np.maximum(np.abs(x), np.abs(y))
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.where(both_nan, 0.0, np.abs(x - y) / np.where(scale > 0, scale, 1.0))
            rel = np.nan_to_num(rel, nan=np.inf)
            if rel.size:
                worst = max(worst, float(rel.max()))
            if not np.all(both_nan | np.isclose(x, y, rtol=rtol, atol=atol)):
                differences.append(f"column {column}: max relative difference "
                                   f"{float(rel.max()):.3g}")
        elif not np.array_equal(x.astype(str), y.astype(str)):
            differences.append(f"column {column}: values differ")
            worst = np.inf
    return worst, differences


def reproduce(result_path, *, data=None, rtol=1e-9, atol=0.0, check_environment=True):
    """Reload a saved result, re-run its recorded steps from the recorded source (or
    ``data``) with the recorded settings, and compare.

    Returns
    -------
    ReproduceReport
        ``identical`` (every table value within ``rtol``/``atol``), ``max_rel_diff``,
        per-step status, the differing columns, and the environment differences.

    Raises
    ------
    EvoparamVersionError
        A recorded preset version no longer exists.
    """

    from evoparam import presets
    from evoparam.data.serialize import load_object

    saved = load_object(result_path)
    provenance = saved.provenance
    for step in provenance.steps:
        preset = (step.get("settings") or {}).get("preset")
        note = (step.get("settings") or {}).get("preset_note") or ""
        if preset and preset in presets.PRESETS and " v" in note:
            version = note.split(" v")[1].split(";")[0].split(" ")[0]
            if version != presets.get(preset).version:
                raise EvoparamVersionError(
                    f"The result was made with preset '{preset}' v{version}; this evoparam "
                    f"has v{presets.get(preset).version}.")
    environment = {}
    if check_environment:
        saved_env = provenance.environment or {}
        from evoparam.data.provenance import capture_environment

        now = capture_environment()
        for key in sorted(set(saved_env) | set(now)):
            if saved_env.get(key) != now.get(key) and key not in ("cpu", "platform"):
                environment[key] = {"saved": saved_env.get(key), "now": now.get(key)}
        if provenance.package_version != __version__:
            warnings.warn(f"The result was made with evoparam {provenance.package_version}; "
                          f"this is {__version__}.", UserWarning, stacklevel=2)
    new, status = replay_steps(provenance.steps, provenance.source, data)
    if new is None or not hasattr(new, "to_dataframe"):
        raise DataError("The recorded steps do not produce a result that can be compared.")
    worst, differences = _compare(saved, new, rtol, atol)
    return ReproduceReport(identical=not differences, max_rel_diff=worst, steps=status,
                           differences=differences, environment=environment, result=new)
