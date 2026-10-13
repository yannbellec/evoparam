# SPDX-License-Identifier: Apache-2.0
"""The seven benchmark tasks, their versioned schemas and their files (spec §8.1, §8.4).

A task **input** is a JSON descriptor ``<cell>.json`` next to an ``.npz`` file of arrays
(``data``: trials x channels x times; ``evoked``: channels x times for T3; ``average``:
conditions-free channel averages for T6). The descriptor holds the sampling rate, the
first time, the channel names, the hashes of the arrays and the task-specific fields
(T4 ``artifact_window``, T5 ``targets``, T6 ``definitions``).

A task **output** is a TSV file ``<cell>.tsv`` with the fixed columns of its schema
(``site, channel, responsive, p_value, ...``), ``n/a`` for a missing value, the required
columns first and the optional ones after them in the order of the schema; for T4 (cleaned
epochs, which do not fit a table) an ``.npz`` file holding ``data``.

Every schema is versioned as ``benchmark-task/<id>/1`` (``schemas/<id>.json``, JSON
Schema 2020-12) and every file is validated when it is read; a file that does not
validate raises :class:`BenchmarkValidationError` naming the first violation (REQ-351).
"""

import copy
import json
import math
import os
import re
from dataclasses import dataclass, field
from importlib import resources
from typing import TYPE_CHECKING

import numpy as np

from evoparam._typing import Any
from evoparam.errors import DataError
from evoparam.utils._optional import require

from ._canon import array_sha256, canonical_bytes, sha256_bytes

if TYPE_CHECKING:
    from evoparam.data.containers import Evoked, TrialSet

__all__ = [
    "TASKS",
    "TASK_IDS",
    "BenchmarkValidationError",
    "TaskInput",
    "load_schema",
    "schema_id",
    "validate",
    "output_columns",
    "output_format",
    "write_input",
    "read_input",
    "write_output",
    "read_output",
    "output_path",
]

TASKS = {
    "T1": "response detection",
    "T2": "early component latency and amplitude",
    "T3": "component decomposition",
    "T4": "artifact removal",
    "T5": "single-trial estimation",
    "T6": "ERP scoring",
    "T7": "time-frequency response",
}
TASK_IDS = tuple(TASKS)
SCHEMA_VERSION = 1
_CACHE: dict[Any, Any]
_CACHE = {}


class BenchmarkValidationError(DataError):
    """A task file does not follow its schema; the message names the first violation."""


def schema_id(task: Any) -> str:
    """``benchmark-task/<id>/1``."""
    _check_task(task)
    return f"benchmark-task/{task}/{SCHEMA_VERSION}"


def _check_task(task: Any) -> None:
    """Raise DataError when ``task`` is not a benchmark task."""
    if task not in TASKS:
        raise DataError(f"Unknown benchmark task {task!r}; the tasks are {list(TASKS)}.")


def load_schema(task: Any) -> Any:
    """Return the JSON Schema of a task (a deep copy of ``schemas/<task>.json``)."""
    _check_task(task)
    if task not in _CACHE:
        text = (
            resources.files("evoparam.benchmark")
            .joinpath("schemas")
            .joinpath(f"{task}.json")
            .read_text("utf-8")
        )
        _CACHE[task] = json.loads(text)
    return copy.deepcopy(_CACHE[task])


def output_format(task: Any) -> Any:
    """``"tsv"``, or ``"npz"`` for T4."""
    return load_schema(task)["x-output-format"]


def output_columns(task: Any) -> tuple[list[Any], list[Any]]:
    """Return the (required, optional) output columns of a TSV task, in their fixed order."""
    columns = load_schema(task).get("x-columns")
    if columns is None:
        raise DataError(f"Task {task} has no table output (its output is an .npz file).")
    return list(columns["required"]), list(columns["optional"])


def output_path(folder: Any, task: Any, cell: Any) -> str:
    """Where the output of ``cell`` is expected in a predictions folder."""
    return os.path.join(folder, f"{cell}.{output_format(task)}")


def _validator(task: Any, role: Any) -> Any:
    """Return the (cached) JSON Schema validator of a task's input, output or row."""
    key = (task, role, "validator")
    if key not in _CACHE:
        jsonschema = require(
            "jsonschema",
            extra="validate",
            min_version="4.18",
            caller="evoparam.benchmark (task schemas)",
        )
        schema = load_schema(task)
        schema["$ref"] = f"#/$defs/{role}"
        _CACHE[key] = jsonschema.Draft202012Validator(schema)
    return _CACHE[key]


def _path_text(error: Any) -> str:
    """Return the place of a schema violation in the document, as a path."""
    parts = list(error.absolute_path)
    return "/".join(str(p) for p in parts) if parts else "(document)"


def _first_error(validator: Any, document: Any) -> Any:
    """Return the first schema violation in document order (None when valid)."""
    errors = list(validator.iter_errors(document))
    if not errors:
        return None
    # the first violation in document order: by the path of the instance, then by the
    # schema keyword, so that the same file always reports the same first violation
    return min(
        errors,
        key=lambda e: (
            [str(p) for p in e.absolute_path],
            [str(p) for p in e.absolute_schema_path],
            e.message,
        ),
    )


def validate(document: Any, task: Any, role: Any, *, where: str = "") -> None:
    """Validate ``document`` against one part of the task's schema.

    ``role`` names the part: ``"input"``, ``"output"`` or ``"output_row"``.

    Raises
    ------
    BenchmarkValidationError
        Naming the place of the first violation, the violation and the schema id.
    """
    error = _first_error(_validator(task, role), document)
    if error is not None:
        place = f"{where}: " if where else ""
        raise BenchmarkValidationError(
            f"{place}{_path_text(error)}: {error.message} (schema {schema_id(task)}, {role})"
        )


###################################################################################################
## Inputs


@dataclass
class TaskInput:
    """One task input: the JSON descriptor and its arrays.

    Attributes
    ----------
    descriptor : dict
        The validated descriptor (schema ``benchmark-task/<task>/1``, role ``input``).
    arrays : dict
        Array name -> ndarray (float64).
    truth : object, optional
        The ground truth, kept in memory by the suite (never written with the input).
    """

    descriptor: dict[str, Any]
    arrays: dict[str, Any]
    truth: Any = field(default=None, repr=False)

    @property
    def task(self) -> Any:
        """Task of the input."""
        return self.descriptor["task"]

    @property
    def cell(self) -> Any:
        """Cell of the input."""
        return self.descriptor["cell"]

    @property
    def times(self) -> Any:
        """Time axis (s)."""
        n_times = next(iter(self.arrays.values())).shape[-1]
        return self.descriptor["tmin"] + np.arange(n_times) / self.descriptor["sfreq"]

    def sha256(self) -> str:
        """SHA-256 of the input.

        The canonical JSON of its descriptor, which holds the hashes of the arrays.
        """
        return sha256_bytes(canonical_bytes(self.descriptor))

    def to_trialset(self) -> "TrialSet":
        """Return a :class:`~evoparam.data.TrialSet` of ``data``.

        It has ``stim_site`` metadata when the input has a site.
        """
        from evoparam.data import TrialSet

        pd = require("pandas", extra="data", caller="evoparam.benchmark")
        data = self.arrays["data"]
        meta = pd.DataFrame({"trial": np.arange(len(data))})
        if self.descriptor.get("site"):
            meta["stim_site"] = self.descriptor["site"]
        return TrialSet(
            data=data,
            times=self.times,
            sfreq=float(self.descriptor["sfreq"]),
            ch_names=list(self.descriptor["ch_names"]),
            units=self.descriptor["units"],
            metadata=meta,
            modality=self.descriptor.get("modality"),
        )

    def to_evoked(self) -> "Evoked":
        """Return an :class:`~evoparam.data.Evoked` of ``evoked`` (T3)."""
        from evoparam.data import Evoked

        return Evoked(
            data=self.arrays["evoked"],
            times=self.times,
            sfreq=float(self.descriptor["sfreq"]),
            ch_names=list(self.descriptor["ch_names"]),
            units=self.descriptor["units"],
            n_trials=self.descriptor["n_trials"],
        )


def make_descriptor(task: Any, arrays: Any, **fields: Any) -> dict[str, Any]:
    """Return a descriptor with the schema, role, task and array hashes filled in."""
    descriptor = {"schema": schema_id(task), "role": "input", "task": task}
    descriptor.update(fields)
    descriptor["arrays"] = {
        name: {"shape": list(np.shape(value)), "sha256": array_sha256(value)}
        for name, value in arrays.items()
    }
    descriptor["data_file"] = f"{fields['cell']}.npz"
    return descriptor


def write_input(task_input: Any, folder: Any) -> str:
    """Write ``<cell>.json`` and ``<cell>.npz`` into ``folder``; returns the JSON path."""
    descriptor = task_input.descriptor
    validate(descriptor, descriptor["task"], "input", where=descriptor["cell"])
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{descriptor['cell']}.json")
    arrays: dict[str, Any] = {
        k: np.asarray(v, dtype=np.float64) for k, v in task_input.arrays.items()
    }
    np.savez(os.path.join(folder, descriptor["data_file"]), **arrays)
    with open(path, "w", encoding="utf-8") as fobj:
        json.dump(descriptor, fobj, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False)
        fobj.write("\n")
    return path


def read_input(path: Any) -> "TaskInput":
    """Read and validate a task input (the ``.json`` descriptor; its ``.npz`` is next to it).

    Raises
    ------
    BenchmarkValidationError
        The descriptor does not follow its schema, or an array differs from its recorded
        shape or hash.
    """
    with open(path, encoding="utf-8") as fobj:
        try:
            descriptor = json.load(fobj)
        except json.JSONDecodeError as err:
            raise BenchmarkValidationError(f"{path}: not valid JSON ({err}).") from None
    task = descriptor.get("task") if isinstance(descriptor, dict) else None
    if task not in TASKS:
        raise BenchmarkValidationError(f"{path}: task: {task!r} is not one of {list(TASKS)}.")
    validate(descriptor, task, "input", where=path)
    npz = os.path.join(os.path.dirname(path), descriptor["data_file"])
    arrays = {}
    with np.load(npz, allow_pickle=False) as archive:
        for name, info in descriptor["arrays"].items():
            if name not in archive:
                raise BenchmarkValidationError(f"{npz}: array {name!r} is missing.")
            value = np.asarray(archive[name], dtype=np.float64)
            if list(value.shape) != info["shape"] or array_sha256(value) != info["sha256"]:
                raise BenchmarkValidationError(
                    f"{npz}: array {name!r} differs from the shape or SHA-256 of its descriptor."
                )
            arrays[name] = value
    return TaskInput(descriptor=descriptor, arrays=arrays)


###################################################################################################
## Outputs


def _format(value: Any) -> str:
    """Return a value as TSV text (``n/a`` for None and non-finite floats, 0/1 for booleans)."""
    if value is None:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return repr(value) if math.isfinite(value) else "n/a"
    text = str(value)
    if "\t" in text or "\n" in text:
        raise DataError(f"TSV value {text!r} holds a tab or a new line.")
    return text


def write_output(task: Any, value: Any, path: Any) -> Any:
    """Write a task output.

    Rows (list of dicts) as TSV with the task's columns, or the cleaned epochs (array) as ``.npz``
    for T4. Returns the path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if output_format(task) == "npz":
        np.savez(path, data=np.asarray(value, dtype=np.float64))
        return path
    required, optional = output_columns(task)
    present = [c for c in optional if any(c in row for row in value)]
    columns = required + present
    with open(path, "w", encoding="utf-8", newline="\n") as fobj:
        fobj.write("\t".join(columns) + "\n")
        for row in value:
            fobj.write("\t".join(_format(row.get(c)) for c in columns) + "\n")
    return path


def _string_columns(task: Any) -> set[Any]:
    """Return the output columns whose values are strings."""
    schema = load_schema(task)
    props = schema["$defs"]["output_row"]["properties"]
    out = set()
    for name, raw_prop in props.items():
        prop = raw_prop
        if "$ref" in prop:
            prop = schema["$defs"][prop["$ref"].rsplit("/", 1)[1]]
        kinds = prop.get("type", [])
        kinds = [kinds] if isinstance(kinds, str) else kinds
        if "string" in kinds:
            out.add(name)
    return out


_INT = re.compile(r"^[+-]?\d+$")


def _parse(text: Any, column: Any, strings: Any, where: Any) -> Any:
    """Return a TSV field typed by the schema (None for ``n/a``)."""
    if text == "n/a":
        return None
    if column in strings:
        return text
    if _INT.match(text):
        return int(text)
    try:
        number = float(text)
    except ValueError:
        return text  # the schema reports the wrong type
    if not math.isfinite(number):
        raise BenchmarkValidationError(
            f"{where}, column '{column}': {text!r} is not a finite "
            "number (write n/a for a missing value)"
        )
    return number


def read_output(task: Any, path: Any) -> Any:
    """Read and validate a task output.

    Returns
    -------
    list of dict or ndarray
        The rows of a TSV output, typed (``n/a`` -> None), or the ``data`` array of a T4
        output.

    Raises
    ------
    BenchmarkValidationError
        The first violation: header (unknown, missing or misplaced column), a row with the
        wrong number of fields, or the first row and column that do not follow the schema.
    """
    _check_task(task)
    if not os.path.isfile(path):
        raise DataError(f"{path}: no such output file.")
    if output_format(task) == "npz":
        return _read_npz_output(task, path)
    return _read_tsv_output(task, path)


def _read_npz_output(task: Any, path: Any) -> Any:
    """Read and validate the ``data`` array of a T4 output (``.npz``)."""
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as err:
        raise BenchmarkValidationError(f"{path}: not an .npz file ({err}).") from None
    with archive:
        names = list(archive.files)
        arrays = {name: archive[name] for name in names}
    data = arrays.get("data")
    description: dict[str, Any]
    description = {"arrays": {}}
    if data is not None:
        description["arrays"]["data"] = {
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "finite": bool(np.issubdtype(data.dtype, np.floating) and np.all(np.isfinite(data))),
        }
    for name in names:
        if name != "data":
            description["arrays"][name] = {}
    validate(description, task, "output", where=path)
    return np.asarray(data, dtype=np.float64)


def _read_tsv_output(task: Any, path: Any) -> list[dict[str, Any]]:
    """Read and validate the typed rows of a TSV output."""
    with open(path, encoding="utf-8") as fobj:
        lines = fobj.read().split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines:
        raise BenchmarkValidationError(f"{path}: empty file (the header line is required).")
    header = [c.strip() for c in lines[0].rstrip("\r").split("\t")]
    _check_header(task, path, header)
    strings = _string_columns(task)
    rows = []
    for number, line in enumerate(lines[1:], start=2):
        fields = line.rstrip("\r").split("\t")
        where = f"{path}: line {number}"
        if len(fields) != len(header):
            raise BenchmarkValidationError(
                f"{where}: {len(fields)} fields for {len(header)} columns"
            )
        row = {c: _parse(v, c, strings, where) for c, v in zip(header, fields, strict=False)}
        error = _first_error(_validator(task, "output_row"), row)
        if error is not None:
            column = next(iter(error.absolute_path), None)
            place = f", column '{column}'" if column is not None else ""
            raise BenchmarkValidationError(
                f"{where}{place}: {error.message} (schema {schema_id(task)})"
            )
        rows.append(row)
    return rows


def _check_header(task: Any, path: Any, header: list[str]) -> None:
    """Raise BenchmarkValidationError on an unknown, missing or misplaced column."""
    required, optional = output_columns(task)
    column: Any
    for column in header:
        if column not in required + optional:
            raise BenchmarkValidationError(
                f"{path}: header: unknown column '{column}' "
                f"(columns of {schema_id(task)}: "
                f"{required + optional})"
            )
    for column in required:
        if column not in header:
            raise BenchmarkValidationError(
                f"{path}: header: missing required column '{column}' (schema {schema_id(task)})"
            )
    expected = [c for c in required + optional if c in header]
    if header != expected:
        raise BenchmarkValidationError(
            f"{path}: header: columns must be in the order {expected}, found {header}"
        )
