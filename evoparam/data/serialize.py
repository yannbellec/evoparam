# SPDX-License-Identifier: Apache-2.0
"""The ``.evoparam`` container: saving and loading containers and results.

A container is a directory (or a zip when the path ends in ``.zip``) holding:

``manifest.json``
    ``schema`` ("evoparam-result/<major>.<minor>"), ``object`` (class name),
    ``package_version``, ``created``, ``table_format`` and ``files``
    (relative path → SHA-256 of every other file).
``structure.json``
    The object's fields; arrays and tables are referenced by key.
``data.npz``
    The arrays (``allow_pickle=False``).
``data.npy``
    The main ``data`` array of a ``TrialSet`` or ``Evoked``, uncompressed so
    that it can be memory-mapped.
``tables/<key>.csv`` or ``tables/<key>.parquet``, with ``tables/<key>.json``
    The tables (parquet when ``pyarrow`` is installed, CSV otherwise); the JSON
    sidecar records column dtypes, the index and the unit of every column.
``provenance.json``, ``settings.json``
    The object's provenance and settings, for reading without evoparam.
``methods.txt``
    Results only: the methods paragraph (``methods_text()``) at saving time (REQ-390).

All JSON is strict RFC 8259 JSON: NaN and infinities are written as ``null``
in ``provenance.json``, ``settings.json`` and the table sidecars, and as
``{"__float__": "nan"}`` inside ``structure.json`` so they round-trip exactly.
"""

import hashlib
import math
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import fields, is_dataclass

import numpy as np

from evoparam.data._jsonutil import dumps, loads
from evoparam.errors import DataError, EvoparamVersionError
from evoparam.version import __version__

__all__ = ["save_object", "load_object", "register_type", "SCHEMA", "SCHEMA_MAJOR", "SCHEMA_MINOR"]

SCHEMA_MAJOR = 1
SCHEMA_MINOR = 0
SCHEMA = f"evoparam-result/{SCHEMA_MAJOR}.{SCHEMA_MINOR}"
SUFFIX = ".evoparam"
MISSING = "\\N"
_SCHEMA_RE = re.compile(r"^evoparam-result/(\d+)(?:\.(\d+))?$")


# Result types defined outside evoparam.data (for example evoparam.sim.GroundTruth), added
# by their module when it is imported; evoparam.data imports nothing from them (spec §4.2).
_EXTRA_TYPES = {}


def register_type(cls):
    """Make a dataclass defined outside :mod:`evoparam.data` savable in a ``.evoparam``
    container (spec §4.3.5). Its module calls this when it is imported.
    """
    _EXTRA_TYPES[cls.__name__] = cls
    return cls


def _registry():
    from evoparam.data import containers, provenance, results

    classes = [containers.TrialSet, containers.Evoked, provenance.Provenance]
    classes += [
        getattr(results, name)
        for name in results.__all__
        if isinstance(getattr(results, name), type)
    ]
    return {**{cls.__name__: cls for cls in classes}, **_EXTRA_TYPES}


def _is_dataframe(obj):
    return type(obj).__name__ == "DataFrame" and type(obj).__module__.startswith("pandas")


def _sha256(path, block=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as fobj:
        for chunk in iter(lambda: fobj.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _have_pyarrow():
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


###################################################################################################
## Encoding


class _Encoder:
    def __init__(self, top):
        self.top = top
        self.arrays = {}
        self.tables = {}
        self.main_array = None

    def _key(self, path, used):
        key = re.sub(r"[^A-Za-z0-9_.-]", "_", path) or "value"
        candidate, number = key, 1
        while candidate in used:
            candidate, number = f"{key}_{number}", number + 1
        return candidate

    def encode(self, obj, path):
        if obj is None or isinstance(obj, (str, bool)):
            return obj
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (int, np.integer)):
            return int(obj)
        if isinstance(obj, (float, np.floating)):
            value = float(obj)
            if math.isfinite(value):
                return value
            return {"__float__": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}
        if isinstance(obj, np.ndarray):
            if obj.dtype.kind in "OV":
                raise DataError(f"{path}: arrays of dtype {obj.dtype} cannot be saved.")
            if (
                obj is getattr(self.top, "data", None)
                and path == "data"
                and type(self.top).__name__ in ("TrialSet", "Evoked")
            ):
                self.main_array = obj
                return {"__npy__": "data.npy"}
            key = self._key(path, self.arrays)
            self.arrays[key] = obj
            return {"__array__": key}
        if _is_dataframe(obj):
            key = self._key(path, self.tables)
            self.tables[key] = obj
            return {"__table__": key}
        if is_dataclass(obj) and not isinstance(obj, type):
            name = type(obj).__name__
            if name not in _registry():
                raise DataError(f"{path}: {name} is not a savable evoparam type.")
            return {
                "__type__": name,
                "fields": {
                    fld.name: self.encode(
                        getattr(obj, fld.name), f"{path}.{fld.name}" if path else fld.name
                    )
                    for fld in fields(obj)
                },
            }
        if isinstance(obj, tuple):
            return {"__tuple__": [self.encode(v, f"{path}.{i}") for i, v in enumerate(obj)]}
        if isinstance(obj, list):
            return [self.encode(v, f"{path}.{i}") for i, v in enumerate(obj)]
        if isinstance(obj, dict):
            if all(isinstance(key, str) for key in obj):
                return {
                    "__dict__": {
                        key: self.encode(value, f"{path}.{key}") for key, value in obj.items()
                    }
                }
            return {
                "__items__": [
                    [self.encode(key, f"{path}.k{i}"), self.encode(value, f"{path}.{i}")]
                    for i, (key, value) in enumerate(obj.items())
                ]
            }
        raise DataError(f"{path}: values of type {type(obj).__name__} cannot be saved.")


def _write_table(table, folder, key, table_format):
    if not all(isinstance(column, str) for column in table.columns):
        raise DataError(f"table {key}: column names must be strings.")
    if table.columns.has_duplicates:
        raise DataError(f"table {key}: column names must be unique.")
    if type(table.index).__name__ == "MultiIndex":
        raise DataError(f"table {key}: a MultiIndex cannot be saved.")
    units = dict(table.attrs.get("units", {}))
    index = table.index
    is_range = type(index).__name__ == "RangeIndex" and index.start == 0 and index.step == 1
    sidecar = {
        "format": table_format,
        "n_rows": int(len(table)),
        "index": {"kind": "range"}
        if is_range
        else {"kind": "column", "name": index.name, "dtype": str(index.dtype)},
        "columns": {},
    }
    out = table if is_range else table.reset_index(names="__index__")
    for column in table.columns:
        dtype = table[column].dtype
        if dtype.kind not in "biufO" and str(dtype) not in ("str", "string"):
            raise DataError(f"table {key}: column '{column}' has unsupported dtype {dtype}.")
        spec = {"dtype": str(dtype), "unit": str(units.get(column, ""))}
        if dtype.kind == "O" or str(dtype) in ("str", "string"):
            raw = table[column].tolist()
            values = [v for v in raw if not _missing(v)]
            if any(not isinstance(v, str) for v in values):
                raise DataError(f"table {key}: column '{column}' mixes strings and other values.")
            if MISSING in values:
                raise DataError(
                    f"table {key}: column '{column}' contains the reserved value {MISSING!r}."
                )
            kinds = {_missing_kind(v) for v in raw if _missing(v)}
            if kinds and kinds != {"none"}:
                # which missing value the column held (None, NaN or pandas NA), restored on
                # reading; one kind per column
                if len(kinds) > 1:
                    raise DataError(
                        f"table {key}: column '{column}' mixes missing values of "
                        f"different kinds ({sorted(kinds)})."
                    )
                spec["missing"] = kinds.pop()
        sidecar["columns"][column] = spec
    if table_format == "parquet":
        out.to_parquet(os.path.join(folder, f"{key}.parquet"), index=False)
    elif len(out.columns):
        out.to_csv(os.path.join(folder, f"{key}.csv"), index=False, na_rep=MISSING)
    else:
        open(os.path.join(folder, f"{key}.csv"), "w", encoding="utf-8").close()
    with open(os.path.join(folder, f"{key}.json"), "w", encoding="utf-8") as fobj:
        fobj.write(dumps(sidecar, indent=1))


def _missing(value):
    return (
        value is None
        or (isinstance(value, float) and math.isnan(value))
        or type(value).__name__ == "NAType"
    )


def _missing_kind(value):
    if value is None:
        return "none"
    if type(value).__name__ == "NAType":
        return "na"
    return "nan"


def _write_tree(obj, folder, table_format):
    encoder = _Encoder(obj)
    structure = encoder.encode(obj, "")
    files = []
    with open(os.path.join(folder, "structure.json"), "w", encoding="utf-8") as fobj:
        fobj.write(dumps(structure, indent=1))
    files.append("structure.json")
    if encoder.main_array is not None:
        np.save(
            os.path.join(folder, "data.npy"),
            np.ascontiguousarray(encoder.main_array),
            allow_pickle=False,
        )
        files.append("data.npy")
    if encoder.arrays:
        np.savez(os.path.join(folder, "data.npz"), allow_pickle=False, **encoder.arrays)
        files.append("data.npz")
    if encoder.tables:
        os.makedirs(os.path.join(folder, "tables"))
        for key, table in encoder.tables.items():
            _write_table(table, os.path.join(folder, "tables"), key, table_format)
            files += [
                f"tables/{key}.{'parquet' if table_format == 'parquet' else 'csv'}",
                f"tables/{key}.json",
            ]
    provenance = getattr(obj, "provenance", None)
    with open(os.path.join(folder, "provenance.json"), "w", encoding="utf-8") as fobj:
        fobj.write(provenance.to_json(indent=1) if provenance is not None else "null")
    with open(os.path.join(folder, "settings.json"), "w", encoding="utf-8") as fobj:
        fobj.write(dumps(getattr(obj, "settings", {}), indent=1))
    files += ["provenance.json", "settings.json"]
    if callable(getattr(obj, "methods_text", None)):
        # the methods paragraph is stored with every saved result, not only produced on
        # request (spec §11.8, REQ-390)
        with open(os.path.join(folder, "methods.txt"), "w", encoding="utf-8") as fobj:
            fobj.write(obj.methods_text() + "\n")
        files.append("methods.txt")
    from evoparam.data.provenance import _now_utc

    manifest = {
        "schema": SCHEMA,
        "object": type(obj).__name__,
        "package_version": __version__,
        "created": _now_utc(),
        "table_format": table_format,
        "files": {name: _sha256(os.path.join(folder, *name.split("/"))) for name in files},
    }
    with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as fobj:
        fobj.write(dumps(manifest, indent=1))


def save_object(obj, path, overwrite=False, table_format="auto"):
    """Save a container or result to a ``.evoparam`` directory or zip.

    Parameters
    ----------
    obj : TrialSet, Evoked or any result dataclass
    path : str or path-like
        Target. ``.evoparam`` is appended to directory names that lack it; a
        path ending in ``.zip`` gives a zip archive.
    overwrite : bool
        Replace an existing target.
    table_format : {"auto", "csv", "parquet"}
        "auto" uses parquet when ``pyarrow`` is installed, CSV otherwise.

    Returns
    -------
    str
        The path written.
    """
    if table_format == "auto":
        table_format = "parquet" if _have_pyarrow() else "csv"
    if table_format not in ("csv", "parquet"):
        raise DataError(f"table_format must be 'auto', 'csv' or 'parquet', got {table_format!r}.")
    path = os.fspath(path)
    as_zip = path.endswith(".zip")
    if not as_zip and not path.endswith(SUFFIX):
        path += SUFFIX
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace it.")
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix=".evoparam-", dir=parent)
    try:
        _write_tree(obj, workdir, table_format)
        if as_zip:
            tmp_zip = workdir + ".zip"
            with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for root, _, names in os.walk(workdir):
                    for name in names:
                        full = os.path.join(root, name)
                        archive.write(full, os.path.relpath(full, workdir).replace(os.sep, "/"))
            if os.path.exists(path):
                os.remove(path)
            os.replace(tmp_zip, path)
        else:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.replace(workdir, path)
    finally:
        if os.path.exists(workdir):
            shutil.rmtree(workdir)
    return path


###################################################################################################
## Decoding


def _read_table(folder, key, n_hint=None):
    import pandas as pd

    with open(os.path.join(folder, f"{key}.json"), encoding="utf-8") as fobj:
        sidecar = loads(fobj.read())
    if sidecar["format"] == "parquet":
        table = pd.read_parquet(os.path.join(folder, f"{key}.parquet"))
    else:
        csv_path = os.path.join(folder, f"{key}.csv")
        if os.path.getsize(csv_path) == 0:
            table = pd.DataFrame(index=pd.RangeIndex(sidecar["n_rows"]))
        else:
            raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_filter=False)
            table = pd.DataFrame(index=pd.RangeIndex(len(raw)))
            specs = dict(sidecar["columns"])
            if sidecar["index"]["kind"] == "column":
                specs = {"__index__": {"dtype": sidecar["index"]["dtype"]}, **specs}
            for column, spec in specs.items():
                table[column] = _convert_column(
                    raw[column].tolist(), spec["dtype"], column, key, spec.get("missing", "none")
                )
    if sidecar["index"]["kind"] == "column":
        table = table.set_index("__index__")
        table.index.name = sidecar["index"]["name"]
    table.attrs["units"] = {column: spec["unit"] for column, spec in sidecar["columns"].items()}
    return table


def _convert_column(values, dtype, column, key, missing="none"):
    import pandas as pd

    if dtype[:1] in "IU" and dtype[1:].startswith(("nt", "Int")) or dtype == "boolean":
        # pandas nullable integers and booleans: missing values are pd.NA
        if dtype == "boolean":
            items = [pd.NA if v == MISSING else v == "True" for v in values]
        else:
            items = [pd.NA if v == MISSING else int(v) for v in values]
        return pd.Series(pd.array(items, dtype=dtype))
    kind = np.dtype(dtype).kind if dtype not in ("str", "string") else "S"
    if kind == "f":
        return pd.Series([np.nan if v == MISSING else float(v) for v in values], dtype=dtype)
    if kind in "iu":
        if MISSING in values:
            raise DataError(f"table {key}: integer column '{column}' has missing values.")
        return pd.Series([int(v) for v in values], dtype=dtype)
    if kind == "b":
        return pd.Series([v == "True" for v in values], dtype=dtype)
    gap = {"none": None, "na": pd.NA, "nan": float("nan")}[missing]
    converted = [gap if v == MISSING else v for v in values]
    return pd.Series(converted, dtype=dtype)


class _Decoder:
    def __init__(self, folder, arrays, mmap):
        self.folder = folder
        self.arrays = arrays
        self.mmap = mmap
        self.registry = _registry()

    def decode(self, node):
        if isinstance(node, list):
            return [self.decode(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "__float__" in node:
            return float(node["__float__"])
        if "__array__" in node:
            return np.array(self.arrays[node["__array__"]])
        if "__npy__" in node:
            return np.load(
                os.path.join(self.folder, node["__npy__"]),
                allow_pickle=False,
                mmap_mode="r" if self.mmap else None,
            )
        if "__table__" in node:
            return _read_table(os.path.join(self.folder, "tables"), node["__table__"])
        if "__tuple__" in node:
            return tuple(self.decode(item) for item in node["__tuple__"])
        if "__dict__" in node:
            return {key: self.decode(value) for key, value in node["__dict__"].items()}
        if "__items__" in node:
            return {self.decode(key): self.decode(value) for key, value in node["__items__"]}
        if "__type__" in node:
            cls = self.registry.get(node["__type__"])
            if cls is None:
                raise DataError(
                    f"Unknown object type {node['__type__']!r} in container "
                    "(import the module that defines it, e.g. evoparam.sim)."
                )
            return cls(**{name: self.decode(value) for name, value in node["fields"].items()})
        raise DataError(f"Unrecognised node in structure.json: {sorted(node)}.")


def _read_manifest(folder):
    manifest_path = os.path.join(folder, "manifest.json")
    if not os.path.exists(manifest_path):
        raise DataError(f"{folder}: not an evoparam container (no manifest.json).")
    with open(manifest_path, encoding="utf-8") as fobj:
        try:
            manifest = loads(fobj.read())
        except ValueError as err:
            raise DataError(f"{folder}: manifest.json is not valid JSON ({err}).") from err
    match = _SCHEMA_RE.match(str(manifest.get("schema", "")))
    if not match:
        raise DataError(f"{folder}: unknown schema {manifest.get('schema')!r}.")
    major = int(match.group(1))
    if major > SCHEMA_MAJOR:
        raise EvoparamVersionError(
            f"{folder}: container schema {manifest['schema']} is newer "
            f"than this evoparam ({SCHEMA}); upgrade evoparam."
        )
    return manifest, major


def _load_folder(folder, mmap, verify):
    manifest, major = _read_manifest(folder)
    if verify:
        for name, digest in manifest["files"].items():
            full = os.path.join(folder, *name.split("/"))
            if not os.path.exists(full):
                raise DataError(f"{folder}: file {name} listed in the manifest is missing.")
            if _sha256(full) != digest:
                raise DataError(
                    f"{folder}: checksum mismatch for {name}; the file was modified or corrupted."
                )
    with open(os.path.join(folder, "structure.json"), encoding="utf-8") as fobj:
        structure = loads(fobj.read())
    applied = []
    if major < SCHEMA_MAJOR:
        from evoparam.data._migrations import migrate

        try:
            structure, applied = migrate(structure, major, SCHEMA_MAJOR)
        except KeyError as err:
            raise EvoparamVersionError(
                f"{folder}: no migration from schema major {major} to {SCHEMA_MAJOR}."
            ) from err
    arrays = {}
    npz_path = os.path.join(folder, "data.npz")
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=False) as npz:
            arrays = {key: npz[key] for key in npz.files}
    obj = _Decoder(folder, arrays, mmap).decode(structure)
    for from_major, to_major in applied:
        obj.provenance = obj.provenance.with_step(
            "evoparam.data.migrate", {"from_schema": from_major, "to_schema": to_major}
        )
    return obj


def load_object(path, mmap=False, verify=True):
    """Load a container or result saved with :func:`save_object`.

    Parameters
    ----------
    path : str or path-like
        ``.evoparam`` directory or ``.zip`` archive.
    mmap : bool
        Memory-map the main ``data`` array read-only (directories only).
    verify : bool
        Check every file against the SHA-256 recorded in the manifest; a
        mismatch raises :class:`~evoparam.errors.DataError`.
    """
    path = os.fspath(path)
    if not os.path.exists(path) and os.path.exists(path + SUFFIX):
        path += SUFFIX
    if os.path.isdir(path):
        return _load_folder(path, mmap, verify)
    if zipfile.is_zipfile(path):
        if mmap:
            raise DataError("mmap=True needs a directory container, not a zip archive.")
        with tempfile.TemporaryDirectory(prefix=".evoparam-") as workdir:
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    target = os.path.realpath(os.path.join(workdir, member))
                    if not target.startswith(os.path.realpath(workdir) + os.sep):
                        raise DataError(f"{path}: unsafe member name {member!r}.")
                archive.extractall(workdir)
            obj = _load_folder(workdir, False, verify)
            return _materialise(obj)
    raise DataError(f"{path}: not an evoparam container.")


def _materialise(obj):
    """Make sure no array still points into a temporary folder."""
    if is_dataclass(obj) and not isinstance(obj, type):
        for fld in fields(obj):
            value = getattr(obj, fld.name)
            if isinstance(value, np.memmap):
                object.__setattr__(obj, fld.name, np.array(value))
    return obj
