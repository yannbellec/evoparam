# SPDX-License-Identifier: Apache-2.0
"""Machine-readable report: one JSON file holding the ``.evoparam`` container (REQ-341)."""

import json
import math
import os
import tempfile

import numpy as np

from evoparam._typing import Any
from evoparam.errors import DataError
from evoparam.utils._optional import require

__all__ = ["to_json", "from_json", "validate_json", "SCHEMA_FILE"]

SCHEMA_FILE = "evoparam-result-1.json"


def _encode_value(value: Any) -> Any:
    """Return a JSON value for a scalar (NaN and infinities as strings)."""
    if isinstance(value, (str, np.str_, bytes)):
        return str(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def _decode_values(values: Any, dtype: Any) -> Any:
    """Return the array of ``dtype`` written by :func:`_encode_value`."""
    if dtype.startswith(("float", "complex")):
        return np.array([float(v) for v in values], dtype=dtype)
    if dtype == "bool":
        return np.array(values, dtype=bool)
    if dtype.startswith(("int", "uint")):
        return np.array(values, dtype=dtype)
    return np.array(values, dtype=dtype)


def to_json(result: Any, path: str | None = None, *, validate: bool | None = None) -> Any:
    """Return the full result as one JSON document (schema ``evoparam-result/1``).

    The document holds the files of the ``.evoparam`` container: ``structure`` (the fields),
    ``arrays`` (dtype, shape and flat values; ``"nan"``, ``"inf"`` and ``"-inf"`` for the
    non-finite numbers, so that the file is strict RFC 8259 JSON), ``tables`` (CSV text and
    its sidecar of dtypes and units), ``provenance`` and ``settings``.

    Parameters
    ----------
    result : any object that :func:`evoparam.data.serialize.save_object` saves
    path : str, optional
        Write the document there (and return the path); else return the text.
    validate : bool, optional
        Validate against the shipped JSON Schema (needs ``jsonschema``; default: when it is
        installed).
    """
    from evoparam.data.serialize import save_object

    with tempfile.TemporaryDirectory(prefix=".evoparam-json-") as work:
        folder = save_object(result, os.path.join(work, "result.evoparam"), table_format="csv")

        def read(name: Any) -> Any:
            """Return the text of file ``name`` of the container."""
            return open(os.path.join(folder, name), encoding="utf-8").read()

        manifest = json.loads(read("manifest.json"))
        document = {
            "schema": manifest["schema"],
            "object": manifest["object"],
            "package_version": manifest["package_version"],
            "created": manifest["created"],
            "structure": json.loads(read("structure.json")),
            "arrays": {},
            "tables": {},
            "provenance": json.loads(read("provenance.json")),
            "settings": json.loads(read("settings.json")),
        }
        for name in ("data.npz", "data.npy"):
            full = os.path.join(folder, name)
            if not os.path.exists(full):
                continue
            if name.endswith(".npz"):
                with np.load(full, allow_pickle=False) as npz:
                    items = {key: npz[key] for key in npz.files}
            else:
                items = {"__main__": np.load(full, allow_pickle=False)}
            for key, array in items.items():
                document["arrays"][key] = {
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "values": [_encode_value(v) for v in array.ravel()],
                }
        tables = os.path.join(folder, "tables")
        if os.path.isdir(tables):
            for name in sorted(os.listdir(tables)):
                if name.endswith(".csv"):
                    key = name[:-4]
                    document["tables"][key] = {
                        "csv": read(os.path.join("tables", name)),
                        "sidecar": json.loads(read(os.path.join("tables", key + ".json"))),
                    }
    text = json.dumps(document, allow_nan=False, indent=None, separators=(",", ":"))
    if validate or (validate is None and _have_jsonschema()):
        validate_json(document)
    if path is None:
        return text
    with open(path, "w", encoding="utf-8") as fobj:
        fobj.write(text)
    return os.fspath(path)


def _have_jsonschema() -> bool:
    """Return whether ``jsonschema`` is installed."""
    try:
        require("jsonschema", extra="validate", caller="evoparam.reports._export._have_jsonschema")
        import jsonschema  # noqa: F401
    except ImportError:
        return False
    return True


def schema() -> Any:
    """Return the shipped JSON Schema (a dict)."""
    from importlib import resources

    return json.loads(
        resources.files("evoparam.reports")
        .joinpath("schema")
        .joinpath(SCHEMA_FILE)
        .read_text("utf-8")
    )


def validate_json(document: Any) -> Any:
    """Validate a document (dict, JSON text or path) against the shipped schema.

    Raises
    ------
    DataError
        The document does not follow the schema (the message names the place).
    ImportError
        ``jsonschema`` is not installed (extra ``validate``).
    """
    try:
        require("jsonschema", extra="validate", caller="evoparam.reports._export.validate_json")
        import jsonschema
    except ImportError as err:
        raise ImportError(
            'Validating evoparam JSON needs jsonschema: pip install "evoparam[validate]".'
        ) from err
    if isinstance(document, (str, os.PathLike)):
        if os.path.exists(document):
            with open(document, encoding="utf-8") as fobj:
                text: Any
                text = fobj.read()
        else:
            text = document
        document = json.loads(text)
    try:
        jsonschema.validate(document, schema())
    except jsonschema.ValidationError as err:
        place = "/".join(str(p) for p in err.absolute_path) or "(top level)"
        raise DataError(
            f"evoparam JSON does not follow {SCHEMA_FILE} at {place}: {err.message}"
        ) from err
    return True


def from_json(source: Any) -> Any:
    """Return the result of a document written by :func:`to_json` (dict, JSON text or path)."""
    from evoparam.data._jsonutil import dumps
    from evoparam.data.serialize import load_object

    if isinstance(source, dict):
        document = source
    else:
        if os.path.exists(os.fspath(source)):
            with open(source, encoding="utf-8") as fobj:
                text = fobj.read()
        else:
            text = source
        document = json.loads(text)
    for key in ("schema", "object", "structure", "arrays", "tables"):
        if key not in document:
            raise DataError(f"not an evoparam JSON document (no '{key}').")
    with tempfile.TemporaryDirectory(prefix=".evoparam-json-") as work:
        folder = os.path.join(work, "result.evoparam")
        os.makedirs(os.path.join(folder, "tables"))

        def write(name: Any, text: Any) -> Any:
            """Write ``text`` to file ``name`` of the container."""
            return open(os.path.join(folder, name), "w", encoding="utf-8").write(text)

        write("structure.json", json.dumps(document["structure"]))
        write("provenance.json", json.dumps(document.get("provenance")))
        write("settings.json", dumps(document.get("settings") or {}))
        arrays = {}
        for key, spec in document["arrays"].items():
            array = _decode_values(spec["values"], spec["dtype"]).reshape(spec["shape"])
            if key == "__main__":
                np.save(os.path.join(folder, "data.npy"), array, allow_pickle=False)
            else:
                arrays[key] = array
        if arrays:
            np.savez(os.path.join(folder, "data.npz"), allow_pickle=False, **arrays)
        for key, table in document["tables"].items():
            write(os.path.join("tables", key + ".csv"), table["csv"])
            write(os.path.join("tables", key + ".json"), json.dumps(table["sidecar"]))
        manifest = {
            "schema": document["schema"],
            "object": document["object"],
            "package_version": document.get("package_version"),
            "created": document.get("created"),
            "table_format": "csv",
            "files": {},
        }
        write("manifest.json", json.dumps(manifest))
        return load_object(folder, verify=False)
