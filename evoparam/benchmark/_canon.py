# SPDX-License-Identifier: Apache-2.0
"""Canonical JSON and hashes of the benchmark (spec §8.4, REQ-352).

The signed result of a scoring run must be reproducible byte for byte from the same
inputs, so everything that is hashed or written goes through one canonical encoding:
keys sorted, no whitespace, UTF-8, floats by Python's shortest round-trip ``repr``, and
NaN or infinite values written as ``null`` (RFC 8259 has no NaN). Arrays are hashed from
their little-endian float64 bytes, C order, after their shape, so that the hash of a task
input does not depend on the container (an ``.npz`` file stores a time stamp).
"""

import hashlib
import json
import math

import numpy as np

__all__ = ["canonical", "canonical_bytes", "sha256_bytes", "sha256_file", "array_sha256", "clean"]


def clean(value):
    """Return a JSON-ready copy.

    Numpy scalars and arrays to Python, NaN and +/-inf to None, tuples to lists, dict keys to str.
    """
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return [clean(v) for v in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def canonical(value):
    """Canonical JSON text of ``value`` (sorted keys, no spaces, NaN as null)."""
    return json.dumps(
        clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def canonical_bytes(value):
    """UTF-8 bytes of :func:`canonical`."""
    return canonical(value).encode("utf-8")


def sha256_bytes(data):
    """Hex SHA-256 of bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    """Hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as fobj:
        for block in iter(lambda: fobj.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array):
    """Hex SHA-256 of an array.

    Its shape (canonical JSON) then its float64 little-endian C-order bytes.
    """
    array = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    digest = hashlib.sha256(canonical_bytes(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
