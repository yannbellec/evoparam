# SPDX-License-Identifier: Apache-2.0
"""Provenance: the record that lets anyone regenerate a result.

Every container and every result carries a :class:`Provenance`. It stores the
package version, the creation time, where the data came from, the ordered list
of processing steps with their full settings, the software environment, the
random seed and a hash of the data.
"""

import copy
import functools
import hashlib
import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata

import numpy as np

from evoparam.version import __version__, algorithm_version
from evoparam.data._compare import DeepEqualMixin
from evoparam.data._jsonutil import dumps, loads, to_jsonable

# Optional packages whose versions are recorded when installed (None when absent).
OPTIONAL_PACKAGES = (
    "pandas",
    "pyarrow",
    "matplotlib",
    "mne",
    "mne-bids",
    "pynwb",
    "neo",
    "pyedflib",
    "xarray",
    "joblib",
    "tqdm",
    "click",
    "statsmodels",
    "jsonschema",
    "pylsl",
    "jax",
    "numpyro",
    "pymc",
    "cupy",
)


def fixed_clock():
    """Seconds since the epoch from ``SOURCE_DATE_EPOCH``, or None.

    ``SOURCE_DATE_EPOCH`` is the convention of reproducible builds. When it is set, every
    creation time is that instant and every duration is recorded as 0, so two runs with
    the same inputs and seed write byte-identical files (REQ-345). Without it, the real
    clock is used.
    """

    value = os.environ.get("SOURCE_DATE_EPOCH", "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def elapsed(start):
    """Seconds since ``start`` (a ``time.perf_counter()`` value); 0 with a fixed clock."""

    import time

    return 0.0 if fixed_clock() is not None else time.perf_counter() - start


def _now_utc():
    epoch = fixed_clock()
    when = (
        datetime.fromtimestamp(epoch, timezone.utc)
        if epoch is not None
        else datetime.now(timezone.utc)
    )
    return when.isoformat(timespec="milliseconds")


def _package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _cpu_name():
    name = platform.processor()
    if name:
        return name
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fobj:
            for line in fobj:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _git_commit():
    """Commit of the package source tree when installed from a git checkout."""

    git_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.pardir, ".git"
    )
    try:
        with open(os.path.join(git_dir, "HEAD"), encoding="utf-8") as fobj:
            head = fobj.read().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(" ", 1)[1]
        ref_path = os.path.join(git_dir, *ref.split("/"))
        if os.path.exists(ref_path):
            with open(ref_path, encoding="utf-8") as fobj:
                return fobj.read().strip()
        with open(os.path.join(git_dir, "packed-refs"), encoding="utf-8") as fobj:
            for line in fobj:
                if line.strip().endswith(ref):
                    return line.split(" ", 1)[0]
    except OSError:
        pass
    return None


def _lock_path():
    """The lock file of this Python minor version (REQ-309 c): ``ci/locks/pyX.Y.lock`` of
    the source tree, or of the folder named by ``EVOPARAM_LOCK_DIR``."""

    folder = os.environ.get("EVOPARAM_LOCK_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ci", "locks"
    )
    return os.path.join(folder, "py{}.{}.lock".format(*sys.version_info[:2]))


def lock_sha256(path=None):
    """SHA-256 of the lock file when the installed packages are exactly its pins for this
    interpreter (every applicable pin installed at its version), else None (REQ-309 d)."""

    path = path or _lock_path()
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as fobj:
        raw = fobj.read()
    try:
        from packaging.markers import Marker
    except ImportError:  # markers cannot be evaluated: no claim
        Marker = None
    from importlib import metadata

    for line in raw.decode("utf-8").splitlines():
        line = line.split("#", 1)[0].strip().rstrip("\\").strip()
        if not line or line.startswith("-") or "==" not in line:
            continue
        pin, _, marker = line.partition(";")
        name, _, version = pin.partition("==")
        if marker.strip():
            if Marker is None:
                return None
            if not Marker(marker.strip()).evaluate():
                continue
        try:
            installed = metadata.version(name.split("[")[0].strip())
        except metadata.PackageNotFoundError:
            return None
        if installed != version.strip():
            return None
    return hashlib.sha256(raw).hexdigest()


@functools.lru_cache(maxsize=1)
def _environment():
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": _cpu_name(),
        "numpy": np.__version__,
        "scipy": _package_version("scipy"),
        "optional": {name: _package_version(name) for name in OPTIONAL_PACKAGES},
        "backend": "numpy",
        "jax_enable_x64": None,
        "n_threads": os.environ.get("OMP_NUM_THREADS"),
        "git_commit": _git_commit(),
        "lock_sha256": lock_sha256(),
        "executable": os.path.basename(sys.executable),
    }
    return env


def capture_environment():
    """Return the software environment as a JSON-serialisable dictionary."""

    # the plugins a result used are added by the functions that use them (models, readers):
    # environment["plugins"] = {name: "dist==version"}, absent without plugins (spec §10.8)
    return copy.deepcopy(_environment())


def compute_data_hash(data, times, ch_names):
    """SHA-256 of the data bytes, the time axis and the channel names.

    The data are hashed one slice of the first axis at a time, so a
    memory-mapped array is never copied whole into memory.
    """

    digest = hashlib.sha256()
    data = np.asarray(data)
    if data.ndim == 0 or data.shape[0] == 0:
        digest.update(np.ascontiguousarray(data).tobytes())
    else:
        for index in range(data.shape[0]):
            digest.update(np.ascontiguousarray(data[index]).tobytes())
    digest.update(np.ascontiguousarray(np.asarray(times, dtype=float)).tobytes())
    digest.update("\0".join(ch_names).encode("utf-8"))
    return digest.hexdigest()


@dataclass(eq=False)
class Provenance(DeepEqualMixin):
    """Record of where a container or result comes from and how it was made.

    Parameters
    ----------
    package_version : str
        evoparam version that created the object.
    created : str
        Creation time, ISO-8601, UTC.
    source : dict
        Origin of the data, for example ``{"path": ..., "format": "brainvision",
        "reader": "evoparam.io.read_brainvision", "sha256": ..., "bytes_read": ...}``.
    steps : list of dict
        Processing steps in order. Each step holds ``function``, ``settings``,
        ``version`` and ``duration_s``.
    environment : dict
        Python, platform and package versions (see :func:`capture_environment`).
    random_state : int or None
        Seed used by the steps that draw random numbers.
    data_hash : str or None
        SHA-256 of the data at creation (see :func:`compute_data_hash`).
    """

    package_version: str = __version__
    created: str = field(default_factory=_now_utc)
    source: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)
    environment: dict = field(default_factory=capture_environment)
    random_state: int | None = None
    data_hash: str | None = None

    def __post_init__(self):
        self.source = to_jsonable(dict(self.source))
        self.steps = to_jsonable(list(self.steps))
        self.environment = to_jsonable(dict(self.environment))
        if self.random_state is not None:
            self.random_state = int(self.random_state)

    def copy(self):
        """Return a deep copy."""

        return copy.deepcopy(self)

    def with_step(self, function, settings=None, duration_s=None, **extra):
        """Return a copy with one more processing step.

        Parameters
        ----------
        function : str
            Full dotted name of the public function, for example
            ``"evoparam.data.TrialSet.crop"``.
        settings : dict, optional
            Every argument that changes the result. Must be JSON-serialisable.
        duration_s : float, optional
            Wall-clock duration of the step, in seconds.
        **extra
            Additional keys stored in the step (for example ``preset``).
        """

        if duration_s is not None and fixed_clock() is not None:
            duration_s = 0.0
        settings = settings or {}
        # REQ-294: every step has the seven keys (function, settings, version, preset,
        # random_state, duration_s, n_jobs); REQ-399: and the version of its algorithm
        step = {
            "function": function,
            "settings": to_jsonable(settings),
            "version": __version__,
            "algorithm_version": algorithm_version(function),
            "preset": None,
            "random_state": settings.get("random_state") if isinstance(settings, dict) else None,
            "duration_s": None if duration_s is None else float(duration_s),
            "n_jobs": settings.get("n_jobs", 1) if isinstance(settings, dict) else 1,
        }
        step.update(to_jsonable(extra))
        new = self.copy()
        new.steps.append(step)
        return new

    def to_dict(self):
        """Return a JSON-serialisable dictionary."""

        return {
            "package_version": self.package_version,
            "created": self.created,
            "source": copy.deepcopy(self.source),
            "steps": copy.deepcopy(self.steps),
            "environment": copy.deepcopy(self.environment),
            "random_state": self.random_state,
            "data_hash": self.data_hash,
        }

    @classmethod
    def from_dict(cls, dct):
        """Rebuild from :meth:`to_dict` output."""

        return cls(**dct)

    def to_json(self, **kwargs):
        """Return strict JSON text (NaN written as null)."""

        return dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, text):
        """Rebuild from :meth:`to_json` output."""

        return cls.from_dict(loads(text))
