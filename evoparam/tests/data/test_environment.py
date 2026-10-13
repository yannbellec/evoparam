# SPDX-License-Identifier: Apache-2.0
"""The environment record of the provenance (spec §9.12, REQ-309 d, REQ-345): the fixed
clock of reproducible builds, the CPU name, the git commit of a source checkout, the
lock-file hash and the data hash."""

import hashlib
import pathlib
from typing import Any

import numpy as np
import pytest

from evoparam.data import provenance

pytestmark = pytest.mark.core


@pytest.mark.parametrize(
    ("value", "expected"), [("", None), ("  ", None), ("1700000000", 1700000000), ("x", None)]
)
def test_fixed_clock(monkeypatch: Any, value: str, expected: Any) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", value)
    assert provenance.fixed_clock() == expected


def test_fixed_clock_gives_zero_durations(monkeypatch: Any) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    assert provenance.elapsed(0.0) == 0.0
    assert provenance._now_utc() == "1970-01-01T00:00:00.000+00:00"


def test_cpu_name_from_proc_when_the_platform_has_none(monkeypatch: Any) -> None:
    monkeypatch.setattr(provenance.platform, "processor", lambda: "Some CPU")
    assert provenance._cpu_name() == "Some CPU"
    monkeypatch.setattr(provenance.platform, "processor", lambda: "")
    proc = pathlib.Path("/proc/cpuinfo")
    if proc.is_file():
        names = [
            line.split(":", 1)[1].strip()
            for line in proc.read_text(encoding="utf-8").splitlines()
            if line.lower().startswith("model name")
        ]
        assert provenance._cpu_name() == (names[0] if names else None)

    def no_file(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("no /proc")

    monkeypatch.setattr("builtins.open", no_file)
    assert provenance._cpu_name() is None


def _checkout(tmp_path: pathlib.Path, monkeypatch: Any) -> pathlib.Path:
    """A fake source checkout: <root>/.git and <root>/evoparam/data/provenance.py."""
    module = tmp_path / "evoparam" / "data" / "provenance.py"
    module.parent.mkdir(parents=True)
    monkeypatch.setattr(provenance, "__file__", str(module))
    git = tmp_path / ".git"
    git.mkdir()
    return git


def test_git_commit_of_a_checkout(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    git = _checkout(tmp_path, monkeypatch)
    sha = "0123456789abcdef0123456789abcdef01234567"
    (git / "HEAD").write_text(sha + "\n", encoding="utf-8")  # detached
    assert provenance._git_commit() == sha
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "refs" / "heads" / "main").write_text(sha + "\n", encoding="utf-8")
    assert provenance._git_commit() == sha
    (git / "refs" / "heads" / "main").unlink()  # packed
    (git / "packed-refs").write_text(
        f"# pack-refs with: peeled\n{'f' * 40} refs/heads/other\n{sha} refs/heads/main\n",
        encoding="utf-8",
    )
    assert provenance._git_commit() == sha
    (git / "packed-refs").write_text(f"{'f' * 40} refs/heads/other\n", encoding="utf-8")
    assert provenance._git_commit() is None


def test_git_commit_without_a_checkout(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    git = _checkout(tmp_path, monkeypatch)
    git.rmdir()
    assert provenance._git_commit() is None


def test_lock_sha256(tmp_path: pathlib.Path) -> None:
    from importlib import metadata

    numpy_pin = f"numpy=={metadata.version('numpy')}"
    lock = tmp_path / "py.lock"
    assert provenance.lock_sha256(str(lock)) is None  # no lock file
    text = (
        "# a comment\n--require-hashes\n"
        f"{numpy_pin} \\\n    --hash=sha256:00\n"
        "evoparam-no-such-package==1.0 ; python_version < '3.0'\n"
    )
    lock.write_text(text, encoding="utf-8")
    assert provenance.lock_sha256(str(lock)) == hashlib.sha256(text.encode()).hexdigest()
    lock.write_text("numpy==0.0.1\n", encoding="utf-8")  # another version installed
    assert provenance.lock_sha256(str(lock)) is None
    lock.write_text("evoparam-no-such-package==1.0\n", encoding="utf-8")  # not installed
    assert provenance.lock_sha256(str(lock)) is None


def test_data_hash_of_an_empty_array() -> None:
    empty = provenance.compute_data_hash(np.zeros((0, 2)), [0.0, 0.1], ["a", "b"])
    assert empty != provenance.compute_data_hash(np.zeros((1, 2)), [0.0, 0.1], ["a", "b"])
    assert empty == provenance.compute_data_hash(np.zeros((0, 2)), [0.0, 0.1], ["a", "b"])
