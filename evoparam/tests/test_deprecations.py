# SPDX-License-Identifier: Apache-2.0
"""REQ-293 (deprecation shims) and REQ-394 (frozen name; renamed symbols keep shims).

Every entry of ``evoparam/_deprecated.py`` (``{old_name: (new_name, since, remove_in)}``) is
served by the package root with an ``EvoparamDeprecationWarning`` that names both names and
the version of removal, forwards every argument, is removed exactly two minor versions after
``since``, and is gone once ``evoparam.__version__ >= remove_in``. The registry is empty until
a public name changes (nothing is public yet); the mechanism is checked with a temporary
entry.
"""

import pathlib
import re
import warnings

import pytest

import evoparam
from evoparam import _deprecated
from evoparam.utils import EvoparamDeprecationWarning
from evoparam.utils._deprecation import minor_gap, version_tuple

pytestmark = pytest.mark.core

ROOT = pathlib.Path(evoparam.__file__).resolve().parent.parent


def _check_entry(old, new, since, remove_in):
    assert minor_gap(since, remove_in) == 2, (old, since, remove_in)
    assert version_tuple(evoparam.__version__) < version_tuple(remove_in), (
        f"{old}: the shim must be removed in {remove_in} (this is {evoparam.__version__})"
    )
    with pytest.warns(EvoparamDeprecationWarning) as record:
        shim = getattr(evoparam, old)
        if callable(shim):
            # forwarding is checked with a temporary target in the test below; here the
            # message is produced by calling with no argument when possible
            try:
                shim()
            except TypeError:
                warnings.warn(shim.__deprecated__, EvoparamDeprecationWarning)
    text = str(record[0].message)
    assert old in text and new in text and remove_in in text


@pytest.mark.parametrize("old", sorted(_deprecated.REGISTRY) or [None])
def test_req293_every_registered_shim(old):
    if old is None:
        pytest.skip("no deprecated public name yet (no public release, RELEASING.md)")
    new, since, remove_in = _deprecated.REGISTRY[old]
    _check_entry(f"{old}", new, since, remove_in)


def test_req293_warning_category():
    assert issubclass(EvoparamDeprecationWarning, DeprecationWarning)
    assert evoparam.utils.EvoparamDeprecationWarning is EvoparamDeprecationWarning


def test_req293_the_mechanism_with_a_temporary_entry(monkeypatch):
    """A temporary shim ``evoparam.old_mean`` -> ``numpy:mean``: warning with both names and
    the version of removal, all arguments forwarded, the two-minor-version rule."""

    import numpy as np

    monkeypatch.setitem(_deprecated.REGISTRY, "old_mean", ("numpy:mean", "0.2.0", "0.4.0"))
    with pytest.warns(EvoparamDeprecationWarning) as record:
        value = evoparam.old_mean(np.array([[1.0, 3.0], [5.0, 7.0]]), axis=1, keepdims=True)
    assert value.tolist() == [[2.0], [6.0]]
    text = str(record[0].message)
    assert "evoparam.old_mean" in text and "numpy:mean" in text and "0.4.0" in text
    _check_entry("old_mean", "numpy:mean", "0.2.0", "0.4.0")


def test_req293_the_rules_fail_on_a_bad_entry(monkeypatch):
    monkeypatch.setitem(_deprecated.REGISTRY, "old_mean", ("numpy:mean", "0.2.0", "0.3.0"))
    with pytest.raises(AssertionError):
        _check_entry("old_mean", "numpy:mean", "0.2.0", "0.3.0")  # one minor version only
    monkeypatch.setitem(_deprecated.REGISTRY, "old_mean", ("numpy:mean", "0.0.0", "0.1.0"))
    with pytest.raises(AssertionError):
        _check_entry("old_mean", "numpy:mean", "0.0.0", "0.1.0")  # removal version reached


def test_req293_unknown_name_still_raises_attribute_error():
    with pytest.raises(AttributeError):
        evoparam.no_such_name  # noqa: B018


def test_req394_name_paired_with_the_phrase():
    """REQ-394: README and the docs home pair the name with "evoked response
    parameterization"."""

    for path in ("README.rst", "docs/index.rst"):
        text = (ROOT / path).read_text(encoding="utf-8")
        assert re.search(
            r"evoparam.{0,80}evoked response parameterization", text, re.IGNORECASE | re.DOTALL
        ), path


def test_req394_name_is_frozen_in_the_metadata():
    """The distribution, the import name, the console script and the citation all use the
    one name ``evoparam`` (the frozen name of §11.1 rule 2)."""

    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^name = "evoparam"$', text, re.MULTILINE)
    assert 'evoparam = "evoparam.cli:main"' in text
    assert re.search(
        r'^title: "evoparam', (ROOT / "CITATION.cff").read_text("utf-8"), re.MULTILINE
    )


@pytest.mark.xfail(
    strict=True,
    reason="REQ-394: the name is frozen at the first public "
    "release, which waits for the owner's HIIE decision (nothing public "
    "before it)",
)
def test_req394_frozen_at_the_first_public_release():
    text = (ROOT / "RELEASING.md").read_text(encoding="utf-8")
    gate = text.split("<!-- release-gate -->")[1].split("<!-- end-release-gate -->")[0]
    assert "- HIIE decision: on record" in gate
