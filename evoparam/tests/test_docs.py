"""Documentation content (spec §10.7, REQ-325 for the streaming tutorial; source tree only)."""

import os
import runpy

import pytest

import evoparam

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(evoparam.__file__)))
STREAM_TUTORIAL = os.path.join(ROOT, "tutorials", "92_online_mapping_with_lsl.py")

pytestmark = pytest.mark.skipif(not os.path.isfile(os.path.join(ROOT, "pyproject.toml")),
                                reason="needs the source tree")


def test_req325_streaming_tutorial_states_intended_use():

    with open(STREAM_TUTORIAL, encoding="utf-8") as fobj:
        text = fobj.read()
    assert "not a medical device" in text
    assert "not validated for clinical" in text


def test_streaming_tutorial_runs(capsys):

    runpy.run_path(STREAM_TUTORIAL, run_name="__main__")
    out = capsys.readouterr().out
    assert "c0: response" in out and "c7: no response" in out
