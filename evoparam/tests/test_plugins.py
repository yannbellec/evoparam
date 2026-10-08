"""REQ-328: plugins through the entry points evoparam.shapes, evoparam.presets and
evoparam.readers.

The dummy distribution of ``tests/plugins/evoparam_dummy`` is made visible to
``importlib.metadata`` without pip: its package folder goes on ``sys.path`` next to a
``.dist-info`` folder holding its metadata and entry points, as an install would write.
"""

import os
import shutil
import sys

import numpy as np
import pytest

from evoparam import _plugins, presets
from evoparam.errors import SettingsError
from evoparam.io.readers import READERS, read
from evoparam.shapes import BACKGROUNDS, SHAPES

HERE = os.path.dirname(os.path.abspath(__file__))
DUMMY = os.path.join(HERE, "plugins", "evoparam_dummy")
ENTRY_POINTS = """[evoparam.shapes]
lorentzian = evoparam_dummy:Lorentzian

[evoparam.presets]
dummy_lorentz = evoparam_dummy:DUMMY_LORENTZ

[evoparam.readers]
dummy = evoparam_dummy:read_dummy
"""


def _install(site, name="evoparam_dummy", version="0.1.0", entry_points=ENTRY_POINTS):
    info = os.path.join(site, f"{name}-{version}.dist-info")
    os.makedirs(info, exist_ok=True)
    with open(os.path.join(info, "METADATA"), "w", encoding="utf-8") as fobj:
        fobj.write(f"Metadata-Version: 2.1\nName: {name.replace('_', '-')}\n"
                   f"Version: {version}\n")
    with open(os.path.join(info, "entry_points.txt"), "w", encoding="utf-8") as fobj:
        fobj.write(entry_points)


def _reload():
    for registry in (SHAPES, BACKGROUNDS, presets.PRESETS, READERS):
        registry.reload()


@pytest.fixture
def site(tmp_path, monkeypatch):
    """A site folder on sys.path with the dummy package; registries reloaded after."""

    folder = tmp_path / "site"
    shutil.copytree(os.path.join(DUMMY, "evoparam_dummy"), folder / "evoparam_dummy")
    monkeypatch.syspath_prepend(str(folder))
    monkeypatch.delenv("EVOPARAM_DISABLE_PLUGINS", raising=False)
    yield str(folder)
    sys.modules.pop("evoparam_dummy", None)
    sys.modules.pop("evoparam_other", None)
    monkeypatch.undo()
    _reload()
    _plugins.USED.clear()


def test_without_plugins_the_registries_hold_the_builtins():

    assert "lorentzian" not in SHAPES.builtin
    assert set(SHAPES.builtin) >= {"gaussian", "gamma"}
    assert "ccep" in presets.PRESETS.builtin


def test_dummy_shape_preset_and_reader(site, tmp_path):

    _install(site)
    _reload()
    origin = "evoparam-dummy==0.1.0"
    assert SHAPES.plugins() == {"lorentzian": origin}
    assert presets.PRESETS.plugins() == {"dummy_lorentz": origin}
    assert READERS.plugins() == {"dummy": origin}

    # the shape fits a simulated Lorentzian
    from evoparam.models import fit_response

    times = np.arange(-0.1, 0.5, 1e-3)
    rng = np.random.default_rng(0)
    y = -40.0 / (1.0 + ((times - 0.12) / 0.015) ** 2) + rng.normal(0, 1.0, times.size)
    result = fit_response(times, y, window=(0.01, 0.45), baseline=(-0.1, -0.005),
                          shapes=("lorentzian",), background="constant", random_state=0)
    assert result.status == "ok"
    comp = max(result.components, key=lambda c: abs(c.amplitude))
    assert comp.shape == "lorentzian"
    assert abs(comp.latency - 0.12) < 0.003 and abs(comp.fwhm - 0.030) < 0.005
    assert result.provenance.environment["plugins"]["lorentzian"] == origin

    # the preset resolves
    preset = presets.get("dummy_lorentz")
    assert preset.shapes == ("lorentzian",) and preset.parent == "ccep"

    # the reader opens a file
    data = rng.normal(size=(5, 2, 300))
    path = tmp_path / "rec.dummy"
    with open(path, "wb") as fobj:
        np.savez(fobj, data=data, sfreq=1000.0, tmin=-0.1)
    trials = read(str(path))
    assert trials.data.shape == (5, 2, 300) and trials.ch_names == ["D1", "D2"]
    assert trials.provenance.source["plugin"] == {"dummy": origin}
    assert read(str(path), format="dummy", picks=["D2"]).ch_names == ["D2"]
    assert trials.provenance.environment["plugins"]["dummy"] == origin


def test_cli_info_lists_the_plugins(site, capsys):

    import json

    from evoparam.cli import main

    _install(site)
    _reload()
    assert main(["info", "--json"]) == 0
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    names = {(p["group"], p["name"]) for p in summary["result"]["plugins"]}
    assert names == {("evoparam.shapes", "lorentzian"), ("evoparam.presets", "dummy_lorentz"),
                     ("evoparam.readers", "dummy")}
    assert "dummy_lorentz" in summary["result"]["presets"]


def test_name_collisions_raise(site):

    # with a built-in
    _install(site, entry_points="[evoparam.shapes]\ngaussian = evoparam_dummy:Lorentzian\n")
    with pytest.raises(SettingsError, match="built-in"):
        SHAPES.reload()
    shutil.rmtree(os.path.join(site, "evoparam_dummy-0.1.0.dist-info"))

    # between two distributions
    shutil.copytree(os.path.join(site, "evoparam_dummy"), os.path.join(site, "evoparam_other"))
    _install(site, entry_points="[evoparam.presets]\ndummy_lorentz = "
                                "evoparam_dummy:DUMMY_LORENTZ\n")
    _install(site, name="evoparam_other", version="2.0",
             entry_points="[evoparam.presets]\ndummy_lorentz = evoparam_other:DUMMY_LORENTZ\n")
    with pytest.raises(SettingsError, match="evoparam-dummy==0.1.0.*evoparam-other==2.0|"
                                            "evoparam-other==2.0.*evoparam-dummy==0.1.0"):
        presets.PRESETS.reload()


def test_bad_objects_are_refused(site):

    _install(site, entry_points="[evoparam.presets]\nnot_a_preset = evoparam_dummy:read_dummy\n")
    with pytest.raises(SettingsError, match="not a Preset"):
        presets.PRESETS.reload()
    _install(site, entry_points="[evoparam.readers]\nbad = evoparam_dummy:DUMMY_LORENTZ\n")
    with pytest.raises(SettingsError, match="not callable"):
        READERS.reload()
    _install(site, entry_points="[evoparam.shapes]\nwrong_name = evoparam_dummy:Lorentzian\n")
    with pytest.raises(SettingsError, match="name"):
        SHAPES.reload()


def test_plugins_can_be_disabled(site, monkeypatch):

    _install(site)
    monkeypatch.setenv("EVOPARAM_DISABLE_PLUGINS", "1")
    _reload()
    assert SHAPES.plugins() == {} and "lorentzian" not in SHAPES
    with pytest.raises(SettingsError):
        presets.get("dummy_lorentz")
