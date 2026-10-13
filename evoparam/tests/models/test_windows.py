# SPDX-License-Identifier: Apache-2.0
"""Named latency windows (spec §4.4, §4.5, §5.5.7): :class:`evoparam.models.ComponentWindows`.

The windows name components in time with an optional polarity; a ``ComponentWindows`` is a
read-only mapping accepted wherever a dictionary of windows is (``FitSettings.windows``).
"""

import json

import pytest

from evoparam.errors import SettingsError
from evoparam.models import ComponentWindows
from evoparam.models._settings import FIT_PRESETS, FitSettings

pytestmark = pytest.mark.core


def test_windows_are_a_read_only_mapping() -> None:
    windows = ComponentWindows({"N1": (0.009, 0.05, -1), "N2": [0.05, 0.3]})
    assert windows["N1"] == (0.009, 0.05, -1)
    assert windows["N2"] == (0.05, 0.3, 0)  # polarity 0: either sign
    assert len(windows) == len(windows.labels)
    assert list(windows) == windows.labels == ["N1", "N2"]
    assert windows.definitions == [(0.009, 0.05, -1), (0.05, 0.3, 0)]
    assert dict(windows) == {"N1": (0.009, 0.05, -1), "N2": (0.05, 0.3, 0)}
    assert repr(windows) == "ComponentWindows({N1: (0.009, 0.05, -1), N2: (0.05, 0.3, 0)})"
    assert json.loads(json.dumps(windows.to_dict())) == {
        "N1": [0.009, 0.05, -1],
        "N2": [0.05, 0.3, 0],
    }
    with pytest.raises(KeyError, match=r"no window named 'P3'; windows: \['N1', 'N2'\]"):
        windows["P3"]
    assert len(ComponentWindows()) == 0


def test_add_and_remove_return_copies() -> None:
    windows = ComponentWindows({"N1": (0.009, 0.05, -1)})
    more = windows.add("P2", (0.1, 0.2, 1))
    assert more.labels == ["N1", "P2"]
    assert windows.labels == ["N1"]
    fewer = more.remove("N1")
    assert fewer.labels == ["P2"]
    assert more.labels == ["N1", "P2"]
    with pytest.raises(SettingsError, match="already exists"):
        more.add("P2", (0.1, 0.3))
    with pytest.raises(KeyError, match="no window named 'P3'"):
        more.remove("P3")


@pytest.mark.parametrize(
    ("name", "definition", "message"),
    [
        ("", (0.0, 0.1), "non-empty string"),
        (3, (0.0, 0.1), "non-empty string"),
        ("N1", (0.0,), r"\(start, end\) or \(start, end, polarity\)"),
        ("N1", (0.0, 0.1, 1, 2), r"\(start, end\) or \(start, end, polarity\)"),
        ("N1", (0.1, 0.1), r"start \(0.1 s\) must be before end \(0.1 s\)"),
        ("N1", (0.0, 0.1, 2), "polarity must be -1, \\+1 or 0"),
    ],
)
def test_invalid_windows_are_refused(name: object, definition: tuple, message: str) -> None:
    with pytest.raises(SettingsError, match=message):
        ComponentWindows({name: definition})


@pytest.mark.parametrize("preset", sorted(FIT_PRESETS))
def test_windows_of_a_preset(preset: str) -> None:
    windows = ComponentWindows.from_preset(preset)
    assert dict(windows) == {
        name: tuple(window) for name, window in (FIT_PRESETS[preset]["windows"] or {}).items()
    }


def test_unknown_preset() -> None:
    with pytest.raises(SettingsError, match="unknown preset 'eeg'"):
        ComponentWindows.from_preset("eeg")


def test_fit_settings_accept_component_windows() -> None:
    windows = ComponentWindows.from_preset("ccep")
    settings = FitSettings(windows=windows).validate()
    assert settings.windows == dict(windows)
    assert settings.windows == FitSettings(windows=windows.to_dict()).validate().windows
