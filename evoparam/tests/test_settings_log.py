# SPDX-License-Identifier: Apache-2.0
"""REQ-331: the settings and the preset are recorded in every result.

``settings`` is the full resolved dictionary, ``_overrides`` names the keys the user set,
``settings["preset"]`` and ``provenance.steps[-1].preset`` are ``{"name", "version"}``,
and ``func(data, **settings)`` gives the same result again.
"""

import dataclasses
import json
import warnings
from typing import Any

import pytest

import evoparam
from evoparam import presets, sim
from evoparam.errors import EvoparamVersionError


@pytest.fixture(scope="module")
def fitted() -> Any:
    with pytest.MonkeyPatch.context() as patch:
        # a fixed clock, so that the two provenance records can be compared whole
        patch.setenv("SOURCE_DATE_EPOCH", "1767225600")
        trials, _ = sim.simulate_ccep(n_trials=8, n_channels=3, sfreq=1024.0, seed=4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            group = evoparam.fit(trials, preset="ccep", min_fwhm=0.003)
            settings = group.settings
            again = evoparam.fit(
                trials, **{k: v for k, v in settings.items() if not k.startswith("_")}
            )
    return trials, group, again


def _without_overrides(group: Any) -> Any:
    """The GroupFit with every ``_overrides`` entry removed (settings of the group, of each
    fit and of the provenance steps): the keys typed by the user differ between the two
    calls by construction (one key, then all of them); everything else must be equal."""

    def clean(settings):
        return {k: v for k, v in settings.items() if k != "_overrides"}

    def clean_prov(prov):
        prov = prov.copy()
        prov.steps = [
            dict(step, settings=clean(step.get("settings") or {})) for step in prov.steps
        ]
        return prov

    fits = [
        dataclasses.replace(f, settings=clean(f.settings), provenance=clean_prov(f.provenance))
        for f in group.fits
    ]
    return dataclasses.replace(
        group, fits=fits, settings=clean(group.settings), provenance=clean_prov(group.provenance)
    )


def test_settings_log(fitted: Any) -> None:

    trials, group, again = fitted
    settings = group.settings
    assert settings["min_fwhm"] == 0.003
    assert settings["_overrides"] == ["min_fwhm"]
    assert settings["preset"] == {"name": "ccep", "version": presets.get("ccep").version}
    fit_step = next(
        step
        for step in group.provenance.steps
        if step["function"] == "evoparam.models.ResponseModel.fit_group"
    )
    assert fit_step["preset"] == settings["preset"]
    json.dumps(settings, allow_nan=False)
    assert all(f.settings["_overrides"] == ["min_fwhm"] for f in group.fits)
    # the full resolved dictionary: every field of the fit settings
    from evoparam.models import FitSettings

    assert set(FitSettings.__dataclass_fields__) <= set(settings)
    # refit with the recorded settings: an equal GroupFit
    assert _without_overrides(again) == _without_overrides(group)
    assert again.to_dataframe("component").equals(group.to_dataframe("component"))


def test_overrides_name_only_what_the_user_set(fitted: Any) -> None:

    trials, group, _ = fitted
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plain = evoparam.fit(trials, preset="ccep")
        two = evoparam.fit(trials, preset="ccep", max_components=2, n_starts=2)
    assert plain.settings["_overrides"] == []
    assert two.settings["_overrides"] == ["max_components", "n_starts"]
    # the methods text states the settings whose value differs from the preset
    assert group.settings["_changed_from_preset"] == []
    assert two.settings["_changed_from_preset"] == ["max_components", "n_starts"]


def test_recorded_preset_version_is_checked(fitted: Any) -> None:

    trials, group, _ = fitted
    settings = {k: v for k, v in group.settings.items() if not k.startswith("_")}
    settings["preset"] = {"name": "ccep", "version": "0.1"}
    with pytest.raises(EvoparamVersionError, match="v0.1"):
        evoparam.fit(trials, **settings)
