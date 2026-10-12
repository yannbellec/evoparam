# SPDX-License-Identifier: Apache-2.0
"""The registry of published rules (REQ-133, REQ-137)."""

import numpy as np
import pytest

from evoparam.data import Evoked
from evoparam.errors import SettingsError
from evoparam.existence import detectors, list_rules, resolve_rule
from evoparam.existence import test_response as run_test

from .helpers import SFREQ, TIMES

###################################################################################################
###################################################################################################

COMPLETE = {
    "vanblooijs2023": {"baseline": (-0.5, -0.05)},
    "vanblooijs2023_code": {"baseline": (-0.5, -0.05)},
    "hays2021": {"baseline": (-0.5, -0.05)},
    "keller2014": {},
    "silverstein2020": {"baseline": (-0.5, -0.05)},
    "ftract": {"k": 5.0},
    "prime2020": {"k": 3.0, "baseline": (-0.5, -0.05)},
}


def _evoked_at(spec, value):
    """A waveform whose rule statistic equals ``value`` exactly (baseline SD 1, mean 0)."""

    base = (spec["baseline"][0] <= TIMES) & (spec["baseline"][1] >= TIMES)
    y = np.zeros(len(TIMES))
    n = base.sum()
    y[np.flatnonzero(base)] = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    y[base] -= y[base].mean()
    y[base] /= y[base].std()
    win = np.flatnonzero((spec["window"][0] <= TIMES) & (spec["window"][1] >= TIMES))
    kind = spec["statistic"]
    if kind == "rms_z":
        y[win] = value
    elif kind in ("negative_peak_over_sd", "negative_peak_z"):
        y[win[len(win) // 2]] = -value
    else:
        y[win[len(win) // 2]] = value
    return Evoked(data=y[None], times=TIMES, sfreq=SFREQ, ch_names=["c0"], units="au")


@pytest.mark.parametrize("name", [n for n in COMPLETE if n != "ftract"])
def test_req133_decision_flips_at_threshold(name):

    spec = resolve_rule(name, **COMPLETE[name])
    for delta, expected in ((1e-6, True), (-1e-6, False)):
        evoked = _evoked_at(spec, spec["k"] + delta)
        test = run_test(evoked, method="threshold", rule=name, rule_overrides=COMPLETE[name])
        assert bool(test.significant[0]) is expected, (name, delta)


def test_req137_unspecified_fields_raise():

    with pytest.raises(SettingsError, match=r"van Blooijs.*\['baseline'\]"):
        resolve_rule("vanblooijs2023")
    with pytest.raises(SettingsError, match=r"\['k'\]"):
        resolve_rule("ftract")
    with pytest.raises(SettingsError, match="Unknown rule"):
        resolve_rule("nope")
    table = list_rules()
    assert set(table["name"]) == set(COMPLETE)


def test_detectors_namespace_and_custom_rule():

    evoked = _evoked_at(resolve_rule("keller2014"), 10.0)
    assert detectors.keller2014(evoked).significant[0]
    custom = {
        "statistic": "abs_peak_z",
        "k": 4.0,
        "window": (0.0, 0.05),
        "baseline": (-0.5, -0.05),
    }
    assert run_test(evoked, method="threshold", rule=custom).significant[0]


def _uv(y):
    y = np.atleast_2d(y)
    return Evoked(
        data=y, times=TIMES, sfreq=SFREQ, ch_names=[f"c{i}" for i in range(len(y))], units="uV"
    )


def test_selective_maxima_matches_the_definition():

    from evoparam.existence.rules import selective_maxima

    x = np.array([0.0, 5.0, 4.0, 4.5, 0.0, 10.0, 9.0, 12.0, 0.0])
    # 4.5 does not rise 2 above its valley (4.0); 10 does not fall 2 before 12
    assert selective_maxima(x, 2.0).tolist() == [1, 7]
    assert selective_maxima(x, 0.0).tolist() == [1, 3, 5, 7]
    # a monotone signal: only an end point, when it stands out by more than sel
    assert selective_maxima(np.arange(5.0), 1.0).tolist() == [4]
    assert selective_maxima(np.arange(5.0), 10.0).tolist() == []


def test_vanblooijs2023_default_is_the_registry_row():
    """F65: the default rule is the row of spec §5.6 as written (no SD floor, the window's
    most negative sample); the published code's fields belong to 'vanblooijs2023_code'."""

    paper = resolve_rule("vanblooijs2023", baseline=(-0.5, -0.05))
    assert paper["sd_floor_uv"] is None
    assert paper["peak"] == "extremum"
    assert not paper.get("exclude_first_sample")
    assert not paper.get("saturation_uv")
    assert (paper["k"], paper["window"], paper["polarity"]) == (3.4, (0.009, 0.1), "negative")
    code = resolve_rule("vanblooijs2023_code")
    assert code["baseline"] == (-2.0, -0.1)
    assert code["sd_floor_uv"] == 50.0
    assert code["peak"] == "local"
    assert code["exclude_first_sample"]
    assert code["saturation_uv"] == 3000.0
    assert code["peak_search_end"] == 0.5


def test_vanblooijs2023_published_code_reading():
    """'vanblooijs2023_code' reads the N1 as the published code does (mnl_ccepAge): a
    monotone tail through the window start or a drift through its end is not a peak; a
    trough is; the 50 uV SD floor leaves out a small trough that the registry row keeps."""

    from .helpers import gaussian

    base = (-0.5, -0.05)
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 10, (4, len(TIMES)))
    noise[:, TIMES > 0] = 0.0
    tail = np.where(TIMES > 0, -600 * np.exp(-TIMES / 0.01), 0.0)
    drift = np.where(TIMES > 0, -400 * np.clip((TIMES - 0.06) / 0.04, 0, None), 0.0)
    trough = -300 * gaussian(0.025, 0.012)
    small = -100 * gaussian(0.025, 0.012)  # under the 170 uV floor
    evoked = _uv(noise + np.array([tail, drift, trough, small]))
    local = run_test(
        evoked, method="threshold", rule="vanblooijs2023_code", rule_overrides={"baseline": base}
    )
    extremum = run_test(
        evoked, method="threshold", rule="vanblooijs2023", rule_overrides={"baseline": base}
    )
    floored = run_test(
        evoked,
        method="threshold",
        rule="vanblooijs2023",
        rule_overrides={"baseline": base, "sd_floor_uv": 50.0},
    )
    assert local.significant.tolist() == [False, False, True, False]
    assert extremum.significant.tolist() == [True, True, True, True]
    assert floored.significant.tolist() == [True, True, True, False]
    assert abs(local.onset[2] - 0.025) <= 1.0 / SFREQ
    # the same reading through the override of the default rule
    override = run_test(
        evoked,
        method="threshold",
        rule="vanblooijs2023",
        rule_overrides={
            "baseline": base,
            "peak": "local",
            "sd_floor_uv": 50.0,
            "exclude_first_sample": True,
            "saturation_uv": 3000.0,
            "peak_search_end": 0.5,
            "peak_selectivity_uv": 20.0,
        },
    )
    assert override.significant.tolist() == local.significant.tolist()
    # a trough beyond 3000 uV is refused as saturated, as in the published code
    big = run_test(
        _uv(noise[0] - 4000 * gaussian(0.025, 0.012)),
        method="threshold",
        rule="vanblooijs2023_code",
        rule_overrides={"baseline": base},
    )
    assert not big.significant[0]
    with pytest.raises(SettingsError, match="peak"):
        resolve_rule("vanblooijs2023", baseline=base, peak="global")
