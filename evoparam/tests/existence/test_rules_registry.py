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

COMPLETE = {"vanblooijs2023": {"baseline": (-0.5, -0.05)},
            "hays2021": {"baseline": (-0.5, -0.05)},
            "keller2014": {},
            "silverstein2020": {"baseline": (-0.5, -0.05)},
            "ftract": {"k": 5.0},
            "prime2020": {"k": 3.0, "baseline": (-0.5, -0.05)}}


def _evoked_at(spec, value):
    """A waveform whose rule statistic equals ``value`` exactly (baseline SD 1, mean 0)."""

    base = (TIMES >= spec["baseline"][0]) & (TIMES <= spec["baseline"][1])
    y = np.zeros(len(TIMES))
    n = base.sum()
    y[np.flatnonzero(base)] = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    y[base] -= y[base].mean()
    y[base] /= y[base].std()
    win = np.flatnonzero((TIMES >= spec["window"][0]) & (TIMES <= spec["window"][1]))
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
        test = run_test(evoked, method="threshold", rule=name,
                             rule_overrides=COMPLETE[name])
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
    custom = {"statistic": "abs_peak_z", "k": 4.0, "window": (0.0, 0.05),
              "baseline": (-0.5, -0.05)}
    assert run_test(evoked, method="threshold", rule=custom).significant[0]
