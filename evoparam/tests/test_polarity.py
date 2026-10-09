"""REQ-003 (polarity-agnostic): no code path in fitting, existence or trial modules discards
a candidate or component because of its sign unless the user sets ``polarity``; a test on
sign-flipped data returns mirrored results (latencies equal, amplitudes negated) to 1e-9."""

import ast
import pathlib
import warnings

import numpy as np
import pytest

import evoparam
from evoparam import sim
from evoparam.data import TrialSet

pytest.importorskip("pandas")

TOL = 1e-9
PACKAGE = pathlib.Path(evoparam.__file__).resolve().parent
SAME = ("latency", "fwhm", "onset", "offset", "latency_ci_low", "latency_ci_high",
        "fwhm_ci_low", "fwhm_ci_high", "existence_p")
NEGATED = (("amplitude", "amplitude"), ("area", "area"),
           ("amplitude_ci_low", "amplitude_ci_high"), ("amplitude_ci_high", "amplitude_ci_low"))


def flip(trials):
    return TrialSet(data=-trials.data, times=trials.times, sfreq=trials.sfreq,
                    ch_names=trials.ch_names, metadata=trials.metadata)


CASES = {"ccep": lambda: sim.simulate_ccep(n_trials=10, n_channels=4, seed=3, artifact=False),
         "erp": lambda: sim.simulate_erp(n_trials=30, n_channels=2, seed=1)}


@pytest.fixture(scope="module", params=sorted(CASES))
def pair(request):
    trials, _ = CASES[request.param]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = evoparam.fit(trials, preset=request.param, random_state=0)
        b = evoparam.fit(flip(trials), preset=request.param, random_state=0)
    return trials, a, b


def _close(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    both = np.isnan(x) & np.isnan(y)
    return np.all(both | (np.abs(x - y) <= TOL))


def test_req003_fit_of_flipped_data_is_mirrored(pair):
    _, a, b = pair
    ta, tb = a.to_dataframe(), b.to_dataframe()
    assert len(ta) == len(tb) and (ta["status"] == "ok").any()
    assert list(ta["status"]) == list(tb["status"])
    assert list(ta["component"]) == list(tb["component"])
    for column in SAME:
        assert _close(ta[column], tb[column]), column
    for left, right in NEGATED:
        assert _close(ta[left], -tb[right]), left
    assert list(ta["existence_significant"]) == list(tb["existence_significant"])


def test_req003_trial_fits_of_flipped_data_are_mirrored(pair):
    trials, a, b = pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ta = evoparam.trials.fit_trials(trials, a, random_state=0).to_dataframe()
        tb = evoparam.trials.fit_trials(flip(trials), b, random_state=0).to_dataframe()
    assert len(ta) == len(tb) > 0
    for column in ("shift", "latency", "fwhm", "r2"):
        assert _close(ta[column], tb[column]), column
    # the trial amplitude and its scale carry the sign of the data
    for column in ("scale", "amplitude"):
        assert _close(ta[column], -tb[column]), column
    assert list(ta["trusted"]) == list(tb["trusted"])


def test_req003_existence_test_of_flipped_data():
    trials, _ = CASES["ccep"]()
    kwargs = dict(window=(0.009, 0.5), baseline=(-0.5, -0.05), by=(), random_state=0)
    a = evoparam.test_response(trials, **kwargs)
    b = evoparam.test_response(flip(trials), **kwargs)
    for name in ("statistic", "p_value", "onset", "offset", "duration"):
        assert _close(getattr(a, name), getattr(b, name)), name
    assert np.array_equal(a.significant, b.significant)


def _sign_rules(path):
    """``if`` statements that drop something (continue, remove, pop) on a comparison of a sign
    (amplitude, beta, sign, polarity of a component) with zero, outside a test of the
    user's ``polarity`` setting."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        signed = any(isinstance(n, ast.Compare) and any(
            isinstance(c, ast.Constant) and c.value == 0 for c in [n.left, *n.comparators])
            for n in ast.walk(node.test)) or "np.sign(" in test
        sized = any(word in test for word in ("amplitude", "beta", "sign", "height", "peak"))
        drops = any(isinstance(n, ast.Continue) or (isinstance(n, ast.Call) and isinstance(
            n.func, ast.Attribute) and n.func.attr in ("remove", "pop"))
            for stmt in node.body for n in ast.walk(stmt))
        if not (signed and sized and drops):
            continue
        guarded, up = "polarity" in test, parents.get(node)
        while up is not None and not guarded:
            if isinstance(up, (ast.If, ast.While)) and "polarity" in ast.unparse(up.test):
                guarded = True
            up = parents.get(up)
        if not guarded:
            hits.append(f"{path.relative_to(PACKAGE.parent)}:{node.lineno}: if {test[:70]}")
    return hits


def test_req003_no_sign_rule_without_polarity():
    hits = []
    for folder in ("models", "existence", "trials"):
        for path in sorted((PACKAGE / folder).rglob("*.py")):
            hits += _sign_rules(path)
    assert not hits, hits


def test_req003_sign_rule_scan_finds_a_mutation(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def f(cands, polarity):\n"
                   "    for c in cands:\n"
                   "        if c.amplitude > 0:\n"
                   "            continue\n"
                   "    if polarity == 'negative':\n"
                   "        for c in cands:\n"
                   "            if c.amplitude > 0:\n"
                   "                continue\n")
    global PACKAGE
    saved, PACKAGE = PACKAGE, tmp_path
    try:
        assert len(_sign_rules(bad)) == 1
    finally:
        PACKAGE = saved
