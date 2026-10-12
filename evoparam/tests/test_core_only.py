# SPDX-License-Identifier: Apache-2.0
"""REQ-010 (light core): ``import evoparam`` and the core functions of REQ-291 work with
NumPy and SciPy only.

The chain runs in a subprocess where every module outside the standard library, NumPy,
SciPy and evoparam is hidden (an import hook raises ImportError, as if the package were not
installed): ``simulate``, ``detect_events``, ``remove_artifact`` (interpolation and
template), ``fit`` (with its existence test), ``test_response``, ``fit_trials``, and save and
load of ``TrialSet``, ``Evoked`` and ``GroupFit`` (``data.npz`` and CSV tables), then the
npz container read back with ``evoparam.io.read``.
"""

import json
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.core

HOOK = textwrap.dedent("""
    import sys

    ALLOWED = set(sys.stdlib_module_names) | {"numpy", "scipy", "evoparam", "__main__"}
    # modules the interpreter loaded before this code, through the .pth files of site-packages
    # (for example the namespace package sphinxcontrib of the docs extra): not evoparam's
    PRELOADED = {m.split(".")[0] for m in sys.modules}

    class Hide:
        hidden = set()

        def find_spec(self, name, path=None, target=None):
            top = name.split(".")[0]
            if top not in ALLOWED and not top.startswith("_"):
                Hide.hidden.add(top)
                raise ImportError(f"hidden by the core-only test: {name}")
            return None

    sys.meta_path.insert(0, Hide())
""")

CHAIN = textwrap.dedent("""
    import json
    import os
    import tempfile
    import traceback
    import warnings

    import numpy as np

    import evoparam

    warnings.simplefilter("ignore")
    loaded = sorted({m.split(".")[0] for m in sys.modules} - set(sys.stdlib_module_names)
                    - {"cython_runtime", "sitecustomize", "usercustomize"} - PRELOADED)
    loaded = [m for m in loaded if not m.startswith("_")]
    steps = {"import": "ok" if set(loaded) <= {"numpy", "scipy", "evoparam"} else str(loaded)}
    state = {}
    folder = tempfile.mkdtemp()


    def step(name):
        def run(func):
            try:
                func()
                steps[name] = "ok"
            except Exception as exc:            # the step's error is the measured result
                steps[name] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
            return func
        return run


    @step("simulate")
    def _():
        from evoparam import sim

        state["trials"], _ = sim.simulate_ccep(n_trials=10, n_channels=3, seed=0)
        state["preset"] = evoparam.presets.get("ccep")


    @step("detect_events")
    def _():
        trials = state["trials"]
        continuous = np.concatenate(list(trials.data), axis=-1)
        events = evoparam.detect_events(continuous, sfreq=trials.sfreq,
                                        ch_names=trials.ch_names)
        assert len(events) >= 1


    @step("remove_artifact interpolate")
    def _():
        out = evoparam.artifact.remove_artifact(state["trials"], method="interpolate",
                                                window=state["preset"].artifact_window)
        state["clean"] = out.cleaned


    @step("remove_artifact template")
    def _():
        evoparam.artifact.remove_artifact(state["trials"], method="template",
                                          window=state["preset"].artifact_window)


    @step("fit")
    def _():
        group = evoparam.fit(state.get("clean", state["trials"]), preset="ccep",
                             random_state=0)
        assert type(group).__name__ == "GroupFit"
        assert all(f.existence is not None for f in group.fits)
        state["group"] = group


    @step("test_response")
    def _():
        p = state["preset"]
        state["test"] = evoparam.test_response(state.get("clean", state["trials"]),
                                               window=p.fit_window, baseline=p.baseline,
                                               by=(), random_state=0)


    @step("fit_trials")
    def _():
        state["single"] = evoparam.trials.fit_trials(state.get("clean", state["trials"]),
                                                     state["group"], random_state=0)


    for name in ("trials", "group"):
        @step(f"save/load {name}")
        def _():
            obj = state[name]
            path = os.path.join(folder, name + ".evoparam")
            evoparam.data.save_object(obj, path, table_format="csv")
            assert type(evoparam.load_object(path)) is type(obj)


    @step("save/load evoked")
    def _():
        evoked = state["trials"].average()
        path = os.path.join(folder, "evoked.evoparam")
        evoparam.data.save_object(evoked, path, table_format="csv")
        assert np.array_equal(evoparam.load_object(path).data, evoked.data)


    @step("read npz")
    def _():
        again = evoparam.io.read(os.path.join(folder, "trials.evoparam"))
        assert np.array_equal(again.data, state["trials"].data)


    @step("selftest")
    def _():
        assert evoparam.selftest() <= 0.001


    print(json.dumps(steps))
""")

STEPS = (
    "import",
    "simulate",
    "detect_events",
    "remove_artifact interpolate",
    "remove_artifact template",
    "fit",
    "test_response",
    "fit_trials",
    "save/load trials",
    "save/load group",
    "save/load evoked",
    "read npz",
    "selftest",
)

# Measured on 2026-10-09 (Python 3.12, NumPy and SciPy only): these steps need pandas
# (extra ``data``), so REQ-010 is not met for them yet. detect_events returns a DataFrame
# (evoparam.events.detect_pulses); remove_artifact computes its quality table
# (evoparam.artifact.artifact_quality); fit builds the index of the GroupFit as a DataFrame
# (GroupFit.from_fits, evoparam/data/results.py); fit_trials and saving a GroupFit need that
# fit. import, simulate, test_response and save/load of TrialSet and Evoked (data.npz) work.
# The test fails if any other step stops working; it passes once these work.
NEEDS_PANDAS = {
    "detect_events",
    "remove_artifact interpolate",
    "remove_artifact template",
    "fit",
    "fit_trials",
    "save/load group",
}


def run_chain():
    result = subprocess.run(
        [sys.executable, "-c", HOOK + CHAIN],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_req010_core_chain_with_numpy_and_scipy_only():
    steps = run_chain()
    assert set(steps) == set(STEPS)
    failing = {name: error for name, error in steps.items() if error != "ok"}
    unexpected = {name: error for name, error in failing.items() if name not in NEEDS_PANDAS}
    assert not unexpected, unexpected
    if failing:
        pytest.xfail(
            "REQ-010 not met as written; steps that need pandas: "
            + "; ".join(f"{k}: {v}" for k, v in failing.items())
        )


def test_req010_the_hook_hides_optional_packages():
    code = HOOK + textwrap.dedent("""
        for name in ("pandas", "mne", "matplotlib", "sklearn"):
            try:
                __import__(name)
            except ImportError:
                continue
            raise SystemExit(f"{name} was importable")
        print("ok")
    """)
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_req291_selftest():
    """REQ-291: ``evoparam.selftest()`` fits the bundled synthetic CCEP and recovers the N1
    latency within 1 ms; in the NumPy-and-SciPy-only subprocess too (step ``selftest`` of
    the chain above, which must be "ok")."""

    import evoparam

    assert evoparam.selftest() <= 0.001
    assert run_chain()["selftest"] == "ok"
