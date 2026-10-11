# SPDX-License-Identifier: Apache-2.0
"""evoparam.fit with a preset, then the tidy table and the methods paragraph."""

import evoparam
from evoparam import sim
from evoparam.data import GroupFit, TrialSet


def analyse(trials: TrialSet) -> str:
    group: GroupFit = evoparam.fit(trials, preset="ccep", random_state=0)
    first = group.fits[0]
    latency: float = first.components[0].latency if first.components else float("nan")
    return f"{latency:.4f} s; " + group.methods_text()


def simulated() -> TrialSet:
    trials, _truth = sim.simulate_ccep(n_trials=10, n_channels=4, seed=0)
    return trials
