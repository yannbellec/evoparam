# SPDX-License-Identifier: Apache-2.0
"""Artifact removal, then trial-by-trial fits."""

import evoparam
from evoparam.artifact import remove_artifact
from evoparam.data import ArtifactResult, TrialSet
from evoparam.trials import fit_trials


def clean(trials: TrialSet) -> TrialSet:
    result: ArtifactResult = remove_artifact(trials, method="interpolate",
                                             window=(-0.002, 0.006))
    return result.cleaned


def jitter(trials: TrialSet) -> object:
    group = evoparam.fit(trials, preset="ccep", random_state=0)
    return fit_trials(trials, group, random_state=0)
