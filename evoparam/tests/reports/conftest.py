"""Results of every type, made once for the report and plot tests."""

import warnings

import numpy as np
import pytest


@pytest.fixture(scope="session")
def results():
    import matplotlib

    matplotlib.use("Agg")
    from evoparam.artifact import remove_artifact
    from evoparam.existence import test_response
    from evoparam.models import fit
    from evoparam.spatial import shared_shapes
    from evoparam.tests.trials.helpers import simulate
    from evoparam.tf import tf_fit
    from evoparam.trials import fit_trials

    ts, _, _ = simulate(n=20, channels=3, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = fit(ts, "ccep")
        test = test_response(ts, n_permutations=50, random_state=0)
        trials = fit_trials(ts, group)
        tf = tf_fit(ts, freqs=(4, 200, 12), zones="auto", random_state=0)
        artifact = remove_artifact(ts, method="interpolate", window=(-0.002, 0.008))
        try:
            basis = shared_shapes(ts, random_state=0)
        except Exception:
            basis = None
    return {"ts": ts, "group": group, "fit": group.fits[0], "test": test, "trials": trials,
            "tf": tf, "artifact": artifact, "basis": basis}
