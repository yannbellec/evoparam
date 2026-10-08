"""Results of every type, made once for the report and plot tests."""

import warnings

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


@pytest.fixture(scope="session")
def erp_run():
    """The ERP CORE P3 reference pipeline on its stand-in input (1 channel, about 1 s)."""

    from evoparam import reports

    return reports.run_reference_pipeline("erpcore_p3")


@pytest.fixture(scope="session")
def ccep_run():
    """A small CCEP pipeline: artifact removal, reference, baseline, fit, existence test and
    trial fits on 3 simulated channels."""

    from evoparam import reports, sim

    trials, _ = sim.simulate_ccep(n_trials=8, n_channels=3, sfreq=1024.0, seed=1)
    with warnings.catch_warnings():
        # three channels are too few for the adjusted common average (plain average used)
        warnings.simplefilter("ignore", UserWarning)
        return reports.run_pipeline(trials, "ccep", trial_fits=True)
