# SPDX-License-Identifier: Apache-2.0
"""The lean trust-region reflective optimiser of the engine (models/_trf.py) takes the steps of
scipy.optimize.least_squares(method="trf", x_scale="jac"): the same iterates, to rounding."""

import numpy as np
import pytest
from scipy.optimize import least_squares

from evoparam.models._trf import least_squares_trf

###################################################################################################
###################################################################################################


def _problem(seed):
    """Two Gaussians and an exponential tail on 400 samples, bounded parameters."""

    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 400)
    truth = np.array([0.3, 0.05, 0.6, 0.1, 0.2])

    def model(p):
        return (
            np.exp(-0.5 * ((t - p[0]) / p[1]) ** 2)
            - 0.7 * np.exp(-0.5 * ((t - p[2]) / p[3]) ** 2)
            + 0.5 * np.exp(-t / p[4])
        )

    def jac(p):
        g1 = np.exp(-0.5 * ((t - p[0]) / p[1]) ** 2)
        g2 = np.exp(-0.5 * ((t - p[2]) / p[3]) ** 2)
        z1, z2 = (t - p[0]) / p[1], (t - p[2]) / p[3]
        e = np.exp(-t / p[4])
        return np.column_stack(
            [
                g1 * z1 / p[1],
                g1 * z1**2 / p[1],
                -0.7 * g2 * z2 / p[3],
                -0.7 * g2 * z2**2 / p[3],
                0.5 * e * t / p[4] ** 2,
            ]
        )

    y = model(truth) + rng.normal(0, 0.05, len(t))
    lower = np.array([0.0, 0.01, 0.0, 0.01, 0.01])
    upper = np.array([1.0, 0.3, 1.0, 0.3, 0.25])
    x0 = np.clip(truth * rng.uniform(0.6, 1.4, 5), lower + 1e-3, upper - 1e-3)
    return (lambda p: model(p) - y), jac, x0, lower, upper


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("tol", [1e-8, 1e-4])
def test_same_iterates_as_scipy(seed, tol):

    fun, jac, x0, lower, upper = _problem(seed)
    ref = least_squares(
        fun,
        x0,
        jac=jac,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        ftol=tol,
        xtol=tol,
        gtol=tol,
        max_nfev=2000,
    )
    for reduced in (False, True):
        x, cost, nfev, status = least_squares_trf(
            fun,
            jac,
            x0,
            lower,
            upper,
            ftol=tol,
            xtol=tol,
            gtol=tol,
            max_nfev=2000,
            reduced=reduced,
        )
        assert nfev == ref.nfev and status == ref.status
        assert np.allclose(x, ref.x, rtol=1e-10, atol=0)
        assert np.isclose(cost, ref.cost, rtol=1e-12)


def test_errors_as_scipy():

    fun, jac, x0, lower, upper = _problem(0)
    with pytest.raises(ValueError, match="outside of provided bounds"):
        least_squares_trf(fun, jac, upper + 1, lower, upper)
    with pytest.raises(ValueError, match="not finite"):
        least_squares_trf(
            lambda p: np.full(3, np.nan), lambda p: np.ones((3, 5)), x0, lower, upper
        )
