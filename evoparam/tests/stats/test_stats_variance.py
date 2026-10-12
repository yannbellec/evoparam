# SPDX-License-Identifier: Apache-2.0
"""REML variance components of nested random intercepts (dependability, and the hierarchical
models of §5.8 that build on them)."""

import numpy as np
import pandas as pd
import pytest

from evoparam.errors import DataError, SettingsError
from evoparam.stats import reml, variance_components

###################################################################################################
###################################################################################################


def _dense(y, z_list, variances, residual, se2, xmat):
    """Independent dense computation: V, beta, BLUPs and prediction-error variances."""

    n = len(y)
    gmat = np.diag(
        np.concatenate(
            [np.full(zmat.shape[1], v) for zmat, v in zip(z_list, variances, strict=False)]
        )
    )
    zmat = np.hstack(z_list)
    vmat = zmat @ gmat @ zmat.T + np.diag(residual + se2)
    vmat_inv = np.linalg.inv(vmat)
    cmat = np.linalg.inv(xmat.T @ vmat_inv @ xmat)
    beta = cmat @ xmat.T @ vmat_inv @ y
    pmat = vmat_inv - vmat_inv @ xmat @ cmat @ xmat.T @ vmat_inv
    blup = gmat @ zmat.T @ vmat_inv @ (y - xmat @ beta)
    pev = np.diag(gmat - gmat @ zmat.T @ pmat @ zmat @ gmat)
    sign, logdet = np.linalg.slogdet(vmat)
    r = y - xmat @ beta
    loglik = -0.5 * (
        logdet
        + np.linalg.slogdet(xmat.T @ vmat_inv @ xmat)[1]
        + r @ vmat_inv @ r
        + (n - xmat.shape[1]) * np.log(2 * np.pi)
    )
    return beta, blup, pev, loglik


def _indicators(codes):
    return (codes[:, None] == np.unique(codes)[None, :]).astype(float)


def test_one_level_matches_dense_computation():

    rng = np.random.default_rng(0)
    sizes = rng.integers(3, 25, 18)
    subject = np.repeat(np.arange(18), sizes)
    y = 2.0 + 1.5 * rng.normal(size=18)[subject] + rng.normal(size=subject.size)
    fast = reml(y, [subject])  # closed-form profile
    general = reml(y, [subject], se=np.zeros(y.size))  # numerical optimisation
    assert general["variances"][0] == pytest.approx(fast["variances"][0], rel=1e-4)
    assert general["residual"] == pytest.approx(fast["residual"], rel=1e-4)
    xmat = np.ones((y.size, 1))
    beta, blup, pev, loglik = _dense(
        y, [_indicators(subject)], fast["variances"], fast["residual"], np.zeros(y.size), xmat
    )
    assert fast["beta"][0] == pytest.approx(beta[0], rel=1e-9)
    assert np.allclose(fast["blup"][0], blup, atol=1e-9)
    assert np.allclose(fast["blup_se"][0], np.sqrt(pev), atol=1e-9)
    assert fast["loglik"] == pytest.approx(loglik, rel=1e-9)
    # the REML maximum: nudging either variance lowers the restricted likelihood
    for factor in (0.9, 1.1):
        for which in (0, 1):
            v = list(fast["variances"]) + [fast["residual"]]
            v[which] *= factor
            assert (
                _dense(y, [_indicators(subject)], v[:1], v[1], np.zeros(y.size), xmat)[3] < loglik
            )


def test_nested_levels_and_known_sampling_variances():
    """Channels within subjects, observations with known standard errors (the two-level
    meta-analytic model of the empirical-Bayes hierarchical engine)."""

    rng = np.random.default_rng(1)
    n_sub, n_ch = 40, 25
    subject = np.repeat(np.arange(n_sub), n_ch)
    channel = np.tile(np.arange(n_ch), n_sub)  # names repeat across subjects
    se = rng.uniform(0.5, 2.0, subject.size)
    y = (
        10
        + 2.0 * rng.normal(size=n_sub)[subject]
        + 1.0 * rng.normal(size=subject.size)
        + se * rng.normal(size=subject.size)
    )
    out = reml(y, [subject, channel], se=se, residual=False)
    assert np.sqrt(out["variances"][0]) == pytest.approx(2.0, rel=0.2)
    assert np.sqrt(out["variances"][1]) == pytest.approx(1.0, rel=0.2)
    assert out["residual"] == 0.0
    assert out["converged"]
    assert len(out["labels"][1]) == n_sub * n_ch
    zmats = [_indicators(subject), _indicators(subject * 1000 + channel)]
    beta, blup, pev, loglik = _dense(y, zmats, out["variances"], 0.0, se**2, np.ones((y.size, 1)))
    assert out["beta"][0] == pytest.approx(beta[0], rel=1e-8)
    assert np.allclose(np.concatenate(out["blup"]), blup, atol=1e-7)
    assert np.allclose(np.concatenate(out["blup_se"]), np.sqrt(pev), atol=1e-7)
    assert out["loglik"] == pytest.approx(loglik, rel=1e-9)


def test_table_interface_with_fixed_effects():

    rng = np.random.default_rng(2)
    subject = np.repeat([f"s{i}" for i in range(30)], 20)
    condition = np.tile(["a", "b"], 300)
    level = dict(zip(np.unique(subject), 3 * rng.normal(size=30), strict=False))
    y = (
        np.array([level[s] for s in subject])
        + np.where(condition == "b", 2.0, 0.0)
        + 2 * rng.normal(size=600)
    )
    df = pd.DataFrame({"subject": subject, "condition": condition, "amp": y})
    df.loc[5, "amp"] = np.nan
    vc = variance_components(df, value="amp", groups=["subject"], fixed=["condition"])
    assert vc.table["level"].tolist() == ["subject", "residual"]
    assert vc.table["fraction"].sum() == pytest.approx(1.0)
    assert vc.variance("subject") == pytest.approx(9.0, rel=0.5)
    b = vc.fixed.set_index("term").loc["condition_b"]
    assert b["estimate"] == pytest.approx(2.0, abs=3 * b["se"])
    effects = vc.random_effects["subject"]
    assert len(effects) == 30
    assert (effects["se"] > 0).all()
    assert list(effects.columns) == [
        "subject",
        "n_obs",
        "estimate",
        "se",
        "shrunken",
        "shrunken_se",
        "shrinkage",
    ]
    assert effects["n_obs"].sum() == 599
    assert effects["shrinkage"].between(0, 1).all()
    assert vc.settings["n_excluded"] == 1
    assert vc.provenance.steps[-1]["function"] == "evoparam.stats.variance_components"
    with pytest.raises(KeyError):
        vc.variance("channel")
    with pytest.raises(SettingsError, match="missing"):
        variance_components(df, value="latency", groups=["subject"])


def test_boundary_and_errors():

    rng = np.random.default_rng(3)
    subject = np.repeat(np.arange(20), 10)
    y = rng.normal(size=200)  # no subject variance
    out = reml(y, [subject])
    assert out["variances"][0] < 0.05
    with pytest.raises(SettingsError):
        reml(y, [subject], residual=False)
    with pytest.raises(DataError):
        reml(np.r_[y[:-1], np.nan], [subject])
    with pytest.raises(DataError):
        reml(y, [subject[:-1]])


def test_shrunken_unit_estimates_match_henderson_equations():
    """The shrunken subject and channel estimates and their standard errors (the closed-form
    outputs of the empirical-Bayes hierarchical engine) against the inverse of Henderson's
    mixed-model equations, with a subject covariate, known se and a residual variance."""

    rng = np.random.default_rng(4)
    n_sub, n_ch = 12, 6
    sizes = rng.integers(1, 6, n_sub * n_ch)  # trials per channel, unbalanced
    pair = np.repeat(np.arange(n_sub * n_ch), sizes)
    subject, channel = pair // n_ch, pair % n_ch
    age = rng.uniform(5, 30, n_sub)
    xmat = np.column_stack([np.ones(pair.size), age[subject]])
    se = rng.uniform(0.5, 1.5, pair.size)
    y = (
        xmat @ [20.0, -0.3]
        + 1.5 * rng.normal(size=n_sub)[subject]
        + 1.0 * rng.normal(size=n_sub * n_ch)[pair]
        + 0.7 * rng.normal(size=pair.size)
        + se * rng.normal(size=pair.size)
    )
    out = reml(y, [subject, channel], se=se, X=xmat)
    assert all(v > 0 for v in out["variances"])
    assert out["residual"] > 0

    zs, zc = _indicators(subject), _indicators(pair)
    zmat = np.hstack([zs, zc])
    gmat = np.r_[np.full(n_sub, out["variances"][0]), np.full(n_sub * n_ch, out["variances"][1])]
    r_inv = 1.0 / (out["residual"] + se**2)
    mme = np.block(
        [
            [xmat.T @ (xmat * r_inv[:, None]), xmat.T @ (zmat * r_inv[:, None])],
            [
                zmat.T @ (xmat * r_inv[:, None]),
                zmat.T @ (zmat * r_inv[:, None]) + np.diag(1 / gmat),
            ],
        ]
    )
    cmat = np.linalg.inv(mme)
    sol = cmat @ np.r_[xmat.T @ (y * r_inv), zmat.T @ (y * r_inv)]
    assert np.allclose(sol[:2], out["beta"], rtol=1e-8)
    assert np.allclose(sol[2:], np.concatenate(out["blup"]), atol=1e-8)

    # subject s: mean design row of its observations, plus u_s
    l_sub = np.hstack(
        [zs.T @ xmat / zs.sum(0)[:, None], np.eye(n_sub), np.zeros((n_sub, n_sub * n_ch))]
    )
    # channel c of subject s: mean design row, plus u_s and v_sc
    l_ch = np.hstack(
        [zc.T @ xmat / zc.sum(0)[:, None], (zc.T @ zs > 0).astype(float), np.eye(n_sub * n_ch)]
    )
    for level, lmat in enumerate((l_sub, l_ch)):
        assert np.allclose(out["unit_mean"][level], lmat @ sol, atol=1e-8)
        se_dense = np.sqrt(np.einsum("ij,jk,ik->i", lmat, cmat, lmat))
        assert np.allclose(out["unit_mean_se"][level], se_dense, atol=1e-8)
        assert out["n_obs"][level].sum() == y.size
    # shrinkage: 1 - Var(u | y, beta) / sigma^2, the conditional part of Henderson's inverse
    zr = zmat * np.sqrt(r_inv)[:, None]
    cond = np.diag(np.linalg.inv(zr.T @ zr + np.diag(1 / gmat)))
    assert np.allclose(np.concatenate(out["shrinkage"]), 1 - cond / gmat, atol=1e-8)


def test_one_level_shrinkage_is_the_empirical_bayes_weight():
    """One level with known se: the shrunken estimate is mu + B (y - mu) with
    B = tau^2 / (tau^2 + se^2), and the shrinkage factor is B."""

    rng = np.random.default_rng(5)
    se = rng.uniform(0.5, 3.0, 50)
    y = 4.0 + 2.0 * rng.normal(size=50) + se * rng.normal(size=50)
    out = reml(y, [np.arange(50)], se=se, residual=False)
    tau2 = out["variances"][0]
    shrink_b = tau2 / (tau2 + se**2)
    mu = out["beta"][0]
    assert np.allclose(out["shrinkage"][0], shrink_b, atol=1e-10)
    assert np.allclose(out["unit_mean"][0], mu + shrink_b * (y - mu), atol=1e-10)
    # the classical EB standard error: sqrt(tau^2 (1 - B) + (1 - B)^2 Var(mu_hat))
    var_mu = 1.0 / np.sum(1.0 / (tau2 + se**2))
    assert np.allclose(
        out["unit_mean_se"][0],
        np.sqrt(tau2 * (1 - shrink_b) + (1 - shrink_b) ** 2 * var_mu),
        atol=1e-10,
    )
