# SPDX-License-Identifier: Apache-2.0
"""Full Bayesian engines of the hierarchical model (spec §5.8, step 3): numpyro and PyMC.

The same model in both engines, sampled with NUTS and initialised at the empirical-Bayes
solution. It is written on the statistics the empirical-Bayes engine uses (channel
estimates with their standard errors, trial latencies with their noise), not on the raw
samples of every trial: a waveform-level likelihood (spec §5.8, step 1) would make every
NUTS step evaluate every component at every sample of every trial.

Per component (latencies in ms, amplitudes in units of their channel SD ``A``)::

    latency:   L_s = mu_L + x_s' beta + s_LS z_s          (subjects; non-centred)
               L_c ~ N(L_s, s_LC)                          (channels; centred, as the data
                                                            pin each channel down: a
                                                            non-centred channel level gave
                                                            r_hat 1.02 with PyMC)
               Lhat_c ~ N(L_c, se_c)                       (fitted channel latency)
               l_ct ~ ExGaussian(L_c - tau, sqrt(s_d^2 + nu_c^2), tau)   (trusted trials;
                    nu_c the latency noise of the channel; Normal with latency_dist="normal")
    amplitude: a_s = mu_A + s_AS u_s;  a_c ~ N(a_s, s_AC);  ahat_c ~ N(a_c, se_c)
    existence: y_c ~ (1 - pi_s) N(0, e_c) + pi_s N(m_s, sqrt(t^2 + e_c^2)),
               m_s = mu_E + s_ES r_s,  pi_s ~ Beta(1, 1)
               (y_c, e_c: mean and standard error of the template amplitude of every
               channel; P(response) is the posterior mean of each channel's responsibility)

Weakly informative priors: centres at the empirical-Bayes values, half-normal scales of
the order of the data spread.
"""

import numpy as np

from evoparam.errors import warn as _warn
from evoparam.utils._optional import require

__all__ = ["require_engine", "sample", "loo_table", "prepare"]

EXTRAS = {"numpyro": "bayes", "pymc": "bayes"}
MIN_TAU_MS = 0.01
DIVERGENCE_WARNING = 0.01


def require_engine(engine):
    """Import the engine's packages, or raise ImportError naming what to install."""
    if engine == "numpyro":
        require("jax", extra=EXTRAS[engine], caller="engine='numpyro'")
        return require("numpyro", extra=EXTRAS[engine], caller="engine='numpyro'")
    return require("pymc", extra=EXTRAS[engine], caller="engine='pymc'")


###################################################################################################
## Data


def prepare(channel, trial, label, formula, covariates, subjects):
    """Arrays of one component for the samplers (scaled), and the scales."""
    from . import eb

    ch = channel[channel["component"] == label].reset_index(drop=True)
    sub_names = list(subjects)
    sub_index = {n: i for i, n in enumerate(sub_names)}
    use = (
        ch["has_component"]
        & ch["flag"].ne("prior_dominated")
        & np.isfinite(ch["latency_raw"])
        & np.isfinite(ch["latency_raw_se"])
        & (ch["latency_raw_se"] > 0)
        & np.isfinite(ch["amplitude_raw"])
        & np.isfinite(ch["amplitude_raw_se"])
        & (ch["amplitude_raw_se"] > 0)
    )
    fitted = ch[use].reset_index(drop=True)
    amp_scale = float(np.std(fitted["amplitude_raw"])) or 1.0
    data = {
        "n_subjects": len(sub_names),
        "ch_subject": np.array([sub_index[s] for s in fitted["subject"]]),
        "lat_hat": fitted["latency_raw"].to_numpy() * 1e3,
        "lat_se": fitted["latency_raw_se"].to_numpy() * 1e3,
        "amp_hat": fitted["amplitude_raw"].to_numpy() / amp_scale,
        "amp_se": fitted["amplitude_raw_se"].to_numpy() / amp_scale,
    }
    keys = list(zip(fitted["subject"], fitted["channel"]))
    lookup = {k: i for i, k in enumerate(keys)}
    tr = trial[trial["label"] == label] if len(trial) else trial
    rows = (
        [(lookup.get((s, c)), l) for s, c, l in zip(tr["subject"], tr["channel"], tr["latency"])]
        if len(tr)
        else []
    )
    rows = [(i, l) for i, l in rows if i is not None and np.isfinite(l)]
    noise = fitted["latency_noise_sd"].to_numpy() * 1e3
    data["trial_channel"] = np.array([i for i, _ in rows], int)
    data["trial_lat"] = np.array([l for _, l in rows], float) * 1e3
    data["trial_noise"] = (
        np.nan_to_num(noise[data["trial_channel"]], nan=np.nanmedian(noise))
        if rows
        else np.zeros(0)
    )
    exist = ch[
        np.isfinite(ch["existence_amplitude"])
        & np.isfinite(ch["existence_amplitude_se"])
        & (ch["existence_amplitude_se"] > 0)
    ]
    data["ex_subject"] = np.array([sub_index[s] for s in exist["subject"]])
    data["ex_y"] = exist["existence_amplitude"].to_numpy() / amp_scale
    data["ex_se"] = exist["existence_amplitude_se"].to_numpy() / amp_scale
    parameter, terms = formula
    if parameter == "latency" and terms:
        X, names = eb.design_matrix([covariates[s] for s in sub_names], terms)
        data["X"], data["X_names"] = X[:, 1:], names[1:]
    else:
        data["X"], data["X_names"] = np.zeros((len(sub_names), 0)), []
    data["keys"] = keys
    data["exist_keys"] = list(zip(exist["subject"], exist["channel"]))
    data["subjects"] = sub_names
    data["amp_scale"] = amp_scale
    lat_sd = float(np.std(data["lat_hat"])) or 1.0
    data["prior"] = {
        "lat_centre": float(np.median(data["lat_hat"])),
        "lat_scale": 5 * lat_sd,
        "jitter_scale": 5
        * max(
            float(np.median(data["lat_se"]))
            * np.sqrt(max(np.median(np.bincount(data["trial_channel"])) if rows else 1, 1)),
            0.5,
        ),
        "amp_centre": float(np.median(data["amp_hat"])),
    }
    return data


###################################################################################################
## numpyro


def _exgauss_logpdf_jax(x, mu, sigma, tau):
    import jax.numpy as jnp
    from jax.scipy.stats import norm

    return (
        -jnp.log(tau)
        + (mu - x) / tau
        + sigma**2 / (2 * tau**2)
        + norm.logcdf((x - mu) / sigma - sigma / tau)
    )


def _numpyro_model(data, latency_dist):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from jax.scipy.special import logsumexp

    p = data["prior"]
    n_s = data["n_subjects"]
    mu_l = numpyro.sample("mu_latency", dist.Normal(p["lat_centre"], p["lat_scale"]))
    s_ls = numpyro.sample("sd_latency_subject", dist.HalfNormal(p["lat_scale"]))
    s_lc = numpyro.sample("sd_latency_channel", dist.HalfNormal(p["lat_scale"]))
    n_x = data["X"].shape[1]
    beta = (
        numpyro.sample("beta", dist.Normal(0, p["lat_scale"]).expand([n_x]))
        if n_x
        else jnp.zeros(0)
    )
    z = numpyro.sample("z_subject", dist.Normal(0, 1).expand([n_s]))
    lat_s = mu_l + jnp.asarray(data["X"]) @ beta + s_ls * z
    numpyro.deterministic("latency_subject", lat_s)
    lat_c = numpyro.sample("latency_channel", dist.Normal(lat_s[data["ch_subject"]], s_lc))
    numpyro.sample("lat_hat", dist.Normal(lat_c, data["lat_se"]), obs=data["lat_hat"])

    s_d = numpyro.sample("sd_jitter", dist.HalfNormal(p["jitter_scale"]))
    if len(data["trial_lat"]):
        scale = jnp.sqrt(s_d**2 + data["trial_noise"] ** 2)
        centre = lat_c[data["trial_channel"]]
        if latency_dist == "exgaussian":
            tau = MIN_TAU_MS + numpyro.sample("tau_jitter_raw", dist.HalfNormal(p["jitter_scale"]))
            numpyro.deterministic("tau_jitter", tau)
            numpyro.factor(
                "trial_lat",
                jnp.sum(_exgauss_logpdf_jax(data["trial_lat"], centre - tau, scale, tau)),
            )
        else:
            numpyro.sample("trial_lat", dist.Normal(centre, scale), obs=data["trial_lat"])

    mu_a = numpyro.sample("mu_amplitude", dist.Normal(p["amp_centre"], 5.0))
    s_as = numpyro.sample("sd_amplitude_subject", dist.HalfNormal(2.0))
    s_ac = numpyro.sample("sd_amplitude_channel", dist.HalfNormal(2.0))
    u = numpyro.sample("u_subject", dist.Normal(0, 1).expand([n_s]))
    a_s = mu_a + s_as * u
    numpyro.deterministic("amplitude_subject", a_s)
    a_c = numpyro.sample("amplitude_channel", dist.Normal(a_s[data["ch_subject"]], s_ac))
    numpyro.sample("amp_hat", dist.Normal(a_c, data["amp_se"]), obs=data["amp_hat"])

    if len(data["ex_y"]):
        mu_e = numpyro.sample("mu_existence", dist.Normal(p["amp_centre"], 5.0))
        s_es = numpyro.sample("sd_existence_subject", dist.HalfNormal(2.0))
        t_e = numpyro.sample("sd_existence_channel", dist.HalfNormal(2.0))
        r = numpyro.sample("r_subject", dist.Normal(0, 1).expand([n_s]))
        pi = numpyro.sample("pi", dist.Beta(1.0, 1.0).expand([n_s]))
        m = (mu_e + s_es * r)[data["ex_subject"]]
        pis = pi[data["ex_subject"]]
        se = data["ex_se"]
        l1 = jnp.log(pis) + dist.Normal(m, jnp.sqrt(t_e**2 + se**2)).log_prob(data["ex_y"])
        l0 = jnp.log1p(-pis) + dist.Normal(0.0, se).log_prob(data["ex_y"])
        total = logsumexp(jnp.stack([l1, l0]), axis=0)
        numpyro.factor("existence", jnp.sum(total))
        numpyro.deterministic("responsibility", jnp.exp(l1 - total))


EB_KEYS = {
    ("latency", "subject"): ("sd_latency_subject", 1e3),
    ("latency", "channel"): ("sd_latency_channel", 1e3),
    ("latency", "trial"): ("sd_jitter", 1e3),
    ("amplitude", "subject"): ("sd_amplitude_subject", None),
    ("amplitude", "channel"): ("sd_amplitude_channel", None),
}


def _init_values(data, eb_variance, label):
    """Starting point: the empirical-Bayes means and standard deviations."""
    out = {"mu_latency": data["prior"]["lat_centre"], "mu_amplitude": data["prior"]["amp_centre"]}
    if eb_variance is None or not len(eb_variance):
        return out
    rows = eb_variance[eb_variance["component"] == label]
    for (parameter, level), (key, scale) in EB_KEYS.items():
        match = rows[(rows["parameter"] == parameter) & (rows["level"] == level)]
        if len(match):
            value = float(match["sd"].iloc[0]) * (scale or 1.0 / data["amp_scale"])
            # a zero SD is on the boundary of the half-normal: start slightly inside
            out[key] = max(value, 1e-3 * (data["prior"]["lat_scale"] if scale else 1.0))
    return out


def _sample_numpyro(data, latency_dist, n_chains, n_warmup, n_draws, target_accept, seed, init):
    import jax
    from numpyro.infer import MCMC, NUTS, init_to_value

    jax.config.update("jax_enable_x64", True)
    kernel = NUTS(
        lambda: _numpyro_model(data, latency_dist),
        target_accept_prob=target_accept,
        init_strategy=init_to_value(values=init),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=n_warmup,
        num_samples=n_draws,
        num_chains=n_chains,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(int(seed)), extra_fields=("diverging",))
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    diverging = np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"])
    loglik = None
    try:
        from numpyro.infer import log_likelihood

        flat = mcmc.get_samples()
        ll = log_likelihood(lambda: _numpyro_model(data, latency_dist), flat)
        loglik = {k: np.asarray(v).reshape(n_chains, n_draws, -1) for k, v in ll.items()}
    except Exception:  # log-likelihood is only needed for LOO
        loglik = None
    return samples, diverging, loglik


###################################################################################################
## PyMC


def _sample_pymc(data, latency_dist, n_chains, n_warmup, n_draws, target_accept, seed, init):
    import pymc as pm
    import pytensor.tensor as pt

    p = data["prior"]
    n_s = data["n_subjects"]
    with pm.Model():
        mu_l = pm.Normal("mu_latency", p["lat_centre"], p["lat_scale"])
        s_ls = pm.HalfNormal("sd_latency_subject", p["lat_scale"])
        s_lc = pm.HalfNormal("sd_latency_channel", p["lat_scale"])
        n_x = data["X"].shape[1]
        beta = pm.Normal("beta", 0, p["lat_scale"], shape=n_x) if n_x else pt.zeros(0)
        z = pm.Normal("z_subject", 0, 1, shape=n_s)
        lat_s = pm.Deterministic(
            "latency_subject", mu_l + pt.dot(np.asarray(data["X"]), beta) + s_ls * z
        )
        lat_c = pm.Normal(
            "latency_channel", lat_s[data["ch_subject"]], s_lc, shape=len(data["lat_hat"])
        )
        pm.Normal("lat_hat", lat_c, data["lat_se"], observed=data["lat_hat"])
        s_d = pm.HalfNormal("sd_jitter", p["jitter_scale"])
        if len(data["trial_lat"]):
            scale = pt.sqrt(s_d**2 + data["trial_noise"] ** 2)
            centre = lat_c[data["trial_channel"]]
            if latency_dist == "exgaussian":
                tau_raw = pm.HalfNormal("tau_jitter_raw", p["jitter_scale"])
                tau = pm.Deterministic("tau_jitter", MIN_TAU_MS + tau_raw)
                pm.ExGaussian(
                    "trial_lat", mu=centre - tau, sigma=scale, nu=tau, observed=data["trial_lat"]
                )
            else:
                pm.Normal("trial_lat", centre, scale, observed=data["trial_lat"])
        mu_a = pm.Normal("mu_amplitude", p["amp_centre"], 5.0)
        s_as = pm.HalfNormal("sd_amplitude_subject", 2.0)
        s_ac = pm.HalfNormal("sd_amplitude_channel", 2.0)
        u = pm.Normal("u_subject", 0, 1, shape=n_s)
        a_s = pm.Deterministic("amplitude_subject", mu_a + s_as * u)
        a_c = pm.Normal(
            "amplitude_channel", a_s[data["ch_subject"]], s_ac, shape=len(data["amp_hat"])
        )
        pm.Normal("amp_hat", a_c, data["amp_se"], observed=data["amp_hat"])
        if len(data["ex_y"]):
            mu_e = pm.Normal("mu_existence", p["amp_centre"], 5.0)
            s_es = pm.HalfNormal("sd_existence_subject", 2.0)
            t_e = pm.HalfNormal("sd_existence_channel", 2.0)
            r = pm.Normal("r_subject", 0, 1, shape=n_s)
            pi = pm.Beta("pi", 1.0, 1.0, shape=n_s)
            m = (mu_e + s_es * r)[data["ex_subject"]]
            pis = pi[data["ex_subject"]]
            se = data["ex_se"]
            l1 = pt.log(pis) + pm.logp(pm.Normal.dist(m, pt.sqrt(t_e**2 + se**2)), data["ex_y"])
            l0 = pt.log1p(-pis) + pm.logp(pm.Normal.dist(0.0, se), data["ex_y"])
            total = pt.logaddexp(l1, l0)
            pm.Potential("existence", pt.sum(total))
            pm.Deterministic("responsibility", pt.exp(l1 - total))
        idata = pm.sample(
            draws=n_draws,
            tune=n_warmup,
            chains=n_chains,
            cores=1,
            target_accept=target_accept,
            random_seed=int(seed),
            initvals={k: v for k, v in init.items()},
            progressbar=False,
            compute_convergence_checks=False,
            idata_kwargs={"log_likelihood": True},
        )
    post = idata.posterior
    samples = {k: np.asarray(post[k]) for k in post.data_vars}
    diverging = np.asarray(idata.sample_stats["diverging"])
    loglik = {
        k: np.asarray(idata.log_likelihood[k]).reshape(n_chains, n_draws, -1)
        for k in idata.log_likelihood.data_vars
    }
    return samples, diverging, loglik


###################################################################################################
## Summaries


def _summary(draws):
    """Mean, SD and 95 % interval over chains and draws (axis 0 and 1)."""
    flat = draws.reshape(-1, *draws.shape[2:])
    return (
        flat.mean(0),
        flat.std(0, ddof=1),
        np.percentile(flat, 2.5, axis=0),
        np.percentile(flat, 97.5, axis=0),
    )


def _from_dict(az, groups):
    """``InferenceData`` from ``{group: {variable: draws}}``: one dict of groups with
    arviz >= 1, one keyword per group with arviz 0.x (where a dict alone would be read as
    a single variable named after the group).
    """
    if int(str(az.__version__).split(".")[0]) >= 1:
        return az.from_dict(groups)
    return az.from_dict(**groups)


def _diagnostics(samples, diverging):
    import arviz as az

    names = [k for k in samples if not k.startswith(("responsibility",))]
    data = _from_dict(az, {"posterior": {k: samples[k] for k in names}})
    rhat = az.rhat(data)
    ess = az.ess(data)
    worst_rhat = max(float(np.nanmax(np.asarray(rhat[k]))) for k in names)
    worst_ess = min(float(np.nanmin(np.asarray(ess[k]))) for k in names)
    per = {
        k: {
            "r_hat_max": float(np.nanmax(np.asarray(rhat[k]))),
            "ess_min": float(np.nanmin(np.asarray(ess[k]))),
        }
        for k in names
    }
    return {
        "r_hat_max": worst_rhat,
        "ess_min": worst_ess,
        "divergences": int(diverging.sum()),
        "divergent_fraction": float(diverging.mean()),
        "per_parameter": per,
    }


def _ppc(samples, data):
    """Posterior-predictive p-value of the spread of the channel latencies."""
    rng = np.random.default_rng(0)
    lat_c = samples["latency_channel"].reshape(-1, len(data["lat_hat"]))
    rep = lat_c + rng.normal(size=lat_c.shape) * data["lat_se"]
    observed = np.std(data["lat_hat"])
    return float(np.mean(np.std(rep, axis=1) >= observed))


def sample(
    engine,
    level_tables,
    settings,
    *,
    n_chains,
    n_warmup,
    n_draws,
    target_accept,
    random_state,
    formula,
    covariates,
    eb_variance=None,
):
    """Sample every component and return the posterior tables."""
    import pandas as pd

    require_engine(engine)
    channel = level_tables["channel"].copy()
    trial = level_tables["trial"]
    subjects = settings["subjects"]
    seed = 0 if random_state is None else int(random_state)
    var_rows, effect_rows, sub_rows, diag = [], [], [], {"bayes": {}}
    for label in settings["components"]:
        if not (channel["component"] == label).any():
            continue
        data = prepare(channel, trial, label, formula, covariates, subjects)
        if len(data["lat_hat"]) < 2:
            diag["bayes"][label] = {"skipped": "fewer than 2 channels"}
            continue
        init = _init_values(data, eb_variance, label)
        sampler = _sample_numpyro if engine == "numpyro" else _sample_pymc
        samples, diverging, loglik = sampler(
            data, settings["latency_dist"], n_chains, n_warmup, n_draws, target_accept, seed, init
        )
        d = _diagnostics(samples, diverging)
        d["ppc_p_latency_spread"] = _ppc(samples, data)
        d["loglik"] = loglik
        if d["divergent_fraction"] > DIVERGENCE_WARNING:
            _warn(
                f"component {label!r}: {100 * d['divergent_fraction']:.1f} % divergent "
                f"transitions; raise target_accept.",
                UserWarning,
                stacklevel=3,
            )
        diag["bayes"][label] = d
        A = data["amp_scale"]
        # channel posteriors
        rows = channel["component"] == label
        index = {k: i for i, k in enumerate(data["keys"])}
        lat = _summary(samples["latency_channel"] / 1e3)
        amp = _summary(samples["amplitude_channel"] * A)
        for name, stats in (("latency", lat), ("amplitude", amp)):
            for column in ("", "_se", "_ci_low", "_ci_high"):
                channel.loc[rows, f"{name}{column}"] = np.nan
            for idx in channel.index[rows]:
                key = (channel.at[idx, "subject"], channel.at[idx, "channel"])
                if key in index:
                    i = index[key]
                    channel.at[idx, name] = stats[0][i]
                    channel.at[idx, f"{name}_se"] = stats[1][i]
                    channel.at[idx, f"{name}_ci_low"] = stats[2][i]
                    channel.at[idx, f"{name}_ci_high"] = stats[3][i]
        if "responsibility" in samples:
            resp = samples["responsibility"].reshape(-1, len(data["ex_y"])).mean(0)
            eindex = {k: i for i, k in enumerate(data["exist_keys"])}
            for idx in channel.index[rows]:
                key = (channel.at[idx, "subject"], channel.at[idx, "channel"])
                channel.at[idx, "p_response"] = resp[eindex[key]] if key in eindex else np.nan
        # The draws themselves, for HierarchicalFit.to_arviz (chain x draw x site/subject)
        diag.setdefault("draws", {})[label] = {
            "sites": [f"{a}/{b}" for a, b in data["keys"]],
            "subjects": list(subjects),
            "latency_channel": np.asarray(samples["latency_channel"]) / 1e3,
            "amplitude_channel": np.asarray(samples["amplitude_channel"]) * A,
            "latency_subject": np.asarray(samples["latency_subject"]) / 1e3,
            "amplitude_subject": np.asarray(samples["amplitude_subject"]) * A,
        }
        sl = _summary(samples["latency_subject"] / 1e3)
        sa = _summary(samples["amplitude_subject"] * A)
        for i, name in enumerate(subjects):
            sub_rows.append(
                {
                    "subject": name,
                    "component": label,
                    "latency": sl[0][i],
                    "latency_se": sl[1][i],
                    "latency_ci_low": sl[2][i],
                    "latency_ci_high": sl[3][i],
                    "amplitude": sa[0][i],
                    "amplitude_se": sa[1][i],
                    "amplitude_ci_low": sa[2][i],
                    "amplitude_ci_high": sa[3][i],
                    **covariates[name],
                }
            )
        # total trial-to-trial SD (the ex-Gaussian's: sqrt(sigma^2 + tau^2))
        samples["sd_jitter_total"] = np.sqrt(
            samples["sd_jitter"] ** 2 + samples.get("tau_jitter", 0.0) ** 2
        )
        for key, parameter, level, scale in (
            ("sd_latency_subject", "latency", "subject", 1e-3),
            ("sd_latency_channel", "latency", "channel", 1e-3),
            ("sd_jitter_total", "latency", "trial", 1e-3),
            ("sd_jitter", "latency", "trial (ex-Gaussian sigma)", 1e-3),
            ("tau_jitter", "latency", "trial (ex-Gaussian tau)", 1e-3),
            ("sd_amplitude_subject", "amplitude", "subject", A),
            ("sd_amplitude_channel", "amplitude", "channel", A),
            ("sd_existence_channel", "amplitude", "channel (existence mixture)", A),
        ):
            if key in samples:
                m, s, lo, hi = _summary(samples[key] * scale)
                var_rows.append(
                    {
                        "component": label,
                        "parameter": parameter,
                        "level": level,
                        "sd": float(m),
                        "sd_se": float(s),
                        "ci_low": float(lo),
                        "ci_high": float(hi),
                        "method": f"NUTS ({engine})",
                    }
                )
        if "beta" in samples and len(data["X_names"]):
            m, s, lo, hi = _summary(samples["beta"] / 1e3)
            for j, term in enumerate(data["X_names"]):
                effect_rows.append(
                    {
                        "component": label,
                        "parameter": "latency",
                        "term": term,
                        "estimate": m[j],
                        "se": s[j],
                        "ci_low": lo[j],
                        "ci_high": hi[j],
                    }
                )
        diag["bayes"][label]["posterior"] = {
            k: _summary(v)[0].tolist() if np.ndim(v) > 2 else float(_summary(v)[0])
            for k, v in samples.items()
            if k.startswith(("mu_", "sd_", "tau_", "beta"))
        }
    out = {
        "level_tables": {"channel": channel, "subject": pd.DataFrame(sub_rows)},
        "diagnostics": diag,
        "variance_components": pd.DataFrame(var_rows),
        "p_response": channel[["subject", "channel", "component", "p_response"]].copy(),
    }
    if effect_rows:
        out["effects"] = pd.DataFrame(effect_rows)
    return out


def loo_table(fits, var_name="lat_hat"):
    """PSIS-LOO of each fit on the channel latencies (``arviz.compare``), best first."""
    import arviz as az
    import pandas as pd

    idatas = {}
    for i, fit in enumerate(fits):
        bayes = fit.diagnostics.get("bayes", {})
        for label, d in bayes.items():
            ll = (d.get("loglik") or {}).get(var_name)
            if ll is None:
                continue
            name = f"fit{i}:{label}"
            idatas[name] = _from_dict(
                az, {"posterior": {"_": np.zeros(ll.shape[:2])}, "log_likelihood": {var_name: ll}}
            )
    if len(idatas) < 2:
        return pd.DataFrame()
    return az.compare(idatas, var_name=var_name)
