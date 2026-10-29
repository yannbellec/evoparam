# SPDX-License-Identifier: Apache-2.0
"""
17: Statistics
==============

Put error bars on a parameter, compare conditions, and plan how many trials you need.
"""  # noqa: D205, D400  (sphinx-gallery: the docstring opens with an RST title)

###################################################################################################
# Introduction
# ------------
#
# A fit gives numbers; statistics tell you how much to trust them and how to compare them.
# ``evoparam.stats`` covers the evoked-response questions:
#
# - how precise is one parameter (a bootstrap confidence interval);
# - does a parameter differ between two conditions (a permutation contrast across
#   subjects);
# - how reliable is the measure, and how many trials would reach a target reliability;
# - a tidy long table to hand to a mixed-effects model (``statsmodels``, ``lme4``).
#
# We work on made-up data so the truth is known, and keep the permutation counts small so
# the tutorial runs quickly; raise them for real work.

###################################################################################################

# Import what we need
import warnings

import numpy as np
import pandas as pd

from evoparam import models
from evoparam.io import from_array
from evoparam.sim import simulate_ccep
from evoparam.stats import (
    bootstrap_interval,
    compare_conditions,
    reliability,
    to_long,
)

###################################################################################################
# 1. A confidence interval on one parameter
# -----------------------------------------
#
# The fit already returns a standard error and an interval for every parameter (tutorial
# 06). ``bootstrap_interval`` is for a parameter you build yourself, for example the
# **mean N1 amplitude over a group of channels**. You give it the estimate and a set of
# bootstrap values (resampled means); it returns a bias-corrected and accelerated (BCa)
# interval, which is more accurate than estimate plus or minus two standard errors when
# the values are skewed.

###################################################################################################

trials, _ = simulate_ccep(
    20, n_channels=12, p_response=1.0, p_n2=1.0, artifact=False, layout="grid",
    sfreq=512.0, seed=0,
)  # fmt: skip
group = models.fit(trials, "ccep", random_state=0)
table = group.to_dataframe()
n1_amp = table.loc[table["label"] == "N1", "amplitude"].to_numpy()

rng = np.random.default_rng(0)
boot = np.array([rng.choice(n1_amp, n1_amp.size, replace=True).mean() for _ in range(2000)])
interval = bootstrap_interval(float(n1_amp.mean()), boot, ci_level=0.95)
print(f"mean N1 amplitude: {n1_amp.mean():.1f} uV, "
      f"95% CI [{interval['low']:.1f}, {interval['high']:.1f}] ({interval['method']})")

###################################################################################################
# 2. Comparing two conditions
# ---------------------------
#
# The real question is usually a contrast: is the N1 larger at a high stimulation current
# than a low one? ``compare_conditions`` takes a wide table of parameters with a
# ``condition`` column and a ``subject`` column, and runs a **paired permutation test**
# across subjects, correcting for the parameters tested. We simulate 8 subjects, each with
# a low and a high condition, and make the high condition's N1 truly larger.

###################################################################################################

tables = []
for subject in range(8):
    for condition, amplitude in [("low", -40.0), ("high", -75.0)]:
        subj_trials, _ = simulate_ccep(
            20, n_channels=3, p_response=1.0, p_n2=1.0, artifact=False, layout="grid",
            sfreq=512.0, seed=subject, n1={"amplitude": amplitude},
        )  # fmt: skip
        fit = models.fit(subj_trials, "ccep", random_state=0)
        wide = to_long(fit, wide=True)
        wide["subject"] = f"sub-{subject + 1}"
        wide["condition"] = condition
        tables.append(wide)
long_table = pd.concat(tables, ignore_index=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    contrast = compare_conditions(
        long_table, parameter="amplitude", condition="condition", unit="subject",
        by=("label",), random_state=0,
    )  # fmt: skip

columns = ["label", "mean_a", "mean_b", "difference", "ci_low", "ci_high", "p_value", "hedges_g"]
main = contrast[contrast["label"].isin(["N1", "N2"])]
print(main[columns].round(3).to_string(index=False))

###################################################################################################
#
# The N1 difference is large and significant, with a big effect size (``hedges_g``); the
# N2, which we did not change, is not. ``mean_a`` and ``mean_b`` are the condition means,
# ``difference`` is their contrast with its interval, and ``p_value`` is corrected across
# the two components. For a latency the table also gives the difference in milliseconds.

###################################################################################################
# 3. Reliability and how many trials you need
# -------------------------------------------
#
# How repeatable is the measure, and how many trials would you need for a given
# reliability? ``reliability`` splits the trials in half many times and measures how well
# the two halves agree across the units (here subjects). It returns the **dependability**
# (a generalisability-theory reliability) and the number of trials to reach a target. We
# build 10 subjects whose true N1 amplitude differs, 16 trials each.

###################################################################################################

sfreq = 512.0
times = np.arange(-0.2, 0.4, 1.0 / sfreq)
rng = np.random.default_rng(0)
blocks, subjects = [], []
for subject in range(10):
    true_amplitude = -(30.0 + rng.uniform(0.0, 50.0))  # each subject has their own N1 size
    bump = true_amplitude * np.exp(-0.5 * ((times - 0.025) / 0.004) ** 2)
    blocks.append(rng.normal(0.0, 12.0, size=(16, 1, times.size)) + bump)
    subjects += [f"sub-{subject + 1}"] * 16
data = np.concatenate(blocks, axis=0)
rel_trials = from_array(data, sfreq=sfreq, times=times, ch_names=["Cz"], units="uV",
                        metadata=pd.DataFrame({"subject": subjects}))  # fmt: skip

rel = reliability(rel_trials, scorer="mean_amplitude", facets=("trial", "subject"),
                  window=(0.009, 0.1), n_splits=500, target_phi=0.7, random_state=0)  # fmt: skip
columns = ["n_units", "n_trials_mean", "dependability", "trials_needed", "target_phi", "sme_rms"]
print(rel[columns].round(3).to_string(index=False))

###################################################################################################
#
# ``dependability`` near 0.9 means the measure ranks the subjects consistently;
# ``trials_needed`` is how many trials per subject reach the target dependability (0.7
# here), and ``sme_rms`` is the measurement error in microvolts. ``trials_needed`` on a
# single ``TrialSet`` answers the same planning question per channel, by resampling fewer
# and fewer trials and extrapolating.

###################################################################################################
# 4. Handing off to a mixed-effects model
# ---------------------------------------
#
# For a model evoparam does not fit itself (random slopes, crossed effects), export a tidy
# long table with ``to_long`` and give it to ``statsmodels`` or ``lme4``. Each row is one
# parameter of one component on one channel, with its standard error, so a weighted model
# can use the measurement uncertainty.
#
# .. code-block:: python
#
#     from evoparam.stats import mixed_formula, fit_mixed
#
#     long = to_long(group)                          # one row per parameter
#     formula = mixed_formula("amplitude", by="subject")   # amplitude ~ 1 + (1 | subject)
#     result = fit_mixed(long, formula)              # statsmodels mixed model

###################################################################################################
# Conclusion
# ----------
#
# ``bootstrap_interval`` puts an honest interval on a derived number,
# ``compare_conditions`` tests a contrast across subjects with a permutation test and an
# effect size, and ``reliability`` and ``trials_needed`` tell you how repeatable the
# measure is and how many trials a future study should collect. When you need a richer
# model, ``to_long`` hands the parameters, with their uncertainty, to a mixed-effects tool.
