# SPDX-License-Identifier: Apache-2.0
"""Statistics: intervals, comparisons, reliability and export to mixed models (spec §5.11).

- :func:`bootstrap`: percentile and BCa intervals of any statistic, by resampling trials
  within strata (stimulation site, condition).
- :func:`jackknife_latency`: leave-one-out grand averages of participants, the corrected
  statistics (:func:`jackknife_t`, :func:`jackknife_f`) and the retrieved individual
  scores.
- :func:`compare_conditions`: permutation contrasts at the level of the exchangeable unit
  (trial, channel, subject), Hedges' g and the difference with its interval.
- :func:`reliability`: split-half (odd/even and random splits, Spearman-Brown) and
  dependability from REML variance components, with the trials needed for a target;
  :func:`trials_needed`: detection probability and precision against the number of trials.
- :func:`variance_components` / :func:`reml`: nested random-intercept models by REML, with
  known sampling variances (the variance components of the hierarchical models, §5.8).
- :func:`to_long`, :func:`column_dictionary`, :func:`mixed_formula`, :func:`fit_mixed`,
  :func:`group_summary`: tidy exports, model formulas for statsmodels, lme4 and brms, and
  summaries that count every fit status.
- :func:`waveform_band`, :func:`strip_jitter`: the average of a set of waveforms with its
  spread band, and the positions of a strip plot (the numbers of the signal plots).

Everything runs on numpy and scipy (tables need pandas); only :func:`fit_mixed` needs the
extra ``stats`` (statsmodels).
"""

from ._scorers import SCORES, score_waveforms
from .bootstrap import BOOTSTRAP_COLUMNS, bootstrap, bootstrap_interval
from .compare import COMPARE_COLUMNS, UNITS, compare_conditions, hedges_g
from .export import (LONG_COLUMNS, PARAMETERS, column_dictionary, fit_mixed, group_summary,
                     mixed_formula, to_long)
from .jackknife import JACKKNIFE_COLUMNS, jackknife_f, jackknife_latency, jackknife_t
from .reliability import (RELIABILITY_COLUMNS, TRIALS_NEEDED_COLUMNS, dependability,
                          reliability, spearman_brown, trials_needed)
from .variance import VarianceResult, reml, variance_components
from .waveforms import strip_jitter, waveform_band

__all__ = ["bootstrap", "bootstrap_interval", "jackknife_latency", "jackknife_t",
           "jackknife_f", "compare_conditions", "hedges_g", "reliability", "trials_needed",
           "spearman_brown", "dependability", "variance_components", "reml", "VarianceResult",
           "to_long", "column_dictionary", "mixed_formula", "fit_mixed", "group_summary",
           "score_waveforms", "SCORES", "UNITS", "PARAMETERS", "LONG_COLUMNS",
           "BOOTSTRAP_COLUMNS", "COMPARE_COLUMNS", "JACKKNIFE_COLUMNS", "RELIABILITY_COLUMNS",
           "TRIALS_NEEDED_COLUMNS", "waveform_band", "strip_jitter"]
