"""
scikit-learn: evoked parameters as features
===========================================

``TrialFitter`` learns each channel's response on the trial average (``fit``) and gives every
trial's shift, scale and latency per component (``transform``); ``ArtifactRemover`` removes
the stimulation artifact; ``ResponseModel`` fits waveforms. All three follow the
scikit-learn conventions (``get_params`` / ``set_params``, ``clone``, ``Pipeline``,
``NotFittedError``) and pass scikit-learn's ``check_estimator``. Here they decode, trial by
trial, which of two currents was used.

Data: simulated CCEPs at 2 and 6 mA; the N1 is three times larger at 6 mA (an amplitude
growing with the current, as the input-output curves of CCEPs do), with a 30 % trial-to-trial
variability of the amplitude.
"""

# %%
# Two currents
# ------------

import numpy as np
import pandas as pd

import evoparam
from evoparam import sim
from evoparam.artifact import ArtifactRemover
from evoparam.trials import TrialFitter

n1_low = {"amplitude": (-100.0, -100.0), "latency": (0.025, 0.025)}
n1_high = {"amplitude": (-300.0, -300.0), "latency": (0.025, 0.025)}
low, _ = sim.simulate_ccep(n_trials=40, n_channels=3, stim_current_mA=2.0, seed=0,
                           p_response=1.0, n1=n1_low, amplitude_cv=0.3)
high, _ = sim.simulate_ccep(n_trials=40, n_channels=3, stim_current_mA=6.0, seed=1,
                            p_response=1.0, n1=n1_high, amplitude_cv=0.3)
X = np.concatenate([low.data, high.data])               # (trials, channels, times), µV
y = np.r_[np.zeros(len(low.data)), np.ones(len(high.data))]
sfreq, tmin = low.sfreq, float(low.times[0])
print(X.shape)

# %%
# A pipeline
# ----------

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(
    ArtifactRemover("template", sfreq=sfreq, tmin=tmin, window=(-0.002, 0.008)),
    TrialFitter(sfreq=sfreq, tmin=tmin, fit_window=(0.009, 0.5), max_components=2,
                random_state=0),
    SimpleImputer(), StandardScaler(), LogisticRegression())
print(pipe)
pipe.set_params(trialfitter__max_components=2)       # nested parameters, as in scikit-learn

# %%
# Cross-validated decoding
# ------------------------
# The fitter is refitted on the training trials of each fold.

scores = cross_val_score(pipe, X, y, cv=4)
print("accuracy per fold:", scores.round(2), "mean", scores.mean().round(2))

# %%
# The features
# ------------

fitter = pipe.fit(X, y).named_steps["trialfitter"]
features = pd.DataFrame(pipe[:2].transform(X), columns=fitter.get_feature_names_out())
print(features.groupby(y).mean().T.round(3).head(9))

# %%
# Methods paragraph (the template fit of the fitter)
# --------------------------------------------------

print(evoparam.reports.methods_text(fitter.group_fit_))
