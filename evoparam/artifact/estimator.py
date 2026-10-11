# SPDX-License-Identifier: Apache-2.0
"""``ArtifactRemover``: stimulation-artifact removal with the scikit-learn conventions."""

import numpy as np

from evoparam.errors import DataError, SettingsError, warn as _warn
from evoparam.utils._estimator import (BaseEstimatorMixin, TransformerMixin, check_is_fitted,
                                       validate_array)

__all__ = ["ArtifactRemover"]


class ArtifactRemover(BaseEstimatorMixin, TransformerMixin):
    """``evoparam.artifact.remove_artifact`` as a transformer: the output has the type and
    shape of X (a TrialSet keeps its masks of replaced samples).

    Parameters
    ----------
    method : str
        A method of ``remove_artifact``.
    window : (float, float)
        Artifact window (s).
    sfreq, tmin : float, optional
        Sampling rate and first time when X is an array (n_trials, n_channels, n_times).
    baseline : (float, float)
    random_state : int, optional
    """

    def __init__(self, method="template", *, window=(-0.002, 0.008), sfreq=None, tmin=None,
                 baseline=(-0.5, -0.05), random_state=None):
        self.method = method
        self.window = window
        self.sfreq = sfreq
        self.tmin = tmin
        self.baseline = baseline
        self.random_state = random_state

    def _trials(self, X, *, reset):
        """(TrialSet, shape of the array input or None for a TrialSet). An array is
        (n_trials, n_times) or (n_trials, n_channels, n_times), checked the scikit-learn way
        (dense, real, finite; after ``fit`` as many time points as in ``fit``)."""

        from evoparam.data import TrialSet

        if type(X).__module__.split(".")[0] == "mne":
            X = TrialSet.from_mne(X)          # mne.Epochs (the only MNE object accepted)
        if isinstance(X, TrialSet):
            if reset:
                self.n_features_in_ = int(X.n_times)
            elif X.n_times != self.n_features_in_:
                raise DataError(f"X has {X.n_times} time points, but ArtifactRemover is "
                                f"expecting {self.n_features_in_}.")
            return X, None
        data = validate_array(self, X, reset=reset, ndim=3)
        shape = data.shape
        if data.ndim == 2:
            data = data[:, None, :]
        if self.sfreq is None:
            raise SettingsError("sfreq is required when X is an array.")
        tmin = 0.0 if self.tmin is None else self.tmin
        times = tmin + np.arange(data.shape[-1]) / self.sfreq
        ts = TrialSet(data=data, times=times, sfreq=self.sfreq,
                      ch_names=[f"ch{i}" for i in range(data.shape[1])])
        return ts, shape

    def fit(self, X, y=None):
        """Nothing is learned from X but its time axis (``times_``, ``n_features_in_``);
        ``y`` is ignored."""

        trials, _ = self._trials(X, reset=True)
        self.times_ = np.array(trials.times)
        return self

    def transform(self, X):
        """X without the artifact. When the artifact window does not fit inside the epoch
        (it needs 3 samples on each side) X is returned unchanged with a warning."""

        from evoparam.artifact import remove_artifact

        check_is_fitted(self)
        trials, shape = self._trials(X, reset=False)
        times = np.asarray(trials.times, float)
        step = 1.0 / float(trials.sfreq)
        window = tuple(float(v) for v in self.window)
        if window[0] - 3 * step < times[0] or window[1] + 3 * step > times[-1]:
            _warn(f"ArtifactRemover: the window {window} needs 3 samples on each side "
                  f"inside the epoch ({times[0]:.4f} to {times[-1]:.4f} s); X is "
                  f"returned unchanged.", UserWarning, stacklevel=2)
            return trials.copy() if shape is None else np.array(trials.data).reshape(shape)
        baseline = (max(self.baseline[0], float(times[0])),
                    min(self.baseline[1], float(times[-1])))
        result = remove_artifact(trials, method=self.method, window=window,
                                 baseline=baseline, fallback=True,
                                 random_state=self.random_state)
        if shape is None:
            return result.cleaned
        return np.nan_to_num(result.cleaned.data).reshape(shape)
