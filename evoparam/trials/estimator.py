"""``TrialFitter``: trial-by-trial parameters with the scikit-learn conventions (spec §10.4)."""

import numpy as np

from evoparam.errors import DataError, SettingsError
from evoparam.utils._estimator import (BaseEstimatorMixin, TransformerMixin, check_is_fitted,
                                       validate_array)

__all__ = ["TrialFitter"]


class TrialFitter(BaseEstimatorMixin, TransformerMixin):
    """Learn each channel's template on the trial average (``fit``), then give every trial's
    shift, scale and latency per component (``transform``).

    Parameters
    ----------
    preset : str or Preset, optional
        Fitting preset (default: the package defaults with ``fit_window``).
    sfreq, tmin : float, optional
        Sampling rate and first time when X is an array (n_trials, n_channels, n_times).
    fit_window : (float, float), optional
    max_components : int, optional
        Component slots per channel in the feature matrix (default: the most found).
    mode : str
        Trial model of ``evoparam.trials.fit_trials``.
    random_state : int, optional
    """

    def __init__(self, preset=None, *, sfreq=None, tmin=None, fit_window=None,
                 max_components=None, mode="shift_scale", random_state=None):
        self.preset = preset
        self.sfreq = sfreq
        self.tmin = tmin
        self.fit_window = fit_window
        self.max_components = max_components
        self.mode = mode
        self.random_state = random_state

    def _trials(self, X, *, reset):
        """X as a TrialSet. An array is (n_trials, n_times) for one channel or (n_trials,
        n_channels, n_times), checked the scikit-learn way (dense, real, finite; after
        ``fit`` as many time points as in ``fit``)."""

        from evoparam.data import TrialSet
        from evoparam.models.api import _from_mne

        X = _from_mne(X)
        if isinstance(X, TrialSet):
            if reset:
                self.n_features_in_ = int(X.n_times)
            elif X.n_times != self.n_features_in_:
                raise DataError(f"X has {X.n_times} time points, but TrialFitter is "
                                f"expecting {self.n_features_in_}.")
            return X
        data = validate_array(self, X, reset=reset, ndim=3)
        if data.ndim == 2:
            data = data[:, None, :]
        if self.sfreq is None:
            raise SettingsError("sfreq is required when X is an array.")
        tmin = 0.0 if self.tmin is None else self.tmin
        times = tmin + np.arange(data.shape[-1]) / self.sfreq
        return TrialSet(data=data, times=times, sfreq=self.sfreq,
                        ch_names=[f"ch{i}" for i in range(data.shape[1])])

    def fit(self, X, y=None):
        """Fit the trial average of each channel (``group_fit_``); ``y`` is ignored. Without
        ``fit_window`` and preset the window is the data range."""

        from evoparam.models import ResponseModel

        trials = self._trials(X, reset=True)
        model = ResponseModel(self.preset, fit_window=self.fit_window,
                              max_components=self.max_components,
                              random_state=self.random_state).fit(trials)
        self.group_fit_ = model.fit_
        found = [len(f.components) for f in self.group_fit_.fits]
        self.n_components_ = int(self.max_components or max(found + [1]))
        self.ch_names_ = list(trials.ch_names)
        return self

    def transform(self, X):
        """(n_trials, n_channels * n_components * 3): shift, scale and latency of each
        component slot of each channel; NaN where a channel has fewer components."""

        from evoparam.trials import fit_trials

        check_is_fitted(self)
        trials = self._trials(X, reset=False)
        if trials.n_channels != len(self.ch_names_):
            raise DataError(f"X has {trials.n_channels} channels, but TrialFitter was fitted "
                            f"on {len(self.ch_names_)}.")
        out = np.full((trials.n_trials, len(self.ch_names_), self.n_components_, 3), np.nan)
        if any(f.components for f in self.group_fit_.fits):
            result = fit_trials(trials, self.group_fit_, mode=self.mode, by=(),
                                random_state=self.random_state, verbose=False)
            table = result.table
            for row in table.to_dict("records"):
                ch = self.ch_names_.index(row["channel"])
                k = int(row["component"])
                if k < self.n_components_:
                    out[int(row["trial"]), ch, k] = (row["shift"], row["scale"], row["latency"])
        return out.reshape(trials.n_trials, -1)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{ch}_c{k}_{name}" for ch in self.ch_names_
                         for k in range(self.n_components_)
                         for name in ("shift", "scale", "latency")])
