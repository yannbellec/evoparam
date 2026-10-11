# SPDX-License-Identifier: Apache-2.0
"""Example evoparam plugin (spec §10.8): a shape, a preset and a reader.

* ``Lorentzian``: ``s(t) = 1 / (1 + ((t - mu) / gamma)²)``, parameters ``(mu, gamma)`` in s,
  FWHM ``2 gamma``; it satisfies ``evoparam.shapes.ComponentShape`` and passes the checks
  of ``register_shape`` (unit amplitude, analytic Jacobian).
* ``DUMMY_LORENTZ``: the ``ccep`` preset fitting Lorentzians.
* ``read_dummy``: reads ``.dummy`` files, a NumPy ``.npz`` archive holding ``data``
  (n_trials, n_channels, n_times, µV), ``sfreq`` and ``tmin``.
"""

import dataclasses
from typing import ClassVar

import numpy as np

__all__ = ["Lorentzian", "DUMMY_LORENTZ", "read_dummy"]


@dataclasses.dataclass(frozen=True)
class Lorentzian:
    """Lorentzian bump with unit peak."""

    name: ClassVar[str] = "lorentzian"
    param_names: ClassVar[tuple] = ("mu", "gamma")
    n_params: ClassVar[int] = 2
    latency_param: ClassVar[str] = "mu"
    min_fwhm: float | None = None
    max_fwhm: float | None = None

    @staticmethod
    def _split(params):
        params = np.asarray(params, dtype=float)
        return params[..., 0:1], params[..., 1:2]

    def evaluate(self, times, params):
        mu, gamma = self._split(params)
        z = (np.asarray(times, float) - mu) / gamma
        return 1.0 / (1.0 + z ** 2)

    def jacobian(self, times, params):
        mu, gamma = self._split(params)
        z = (np.asarray(times, float) - mu) / gamma
        s2 = (1.0 / (1.0 + z ** 2)) ** 2
        return np.stack([2.0 * z * s2 / gamma, 2.0 * z ** 2 * s2 / gamma], axis=-1)

    def _widths(self, times):
        times = np.asarray(times, float)
        step = float(np.median(np.diff(times)))
        low = 2.0 * step if self.min_fwhm is None else float(self.min_fwhm)
        high = (times[-1] - times[0]) / 2.0 if self.max_fwhm is None else float(self.max_fwhm)
        return times, low, high

    def default_bounds(self, times, data_range=None):
        times, low, high = self._widths(times)
        return (np.array([times[0], low / 2.0]), np.array([times[-1], high / 2.0]))

    def initial_guess(self, times, waveform, peak_index):
        lower, upper = self.default_bounds(times)
        times = np.asarray(times, float)
        guess = np.array([times[int(peak_index)], 0.5 * (lower[1] + upper[1]) / 4.0])
        return np.clip(guess, lower, upper)

    def latency(self, params):
        return np.asarray(params, float)[..., 0]

    extremum = latency

    def features(self, times, params, amplitude=1.0):
        from evoparam.shapes import compute_features

        return compute_features(self, times, params, amplitude)


def _preset():
    from evoparam import presets

    # built on the ccep preset; presets.derive keeps the parent name in the record
    return presets.derive("ccep", name="dummy_lorentz", shapes=("lorentzian",),
                          validation="simulation_only")


DUMMY_LORENTZ = _preset()


def read_dummy(path, *, picks=None):
    """TrialSet of a ``.dummy`` file (``np.savez`` archive with data, sfreq, tmin)."""

    from evoparam.data import Provenance, TrialSet

    with open(path, "rb") as fobj:
        archive = np.load(fobj)
        data = np.asarray(archive["data"], float)
        sfreq, tmin = float(archive["sfreq"]), float(archive["tmin"])
    names = [f"D{i + 1}" for i in range(data.shape[1])]
    if picks is not None:
        index = [names.index(p) for p in picks]
        data, names = data[:, index], [names[i] for i in index]
    times = tmin + np.arange(data.shape[-1]) / sfreq
    return TrialSet(data=data, times=times, sfreq=sfreq, ch_names=names,
                    provenance=Provenance(source={"format": "dummy", "path": str(path)}))


read_dummy.extensions = (".dummy",)
