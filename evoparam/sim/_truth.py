# SPDX-License-Identifier: Apache-2.0
"""GroundTruth: what a simulator put in the data, exactly.

The truth stores every component once (its nominal parameters, channel, role
and requested signal-to-noise ratio) and, per trial, how it was moved and
scaled (latency shift, amplitude scale, presence). A trial's component is the
nominal component with its location parameter moved by the shift and its
amplitude multiplied by the scale, so every feature of every trial is known
without rounding. The noise and artifact parameters, the seed, the settings and
(optionally) the noise-free signal and the artifact are kept with it.
"""

import copy
from dataclasses import dataclass, field

import numpy as np

from evoparam._typing import Any, FloatArray, NDArray
from evoparam.data import Provenance
from evoparam.data._compare import DeepEqualMixin
from evoparam.data._jsonutil import dumps, loads, to_jsonable
from evoparam.data.results import COMPONENT_COLUMNS
from evoparam.data.serialize import register_type
from evoparam.errors import DataError
from evoparam.utils._optional import require

__all__ = ["GroundTruth", "TRUTH_COLUMNS", "PARAM_UNITS"]

# Features shared with ResponseFit / GroupFit tables, with the same units.
FEATURES = (
    "latency",
    "amplitude",
    "polarity",
    "fwhm",
    "onset",
    "offset",
    "area",
    "rise_time",
    "decay_time",
    "symmetry",
    "sharpness",
)
# Features that move with the latency shift, and those that scale with the amplitude.
_SHIFTED = ("latency", "onset", "offset", "peak_time")
_SCALED = ("amplitude", "area")

# Columns of GroundTruth.to_dataframe(): the component columns of GroupFit.to_dataframe()
# (same names and units, without "status"), then the truth-only columns.
TRUTH_COLUMNS = {
    key: unit for key, unit in COMPONENT_COLUMNS.items() if key not in ("status", "resolvable")
}
TRUTH_COLUMNS.update(
    {"trial": "", "role": "", "event": "", "snr": "", "shift": "s", "scale": "", "n_trials": ""}
)

# Units of the shape parameters (columns "param_<name>").
PARAM_UNITS = {
    "mu": "s",
    "sigma": "s",
    "xi": "s",
    "omega": "s",
    "alpha": "",
    "tau": "s",
    "t0": "s",
    "kappa": "",
    "theta": "s",
    "tau_rise": "s",
    "tau_decay": "s",
    "freq": "Hz",
    "phase": "rad",
    "shift": "s",
    "stretch": "",
}


@dataclass(eq=False)
class GroundTruth(DeepEqualMixin):
    """Exact content of a simulation: components per trial and channel, noise, artifact, seed.

    Parameters
    ----------
    generator : str
        Full name of the function that made the data.
    ch_names : list of str
        Channels of the simulated :class:`~evoparam.data.TrialSet`.
    times : ndarray
        Time axis (s).
    sfreq : float
        Sampling frequency (Hz).
    units : str
        Amplitude unit.
    components : list of dict
        One entry per component and channel: ``label``, ``shape``, ``param_names``,
        ``params`` (nominal), ``amplitude`` (nominal linear amplitude), ``channel``,
        ``role``, ``event`` (0 for the trial's own event, -1 / +1 for the previous /
        next event, ``k`` for the ``k``-th pulse of a train), ``loc_param`` (the
        parameter moved by a latency shift), ``snr`` (requested signal-to-noise ratio,
        None when the amplitude was given) and ``features`` (of the nominal component).
    shift : ndarray, shape (n_trials, n_components)
        Latency shift of each component in each trial (s).
    scale : ndarray, shape (n_trials, n_components)
        Amplitude scale of each component in each trial.
    present : ndarray of bool, shape (n_trials, n_components)
        Whether the component is in the trial.
    conditions : list of str
        Condition of each trial ("" when there is none).
    noise : dict
        Noise model and parameters; ``baseline_sd`` is the per-trial noise SD measured
        on the baseline, averaged over trials, per channel; real noise adds the record
        of every segment.
    artifact : dict or None
        Artifact model and parameters.
    seed : int
        Seed that regenerates the simulation.
    settings : dict
        Every argument of the generator.
    signal : ndarray or None
        Noise-free sum of the components, shape (n_trials, n_channels, n_times).
    artifact_data : ndarray or None
        The artifact added to the data, same shape.
    provenance : Provenance
    """

    generator: str
    ch_names: list[str]
    times: NDArray[Any]
    sfreq: float
    units: str
    components: list[Any]
    shift: NDArray[Any]
    scale: NDArray[Any]
    present: NDArray[Any]
    conditions: list[Any]
    noise: dict[str, Any]
    artifact: dict[str, Any] | None
    seed: int
    settings: dict[str, Any] = field(default_factory=dict)
    signal: NDArray[Any] | None = None
    artifact_data: NDArray[Any] | None = None
    provenance: Provenance = field(default_factory=Provenance)

    def __post_init__(self) -> None:
        """Coerce the fields to their types; check the per-trial arrays (trials x components)."""
        self.ch_names = [str(name) for name in self.ch_names]
        self.times = np.asarray(self.times, dtype=np.float64)
        self.sfreq = float(self.sfreq)
        self.conditions = [str(c) for c in self.conditions]
        shape = (len(self.conditions), len(self.components))
        try:
            self.shift = np.asarray(self.shift, dtype=np.float64).reshape(shape)
            self.scale = np.asarray(self.scale, dtype=np.float64).reshape(shape)
            self.present = np.asarray(self.present, dtype=bool).reshape(shape)
        except ValueError:
            raise DataError(
                f"GroundTruth: shift, scale and present must have shape {shape} "
                "(trials × components)."
            ) from None
        self.seed = None if self.seed is None else int(self.seed)
        for name in ("signal", "artifact_data"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, np.asarray(value, dtype=np.float64))

    @property
    def n_trials(self) -> Any:
        """Number of simulated trials."""
        return int(self.shift.shape[0])

    @property
    def n_components(self) -> Any:
        """Number of simulated components (all channels)."""
        return len(self.components)

    def copy(self) -> "GroundTruth":
        """Return a deep copy."""
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        """Return a one-line summary: generator, trials, channels, components and seed."""
        return (
            f"<GroundTruth from {self.generator}: {self.n_trials} trials, "
            f"{len(self.ch_names)} channels, {self.n_components} components, "
            f"seed {self.seed}>"
        )

    ###############################################################################################
    ## Trial-level values

    def trial_params(self, index: Any) -> FloatArray:
        """Shape parameters of component ``index`` in every trial, shape (n_trials, n_params)."""
        comp = self.components[index]
        params = np.tile(np.asarray(comp["params"], dtype=np.float64), (self.n_trials, 1))
        loc = comp["param_names"].index(comp["loc_param"])
        params[:, loc] += self.shift[:, index]
        return params

    def waveform(self, index: Any, trial: Any = None) -> Any:
        """Curve of component ``index`` (amplitude included) on ``times``.

        Returns shape (n_times,) for one ``trial``, else (n_trials, n_times).
        """
        from evoparam.shapes import get_shape

        comp = self.components[index]
        shape = get_shape(comp["shape"])
        params = self.trial_params(index)
        amplitude = comp["amplitude"] * self.scale[:, index] * self.present[:, index]
        curves = amplitude[:, None] * shape.evaluate(self.times, params)
        return curves if trial is None else curves[trial]

    ###############################################################################################
    ## Tables

    def _component_order(self) -> NDArray[Any]:
        """Position of each component among those of its channel, by nominal latency."""
        order = np.zeros(self.n_components, dtype=np.int64)
        by_channel: dict[Any, Any]
        by_channel = {}
        for index, comp in enumerate(self.components):
            by_channel.setdefault(comp["channel"], []).append(index)
        for indices in by_channel.values():
            ranked = sorted(indices, key=lambda i: (self.components[i]["features"]["latency"], i))
            for position, index in enumerate(ranked):
                order[index] = position
        return order

    def _row(self, index: Any, position: Any, shift: Any, scale: Any) -> dict[str, Any]:
        """Return the table row of component ``index`` with a latency shift and a scale."""
        comp = self.components[index]
        feats = {
            key: np.nan if value is None else value for key, value in comp["features"].items()
        }
        for name in _SHIFTED:
            if name in feats:
                feats[name] = feats[name] + shift
        for name in _SCALED:
            feats[name] = feats[name] * scale
        feats["polarity"] = int(np.sign(feats["amplitude"])) or int(feats["polarity"])
        row = {
            "channel": comp["channel"],
            "component": int(position),
            "label": comp["label"],
            "shape": comp["shape"],
        }
        row.update({name: feats[name] for name in FEATURES})
        row.update(
            {
                "role": comp["role"],
                "event": int(comp["event"]),
                "snr": np.nan if comp["snr"] is None else float(comp["snr"]) * scale,
                "shift": float(shift),
                "scale": float(scale),
            }
        )
        params = np.asarray(comp["params"], dtype=np.float64).copy()
        params[comp["param_names"].index(comp["loc_param"])] += shift
        row.update(
            {
                f"param_{name}": float(value)
                for name, value in zip(comp["param_names"], params, strict=False)
            }
        )
        return row

    def to_dataframe(self, level: str = "component") -> Any:
        """Tidy table of the components, in the column schema of ``GroupFit.to_dataframe()``.

        Parameters
        ----------
        level : {"component", "trial"}
            ``"trial"``: one row per trial, channel and component present in the
            trial, exact. ``"component"``: one row per channel, condition and
            component, the expected component of the condition's average when trials
            are not jittered: latency moved by the mean shift of the trials that hold
            the component, amplitude and area multiplied by the mean scale over all
            trials of the condition (a trial without the component counts as 0). With
            latency jitter the average is also smeared, which this level ignores.

        Returns
        -------
        pandas.DataFrame
            Columns ``channel``, ``condition``, ``component``, ``label``, ``shape``,
            the features ``latency`` … ``sharpness`` (as in ``GroupFit``), then
            ``trial`` (trial level) or ``n_trials`` (component level), ``role``,
            ``event``, ``snr`` (the requested SNR times ``scale``, so that
            ``|amplitude| / baseline SD == snr``; NaN when the amplitude was given in
            data units), ``shift`` (s), ``scale`` and one ``param_<name>`` column per
            shape parameter. Units are in ``table.attrs["units"]`` ("data" is the
            amplitude unit, also in ``table.attrs["data_units"]``).
        """
        pd = require("pandas", extra="data", caller="evoparam.sim.GroundTruth.to_dataframe")
        if level not in ("component", "trial"):
            raise DataError(
                f"GroundTruth.to_dataframe: level must be 'component' or 'trial', got {level!r}."
            )
        order = self._component_order()
        conditions = np.asarray(self.conditions, dtype=object)
        rows = []
        index: Any
        if level == "trial":
            for trial in range(self.n_trials):
                for index in np.flatnonzero(self.present[trial]):
                    row = self._row(
                        index, order[index], self.shift[trial, index], self.scale[trial, index]
                    )
                    row.update(trial=trial, condition=self.conditions[trial])
                    rows.append(row)
        else:
            for condition in dict.fromkeys(self.conditions):
                trials = np.flatnonzero(conditions == condition)
                for index in range(self.n_components):
                    held = self.present[trials, index]
                    if not held.any():
                        continue
                    shift = float(np.mean(self.shift[trials[held], index]))
                    scale = float(np.mean(self.scale[trials, index] * held))
                    row = self._row(index, order[index], shift, scale)
                    row.update(n_trials=int(held.sum()), condition=condition)
                    rows.append(row)
        param_columns = sorted({key for row in rows for key in row if key.startswith("param_")})
        base = [
            key for key in TRUTH_COLUMNS if key != ("n_trials" if level == "trial" else "trial")
        ]
        table = pd.DataFrame(rows, columns=base + param_columns)
        for column in ("component", "polarity", "event") + (
            ("trial",) if level == "trial" else ("n_trials",)
        ):
            table[column] = table[column].astype("int64")
        units = {key: TRUTH_COLUMNS[key] for key in base}
        units.update({key: PARAM_UNITS.get(key[6:], "") for key in param_columns})
        table.attrs["units"] = units
        table.attrs["data_units"] = self.units
        return table

    ###############################################################################################
    ## Serialisation

    def to_dict(self, arrays: bool = True) -> Any:
        """JSON-serialisable dictionary.

        ``arrays=False`` leaves out ``signal`` and ``artifact_data``.
        """
        out = {
            "generator": self.generator,
            "ch_names": list(self.ch_names),
            "times": self.times.tolist(),
            "sfreq": self.sfreq,
            "units": self.units,
            "components": copy.deepcopy(self.components),
            "shift": self.shift.tolist(),
            "scale": self.scale.tolist(),
            "present": self.present.tolist(),
            "conditions": list(self.conditions),
            "noise": copy.deepcopy(self.noise),
            "artifact": copy.deepcopy(self.artifact),
            "seed": self.seed,
            "settings": copy.deepcopy(self.settings),
            "provenance": self.provenance.to_dict(),
        }
        if arrays:
            out["signal"] = None if self.signal is None else self.signal.tolist()
            out["artifact_data"] = (
                None if self.artifact_data is None else self.artifact_data.tolist()
            )
        return to_jsonable(out)

    @classmethod
    def from_dict(cls, dct: Any) -> "GroundTruth":
        """Rebuild from :meth:`to_dict` output."""
        dct = dict(dct)
        dct["provenance"] = Provenance.from_dict(dct["provenance"])
        components = copy.deepcopy(dct["components"])
        for comp in components:
            # Strict JSON writes NaN as null; features are numbers, so null means NaN.
            comp["features"] = {
                key: np.nan if value is None else value for key, value in comp["features"].items()
            }
        dct["components"] = components
        for name in ("signal", "artifact_data"):
            if dct.get(name) is not None:
                dct[name] = np.asarray(dct[name], dtype=np.float64)
        return cls(**dct)

    def save(self, path: Any, overwrite: bool = False, table_format: Any = "auto") -> str:
        """Save to a ``.evoparam`` container (spec §4.3.5), arrays included.

        Returns
        -------
        str
            The path written.
        """
        from evoparam.data.serialize import save_object

        return save_object(self, path, overwrite=overwrite, table_format=table_format)

    @classmethod
    def load(cls, path: Any, verify: bool = True) -> "GroundTruth":
        """Load a container written by :meth:`save` (also :func:`evoparam.load_object`)."""
        from evoparam.data.serialize import load_object

        obj = load_object(path, verify=verify)
        if not isinstance(obj, cls):
            raise DataError(f"{path}: holds a {type(obj).__name__}, not a GroundTruth.")
        return obj

    def to_json(self, arrays: bool = False, **kwargs: Any) -> str:
        """Strict JSON text (NaN written as null); arrays left out by default."""
        return dumps(self.to_dict(arrays=arrays), **kwargs)

    @classmethod
    def from_json(cls, text: Any) -> "GroundTruth":
        """Rebuild from :meth:`to_json` output."""
        return cls.from_dict(loads(text))


register_type(GroundTruth)
