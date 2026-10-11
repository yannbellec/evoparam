# SPDX-License-Identifier: Apache-2.0
"""Result dataclasses returned by the fitting, testing and analysis modules.

Results are plain data: they hold numbers, tables, the resolved settings and a
:class:`~evoparam.data.provenance.Provenance`. They compare with a NaN-aware
``==`` and save to, and load from, the ``.evoparam`` container
(:mod:`evoparam.data.serialize`).
"""

import copy
from dataclasses import dataclass, field, fields, replace

import numpy as np

from evoparam.errors import DataError
from evoparam.data._compare import DeepEqualMixin
from evoparam.data.provenance import Provenance

__all__ = ["Component", "GoodnessOfFit", "ResponseFit", "GroupFit", "TrialFit",
           "HierarchicalFit", "ResponseTest", "ArtifactResult", "ShapeBasis", "TFFit",
           "FIT_STATUSES", "COMPONENT_COLUMNS", "INTERVAL_COLUMNS", "EXISTENCE_COLUMNS",
           "FIT_COLUMNS"]

FIT_STATUSES = ("ok", "no_response", "failed", "skipped")

# Columns of ResponseFit.to_dataframe() / GroupFit.to_dataframe("component") and their units.
# "data" stands for the amplitude unit of the fitted data.
COMPONENT_COLUMNS = {
    "channel": "", "condition": "", "component": "", "label": "", "shape": "",
    "latency": "s", "amplitude": "data", "polarity": "", "fwhm": "s", "onset": "s",
    "offset": "s", "area": "data·s", "rise_time": "s", "decay_time": "s", "symmetry": "",
    "sharpness": "", "resolvable": "", "status": "",
}

# Interval of each reported feature, in the same row (REQ-006): the bounds of
# ``Component.ci`` at the fit's ``ci_level`` (NaN when the fit has no interval).
INTERVAL_FEATURES = ("latency", "amplitude", "fwhm", "onset", "offset", "area")
INTERVAL_COLUMNS = {f"{name}_ci_{side}": COMPONENT_COLUMNS[name]
                    for name in INTERVAL_FEATURES for side in ("low", "high")}

# Columns appended to the component table by the existence test of each channel (§5.6).
EXISTENCE_COLUMNS = {"existence_p": "", "existence_significant": ""}

# Columns of GroupFit.to_dataframe("fit"): one row per channel × condition.
FIT_COLUMNS = {
    "channel": "", "condition": "", "status": "", "message": "", "n_components": "",
    "r2": "", "adj_r2": "", "rmse": "data", "mae": "data", "nrmse": "", "aic": "", "aicc": "",
    "bic": "", "n_params": "", "n_samples": "", "n_eff": "", "residual_lag1_autocorr": "",
    "baseline_sd": "data",
}

COMPONENT_FEATURES = ("latency", "amplitude", "polarity", "fwhm", "onset", "offset", "area",
                      "rise_time", "decay_time", "symmetry", "sharpness")


class _ResultMixin(DeepEqualMixin):
    """copy, to_dict / from_dict and save / load for result dataclasses."""

    def copy(self):
        """Return a deep copy."""

        return copy.deepcopy(self)

    def to_dict(self):
        """Return the fields as a dictionary of deep copies."""

        return {fld.name: copy.deepcopy(getattr(self, fld.name)) for fld in fields(self)}

    @classmethod
    def from_dict(cls, dct):
        """Build from :meth:`to_dict` output."""

        return cls(**dct)

    def methods_text(self):
        """Methods paragraph built from ``provenance`` and ``settings`` only (REQ-338);
        see :func:`evoparam.reports.methods_text`."""

        from evoparam.data._methods import methods_text
        return methods_text(self)

    def save(self, path, overwrite=False):
        """Save to a ``.evoparam`` container (directory, or zip when ``path`` ends in .zip)."""

        from evoparam.data.serialize import save_object
        return save_object(self, path, overwrite=overwrite)

    @classmethod
    def load(cls, path, *, verify=True):
        """Load from a ``.evoparam`` container and check its type."""

        from evoparam.data.serialize import load_object
        obj = load_object(path, verify=verify)
        if not isinstance(obj, cls):
            raise DataError(f"{path} holds a {type(obj).__name__}, not a {cls.__name__}.")
        return obj


def _as_float_array(value):
    return None if value is None else np.asarray(value, dtype=np.float64)


###################################################################################################
## Single-waveform results

@dataclass(eq=False)
class Component(_ResultMixin):
    """One fitted component and the features computed from its fitted curve.

    Times are in seconds, amplitudes in the data unit. ``onset`` and ``offset``
    are where the component reaches ``onset_fraction`` (default 0.10) of its
    absolute amplitude; ``symmetry`` is (decay − rise) / (decay + rise);
    ``sharpness`` is the curvature of the unit-amplitude shape at the peak times
    fwhm², dimensionless.
    """

    shape: str
    params: np.ndarray
    param_names: tuple
    latency: float
    amplitude: float
    polarity: int
    fwhm: float
    onset: float
    offset: float
    area: float
    rise_time: float
    decay_time: float
    symmetry: float
    sharpness: float
    label: str | None = None
    se: dict | None = None
    ci: dict | None = None
    posterior: dict | None = None
    # False when the component cannot be told apart from a neighbour: their amplitudes (or
    # latencies) correlate beyond 0.95, or two components of opposite sign overlap and
    # cancel (spec §5.5.3 step 6, risk S1): its numbers are not to be read alone
    resolvable: bool = True

    def __post_init__(self):
        self.params = np.asarray(self.params, dtype=np.float64)
        self.resolvable = bool(self.resolvable)
        self.param_names = tuple(self.param_names)
        if self.params.shape != (len(self.param_names),):
            raise DataError(f"Component: {self.params.size} params for "
                            f"{len(self.param_names)} param_names.")
        if self.polarity not in (1, -1):
            raise DataError(f"Component: polarity must be +1 or -1, got {self.polarity!r}.")
        self.polarity = int(self.polarity)
        for name in COMPONENT_FEATURES:
            if name != "polarity":
                setattr(self, name, float(getattr(self, name)))
        if self.ci is not None:
            self.ci = {key: tuple(float(v) for v in value) for key, value in self.ci.items()}
        if self.se is not None:
            self.se = {key: float(value) for key, value in self.se.items()}


@dataclass(eq=False)
class GoodnessOfFit(_ResultMixin):
    """Fit quality on the fit window. ``r2`` is 1 − SS_res / SS_tot (not a squared correlation)."""

    r2: float
    adj_r2: float
    rmse: float
    mae: float
    nrmse: float
    aic: float
    aicc: float
    bic: float
    n_params: int
    n_samples: int
    n_eff: float
    residual_lag1_autocorr: float
    baseline_sd: float

    def __post_init__(self):
        for fld in fields(self):
            value = getattr(self, fld.name)
            setattr(self, fld.name, int(value) if fld.name in ("n_params", "n_samples")
                    else float(value))


@dataclass(eq=False)
class ResponseTest(_ResultMixin):
    """Output of the existence test: is there a response on each channel?"""

    method: str
    statistic: np.ndarray
    p_value: np.ndarray
    p_value_raw: np.ndarray
    significant: np.ndarray
    duration: np.ndarray
    onset: np.ndarray
    offset: np.ndarray
    snr_db: np.ndarray
    rule: str | None
    alpha: float
    correction: str
    n_permutations: int
    window: tuple
    settings: dict
    provenance: Provenance = field(default_factory=Provenance)
    curve_times: np.ndarray | None = None
    curve: np.ndarray | None = None
    null_band: np.ndarray | None = None

    def __post_init__(self):
        for name in ("statistic", "p_value", "p_value_raw", "duration", "onset", "offset",
                     "snr_db"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float64))
        self.significant = np.asarray(self.significant, dtype=bool)
        self.window = tuple(float(v) for v in self.window)
        n_channels = self.statistic.shape
        for name in ("p_value", "p_value_raw", "significant", "duration", "onset", "offset",
                     "snr_db"):
            if getattr(self, name).shape != n_channels:
                raise DataError(f"ResponseTest: '{name}' has shape {getattr(self, name).shape}, "
                                f"expected {n_channels}.")
        # The statistic over time (spec §5.13): curve [*statistic.shape, n_times] on
        # curve_times, and its null band [*statistic.shape, n_times, 2] (low, high; NaN on
        # a side the test does not look at). None for a method without a time course.
        for name in ("curve_times", "curve", "null_band"):
            if getattr(self, name) is not None:
                setattr(self, name, np.asarray(getattr(self, name), dtype=np.float64))
        if self.curve is not None:
            n_times = 0 if self.curve_times is None else self.curve_times.size
            if self.curve.shape != n_channels + (n_times,):
                raise DataError(f"ResponseTest: 'curve' has shape {self.curve.shape}, "
                                f"expected {n_channels + (n_times,)}.")
            if self.null_band is not None and self.null_band.shape != self.curve.shape + (2,):
                raise DataError(f"ResponseTest: 'null_band' has shape {self.null_band.shape},"
                                f" expected {self.curve.shape + (2,)}.")


@dataclass(eq=False)
class ResponseFit(_ResultMixin):
    """Result of fitting one waveform (one channel, one condition)."""

    components: list
    background: dict
    gof: GoodnessOfFit | None
    status: str
    message: str
    window: tuple
    baseline: tuple
    settings: dict
    classical: dict = field(default_factory=dict)
    model: np.ndarray | None = None
    residual: np.ndarray | None = None
    existence: ResponseTest | None = None
    warnings: list = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in FIT_STATUSES:
            raise DataError(f"ResponseFit: status must be one of {FIT_STATUSES}, "
                            f"got {self.status!r}.")
        if self.status != "ok" and not self.message:
            raise DataError(f"ResponseFit: status '{self.status}' needs a message.")
        self.components = sorted(self.components, key=lambda comp: comp.latency)
        self.window = tuple(float(v) for v in self.window)
        self.baseline = tuple(float(v) for v in self.baseline)
        self.model = _as_float_array(self.model)
        self.residual = _as_float_array(self.residual)
        self.warnings = list(self.warnings)

    def to_dataframe(self, channel="", condition=""):
        """Tidy table, one row per component (one row with NaN features when there is none).

        Column units are in ``table.attrs["units"]`` (see :data:`COMPONENT_COLUMNS`).
        """

        import pandas as pd
        rows = []
        for position, comp in enumerate(self.components):
            row = {"channel": channel, "condition": condition, "component": position,
                   "label": comp.label if comp.label is not None else "", "shape": comp.shape}
            row.update({name: getattr(comp, name) for name in COMPONENT_FEATURES})
            row["resolvable"] = comp.resolvable
            row["status"] = self.status
            for name in INTERVAL_FEATURES:
                low, high = (comp.ci or {}).get(name, (np.nan, np.nan))
                row[f"{name}_ci_low"], row[f"{name}_ci_high"] = float(low), float(high)
            rows.append(row)
        if not rows:
            rows.append({"channel": channel, "condition": condition, "component": -1,
                         "label": "", "shape": "", **{name: np.nan for name in COMPONENT_FEATURES},
                         "resolvable": pd.NA, "status": self.status,
                         **{name: np.nan for name in INTERVAL_COLUMNS}})
            rows[0]["polarity"] = 0
        table = pd.DataFrame(rows, columns=list(COMPONENT_COLUMNS) + list(INTERVAL_COLUMNS))
        table["polarity"] = table["polarity"].astype("int64")
        table["component"] = table["component"].astype("int64")
        table["resolvable"] = pd.array(table["resolvable"].tolist(), dtype="boolean")
        # The existence test of this channel, next to the fitted components, so that a
        # component on a channel without a significant response is never hidden
        if self.existence is not None:
            table["existence_p"] = float(np.ravel(self.existence.p_value)[0])
            table["existence_significant"] = bool(np.ravel(self.existence.significant)[0])
        else:
            table["existence_p"] = np.nan
            table["existence_significant"] = pd.array([pd.NA] * len(table), dtype="boolean")
        table.attrs["units"] = {**COMPONENT_COLUMNS, **INTERVAL_COLUMNS, **EXISTENCE_COLUMNS}
        return table


@dataclass(eq=False)
class GroupFit(_ResultMixin):
    """One :class:`ResponseFit` per channel (and per condition when the input was grouped).

    ``index`` is a table with columns ``channel``, ``condition`` and
    ``position`` (the position of the fit in ``fits``).
    """

    fits: list
    ch_names: list
    conditions: list | None
    index: object
    settings: dict = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)

    def __post_init__(self):
        self.fits = list(self.fits)
        self.ch_names = [str(name) for name in self.ch_names]
        if self.conditions is not None:
            self.conditions = [str(cond) for cond in self.conditions]
        if len(self.index) != len(self.fits):
            raise DataError(f"GroupFit: index has {len(self.index)} rows for "
                            f"{len(self.fits)} fits.")

    @classmethod
    def from_fits(cls, fits, ch_names, conditions=None, settings=None, provenance=None):
        """Build the ``index`` table from parallel lists of fits, channels and conditions."""

        import pandas as pd
        if len(fits) != len(ch_names) or (conditions is not None and len(conditions) != len(fits)):
            raise DataError("GroupFit.from_fits: fits, ch_names and conditions must have "
                            "the same length.")
        index = pd.DataFrame({"channel": [str(name) for name in ch_names],
                              "condition": [str(c) for c in conditions] if conditions is not None
                              else [""] * len(fits),
                              "position": np.arange(len(fits), dtype=np.int64)})
        unique_channels = list(dict.fromkeys(index["channel"]))
        unique_conditions = None if conditions is None else list(dict.fromkeys(index["condition"]))
        return cls(fits=list(fits), ch_names=unique_channels, conditions=unique_conditions,
                   index=index, settings=settings or {},
                   provenance=provenance if provenance is not None else Provenance())

    def get(self, channel, condition=None):
        """Return the fit of one channel (and condition)."""

        rows = self.index[self.index["channel"] == channel]
        if condition is not None:
            rows = rows[rows["condition"] == str(condition)]
        if len(rows) != 1:
            raise KeyError(f"{len(rows)} fits match channel={channel!r}, condition={condition!r}.")
        return self.fits[int(rows["position"].iloc[0])]

    def to_dataframe(self, level="component"):
        """Tidy table: ``"component"`` (one row per component) or ``"fit"`` (one row per fit).

        Column units are in ``table.attrs["units"]``.
        """

        import pandas as pd
        if level == "component":
            parts = [self.fits[int(pos)].to_dataframe(channel=ch, condition=cond)
                     for ch, cond, pos in self.index[["channel", "condition", "position"]]
                     .itertuples(index=False)]
            table = pd.concat(parts, ignore_index=True) if parts else \
                pd.DataFrame(columns=list(COMPONENT_COLUMNS) + list(INTERVAL_COLUMNS)
                             + list(EXISTENCE_COLUMNS))
            table.attrs["units"] = {**COMPONENT_COLUMNS, **INTERVAL_COLUMNS,
                                    **EXISTENCE_COLUMNS}
            return table
        if level == "fit":
            rows = []
            for ch, cond, pos in self.index[["channel", "condition", "position"]] \
                    .itertuples(index=False):
                fit = self.fits[int(pos)]
                row = {"channel": ch, "condition": cond, "status": fit.status,
                       "message": fit.message, "n_components": len(fit.components)}
                for name in list(FIT_COLUMNS)[5:]:
                    row[name] = getattr(fit.gof, name) if fit.gof is not None else np.nan
                rows.append(row)
            table = pd.DataFrame(rows, columns=list(FIT_COLUMNS))
            table.attrs["units"] = dict(FIT_COLUMNS)
            return table
        raise DataError(f"GroupFit.to_dataframe: level must be 'component' or 'fit', got {level!r}.")

    def to_annotations(self, *, what="components", orig_time=None):
        """``mne.Annotations`` of the components, for display on MNE plots: one per
        component, onset to offset, description ``"<label>|<channel>"`` (extra ``mne``)."""

        import mne

        onsets, durations, descriptions, channels = [], [], [], []
        for ch, cond, pos in self.index[["channel", "condition", "position"]].itertuples(
                index=False):
            for k, comp in enumerate(self.fits[int(pos)].components):
                if not (np.isfinite(comp.onset) and np.isfinite(comp.offset)):
                    continue
                onsets.append(float(comp.onset))
                durations.append(float(comp.offset - comp.onset))
                descriptions.append(f"{comp.label or f'c{k}'}|{ch}")
                channels.append((ch,))
        return mne.Annotations(onsets, durations, descriptions, orig_time=orig_time,
                               ch_names=channels)

    def to_evoked_model(self, *, sfreq=None):
        """The fitted curves (fit window, zero elsewhere) as an ``mne.EvokedArray`` whose
        comment is ``"evoparam model"`` (extra ``mne``)."""

        import mne

        fits = [self.fits[int(p)] for p in self.index["position"]]
        lengths = [len(f.model) for f in fits if f.model is not None]
        if not lengths:
            raise DataError("to_evoked_model: the fits hold no model curve (saved light?).")
        starts = [f.window[0] for f in fits]
        stops = [f.window[1] for f in fits]
        ref = next(f for f in fits if f.model is not None)
        rate = sfreq or (len(ref.model) - 1) / (ref.window[1] - ref.window[0])
        times = np.arange(min(starts), max(stops) + 0.5 / rate, 1 / rate)
        data = np.zeros((len(fits), times.size))
        for i, f in enumerate(fits):
            if f.model is not None and len(f.model):
                grid = np.linspace(f.window[0], f.window[1], len(f.model))
                inside = (times >= f.window[0]) & (times <= f.window[1])
                data[i, inside] = np.interp(times[inside], grid, f.model)
        names = [f"{c}|{d}" if d else str(c) for c, d in
                 zip(self.index["channel"], self.index["condition"])]
        info = mne.create_info(names, float(rate), "misc")
        return mne.EvokedArray(data, info, tmin=float(times[0]), comment="evoparam model",
                               verbose="error")

    def save(self, path, overwrite=False, light=False):
        """Save to a ``.evoparam`` container; ``light=True`` drops fitted curves and residuals."""

        obj = self
        if light:
            obj = replace(self, fits=[replace(fit, model=None, residual=None) for fit in self.fits])
        from evoparam.data.serialize import save_object
        return save_object(obj, path, overwrite=overwrite)


###################################################################################################
## Trial-level and hierarchical results

@dataclass(eq=False)
class TrialFit(_ResultMixin):
    """Trial-level parameters for one channel (and condition)."""

    channel: str
    template: ResponseFit
    shift: np.ndarray
    scale: np.ndarray
    params: np.ndarray
    latency: np.ndarray
    amplitude: np.ndarray
    fwhm: np.ndarray
    gof: np.ndarray
    converged: np.ndarray
    jitter_sd: dict
    jitter_sd_null: dict
    amplitude_jitter_corrected: dict
    trusted: np.ndarray
    table: object
    covariates: object = None
    realigned: object = None
    settings: dict = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)

    def __post_init__(self):
        for name in ("shift", "scale", "params", "latency", "amplitude", "fwhm", "gof"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float64))
        self.converged = np.asarray(self.converged, dtype=bool)
        self.trusted = np.asarray(self.trusted, dtype=bool)
        n_trials = self.shift.shape[0]
        for name in ("scale", "params", "latency", "amplitude", "fwhm", "gof", "converged",
                     "trusted"):
            if getattr(self, name).shape[:1] != (n_trials,):
                raise DataError(f"TrialFit: '{name}' must have {n_trials} rows.")


@dataclass(eq=False)
class HierarchicalFit(_ResultMixin):
    """Trial → channel → subject model summary."""

    levels: tuple
    summary: object
    level_tables: dict
    p_response: object
    variance_components: object
    effects: object
    engine: str
    diagnostics: dict
    settings: dict = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)

    def __post_init__(self):
        self.levels = tuple(self.levels)
        if self.engine not in ("eb", "numpyro", "pymc"):
            raise DataError(f"HierarchicalFit: engine must be 'eb', 'numpyro' or 'pymc', "
                            f"got {self.engine!r}.")

    def _normal_draws(self, n_draws, random_state):
        """Draws from the normal approximation of the empirical-Bayes tables, in the layout
        of ``diagnostics["draws"]`` (one chain)."""

        rng = np.random.default_rng(random_state)
        out = {}
        tables = self.level_tables or {}
        for level, column in (("channel", "sites"), ("subject", "subjects")):
            table = tables.get(level)
            if table is None or len(table) == 0 or "component" not in table.columns:
                continue
            for label in dict.fromkeys(table["component"]):
                part = table[table["component"] == label]
                block = out.setdefault(str(label), {})
                if level == "channel":
                    block[column] = [f"{a}/{b}" for a, b in zip(part["subject"], part["channel"])]
                else:
                    block[column] = [str(v) for v in part["subject"]]
                for parameter in ("latency", "amplitude"):
                    if parameter not in part.columns:
                        continue
                    mean = part[parameter].to_numpy(dtype=float)
                    se = (part[f"{parameter}_se"].to_numpy(dtype=float)
                          if f"{parameter}_se" in part.columns else np.zeros(len(part)))
                    se = np.where(np.isfinite(se), se, 0.0)
                    block[f"{parameter}_{level}"] = (
                        mean[None, None, :]
                        + se[None, None, :] * rng.standard_normal((1, n_draws, len(part))))
        return out

    def to_arviz(self, *, n_draws=1000, random_state=None):
        """The posterior as an ArviZ ``InferenceData`` (with ArviZ 1.x, the ``DataTree``
        that replaces it), for ``arviz`` summaries and plots (extra ``bayes``).

        Variables, per component label ``<c>``: ``<c>_latency_channel`` and
        ``<c>_amplitude_channel`` (dim ``site``, one per subject and channel, in the order of
        the channel table) and ``<c>_latency_subject``, ``<c>_amplitude_subject`` (dim
        ``subject``), in seconds and in the data unit.

        With a sampling engine (``numpyro``, ``pymc``) these are the posterior draws kept in
        ``diagnostics["draws"]`` (per label: arrays chain x draw x site or subject, with the
        ``sites`` and ``subjects`` names). The empirical-Bayes engine (``eb``) has no draws:
        they are drawn from its normal approximation (estimate and standard error of each row of the
        level tables; ``n_draws``, ``random_state``), which the attribute ``approximation``
        states.
        """

        try:
            import arviz as az
        except ImportError as err:
            raise ImportError("HierarchicalFit.to_arviz needs ArviZ, in the extra 'bayes': "
                              "pip install \"evoparam[bayes]\".") from err
        draws = (self.diagnostics or {}).get("draws")
        approximation = "none (posterior draws)"
        if not draws:
            approximation = "normal (empirical-Bayes estimate and standard error)"
            draws = self._normal_draws(int(n_draws), random_state)
        posterior, coords, dims = {}, {}, {}
        for label, block in draws.items():
            for level, dim, names in (("channel", f"site_{label}", block.get("sites")),
                                      ("subject", f"subject_{label}", block.get("subjects"))):
                for parameter in ("latency", "amplitude"):
                    values = block.get(f"{parameter}_{level}")
                    if values is None:
                        continue
                    values = np.asarray(values, dtype=np.float64)
                    key = f"{label}_{parameter}_{level}"
                    posterior[key] = values
                    dims[key] = [dim]
                    coords[dim] = (list(names) if names is not None
                                   and len(names) == values.shape[-1]
                                   else list(range(values.shape[-1])))
        if not posterior:
            raise DataError("HierarchicalFit.to_arviz: no posterior to export (empty tables).")
        if int(str(az.__version__).split(".")[0]) >= 1:
            data = az.from_dict({"posterior": posterior}, coords=coords, dims=dims)
        else:
            data = az.from_dict(posterior=posterior, coords=coords, dims=dims)
        try:
            data.attrs["approximation"] = approximation
            data.attrs["engine"] = self.engine
        except (AttributeError, TypeError):
            pass
        return data


###################################################################################################
## Other module results

@dataclass(eq=False)
class ArtifactResult(_ResultMixin):
    """Output of stimulation-artifact removal."""

    cleaned: object
    method: str
    window: tuple
    template: np.ndarray | None
    quality: np.ndarray
    quality_components: object
    basis: np.ndarray | None = None
    settings: dict = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)

    def __post_init__(self):
        self.window = tuple(float(v) for v in self.window)
        self.template = _as_float_array(self.template)
        self.basis = _as_float_array(self.basis)
        self.quality = np.asarray(self.quality, dtype=np.float64)

    def quality_summary(self):
        """Summary of the per-trial quality scores: ``{"n_trials", "median", "p05"}``
        (median and 5th percentile of the finite scores; None when there is none)."""

        quality = np.ravel(self.quality)
        finite = quality[np.isfinite(quality)]
        return {"n_trials": int(quality.size),
                "median": float(np.median(finite)) if finite.size else None,
                "p05": float(np.percentile(finite, 5)) if finite.size else None}


@dataclass(eq=False)
class ShapeBasis(_ResultMixin):
    """Shared response shapes across channels or stimulation sites."""

    curves: np.ndarray
    times: np.ndarray
    weights: np.ndarray
    r2: np.ndarray
    assignment: object
    stability: np.ndarray
    n_shapes: int
    method: str
    settings: dict = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)

    def __post_init__(self):
        for name in ("curves", "times", "weights", "r2", "stability"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float64))
        self.n_shapes = int(self.n_shapes)
        if self.curves.shape != (self.n_shapes, self.times.size):
            raise DataError(f"ShapeBasis: curves shape {self.curves.shape} differs from "
                            f"({self.n_shapes}, {self.times.size}).")


@dataclass(eq=False)
class TFFit(_ResultMixin):
    """Time-frequency maps, significance and data-driven zones."""

    freqs: np.ndarray
    times: np.ndarray
    maps: dict
    tmaps: np.ndarray
    significant: np.ndarray
    zones: object
    zone_metrics: object
    method: str
    settings: dict = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)

    def __post_init__(self):
        self.freqs = np.asarray(self.freqs, dtype=np.float64)
        self.times = np.asarray(self.times, dtype=np.float64)
        self.tmaps = np.asarray(self.tmaps, dtype=np.float64)
        self.significant = np.asarray(self.significant, dtype=bool)
        self.maps = {key: np.asarray(value, dtype=np.float64) for key, value in self.maps.items()}
        if self.significant.shape != self.tmaps.shape:
            raise DataError("TFFit: significant and tmaps must have the same shape.")

    def to_mne(self, map="total", *, ch_type="misc"):
        """One map as ``mne.time_frequency.AverageTFR`` (extra ``mne``).

        Parameters
        ----------
        map : str, default: "total"
            Key of :attr:`maps` holding channels x freqs x times (``"total"``,
            ``"total_norm"``, ``"evoked_footprint"``, ``"residual"``, ``"cross"``, ...).
        ch_type : str, default: "misc"
            MNE channel type of every row.

        Returns
        -------
        mne.time_frequency.AverageTFR
            One channel per row of the fit (``"<channel>"``, or ``"<channel>|<group>"``
            when the fit was split by condition), ``nave`` the smallest number of trials
            of a row, ``comment`` the map name and ``method`` the transform.
        """

        try:
            import mne
        except ImportError as err:
            raise ImportError("TFFit.to_mne needs the extra 'mne': pip install "
                              "\"evoparam[mne]\".") from err
        if map not in self.maps:
            raise DataError(f"TFFit.to_mne: unknown map {map!r}; available: "
                            f"{sorted(self.maps)}.")
        data = np.asarray(self.maps[map], dtype=np.float64)
        if data.ndim != 3 or data.shape[1:] != (self.freqs.size, self.times.size):
            raise DataError(f"TFFit.to_mne: map {map!r} is not channels x freqs x times.")
        rows = self.settings.get("rows") or [[str(i), ""] for i in range(data.shape[0])]
        names = [f"{row[0]}|{row[1]}" if len(row) > 1 and row[1] else str(row[0])
                 for row in rows]
        sfreq = 1.0 / float(np.median(np.diff(self.times))) if self.times.size > 1 else 1.0
        info = mne.create_info(names, sfreq, ch_type)
        n_trials = self.settings.get("n_trials") or [1]
        nave = int(min(n_trials))
        method = str(self.settings.get("transform", "unknown"))
        if hasattr(mne.time_frequency, "AverageTFRArray"):
            return mne.time_frequency.AverageTFRArray(
                info=info, data=data, times=self.times, freqs=self.freqs, nave=nave,
                comment=map, method=method)
        return mne.time_frequency.AverageTFR(info, data, self.times, self.freqs, nave,
                                             comment=map, method=method)

    def methods_text(self):
        """Methods paragraph: transform and its parameters, baseline window and mode, the
        statistic, alpha and the zone method (spec REQ-219)."""

        s = self.settings or {}
        tp = s.get("transform_params") or {}

        def _g(value):                   # a value missing from the settings is named as such
            return f"{value:g}" if isinstance(value, (int, float)) else "not recorded"

        transform = s.get("transform", "?")
        if transform == "morse":
            what = (f"generalised Morse wavelets (gamma = {_g(tp.get('gamma'))}, "
                    f"beta = {_g(tp.get('beta'))})")
        elif transform == "morlet":
            cycles = np.unique(np.round(tp.get("n_cycles", [np.nan]), 3))
            what = f"complex Morlet wavelets ({', '.join(f'{c:g}' for c in cycles)} cycles)"
        elif transform == "multitaper":
            what = (f"DPSS multitapers (time-bandwidth product {_g(tp.get('time_bandwidth'))}, "
                    f"{tp.get('n_tapers')} tapers)")
        else:
            what = f"a short-time Fourier transform (Hann window of {_g(tp.get('stft_window'))} s)"
        freqs = s.get("freqs", [np.nan])
        base = s.get("baseline", [np.nan, np.nan])
        win = s.get("window", [np.nan, np.nan])
        level = {"single_trial": "each trial by its own baseline",
                 "evoked": "the trial-average power by its baseline",
                 "full_epoch": "each trial by its own full-epoch mean"}.get(
                     s.get("baseline_level"), s.get("baseline_level"))
        stat = {"fdr": "pixel-wise one-sample t-tests across trials with Benjamini-Hochberg "
                       "false-discovery-rate control",
                "cluster": f"a cluster-mass sign-flip permutation test "
                           f"({s.get('n_permutations')} permutations)",
                "tfce": f"threshold-free cluster enhancement (E = 0.5, H = 2) with "
                        f"{s.get('n_permutations')} sign-flip permutations",
                "baseline_normal": "a comparison of each pixel of the trial-average dB power "
                                   "with a normal distribution fitted to the baseline of its "
                                   "frequency (two-sided), with Benjamini-Hochberg "
                                   "false-discovery-rate control",
                "none": "no statistical test"}.get(s.get("stats"), s.get("stats"))
        zones = {"nmf": "data-driven zones were found by non-negative matrix factorisation of "
                        "the significant group map, with the rank chosen by split-half "
                        "stability (congruence >= 0.8, at most "
                        f"{s.get('max_zones')} zones)",
                 "cp": "data-driven zones were found by a non-negative canonical-polyadic "
                       "decomposition (frequency x time x channel), with the rank chosen by "
                       "split-half stability over the trials of each channel (congruence "
                       ">= 0.8, at most "
                       f"{s.get('max_zones')} zones)",
                 "published": f"the published zone '{(s.get('published_zone') or {}).get('name')}'"
                              f" ({(s.get('published_zone') or {}).get('source')}) was used",
                 "none": "no zones were computed"}.get(self.method, self.method)
        across = {
            "fdr_bh": "the channels were selected by Benjamini-Hochberg false-discovery-rate "
                      "control on their smallest adjusted p-value, and the pixels of each "
                      "selected channel were tested at the level of that selection "
                      "(Benjamini and Bogomolov 2014)",
            "holm": "the level of each channel's pixels was set by the Holm step-down "
                    "correction of the channels' smallest adjusted p-values",
            "bonferroni": "each channel was tested at alpha divided by the number of "
                          "channels (Bonferroni)",
            "none": "no correction was applied across channels"}.get(
                s.get("channel_correction"),
                f"{s.get('channel_correction')} correction across channels")
        text = (f"Time-frequency power was computed with {what} at {len(freqs)} frequencies "
                f"from {freqs[0]:.3g} to {freqs[-1]:.3g} Hz; pixels within the wavelet "
                f"support of an epoch edge or of the artifact mask were excluded. Power was "
                f"normalised ({s.get('baseline_mode')}) {level}, baseline {base[0] * 1000:g} "
                f"to {base[1] * 1000:g} ms. Pixels from {win[0] * 1000:g} to "
                f"{win[1] * 1000:g} ms were tested with {stat} (alpha = {_g(s.get('alpha'))}); "
                f"{across}. "
                f"{zones[0].upper() + zones[1:]}.")
        if s.get("footprint"):
            text += (" The power of the fitted evoked components (footprint) was separated "
                     "from the residual power; the cross term is reported.")
        return text + f" (evoparam {s.get('package_version')}, seed {s.get('random_state')})."
