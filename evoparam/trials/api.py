# SPDX-License-Identifier: Apache-2.0
"""Trial-by-trial parameters, jitter, realignment and covariates (spec §5.7)."""

import dataclasses
import functools
import logging
import os
import time as _time
import warnings
from types import SimpleNamespace
from typing import NamedTuple, cast

import numpy as np

from evoparam._typing import Any, FloatArray, NDArray
from evoparam.data import Evoked, Provenance, TrialFit, TrialSet
from evoparam.data._table import concat_tables, new_table, row_labels
from evoparam.errors import DataError, SettingsError
from evoparam.errors import warn as _warn
from evoparam.models.component_shapes import (
    FWHM_PER_SIGMA,
    Template,
    compute_features,
    get_background,
    get_shape,
)
from evoparam.utils._failures import (
    PROPAGATE,
    check_on_error,
    fit_error,
    record_failure,
)
from evoparam.utils._logging import verbose
from evoparam.utils._optional import require
from evoparam.utils._parallel import parallel_map
from evoparam.version import __version__

from ._model import ChannelTemplate, fit_trial, fit_trials_batch

# Fewest samples of a component's own support before the support falls back to its peak
_MIN_SUPPORT = 4
# Fewest trials for leave-one-trial-out ridge selection
_MIN_LOTO_TRIALS = 3
# Fewest baseline samples of a trial to build null segments from
_MIN_NOISE_SAMPLES = 4
# Fewest values for a standard deviation of shifts (jitter, null jitter, bootstrap)
_MIN_SD_VALUES = 3
# The F quantile of the jitter refusal needs more than this many trials on both sides
_F_MIN_TRIALS = 2
# Fewest trusted trials for a trend (habituation) fit
_MIN_TREND_TRIALS = 3
# Fewest trials per state, and states, for the state-dependence tests
_MIN_STATE_TRIALS = 2
_MIN_STATES = 2
# Fewest doses for a dose-response fit, and for the "auto" model to choose the Hill model
_MIN_DOSES = 2
_HILL_MIN_DOSES = 4


logger = logging.getLogger(__name__)

__all__ = [
    "fit_trials",
    "TrialFitSet",
    "jitter",
    "realign",
    "dose_response",
    "prestimulus_dependence",
    "MODES",
    "state_dependence",
    "COVARIATES",
]

MODES = ("shift_scale", "free", "template", "regularized")
COVARIATES = ("trend", "dose", "prestim", "state")
SCALP_PRESETS = ("erp", "p300_bci", "tep", "sep")
JITTER_REFUSE = 1.2
NULL_REPEATS = 5
RIDGE_GRID = (0.0, 0.01, 0.1, 1.0, 10.0)
RIDGE_CV = ("loto", "split")
NULL_RULES = ("ar", "segments", "spec")
# Order of the autoregressive noise model of null_rule="ar" (fewer with a short baseline)
NULL_AR_ORDER = 8
# Simulated noise segments per trial of null_rule="ar" (the null SD to about 2.5 % with 40 trials)
NULL_AR_REPEATS = 20
TRIAL_MODELS = ("all", "anchored", "selected")
MODALITY_PRESETS = {
    "spes": "ccep",
    "erp": "erp",
    "tep": "tep",
    "erna": "erna",
    "sep": "sep",
    "ecap": "ecap",
    "lfp_ep": "rodent_ep",
}


###################################################################################################
## Container


@dataclasses.dataclass(eq=False)
class TrialFitSet:
    """Trial fits of every channel (and condition).

    ``fits`` (list of :class:`TrialFit`), the tidy ``table`` of all of them, ``skipped_channels``
    and ``settings``.
    """

    fits: list[Any]
    table: Any
    skipped_channels: list[Any]
    settings: dict[str, Any]
    provenance: Provenance = dataclasses.field(default_factory=Provenance)

    def to_dataframe(self) -> Any:
        """Return a copy of the tidy table of all the trial fits."""
        return self.table.copy()

    def get(self, channel: Any, condition: Any = None) -> Any:
        """Return the trial fit of ``channel`` (and ``condition``)."""
        for fit in self.fits:
            if fit.channel == channel and (
                condition is None or fit.settings.get("condition") == condition
            ):
                return fit
        raise KeyError(f"No trial fit for channel {channel!r}.")

    def save(self, path: Any, overwrite: bool = False) -> Any:
        """Save to a folder: one ``.evoparam`` container per channel fit, and the table."""
        import json

        if os.path.exists(path) and not overwrite and os.listdir(path):
            raise DataError(f"{path} exists and is not empty (overwrite=True to replace).")
        os.makedirs(path, exist_ok=True)
        for i, fit in enumerate(self.fits):
            fit.save(os.path.join(path, f"fit_{i:04d}.evoparam"), overwrite=True)
        self.table.to_csv(os.path.join(path, "table.csv"), index=False)
        from evoparam.data.serialize import _Encoder

        # settings and provenance in the typed encoding of the containers (tuples and NaN
        # come back as they were)
        encoder = _Encoder(None)
        content = {
            "n_fits": len(self.fits),
            "skipped_channels": self.skipped_channels,
            "package_version": __version__,
            "settings": encoder.encode(self.settings, "settings"),
            "provenance": encoder.encode(self.provenance, "provenance"),
        }
        if encoder.arrays or encoder.tables:
            raise DataError("TrialFitSet.save: settings must not hold arrays or tables.")
        with open(os.path.join(path, "set.json"), "w", encoding="utf-8") as fobj:
            json.dump(content, fobj)
        return path

    @classmethod
    def load(cls, path: Any) -> "TrialFitSet":
        """Load a set saved by :meth:`save`."""
        import json

        pd = require("pandas", extra="data", caller="evoparam.trials.TrialFitSet.load")
        from evoparam.data.serialize import _Decoder

        with open(os.path.join(path, "set.json"), encoding="utf-8") as fobj:
            meta = json.load(fobj)
        decoder = _Decoder(path, {}, False)
        fits = [
            TrialFit.load(os.path.join(path, f"fit_{i:04d}.evoparam"))
            for i in range(meta["n_fits"])
        ]
        table = (
            pd.concat([f.table for f in fits], ignore_index=True)
            if fits
            else pd.read_csv(os.path.join(path, "table.csv"))
        )
        settings = decoder.decode(meta["settings"]) if "settings" in meta else {}
        provenance = decoder.decode(meta["provenance"]) if "provenance" in meta else Provenance()
        return cls(
            fits=fits,
            table=table,
            skipped_channels=meta["skipped_channels"],
            settings=settings,
            provenance=provenance,
        )


###################################################################################################
## Helpers


def _shape_for(
    comp: Any, settings: Any
) -> Any:  # a ComponentShape (shapes is reached through models only)
    """Return the shape of ``comp`` with the width limits of the channel fit's settings."""
    name = comp.shape
    fields = {f.name for f in dataclasses.fields(cast(Any, type(get_shape(name))))}
    kwargs = {k: settings[k] for k in ("min_fwhm", "max_fwhm") if k in fields and k in settings}
    return get_shape(name, **kwargs)


def _amplitude_of(comp: Any) -> Any:
    """Return the amplitude that scales the shape of ``comp``: its CI centre when it has a CI."""
    ci = comp.ci or {}
    return 0.5 * (ci["A"][0] + ci["A"][1]) if "A" in ci else comp.amplitude


def _template_for(
    fit: Any, picks: Any, times: Any, max_shift: Any, factor: Any, moving: Any = None
) -> "ChannelTemplate":
    """Channel template restricted to the chosen components."""
    settings = fit.settings
    window = fit.window
    tw = times[(times >= window[0]) & (times <= window[1])]
    shapes, params, locs, widths = [], [], [], []
    for k in picks:
        comp = fit.components[k]
        shape = _shape_for(comp, settings)
        shapes.append(shape)
        params.append(np.array(comp.params, float))
        locs.append(shape.param_names.index(shape.latency_param))
        widths.append(comp.fwhm)
    if max_shift == "auto":
        shifts = np.array([factor * w for w in widths], float)
    else:
        shifts = np.full(len(picks), float(max_shift))
    # A trial's component must not reach its neighbour: at most half the distance
    lats = [fit.components[k].latency for k in picks]
    for i in range(len(picks)):
        if moving is not None and i not in moving:
            continue
        gaps = [
            abs(lats[i] - lats[j])
            for j in range(len(picks))
            if j != i and (moving is None or j in moving)
        ]
        if gaps and shifts[i] > 0.5 * min(gaps):
            _warn(
                f"max_shift of component {picks[i]} clipped to half the distance to "
                "its neighbour.",
                UserWarning,
                # _fit_unit, _fit_channel, fit_trials, then its caller (as before the move)
                stacklevel=5,
            )
            shifts[i] = 0.5 * min(gaps)
    bg = fit.background or {"name": "none"}
    background = get_background(bg["name"])
    nl = np.array([bg[n] for n in getattr(background, "nonlinear_names", ())], float)
    return ChannelTemplate(shapes, params, locs, np.maximum(shifts, 1e-6), background, nl, tw)


def _template_mode(template: Any, fit: Any, average: Any, times: Any, picks: Any) -> Any:
    """Replace each component by the channel average over its support (Woody template)."""
    tw = template.tw
    win = (times >= tw[0]) & (times <= tw[-1])
    y = average[win]
    curves = [
        s.evaluate(tw, p) * _amplitude_of(fit.components[k])
        for s, p, k in zip(template.shapes, template.params, picks, strict=False)
    ]
    total = np.sum(curves, axis=0)
    shapes, params, locs = [], [], []
    for i, k in enumerate(picks):
        comp = fit.components[k]
        own = y - (total - curves[i])
        support = (tw >= comp.onset) & (tw <= comp.offset)
        if support.sum() < _MIN_SUPPORT:
            support = np.abs(curves[i]) >= 0.1 * np.max(np.abs(curves[i]))
        values = np.where(support, own, 0.0)
        values[0] = values[-1] = 0.0
        shape = Template(
            template_times=tw, template_values=values, max_shift=float(template.max_shift[i])
        )
        shapes.append(shape)
        params.append(np.array([0.0]))
        locs.append(0)
    return dataclasses.replace(template, shapes=shapes, params=params, loc_index=locs)


def _fit_all(
    template: Any,
    data: Any,
    valid: Any,
    mode: Any,
    ridge: Any,
    support: Any = None,
    movable: Any = None,
) -> Any:
    """Return the fit of every trial: one vectorised solve, or one fit per trial."""
    if mode == "shift_scale" and not ridge:
        # every trial at once (the same model, a vectorised solver)
        return fit_trials_batch(template, data, valid=valid, support=support, movable=movable)
    return [
        fit_trial(
            template,
            data[i],
            mode=mode,
            ridge=ridge,
            valid=valid[i],
            support=support,
            movable=movable,
        )
        for i in range(len(data))
    ]


def _choose_ridge(template: Any, data: Any, valid: Any, movable: Any = None) -> float:
    """Penalty chosen by split-sample prediction error: fit on even samples, score on odd."""
    even = np.zeros(data.shape[1], bool)
    even[::2] = True
    best, best_err = 0.0, np.inf
    for lam in RIDGE_GRID:
        err = 0.0
        for i in range(len(data)):
            res = fit_trial(
                template,
                data[i],
                mode="regularized",
                ridge=lam,
                valid=valid[i] & even,
                tol=1e-6,
                movable=movable,
            )
            odd = valid[i] & ~even
            err += np.sum((data[i][odd] - res["fitted"][odd]) ** 2)
        if err < best_err:
            best, best_err = lam, err
    return best


def _choose_ridge_loto(template: Any, data: Any, valid: Any, movable: Any = None) -> float:
    """Penalty chosen by leave-one-trial-out prediction error (spec §5.7).

    For each trial, the centre of the penalty is the channel model refitted to the average of the
    *other* trials (``free`` fit of the channel components, warm-started at the channel values);
    the trial is fitted with each penalty towards that centre and scored on its own samples that
    the fit did not see (even samples to fit, odd to score: a penalised fit scored on the samples
    it was fitted to always prefers no penalty).
    """
    n = len(data)
    if n < _MIN_LOTO_TRIALS:
        return _choose_ridge(template, data, valid, movable)
    even = np.zeros(data.shape[1], bool)
    even[::2] = True
    total = np.where(valid, data, 0.0).sum(axis=0)
    count = valid.sum(axis=0)
    errors = np.zeros(len(RIDGE_GRID))
    for i in range(n):
        others = count - valid[i]
        average = (total - np.where(valid[i], data[i], 0.0)) / np.maximum(others, 1)
        centre = fit_trial(
            template, average, mode="free", valid=others > 0, tol=1e-6, movable=movable
        )
        loo = dataclasses.replace(
            template, params=[np.asarray(p, float) for p in centre["params"]]
        )
        odd = valid[i] & ~even
        for j, lam in enumerate(RIDGE_GRID):
            res = fit_trial(
                loo,
                data[i],
                mode="regularized",
                ridge=lam,
                valid=valid[i] & even,
                tol=1e-6,
                movable=movable,
            )
            errors[j] += np.sum((data[i][odd] - res["fitted"][odd]) ** 2)
    return RIDGE_GRID[int(np.argmin(errors))]


@dataclasses.dataclass(frozen=True)
class _NullRule:
    """How :func:`_null_jitter` fits and trusts its noise-only segments.

    ``support`` and ``report`` are those of the real trials (the samples that count, the
    components that move), ``r2_min`` their trust threshold, and ``repeats`` the number of
    baseline segments per trial.
    """

    support: Any = None
    r2_min: float = 0.0
    report: Any = None
    repeats: Any = NULL_REPEATS
    noise: str = "segments"
    seed: Any = 0


def _null_jitter(
    template: Any, fit: Any, trials: Any, times: Any, baseline: Any, picks: Any, rule: _NullRule
) -> Any:
    """Jitter found by the same procedure on noise only.

    The channel curve plus each trial's baseline segment (no response).
    """
    support, r2_min, report, repeats = rule.support, rule.r2_min, rule.report, rule.repeats
    tw = template.tw
    base = (times >= baseline[0]) & (times <= baseline[1])
    n_win = len(tw)
    model = sum(
        s.evaluate(tw, p) * _amplitude_of(fit.components[k])
        for s, p, k in zip(template.shapes, template.params, picks, strict=False)
    )
    noise = np.nan_to_num(trials[:, base])
    if noise.shape[1] < _MIN_NOISE_SAMPLES:
        return np.full(len(picks), np.nan), 0
    segments = []
    if rule.noise == "ar":
        segments = [model + e for e in _ar_noise(noise, len(trials) * repeats, n_win, rule.seed)]
    for x in noise if rule.noise != "ar" else ():
        noise_all = x
        # Several segments of each trial's baseline (circular offsets): the null SD is then
        # estimated from more values than the trial SD it is compared with
        for offset in np.linspace(0, len(noise_all), repeats, endpoint=False).astype(int):
            idx = (offset + np.arange(n_win)) % len(noise_all)
            segments.append(model + noise_all[idx] - noise_all.mean())
    # only the reported components move: the others stay at their channel values
    results = fit_trials_batch(
        template, np.array(segments), tol=1e-6, support=support, movable=report
    )
    shifts = []
    for res in results:
        bound = res["at_bound_k"][report].any() if report is not None else res["at_bound"]
        # the same trust rule as the real trials, so that both SDs describe the same set
        if res["converged"] and not bound and np.nan_to_num(res["r2"], nan=-1) >= r2_min:
            shifts.append(res["shift"])
    if len(shifts) < _MIN_SD_VALUES:
        return np.full(len(picks), np.nan), 0
    # Repeats from one trial's baseline are not independent: count trials, not repeats
    return np.nanstd(np.array(shifts), axis=0, ddof=1), int(np.ceil(len(shifts) / repeats))


def _ar_noise(noise: Any, n_seg: int, n_win: int, seed: Any) -> FloatArray:
    """Return ``n_seg`` noise segments of ``n_win`` samples from a model of the baseline.

    One autoregressive model for all trials, fitted by least squares.
    A baseline shorter than the trial window, repeated end to end, has no power below
    1 / its duration; the latency of a slow component (P3) lives there, so the circular
    segments of ``null_rule="segments"`` give too small a null jitter. The model keeps the
    level and the spectrum of the baseline noise at every frequency.
    """
    x = noise - noise.mean(axis=1, keepdims=True)
    order = int(min(NULL_AR_ORDER, max(1, x.shape[1] // 8)))
    lagged = np.concatenate(
        [np.stack([xi[order - j - 1 : len(xi) - j - 1] for j in range(order)], -1) for xi in x]
    )
    target = np.concatenate([xi[order:] for xi in x])
    coef = np.linalg.lstsq(lagged, target, rcond=None)[0]
    roots = np.roots(np.concatenate([[1.0], -coef]))
    if roots.size and np.max(np.abs(roots)) >= 1.0:  # not stationary: white noise of that SD
        coef = np.zeros(order)
    innov = target - lagged @ coef
    rng = np.random.default_rng(seed if isinstance(seed, (int, np.integer)) else 0)
    burn = 10 * order + 50
    e = rng.normal(0.0, float(np.std(innov)), (n_seg, n_win + burn))
    from scipy.signal import lfilter

    return np.asarray(lfilter([1.0], np.concatenate([[1.0], -coef]), e, axis=1)[:, burn:])


def _prestim(
    trials: Any, times: Any, band: Any, window: Any
) -> tuple[FloatArray, NDArray[Any], FloatArray]:
    """Pre-stimulus phase (three estimators), its circular SD, and power, per trial."""
    from scipy.signal import butter, hilbert, lfilter, sosfiltfilt

    inside = (times >= window[0]) & (times <= window[1])
    sfreq = 1.0 / np.median(np.diff(times))
    if inside.sum() / sfreq < 3.0 / band[0]:
        raise SettingsError(
            "prestim: the pre-stimulus window is shorter than three cycles of "
            "the band's lowest frequency."
        )
    seg = np.nan_to_num(trials[:, inside])
    b, a = butter(2, band, btype="bandpass", fs=sfreq)
    causal = lfilter(b, a, seg, axis=-1)
    phase1 = np.angle(hilbert(causal, axis=-1))[:, -1]
    mirrored = np.concatenate([seg, seg[:, ::-1]], axis=-1)
    sos = butter(4, band, btype="bandpass", fs=sfreq, output="sos")
    phase2 = np.angle(hilbert(sosfiltfilt(sos, mirrored, axis=-1), axis=-1))[:, seg.shape[1] - 1]
    # Autoregressive forward prediction of the filtered signal, one cycle past the end
    # (order 8, least squares; all trials at once)
    order = 8
    n_seg = causal.shape[1]
    lagged = np.stack([causal[:, order - j - 1 : n_seg - j - 1] for j in range(order)], axis=-1)
    q, r = np.linalg.qr(lagged)
    rhs = np.matmul(np.swapaxes(q, 1, 2), causal[:, order:, None])
    try:
        coef = np.linalg.solve(r, rhs)[..., 0]
    except np.linalg.LinAlgError:
        coef = np.array(
            [
                np.linalg.lstsq(lagged[i], causal[i, order:], rcond=None)[0]
                for i in range(len(causal))
            ]
        )
    n_ahead = int(sfreq / band[0])
    ext = np.concatenate([causal, np.zeros((len(causal), n_ahead))], axis=1)
    for step in range(n_ahead):
        pos = n_seg + step
        ext[:, pos] = np.sum(coef * ext[:, pos - 1 : pos - order - 1 : -1], axis=1)
    phase3 = np.angle(hilbert(ext, axis=-1))[:, n_seg - 1]
    phases = np.column_stack([phase1, phase2, phase3])
    resultant = np.abs(np.mean(np.exp(1j * phases), axis=1))
    circ_sd = np.sqrt(-2 * np.log(np.clip(resultant, 1e-12, 1)))
    phase: Any = np.angle(np.mean(np.exp(1j * phases), axis=1))  # per trial
    power = np.mean(np.abs(hilbert(sosfiltfilt(sos, seg, axis=-1), axis=-1)) ** 2, axis=-1)
    return phase, circ_sd, power


###################################################################################################
## Main entry point


@verbose
def fit_trials(  # noqa: PLR0913  (spec §5.7 fixes 16 parameters, plus preset, trial_model, null_rule, ridge_cv and on_error; DECISIONS.md §7.2)
    ts: Any,
    group_fit: Any = None,
    *,
    mode: str = "shift_scale",
    components: Any = None,
    max_shift: Any = "auto",
    stretch: bool = False,
    r2_min: float = 0.2,
    realign: bool = False,
    realign_on: Any = None,
    covariates: Any = (),
    prestim_band: Any = (8.0, 13.0),
    prestim_window: Any = (-0.5, -0.01),
    by: Any = ("stim_site",),
    preset: Any = None,
    trial_model: str = "anchored",
    null_rule: str = "ar",
    ridge_cv: str = "loto",
    n_jobs: int = 1,
    random_state: Any = None,
    on_error: str = "record",
    verbose: Any = None,
) -> "TrialFitSet":
    """Latency, amplitude (and in the free modes, width) of each component in every trial.

    The channel fit (``group_fit``, or ``evoparam.models.fit`` of the trial average with
    ``preset``, by default the one of the TrialSet's modality) gives the components; each
    trial is then fitted with them, shifted (``max_shift``, by default a quarter of the
    component's width for intracranial presets and half for scalp ones, never more than half
    the distance to a neighbour) and scaled:

    - ``"shift_scale"``: shift and amplitude only (Woody filtering with a parametric template);
    - ``"free"``: every parameter within +/-25 % of the channel value;
    - ``"template"``: the component's own segment of the channel average as template;
    - ``"regularized"``: ``"free"`` with a ridge penalty towards the channel values, the
      penalty chosen by prediction error (``ridge_cv``): ``"loto"`` (default),
      leave-one-trial-out (spec §5.7: the centre of the penalty is the channel model
      refitted without the trial, and the trial is scored on samples its fit did not see),
      or ``"split"``, split samples (even samples to fit, odd to score, towards the channel
      model of all trials).

    ``trial_model``: ``"anchored"`` (default) keeps every component of the channel in the
    trial model but only those of ``components`` move (the others keep their channel latency
    and shape; their amplitude is fitted), so ``max_shift`` is clipped against moving
    neighbours only; ``"all"`` lets every component move, each with its own shift;
    ``"selected"`` (spec) puts only the components of ``components`` in the model. Only the
    components of ``components`` are reported. ``null_rule``: ``"ar"`` (default) estimates
    ``jitter_sd_null`` on five noise segments per trial drawn from an autoregressive model of
    the baseline (its level and spectrum, over the whole trial window) and refuses realignment
    unless ``jitter_sd`` exceeds both 1.2 x ``jitter_sd_null`` and the 95 % quantile of the
    ratio of two noise SDs (F distribution); ``"segments"`` takes the five segments from the
    baseline itself, repeated end to end (too small a null for components slower than the
    baseline: DECISIONS.md); ``"spec"`` uses one baseline segment per trial and the 1.2 x rule
    alone (spec §5.7). :func:`jitter` reports the test (``jitter_detected``, the smallest
    detectable jitter and ``jitter_status``), and the trial table adds ``latency_se`` and the
    empirical-Bayes ``latency_shrunk``. The measurements behind the defaults are in the CHANGELOG.

    A trial is ``converged`` when the optimiser ends normally with finite numbers, and
    ``trusted`` when also its r² >= ``r2_min`` and its shift is not at a bound; untrusted
    trials keep their numbers. ``jitter_sd`` (SD of the shifts over trusted trials) is
    reported with ``jitter_sd_null``, the same procedure on the baseline noise added to the
    channel curve; realignment is refused (warning) under the rule of ``null_rule``. In
    ``shift_scale`` mode the trials of a channel are solved together (one vectorised
    Levenberg-Marquardt iteration for the same model and bounds).

    Covariates: ``"trend"`` (amplitude against the position in the train), ``"dose"``
    (against ``stim_current_mA``, see :func:`dose_response`), ``"prestim"`` (pre-stimulus
    phase with its uncertainty across three estimators, and power, in ``prestim_band``),
    ``"state"`` (amplitude and latency per level of ``metadata["state"]``, see
    :func:`state_dependence`; a default covariate of the ``rodent_ep`` and ``primate_ep``
    presets, spec §6.2).

    ``n_jobs`` fits the channels (and conditions) in parallel: ``joblib`` processes when the
    extra ``parallel`` is installed, else threads. The fit of a channel is deterministic,
    so the result does not depend on ``n_jobs``; the warnings and log records of each
    channel reach the caller in the order of a serial run. ``verbose`` sets the level of the
    ``evoparam`` logger for the call (one INFO record at the end; ``"progress"`` adds a
    progress bar over the channels); ``verbose=False`` also silences the warnings of the
    template step, as before.

    A channel whose fit fails (REQ-307 ii) is kept with ``status="failed"`` and the exception
    in ``message``, NaN numbers and no trusted trial, and leaves no row in ``table``; one
    WARNING record names the channel (``on_error="record"``, the default).
    ``on_error="raise"`` raises the first :class:`~evoparam.errors.FitError` instead. Errors
    of the data or of the settings always propagate.

    Returns
    -------
    TrialFitSet
    """
    start = _time.perf_counter()
    _check_fit_trials(ts, mode, on_error, trial_model, null_rule, ridge_cv, covariates)
    if group_fit is None:
        group_fit = _default_group_fit(ts, preset, by)
    settings0 = group_fit.settings or {}
    preset0 = settings0.get("preset")
    preset0 = preset0.get("name") if isinstance(preset0, dict) else preset0
    options = _Options(
        mode=mode,
        trial_model=trial_model,
        max_shift=max_shift,
        factor=0.5 if preset0 in SCALP_PRESETS else 0.25,
        stretch=stretch,
        r2_min=r2_min,
        realign=realign,
        realign_on=realign_on,
        covariates=covariates,
        prestim_band=prestim_band,
        prestim_window=prestim_window,
        null_rule=null_rule,
        ridge_cv=ridge_cv,
        baseline=tuple(settings0.get("baseline") or (ts.times[0], ts.times[0] + 0.1)),
        verbose=verbose,
        random_state=random_state,
    )
    units, skipped = _channel_units(ts, group_fit, components, by, options)
    fits, rows = [], []
    channel = functools.partial(_fit_channel, on_error=on_error)
    for frame, trial_fit in parallel_map(channel, units, n_jobs, desc="fit_trials"):
        if trial_fit.status == "ok":  # a failed channel has no trial rows
            rows.append(frame)
        fits.append(trial_fit)
    table = concat_tables(rows) if rows else new_table({})
    settings = _set_settings(table, options, by)
    logger.info(
        "fit_trials (%s): %d channel fit(s), %d trial row(s), %d channel(s) skipped, %.2f s",
        mode,
        len(fits),
        len(table),
        len(set(skipped)),
        _time.perf_counter() - start,
    )
    # REQ-294: the record continues the group fit's when the group fit was made from these
    # trials (its steps start with theirs: read, cleaning, ..., fit, existence test), else
    # the trials'; one fit_trials step, with every keyword argument that changes the result
    step_settings = {k: v for k, v in settings.items() if k != "covariate_tables"}
    step_settings.update(
        {
            "components": None if components is None else list(components),
            "stretch": bool(stretch),
            "realign": bool(realign),
            "realign_on": realign_on,
            "prestim_band": [float(f) for f in prestim_band],
            "prestim_window": [float(t) for t in prestim_window],
            "n_jobs": n_jobs,
        }
    )
    base = (
        group_fit.provenance
        if _continues(group_fit.provenance.steps, ts.provenance.steps)
        else ts.provenance
    )
    provenance = base.with_step(
        "evoparam.trials.fit_trials",
        step_settings,
        duration_s=_time.perf_counter() - start,
        random_state=_recorded_seed(random_state),
    )
    for trial_fit in fits:
        trial_fit.provenance = provenance
    return TrialFitSet(
        fits=fits,
        table=table,
        skipped_channels=sorted(set(skipped)),
        settings=settings,
        provenance=provenance,
    )


class _Options(NamedTuple):
    """The settings of :func:`fit_trials` that every channel's fit reads."""

    mode: str
    trial_model: str
    max_shift: Any
    factor: float
    stretch: bool
    r2_min: float
    realign: bool
    realign_on: Any
    covariates: Any
    prestim_band: Any
    prestim_window: Any
    null_rule: str
    ridge_cv: str
    baseline: Any
    verbose: Any
    random_state: Any = None


class _Unit(NamedTuple):
    """One channel and condition: the unit of work of ``n_jobs`` in :func:`fit_trials`."""

    fit: Any
    picks: list[int]
    name: Any
    label: Any
    trial_idx: Any
    trials: Any
    trial_masks: Any
    meta_rows: Any
    times: Any
    sfreq: Any
    context: Any
    options: _Options


def _check_fit_trials(
    ts: Any,
    mode: Any,
    on_error: Any,
    trial_model: Any,
    null_rule: Any,
    ridge_cv: Any,
    covariates: Any,
) -> None:
    """Raise when the data or a setting of :func:`fit_trials` is not valid."""
    if not isinstance(ts, TrialSet):
        raise DataError("fit_trials expects a TrialSet.")
    if mode not in MODES:
        raise SettingsError(f"mode must be one of {MODES}, not {mode!r}.")
    check_on_error(on_error)
    for name, value, allowed in (
        ("trial_model", trial_model, TRIAL_MODELS),
        ("null_rule", null_rule, NULL_RULES),
        ("ridge_cv", ridge_cv, RIDGE_CV),
    ):
        if value not in allowed:
            raise SettingsError(f"{name} must be one of {allowed}, not {value!r}.")
    unknown = set(covariates) - set(COVARIATES)
    if unknown:
        raise SettingsError(f"Unknown covariates {sorted(unknown)}.")


def _default_group_fit(ts: Any, preset: Any, by: Any) -> Any:
    """Fit the trial average with ``preset``, by default the one of the modality."""
    from evoparam.models import fit as fit_models

    preset = preset or (MODALITY_PRESETS.get(ts.modality) if ts.modality else None)
    if preset is None:
        raise SettingsError("Give group_fit, or preset= (the TrialSet has no modality).")
    return fit_models(
        ts,
        preset,
        by=[c for c in by if ts.metadata is not None and c in ts.metadata.columns] or None,
    )


def _channel_fit(group_fit: Any, name: Any, label: Any) -> Any:
    """Return the channel fit of ``name`` in condition ``label``, else of all trials, or None."""
    try:
        return group_fit.get(name, condition=label if label else None)
    except KeyError:
        try:
            return group_fit.get(name)
        except KeyError:
            return None


def _unit_picks(fit: Any, components: Any) -> list[int]:
    """Return the components of ``fit`` to report (none when the fit has none to give)."""
    if fit.status != "ok" or not fit.components:
        return []
    if components is None:
        return list(range(len(fit.components)))
    return [k for k, c in enumerate(fit.components) if k in components or c.label in components]


def _channel_units(
    ts: Any, group_fit: Any, components: Any, by: Any, options: _Options
) -> tuple[list[_Unit], list[Any]]:
    """Return the channels' units of work, in the order of the results, and the skipped ones.

    The order is condition, then channel; a channel without a usable fit, or without one of
    ``components``, is skipped.
    """
    meta = ts.metadata
    columns = [c for c in (by or ()) if meta is not None and c in meta.columns]
    labels = row_labels(meta, columns) if columns else np.array([""] * ts.n_trials)
    units: list[_Unit] = []
    skipped = []
    context = SimpleNamespace(sfreq=ts.sfreq, units=ts.units, provenance=ts.provenance)
    for label in dict.fromkeys(labels):
        trial_idx = np.flatnonzero(labels == label)
        sub = ts.data[trial_idx]
        sub_masks = None if ts.masks is None else ts.masks[trial_idx]
        for ch, name in enumerate(ts.ch_names):
            fit = _channel_fit(group_fit, name, label)
            if fit is None:
                continue
            picks = _unit_picks(fit, components)
            if not picks:
                skipped.append(name)
                continue
            units.append(
                _Unit(
                    fit,
                    picks,
                    name,
                    label,
                    trial_idx,
                    sub[:, ch],
                    None if sub_masks is None else sub_masks[:, ch],
                    None if meta is None else meta.iloc[trial_idx],
                    ts.times,
                    ts.sfreq,
                    context,
                    options,
                )
            )
    return units, skipped


def _set_settings(table: Any, options: _Options, by: Any) -> dict[str, Any]:
    """Return the settings of the :class:`TrialFitSet`, with its covariate tables."""
    covariates = options.covariates
    covariate_tables = {}
    if "trend" in covariates and len(table):
        covariate_tables["trend"] = _trend(table)
    if "dose" in covariates and len(table):
        covariate_tables["dose"] = dose_response(table)
    if "state" in covariates and len(table):
        covariate_tables["state"] = state_dependence(table)
    settings = {
        "mode": options.mode,
        "max_shift": options.max_shift,
        "r2_min": options.r2_min,
        "by": list(by or ()),
        "trial_model": options.trial_model,
        "null_rule": options.null_rule,
        "ridge_cv": options.ridge_cv,
        "covariates": list(covariates),
        "package_version": __version__,
        "covariate_tables": {k: v.to_dict("list") for k, v in covariate_tables.items()},
    }
    if "prestim" in covariates:
        # the band and window of the pre-stimulus phases (read by prestimulus_dependence)
        settings["prestim_band"] = [float(f) for f in options.prestim_band]
        settings["prestim_window"] = [float(t) for t in options.prestim_window]
    return settings


def _recorded_seed(random_state: Any) -> Any:
    """Return the seed of the call as recorded with the step (REQ-294)."""
    return (
        random_state
        if random_state is None or isinstance(random_state, (int, np.integer))
        else type(random_state).__name__
    )


def _continues(steps: Any, prefix: Any) -> bool:
    """Return True when the record ``steps`` starts with the steps ``prefix``.

    The same functions with the same settings, in order.
    """
    import json

    def key(step: Any) -> str:
        """Return the function and settings of ``step`` as one comparable string."""
        return json.dumps(
            [step.get("function"), step.get("settings")], sort_keys=True, default=str
        )

    return len(steps) >= len(prefix) and [key(s) for s in steps[: len(prefix)]] == [
        key(s) for s in prefix
    ]


def _fit_channel(unit: _Unit, on_error: str = "raise") -> tuple[Any, "TrialFit"]:
    """Fit the trials of one channel and condition (one unit of ``n_jobs`` in fit_trials).

    Returns the trial table of the channel and its :class:`TrialFit`. Deterministic: the
    result does not depend on the worker that runs it. A failure of the channel is
    recorded or raised as ``on_error`` says (REQ-307 ii); errors of the data or of the
    settings propagate.
    """
    try:
        return _fit_unit(unit)
    except PROPAGATE:
        raise
    except Exception as exc:  # the failure of this channel only
        name, label = unit.name, unit.label
        where = f"{name} ({label})" if label else str(name)
        if on_error == "raise":
            raise fit_error("fit_trials", where, exc) from exc
        return _failed_channel(unit, record_failure(logger, "fit_trials", where, exc))


def _failed_channel(unit: _Unit, message: str) -> tuple[Any, "TrialFit"]:
    """Return the :class:`TrialFit` of a failed channel: NaN numbers, no trusted trial."""
    fit, picks, name, label, trial_idx = unit[:5]
    options = unit.options
    n_trials, n_picks = len(trial_idx), len(picks)
    n_params = sum(len(fit.components[k].params) for k in picks)
    nothing = dict.fromkeys(picks, np.nan)
    return new_table({}), TrialFit(
        channel=name,
        template=fit,
        shift=np.full((n_trials, n_picks), np.nan),
        scale=np.full((n_trials, n_picks), np.nan),
        params=np.full((n_trials, n_params), np.nan),
        latency=np.full((n_trials, n_picks), np.nan),
        amplitude=np.full((n_trials, n_picks), np.nan),
        fwhm=np.full((n_trials, n_picks), np.nan),
        gof=np.full(n_trials, np.nan),
        converged=np.zeros(n_trials, bool),
        jitter_sd=dict(nothing),
        jitter_sd_null=dict(nothing),
        amplitude_jitter_corrected=dict(nothing),
        trusted=np.zeros(n_trials, bool),
        table=new_table({}),
        settings={
            "mode": options.mode,
            "condition": label,
            "picks": picks,
            "r2_min": options.r2_min,
            "package_version": __version__,
        },
        status="failed",
        message=message,
    )


def _fit_unit(unit: _Unit) -> tuple[Any, "TrialFit"]:
    """Fit one channel and condition (:func:`_fit_channel` without the guard)."""
    fit, picks, name, trials, times, o = (
        unit.fit,
        unit.picks,
        unit.name,
        unit.trials,
        unit.times,
        unit.options,
    )
    # trial_model "anchored" and "all": the trial model keeps every component of the
    # channel (a complete model) and only the chosen ones are reported; "selected"
    # (spec): the model has the chosen components only
    every = list(range(len(fit.components))) if o.trial_model != "selected" else list(picks)
    report = [every.index(k) for k in picks]
    # "anchored": the other components stay at their channel latency (scale only),
    # so max_shift is clipped against the moving components only
    movable = report if o.trial_model == "anchored" else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore" if o.verbose is False else "default")
        template = _template_for(fit, every, times, o.max_shift, o.factor, moving=movable)
    support = _report_support(template.tw, fit, picks)
    win = (times >= template.tw[0]) & (times <= template.tw[-1])
    x = np.nan_to_num(trials[:, win])
    valid = np.ones(x.shape, bool) if unit.trial_masks is None else ~unit.trial_masks[:, win]
    channel_template = template
    if o.mode == "template":
        average = np.nanmean(trials, axis=0)
        template = _template_mode(template, fit, average, times, every)
    fit_mode = "free" if (o.stretch and o.mode == "shift_scale") or o.mode == "free" else o.mode
    ridge = 0.0
    if o.mode == "regularized":
        choose = _choose_ridge if o.ridge_cv == "split" else _choose_ridge_loto
        ridge = choose(template, x, valid, movable)
    results = _fit_all(
        template,
        x,
        valid,
        "shift_scale" if o.mode == "template" else fit_mode,
        ridge,
        support,
        movable=movable,
    )
    m = _measures(results, fit, picks, report, o.r2_min)
    if fit_mode in ("free", "regularized") and o.mode != "template":
        _free_fwhm(m.fwhm, results, report, channel_template, unit.sfreq)

    jit = {k: _shift_sd(m.shift[:, j], m.trusted) for j, k in enumerate(picks)}
    rule = _NullRule(
        support,
        o.r2_min,
        report,
        {"spec": 1, "ar": NULL_AR_REPEATS}.get(o.null_rule, NULL_REPEATS),
        noise="ar" if o.null_rule == "ar" else "segments",
        seed=o.random_state,
    )
    null, n_null = _null_jitter(channel_template, fit, trials, times, o.baseline, every, rule)
    null = null[report]
    jit_null = {k: float(null[j]) for j, k in enumerate(picks)}
    corrected = _jitter_corrected(fit, picks, jit)
    limit = _refusal_limit(o.null_rule, int(m.trusted.sum()), n_null)
    _shrunk_latency(m, picks, jit, jit_null, limit)

    realigned = None
    if o.realign:
        target = _realign_target(fit, picks, o.realign_on)
        j = picks.index(target)
        if not jit[target] > limit * jit_null[target]:
            _warn(
                f"{name}: jitter {jit[target]:.4f} s is not above "
                f"{limit:.2f} x the noise-only jitter "
                f"{jit_null[target]:.4f} s; realignment refused.",
                UserWarning,
                stacklevel=4,  # _fit_unit, _fit_channel, fit_trials, then its caller
            )
        else:
            realigned = _realigned(times, trials, m.shift[:, j], m.trusted, name, unit.context)

    frame = _trial_table(unit, m)
    if "prestim" in o.covariates:
        phase, circ_sd, power = _prestim(trials, times, o.prestim_band, o.prestim_window)
        frame["prestim_phase"] = np.repeat(phase, len(picks))
        frame["prestim_phase_sd"] = np.repeat(circ_sd, len(picks))
        frame["prestim_power"] = np.repeat(power, len(picks))
    return frame, (
        TrialFit(
            channel=name,
            template=fit,
            shift=m.shift,
            scale=m.scale,
            params=m.params,
            latency=m.latency,
            amplitude=m.amplitude,
            fwhm=m.fwhm,
            gof=m.r2,
            converged=m.converged,
            jitter_sd=jit,
            jitter_sd_null=jit_null,
            amplitude_jitter_corrected=corrected,
            trusted=m.trusted,
            table=frame,
            realigned=realigned,
            settings={
                "mode": o.mode,
                "condition": unit.label,
                "picks": picks,
                "ridge": ridge,
                "max_shift": channel_template.max_shift.tolist(),
                "r2_min": o.r2_min,
                "jitter_limit": limit,
                "package_version": __version__,
            },
        )
    )


def _report_support(tw: Any, fit: Any, picks: Any) -> Any:
    """Return the samples of ``tw`` within one width of a reported component."""
    support = np.zeros(len(tw), bool)
    for k in picks:
        comp = fit.components[k]
        support |= (tw >= comp.onset - comp.fwhm) & (tw <= comp.offset + comp.fwhm)
    return support


@dataclasses.dataclass
class _Measures:
    """Per-trial numbers of the reported components of one channel (rows are trials)."""

    shift: Any
    scale: Any
    r2: Any
    converged: Any
    trusted: Any
    amplitude: Any
    latency: Any
    fwhm: Any
    params: Any
    latency_shrunk: Any = None
    latency_se: Any = None


def _measures(results: Any, fit: Any, picks: Any, report: Any, r2_min: float) -> _Measures:
    """Return the trial numbers of the reported components from the solver's ``results``."""
    shift = np.array([r["shift"] for r in results])[:, report]
    scale = np.array([r["scale"] for r in results])[:, report]
    r2 = np.array([r["r2"] for r in results])
    converged = np.array([r["converged"] for r in results])
    # Trust looks at the reported components only: a neighbour held by a tight bound
    # does not make the trial's N1 untrustworthy
    at_bound = np.array([r["at_bound_k"][report].any() for r in results])
    trusted = converged & (np.nan_to_num(r2, nan=-1) >= r2_min) & ~at_bound
    peaks = np.array(
        [
            fit.components[k].amplitude / _amplitude_of(fit.components[k])
            if _amplitude_of(fit.components[k])
            else 1.0
            for k in picks
        ]
    )
    return _Measures(
        shift=shift,
        scale=scale,
        r2=r2,
        converged=converged,
        trusted=trusted,
        amplitude=scale * peaks[None],
        latency=shift + np.array([fit.components[k].latency for k in picks])[None],
        fwhm=np.tile([fit.components[k].fwhm for k in picks], (len(results), 1)),
        params=np.array([np.concatenate([r["params"][k] for k in report]) for r in results]),
    )


def _free_fwhm(fwhm: Any, results: Any, report: Any, template: Any, sfreq: Any) -> None:
    """Replace ``fwhm`` (in place) by the widths of each trial's own fitted shapes."""
    for i, r in enumerate(results):
        for j, k in enumerate(report):
            f = compute_features(template.shapes[k], template.tw, r["params"][k], 1.0, sfreq=sfreq)
            fwhm[i, j] = f.fwhm


def _shrunk_latency(m: _Measures, picks: Any, jit: Any, jit_null: Any, limit: float) -> None:
    """Set the empirical-Bayes latencies of ``m`` (in place) and their standard errors.

    Each trial's shift is drawn towards the mean of the trusted shifts by the weight
    ``tau² / (tau² + null²)``, where ``null`` is the noise-only jitter (the spread that
    noise alone gives to one trial's shift) and ``tau² = jitter_sd² - null²`` the true
    jitter variance; ``tau`` is 0 when the jitter is not above ``limit`` x the noise-only
    jitter (no measurable jitter: every trial gets the mean latency). The standard error of
    the raw latency of a trial is ``null``.
    """
    m.latency_shrunk = np.full(m.latency.shape, np.nan)
    m.latency_se = np.full(m.latency.shape, np.nan)
    for j, k in enumerate(picks):
        raw, null = jit[k], jit_null[k]
        if not (np.isfinite(raw) and np.isfinite(null) and m.trusted.any()):
            continue
        tau2 = raw**2 - null**2 if raw > limit * null else 0.0
        w = tau2 / (tau2 + null**2) if tau2 + null**2 > 0 else 0.0
        mean = np.mean(m.latency[m.trusted, j])
        m.latency_shrunk[:, j] = mean + w * (m.latency[:, j] - mean)
        m.latency_se[:, j] = null


def _shift_sd(shifts: Any, trusted: Any) -> float:
    """Return the SD of the shifts of the trusted trials (NaN with too few of them)."""
    return float(np.std(shifts[trusted], ddof=1)) if trusted.sum() >= _MIN_SD_VALUES else np.nan


def _jitter_corrected(fit: Any, picks: Any, jit: Any) -> dict[Any, float]:
    """Return each component's amplitude corrected for the smearing of its jitter."""
    corrected = {}
    for k in picks:
        comp = fit.components[k]
        sigma = comp.fwhm / FWHM_PER_SIGMA
        jj = jit[k] if np.isfinite(jit[k]) else 0.0
        corrected[k] = float(comp.amplitude * np.sqrt(sigma**2 + jj**2) / sigma)
    return corrected


def _realign_target(fit: Any, picks: Any, realign_on: Any) -> Any:
    """Return the component to realign on: ``realign_on`` (index or label), else the first."""
    if realign_on is None:
        return picks[0]
    if isinstance(realign_on, int):
        return realign_on
    return next(k for k in picks if fit.components[k].label == realign_on)


def _refusal_limit(null_rule: str, n_t: int, n_null: int) -> float:
    """Return the ratio to the noise-only jitter that realignment requires.

    Refuse unless the jitter clearly exceeds what noise alone gives: above 1.2 x the
    noise-only SD, and above the 95 % quantile of the ratio of two SDs of noise (F
    distribution), so that jitter-free data are refused in 95 % of runs whatever the
    number of trials.
    """
    from scipy.stats import f as f_dist

    limit = JITTER_REFUSE
    if null_rule in ("ar", "segments") and n_t > _F_MIN_TRIALS and n_null > _F_MIN_TRIALS:
        limit = max(limit, float(np.sqrt(f_dist.ppf(0.95, n_t - 1, n_null - 1))))
    return limit


def _trial_table(unit: _Unit, m: _Measures) -> Any:
    """Return the trial table of one channel: one row per trial and reported component."""
    fit, meta_rows = unit.fit, unit.meta_rows
    table = []
    for i, trial in enumerate(unit.trial_idx):
        for j, k in enumerate(unit.picks):
            row = {
                "trial": int(trial),
                "channel": unit.name,
                "condition": unit.label,
                "component": int(k),
                "label": fit.components[k].label or "",
                "shift": m.shift[i, j],
                "scale": m.scale[i, j],
                "latency": m.latency[i, j],
                "latency_se": np.nan if m.latency_se is None else m.latency_se[i, j],
                "latency_shrunk": np.nan if m.latency_shrunk is None else m.latency_shrunk[i, j],
                "amplitude": m.amplitude[i, j],
                "fwhm": m.fwhm[i, j],
                "r2": m.r2[i],
                "converged": bool(m.converged[i]),
                "trusted": bool(m.trusted[i]),
            }
            if meta_rows is not None:
                for col in meta_rows.columns:
                    row.setdefault(col, meta_rows[col].iloc[i])
            table.append(row)
    return new_table(table)


def _realigned(times: Any, trials: Any, shifts: Any, trusted: Any, name: Any, ts: Any) -> "Evoked":
    """Return the average of the trusted trials, each shifted back by its own shift."""
    data: Any
    data = []
    for x, d in zip(trials[trusted], shifts[trusted], strict=False):
        data.append(np.interp(times, times - d, np.nan_to_num(x)))
    data = np.array(data)
    return Evoked(
        data=data.mean(axis=0)[None],
        times=times,
        sfreq=ts.sfreq,
        ch_names=[name],
        units=ts.units,
        n_trials=int(trusted.sum()),
        provenance=ts.provenance.with_step("evoparam.trials.realign", {"channel": name}),
    )


###################################################################################################
## Companions


def jitter(trial_fit: Any, component: int = 0) -> Any:
    """Return the jitter statistics of one component, per channel.

    ``jitter_sd`` (with a bootstrap interval over trusted trials), ``jitter_sd_null``,
    ``jitter_sd_excess = sqrt(jitter_sd² - jitter_sd_null²)`` (the jitter left once the part that
    noise alone produces is removed; the best estimate of the true jitter at low SNR) and
    ``amplitude_jitter_corrected`` of one component (index or label).

    The test of the jitter (the idea of Pham et al. 1987, Biometrika 74:525): ``jitter_detected``
    is True when ``jitter_sd`` exceeds the refusal limit of :func:`fit_trials` (``limit`` x
    ``jitter_sd_null``); ``jitter_min_detectable = jitter_sd_null sqrt(limit² - 1)`` is the
    smallest true jitter that passes the test with these trials and this noise;
    ``jitter_sd_eb`` is ``jitter_sd_excess`` when the jitter is detected and 0 otherwise, and
    ``jitter_status`` says ``"measured"`` or ``"not measurable (< x ms)"``.
    """
    pd = require("pandas", extra="data", caller="evoparam.trials.jitter")
    rng = np.random.default_rng(0)
    rows = []
    for fit in trial_fit.fits:
        picks = fit.settings["picks"]
        if isinstance(component, str):
            matches = [k for k in picks if fit.template.components[k].label == component]
            if not matches:
                continue
            k = matches[0]
        else:
            if component not in picks:
                continue
            k = component
        j = picks.index(k)
        values = fit.shift[fit.trusted, j]
        boots = (
            [np.std(rng.choice(values, len(values)), ddof=1) for _ in range(500)]
            if len(values) >= _MIN_SD_VALUES
            else [np.nan]
        )
        raw, null = fit.jitter_sd[k], fit.jitter_sd_null[k]
        excess = (
            float(np.sqrt(max(raw**2 - null**2, 0.0)))
            if np.isfinite(raw) and np.isfinite(null)
            else np.nan
        )
        limit = fit.settings.get("jitter_limit", JITTER_REFUSE)
        detected = bool(np.isfinite(raw) and np.isfinite(null) and raw > limit * null)
        min_detectable = float(null * np.sqrt(limit**2 - 1)) if np.isfinite(null) else np.nan
        status = (
            "measured"
            if detected
            else f"not measurable (< {1e3 * min_detectable:.1f} ms)"
            if np.isfinite(min_detectable)
            else "not measurable"
        )
        rows.append(
            {
                "channel": fit.channel,
                "condition": fit.settings["condition"],
                "component": k,
                "jitter_sd": raw,
                "jitter_sd_excess": excess,
                "jitter_sd_eb": excess if detected else 0.0,
                "jitter_detected": detected,
                "jitter_min_detectable": min_detectable,
                "jitter_status": status,
                "ci_low": float(np.nanpercentile(boots, 2.5)),
                "ci_high": float(np.nanpercentile(boots, 97.5)),
                "jitter_sd_null": fit.jitter_sd_null[k],
                "amplitude_jitter_corrected": fit.amplitude_jitter_corrected[k],
                "n_trusted": int(fit.trusted.sum()),
            }
        )
    return pd.DataFrame(rows)


def realign(ts: Any, trial_fit: Any, component: int = 0) -> list[Any]:
    """Average the trusted trials, each shifted back by its latency shift of ``component``.

    The Woody-filter average; one Evoked per channel fit.
    """
    out = []
    for fit in trial_fit.fits:
        picks = fit.settings["picks"]
        k = (
            component
            if isinstance(component, int)
            else next((p for p in picks if fit.template.components[p].label == component), None)
        )
        if k not in picks:
            continue
        ch = list(ts.ch_names).index(fit.channel)
        out.append(
            _realigned(
                ts.times,
                ts.data[:, ch],
                fit.shift[:, picks.index(k)],
                fit.trusted,
                fit.channel,
                ts,
            )
        )
    return out


def _trend(table: Any) -> Any:
    """Amplitude scale against the position in the train, per channel, condition and component.

    ``a_i = a (1 + c i)`` by least squares over trusted trials. ``slope`` is ``c`` (relative change
    per pulse), ``p_value`` the two-sided t test of the slope.
    """
    from scipy.stats import t as t_dist

    pd = require("pandas", extra="data", caller="evoparam.trials")
    order_col = "index_in_train" if "index_in_train" in table.columns else "trial"
    rows = []
    for (ch, cond, k), g in table[table["trusted"]].groupby(["channel", "condition", "component"]):
        if len(g) < _MIN_TREND_TRIALS:
            continue
        x = g[order_col].to_numpy(float)
        y = g["scale"].to_numpy(float)
        x = x - x.min()
        slope, intercept = np.polyfit(x, y, 1)
        resid = y - (intercept + slope * x)
        sxx = np.sum((x - x.mean()) ** 2)
        dof = len(y) - 2
        if sxx > 0 and dof > 0:
            se = np.sqrt(np.sum(resid**2) / dof / sxx)
            p_value = float(2 * t_dist.sf(abs(slope) / se, dof)) if se > 0 else 0.0
        else:
            p_value = np.nan
        rows.append(
            {
                "channel": ch,
                "condition": cond,
                "component": k,
                "slope": slope / intercept if intercept else np.nan,
                "intercept": intercept,
                "p_value": p_value,
                "n_trials": len(y),
                "x": order_col,
            }
        )
    return pd.DataFrame(rows)


def state_dependence(trial_fit: Any, *, column: str = "state", reference: Any = None) -> Any:
    """Return the amplitude and latency per behavioural or brain state.

    The state is ``metadata["state"]`` (sleep, wake, anaesthesia...); per channel and component, on
    the trusted trials.

    State changes the gain of evoked field potentials up to about three-fold (spec §6.2,
    rodent and primate presets), so the trial parameters are summarised per state level:
    ``n_trials``, the median and mean amplitude and latency, the ``gain`` (median absolute
    amplitude over that of the ``reference`` state, by default the most frequent one) and,
    on the reference row's channel and component, the Kruskal-Wallis p-values of amplitude
    and latency across the states (``p_amplitude``, ``p_latency``; NaN with one state).

    Raises
    ------
    SettingsError
        The trial table has no ``column``.
    """
    from scipy.stats import kruskal

    pd = require("pandas", extra="data", caller="evoparam.trials.state_dependence")
    table = trial_fit.table if isinstance(trial_fit, TrialFitSet) else trial_fit
    if column not in table.columns:
        raise SettingsError(
            f"state_dependence: the trial table has no column {column!r} (metadata['{column}'])."
        )
    columns = [
        "channel",
        "component",
        "label",
        column,
        "n_trials",
        "amplitude_median",
        "amplitude_mean",
        "latency_median",
        "latency_mean",
        "gain",
        "reference",
        "p_amplitude",
        "p_latency",
    ]
    rows = []
    trusted = table[table["trusted"].astype(bool) & table[column].notna()]
    for (ch, k), g in trusted.groupby(["channel", "component"], sort=False):
        states = g[column].astype(str)
        counts = states.value_counts()
        ref = str(reference) if reference is not None else str(counts.index[0])
        groups = {state: g[states == state] for state in sorted(counts.index)}
        ref_amp = (
            np.median(np.abs(groups[ref]["amplitude"].to_numpy(float)))
            if ref in groups
            else np.nan
        )
        p_amp = p_lat = np.nan
        usable = [v for v in groups.values() if len(v) >= _MIN_STATE_TRIALS]
        if len(usable) >= _MIN_STATES:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    p_amp = float(
                        kruskal(*[v["amplitude"].to_numpy(float) for v in usable]).pvalue
                    )
                    p_lat = float(kruskal(*[v["latency"].to_numpy(float) for v in usable]).pvalue)
                except ValueError:  # all values identical
                    pass
        for state, v in groups.items():
            amp, lat = v["amplitude"].to_numpy(float), v["latency"].to_numpy(float)
            rows.append(
                {
                    "channel": ch,
                    "component": int(k),
                    "label": str(v["label"].iloc[0]) if "label" in v else "",
                    column: state,
                    "n_trials": int(len(v)),
                    "amplitude_median": float(np.median(amp)),
                    "amplitude_mean": float(np.mean(amp)),
                    "latency_median": float(np.median(lat)),
                    "latency_mean": float(np.mean(lat)),
                    "gain": float(np.median(np.abs(amp)) / ref_amp) if ref_amp else np.nan,
                    "reference": ref,
                    "p_amplitude": p_amp,
                    "p_latency": p_lat,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _hill(i: Any, a_max: Any, i50: Any, h: Any) -> Any:
    """Return the Hill function ``a_max i^h / (i50^h + i^h)`` at the doses ``i``."""
    return a_max * i**h / (i50**h + i**h)


def dose_response(trial_fit: Any, *, x: str = "stim_current_mA", model: Any = "auto") -> Any:
    """Amplitude against stimulation dose per channel and component.

    ``model="hill"``: ``a(I) = a_max I^h / (I50^h + I^h)``, ``h`` in [0.5, 6], with the
    threshold current ``I_thr = I50 (0.1/0.9)^(1/h)``; ``"linear"``: a straight line;
    ``"auto"``: Hill with four currents or more, else linear.
    """
    from scipy.optimize import curve_fit

    pd = require("pandas", extra="data", caller="evoparam.trials.dose_response")
    table = trial_fit.table if isinstance(trial_fit, TrialFitSet) else trial_fit
    if x not in table.columns:
        raise SettingsError(f"dose_response: the trial table has no column {x!r}.")
    rows = []
    for (ch, k), g in table[table["trusted"]].groupby(["channel", "component"]):
        dose = g[x].to_numpy(float)
        amp = np.abs(g["amplitude"].to_numpy(float))
        levels = np.unique(dose[np.isfinite(dose)])
        if len(levels) < _MIN_DOSES:
            raise SettingsError("dose_response needs at least two doses.")
        kind = (
            model if model != "auto" else ("hill" if len(levels) >= _HILL_MIN_DOSES else "linear")
        )
        row = {"channel": ch, "component": k, "model": kind, "n_doses": len(levels)}
        if kind == "hill":
            try:
                popt, _ = curve_fit(
                    _hill,
                    dose,
                    amp,
                    p0=[amp.max(), np.median(levels), 2.0],
                    bounds=(
                        [0, levels.min() / 10, 0.5],
                        [10 * amp.max() + 1e-12, levels.max() * 10, 6.0],
                    ),
                    maxfev=20000,
                )
                a_max, i50, h = popt
                row.update(a_max=a_max, i50=i50, h=h, i_thr=i50 * (0.1 / 0.9) ** (1 / h))
            except RuntimeError:
                row.update(a_max=np.nan, i50=np.nan, h=np.nan, i_thr=np.nan)
        else:
            slope, intercept = np.polyfit(dose, amp, 1)
            row.update(slope=slope, intercept=intercept)
        rows.append(row)
    return pd.DataFrame(rows)


def _harmonic(phase: Any, amp: Any) -> Any:
    """Return ``(c0, c1 cos phi0, c1 sin phi0)``: least squares of ``amp`` on the phase."""
    design = np.column_stack([np.ones_like(phase), np.cos(phase), np.sin(phase)])
    coef, *_ = np.linalg.lstsq(design, amp, rcond=None)
    return coef


def prestimulus_dependence(
    trial_fit: Any,
    *,
    band: Any = (8.0, 13.0),
    n_permutations: int = 1000,
    random_state: Any = None,
) -> Any:
    """Amplitude against pre-stimulus phase.

    ``a = c0 + c1 cos(phi - phi0)``, with a permutation test of ``c1`` (phases shuffled across
    trials). Needs ``covariates=("prestim",)`` in :func:`fit_trials`.

    ``band`` (Hz) is the frequency band of the phases: :func:`fit_trials` estimates them in
    its ``prestim_band`` and records it, and a ``band`` that differs from the recorded one
    raises :class:`~evoparam.errors.SettingsError` (the phases of another band cannot be
    read from the table). With a table instead of a TrialFitSet the band cannot be
    checked; it is recorded in ``attrs["band"]`` of the result either way.
    """
    pd = require("pandas", extra="data", caller="evoparam.trials.prestimulus_dependence")
    band = _check_band(band)
    table = trial_fit.table if isinstance(trial_fit, TrialFitSet) else trial_fit
    if "prestim_phase" not in table.columns:
        raise SettingsError("Run fit_trials with covariates=('prestim',) first.")
    recorded = (
        (trial_fit.settings or {}).get("prestim_band")
        if isinstance(trial_fit, TrialFitSet)
        else None
    )
    if recorded is not None and not np.allclose(np.asarray(recorded, float), band):
        raise SettingsError(
            f"prestimulus_dependence: band={tuple(band)} Hz, but fit_trials estimated the "
            f"pre-stimulus phases in prestim_band={tuple(recorded)} Hz; run fit_trials with "
            f"prestim_band={tuple(band)}, or pass band={tuple(recorded)}."
        )
    rng = np.random.default_rng(random_state)
    rows = []
    for (ch, k), g in table[table["trusted"]].groupby(["channel", "component"]):
        phi = g["prestim_phase"].to_numpy(float)
        amp = g["scale"].to_numpy(float)
        c0, cc, cs = _harmonic(phi, amp)
        c1 = np.hypot(cc, cs)
        null = np.array(
            [np.hypot(*_harmonic(rng.permutation(phi), amp)[1:]) for _ in range(n_permutations)]
        )
        rows.append(
            {
                "channel": ch,
                "component": k,
                "c0": c0,
                "c1": c1,
                "modulation": c1 / abs(c0) if c0 else np.nan,
                "phi0": float(np.arctan2(cs, cc)),
                "p_value": (1 + np.sum(null >= c1)) / (1 + n_permutations),
                "n_trials": len(g),
            }
        )
    result = pd.DataFrame(rows)
    result.attrs["band"] = [float(f) for f in band]
    return result


def _check_band(band: Any) -> FloatArray:
    """Return ``band`` as two increasing positive frequencies (Hz), or raise SettingsError."""
    message = f"band must be (low, high) in Hz with 0 < low < high, not {band!r}."
    try:
        low, high = (float(f) for f in np.asarray(band, dtype=float).reshape(-1))
    except (TypeError, ValueError) as error:
        raise SettingsError(message) from error
    if not (np.isfinite(high) and 0 < low < high):
        raise SettingsError(message)
    return np.array([low, high])
