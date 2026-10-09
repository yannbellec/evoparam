"""The preset pipeline (artifact → reference → baseline → fit → existence → trials) and the
three reference pipelines whose methods text is frozen (spec §5.13, REQ-340).

:func:`run_pipeline` only calls the public functions of the other modules, in the order of
spec §4.1 principle 1, with the values of a preset: it is the Python twin of
``evoparam fit`` (the command line calls it), and it computes nothing itself (REQ-349).
"""

import dataclasses

import numpy as np

import evoparam.presets as _presets
from evoparam.errors import SettingsError

__all__ = ["PipelineResult", "run_pipeline", "REFERENCE_PIPELINES", "reference_input",
           "run_reference_pipeline", "artifact_steps", "reference_scheme"]


@dataclasses.dataclass
class PipelineResult:
    """What :func:`run_pipeline` produced: the cleaned trials, the artifact results (one per
    artifact step), the channel fits with their existence tests, the trial fits, and the
    reference-sensitivity report (the kappa table of
    :func:`evoparam.prep.reference_sensitivity`, for a preset that runs it: ``ccep_seeg``)."""

    trials: object
    artifact: list
    group: object
    trial_fits: object = None
    test: object = None
    reference_report: object = None

    @property
    def results(self):
        """The results to write as BIDS derivatives (spec §10.2): the group fit, the
        existence test, the trial fits and the artifact results."""

        out = [self.group]
        if self.test is not None:
            out.append(self.test)
        if self.trial_fits is not None:
            out.append(self.trial_fits)
        return out + list(self.artifact)


def _artifact_baseline(ts, preset, window):
    """Baseline of the artifact-quality score: the preset's baseline when the epoch holds
    it, else the part of the epoch before the artifact window (the short epochs of the
    ECAP presets have no sample in the default -0.5 to -0.05 s)."""

    times = ts.times
    if preset.baseline is not None:
        lo, hi = preset.baseline
        if ((times >= lo) & (times <= hi)).sum() >= 3:
            return tuple(preset.baseline)
    stop = (window[0] if window is not None else 0.0) - 1.0 / ts.sfreq
    return (float(times[0]), float(stop))


def artifact_steps(preset):
    """The ``remove_artifact`` keyword sets that the preset's ``artifact_method`` stands for.

    ``"template+tail:joint"`` → one template step with ``tail="joint"``;
    ``"interpolate+subspace"`` → interpolation then subspace projection;
    ``"paradigm:<scheme>"`` → paradigm subtraction with that scheme; ``None`` → none.
    """

    p = _presets.get(preset)
    method = p.artifact_method
    if method in (None, "", "none"):
        return []
    window = tuple(p.artifact_window) if p.artifact_window is not None else None
    steps = []
    for part in method.split("+"):
        if part.startswith("tail:"):
            if not steps:
                raise SettingsError(f"preset {p.name!r}: a tail option needs a method first.")
            steps[-1]["tail"] = part.split(":", 1)[1]
            continue
        name, _, option = part.partition(":")
        kwargs = {"method": name}
        if window is not None:
            kwargs["window"] = window
        if name == "paradigm":
            kwargs["scheme"] = option
        steps.append(kwargs)
    return steps


def reference_scheme(preset):
    """The ``rereference`` scheme of the preset, or None to keep the recording reference."""

    ref = _presets.get(preset).reference
    if ref in (None, "", "as_recorded", "as recorded", "monopolar"):
        return None
    return {"average": "car"}.get(ref, ref)


def _test_window(ts, window):
    """The existence-test window: the fit window, starting after the last sample that the
    artifact removal replaced in every trial (the test refuses such samples; with the
    interpolation buffer they can reach past the artifact window)."""

    if ts.masks is None:
        return window
    inside = (ts.times >= window[0]) & (ts.times <= window[1])
    replaced = ts.masks[:, :, inside].all(axis=0).any(axis=0)
    if not replaced.any():
        return window
    last = ts.times[inside][np.flatnonzero(replaced)[-1]]
    later = ts.times[ts.times > last]
    return (float(later[0]), float(window[1])) if later.size else window


def run_pipeline(trials, preset, *, artifact=True, reference=True, baseline=True,
                 existence=True, trial_fits=None, random_state=0, existence_kwargs=None,
                 n_jobs=1, reference_sensitivity=None, continuous=None, **overrides):
    """Run the preset's chain on a TrialSet.

    Parameters
    ----------
    trials : TrialSet
        Epochs around the stimulation (or event) onsets.
    preset : str or Preset
    artifact, reference, baseline, existence : bool
        Run the preset's artifact removal, re-referencing, baseline correction and
        existence test (each only when the preset defines it).
    trial_fits : bool or None
        Fit every trial (``evoparam.trials.fit_trials`` with the preset's trial model);
        None follows the preset (``trial_fit``).
    random_state : int
        Seed of every step that draws random numbers.
    existence_kwargs : dict, optional
        Passed to the existence test (default: the preset's method and alpha).
    reference_sensitivity : bool or None
        Run :func:`evoparam.prep.reference_sensitivity` on the cleaned trials before the
        re-referencing and attach its kappa report (``PipelineResult.reference_report``);
        None follows the preset (``ccep_seeg``: yes, spec §6.2).
    continuous : array, TrialSet or tuple, optional
        The continuous recording, for a preset whose baseline is taken per train before the
        train (``erna``, :func:`evoparam.prep.train_baseline`): a TrialSet with one
        continuous trial, or ``(data, sfreq, t0)``. Without it the baseline comes from the
        epochs when they start early enough, else it is not applied (warning).
    **overrides
        Fit settings that replace the preset's (recorded in the methods text).

    Returns
    -------
    PipelineResult
    """

    from evoparam.artifact import remove_artifact
    from evoparam.errors import DataError, warn
    from evoparam.existence import attach_test, test_response
    from evoparam.models import fit as fit_models
    from evoparam.prep import baseline_correct, rereference
    from evoparam.prep import reference_sensitivity as sensitivity
    from evoparam.prep import train_baseline

    p = _presets.get(preset)
    ts = trials
    artifacts = []
    if artifact:
        for kwargs in artifact_steps(p):
            kwargs.setdefault("baseline", _artifact_baseline(ts, p, kwargs.get("window")))
            result = remove_artifact(ts, random_state=random_state, **kwargs)
            artifacts.append(result)
            ts = result.cleaned
    report = None
    if reference_sensitivity if reference_sensitivity is not None else \
            (reference and p.reference_sensitivity):
        # spec §6.2 (ccep_seeg): the detections under several references, with their kappa
        try:
            report = sensitivity(ts, reference_kwargs={"random_state": random_state})
        except (DataError, SettingsError) as exc:
            warn(f"preset {p.name!r}: reference_sensitivity could not run ({exc}); no kappa "
                 "report is attached.", UserWarning, stacklevel=2)
    scheme = reference_scheme(p)
    if reference and scheme is not None:
        extra = {"random_state": random_state} if scheme == "car_adaptive" else {}
        ts = rereference(ts, scheme, **extra)
    if baseline and p.baseline is not None:
        ts = baseline_correct(ts, tuple(p.baseline))
    elif baseline and p.train_baseline is not None:
        # spec §6.1 (erna): the last 50 ms before each train
        raw = continuous
        kwargs = {}
        if isinstance(continuous, tuple):
            raw, kwargs = continuous[0], {"sfreq": continuous[1], "t0": continuous[2]}
        try:
            ts = train_baseline(ts, p.train_baseline, raw=raw, **kwargs)
        except DataError as exc:
            warn(f"preset {p.name!r}: the baseline per train was not applied ({exc}).",
                 UserWarning, stacklevel=2)
    test_kwargs = {"method": p.existence_method, "alpha": p.existence_alpha}
    window = _test_window(ts, tuple(overrides.get("fit_window") or p.fit_window))
    if window != tuple(overrides.get("fit_window") or p.fit_window):
        test_kwargs["window"] = window
    test_kwargs.update(existence_kwargs or {})
    group = fit_models(ts, preset, random_state=random_state, n_jobs=n_jobs, **overrides)
    test = None
    if existence:
        # as evoparam.existence.fit(existence=True), keeping the test for the derivatives
        settings = group.settings
        kwargs = dict(window=tuple(settings["fit_window"]),
                      baseline=tuple(settings["baseline"]) if settings.get("baseline") else
                      (-0.5, -0.05), by=(), random_state=random_state)
        kwargs.update(test_kwargs)
        test = test_response(ts, **kwargs)
        group = attach_test(group, test)
    fits = None
    if trial_fits if trial_fits is not None else p.trial_fit:
        from evoparam.trials import fit_trials
        # the preset's default covariates (rodent_ep, primate_ep: metadata["state"])
        covariates = [c for c in p.trial_covariates
                      if ts.metadata is not None and c in ts.metadata.columns]
        for missing in sorted(set(p.trial_covariates) - set(covariates)):
            warn(f"preset {p.name!r}: metadata[{missing!r}] is missing, so the default "
                 f"covariate {missing!r} of the trial fits is not computed.", UserWarning,
                 stacklevel=2)
        fits = fit_trials(ts, group, mode=p.trial_mode or "shift_scale",
                          random_state=random_state, n_jobs=n_jobs, covariates=covariates)
    return PipelineResult(trials=ts, artifact=artifacts, group=group, trial_fits=fits,
                          test=test, reference_report=report)


###################################################################################################
## Reference pipelines (REQ-340)

# name -> preset, the published data it stands for, the stand-in used when those data are
# not at hand, and the options of run_pipeline. The methods text depends only on the
# settings, never on the data, so the stand-in gives the words of the real pipeline.
REFERENCE_PIPELINES = {
    "erpcore_p3": {
        "preset": "erp",
        "data": "ERP CORE P3 (active visual oddball), channel Pz, all epochs of one "
                "participant (Kappenman et al. 2021; OSF node etdkz, CC-BY-4.0; downloaded "
                "by the network test, not redistributed)",
        "stand_in": {"generator": "simulate_erp",
                     "kwargs": {"n_trials": 30, "n_channels": 1, "components": ["P3"],
                                "ch_names": ["Pz"], "seed": 0}},
        "options": {},
    },
    "ds004080_ccep": {
        "preset": "ccep",
        "data": "OpenNeuro ds004080, sub-ccepAgeUMCU01 run-021448, stimulation site "
                "PT01-PT02 (van Blooijs et al. 2023; CC0)",
        "stand_in": {"generator": "simulate_ccep",
                     "kwargs": {"n_trials": 10, "n_channels": 10, "sfreq": 1024.0,
                                "seed": 0}},
        "options": {"trial_fits": True},
    },
    "sim_tep": {
        "preset": "tep",
        "data": "evoparam.sim.simulate_tep, 20 trials, 3 channels (stimulated site Cz), "
                "epochs -1 to 1 s, seed 0",
        "stand_in": {"generator": "simulate_tep",
                     "kwargs": {"n_trials": 20, "n_channels": 3, "stim_site": "Cz",
                                "tmin": -1.0, "tmax": 1.0, "seed": 0}},
        "options": {},
    },
}


def reference_input(name):
    """The stand-in TrialSet of a reference pipeline (simulated; see REFERENCE_PIPELINES)."""

    from evoparam import sim

    spec = REFERENCE_PIPELINES[name]["stand_in"]
    ts, _ = getattr(sim, spec["generator"])(**spec["kwargs"])
    return ts


def run_reference_pipeline(name, trials=None, *, random_state=0):
    """Run reference pipeline ``name`` on ``trials`` (default: its stand-in input)."""

    if name not in REFERENCE_PIPELINES:
        raise SettingsError(f"Unknown reference pipeline {name!r}; known: "
                            f"{sorted(REFERENCE_PIPELINES)}.")
    spec = REFERENCE_PIPELINES[name]
    ts = reference_input(name) if trials is None else trials
    return run_pipeline(ts, spec["preset"], random_state=random_state, **spec["options"])
