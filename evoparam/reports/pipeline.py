# SPDX-License-Identifier: Apache-2.0
"""The preset pipeline and the three reference pipelines (spec §5.13, REQ-340).

The preset pipeline runs artifact → reference → baseline → fit → existence → trials; the methods
text of the three reference pipelines is frozen.

:func:`run_pipeline` only calls the public functions of the other modules, in the order of
spec §4.1 principle 1, with the values of a preset: it is the Python twin of
``evoparam fit`` (the command line calls it), and it computes nothing itself (REQ-349).
"""

import dataclasses
from typing import TYPE_CHECKING, cast

import numpy as np

import evoparam.presets as _presets
from evoparam._typing import Any
from evoparam.errors import SettingsError

if TYPE_CHECKING:
    from evoparam.data.containers import TrialSet

__all__ = [
    "PipelineResult",
    "run_pipeline",
    "REFERENCE_PIPELINES",
    "reference_input",
    "run_reference_pipeline",
    "artifact_steps",
    "reference_scheme",
]

# Fewest samples of the preset's baseline for the artifact-quality score.
_MIN_BASELINE_SAMPLES = 3


@dataclasses.dataclass
class PipelineResult:
    """What :func:`run_pipeline` produced.

    The cleaned trials, the artifact results (one per artifact step), the channel fits with their
    existence tests, the trial fits, and the reference-sensitivity report (the kappa table of
    :func:`evoparam.prep.reference_sensitivity`, for a preset that runs it: ``ccep_seeg``).
    """

    trials: Any
    artifact: list[Any]
    group: Any
    trial_fits: Any = None
    test: Any = None
    reference_report: Any = None

    @property
    def results(self) -> Any:
        """The results to write as BIDS derivatives (spec §10.2).

        The group fit, the existence test, the trial fits and the artifact results.
        """
        out = [self.group]
        if self.test is not None:
            out.append(self.test)
        if self.trial_fits is not None:
            out.append(self.trial_fits)
        return out + list(self.artifact)


def _artifact_baseline(ts: Any, preset: Any, window: Any) -> Any:
    """Baseline of the artifact-quality score.

    The preset's baseline when the epoch holds it, else the part of the epoch before the artifact
    window (the short epochs of the ECAP presets have no sample in the default -0.5 to -0.05 s).
    """
    times = ts.times
    if preset.baseline is not None:
        lo, hi = preset.baseline
        if ((times >= lo) & (times <= hi)).sum() >= _MIN_BASELINE_SAMPLES:
            return tuple(preset.baseline)
    stop = (window[0] if window is not None else 0.0) - 1.0 / ts.sfreq
    return (float(times[0]), float(stop))


def artifact_steps(preset: Any) -> list[Any]:
    """Return the ``remove_artifact`` keyword sets of the preset's ``artifact_method``.

    ``"template+tail:joint"`` → one template step with ``tail="joint"``;
    ``"interpolate+subspace"`` → interpolation then subspace projection;
    ``"paradigm:<scheme>"`` → paradigm subtraction with that scheme; ``None`` → none.
    """
    p = _presets.get(preset)
    method = p.artifact_method
    if method in (None, "", "none"):
        return []
    window = tuple(p.artifact_window) if p.artifact_window is not None else None
    steps: list[Any]
    steps = []
    for part in method.split("+"):
        if part.startswith("tail:"):
            if not steps:
                raise SettingsError(f"preset {p.name!r}: a tail option needs a method first.")
            steps[-1]["tail"] = part.split(":", 1)[1]
            continue
        name, _, option = part.partition(":")
        kwargs: dict[str, Any] = {"method": name}
        if window is not None:
            kwargs["window"] = window
        if name == "paradigm":
            kwargs["scheme"] = option
        if name == "subspace":
            # the specification's projection of every sample, after the interpolation of
            # the window: the two-stage logic of the TMS pipelines (§5.6, tep), where the
            # projection is meant for the decay beyond the window (the default of
            # remove_artifact projects the window only; not measured on TEPs)
            kwargs.update(n_components="energy", subspace_extent="all")
        steps.append(kwargs)
    return steps


def reference_scheme(preset: Any) -> Any:
    """Return the ``rereference`` scheme of the preset, or None to keep the recording reference."""
    ref = _presets.get(preset).reference
    if ref in (None, "", "as_recorded", "as recorded", "monopolar"):
        return None
    return {"average": "car"}.get(ref, ref)


def _test_window(ts: Any, window: Any) -> Any:
    """Return the existence-test window.

    The fit window, starting after the last sample that the artifact removal replaced in every
    trial (the test refuses such samples; with the interpolation buffer they can reach past the
    artifact window).
    """
    if ts.masks is None:
        return window
    inside = (ts.times >= window[0]) & (ts.times <= window[1])
    replaced = ts.masks[:, :, inside].all(axis=0).any(axis=0)
    if not replaced.any():
        return window
    last = ts.times[inside][np.flatnonzero(replaced)[-1]]
    later = ts.times[ts.times > last]
    return (float(later[0]), float(window[1])) if later.size else window


def run_pipeline(  # noqa: PLR0913  (public keywords; DECISIONS.md §7.2)
    trials: "TrialSet",
    preset: Any,
    *,
    artifact: bool = True,
    reference: bool = True,
    baseline: bool = True,
    existence: bool = True,
    trial_fits: bool | None = None,
    random_state: int = 0,
    existence_kwargs: dict[str, Any] | None = None,
    n_jobs: int = 1,
    reference_sensitivity: bool | None = None,
    continuous: Any = None,
    **overrides: Any,
) -> "PipelineResult":
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
    from evoparam.models import fit as fit_models
    from evoparam.prep import rereference

    p = _presets.get(preset)
    ts, artifacts = _artifact_steps(trials, p, random_state) if artifact else (trials, [])
    report = None
    if (
        reference_sensitivity
        if reference_sensitivity is not None
        else (reference and p.reference_sensitivity)
    ):
        report = _reference_report(ts, p, random_state)
    scheme = reference_scheme(p)
    if reference and scheme is not None:
        extra: dict[str, Any] = {"random_state": random_state} if scheme == "car_adaptive" else {}
        ts = rereference(ts, scheme, **extra)
    if baseline:
        ts = _baseline(ts, p, continuous)
    test_kwargs = {"method": p.existence_method, "alpha": p.existence_alpha}
    window = _test_window(ts, tuple(overrides.get("fit_window") or p.fit_window))
    if window != tuple(overrides.get("fit_window") or p.fit_window):
        test_kwargs["window"] = window
    test_kwargs.update(existence_kwargs or {})
    group = fit_models(ts, preset, random_state=random_state, n_jobs=n_jobs, **overrides)
    test = None
    if existence:
        group, test = _existence(ts, group, test_kwargs, random_state)
    fits = None
    if trial_fits if trial_fits is not None else p.trial_fit:
        fits = _trial_fits(ts, group, p, random_state, n_jobs)
    return PipelineResult(
        trials=ts,
        artifact=artifacts,
        group=group,
        trial_fits=fits,
        test=test,
        reference_report=report,
    )


def _artifact_steps(ts: Any, p: Any, random_state: Any) -> tuple[Any, list[Any]]:
    """Run the artifact steps of preset ``p``; return the cleaned trials and the results."""
    from evoparam.artifact import remove_artifact

    artifacts = []
    for kwargs in artifact_steps(p):
        kwargs.setdefault("baseline", _artifact_baseline(ts, p, kwargs.get("window")))
        result = remove_artifact(ts, random_state=random_state, **kwargs)
        artifacts.append(result)
        ts = result.cleaned
    return ts, artifacts


def _reference_report(ts: Any, p: Any, random_state: Any) -> Any:
    """Return the kappa report of the detections under several references (spec §6.2).

    None, with a warning, when it cannot run.
    """
    from evoparam.errors import DataError, warn
    from evoparam.prep import reference_sensitivity as sensitivity

    try:
        return sensitivity(ts, reference_kwargs={"random_state": random_state})
    except (DataError, SettingsError) as exc:
        warn(
            f"preset {p.name!r}: reference_sensitivity could not run ({exc}); no kappa "
            "report is attached.",
            UserWarning,
            stacklevel=3,
        )
    return None


def _baseline(ts: Any, p: Any, continuous: Any) -> Any:
    """Correct the baseline of preset ``p``: a window of the epoch, or the time before each train.

    The baseline per train (spec §6.1, erna: the last 50 ms before each train) uses the
    continuous recording when given; when it cannot be applied, a warning says so.
    """
    from evoparam.errors import DataError, warn
    from evoparam.prep import baseline_correct, train_baseline

    if p.baseline is not None:
        return baseline_correct(ts, cast("tuple[float, float]", tuple(p.baseline)))
    if p.train_baseline is not None:
        raw = continuous
        kwargs = {}
        if isinstance(continuous, tuple):
            raw, kwargs = continuous[0], {"sfreq": continuous[1], "t0": continuous[2]}
        try:
            return train_baseline(ts, p.train_baseline, raw=raw, **kwargs)
        except DataError as exc:
            warn(
                f"preset {p.name!r}: the baseline per train was not applied ({exc}).",
                UserWarning,
                stacklevel=3,
            )
    return ts


def _existence(ts: Any, group: Any, test_kwargs: Any, random_state: Any) -> tuple[Any, Any]:
    """Test the channels of ``group`` and attach the test; return the group fit and the test.

    As ``evoparam.existence.fit(existence=True)``, keeping the test for the derivatives.
    """
    from evoparam.existence import attach_test, test_response

    settings = group.settings
    kwargs = {
        "window": tuple(settings["fit_window"]),
        "baseline": tuple(settings["baseline"]) if settings.get("baseline") else (-0.5, -0.05),
        "by": (),
        "random_state": random_state,
    }
    kwargs.update(test_kwargs)
    test = test_response(ts, **kwargs)
    return attach_test(group, test), test


def _trial_fits(ts: Any, group: Any, p: Any, random_state: Any, n_jobs: int) -> Any:
    """Fit every trial with the trial model and default covariates of preset ``p``.

    A default covariate whose metadata column is missing is left out, with a warning.
    """
    from evoparam.errors import warn
    from evoparam.trials import fit_trials

    # the preset's default covariates (rodent_ep, primate_ep: metadata["state"])
    covariates = [
        c for c in p.trial_covariates if ts.metadata is not None and c in ts.metadata.columns
    ]
    for missing in sorted(set(p.trial_covariates) - set(covariates)):
        warn(
            f"preset {p.name!r}: metadata[{missing!r}] is missing, so the default "
            f"covariate {missing!r} of the trial fits is not computed.",
            UserWarning,
            stacklevel=3,
        )
    return fit_trials(
        ts,
        group,
        mode=p.trial_mode or "shift_scale",
        random_state=random_state,
        n_jobs=n_jobs,
        covariates=covariates,
    )


###################################################################################################
## Reference pipelines (REQ-340)

# name -> preset, the published data it stands for, the stand-in used when those data are
# not at hand, and the options of run_pipeline. The methods text depends on the settings and
# on the noise reference the fits resolved (noise_source="auto" switches to the plus-minus
# average when the baseline holds fewer than 30 effective samples), so each stand-in has the
# sampling of its real recording and resolves the same reference: it gives the words of the
# real pipeline.
REFERENCE_PIPELINES: dict[str, dict[str, Any]] = {
    "erpcore_p3": {
        "preset": "erp",
        "data": "ERP CORE P3 (active visual oddball), channel Pz, all epochs of one "
        "participant (Kappenman et al. 2021; OSF node etdkz, CC-BY-4.0; downloaded "
        "by the network test, not redistributed)",
        "stand_in": {
            "generator": "simulate_erp",
            "kwargs": {
                "n_trials": 30,
                "n_channels": 1,
                "components": ["P3"],
                "ch_names": ["Pz"],
                "sfreq": 256.0,
                "seed": 0,
            },
        },
        "options": {},
    },
    "ds004080_ccep": {
        "preset": "ccep",
        "data": "OpenNeuro ds004080, sub-ccepAgeUMCU01 run-021448, stimulation site "
        "PT01-PT02 (van Blooijs et al. 2023; CC0)",
        "stand_in": {
            "generator": "simulate_ccep",
            "kwargs": {"n_trials": 10, "n_channels": 10, "sfreq": 1024.0, "seed": 0},
        },
        "options": {"trial_fits": True},
    },
    "sim_tep": {
        "preset": "tep",
        "data": "evoparam.sim.simulate_tep, 20 trials, 3 channels (stimulated site Cz), "
        "epochs -1 to 1 s, seed 0",
        "stand_in": {
            "generator": "simulate_tep",
            "kwargs": {
                "n_trials": 20,
                "n_channels": 3,
                "stim_site": "Cz",
                "tmin": -1.0,
                "tmax": 1.0,
                "seed": 0,
            },
        },
        "options": {},
    },
}


def reference_input(name: Any) -> Any:
    """Return the stand-in TrialSet of a reference pipeline.

    The TrialSet is simulated; see REFERENCE_PIPELINES.
    """
    from evoparam import sim

    spec = REFERENCE_PIPELINES[name]["stand_in"]
    ts, _ = getattr(sim, spec["generator"])(**spec["kwargs"])
    return ts


def run_reference_pipeline(name: Any, trials: Any = None, *, random_state: int = 0) -> Any:
    """Run reference pipeline ``name`` on ``trials`` (default: its stand-in input)."""
    if name not in REFERENCE_PIPELINES:
        raise SettingsError(
            f"Unknown reference pipeline {name!r}; known: {sorted(REFERENCE_PIPELINES)}."
        )
    spec = REFERENCE_PIPELINES[name]
    ts = reference_input(name) if trials is None else trials
    return run_pipeline(ts, spec["preset"], random_state=random_state, **spec["options"])
