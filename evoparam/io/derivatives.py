"""BIDS derivatives of evoparam results (spec §10.2, REQ-315, REQ-316, REQ-336).

Layout::

    <out>/
    ├── dataset_description.json   (DatasetType "derivative", GeneratedBy, SourceDatasets)
    ├── descriptions.tsv
    ├── README
    └── sub-<label>/[ses-<label>/]<datatype>/
        ├── <stem>_desc-<desc>_responses.tsv / .json   (GroupFit)
        ├── <stem>_desc-<desc>_existence.tsv / .json   (ResponseTest)
        ├── <stem>_desc-<desc>_trials.tsv / .json      (trial fits)
        └── <stem>_desc-<desc>_artifact.json           (ArtifactResult)

``n/a`` marks missing values, as BIDS requires. Every TSV has a JSON sidecar describing its
columns (``Description``, ``Units``, ``Levels``), with ``Settings``, ``Preset``,
``PackageVersion`` and ``Provenance``. Non-standard stimulation columns carry the
``evoparam_`` prefix.
"""

import json
import math
import os

import numpy as np

from evoparam.data._jsonutil import to_jsonable
from evoparam.errors import DataError
from evoparam.version import __version__

__all__ = ["write_bids_derivatives", "RESPONSE_COLUMNS", "EXISTENCE_COLUMNS_BIDS",
           "TRIAL_COLUMNS_BIDS"]

# column -> (description, unit); "data" is the amplitude unit of the result
RESPONSE_COLUMNS = {
    "channel": ("Channel name, as in channels.tsv.", ""),
    "stim_site": ("Stimulation site (contact pair) of the averaged trials.", ""),
    "condition": ("Condition of the averaged trials (stimulation site, current, ...).", ""),
    "component": ("Component label from the preset's latency windows (for example N1), or "
                  "c<k> when none applies.", ""),
    "component_index": ("Position of the component by latency (0 first; -1: no component).",
                        ""),
    "shape": ("Shape function of the fitted component.", ""),
    "status": ("Outcome of the fit of the channel: ok, no_response, failed or skipped.", ""),
    "present": ("Existence decision of the channel's test (true, false, or n/a when no test "
                "was run).", ""),
    "latency": ("Latency of the component extremum.", "s"),
    "latency_se": ("Standard error of the latency.", "s"),
    "latency_ci_low": ("Lower bound of the latency interval.", "s"),
    "latency_ci_high": ("Upper bound of the latency interval.", "s"),
    "amplitude": ("Signed amplitude of the component extremum.", "data"),
    "amplitude_se": ("Standard error of the amplitude.", "data"),
    "amplitude_ci_low": ("Lower bound of the amplitude interval.", "data"),
    "amplitude_ci_high": ("Upper bound of the amplitude interval.", "data"),
    "polarity": ("Sign of the extremum: 1 or -1.", ""),
    "fwhm": ("Full width at half maximum of the component.", "s"),
    "onset": ("Time where the component reaches 10 % of its amplitude, before the peak.", "s"),
    "offset": ("Time where the component returns to 10 % of its amplitude.", "s"),
    "area": ("Area of the component.", "data*s"),
    "rise_time": ("Onset to peak.", "s"),
    "decay_time": ("Peak to offset.", "s"),
    "symmetry": ("(decay - rise) / (decay + rise).", ""),
    "sharpness": ("Curvature of the unit-amplitude shape at the peak times FWHM squared.", ""),
    "r2": ("1 - residual sum of squares / total sum of squares in the fit window.", ""),
    "rmse": ("Root mean square of the residual.", "data"),
    "nrmse": ("RMSE divided by the range of the data in the window.", ""),
    "n_trials": ("Number of trials averaged.", ""),
    "window_start": ("Start of the fit window.", "s"),
    "window_stop": ("End of the fit window.", "s"),
}
EXISTENCE_COLUMNS_BIDS = {
    "channel": ("Channel name.", ""), "stim_site": ("Stimulation site.", ""),
    "condition": ("Condition.", ""), "method": ("Existence-test method.", ""),
    "statistic": ("Test statistic.", ""), "p_value": ("Corrected p-value.", ""),
    "p_value_raw": ("Uncorrected p-value.", ""),
    "significant": ("Decision at alpha after correction.", ""),
    "onset": ("Start of the response.", "s"), "offset": ("End of the response.", "s"),
    "duration": ("Duration of the response.", "s"),
    "snr_db": ("Signal-to-noise ratio of the response.", "dB"),
    "alpha": ("Significance level.", ""), "correction": ("Correction across channels.", ""),
    "n_permutations": ("Number of permutations.", ""), "n_trials": ("Number of trials.", ""),
}
TRIAL_COLUMNS_BIDS = {
    "trial": ("Index of the trial in the source run.", ""),
    "onset": ("Onset of the trial in the source recording.", "s"),
    "channel": ("Channel name.", ""), "stim_site": ("Stimulation site.", ""),
    "component": ("Component label.", ""), "shift": ("Latency shift from the channel fit.", "s"),
    "scale": ("Amplitude scale of the trial.", ""), "latency": ("Trial latency.", "s"),
    "amplitude": ("Trial amplitude.", "data"), "fwhm": ("Width.", "s"),
    "r2": ("Goodness of fit of the trial around its components.", ""),
    "converged": ("The trial fit converged.", ""),
    "rejected": ("The trial is not trusted (not converged, at a bound or poor fit).", ""),
    "reject_reason": ("Why the trial is not trusted.", ""),
    "quality": ("Artifact-removal quality score of the trial (0 to 1).", ""),
}
STIM_COLUMNS = {
    "evoparam_stim_current_mA": ("Stimulation current.", "mA"),
    "evoparam_stim_pulse_width_us": ("Pulse width.", "us"),
    "evoparam_stim_polarity": ("Pulse polarity (Table 4.1 vocabulary).", ""),
    "evoparam_stim_n_pulses": ("Pulses per train.", ""),
    "evoparam_stim_frequency_hz": ("Stimulation frequency.", "Hz"),
    "evoparam_stim_charge_uC": ("Charge per phase.", "uC"),
}


def _cell(value):
    if value is None:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        return "n/a" if not math.isfinite(float(value)) else repr(float(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    text = str(value)
    return "n/a" if text in ("", "nan", "None") else text.replace("\t", " ")


def _write_tsv(path, columns, rows):
    with open(path, "w", encoding="utf-8") as fobj:
        fobj.write("\t".join(columns) + "\n")
        for row in rows:
            fobj.write("\t".join(_cell(row.get(c)) for c in columns) + "\n")


def _sidecar(columns, units, extra):
    out = {}
    for name in columns:
        description, unit = columns[name]
        entry = {"Description": description}
        # every column states its unit (REQ-342); "n/a" for labels and dimensionless values
        entry["Units"] = (units if unit == "data" else unit.replace("data", units)) \
            if unit else "n/a"
        out[name] = entry
    out.update(extra)
    return out


def _write_json(path, content):
    with open(path, "w", encoding="utf-8") as fobj:
        fobj.write(json.dumps(to_jsonable(content), indent=1, allow_nan=False))


def _stem(subject, session, task, run):
    parts = [f"sub-{subject}"]
    if session:
        parts.append(f"ses-{session}")
    if task:
        parts.append(f"task-{task}")
    if run:
        parts.append(f"run-{run}")
    return "_".join(parts)


def _strip(value, prefix):
    if value is None:
        return None
    value = str(value)
    return value[len(prefix):] if value.startswith(prefix) else value


def _provenance(result):
    provenance = getattr(result, "provenance", None)
    return provenance.to_dict() if provenance is not None else None


def _preset(settings):
    name = (settings or {}).get("preset")
    if not name:
        return None
    note = (settings or {}).get("preset_note") or ""
    version = note.split(" v")[1].split(" ")[0].rstrip(";") if " v" in note else None
    return {"Name": name, "Version": version}


def _response_rows(group, test=None):
    present = {}
    if test is not None:
        names = test.settings.get("channels") or []
        conditions = test.settings.get("conditions") or [""] * len(names)
        for name, cond, sig in zip(names, conditions, np.ravel(test.significant)):
            present[(name, cond)] = bool(sig)
    rows = []
    for ch, cond, pos in group.index[["channel", "condition", "position"]].itertuples(index=False):
        fit = group.fits[int(pos)]
        gof = fit.gof
        base = {"channel": ch, "stim_site": cond or None, "condition": cond or None,
                "status": fit.status,
                "present": present.get((ch, cond), present.get((ch, ""))),
                "r2": getattr(gof, "r2", None), "rmse": getattr(gof, "rmse", None),
                "nrmse": getattr(gof, "nrmse", None),
                "n_trials": (fit.settings or {}).get("n_trials"),
                "window_start": fit.window[0], "window_stop": fit.window[1]}
        if not fit.components:
            rows.append(dict(base, component=None, component_index=-1))
            continue
        for k, comp in enumerate(fit.components):
            row = dict(base, component=comp.label or f"c{k}", component_index=k,
                       shape=comp.shape)
            for name in ("latency", "amplitude", "polarity", "fwhm", "onset", "offset", "area",
                         "rise_time", "decay_time", "symmetry", "sharpness"):
                row[name] = getattr(comp, name)
            for name in ("latency", "amplitude"):
                row[f"{name}_se"] = (comp.se or {}).get(name)
                ci = (comp.ci or {}).get(name)
                if ci is not None:
                    row[f"{name}_ci_low"], row[f"{name}_ci_high"] = ci
            rows.append(row)
    return rows


def write_bids_derivatives(results, bids_root=None, *, subject, session=None, task=None,
                           run=None, datatype="ieeg", out=None, desc="evoparam",
                           pipeline_name="evoparam", units="uV", source_dataset=None,
                           overwrite=False):
    """Write results as BIDS derivatives (spec §10.2).

    Parameters
    ----------
    results : GroupFit, ResponseTest, ArtifactResult, trial fits, or a list of them
    bids_root : str, optional
        The raw dataset; its ``dataset_description.json`` gives ``SourceDatasets``.
    subject, session, task, run : str
        BIDS entities of the source run (``sub-`` prefixes accepted).
    out : str, optional
        Derivative folder (default ``<bids_root>/derivatives/evoparam-<version>``).
    units : str
        Amplitude unit of the results.

    Returns
    -------
    list of str
        The files written.
    """

    from evoparam.data import ArtifactResult, GroupFit, ResponseTest

    items = list(results) if isinstance(results, (list, tuple)) else [results]
    if out is None:
        if bids_root is None:
            raise DataError("write_bids_derivatives needs out= or bids_root=.")
        out = os.path.join(bids_root, "derivatives", f"{pipeline_name}-{__version__}")
    subject, session = _strip(subject, "sub-"), _strip(session, "ses-")
    task, run = _strip(task, "task-"), _strip(run, "run-")
    folder = os.path.join(out, f"sub-{subject}", *([f"ses-{session}"] if session else []),
                          datatype)
    os.makedirs(folder, exist_ok=True)
    stem = _stem(subject, session, task, run) + f"_desc-{desc}"
    written = []

    def target(suffix):
        path = os.path.join(folder, f"{stem}_{suffix}")
        if os.path.exists(path) and not overwrite:
            raise DataError(f"{path} exists; pass overwrite=True.")
        written.append(path)
        return path

    groups = [r for r in items if isinstance(r, GroupFit)]
    tests = [r for r in items if isinstance(r, ResponseTest)]
    artifacts = [r for r in items if isinstance(r, ArtifactResult)]
    trial_sets = [r for r in items if hasattr(r, "table") and hasattr(r, "fits")
                  and not isinstance(r, GroupFit)]
    for group in groups:
        rows = _response_rows(group, tests[0] if tests else None)
        _write_tsv(target("responses.tsv"), list(RESPONSE_COLUMNS), rows)
        _write_json(target("responses.json"), _sidecar(
            RESPONSE_COLUMNS, units,
            {"Settings": group.settings, "Preset": _preset(group.settings),
             "PackageVersion": __version__, "Provenance": _provenance(group)}))
    for test in tests:
        names = test.settings.get("channels") or [f"ch{i}" for i in range(len(test.statistic))]
        conditions = test.settings.get("conditions") or [""] * len(names)
        rows = []
        for i, (name, cond) in enumerate(zip(names, conditions)):
            rows.append({"channel": name, "stim_site": cond or None, "condition": cond or None,
                         "method": test.method, "statistic": test.statistic[i],
                         "p_value": test.p_value[i], "p_value_raw": test.p_value_raw[i],
                         "significant": bool(test.significant[i]), "onset": test.onset[i],
                         "offset": test.offset[i], "duration": test.duration[i],
                         "snr_db": test.snr_db[i], "alpha": test.alpha,
                         "correction": test.correction, "n_permutations": test.n_permutations,
                         "n_trials": test.settings.get("n_trials")})
        _write_tsv(target("existence.tsv"), list(EXISTENCE_COLUMNS_BIDS), rows)
        _write_json(target("existence.json"), _sidecar(
            EXISTENCE_COLUMNS_BIDS, units,
            {"Settings": test.settings, "PackageVersion": __version__,
             "Provenance": _provenance(test)}))
    for tset in trial_sets:
        table = tset.table
        rows = []
        for record in table.to_dict("records"):
            rows.append({"trial": record.get("trial"), "onset": record.get("onset"),
                         "channel": record.get("channel"),
                         "stim_site": record.get("condition") or None,
                         "component": record.get("label") or f"c{record.get('component')}",
                         "shift": record.get("shift"), "scale": record.get("scale"),
                         "latency": record.get("latency"), "amplitude": record.get("amplitude"),
                         "fwhm": record.get("fwhm"), "r2": record.get("r2"),
                         "converged": record.get("converged"),
                         "rejected": not bool(record.get("trusted", True)),
                         "reject_reason": None if record.get("trusted", True) else "not_trusted",
                         "quality": record.get("quality")})
        _write_tsv(target("trials.tsv"), list(TRIAL_COLUMNS_BIDS), rows)
        _write_json(target("trials.json"), _sidecar(
            TRIAL_COLUMNS_BIDS, units,
            {"Settings": tset.settings, "PackageVersion": __version__,
             "Provenance": _provenance(tset)}))
    records = []
    for artifact in artifacts:
        replaced = []
        masks = getattr(artifact.cleaned, "masks", None)
        if masks is not None:
            times = artifact.cleaned.times
            for trial, channel in np.ndindex(*masks.shape[:2]):
                edges = np.flatnonzero(np.diff(np.r_[0, masks[trial, channel].astype(int), 0]))
                for start, stop in zip(edges[::2], edges[1::2]):
                    replaced.append({"trial": int(trial),
                                     "channel": artifact.cleaned.ch_names[channel],
                                     "start": float(times[start]),
                                     "stop": float(times[stop - 1])})
        quality = artifact.quality_summary()
        records.append({
            "Method": artifact.method, "Window": list(artifact.window),
            "Parameters": artifact.settings, "PackageVersion": __version__,
            "QualitySummary": {"median": quality["median"], "p05": quality["p05"]},
            "ReplacedSamples": replaced, "Provenance": _provenance(artifact)})
    if records:
        # One record per run: the last artifact step (its cleaned trials were fitted), and
        # every step in order when the preset chains several (e.g. interpolate+subspace).
        record = dict(records[-1])
        if len(records) > 1:
            record["Steps"] = [{k: v for k, v in r.items() if k != "Provenance"}
                               for r in records]
        _write_json(target("artifact.json"), record)
    # dataset level
    source = {}
    if bids_root is not None and os.path.exists(os.path.join(bids_root,
                                                             "dataset_description.json")):
        with open(os.path.join(bids_root, "dataset_description.json"), encoding="utf-8") as f:
            raw = json.load(f)
        source = {k: v for k, v in (("DOI", raw.get("DatasetDOI")),
                                    ("URL", bids_root if os.path.isdir(bids_root) else None),
                                    ("Version", raw.get("DatasetVersion"))) if v}
    if source_dataset:
        source.update(source_dataset)
    main = (groups or tests or trial_sets or artifacts or [None])[0]
    generated = {"Name": pipeline_name, "Version": __version__,
                 "Description": "Parameterisation of evoked responses",
                 "CodeURL": "https://github.com/yannbellec/evoparam"}
    if main is not None and getattr(main, "provenance", None) is not None:
        # the provenance of the run written last (REQ-336); each sidecar holds its own
        generated["Provenance"] = _provenance(main)
    description = {"Name": pipeline_name, "BIDSVersion": "1.9.0", "DatasetType": "derivative",
                   "GeneratedBy": [generated],
                   "SourceDatasets": [source] if source else []}
    path = os.path.join(out, "dataset_description.json")
    _write_json(path, description)
    with open(os.path.join(out, "descriptions.tsv"), "w", encoding="utf-8") as fobj:
        fobj.write("desc_id\tdescription\n")
        fobj.write(f"desc-{desc}\tevoparam results: fitted components, existence tests, "
                   f"trial parameters and artifact records\n")
    # The evoparam tables have no BIDS suffix yet (no standard for evoked-response
    # parameters, spec §10.2): they are declared to the validator, as BIDS prescribes for
    # files the specification does not cover; so are the result containers saved with them.
    ignore = os.path.join(out, ".bidsignore")
    with open(ignore, "w", encoding="utf-8") as fobj:
        fobj.write("\n".join([f"*_desc-{desc}_{suffix}" for suffix in (
            "responses.tsv", "responses.json", "existence.tsv", "existence.json",
            "trials.tsv", "trials.json", "artifact.json")] + ["*.evoparam/", "*.evoparam/**"])
                   + "\n")
    readme = [f"evoparam {__version__} derivatives.", "",
              "*_responses.tsv: fitted components, one row per channel x condition x "
              "component; *_existence.tsv: existence tests; *_trials.tsv: trial-by-trial "
              "parameters; *_artifact.json: artifact removal and the replaced samples.",
              "Every TSV has a JSON sidecar with column descriptions, units, settings and "
              "provenance."]
    for group in groups:
        note = (group.settings or {}).get("preset_note")
        if note:
            readme.append(f"Preset: {note}.")
    with open(os.path.join(out, "README"), "w", encoding="utf-8") as fobj:
        fobj.write("\n".join(readme) + "\n")
    return written + [path, os.path.join(out, "descriptions.tsv"), ignore,
                      os.path.join(out, "README")]
