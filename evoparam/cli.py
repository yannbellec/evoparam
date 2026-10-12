# SPDX-License-Identifier: Apache-2.0
"""The ``evoparam`` command (spec §5.13, §10.5): standard-library ``argparse`` only.

Every sub-command accepts ``--json`` (one JSON summary object on stdout, logs on stderr),
``--verbose``, ``--n-jobs``, ``--random-state`` and ``--config`` (a JSON file whose keys are
the long option names; command-line values win), and writes nothing outside ``--out``.

Exit codes: 0 success; 1 runtime error; 2 usage error; 3 partial success (a channel with
``status="failed"``); 4 missing optional dependency; 5 input error (``DataError``,
``ReaderError``); 6 reproduction mismatch; 130 interrupted.
"""

import argparse
import json
import os
import sys
import time
import warnings

from evoparam.errors import DataError, EvoparamError, ReaderError, SettingsError
from evoparam.version import __version__

__all__ = ["main", "run_fit", "EXIT_CODES"]

EXIT_CODES = {
    "ok": 0,
    "error": 1,
    "usage": 2,
    "partial": 3,
    "missing_dependency": 4,
    "input": 5,
    "mismatch": 6,
    "interrupted": 130,
}
RESEARCH_USE = (
    "For research use only: not a medical device; outputs are not intended for "
    "diagnosis or treatment decisions."
)


class _Outcome:
    def __init__(self):
        self.outputs, self.warnings = [], []
        self.n_units = {"ok": 0, "no_response": 0, "failed": 0, "skipped": 0}
        self.code = 0


def _log(args, message):
    if getattr(args, "verbose", "warning") in ("info", "debug") or not getattr(
        args, "json", False
    ):
        print(message, file=sys.stderr)


def _entities(args):
    return {"session": args.ses, "task": args.task, "run": args.run}


def _datatype(root, subject, wanted):
    if wanted:
        return wanted
    for datatype in ("ieeg", "eeg"):
        if os.path.isdir(os.path.join(root, f"sub-{subject}", datatype)) or any(
            os.path.isdir(os.path.join(root, f"sub-{subject}", s, datatype))
            for s in (
                os.listdir(os.path.join(root, f"sub-{subject}"))
                if os.path.isdir(os.path.join(root, f"sub-{subject}"))
                else []
            )
        ):
            return datatype
    return "ieeg"


SINGLE_FILES = (".vhdr", ".edf", ".bdf", ".fif", ".fif.gz", ".nwb", ".npz")


def _preset(args):
    """The preset of --preset, else the one of the datatype (ieeg: ccep, eeg: erp)."""
    from evoparam import presets

    if getattr(args, "preset", None):
        return presets.get(args.preset)
    if _is_single(args.input):
        return presets.get(
            "erp" if args.input.lower().endswith((".edf", ".bdf", ".fif", ".fif.gz")) else "ccep"
        )
    subject = (args.sub or "").removeprefix("sub-")
    return presets.get(
        "erp"
        if _datatype(args.input, subject, getattr(args, "datatype", None)) == "eeg"
        else "ccep"
    )


def _is_single(path):
    return os.path.isfile(path) and path.lower().endswith(SINGLE_FILES)


def _file_entities(path):
    """BIDS entities of a file name (sub, ses, task, run), for the derivative names."""
    import re

    found = dict(re.findall(r"(sub|ses|task|run)-([a-zA-Z0-9]+)", os.path.basename(path)))
    return found


def _events_for(path, args):
    """Onsets (s) and metadata of a single recording: --events, else the BIDS events.tsv
    next to it, else the annotations of an EDF+ file.
    """
    import pandas as pd

    from evoparam.data._harmonize import harmonize_stim_events

    table = None
    if getattr(args, "events", None):
        table = pd.read_csv(args.events, sep="\t", na_values=["n/a"], keep_default_na=False)
    else:
        base = os.path.basename(path)
        for suffix in ("_ieeg", "_eeg", "_meg"):
            if suffix + "." in base:
                candidate = os.path.join(
                    os.path.dirname(path), base.split(suffix + ".")[0] + "_events.tsv"
                )
                if os.path.isfile(candidate):
                    table = pd.read_csv(
                        candidate, sep="\t", na_values=["n/a"], keep_default_na=False
                    )
                break
    if table is None and path.lower().endswith((".edf", ".bdf")):
        from evoparam.io import read_events_edf

        table = pd.DataFrame(read_events_edf(path))
    if table is None or "onset" not in table:
        raise DataError(
            f"{path}: no events; give --events <events.tsv> (BIDS columns onset, "
            f"duration, trial_type)."
        )
    harmonized, _ = harmonize_stim_events(table)
    if "trial_type" in harmonized:
        stim = harmonized["trial_type"].astype(str) == "electrical_stimulation"
        if stim.any():
            harmonized = harmonized[stim].reset_index(drop=True)
    return harmonized["onset"].to_numpy(float), harmonized


def _read_single(path, args, epoch):
    """A TrialSet from one file: .nwb (its events), .fif (MNE epochs, or raw with its
    events), .npz (data [trials, channels, times], sfreq, tmin, optional ch_names), .vhdr /
    .edf / .bdf (epochs around the events of --events, the BIDS events.tsv next to the file,
    or the EDF+ annotations).
    """
    from evoparam.io import read, read_nwb

    lower = path.lower()
    if lower.endswith(".nwb"):
        return read_nwb(path, tmin=epoch[0], tmax=epoch[1])
    if lower.endswith((".fif", ".fif.gz")):
        from evoparam.data import TrialSet
        from evoparam.utils._optional import require

        mne = require("mne", extra="mne", caller="evoparam fit <file>.fif")
        try:
            epochs = mne.read_epochs(path, verbose="error")
        except (ValueError, RuntimeError, OSError):
            raw = mne.io.read_raw_fif(path, preload=False, verbose="error")
            try:
                events = mne.find_events(raw, verbose="error")
            except ValueError:
                events, _ = mne.events_from_annotations(raw, verbose="error")
            epochs = mne.Epochs(
                raw,
                events,
                tmin=epoch[0],
                tmax=epoch[1] - 1 / raw.info["sfreq"],
                baseline=None,
                preload=True,
                verbose="error",
            )
        return TrialSet.from_mne(epochs)
    if lower.endswith(".npz"):
        import numpy as np

        from evoparam.data import Provenance, TrialSet

        with np.load(path, allow_pickle=False) as archive:
            data = np.asarray(archive["data"], float)
            if data.ndim == 2:
                data = data[None]
            sfreq = float(archive["sfreq"])
            tmin = float(archive["tmin"]) if "tmin" in archive else float(epoch[0])
            names = (
                [str(n) for n in archive["ch_names"]]
                if "ch_names" in archive
                else [f"ch{i}" for i in range(data.shape[1])]
            )
        times = tmin + np.arange(data.shape[-1]) / sfreq
        return TrialSet(
            data=data,
            times=times,
            sfreq=sfreq,
            ch_names=names,
            provenance=Provenance(source={"path": path, "format": "npz"}),
        )
    onsets, metadata = _events_for(path, args)
    return read(path, onsets=onsets, epoch=tuple(epoch), metadata=metadata)


def _read(args, preset):
    from evoparam.io import read_bids

    epoch = tuple(args.epoch) if getattr(args, "epoch", None) else preset.epoch
    if _is_single(args.input):
        ts = _read_single(args.input, args, epoch)
        datatype = (
            "eeg" if args.input.lower().endswith((".edf", ".bdf", ".fif", ".fif.gz")) else "ieeg"
        )
        return ts, getattr(args, "datatype", None) or datatype
    if not args.sub:
        raise DataError("--sub is needed with a BIDS dataset.")
    subject = args.sub.removeprefix("sub-")
    datatype = _datatype(args.input, subject, getattr(args, "datatype", None))
    return read_bids(
        args.input,
        subject,
        datatype=datatype,
        epoch=epoch,
        remote=bool(getattr(args, "remote", False)),
        **_entities(args),
    ), datatype


def _target(args):
    """(bids_root for SourceDatasets, subject, entities) of the derivatives."""
    if _is_single(args.input):
        found = _file_entities(args.input)
        subject = (args.sub or found.get("sub") or "01").removeprefix("sub-")
        entities = {
            "session": args.ses or found.get("ses"),
            "task": args.task or found.get("task") or "unknown",
            "run": args.run or found.get("run"),
        }
        return None, subject, entities
    return args.input, args.sub, _entities(args)


def run_fit(ts, preset, *, random_state=None, n_jobs=1, existence=True, trials=None, log=None):
    """The chain of ``evoparam fit``: :func:`evoparam.reports.run_pipeline` (artifact and
    reference of the preset, baseline, fit, existence test with the preset's method, trial
    fits when the preset says so).

    Returns ``(results, notes)``: the list of results for :func:`write_bids_derivatives`
    (GroupFit, ResponseTest, trial fits, ArtifactResult) and the steps skipped.
    """
    from evoparam.reports import run_pipeline

    result = run_pipeline(
        ts,
        preset,
        random_state=random_state,
        n_jobs=n_jobs,
        existence=existence,
        trial_fits=trials,
    )
    return result.results, []


def _count(outcome, group):
    for fit in group.fits:
        outcome.n_units[fit.status] = outcome.n_units.get(fit.status, 0) + 1
        outcome.warnings += [w for w in fit.warnings]


def cmd_fit(args, outcome):
    import numpy as np

    from evoparam.data import GroupFit
    from evoparam.io import write_bids_derivatives
    from evoparam.reports import methods_text

    preset = _preset(args)
    ts, datatype = _read(args, preset)
    results, notes = run_fit(
        ts,
        preset,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        existence=not args.no_existence,
    )
    outcome.warnings += notes
    group = next(r for r in results if isinstance(r, GroupFit))
    _count(outcome, group)
    failed_trials = 0
    for trial_fits in results:
        for fit in getattr(trial_fits, "fits", []) if not isinstance(trial_fits, GroupFit) else []:
            failed_trials += int(np.sum(~np.asarray(fit.converged, bool)))
    if failed_trials:
        # a trial whose fit did not converge is a failed trial (exit code 3, spec §10.5)
        outcome.warnings.append(f"{failed_trials} trial fit(s) did not converge")
    out = args.out
    root, subject, entities = _target(args)
    written = write_bids_derivatives(
        results,
        root,
        subject=subject,
        datatype=datatype,
        out=out,
        overwrite=True,
        trials=ts,
        **entities,
    )
    # the group fit is saved by write_bids_derivatives next to its table
    # (<stem>_desc-evoparam_responses.evoparam), as every result of the run
    outcome.outputs += written
    _log(args, methods_text(results))
    if outcome.n_units.get("failed") or failed_trials:
        outcome.code = EXIT_CODES["partial"]


def cmd_test(args, outcome):
    from evoparam.existence import test_response
    from evoparam.io import write_bids_derivatives
    from evoparam.prep import baseline_correct

    preset = _preset(args)
    ts, datatype = _read(args, preset)
    if preset.baseline is not None:
        ts = baseline_correct(ts, preset.baseline, check=False)
    test = test_response(
        ts,
        method=args.method,
        window=preset.fit_window,
        baseline=preset.baseline or (ts.times[0], 0.0),
        random_state=args.random_state,
        by=(),
    )
    root, subject, entities = _target(args)
    outcome.outputs += write_bids_derivatives(
        test,
        root,
        subject=subject,
        datatype=datatype,
        out=args.out,
        overwrite=True,
        trials=ts,
        **entities,
    )


def cmd_clean(args, outcome):
    from evoparam.artifact import remove_artifact
    from evoparam.data.serialize import save_object
    from evoparam.io import write_bids_derivatives

    preset = _preset(args)
    ts, datatype = _read(args, preset)
    window = tuple(args.window) if args.window else preset.artifact_window
    if window is None:
        raise DataError("clean: give --window TMIN TMAX (the preset has no artifact window).")
    result = remove_artifact(ts, method=args.method, window=window, random_state=args.random_state)
    root, subject, entities = _target(args)
    written = write_bids_derivatives(
        result,
        root,
        subject=subject,
        datatype=datatype,
        out=args.out,
        overwrite=True,
        trials=ts,
        **entities,
    )
    outcome.outputs += written
    folder = os.path.dirname(next(p for p in written if p.endswith("_artifact.json")))
    stem = os.path.basename(next(p for p in written if p.endswith("_artifact.json")))
    stem = stem[: -len("_artifact.json")]
    # the cleaned epochs: FIF when mne is installed, else npz (spec §10.5); and the exact
    # container
    try:
        import mne  # noqa: F401

        path = os.path.join(folder, f"{stem}_epo.fif")
        result.cleaned.to_mne().save(path, overwrite=True, verbose="error")
    except ImportError:
        import numpy as np

        path = os.path.join(folder, f"{stem}_epo.npz")
        np.savez(
            path,
            data=result.cleaned.data,
            sfreq=result.cleaned.sfreq,
            tmin=result.cleaned.times[0],
            ch_names=np.array(result.cleaned.ch_names),
        )
    outcome.outputs.append(path)
    container = os.path.join(args.out, f"sub-{subject.removeprefix('sub-')}", "cleaned.evoparam")
    outcome.outputs.append(save_object(result.cleaned, container, overwrite=True))


def _containers(path):
    if path.endswith(".evoparam") or path.endswith(".zip"):
        return [path]
    found = []
    for folder, dirs, files in os.walk(path):
        for d in list(dirs):
            if d.endswith(".evoparam") and os.path.isfile(
                os.path.join(folder, d, "manifest.json")
            ):
                found.append(os.path.join(folder, d))
                dirs.remove(d)
    if not found:
        raise DataError(f"No .evoparam result under {path}.")
    return sorted(found)


def cmd_report(args, outcome):
    from evoparam import reports
    from evoparam.data.serialize import load_object

    for container in _containers(args.input):
        result = load_object(container)
        base = os.path.join(
            args.out or os.path.dirname(container),
            os.path.basename(container).rsplit(".", 1)[0] + "_report",
        )
        os.makedirs(os.path.dirname(base), exist_ok=True)
        if args.format == "html":
            outcome.outputs.append(reports.html(result, base + ".html"))
        elif args.format == "pdf":
            outcome.outputs.append(reports.pdf(result, base + ".pdf"))
        elif args.format == "json":
            outcome.outputs.append(reports.to_json(result, base + ".json"))
        else:
            with open(base + ".md", "w", encoding="utf-8") as fobj:
                fobj.write(reports.markdown(result))
            outcome.outputs.append(base + ".md")


def cmd_simulate(args, outcome):
    from evoparam.sim import GENERATORS
    from evoparam.sim.bids import write_bids

    if args.modality not in GENERATORS:
        raise DataError(f"modality must be one of {sorted(GENERATORS)}.")
    seed = args.seed if args.seed is not None else args.random_state
    kwargs = {"n_trials": args.n_trials, "seed": seed}
    if args.n_channels:
        kwargs["n_channels"] = args.n_channels
    ts, truth = GENERATORS[args.modality](**kwargs)
    outcome.outputs.append(
        write_bids(
            ts,
            truth,
            args.out,
            subject=args.sub.removeprefix("sub-"),
            task=args.task or args.modality,
        )
    )


def cmd_detect_events(args, outcome):
    import fnmatch

    from evoparam.events import detect_events
    from evoparam.io import read_brainvision

    data, sfreq, names, *_ = read_brainvision(args.input)
    picks = [
        i for i, n in enumerate(names) if not args.channels or fnmatch.fnmatch(n, args.channels)
    ]
    events = detect_events(data[picks], sfreq=sfreq, ch_names=[names[i] for i in picks])
    # A BIDS events file starts with onset and duration (s).
    events = events.copy()
    if "onset" not in events.columns:
        events.insert(0, "onset", events["onset_time"].to_numpy(float))
    if "duration" not in events.columns:
        events.insert(1, "duration", 0.0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    events.to_csv(args.out, sep="\t", index=False, na_rep="n/a", lineterminator="\n")
    sidecar = args.out.rsplit(".", 1)[0] + ".json"
    with open(sidecar, "w", encoding="utf-8") as fobj:
        json.dump(
            {
                c: {
                    "Description": f"evoparam event column {c} (evoparam.events.detect_events)",
                    **(
                        {"Units": "s"} if c in ("onset", "duration") or c.endswith("_time") else {}
                    ),
                }
                for c in events.columns
            },
            fobj,
            indent=1,
        )
    outcome.n_units["ok"] = int(len(events))
    outcome.outputs += [args.out, sidecar]


def cmd_reproduce(args, outcome):
    from evoparam.reports import reproduce

    report = reproduce(args.input, rtol=args.rtol)
    outcome.extra = report.to_dict()
    if not report.identical:
        outcome.code = EXIT_CODES["mismatch"]
        outcome.warnings += [
            f"{d['field']} differs (relative difference {d['rel_diff']})"
            for d in report.differences
        ]


def cmd_migrate(args, outcome):
    from evoparam.data.serialize import load_object, save_object
    from evoparam.reports import migrate

    if args.out:
        # a migrated copy; the original is left as it is
        outcome.outputs.append(save_object(load_object(args.input), args.out, overwrite=True))
        return
    outcome.extra = migrate(args.input)
    outcome.outputs.append(outcome.extra["path"])


def cmd_info(args, outcome):
    from importlib import metadata

    from evoparam import presets

    extras = {}
    for name in (
        "pandas",
        "matplotlib",
        "mne",
        "jsonschema",
        "numpyro",
        "pymc",
        "pyarrow",
        "edfio",
        "pynwb",
        "pylsl",
    ):
        try:
            extras[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            extras[name] = None
    from evoparam.io.readers import READERS
    from evoparam.shapes import BACKGROUNDS, SHAPES

    plugins = [
        {"group": group, "name": name, "distribution": origin}
        for group, registry in (
            ("evoparam.shapes", SHAPES),
            ("evoparam.shapes", BACKGROUNDS),
            ("evoparam.presets", presets.PRESETS),
            ("evoparam.readers", READERS),
        )
        for name, origin in sorted(registry.plugins().items())
    ]

    def installed(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    # numerical backends: the fitting engine (NumPy/SciPy; the GPU backends of spec §9.1
    # are not in this version) and the engines of the hierarchical model, with whether the
    # packages they need are installed
    backends = {
        "fit": {"numpy": True},
        "hierarchical": {
            "eb": True,
            "numpyro": bool(installed("numpyro") and installed("jax")),
            "pymc": bool(installed("pymc")),
        },
    }
    outcome.extra = {
        "evoparam": __version__,
        "python": sys.version.split()[0],
        "numpy": metadata.version("numpy"),
        "scipy": metadata.version("scipy"),
        "optional": extras,
        "presets": sorted(presets.PRESETS),
        "plugins": plugins,
        "backends": backends,
        "intended_use": RESEARCH_USE,
    }
    if not args.json:
        for key, value in outcome.extra.items():
            print(f"{key}: {value}")


def cmd_presets(args, outcome):
    from evoparam import presets

    if args.action == "list":
        table = presets.list()
        outcome.extra = {"presets": table.to_dict("records")}
        if not args.json:
            print(table.to_string(index=False))
    else:
        if not args.name:
            raise DataError("presets show needs a preset name.")
        text = presets.to_json(args.name)
        outcome.extra = {"preset": json.loads(text)}
        if not args.json:
            print(text)
            print(presets.methods_note(args.name))


def cmd_cite(args, outcome):
    from evoparam.reports import citation

    text = citation(bibtex=args.bibtex)
    outcome.extra = {"citation": text}
    if not args.json:
        print(text)


def cmd_validate_bids(args, outcome):
    from evoparam.io import format_stim_report, validate_stim_metadata

    table = validate_stim_metadata(args.input, subject=args.sub)
    rows = table.astype(object).where(table.notna(), None)  # JSON has no NaN
    outcome.extra = {
        "fields": rows.to_dict("records"),
        "counts": {k: int(v) for k, v in table["status"].value_counts().items()},
    }
    if not args.json:
        print(format_stim_report(table))
    missing = table[table["status"] == "missing"]
    outcome.warnings += [f"missing: {r.events_file} {r.field}" for r in missing.itertuples()]
    if args.strict and len(missing):
        # the report is printed first; the error gives the exit code 5 (input error)
        raise ReaderError(f"{len(missing)} stimulation metadata field(s) missing (--strict).")


def _csv(text):
    return [t.strip() for t in text.split(",") if t.strip()] if text is not None else None


def cmd_benchmark(args, outcome):
    """``evoparam benchmark run | score | leaderboard | prepare-deposit`` (spec §8)."""
    from evoparam import benchmark

    log = (lambda message: _log(args, message)) if args.verbose in ("info", "debug") else None
    if args.action == "run" and args.suite in ("perf", "ci"):
        # the performance workloads P1-P6 (spec §9.1, REQ-290)
        from evoparam.benchmark import timing

        # --tasks selects workloads here (for example --task P1,P6)
        document = timing.run_suite(
            args.suite, only=_csv(args.tasks), log=log or (lambda line: _log(args, line))
        )
        path = timing.write_results(
            document,
            timing.results_path(document, args.out or os.path.join("benchmarks", "results")),
        )
        outcome.outputs.append(path)
        outcome.extra = {
            "suite": args.suite,
            "results": {
                timing._key(r): {"median_s": r["median_s"], "target_met": r["target_met"]}
                for r in document["results"]
            },
        }
    elif args.action == "run":
        if not args.out:
            raise SettingsError("benchmark run --suite core|full|<file> needs --out.")
        summary = benchmark.run_suite(
            args.suite,
            args.out,
            tasks=_csv(args.tasks),
            methods=_csv(args.methods),
            adapters=args.adapter or (),
            hidden_seeds=args.hidden_seeds,
            split=args.split,
            n_jobs=args.n_jobs,
            log=log,
        )
        outcome.extra = {
            k: summary[k]
            for k in (
                "suite",
                "split",
                "tasks",
                "methods",
                "adapters",
                "n_cells",
                "duration_s",
                "failures",
            )
        }
        outcome.outputs += (
            summary["results"] + summary["pages"] + [os.path.join(args.out, "summary.json")]
        )
        outcome.warnings += [
            f"{f['method']} failed on {f['cell']}: {f['error']}" for f in summary["failures"]
        ]
        if summary["failures"]:
            outcome.code = EXIT_CODES["partial"]
        _log(
            args,
            f"benchmark run: {summary['n_cells']} cells in {summary['duration_s']['total']:.1f} s",
        )
    elif args.action == "score":
        method = {
            "name": args.method_name,
            "version": args.method_version,
            "citation": args.citation,
            "kind": "submission",
            "settings": {},
        }
        document = benchmark.score(
            args.task,
            args.predictions,
            args.suite,
            method=method,
            hidden_seeds=args.hidden_seeds,
            split=args.split,
            allow_missing=args.allow_missing,
        )
        out = args.out or os.path.join(args.predictions, f"{args.method_name}__{args.task}.json")
        outcome.outputs.append(benchmark.write_result(document, out))
        outcome.extra = {
            "task": args.task,
            "metrics": document["metrics"],
            "signature": document["signature"]["digest"],
        }
        if not args.json:
            from evoparam.benchmark.report import format_metric, metric_label

            for key, value in document["headline"].items():
                print(f"{metric_label(key)}\t{format_metric(key, value)}")
    elif args.action == "leaderboard":
        outcome.outputs.append(benchmark.write_leaderboard(args.results, args.out))
        if args.report:
            outcome.outputs.append(benchmark.write_report(args.results, args.report))
    else:  # prepare-deposit
        path = benchmark.prepare_deposit(
            args.suite,
            args.out,
            formats=_csv(args.formats),
            tasks=_csv(args.tasks),
            hidden_seeds=args.hidden_seeds,
            overwrite=args.overwrite,
            log=log,
        )
        with open(path, encoding="utf-8") as fobj:
            manifest = json.load(fobj)
        outcome.outputs.append(path)
        outcome.extra = {
            "status": manifest["status"],
            "doi": manifest["doi"],
            "n_public_cells": manifest["public_split"]["n_cells"],
            "hidden_commitment_sha256": manifest["hidden_split"]["commitment_sha256"],
        }


def _benchmark_parser(sub, common):
    p = sub.add_parser(
        "benchmark", help="public benchmark: run, score, leaderboard, prepare-deposit (spec §8)"
    )
    actions = p.add_subparsers(dest="action", required=True)
    q = actions.add_parser(
        "run",
        parents=[common],
        help="run baselines (and adapters) on a "
        "suite, score them, write results, report and leaderboard",
    )
    q.add_argument(
        "--suite",
        default="core",
        help="core, full, or a suite file; perf or "
        "ci for the performance workloads P1-P6 (spec §9.1)",
    )
    q.add_argument(
        "--tasks",
        "--task",
        dest="tasks",
        help="comma-separated tasks (default: all); with --suite perf or ci, workload ids P1-P6",
    )
    q.add_argument(
        "--methods",
        "--method",
        dest="methods",
        help="comma-separated baselines (default: all; '' writes the task inputs "
        "only); --task T --method M reproduces one leaderboard entry",
    )
    q.add_argument(
        "--adapter",
        action="append",
        help="adapter specification (JSON) of an "
        "external tool, run in a separate process (repeatable)",
    )
    q.add_argument(
        "--hidden-seeds", help="hidden seeds document (checked against the suite's commitment)"
    )
    q.add_argument("--split", default="public", choices=("public", "hidden", "all"))
    q.add_argument(
        "--out",
        help="output folder (required for core, full or a suite file; "
        "perf and ci: default benchmarks/results)",
    )
    q = actions.add_parser(
        "score", parents=[common], help="score predictions of one task and write the signed result"
    )
    q.add_argument("--task", required=True, choices=("T1", "T2", "T3", "T4", "T5", "T6", "T7"))
    q.add_argument("--predictions", required=True, help="folder of <cell>.tsv (.npz for T4)")
    q.add_argument("--suite", default="core")
    q.add_argument("--method-name", default="submission")
    q.add_argument("--method-version", default="")
    q.add_argument("--citation", default="")
    q.add_argument("--hidden-seeds")
    q.add_argument("--split", default="public", choices=("public", "hidden", "all"))
    q.add_argument("--allow-missing", action="store_true")
    q.add_argument("--out", help="result file (default: <predictions>/<method>__<task>.json)")
    q = actions.add_parser(
        "leaderboard", parents=[common], help="static leaderboard page from result JSONs"
    )
    q.add_argument("--results", required=True, help="folder of result JSONs")
    q.add_argument("--out", required=True, help="HTML page to write")
    q.add_argument("--report", help="also write the Markdown report here")
    q = actions.add_parser(
        "prepare-deposit",
        parents=[common],
        help="write the public "
        "split as NWB/BIDS files with its manifest (local only; nothing "
        "is uploaded)",
    )
    q.add_argument("--suite", default="core")
    q.add_argument("--out", required=True)
    q.add_argument("--formats", default="bids,nwb")
    q.add_argument("--tasks")
    q.add_argument("--hidden-seeds")
    q.add_argument("--overwrite", action="store_true")
    for parser in actions.choices.values():
        parser.set_defaults(func=cmd_benchmark)


def _parser():
    parser = argparse.ArgumentParser(
        prog="evoparam", description="Parameterise evoked neural responses."
    )
    parser.add_argument(
        "--version", action="store_true", help="print the version and the research-use statement"
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="print one JSON summary on stdout")
    common.add_argument("--verbose", default="warning", choices=("warning", "info", "debug"))
    common.add_argument("--n-jobs", type=int, default=1)
    common.add_argument("--random-state", type=int, default=0)
    common.add_argument(
        "--config",
        help="JSON or TOML file with option values (the long "
        "option names; command-line values win)",
    )
    bids = argparse.ArgumentParser(add_help=False)
    bids.add_argument(
        "input", help="BIDS root, or one recording (.vhdr, .edf, .bdf, .fif, .nwb, .npz)"
    )
    bids.add_argument("--sub", help="subject (required with a BIDS root)")
    bids.add_argument("--ses")
    bids.add_argument("--task")
    bids.add_argument("--run")
    bids.add_argument("--datatype", choices=("ieeg", "eeg"))
    bids.add_argument("--epoch", type=float, nargs=2, metavar=("TMIN", "TMAX"))
    bids.add_argument("--remote", action="store_true", help="read from OpenNeuro")
    bids.add_argument("--preset", help="default: ccep for iEEG, erp for EEG")
    bids.add_argument(
        "--events",
        help="events.tsv of a single recording (default: the BIDS "
        "events.tsv next to it, or the EDF+ annotations)",
    )
    bids.add_argument("--out", required=True)
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("fit", parents=[common, bids], help="read, clean, fit, test, write")
    p.add_argument("--no-existence", action="store_true")
    p.set_defaults(func=cmd_fit)
    p = sub.add_parser("test", parents=[common, bids], help="existence test only")
    p.add_argument("--method", default="baseline_permutation")
    p.set_defaults(func=cmd_test)
    p = sub.add_parser("clean", parents=[common, bids], help="artifact removal only")
    p.add_argument("--method", default="template")
    p.add_argument("--window", type=float, nargs=2)
    p.set_defaults(func=cmd_clean)
    p = sub.add_parser("report", parents=[common], help="HTML, PDF, Markdown or JSON report")
    p.add_argument("input")
    p.add_argument("--format", default="html", choices=("html", "pdf", "md", "json"))
    p.add_argument("--out")
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("simulate", parents=[common], help="write a synthetic BIDS dataset")
    p.add_argument("--modality", default="ccep")
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--n-channels", type=int)
    p.add_argument("--seed", type=int, help="seed of the simulation (default: --random-state)")
    p.add_argument("--sub", default="01")
    p.add_argument("--task")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_simulate)
    p = sub.add_parser(
        "detect-events",
        parents=[common],
        help="stimulation events of a continuous BrainVision recording",
    )
    p.add_argument("input")
    p.add_argument("--channels", help="channel name pattern, for example 'PT*'")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_detect_events)
    p = sub.add_parser("reproduce", parents=[common], help="re-run a saved result")
    p.add_argument("input")
    p.add_argument("--rtol", type=float, default=1e-9)
    p.set_defaults(func=cmd_reproduce)
    p = sub.add_parser("migrate", parents=[common], help="rewrite a result in the current schema")
    p.add_argument("input")
    p.add_argument("--out")
    p.set_defaults(func=cmd_migrate)
    p = sub.add_parser("info", parents=[common], help="versions, extras, presets")
    p.set_defaults(func=cmd_info)
    p = sub.add_parser("presets", parents=[common], help="list or show presets")
    p.add_argument("action", choices=("list", "show"))
    p.add_argument("name", nargs="?")
    p.set_defaults(func=cmd_presets)
    p = sub.add_parser("cite", parents=[common], help="how to cite evoparam")
    p.add_argument("--bibtex", action="store_true")
    p.set_defaults(func=cmd_cite)
    _benchmark_parser(sub, common)
    p = sub.add_parser(
        "validate-bids",
        parents=[common],
        help="stimulation metadata "
        "validator: which fields each run states (present, inferred, missing, "
        "inconsistent)",
    )
    p.add_argument("input")
    p.add_argument("--sub", default=None, help="one subject (default: all)")
    p.add_argument("--strict", action="store_true", help="exit 5 when a field is missing")
    p.set_defaults(func=cmd_validate_bids)
    return parser


def _read_config(path):
    """Options of a JSON or TOML file (``.toml``: ``tomllib``, Python 3.11+, else ``tomli``).
    Top-level keys apply to every command; a table ``[evoparam]`` too, and a table named
    after the command (``[fit]``) only to it.
    """
    if path.lower().endswith(".toml"):
        try:
            import tomllib
        except ImportError:  # Python 3.10
            from evoparam.utils._optional import require

            tomllib = require("tomli", extra="cli", caller="evoparam --config <file>.toml")
        with open(path, "rb") as fobj:
            config = tomllib.load(fobj)
    else:
        with open(path, encoding="utf-8") as fobj:
            config = json.load(fobj)
    if not isinstance(config, dict):
        raise DataError(f"{path}: the configuration must be a table of options.")
    return config


def _apply_config(parser, args, argv):
    if not getattr(args, "config", None):
        return args
    config = _read_config(args.config)
    given = {a.split("=")[0].lstrip("-").replace("-", "_") for a in argv if a.startswith("--")}
    flat = {k: v for k, v in config.items() if not isinstance(v, dict)}
    for table in ("evoparam", args.command):  # [evoparam] then [<command>] tables
        if isinstance(config.get(table), dict):
            flat.update(config[table])
    for key, value in flat.items():
        attr = key.replace("-", "_")
        if attr not in given:
            setattr(args, attr, value)
    return args


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as stop:
        return int(stop.code or 0)
    if args.version:
        print(f"evoparam {__version__}\n{RESEARCH_USE}")
        return 0
    if not args.command:
        parser.print_help(sys.stderr)
        return EXIT_CODES["usage"]
    args = _apply_config(parser, args, argv)
    outcome = _Outcome()
    outcome.extra = None
    start = time.perf_counter()
    status = "ok"
    try:
        with warnings.catch_warnings():
            if args.verbose == "warning":
                warnings.simplefilter("ignore")
            args.func(args, outcome)
        code = outcome.code
        status = {0: "ok", 3: "partial", 6: "mismatch"}.get(code, "error")
    except KeyboardInterrupt:
        code, status = EXIT_CODES["interrupted"], "interrupted"
    except ImportError as err:
        code, status = EXIT_CODES["missing_dependency"], "error"
        print(f"evoparam: {err}", file=sys.stderr)
    except (DataError, ReaderError) as err:
        code, status = EXIT_CODES["input"], "error"
        print(f"evoparam: {err}", file=sys.stderr)
    except EvoparamError as err:
        code, status = EXIT_CODES["error"], "error"
        print(f"evoparam: {err}", file=sys.stderr)
    except Exception as err:  # every other failure is a runtime error with exit code 1
        code, status = EXIT_CODES["error"], "error"
        print(f"evoparam: {type(err).__name__}: {err}", file=sys.stderr)
    if args.json:
        summary = {
            "command": args.command,
            "version": __version__,
            "status": status,
            "exit_code": code,
            "outputs": [str(o) for o in outcome.outputs],
            "n_units": outcome.n_units,
            "warnings": sorted(set(outcome.warnings)),
            "duration_s": time.perf_counter() - start,
        }
        if outcome.extra is not None:
            summary["result"] = outcome.extra
        print(json.dumps(summary, default=str, allow_nan=False))
    elif outcome.outputs:
        for path in outcome.outputs:
            print(path)
    return code


if __name__ == "__main__":
    sys.exit(main())
