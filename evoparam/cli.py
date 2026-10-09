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

from evoparam.errors import DataError, EvoparamError, ReaderError
from evoparam.version import __version__

__all__ = ["main", "run_fit", "EXIT_CODES"]

EXIT_CODES = {"ok": 0, "error": 1, "usage": 2, "partial": 3, "missing_dependency": 4,
              "input": 5, "mismatch": 6, "interrupted": 130}
RESEARCH_USE = ("For research use only: not a medical device; outputs are not intended for "
                "diagnosis or treatment decisions.")


class _Outcome:
    def __init__(self):
        self.outputs, self.warnings = [], []
        self.n_units = {"ok": 0, "no_response": 0, "failed": 0, "skipped": 0}
        self.code = 0


def _log(args, message):
    if getattr(args, "verbose", "warning") in ("info", "debug") or not getattr(args, "json",
                                                                               False):
        print(message, file=sys.stderr)


def _entities(args):
    return {"session": args.ses, "task": args.task, "run": args.run}


def _datatype(root, subject, wanted):
    if wanted:
        return wanted
    for datatype in ("ieeg", "eeg"):
        if os.path.isdir(os.path.join(root, f"sub-{subject}", datatype)) or any(
                os.path.isdir(os.path.join(root, f"sub-{subject}", s, datatype))
                for s in (os.listdir(os.path.join(root, f"sub-{subject}"))
                          if os.path.isdir(os.path.join(root, f"sub-{subject}")) else [])):
            return datatype
    return "ieeg"


def _read(args, preset):
    from evoparam.io import read_bids

    subject = args.sub.removeprefix("sub-")
    datatype = _datatype(args.input, subject, getattr(args, "datatype", None))
    epoch = tuple(args.epoch) if getattr(args, "epoch", None) else preset.epoch
    return read_bids(args.input, subject, datatype=datatype, epoch=epoch,
                     remote=bool(getattr(args, "remote", False)), **_entities(args)), datatype


def run_fit(ts, preset, *, random_state=None, n_jobs=1, existence=True, trials=None,
            log=None):
    """The chain of ``evoparam fit``: :func:`evoparam.reports.run_pipeline` (artifact and
    reference of the preset, baseline, fit, existence test with the preset's method, trial
    fits when the preset says so).

    Returns ``(results, notes)``: the list of results for :func:`write_bids_derivatives`
    (GroupFit, ResponseTest, trial fits, ArtifactResult) and the steps skipped.
    """

    from evoparam.reports import run_pipeline

    result = run_pipeline(ts, preset, random_state=random_state, n_jobs=n_jobs,
                          existence=existence, trial_fits=trials)
    return result.results, []


def _count(outcome, group):
    for fit in group.fits:
        outcome.n_units[fit.status] = outcome.n_units.get(fit.status, 0) + 1
        outcome.warnings += [w for w in fit.warnings]


def cmd_fit(args, outcome):
    from evoparam.data import GroupFit
    from evoparam.data.serialize import save_object
    from evoparam.io import write_bids_derivatives
    from evoparam.reports import methods_text

    from evoparam import presets

    preset = presets.get(args.preset)
    ts, datatype = _read(args, preset)
    results, notes = run_fit(ts, preset, random_state=args.random_state, n_jobs=args.n_jobs,
                             existence=not args.no_existence)
    outcome.warnings += notes
    group = next(r for r in results if isinstance(r, GroupFit))
    _count(outcome, group)
    out = args.out
    written = write_bids_derivatives(results, args.input, subject=args.sub, datatype=datatype,
                                     out=out, overwrite=True, **_entities(args))
    container = os.path.join(out, f"sub-{args.sub.removeprefix('sub-')}",
                             f"{os.path.basename(written[0]).rsplit('_', 1)[0]}.evoparam")
    save_object(group, container, overwrite=True)
    outcome.outputs += written + [container]
    _log(args, methods_text(results))
    if outcome.n_units.get("failed"):
        outcome.code = EXIT_CODES["partial"]


def cmd_test(args, outcome):
    from evoparam import presets
    from evoparam.existence import test_response
    from evoparam.io import write_bids_derivatives
    from evoparam.prep import baseline_correct

    preset = presets.get(args.preset)
    ts, datatype = _read(args, preset)
    if preset.baseline is not None:
        ts = baseline_correct(ts, preset.baseline, check=False)
    test = test_response(ts, method=args.method, window=preset.fit_window,
                         baseline=preset.baseline or (ts.times[0], 0.0),
                         random_state=args.random_state, by=())
    outcome.outputs += write_bids_derivatives(test, args.input, subject=args.sub,
                                              datatype=datatype, out=args.out, overwrite=True,
                                              **_entities(args))


def cmd_clean(args, outcome):
    from evoparam.artifact import remove_artifact
    from evoparam.data.serialize import save_object
    from evoparam.io import write_bids_derivatives

    from evoparam import presets

    preset = presets.get(args.preset)
    ts, datatype = _read(args, preset)
    window = tuple(args.window) if args.window else preset.artifact_window
    result = remove_artifact(ts, method=args.method, window=window,
                             random_state=args.random_state)
    outcome.outputs += write_bids_derivatives(result, args.input, subject=args.sub,
                                              datatype=datatype, out=args.out, overwrite=True,
                                              **_entities(args))
    path = os.path.join(args.out, f"sub-{args.sub.removeprefix('sub-')}", "cleaned.evoparam")
    outcome.outputs.append(save_object(result.cleaned, path, overwrite=True))


def _containers(path):
    if path.endswith(".evoparam") or path.endswith(".zip"):
        return [path]
    found = []
    for folder, dirs, _ in os.walk(path):
        for d in list(dirs):
            if d.endswith(".evoparam"):
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
        base = os.path.join(args.out or os.path.dirname(container),
                            os.path.basename(container).rsplit(".", 1)[0] + "_report")
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
    kwargs = {"n_trials": args.n_trials, "seed": args.random_state}
    if args.n_channels:
        kwargs["n_channels"] = args.n_channels
    ts, truth = GENERATORS[args.modality](**kwargs)
    outcome.outputs.append(write_bids(ts, truth, args.out, subject=args.sub.removeprefix("sub-"),
                                      task=args.task or args.modality))


def cmd_detect_events(args, outcome):
    import fnmatch

    from evoparam.events import detect_events
    from evoparam.io import read_brainvision

    data, sfreq, names, *_ = read_brainvision(args.input)
    picks = [i for i, n in enumerate(names)
             if not args.channels or fnmatch.fnmatch(n, args.channels)]
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
        json.dump({c: {"Description": f"evoparam event column {c} "
                       "(evoparam.events.detect_events)",
                       **({"Units": "s"} if c in ("onset", "duration") or c.endswith("_time")
                          else {})} for c in events.columns}, fobj, indent=1)
    outcome.n_units["ok"] = int(len(events))
    outcome.outputs += [args.out, sidecar]


def cmd_reproduce(args, outcome):
    from evoparam.reports import reproduce

    report = reproduce(args.input, rtol=args.rtol)
    outcome.extra = report.to_dict()
    if not report.identical:
        outcome.code = EXIT_CODES["mismatch"]
        outcome.warnings += [f"{d['field']} differs (relative difference {d['rel_diff']})"
                             for d in report.differences]


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
    for name in ("pandas", "matplotlib", "mne", "jsonschema", "numpyro", "pymc", "pyarrow",
                 "edfio", "pynwb", "pylsl"):
        try:
            extras[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            extras[name] = None
    from evoparam.io.readers import READERS
    from evoparam.shapes import BACKGROUNDS, SHAPES

    plugins = [{"group": group, "name": name, "distribution": origin}
               for group, registry in (("evoparam.shapes", SHAPES),
                                       ("evoparam.shapes", BACKGROUNDS),
                                       ("evoparam.presets", presets.PRESETS),
                                       ("evoparam.readers", READERS))
               for name, origin in sorted(registry.plugins().items())]
    outcome.extra = {"evoparam": __version__, "python": sys.version.split()[0],
                     "numpy": metadata.version("numpy"), "scipy": metadata.version("scipy"),
                     "optional": extras, "presets": sorted(presets.PRESETS),
                     "plugins": plugins, "intended_use": RESEARCH_USE}
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
    rows = table.astype(object).where(table.notna(), None)          # JSON has no NaN
    outcome.extra = {"fields": rows.to_dict("records"),
                     "counts": {k: int(v) for k, v in table["status"].value_counts().items()}}
    if not args.json:
        print(format_stim_report(table))
    missing = table[table["status"] == "missing"]
    outcome.warnings += [f"missing: {r.events_file} {r.field}" for r in missing.itertuples()]
    if args.strict and len(missing):
        # the report is printed first; the error gives the exit code 5 (input error)
        raise ReaderError(f"{len(missing)} stimulation metadata field(s) missing "
                          f"(--strict).")


def cmd_later(step):
    def command(args, outcome):
        raise EvoparamError(f"'evoparam {args.command}' comes with build step {step} "
                            f"(not in evoparam {__version__}).")
    return command


def _parser():
    parser = argparse.ArgumentParser(prog="evoparam",
                                     description="Parameterise evoked neural responses.")
    parser.add_argument("--version", action="store_true",
                        help="print the version and the research-use statement")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="print one JSON summary on stdout")
    common.add_argument("--verbose", default="warning", choices=("warning", "info", "debug"))
    common.add_argument("--n-jobs", type=int, default=1)
    common.add_argument("--random-state", type=int, default=0)
    common.add_argument("--config", help="JSON file with option values")
    bids = argparse.ArgumentParser(add_help=False)
    bids.add_argument("input", help="BIDS root")
    bids.add_argument("--sub", required=True)
    bids.add_argument("--ses")
    bids.add_argument("--task")
    bids.add_argument("--run")
    bids.add_argument("--datatype", choices=("ieeg", "eeg"))
    bids.add_argument("--epoch", type=float, nargs=2, metavar=("TMIN", "TMAX"))
    bids.add_argument("--remote", action="store_true", help="read from OpenNeuro")
    bids.add_argument("--preset", required=True)
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
    p.add_argument("--sub", default="01")
    p.add_argument("--task")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_simulate)
    p = sub.add_parser("detect-events", parents=[common], help="stimulation events of a "
                       "continuous BrainVision recording")
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
    p = sub.add_parser("benchmark", parents=[common], help="benchmark suites (step S19)")
    p.add_argument("action", nargs="?", default="run")
    p.set_defaults(func=cmd_later("S19"))
    p = sub.add_parser("validate-bids", parents=[common], help="stimulation metadata "
                       "validator: which fields each run states (present, inferred, missing, "
                       "inconsistent)")
    p.add_argument("input")
    p.add_argument("--sub", default=None, help="one subject (default: all)")
    p.add_argument("--strict", action="store_true",
                   help="exit 5 when a field is missing")
    p.set_defaults(func=cmd_validate_bids)
    return parser


def _apply_config(parser, args, argv):
    if not getattr(args, "config", None):
        return args
    with open(args.config, encoding="utf-8") as fobj:
        config = json.load(fobj)
    given = {a.split("=")[0].lstrip("-").replace("-", "_") for a in argv if a.startswith("--")}
    for key, value in config.items():
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
        summary = {"command": args.command, "version": __version__, "status": status,
                   "exit_code": code, "outputs": [str(o) for o in outcome.outputs],
                   "n_units": outcome.n_units, "warnings": sorted(set(outcome.warnings)),
                   "duration_s": time.perf_counter() - start}
        if outcome.extra is not None:
            summary["result"] = outcome.extra
        print(json.dumps(summary, default=str, allow_nan=False))
    elif outcome.outputs:
        for path in outcome.outputs:
            print(path)
    return code


if __name__ == "__main__":
    sys.exit(main())
