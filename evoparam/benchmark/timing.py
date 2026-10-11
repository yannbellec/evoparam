# SPDX-License-Identifier: Apache-2.0
"""Performance workloads P1–P6 of the specification (§9.1, REQ-290, REQ-299).

``run_suite("perf")`` runs the six workloads at the sizes of the specification and is the
release measurement on the reference machine (``benchmarks/README.md``); ``run_suite("ci")``
runs the same workloads at sizes a CI runner finishes in a few minutes and is compared with
``benchmarks/baseline-ci.json`` by the ``perf`` workflow. Fixtures are generated
deterministically with :mod:`evoparam.sim` (``random_state=0``).

==== ======================================================== ==================================
id   workload (``perf``)                                      target
==== ======================================================== ==================================
P1   one averaged waveform, 2 048 Hz, 1 s fit window,         median < 50 ms (20 repetitions after
     ``max_components=6``, preset ``ccep`` (3 starts)         one warm-up call)
P2   ``fit_trials``, 100 channels x 50 trials, ``n_jobs=4``   < 60 s (one run)
P3   ``evoparam.fit`` of 500 x 256 x 4 096 float64 trials     peak RSS increase < 2 x data size;
     (4.19 GB), and of the same set memory-mapped             mapped: < 0.25 x data + 512 MB
P4   GPU backend against the CPU on P1 and P2                 equal within 1e-6 relative
P5   ``evoparam.reproduce`` of a saved 256-channel GroupFit   < 1.5 x the original fit
P6   ``python -X importtime -c "import evoparam"``            median of 10 < 300 ms
==== ======================================================== ==================================

Each result is ``{id, workload, median_s, p90_s, peak_rss_bytes, n_repeats, target,
target_met, environment, ...}``; ``peak_rss_bytes`` is the increase of the resident set
size during the call, sampled every 50 ms from ``/proc/self/statm`` (``resource.getrusage``
where ``/proc`` is missing).
"""

import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
import warnings

import numpy as np

__all__ = ["SUITES", "run_suite", "compare", "write_results", "results_path"]

SCHEMA = "evoparam-perf/1"
SUITES = {
    "perf": {"P1": {"repeats": 20}, "P2": {"n_channels": 100, "n_trials": 50, "n_jobs": 4},
             "P3": {"n_trials": 500, "n_channels": 256, "n_times": 4096, "mmap": True},
             "P4": {}, "P5": {"n_channels": 256}, "P6": {"repeats": 10}},
    "ci": {"P1": {"repeats": 5}, "P2": {"n_channels": 8, "n_trials": 50, "n_jobs": 1},
           "P3": {"n_trials": 50, "n_channels": 64, "n_times": 4096, "mmap": True},
           "P4": {}, "P5": {"n_channels": 16}, "P6": {"repeats": 5}},
}
TARGETS = {"P1": "median < 0.050 s", "P2": "< 60 s on 4 cores",
           "P3": "peak RSS increase < 2 x data size; memory-mapped < 0.25 x data size + 512 MB",
           "P4": "GPU equals CPU within 1e-6 relative", "P5": "< 1.5 x the original fit",
           "P6": "median < 0.300 s"}
SFREQ = 2048.0


###################################################################################################
## Measurement helpers

def _rss():
    try:
        with open("/proc/self/statm", encoding="ascii") as fobj:
            return int(fobj.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, AttributeError):
        import resource

        scale = 1 if sys.platform == "darwin" else 1024
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale


class _PeakRSS:
    """Peak resident set size above the level at entry, sampled every 50 ms."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.start = self.peak = 0
        self._stop = threading.Event()

    def __enter__(self):
        self.start = self.peak = _rss()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self):
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, _rss())

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, _rss())

    @property
    def increase(self):
        return int(self.peak - self.start)


def _timed(func, repeats, warmup=True):
    if warmup:
        func()
    durations = []
    with _PeakRSS() as rss:
        for _ in range(repeats):
            start = time.perf_counter()
            func()
            durations.append(time.perf_counter() - start)
    return durations, rss.increase


def _row(id_, workload, durations, peak, **extra):
    durations = np.asarray(durations, dtype=float)
    return {"id": id_, "workload": workload, "median_s": float(np.median(durations)),
            "p90_s": float(np.percentile(durations, 90)), "peak_rss_bytes": int(peak),
            "n_repeats": int(durations.size), "target": TARGETS[id_], **extra}


def environment():
    """Machine and library versions of a run."""

    from evoparam.data.provenance import capture_environment
    from evoparam.version import __version__

    env = capture_environment()
    commit = env.get("git_commit")
    if not commit:
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True, cwd=os.path.dirname(os.path.abspath(__file__)),
                                    timeout=10).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            commit = None
    return {"evoparam": __version__, "python": platform.python_version(),
            "platform": platform.platform(), "machine": platform.machine(),
            "cpu": env.get("cpu"), "cpu_count": os.cpu_count(), "numpy": np.__version__,
            "scipy": env.get("scipy"), "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "git_commit": commit,
            "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None}


###################################################################################################
## Fixtures (evoparam.sim, random_state=0)

def _ccep(n_trials, n_channels, tmin=-0.5, tmax=1.1, seed=0):
    from evoparam import sim

    trials, _ = sim.simulate_ccep(n_trials=n_trials, n_channels=max(n_channels, 3), seed=seed,
                                  artifact=False, p_response=1.0, tmin=tmin, tmax=tmax,
                                  sfreq=SFREQ, layout="grid")
    if n_channels < trials.data.shape[1]:
        trials = trials.select(picks=trials.ch_names[:n_channels])
    return trials


###################################################################################################
## The six workloads

def p1(repeats=20):
    """One averaged waveform of 2 048 samples (1 s fit window), ccep preset."""

    from evoparam.models import ResponseModel

    trials = _ccep(10, 3)
    waveform = trials.data[:, 0].mean(axis=0)
    model = ResponseModel("ccep", max_components=6, fit_window=(0.009, 0.009 + 2047 / SFREQ),
                          random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        durations, peak = _timed(lambda: model.fit_waveform(trials.times, waveform), repeats)
    median = float(np.median(durations))
    return _row("P1", "1 waveform, 2048 samples, max_components=6, ccep, 3 starts",
                durations, peak, target_met=median < 0.050)


def p2(n_channels=100, n_trials=50, n_jobs=4):
    """Trial-by-trial fits with the channel template as warm start."""

    from evoparam import existence  # not the package root (import graph, §4.2)
    from evoparam.trials import fit_trials

    trials = _ccep(n_trials, n_channels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = existence.fit(trials, preset="ccep", existence=False, random_state=0)
        with _PeakRSS() as rss:
            start = time.perf_counter()
            fit_trials(trials, group, n_jobs=n_jobs, random_state=0)
            elapsed = time.perf_counter() - start
    full = n_channels == 100 and n_trials == 50
    return _row("P2", f"fit_trials, {n_channels} channels x {n_trials} trials, n_jobs={n_jobs}",
                [elapsed], rss.increase, n_fits=n_channels * n_trials,
                target_met=(elapsed < 60.0) if full else None)


def _p3_data(n_trials, n_channels, n_times, out=None):
    """Deterministic trials: the simulated responses of 16 channels, tiled, plus noise."""

    base = _ccep(n_trials, 16, tmin=-1.0, tmax=-1.0 + (n_times + 1) / SFREQ)
    data = out if out is not None else np.empty((n_trials, n_channels, n_times))
    rng = np.random.default_rng(0)
    for start in range(0, n_channels, 16):
        stop = min(start + 16, n_channels)
        data[:, start:stop] = base.data[:, :stop - start, :n_times] + \
            rng.normal(0.0, 5.0, (n_trials, stop - start, n_times))
    return data, base.times[:n_times]


def p3(n_trials=500, n_channels=256, n_times=4096, mmap=True):
    """Peak memory of ``evoparam.fit`` on a large TrialSet, in memory and memory-mapped."""

    import inspect

    from evoparam import existence  # not the package root (import graph, §4.2)
    from evoparam.data import TrialSet
    from evoparam.models import fit as models_fit

    chunk = "chunk" in inspect.signature(models_fit).parameters
    kwargs = {"chunk": "auto"} if chunk else {}
    note = None if chunk else ("evoparam.fit has no chunk argument (spec §9.1 chunking "
                               "rule); measured without it")
    rows = []
    data, times = _p3_data(n_trials, n_channels, n_times)
    nbytes = data.nbytes
    ts = TrialSet(data=data, times=times, sfreq=SFREQ,
                  ch_names=[f"C{i:03d}" for i in range(n_channels)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with _PeakRSS() as rss:
            start = time.perf_counter()
            existence.fit(ts, preset="ccep", random_state=0, **kwargs)
            elapsed = time.perf_counter() - start
    rows.append(_row("P3", f"evoparam.fit, {n_trials} x {n_channels} x {n_times} float64 "
                     f"({nbytes / 1e9:.2f} GB) in memory", [elapsed], rss.increase,
                     data_bytes=int(nbytes), bound_bytes=int(2 * nbytes),
                     target_met=rss.increase < 2 * nbytes, note=note))
    del ts, data
    if mmap:
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "p3.npy")
            mapped = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64,
                                               shape=(n_trials, n_channels, n_times))
            _p3_data(n_trials, n_channels, n_times, out=mapped)
            mapped.flush()
            del mapped
            saved = os.path.join(folder, "p3.evoparam")
            data = np.load(path, mmap_mode="r")
            TrialSet(data=data, times=times, sfreq=SFREQ,
                     ch_names=[f"C{i:03d}" for i in range(n_channels)]).save(saved)
            del data
            ts = TrialSet.load(saved, mmap=True)
            bound = 0.25 * nbytes + 512 * 2 ** 20
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with _PeakRSS() as rss:
                    start = time.perf_counter()
                    existence.fit(ts, preset="ccep", random_state=0, **kwargs)
                    elapsed = time.perf_counter() - start
            del ts
        rows.append(_row("P3", "evoparam.fit, the same set memory-mapped "
                         "(TrialSet.load(path, mmap=True))", [elapsed], rss.increase,
                         variant="mmap", data_bytes=int(nbytes), bound_bytes=int(bound),
                         target_met=rss.increase < bound, note=note))
    return rows


def p4():
    """GPU against CPU: the GPU backends of §9.1 are not in this version."""

    return {"id": "P4", "workload": "GPU backend against the CPU on P1 and P2",
            "median_s": None, "p90_s": None, "peak_rss_bytes": None, "n_repeats": 0,
            "target": TARGETS["P4"], "target_met": None, "status": "not_available",
            "note": "no GPU backend in this version (the fitting engine is NumPy/SciPy; "
                    "Provenance.environment['backend'] is 'numpy')"}


def p5(n_channels=256):
    """``evoparam.reproduce`` of a saved GroupFit against the time of the original fit."""

    from evoparam import existence, reports  # not the package root (import graph, §4.2)

    trials = _ccep(10, n_channels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.perf_counter()
        group = existence.fit(trials, preset="ccep", random_state=0)
        original = time.perf_counter() - start
        with tempfile.TemporaryDirectory() as folder:
            path = group.save(os.path.join(folder, "group"))
            with _PeakRSS() as rss:
                start = time.perf_counter()
                report = reports.reproduce(path, data=trials)
                elapsed = time.perf_counter() - start
    ratio = elapsed / original
    return _row("P5", f"evoparam.reproduce of a saved GroupFit of {n_channels} channels",
                [elapsed], rss.increase, original_s=original, ratio=ratio,
                identical=bool(report.identical), target_met=ratio < 1.5)


_IMPORTTIME = re.compile(r"import time:\s+\d+\s+\|\s+(\d+)\s+\|\s+evoparam$")


def p6(repeats=10):
    """Import time of the package (``python -X importtime``), median of ``repeats``."""

    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [root, env.get("PYTHONPATH")]))
    durations, wall = [], []
    for _ in range(repeats + 1):                     # the first one fills the caches
        start = time.perf_counter()
        done = subprocess.run([sys.executable, "-X", "importtime", "-c", "import evoparam"],
                              capture_output=True, text=True, env=env, check=True)
        wall.append(time.perf_counter() - start)
        found = [int(m.group(1)) for line in done.stderr.splitlines()
                 if (m := _IMPORTTIME.search(line.strip()))]
        durations.append(found[-1] / 1e6 if found else np.nan)
    durations, wall = durations[1:], wall[1:]
    if not np.isfinite(durations).any():             # no importtime line: the wall time
        durations = wall
    median = float(np.nanmedian(durations))
    loaded = subprocess.run([sys.executable, "-c", "import sys, evoparam; "
                             "print(','.join(sorted(m for m in ('pandas', 'matplotlib', 'mne',"
                             " 'pynwb', 'h5py', 'numba', 'jax', 'sklearn') if m in "
                             "sys.modules)))"],
                            capture_output=True, text=True, env=env, check=True).stdout.strip()
    return _row("P6", "python -X importtime -c 'import evoparam' (cumulative time of "
                "evoparam)", durations, 0, process_wall_median_s=float(np.median(wall)),
                optional_modules_loaded=loaded.split(",") if loaded else [],
                target_met=median < 0.300)


WORKLOADS = {"P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5, "P6": p6}


###################################################################################################
## Suite, comparison, files

def run_suite(suite="perf", *, only=None, repeat_runs=1, log=None):
    """Run the workloads of ``suite`` (``"perf"`` or ``"ci"``).

    Parameters
    ----------
    suite : {"perf", "ci"}
    only : sequence of str, optional
        Workload ids to run (default: all six).
    repeat_runs : int
        Runs of the whole suite; each figure is the median over the runs (the CI baseline
        is written from three).
    log : callable, optional
        Called with one line per workload.

    Returns
    -------
    dict
        ``{"schema", "suite", "environment", "results": [...]}``.
    """

    from evoparam.errors import SettingsError

    if suite not in SUITES:
        raise SettingsError(f"Unknown performance suite {suite!r}: {sorted(SUITES)}.")
    ids = [i for i in SUITES[suite] if only is None or i in only]
    runs = []
    for _ in range(max(1, int(repeat_runs))):
        rows = []
        for id_ in ids:
            out = WORKLOADS[id_](**SUITES[suite][id_])
            for row in out if isinstance(out, list) else [out]:
                rows.append(row)
                if log:
                    log(_line(row))
        runs.append(rows)
    results = runs[0] if len(runs) == 1 else _median_of_runs(runs)
    env = environment()
    for row in results:
        row["environment"] = env
    return {"schema": SCHEMA, "suite": suite, "repeat_runs": int(repeat_runs),
            "environment": env, "results": results}


def _key(row):
    return row["id"] + ("/" + row["variant"] if row.get("variant") else "")


def _median_of_runs(runs):
    out = []
    for rows in zip(*runs, strict=True):
        row = dict(rows[0])
        for field in ("median_s", "p90_s", "peak_rss_bytes"):
            values = [r[field] for r in rows if r[field] is not None]
            if values:
                row[field] = float(np.median(values)) if field != "peak_rss_bytes" else \
                    int(np.median(values))
        row["n_runs"] = len(rows)
        out.append(row)
    return out


def _line(row):
    if row.get("median_s") is None:
        return f"{_key(row)}: {row.get('status', 'not run')} ({row.get('note', '')})"
    met = {True: "met", False: "NOT met", None: "not checked at this size"}[row["target_met"]]
    return (f"{_key(row)}: median {row['median_s']:.4f} s, p90 {row['p90_s']:.4f} s, peak RSS "
            f"+{row['peak_rss_bytes'] / 2 ** 20:.0f} MiB ({row['n_repeats']} runs); target "
            f"{row['target']}: {met}")


def compare(document, baseline, *, time_tolerance=1.25, memory_tolerance=1.10,
            warn_above=1.10):
    """Compare a run with a baseline run of the same suite.

    Returns ``(failures, warnings)``: lists of messages naming the workload and the ratio. A
    median time above ``time_tolerance`` x its baseline fails, between ``warn_above`` and
    ``time_tolerance`` warns; a peak memory above ``memory_tolerance`` x its baseline fails,
    and so does a P3 figure above its absolute bound.
    """

    base = {_key(r): r for r in baseline["results"]}
    failures, notes = [], []
    for row in document["results"]:
        key = _key(row)
        if row["id"] == "P3" and row.get("target_met") is False:
            failures.append(f"{key}: peak RSS increase {row['peak_rss_bytes'] / 2 ** 20:.0f} "
                            f"MiB above its bound {row['bound_bytes'] / 2 ** 20:.0f} MiB")
        ref = base.get(key)
        if ref is None or row.get("median_s") is None or not ref.get("median_s"):
            continue
        ratio = row["median_s"] / ref["median_s"]
        if ratio > time_tolerance:
            failures.append(f"{key} ({row['workload']}): median {row['median_s']:.4f} s is "
                            f"{ratio:.2f} x the baseline {ref['median_s']:.4f} s (tolerance "
                            f"{time_tolerance:.2f} x)")
        elif ratio > warn_above:
            notes.append(f"{key}: median {ratio:.2f} x the baseline (warning above "
                         f"{warn_above:.2f} x)")
        if ref.get("peak_rss_bytes") and row.get("peak_rss_bytes") and \
                ref["peak_rss_bytes"] > 64 * 2 ** 20:
            mem = row["peak_rss_bytes"] / ref["peak_rss_bytes"]
            if mem > memory_tolerance:
                failures.append(f"{key}: peak memory {mem:.2f} x the baseline (tolerance "
                                f"{memory_tolerance:.2f} x)")
    return failures, notes


def results_path(document, folder):
    """``<folder>/<platform>-<python>-<commit>.json`` (spec §9.1); a run of the ``ci``
    suite ends in ``-ci.json``."""

    env = document["environment"]
    commit = (env.get("git_commit") or "nocommit")[:12]
    suffix = "" if document.get("suite", "perf") == "perf" else "-" + document["suite"]
    name = f"{sys.platform}-{env['python']}-{commit}{suffix}.json"
    return os.path.join(folder, name)


def write_results(document, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fobj:
        json.dump(document, fobj, indent=1, allow_nan=False, default=_jsonable)
        fobj.write("\n")
    return path


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")
