# Benchmark adapters (GPL, outside the evoparam package)

Adapters run an external tool on the task files of the evoparam benchmark (spec §8.3
item 6, REQ-354). evoparam starts each adapter as a separate process
(`evoparam.benchmark.Adapter`, through `subprocess`): the only exchange is files, the task
input (`<cell>.json` and `<cell>.npz`) and the task output (`<cell>.tsv`). Nothing here
is imported by evoparam, and this folder is not shipped with it (`MANIFEST.in`:
`prune adapters`). Licence: GPL-3.0-or-later, see `LICENSE`.

| file | tool | licence of the tool | tasks |
|---|---|---|---|
| `erdetect.json`, `erdetect_adapter.py` | ER-detect (van Blooijs et al. 2023), Python | GPL-3.0 | T1, T2 |
| `crp.json`, `crp_adapter.py`, `crp_wrapper.m` | canonical response parameterization (Miller et al. 2023), MATLAB | GPL | T1, T2 |
| `_taskio.py` | reader of task inputs and writer of TSV outputs, NumPy only | (this folder) | all |

Run an adapter on the `core` suite:

    evoparam benchmark run --suite core --tasks T1,T2 --methods "" \
        --adapter adapters/erdetect.json --out results-erdetect

or on one task input, by hand:

    python adapters/erdetect_adapter.py inputs/T1/ccep_detection-p000.json out.tsv

The tools are not installed with evoparam: install ER-detect in the environment that runs
the adapter (`pip install erdetect`), and MATLAB (or GNU Octave) with the CRP scripts on
its path for the CRP adapter.

**Status: not verified against the real tools.** Neither tool is installed on the
machine where these scripts were written. The calls (`erdetect.core.detection.ieeg_detect_er`
and its arguments; `CRP_method` and the fields of its output) follow the tools'
documentation as read when writing, and must be checked against an installed copy before
any result of these adapters is published. The input reading and the output writing are
tested (`evoparam/tests/benchmark/test_adapters.py` runs `_taskio.py` through a fake tool).

Each specification file (`*.json`) gives the command, the tasks, the version of the tool
and its citation; `{python}` is the running interpreter and `{here}` this folder. Change
`command` to run the tool in another environment (a container, another interpreter).
