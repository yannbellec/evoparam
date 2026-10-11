# SPDX-License-Identifier: Apache-2.0
"""Public benchmark: tasks, frozen suites, baselines, adapters, scoring and leaderboard
(spec §8).

Seven tasks (:data:`TASKS`: T1 response detection, T2 early component latency and
amplitude, T3 component decomposition, T4 artifact removal, T5 single-trial estimation,
T6 ERP scoring, T7 time-frequency response) with versioned JSON Schemas
(``benchmark-task/<id>/1``); simulated suites ``core`` and ``full`` generated on the fly by
:mod:`evoparam.sim` from frozen seeds, with a hidden split committed by its SHA-256;
real-data suites as index files and loaders; baselines (the published threshold rules, the
cross-projection test, classical measures, template alignment, evoparam presets);
adapters that run external tools in a separate process; deterministic scoring with a
signed result; a Markdown report and a static leaderboard page generated from the results
only; and the local preparation of the deposit of the public split.

Command line: ``evoparam benchmark run | score | leaderboard | prepare-deposit``.

Examples
--------
>>> from evoparam import benchmark
>>> benchmark.load_suite("core").task_ids()
['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']
>>> benchmark.schema_id("T1")
'benchmark-task/T1/1'
"""

from .adapters import Adapter, AdapterError, load_adapter
from .baselines import BASELINES, RULES, Baseline, baselines_for, get_baseline
from .deposit import prepare_deposit, write_trials_nwb
from .leaderboard import leaderboard_html, write_leaderboard
from .realdata import list_indices, load_index, real_inputs
from .report import load_results, markdown, write_report
from .runner import run_suite
from .scoring import HEADLINE, read_result, score, score_outputs, verify_result, write_result
from .suite import (Cell, Suite, hidden_commitment, list_suites, load_hidden_seeds, load_suite,
                    make_inputs, simulate_cell)
from .tasks import (TASKS, BenchmarkValidationError, TaskInput, load_schema, read_input,
                    read_output, schema_id, validate, write_input, write_output)

__all__ = ["TASKS", "TaskInput", "BenchmarkValidationError", "load_schema", "schema_id",
           "validate", "read_input", "write_input", "read_output", "write_output",
           "Suite", "Cell", "load_suite", "list_suites", "make_inputs", "simulate_cell",
           "hidden_commitment", "load_hidden_seeds", "score", "score_outputs", "write_result",
           "read_result", "verify_result", "HEADLINE", "Baseline", "BASELINES", "RULES",
           "get_baseline", "baselines_for", "Adapter", "AdapterError", "load_adapter",
           "run_suite", "load_results", "markdown", "write_report", "leaderboard_html",
           "write_leaderboard", "prepare_deposit", "write_trials_nwb", "list_indices",
           "load_index", "real_inputs"]
