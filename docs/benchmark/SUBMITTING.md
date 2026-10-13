# Submitting a result to the evoparam benchmark

Spec §8.4, REQ-392. The leaderboard (`leaderboard.html`, `report-core.md`) is generated from
the signed result files of `docs/benchmark/results/` only (REQ-357). Any method, written in
any language, can enter it.

## 1. Get the task inputs

```
evoparam benchmark run --suite core --tasks T1,T2 --methods "" --out work
```

writes, for every cell of the public split, the input `work/inputs/<task>/<cell>.json` (the
descriptor: sampling rate, first time, channel names, hashes and the task-specific fields)
and its arrays `<cell>.npz` (`data`: trials x channels x times; `evoked` for T3; `average`
for T6). Nothing else is needed: the truth stays inside the suite.

## 2. Write your predictions

For each cell, one file in a predictions folder:

- T1, T2, T3, T5, T6, T7: `<cell>.tsv`, the columns of the task schema
  (`evoparam/benchmark/schemas/<task>.json`, `benchmark-task/<task>/1`), required columns
  first, `n/a` for a missing value;
- T4: `<cell>.npz` holding `data` (the cleaned epochs).

Every file is validated against its schema when it is read; the first violation is reported.

## 3. Score and sign

```
evoparam benchmark score --task T1 --predictions my_predictions --suite core \
    --method-name my-method --method-version 1.0 --citation "Author et al. 2027"
```

writes `my_predictions/my-method__T1.json`: the metrics, the headline numbers, the SHA-256
of the inputs and of your outputs, and a signature (SHA-256 of the canonical content). A
result edited by hand no longer matches its signature and is refused.

## 4. Open a pull request

Add the result file to `docs/benchmark/results/`, with the code or the container that
produced the predictions (or a link to them), and run

```
python tools/benchmark_leaderboard.py
```

to regenerate the page and the report. The CI job `benchmark-results` checks every
signature and that the page and the report are what the results give
(`python tools/benchmark_leaderboard.py --check`).

## Reproducing an entry

Every entry of evoparam's own methods (the baselines, the presets) is reproduced from a
tagged version by one command, for example

```
evoparam benchmark run --suite core --task T1 --method vanblooijs2023 --out check
```

whose `check/results/vanblooijs2023__T1.json` has the scores, the output hashes and the
signature of `docs/benchmark/results/vanblooijs2023__T1.json`.

The public archive of the suites (NWB/BIDS files with a DOI) and the public leaderboard
come with the first public release, after the institutional decision recorded in
`GOVERNANCE.md` (spec REQ-355, REQ-398).
