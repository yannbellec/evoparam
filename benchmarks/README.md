# Benchmarks

## Reference machine

The performance targets of the specification (§9.1, REQ-290; REQ-324 and REQ-347 for the
online mode) are stated "on the reference machine". It is:

| | |
|---|---|
| CPU | Intel Xeon Processor @ 2.80 GHz (family 6, model 85, stepping 7; AVX-512), KVM virtual machine |
| cores | 4 (1 thread per core, 1 socket) |
| RAM | 15 GB |
| OS | Ubuntu 24.04.5 LTS, Linux 6.18 |
| Python | 3.13.16 (GCC 13.3.0) |
| NumPy | 2.5.3, BLAS and LAPACK: scipy-openblas 0.3.34 (OpenBLAS, DYNAMIC_ARCH, Haswell kernels, 64-bit integers) |
| threads | `OMP_NUM_THREADS=1` (one core) |

CI machines are about twice as slow: the timed tests run only with
`EVOPARAM_PERF_TESTS=1`. The scripts: `bench_artifact.py` (REQ-068), `bench_events.py`
(REQ-045), `bench_tf.py` (REQ-218) and `stream_latency.py` (REQ-324, REQ-347); each
docstring gives its target and the test that checks it.

## Performance targets P1–P6: `run_benchmarks.py`

```
OMP_NUM_THREADS=1 python benchmarks/run_benchmarks.py --suite perf
python benchmarks/run_benchmarks.py --suite ci --compare benchmarks/baseline-ci.json
```

(also `evoparam benchmark run --suite perf`). The workloads of spec §9.1 are generated with
`evoparam.sim` (`random_state=0`). The results go to
`results/<platform>-<python>-<commit>.json`, or `...-ci.json` for the `ci` suite, with
`{id, median_s, p90_s, peak_rss_bytes, n_repeats, environment}` per workload. Peak memory is
the increase of the resident set during the call, sampled every 50 ms.

Measured on the reference machine, one thread, load average 9.7 on the 4 cores
(`results/linux-3.13.16-bddedc9aa35f.json`):

| id | workload | measured | target | met |
|---|---|---|---|---|
| P1 | one waveform, 2 048 Hz, `max_components=6`, `ccep` | median 1.30 s (20 runs) | < 50 ms | no |
| P2 | `fit_trials`, 100 channels x 50 trials, `n_jobs=4` | 53.7 s | < 60 s | yes |
| P3 | `evoparam.fit`, in memory and memory-mapped | ci size only, see below | < 2x data; mapped < 0.25x + 512 MB | in memory no, mapped yes |
| P4 | GPU backend against the CPU | no GPU backend in this version | 1e-6 relative | not available |
| P5 | `evoparam.reproduce`, 256 channels | 266.5 s for a 270.7 s fit (0.98x), identical | < 1.5x | yes |
| P6 | `import evoparam` | median 0.909 s (10 runs) | < 300 ms | no |

P3 was run at the `ci` size (50 x 64 x 4 096 float64, 105 MB): in memory, peak RSS
+331 MiB (347 MB, 3.3x the data; bound 2x = 210 MB); memory-mapped, the same +331 MiB against
its bound of 563 MB (0.25 x data + 512 MiB); medians of the three runs of `baseline-ci.json`.
The full size (4.19 GB) was not run: at 3.3x it would exceed the machine's 15 GB.
`evoparam/tests/benchmark/test_timing.py` reads these files: the unmet targets are strict
expected failures that give the measured values.

`baseline-ci.json` is the median of three runs of the `ci` suite
(`--repeat-runs 3 --write-baseline`). This one was written on the reference machine, so it
holds until the `perf` workflow's `baseline-update` job (label `perf-baseline-update`)
regenerates it on the CI runner. Its medians: P1 1.20 s, P2 (8 channels x 50
trials, one job) 2.68 s, P3 55.7 s in memory and 53.8 s memory-mapped, P5 29.4 s (1.04x
its fit), P6 0.904 s. The `ci` run of P3 in memory is above its bound (3.3x the data), so
the `perf` job fails on it until the memory use of `evoparam.fit` is reduced.

## Online mode: `stream_latency.py`

```
OMP_NUM_THREADS=1 python benchmarks/stream_latency.py
```

1 000 simulated pulses, one per second, 64 channels at 2 048 Hz (stimulation artifact on
every channel, a response on half of them) pushed through `evoparam.stream.StreamFitter`:
threshold detection of the pulses, a 0.5 s response window (0 to 0.5 s, the artifact
window -2 to 8 ms interpolated), `max_shift` 5 ms, the running existence test with a look
after every pulse. The latency of a pulse is the wall time of the `push` call that
delivers the last sample of its window (detection, fit, existence test and the estimates
included). Budget (REQ-324): p99 < 20 ms.

Measured on the reference machine with other jobs running (load average 4 to 7 on the 4
cores), times in ms:

| template | mode | chunk | p50 | p95 | p99 | max | compute p99 |
|---|---|---|---|---|---|---|---|
| N1 | default (`shift_scale`) | 1 | 5.31 | 7.54 | 10.41 | 20.65 | 9.91 |
| N1 | default (`shift_scale`) | 32 | 4.75 | 6.58 | 8.56 | 11.28 | 8.09 |
| N1 + N2 | `common` (spec §10.6) | 32 | 5.70 | 7.49 | 10.73 | 18.41 | 10.10 |
| N1 + N2 | `scale` | 32 | 6.25 | 7.74 | 10.26 | 14.69 | 9.72 |
| N1 + N2 | default (`shift_scale`) | 32 | 7.49 | 10.01 | 11.54 | 17.91 | 10.91 |

With a single component the three modes are the same model. "compute" is
`StreamFitter.latency_stats` (the estimation of a pulse, without the rest of the `push`
call). `evoparam/tests/stream/test_stream.py::test_req347_latency_budget` checks the p99
of the default mode with one and two components when `EVOPARAM_PERF_TESTS=1`.
