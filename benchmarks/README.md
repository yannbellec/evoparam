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
