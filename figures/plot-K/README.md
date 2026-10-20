# Plot K: how the N1 threshold k is chosen

Ground truth is needed to choose k, and real recordings do not provide it. These figures
build it from real data: for each of the 51 ds004080 patients (van Blooijs et al. 2023),
the 2 s **before** each pulse (no response, no artifact, real noise) are relabelled as a
-1..+1 s epoch. Half of the contacts (fixed seed) receive a known negative N1 (Gaussian,
latency 15-45 ms, width at half height 5-15 ms, peak 20-500 µV log-uniform, ±2 ms
jitter between trials); the other half stay pure noise. The published rule as written
(registry row `vanblooijs2023`, the gating reading, D8) is then applied on the trial average
for k = 1.0 to 10.0 (step 0.1).

- **K1**: % of injected N1s found (blue) and % of noise-only contacts wrongly flagged
  (orange) vs k; mean over patients, band = 95 % bootstrap CI over patients.
  At k = 3.4: 56.7 % found, 2.5 % invented. Best compromise (max found − invented):
  k = 2.7, 63.7 % found, 6.7 % invented. Invented ≤ 5 % from k = 3.0, ≤ 1 % from k = 4.2.
- **K2**: % of injected N1s found vs their size relative to the noise (peak / baseline SD).
  At k = 3.4 an N1 is found 81 % of the time at 4-6 SD and ~100 % above 6 SD.

The orange curve does not depend on the injected N1s (pure noise). The blue curve's level
depends on the assumed spread of N1 sizes; K2 gives the assumption-free view.

Rebuild (needs the ds004080 cache of `fetch_ds004080.py`):

    python benchmarks/plots/plot_k_compute.py
    python benchmarks/plots/plot_k_figures.py
