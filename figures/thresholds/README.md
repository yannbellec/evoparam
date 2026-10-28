# Threshold figures: how each central setting of the ccep preset is justified

Data: ds004080 (van Blooijs et al. 2023), the 51 patients with a 2048 Hz SPESclin run and a
stimulated pair of at least 10 trials (`benchmarks/plots/ds004080_subjects.tsv`). The
patient is the statistical unit: one value per patient, thin grey line per patient, thick
line = mean over patients, band = 95 % bootstrap CI over patients (5000 resamples).

**Known truth** (T3-T8): the -2..0 s part of every trial (before the pulse: real noise, no
response, no artifact) becomes a -1..+1 s epoch. Half of the contacts (random) receive a
negative Gaussian N1 (latency 15-45 ms, FWHM 5-15 ms, peak 20-500 µV log-uniform) and N2
(80-200 ms, FWHM 40-100 ms, 20-300 µV), ±2 ms latency jitter per trial; the other half stay
pure noise (`common.inject`). "Found" = evoparam (ccep preset) returns a bump labelled N1
(9-50 ms) or N2 (50-300 ms) on an injected contact; "invented" = it returns one on a
noise-only contact.

Noise SD: standard deviation of the trial average over the baseline (-500..-20 ms unless
stated), in µV. Trial average: per-trial median of the baseline removed, then mean.

| Figure | Setting (ccep value) | x | y | Result |
| --- | --- | --- | --- | --- |
| T1a | end of the masked artifact window (9 ms) | time after the pulse (ms) | median over contacts of \|average\| / noise SD | the artifact peaks at ~100 × noise at 2 ms, falls under 3.4 × noise at 6 ms; a second small bump at 8 ms; flat from 9-10 ms |
| T1b | same | start of the N1 search window (ms) | % contacts with an N1 (published rule, k = 3.4) and % with the N1 in the first 1 ms of the window | edge "N1s" peak at 8 ms (3.3 %), 2.4 % at 9 ms, 0.8 % at 10 ms: 10 ms would be safer for the rule on raw data |
| T2a | end of the baseline (-20 ms) | baseline end (ms) | SD(-500..end) / SD(-500..-100) | +1.2 % at -20 ms, +2.6 % at 0 ms: little leakage before the pulse |
| T2b | baseline length (480 ms) | length (ms, ending at -20 ms) | % N1 decisions equal to those with 980 ms | 79 % at 50 ms, 96 % at 480 ms, 98 % at 700 ms |
| T3 | reference (adaptive CAR) | none / CAR / adaptive CAR | N1 and N2 found, invented, amplitude found / true | adaptive CAR finds the most N1s (65 % vs 61 %); plain CAR shrinks N1 to 87 % and N2 to 95 % of their true size and loses N2s (39 % vs 44-45 %) |
| T4a | largest background τ (50 ms) | fitted τ (ms), real data, τ allowed up to 500 ms | contacts | 32 % of contacts at the 0.5 ms lower bound (no tail left after 9 ms); median 3 ms; 15 % above 50 ms; 2.4 % at the 500 ms bound |
| T4b-c | same | τ_max allowed (ms) | N1 / N2 found and invented (known truth + exponential tails with the τ of T4a) | flat: N1 62 %, N2 45 % from 10 to 500 ms; N2 invented 2.7 % at 50 ms vs 2.2 % at 500 ms. With exact exponential tails τ_max barely matters; its role is with real tails |
| T5 | ΔBIC to keep a bump (10) | ΔBIC | N1 / N2 found and invented; bumps per noise-only contact | from 10 to 2: N1 found 62.5 → 65 %, N2 44.5 → 51 %, N2 invented 2.1 → 3.6 %; bumps on pure noise 0.08 → 0.14 per contact |
| T6 | response test (α = 0.05, FDR) | α | % responding contacts found; % noise-only contacts flagged | at 0.05: FDR 48 % found / 5.7 % flagged; Holm 3.7 % / 0.1 %; none 66 % / 14.6 %. With 10 trials a sign-flip test cannot give p < 1/1024, so Holm almost never passes |
| T7 | number of trials (≥ 5 for the test) | trials averaged | N1 / N2 found and invented; median \|latency error\| | N1 found 44 % (3 trials), 50 % (5), 61 % (10); N2 latency error 8.7 ms (3), 3.7 ms (5), 2.4 ms (10) |
| T8 | onset at 10 % of the peak | % of the peak | median \|fitted onset - true onset\| (ms) | N1 1.0 ms at 10 %, 0.64 ms at 50 %; N2 5.3 ms at 10 %, 3.9 ms at 50 % |

Rebuild (the ds004080 cache is filled by `benchmarks/plots/fetch_ds004080.py`):

    cd benchmarks/plots/thresholds
    python t1_artifact_window.py   # same for t2 ... t8; each writes its CSV, then the figure
