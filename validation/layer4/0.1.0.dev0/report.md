# Layer-4 robustness, evoparam 0.1.0.dev0

2026-10-11; Linux-6.18.44-fc-v114-x86_64-with-glibc2.39, 4 CPUs; 77.8 min with 2 processes. Analyses: montage, preset, sampling, trials, platform.

## 1. Montage (existence test under three references)

| pair | runs/sites | median kappa | lowest kappa | median latency diff (ms) |
|---|---|---|---|---|
| bipolar-car | 11 | 0.23 | 0.00 | 0.00 |
| bipolar-car_adaptive | 11 | 0.38 | 0.00 | 0.21 |
| car-car_adaptive | 11 | 0.76 | 0.04 | 0.00 |

## 2. Preset choice fields (x0.5, x1.5)

| preset | field | factor | recall | change | median latency error (ms) | change (ms) | sensitive |
|---|---|---|---|---|---|---|---|
| ccep | (as set) | 1.0 | 1.000 | +0.000 | 0.417 | +0.000 |  |
| ccep | artifact_window | 1.0 | 1.000 | +0.000 | 0.058 | +0.000 |  |
| ccep | artifact_window | 0.5 | 1.000 | +0.000 | 0.058 | +0.000 |  |
| ccep | artifact_window | 1.5 | 0.985 | -0.015 | 0.148 | +0.090 |  |
| ccep_seeg | (as set) | 1.0 | 1.000 | +0.000 | 0.417 | +0.000 |  |
| ccep_seeg | artifact_window | 1.0 | 1.000 | +0.000 | 0.058 | +0.000 |  |
| ccep_seeg | artifact_window | 0.5 | 1.000 | +0.000 | 0.058 | +0.000 |  |
| ccep_seeg | artifact_window | 1.5 | 0.985 | -0.015 | 0.228 | +0.170 |  |
| ecap | (as set) | 1.0 | 0.875 | +0.000 | 0.024 | +0.000 |  |
| ecap | artifact_window | 1.0 | 1.000 | +0.000 | 0.002 | +0.000 |  |
| ecap | component_labels | 1.0 | 0.875 | +0.000 | 0.024 | +0.000 |  |
| ecap | epoch | 0.5 | 0.000 | -0.875 |  |  | yes |
| ecap | epoch | 1.5 | 0.950 | +0.075 | 0.021 | -0.004 |  |
| ecap | baseline | 0.5 | 0.875 | +0.000 | 0.025 | +0.000 |  |
| ecap | baseline | 1.5 | 0.875 | +0.000 | 0.025 | +0.000 |  |
| ecap | artifact_window | 0.5 | 1.000 | +0.000 | 0.002 | +0.000 |  |
| ecap | artifact_window | 1.5 | 1.000 | +0.000 | 0.002 | +0.000 |  |
| ecap | fit_window | 0.5 | 0.750 | -0.125 | 0.032 | +0.007 | yes |
| ecap | fit_window | 1.5 | 0.725 | -0.150 | 0.025 | +0.001 | yes |
| ecap | component_labels | 0.5 | 0.000 | -0.875 | 0.024 | +0.000 | yes |
| ecap | component_labels | 1.5 | 0.875 | +0.000 | 0.024 | +0.000 |  |
| ecap | min_fwhm | 0.5 | 0.825 | -0.050 | 0.018 | -0.006 |  |
| ecap | min_fwhm | 1.5 | 0.825 | -0.050 | 0.021 | -0.003 |  |
| ecap | max_fwhm | 0.5 | 0.850 | -0.025 | 0.026 | +0.002 |  |
| ecap | max_fwhm | 1.5 | 0.875 | +0.000 | 0.025 | +0.000 |  |
| erna | (as set) | 1.0 | 0.850 | +0.000 | 0.526 | +0.000 |  |
| erna | max_components | 0.5 | 0.850 | +0.000 | 0.421 | -0.105 |  |
| erna | max_components | 1.5 | 0.850 | +0.000 | 0.534 | +0.009 |  |
| erp | (as set) | 1.0 | 1.000 | +0.000 | 0.939 | +0.000 |  |
| erp | min_fwhm | 0.5 | 1.000 | +0.000 | 0.939 | -0.000 |  |
| erp | min_fwhm | 1.5 | 1.000 | +0.000 | 0.939 | -0.000 |  |
| sep | (as set) | 1.0 | 1.000 | +0.000 | 0.076 | +0.000 |  |
| sep | max_fwhm | 0.5 | 1.000 | +0.000 | 0.076 | -0.000 |  |
| sep | max_fwhm | 1.5 | 1.000 | +0.000 | 0.076 | +0.000 |  |

## 3. Sampling rate (ds004080, N1 against the published rule)

| reading | rate | kappa | recall | median abs latency diff (ms) |
|---|---|---|---|---|
| paper | native | 0.410 | 0.429 | 8.41 |
| paper | 1024 | 0.385 | 0.379 | 10.19 |
| paper | 512 | 0.385 | 0.447 | 7.22 |
| code | native | 0.398 | 0.882 | 3.47 |
| code | 1024 | 0.407 | 0.794 | 4.45 |
| code | 512 | 0.315 | 0.824 | 2.37 |

This section was recomputed alone on 2026-10-11 (19.4 min; Linux-6.18.44-fc-v114-x86_64-with-glibc2.39, 4 CPUs).

The published rule's decisions were recomputed on 2026-10-13 (each reading in its own frame: paper baseline [-1.0, -0.1] s, time shift 0 sample(s); code baseline [-2.0, -0.1] s, time shift 1 sample(s)); the model's decisions are those of the run above.

At 1024 Hz, recorded at or below the rate and not resampled: ccepAgeUMCU01 run 021448 (512 Hz).

At 512 Hz, recorded at or below the rate and not resampled: ccepAgeUMCU01 run 021448 (512 Hz).

paper: recall drop at 1024 Hz +4.97 points (criterion <= 5.0): met.

code: recall drop at 1024 Hz +8.82 points (criterion <= 5.0): NOT met.

## 4. Trial count (existence decisions against all trials)

| trials | draws | decisions changed | median kappa |
|---|---|---|---|
| 5 | 35 | 0.259 | 0.000 |
| 7 | 35 | 0.115 | 0.594 |
| 10 | 35 | 0.000 | 1.000 |

## 5. Platform

Linux-6.18.44-fc-v114-x86_64-with-glibc2.39: fixture fit within rtol 1e-09 of the reference: yes (largest relative difference 0). macOS and Windows (x86-64, arm64) are the cells of the tests matrix of .github/workflows/ci.yml (tests/test_determinism.py); not run on this machine.

