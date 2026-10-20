# Plot A: N1 detection threshold and the share of contacts with an N1

Data: ds004080 (van Blooijs et al. 2023, OpenNeuro), 51 of the 74 patients: those with a
SPESclin run recorded at 2048 Hz that has a stimulation pair of at least 10 trials (the
other 23 have no such run). Per patient: the first such run (listing order) and its first
such pair; every trial of the pair (both polarities), -2 to 3 s, the good non-stimulated
ECoG channels of channels.tsv, raw monopolar data (`benchmarks/plots/ds004080_subjects.tsv`).

Average: median of -1 to -0.1 s removed from each trial, mean over trials, median of -2 to
-0.1 s removed from the average (mnl_ccepAge). Rule: a negative peak of the average whose
magnitude exceeds k times the SD of the baseline, read in the N1 window 9-50 ms, for k from
1 to 10 in steps of 0.1. Two readings of the published rule:

* rule as written (registry `vanblooijs2023`, strict local minimum, SD of -1 to -0.1 s);
* published code (registry `vanblooijs2023_code`: SD floor 50 uV, 20 uV peak selectivity,
  SD of -2 to -0.1 s).

The unit of statistics is the patient: per patient, the % of its contacts with an N1; the
mean over patients with a 95 % bootstrap interval over patients (5000 resamples, seed 0).

| File | What it shows |
| --- | --- |
| `A1_threshold_sweep` | k on x, % of contacts with an N1 on y; grey: each patient (rule as written); bands: mean and 95 % CI over patients for both readings |
| `A2_ridgeline_per_patient` | distribution over patients of the % at k = 2, 2.6, 3, 3.4, 4, 5, 6 (rule as written); black tick: median |
| `A3_paired_estimation` | paired estimation plot (DABEST): change of the % per patient when k goes from 3.4 to 2.6 or 4.2; `A3_paired_differences.csv` holds the mean differences, BCa intervals and permutation p-values |
| `A4_example_patient_traces` | one patient (ccepAgeUMCU01, PT01-PT02): stacked averages, N1 found at k = 3.4 (filled) or only at k = 2.6 (open) |

Values (rule as written, mean over 51 patients): 18.7 % of contacts at k = 3.4; +5.7 points
at k = 2.6 (95 % CI 4.6 to 7.4); -2.8 points at k = 4.2 (95 % CI -3.6 to -2.3). Published
code reading: 9.6 % at k = 3.4.

Style: SciencePlots (`science`, `no-latex`), batlow (cmcrameri) for the ridgeline, DABEST
for the estimation plot.

Rebuild (needs `pip install scienceplots cmcrameri dabest seaborn`; the data are read from
OpenNeuro on the first run and cached):

```
python benchmarks/plots/fetch_ds004080.py
python benchmarks/plots/plot_a_compute.py
python benchmarks/plots/plot_a_figures.py
```
