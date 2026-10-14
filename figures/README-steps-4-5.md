# Threshold figures, steps 4 and 5

Each figure changes one threshold and keeps every other setting at the package value (evoparam, main at 013d7b7).
Data: CCEP = OpenNeuro ds004080, 14 subjects, one run each, one stimulation pair, 10 trials, every ECoG channel (1051 channels in all).
TMS-EEG = OpenNeuro ds002094, 600 trials and 30 electrodes per subject, the tep preset chain without ICA.
In the paired plots, a gray line is one subject and the colored dot is the mean ± SD over subjects. The p-values are paired Wilcoxon tests across subjects.
Each PNG has a CSV next to it with the plotted numbers per subject.

**Which value is the code default.** The figures label the first value "proposed", following the request. In the code (evoparam/presets/*.json on main), the starting values are: baseline ccep −500 to −20 ms, tep −800 to −100 ms, erp −200 to 0 ms; reference ccep adaptive CAR, tep average, erp as recorded. So in 4-01 the code default is the −500 to −20 ms bar, and in 4-04 it is the adaptive CAR bar. In every other figure, the "proposed" bar is also the code default.

## Step 4, preparation

**4-01-baseline-window.png**: baseline −100 to −5 ms against −500 to −20 ms (code default for ccep) and −50 to −5 ms, on CCEP, 14 subjects. The same window is used for the baseline correction, for the SD of the N1 rule and for the noise of the model.
The estimated noise is 10.8 µV (−100/−5), 20.5 µV (−500/−20) and 6.9 µV (−50/−5) (median per subject, then mean). Channels with an N1 by the published rule: 46.2 %, 26.6 % and 56.3 %; N1 found by the model: 22.4 %, 18.0 % and 28.2 %. The permutation test hardly moves (27.3 %, 23.5 %, 26.3 %; p = 0.14 and 0.59).
In short: a shorter baseline looks less noisy, so threshold rules find many more N1s; the permutation test does not depend on it.

**4-02-noise-source.png**: noise source "auto" (baseline or plus-minus) against "always the baseline", on CCEP, 14 subjects, 1051 channels.
With 10 trials at 2048 Hz, auto picks plus-minus on 100 % of channels (the baseline is too correlated). Plus-minus noise is a median 1.23 times the baseline noise (quartiles 0.94–1.57). The noise the fit uses goes from 20.5 µV to 24.9 µV (p < 0.001), but the components kept per channel barely change (1.91 against 1.89, p = 0.09).

**4-03-noise-model.png**: noise model AR whitening (code default) against n_eff correction against none, on CCEP, 14 subjects, 1051 channels. False components are counted on the same window read 1 s before the stimulation, where there is no response.
After stimulation: 1.89 components per channel with AR, 0.01 with n_eff (9 of 1051 channels keep any component), 4.11 with none. Before stimulation: 4.7 % of segments get a false component with AR, 0 % with n_eff, 55.9 % with none (all p ≤ 0.006).
In short: without a noise model the fit invents components; n_eff removes almost everything, real responses included; AR sits between.

**4-04-reference.png**: reference bipolar against common average (CAR) against adaptive CAR (code default for ccep), on CCEP, 14 subjects, with example traces from subject S01. Bipolar channels are different channels, so the comparison is per subject.
N1 by the published rule: 22.3 % (bipolar), 23.3 % (CAR), 22.2 % (adaptive CAR), no clear difference. Median |N1 amplitude| of detected N1s: 146, 104 and 123 µV. The permutation test reacts much more: 20.6 %, 38.4 % and 23.5 % of channels; plain CAR makes some subjects close to 100 % responsive (bipolar vs CAR p = 0.008).

**4-05-trial-rejection.png**: z threshold 5 (code default) against 3, 4, 6 and 8, on TMS-EEG, 20 subjects (CCEP has only 10 trials per run, too few for this). Only ptp_z and rms_z change. These data have no trial quality score, so the "quality < 0.5" part does not act.
Trials rejected in full: 11.1 %, 6.7 %, 4.4 %, 2.7 % and 1.1 % for z > 3, 4, 5, 6, 8. Trial-channel pairs masked: 14.3 %, 9.9 %, 7.3 %, 5.5 %, 3.3 %. The baseline noise of the average does not go down: it rises by 19.1 %, 9.9 %, 5.0 %, 3.0 % and 0 % against no rejection, because fewer trials are averaged.

## Step 5, testing for a response

**5-01-alpha.png**: α = 5 % against 1 % and 10 %, permutation test with FDR, on CCEP, 14 subjects, every channel. Only α changes.
Channels declared responsive: 23.5 % at 5 %, 9.2 % at 1 %, 32.1 % at 10 % (both p = 0.002). The spread between subjects is large (SD about 20 points).

**5-02-permutations.png**: 1 000 permutations against 5 000 and 10 000; p-value stability across random seeds and computing time, on CCEP (14 subjects) and TMS-EEG (7 subjects: sub-283, 711, 718, 728, 734, 735, 739).
CCEP has 10 trials, so only 1 024 sign flips exist: with 5 000 or 10 000 the test lists them all and the p-value is exact (no spread). With 1 000, the least stable channel moves by SD 0.020. TMS-EEG: SD 0.0028, 0.0006 and 0.0004. Time per test: CCEP 3.8 to 3.9 s; TMS-EEG 13.7, 16.2 and 19.5 s.

**5-03-correction.png**: correction across electrodes FDR against Bonferroni against none, on CCEP, 14 subjects. Same permutations, α = 5 %.
Responsive channels: 23.5 % (FDR), 0.2 % (Bonferroni), 37.6 % (none). With 10 trials the smallest p-value is about 0.001 (median per subject 0.0011), while Bonferroni over about 66 channels needs p ≤ 0.0008, so it almost never passes.

**5-04-projection-pairs.png**: projection test with sign flip (code default) against disjoint pairs against half pairs, on CCEP, 14 subjects, 1051 channels.
Responsive channels: 35.8 % (sign flip), 19.7 % (disjoint), 59.8 % (half pairs). Sign flip agrees with disjoint on 84 % of channels (κ = 0.58) and with half pairs on 76 % (κ = 0.54); disjoint and half pairs agree on 61 % (κ = 0.28). The package documentation says half pairs wrongly reject 33 to 44 % of pure-noise cases at α = 5 %.

**5-05-n1-rule.png**: the classic N1 rule (3.4 SD, 9–100 ms, registry rule vanblooijs2023 with a strict negative peak) compared with the model's N1 (ccep preset, negative component at 9–50 ms), and the factor varied (2.5, 3.4, 5), on CCEP, 14 subjects, 1051 channels, same preparation for both.
At 3.4 SD the two agree on 84 % of channels (κ = 0.49): 113 both, 104 rule only, 60 model only, 774 neither. The rule finds an N1 on 22.2 % of channels at 3.4 SD, 32.8 % at 2.5 SD and 14.3 % at 5 SD; the model on 18.0 %. Overall κ with the model: 0.49, 0.40 and 0.48. Bottom row: one example of each case from subject S01.

## Limits

- CCEP: 10 trials per run, so permutation p-values cannot go below about 0.001 and trial rejection is shown on TMS-EEG instead.
- 4-01, 4-03, 4-05 and 5-05 use the model fit on the trial average (as the repository's N1 tests do), so the model's noise comes from the baseline. 4-02 fits on the trials, where "auto" can choose plus-minus.
- 5-02, TMS-EEG: the test window starts at 13 ms, not at the preset's 12 ms, because 12 ms is the last sample of the artifact window and the test refuses it.
