# Step 6 — Describe the response: effect of each threshold on real data (version 1)

Each figure compares several values of one setting; every other setting keeps its proposed
value. Red: the value proposed today. Grey: the alternatives.

This is version 1, made for a first presentation: real data only, channels pooled, no confidence
intervals, no known ground truth. Version 2, in `figures/v2/etape-6/`, adds those.

Data (public, already on disk):

- **CCEP**: ds004080, 6 patients and 7 runs (the N1 validation list), and ds004977, 4 patients.
  Trial averages. In each run, the 10 most responsive channels and 10 channels drawn at random
  (seed 0): 220 channels. Preset `ccep` 1.1.
- **ERP CORE**: the published difference waves of the 7 components, 40 subjects. That gives 279
  waves, because one subject has no wave for one component. The reference is the published
  latency: the 50 % area latency of the raw wave.
- **TEP**: ds002094, 20 subjects, published (TESA) cleaning. In each subject, the channel with
  the largest response from 38 to 140 ms. Preset `tep` 1.1.

To regenerate: `python build_erp.py` (cache of the ERP CORE waves), then `python make_etape6.py`.

---

## 6-01-shapes.png — Allowed component shapes

Compares Gaussian only, skewed only, and both (proposed), on 220 CCEP channels from 10 patients.
With Gaussian only, slightly more components are needed (2.4 against 2.3), and the fit is worse
on part of the channels (lower R²).

## 6-02-background.png — Background model

Compares none, constant, linear and exponential (proposed), on 220 CCEP channels from 10
patients. Without a background, the model adds components to follow the drift: about 3 per
channel against 2.3. The exponential background needs the fewest components and gives the
highest R².

## 6-03-width-and-max-components.png — Maximum number of components and maximum width

Maximum number of components from 2 to 10, and maximum FWHM from 50 to 500 ms, on 220 CCEP
channels from 10 patients. At a maximum of 6 (proposed), fewer than 2 % of the channels reach
it, so the bound almost never constrains the fit. A maximum FWHM of 50 or 100 ms cuts slow
components into pieces: 3.2 components against 2.3.

## 6-04-delta-bic.png — Rule to keep one more component (ΔBIC)

ΔBIC threshold of 2, 5, 10 (proposed), 20, 40, and "calibrated" (set on simulations), on 220
CCEP channels from 10 patients. The stricter the threshold, the fewer components are kept:
on random channels, from 0.66 (threshold 2) to 0.26 (threshold 40). An N1 appears on 5.5 % of
the random channels at threshold 2, against 2 % at threshold 10. On responsive channels, the
share with an N1 barely moves up to threshold 20, then drops at 40 (28 %).

## 6-05-candidate-threshold.png — Starting-candidate threshold

Threshold for a starting candidate from 1 to 4 σ, on 220 CCEP channels from 10 patients. The
effect on the number of components (2.4 to 2.1) and on the N1s (42 % to 38 %) is small. The
main effect is on computation time, which drops by more than half from 1 σ to 4 σ.

## 6-06-merge.png — Merging two close components

Merge factor c from 0 to 2: two components merge if their gap is below c × (σ₁ + σ₂). Tested on
220 CCEP channels from 10 patients. On these data the setting changes almost nothing: about 2.3
components for every value, and the same gaps between components. It is a low-sensitivity
choice.

## 6-07-cancelling-pairs.png — Cancelling pairs of opposite-sign components

Keeping or rejecting close + / − component pairs that cancel. Tested on 220 CCEP channels and on
the stimulus-locked ERP CORE waves (full `erp` preset). Rejecting them does not change the mean
number of components. It lowers the share of waves that contain such a pair: from 14.5 % to
9.5 % in CCEP, and from 6.5 % to 3.5 % in ERP.

## 6-08-latency-definition.png — Latency definition

Raw peak, model component peak, and 50 % area of the model (proposed for ERPs), on 279 ERP
CORE waves (40 subjects × 7 components). With the 50 % area of the model, 92 to 100 % of the
subjects are within ± 10 ms of the published value. The raw peak and the model peak are far from
it for P3 and N400, with only 7 to 17 % of subjects within ± 10 ms. The definition alone can
move the latency by tens of milliseconds.

## 6-09-onset-offset.png — Onset and offset

Onset and offset threshold of 5, 10 (proposed), 20 or 50 % of the peak, on 279 ERP CORE waves
(40 subjects × 7 components). The reported duration depends strongly on this threshold: for
P3, from about 430 ms (5 %) to 195 ms (50 %). The threshold must be fixed before durations are
compared across studies.

## 6-10-targeted-measure.png — Targeted measure of an ERP component

Margin around the published window from 0 to 50 ms, and 1 or 3 parts, on 279 ERP CORE waves
(40 subjects). Three parts agree better than one (P3: 92 % against 75 %). The ± 25 ms margin
(proposed) gives the best overall agreement. The right panel shows that above 10 ms, a wider
tolerance adds almost no subjects.

## 6-11-tep-windows.png — TEP component windows N45 and N100

Three sets of windows, on 20 TMS-EEG subjects (ds002094, TESA cleaning):

- proposed: N45 from 38 to 55 ms, N100 from 80 to 140 ms;
- narrow: N45 from 40 to 50 ms, N100 from 90 to 120 ms;
- wide: N45 from 30 to 60 ms, N100 from 70 to 160 ms.

Only 10 to 20 % of the subjects have a negative component in the N45 window, whatever the
window, so the window is not the main problem. For N100, the share goes from 45 % (proposed) to
55 % (wide).

## 6-12-trial-by-trial.png — Maximum shift, trial by trial

Maximum single-trial shift from 0.1 to 2 × the component FWHM, on the first component of 89
responsive CCEP channels (10 patients, 10 to 12 trials each). At 0.25 × FWHM (proposed for
intracranial data), 17 % of the trials sit at the bound; at 0.5 × FWHM, 9 %. Whatever the
bound, the variability found stays close to the variability that noise alone produces: with
10 to 12 trials, real latency variability cannot be measured.
