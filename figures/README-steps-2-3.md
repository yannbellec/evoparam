# Threshold figures: steps 2 and 3

Each figure compares the proposed value (in blue or bold) with other values. Only one value changes at a time. Everything else keeps the default setting of the evoparam package. All data are real (OpenNeuro). The numbers below were measured for these figures.

## Step 2: find the pulses

**etape-2/2-01-pulse-z-threshold.png**: detection threshold (robust z > 10, 15, 20 proposed, 30).
Data: 62 raw-signal segments (ds004080: 25 recordings; ds003708: 3; ds004457: 15; ds004696: 15; ds004977: 4) and 372 true pulses from events.tsv.
Result: at z > 20, 371 of the 372 pulses are found, with 2 false detections. At z > 15 there are 37 false detections. At z > 10 there are 33 false detections and 4 missed pulses. At z > 30 the result is the same as at z > 20.

**etape-2/2-02-size-filter.png**: size filter (a candidate must be at least 25% of the size of clear pulses; compared with 0% (no filter), 10% and 50%).
Data: the same 62 segments and 372 pulses.
Result: there are 157 false detections without the filter and 51 at 10%. At 25%, 2 false detections and 1 missed pulse remain. At 50%, 6 pulses are missed.

**etape-2/2-03-fallback-90th-percentile.png**: fallback measure when the median of the electrodes finds nothing (90th percentile proposed; compared with no fallback and with the maximum).
Data: the same 62 segments. The fallback is used on only 2 segments (ds004977, subjects 1 and 4, 5 pulses each). On the other 60 segments, the three settings give the same result.
Result: without the fallback, 0 of 5 pulses are found; with the 90th percentile or the maximum, 5 of 5. With the median, the weakest pulse reaches z = 9 and z = 7. With the 90th percentile it reaches 218 and 54, and with the maximum 166 and 1,797. No peak away from the pulses goes above z = 8.

## Step 3: remove the artifact

**etape-3/3-01-masked-window-ccep.png**: masked window for CCEP (−2 to 8 ms proposed; compared with −1 to 5 ms and −2 to 12 ms).
Data: ds004080, 14 subjects (one stimulated pair per subject, 10 trials) and 408 electrodes with a clear N1. Cleaning is that of the ccep preset. The N1 is measured simply, as the minimum of the average between 9 and 50 ms (no model fit).
Result: the short window leaves some artifact between 5 and 8 ms, but the N1 does not change. Latency is identical on all 408 electrodes, and amplitude changes by more than 10% on only 1 electrode. With the long window, amplitude changes by more than 10% on 109 electrodes and latency changes on 120 electrodes (median shift 2.9 ms). Per subject, the median amplitude goes from −64.8 to −66.7 µV (Wilcoxon, p = 0.013) and the latency from 29.1 to 27.8 ms (p = 0.26). Part of the difference comes from the adaptive common average, which picks other reference electrodes when the window changes (see panel C).

**etape-3/3-02-masked-window-tms.png**: masked window for TMS (−2 to 12 ms proposed; compared with −2 to 8, −2 to 16 and −2 to 20 ms).
Data: ds002094, 20 subjects. Only 20 s of raw signal per subject are available locally, which gives 2 to 5 pulses per subject (4 for most). Cleaning is that of the tep preset (interpolation, then projection). The measure is the global activity: the standard deviation of the average across electrodes.
Result: between 8 and 12 ms, the 8 ms window leaves 33 µV (median of subjects), against 1.3 µV with the 12 ms window (p < 0.001). Between 20 and 30 ms, activity drops as the window gets longer: 9.0 µV at 12 ms, 8.1 at 16 ms and 5.4 at 20 ms. With the 8 ms window, which leaves residual artifact, it is 17.2 µV. Between 30 and 60 ms, the values stay close: 10.8, 9.5, 9.7 and 9.8 µV.

**etape-3/3-03-cleaning-method.png**: cleaning method (interpolation proposed; compared with setting to zero, template subtraction, learned basis (PCA) and no cleaning).
The package has no ICA, so ICA is not shown. Setting to zero is not a method of the package, so it was done by hand.
Data: top, CCEP ds004080 with 14 subjects and the −2 to 8 ms window. Bottom, TMS ds002094 with 19 subjects and the −2 to 12 ms window. One subject has only 2 pulses, too few for the template, and is left out.
CCEP result: the method changes only the inside of the window, and the N1 stays the same (median −64.8 µV for all four methods). What is left inside the window is 31 µV with interpolation, 19 µV with the template, 54 µV with the learned basis and 301 µV without cleaning.
TMS result: after the full cleaning (including the projection), activity between 12 and 40 ms is 9.1 µV with interpolation, against 25 to 32 µV with the other methods (p < 0.001 for each).

Statistics: paired Wilcoxon test across subjects, proposed value against each other value.
