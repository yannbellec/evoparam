# The two readings of the published N1 rule, drawn

One made-up trial average (an N1 of about 130 µV at 22 ms over noise of SD ≈ 18 µV)
judged by both readings of the rule of van Blooijs et al. 2023 (ds004080):

- `rule_as_written`: the rule as the paper states it (registry row `vanblooijs2023`):
  SD and mean of the average over −1 to −0.1 s; deepest strict local minimum in 9-100 ms;
  N1 if it lies more than k × SD (k = 3.4) below the mean. Here: N1.
- `published_code`: the authors' MATLAB code (`ccep_detect_n1peak_ECoG.m`, mnl_ccepAge;
  registry row `vanblooijs2023_code`): SD over −2 to −0.1 s with a 50 µV floor, median level,
  dips under 20 µV deep ignored, first sample skipped, > 3000 µV refused. With the floor the
  trough must exceed 170 µV at k = 3.4. Same average: no N1.

The curve is illustrative, not a recording. Rebuild: `python benchmarks/plots/plot_rules_schematic.py`.
