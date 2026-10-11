# Algorithms written from papers

Spec REQ-396. Every algorithm of evoparam that comes from a publication is listed here with
the publication it was written from, the module that implements it, and any patent
publication known when it was written (spec §13.4). Citation keys `[LIT-<axis>-<n>]` resolve
in `docs/references.bib` (and `docs/spec/LITERATURE.md`).

**No code was copied.** Each algorithm below was written from the equations and the text of
the publication. Where a reference implementation exists under the GPL or without a licence
(CRP, BPC, ER-detect, PARRM, MPARRM, the ERA toolbox, ERPLAB), it was neither imported nor
copied nor translated; GPL tools are reached only through the separately licensed adapters
of `adapters/` (spec REQ-395). The code that evoparam inherited is named in `NOTICE` and
marked in its files ("Modified from ..."); it is not listed here.

Patent column: "none known" means that no patent publication was found in the literature
review of the specification (`docs/spec/LITERATURE.md`), not that none exists. The freedom to
operate of the two methods with a known publication is a question for counsel before they
ship in a public release (spec §13.4).

## Stimulation-artifact removal (`evoparam.artifact`)

| method | module | written from | known patent publication |
|---|---|---|---|
| `period_template`: period-based reconstruction (PARRM) | `artifact/period.py` | Dastin-van Rijn et al. 2021, Cell Reports Methods [LIT-artifact-1] (open-access description) | US 2022/0338786 A1, Brown University, pending when read [LIT-artifact-4] |
| `matching_pursuit`: per-pulse decomposition (line noise, evoked response, artifact atoms) | `artifact/pursuit.py` | Xie et al. 2023 (MPARRM) [LIT-artifact-5], public abstract and claim summary only | US 2023/0240618 A1, Washington University [LIT-artifact-25] |
| `lowrank`: sequential low-rank regression across channels, pulses and trials (ERAASR) | `artifact/lowrank.py` | O'Shea and Shenoy 2018, J Neural Eng [LIT-artifact-6] | none known |
| `subspace`: pre-whitening and null projection | `artifact/subspace.py` | Lim et al. 2023, J Neural Eng [LIT-artifact-14] | none known |
| `polynomial`: local cubic fit with depegging (SALPA) | `artifact/polynomial.py` | Wagenaar and Potter 2002, J Neurosci Methods [LIT-artifact-27] | none known |
| `paradigm`: alternating polarity and forward masking | `artifact/paradigm.py` | Baudhuin et al. 2016 [LIT-artifact-36]; Brown et al. 1990 [LIT-artifact-37] | none known |

## Existence of a response (`evoparam.existence`)

| method | module | written from | known patent publication |
|---|---|---|---|
| `projection`: cross-projection test and response duration (canonical response parameterisation, CRP) | `existence/projection.py` | Miller et al. 2023, PLOS Comput Biol [LIT-fooof-92]; the reference implementation is GPL | none known |
| `threshold`: registry of published detection rules (`vanblooijs2023`, `vanblooijs2023_code`, `hays2021`, `keller2014`, `silverstein2020`, `ftract`, `prime2020`, ...), each applied to the letter | `existence/rules.py`, `existence/rules.json`, `benchmark/baselines.py` | the publication named in each rule's `source` field (for example van Blooijs et al. 2023 [LIT-ccep-7]; the ER-detect pipeline [LIT-datasets-62] is GPL and only reached through `adapters/`) | none known |
| permutation p-values with a tail approximation | `existence/permutation.py` | Knijnenburg et al. 2009 | none known |
| sequential sign-flip test with an alpha-spending function | `existence/sequential.py` | Lan and DeMets 1983 | none known |
| false-discovery-rate and family-wise corrections (Benjamini-Hochberg, Benjamini-Yekutieli, Holm, Bonferroni; selective inference across channels) | `existence/corrections.py`, `tf/stats.py` | Benjamini and Hochberg 1995; Benjamini and Yekutieli 2001; Holm 1979; Benjamini and Bogomolov 2014 | none known |

## Referencing and spatial analysis (`evoparam.prep`, `evoparam.spatial`)

| method | module | written from | known patent publication |
|---|---|---|---|
| `car_adaptive`: adjusted common average (CARLA) | `prep/reference.py` | Huang et al. 2024, J Neurosci Methods [LIT-ccep-18] | none known |
| basis profile curves (BPC) | `spatial/bpc.py` | Miller, Müller and Hermes 2021, PLOS Comput Biol [LIT-ccep-16]; the reference implementation is GPL | none known |

## Fitting, measures and statistics (`evoparam.models`, `evoparam.stats`)

| method | module | written from | known patent publication |
|---|---|---|---|
| classical ERP measures (peak, mean amplitude, area, fractional-area latency) with the definitions of the ERP tools | `models/classical.py` | Lopez-Calderon and Luck 2014 (ERPLAB definitions) [LIT-erptools-1] | none known |
| jackknife latency scoring and its corrections | `stats/jackknife.py` | Miller, Patterson and Ulrich 1998 [LIT-erpmeas-4]; Ulrich and Miller 2001 [LIT-erpmeas-5]; Smulders 2010 [LIT-erpmeas-6] | none known |
| split-half reliability and dependability | `stats/reliability.py` | Clayson and Miller 2017 (ERA toolbox) [LIT-erpmeas-36]; Parsons, Kruijt and Fox 2019 | none known |
| bias-corrected accelerated bootstrap intervals | `stats/bootstrap.py` | Efron 1987 | none known |

## Time-frequency analysis (`evoparam.tf`)

| method | module | written from | known patent publication |
|---|---|---|---|
| generalised Morse wavelets | `tf/transforms.py` | Lilly and Olhede 2012 [LIT-tf-24] | none known |
| single-trial baseline normalisation | `tf/normalize.py` | Grandchamp and Delorme 2011 [LIT-tf-12]; Ciuparu and Muresan 2016 [LIT-tf-36] | none known |
| cluster-mass permutation test | `tf/stats.py` | Maris and Oostenveld 2007 [LIT-tf-10] | none known |
| threshold-free cluster enhancement | `tf/stats.py` | Smith and Nichols 2009 [LIT-tf-38] | none known |
| `baseline_normal` pixel test of the eleven-zone work | `tf/stats.py` | Brinyark et al. 2026 [LIT-tf-4] | none known |
| data-driven zones by non-negative matrix factorisation and non-negative canonical-polyadic decomposition | `tf/zones.py` | Lee and Seung 1999 [LIT-tf-45]; Morup et al. 2006 [LIT-tf-42] | none known |

`evoparam/tests/test_spdx_headers.py` checks that every module whose docstring says it was
implemented from a publication is listed above, and that the two methods with a known patent
publication name it.
