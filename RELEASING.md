# Releasing evoparam

Nothing about evoparam is public before the written decision of HIIE (the institution's
innovation office) is on record: no public repository, no package upload (PyPI, conda-forge,
TestPyPI), no Zenodo deposit, no preprint, no abstract, no poster, no talk, and no
conversation with outside groups about the specific method (REQ-019, REQ-398; specification
§11.4 and §13.7). A disclosure before the decision can forfeit patent rights outside the
United States.

`evoparam/tests/test_release_gate.py` reads the gate below and fails when something in the
repository would make a public step possible before the decision is on record.

## Gate

<!-- release-gate -->
- HIIE decision: not on record
- Invention disclosure (M0): not filed
- Decision record (M1): none
<!-- end-release-gate -->

When the decision arrives, replace the three lines with the date and the reference of each
record, for example `- HIIE decision: on record, 2027-03-01, HIIE ref. 2027-123 (patent: no;
licence: Apache-2.0; trademark owner: ...)`, then follow the steps below. Until then the
package carries the classifier `Private :: Do Not Upload`, which PyPI refuses, and no
workflow publishes anything.

## Sequence of public steps (§11.4)

| step | content | requires |
|---|---|---|
| M0 | invention disclosure filed | private repository with SPDX headers and NOTICE; candidate-novelty list (§13.4) |
| M1 | HIIE written decision: patent or not, licence, trademark owner | M0 |
| M2 | first public release 0.x: GitHub, PyPI, Zenodo DOI the same day; CITATION.cff; docs site | M1; the §7 validation suite green; four presets; `methods_text()` |
| M3 | tutorials and docs complete; conda-forge feedstock | M2 |
| M4 | software paper (JOSS) and preprint | M3; coverage at the REQ-298 thresholds |
| M5 | conference abstracts | M2, never before |
| M6 | 1.0 feature-complete, methods paper preprint | public benchmark leaderboard (REQ-392); figures regenerated from a tag |
| M7 | methods paper submitted | M6 |
| M8 | benchmark paper | labels and lab cohorts cleared for release |

If HIIE files a provisional patent at M1, M2 follows within the twelve months of the
provisional, and foreign filing decisions are taken before M2.

## Making a release (from M2 on)

1. Update the gate above with the decision record; remove the `Private :: Do Not Upload`
   classifier from `pyproject.toml`.
2. Run the whole suite with `python tools/run_all_tests.py --venv .venv-3.12 --python
   python3.12` (network, slow and performance tests included) and the traceability check
   `python tools/check_traceability.py --files`.
3. Update `CHANGELOG.md`, `CITATION.cff` and the version; tag the release.
4. Build and check the distribution (`make dist check-dist`), then upload (`make publish`
   prints the commands).
