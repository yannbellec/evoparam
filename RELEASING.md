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

Each release is produced by this checklist (REQ-308) and yields a tag `vX.Y.Z`, a GitHub
release, a PyPI upload of the sdist and the wheel by the `release` workflow (trusted
publishing, no stored token), a Zenodo DOI under the concept DOI of `CITATION.cff` (version
DOI added to the release notes) and a CHANGELOG section.

1. Record the decision: update the gate above and the `hiie-decision` block of
   `GOVERNANCE.md` (reference and date; `python tools/release_gate.py` must pass); remove the
   `Private :: Do Not Upload` classifier from `pyproject.toml`.
2. Freeze the CHANGELOG: rename `## Unreleased` to `## X.Y.Z (YYYY-MM-DD)`; every layer-3
   regression beyond its tolerance has its line `Layer-3 regression explained: <metric id>:
   <reason>` (REQ-376).
3. Bump the version in `evoparam/version.py` and in `CITATION.cff` (`version`,
   `date-released`); `evoparam/tests/test_citation.py` checks both.
4. Run `python benchmarks/run_benchmarks.py --suite perf` on the reference machine named in
   `benchmarks/README.md` and record P1–P6 in the CHANGELOG (REQ-290); the results file is
   attached to the release.
5. Run the robustness layer `python validation/robustness.py --jobs 4` (the five analyses of
   spec §7.4, REQ-375) and the gate `python tools/layer3_gate.py` against the previous tag
   (REQ-376); a blocked gate stops the release. Both reports go into the release notes.
6. Run the whole suite (`python tools/run_all_tests.py`, network, slow and performance tests
   included), `python tools/check_traceability.py --strict` and `python tools/gen_soup.py
   --check` (REQ-399).
7. Build the docs with `sphinx-build -W --keep-going -b html docs docs/_build/html`
   (warnings are errors, REQ-302).
8. Tag the release (`git tag -s vX.Y.Z`) and push the tag: the `release` workflow runs the
   gate, builds and checks the distribution and the validation reports.
9. After M1 only: add the upload job of `ci/release-publish-job.yml` to
   `.github/workflows/release.yml` (PyPI trusted publishing, environment `pypi`), so that the
   tag publishes the distribution.
10. Verify the PyPI smoke test (`smoke` workflow: clean environment, `pip install
    evoparam==X.Y.Z`, `evoparam.selftest()`, `evoparam fit --help`, REQ-300) and link its
    log from the release notes.
11. Verify the Zenodo DOI of the version (GitHub–Zenodo integration) and add it to the
    release notes and to `CITATION.cff` (`identifiers`).
12. Announce (release notes, mailing list, Neurostars), only after steps 10 and 11.

## Checks for every release (from M2 on)

1. The full test suite passes on a clean environment (`pip install -e ".[all,tests]"`, then
   `pytest`), with `EVOPARAM_NETWORK_TESTS=1` and `EVOPARAM_SLOW_TESTS=1` for a minor release.
2. `CHANGELOG.md` has the release's section; `evoparam/version.py` and `CITATION.cff` carry
   the version; the methods texts of the reference pipelines are unchanged or their change
   is in the CHANGELOG.
3. The documentation builds (`sphinx-build -W -b html docs docs/_build/html`) and every
   tutorial runs.
4. `python -m build`, `twine check dist/*`; the tag `vX.Y.Z` is signed by the author.
5. Zenodo archives the tag (DOI recorded in `CITATION.cff`).

## 1.0 checklist (REQ-329)

The integration targets of spec §10.8, each with a tracking issue on the roadmap board and an
owner. These three are required for 1.0:

- [ ] **MNE example**: a gallery example in the MNE documentation using `evoparam.fit` on an
      `Evoked` (evoparam's own: `examples/mne/plot_fit_mne_sample.py`).
      Link: _to add when the MNE pull request is opened_
- [ ] **DANDI / NWB listing**: `ndx-evoparam` (`evoparam/io/ndx_evoparam`, namespace 0.1.0)
      released to PyPI and submitted to the NDX catalog, an example notebook on a dandiset,
      and the listing on the NWB tools page.
      Links: NDX catalog _to add_; NWB tools page _to add_
- [ ] **conda-forge feedstock**: `evoparam` through `staged-recipes`.
      Link: _to add_

The other targets of the roadmap board (spec §10.8): MNE-BIDS, SpikeInterface, pynapple,
Elephant / Neo, MOABB, MATLAB users.

## Upstream helpers (REQ-312)

Code proposed to MNE or MNE-BIDS lives in the separate repository `evoparam-upstream`
(BSD-3-Clause), written new: its CI runs `tools/check_no_inherited_code.py` (copied as
`scripts/check_no_inherited_code.py`), and its pull-request checklist states "not derived
from Apache-2.0 inherited files".
