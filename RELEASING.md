# Releasing evoparam

Release steps follow the publication sequence of the specification (§11.4): public steps
(repository, PyPI, preprint) are decided by the project author and are independent of the
build steps.

## Every release

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
