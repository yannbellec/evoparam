# Contributing Guidelines

Thank you for your interest in contributing to `evoparam`!

This project follows a [code of conduct](CODE_OF_CONDUCT.md) that you are expected to uphold.
How decisions are made is described in [GOVERNANCE.md](GOVERNANCE.md); where to ask questions
in [SUPPORT.md](SUPPORT.md); how to report a vulnerability in [SECURITY.md](SECURITY.md).

## Reporting a problem

Please open an issue on the [issue tracker](https://github.com/yannbellec/evoparam/issues)
with the *Bug report* form, with:

1. A short summary of the issue (1-2 sentences)
2. A short, self-contained code snippet that reproduces it
3. The actual outcome
4. The expected outcome
5. The output of `evoparam info`

Questions go to GitHub Discussions or the Neurostars tag `evoparam`, not to the tracker.

## Project scope

`evoparam` parameterizes evoked neural responses (for example ERPs, and
cortico-cortical evoked potentials, CCEPs): model fitting, management and
analysis of the resulting parameters, visualization, and simulation of evoked
responses. Human and animal recordings are both in scope.

## Development install

```console
$ git clone https://github.com/yannbellec/evoparam && cd evoparam
$ uv venv && uv pip install -e ".[all,tests,docs]"     # or: pip install -e ".[all,tests,docs]"
```

## Tests and docs

```console
$ pytest                                   # the test suite (network tests are skipped)
$ EVOPARAM_NETWORK_TESTS=1 pytest -m network   # tests that read public datasets
$ EVOPARAM_SLOW_TESTS=1 pytest             # long simulation grids
$ pytest --doctest-modules --ignore=evoparam/tests evoparam
```

Run one pytest process at a time: the legacy tests share a temporary folder.

## Coding rules

* `ruff check` and `ruff format` (line length 99), numpy-style docstrings.
* Every public function records a provenance step; settings are recorded in full.
* No silent failure: name the exception class in every `except` clause; a failed channel
  carries `status="failed"` and a message.
* No new hard dependency: optional packages are imported lazily and listed as extras.
* Modules import each other only along the dependency table of the specification (§4.2);
  `evoparam/tests/test_import_graph.py` checks it.

## Adding a preset, a shape, a reader or a dataset

* **Shape, preset or reader**: publish it as a plugin through the entry points
  `evoparam.shapes`, `evoparam.presets` and `evoparam.readers`; nothing in evoparam has to
  change. See [docs/contributing/add_shape.rst](docs/contributing/add_shape.rst) and the
  example plugin `evoparam/tests/plugins/evoparam_dummy`.
* **Preset in evoparam itself**: open a *New preset* issue with the literature keys of each
  value and a test dataset, then add `evoparam/presets/<name>.json`.
* **Dataset**: open a *New dataset* issue (DOI, licence, size); entries of
  `evoparam/io/registry.json` carry a sha256 for each file.

## Making a contribution

1. Open an issue describing the change and why.
2. Fork the repository and create a branch.
3. Add tests for new code, a CHANGELOG entry, and run `pytest` before opening a pull request.
4. Sign off every commit (Developer Certificate of Origin, below).

## Developer Certificate of Origin (DCO)

Contributions are accepted under the licence of the project (Apache-2.0, inbound =
outbound), with the [Developer Certificate of Origin 1.1](https://developercertificate.org/):
by adding `Signed-off-by: Your Name <you@example.org>` to a commit message (`git commit -s`)
you certify that you wrote the change or have the right to submit it under the project's
licence. There is no contributor licence agreement during 0.x; any change to this policy is
a steering-group decision recorded in [GOVERNANCE.md](GOVERNANCE.md).
