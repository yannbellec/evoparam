# Governance

## During 0.x

The project author, Yann V. Bellec, is the final decision maker and the sole release
manager (a "benevolent dictator for life" model while the project is young). Decisions are
discussed in issues and pull requests before they are taken.

## From 1.0

A **steering group of three people** takes decisions: the project author plus two
maintainers elected by the maintainers who merged at least 10 pull requests in the previous
12 months. Elected members serve two-year terms.

* Decisions are taken by lazy consensus with a 7-day window; when consensus fails, by
  majority vote of the steering group.
* The steering group owns releases, the roadmap and the code of conduct.

## Roadmap and releases

The roadmap is a public GitHub Project board with the columns `Next minor`, `Later`,
`Needs data` and `Needs decision`; each item links to a requirement (REQ) of the
specification. During 0.x a minor release comes about every three months, patch releases as
needed, and each minor release carries at most one set of breaking changes, announced in the
CHANGELOG with the deprecation rules of the specification.

## Contributions and licence

Inbound = outbound (Apache-2.0) with the Developer Certificate of Origin (`Signed-off-by` on
every commit, checked on pull requests); no contributor licence agreement during 0.x. Any
change of this policy is a steering-group decision recorded in this file.

## Support

GitHub Discussions (questions, show-and-tell), the Neurostars tag `evoparam`, and the
issue tracker for bugs only. The target is a median first response within 7 days, reported
in a yearly status post.

## Institutional decision (HIIE)

The invention disclosure and the written decision of HIIE (patent or not; licence; trademark
owner) come before any public step (specification §11.4 and §13.7, REQ-398). The release
workflow (`.github/workflows/release.yml`, `tools/release_gate.py`) reads the two lines
below and refuses to go on until both carry a reference and a date.

<!-- hiie-decision -->
- Invention disclosure: not filed
- HIIE decision reference: none
<!-- end-hiie-decision -->

Once recorded, each line reads, for example, `- HIIE decision reference: HIIE ref. 2027-123,
2027-03-01 (patent: no; licence: Apache-2.0; trademark owner: ...)`.
