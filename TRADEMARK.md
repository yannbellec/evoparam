# Trademark policy for the name "evoparam"

Spec REQ-397. This policy covers the name **evoparam** (and its pairing "evoparam: evoked
response parameterization"), whatever the licence of the code.

## Who owns the name

The owner of the name is set by the written decision of the University of Alabama at
Birmingham technology-transfer office (HIIE / UABRF) on the invention disclosure, recorded in
`GOVERNANCE.md` (section "Institutional decision (HIIE)"). Until that decision is on record
the name is a working name: it is not used publicly, and nothing is published under it
(spec REQ-019, REQ-398). This section is completed with the owner's name when the decision
is recorded, and the policy is reviewed at each major release (`RELEASING.md`).

## The licence grants no trademark rights

evoparam is distributed under the Apache License, Version 2.0. Section 6 of that licence
("Trademarks") states that it "does not grant permission to use the trade names, trademarks,
service marks, or product names of the Licensor, except as required for reasonable and
customary use in describing the origin of the Work and reproducing the content of the NOTICE
file". The permissions below are the only other uses allowed.

## Forks

You may fork the code under the licence. A fork that is distributed (on a package index, as
a download, or as a hosted service) must take a different name that does not contain
"evoparam" and is not confusingly similar to it. The fork may say that it is "derived from
evoparam" or "a fork of evoparam" in its documentation, and must keep the `NOTICE` file.

## Plugins, extensions and products

A package that extends evoparam through its plugin entry points (spec §10.8) or that uses it
as a library may refer to the name in these forms only:

- "**for evoparam**", for a plugin or an extension (for example "Lorentzian shapes for
  evoparam"); a package name of the form `evoparam-<name>` is allowed for a plugin, provided
  its description says that it is not part of evoparam;
- "**built on evoparam**", for a product or a service that uses evoparam as a library,
  including a paid product (spec §13.6).

Neither form may suggest that the plugin or product is made, endorsed or certified by the
owner of the name, and neither may use the name as the product's own name.

## Scientific use

Naming evoparam in a paper, a poster, a talk or a methods section, with its version, is
always allowed, and expected (`CITATION.cff`, `evoparam.cite()`).

## Contact

Questions about this policy go to the project author named in `GOVERNANCE.md`.
