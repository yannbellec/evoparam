Reporting and citing
====================

What to report
--------------

A result of evoparam can be reproduced only when the paper states how it was obtained. Report
at least:

- the **settings**: the preset, its version, and every setting changed from the preset;
- the **package version** (and, once releases are archived, its versioned DOI);
- the **goodness of fit** of the channels you analyse (``r2``, ``nrmse``, and the model
  selection criterion);
- the **existence test**: method, alpha, multiple-comparison correction, and how many
  channels were found responsive;
- the **artifact method** and its window;
- the **montage** (reference) used before fitting.

In a 2025 systematic review of clinical studies that used the ancestor tool, 30 % reported a
goodness-of-fit check and 56 % documented the settings, although the tool could print them
(Donoghue 2025, [LIT-fooof-4]). evoparam therefore writes this paragraph for you by default,
and stores it with the result.

How ``methods_text()`` fills it
-------------------------------

Every result (``ResponseFit``, ``GroupFit``, ``TrialFit``, ``HierarchicalFit``,
``ResponseTest``, ``ArtifactResult``, ``TFFit``) has a ``methods_text()`` method that builds the
paragraph from the result's provenance and settings only: the package version, the preset
and its version, the settings that differ from the preset, the fitting and model-selection
rules, the uncertainty method, the existence test with its alpha and correction, the
artifact method, the montage, the random seed, the active warnings, and the citations of the
methods used.

.. code-block:: python

   result = evoparam.fit(trials, preset="ccep", random_state=0)
   print(result.methods_text())

The paragraph is stored with the result:

- ``result.save(path)`` writes it to ``methods.txt`` inside the ``.evoparam`` container
  (listed with its SHA-256 in ``manifest.json``); ``evoparam.load_object(path)`` gives back a
  result whose ``methods_text()`` is byte-identical;
- ``evoparam.io.write_bids_derivatives`` writes it to the derivative's ``README``, and each
  result container next to its table holds its own ``methods.txt``;
- ``evoparam.io.write_nwb`` writes it, with the settings and provenance, to the description
  of the ``evoparam`` processing module.

The tutorial *Reporting your results* (:doc:`auto_tutorials/index`) shows the full chain:
methods paragraph, tidy table with units, saving, and ``evoparam.reproduce`` regenerating the
result from its provenance.

Citing evoparam
---------------

Cite the software and, when there is one, the paper:

- **the software**: ``CITATION.cff`` at the root of the repository (and inside the installed
  package) is the single source. ``evoparam.cite()`` in Python and ``evoparam cite`` on the
  command line print it with the version you use; ``bibtex=True`` (``--bibtex``) gives a
  BibTeX entry. Every release will be archived on Zenodo with a versioned DOI under one
  concept DOI; ``evoparam.cite()`` then prints both DOIs;
- **the paper**: when the methods paper is published, ``CITATION.cff`` gets a
  ``preferred-citation`` and ``evoparam.cite()`` prints it as well.

The archive, its DOIs and the paper come with the first public release, which waits for the
institutional decision recorded in ``GOVERNANCE.md`` (nothing is public before it). Until
then ``evoparam.cite()`` prints the software citation with its version only.

The methods paragraph also cites the published methods evoparam implements (for example the
false-discovery-rate correction, or the published N1 rule when it is used); keep those
citations in your reference list.
