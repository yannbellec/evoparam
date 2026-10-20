=========================================================
evoparam: evoked response parameterization
=========================================================

**Website and tutorials:** https://yannbellec.github.io/evoparam

evoparam (evoked response parameterization) is a tool to parameterize neural evoked
responses: it describes each response (ERP, CCEP, ...) as a small set of peaks with a
latency, an amplitude and a width.

Status: early development (version 0.1, working name). Origin of the code and licence:
see ``NOTICE`` and ``LICENSE``; third-party packages: ``THIRD_PARTY.md``; algorithms written
from papers: ``REIMPLEMENTED.md``; use of the name: `TRADEMARK.md <TRADEMARK.md>`_.

What it does
------------

evoparam covers the whole path from a raw recording to response parameters:

* ``evoparam.io``: read BIDS events and BrainVision recordings (local files, or only the
  needed seconds of a remote file), and cut epochs around stimulations.
* ``evoparam.events``: find stimulation pulses from the signal itself, when the events file
  is missing or wrong (robust threshold, strength gate and template matching), flag
  amplifier saturation, guess the stimulated contact pair, group pulses into trains, and
  score detections against reference onsets.
* ``evoparam.artifact``: remove the stimulation artifact (interpolation, template, recovery
  tail, period-based reconstruction for high-frequency trains, low-rank regression,
  subspace projection, local polynomial, matching pursuit, paradigm-based cancellation,
  learned basis), mark the replaced samples and score each trial.
* ``evoparam.prep``: baseline correction (with a check against the previous pulse's
  delayed responses), re-referencing (bipolar, common average, adjusted common average,
  Laplacian), montage-sensitivity report, trial rejection, resampling, averaging, filters.
* ``evoparam.models``: the fitting engine. ``evoparam.fit(trials, preset="ccep")`` describes
  each channel's response as a background plus components (latency, amplitude, width,
  shape) chosen by model comparison, with uncertainties, conditioning diagnostics,
  classical measures and labels (N1, N2, ...).
* ``evoparam.existence``: is there a response, and how long does it last? Sign-flip
  permutation test (max-t or cluster), cross-projection test, model comparison, and a
  registry of published CCEP detection rules reproduced to the letter; corrections across
  channels; ``evoparam.fit(..., existence=True)`` reports the test next to the fit.
* ``evoparam.trials``: trial-by-trial latency and amplitude of each component (shifted and
  scaled channel model, warm-started), latency jitter against the jitter that noise alone
  gives, realigned averages, and covariates (position in the train, dose-response,
  pre-stimulus phase and power).
* ``evoparam.stats``: the numbers a paper reports. Bootstrap intervals over trials
  (percentile and BCa), jackknife latencies over participants with the retrieved
  individual scores, permutation contrasts between conditions at the right unit (trial,
  channel or subject), split-half reliability and dependability with the trials needed,
  REML variance components, and tidy exports with model formulas for statsmodels, lme4 and
  brms.
* ``evoparam.hierarchical``: one model for trials, channels and subjects. The probability
  that each channel responds, shrunken channel and subject estimates with their standard
  errors, between-channel and between-subject SDs, the trial-to-trial latency jitter, and
  subject covariates (``"latency ~ age"``). Empirical Bayes by default (no extra
  dependency); NUTS with numpyro or PyMC as an option (extra ``bayes``).
* ``evoparam.tf``: time-frequency response (generalised Morse or Morlet wavelets,
  multitaper, STFT), baseline normalisation per trial or on the average, pixel tests (FDR,
  cluster mass, TFCE), the power of the fitted evoked components separated exactly from the
  residual power, data-driven time-frequency zones (non-negative factorisations with the
  rank chosen by split-half stability) and published fixed zones.
* ``evoparam.spatial``: the few response shapes shared across stimulation sites into one
  channel, or across channels for one site (basis profile curves, re-implemented from the
  published description), fits whose shape parameters are shared across channels (with
  per-contact latency offsets and an optional smoothness penalty over neighbouring
  contacts), per (site, channel) parameter maps with coordinates, the (site x channel)
  connectivity export, and a volume-conduction screen (polarity reversal, distance decay,
  reference sensitivity).
* ``evoparam.presets``: eleven presets (ERP, CCEP on ECoG and sEEG, SEP, TEP, ERNA, spinal
  and cochlear ECAP, rodent and primate LFP, P300 BCI) as readable JSON files with their
  sources; ``presets.derive`` makes a variant that records its parent (``docs/presets.md``).
* ``evoparam.reports`` and ``evoparam.plots``: methods paragraph from the provenance,
  JSON export with its schema, quality summary, self-contained HTML, PDF and Markdown
  reports, figures of every result, ``evoparam.reproduce``, and BIDS derivatives
  (``evoparam.io.write_bids_derivatives``).
* The command line: ``evoparam fit <bids_root> --sub 01 --preset ccep --out derivatives/``,
  ``evoparam report``, ``evoparam simulate``, ``evoparam presets list``, ``evoparam cite``,
  ``evoparam reproduce`` (``evoparam --help`` lists them all).

Installation
------------

.. code-block:: bash

    git clone https://github.com/yannbellec/evoparam
    cd evoparam
    pip install -e ".[all]"
    pytest

``import evoparam`` needs only NumPy and SciPy; the extras (``mne``, ``bids``, ``nwb``,
``plot``, ``data``, ...) add readers, figures and tables. ``python tools/run_all_tests.py``
runs the whole test suite, network and slow tests included, on the Python you choose.

Full example on public data
---------------------------

``examples/ccep_pipeline.py`` runs the four steps on one patient of the open CCEP dataset
OpenNeuro ds004080 (it reads about 55 MB of a 2 GB recording):

.. code-block:: bash

    python examples/ccep_pipeline.py

Quickstart
----------

One call fits every channel with a preset and tests whether each channel responds:

.. code-block:: python

    import evoparam

    trials = evoparam.io.read("ds004080", subject="ccepAgeUMCU01")   # epochs of a BIDS
    group = evoparam.fit(trials, preset="ccep")       # dataset; or mne.Epochs, an Evoked
    table = group.to_dataframe()      # one row per channel, condition and component,
                                      # with intervals and the existence test
    print(group.methods_text())       # the methods paragraph, with citations

A NumPy array works too: ``evoparam.fit(data, preset="erp", times=times)`` with ``data`` of
shape (n_channels, n_times) and ``times`` in seconds. Settings that differ from the preset
are passed by name (``evoparam.fit(trials, preset="ccep", max_components=2)``) and are
recorded with the result; ``evoparam.reproduce`` refits a saved result from its record.

Research use
------------

For research use only; not a medical device. evoparam's outputs are not intended for
diagnosis or treatment decisions.

Community
---------

* `CONTRIBUTING.md <CONTRIBUTING.md>`_: development install, tests, coding rules, adding a
  preset, shape, reader (plugins, `docs/contributing/add_shape.rst
  <docs/contributing/add_shape.rst>`_) or dataset, Developer Certificate of Origin.
* `CODE_OF_CONDUCT.md <CODE_OF_CONDUCT.md>`_, `GOVERNANCE.md <GOVERNANCE.md>`_,
  `SUPPORT.md <SUPPORT.md>`_, `SECURITY.md <SECURITY.md>`_, `TRADEMARK.md <TRADEMARK.md>`_.
* Questions: GitHub Discussions or the Neurostars tag ``evoparam``; bugs: the issue tracker.
