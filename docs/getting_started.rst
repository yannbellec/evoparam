Getting started
===============

Install
-------

evoparam needs Python 3.10 or later, NumPy and SciPy. Everything else is an optional extra,
installed only when you need it (the error message of a function that needs one names it):

.. code-block:: bash

   pip install evoparam                 # core: numpy and scipy only
   pip install "evoparam[mne]"          # one extra (mne, bids, nwb, cli, plot, ...)
   pip install "evoparam[all]"          # every extra that needs neither a GPU nor a sampler

Until the first public release, install from a source checkout with ``pip install .``.
``python -c "import evoparam; evoparam.selftest()"`` fits one bundled synthetic CCEP and
checks the result: it is the check run after every installation.

A ten-line example
------------------

.. code-block:: python

   import evoparam
   from evoparam import sim

   trials, truth = sim.simulate_ccep(n_trials=20, n_channels=4, seed=0, artifact=False,
                                     p_response=1.0)            # or evoparam.read(path)
   result = evoparam.fit(trials, preset="ccep", random_state=0)  # one fit per channel
   table = result.to_dataframe()                                 # one row per component
   print(table[["channel", "label", "latency", "amplitude", "fwhm", "polarity"]])
   print(result.fits[0].gof.r2, result.fits[0].status)
   print(result.methods_text())                                  # the methods paragraph

``evoparam.fit`` reads a :class:`~evoparam.data.TrialSet` (trials × channels × time), an
:class:`~evoparam.data.Evoked`, or MNE objects. With a preset it applies the whole chain the
preset describes (baseline, montage, existence test, model fit, labels); every setting can be
overridden by keyword. Real recordings come from ``evoparam.read`` (BrainVision, EDF, BIDS,
NWB, MEF, Neo, MNE) or ``evoparam.io.read_bids``.

What the numbers mean
---------------------

Each channel's average becomes a background plus a small number of *components*. For each
component:

``latency``
   time of the component's extremum, in seconds after the event onset;
``amplitude``
   signed value at the extremum, in the data's units (``table.attrs["units"]``);
``polarity``
   +1 or -1, kept apart from the amplitude so that responses that change sign with the
   montage can be compared (see :doc:`faq`);
``fwhm``
   full width at half maximum, in seconds; ``onset`` and ``offset`` are where the component
   reaches 10 % of its peak amplitude;
``label``
   the name given by the preset's latency windows (``N1``, ``N2``, ``P300`` ...), empty when
   the component falls in no window;
``se`` and ``ci``
   standard errors and confidence intervals of every parameter.

For each channel, ``status`` says whether the fit is usable (``ok``), whether the existence
test found no response (``no_response``) or whether the fit failed (``failed``, with the
reason in ``message``); ``gof`` holds the goodness of fit (``r2`` is the coefficient of
determination, not a squared correlation), ``existence`` the test of whether there is a
response at all, and ``classical`` the classical ERP measures (peak, mean amplitude, area)
computed on the same window. :doc:`glossary` defines every field.

Next
----

The tutorials (one per preset and one per integration) start from data like yours; the
:doc:`reporting` page lists what to report and how evoparam writes it for you.
