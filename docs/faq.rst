Frequently asked questions
==========================

.. contents::
   :local:
   :depth: 1

Which preset for my data?
-------------------------

Pick the preset of your recording and paradigm; every preset is a starting point whose values
are cited (:doc:`presets`), and every setting can be overridden by keyword.

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - preset
     - data
   * - ``ccep``
     - cortico-cortical evoked potentials from single-pulse electrical stimulation (SPES),
       ECoG grids and strips
   * - ``ccep_seeg``
     - the same with stereo-EEG depth electrodes (bipolar montage)
   * - ``erp``
     - scalp EEG or MEG event-related potentials
   * - ``p300_bci``
     - scalp P300 in brain-computer-interface paradigms
   * - ``sep``
     - somatosensory evoked potentials, scalp or epidural
   * - ``tep``
     - TMS-evoked potentials (TMS-EEG)
   * - ``erna``
     - evoked resonant neural activity in deep-brain-stimulation LFPs
   * - ``ecap``
     - evoked compound action potentials of spinal-cord stimulation
   * - ``ecap_ci``
     - electrically evoked compound action potentials of cochlear implants
   * - ``rodent_ep``
     - evoked potentials in mouse or rat LFP/ECoG
   * - ``primate_ep``
     - evoked potentials in macaque LFP/ECoG

The tutorial of each preset (:doc:`auto_tutorials/index`) runs it on data like yours, and
:doc:`preset_sensitivity` shows how much each preset's results move when its windows change by
±50 %. When no preset fits, start from the closest one with
``dataclasses.replace(evoparam.presets.PRESETS["erp"], name="mine", ...)`` and say so in the
methods: the paragraph lists every changed setting.

Why is polarity a separate field?
---------------------------------

Intracranial responses change sign with the montage and with the depth of the contact
relative to the generator: the same N1 can be negative on one contact and positive on the
next. evoparam fits signed data, but reports ``amplitude`` as the signed value at the extremum
*and* ``polarity`` (+1 or -1) as its own field, so that you can compare magnitudes across
contacts and montages (``abs(amplitude)``) and still keep the sign. Every preset for
intracranial data searches both signs; labels such as ``N1`` state the polarity they expect.

Why is ``r2`` not the inherited correlation?
--------------------------------------------

The ancestor tool reported ``r_squared_`` as the squared Pearson correlation between the data
and the model. A correlation ignores offsets and scale: a model at the wrong amplitude can
have a correlation of 1. evoparam's ``gof.r2`` is the coefficient of determination,
``1 - SS_residual / SS_total`` on the fit window, which drops when the model misses the data's
amplitude or offset. The goodness of fit also holds ``rmse``, ``nrmse``, the information
criteria (``aic``, ``aicc``, ``bic``) and the effective sample count used to compute them.

My N1 is missing
----------------

Check, in this order:

1. ``status``: ``no_response`` means the existence test found no response on that channel at
   the preset's alpha (after the correction across channels); the fit is not reported. See
   ``result.fits[i].existence`` for the p-value.
2. the channel itself: stimulating contacts (``ch_info.stim_contact``, or the pair named in
   ``metadata["stim_site"]``) and channels marked ``bad`` are left out of the common
   reference and carry the stimulation artifact; their fits are not N1s;
3. the artifact window (``ccep``: -2 to 8 ms) and the fit window (``ccep``: from 9 ms) must
   leave the N1 visible; an N1 earlier than 9 ms is cut;
4. the label window: ``N1`` is a negative component in 9 to 50 ms for ``ccep``; a component
   found outside it, or positive after your montage, is fitted but not labelled ``N1``
   (look at the unlabelled rows of ``to_dataframe()``);
5. the montage: ``evoparam.prep.reference_sensitivity`` shows which responses survive a
   change of reference;
6. the number of trials: with fewer than about 10 trials the existence test has little power.

GPU gives slightly different numbers
------------------------------------

This version fits on the CPU only (NumPy and SciPy); the GPU backends of the specification
are not in it, and ``Provenance.environment["backend"]`` records ``"numpy"``. When they come,
a GPU fit must equal the CPU fit within 1e-6 relative on every parameter, or it is a bug to
report. Across machines on the CPU, results agree within 1e-9 relative (the reference values
of ``evoparam/tests/reference``); differences at that level come from the linear-algebra
library and the number of threads, and are recorded in the provenance.

How do I add a shape?
---------------------

Write a class that satisfies the ``evoparam.shapes.ComponentShape`` protocol (``name``,
``param_names``, ``n_params``, ``latency_param``, and the methods ``evaluate``,
``jacobian``, ``default_bounds``, ``initial_guess`` and ``features``) and
register it, either at run time with ``evoparam.shapes.register_shape`` or from your own
package through the ``evoparam.shapes`` entry point. :doc:`contributing/add_shape` walks
through a complete plugin.

Is this a medical device?
-------------------------

No. evoparam is for research use only: it is not intended for diagnosis, monitoring or
treatment decisions, and makes no clinical claim. Software that informs a clinical decision
is regulated as a medical device (for example, evoked-potential analysis systems are class II
devices in the United States, and fall under Rule 11 of the European MDR); using evoparam
that way would need a separate, regulated product. Section 13 of the specification
(``docs/spec/SPEC.md``, "Valorisation", §13.3 "Regulatory path") explains where the line
lies.
