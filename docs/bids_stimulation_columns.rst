Stimulation columns proposed for BIDS
=====================================

BIDS describes electrical stimulation in iEEG through the sidecar fields
``ElectricalStimulation`` (boolean) and ``ElectricalStimulationParameters`` (free text) and
three example ``events.tsv`` columns, ``electrical_stimulation_type``, ``_site`` and
``_current``, which are not in the column registry of the schema. Units, pulse width,
polarity, pulse count, frequency and charge have no standard column, and deposits encode
them differently (ds004080 writes currents in A and pulse widths in s without units; the
Mayo Clinic datasets ds003708 and ds004977 write ``"6.0 mA"`` and give no pulse width).

Until a BIDS Extension Proposal defines them, evoparam writes these fields in its
derivatives with the ``evoparam_`` prefix and defines each one in the JSON sidecar
(``LongName``, ``Description``, ``Units``, ``Levels``). The standard example columns are
copied unchanged when the source has them. ``evoparam.io.harmonize_stim_events`` reads
every shape of deposit into the same columns, and ``evoparam validate-bids`` reports, run by
run, which fields a dataset states.

Proposed columns
----------------

======================================  =======  ==============================================
column                                  unit     meaning
======================================  =======  ==============================================
``evoparam_stim_current_mA``            mA       current amplitude
``evoparam_stim_pulse_width_us``        us       width of one phase
``evoparam_stim_polarity``              —        vocabulary below
``evoparam_stim_n_pulses``              —        pulses in the train of the event (integer)
``evoparam_stim_frequency_hz``          Hz       pulse frequency in a train; n/a for single
                                                 pulses
``evoparam_stim_charge_uC``             uC       charge per phase: current x pulse width
``evoparam_stim_type_raw``              —        the source value, kept when its phase order
                                                 could not be mapped
======================================  =======  ==============================================

Polarity vocabulary (``Levels``):

* ``monophasic_cathodic``: one cathodic phase;
* ``monophasic_anodic``: one anodic phase;
* ``biphasic_cathodic_first``: two phases, cathodic first;
* ``biphasic_anodic_first``: two phases, anodic first;
* ``alternating``: polarity alternating across pulses;
* ``unknown``: the phase order is not stated by the source.

Unit rules used to read deposits
--------------------------------

* Current: ``Units`` of the sidecar (``A``, ``mA``, ``uA``), else a unit written with the
  value (``"6.0 mA"``), else values below 0.1 are read as A (warning ``unit_inferred``).
* Pulse width: ``Units`` (``s``, ``ms``, ``us``), else inline, else values below 0.01 are
  read as s.
* Polarity: mapped when the phase order is stated in the value or in
  ``ElectricalStimulationParameters``; otherwise ``unknown``.

Proposal to the BIDS community
------------------------------

The column set, its units and the polarity vocabulary are offered as a worked proposal to
the BEP021 (electrophysiology derivatives) and BEP037 (non-invasive brain stimulation)
groups.

BEP issue: not opened yet. Opening it is an action of the project author on the BIDS
repositories; its link replaces this sentence when it exists.
