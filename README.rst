=========================================
evoparam
=========================================

evoparam is a tool to parameterize neural evoked responses: it describes each
response (ERP, CCEP, ...) as a small set of peaks with a latency, an amplitude and a width.

Status: early development (version 0.1, working name). Origin of the code and licence:
see ``NOTICE`` and ``LICENSE``.

What it does
------------

evoparam covers the whole path from a raw recording to response parameters:

* ``evoparam.io``: read BIDS events and BrainVision recordings (local files, or only the
  needed seconds of a remote file), and cut epochs around stimulations.
* ``evoparam.detect``: find stimulation onsets from the signal itself, when the events file
  is missing or wrong, and score detections against reference onsets.
* ``evoparam.clean``: remove the stimulation artifact (interpolation, for now).
* ``evoparam.param``: describe each response as a set of peaks (latency, amplitude, width).

Methods planned for the next versions (trial-by-trial fitting, uncertainty, more artifact
removal methods) have their place in this structure, but are not available yet.

Installation
------------

.. code-block:: bash

    git clone https://github.com/yannbellec/evoparam
    cd evoparam
    pip install -e ".[all]"
    pytest

Full example on public data
---------------------------

``examples/ccep_pipeline.py`` runs the four steps on one patient of the open CCEP dataset
OpenNeuro ds004080 (it reads about 55 MB of a 2 GB recording):

.. code-block:: bash

    python examples/ccep_pipeline.py

Quickstart: fitting the model directly
--------------------------------------

**Fitting a Single Evoked Response**

With an event-related potential (ERP) loaded (with 'time' storing time points, and 'signal' storing
the ERP, evoparam can be used as follows:

.. code-block:: python

    # Import the evoparam object
    from evoparam import evoparam

    # Initialize evoparam object
    params = evoparam()

    # Define time range within which to model the ERP
    time_range = [0, 1]

    # Model the ERP with evoparam, and print out a report
    params.report(time, signal, time_range)

evoparam.report() fits the model, plots the original (or rectified) ERP with the associated evoparam model fit,
and prints out the parameters of the model fit.

**Defining the model Settings**

The settings for the algorithm are:

* ``peak_width_limits`` sets the possible lower- and upper-bounds for the fitted peak widths.
* ``max_n_peaks`` sets the maximum number of peaks to fit.
* ``min_peak_height`` sets an absolute limit on the minimum height for any extracted peak.
* ``peak_threshold`` sets a relative threshold above which a peak height must cross to be included in the model.

These settings can be defined when initializing the model, for example:

.. code-block:: python

    # Initialize an evoparam model object with defined settings
    params = evoparam(peak_width_limits=(0.01, 0.5), peak_threshold=2, min_peak_height=0, max_n_peaks=3,)


**Fitting a Group of ERPs**

Next is an example workflow for fitting a group of ERPs.
In this case, 'time' is again a 1D array of time points, and 'signal' is a 2D array of ERPs.
We can fit the group of ERPs by doing:

.. code-block:: python

    # Initialize a evoparamGroup object, specifying some parameters
    params = evoparamGroup(peak_width_limits=(0.01, 0.5), peak_threshold=2, min_peak_height=0, max_n_peaks=3)

    # Fit evoparam model across the matrix of ERPs
    params.fit(time, signals)

    # Create and save out a report summarizing the results across the group of ERPs
    params.save_report('example_report')

    # Save out evoparamGroup results for further analysis later
    params.save(file_name='group_results', save_results=True)
