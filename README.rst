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
