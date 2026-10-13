Result tables
=============

The tables returned by ``to_dataframe()`` and written by ``save()`` are tidy: one row
per channel × condition × component, one column per quantity (spec REQ-332). The unit
of every column is in ``table.attrs["units"]`` and, in a saved container, in the
sidecar ``tables/<name>.json`` as ``{"columns": {"<column>": {"unit": ...}}}``.
"data unit" is the amplitude unit of the fitted data (for example uV).
``tests/test_tidy.py`` checks that the columns below are exactly those of the code.

.. _table-component:

``GroupFit.to_dataframe("component")``
---------------------------------------

Also ``ResponseFit.to_dataframe()``. Every reported latency, amplitude, width, onset,
offset and area has its interval in the same row (REQ-006); the existence decision has
its p-value.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - column
     - unit
     - description
   * - ``channel``
     - (none)
     - Channel name.
   * - ``condition``
     - (none)
     - Condition label (empty when the input was not grouped).
   * - ``component``
     - (none)
     - Position of the component in the fit, from 0; -1 on the row of a fit without components.
   * - ``label``
     - (none)
     - Name from the latency windows (for example N1), or C<k> outside every window.
   * - ``shape``
     - (none)
     - Component shape (for example gaussian).
   * - ``latency``
     - s
     - Time of the extremum of the component.
   * - ``amplitude``
     - data unit
     - Signed amplitude at the extremum.
   * - ``polarity``
     - (none)
     - +1 or -1, the sign of the amplitude (0 on the row of a fit without components).
   * - ``fwhm``
     - s
     - Full width at half maximum.
   * - ``onset``
     - s
     - First time the component reaches onset_fraction of its absolute amplitude.
   * - ``offset``
     - s
     - Last time the component is at onset_fraction of its absolute amplitude.
   * - ``area``
     - data unit × s
     - Signed area of the component.
   * - ``rise_time``
     - s
     - Time from onset to the extremum.
   * - ``decay_time``
     - s
     - Time from the extremum to offset.
   * - ``symmetry``
     - (none)
     - (decay - rise) / (decay + rise).
   * - ``sharpness``
     - (none)
     - Curvature of the unit-amplitude shape at the peak times fwhm squared.
   * - ``resolvable``
     - (none)
     - False when the component cannot be told apart from a neighbour (amplitudes or latencies correlated beyond 0.95, or a cancelling pair of opposite signs, spec §5.5.3 step 6): its numbers are not to be read alone. Empty on the row of a fit without components.
   * - ``status``
     - (none)
     - Status of the fit: ok, no_response, failed or skipped.
   * - ``latency_ci_low``
     - s
     - Lower bound of the interval of latency (at the ci_level of the fit).
   * - ``latency_ci_high``
     - s
     - Upper bound of the interval of latency (at the ci_level of the fit).
   * - ``amplitude_ci_low``
     - data unit
     - Lower bound of the interval of amplitude (at the ci_level of the fit).
   * - ``amplitude_ci_high``
     - data unit
     - Upper bound of the interval of amplitude (at the ci_level of the fit).
   * - ``fwhm_ci_low``
     - s
     - Lower bound of the interval of fwhm (at the ci_level of the fit).
   * - ``fwhm_ci_high``
     - s
     - Upper bound of the interval of fwhm (at the ci_level of the fit).
   * - ``onset_ci_low``
     - s
     - Lower bound of the interval of onset (at the ci_level of the fit).
   * - ``onset_ci_high``
     - s
     - Upper bound of the interval of onset (at the ci_level of the fit).
   * - ``offset_ci_low``
     - s
     - Lower bound of the interval of offset (at the ci_level of the fit).
   * - ``offset_ci_high``
     - s
     - Upper bound of the interval of offset (at the ci_level of the fit).
   * - ``area_ci_low``
     - data unit × s
     - Lower bound of the interval of area (at the ci_level of the fit).
   * - ``area_ci_high``
     - data unit × s
     - Upper bound of the interval of area (at the ci_level of the fit).
   * - ``existence_p``
     - (none)
     - p-value of the existence test of the channel (NaN without a test).
   * - ``existence_significant``
     - (none)
     - Decision of the existence test (empty without a test).

.. _table-fit:

``GroupFit.to_dataframe("fit")``
---------------------------------

One row per channel × condition.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - column
     - unit
     - description
   * - ``channel``
     - (none)
     - Channel name.
   * - ``condition``
     - (none)
     - Condition label.
   * - ``status``
     - (none)
     - ok, no_response, failed or skipped.
   * - ``message``
     - (none)
     - Why the status is not ok.
   * - ``n_components``
     - (none)
     - Number of fitted components.
   * - ``r2``
     - (none)
     - Coefficient of determination in the fit window.
   * - ``adj_r2``
     - (none)
     - Adjusted coefficient of determination.
   * - ``rmse``
     - data unit
     - Root mean square residual.
   * - ``mae``
     - data unit
     - Mean absolute residual.
   * - ``nrmse``
     - (none)
     - RMSE divided by the range of the data.
   * - ``aic``
     - (none)
     - Akaike information criterion.
   * - ``aicc``
     - (none)
     - AIC with the small-sample correction.
   * - ``bic``
     - (none)
     - Bayesian information criterion.
   * - ``n_params``
     - (none)
     - Number of free parameters.
   * - ``n_samples``
     - (none)
     - Number of samples in the window.
   * - ``n_eff``
     - (none)
     - Effective number of samples (autocorrelated noise).
   * - ``residual_lag1_autocorr``
     - (none)
     - Lag-1 autocorrelation of the residual.
   * - ``baseline_sd``
     - data unit
     - Standard deviation of the baseline.
