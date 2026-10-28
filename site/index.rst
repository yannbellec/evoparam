evoparam
========

*evoparam: evoked response parameterization.*

evoparam describes an evoked response as a few bumps on a slow background, and gives the
timing, height and width of each bump.

Install
-------

.. code-block:: bash

   pip install "evoparam[plot] @ git+https://github.com/yannbellec/evoparam"

In a notebook (Jupyter, Colab), run the same line in a cell with ``!`` in front, then
restart the session.

To update evoparam later, reinstall evoparam alone and leave the other packages as they
are:

.. code-block:: bash

   pip install -U --force-reinstall --no-deps "evoparam @ git+https://github.com/yannbellec/evoparam"

Do not use ``--force-reinstall`` without ``--no-deps``: it reinstalls numpy, pillow and the
other dependencies at their newest versions, which breaks packages that Colab installs
with tighter limits (numba, gradio).

Get started
-----------

:doc:`auto_tutorials/plot_01_model_description`

:doc:`auto_tutorials/plot_02_reading_results`

:doc:`auto_tutorials/plot_03_fitting_algorithm`

:doc:`auto_tutorials/plot_04_component_fitting`

:doc:`auto_tutorials/plot_05_background_fitting`

:doc:`auto_tutorials/plot_06_metrics`

:doc:`auto_tutorials/plot_07_fit_object`

:doc:`auto_tutorials/plot_08_groups_real_data`

:doc:`auto_tutorials/plot_09_trial_by_trial`

:doc:`auto_tutorials/plot_10_tuning_the_fit`

:doc:`auto_tutorials/plot_11_further_analysis`

:doc:`auto_tutorials/plot_12_reporting`

.. toctree::
   :hidden:

   auto_tutorials/index
