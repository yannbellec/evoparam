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

.. toctree::
   :hidden:

   auto_tutorials/index
