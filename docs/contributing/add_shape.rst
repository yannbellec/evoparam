Adding a shape, a preset or a reader as a plugin
================================================

evoparam finds extensions through Python *entry points* (spec §10.8): a separate package
declares, in its ``pyproject.toml``, the objects it adds, and evoparam loads them the first
time its registries (``evoparam.shapes.SHAPES``, ``evoparam.presets.PRESETS``,
``evoparam.io.read``) are used. Nothing in evoparam changes.

The example below is the dummy plugin of the test suite,
``evoparam/tests/plugins/evoparam_dummy``.

1. The package
--------------

.. code-block:: toml

   [project]
   name = "evoparam-dummy"
   version = "0.1.0"
   dependencies = ["evoparam", "numpy"]

   [project.entry-points."evoparam.shapes"]
   lorentzian = "evoparam_dummy:Lorentzian"

   [project.entry-points."evoparam.presets"]
   dummy_lorentz = "evoparam_dummy:DUMMY_LORENTZ"

   [project.entry-points."evoparam.readers"]
   dummy = "evoparam_dummy:read_dummy"

The entry-point name must be the object's ``name`` (shapes, presets) and must not be the
name of a built-in or of another installed plugin: evoparam raises ``SettingsError`` naming
both distributions.

2. A shape
----------

A shape satisfies the ``evoparam.shapes.ComponentShape`` protocol: ``name``,
``param_names``, ``n_params``, ``latency_param``, and the methods ``evaluate`` (unit
peak), ``jacobian`` (analytic), ``default_bounds``, ``initial_guess`` and ``features``. It
is a dataclass whose fields may include ``min_fwhm`` and ``max_fwhm`` (filled from the fit
settings). The dummy plugin's Lorentzian:

.. code-block:: python

   @dataclasses.dataclass(frozen=True)
   class Lorentzian:
       name: ClassVar[str] = "lorentzian"
       param_names: ClassVar[tuple] = ("mu", "gamma")
       n_params: ClassVar[int] = 2
       latency_param: ClassVar[str] = "mu"
       min_fwhm: float | None = None
       max_fwhm: float | None = None

       def evaluate(self, times, params):          # 1 / (1 + ((t - mu) / gamma)^2)
           ...

When loaded, the shape goes through the checks of ``register_shape``: protocol, finite
bounds, output shapes, unit amplitude, analytic Jacobian against finite differences
(relative error below 1e-6), initial guess inside the bounds, and the feature fields. Run
them while developing:

.. code-block:: python

   from evoparam.shapes import check_shape
   check_shape(Lorentzian())

A background is declared in the same group with ``kind = "background"`` and the
``evoparam.shapes.Background`` protocol.

3. A preset
-----------

An instance of ``evoparam.presets.Preset``, most simply derived from a built-in one:

.. code-block:: python

   DUMMY_LORENTZ = presets.derive("ccep", name="dummy_lorentz", shapes=("lorentzian",),
                                  validation="simulation_only")

4. A reader
-----------

A callable taking the path first and returning a ``TrialSet``, with the extensions it
reads:

.. code-block:: python

   def read_dummy(path, *, picks=None):
       ...
       return TrialSet(data=data, times=times, sfreq=sfreq, ch_names=names)

   read_dummy.extensions = (".dummy",)

``evoparam.io.read("rec.dummy")`` then calls it (or ``read(path, format="dummy")``). The
built-in extensions (``.vhdr``, ``.edf``, ``.bdf``, ``.nwb``, ``.mefd``, ``.evoparam``,
``.zip``) cannot be claimed.

5. Install and check
--------------------

.. code-block:: console

   $ pip install -e evoparam/tests/plugins/evoparam_dummy
   $ evoparam info --json        # "plugins" lists the three entries

Every plugin object a result used is recorded in ``provenance.environment["plugins"]``
(``{"lorentzian": "evoparam-dummy==0.1.0"}``), so that ``evoparam reproduce`` can name a
missing plugin. ``EVOPARAM_DISABLE_PLUGINS=1`` turns every plugin off.
