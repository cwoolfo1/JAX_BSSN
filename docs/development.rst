Development guide
=================

Local setup
-----------

Create an isolated editable environment from the repository root:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e ".[demos,test]"
   python -m pip install -r docs/requirements.in

Use the JAX installation procedure appropriate for the intended accelerator if
the ordinary dependency resolver does not select the required build.

Run tests
---------

The complete suite includes unittest classes and pytest-style function tests:

.. code-block:: bash

   python -m pytest

Useful focused runs include:

.. code-block:: bash

   python -m pytest tests/test_bssn_equation_regressions.py
   python -m pytest tests/test_evolve.py
   python -m pytest tests/test_refinement.py tests/test_fmr_linear_wave.py
   python -m pytest tests/test_openpmd.py

Gauge-wave and FMR convergence tests compile and evolve three-dimensional
states, so they take longer than algebraic unit tests.

Build documentation
-------------------

Treat every Sphinx warning as an error:

.. code-block:: bash

   sphinx-build -W --keep-going -b html docs docs/_build/html
   sphinx-build -W --keep-going -b linkcheck docs docs/_build/linkcheck

For live editing, install ``sphinx-autobuild`` separately and run:

.. code-block:: bash

   sphinx-autobuild docs docs/_build/html --port 8008

Numerical-change checklist
--------------------------

Before changing an equation, stencil, boundary, projection, or transfer:

* Record representative unigrid and FMR inputs and outputs.
* Check the ordering of ``BSSNVariables`` leaves.
* Verify derivative axes for scalar, vector, and tensor leading dimensions.
* Preserve the distinction between Sommerfeld RHS treatment and
  Super-Gaussian state filtering.
* Check every RK stage, not only the final state.
* Measure gauge-wave and FMR linear-wave convergence.
* Compare array dtypes and weak-type behavior.

Repository ownership
--------------------

Add mathematical quantities to ``bssn`` only when they belong to the BSSN
system or are shared by multiple equations. Put timestep ordering and boundary
placement in ``evolution``. Keep FMR transfers in ``fmr/refinement.py`` until
multiple patches or subcycling create a genuine split. Put reductions and
side effects in ``diagnostics``.

Physical initial data belongs in a specific demo. Test states belong in
``tests/initial_data.py``. Do not restore an installed initial-data dispatcher
or make tests import executable demos.

Debugging JAX kernels
---------------------

Start with the CPU backend and a small grid. Disable outer Python progress and
output while isolating a kernel. Because compilation is shape-sensitive, keep
field rank and component-axis order identical to production when constructing
a reduced case. Call ``jax.block_until_ready`` around timing or before checking
for asynchronous device errors.
