Installation
============

Requirements
------------

JAX BSSN is a Python package built around JAX and NumPy. Matplotlib supports
the plotting diagnostics, and ``openPMD-api`` supports mesh output. Install a
JAX build appropriate for the intended CPU or accelerator before running a
large simulation.

Editable development setup
--------------------------

From the repository root:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e .

Install the test-only dependencies, including pytest and SciPy, with:

.. code-block:: bash

   python -m pip install -e ".[test]"

SciPy is not a runtime solver dependency. It is currently used only by the
derivative regression tests.

Documentation dependencies
--------------------------

.. code-block:: bash

   python -m pip install -r docs/requirements.in
   sphinx-build -W --keep-going -b html docs docs/_build/html

Read the Docs uses Ubuntu 24.04 and Python 3.12 as configured in
``.readthedocs.yaml``.
