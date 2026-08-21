Demos
=====

The supported physical examples are exactly gauge wave, linear wave, and
single puncture. Each demo owns its coordinates, initial data, parameters, and
run loop. Run commands from an editable checkout; there is no generic JAX BSSN
CLI or initial-data dispatcher.

Install demo support first:

.. code-block:: bash

   python -m pip install -e ".[demos]"

Gauge wave
----------

The gauge-wave demo evolves the analytic positive-x harmonic gauge wave on a
periodic uniform grid with harmonic slicing and fixed zero shift. It reports
RMS errors against the analytic solution at the final time.

.. code-block:: bash

   python demos/gauge_wave/gauge_wave.py

For a shorter exploratory run, call its run function explicitly:

.. code-block:: python

   from demos.gauge_wave.gauge_wave import run_gauge_wave

   state, run = run_gauge_wave(
       grid_size=32,
       final_time=0.1,
       show_progress=True,
   )

The default configuration uses ``grid_size=60``, amplitude ``0.1``, wavelength
``1``, CFL ``1``, and one wavelength of evolution.

Linear wave with FMR
--------------------

The linear-wave demo evolves a plus-polarized wave through one centered 2:1
refinement patch. It uses stage-synchronous coarse/fine RK4, coarse MAD, and
ordinary fourth-order fine derivatives. Diagnostics split error into native
fine, uncovered coarse, and interface regions.

.. code-block:: bash

   python demos/linear_wave/linear_wave.py

The default run writes a composite openPMD series to
``demos/linear_wave/output/linear_wave_fmr.h5``. Configure it from Python:

.. code-block:: python

   from demos.linear_wave.linear_wave import run_linear_wave

   coarse, fine, run = run_linear_wave(
       grid_size=16,
       final_time=0.02,
       output_iterations=2,
       output_path="output/linear_wave_smoke.h5",
       show_progress=False,
   )

``grid_size`` must be divisible by four so the centered inclusive patch bounds
land on coarse vertices. The demo rejects a final normalized coarse or fine
RMS wave error at or above five percent.

Single puncture
---------------

The single-puncture demo constructs time-symmetric Schwarzschild puncture data
with :math:`W=\psi^{-2}` and evolves it with 1+log lapse and an evolved
Gamma-driver shift. Its grid avoids sampling :math:`R=0` directly.

Run from the demo directory so its snapshots and the movie helper agree on the
output location:

.. code-block:: bash

   cd demos/single_puncture
   python single_puncture.py

The default run writes per-field ``.npy`` snapshots and
``constraint_l2.txt`` under ``output/``, plus a final field plot. Grid size,
domain width, CFL, final time, and snapshot cadence are explicit near the top
of ``main``.

After a run, optional MP4 rendering requires an ``ffmpeg`` executable visible
to Matplotlib:

.. code-block:: bash

   python make_movies.py

The renderer produces lapse, shift, and conformal-factor movies from the
synchronized snapshots.

Resource expectations
---------------------

The default three-dimensional runs compile substantial JAX kernels and can
use significant memory. Start with the shorter function calls above when
checking a new environment. Enable ``jax_enable_x64`` before constructing
state if a custom driver requires double precision; the provided demo modules
do this at import time.
