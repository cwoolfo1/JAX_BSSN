Constraints and diagnostics
===========================

Diagnostics observe solver state. They do not participate in the RK4 call
graph, with one deliberate ownership exception: raw mathematical constraints
live in the BSSN package because the momentum constraint is used by an
evolution equation.

Raw constraint fields
---------------------

``JAX_BSSN.bssn.constraints`` produces spatially resolved arrays:

.. list-table:: Constraint fields
   :header-rows: 1
   :widths: 24 22 54

   * - Field
     - Shape
     - Meaning
   * - ``hamiltonian``
     - ``(Nx, Ny, Nz)``
     - Vacuum Hamiltonian residual, using the denominator-free trace of
       :math:`W^2R_{ij}`.
   * - ``momentum``
     - ``(3, Nx, Ny, Nz)``
     - Covariant momentum residual used by both diagnostics and evolution.
   * - ``det_gamma``
     - ``(Nx, Ny, Nz)``
     - :math:`\det\tilde{\gamma}-1`.
   * - ``trace_A``
     - ``(Nx, Ny, Nz)``
     - :math:`\tilde{\gamma}^{ij}\tilde{A}_{ij}`.
   * - ``gamma_condition``
     - ``(Nx, Ny, Nz)`` currently
     - Placeholder returned by the current Gamma-constraint implementation.

The Gamma constraint is not yet implemented and currently returns a scalar
zero field with the lapse shape. A reported zero Gamma norm must not be
interpreted as a computed physical residual.

Reduction and reporting
-----------------------

``JAX_BSSN.diagnostics.constraints`` reduces raw fields to L2 and infinity
norms, prints summaries, and provides a basic finite-value/constraint-threshold
health check. These Python-side decisions remain outside the evolution
equations.

Plotting and NumPy snapshots
----------------------------

``diagnostics.plotting`` provides central-slice plots, constraint-history
plots, ``.npz`` state snapshots, and a callback-based text writer. Plotting and
filesystem output are side effects and should be called outside compiled
timesteps.

openPMD output
--------------

``OpenPMDWriter`` writes synchronous Cartesian mesh iterations. Scalar fields
are three-dimensional arrays. Vector fields are three-component tuples or
lists whose components share one three-dimensional shape.

Mesh metadata preserves:

* C data order.
* Axis labels ``x``, ``y``, and ``z`` in array order.
* ``grid_spacing``, ``grid_global_offset``, and ``grid_position``.
* Scalar record components and vector component names ``x``, ``y``, ``z``.
* Floating-point dtype where possible.

``ghost_cells`` may be a scalar or three-tuple and is stripped before geometry
and output are established.

For refinement output, ``write_levels`` defaults to one composite finest-grid
mesh for each physical field. All levels must have matching field names,
matching scalar/vector kinds, aligned physical offsets, integer spacing ratios,
and a shared grid position. Coarse vertex values are repeated piecewise
constantly into finest-index space, then finer chunks overwrite the covered
region. ``composite=False`` retains separate level-prefixed mesh records.

Writes are synchronous. The implementation retains host buffers until each
openPMD flush completes; there is no asynchronous output queue.

API reference
-------------

.. automodule:: JAX_BSSN.bssn.constraints
   :members:

.. automodule:: JAX_BSSN.diagnostics.constraints
   :members:

.. automodule:: JAX_BSSN.diagnostics.openpmd
   :members:

.. automodule:: JAX_BSSN.diagnostics.plotting
   :members:
