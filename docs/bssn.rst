The BSSN system
===============

JAX BSSN evolves the vacuum Cartesian BSSN system in the ``W`` conformal-factor
formulation. Every array ends with the three spatial axes ``(Nx, Ny, Nz)``.
Tensor and vector component axes precede those spatial axes.

Evolved variables
-----------------

The field order is a numerical contract because RK4 and FMR combine states by
tuple position.

.. list-table:: ``BSSNVariables`` layout
   :header-rows: 1
   :widths: 22 20 26 32

   * - Field
     - Symbol
     - Shape
     - Meaning
   * - ``conformal_metric``
     - :math:`\tilde{\gamma}_{ij}`
     - ``(3, 3, Nx, Ny, Nz)``
     - Symmetric conformal spatial metric, projected to determinant one.
   * - ``conformal_factor``
     - :math:`W`
     - ``(Nx, Ny, Nz)``
     - Conformal factor with :math:`\gamma_{ij}=W^{-2}\tilde{\gamma}_{ij}`.
   * - ``traceless_K``
     - :math:`\tilde{A}_{ij}`
     - ``(3, 3, Nx, Ny, Nz)``
     - Conformal trace-free extrinsic curvature.
   * - ``trace_K``
     - :math:`K`
     - ``(Nx, Ny, Nz)``
     - Trace of the physical extrinsic curvature.
   * - ``conformal_connection``
     - :math:`\tilde{\Gamma}^{i}`
     - ``(3, Nx, Ny, Nz)``
     - Evolved conformal connection functions.
   * - ``lapse``
     - :math:`\alpha`
     - ``(Nx, Ny, Nz)``
     - Lapse function.
   * - ``shift``
     - :math:`\beta^{i}`
     - ``(3, Nx, Ny, Nz)``
     - Contravariant shift vector.

Equation ownership
------------------

.. list-table:: Time-derivative ownership
   :header-rows: 1
   :widths: 32 38 30

   * - Evolved field
     - Function
     - Module
   * - :math:`\tilde{\gamma}_{ij}`
     - ``evolve_conformal_metric``
     - ``bssn.spatial_metric``
   * - :math:`W`
     - ``evolve_conformal_factor``
     - ``bssn.spatial_metric``
   * - :math:`\tilde{A}_{ij}`
     - ``evolve_traceless_extrinsic_curvature``
     - ``bssn.extrinsic_curvature``
   * - :math:`K`
     - ``evolve_trace_extrinsic_curvature``
     - ``bssn.extrinsic_curvature``
   * - :math:`\tilde{\Gamma}^{i}`
     - ``evolve_conformal_connection``
     - ``bssn.conformal_connection``
   * - :math:`\alpha`
     - ``evolve_lapse``
     - ``bssn.shift_and_lapse``
   * - :math:`\beta^{i}`
     - ``evolve_shift``
     - ``bssn.shift_and_lapse``

Shared geometry
---------------

``bssn.geometry`` reconstructs the physical metric, packs and unpacks
symmetric tensors, and computes the curvature and lapse-Hessian sources shared
by the extrinsic-curvature equations.

The physical metric uses

.. math::

   \gamma_{ij} = W^{-2}\tilde{\gamma}_{ij}.

Inverse powers used during physical-metric reconstruction and conformal-
connection evolution apply ``W_FLOOR_VALUE = 1e-12``. This protects only the
inverse operation: the evolved ``W`` array is not clamped or overwritten.

The curvature source is assembled directly as the denominator-free scaled
tensor :math:`W^2 R_{ij}`. The lapse source is likewise assembled as
:math:`W^2 D_iD_j\alpha`. The traceless-curvature RHS consumes these scaled
quantities without first forming a singular unscaled tensor near a puncture.

Gauge choices
-------------

``BSSNParameters.gauge`` selects the lapse source inside JIT-compatible control
flow:

* ``0``: harmonic slicing,
  :math:`\partial_t\alpha-\beta^i\partial_i\alpha=-\alpha^2K`.
* ``1``: 1+log slicing,
  :math:`\partial_t\alpha-\beta^i\partial_i\alpha=-2\alpha K`.

The shift uses a single-variable Gamma driver with advection and linear
damping. ``zero_shift = 1`` suppresses the shift RHS at every RK stage;
otherwise the configured driver is evolved.

Algebraic constraints
---------------------

The timestep enforces

.. math::

   \det\tilde{\gamma}=1,
   \qquad
   \tilde{\gamma}^{ij}\tilde{A}_{ij}=0

before each RHS evaluation and after the final RK4 combination. These
projections are distinct from the raw Hamiltonian, momentum, Gamma,
determinant, and trace constraint fields described in :doc:`diagnostics`.

API reference
-------------

.. automodule:: JAX_BSSN.bssn.variables
   :members:
   :undoc-members:

.. automodule:: JAX_BSSN.bssn.geometry
   :members:

.. automodule:: JAX_BSSN.bssn.spatial_metric
   :members:

.. automodule:: JAX_BSSN.bssn.extrinsic_curvature
   :members:

.. automodule:: JAX_BSSN.bssn.conformal_connection
   :members:

.. automodule:: JAX_BSSN.bssn.shift_and_lapse
   :members:

.. automodule:: JAX_BSSN.bssn.tensor_algebra
   :members:
