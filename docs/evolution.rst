Evolution and boundaries
========================

The evolution package owns the discretization and the placement of boundary
and projection operations around the BSSN equations. See :doc:`architecture`
for the package-level dependency flow.

First derivatives and MAD
-------------------------

``diff1_field`` treats ``direction`` as an absolute array axis. Scalar fields
therefore use directions 0, 1, and 2; vectors and tensors add their leading
component axes before selecting a spatial direction.

With ``mad_q = 1`` the result is exactly the historical centered fourth-order
operator

.. math::

   D_4 f_i = \frac{2}{3h}(f_{i+1}-f_{i-1})
             -\frac{1}{12h}(f_{i+2}-f_{i-2}).

For other values, mesh-adapted differencing blends that operator with the
centered sixth-order-accurate first derivative:

.. math::

   D_{\mathrm{MAD}} = qD_4 + (1-q)D_6.

FMR derives the coarse value of ``q`` from the spacing ratio and forces the
fine value to one. On a physical Sommerfeld axis, first derivatives
deliberately retain the fourth-order operator because nonperiodic closures for
the wider MAD stencil are not implemented.

Kreiss--Oliger dissipation
--------------------------

``diff6_field`` computes the centered seven-point sixth derivative. Each
evolution equation adds the existing dissipation term

.. math::

   \frac{\nu}{64}h^5
   \left(\partial_x^6+
         \partial_y^6+
         \partial_z^6\right)u.

Periodic axes use wrapped stencils. Sommerfeld faces replace the outer closure
points with the implemented lopsided coefficients. Leading component axes are
preserved.

Boundary codes
--------------

Each low/high face has an integer code in ``BSSNParameters``:

.. list-table:: Boundary families
   :header-rows: 1
   :widths: 18 25 57

   * - Code
     - Name
     - Numerical role
   * - ``0``
     - Periodic
     - Centered derivative stencils wrap with ``jnp.roll``.
   * - ``1``
     - Super-Gaussian
     - Filter the state toward its flat-space value in a finite-width layer.
   * - ``2``
     - Sommerfeld
     - Use nonperiodic derivative closures and replace the assembled RHS on
       the selected outer plane.

The Super-Gaussian weight combines all active faces, including edges and
corners, and uses ``bc_width``, ``bc_order``, and ``bc_strength``. Metric
diagonal components approach one; off-diagonal metric components,
extrinsic-curvature variables, connections, and shift approach zero; ``W`` and
the lapse approach one.

Sommerfeld uses the coordinate origin supplied through ``x_min``, ``y_min``,
and ``z_min``. It evaluates

.. math::

   \partial_t u = -\left(\partial_r u + \frac{u-u_0}{r}\right)

on the union of configured face planes. The union mask ensures an edge or
corner is replaced once, independent of face ordering.

RHS assembly
------------

``compute_bssn_rhs`` calls the seven equation owners in ``BSSNVariables``
field order. It then applies active Sommerfeld face replacement to the complete
RHS exactly once.

``enforce_boundaries_and_trace_free_A`` is the state-side operation. Its order
is fixed:

1. Super-Gaussian filtering.
2. Determinant-one conformal-metric projection.
3. Trace-free conformal extrinsic-curvature projection.

RK4 placement
-------------

Classical RK4 forms ``k1``, two midpoint states, and one endpoint state. Every
stage state is passed through the state-side operation before its RHS is
evaluated. The weighted final state is passed through it again. Kreiss--Oliger
dissipation is already part of each field RHS, so it follows the same RK time
centering as the physical source terms.

API reference
-------------

.. automodule:: JAX_BSSN.evolution.derivatives
   :members:

.. automodule:: JAX_BSSN.evolution.boundaries
   :members:

.. automodule:: JAX_BSSN.evolution.time_evolve
   :members:
