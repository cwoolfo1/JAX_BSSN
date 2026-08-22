Evolution and boundaries
========================

The evolution package owns the discretization and the placement of boundary
and algebraic constraint operations for the BSSN equations.


First and second derivatives with MAD
-------------------------------------

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

``diff2_field`` applies the same MAD weight to dedicated centered second-
derivative stencils.  With ``mad_q = 1`` it uses the five-point fourth-order
operator; ``mad_q = 0`` selects the seven-point sixth-order operator.  Pure
second derivatives never compose two centered first derivatives.  Mixed
derivatives across distinct axes continue to compose the corresponding first
derivative operators.

FMR derives the coarse value of ``q`` from the spacing ratio and forces the
fine value to one. On a physical Sommerfeld axis, first and second derivatives
deliberately retain their fourth-order operators.

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
     - Sommerfeld
     - Use nonperiodic derivative closures and replace the assembled RHS on
       the selected outer plane.

Sommerfeld uses the coordinate origin supplied through ``x_min``, ``y_min``,
and ``z_min``. It evaluates

.. math::

   \partial_t u = -\left(\partial_r u + \frac{u-u_0}{r}\right)

on the union of configured face planes. The union mask ensures an edge or
corner is replaced once, independent of face ordering.

RHS assembly
------------

``compute_bssn_rhs`` calls the seven time derivatives. It then applies 
active Sommerfeld face replacement to the complete RHS exactly once.

``enforce_algebraic_constraints`` applies the two BSSN algebraic projections
in this fixed order:

1. Determinant-one conformal-metric projection.
2. Trace-free conformal extrinsic-curvature projection.
