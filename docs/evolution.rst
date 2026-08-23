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

Upwind shift advection
----------------------

``diff1_upwind_field`` is the first-derivative operator used only for shift
advection. Its ``rhs_coefficient`` argument is the coefficient :math:`c` in

.. math::

   \partial_t u = \cdots + c\,\partial_i u.

This convention is important: the corresponding transport velocity is
:math:`-c`. A positive RHS coefficient therefore selects the forward-biased
fourth-order stencil

.. math::

   D^{+}_4 f_i = \frac{-3f_{i-1}-10f_i+18f_{i+1}
                     -6f_{i+2}+f_{i+3}}{12h},

while a negative coefficient selects

.. math::

   D^{-}_4 f_i = \frac{-f_{i-3}+6f_{i-2}-18f_{i-1}
                     +10f_i+3f_{i+1}}{12h}.

As with centered first derivatives, MAD blends fourth- and sixth-order
operators:

.. math::

   D_{\mathrm{upwind,MAD}} = qD_{4,\mathrm{upwind}}
                              +(1-q)D_{6,\mathrm{upwind}}.

The forward sixth-order stencil uses offsets ``(-2,-1,0,1,2,3,4)`` and
coefficients ``(2,-24,-35,80,-30,8,-1)/(60h)``. The backward stencil uses
offsets ``(-4,-3,-2,-1,0,1,2)`` and coefficients
``(1,-8,30,-80,35,24,-2)/(60h)``. At a zero coefficient the selection falls
back to the centered derivative; multiplication by that coefficient makes
the complete advection contribution exactly zero.

``mad_q = 1`` takes a dedicated D4 fast path, so the wider D6 stencil is not
evaluated. On an axis with a physical Sommerfeld face the operator also uses
D4 regardless of ``mad_q``. Within the three grid points adjacent to each
configured Sommerfeld face, its biased result is replaced by the existing
boundary-aware centered D4 derivative. This prevents a rolled biased stencil
from importing values across a nonperiodic face.

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
