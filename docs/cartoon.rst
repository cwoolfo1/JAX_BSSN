Spherical Cartoon evolution
===========================

``JAX_BSSN.cartoon`` evolves a spherically symmetric state while reusing the
Cartesian BSSN equations and finite-difference operators. The compact state
stores only positive half-cell radii plus the reflected samples required by
the Cartesian stencils. A temporary Cartesian support grid is reconstructed
for every RK stage; it is never an independently evolved mesh.

Compact radial storage
----------------------

For :math:`N_r` independent points, the positive coordinates are

.. math::

   r_i = (i + 1/2)\Delta x,
   \qquad i=0,\ldots,N_r-1.

The allocated field shape is ``(Nr + 4, 1, 1)``. The first four entries are
negative-:math:`x` parity ghosts and the remaining entries are the
authoritative positive-radius data. The half-cell placement avoids sampling
the coordinate origin or a puncture at :math:`r=0`.

Scalars are even under reflection. For Cartesian vectors the :math:`x`
component is odd and the transverse components are even. Rank-two tensor
parity is the outer product of those vector parities. ``fill_cartoon_ghosts``
recreates every ghost value from the first four positive samples.

Spherical reconstruction
------------------------

``reconstruct_cartoon_support`` directly interpolates the parity-filled
compact centerline and reconstructs a Cartesian ``(Nr + 4) x 9 x 9`` support
grid. It does not create a separate radial-profile representation or average
the ``yy`` and ``zz`` tensor components.

Axis fields are evaluated at
:math:`r=\sqrt{x^2+y^2+z^2}` with six-point, degree-five nonperiodic Lagrange
interpolation. Vectors use :math:`V^i=V_r n^i`; tensors use

.. math::

   T_{ij} = T_t\delta_{ij} + (T_r-T_t)n_i n_j.

For the signed compact axis the continuous interpolation coordinate is

.. math::

   q = r/\Delta x + N_{\mathrm{ghost}} - 1/2 = r/\Delta x + 3.5.

Thus the first positive half-cell naturally selects a centered stencil that
includes the ``-1.5dx`` and ``-0.5dx`` parity ghosts. Three temporary outer
samples are appended for reconstruction only. They continue each field's
deviation from its flat-space value as ``1/r``, consistent with the radial
falloff in the Sommerfeld boundary condition. Out-of-domain interpolation
returns NaN, so a missing buffer cannot silently become polynomial
extrapolation.

Only the positive centerline RHS is retained after the Cartesian equations
are evaluated. Four reflected ghosts are then regenerated from that result.

Evolution contract
------------------

``cartoon_rk4_step`` is a separate entry point from Cartesian ``rk4_step``.
For each classical RK4 stage it performs this sequence:

1. Enforce determinant-one and trace-free BSSN algebraic constraints.
2. Refresh compact origin ghosts from reflection parity.
3. Reconstruct Cartesian support from the parity-filled signed axis.
4. Evaluate the ordinary Cartesian BSSN RHS and its outer Sommerfeld face.
5. Project the support RHS back onto compact radial storage.

The compact grid uses a non-Sommerfeld left ``x`` code because those entries
are internal parity ghosts, not a physical face. The outer ``x`` face is
Sommerfeld and all temporary ``y``/``z`` faces are periodic. Coordinate minima
must be ``(-3.5dx, -4dx, -4dx)``.

The current ``9 x 9`` support conservatively accommodates the fourth-order
first-, second-, mixed-, and upwind shift-advection stencils. In particular,
the D4 upwind operator reaches three points toward the transport direction;
the centerline has four reconstructed samples available on either side.
Cartoon requires ``mad_q = 1``, which takes the D4 fast path and does not
evaluate the wider D6 upwind stencil. Setup rejects other MAD weights rather
than allowing transverse stencils to exceed the supported evolution
contract. The outer radial Sommerfeld face additionally uses the centered D4
fallback in its three adjacent points.

Constraints and output
----------------------

``compute_cartoon_constraints`` reconstructs support once, evaluates the raw
Cartesian constraints, and projects each result back to compact storage.
``compute_cartoon_constraint_norms`` reduces only independent positive-radius
samples and excludes four outer samples by default.
``compute_spherical_symmetry_norms`` reports L2 and L-infinity norms of
``gamma_yy-gamma_zz``, ``A_yy-A_zz``, and reference-axis components that
should vanish.

``cartoon_axis_output_fields`` expands compact fields and constraints onto a
complete reflected ``2*Nr x 1 x 1`` signed axis. Its mapping can be passed
directly to ``OpenPMDWriter``; no support-plane values or compact ghosts are
written.
