Constraints and diagnostics
===========================

Diagnostics include hamiltonian and momentum constraint monitoring,
as well as plotting and openPMD output.

Raw constraint fields
---------------------

``JAX_BSSN.bssn.constraints`` produces arrays:

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
Apparent horizons
-----------------

``JAX_BSSN.utilities.find_apparent_horizon`` solves
:math:`\Theta = D_i s^i + K_{ij}s^i s^j - K = 0` on a star-shaped surface
:math:`r = h(\theta,\phi)` about a center, for physical covariant
:math:`\gamma_{ij}` and :math:`K_{ij}` taken off a finished slice. The unknowns
are the radii themselves on a cell-centered finite-difference angular grid, not
the coefficients of an angular basis, so one solver covers every symmetry the
code evolves. Which one is in use follows from the coordinate axes passed:

.. list-table:: Horizon input layouts
   :header-rows: 1
   :widths: 22 24 24 30

   * - Symmetry
     - Coordinates
     - ``gamma``/``Kij`` shape
     - Unknowns
   * - ``spherical``
     - ``(r,)``
     - ``(3, 3, nr)``
     - a single radius
   * - ``axisymmetric``
     - ``(x, z)``
     - ``(3, 3, nx, nz)``
     - :math:`h(\theta)`, center on the axis
   * - ``3d``
     - ``(x, y, z)``
     - ``(3, 3, nx, ny, nz)``
     - :math:`h(\theta,\phi)`, free center

The reduced layouts are not a reduced solver. A geometry provider rotates the
stored plane or line onto whatever Cartesian point the surface asks for, and
forward-mode AD of that interpolation supplies :math:`\partial_k\gamma_{ij}`,
including the azimuthal terms the reduced storage never holds explicitly.

The poles are never sampled and the :math:`\theta` ghost rows are the exact
shift :math:`h(-\theta,\phi) = h(\theta,\phi+\pi)`, so no pole regularity
condition enters. Newton updates use an exact AD Jacobian: dense below
``DENSE_JACOBIAN_LIMIT`` unknowns, and otherwise matrix-free GMRES
preconditioned by the flat-space linearization of :math:`\Theta`.

Every solve is audited. The center is moved onto the surface centroid along the
axes the symmetry leaves free, the angular grid is doubled until the area mass
stops moving, and the converged surface is re-evaluated on an independent
angular grid. Failures raise: there is no fixed-mass or spherical fallback, and
the returned dictionary carries the convergence history, both residuals and the
seeds that did not converge.
