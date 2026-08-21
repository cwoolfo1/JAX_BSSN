Architecture
============

The overall architecture of JAX BSSN keeps time derivative equations 
as separate pure functions, and the evolution package decides when to call them. 
This allows future experimentation with different timestepping algorithms and 
boundary treatments without changing the equations themselves.


Package map
-----------

``JAX_BSSN.bssn``
   State definitions, tensor algebra, shared geometry, the individual BSSN
   time-derivative equations, and raw constraint fields.

``JAX_BSSN.evolution``
   Finite-difference operators, physical boundary treatments, complete RHS
   assembly, algebraic projections, and unigrid RK4.

``JAX_BSSN.fmr``
   Geometry and transfers for one vertex-centered refinement patch, plus the
   stage-synchronous coarse/fine RK4 step.

``JAX_BSSN.diagnostics``
   Constraint reductions and reporting, plotting, NumPy snapshots, and
   synchronous openPMD output. Evolution equations never import this package.

``demos``
   The three supported physical configurations and their initial data. Physical
   initial data are deliberately not part of the installed package.

``tests``
   Regression, convergence, and transfer tests. Gauge- and linear-wave test
   data live in ``tests/initial_data.py`` rather than executable demo modules.

Dependency direction
--------------------

The intended dependency flow is:

.. code-block:: text

   bssn.variables
        |
        +--> evolution.derivatives
        +--> bssn tensor/geometry/equation modules
        +--> evolution.boundaries
        +--> evolution.time_evolve
        +--> fmr.refinement
        +--> diagnostics

Raw constraints remain in ``bssn.constraints`` because the momentum
constraint participates in the evolution equations. Norms, printing, and
health decisions remain in ``diagnostics.constraints`` so the solver does not
depend on reporting code.

Unigrid execution flow
----------------------

``JAX_BSSN.evolution.time_evolve.rk4_step`` performs one update in this exact
order:

1. Apply any active Super-Gaussian state filter.
2. Rescale the conformal metric to determinant one.
3. Project conformal extrinsic curvature to its trace-free part.
4. Assemble ``k1`` from all seven equation functions and replace the RHS on
   active Sommerfeld faces.
5. Form the ``k2`` midpoint state, then repeat steps 1--4.
6. Form the ``k3`` midpoint state, then repeat steps 1--4.
7. Form the ``k4`` endpoint state, then repeat steps 1--4.
8. Combine the classical RK4 stages and apply the state filter and both
   algebraic projections once more.

Sommerfeld is therefore an RHS boundary treatment. Super-Gaussian damping is
a state filter. Moving either operation across the RHS/projection boundary
would define a different numerical algorithm.
