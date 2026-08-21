Architecture
============

JAX BSSN separates mathematical ownership from timestep orchestration and
diagnostic side effects. The solver state is an immutable ``NamedTuple`` of
JAX arrays. Equation functions return right-hand-side arrays; the evolution
package decides when those functions, boundary treatments, and algebraic
projections run.

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

JAX execution model
-------------------

The production kernels are pure functions over ``BSSNVariables`` and
``BSSNParameters``. JAX array updates use functional ``.at[...]`` operations,
and runtime choices such as slicing gauge, zero shift, and active boundary
families use ``jax.lax.cond`` inside compiled code.

Derivative directions are static JIT arguments because they select an array
axis and stencil. Parameters remain scalar leaves of ``BSSNParameters``. Field
arrays retain strong floating-point dtypes; callers should enable 64-bit JAX
before constructing production data when double precision is required.

Current boundaries
------------------

The current architecture intentionally stops at one uniform grid or one
refinement patch. There is no hierarchy manager, dynamic regridding, multiple
patch ownership, mesh motion, matter source system, or Berger--Oliger
subcycling. See :doc:`fmr` for the exact implemented refinement contract.
