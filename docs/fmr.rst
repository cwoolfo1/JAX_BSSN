Fixed mesh refinement
=====================

The current FMR implementation is intentionally narrow: one immutable,
vertex-centered refinement patch with a fixed 2:1 spacing ratio and four guard
cells. ``JAX_BSSN.fmr.refinement`` keeps geometry, interpolation, guard filling,
restriction, and the synchronized RK4 step together so their ownership is
visible.

Patch geometry
--------------

``FMRPatchSpec.coarse_lo`` and ``coarse_hi`` are inclusive coarse-grid index
triples. For one axis with bounds ``lo`` and ``hi``, the number of active fine
vertices is

.. math::

   N_f = 2(\mathrm{hi}-\mathrm{lo})+1.

With guard width :math:`g=4`, the allocated fine extent is ``Nf + 2*g`` along
each axis. The active slice is therefore ``g:g+Nf``. ``fine_coordinates``
returns coordinates for the complete padded allocation; ``fine_active_view``
selects only the physical patch.

The default and currently supported geometry is:

.. code-block:: text

   refinement_ratio = 2
   ghost_width = 4
   coarse bounds = inclusive
   centering = vertex centered

Prolongation and guards
-----------------------

``prolongate_to_fine`` uses a tensor product of six-point, degree-five
Lagrange interpolation matrices. The same operator supports scalar, vector,
and tensor fields because it acts on the trailing three spatial axes and
preserves all leading axes.

``fill_fine_ghosts`` prolongates the coarse state over the padded fine
allocation and copies only the guard region. Active fine values are preserved.
Faces, edges, and corners are filled by the same tensor-product result, avoiding
an ordering dependency between separate face passes.

The coarse interpolation indices wrap periodically. The current FMR path is
therefore exercised with periodic physical boundaries; nonperiodic physical
FMR boundaries are not yet a supported hierarchy contract.

Restriction and injection
-------------------------

After a complete RK4 step, ``restrict_to_coarse`` selects coincident active
fine vertices with ``[::2, ::2, ::2]`` and injects them into the inclusive
coarse patch box. This is point injection, not volume averaging. Coarse points
outside the covered box remain unchanged.

Stage-synchronous RK4
---------------------

Coarse and fine levels use one shared timestep. There is no subcycling. The
implemented sequence is:

1. Project both level states to determinant-one and trace-free form.
2. Fill fine guard cells from the matching coarse stage.
3. Evaluate the coarse and fine BSSN right-hand sides.
4. Form matching coarse/fine ``k2``, ``k3``, and ``k4`` states and repeat
   steps 1--3 for each stage.
5. Combine both RK4 solutions.
6. Project the fine result.
7. Inject coincident active fine points into the coarse result and project the
   coarse state.
8. Refill final fine guard cells from the updated coarse state.

Both parameter sets must have equal ``dt``, and the fine ``dx`` must be exactly
half the coarse ``dx``. When ``use_mad`` is true, the coarse derivative
weight is derived inside the timestep as

.. math::

   q_{\mathrm{coarse}}
   = \left(\frac{h_{\mathrm{fine}}}{h_{\mathrm{coarse}}}\right)^4.

The fine level always uses ``mad_q = 1``, selecting the ordinary fourth-order
first- and second-derivative operators.
