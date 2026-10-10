Einstein--Maxwell coupling
=========================

Time integration
----------------

The full Cartesian ``JAX_BSSN.EM.first_order`` solver advances BSSN with
classical RK4 and retains the second-order doubled-leapfrog Maxwell
history. It evaluates the gravity RHS four times per step: at the initial
time, at two midpoint stages, and at an endpoint stage. The three Maxwell
RHS evaluations use the initial metric, the second midpoint metric, and the
final projected metric. The coupled staggered method remains second order
overall. Spatial derivatives are unchanged.

EM evolution is supported on Cartesian grids. The second-order Maxwell wave
formulation and EM Cartoon reductions have been removed. Vacuum BSSN Cartoon
and FMR integrators continue to use RK4.

Cartesian conducting walls
-------------------------

``JAX_BSSN.EM.first_order.pec`` ports the static-metric PEC implementation
from the local PyPIC3D project, including its coupled electric edge solves,
magnetic ghost edge solves, and ordered corner reflections. It has no
runtime dependency on PyPIC3D. The public functions accept *densitized*
displacement and magnetic fields. The metric and volume weights are rebuilt
from the current BSSN stage, rather than held fixed during evolution.

The wall contract is PyPIC3D's stationary FIDO-frame condition. For a face
normal to coordinate direction ``a``, the electric displacement retains its
metric-normal part, ``D^i = gamma^{ia} D^a / gamma^{aa}``, and the normal
magnetic flux vanishes. Off-diagonal metrics can produce nonzero coordinate
tangential components of D. Intersections of independent electric wall
normals are zero. With nonzero shift this condition does not require the
coordinate covector ``E_i`` to have zero tangential components.

Select paired walls using ``PECBoundary(axes=(0, 1, 2), guard_cells=3)``.
The axes identify Cartesian x/y/z directions. This setting is independent
of the gravity boundary codes in ``BSSNParameters`` and applies to the
Cartesian staggered EM fields.

Each selected axis must include explicit exterior layers. For ``N`` physical
cell intervals and ``g`` guard layers, allocate ``N + 1 + 2*g`` samples.
The walls lie on C nodes at indices ``g`` and ``size-g-1``. JAX BSSN places
V samples half a cell below C samples; their owned indices run from ``g+1``
through ``size-g-1``. At least two guard layers and ``N > g`` are required.
The original grid coordinates and state array shapes are preserved.

For example, given BSSN and physical native Yee fields on this padded grid::

    from JAX_BSSN.EM.first_order import (
        PECBoundary,
        initialize_first_order_einstein_maxwell_state,
        first_order_einstein_maxwell_step,
    )

    walls = PECBoundary(axes=(0,), guard_cells=3)
    # For a left wall at x=0, use params.x_min = -3 * params.dx.
    state = initialize_first_order_einstein_maxwell_state(
        bssn, physical_D, physical_B, params, pec_boundary=walls,
    )
    state = first_order_einstein_maxwell_step(
        state, params, time=t, pec_boundary=walls,
    )

Supply the intended spacetime ghost data through the existing
``stage_boundary(bssn, D, B, time)`` callback when needed. PEC conditions only
the EM fields and runs after that callback. Exterior samples are stencil
data and should be excluded from physical-domain diagnostics. The explicit
layers keep constitutive transfers, curls, and matter-source stencils away
from periodic wraparound at the allocation edges.

Initialization projects the common-time fields before computing their
derivatives and conditions the bootstrapped history. During stepping, the
fields use their metric at each RK4 stage and at the final projected state.
History projections use linear metric predictions at their integer or
half-integer times. The forward half-time predictor uses the endpoint
derivative ``k4``, retaining second-order history accuracy without additional
gravity RHS evaluations.

For fixed-background use, ``apply_pec_boundaries(D, B, bssn, params,
boundary=walls)`` returns a conditioned pair. ``enforce_pec_D`` and
``enforce_pec_B`` provide the individual operations. Refresh fields before
constitutive or curl evaluations and after updates. Do not separately zero
the computed auxiliary E/H fields: that would change the imported wall
contract.
