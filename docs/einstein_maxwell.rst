Einstein--Maxwell coupling
=========================

PyPIC3D dependency and ownership
------------------------------

``JAX_BSSN.EM.first_order`` imports PyPIC3D's static-metric Maxwell loop,
constitutive fields, boundary conditions, interpolation, and diagnostics.
There is no second Maxwell solver in JAX_BSSN. The required dependency is
pinned to PyPIC3D commit ``9124995c8eac57f92957a6a59089d04dac1a3392``, which
supports vacuum evolution with empty particle storage. No particles, charge,
current, or external fields participate in this coupling.

Install normally with ``python -m pip install -e '.[test]'``. To develop
against an already provisioned sibling checkout instead of the pinned Git
source, install both editable packages without dependency resolution::

    python -m pip install --no-deps --no-build-isolation -e ../PyPIC3D
    python -m pip install --no-deps --no-build-isolation -e '.[test]'

This development override assumes the runtime and test dependencies are
already installed. It does not modify Python's import path in source code.

Time integration and state
--------------------------

One coupled step consists of an imported vacuum Maxwell step with
``h = params.dt / 2`` and the initial metric frozen, an RK4 BSSN step of
``params.dt`` with fixed synchronized densities, and a second vacuum Maxwell
step with the final metric frozen. Electromagnetic stress-energy sources
BSSN. The quadratic native-field moments are frozen during RK4; their
conversion to physical Eulerian stress-energy uses each RK4 stage metric.
The existing rationalized normalization, with flat-space energy density
``(D**2 + B**2)/2`` and gravitational coupling ``8*pi``, is retained.

``EinsteinMaxwellVariables`` contains BSSN and ``DensitizedMaxwellState``.
The latter stores PyPIC3D's full nine-slot ``fields`` tuple, a separately
computed ``synchronized_magnetic`` tuple, and the history spacing ``half_dt``.
At the common integer time t, the runtime tuple contains D(t), B(t-h/2),
and previous fields D(t-h), B(t-3h/2). Histories are initialized once from
common-time data. Keep the timestep fixed after initialization; changing it
requires initializing a new history.

After each imported loop call, JAX_BSSN reconstructs B at the incoming
integer time using saved incoming histories and ``compute_covariant_E`` /
``update_B``. A second full-h ``update_B`` uses the returned staggered B and
the average of incoming and returned D to produce synchronized endpoint B.
The loop's returned B and history are not replaced by that synchronized view.
``common_densitized_fields`` returns native tiled D/B at the common time;
``common_physical_fields`` returns component-first physical fields at BSSN
C nodes, using PyPIC3D interpolation and the current metric volume.

History alignment at a metric change
-----------------------------------

A frozen-metric switch changes the Maxwell time derivatives. Merely replacing
the metric while keeping numerical history samples unchanged gives first-order
error, even with Strang ordering. Before evolution with the new metric, the
adapter evaluates old and new Maxwell derivatives on the synchronized fields,
using only imported constitutive and update functions. With
``delta_F = F_new - F_old``, it corrects the retained samples by::

    previous_D -= h * delta_F_D
    staggered_B -= (h / 2) * delta_F_B
    previous_B -= (3 * h / 2) * delta_F_B

These corrections align the negative-time samples with the new frozen
operator. They do not reset histories or change common-time owned D/B.
For a smooth metric change of order h, the omitted higher-order correction
is order h cubed. PyPIC3D boundary refresh follows the correction; conducting
boundary values may be projected onto the new metric's wall condition.
Both vacuum loop calls themselves remain unchanged.

Grid, metrics, and boundaries
-----------------------------

Construct ``make_em_grid(shape, params, boundary_conditions=...)`` outside
JIT and pass the resulting static ``EMGrid`` to initialization, stepping,
and diagnostics. This version uses one Cartesian tile with three EM guard
cells. BSSN and native-field input arrays contain physical points only;
the adapter creates the EM guards. Native vectors have input shape
``(3,) + grid.shape``. Use ``grid.coordinates(params, location)`` and
PyPIC3D's ``D_FIELD_LOCATIONS`` / ``B_FIELD_LOCATIONS`` for initial data.

PyPIC3D C nodes coincide with BSSN points, and V nodes are half a cell
*above* C nodes. This differs from the previous JAX_BSSN EM convention and
from the separate legacy coordinate utilities used by vacuum Cartoon code.
Do not use those legacy V coordinates to initialize the new EM backend.

Periodic axes contain N physical C nodes for N cells. Conducting axes
contain N+1 C nodes, including both walls, for N cells. The uppermost
allocated V sample on a conducting axis is exterior data; exclude it from
native-field physical-domain norms. Active axes need more than three cells;
singleton axes must be periodic. Metric interpolation derives a consistent
physical metric, inverse, and volume after projecting the interpolated
conformal metric to unit determinant and applying the existing W floor.
An additional metric halo prevents wraparound in exterior interpolation.

EM boundary names are ``periodic`` (default), ``conducting``, ``absorbing``
(PyPIC3D's zero-exterior policy), and ``constant``. They are independent of
BSSN's per-face gravity boundary codes. Integer codes and EM ``sommerfeld``
are rejected; gravity Sommerfeld remains supported. Absorbing exterior
values are not a Sommerfeld radiation condition or an absorbing layer.
PML, supergaussian absorber layers, and horizon policies are disabled.

A conducting cavity can be initialized as follows::

    import jax.numpy as jnp
    from JAX_BSSN.EM.first_order import (
        D_FIELD_LOCATIONS, make_em_grid,
        initialize_first_order_einstein_maxwell_state,
        first_order_einstein_maxwell_step, electromagnetic_output_fields,
    )

    # bssn has physical shape (N+1, 1, 1); x_min is the left wall.
    grid = make_em_grid(
        bssn.lapse.shape, params,
        boundary_conditions=("conducting", "periodic", "periodic"),
    )
    x, _, _ = grid.coordinates(params, D_FIELD_LOCATIONS[1])
    length = (grid.shape[0] - 1) * params.dx
    zero = jnp.zeros((3,) + grid.shape)
    D = zero.at[1].set(1.e-5 * jnp.sin(jnp.pi * (x - params.x_min) / length))
    state = initialize_first_order_einstein_maxwell_state(bssn, D, zero, params, grid)
    state = first_order_einstein_maxwell_step(state, params, grid, time=0.)
    output = electromagnetic_output_fields(state, grid)

The optional ``gravity_boundary(bssn, time)`` callback conditions BSSN only.
``prescribed_gauge(bssn, time)`` returns lapse, shift, lapse derivative, and
shift derivative. Both callbacks are pure JAX functions; RK4 evaluates the
gauge at its initial, midpoint, and endpoint stage times. Algebraic BSSN
constraints are enforced at every stage. EM boundary operations belong to
PyPIC3D and cannot modify the frozen matter moments during RK4.

Migration and validation
------------------------

The old four-history EM state, ``PECBoundary``, combined ``stage_boundary``
callback, and local Maxwell-kernel interfaces are removed. Old checkpoints
are not compatible: reconstruct physical common-time initial data at the
new native coordinates and initialize a new state. openPMD output continues
to contain physical synchronized D/B at BSSN C nodes.

Tests compare the adapter to direct PyPIC3D calls, audit volumes and endpoints,
check density Gauss constraints and conducting cavities, and exercise JIT
and ``lax.scan``. Temporal tests use an exact semidiscrete Fourier mode to
separate timestep error from spatial error, including a changing lapse that
exposes history misalignment. Einstein--Maxwell packet and coupled convergence
tests exercise backreaction, and zero-field tests compare against vacuum RK4.
