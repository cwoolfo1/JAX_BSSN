Einstein--Maxwell coupling
=========================

PyPIC3D dependency and ownership
------------------------------

``JAX_BSSN.EM.first_order`` imports PyPIC3D's static-metric Maxwell loop,
constitutive fields, boundary conditions, interpolation, and diagnostics.
There is no second Maxwell solver in JAX_BSSN. The PyPIC3D dependency supports
vacuum evolution with empty particle storage. This coupling has no
particle/current deposition or externally imposed fields. The charged
puncture's charge is encoded in the initial densitized electric flux.

The reference sponge and unlimited second-order shift upwinding described
below require the corresponding modified PyPIC3D checkout. Use both
companion checkouts when reproducing the current charged-puncture runs;
the historical pinned dependency did not provide these developments.

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
singleton axes must be periodic. BSSN lapse, shift, W, and conformal metric
components are transferred to native Yee locations by the original sequential
two-point arithmetic averaging. Along an axis changing from C to V, the
half-cell value is ``(f[i] + f[i+1]) / 2``. Applying this rule along each V axis
gives equal averages of 2, 4, or 8 samples for locations with one, two, or
three V coordinates. Unchanged C axes are not interpolated, and the CCC
sample is unchanged. These are equal sample averages, not volume-weighted
averages.

The metric builder applies this transfer unconditionally at initialization
and every coupled geometry rebuild. It then applies the existing W floor,
projects the averaged conformal metric to unit determinant, and derives the
physical metric, inverse, and volume from those values. The physical metric
and its volume are therefore derived after averaging W and the conformal
components, rather than averaged independently.

One additional temporary metric halo layer supplies the upper neighbor for
every retained EM ghost. The temporary layer is removed after the transfers;
the physical domain and retained EM guard allocation are unchanged.
Periodic copying and linear nonperiodic metric extrapolation retain their
existing conventions. Smooth interior transfer is second order. Native EM
density transfers, quadratic source averaging, Maxwell curls, time staggering,
and the boundary sponge are unchanged.

There is no production metric-interpolation selector. The shared Hermite
utility in ``utilities/hermite_interpolation.py`` remains available for other
interpolation tasks and archived comparison experiments; the production
Yee metric builder does not use it.

EM boundary names are ``periodic`` (default), ``conducting``, ``absorbing``
(PyPIC3D's zero-exterior policy), and ``constant``. They are independent of
BSSN's per-face gravity boundary codes. Integer codes and EM ``sommerfeld``
are rejected; gravity Sommerfeld remains supported. Absorbing exterior
values are not a Sommerfeld radiation condition or an absorbing layer.
PML, supergaussian absorber layers, and horizon policies are disabled.

Reference sponge for the charged puncture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The charged-puncture demo uses an additional runtime reference sponge, not
the ``absorbing`` zero-exterior flag. BSSN retains its Sommerfeld condition.
On all six EM faces a layer of coordinate width 3 damps deviations from the
initial densitized Yee fields. The native references include analytically
sampled exterior and face values and remain fixed when the metric changes.
The original domain and interior Maxwell curls are unchanged.

For distance d inward from a face, the rate is
``sigma_max * clip(1-d/width, 0, 1)**order``. Face rates add at edges and
corners. The defaults are order 4 and
``sigma_max = -5*log(1e-3)/3``. The nominal attenuation is for a unit-speed
crossing; it is not a guaranteed reflection coefficient. Coordinates are
evaluated at each field component's native Yee sites. Exterior ghosts and
face-coincident V nodes are fixed to the reference; C sites inside the faces
remain evolved. There is no interior transverse EM dissipation or PML.

``DensitizedMaxwellState.absorber`` optionally carries two tuples, one for D
and one for B, each containing component tuples of rates, references, and
exterior masks. All are runtime JAX arrays. PyPIC3D's GR Yee ``update_D``,
``update_B``, and ``time_loop_static_metric`` accept optional absorber data;
callers that omit it retain their original behavior. With midpoint curl C,
each update integrates local relaxation exponentially::

    F_new = F0 + exp(-sigma*dt)*(F_old-F0) + (-expm1(-sigma*dt)/sigma)*C

At zero rate this is the original ``F_old + dt*C``. The same update is used
for predictors, staggered history initialization, metric rephasing, and the
synchronized magnetic field. References are never independently damped or
redensitized. Negative steps used in history initialization are supported.

Spatially varying relaxation does not preserve discrete Gauss constraints:
the derivative of the rate contributes a source in the layer. Diagnostics
therefore distinguish sponge, undamped, and resolved-undamped regions, and
report the electric flux through a coordinate cube of half-width 4.
``tests/EM/test_em_sponge.py`` checks relaxation and time convergence;
``python -m tests.EM.sponge_wave_validation OUTPUT_DIRECTORY`` measures
reflection and constraint errors against a larger-domain reference.

Upwind shift products
~~~~~~~~~~~~~~~~~~~~~

The production E/H relations use ordinary arithmetic field interpolation
for the lapse-metric term, followed by division by the target
``sqrt_gamma``, contraction with the target spatial metric, and multiplication
by the target lapse. Metric geometry itself retains its original two-point
arithmetic transfer from the BSSN grid.

Only the shift cross products use upstream reconstruction. For a native
scalar density F, the unlimited slope and interface states are::

    s_i = (F_{i+1} - F_{i-1}) / 2
    left  = F_i     + s_i / 2
    right = F_{i+1} - s_{i+1} / 2

These brackets are shifted down one index for V-to-C transfer. Each
cross-product term transfers along one axis. The transport velocity is
``-beta``: positive target beta chooses the right state, negative beta the
left. Multiplication by target beta gives zero at zero shift. E contains
``+beta x B_density`` and H contains ``-beta x D_density``. There is no slope
limiter or alternative production transport selector.

This changes numerical dissipation and dispersion in the shift terms.
The native curls, time staggering, retained histories, current normalization,
source quadrature, and fixed-reference sponge are unchanged. The curl
increment still satisfies discrete divergence-of-curl cancellation; the
spatially varying sponge can introduce Gauss errors. No reciprocal
quadratic-energy conservation is asserted for this operator. This is
componentwise directional reconstruction, not a characteristic UCT scheme.

The same E/H functions are called during initialization, native predictions
and advances, metric-history rephasing, and common-time B reconstruction.
The stencil requires at least two guard cells; the evolution entry point
checks this, and the JAX adapter supplies three. Existing field boundary
refresh and conducting projections remain responsible for boundary values.

Restarting a historical method requires common-time D and synchronized B,
then fresh history initialization with this operator. Historical retained
histories cannot be reused by changing metadata or rephasing the metric.
Retired experiments and their dependencies are archived under
``demos/single_charged_blackhole/validation/sole_shift_upwind_20261020/before``.
They are excluded from active test discovery. The fixed production method
and source provenance are recorded in new outputs.

A controlled 96-cubed charged-puncture run using these shift products and
original target-metric terms completed 200M without the previous EM
runaway. Completion does not establish stationary RN accuracy, nor does it
qualify every metric and boundary configuration. Report growth, charge flux,
constraints, and analytical accuracy separately.

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
