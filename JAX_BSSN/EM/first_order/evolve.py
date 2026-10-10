"""Strang orchestration of unchanged PyPIC3D vacuum steps and BSSN RK4."""

from functools import partial
import jax
from PyPIC3D.relativity.field_state import densitize_vector
from JAX_BSSN.evolution.time_evolve import (
    compute_bssn_rhs_with_matter,
    enforce_algebraic_constraints,
)
from JAX_BSSN.EM.variables import EinsteinMaxwellVariables
from .geometry import build_yee_metric
from .coupling import quadratic_moments, sources_from_moments
from .energy_momentum import physical_fields_at_centers
from .maxwell import initialize_native_state, maxwell_half_step, rephase_state_metric


def common_densitized_fields(em):
    """Synchronized native tiled densities, at the integer D time."""
    return em.fields[0], em.synchronized_magnetic


@partial(jax.jit, static_argnames=("grid",))
def common_physical_fields(state, grid):
    return physical_fields_at_centers(
        *common_densitized_fields(state.em), state.bssn, grid
    )


@partial(jax.jit, static_argnames=("grid",))
def initialize_densitized_maxwell_state(displacement, magnetic, bssn, params, grid):
    """Initialize from component-first native densities on the physical grid."""
    metric = build_yee_metric(enforce_algebraic_constraints(bssn), grid)
    return initialize_native_state(
        tuple(grid.to_tile(v) for v in displacement),
        tuple(grid.to_tile(v) for v in magnetic),
        metric,
        params,
        grid,
    )


@partial(jax.jit, static_argnames=("grid",))
def initialize_first_order_einstein_maxwell_state(
    bssn, physical_displacement, physical_magnetic, params, grid
):
    """Initialize physical contravariant fields at PyPIC3D's native Yee sites.

    Inputs have shape (3,) + grid.shape; they do not contain EM guard cells.
    History is initialized once with spacing h=params.dt/2.
    """
    bssn = enforce_algebraic_constraints(bssn)
    metric = build_yee_metric(bssn, grid)
    d = densitize_vector(
        tuple(grid.to_tile(v) for v in physical_displacement), metric.D
    )
    b = densitize_vector(tuple(grid.to_tile(v) for v in physical_magnetic), metric.B)
    return EinsteinMaxwellVariables(
        bssn, initialize_native_state(d, b, metric, params, grid)
    )


def _add(bssn, rhs, scale):
    return jax.tree_util.tree_map(lambda q, r: q + scale * r, bssn, rhs)


@partial(jax.jit, static_argnames=("grid", "prescribed_gauge", "gravity_boundary"))
def first_order_einstein_maxwell_step(
    state, params, grid, *, time=0.0, prescribed_gauge=None, gravity_boundary=None
):
    """Two vacuum half-steps surrounding RK4 with fixed midpoint densities.

    prescribed_gauge(bssn,time) returns lapse, shift, dt_lapse, dt_shift.
    gravity_boundary(bssn,time) returns BSSN boundary data only. Callbacks
    are pure/JAX-compatible and static under JIT. Keep params.dt fixed after
    initialization: retained PyPIC3D history has spacing params.dt/2.
    """

    def prepare(bssn, t):
        if gravity_boundary is not None:
            bssn = gravity_boundary(bssn, t)
        if prescribed_gauge is not None:
            lapse, shift, _, _ = prescribed_gauge(bssn, t)
            bssn = bssn._replace(lapse=lapse, shift=shift)
        return enforce_algebraic_constraints(bssn)

    bssn = prepare(state.bssn, time)
    em = rephase_state_metric(state.em, build_yee_metric(bssn, grid), params, grid)
    midpoint = maxwell_half_step(em, params, grid)
    moments = quadratic_moments(*common_densitized_fields(midpoint), grid)

    def rhs(q, t):
        result = compute_bssn_rhs_with_matter(
            q, params, *sources_from_moments(moments, q)
        )
        if prescribed_gauge is not None:
            _, _, lapse_dot, shift_dot = prescribed_gauge(q, t)
            result = result._replace(lapse=lapse_dot, shift=shift_dot)
        return result

    dt = params.dt
    k1 = rhs(bssn, time)
    k2 = rhs(prepare(_add(bssn, k1, dt / 2), time + dt / 2), time + dt / 2)
    k3 = rhs(prepare(_add(bssn, k2, dt / 2), time + dt / 2), time + dt / 2)
    k4 = rhs(prepare(_add(bssn, k3, dt), time + dt), time + dt)
    increment = jax.tree_util.tree_map(
        lambda a, b, c, d: (a + 2 * b + 2 * c + d) / 6, k1, k2, k3, k4
    )
    final_bssn = prepare(_add(bssn, increment, dt), time + dt)
    em = rephase_state_metric(
        midpoint, build_yee_metric(final_bssn, grid), params, grid
    )
    return EinsteinMaxwellVariables(final_bssn, maxwell_half_step(em, params, grid))
