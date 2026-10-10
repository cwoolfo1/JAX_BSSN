"""Call PyPIC3D unchanged, then construct a separate common-time B view."""

from functools import partial

import jax
import jax.numpy as jnp
from PyPIC3D.boundary_conditions.staggered import refresh_fields
from PyPIC3D.particles.particle_class import TiledParticles, SpeciesConfig
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS
from PyPIC3D.solvers.GR_yee.static_metric import (
    compute_covariant_E,
    compute_covariant_H,
    update_B,
    update_D,
)
from PyPIC3D.solvers.GR_yee.time_loop import time_loop_static_metric
from .variables import DensitizedMaxwellState


def _average(a, b):
    return tuple(0.5 * (x + y) for x, y in zip(a, b))


def empty_particles(dtype):
    position = jnp.zeros((1, 1, 1, 0, 0, 3), dtype=dtype)
    particles = TiledParticles(
        position, position, jnp.zeros(position.shape[:-1], dtype=bool)
    )
    zero = jnp.zeros((0,), dtype=dtype)
    return particles, SpeciesConfig(zero, zero, zero, jnp.zeros((0, 3), dtype=bool))


def refresh_state_metric(em, metric, grid):
    """Replace geometry and refresh boundary data without redensitizing."""

    def D(v):
        return refresh_fields(v, grid.static, D_FIELD_LOCATIONS, "D", metric)

    def B(v):
        return refresh_fields(v, grid.static, B_FIELD_LOCATIONS, "B", metric)

    d, b, current, rho, phi, external, _, previous, overflow = em.fields
    fields = (
        D(d),
        B(b),
        current,
        rho,
        phi,
        external,
        metric,
        (D(previous[0]), B(previous[1])),
        overflow,
    )
    return em._replace(fields=fields, synchronized_magnetic=B(em.synchronized_magnetic))


def rephase_state_metric(em, metric, params, grid):
    """Align retained history with a changed frozen Maxwell generator.

    At a metric switch, a sample at offset tau needs tau*(F_new-F_old).
    Here tau is -h for previous D, -h/2 for B, and -3h/2 for previous B.
    Since the metric change is O(h), the omitted quadratic difference is
    O(h^3). Common-time owned densities are preserved, not reinitialized.
    This correction is necessary for second-order Strang convergence.
    """
    dynamic = grid.dynamic(params, em.half_dt)
    refreshed = refresh_state_metric(em, metric, grid)

    def increments(state):
        d, b, m = state.fields[0], state.synchronized_magnetic, state.fields[6]
        zero = tuple(jnp.zeros_like(q) for q in d)
        ed = compute_covariant_E(d, b, m)
        hb = compute_covariant_H(d, b, m)
        next_d = update_D(d, hb, zero, m, grid.static, dynamic, dynamic.dt)
        next_b = update_B(ed, b, m, grid.static, dynamic, dynamic.dt)
        return (
            tuple(a - q for a, q in zip(next_d, d)),
            tuple(a - q for a, q in zip(next_b, b)),
        )

    old, new = increments(em), increments(refreshed)
    delta_d = tuple(n - o for n, o in zip(new[0], old[0]))
    delta_b = tuple(n - o for n, o in zip(new[1], old[1]))
    d, b, j, rho, phi, external, m, previous, overflow = refreshed.fields
    b = tuple(q - 0.5 * delta for q, delta in zip(b, delta_b))
    previous = (
        tuple(q - delta for q, delta in zip(previous[0], delta_d)),
        tuple(q - 1.5 * delta for q, delta in zip(previous[1], delta_b)),
    )
    shifted = refreshed._replace(
        fields=(d, b, j, rho, phi, external, m, previous, overflow)
    )
    return refresh_state_metric(shifted, metric, grid)


def initialize_native_state(displacement, magnetic, metric, params, grid):
    """One-time O(h^2) history initialization from common-time densities."""
    static = grid.static
    h = params.dt / 2
    dynamic = grid.dynamic(params, h)
    d = refresh_fields(displacement, static, D_FIELD_LOCATIONS, "D", metric)
    b = refresh_fields(magnetic, static, B_FIELD_LOCATIONS, "B", metric)
    zero = tuple(jnp.zeros_like(v) for v in d)
    e, H = compute_covariant_E(d, b, metric), compute_covariant_H(d, b, metric)
    previous_d = update_D(d, H, zero, metric, static, dynamic, -h)
    half_b = update_B(e, b, metric, static, dynamic, -h / 2)
    previous_b = update_B(e, b, metric, static, dynamic, -3 * h / 2)
    fields = (
        d,
        half_b,
        zero,
        zero[0],
        zero[0],
        (zero, zero),
        metric,
        (previous_d, previous_b),
        jnp.asarray(False),
    )
    return DensitizedMaxwellState(fields, b, jnp.asarray(h))


def synchronized_magnetic_after_step(before, after, grid, dynamic):
    """Endpoint B, centered in time, using only imported Maxwell operations.

    Reconstruct B^n from the saved incoming history, then advance B^n by h
    with E evaluated at n+1/2. Neither before nor after is modified.
    """
    d, b = before[:2]
    dp, bp = before[7]
    metric = before[6]
    half_d = _average(d, dp)
    integer_b = update_B(
        compute_covariant_E(half_d, b, metric),
        _average(b, bp),
        metric,
        grid.static,
        dynamic,
        dynamic.dt,
    )
    midpoint_d = _average(d, after[0])
    return update_B(
        compute_covariant_E(midpoint_d, after[1], metric),
        integer_b,
        metric,
        grid.static,
        dynamic,
        dynamic.dt,
    )


@partial(jax.jit, static_argnames=("grid",))
def maxwell_half_step(em, params, grid):
    """One h=dt/2 PyPIC3D vacuum call, preserving its entire returned state.

    The timestep must match the spacing used to initialize the history.
    This function assumes the metric/boundaries in em.fields are refreshed.
    """
    dynamic = grid.dynamic(params, em.half_dt)
    particles, species = empty_particles(em.fields[0][0].dtype)
    _, fields = time_loop_static_metric(
        particles, species, em.fields, grid.static, dynamic
    )
    synchronized_b = synchronized_magnetic_after_step(em.fields, fields, grid, dynamic)
    return DensitizedMaxwellState(fields, synchronized_b, em.half_dt)
