"""Synchronized BSSN RK4 and doubled-leapfrog Maxwell evolution."""

import jax
import jax.numpy as jnp
from functools import partial

from JAX_BSSN.bssn import BSSNParameters, BSSNVariables
from JAX_BSSN.evolution.time_evolve import (
    compute_bssn_rhs_with_matter,
    enforce_algebraic_constraints,
)

from JAX_BSSN.EM.first_order.energy_momentum import (
    compute_densitized_electromagnetic_energy_momentum,
    physical_fields_at_centers,
)
from JAX_BSSN.EM.first_order.equations import (
    densitized_maxwell_rhs,
)
from JAX_BSSN.EM.first_order.geometry import (
    _conformal_factors_at_locations,
)
from JAX_BSSN.EM.first_order.staggering import (
    DISPLACEMENT_FIELD_LOCATIONS,
    MAGNETIC_FIELD_LOCATIONS,
)
from JAX_BSSN.EM.first_order.variables import DensitizedMaxwellState
from JAX_BSSN.EM.variables import EinsteinMaxwellVariables


def _add_bssn_scaled(
    bssn: BSSNVariables, rhs: BSSNVariables, scale
) -> BSSNVariables:
    return jax.tree_util.tree_map(
        lambda field, derivative: field + scale * derivative,
        bssn,
        rhs,
    )


def _bssn_rhs_from_densities(
    bssn: BSSNVariables,
    densitized_displacement: jnp.ndarray,
    densitized_magnetic: jnp.ndarray,
    params: BSSNParameters,
) -> BSSNVariables:
    energy_density, momentum_density, spatial_stress = (
        compute_densitized_electromagnetic_energy_momentum(
            densitized_displacement,
            densitized_magnetic,
            bssn,
            params,
        )
    )
    return compute_bssn_rhs_with_matter(
        bssn,
        params,
        energy_density,
        momentum_density,
        spatial_stress,
    )


@jax.jit
def common_densitized_fields(
    em: DensitizedMaxwellState,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(mathcal D^n, mathcal B^n)`` at the common integer time."""

    densitized_displacement = 0.5 * (
        em.displacement_left_half + em.displacement_right_half
    )
    return densitized_displacement, em.magnetic_current


@jax.jit
def common_physical_fields(
    state: EinsteinMaxwellVariables[DensitizedMaxwellState],
    params: BSSNParameters,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return cell-centered physical contravariant fields at the common time."""

    densitized_displacement, densitized_magnetic = common_densitized_fields(
        state.em
    )
    return physical_fields_at_centers(
        densitized_displacement,
        densitized_magnetic,
        state.bssn,
        params,
    )


@jax.jit
def bootstrap_densitized_maxwell_state(
    densitized_displacement: jnp.ndarray,
    densitized_magnetic: jnp.ndarray,
    densitized_displacement_dot: jnp.ndarray,
    densitized_magnetic_dot: jnp.ndarray,
    dt: float,
) -> DensitizedMaxwellState:
    """Build second-order leapfrog history from common-time fields and dots."""

    half_step = 0.5 * dt
    return DensitizedMaxwellState(
        magnetic_previous=densitized_magnetic - dt * densitized_magnetic_dot,
        magnetic_current=densitized_magnetic,
        displacement_left_half=(
            densitized_displacement - half_step * densitized_displacement_dot
        ),
        displacement_right_half=(
            densitized_displacement + half_step * densitized_displacement_dot
        ),
    )


@jax.jit
def initialize_densitized_maxwell_state(
    densitized_displacement: jnp.ndarray,
    densitized_magnetic: jnp.ndarray,
    bssn: BSSNVariables,
    params: BSSNParameters,
) -> DensitizedMaxwellState:
    """Bootstrap the doubled leapfrog from common-time density fields."""

    bssn = enforce_algebraic_constraints(bssn)
    displacement_rhs, magnetic_rhs = densitized_maxwell_rhs(
        densitized_displacement,
        densitized_magnetic,
        bssn,
        params,
    )
    return bootstrap_densitized_maxwell_state(
        densitized_displacement,
        densitized_magnetic,
        displacement_rhs,
        magnetic_rhs,
        params.dt,
    )


@jax.jit
def initialize_first_order_einstein_maxwell_state(
    bssn: BSSNVariables,
    physical_displacement: jnp.ndarray,
    physical_magnetic: jnp.ndarray,
    params: BSSNParameters,
) -> EinsteinMaxwellVariables[DensitizedMaxwellState]:
    """Initialize from physical contravariant fields on their native Yee sites."""

    bssn = enforce_algebraic_constraints(bssn)
    displacement_W = _conformal_factors_at_locations(
        bssn, DISPLACEMENT_FIELD_LOCATIONS, params
    )
    magnetic_W = _conformal_factors_at_locations(
        bssn, MAGNETIC_FIELD_LOCATIONS, params
    )
    densitized_displacement = displacement_W**-3 * physical_displacement
    densitized_magnetic = magnetic_W**-3 * physical_magnetic
    em = initialize_densitized_maxwell_state(
        densitized_displacement,
        densitized_magnetic,
        bssn,
        params,
    )
    return EinsteinMaxwellVariables(bssn=bssn, em=em)


@partial(jax.jit, static_argnames=("prescribed_gauge", "stage_boundary"))
def first_order_einstein_maxwell_step(
    state: EinsteinMaxwellVariables[DensitizedMaxwellState],
    params: BSSNParameters,
    *,
    time=0.0,
    prescribed_gauge=None,
    stage_boundary=None,
) -> EinsteinMaxwellVariables[DensitizedMaxwellState]:
    """Advance one two-way coupled RK4/doubled-leapfrog timestep.

    Optional pure, JAX-compatible callbacks are static compilation arguments.
    ``stage_boundary(bssn, D, B, time)`` returns boundary-conditioned
    ``(bssn, D, B)`` at one common time. ``prescribed_gauge(bssn, time)``
    returns ``(lapse, shift, dt_lapse, dt_shift)`` at that time. The latter
    replaces both gauge values and their RHS, without changing physical RHSs.
    History fields are conditioned at their own integer/half-integer times.
    During history-only calls, other inputs retain current-stage values and
    their returned values are discarded. Prescribed boundary callbacks should
    therefore obtain boundary data from time/coordinates, not those inputs.
    With neither callback, the historical algorithm is unchanged.
    """

    def prepare(bssn, displacement, magnetic, t):
        if stage_boundary is not None:
            bssn, displacement, magnetic = stage_boundary(bssn, displacement, magnetic, t)
        if prescribed_gauge is not None:
            lapse, shift, _, _ = prescribed_gauge(bssn, t)
            bssn = bssn._replace(lapse=lapse, shift=shift)
        return bssn, displacement, magnetic

    def gravity_rhs(bssn, displacement, magnetic, t):
        rhs = _bssn_rhs_from_densities(bssn, displacement, magnetic, params)
        if prescribed_gauge is not None:
            _, _, lapse_dot, shift_dot = prescribed_gauge(bssn, t)
            rhs = rhs._replace(lapse=lapse_dot, shift=shift_dot)
        return rhs

    dt = params.dt
    bssn_n = enforce_algebraic_constraints(state.bssn)
    em = state.em
    displacement_n, magnetic_n = common_densitized_fields(em)
    bssn_n, displacement_n, magnetic_n = prepare(bssn_n, displacement_n, magnetic_n, time)

    # BSSN k1 and the first member of the doubled magnetic leapfrog.
    k1 = gravity_rhs(
        bssn_n, displacement_n, magnetic_n, time
    )
    _, magnetic_rhs_n = densitized_maxwell_rhs(
        displacement_n, magnetic_n, bssn_n, params
    )
    magnetic_previous = em.magnetic_previous
    if stage_boundary is not None:
        _, _, magnetic_previous = stage_boundary(bssn_n, displacement_n, magnetic_previous, time - dt)
    magnetic_left_half = 0.5 * (magnetic_previous + magnetic_n)
    magnetic_right_half = magnetic_left_half + dt * magnetic_rhs_n

    # Both RK4 midpoint stages represent the same physical half time, but use
    # their own projected BSSN metrics for stress-energy and constitutive fields.
    bssn_midpoint_1 = enforce_algebraic_constraints(
        _add_bssn_scaled(bssn_n, k1, 0.5 * dt)
    )
    bssn_midpoint_1, displacement_half, magnetic_right_half = prepare(
        bssn_midpoint_1, em.displacement_right_half, magnetic_right_half, time + 0.5 * dt
    )
    k2 = gravity_rhs(
        bssn_midpoint_1,
        displacement_half,
        magnetic_right_half,
        time + 0.5 * dt,
    )

    bssn_midpoint_2 = enforce_algebraic_constraints(
        _add_bssn_scaled(bssn_n, k2, 0.5 * dt)
    )
    bssn_midpoint_2, displacement_half, magnetic_right_half = prepare(
        bssn_midpoint_2, displacement_half, magnetic_right_half, time + 0.5 * dt
    )
    k3 = gravity_rhs(
        bssn_midpoint_2,
        displacement_half,
        magnetic_right_half,
        time + 0.5 * dt,
    )
    displacement_rhs_midpoint, magnetic_rhs_midpoint = densitized_maxwell_rhs(
        displacement_half,
        magnetic_right_half,
        bssn_midpoint_2,
        params,
    )
    magnetic_next = magnetic_n + dt * magnetic_rhs_midpoint
    displacement_next = displacement_n + dt * displacement_rhs_midpoint

    bssn_endpoint = enforce_algebraic_constraints(
        _add_bssn_scaled(bssn_n, k3, dt)
    )
    bssn_endpoint, displacement_next, magnetic_next = prepare(
        bssn_endpoint, displacement_next, magnetic_next, time + dt
    )
    k4 = gravity_rhs(
        bssn_endpoint, displacement_next, magnetic_next, time + dt
    )
    bssn_increment = jax.tree_util.tree_map(
        lambda d1, d2, d3, d4: (d1 + 2.0 * d2 + 2.0 * d3 + d4)
        / 6.0,
        k1,
        k2,
        k3,
        k4,
    )
    bssn_next = enforce_algebraic_constraints(
        _add_bssn_scaled(bssn_n, bssn_increment, dt)
    )
    bssn_next, displacement_next, magnetic_next = prepare(
        bssn_next, displacement_next, magnetic_next, time + dt
    )

    # Finish the second electric leapfrog with the final projected spacetime.
    displacement_rhs_endpoint, _ = densitized_maxwell_rhs(
        displacement_next,
        magnetic_next,
        bssn_next,
        params,
    )
    displacement_next_half = displacement_half + dt * (
        displacement_rhs_endpoint
    )
    if stage_boundary is not None:
        _, displacement_next_half, _ = stage_boundary(
            bssn_next, displacement_next_half, magnetic_next, time + 1.5 * dt
        )

    return EinsteinMaxwellVariables(
        bssn=bssn_next,
        em=DensitizedMaxwellState(
            magnetic_previous=magnetic_n,
            magnetic_current=magnetic_next,
            displacement_left_half=displacement_half,
            displacement_right_half=displacement_next_half,
        ),
    )


__all__ = [
    "bootstrap_densitized_maxwell_state",
    "common_densitized_fields",
    "common_physical_fields",
    "first_order_einstein_maxwell_step",
    "initialize_densitized_maxwell_state",
    "initialize_first_order_einstein_maxwell_state",
]
