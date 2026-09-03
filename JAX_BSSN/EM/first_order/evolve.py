"""Synchronized BSSN RK4 and doubled-leapfrog Maxwell evolution."""

import jax
import jax.numpy as jnp

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
from JAX_BSSN.EM.first_order.geometry import compute_bssn_yee_geometry
from JAX_BSSN.EM.first_order.variables import (
    DensitizedMaxwellState,
    FirstOrderEinsteinMaxwellState,
)


def _add_bssn_scaled(
    bssn: BSSNVariables, rhs: BSSNVariables, scale
) -> BSSNVariables:
    return BSSNVariables(
        *(field + scale * derivative for field, derivative in zip(bssn, rhs))
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
    state: FirstOrderEinsteinMaxwellState,
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
    geometry = compute_bssn_yee_geometry(bssn, params)
    displacement_rhs, magnetic_rhs = densitized_maxwell_rhs(
        densitized_displacement,
        densitized_magnetic,
        geometry,
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
) -> FirstOrderEinsteinMaxwellState:
    """Initialize from physical contravariant fields on their native Yee sites."""

    bssn = enforce_algebraic_constraints(bssn)
    geometry = compute_bssn_yee_geometry(bssn, params)
    densitized_displacement = jnp.stack(
        tuple(
            geometry.displacement[i].W**-3 * physical_displacement[i]
            for i in range(3)
        ),
        axis=0,
    )
    densitized_magnetic = jnp.stack(
        tuple(
            geometry.magnetic[i].W**-3 * physical_magnetic[i]
            for i in range(3)
        ),
        axis=0,
    )
    em = initialize_densitized_maxwell_state(
        densitized_displacement,
        densitized_magnetic,
        bssn,
        params,
    )
    return FirstOrderEinsteinMaxwellState(bssn=bssn, em=em)


@jax.jit
def first_order_einstein_maxwell_step(
    state: FirstOrderEinsteinMaxwellState,
    params: BSSNParameters,
) -> FirstOrderEinsteinMaxwellState:
    """Advance one two-way coupled RK4/doubled-leapfrog timestep."""

    dt = params.dt
    bssn_n = enforce_algebraic_constraints(state.bssn)
    em = state.em
    displacement_n, magnetic_n = common_densitized_fields(em)

    # BSSN k1 and the first member of the doubled magnetic leapfrog.
    k1 = _bssn_rhs_from_densities(
        bssn_n, displacement_n, magnetic_n, params
    )
    geometry_n = compute_bssn_yee_geometry(bssn_n, params)
    _, magnetic_rhs_n = densitized_maxwell_rhs(
        displacement_n, magnetic_n, geometry_n, params
    )
    magnetic_left_half = 0.5 * (
        em.magnetic_previous + em.magnetic_current
    )
    magnetic_right_half = magnetic_left_half + dt * magnetic_rhs_n

    # Both RK4 midpoint stages represent the same physical half time, but use
    # their own projected BSSN metrics for stress-energy and constitutive fields.
    bssn_midpoint_1 = enforce_algebraic_constraints(
        _add_bssn_scaled(bssn_n, k1, 0.5 * dt)
    )
    k2 = _bssn_rhs_from_densities(
        bssn_midpoint_1,
        em.displacement_right_half,
        magnetic_right_half,
        params,
    )

    bssn_midpoint_2 = enforce_algebraic_constraints(
        _add_bssn_scaled(bssn_n, k2, 0.5 * dt)
    )
    k3 = _bssn_rhs_from_densities(
        bssn_midpoint_2,
        em.displacement_right_half,
        magnetic_right_half,
        params,
    )
    midpoint_geometry = compute_bssn_yee_geometry(
        bssn_midpoint_2, params
    )
    displacement_rhs_midpoint, magnetic_rhs_midpoint = densitized_maxwell_rhs(
        em.displacement_right_half,
        magnetic_right_half,
        midpoint_geometry,
        params,
    )
    magnetic_next = magnetic_n + dt * magnetic_rhs_midpoint
    displacement_next = displacement_n + dt * displacement_rhs_midpoint

    bssn_endpoint = enforce_algebraic_constraints(
        _add_bssn_scaled(bssn_n, k3, dt)
    )
    k4 = _bssn_rhs_from_densities(
        bssn_endpoint, displacement_next, magnetic_next, params
    )
    bssn_increment = BSSNVariables(
        *(
            (d1 + 2.0 * d2 + 2.0 * d3 + d4) / 6.0
            for d1, d2, d3, d4 in zip(k1, k2, k3, k4)
        )
    )
    bssn_next = enforce_algebraic_constraints(
        _add_bssn_scaled(bssn_n, bssn_increment, dt)
    )

    # Finish the second electric leapfrog with the final projected spacetime.
    endpoint_geometry = compute_bssn_yee_geometry(bssn_next, params)
    displacement_rhs_endpoint, _ = densitized_maxwell_rhs(
        displacement_next,
        magnetic_next,
        endpoint_geometry,
        params,
    )
    displacement_next_half = em.displacement_right_half + dt * (
        displacement_rhs_endpoint
    )

    return FirstOrderEinsteinMaxwellState(
        bssn=bssn_next,
        em=DensitizedMaxwellState(
            magnetic_previous=magnetic_n,
            magnetic_current=magnetic_next,
            displacement_left_half=em.displacement_right_half,
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
