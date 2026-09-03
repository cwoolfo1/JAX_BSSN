"""Synchronized Einstein--Maxwell time integration."""

import jax

from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.evolution.time_evolve import (
    compute_bssn_rhs_with_matter,
    enforce_algebraic_constraints,
)

from JAX_BSSN.EM.equations import compute_em_rhs
from JAX_BSSN.EM.energy_momentum import (
    compute_electromagnetic_energy_momentum,
)
from JAX_BSSN.EM.variables import EinsteinMaxwellVariables


def _add_scaled(state, rhs, scale):
    return jax.tree_util.tree_map(
        lambda value, derivative: value + scale * derivative,
        state,
        rhs,
    )


def _project_bssn(state: EinsteinMaxwellVariables) -> EinsteinMaxwellVariables:
    return EinsteinMaxwellVariables(
        bssn=enforce_algebraic_constraints(state.bssn),
        em=state.em,
    )


@jax.jit
def compute_einstein_maxwell_rhs(
    state: EinsteinMaxwellVariables, params: BSSNParameters
) -> EinsteinMaxwellVariables:
    """Return synchronized two-way Einstein-Maxwell stage derivatives."""

    rho, momentum_density, spatial_stress = (
        compute_electromagnetic_energy_momentum(
            state.em.electric_field,
            state.em.magnetic_field,
            state.bssn,
        )
    )
    bssn_rhs = compute_bssn_rhs_with_matter(
        state.bssn,
        params,
        rho,
        momentum_density,
        spatial_stress,
    )
    em_rhs = compute_em_rhs(
        state.em,
        state.bssn,
        bssn_rhs,
        params,
        backreaction_sources=(rho, momentum_density, spatial_stress),
    )
    return EinsteinMaxwellVariables(bssn=bssn_rhs, em=em_rhs)


@jax.jit
def einstein_maxwell_rk4_step(
    state: EinsteinMaxwellVariables, params: BSSNParameters
) -> EinsteinMaxwellVariables:
    """Advance BSSN and Maxwell variables through the same RK4 stages."""

    dt = params.dt
    state = _project_bssn(state)

    k1 = compute_einstein_maxwell_rhs(state, params)
    midpoint = _project_bssn(_add_scaled(state, k1, 0.5 * dt))

    k2 = compute_einstein_maxwell_rhs(midpoint, params)
    midpoint = _project_bssn(_add_scaled(state, k2, 0.5 * dt))

    k3 = compute_einstein_maxwell_rhs(midpoint, params)
    endpoint = _project_bssn(_add_scaled(state, k3, dt))

    k4 = compute_einstein_maxwell_rhs(endpoint, params)
    increment = jax.tree_util.tree_map(
        lambda rhs1, rhs2, rhs3, rhs4: (
            rhs1 + 2.0 * rhs2 + 2.0 * rhs3 + rhs4
        )
        / 6.0,
        k1,
        k2,
        k3,
        k4,
    )

    return _project_bssn(_add_scaled(state, increment, dt))


@jax.jit
def evolve_einstein_maxwell_steps(
    state: EinsteinMaxwellVariables,
    params: BSSNParameters,
    num_steps: int,
) -> EinsteinMaxwellVariables:
    """Advance a fixed number of coupled steps without storing the trajectory."""

    return jax.lax.fori_loop(
        0,
        num_steps,
        lambda _, current: einstein_maxwell_rk4_step(current, params),
        state,
    )


__all__ = [
    "compute_einstein_maxwell_rhs",
    "einstein_maxwell_rk4_step",
    "evolve_einstein_maxwell_steps",
]
