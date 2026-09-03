"""Electromagnetic stress-energy from densitized first-order fields."""

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNParameters, BSSNVariables
from JAX_BSSN.bssn.geometry import W_FLOOR_VALUE
from JAX_BSSN.EM.first_order.equations import (
    LEVI_CIVITA_SYMBOL,
    collocate_densitized_fields,
)


@jax.jit
def physical_fields_at_centers(
    densitized_displacement: jnp.ndarray,
    densitized_magnetic: jnp.ndarray,
    bssn: BSSNVariables,
    params: BSSNParameters,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return physical contravariant ``(D^i, B^i)`` at BSSN centers."""

    displacement_density, magnetic_density = collocate_densitized_fields(
        densitized_displacement, densitized_magnetic, params
    )
    W = jnp.maximum(bssn.conformal_factor, W_FLOOR_VALUE)
    return W**3 * displacement_density, W**3 * magnetic_density


@jax.jit
def compute_densitized_electromagnetic_energy_momentum(
    densitized_displacement: jnp.ndarray,
    densitized_magnetic: jnp.ndarray,
    bssn: BSSNVariables,
    params: BSSNParameters,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return Eulerian ``(rho, S_i, S_ij)`` without a determinant solve."""

    displacement_up, magnetic_up = physical_fields_at_centers(
        densitized_displacement, densitized_magnetic, bssn, params
    )
    W = jnp.maximum(bssn.conformal_factor, W_FLOOR_VALUE)
    metric = bssn.conformal_metric / W**2

    displacement_down = jnp.einsum(
        "ij...,j...->i...", metric, displacement_up
    )
    magnetic_down = jnp.einsum(
        "ij...,j...->i...", metric, magnetic_up
    )
    displacement_squared = jnp.einsum(
        "i...,i...->...", displacement_down, displacement_up
    )
    magnetic_squared = jnp.einsum(
        "i...,i...->...", magnetic_down, magnetic_up
    )
    energy_density = 0.5 * (displacement_squared + magnetic_squared)

    epsilon = LEVI_CIVITA_SYMBOL.astype(displacement_up.dtype)[
        :, :, :, None, None, None
    ] * W**-3
    momentum_density = jnp.einsum(
        "ijk...,j...,k...->i...", epsilon, displacement_up, magnetic_up
    )
    field_outer = jnp.einsum(
        "i...,j...->ij...", displacement_down, displacement_down
    ) + jnp.einsum(
        "i...,j...->ij...", magnetic_down, magnetic_down
    )
    spatial_stress = metric * energy_density - field_outer
    return energy_density, momentum_density, spatial_stress


__all__ = [
    "compute_densitized_electromagnetic_energy_momentum",
    "physical_fields_at_centers",
]
