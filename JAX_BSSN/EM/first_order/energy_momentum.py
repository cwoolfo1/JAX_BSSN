"""Electromagnetic stress-energy from densitized first-order fields."""

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNParameters, BSSNVariables
from JAX_BSSN.bssn.geometry import W_FLOOR_VALUE
from JAX_BSSN.EM.first_order.coupling import quadratic_moments, sources_from_moments
from JAX_BSSN.EM.first_order.equations import (
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
    """Return Eulerian ``(rho, S_i, S_ij)`` from native quadratic moments."""

    return sources_from_moments(
        quadratic_moments(densitized_displacement, densitized_magnetic, params),
        bssn,
    )


__all__ = [
    "compute_densitized_electromagnetic_energy_momentum",
    "physical_fields_at_centers",
]
