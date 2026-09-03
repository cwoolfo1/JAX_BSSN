"""Electromagnetic stress-energy projections for Einstein-Maxwell coupling."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNVariables
from JAX_BSSN.bssn.geometry import W_FLOOR_VALUE
from JAX_BSSN.bssn.tensor_algebra import (
    determinant_3x3_metric,
    invert_3x3_metric,
)


LEVI_CIVITA_SYMBOL = jnp.asarray(
    [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ]
)


class ElectromagneticStressEnergy(NamedTuple):
    """Eulerian electromagnetic stress and conformally scaled stress."""

    energy_density: jnp.ndarray
    momentum_density: jnp.ndarray
    spatial_stress: jnp.ndarray
    stress_trace: jnp.ndarray
    anisotropic_stress: jnp.ndarray
    scaled_spatial_stress: jnp.ndarray


def _metric_from_bssn(bssn: BSSNVariables):
    """Return physical metric data without differentiating inverse powers of W."""

    W = jnp.maximum(bssn.conformal_factor, W_FLOOR_VALUE)
    inverse_conformal_metric = invert_3x3_metric(bssn.conformal_metric)
    metric = bssn.conformal_metric / W**2
    inverse_metric = W**2 * inverse_conformal_metric

    sqrt_conformal_determinant = jnp.sqrt(
        determinant_3x3_metric(bssn.conformal_metric)
    )
    sqrt_metric_determinant = sqrt_conformal_determinant / W**3
    return W, metric, inverse_metric, sqrt_metric_determinant


@jax.jit
def compute_electromagnetic_stress_energy(
    electric_field: jnp.ndarray,
    magnetic_field: jnp.ndarray,
    bssn: BSSNVariables,
) -> ElectromagneticStressEnergy:
    """Return all 3+1 electromagnetic stress-energy projections.

    ``electric_field`` and ``magnetic_field`` are physical spatial covectors.
    Lorentz-Heaviside units are used, while the Einstein ``8 pi`` coupling is
    applied by the BSSN matter evolution rather than in this function.
    """

    W, metric, inverse_metric, sqrt_metric_determinant = _metric_from_bssn(bssn)

    electric_field_up = jnp.einsum(
        "ij...,j...->i...", inverse_metric, electric_field
    )
    magnetic_field_up = jnp.einsum(
        "ij...,j...->i...", inverse_metric, magnetic_field
    )

    electric_field_squared = jnp.einsum(
        "i...,i...->...", electric_field, electric_field_up
    )
    magnetic_field_squared = jnp.einsum(
        "i...,i...->...", magnetic_field, magnetic_field_up
    )
    energy_density = 0.5 * (
        electric_field_squared + magnetic_field_squared
    )

    levi_civita = (
        LEVI_CIVITA_SYMBOL.astype(metric.dtype)[:, :, :, None, None, None]
        * sqrt_metric_determinant
    )
    momentum_density = jnp.einsum(
        "ijk...,j...,k...->i...",
        levi_civita,
        electric_field_up,
        magnetic_field_up,
    )

    field_outer = jnp.einsum(
        "i...,j...->ij...", electric_field, electric_field
    ) + jnp.einsum(
        "i...,j...->ij...", magnetic_field, magnetic_field
    )
    spatial_stress = metric * energy_density - field_outer

    # Electromagnetic stress is trace free in four dimensions, so S=rho.
    stress_trace = energy_density
    anisotropic_stress = spatial_stress - metric * stress_trace / 3.0

    # This is the combination consumed by the conformal A_ij equation.  Forming
    # it directly avoids a second inverse power of W near puncture-like regions.
    scaled_spatial_stress = (
        bssn.conformal_metric * energy_density - W**2 * field_outer
    )

    return ElectromagneticStressEnergy(
        energy_density=energy_density,
        momentum_density=momentum_density,
        spatial_stress=spatial_stress,
        stress_trace=stress_trace,
        anisotropic_stress=anisotropic_stress,
        scaled_spatial_stress=scaled_spatial_stress,
    )


@jax.jit
def compute_electromagnetic_energy_momentum(
    electric_field: jnp.ndarray,
    magnetic_field: jnp.ndarray,
    bssn: BSSNVariables,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``(rho, S_i, S_ij)`` for the supplied BSSN geometry."""

    stress_energy = compute_electromagnetic_stress_energy(
        electric_field, magnetic_field, bssn
    )
    return (
        stress_energy.energy_density,
        stress_energy.momentum_density,
        stress_energy.spatial_stress,
    )


__all__ = [
    "ElectromagneticStressEnergy",
    "compute_electromagnetic_energy_momentum",
    "compute_electromagnetic_stress_energy",
]
