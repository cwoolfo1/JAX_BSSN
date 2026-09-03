"""Diagnostics for densitized first-order Maxwell states."""

import jax

from JAX_BSSN.bssn import BSSNParameters

from JAX_BSSN.EM.first_order.equations import (
    densitized_displacement_divergence,
    densitized_magnetic_divergence,
)
from JAX_BSSN.EM.first_order.evolve import (
    common_densitized_fields,
    common_physical_fields,
)
from JAX_BSSN.EM.first_order.variables import DensitizedMaxwellState
from JAX_BSSN.EM.variables import EinsteinMaxwellVariables


@jax.jit
def first_order_constraint_divergences(
    state: EinsteinMaxwellVariables[DensitizedMaxwellState],
    params: BSSNParameters,
):
    """Return the source-free densitized Gauss constraints."""

    densitized_displacement, densitized_magnetic = common_densitized_fields(
        state.em
    )
    return (
        densitized_displacement_divergence(densitized_displacement, params),
        densitized_magnetic_divergence(densitized_magnetic, params),
    )


def electromagnetic_output_fields(
    state: EinsteinMaxwellVariables[DensitizedMaxwellState],
    params: BSSNParameters,
) -> dict:
    """Return cell-centered physical contravariant D and B components."""

    displacement, magnetic = common_physical_fields(state, params)
    return {
        "D": tuple(displacement[i] for i in range(3)),
        "B": tuple(magnetic[i] for i in range(3)),
    }


__all__ = [
    "electromagnetic_output_fields",
    "first_order_constraint_divergences",
]
