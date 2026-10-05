"""Reflect native Yee samples while retaining equal-sized component arrays."""

import jax.numpy as jnp

from JAX_BSSN.EM.first_order.staggering import (
    DISPLACEMENT_FIELD_LOCATIONS,
    MAGNETIC_FIELD_LOCATIONS,
)
from JAX_BSSN.EM.first_order.variables import DensitizedMaxwellState


def expand_native_state(state, ghost_cells, parity):
    """Expand each history field using its radial location and vector parity."""
    fields = []
    locations_by_field = (
        MAGNETIC_FIELD_LOCATIONS,
        MAGNETIC_FIELD_LOCATIONS,
        DISPLACEMENT_FIELD_LOCATIONS,
        DISPLACEMENT_FIELD_LOCATIONS,
    )
    for field, locations in zip(state, locations_by_field):
        components = []
        for component, location in enumerate(locations):
            positive = field[component, ghost_cells:]
            if location[0] == "V":
                if parity[component] == -1:
                    positive = positive.at[0].set(0.0)
                outer = 2.0 * positive[-1:] - positive[-2:-1]
                reflected = jnp.concatenate((positive[1:], outer), axis=0)
            else:
                reflected = positive
            negative = parity[component] * jnp.flip(reflected, axis=0)
            components.append(jnp.concatenate((negative, positive), axis=0))
        fields.append(jnp.stack(components, axis=0))
    return DensitizedMaxwellState(*fields)
