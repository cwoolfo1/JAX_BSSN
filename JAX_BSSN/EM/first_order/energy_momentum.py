"""BSSN sources and physical output views of PyPIC3D densities."""

from functools import partial
import jax
import jax.numpy as jnp
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS
from PyPIC3D.relativity.field_interpolation import reconstruct_vector
from JAX_BSSN.bssn.geometry import W_FLOOR_VALUE
from .grid import CENTER_LOCATION
from .coupling import quadratic_moments, sources_from_moments


@partial(jax.jit, static_argnames=("grid",))
def physical_fields_at_centers(displacement, magnetic, bssn, grid):
    W = jnp.maximum(bssn.conformal_factor, W_FLOOR_VALUE)

    def center(vector, locations):
        return W**3 * jnp.stack(
            tuple(
                grid.from_tile(v)
                for v in reconstruct_vector(vector, locations, CENTER_LOCATION)
            )
        )

    return center(displacement, D_FIELD_LOCATIONS), center(magnetic, B_FIELD_LOCATIONS)


@partial(jax.jit, static_argnames=("grid",))
def compute_densitized_electromagnetic_energy_momentum(
    displacement, magnetic, bssn, grid
):
    return sources_from_moments(quadratic_moments(displacement, magnetic, grid), bssn)
