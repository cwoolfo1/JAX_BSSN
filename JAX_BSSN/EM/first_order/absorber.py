"""Boundary-local relaxation about fixed initial native Yee densities."""

import math
import jax.numpy as jnp
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS


def make_reference_sponge(grid, params, displacement, magnetic, *, width, order=4,
                          target_attenuation=1.0e-3):
    """Return D/B tuples of (rates, references, exterior masks).

    References include all native ghost samples and stay fixed as the metric
    evolves. Width is a coordinate length. The nominal attenuation assumes a
    unit-speed ray through one face; it does not specify actual reflection.
    """
    if width <= 0 or order < 1 or not 0 < target_attenuation < 1:
        raise ValueError('Sponge needs positive width/order and attenuation between zero and one')
    active = tuple(bc != 0 for bc in grid.static.boundary_conditions)
    if any(bc not in (0, 3) for bc in grid.static.boundary_conditions):
        raise ValueError('Reference sponge requires periodic or constant exterior axes')
    if any(on and 2 * width >= n * params.dx for on, n in zip(active, grid.shape)):
        raise ValueError('Opposite sponge layers must leave an undamped interior')
    sigma_max = -(order + 1) * math.log(target_attenuation) / width
    origins = (params.x_min, params.y_min, params.z_min)
    result = []
    for reference, locations in ((displacement, D_FIELD_LOCATIONS), (magnetic, B_FIELD_LOCATIONS)):
        rates, masks = [], []
        for initial, location in zip(reference, locations):
            sigma = jnp.zeros_like(initial)
            exterior = jnp.zeros(initial.shape, dtype=bool)
            coordinates = grid.coordinates(params, location, tiled=True)
            for axis, (x, origin, n, on) in enumerate(zip(coordinates, origins, grid.shape, active)):
                if not on:
                    continue
                lower = origin - params.dx / 2
                upper = lower + n * params.dx
                for distance in (x - lower, upper - x):
                    eta = jnp.clip(1 - distance / width, 0, 1)
                    sigma = sigma + sigma_max * eta**order
                # Index tests include face-coincident V sites without roundoff
                # dependence on the coordinate origin or grid spacing.
                index = jnp.arange(initial.shape[axis + 3]) - grid.guard_cells
                dims = [1] * 6
                dims[axis + 3] = index.size
                edge = ((index < 0) | (index >= n - (location[axis] == 'V')))
                exterior = exterior | edge.reshape(dims)
            rates.append(sigma)
            masks.append(exterior)
        result.append((tuple(rates), tuple(reference), tuple(masks)))
    return tuple(result)
