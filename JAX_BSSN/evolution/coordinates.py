"""Uniform physical Cartesian coordinates, including staggered Yee sites."""

import jax.numpy as jnp


def axis_coordinates(size, direction, params, dtype, site="C"):
    """Return centers or their lower vertices, including parity ghosts."""
    minimum = (params.x_min, params.y_min, params.z_min)[direction]
    offset = -0.5 if site == "V" else 0.0
    return jnp.asarray(minimum, dtype=dtype) + jnp.asarray(params.dx, dtype=dtype) * (
        jnp.arange(size, dtype=dtype) + offset
    )


def grid_coordinates(shape, params, dtype, location=("C", "C", "C")):
    """Broadcast physical sample coordinates over three spatial dimensions."""
    return tuple(
        axis_coordinates(size, d, params, dtype, site).reshape(
            (1,) * d + (size,) + (1,) * (2 - d)
        )
        for d, (size, site) in enumerate(zip(shape, location))
    )


def cylindrical_volume_weights(shape, params, dtype, location=("C", "C", "C")):
    """Cylindrical coordinate-volume weights; common cell factors cancel in norms."""
    rho = axis_coordinates(shape[0], 0, params, dtype, location[0])
    return jnp.broadcast_to(jnp.abs(rho)[:, None], (shape[0], shape[-1]))
