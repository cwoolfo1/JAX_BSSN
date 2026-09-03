"""Native-grid periodic and Sommerfeld boundaries for vector densities."""

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.bssn.variables import get_boundary_codes
from JAX_BSSN.evolution.boundaries import SOMMERFELD_BC

from JAX_BSSN.EM.first_order.staggering import (
    DISPLACEMENT_FIELD_LOCATIONS,
    MAGNETIC_FIELD_LOCATIONS,
)
from JAX_BSSN.EM.first_order.variables import BSSNYeeGeometry


def _spatial_axis(field, direction):
    return field.ndim - 3 + direction


def _scalar_gradient(field, params):
    derivatives = []
    for direction in range(3):
        axis = _spatial_axis(field, direction)
        if field.shape[axis] == 1:
            derivatives.append(jnp.zeros_like(field))
            continue

        derivative = (
            jnp.roll(field, -1, axis=axis)
            - jnp.roll(field, 1, axis=axis)
        ) / (2.0 * params.dx)
        left = (
            -3.0 * jnp.take(field, 0, axis=axis)
            + 4.0 * jnp.take(field, 1, axis=axis)
            - jnp.take(field, 2, axis=axis)
        ) / (2.0 * params.dx)
        right = (
            3.0 * jnp.take(field, -1, axis=axis)
            - 4.0 * jnp.take(field, -2, axis=axis)
            + jnp.take(field, -3, axis=axis)
        ) / (2.0 * params.dx)
        derivative = derivative.at[
            (slice(None),) * axis + (0,)
        ].set(left)
        derivative = derivative.at[
            (slice(None),) * axis + (-1,)
        ].set(right)
        derivatives.append(derivative)
    return jnp.stack(tuple(derivatives), axis=0)


def _outward_boundary_covector(shape, params, dtype):
    indices = tuple(
        jnp.arange(size).reshape(
            (1,) * direction + (size,) + (1,) * (2 - direction)
        )
        for direction, size in enumerate(shape)
    )
    components = []
    for direction, (index, size) in enumerate(zip(indices, shape)):
        left_bc, right_bc = get_boundary_codes(params, direction)
        component = jnp.zeros(shape, dtype=dtype)
        component += jnp.where(
            (left_bc == SOMMERFELD_BC) & (index == 0), -1.0, 0.0
        )
        component += jnp.where(
            (right_bc == SOMMERFELD_BC) & (index == size - 1), 1.0, 0.0
        )
        components.append(component)
    return jnp.stack(tuple(components), axis=0)


def _native_radius(shape, location, params, dtype):
    minima = (params.x_min, params.y_min, params.z_min)
    coordinates = []
    for direction, (size, site, minimum) in enumerate(
        zip(shape, location, minima)
    ):
        offset = -0.5 if site == "V" else 0.0
        coordinate = jnp.asarray(minimum, dtype=dtype) + params.dx * (
            jnp.arange(size, dtype=dtype) + offset
        )
        coordinates.append(
            coordinate.reshape(
                (1,) * direction + (size,) + (1,) * (2 - direction)
            )
        )
    return jnp.sqrt(sum(coordinate**2 for coordinate in coordinates))


def _apply_component_boundary(
    field,
    rhs,
    metric,
    location,
    params,
):
    shape = field.shape[-3:]
    boundary_covector = _outward_boundary_covector(
        shape, params, field.dtype
    )
    norm_squared = jnp.einsum(
        "ij...,i...,j...->...",
        metric.W**2 * metric.inverse_conformal_metric,
        boundary_covector,
        boundary_covector,
    )
    mask = norm_squared > 0.0
    safe_norm = jnp.sqrt(jnp.where(mask, norm_squared, 1.0))
    outward_normal = jnp.einsum(
        "ij...,j...->i...",
        metric.W**2 * metric.inverse_conformal_metric,
        boundary_covector,
    ) / safe_norm

    gradient = _scalar_gradient(field, params)
    characteristic_velocity = metric.shift - metric.lapse * outward_normal
    boundary_rhs = jnp.einsum(
        "i...,i...->...", characteristic_velocity, gradient
    )

    # A zero-asymptotic 1/r Sommerfeld falloff is used away from the origin.
    radius = _native_radius(shape, location, params, field.dtype)
    safe_radius = jnp.where(radius > params.dx, radius, 1.0)
    boundary_rhs -= jnp.where(
        radius > params.dx, metric.lapse * field / safe_radius, 0.0
    )
    return jnp.where(mask, boundary_rhs, rhs)


@jax.jit
def apply_densitized_sommerfeld_boundaries(
    densitized_displacement,
    densitized_magnetic,
    displacement_rhs,
    magnetic_rhs,
    geometry: BSSNYeeGeometry,
    params: BSSNParameters,
):
    """Replace active outer-face RHS values on each native Yee grid."""

    displacement_rhs = jnp.stack(
        tuple(
            _apply_component_boundary(
                densitized_displacement[i],
                displacement_rhs[i],
                geometry.displacement[i],
                DISPLACEMENT_FIELD_LOCATIONS[i],
                params,
            )
            for i in range(3)
        ),
        axis=0,
    )
    magnetic_rhs = jnp.stack(
        tuple(
            _apply_component_boundary(
                densitized_magnetic[i],
                magnetic_rhs[i],
                geometry.magnetic[i],
                MAGNETIC_FIELD_LOCATIONS[i],
                params,
            )
            for i in range(3)
        ),
        axis=0,
    )
    return displacement_rhs, magnetic_rhs


__all__ = ["apply_densitized_sommerfeld_boundaries"]
