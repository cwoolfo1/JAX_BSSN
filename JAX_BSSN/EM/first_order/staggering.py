"""Logical FPIC Yee locations and second-order interpolation."""

from functools import partial

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.bssn.variables import get_boundary_codes
from JAX_BSSN.evolution.boundaries import SOMMERFELD_BC


DISPLACEMENT_FIELD_LOCATIONS = (
    ("V", "C", "C"),
    ("C", "V", "C"),
    ("C", "C", "V"),
)
MAGNETIC_FIELD_LOCATIONS = (
    ("C", "V", "V"),
    ("V", "C", "V"),
    ("V", "V", "C"),
)
CENTER_LOCATION = ("C", "C", "C")


def _axis_index(array: jnp.ndarray, spatial_axis: int) -> int:
    return array.ndim - 3 + spatial_axis


def _center_to_vertex(
    array: jnp.ndarray,
    spatial_axis: int,
    params: BSSNParameters,
) -> jnp.ndarray:
    """Interpolate centers to the lower bounding vertex on one axis."""

    axis = _axis_index(array, spatial_axis)
    if array.shape[axis] == 1:
        return array

    interpolated = 0.5 * (array + jnp.roll(array, 1, axis=axis))
    left_bc, _ = get_boundary_codes(params, spatial_axis)

    first = jnp.take(array, 0, axis=axis)
    second = jnp.take(array, 1, axis=axis)
    lower_face = 1.5 * first - 0.5 * second

    return jax.lax.cond(
        left_bc == SOMMERFELD_BC,
        lambda values: values.at[
            (slice(None),) * axis + (0,)
        ].set(lower_face),
        lambda values: values,
        interpolated,
    )


def _vertex_to_center(
    array: jnp.ndarray,
    spatial_axis: int,
    params: BSSNParameters,
) -> jnp.ndarray:
    """Interpolate lower-face vertices to the adjacent cell centers."""

    axis = _axis_index(array, spatial_axis)
    if array.shape[axis] == 1:
        return array

    interpolated = 0.5 * (array + jnp.roll(array, -1, axis=axis))
    _, right_bc = get_boundary_codes(params, spatial_axis)

    last = jnp.take(array, -1, axis=axis)
    previous = jnp.take(array, -2, axis=axis)
    upper_center = 1.5 * last - 0.5 * previous

    return jax.lax.cond(
        right_bc == SOMMERFELD_BC,
        lambda values: values.at[
            (slice(None),) * axis + (-1,)
        ].set(upper_center),
        lambda values: values,
        interpolated,
    )


@partial(jax.jit, static_argnames=("source_location", "target_location"))
def interpolate_between_locations(
    array: jnp.ndarray,
    source_location: tuple[str, str, str],
    target_location: tuple[str, str, str],
    params: BSSNParameters,
) -> jnp.ndarray:
    """Move an array between equally sized logical Yee grids."""

    result = array
    for spatial_axis, (source_axis, target_axis) in enumerate(
        zip(source_location, target_location)
    ):
        if source_axis == target_axis:
            continue
        if source_axis == "C":
            result = _center_to_vertex(result, spatial_axis, params)
        else:
            result = _vertex_to_center(result, spatial_axis, params)

    return result


def vector_at_location(
    vector: jnp.ndarray,
    source_locations,
    target_location,
    params: BSSNParameters,
) -> jnp.ndarray:
    """Interpolate all components of a staggered vector to one Yee site."""

    return jnp.stack(
        tuple(
            interpolate_between_locations(
                vector[component],
                source_locations[component],
                target_location,
                params,
            )
            for component in range(3)
        ),
        axis=0,
    )


__all__ = [
    "CENTER_LOCATION",
    "DISPLACEMENT_FIELD_LOCATIONS",
    "MAGNETIC_FIELD_LOCATIONS",
    "interpolate_between_locations",
    "vector_at_location",
]
