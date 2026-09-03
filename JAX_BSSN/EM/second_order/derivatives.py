"""Second-order Cartesian derivatives and spatial tensor operators."""

from functools import partial

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn.variables import BSSNParameters, get_boundary_codes
from JAX_BSSN.evolution.boundaries import SOMMERFELD_BC


@partial(jax.jit, static_argnames=("direction",))
def first_derivative(
    field: jnp.ndarray,
    direction: int,
    dx: float,
    left_bc: int,
    right_bc: int,
) -> jnp.ndarray:
    """Return a centered second-order first derivative along one array axis."""

    derivative = (
        jnp.roll(field, -1, axis=direction)
        - jnp.roll(field, 1, axis=direction)
    ) / (2.0 * dx)

    field_axis_last = jnp.moveaxis(field, direction, -1)
    derivative_axis_last = jnp.moveaxis(derivative, direction, -1)

    left_value = (
        -3.0 * field_axis_last[..., 0]
        + 4.0 * field_axis_last[..., 1]
        - field_axis_last[..., 2]
    ) / (2.0 * dx)
    derivative_axis_last = jax.lax.cond(
        left_bc == SOMMERFELD_BC,
        lambda values: values.at[..., 0].set(left_value),
        lambda values: values,
        derivative_axis_last,
    )

    right_value = (
        3.0 * field_axis_last[..., -1]
        - 4.0 * field_axis_last[..., -2]
        + field_axis_last[..., -3]
    ) / (2.0 * dx)
    derivative_axis_last = jax.lax.cond(
        right_bc == SOMMERFELD_BC,
        lambda values: values.at[..., -1].set(right_value),
        lambda values: values,
        derivative_axis_last,
    )

    return jnp.moveaxis(derivative_axis_last, -1, direction)


@partial(jax.jit, static_argnames=("direction",))
def second_derivative(
    field: jnp.ndarray,
    direction: int,
    dx: float,
    left_bc: int,
    right_bc: int,
) -> jnp.ndarray:
    """Return a centered second-order pure second derivative."""

    derivative = (
        jnp.roll(field, -1, axis=direction)
        - 2.0 * field
        + jnp.roll(field, 1, axis=direction)
    ) / dx**2

    field_axis_last = jnp.moveaxis(field, direction, -1)
    derivative_axis_last = jnp.moveaxis(derivative, direction, -1)

    left_value = (
        2.0 * field_axis_last[..., 0]
        - 5.0 * field_axis_last[..., 1]
        + 4.0 * field_axis_last[..., 2]
        - field_axis_last[..., 3]
    ) / dx**2
    derivative_axis_last = jax.lax.cond(
        left_bc == SOMMERFELD_BC,
        lambda values: values.at[..., 0].set(left_value),
        lambda values: values,
        derivative_axis_last,
    )

    right_value = (
        2.0 * field_axis_last[..., -1]
        - 5.0 * field_axis_last[..., -2]
        + 4.0 * field_axis_last[..., -3]
        - field_axis_last[..., -4]
    ) / dx**2
    derivative_axis_last = jax.lax.cond(
        right_bc == SOMMERFELD_BC,
        lambda values: values.at[..., -1].set(right_value),
        lambda values: values,
        derivative_axis_last,
    )

    return jnp.moveaxis(derivative_axis_last, -1, direction)


def spatial_derivatives(
    field: jnp.ndarray, params: BSSNParameters
) -> jnp.ndarray:
    """Return partial derivatives with derivative direction as the first axis."""

    spatial_start = field.ndim - 3
    return jnp.stack(
        [
            first_derivative(
                field,
                spatial_start + direction,
                params.dx,
                *get_boundary_codes(params, direction),
            )
            for direction in range(3)
        ],
        axis=0,
    )


def partial_second_derivatives(
    field: jnp.ndarray, params: BSSNParameters
) -> jnp.ndarray:
    """Return partial_j partial_k field with two leading derivative axes."""

    spatial_start = field.ndim - 3
    first = spatial_derivatives(field, params)
    rows = []
    for j in range(3):
        row = []
        for k in range(3):
            if j == k:
                derivative = second_derivative(
                    field,
                    spatial_start + k,
                    params.dx,
                    *get_boundary_codes(params, k),
                )
            else:
                derivative = first_derivative(
                    first[k],
                    spatial_start + j,
                    params.dx,
                    *get_boundary_codes(params, j),
                )
            row.append(derivative)
        rows.append(jnp.stack(row, axis=0))

    return jnp.stack(rows, axis=0)


def covariant_derivative_covector(
    field: jnp.ndarray,
    christoffel: jnp.ndarray,
    params: BSSNParameters,
) -> jnp.ndarray:
    """Return D_j field_i for a spatial covector."""

    partial_field = spatial_derivatives(field, params)
    connection_term = jnp.einsum(
        "mji...,m...->ji...", christoffel, field
    )
    return partial_field - connection_term


def covariant_derivative_tensor2(
    tensor: jnp.ndarray,
    christoffel: jnp.ndarray,
    params: BSSNParameters,
) -> jnp.ndarray:
    """Return D_k tensor_ij for a spatial rank-two covariant tensor."""

    partial_tensor = spatial_derivatives(tensor, params)
    first_connection = jnp.einsum(
        "mki...,mj...->kij...", christoffel, tensor
    )
    second_connection = jnp.einsum(
        "mkj...,im...->kij...", christoffel, tensor
    )
    return partial_tensor - first_connection - second_connection


def covariant_vector_laplacian(
    field: jnp.ndarray,
    inverse_metric: jnp.ndarray,
    christoffel: jnp.ndarray,
    params: BSSNParameters,
) -> jnp.ndarray:
    """Return gamma^jk D_j D_k field_i for a spatial covector."""

    partial_field = spatial_derivatives(field, params)
    second_partial = partial_second_derivatives(field, params)
    christoffel_derivatives = spatial_derivatives(christoffel, params)

    hessian = second_partial
    hessian = hessian - jnp.einsum(
        "jmki...,m...->jki...", christoffel_derivatives, field
    )
    hessian = hessian - jnp.einsum(
        "mki...,jm...->jki...", christoffel, partial_field
    )
    hessian = hessian - jnp.einsum(
        "mjk...,mi...->jki...", christoffel, partial_field
    )
    hessian = hessian + jnp.einsum(
        "mjk...,nmi...,n...->jki...", christoffel, christoffel, field
    )
    hessian = hessian - jnp.einsum(
        "mji...,km...->jki...", christoffel, partial_field
    )
    hessian = hessian + jnp.einsum(
        "mji...,nkm...,n...->jki...", christoffel, christoffel, field
    )

    return jnp.einsum("jk...,jki...->i...", inverse_metric, hessian)


def lie_derivative_covector(
    field: jnp.ndarray,
    shift: jnp.ndarray,
    params: BSSNParameters,
) -> jnp.ndarray:
    """Return the spatial Lie derivative of a covector along the shift."""

    partial_field = spatial_derivatives(field, params)
    partial_shift = spatial_derivatives(shift, params)

    advection = jnp.einsum("j...,ji...->i...", shift, partial_field)
    basis_change = jnp.einsum("j...,ij...->i...", field, partial_shift)
    return advection + basis_change
