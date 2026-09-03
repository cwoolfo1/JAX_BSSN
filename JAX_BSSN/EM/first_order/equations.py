"""Source-free densitized Maxwell equations on the FPIC Yee grid."""

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNParameters, BSSNVariables
from JAX_BSSN.bssn.variables import get_boundary_codes
from JAX_BSSN.evolution.boundaries import SOMMERFELD_BC

from JAX_BSSN.EM.first_order.staggering import (
    CENTER_LOCATION,
    DISPLACEMENT_FIELD_LOCATIONS,
    MAGNETIC_FIELD_LOCATIONS,
    vector_at_location,
)
from JAX_BSSN.EM.first_order.boundaries import (
    _apply_densitized_sommerfeld_boundaries_with_geometry,
)
from JAX_BSSN.EM.first_order.geometry import (
    _metric_fields_at_location,
    _metric_fields_on_yee_sites,
)


LEVI_CIVITA_SYMBOL = jnp.asarray(
    [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ]
)


def _physical_contravariant(density, W):
    # sqrt(gamma)=W^-3, hence V^i=W^3 mathcal(V)^i.
    return W**3 * density


def _physical_covariant(contravariant, W, conformal_metric):
    # gamma_ij=W^-2 conformal_gamma_ij.
    return W**-2 * jnp.einsum(
        "ij...,j...->i...", conformal_metric, contravariant
    )


def _compute_covariant_E(
    densitized_displacement: jnp.ndarray,
    densitized_magnetic: jnp.ndarray,
    displacement_geometry,
    params: BSSNParameters,
) -> jnp.ndarray:
    components = []
    epsilon = LEVI_CIVITA_SYMBOL.astype(densitized_displacement.dtype)
    for target_component, target_location in enumerate(
        DISPLACEMENT_FIELD_LOCATIONS
    ):
        W, lapse, shift, conformal_metric, _ = displacement_geometry[
            target_component
        ]
        displacement_density = vector_at_location(
            densitized_displacement,
            DISPLACEMENT_FIELD_LOCATIONS,
            target_location,
            params,
        )
        magnetic_density = vector_at_location(
            densitized_magnetic,
            MAGNETIC_FIELD_LOCATIONS,
            target_location,
            params,
        )

        displacement_up = _physical_contravariant(
            displacement_density, W
        )
        magnetic_up = _physical_contravariant(magnetic_density, W)
        displacement_down = _physical_covariant(
            displacement_up, W, conformal_metric
        )
        sqrt_gamma = W**-3
        shift_cross_magnetic = jnp.einsum(
            "jk,j...,k...->...",
            epsilon[target_component],
            shift,
            magnetic_up,
        )
        components.append(
            lapse * displacement_down[target_component]
            + sqrt_gamma * shift_cross_magnetic
        )

    return jnp.stack(tuple(components), axis=0)


@jax.jit
def compute_covariant_E(
    densitized_displacement: jnp.ndarray,
    densitized_magnetic: jnp.ndarray,
    bssn: BSSNVariables,
    params: BSSNParameters,
) -> jnp.ndarray:
    """Return Entity ``E_i`` on the three displacement Yee sites."""

    displacement_geometry = tuple(
        _metric_fields_at_location(bssn, location, params)
        for location in DISPLACEMENT_FIELD_LOCATIONS
    )
    return _compute_covariant_E(
        densitized_displacement,
        densitized_magnetic,
        displacement_geometry,
        params,
    )


def _compute_covariant_H(
    densitized_displacement: jnp.ndarray,
    densitized_magnetic: jnp.ndarray,
    magnetic_geometry,
    params: BSSNParameters,
) -> jnp.ndarray:

    components = []
    epsilon = LEVI_CIVITA_SYMBOL.astype(densitized_displacement.dtype)
    for target_component, target_location in enumerate(
        MAGNETIC_FIELD_LOCATIONS
    ):
        W, lapse, shift, conformal_metric, _ = magnetic_geometry[
            target_component
        ]
        displacement_density = vector_at_location(
            densitized_displacement,
            DISPLACEMENT_FIELD_LOCATIONS,
            target_location,
            params,
        )
        magnetic_density = vector_at_location(
            densitized_magnetic,
            MAGNETIC_FIELD_LOCATIONS,
            target_location,
            params,
        )

        displacement_up = _physical_contravariant(
            displacement_density, W
        )
        magnetic_up = _physical_contravariant(magnetic_density, W)
        magnetic_down = _physical_covariant(
            magnetic_up, W, conformal_metric
        )
        sqrt_gamma = W**-3
        shift_cross_displacement = jnp.einsum(
            "jk,j...,k...->...",
            epsilon[target_component],
            shift,
            displacement_up,
        )
        components.append(
            lapse * magnetic_down[target_component]
            - sqrt_gamma * shift_cross_displacement
        )

    return jnp.stack(tuple(components), axis=0)


@jax.jit
def compute_covariant_H(
    densitized_displacement: jnp.ndarray,
    densitized_magnetic: jnp.ndarray,
    bssn: BSSNVariables,
    params: BSSNParameters,
) -> jnp.ndarray:
    """Return Entity ``H_i`` on the three magnetic Yee sites."""

    magnetic_geometry = tuple(
        _metric_fields_at_location(bssn, location, params)
        for location in MAGNETIC_FIELD_LOCATIONS
    )
    return _compute_covariant_H(
        densitized_displacement,
        densitized_magnetic,
        magnetic_geometry,
        params,
    )


def _backward_difference(
    field: jnp.ndarray,
    spatial_axis: int,
    params: BSSNParameters,
) -> jnp.ndarray:
    axis = field.ndim - 3 + spatial_axis
    if field.shape[axis] == 1:
        return jnp.zeros_like(field)

    derivative = (field - jnp.roll(field, 1, axis=axis)) / params.dx
    left_bc, _ = get_boundary_codes(params, spatial_axis)
    first = jnp.take(field, 0, axis=axis)
    second = jnp.take(field, 1, axis=axis)
    third = jnp.take(field, 2, axis=axis)
    left_derivative = (-3.0 * first + 4.0 * second - third) / (
        2.0 * params.dx
    )
    return jax.lax.cond(
        left_bc == SOMMERFELD_BC,
        lambda values: values.at[
            (slice(None),) * axis + (0,)
        ].set(left_derivative),
        lambda values: values,
        derivative,
    )


def _forward_difference(
    field: jnp.ndarray,
    spatial_axis: int,
    params: BSSNParameters,
) -> jnp.ndarray:
    axis = field.ndim - 3 + spatial_axis
    if field.shape[axis] == 1:
        return jnp.zeros_like(field)

    derivative = (jnp.roll(field, -1, axis=axis) - field) / params.dx
    _, right_bc = get_boundary_codes(params, spatial_axis)
    last = jnp.take(field, -1, axis=axis)
    previous = jnp.take(field, -2, axis=axis)
    previous_two = jnp.take(field, -3, axis=axis)
    right_derivative = (3.0 * last - 4.0 * previous + previous_two) / (
        2.0 * params.dx
    )
    return jax.lax.cond(
        right_bc == SOMMERFELD_BC,
        lambda values: values.at[
            (slice(None),) * axis + (-1,)
        ].set(right_derivative),
        lambda values: values,
        derivative,
    )


@jax.jit
def curl_E_to_densitized_B(
    electric_covector: jnp.ndarray, params: BSSNParameters
) -> jnp.ndarray:
    """Return ``[ijk] partial_j E_k`` on magnetic Yee sites."""

    E1, E2, E3 = electric_covector
    return jnp.stack(
        (
            _backward_difference(E3, 1, params)
            - _backward_difference(E2, 2, params),
            _backward_difference(E1, 2, params)
            - _backward_difference(E3, 0, params),
            _backward_difference(E2, 0, params)
            - _backward_difference(E1, 1, params),
        ),
        axis=0,
    )


@jax.jit
def curl_H_to_densitized_D(
    magnetic_covector: jnp.ndarray, params: BSSNParameters
) -> jnp.ndarray:
    """Return ``[ijk] partial_j H_k`` on displacement Yee sites."""

    H1, H2, H3 = magnetic_covector
    return jnp.stack(
        (
            _forward_difference(H3, 1, params)
            - _forward_difference(H2, 2, params),
            _forward_difference(H1, 2, params)
            - _forward_difference(H3, 0, params),
            _forward_difference(H2, 0, params)
            - _forward_difference(H1, 1, params),
        ),
        axis=0,
    )


@jax.jit
def densitized_displacement_divergence(
    densitized_displacement: jnp.ndarray, params: BSSNParameters
) -> jnp.ndarray:
    """Return ``partial_i mathcal(D)^i`` at cell centers."""

    return sum(
        _forward_difference(densitized_displacement[i], i, params)
        for i in range(3)
    )


@jax.jit
def densitized_magnetic_divergence(
    densitized_magnetic: jnp.ndarray, params: BSSNParameters
) -> jnp.ndarray:
    """Return ``partial_i mathcal(B)^i`` at grid vertices."""

    return sum(
        _backward_difference(densitized_magnetic[i], i, params)
        for i in range(3)
    )


@jax.jit
def densitized_maxwell_rhs(
    densitized_displacement: jnp.ndarray,
    densitized_magnetic: jnp.ndarray,
    bssn: BSSNVariables,
    params: BSSNParameters,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return source-free ``(dot(mathcal D), dot(mathcal B))``."""

    displacement_geometry, magnetic_geometry = _metric_fields_on_yee_sites(
        bssn, params
    )
    electric_covector = _compute_covariant_E(
        densitized_displacement,
        densitized_magnetic,
        displacement_geometry,
        params,
    )
    magnetic_covector = _compute_covariant_H(
        densitized_displacement,
        densitized_magnetic,
        magnetic_geometry,
        params,
    )
    displacement_rhs = curl_H_to_densitized_D(magnetic_covector, params)
    magnetic_rhs = -curl_E_to_densitized_B(electric_covector, params)
    return _apply_densitized_sommerfeld_boundaries_with_geometry(
        densitized_displacement,
        densitized_magnetic,
        displacement_rhs,
        magnetic_rhs,
        displacement_geometry,
        magnetic_geometry,
        params,
    )


@jax.jit
def collocate_densitized_fields(
    densitized_displacement: jnp.ndarray,
    densitized_magnetic: jnp.ndarray,
    params: BSSNParameters,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Move both densitized vectors to BSSN cell centers."""

    return (
        vector_at_location(
            densitized_displacement,
            DISPLACEMENT_FIELD_LOCATIONS,
            CENTER_LOCATION,
            params,
        ),
        vector_at_location(
            densitized_magnetic,
            MAGNETIC_FIELD_LOCATIONS,
            CENTER_LOCATION,
            params,
        ),
    )


__all__ = [
    "LEVI_CIVITA_SYMBOL",
    "collocate_densitized_fields",
    "compute_covariant_E",
    "compute_covariant_H",
    "curl_E_to_densitized_B",
    "curl_H_to_densitized_D",
    "densitized_displacement_divergence",
    "densitized_magnetic_divergence",
    "densitized_maxwell_rhs",
]
