"""Curved source-free second-order Maxwell wave equations."""

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNParameters, BSSNVariables

from JAX_BSSN.EM.boundaries import apply_planar_sommerfeld_boundaries
from JAX_BSSN.EM.derivatives import (
    covariant_derivative_covector,
    covariant_derivative_tensor2,
    covariant_vector_laplacian,
    lie_derivative_covector,
    spatial_derivatives,
)
from JAX_BSSN.EM.geometry import compute_bssn_em_geometry, tracefree
from JAX_BSSN.EM.variables import BSSNEMGeometry, EMVariables


def _raise_covector(
    field: jnp.ndarray, inverse_metric: jnp.ndarray
) -> jnp.ndarray:
    return jnp.einsum("ij...,j...->i...", inverse_metric, field)


def _raise_tensor2(
    tensor: jnp.ndarray, inverse_metric: jnp.ndarray
) -> jnp.ndarray:
    return jnp.einsum(
        "ik...,jl...,kl...->ij...", inverse_metric, inverse_metric, tensor
    )


def _pstf_vector_gradient(
    field: jnp.ndarray,
    geometry: BSSNEMGeometry,
    params: BSSNParameters,
) -> jnp.ndarray:
    derivative = covariant_derivative_covector(
        field, geometry.christoffel, params
    )
    return tracefree(derivative, geometry.metric, geometry.inverse_metric)


def _curl_covector(
    field: jnp.ndarray,
    geometry: BSSNEMGeometry,
    params: BSSNParameters,
) -> jnp.ndarray:
    derivative = covariant_derivative_covector(
        field, geometry.christoffel, params
    )
    epsilon_mixed = jnp.einsum(
        "jm...,kn...,imn...->ijk...",
        geometry.inverse_metric,
        geometry.inverse_metric,
        geometry.levi_civita,
    )
    return jnp.einsum("ijk...,jk...->i...", epsilon_mixed, derivative)


def source_free_projected_field_dots(
    electric_field: jnp.ndarray,
    magnetic_field: jnp.ndarray,
    bssn: BSSNVariables,
    geometry: BSSNEMGeometry,
    params: BSSNParameters,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the source-free first-order 1+3 Maxwell propagation RHS.

    The returned covectors are the Eulerian projected derivatives used by
    ``EMVariables``.  With ``Theta=-K`` and ``sigma_ij=-A_ij``, the
    propagation equations are

    ``dot(E)_i = -A_i^j E_j + 2 K E_i / 3 + curl(H)_i
                  + epsilon_ijk a^j H^k``

    and its electric-magnetic dual.
    """

    inverse_metric = geometry.inverse_metric
    A_mixed = jnp.einsum(
        "ik...,kj...->ij...",
        geometry.traceless_extrinsic_curvature,
        inverse_metric,
    )
    acceleration_up = _raise_covector(geometry.acceleration, inverse_metric)
    electric_field_up = _raise_covector(electric_field, inverse_metric)
    magnetic_field_up = _raise_covector(magnetic_field, inverse_metric)

    acceleration_cross_magnetic = jnp.einsum(
        "ijk...,j...,k...->i...",
        geometry.levi_civita,
        acceleration_up,
        magnetic_field_up,
    )
    acceleration_cross_electric = jnp.einsum(
        "ijk...,j...,k...->i...",
        geometry.levi_civita,
        acceleration_up,
        electric_field_up,
    )

    electric_field_dot = (
        -jnp.einsum("ij...,j...->i...", A_mixed, electric_field)
        + 2.0 * bssn.trace_K * electric_field / 3.0
        + _curl_covector(magnetic_field, geometry, params)
        + acceleration_cross_magnetic
    )
    magnetic_field_dot = (
        -jnp.einsum("ij...,j...->i...", A_mixed, magnetic_field)
        + 2.0 * bssn.trace_K * magnetic_field / 3.0
        - _curl_covector(electric_field, geometry, params)
        - acceleration_cross_electric
    )
    return electric_field_dot, magnetic_field_dot


def curved_source_terms(
    wave: EMVariables,
    bssn: BSSNVariables,
    geometry: BSSNEMGeometry,
    params: BSSNParameters,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the electric and magnetic sources in the rendered equations."""

    E = wave.electric_field
    P = wave.electric_field_dot
    H = wave.magnetic_field
    Q = wave.magnetic_field_dot

    inverse_metric = geometry.inverse_metric
    A = geometry.traceless_extrinsic_curvature
    acceleration = geometry.acceleration
    acceleration_dot = geometry.acceleration_dot
    epsilon = geometry.levi_civita
    K = bssn.trace_K

    E_up = _raise_covector(E, inverse_metric)
    P_up = _raise_covector(P, inverse_metric)
    H_up = _raise_covector(H, inverse_metric)
    Q_up = _raise_covector(Q, inverse_metric)
    acceleration_up = _raise_covector(acceleration, inverse_metric)
    acceleration_dot_up = _raise_covector(
        acceleration_dot, inverse_metric
    )

    A_up_up = _raise_tensor2(A, inverse_metric)
    A_lower_up = jnp.einsum("im...,mk...->ik...", A, inverse_metric)
    A_up_lower = jnp.einsum("jm...,ml...->jl...", inverse_metric, A)

    dK = spatial_derivatives(K, params)
    dK_up = _raise_covector(dK, inverse_metric)
    dA = covariant_derivative_tensor2(A, geometry.christoffel, params)
    dA_up_up_up = jnp.einsum(
        "jp...,km...,ln...,pmn...->jkl...",
        inverse_metric,
        inverse_metric,
        inverse_metric,
        dA,
    )

    pstf_E = _pstf_vector_gradient(E, geometry, params)
    pstf_H = _pstf_vector_gradient(H, geometry, params)
    pstf_E_up_up = _raise_tensor2(pstf_E, inverse_metric)
    pstf_H_up_up = _raise_tensor2(pstf_H, inverse_metric)
    curl_E_up = _raise_covector(
        _curl_covector(E, geometry, params), inverse_metric
    )
    curl_H_up = _raise_covector(
        _curl_covector(H, geometry, params), inverse_metric
    )

    acceleration_squared = jnp.einsum(
        "i...,i...->...", acceleration, acceleration_up
    )
    A_squared = jnp.einsum("ij...,ij...->...", A, A_up_up)

    electric_source = (
        (
            acceleration_squared
            - 4.0 * K**2 / 9.0
            + A_squared
            + 2.0 * geometry.normal_normal_ricci / 3.0
        )
        * E
        + K * jnp.einsum("ij...,j...->i...", A, E_up) / 3.0
        - jnp.einsum(
            "ik...,bk...,b...->i...", A_lower_up, A, E_up
        )
        + 5.0 * K * P / 3.0
        - jnp.einsum("ij...,j...->i...", A, P_up)
        + jnp.einsum("j...,ij...->i...", acceleration_up, pstf_E)
        - jnp.einsum("ij...,j...->i...", geometry.spatial_ricci, E_up)
        - jnp.einsum("ijk...,l...,jkl...->i...", epsilon, H, dA_up_up_up)
        - 2.5
        * jnp.einsum(
            "ijk...,j...,k...->i...", epsilon, acceleration_up, curl_E_up
        )
        + jnp.einsum(
            "ijk...,j...,k...->i...",
            epsilon,
            acceleration_dot_up,
            H_up,
        )
        - 2.0
        * jnp.einsum("ijk...,j...,k...->i...", epsilon, H_up, dK_up)
        / 3.0
        - 3.0
        * jnp.einsum(
            "ijk...,j...,kl...,l...->i...",
            epsilon,
            acceleration_up,
            A_up_up,
            H,
        )
        + 2.0
        * jnp.einsum(
            "ijk...,jl...,kl...->i...", epsilon, A_up_lower, pstf_H_up_up
        )
        - jnp.einsum("ij...,j...->i...", geometry.electric_weyl, E_up)
        + jnp.einsum("ij...,j...->i...", geometry.magnetic_weyl, H_up)
    )

    magnetic_source = (
        (
            acceleration_squared
            - 4.0 * K**2 / 9.0
            + A_squared
            + 2.0 * geometry.normal_normal_ricci / 3.0
        )
        * H
        + K * jnp.einsum("ij...,j...->i...", A, H_up) / 3.0
        - jnp.einsum(
            "ik...,bk...,b...->i...", A_lower_up, A, H_up
        )
        + 5.0 * K * Q / 3.0
        - jnp.einsum("ij...,j...->i...", A, Q_up)
        + jnp.einsum("j...,ij...->i...", acceleration_up, pstf_H)
        - jnp.einsum("ij...,j...->i...", geometry.spatial_ricci, H_up)
        + jnp.einsum("ijk...,l...,jkl...->i...", epsilon, E, dA_up_up_up)
        - 2.5
        * jnp.einsum(
            "ijk...,j...,k...->i...", epsilon, acceleration_up, curl_H_up
        )
        - jnp.einsum(
            "ijk...,j...,k...->i...",
            epsilon,
            acceleration_dot_up,
            E_up,
        )
        + 2.0
        * jnp.einsum("ijk...,j...,k...->i...", epsilon, E_up, dK_up)
        / 3.0
        + 3.0
        * jnp.einsum(
            "ijk...,j...,kl...,l...->i...",
            epsilon,
            acceleration_up,
            A_up_up,
            E,
        )
        - 2.0
        * jnp.einsum(
            "ijk...,jl...,kl...->i...", epsilon, A_up_lower, pstf_E_up_up
        )
        - jnp.einsum("ij...,j...->i...", geometry.electric_weyl, H_up)
        - jnp.einsum("ij...,j...->i...", geometry.magnetic_weyl, E_up)
    )

    return electric_source, magnetic_source


def _coordinate_field_rhs(
    field: jnp.ndarray,
    field_dot: jnp.ndarray,
    bssn: BSSNVariables,
    geometry: BSSNEMGeometry,
    params: BSSNParameters,
) -> jnp.ndarray:
    """Convert a projected first derivative into a coordinate-time RHS."""

    mixed_K = jnp.einsum(
        "ik...,kj...->ij...",
        geometry.extrinsic_curvature,
        geometry.inverse_metric,
    )
    return lie_derivative_covector(field, bssn.shift, params) + bssn.lapse * (
        field_dot - jnp.einsum("ij...,j...->i...", mixed_K, field)
    )


def _coordinate_field_dot_rhs(
    field: jnp.ndarray,
    field_dot: jnp.ndarray,
    source: jnp.ndarray,
    bssn: BSSNVariables,
    geometry: BSSNEMGeometry,
    params: BSSNParameters,
) -> jnp.ndarray:
    """Convert the projected wave equation into a coordinate-time RHS."""

    laplacian = covariant_vector_laplacian(
        field,
        geometry.inverse_metric,
        geometry.christoffel,
        params,
    )
    mixed_K = jnp.einsum(
        "ik...,kj...->ij...",
        geometry.extrinsic_curvature,
        geometry.inverse_metric,
    )
    return lie_derivative_covector(
        field_dot, bssn.shift, params
    ) + bssn.lapse * (
        laplacian
        + source
        - jnp.einsum("ij...,j...->i...", mixed_K, field_dot)
    )


@jax.jit
def compute_em_rhs(
    em: EMVariables,
    bssn: BSSNVariables,
    bssn_rhs: BSSNVariables,
    params: BSSNParameters,
    backreaction_sources=None,
) -> EMVariables:
    """Assemble the curved bulk RHS and apply planar Sommerfeld faces."""

    geometry = compute_bssn_em_geometry(
        bssn,
        bssn_rhs,
        params,
        backreaction_sources=backreaction_sources,
    )
    electric_source, magnetic_source = curved_source_terms(
        em, bssn, geometry, params
    )

    rhs = EMVariables(
        electric_field=_coordinate_field_rhs(
            em.electric_field,
            em.electric_field_dot,
            bssn,
            geometry,
            params,
        ),
        electric_field_dot=_coordinate_field_dot_rhs(
            em.electric_field,
            em.electric_field_dot,
            electric_source,
            bssn,
            geometry,
            params,
        ),
        magnetic_field=_coordinate_field_rhs(
            em.magnetic_field,
            em.magnetic_field_dot,
            bssn,
            geometry,
            params,
        ),
        magnetic_field_dot=_coordinate_field_dot_rhs(
            em.magnetic_field,
            em.magnetic_field_dot,
            magnetic_source,
            bssn,
            geometry,
            params,
        ),
    )

    return apply_planar_sommerfeld_boundaries(
        em, rhs, bssn, geometry, params
    )


__all__ = [
    "compute_em_rhs",
    "curved_source_terms",
    "source_free_projected_field_dots",
]
