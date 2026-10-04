"""Conformal-connection evolution equation."""

import jax.numpy as jnp

from JAX_BSSN.evolution.spatial_derivatives import (
    diff1_physical, diff2_physical, ko_dissipation,
)
from jax import jit

from JAX_BSSN.bssn.geometry import W_FLOOR_VALUE
from JAX_BSSN.bssn.shift_and_lapse import (
    compute_shift_advection,
    compute_shift_derivatives,
)
from JAX_BSSN.bssn.tensor_algebra import christoffel_symbols_second_kind, invert_3x3_metric
from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables


@jit
def evolve_conformal_connection(vars: BSSNVariables,
                               params: BSSNParameters) -> jnp.ndarray:
    """
    Evolve conformal connection functions Γ̃^i.


    Args:
        vars: Current BSSN variables
        params: Evolution parameters

    Returns:
        Time derivative of conformal connection
    """

    alpha = vars.lapse
    K = vars.trace_K
    A_ij = vars.traceless_K
    W    = vars.conformal_factor
    W_floor = jnp.maximum(W, W_FLOOR_VALUE)
    gamma = vars.conformal_metric
    inv_gamma = invert_3x3_metric(gamma)

    dWdi = jnp.stack(
        [
            diff1_physical(W, d, params)
            for d in range(3)
        ],
        axis=0,
    )
    # first derivatives of W

    dalphadi = jnp.stack(
        [
            diff1_physical(alpha, d, params)
            for d in range(3)
        ],
        axis=0,
    )
    # first derivatives of alpha

    dKdi = jnp.stack(
        [
            diff1_physical(K, d, params)
            for d in range(3)
        ],
        axis=0,
    )
    # first derivatives of K

    shift = vars.shift
    # unpack the shift vector

    d_shift = compute_shift_derivatives(shift, params)
    # d_shift[i, j] = partial_j beta^i

    div_shift = jnp.einsum("ii...->...", d_shift)
    # divergence of the shift vector

    first_term = -4/3 * alpha * jnp.einsum('ij...,j...->i...', inv_gamma, dKdi)
    # first term


    metric_derivs = jnp.stack(
        [
            diff1_physical(gamma, d + 2, params)
            for d in range(3)
        ],
        axis=0,
    )
    # shape (3, 3, 3, ni, nj, nk)
    christoffel_second = christoffel_symbols_second_kind(inv_gamma, metric_derivs)
    # compute Christoffel symbols of the second kind

    A_ij_raised = jnp.einsum('ik...,jl...,kl...->ij...', inv_gamma, inv_gamma, A_ij)
    # raise indices of A_ij
    second_term = 2 * alpha * jnp.einsum('ijk...,jk...->i...', christoffel_second, A_ij_raised)
    # second term

    third_term = -6 * alpha / W_floor * jnp.einsum('ij...,j...->i...', A_ij_raised, dWdi)
    # third term

    fourth_term = -2 * jnp.einsum('ij...,j...->i...', A_ij_raised, dalphadi)
    # fourth term

    fifth_term = compute_shift_advection(
        vars.conformal_connection, shift, params
    )
    # upwinded advection of the conformal connection by the shift

    sixth_term = (2.0 / 3.0) * vars.conformal_connection * div_shift
    # conformal-weight correction from div(beta)

    seventh_term = -jnp.einsum('m...,im...->i...', vars.conformal_connection, d_shift)
    # -Gamma^m partial_m beta^i

    d2_shift = jnp.zeros((3, 3, 3) + shift.shape[1:])
    for i in range(3):
        for m in range(3):
            for n in range(3):
                if m == n:
                    second_derivative = diff2_physical(shift[i], m, params)
                else:
                    second_derivative = diff1_physical(d_shift[i, n], m, params)
                d2_shift = d2_shift.at[i, m, n].set(second_derivative)
    # d2_shift[i, m, n] = partial_m partial_n beta^i

    eighth_term = jnp.einsum('mn...,imn...->i...', inv_gamma, d2_shift)
    # gamma^mn partial_m partial_n beta^i

    div_shift_deriv = jnp.zeros((3,) + shift.shape[1:], dtype=shift.dtype)
    for m in range(3):
        for n in range(3):
            if m == n:
                derivative = diff2_physical(shift[n], m, params)
            else:
                derivative = diff1_physical(d_shift[n, n], m, params)
            div_shift_deriv = div_shift_deriv.at[m].add(derivative)
    # partial_m partial_n beta^n = partial_m div(beta)

    ninth_term = (1.0 / 3.0) * jnp.einsum(
        'im...,m...->i...', inv_gamma, div_shift_deriv
    )
    # 1/3 gamma^im partial_m partial_n beta^n

    dt_Gamma = (
        first_term
        + second_term
        + third_term
        + fourth_term
        + fifth_term
        + sixth_term
        + seventh_term
        + eighth_term
        + ninth_term
    )
    # compute dt_Gamma

    # Gamma is shape (3, ni, nj, nk)

    dissipation_term = ko_dissipation(vars.conformal_connection, params)
    # compute dissipation term

    return dt_Gamma + dissipation_term


@jit
def evolve_conformal_connection_with_matter(
    vars: BSSNVariables,
    params: BSSNParameters,
    momentum_density: jnp.ndarray,
) -> jnp.ndarray:
    """Evolve ``Gamma_tilde^i`` with covariant momentum density ``S_i``."""

    vacuum_rhs = evolve_conformal_connection(vars, params)
    inverse_conformal_metric = invert_3x3_metric(vars.conformal_metric)
    matter_source = -16.0 * jnp.pi * vars.lapse * jnp.einsum(
        'ij...,j...->i...', inverse_conformal_metric, momentum_density
    )
    return vacuum_rhs + matter_source
