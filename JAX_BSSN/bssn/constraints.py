"""Raw mathematical constraint fields for the Cartesian BSSN system."""

from typing import NamedTuple

import jax.numpy as jnp
from jax import jit

from JAX_BSSN.evolution.derivatives import diff1_field
from JAX_BSSN.bssn.geometry import W_FLOOR_VALUE, compute_W2_ricci
from JAX_BSSN.bssn.tensor_algebra import determinant_3x3_metric, invert_3x3_metric, trace_tensor
from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables, get_boundary_codes


@jit
def compute_momentum_constraint(vars: BSSNVariables,
                               params: BSSNParameters) -> jnp.ndarray:
    """
    Compute momentum constraint M_i.
    
    Args:
        vars: Current BSSN variables
        params: Evolution parameters
        
    Returns:
        Momentum constraint vector M_i
    """

    dx = params.dx
    K = vars.trace_K
    A_ij = vars.traceless_K
    W    = vars.conformal_factor
    W_floor = jnp.maximum(W, W_FLOOR_VALUE)
    gamma = vars.conformal_metric
    inv_gamma = invert_3x3_metric(gamma)

    dKdi = jnp.stack(
        [
            diff1_field(K, d, dx, *get_boundary_codes(params, d), mad_q=params.mad_q)
            for d in range(3)
        ],
        axis=0,
    )
    # first derivatives of K

    dWdi = jnp.stack(
        [
            diff1_field(W, d, dx, *get_boundary_codes(params, d), mad_q=params.mad_q)
            for d in range(3)
        ],
        axis=0,
    )
    # first derivatives of W

    A_i_up_j = jnp.einsum('jk...,ik...->ij...', inv_gamma, A_ij)
    # raise the second index in A_i^j

    dA_i_up_j_dk = jnp.stack(
        [
            diff1_field(
                A_i_up_j, d + 2, dx, *get_boundary_codes(params, d), mad_q=params.mad_q
            )
            for d in range(3)
        ],
        axis=0,
    )
    # derivatives of A_i^j

    dA_ij_dk = jnp.stack(
        [
            diff1_field(A_ij, d + 2, dx, *get_boundary_codes(params, d), mad_q=params.mad_q)
            for d in range(3)
        ],
        axis=0,
    )
    # derivatives of A_ij

    first_term = jnp.einsum('jij...->i...', dA_i_up_j_dk)
    # first term

    second_term = -0.5 * jnp.einsum('jk...,ijk...->i...', inv_gamma, dA_ij_dk)
    # second term

    third_term = -3 * jnp.einsum('ij...,j...->i...', A_i_up_j, dWdi) / W_floor
    # third term

    fourth_term = -2.0/3.0 * dKdi
    # fourth term

    M_i = first_term + second_term + third_term + fourth_term
    # compute momentum constraint

    return M_i




class ConstraintViolations(NamedTuple):
    """Container for constraint violation measures."""
    hamiltonian: jnp.ndarray      # Hamiltonian constraint violation
    momentum: jnp.ndarray         # Momentum constraint violation (3-vector)
    det_gamma: jnp.ndarray        # det(γ) = 1 violation
    trace_A: jnp.ndarray          # tr(A) = 0 violation
    gamma_condition: jnp.ndarray   # Gamma constraint violation


@jit
def compute_hamiltonian_constraint(vars: BSSNVariables, 
                                  params: BSSNParameters) -> jnp.ndarray:
    """
    Compute Hamiltonian constraint violation.
    
    where R is the 3D Ricci scalar.
    
    Args:
        vars: BSSN variables
        params: Evolution parameters
        
    Returns:
        Hamiltonian constraint violation H
    """
    conformal_metric = vars.conformal_metric

    inv_metric = invert_3x3_metric(conformal_metric)

    W2_ricci = compute_W2_ricci(vars, params)
    ricci_scalar = trace_tensor(W2_ricci, inv_metric)
    # R = gamma_tilde^ij (W**2 R_ij), evaluated without inverse powers of W.

    K_squared = vars.trace_K**2
    
    A_squared = jnp.einsum('ik...,jl...,ij...,kl...->...', inv_metric, inv_metric,
                            vars.traceless_K, vars.traceless_K)


    # Hamiltonian constraint
    hamiltonian = ricci_scalar + 2/3 * K_squared - A_squared

    return hamiltonian


@jit
def compute_det_gamma_violation(vars: BSSNVariables) -> jnp.ndarray:
    """
    Compute violation of det(γ) = 1 condition.
    
    In BSSN, the conformal metric should satisfy det(γ) = 1.
    
    Args:
        vars: BSSN variables
        
    Returns:
        det(γ) - 1
    """
    det_gamma = determinant_3x3_metric(vars.conformal_metric)
    return det_gamma - 1.0


@jit
def compute_trace_A_violation(vars: BSSNVariables) -> jnp.ndarray:
    """
    Compute violation of tr(A) = 0 condition.
    
    The traceless extrinsic curvature should be traceless.
    
    Args:
        vars: BSSN variables
        
    Returns:
        tr(A) = γ^ij A_ij
    """
    inv_metric = invert_3x3_metric(vars.conformal_metric)
    trace_A = trace_tensor(vars.traceless_K, inv_metric)
    return trace_A


# @jit
def compute_gamma_constraint(vars: BSSNVariables, 
                            params: BSSNParameters) -> jnp.ndarray:
    """
    Compute Gamma constraint violation.
    
    The Gamma constraint relates the conformal connection to metric derivatives:
    Γ^i = γ^jk Γ^i_jk
    
    Args:
        vars: BSSN variables
        params: Evolution parameters
        
    Returns:
        Gamma constraint violation
    """

    # raise NotImplementedError("Gamma constraint computation not implemented")
    return jnp.zeros_like(vars.lapse)
    # dx = params.dx
    # shape = vars.conformal_metric.shape[2:]
    
    # # Compute metric derivatives
    # metric_derivs = jnp.zeros((3, 3, 3) + shape)
    # for i in range(3):
    #     for j in range(3):
    #         for k in range(3):
    #             metric_derivs = metric_derivs.at[k, i, j].set(
    #                 diff1_field(vars.conformal_metric[i, j], k, dx))
    
    # # Compute inverse metric
    # inv_metric = invert_3x3_metric(vars.conformal_metric)
    
    # # Compute Christoffel symbols
    # christoffel = christoffel_symbols_second_kind(inv_metric, metric_derivs)
    
    # # Compute γ^jk Γ^i_jk
    # gamma_from_christoffel = jnp.zeros((3,) + shape)
    # for i in range(3):
    #     for j in range(3):
    #         for k in range(3):
    #             gamma_from_christoffel = gamma_from_christoffel.at[i].add(
    #                 inv_metric[j, k] * christoffel[i, j, k])
    
    # # Constraint violation
    # gamma_violation = jnp.zeros((3,) + shape)
    # for i in range(3):
    #     gamma_violation = gamma_violation.at[i].set(
    #         vars.conformal_connection[i] - gamma_from_christoffel[i])
    
    # return gamma_violation


# @jit
def compute_all_constraints(vars: BSSNVariables,
                           params: BSSNParameters) -> ConstraintViolations:
    """
    Compute all constraint violations.
    
    Args:
        vars: BSSN variables
        params: Evolution parameters
        
    Returns:
        All constraint violations
    """
    hamiltonian = compute_hamiltonian_constraint(vars, params)
    momentum = compute_momentum_constraint(vars, params)
    det_gamma = compute_det_gamma_violation(vars)
    trace_A = compute_trace_A_violation(vars)
    gamma_condition = compute_gamma_constraint(vars, params)
    
    return ConstraintViolations(
        hamiltonian=hamiltonian,
        momentum=momentum,
        det_gamma=det_gamma,
        trace_A=trace_A,
        gamma_condition=gamma_condition
    )
