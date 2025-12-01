"""
BSSN (Baumgarte-Shapiro-Shibata-Nakamura) evolution equations.

This module implements the BSSN formulation for 3+1 numerical relativity.
The BSSN variables are:
- conformal metric γ_ij 
- conformal factor W (or φ)
- traceless extrinsic curvature A_ij
- trace of extrinsic curvature K
- conformal connection functions Γ^i

All evolution equations are JIT-compiled with JAX for performance.
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple, NamedTuple
import numpy as np

from JAX_BSSN.derivatives import (diff1_field, compute_all_derivatives, 
                        laplacian_3d, divergence_3d)
from JAX_BSSN.tensor_algebra import (invert_3x3_metric, determinant_3x3_metric,
                           christoffel_symbols_second_kind,
                           christoffel_symbols_first_kind,
                           ricci_tensor, ricci_scalar, trace_tensor,
                           traceless_part, lie_derivative_conformal_metric,
                           raise_index, lower_index)


class BSSNVariables(NamedTuple):
    """Container for BSSN evolution variables."""
    conformal_metric: jnp.ndarray      # γ_ij (3x3 symmetric)
    conformal_factor: jnp.ndarray      # W or φ
    traceless_K: jnp.ndarray          # A_ij (3x3 traceless)
    trace_K: jnp.ndarray              # K (scalar)
    conformal_connection: jnp.ndarray  # Γ^i (3-vector)
    lapse: jnp.ndarray                # α (scalar)
    shift: jnp.ndarray                # β^i (3-vector)


class BSSNParameters(NamedTuple):
    """Parameters for BSSN evolution."""
    eta: float = 2.0          # Damping parameter for Γ^i evolution
    f: float = 2.0            # Multiple of 1+log slicing
    g: float = 0.75           # Gamma driver shift parameter
    dx: float = 0.1           # Grid spacing
    dt: float = 0.001         # Time step


@jit
def pack_symmetric_3x3(tensor: jnp.ndarray) -> jnp.ndarray:
    """
    Pack symmetric 3x3 tensor into 6-component array.
    Order: [00, 01, 02, 11, 12, 22]
    
    Args:
        tensor: Array with shape (3, 3, ni, nj, nk)
        
    Returns:
        Packed array with shape (6, ni, nj, nk)
    """
    shape = tensor.shape[2:]
    packed = jnp.zeros((6,) + shape)
    
    # Pack symmetric components
    packed = packed.at[0].set(tensor[0, 0])  # γ_xx
    packed = packed.at[1].set(tensor[0, 1])  # γ_xy  
    packed = packed.at[2].set(tensor[0, 2])  # γ_xz
    packed = packed.at[3].set(tensor[1, 1])  # γ_yy
    packed = packed.at[4].set(tensor[1, 2])  # γ_yz
    packed = packed.at[5].set(tensor[2, 2])  # γ_zz
    
    return packed


@jit
def unpack_symmetric_3x3(packed: jnp.ndarray) -> jnp.ndarray:
    """
    Unpack 6-component array into symmetric 3x3 tensor.
    
    Args:
        packed: Array with shape (6, ni, nj, nk)
        
    Returns:
        Tensor with shape (3, 3, ni, nj, nk)
    """
    shape = packed.shape[1:]
    tensor = jnp.zeros((3, 3) + shape)
    
    # Unpack symmetric components
    tensor = tensor.at[0, 0].set(packed[0])  # γ_xx
    tensor = tensor.at[0, 1].set(packed[1])  # γ_xy
    tensor = tensor.at[1, 0].set(packed[1])  # γ_xy  
    tensor = tensor.at[0, 2].set(packed[2])  # γ_xz
    tensor = tensor.at[2, 0].set(packed[2])  # γ_xz
    tensor = tensor.at[1, 1].set(packed[3])  # γ_yy
    tensor = tensor.at[1, 2].set(packed[4])  # γ_yz
    tensor = tensor.at[2, 1].set(packed[4])  # γ_yz
    tensor = tensor.at[2, 2].set(packed[5])  # γ_zz
    
    return tensor


@jit
def compute_physical_metric(conformal_metric: jnp.ndarray, 
                           conformal_factor: jnp.ndarray) -> jnp.ndarray:
    """
    Compute physical metric from conformal metric and conformal factor.
    
    g_ij = W^4 * γ_ij  (W-formulation)
    
    Args:
        conformal_metric: Conformal metric γ_ij with shape (3, 3, ni, nj, nk)
        conformal_factor: Conformal factor W with shape (ni, nj, nk)
        
    Returns:
        Physical metric with shape (3, 3, ni, nj, nk)
    """
    W4 = conformal_factor**4
    # Scale conformal metric by W^4

    physical_metric = W4 * conformal_metric
    # compute physical metric
    
    return physical_metric


@jit
def compute_conformal_ricci(vars: BSSNVariables,
                           params: BSSNParameters) -> jnp.ndarray:
    """
    Compute conformal Ricci tensor R̃_ij.
    
    This is the most computationally intensive part of BSSN evolution.
    
    Args:
        conformal_metric: γ_ij with shape (3, 3, ni, nj, nk)
        conformal_connection: Γ̃^i with shape (3, ni, nj, nk)
        params: BSSN parameters
        
    Returns:
        Conformal Ricci tensor with shape (3, 3, ni, nj, nk)
    """
    dx = params.dx
    conformal_metric = vars.conformal_metric
    conformal_connection = vars.conformal_connection
    shape = conformal_metric.shape[2:]
    
    metric_derivs = jnp.stack( [diff1_field(conformal_metric, d+2, dx) for d in range(3)], axis=0) 
    # shape (3, 3, 3, ni, nj, nk)

    # Compute inverse conformal metric
    inv_metric = invert_3x3_metric(conformal_metric)
    
    # Compute Christoffel symbols
    christoffel_first  = christoffel_symbols_first_kind(metric_derivs)
    christoffel_second = christoffel_symbols_second_kind(inv_metric, metric_derivs)

    mixed_derivatives = jnp.zeros((3, 3, 3, 3) + shape)
    for m in range(3):
        for n in range(3):
            mixed_derivatives = mixed_derivatives.at[m, n, ...].set(
                diff1_field( metric_derivs[m, ...], n+2, dx))
            
    term_1 = -0.5 * jnp.einsum('mn...,mnij...->ij...', inv_metric, mixed_derivatives)
    # compute first term of Ricci tensor

    connection_derivs = jnp.stack( [diff1_field(conformal_connection, d+1, dx) for d in range(3)], axis=0)
    # shape (3, 3, ni, nj, nk)

    term_2 = (jnp.einsum('mi...,jm...->ij...', inv_metric, connection_derivs) + jnp.einsum('mj...,im...->ij...', inv_metric, connection_derivs)) / 2.0
    # compute second term of Ricci tensor

    term_3 = ( jnp.einsum('m...,ijm...->ij...', conformal_connection, christoffel_first) + jnp.einsum('m...,jim...->ij...', conformal_connection, christoffel_first) ) / 2.0
    # compute third term of Ricci tensor

    term_4 = jnp.einsum('mn...,kmi...,jkn...->ij...', inv_metric, christoffel_second, christoffel_first) + \
        jnp.einsum('mn...,kmj...,kin...->ji...', inv_metric, christoffel_second, christoffel_first)
    # compute fourth term of Ricci tensor

    term_5 = jnp.einsum('mn...,kim...,kjn...->ij...', inv_metric, christoffel_second, christoffel_first)
    # compute fifth term of Ricci tensor

    conformal_ricci = term_1 + term_2 + term_3 + term_4 + term_5
    
    return conformal_ricci


@jit 
def compute_ricci_with_matter(vars: BSSNVariables,
                             params: BSSNParameters) -> jnp.ndarray:
    """
    Compute full Ricci tensor including conformal factor contributions.
    
    R_ij = R̃_ij + R_ij^φ
    
    where R_ij^φ contains terms from the conformal factor.
    
    Args:
        conformal_ricci: Conformal Ricci tensor R̃_ij
        conformal_metric: γ_ij  
        conformal_factor: W
        params: BSSN parameters
        
    Returns:
        Full Ricci tensor
    """

    dx = params.dx
    alpha = vars.lapse
    K = vars.trace_K
    A_ij = vars.traceless_K
    W    = vars.conformal_factor
    gamma = vars.conformal_metric
    inv_gamma = invert_3x3_metric(gamma)


    dWdi = jnp.stack( [diff1_field(W, d, dx) for d in range(3)], axis=0)
    dWdij = jnp.zeros((3,3) + W.shape)
    for i in range(3):
        for j in range(3):
            dWdij = dWdij.at[i,j].set( diff1_field( dWdi[i], j, dx) )
    # second derivatives of W


    first_term = dWdij / W
    # first term

    second_term = jnp.einsum('ij...,mn...,nm...->ij...', gamma, inv_gamma, dWdij) / W
    # second term

    third_term = -2 * jnp.einsum('ij...,mn...,m...,n...->ij...', gamma, inv_gamma, dWdi, dWdi) / W**2
    # third term

    R_ij_W = first_term + second_term + third_term
    # conformal factor contribution to Ricci tensor


    R_ij = compute_conformal_ricci(vars, params) + R_ij_W
    # full Ricci tensor

    return R_ij


@jit
def evolve_conformal_metric(vars: BSSNVariables, 
                           params: BSSNParameters) -> jnp.ndarray:
    """
    Evolve conformal metric γ_ij.
    
    Args:
        vars: Current BSSN variables
        params: Evolution parameters
        
    Returns:
        Time derivative of conformal metric
    """
    # Note: this is only valid in the harmonic gauge where shift is 0
    
    # First term: -2α A_ij
    dt_gamma = -2.0 * vars.lapse * vars.traceless_K

    return dt_gamma


@jit
def evolve_conformal_factor(vars: BSSNVariables,
                           params: BSSNParameters) -> jnp.ndarray:
    """
    Evolve conformal factor W.
    
    Args:
        vars: Current BSSN variables
        params: Evolution parameters
        
    Returns:
        Time derivative of conformal factor
    """
    
    # Note: this is only valid in the harmonic gauge where shift is 0
    
    # First term: (1/3) α W K
    dt_W = (1.0/3.0) * vars.lapse * vars.conformal_factor * vars.trace_K

    
    return dt_W


@jit
def evolve_trace_extrinsic_curvature(vars: BSSNVariables,
                                    params: BSSNParameters) -> jnp.ndarray:
    """
    Evolve trace of extrinsic curvature K.
    
    Args:
        vars: Current BSSN variables
        params: Evolution parameters
        
    Returns:
        Time derivative of trace K
    """

    # Note: this is only valid in the harmonic gauge where shift is 0

    dx = params.dx
    alpha = vars.lapse
    K = vars.trace_K
    A_ij = vars.traceless_K
    W    = vars.conformal_factor
    gamma = vars.conformal_metric
    inv_gamma = invert_3x3_metric(gamma)


    dalphadi = jnp.stack( [diff1_field(alpha, d, dx) for d in range(3)], axis=0)
    dalphadij = jnp.zeros((3,3) + alpha.shape)
    for i in range(3):
        for j in range(3):
            dalphadij = dalphadij.at[i,j].set( diff1_field( dalphadi[i], j, dx) )
    # second derivatives of alpha

    dWdi = jnp.stack( [diff1_field(W, d, dx) for d in range(3)], axis=0)
    # first derivatives of W

    DiDj_alpha = dalphadij
    DiDj_alpha = DiDj_alpha + 1/W * jnp.einsum('i...,j...->ij...', dWdi, dalphadi)
    DiDj_alpha = DiDj_alpha + 1/W * jnp.einsum('j...,i...->ij...', dWdi, dalphadi)
    DiDj_alpha = DiDj_alpha - 1/W * jnp.einsum('ij...,mn...,m...,n...->ij...', gamma, inv_gamma, dWdi, dalphadi)
    # full covariant second derivative of alpha

    first_term = - W**2 * jnp.einsum('ij...,ij...->...', inv_gamma, DiDj_alpha)
    # first term

    second_term = alpha * jnp.einsum('ij...,kl...,ik...,jl...->...', inv_gamma, inv_gamma, A_ij, A_ij)
    # second term

    third_term = alpha * K**2 / 3.0
    # third term
    
    dt_K = first_term + second_term + third_term
    # compute dt_K
    
    return dt_K


@jit
def evolve_traceless_extrinsic_curvature(vars: BSSNVariables,
                                        params: BSSNParameters) -> jnp.ndarray:
    """
    Evolve traceless extrinsic curvature A_ij.
    
    Args:
        vars: Current BSSN variables  
        params: Evolution parameters
        
    Returns:
        Time derivative of traceless extrinsic curvature
    """

    # Note this is only valid in holonomic gauge (shift=0)

    dx = params.dx
    alpha = vars.lapse
    K = vars.trace_K
    A_ij = vars.traceless_K
    W    = vars.conformal_factor
    gamma = vars.conformal_metric
    inv_gamma = invert_3x3_metric(gamma)


    first_term = alpha * K * A_ij
    # first term

    second_term = -2 * alpha * jnp.einsum('ik...,kl...,lj...->ij...', A_ij, inv_gamma, A_ij)
    # second term

    dalphadi = jnp.stack( [diff1_field(alpha, d, dx) for d in range(3)], axis=0)
    dalphadij = jnp.zeros((3,3) + alpha.shape)
    for i in range(3):
        for j in range(3):
            dalphadij = dalphadij.at[i,j].set( diff1_field( dalphadi[i], j, dx) )
    # second derivatives of alpha

    dWdi = jnp.stack( [diff1_field(W, d, dx) for d in range(3)], axis=0)
    # first derivatives of W

    DiDj_alpha = dalphadij
    DiDj_alpha = DiDj_alpha + 1/W * jnp.einsum('i...,j...->ij...', dWdi, dalphadi)
    DiDj_alpha = DiDj_alpha + 1/W * jnp.einsum('j...,i...->ij...', dWdi, dalphadi)
    DiDj_alpha = DiDj_alpha - 1/W * jnp.einsum('ij...,mn...,m...,n...->ij...', gamma, inv_gamma, dWdi, dalphadi)
    # full covariant second derivative of alpha
        
    # Compute full Ricci tensor
    ricci = compute_ricci_with_matter(vars, params)

    
    third_term = alpha * ricci - DiDj_alpha
    third_term = traceless_part(third_term, vars.conformal_metric, inv_gamma)
    third_term = W**2 * third_term
    # third term
    
    dt_A = first_term + second_term + third_term
    # compute dt_A

    return dt_A


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

    dx = params.dx
    alpha = vars.lapse
    K = vars.trace_K
    A_ij = vars.traceless_K
    W    = vars.conformal_factor
    gamma = vars.conformal_metric
    inv_gamma = invert_3x3_metric(gamma)

    dWdi = jnp.stack( [diff1_field(W, d, dx) for d in range(3)], axis=0)
    # first derivatives of W

    dalphadi = jnp.stack( [diff1_field(alpha, d, dx) for d in range(3)], axis=0)
    # first derivatives of alpha

    dKdi = jnp.stack( [diff1_field(K, d, dx) for d in range(3)], axis=0)
    # first derivatives of K

    first_term = -4/3 * alpha * jnp.einsum('ij...,j...->i...', inv_gamma, dKdi)
    # first term


    metric_derivs = jnp.stack( [diff1_field(gamma, d+2, dx) for d in range(3)], axis=0) 
    # shape (3, 3, 3, ni, nj, nk)
    christoffel_second = christoffel_symbols_second_kind(inv_gamma, metric_derivs)
    # compute Christoffel symbols of the second kind

    A_ij_raised = jnp.einsum('ik...,kl...,lj...->ij...', A_ij, inv_gamma, inv_gamma)
    # raise indices of A_ij
    second_term = 2 * alpha * jnp.einsum('ijk...,jk...->i...', christoffel_second, A_ij_raised)
    # second term

    third_term = -6 * alpha / W * jnp.einsum('ij...,j...->i...', A_ij_raised, dWdi)
    # third term

    fourth_term = -2 * jnp.einsum('ij...,j...->i...', A_ij_raised, dalphadi)
    # fourth term

    dt_Gamma = first_term + second_term + third_term + fourth_term
    # compute dt_Gamma

    return dt_Gamma


@jit
def evolve_lapse(vars: BSSNVariables, params: BSSNParameters) -> jnp.ndarray:
    """
    Evolve lapse function α using 1+log slicing.
    
    ∂_t α = -f α^2 K + £_β α
    
    Args:
        vars: Current BSSN variables
        params: Evolution parameters
        
    Returns:
        Time derivative of lapse
    """
    dx = params.dx
    f = params.f
    
    # 1+log slicing evolution
    dt_alpha = -f * vars.lapse**2 * vars.trace_K
    
    # Add shift advection
    for k in range(3):
        dalpha_dk = diff1_field(vars.lapse, k, dx)
        dt_alpha += vars.shift[k] * dalpha_dk

    
    return dt_alpha


@jit
def evolve_shift(vars: BSSNVariables, params: BSSNParameters) -> jnp.ndarray:
    """
    Evolve shift vector β^i using Gamma driver.
    
    ∂_t β^i = (3/4) Γ̃^i
    
    Args:
        vars: Current BSSN variables
        params: Evolution parameters
        
    Returns:
        Time derivative of shift
    """
    g = params.g
    
    # Gamma driver evolution
    dt_beta = g * vars.conformal_connection
    
    return dt_beta