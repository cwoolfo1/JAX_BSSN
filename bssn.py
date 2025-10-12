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

from derivatives import (diff1_field, compute_all_derivatives, 
                        laplacian_3d, divergence_3d)
from tensor_algebra import (invert_3x3_metric, determinant_3x3_metric,
                           christoffel_symbols_second_kind,
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
def compute_conformal_ricci(conformal_metric: jnp.ndarray, 
                           conformal_connection: jnp.ndarray,
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
    shape = conformal_metric.shape[2:]
    
    metric_derivs = jnp.stack( [diff1_field(conformal_metric, d, dx) for d in range(3)], axis=0) 
    # shape (3, 3, 3, ni, nj, nk)

    print("  Completed metric derivatives computation")

    # Compute inverse conformal metric
    inv_metric = invert_3x3_metric(conformal_metric)

    print("  Completed inverse metric computation")
    
    # Compute Christoffel symbols
    christoffel = christoffel_symbols_second_kind(inv_metric, metric_derivs)
    
    print("  Completed Christoffel symbols computation")
    # Compute conformal connection derivatives
    connection_derivs = jnp.zeros((3, 3) + shape)
    for i in range(3):
        for j in range(3):
            connection_derivs = connection_derivs.at[i, j].set(
                diff1_field(conformal_connection[i], j, dx))
    
    # Conformal Ricci tensor calculation
    # This is a simplified version - full implementation would include
    # second derivatives and more complex terms
    conformal_ricci = jnp.zeros((3, 3) + shape)
    
    for i in range(3):
        for j in range(3):
            # Leading order terms
            term1 = 0.0
            for k in range(3):
                term1 += connection_derivs[k, j] * inv_metric[k, i]
                term1 += connection_derivs[k, i] * inv_metric[k, j]
            
            term2 = 0.0
            for k in range(3):
                for l in range(3):
                    term2 += inv_metric[k, l] * christoffel[k, i, j] * conformal_connection[l]
            
            conformal_ricci = conformal_ricci.at[i, j].set(-0.5 * term1 + term2)
    
    return conformal_ricci


@jit 
def compute_ricci_with_matter(conformal_ricci: jnp.ndarray,
                             conformal_metric: jnp.ndarray,
                             conformal_factor: jnp.ndarray,
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
    shape = conformal_metric.shape[2:]
    
    # Compute conformal factor derivatives
    W_derivs = compute_all_derivatives(conformal_factor, dx)
    
    # Compute conformal factor Laplacian
    W_laplacian = laplacian_3d(conformal_factor, dx)
    
    # Compute inverse conformal metric
    inv_metric = invert_3x3_metric(conformal_metric)
    
    # Conformal factor contribution to Ricci tensor
    ricci_phi = jnp.zeros_like(conformal_ricci)
    
    for i in range(3):
        for j in range(3):
            # ∇_i ∇_j φ term (simplified)
            term1 = diff1_field(W_derivs[i], j, dx)
            
            # Connection term corrections
            term2 = 0.0
            for k in range(3):
                # Simplified Christoffel correction
                term2 += W_derivs[k] * diff1_field(conformal_metric[i, j], k, dx)
            
            ricci_phi = ricci_phi.at[i, j].set(term1 - 0.5 * term2)
    
    return conformal_ricci + ricci_phi


@jit
def evolve_conformal_metric(vars: BSSNVariables, 
                           params: BSSNParameters) -> jnp.ndarray:
    """
    Evolve conformal metric γ_ij.
    
    ∂_t γ_ij = -2α A_ij + £_β γ_ij
    
    Args:
        vars: Current BSSN variables
        params: Evolution parameters
        
    Returns:
        Time derivative of conformal metric
    """
    shape = vars.conformal_metric.shape[2:]
    dt_gamma = jnp.zeros_like(vars.conformal_metric)
    
    # First term: -2α A_ij
    dt_gamma = dt_gamma = -2.0 * vars.lapse * vars.traceless_K
    
    # Second term: Lie derivative with respect to shift
    lie_deriv = lie_derivative_conformal_metric(vars.shift, vars.conformal_metric, params.dx)
    
    return dt_gamma + lie_deriv


@jit
def evolve_conformal_factor(vars: BSSNVariables,
                           params: BSSNParameters) -> jnp.ndarray:
    """
    Evolve conformal factor W.
    
    ∂_t W = -(1/3) α W K + £_β W
    
    Args:
        vars: Current BSSN variables
        params: Evolution parameters
        
    Returns:
        Time derivative of conformal factor
    """
    dx = params.dx
    
    # First term: -(1/3) α W K
    dt_W = -(1.0/3.0) * vars.lapse * vars.conformal_factor * vars.trace_K
    
    # Second term: shift advection
    for k in range(3):
        dW_dk = diff1_field(vars.conformal_factor, k, dx)
        dt_W += vars.shift[k] * dW_dk
    
    return dt_W


@jit
def evolve_traceless_extrinsic_curvature(vars: BSSNVariables,
                                        params: BSSNParameters) -> jnp.ndarray:
    """
    Evolve traceless extrinsic curvature A_ij.
    
    ∂_t A_ij = -∇_i ∇_j α + α(R_ij - 8πS_ij) + K A_ij + £_β A_ij
    
    Args:
        vars: Current BSSN variables  
        params: Evolution parameters
        
    Returns:
        Time derivative of traceless extrinsic curvature
    """
    dx = params.dx
    shape = vars.traceless_K.shape[2:]
    dt_A = jnp.zeros_like(vars.traceless_K)
    print("  Computing conformal Ricci tensor")
    
    # Compute conformal Ricci tensor
    conformal_ricci = compute_conformal_ricci(
        vars.conformal_metric, vars.conformal_connection, params)
    print("  Completed conformal Ricci tensor computation")
    # Compute full Ricci tensor  
    ricci = compute_ricci_with_matter(
        conformal_ricci, vars.conformal_metric, vars.conformal_factor, params)
    print("  Completed full Ricci tensor computation")
    
    # Compute inverse conformal metric
    inv_metric = invert_3x3_metric(vars.conformal_metric)
    
    # Make Ricci tensor traceless
    ricci_traceless = traceless_part(ricci, vars.conformal_metric, inv_metric)


    # α R_ij^TF term
    alpha_ricci = vars.lapse * ricci_traceless
    # K A_ij term
    K_A = vars.trace_K * vars.traceless_K
    
    # Evolution equation terms
    for i in range(3):
        for j in range(3):
            # Second derivative of lapse (simplified)
            d2_alpha = diff1_field(diff1_field(vars.lapse, i, dx), j, dx)

    dt_A = alpha_ricci + K_A - d2_alpha
    
    # Add Lie derivative with respect to shift
    lie_deriv = lie_derivative_conformal_metric(vars.shift, vars.traceless_K, dx)
    
    return dt_A + lie_deriv


@jit
def evolve_trace_extrinsic_curvature(vars: BSSNVariables,
                                    params: BSSNParameters) -> jnp.ndarray:
    """
    Evolve trace of extrinsic curvature K.
    
    ∂_t K = -∇^2 α + α(A_ij A^ij + K^2/3) + £_β K
    
    Args:
        vars: Current BSSN variables
        params: Evolution parameters
        
    Returns:
        Time derivative of trace K
    """
    dx = params.dx
    
    # Laplacian of lapse
    lapl_alpha = laplacian_3d(vars.lapse, dx)
    
    # A_ij A^ij term
    inv_metric = invert_3x3_metric(vars.conformal_metric)
    A_squared = 0.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    A_squared += (inv_metric[i, k] * inv_metric[j, l] * 
                                vars.traceless_K[i, j] * vars.traceless_K[k, l])


    # Evolution equation
    dt_K = (-lapl_alpha + vars.lapse * (A_squared + vars.trace_K**2 / 3.0))
    
    # Add shift advection
    for k in range(3):
        dK_dk = diff1_field(vars.trace_K, k, dx)
        dt_K += vars.shift[k] * dK_dk
    
    return dt_K


@jit
def evolve_conformal_connection(vars: BSSNVariables,
                               params: BSSNParameters) -> jnp.ndarray:
    """
    Evolve conformal connection functions Γ̃^i.
    
    ∂_t Γ̃^i = -2 A^ij ∇_j α + 2α(Γ̃^i_jk A^jk - 2/3 γ^ij ∇_j K) + £_β Γ̃^i
    
    Args:
        vars: Current BSSN variables
        params: Evolution parameters
        
    Returns:
        Time derivative of conformal connection
    """
    dx = params.dx
    eta = params.eta
    shape = vars.conformal_connection.shape[1:]
    dt_Gamma = jnp.zeros_like(vars.conformal_connection)
    
    # Compute inverse metric
    inv_metric = invert_3x3_metric(vars.conformal_metric)
    
    # Raise indices of A_ij
    A_raised = raise_index(vars.traceless_K, inv_metric, 1)
    
    for i in range(3):
        # First term: -2 A^ij ∇_j α
        term1 = 0.0
        for j in range(3):
            dalpha_dj = diff1_field(vars.lapse, j, dx)
            term1 += A_raised[i, j] * dalpha_dj
        term1 *= -2.0
        
        # Second term: 2α(-2/3 γ^ij ∇_j K) (simplified)
        term2 = 0.0
        for j in range(3):
            dK_dj = diff1_field(vars.trace_K, j, dx)
            term2 += inv_metric[i, j] * dK_dj
        term2 *= vars.lapse * (-4.0/3.0)
        
        # Damping term (Gamma driver)
        damping = -eta * vars.conformal_connection[i]
        
        dt_Gamma = dt_Gamma.at[i].set(term1 + term2 + damping)
    
    # Add shift advection
    for i in range(3):
        for k in range(3):
            dGamma_dk = diff1_field(vars.conformal_connection[i], k, dx)
            dt_Gamma = dt_Gamma.at[i].add(vars.shift[k] * dGamma_dk)
    
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


@jit
def bssn_evolution_step(vars: BSSNVariables, 
                       params: BSSNParameters) -> BSSNVariables:
    """
    Compute one evolution step of BSSN equations.
    
    Args:
        vars: Current BSSN variables
        params: Evolution parameters
        
    Returns:
        Updated BSSN variables
    """
    dt = params.dt
    
    print("  Computing BSSN evolution step")
    # Compute time derivatives
    dt_gamma = evolve_conformal_metric(vars, params)
    print("  Completed conformal metric evolution")
    dt_W = evolve_conformal_factor(vars, params)
    print("  Completed conformal factor evolution")
    dt_A = evolve_traceless_extrinsic_curvature(vars, params)
    print("  Completed traceless extrinsic curvature evolution")
    dt_K = evolve_trace_extrinsic_curvature(vars, params)
    dt_Gamma = evolve_conformal_connection(vars, params)
    dt_alpha = evolve_lapse(vars, params)
    dt_beta = evolve_shift(vars, params)

    print("  Completed BSSN evolution step")
    
    # Update variables using forward Euler (can be upgraded to RK4)
    new_vars = BSSNVariables(
        conformal_metric=vars.conformal_metric + dt * dt_gamma,
        conformal_factor=vars.conformal_factor + dt * dt_W,
        traceless_K=vars.traceless_K + dt * dt_A,
        trace_K=vars.trace_K + dt * dt_K,
        conformal_connection=vars.conformal_connection + dt * dt_Gamma,
        lapse=vars.lapse + dt * dt_alpha,
        shift=vars.shift + dt * dt_beta
    )

    print("  Completed BSSN variable update")

    return new_vars