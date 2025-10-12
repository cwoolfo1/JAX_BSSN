"""
Kreiss-Oliger dissipation for numerical stability.

This module implements Kreiss-Oliger artificial dissipation to suppress
high-frequency numerical instabilities in the BSSN evolution.
All functions are JIT-compiled with JAX.
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple
import numpy as np

from derivatives import get_stencil_indices, diff6_field
from bssn import BSSNVariables


@jit
def apply_ko_dissipation_scalar(field: jnp.ndarray, sigma: float, dx: float) -> jnp.ndarray:
    """
    Apply Kreiss-Oliger dissipation to a scalar field.
    
    The dissipation term is: -σ Δx^5 ∂^6f/∂x^6
    
    Args:
        field: Scalar field with shape (ni, nj, nk)
        sigma: Dissipation coefficient (typically 0.01-0.1)
        dx: Grid spacing
        
    Returns:
        Dissipation term to be added to evolution equation
    """
    ni, nj, nk = field.shape

    d6f_dx1 = diff6_field(field, 0, dx)
    d6f_dx2 = diff6_field(field, 1, dx)
    d6f_dx3 = diff6_field(field, 2, dx)
    # compute the 6th derivative in each direction
    dissipation = -sigma * dx**5 * (d6f_dx1 + d6f_dx2 + d6f_dx3)
    # Combine contributions from all three spatial directions
    
    return dissipation


@jit
def apply_ko_dissipation_tensor(tensor: jnp.ndarray, sigma: float, dx: float) -> jnp.ndarray:
    """
    Apply Kreiss-Oliger dissipation to a tensor field.
    
    Args:
        tensor: Tensor field with shape (n1, n2, ..., ni, nj, nk)
        sigma: Dissipation coefficient
        dx: Grid spacing
        
    Returns:
        Dissipation term for the tensor
    """
    # Get tensor dimensions and grid shape
    tensor_dims = tensor.shape[:-3]
    grid_shape = tensor.shape[-3:]
    
    # Flatten tensor indices for easier processing
    if len(tensor_dims) == 0:
        # Scalar field
        return apply_ko_dissipation_scalar(tensor, sigma, dx)
    
    # Initialize dissipation tensor
    dissipation = jnp.zeros_like(tensor)
    
    # Apply dissipation to each tensor component
    if len(tensor_dims) == 1:
        # Vector field
        for i in range(tensor_dims[0]):
            component_dissipation = apply_ko_dissipation_scalar(tensor[i], sigma, dx)
            dissipation = dissipation.at[i].set(component_dissipation)
    
    elif len(tensor_dims) == 2:
        # Rank-2 tensor field
        for i in range(tensor_dims[0]):
            for j in range(tensor_dims[1]):
                component_dissipation = apply_ko_dissipation_scalar(tensor[i, j], sigma, dx)
                dissipation = dissipation.at[i, j].set(component_dissipation)
    
    elif len(tensor_dims) == 3:
        # Rank-3 tensor field
        for i in range(tensor_dims[0]):
            for j in range(tensor_dims[1]):
                for k in range(tensor_dims[2]):
                    component_dissipation = apply_ko_dissipation_scalar(tensor[i, j, k], sigma, dx)
                    dissipation = dissipation.at[i, j, k].set(component_dissipation)
    
    return dissipation

@jit
def apply_ko_dissipation_bssn(vars: BSSNVariables, sigma_avg: float, 
                             dx: float) -> BSSNVariables:
    """
    Apply Kreiss-Oliger dissipation to all BSSN variables.
    
    Args:
        vars: Current BSSN variables
        sigma: Base dissipation coefficient
        dx: Grid spacing        
    Returns:
        Dissipation terms for all BSSN variables
    """
    
    # Apply dissipation to each BSSN variable
    dissipation_gamma = apply_ko_dissipation_tensor(vars.conformal_metric, sigma_avg, dx)
    dissipation_W = apply_ko_dissipation_scalar(vars.conformal_factor, sigma_avg, dx)
    dissipation_A = apply_ko_dissipation_tensor(vars.traceless_K, sigma_avg, dx)
    dissipation_K = apply_ko_dissipation_scalar(vars.trace_K, sigma_avg, dx)
    dissipation_Gamma = apply_ko_dissipation_tensor(vars.conformal_connection, sigma_avg, dx)
    dissipation_alpha = apply_ko_dissipation_scalar(vars.lapse, sigma_avg, dx)
    dissipation_beta = apply_ko_dissipation_tensor(vars.shift, sigma_avg, dx)
    
    return BSSNVariables(
        conformal_metric=dissipation_gamma,
        conformal_factor=dissipation_W,
        traceless_K=dissipation_A,
        trace_K=dissipation_K,
        conformal_connection=dissipation_Gamma,
        lapse=dissipation_alpha,
        shift=dissipation_beta
    )

@jit
def compute_dissipation_timestep_limit(sigma: float, dx: float) -> float:
    """
    Compute timestep limit imposed by Kreiss-Oliger dissipation.
    
    The dissipation term has a stability constraint similar to diffusion.
    
    Args:
        sigma: Dissipation coefficient
        dx: Grid spacing
        
    Returns:
        Maximum stable timestep
    """
    # Stability limit for 6th-order dissipation
    # This is a conservative estimate
    dt_limit = 0.1 * dx**6 / (sigma * dx**5)
    return dt_limit


def get_optimal_dissipation_coefficient(dx: float, dt: float, 
                                      cfl_factor: float = 0.1) -> float:
    """
    Get optimal dissipation coefficient based on grid and timestep.
    
    Args:
        dx: Grid spacing
        dt: Timestep
        cfl_factor: CFL-like factor for dissipation
        
    Returns:
        Optimal dissipation coefficient
    """
    # Balance between numerical stability and physical accuracy
    sigma = cfl_factor * dt / dx**5
    return min(sigma, 0.1)  # Cap at reasonable value
