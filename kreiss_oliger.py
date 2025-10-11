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
def adaptive_dissipation_coefficient(field: jnp.ndarray, base_sigma: float, 
                                   dx: float) -> jnp.ndarray:
    """
    Compute adaptive dissipation coefficient based on local field variations.
    
    The idea is to apply more dissipation where gradients are large.
    
    Args:
        field: Scalar field with shape (ni, nj, nk)
        base_sigma: Base dissipation coefficient
        dx: Grid spacing
        
    Returns:
        Position-dependent dissipation coefficient
    """
    ni, nj, nk = field.shape
    sigma = jnp.full_like(field, base_sigma)
    
    # Compute local gradient magnitude
    grad_magnitude = jnp.zeros_like(field)
    
    for i in range(1, ni - 1):
        for j in range(1, nj - 1):
            for k in range(1, nk - 1):
                # Compute gradient using central differences
                dx_field = (field[i+1, j, k] - field[i-1, j, k]) / (2 * dx)
                dy_field = (field[i, j+1, k] - field[i, j-1, k]) / (2 * dx)
                dz_field = (field[i, j, k+1] - field[i, j, k-1]) / (2 * dx)
                
                grad_mag = jnp.sqrt(dx_field**2 + dy_field**2 + dz_field**2)
                grad_magnitude = grad_magnitude.at[i, j, k].set(grad_mag)
    
    # Scale dissipation coefficient with gradient magnitude
    max_grad = jnp.max(grad_magnitude)
    if max_grad > 1e-12:
        sigma = base_sigma * (1 + grad_magnitude / max_grad)
    
    return sigma


@jit
def apply_ko_dissipation_bssn(vars: BSSNVariables, sigma: float, 
                             dx: float, adaptive: bool = False) -> BSSNVariables:
    """
    Apply Kreiss-Oliger dissipation to all BSSN variables.
    
    Args:
        vars: Current BSSN variables
        sigma: Base dissipation coefficient
        dx: Grid spacing
        adaptive: Whether to use adaptive dissipation
        
    Returns:
        Dissipation terms for all BSSN variables
    """
    if adaptive:
        # Use trace of conformal metric to determine dissipation strength
        metric_trace = vars.conformal_metric[0, 0] + vars.conformal_metric[1, 1] + vars.conformal_metric[2, 2]
        sigma_field = adaptive_dissipation_coefficient(metric_trace, sigma, dx)
        
        # For simplicity, use average value
        sigma_avg = jnp.mean(sigma_field)
    else:
        sigma_avg = sigma
    
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
def ramp_function(x: float, x_start: float, x_end: float) -> float:
    """
    Smooth ramp function that goes from 0 to 1.
    
    Args:
        x: Input value
        x_start: Start of ramp
        x_end: End of ramp
        
    Returns:
        Ramp value between 0 and 1
    """
    if x <= x_start:
        return 0.0
    elif x >= x_end:
        return 1.0
    else:
        # Smooth cubic polynomial
        t = (x - x_start) / (x_end - x_start)
        return t * t * (3.0 - 2.0 * t)


@jit
def boundary_dissipation(field: jnp.ndarray, sigma: float, dx: float,
                        boundary_width: int = 10) -> jnp.ndarray:
    """
    Apply enhanced dissipation near boundaries to prevent reflections.
    
    Args:
        field: Scalar field with shape (ni, nj, nk)
        sigma: Base dissipation coefficient
        dx: Grid spacing
        boundary_width: Width of boundary region in grid points
        
    Returns:
        Enhanced dissipation near boundaries
    """
    ni, nj, nk = field.shape
    dissipation = jnp.zeros_like(field)
    
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                # Distance from boundaries
                dist_x = min(i, ni - 1 - i)
                dist_y = min(j, nj - 1 - j)
                dist_z = min(k, nk - 1 - k)
                
                min_dist = min(dist_x, dist_y, dist_z)
                
                if min_dist < boundary_width:
                    # Enhanced dissipation near boundaries
                    ramp = ramp_function(min_dist, 0, boundary_width)
                    enhanced_sigma = sigma * (1 + 10 * (1 - ramp))
                    
                    # Apply dissipation in direction perpendicular to nearest boundary
                    if dist_x == min_dist and i >= 3 and i < ni - 3:
                        d6 = diff6th_dissipation(field, i, j, k, 0, dx)
                        dissipation = dissipation.at[i, j, k].add(-enhanced_sigma * dx**5 * d6)
                    
                    if dist_y == min_dist and j >= 3 and j < nj - 3:
                        d6 = diff6th_dissipation(field, i, j, k, 1, dx)
                        dissipation = dissipation.at[i, j, k].add(-enhanced_sigma * dx**5 * d6)
                    
                    if dist_z == min_dist and k >= 3 and k < nk - 3:
                        d6 = diff6th_dissipation(field, i, j, k, 2, dx)
                        dissipation = dissipation.at[i, j, k].add(-enhanced_sigma * dx**5 * d6)
    
    return dissipation


@jit
def apply_boundary_dissipation_bssn(vars: BSSNVariables, sigma: float, 
                                   dx: float, boundary_width: int = 10) -> BSSNVariables:
    """
    Apply boundary dissipation to all BSSN variables.
    
    Args:
        vars: Current BSSN variables
        sigma: Dissipation coefficient
        dx: Grid spacing
        boundary_width: Width of boundary region
        
    Returns:
        Boundary dissipation terms
    """
    # Apply boundary dissipation to key variables
    dissipation_gamma = jnp.zeros_like(vars.conformal_metric)
    for i in range(3):
        for j in range(3):
            dissipation_gamma = dissipation_gamma.at[i, j].set(
                boundary_dissipation(vars.conformal_metric[i, j], sigma, dx, boundary_width))
    
    dissipation_W = boundary_dissipation(vars.conformal_factor, sigma, dx, boundary_width)
    
    dissipation_A = jnp.zeros_like(vars.traceless_K)
    for i in range(3):
        for j in range(3):
            dissipation_A = dissipation_A.at[i, j].set(
                boundary_dissipation(vars.traceless_K[i, j], sigma, dx, boundary_width))
    
    dissipation_K = boundary_dissipation(vars.trace_K, sigma, dx, boundary_width)
    
    dissipation_Gamma = jnp.zeros_like(vars.conformal_connection)
    for i in range(3):
        dissipation_Gamma = dissipation_Gamma.at[i].set(
            boundary_dissipation(vars.conformal_connection[i], sigma, dx, boundary_width))
    
    dissipation_alpha = boundary_dissipation(vars.lapse, sigma, dx, boundary_width)
    
    dissipation_beta = jnp.zeros_like(vars.shift)
    for i in range(3):
        dissipation_beta = dissipation_beta.at[i].set(
            boundary_dissipation(vars.shift[i], sigma, dx, boundary_width))
    
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
