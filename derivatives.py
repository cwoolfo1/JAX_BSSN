"""
Finite difference derivatives with JAX JIT compilation.

This module provides finite difference operators for computing spatial derivatives
in numerical relativity simulations. All functions are JIT-compiled for performance.
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple, Union
import numpy as np
from functools import partial

@jit
def periodic_indexing(idx: int, size: int) -> int:
    """Handle periodic boundary conditions for array indexing."""
    return jnp.where(idx < 0, idx + size, jnp.where(idx >= size, idx - size, idx))

@jit
def get_stencil_indices(i: int, j: int, k: int, direction: int, offset: int, 
                       shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Get indices for finite difference stencil with periodic boundaries."""
    ni, nj, nk = shape
    
    if direction == 0:  # x-direction
        new_i = periodic_indexing(i + offset, ni)
        return new_i, j, k
    elif direction == 1:  # y-direction
        new_j = periodic_indexing(j + offset, nj)
        return i, new_j, k
    else:  # z-direction
        new_k = periodic_indexing(k + offset, nk)
        return i, j, new_k

@partial(jit, static_argnames=['direction'])
def diff1_field(field: jnp.ndarray, direction: int, dx: float) -> jnp.ndarray:
    """
    Compute first derivative of entire field in given direction.
    
    Args:
        field: 3D array containing the field values
        direction: Direction of derivative (0=x, 1=y, 2=z)
        dx: Grid spacing
        
    Returns:
        3D array containing the derivative
    """

    # Uses the stencil: (-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[i-2]) / (12*dx)

    field_forward1 = jnp.roll(field, -1, axis=direction)
    field_forward2 = jnp.roll(field, -2, axis=direction)
    field_backward1 = jnp.roll(field, 1, axis=direction)
    field_backward2 = jnp.roll(field, 2, axis=direction)


    return ( (-1/12)*field_forward2 + (2/3)*field_forward1 + (-2/3)*field_backward1 + (1/12)*field_backward2 ) / dx
    # Using 4th-order central difference

@jit
def diff6th_dissipation(field: jnp.ndarray, i: int, j: int, k: int, 
                       direction: int, dx: float) -> float:
    """
    Compute 6th-order derivative for Kreiss-Oliger dissipation.
    
    Uses 7-point stencil for 6th-order accuracy.
    
    Args:
        field: 3D array containing the field values
        i, j, k: Grid point indices  
        direction: Direction of derivative (0=x, 1=y, 2=z)
        dx: Grid spacing
        
    Returns:
        6th derivative at the given point
    """
    shape = field.shape
    
    # Get 7-point stencil
    points = []
    for offset in range(-3, 4):
        idx = get_stencil_indices(i, j, k, direction, offset, shape)
        points.append(field[idx])
    
    # 6th order finite difference coefficients
    # [1, -6, 15, -20, 15, -6, 1]
    result = (points[0] - 6*points[1] + 15*points[2] - 20*points[3] + 
              15*points[4] - 6*points[5] + points[6])
    
    return result / (dx**6)

@jit
def compute_all_derivatives(field: jnp.ndarray, dx: float) -> jnp.ndarray:
    """
    Compute all first derivatives of a field.
    
    Args:
        field: 3D array containing the field values
        dx: Grid spacing
        
    Returns:
        4D array with shape (3, ni, nj, nk) containing derivatives in each direction
    """

    x1_derivative = diff1_field(field, 0, dx)
    y1_derivative = diff1_field(field, 1, dx)
    z1_derivative = diff1_field(field, 2, dx)
    # compute the first derivatives in all three directions

    derivatives = jnp.stack([x1_derivative, y1_derivative, z1_derivative], axis=0)
    # combine them into a single array with shape (3, ni, nj, nk)
    
    return derivatives

@jit
def gradient_3d(field: jnp.ndarray, dx: float) -> jnp.ndarray:
    """
    Compute 3D gradient using 4th-order finite differences.
    
    Args:
        field: 3D scalar field
        dx: Grid spacing (assumed uniform)
        
    Returns:
        4D array with shape (3, ni, nj, nk) containing gradient components
    """
    return compute_all_derivatives(field, dx)

@jit  
def divergence_3d(vector_field: jnp.ndarray, dx: float) -> jnp.ndarray:
    """
    Compute divergence of a 3D vector field.
    
    Args:
        vector_field: 4D array with shape (3, ni, nj, nk)
        dx: Grid spacing
        
    Returns:
        3D array containing the divergence
    """

    df1dx1 = diff1_field(vector_field[0], 0, dx)
    df2dx2 = diff1_field(vector_field[1], 1, dx)
    df3dx3 = diff1_field(vector_field[2], 2, dx)
    # compute the partial derivatives

    div = df1dx1 + df2dx2 + df3dx3
    # sum them to get the divergence

    return div

@jit
def laplacian_3d(field: jnp.ndarray, dx: float) -> jnp.ndarray:
    """
    Compute 3D Laplacian using finite differences.
    
    Args:
        field: 3D scalar field
        dx: Grid spacing
        
    Returns:
        3D array containing the Laplacian
    """

    dfdx1 = diff1_field(field, 0, dx)
    dfdx2 = diff1_field(field, 1, dx)
    dfdx3 = diff1_field(field, 2, dx)
    # compute the first derivatives

    d2fdx1dx1 = diff1_field(dfdx1, 0, dx)
    d2fdx2dx2 = diff1_field(dfdx2, 1, dx)
    d2fdx3dx3 = diff1_field(dfdx3, 2, dx)
    # compute the second derivatives

    lapl = d2fdx1dx1 + d2fdx2dx2 + d2fdx3dx3
    # sum them to get the Laplacian
    
    return lapl