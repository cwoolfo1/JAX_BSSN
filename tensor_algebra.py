"""
Tensor algebra operations for numerical relativity.

This module provides tensor operations commonly used in numerical relativity,
including Christoffel symbols, Riemann tensor components, and other geometric
quantities. All operations are JIT-compiled with JAX.
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple
import numpy as np
from derivatives import diff1_field, compute_all_derivatives


@jit
def invert_3x3_metric(metric: jnp.ndarray) -> jnp.ndarray:
    """
    Invert a 3x3 metric tensor field.
    
    Args:
        metric: Array with shape (3, 3, ni, nj, nk) containing metric components
        
    Returns:
        Inverse metric with same shape
    """
    ni, nj, nk = metric.shape[2:]

    flatten_g = metric.reshape(3, 3, -1)
    # flatten the last three dimensions for vectorized inversion

    vectorized_inv = jax.vmap(jnp.linalg.inv, in_axes=2, out_axes=2)
    inv_flat = vectorized_inv(flatten_g)
    # vmap over the last dimension to invert each 3x3 matrix

    inv_metric = inv_flat.reshape(3, 3, ni, nj, nk)
    # Reshape back to original grid shape
    
    return inv_metric


@jit
def determinant_3x3_metric(metric: jnp.ndarray) -> jnp.ndarray:
    """
    Compute determinant of 3x3 metric tensor field.
    
    Args:
        metric: Array with shape (3, 3, ni, nj, nk) containing metric components
        
    Returns:
        3D array containing determinant at each grid point
    """
    ni, nj, nk = metric.shape[2:]

    g_flatten = metric.reshape(3, 3, -1)
    # flatten the last three dimensions for vectorized determinant
    vectorized_det = jax.vmap(jnp.linalg.det, in_axes=2, out_axes=0)
    det_flat = vectorized_det(g_flatten)
    # vmap over the last dimension to compute determinant of each 3x3 matrix

    det = det_flat.reshape(ni, nj, nk)
    # Reshape back to original grid shape
    
    return det


@jit
def raise_index(tensor: jnp.ndarray, inverse_metric: jnp.ndarray, 
                index_position: int) -> jnp.ndarray:
    """
    Raise an index of a rank-2 tensor using the inverse metric.
    
    Args:
        tensor: Rank-2 tensor with shape (3, 3, ni, nj, nk)
        inverse_metric: Inverse metric with shape (3, 3, ni, nj, nk)
        index_position: Which index to raise (0 for first, 1 for second)
        
    Returns:
        Tensor with raised index
    """
    ni, nj, nk = tensor.shape[2:]
    raised = jnp.zeros_like(tensor)

    raised = jax.lax.cond(index_position == 0,
                            lambda _: jnp.einsum('ik...,kj...->ij...', inverse_metric, tensor),
                            lambda _: jnp.einsum('ik...,jk...->ij...', tensor, inverse_metric),
                            operand=None)
                            # T^i_j = g^ik T_kj or T_i^j = T_i^k g^jk

    # if index_position == 0:
    #     # Raise first index: T^i_j = g^ik T_kj
    #     for i in range(3):
    #         for j in range(3):
    #             for k in range(3):
    #                 raised = raised.at[i, j].add(
    #                     inverse_metric[i, k] * tensor[k, j])
    # else:
    #     # Raise second index: T_i^j = g^jk T_ik  
    #     for i in range(3):
    #         for j in range(3):
    #             for k in range(3):
    #                 raised = raised.at[i, j].add(
    #                     tensor[i, k] * inverse_metric[k, j])
    
    return raised


@jit
def lower_index(tensor: jnp.ndarray, metric: jnp.ndarray, 
                index_position: int) -> jnp.ndarray:
    """
    Lower an index of a rank-2 tensor using the metric.
    
    Args:
        tensor: Rank-2 tensor with shape (3, 3, ni, nj, nk)
        metric: Metric with shape (3, 3, ni, nj, nk)
        index_position: Which index to lower (0 for first, 1 for second)
        
    Returns:
        Tensor with lowered index
    """
    ni, nj, nk = tensor.shape[2:]
    lowered = jnp.zeros_like(tensor)
    
    lowered = jax.lax.cond(index_position == 0,
                            lambda _: jnp.einsum('ik...,kj...->ij...', metric, tensor),
                            lambda _: jnp.einsum('ik...,jk...->ij...', tensor, metric),
                            operand=None)
                            # T_ij = g_ik T^k_j or T_i_j = T_i^k g_kj
    
    # if index_position == 0:
    #     # Lower first index: T_ij = g_ik T^k_j
    #     for i in range(3):
    #         for j in range(3):
    #             for k in range(3):
    #                 lowered = lowered.at[i, j].add(
    #                     metric[i, k] * tensor[k, j])
    # else:
    #     # Lower second index: T_i_j = T_i^k g_kj
    #     for i in range(3):
    #         for j in range(3):
    #             for k in range(3):
    #                 lowered = lowered.at[i, j].add(
    #                     tensor[i, k] * metric[k, j])
    
    return lowered


@jit
def christoffel_symbols_first_kind(metric_derivatives: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Christoffel symbols of the first kind.
    
    Γ_ijk = (1/2) * (∂g_ij/∂x^k + ∂g_ik/∂x^j - ∂g_jk/∂x^i)
    
    Args:
        metric_derivatives: Array with shape (3, 3, 3, ni, nj, nk)
                           where first index is derivative direction
        
    Returns:
        Christoffel symbols of first kind with shape (3, 3, 3, ni, nj, nk)
    """
    shape = metric_derivatives.shape[3:]
    christoffel_1 = jnp.zeros((3, 3, 3) + shape)
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                term1 = metric_derivatives[k, i, j]  # ∂g_ij/∂x^k
                term2 = metric_derivatives[j, i, k]  # ∂g_ik/∂x^j  
                term3 = metric_derivatives[i, j, k]  # ∂g_jk/∂x^i
                
                christoffel_1 = christoffel_1.at[i, j, k].set(
                    0.5 * (term1 + term2 - term3))
    
    return christoffel_1


@jit
def christoffel_symbols_second_kind(inverse_metric: jnp.ndarray,
                                   metric_derivatives: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Christoffel symbols of the second kind.
    
    Γ^l_ijk = g^lm * Γ_ijk,m
    
    Args:
        inverse_metric: Inverse metric with shape (3, 3, ni, nj, nk)
        metric_derivatives: Metric derivatives with shape (3, 3, 3, ni, nj, nk)
        
    Returns:
        Christoffel symbols of second kind with shape (3, 3, 3, ni, nj, nk)
    """
    # First compute first kind
    christoffel_1 = christoffel_symbols_first_kind(metric_derivatives)
    
    shape = inverse_metric.shape[2:]
    christoffel_2 = jnp.zeros((3, 3, 3) + shape)
    
    for l in range(3):
        for i in range(3):
            for j in range(3):
                for m in range(3):
                    christoffel_2 = christoffel_2.at[l, i, j].add(
                        inverse_metric[l, m] * christoffel_1[m, i, j])
    
    return christoffel_2


@jit
def riemann_tensor(christoffel: jnp.ndarray, christoffel_derivatives: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the Riemann curvature tensor.
    
    R^l_ijk = ∂Γ^l_ik/∂x^j - ∂Γ^l_ij/∂x^k + Γ^l_mj Γ^m_ik - Γ^l_mk Γ^m_ij
    
    Args:
        christoffel: Christoffel symbols with shape (3, 3, 3, ni, nj, nk)
        christoffel_derivatives: Derivatives of Christoffel symbols
        
    Returns:
        Riemann tensor with shape (3, 3, 3, 3, ni, nj, nk)
    """
    shape = christoffel.shape[3:]
    riemann = jnp.zeros((3, 3, 3, 3) + shape)
    
    for l in range(3):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # Derivative terms
                    term1 = christoffel_derivatives[j, l, i, k]  # ∂Γ^l_ik/∂x^j
                    term2 = christoffel_derivatives[k, l, i, j]  # ∂Γ^l_ij/∂x^k
                    
                    # Product terms
                    term3 = 0.0
                    term4 = 0.0
                    for m in range(3):
                        term3 += christoffel[l, m, j] * christoffel[m, i, k]
                        term4 += christoffel[l, m, k] * christoffel[m, i, j]
                    
                    riemann = riemann.at[l, i, j, k].set(term1 - term2 + term3 - term4)
    
    return riemann


@jit
def ricci_tensor(riemann: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the Ricci tensor from the Riemann tensor.
    
    R_ij = R^k_ikj
    
    Args:
        riemann: Riemann tensor with shape (3, 3, 3, 3, ni, nj, nk)
        
    Returns:
        Ricci tensor with shape (3, 3, ni, nj, nk)
    """
    shape = riemann.shape[4:]
    ricci = jnp.zeros((3, 3) + shape)
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ricci = ricci.at[i, j].add(riemann[k, i, k, j])
    
    return ricci


@jit
def ricci_scalar(ricci: jnp.ndarray, inverse_metric: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the Ricci scalar from the Ricci tensor.
    
    R = g^ij R_ij
    
    Args:
        ricci: Ricci tensor with shape (3, 3, ni, nj, nk)
        inverse_metric: Inverse metric with shape (3, 3, ni, nj, nk)
        
    Returns:
        Ricci scalar with shape (ni, nj, nk)
    """
    shape = ricci.shape[2:]
    scalar = jnp.zeros(shape)
    
    for i in range(3):
        for j in range(3):
            scalar += inverse_metric[i, j] * ricci[i, j]
    
    return scalar


@jit
def lie_derivative_metric(vector: jnp.ndarray, metric: jnp.ndarray, dx: float) -> jnp.ndarray:
    """
    Compute Lie derivative of metric tensor with respect to vector field.
    
    £_v g_ij = v^k ∂g_ij/∂x^k + g_kj ∂v^k/∂x^i + g_ik ∂v^k/∂x^j
    
    Args:
        vector: Vector field with shape (3, ni, nj, nk)
        metric: Metric tensor with shape (3, 3, ni, nj, nk)
        dx: Grid spacing
        
    Returns:
        Lie derivative with shape (3, 3, ni, nj, nk)
    """
    shape = metric.shape[2:]
    lie_deriv = jnp.zeros_like(metric)
    
    # Compute derivatives of metric
    metric_derivs = jnp.zeros((3, 3, 3) + shape)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                metric_derivs = metric_derivs.at[k, i, j].set(
                    diff1_field(metric[i, j], k, dx))
    
    # Compute derivatives of vector
    vector_derivs = jnp.zeros((3, 3) + shape)
    for i in range(3):
        for j in range(3):
            vector_derivs = vector_derivs.at[i, j].set(
                diff1_field(vector[i], j, dx))
    
    for i in range(3):
        for j in range(3):
            # First term: v^k ∂g_ij/∂x^k
            term1 = 0.0
            for k in range(3):
                term1 += vector[k] * metric_derivs[k, i, j]
            
            # Second term: g_kj ∂v^k/∂x^i  
            term2 = 0.0
            for k in range(3):
                term2 += metric[k, j] * vector_derivs[k, i]
            
            # Third term: g_ik ∂v^k/∂x^j
            term3 = 0.0
            for k in range(3):
                term3 += metric[i, k] * vector_derivs[k, j]
            
            lie_deriv = lie_deriv.at[i, j].set(term1 + term2 + term3)
    
    return lie_deriv


@jit
def lie_derivative_conformal_metric(vector: jnp.ndarray, metric: jnp.ndarray, 
                                   dx: float) -> jnp.ndarray:
    """
    Compute conformal Lie derivative with weight -2/3.
    
    £_v γ_ij = v^k ∂γ_ij/∂x^k + γ_kj ∂v^k/∂x^i + γ_ik ∂v^k/∂x^j - (2/3) γ_ij ∂v^k/∂x^k
    
    Args:
        vector: Vector field with shape (3, ni, nj, nk)
        metric: Conformal metric with shape (3, 3, ni, nj, nk)
        dx: Grid spacing
        
    Returns:
        Conformal Lie derivative with shape (3, 3, ni, nj, nk)
    """
    # Start with regular Lie derivative
    lie_deriv = lie_derivative_metric(vector, metric, dx)
    
    # Compute divergence of vector
    div_v = jnp.zeros(metric.shape[2:])
    for k in range(3):
        div_v += diff1_field(vector[k], k, dx)
    
    # Subtract conformal weight term
    for i in range(3):
        for j in range(3):
            lie_deriv = lie_deriv.at[i, j].set(
                lie_deriv[i, j] - (2.0/3.0) * metric[i, j] * div_v)
    
    return lie_deriv


@jit
def trace_tensor(tensor: jnp.ndarray, inverse_metric: jnp.ndarray) -> jnp.ndarray:
    """
    Compute trace of a rank-2 tensor.
    
    T = g^ij T_ij
    
    Args:
        tensor: Rank-2 tensor with shape (3, 3, ni, nj, nk)
        inverse_metric: Inverse metric with shape (3, 3, ni, nj, nk)
        
    Returns:
        Trace with shape (ni, nj, nk)
    """
    shape = tensor.shape[2:]
    trace = jnp.zeros(shape)
    
    for i in range(3):
        for j in range(3):
            trace += inverse_metric[i, j] * tensor[i, j]
    
    return trace


@jit
def traceless_part(tensor: jnp.ndarray, metric: jnp.ndarray, 
                   inverse_metric: jnp.ndarray) -> jnp.ndarray:
    """
    Extract traceless part of a rank-2 tensor.
    
    A_ij = T_ij - (1/3) γ_ij T
    
    Args:
        tensor: Input tensor with shape (3, 3, ni, nj, nk)
        metric: Metric tensor with shape (3, 3, ni, nj, nk)
        inverse_metric: Inverse metric with shape (3, 3, ni, nj, nk)
        
    Returns:
        Traceless tensor with shape (3, 3, ni, nj, nk)
    """
    trace = trace_tensor(tensor, inverse_metric)
    traceless = jnp.zeros_like(tensor)
    
    for i in range(3):
        for j in range(3):
            traceless = traceless.at[i, j].set(
                tensor[i, j] - (1.0/3.0) * metric[i, j] * trace)
    
    return traceless
