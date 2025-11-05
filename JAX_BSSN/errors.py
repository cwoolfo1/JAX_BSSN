"""
Error analysis and constraint monitoring for BSSN evolution.

This module provides functions to compute and monitor various constraint
violations and convergence measures. All functions are JIT-compiled with JAX.
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple, NamedTuple
import numpy as np

from JAX_BSSN.bssn import BSSNVariables, BSSNParameters
from JAX_BSSN.derivatives import diff1_field, divergence_3d, compute_all_derivatives
from JAX_BSSN.tensor_algebra import (invert_3x3_metric, determinant_3x3_metric,
                           christoffel_symbols_second_kind, ricci_tensor,
                           ricci_scalar, trace_tensor)


class ConstraintViolations(NamedTuple):
    """Container for constraint violation measures."""
    hamiltonian: jnp.ndarray      # Hamiltonian constraint violation
    momentum: jnp.ndarray         # Momentum constraint violation (3-vector)
    det_gamma: jnp.ndarray        # det(γ) = 1 violation
    trace_A: jnp.ndarray          # tr(A) = 0 violation
    gamma_condition: jnp.ndarray   # Gamma constraint violation


# @jit
def compute_hamiltonian_constraint(vars: BSSNVariables, 
                                  params: BSSNParameters) -> jnp.ndarray:
    """
    Compute Hamiltonian constraint violation.
    
    The Hamiltonian constraint is:
    H = R + K² - K_ij K^ij = 0
    
    where R is the 3D Ricci scalar.
    
    Args:
        vars: BSSN variables
        params: Evolution parameters
        
    Returns:
        Hamiltonian constraint violation H
    """
    dx = params.dx
    shape = vars.conformal_metric.shape[2:]
    
    # Compute physical metric
    psi4 = vars.conformal_factor**4
    physical_metric = psi4 * vars.conformal_metric
    
    # Compute physical metric derivatives
    metric_derivs = jnp.stack([diff1_field(physical_metric, k, dx) for k in range(3)], axis=0)
    
    # Compute physical Ricci scalar (simplified calculation)
    inv_physical_metric = invert_3x3_metric(physical_metric)
    christoffel = christoffel_symbols_second_kind(inv_physical_metric, metric_derivs)
    
    # This is a simplified Ricci calculation
    # Full implementation would include all derivative terms
    ricci_3d = jnp.zeros(shape)
    
    # Compute extrinsic curvature terms
    # K² = (tr K)²
    K_squared = vars.trace_K**2
    
    # K_ij K^ij = A_ij A^ij + (1/3) K²
    inv_conformal_metric = invert_3x3_metric(vars.conformal_metric)
    A_squared = 0.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    A_squared += (inv_conformal_metric[i, k] * inv_conformal_metric[j, l] * 
                                vars.traceless_K[i, j] * vars.traceless_K[k, l])
    
    K_ij_K_ij = A_squared + K_squared / 3.0
    
    # Hamiltonian constraint
    hamiltonian = ricci_3d + K_squared - K_ij_K_ij
    
    return hamiltonian


# @jit
def compute_momentum_constraint(vars: BSSNVariables,
                               params: BSSNParameters) -> jnp.ndarray:
    """
    Compute momentum constraint violation.
    
    The momentum constraint is:
    M_i = ∇_j (K^j_i - δ^j_i K) = 0
    
    Args:
        vars: BSSN variables
        params: Evolution parameters
        
    Returns:
        Momentum constraint violation M_i (3-vector)
    """
    dx = params.dx
    shape = vars.conformal_metric.shape[2:]
    
    # Compute inverse conformal metric
    inv_metric = invert_3x3_metric(vars.conformal_metric)
    
    # Compute K^j_i = γ^jk K_ki
    K_mixed = jnp.zeros((3, 3) + shape)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                K_mixed = K_mixed.at[j, i].add(
                    inv_metric[j, k] * vars.traceless_K[k, i])
            # Add trace part
            K_mixed = K_mixed.at[j, i].add((1.0/3.0) * inv_metric[j, i] * vars.trace_K)
    
    # Compute divergence of K^j_i for each i
    momentum = jnp.zeros((3,) + shape)
    for i in range(3):
        # Construct K^j_i - δ^j_i K for this i
        K_term = jnp.zeros((3,) + shape)
        for j in range(3):
            K_term = K_term.at[j].set(K_mixed[j, i])
            if i == j:
                K_term = K_term.at[j].add(-vars.trace_K)
        
        # Compute divergence
        div_K = divergence_3d(K_term, dx)
        momentum = momentum.at[i].set(div_K)
    
    return momentum


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
    dx = params.dx
    shape = vars.conformal_metric.shape[2:]
    
    # Compute metric derivatives
    metric_derivs = jnp.zeros((3, 3, 3) + shape)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                metric_derivs = metric_derivs.at[k, i, j].set(
                    diff1_field(vars.conformal_metric[i, j], k, dx))
    
    # Compute inverse metric
    inv_metric = invert_3x3_metric(vars.conformal_metric)
    
    # Compute Christoffel symbols
    christoffel = christoffel_symbols_second_kind(inv_metric, metric_derivs)
    
    # Compute γ^jk Γ^i_jk
    gamma_from_christoffel = jnp.zeros((3,) + shape)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                gamma_from_christoffel = gamma_from_christoffel.at[i].add(
                    inv_metric[j, k] * christoffel[i, j, k])
    
    # Constraint violation
    gamma_violation = jnp.zeros((3,) + shape)
    for i in range(3):
        gamma_violation = gamma_violation.at[i].set(
            vars.conformal_connection[i] - gamma_from_christoffel[i])
    
    return gamma_violation


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


@jit
def compute_constraint_norms(violations: ConstraintViolations) -> dict:
    """
    Compute various norms of constraint violations.
    
    Args:
        violations: Constraint violations
        
    Returns:
        Dictionary of constraint norms
    """
    norms = {}
    
    # L2 norms
    norms['hamiltonian_l2'] = jnp.sqrt(jnp.mean(violations.hamiltonian**2))
    norms['momentum_l2'] = jnp.sqrt(jnp.mean(violations.momentum**2))
    norms['det_gamma_l2'] = jnp.sqrt(jnp.mean(violations.det_gamma**2))
    norms['trace_A_l2'] = jnp.sqrt(jnp.mean(violations.trace_A**2))
    norms['gamma_l2'] = jnp.sqrt(jnp.mean(violations.gamma_condition**2))
    
    # L∞ norms (maximum values)
    norms['hamiltonian_linf'] = jnp.max(jnp.abs(violations.hamiltonian))
    norms['momentum_linf'] = jnp.max(jnp.abs(violations.momentum))
    norms['det_gamma_linf'] = jnp.max(jnp.abs(violations.det_gamma))
    norms['trace_A_linf'] = jnp.max(jnp.abs(violations.trace_A))
    norms['gamma_linf'] = jnp.max(jnp.abs(violations.gamma_condition))
    
    return norms


@jit
def compute_energy_density(vars: BSSNVariables, params: BSSNParameters) -> jnp.ndarray:
    """
    Compute gravitational energy density.
    
    This is useful for monitoring wave propagation and energy conservation.
    
    Args:
        vars: BSSN variables
        params: Evolution parameters
        
    Returns:
        Energy density at each grid point
    """
    # Simplified energy density based on extrinsic curvature
    inv_metric = invert_3x3_metric(vars.conformal_metric)
    
    # Kinetic energy from extrinsic curvature
    K_energy = 0.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    K_energy += (inv_metric[i, k] * inv_metric[j, l] * 
                                vars.traceless_K[i, j] * vars.traceless_K[k, l])
    
    K_energy += vars.trace_K**2 / 3.0
    
    # Potential energy from conformal factor gradients (simplified)
    dx = params.dx
    W_derivs = compute_all_derivatives(vars.conformal_factor, dx)
    
    potential_energy = 0.0
    for i in range(3):
        for j in range(3):
            potential_energy += inv_metric[i, j] * W_derivs[i] * W_derivs[j]
    
    total_energy = K_energy + potential_energy
    
    return total_energy


@jit
def compute_wave_extraction_quantities(vars: BSSNVariables, 
                                      params: BSSNParameters,
                                      radius: float) -> dict:
    """
    Compute quantities for gravitational wave extraction (simplified).
    
    This extracts the ψ4 Weyl scalar at a given radius, which is related
    to outgoing gravitational radiation.
    
    Args:
        vars: BSSN variables
        params: Evolution parameters
        radius: Extraction radius
        
    Returns:
        Dictionary of wave extraction quantities
    """
    dx = params.dx
    shape = vars.conformal_metric.shape[2:]
    ni, nj, nk = shape
    
    # Create coordinate arrays
    x = jnp.arange(ni) * dx - (ni - 1) * dx / 2
    y = jnp.arange(nj) * dx - (nj - 1) * dx / 2
    z = jnp.arange(nk) * dx - (nk - 1) * dx / 2
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    
    r = jnp.sqrt(X**2 + Y**2 + Z**2)
    
    # Find points near extraction radius
    mask = jnp.abs(r - radius) < dx
    
    # Extract metric perturbations (simplified)
    h_plus = vars.conformal_metric[0, 0] - vars.conformal_metric[1, 1]
    h_cross = vars.conformal_metric[0, 1]
    
    # Compute averages on extraction sphere
    if jnp.sum(mask) > 0:
        h_plus_avg = jnp.sum(h_plus * mask) / jnp.sum(mask)
        h_cross_avg = jnp.sum(h_cross * mask) / jnp.sum(mask)
    else:
        h_plus_avg = 0.0
        h_cross_avg = 0.0
    
    # Time derivatives (would need to store previous timestep)
    # For now, return spatial quantities
    quantities = {
        'h_plus': h_plus_avg,
        'h_cross': h_cross_avg,
        'extraction_radius': radius
    }
    
    return quantities


@jit
def compute_convergence_test(vars_coarse: BSSNVariables, 
                            vars_fine: BSSNVariables,
                            refinement_factor: int = 2) -> dict:
    """
    Compute convergence test between different resolutions.
    
    This compares solutions at different grid resolutions to verify
    convergence to the continuum limit.
    
    Args:
        vars_coarse: BSSN variables on coarse grid
        vars_fine: BSSN variables on fine grid (interpolated to coarse)
        refinement_factor: Factor by which fine grid is refined
        
    Returns:
        Dictionary of convergence measures
    """
    # Compute differences in key quantities
    diff_gamma = vars_fine.conformal_metric - vars_coarse.conformal_metric
    diff_W = vars_fine.conformal_factor - vars_coarse.conformal_factor
    diff_A = vars_fine.traceless_K - vars_coarse.traceless_K
    diff_K = vars_fine.trace_K - vars_coarse.trace_K
    
    # L2 norms of differences
    gamma_diff_norm = jnp.sqrt(jnp.mean(diff_gamma**2))
    W_diff_norm = jnp.sqrt(jnp.mean(diff_W**2))
    A_diff_norm = jnp.sqrt(jnp.mean(diff_A**2))
    K_diff_norm = jnp.sqrt(jnp.mean(diff_K**2))
    
    # Expected convergence rate for 4th-order methods
    expected_rate = refinement_factor**4
    
    convergence = {
        'gamma_diff_norm': gamma_diff_norm,
        'W_diff_norm': W_diff_norm,
        'A_diff_norm': A_diff_norm,
        'K_diff_norm': K_diff_norm,
        'expected_rate': expected_rate,
        'refinement_factor': refinement_factor
    }
    
    return convergence


def print_constraint_summary(violations: ConstraintViolations, time: float):
    """
    Print summary of constraint violations (not JIT-compiled).
    
    Args:
        violations: Constraint violations
        time: Current simulation time
    """
    norms = compute_constraint_norms(violations)
    
    print(f"Time: {time:.4f}")
    print(f"  Hamiltonian L2:  {norms['hamiltonian_l2']:.2e}")
    print(f"  Momentum L2:     {norms['momentum_l2']:.2e}")
    print(f"  det(γ)-1 L2:     {norms['det_gamma_l2']:.2e}")
    print(f"  tr(A) L2:        {norms['trace_A_l2']:.2e}")
    print(f"  Γ constraint L2: {norms['gamma_l2']:.2e}")
    print()


def monitor_simulation_health(vars: BSSNVariables, params: BSSNParameters,
                             time: float, max_constraint_violation: float = 1e-2) -> bool:
    """
    Monitor simulation health and return whether to continue.
    
    Args:
        vars: Current BSSN variables
        params: Evolution parameters  
        time: Current time
        max_constraint_violation: Maximum allowed constraint violation
        
    Returns:
        True if simulation should continue, False if it should stop
    """
    # Check for NaN or infinite values
    for field in [vars.conformal_metric, vars.conformal_factor, vars.traceless_K,
                  vars.trace_K, vars.conformal_connection, vars.lapse, vars.shift]:
        if jnp.any(jnp.isnan(field)) or jnp.any(jnp.isinf(field)):
            print(f"ERROR: NaN or Inf detected at time {time}")
            return False
    
    # Check constraint violations
    violations = compute_all_constraints(vars, params)
    norms = compute_constraint_norms(violations)
    
    max_violation = max(norms['hamiltonian_l2'], norms['momentum_l2'], 
                       norms['det_gamma_l2'], norms['trace_A_l2'])
    
    if max_violation > max_constraint_violation:
        print(f"ERROR: Constraint violation too large at time {time}: {max_violation}")
        return False
    
    return True
