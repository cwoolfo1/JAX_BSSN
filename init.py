"""
Initial data setup for numerical relativity simulations.

This module provides initial data for various spacetimes including
gravitational waves, black holes, and other analytical solutions.
All functions are JIT-compiled with JAX.
"""

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from typing import Tuple, NamedTuple
import math

from bssn import BSSNVariables, BSSNParameters
from tensor_algebra import invert_3x3_metric, determinant_3x3_metric, traceless_part


def create_coordinate_arrays(ni: int, nj: int, nk: int, 
                           dx: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Create coordinate arrays for the computational grid.
    
    Args:
        ni, nj, nk: Grid dimensions
        dx: Grid spacing (assumed uniform)
        
    Returns:
        Tuple of (x, y, z) coordinate arrays
    """
    # Create centered coordinates
    x = jnp.arange(ni) * dx - (ni - 1) * dx / 2
    y = jnp.arange(nj) * dx - (nj - 1) * dx / 2  
    z = jnp.arange(nk) * dx - (nk - 1) * dx / 2
    
    # Create meshgrid
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    
    return X, Y, Z


def flat_spacetime_data(ni: int, nj: int, nk: int, dx: float) -> BSSNVariables:
    """
    Initialize flat Minkowski spacetime data.
    
    Args:
        ni, nj, nk: Grid dimensions
        dx: Grid spacing
        
    Returns:
        BSSN variables for flat spacetime
    """
    shape = (ni, nj, nk)
    
    # Conformal metric = flat 3-metric
    conformal_metric = jnp.zeros((3, 3) + shape)
    conformal_metric = conformal_metric.at[0, 0].set(jnp.ones(shape))  # γ_xx = 1
    conformal_metric = conformal_metric.at[1, 1].set(jnp.ones(shape))  # γ_yy = 1  
    conformal_metric = conformal_metric.at[2, 2].set(jnp.ones(shape))  # γ_zz = 1
    
    # Conformal factor = 1
    conformal_factor = jnp.ones(shape)
    
    # All extrinsic curvature = 0
    traceless_K = jnp.zeros((3, 3) + shape)
    trace_K = jnp.zeros(shape)
    
    # Conformal connection = 0
    conformal_connection = jnp.zeros((3,) + shape)
    
    # Lapse = 1, shift = 0
    lapse = jnp.ones(shape)
    shift = jnp.zeros((3,) + shape)
    
    return BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=conformal_factor,
        traceless_K=traceless_K,
        trace_K=trace_K,
        conformal_connection=conformal_connection,
        lapse=lapse,
        shift=shift
    )


def gravitational_wave_data(ni: int, nj: int, nk: int, dx: float,
                           amplitude: float = 0.1, 
                           wavelength: float = 1.0) -> BSSNVariables:
    """
    Initialize gravitational wave data (linearized gravity).
    
    This implements a simple plane wave solution to linearized gravity:
    h_ij = A * sin(2π(y - x)/λ) for + polarization
    
    Args:
        ni, nj, nk: Grid dimensions  
        dx: Grid spacing
        amplitude: Wave amplitude
        wavelength: Wave wavelength
        
    Returns:
        BSSN variables for gravitational wave
    """
    shape = (ni, nj, nk)
    X, Y, Z = create_coordinate_arrays(ni, nj, nk, dx)
    
    # Wave phase
    phase = 2 * jnp.pi * (Y - X) / wavelength
    h = amplitude * jnp.sin(phase)
    
    # Start with flat spacetime
    vars = flat_spacetime_data(ni, nj, nk, dx)
    
    # Add gravitational wave perturbation
    # For + polarization: h_+ affects xx and yy components
    vars = vars._replace(
        conformal_metric=vars.conformal_metric.at[0, 0].set(1 + h).at[1, 1].set(1 - h)
    )
    
    # Renormalize to maintain det(γ) = 1
    det_gamma = determinant_3x3_metric(vars.conformal_metric)
    conformal_factor = det_gamma**(-1/3)
    
    # Rescale conformal metric
    conformal_metric_rescaled = conformal_factor * vars.conformal_metric

    return vars._replace(
        conformal_metric=conformal_metric_rescaled,
        conformal_factor=conformal_factor
    )


def schwarzschild_data(ni: int, nj: int, nk: int, dx: float,
                      mass: float = 1.0) -> BSSNVariables:
    """
    Initialize Schwarzschild black hole data using isotropic coordinates.
    
    The Schwarzschild metric in isotropic coordinates:
    ds² = -(1-M/2r)²/(1+M/2r)² dt² + (1+M/2r)⁴(dx²+dy²+dz²)
    
    Args:
        ni, nj, nk: Grid dimensions
        dx: Grid spacing  
        mass: Black hole mass
        
    Returns:
        BSSN variables for Schwarzschild black hole
    """
    shape = (ni, nj, nk)
    X, Y, Z = create_coordinate_arrays(ni, nj, nk, dx)
    
    # Isotropic radius
    r = jnp.sqrt(X**2 + Y**2 + Z**2)
    
    # Avoid singularity at origin
    r = jnp.maximum(r, 0.1 * dx)
    
    # Conformal factor ψ = 1 + M/(2r)
    psi = 1 + mass / (2 * r)
    
    # Conformal metric = flat metric (isotropic coordinates)
    conformal_metric = jnp.zeros((3, 3) + shape)
    conformal_metric = conformal_metric.at[0, 0].set(jnp.ones(shape))
    conformal_metric = conformal_metric.at[1, 1].set(jnp.ones(shape))
    conformal_metric = conformal_metric.at[2, 2].set(jnp.ones(shape))
    
    # BSSN conformal factor W = ψ²  
    conformal_factor = psi**2
    
    # Extrinsic curvature = 0 (time symmetric)
    traceless_K = jnp.zeros((3, 3) + shape)
    trace_K = jnp.zeros(shape)
    
    # Conformal connection functions (from conformal factor)
    conformal_connection = jnp.zeros((3,) + shape)
    
    # Lapse function α = (1-M/2r)/(1+M/2r)
    lapse = (1 - mass/(2*r)) / (1 + mass/(2*r))
    
    # Zero shift
    shift = jnp.zeros((3,) + shape)
    
    return BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=conformal_factor,
        traceless_K=traceless_K,
        trace_K=trace_K,
        conformal_connection=conformal_connection,
        lapse=lapse,
        shift=shift
    )


def kerr_schild_data(ni: int, nj: int, nk: int, dx: float,
                    mass: float = 1.0, spin: float = 0.5) -> BSSNVariables:
    """
    Initialize Kerr black hole data using Kerr-Schild coordinates.
    
    This is a simplified version that sets up the basic structure.
    A full implementation would require solving elliptic equations.
    
    Args:
        ni, nj, nk: Grid dimensions
        dx: Grid spacing
        mass: Black hole mass  
        spin: Dimensionless spin parameter a/M
        
    Returns:
        BSSN variables for Kerr black hole (simplified)
    """
    shape = (ni, nj, nk)
    X, Y, Z = create_coordinate_arrays(ni, nj, nk, dx)
    
    # Start with Schwarzschild data
    vars = schwarzschild_data(ni, nj, nk, dx, mass)
    
    # Add simple rotation effects (this is highly simplified)
    r = jnp.sqrt(X**2 + Y**2 + Z**2)
    r = jnp.maximum(r, 0.1 * dx)
    
    # Add angular momentum via shift vector (simplified)
    a = spin * mass
    rho2 = r**2 + a**2 * Z**2 / r**2
    
    # Approximate shift components
    shift_x = -a * Y / rho2
    shift_y = a * X / rho2
    shift_z = jnp.zeros_like(shift_x)
    
    shift = jnp.array([shift_x, shift_y, shift_z])
    
    return vars._replace(shift=shift)


def gaussian_pulse_data(ni: int, nj: int, nk: int, dx: float,
                       amplitude: float = 0.1, width: float = 1.0,
                       center: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> BSSNVariables:
    """
    Initialize Gaussian pulse initial data.
    
    This creates a localized gravitational wave pulse for testing purposes.
    
    Args:
        ni, nj, nk: Grid dimensions
        dx: Grid spacing
        amplitude: Pulse amplitude
        width: Pulse width (standard deviation)
        center: Pulse center coordinates
        
    Returns:
        BSSN variables for Gaussian pulse
    """
    shape = (ni, nj, nk)
    X, Y, Z = create_coordinate_arrays(ni, nj, nk, dx)
    
    # Distance from center
    cx, cy, cz = center
    r2 = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2
    
    # Gaussian profile
    gaussian = amplitude * jnp.exp(-r2 / (2 * width**2))
    
    # Start with flat spacetime
    vars = flat_spacetime_data(ni, nj, nk, dx)
    
    # Add pulse to conformal metric (simplified)
    vars = vars._replace(
        conformal_metric=vars.conformal_metric.at[0, 0].set(1 + gaussian).at[1, 1].set(1 - gaussian)
    )
    
    # Renormalize
    det_gamma = determinant_3x3_metric(vars.conformal_metric)
    conformal_factor = det_gamma**(1/6)
    
    conformal_metric_rescaled = jnp.zeros_like(vars.conformal_metric)
    for i in range(3):
        for j in range(3):
            conformal_metric_rescaled = conformal_metric_rescaled.at[i, j].set(
                vars.conformal_metric[i, j] / conformal_factor**(2/3))
    
    return vars._replace(
        conformal_metric=conformal_metric_rescaled,
        conformal_factor=conformal_factor
    )


def brill_wave_data(ni: int, nj: int, nk: int, dx: float,
                   amplitude: float = 0.1, n_modes: int = 2) -> BSSNVariables:
    """
    Initialize Brill wave initial data.
    
    Brill waves are exact solutions representing cylindrical gravitational waves.
    This is a simplified implementation.
    
    Args:
        ni, nj, nk: Grid dimensions
        dx: Grid spacing
        amplitude: Wave amplitude
        n_modes: Number of wave modes
        
    Returns:
        BSSN variables for Brill wave
    """
    shape = (ni, nj, nk)
    X, Y, Z = create_coordinate_arrays(ni, nj, nk, dx)
    
    # Cylindrical coordinates
    rho = jnp.sqrt(X**2 + Y**2)
    phi = jnp.arctan2(Y, X)
    
    # Avoid singularity at axis
    rho = jnp.maximum(rho, 0.1 * dx)
    
    # Brill wave function (simplified)
    q = jnp.zeros_like(rho)
    for n in range(1, n_modes + 1):
        q += amplitude * jnp.exp(-rho**2) * jnp.cos(n * phi) * jnp.cos(n * Z)
    
    # Start with flat data
    vars = flat_spacetime_data(ni, nj, nk, dx)
    
    # Modify conformal metric with Brill wave
    # This is highly simplified - full Brill waves require solving elliptic equations
    factor = 1 + q
    
    vars = vars._replace(
        conformal_metric=vars.conformal_metric.at[0, 0].set(factor).at[1, 1].set(factor)
    )
    
    # Renormalize
    det_gamma = determinant_3x3_metric(vars.conformal_metric)
    conformal_factor = det_gamma**(1/6)
    
    conformal_metric_rescaled = jnp.zeros_like(vars.conformal_metric)
    for i in range(3):
        for j in range(3):
            conformal_metric_rescaled = conformal_metric_rescaled.at[i, j].set(
                vars.conformal_metric[i, j] / conformal_factor**(2/3))
    
    return vars._replace(
        conformal_metric=conformal_metric_rescaled,
        conformal_factor=conformal_factor
    )


def get_initial_data(data_type: str, ni: int, nj: int, nk: int, 
                    dx: float, **kwargs) -> BSSNVariables:
    """
    Get initial data of specified type.
    
    Args:
        data_type: Type of initial data ('flat', 'wave', 'schwarzschild', 
                   'kerr', 'gaussian', 'brill')
        ni, nj, nk: Grid dimensions
        dx: Grid spacing
        **kwargs: Additional parameters for specific data types
        
    Returns:
        BSSN variables for the specified initial data
    """
    if data_type == 'flat':
        return flat_spacetime_data(ni, nj, nk, dx)
    elif data_type == 'wave':
        return gravitational_wave_data(ni, nj, nk, dx, **kwargs)
    elif data_type == 'schwarzschild':
        return schwarzschild_data(ni, nj, nk, dx, **kwargs)
    elif data_type == 'kerr':
        return kerr_schild_data(ni, nj, nk, dx, **kwargs) 
    elif data_type == 'gaussian':
        return gaussian_pulse_data(ni, nj, nk, dx, **kwargs)
    elif data_type == 'brill':
        return brill_wave_data(ni, nj, nk, dx, **kwargs)
    else:
        raise ValueError(f"Unknown initial data type: {data_type}")


# Utility functions for initial data analysis
def compute_adm_mass(vars: BSSNVariables, dx: float) -> float:
    """
    Compute ADM mass using surface integral at infinity (simplified).
    
    Args:
        vars: BSSN variables
        dx: Grid spacing
        
    Returns:
        Approximate ADM mass
    """
    # This is a very simplified calculation
    # Real ADM mass calculation requires surface integrals
    shape = vars.conformal_factor.shape
    
    # Use conformal factor at boundary as rough estimate
    boundary_average = (jnp.mean(vars.conformal_factor[0, :, :]) +
                       jnp.mean(vars.conformal_factor[-1, :, :]) +
                       jnp.mean(vars.conformal_factor[:, 0, :]) +
                       jnp.mean(vars.conformal_factor[:, -1, :]) +
                       jnp.mean(vars.conformal_factor[:, :, 0]) +
                       jnp.mean(vars.conformal_factor[:, :, -1])) / 6
    
    return jnp.maximum(0.0, boundary_average - 1.0)


def compute_adm_momentum(vars: BSSNVariables, dx: float) -> jnp.ndarray:
    """
    Compute ADM momentum (simplified).
    
    Args:
        vars: BSSN variables
        dx: Grid spacing
        
    Returns:
        3-vector of ADM momentum components
    """
    # Simplified calculation using extrinsic curvature
    momentum = jnp.zeros(3)
    
    for i in range(3):
        # Sum traceless extrinsic curvature components  
        for j in range(3):
            momentum = momentum.at[i].add(jnp.sum(vars.traceless_K[i, j]))
    
    return momentum * dx**3
