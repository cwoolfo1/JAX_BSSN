"""
Initial data setup for numerical relativity simulations.

This module provides initial data for gauge-wave and Gowdy-wave spacetimes
using the harmonic gauge with zero shift.
"""

from typing import Tuple

import jax.numpy as jnp
from jax import jit
from jax.scipy.special import j0, j1

from JAX_BSSN.bssn import BSSNVariables, BSSNParameters
from JAX_BSSN.derivatives import diff1_field
from JAX_BSSN.tensor_algebra import (
    invert_3x3_metric,
    determinant_3x3_metric,
    christoffel_symbols_second_kind,
)
from JAX_BSSN.kreiss_oliger import get_optimal_dissipation_coefficient


# NOTE: GAUGE WAVE AND GOWDY WAVE INITIAL DATA ARE ALIGNED WITH THE HARMONIC GAUGE FORMULATION.


def setup_simulation_parameters():
    """Set up default simulation parameters for harmonic gauge evolution."""
    # Grid parameters
    ni, nj, nk = 64, 64, 64  # Grid size
    dx = 0.1                 # Grid spacing
    dt = dx                  # Time step (CFL ~ 1 for gauge-wave tests)
    t_final = 1.0            # Final time

    # BSSN parameters (harmonic gauge, zero shift)
    bssn_params = BSSNParameters(
        eta=0.0,
        kappa=0.025,
        f=1.0,
        g=0.0,
        dx=dx,
        dt=dt,
    )

    # Dissipation parameters
    ko_sigma = get_optimal_dissipation_coefficient(dx, dt)

    return {
        "grid": (ni, nj, nk, dx),
        "evolution": (dt, t_final),
        "bssn_params": bssn_params,
        "ko_sigma": ko_sigma,
    }


def create_coordinate_arrays(
    ni: int, nj: int, nk: int, dx: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Create coordinate arrays for the computational grid.

    Args:
        ni, nj, nk: Grid dimensions
        dx: Grid spacing (assumed uniform)

    Returns:
        Tuple of (x, y, z) coordinate arrays
    """
    x = jnp.arange(ni) * dx - (ni - 1) * dx / 2
    y = jnp.arange(nj) * dx - (nj - 1) * dx / 2
    z = jnp.arange(nk) * dx - (nk - 1) * dx / 2

    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    return X, Y, Z


def _compute_conformal_connection(conformal_metric: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Compute conformal connection functions from a conformal metric."""
    derivs = jnp.stack(
        [diff1_field(conformal_metric, d + 2, dx) for d in range(3)], axis=0
    )
    inv_conformal_metric = invert_3x3_metric(conformal_metric)
    christoffel_2 = christoffel_symbols_second_kind(inv_conformal_metric, derivs)
    return jnp.einsum("mn..., imn... -> i...", inv_conformal_metric, christoffel_2)


def gauge_wave_data(
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    amplitude: float = 0.1,
    wavelength: float = 1.0,
) -> BSSNVariables:
    """
    Initialize harmonic gauge wave initial data.

    Args:
        ni, nj, nk: Grid dimensions
        dx: Grid spacing
        amplitude: Gauge wave amplitude
        wavelength: Wave wavelength

    Returns:
        BSSN variables for the gauge wave.
    """
    shape = (ni, nj, nk)
    X, _, _ = create_coordinate_arrays(ni, nj, nk, dx)

    k = 2.0 * jnp.pi / wavelength
    H = amplitude * jnp.sin(k * X)

    physical_metric = jnp.zeros((3, 3) + shape)
    physical_metric = physical_metric.at[0, 0].set(1.0 - H)
    physical_metric = physical_metric.at[1, 1].set(jnp.ones(shape))
    physical_metric = physical_metric.at[2, 2].set(jnp.ones(shape))

    lapse = jnp.sqrt(1.0 - H)
    shift = jnp.zeros((3,) + shape)

    det_gamma = determinant_3x3_metric(physical_metric)
    conformal_factor = det_gamma ** (-1.0 / 6.0)
    conformal_metric = physical_metric * conformal_factor**2

    dHdt = -amplitude * k * jnp.cos(k * X)
    dt_g_xx = -dHdt
    extrinsic_curvature = jnp.zeros_like(physical_metric)
    extrinsic_curvature = extrinsic_curvature.at[0, 0].set(-0.5 * dt_g_xx / lapse)

    inv_conformal_metric = invert_3x3_metric(conformal_metric)
    trace_K = jnp.einsum("mn..., mn... -> ...", inv_conformal_metric, extrinsic_curvature)
    traceless_K = conformal_factor**2 * (
        extrinsic_curvature - conformal_metric * trace_K / 3.0
    )

    conformal_connection = _compute_conformal_connection(conformal_metric, dx)

    return BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=conformal_factor,
        traceless_K=traceless_K,
        trace_K=trace_K,
        conformal_connection=conformal_connection,
        lapse=lapse,
        shift=shift,
    )


def gowdy_wave_data(
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    amplitude: float = 0.1,
    wavelength: float = 1.0,
    t0: float = 1.0,
) -> BSSNVariables:
    """
    Initialize polarized Gowdy wave initial data (Q = 0) at a fixed time t0.

    Args:
        ni, nj, nk: Grid dimensions
        dx: Grid spacing
        amplitude: Wave amplitude
        wavelength: Wave wavelength in the z-direction
        t0: Initial time for the Gowdy solution

    Returns:
        BSSN variables for the Gowdy wave.
    """
    shape = (ni, nj, nk)
    _, _, Z = create_coordinate_arrays(ni, nj, nk, dx)

    k = 2.0 * jnp.pi / wavelength
    P = amplitude * j0(k * t0) * jnp.cos(k * Z)
    P_t = -amplitude * k * j1(k * t0) * jnp.cos(k * Z)
    P_z = -amplitude * k * j0(k * t0) * jnp.sin(k * Z)

    lambda_z = 2.0 * t0 * P_t * P_z
    lambda_raw = jnp.cumsum(lambda_z, axis=2) * dx
    lambda_field = lambda_raw - jnp.mean(lambda_raw)
    lambda_t = t0 * (P_t**2 + P_z**2)

    g_xx = t0 * jnp.exp(P)
    g_yy = t0 * jnp.exp(-P)
    g_zz = t0 ** (-0.5) * jnp.exp(0.5 * lambda_field)

    physical_metric = jnp.zeros((3, 3) + shape)
    physical_metric = physical_metric.at[0, 0].set(g_xx)
    physical_metric = physical_metric.at[1, 1].set(g_yy)
    physical_metric = physical_metric.at[2, 2].set(g_zz)

    lapse = t0 ** (-0.25) * jnp.exp(0.25 * lambda_field)
    shift = jnp.zeros((3,) + shape)

    dt_g_xx = g_xx * (1.0 / t0 + P_t)
    dt_g_yy = g_yy * (1.0 / t0 - P_t)
    dt_g_zz = g_zz * (-0.5 / t0 + 0.5 * lambda_t)

    extrinsic_curvature = jnp.zeros_like(physical_metric)
    extrinsic_curvature = extrinsic_curvature.at[0, 0].set(-0.5 * dt_g_xx / lapse)
    extrinsic_curvature = extrinsic_curvature.at[1, 1].set(-0.5 * dt_g_yy / lapse)
    extrinsic_curvature = extrinsic_curvature.at[2, 2].set(-0.5 * dt_g_zz / lapse)

    det_gamma = determinant_3x3_metric(physical_metric)
    conformal_factor = det_gamma ** (-1.0 / 6.0)
    conformal_metric = physical_metric * conformal_factor**2

    inv_conformal_metric = invert_3x3_metric(conformal_metric)
    trace_K = jnp.einsum("mn..., mn... -> ...", inv_conformal_metric, extrinsic_curvature)
    traceless_K = conformal_factor**2 * (
        extrinsic_curvature - conformal_metric * trace_K / 3.0
    )

    conformal_connection = _compute_conformal_connection(conformal_metric, dx)

    return BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=conformal_factor,
        traceless_K=traceless_K,
        trace_K=trace_K,
        conformal_connection=conformal_connection,
        lapse=lapse,
        shift=shift,
    )


def get_initial_data(
    data_type: str, ni: int, nj: int, nk: int, dx: float, **kwargs
) -> BSSNVariables:
    """
    Get initial data of specified type.

    Args:
        data_type: Type of initial data ('gauge_wave', 'gowdy_wave')
        ni, nj, nk: Grid dimensions
        dx: Grid spacing
        **kwargs: Additional parameters for specific data types

    Returns:
        BSSN variables for the specified initial data
    """
    if data_type in {"gauge_wave", "gauge"}:
        return gauge_wave_data(ni, nj, nk, dx, **kwargs)
    if data_type in {"gowdy_wave", "gowdy"}:
        return gowdy_wave_data(ni, nj, nk, dx, **kwargs)

    raise ValueError(f"Unknown initial data type: {data_type}")


def compute_adm_mass(vars: BSSNVariables, dx: float) -> float:
    """
    Compute ADM mass using surface integral at infinity (simplified).

    Args:
        vars: BSSN variables
        dx: Grid spacing

    Returns:
        Approximate ADM mass
    """
    boundary_average = (
        jnp.mean(vars.conformal_factor[0, :, :])
        + jnp.mean(vars.conformal_factor[-1, :, :])
        + jnp.mean(vars.conformal_factor[:, 0, :])
        + jnp.mean(vars.conformal_factor[:, -1, :])
        + jnp.mean(vars.conformal_factor[:, :, 0])
        + jnp.mean(vars.conformal_factor[:, :, -1])
    ) / 6

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
    momentum = jnp.zeros(3)

    for i in range(3):
        for j in range(3):
            momentum = momentum.at[i].add(jnp.sum(vars.traceless_K[i, j]))

    return momentum * dx**3
