"""Exact electromagnetic plane waves on a prescribed flat FLRW background."""

from typing import NamedTuple

import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNParameters, BSSNVariables

from JAX_BSSN.EM.derivatives import covariant_derivative_covector
from JAX_BSSN.EM.geometry import compute_bssn_em_geometry
from JAX_BSSN.EM.variables import EMVariables


class FLRWParameters(NamedTuple):
    """Parameters for a periodic plane wave in conformal FLRW coordinates."""

    H: float
    length_z: float = 1.0
    mode: int = 1
    electric_amplitude: float = 1.0


def flrw_scale_factor(time, params: FLRWParameters):
    """Return a(eta) = 1 + H eta."""

    return 1.0 + params.H * time


def flrw_background(
    time,
    wave: EMVariables,
    params: FLRWParameters,
) -> tuple[BSSNVariables, BSSNVariables]:
    """Return exact FLRW BSSN data and its coordinate-time derivative."""

    shape = wave.electric_field.shape[-3:]
    dtype = wave.electric_field.dtype
    a = jnp.asarray(flrw_scale_factor(time, params), dtype=dtype)
    H = jnp.asarray(params.H, dtype=dtype)

    identity = jnp.eye(3, dtype=dtype)[:, :, None, None, None]
    conformal_metric = jnp.broadcast_to(identity, (3, 3) + shape)
    scalar_one = jnp.ones(shape, dtype=dtype)
    vector_zero = jnp.zeros((3,) + shape, dtype=dtype)
    tensor_zero = jnp.zeros((3, 3) + shape, dtype=dtype)

    # gamma_ij = tilde_gamma_ij / W^2 and
    # K_ij = -(2 alpha)^-1 partial_eta gamma_ij.
    bssn = BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=scalar_one / a,
        traceless_K=tensor_zero,
        trace_K=-3.0 * H * scalar_one / a**2,
        conformal_connection=vector_zero,
        lapse=a * scalar_one,
        shift=vector_zero,
    )

    bssn_rhs = BSSNVariables(
        conformal_metric=tensor_zero,
        conformal_factor=-H * scalar_one / a**2,
        traceless_K=tensor_zero,
        trace_K=6.0 * H**2 * scalar_one / a**3,
        conformal_connection=vector_zero,
        lapse=H * scalar_one,
        shift=vector_zero,
    )
    return bssn, bssn_rhs


def flrw_exact_fields(
    shape: tuple[int, int, int],
    time,
    solver_params: BSSNParameters,
    params: FLRWParameters,
    dtype=jnp.float64,
) -> EMVariables:
    """Return the exact cell-centered covectors (E_i, dot E_i, B_i, dot B_i)."""

    z = solver_params.z_min + solver_params.dx * jnp.arange(
        shape[2], dtype=dtype
    )
    phase = 2.0 * jnp.pi * params.mode * (z - time) / params.length_z
    cosine = jnp.broadcast_to(jnp.cos(phase)[None, None, :], shape)
    sine = jnp.broadcast_to(jnp.sin(phase)[None, None, :], shape)

    a = jnp.asarray(flrw_scale_factor(time, params), dtype=dtype)
    H = jnp.asarray(params.H, dtype=dtype)
    E0 = jnp.asarray(params.electric_amplitude, dtype=dtype)
    k = jnp.asarray(2.0 * jnp.pi * params.mode / params.length_z, dtype=dtype)

    # F_{eta x}=-E0 cos(phi), F_{z x}=E0 cos(phi) gives
    # E_x=B_y=E0 cos(phi)/a in the solver's covariant representation.
    field_profile = E0 * cosine / a

    # The stored dots are projected Eulerian derivatives.  From the solver's
    # coordinate conversion, dot E_i = alpha^-1 partial_eta E_i + K_i^j E_j.
    dot_profile = E0 * (k * sine / a**2 - 2.0 * H * cosine / a**3)

    electric_field = jnp.zeros((3,) + shape, dtype=dtype)
    electric_field = electric_field.at[0].set(field_profile)
    electric_field_dot = jnp.zeros_like(electric_field)
    electric_field_dot = electric_field_dot.at[0].set(dot_profile)

    magnetic_field = jnp.zeros_like(electric_field)
    magnetic_field = magnetic_field.at[1].set(field_profile)
    magnetic_field_dot = jnp.zeros_like(electric_field)
    magnetic_field_dot = magnetic_field_dot.at[1].set(dot_profile)

    return EMVariables(
        electric_field,
        electric_field_dot,
        magnetic_field,
        magnetic_field_dot,
    )


def maxwell_constraint_divergences(
    wave: EMVariables,
    time,
    solver_params: BSSNParameters,
    params: FLRWParameters,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return D_i E^i and D_i B^i using the solver's covariant derivative."""

    bssn, bssn_rhs = flrw_background(time, wave, params)
    geometry = compute_bssn_em_geometry(bssn, bssn_rhs, solver_params)

    def divergence(field):
        derivative = covariant_derivative_covector(
            field, geometry.christoffel, solver_params
        )
        return jnp.einsum(
            "ij...,ij...->...", geometry.inverse_metric, derivative
        )

    return divergence(wave.electric_field), divergence(wave.magnetic_field)
