"""Axisymmetric Cartoon reconstruction and evolution for Maxwell covectors."""

from functools import partial

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.cartoon.axisymmetry import (
    fill_axisymmetric_ghosts,
    project_axisymmetric_rhs,
    reconstruct_axisymmetric_support,
)
from JAX_BSSN.cartoon.axisymmetry.reconstruction import (
    AXISYMMETRIC_CENTER,
    AXISYMMETRIC_GHOST_CELLS,
    AXISYMMETRIC_OUTER_BUFFER_CELLS,
    AXISYMMETRIC_SUPPORT_SIZE,
)
from JAX_BSSN.cartoon.interpolation import lagrange6_nonperiodic
from JAX_BSSN.evolution.boundaries import PERIODIC_BC, SOMMERFELD_BC
from JAX_BSSN.evolution.time_evolve import (
    compute_bssn_rhs_with_matter,
    enforce_algebraic_constraints,
)
from JAX_BSSN.bssn.tensor_algebra import (
    christoffel_symbols_second_kind,
    invert_3x3_metric,
)
from JAX_BSSN.bssn.geometry import W_FLOOR_VALUE

from JAX_BSSN.EM.derivatives import (
    covariant_derivative_covector,
    spatial_derivatives,
)
from JAX_BSSN.EM.equations import compute_em_rhs
from JAX_BSSN.EM.energy_momentum import (
    compute_electromagnetic_energy_momentum,
)
from JAX_BSSN.EM.variables import EinsteinMaxwellVariables, EMVariables


VECTOR_RADIAL_REFLECTION_PARITY = (-1.0, -1.0, 1.0)


def _vector_parity(dtype):
    return jnp.asarray(VECTOR_RADIAL_REFLECTION_PARITY, dtype=dtype)


def _reflect_inner_ghosts(positive_plane):
    parity = _vector_parity(positive_plane.dtype)[:, None, None]
    ghosts = jnp.flip(
        positive_plane[:, :AXISYMMETRIC_GHOST_CELLS, :], axis=-2
    ) * parity
    return jnp.concatenate((ghosts, positive_plane), axis=-2)


def _reflect_full_plane(positive_plane):
    parity = _vector_parity(positive_plane.dtype)[:, None, None]
    negative = jnp.flip(positive_plane, axis=-2) * parity
    return jnp.concatenate((negative, positive_plane), axis=-2)


def _compact_vector(positive_plane):
    return jnp.expand_dims(_reflect_inner_ghosts(positive_plane), axis=-2)


def _expanded_vector(positive_plane):
    return jnp.expand_dims(_reflect_full_plane(positive_plane), axis=-2)


def _reference_plane(field):
    return field[:, :, 0, :]


def _positive_plane(field):
    return _reference_plane(field)[:, AXISYMMETRIC_GHOST_CELLS:, :]


def validate_axisymmetric_wave_grid(
    wave: EMVariables, params: BSSNParameters
) -> None:
    """Validate compact axisymmetric storage and support geometry."""

    shape = wave.electric_field.shape
    if len(shape) != 4 or shape[0] != 3 or shape[2] != 1:
        raise ValueError(
            "axisymmetric wave fields require shape (3, Nrho + 4, 1, Nz)"
        )
    if shape[1] - AXISYMMETRIC_GHOST_CELLS < 6:
        raise ValueError(
            "axisymmetric wave mode requires at least six radial points"
        )
    if shape[3] < 9:
        raise ValueError("axisymmetric wave mode requires at least nine z points")
    for field in wave:
        if field.shape != shape:
            raise ValueError("all axisymmetric wave fields need one shape")

    if params.xl_bc != PERIODIC_BC:
        raise ValueError("axisymmetric x-left is a parity ghost region")
    if params.xr_bc != SOMMERFELD_BC:
        raise ValueError("axisymmetric x-right must use Sommerfeld")
    if params.yl_bc != PERIODIC_BC or params.yr_bc != PERIODIC_BC:
        raise ValueError("axisymmetric reconstructed y faces must be periodic")
    valid_z_codes = (PERIODIC_BC, SOMMERFELD_BC)
    if params.zl_bc not in valid_z_codes or params.zr_bc not in valid_z_codes:
        raise ValueError("axisymmetric z faces must be periodic or Sommerfeld")
    if float(params.mad_q) != 1.0:
        raise ValueError("axisymmetric Cartoon requires mad_q=1")
    if float(params.dx) <= 0.0:
        raise ValueError("axisymmetric Cartoon requires dx > 0")
    if not jnp.isclose(params.x_min, -3.5 * params.dx):
        raise ValueError("axisymmetric Cartoon requires x_min=-3.5*dx")
    if not jnp.isclose(params.y_min, -4.0 * params.dx):
        raise ValueError("axisymmetric Cartoon requires y_min=-4*dx")


def compact_axisymmetric_wave(wave: EMVariables) -> EMVariables:
    """Compact a complete signed x-z Maxwell plane to positive rho."""

    full_nx = wave.electric_field.shape[1]
    if wave.electric_field.shape[2] != 1 or full_nx % 2:
        raise ValueError(
            "axisymmetric initialization requires an even signed x plane"
        )
    positive = slice(full_nx // 2, None)
    return EMVariables(
        *(
            _compact_vector(field[:, positive, 0, :])
            for field in wave
        )
    )


def fill_axisymmetric_wave_ghosts(wave: EMVariables) -> EMVariables:
    """Refresh negative-rho ghosts using rotation-by-pi parity."""

    return EMVariables(
        *(_compact_vector(_positive_plane(field)) for field in wave)
    )


def expand_axisymmetric_wave_plane(wave: EMVariables) -> EMVariables:
    """Expand compact Maxwell fields onto the complete signed x-z plane."""

    return EMVariables(
        *(_expanded_vector(_positive_plane(field)) for field in wave)
    )


def _outer_buffer(reference, params):
    """Append fixed-z samples with zero-asymptotic physical 1/r falloff."""

    nx, nz = reference.shape[-2:]
    num_radial_points = nx - AXISYMMETRIC_GHOST_CELLS
    dtype = reference.dtype
    dx = jnp.asarray(params.dx, dtype=dtype)
    rho_edge = (num_radial_points - 0.5) * dx
    z = jnp.asarray(params.z_min, dtype=dtype) + dx * jnp.arange(
        nz, dtype=dtype
    )
    radius_edge = jnp.sqrt(rho_edge**2 + z**2)
    rho_buffer = rho_edge + dx * jnp.arange(
        1, AXISYMMETRIC_OUTER_BUFFER_CELLS + 1, dtype=dtype
    )
    radius_buffer = jnp.sqrt(rho_buffer[:, None] ** 2 + z[None, :] ** 2)
    ratio = radius_edge[None, :] / radius_buffer
    ratio = ratio.reshape((1,) * (reference.ndim - 2) + ratio.shape)
    buffer = reference[..., -1:, :] * ratio
    return jnp.concatenate((reference, buffer), axis=-2)


def _support_geometry(reference, params):
    nx = reference.shape[-2]
    dtype = reference.dtype
    dx = jnp.asarray(params.dx, dtype=dtype)
    x = (jnp.arange(nx, dtype=dtype) - 3.5) * dx
    y = (
        jnp.arange(AXISYMMETRIC_SUPPORT_SIZE, dtype=dtype)
        - AXISYMMETRIC_CENTER
    ) * dx
    X = x[:, None]
    Y = y[None, :]
    rho = jnp.sqrt(X**2 + Y**2)
    q = rho / dx + 3.5
    return q, X / rho, Y / rho


def _rotation_matrix(cosine, sine, dtype):
    zeros = jnp.zeros_like(cosine)
    ones = jnp.ones_like(cosine)
    return jnp.stack(
        (
            jnp.stack((cosine, -sine, zeros), axis=0),
            jnp.stack((sine, cosine, zeros), axis=0),
            jnp.stack((zeros, zeros, ones), axis=0),
        ),
        axis=0,
    ).astype(dtype)[..., None]


def _reconstruct_vector(field, params):
    reference = _reference_plane(field)
    source = _outer_buffer(reference, params)
    q, cosine, sine = _support_geometry(reference, params)
    radial_values = lagrange6_nonperiodic(source, q, axis=-2)
    rotation = _rotation_matrix(cosine, sine, field.dtype)
    return jnp.einsum("ijxyz,jxyz->ixyz", rotation, radial_values)


def reconstruct_axisymmetric_wave_support(
    wave: EMVariables, params: BSSNParameters
) -> EMVariables:
    """Interpolate and rotate Maxwell covectors onto nine Cartesian y planes."""

    wave = fill_axisymmetric_wave_ghosts(wave)
    return EMVariables(*(_reconstruct_vector(field, params) for field in wave))


def _project_vector(field):
    positive = field[
        :, AXISYMMETRIC_GHOST_CELLS:, AXISYMMETRIC_CENTER, :
    ]
    return _compact_vector(positive)


def _project_scalar(field):
    positive = field[
        AXISYMMETRIC_GHOST_CELLS:, AXISYMMETRIC_CENTER, :
    ]
    ghosts = jnp.flip(
        positive[:AXISYMMETRIC_GHOST_CELLS, :], axis=-2
    )
    return jnp.expand_dims(jnp.concatenate((ghosts, positive), axis=-2), -2)


def project_axisymmetric_wave_rhs(rhs: EMVariables) -> EMVariables:
    """Project a Cartesian support RHS onto the positive-rho reference plane."""

    return EMVariables(*(_project_vector(field) for field in rhs))


def compute_axisymmetric_constraint_divergences(
    wave: EMVariables,
    bssn,
    params: BSSNParameters,
):
    """Return compact D_i E^i and D_i H^i for compact coupled data."""

    support_em = reconstruct_axisymmetric_wave_support(wave, params)
    support_bssn = reconstruct_axisymmetric_support(bssn, params)
    W = jnp.maximum(support_bssn.conformal_factor, W_FLOOR_VALUE)
    inverse_conformal_metric = invert_3x3_metric(
        support_bssn.conformal_metric
    )
    metric = support_bssn.conformal_metric / W**2
    inverse_metric = W**2 * inverse_conformal_metric
    christoffel = christoffel_symbols_second_kind(
        inverse_metric, spatial_derivatives(metric, params)
    )

    def divergence(field):
        derivative = covariant_derivative_covector(
            field, christoffel, params
        )
        support = jnp.einsum(
            "ij...,ij...->...", inverse_metric, derivative
        )
        return _project_scalar(support)

    return divergence(support_em.electric_field), divergence(
        support_em.magnetic_field
    )


def _add_scaled(state, rhs, scale):
    return jax.tree_util.tree_map(
        lambda value, derivative: value + scale * derivative,
        state,
        rhs,
    )


@partial(jax.jit, static_argnames=("background_func",))
def compute_axisymmetric_prescribed_rhs(
    wave,
    time,
    params,
    background_func,
    background_params,
):
    """Evaluate a prescribed Maxwell RHS on axisymmetric support."""

    support_em = reconstruct_axisymmetric_wave_support(wave, params)
    support_bssn, support_bssn_rhs = background_func(
        time, support_em, background_params
    )
    support_rhs = compute_em_rhs(
        support_em, support_bssn, support_bssn_rhs, params
    )
    return project_axisymmetric_wave_rhs(support_rhs)


@partial(jax.jit, static_argnames=("background_func",))
def axisymmetric_prescribed_wave_rk4_step(
    wave,
    time,
    params,
    background_func,
    background_params,
):
    """Advance compact axisymmetric Maxwell fields on prescribed geometry."""

    dt = params.dt
    wave = fill_axisymmetric_wave_ghosts(wave)
    k1 = compute_axisymmetric_prescribed_rhs(
        wave, time, params, background_func, background_params
    )

    midpoint = fill_axisymmetric_wave_ghosts(
        _add_scaled(wave, k1, 0.5 * dt)
    )
    k2 = compute_axisymmetric_prescribed_rhs(
        midpoint,
        time + 0.5 * dt,
        params,
        background_func,
        background_params,
    )

    midpoint = fill_axisymmetric_wave_ghosts(
        _add_scaled(wave, k2, 0.5 * dt)
    )
    k3 = compute_axisymmetric_prescribed_rhs(
        midpoint,
        time + 0.5 * dt,
        params,
        background_func,
        background_params,
    )

    endpoint = fill_axisymmetric_wave_ghosts(_add_scaled(wave, k3, dt))
    k4 = compute_axisymmetric_prescribed_rhs(
        endpoint,
        time + dt,
        params,
        background_func,
        background_params,
    )
    increment = jax.tree_util.tree_map(
        lambda d1, d2, d3, d4: (d1 + 2.0 * d2 + 2.0 * d3 + d4)
        / 6.0,
        k1,
        k2,
        k3,
        k4,
    )
    return fill_axisymmetric_wave_ghosts(_add_scaled(wave, increment, dt))


def _prepare_coupled_stage(state):
    bssn = enforce_algebraic_constraints(state.bssn)
    return EinsteinMaxwellVariables(
        bssn=fill_axisymmetric_ghosts(bssn),
        em=fill_axisymmetric_wave_ghosts(state.em),
    )


@jax.jit
def compute_axisymmetric_einstein_maxwell_rhs(
    state: EinsteinMaxwellVariables, params: BSSNParameters
) -> EinsteinMaxwellVariables:
    """Compute synchronized compact BSSN and axisymmetric Maxwell RHS data."""

    support_bssn = reconstruct_axisymmetric_support(state.bssn, params)
    support_em = reconstruct_axisymmetric_wave_support(state.em, params)
    rho, momentum_density, spatial_stress = (
        compute_electromagnetic_energy_momentum(
            support_em.electric_field,
            support_em.magnetic_field,
            support_bssn,
        )
    )
    support_bssn_rhs = compute_bssn_rhs_with_matter(
        support_bssn,
        params,
        rho,
        momentum_density,
        spatial_stress,
    )
    support_em_rhs = compute_em_rhs(
        support_em,
        support_bssn,
        support_bssn_rhs,
        params,
        backreaction_sources=(rho, momentum_density, spatial_stress),
    )
    return EinsteinMaxwellVariables(
        bssn=project_axisymmetric_rhs(support_bssn_rhs),
        em=project_axisymmetric_wave_rhs(support_em_rhs),
    )


@jax.jit
def axisymmetric_einstein_maxwell_rk4_step(
    state: EinsteinMaxwellVariables, params: BSSNParameters
) -> EinsteinMaxwellVariables:
    """Advance compact axisymmetric BSSN and Maxwell through one RK4 step."""

    dt = params.dt
    state = _prepare_coupled_stage(state)
    k1 = compute_axisymmetric_einstein_maxwell_rhs(state, params)

    midpoint = _prepare_coupled_stage(_add_scaled(state, k1, 0.5 * dt))
    k2 = compute_axisymmetric_einstein_maxwell_rhs(midpoint, params)

    midpoint = _prepare_coupled_stage(_add_scaled(state, k2, 0.5 * dt))
    k3 = compute_axisymmetric_einstein_maxwell_rhs(midpoint, params)

    endpoint = _prepare_coupled_stage(_add_scaled(state, k3, dt))
    k4 = compute_axisymmetric_einstein_maxwell_rhs(endpoint, params)
    increment = jax.tree_util.tree_map(
        lambda d1, d2, d3, d4: (d1 + 2.0 * d2 + 2.0 * d3 + d4)
        / 6.0,
        k1,
        k2,
        k3,
        k4,
    )
    return _prepare_coupled_stage(_add_scaled(state, increment, dt))


__all__ = [
    "axisymmetric_einstein_maxwell_rk4_step",
    "axisymmetric_prescribed_wave_rk4_step",
    "compact_axisymmetric_wave",
    "compute_axisymmetric_constraint_divergences",
    "compute_axisymmetric_einstein_maxwell_rhs",
    "compute_axisymmetric_prescribed_rhs",
    "expand_axisymmetric_wave_plane",
    "fill_axisymmetric_wave_ghosts",
    "project_axisymmetric_wave_rhs",
    "reconstruct_axisymmetric_wave_support",
    "validate_axisymmetric_wave_grid",
]
