"""Spherical Cartoon reconstruction and evolution for Maxwell covectors."""

from functools import partial

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.cartoon.interpolation import lagrange6_nonperiodic
from JAX_BSSN.cartoon.spherical_symmetry import (
    CARTOON_CENTER,
    CARTOON_GHOST_CELLS,
    CARTOON_OUTER_BUFFER_CELLS,
    CARTOON_SUPPORT_SIZE,
    fill_cartoon_ghosts,
    project_cartoon_rhs,
    reconstruct_cartoon_support,
)
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


VECTOR_X_REFLECTION_PARITY = (-1.0, 1.0, 1.0)


def _vector_parity(dtype):
    return jnp.asarray(VECTOR_X_REFLECTION_PARITY, dtype=dtype)


def _reflect_inner_ghosts(positive_axis):
    parity = _vector_parity(positive_axis.dtype)[:, None]
    ghosts = (
        jnp.flip(positive_axis[:, :CARTOON_GHOST_CELLS], axis=-1)
        * parity
    )
    return jnp.concatenate((ghosts, positive_axis), axis=-1)


def _reflect_full_axis(positive_axis):
    parity = _vector_parity(positive_axis.dtype)[:, None]
    negative_axis = jnp.flip(positive_axis, axis=-1) * parity
    return jnp.concatenate((negative_axis, positive_axis), axis=-1)


def _compact_vector(positive_axis):
    return _reflect_inner_ghosts(positive_axis)[:, :, None, None]


def _expanded_vector(positive_axis):
    return _reflect_full_axis(positive_axis)[:, :, None, None]


def _positive_axis(field):
    return field[:, CARTOON_GHOST_CELLS:, 0, 0]


def validate_cartoon_wave_grid(
    wave: EMVariables, params: BSSNParameters
) -> None:
    """Validate compact spherical storage and support boundary choices."""

    shape = wave.electric_field.shape
    if len(shape) != 4 or shape[0] != 3 or shape[-2:] != (1, 1):
        raise ValueError(
            "spherical Cartoon wave fields require shape (3, Nr + 4, 1, 1)"
        )
    if shape[1] - CARTOON_GHOST_CELLS < 6:
        raise ValueError(
            "spherical Cartoon wave mode requires at least six radial points"
        )
    for field in wave:
        if field.shape != shape:
            raise ValueError("all spherical Cartoon wave fields need one shape")

    if params.xl_bc != PERIODIC_BC:
        raise ValueError("Cartoon x-left is a parity ghost region")
    if params.xr_bc != SOMMERFELD_BC:
        raise ValueError("Cartoon x-right must use the Sommerfeld boundary")
    if (
        params.yl_bc != PERIODIC_BC
        or params.yr_bc != PERIODIC_BC
        or params.zl_bc != PERIODIC_BC
        or params.zr_bc != PERIODIC_BC
    ):
        raise ValueError("Cartoon y/z support faces must be periodic")
    if float(params.mad_q) != 1.0:
        raise ValueError("Cartoon reconstruction requires mad_q=1")

    expected_minima = (
        -(CARTOON_GHOST_CELLS - 0.5) * params.dx,
        -CARTOON_GHOST_CELLS * params.dx,
        -CARTOON_GHOST_CELLS * params.dx,
    )
    actual_minima = (params.x_min, params.y_min, params.z_min)
    if any(
        abs(float(actual) - float(expected))
        > 1.0e-12 * max(1.0, abs(float(expected)))
        for actual, expected in zip(actual_minima, expected_minima)
    ):
        raise ValueError(
            "Cartoon coordinate minima must be (-3.5dx, -4dx, -4dx)"
        )


def compact_cartoon_wave(wave: EMVariables) -> EMVariables:
    """Compact a complete signed x-axis Maxwell state."""

    full_nx = wave.electric_field.shape[1]
    if wave.electric_field.shape[-2:] != (1, 1) or full_nx % 2:
        raise ValueError(
            "Cartoon initialization requires an even signed x-axis"
        )
    positive = slice(full_nx // 2, None)
    return EMVariables(
        *(
            _compact_vector(field[:, positive, 0, 0])
            for field in wave
        )
    )


def fill_cartoon_wave_ghosts(wave: EMVariables) -> EMVariables:
    """Refresh the four negative-x parity ghosts in every Maxwell field."""

    return EMVariables(*(_compact_vector(_positive_axis(field)) for field in wave))


def expand_cartoon_wave_axis(wave: EMVariables) -> EMVariables:
    """Expand compact Maxwell fields onto the complete signed x-axis."""

    return EMVariables(
        *(_expanded_vector(_positive_axis(field)) for field in wave)
    )


def _support_geometry(num_x, dx, dtype):
    spacing = jnp.asarray(dx, dtype=dtype)
    x = (
        jnp.arange(num_x, dtype=dtype)
        - CARTOON_GHOST_CELLS
        + 0.5
    ) * spacing
    transverse = (
        jnp.arange(CARTOON_SUPPORT_SIZE, dtype=dtype) - CARTOON_CENTER
    ) * spacing

    X = x[:, None, None]
    Y = transverse[None, :, None]
    Z = transverse[None, None, :]
    shape = (num_x, CARTOON_SUPPORT_SIZE, CARTOON_SUPPORT_SIZE)
    radius = jnp.sqrt(X**2 + Y**2 + Z**2)
    direction = jnp.stack(
        (
            jnp.broadcast_to(X / radius, shape),
            jnp.broadcast_to(Y / radius, shape),
            jnp.broadcast_to(Z / radius, shape),
        )
    )
    return radius, direction


def _outer_buffer(axis, dx):
    """Append reconstruction-only samples with zero-asymptotic 1/r falloff."""

    spacing = jnp.asarray(dx, dtype=axis.dtype)
    outer_radius = (
        axis.shape[-1] - CARTOON_GHOST_CELLS - 0.5
    ) * spacing
    buffer_radius = outer_radius + spacing * jnp.arange(
        1, CARTOON_OUTER_BUFFER_CELLS + 1, dtype=axis.dtype
    )
    buffer = axis[..., -1:] * outer_radius / buffer_radius
    return jnp.concatenate((axis, buffer), axis=-1)


def _reconstruct_vector(field, radius, direction, dx):
    # A spherically symmetric spatial covector is purely radial.  The
    # positive x component is the authoritative radial profile.
    radial_axis = field[0, :, 0, 0]
    source = _outer_buffer(radial_axis, dx)
    q = (
        radius / jnp.asarray(dx, dtype=radius.dtype)
        + CARTOON_GHOST_CELLS
        - 0.5
    )
    radial = lagrange6_nonperiodic(source, q, axis=-1)
    return radial[None, ...] * direction


def reconstruct_cartoon_wave_support(
    wave: EMVariables, params: BSSNParameters
) -> EMVariables:
    """Interpolate radial covectors onto the Cartesian nine-plane support."""

    wave = fill_cartoon_wave_ghosts(wave)
    num_x = wave.electric_field.shape[1]
    radius, direction = _support_geometry(
        num_x, params.dx, wave.electric_field.dtype
    )
    return EMVariables(
        *(
            _reconstruct_vector(field, radius, direction, params.dx)
            for field in wave
        )
    )


def _project_vector(field):
    positive_axis = field[
        :, CARTOON_GHOST_CELLS:, CARTOON_CENTER, CARTOON_CENTER
    ]
    return _compact_vector(positive_axis)


def _project_scalar(field):
    positive_axis = field[
        CARTOON_GHOST_CELLS:, CARTOON_CENTER, CARTOON_CENTER
    ]
    ghosts = jnp.flip(positive_axis[:CARTOON_GHOST_CELLS], axis=-1)
    return jnp.concatenate((ghosts, positive_axis), axis=-1)[:, None, None]


def project_cartoon_wave_rhs(rhs: EMVariables) -> EMVariables:
    """Project a Cartesian support RHS into compact radial storage."""

    return EMVariables(*(_project_vector(field) for field in rhs))


def compute_cartoon_constraint_divergences(
    wave: EMVariables,
    bssn,
    params: BSSNParameters,
):
    """Return compact D_i E^i and D_i H^i for compact coupled data."""

    support_em = reconstruct_cartoon_wave_support(wave, params)
    support_bssn = reconstruct_cartoon_support(bssn, params)
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
def compute_cartoon_prescribed_rhs(
    wave,
    time,
    params,
    background_func,
    background_params,
):
    """Evaluate a prescribed-background Maxwell RHS on spherical support."""

    support_em = reconstruct_cartoon_wave_support(wave, params)
    support_bssn, support_bssn_rhs = background_func(
        time, support_em, background_params
    )
    support_rhs = compute_em_rhs(
        support_em, support_bssn, support_bssn_rhs, params
    )
    return project_cartoon_wave_rhs(support_rhs)


@partial(jax.jit, static_argnames=("background_func",))
def cartoon_prescribed_wave_rk4_step(
    wave,
    time,
    params,
    background_func,
    background_params,
):
    """Advance compact spherical Maxwell fields on prescribed geometry."""

    dt = params.dt
    wave = fill_cartoon_wave_ghosts(wave)
    k1 = compute_cartoon_prescribed_rhs(
        wave, time, params, background_func, background_params
    )

    midpoint = fill_cartoon_wave_ghosts(_add_scaled(wave, k1, 0.5 * dt))
    k2 = compute_cartoon_prescribed_rhs(
        midpoint,
        time + 0.5 * dt,
        params,
        background_func,
        background_params,
    )

    midpoint = fill_cartoon_wave_ghosts(_add_scaled(wave, k2, 0.5 * dt))
    k3 = compute_cartoon_prescribed_rhs(
        midpoint,
        time + 0.5 * dt,
        params,
        background_func,
        background_params,
    )

    endpoint = fill_cartoon_wave_ghosts(_add_scaled(wave, k3, dt))
    k4 = compute_cartoon_prescribed_rhs(
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
    return fill_cartoon_wave_ghosts(_add_scaled(wave, increment, dt))


def _prepare_coupled_stage(state):
    bssn = enforce_algebraic_constraints(state.bssn)
    return EinsteinMaxwellVariables(
        bssn=fill_cartoon_ghosts(bssn),
        em=fill_cartoon_wave_ghosts(state.em),
    )


@jax.jit
def compute_spherical_einstein_maxwell_rhs(
    state: EinsteinMaxwellVariables, params: BSSNParameters
) -> EinsteinMaxwellVariables:
    """Compute synchronized compact BSSN and spherical Maxwell RHS data."""

    support_bssn = reconstruct_cartoon_support(state.bssn, params)
    support_em = reconstruct_cartoon_wave_support(state.em, params)
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
        bssn=project_cartoon_rhs(support_bssn_rhs),
        em=project_cartoon_wave_rhs(support_em_rhs),
    )


@jax.jit
def spherical_einstein_maxwell_rk4_step(
    state: EinsteinMaxwellVariables, params: BSSNParameters
) -> EinsteinMaxwellVariables:
    """Advance compact spherical BSSN and Maxwell data through one RK4 step."""

    dt = params.dt
    state = _prepare_coupled_stage(state)
    k1 = compute_spherical_einstein_maxwell_rhs(state, params)

    midpoint = _prepare_coupled_stage(_add_scaled(state, k1, 0.5 * dt))
    k2 = compute_spherical_einstein_maxwell_rhs(midpoint, params)

    midpoint = _prepare_coupled_stage(_add_scaled(state, k2, 0.5 * dt))
    k3 = compute_spherical_einstein_maxwell_rhs(midpoint, params)

    endpoint = _prepare_coupled_stage(_add_scaled(state, k3, dt))
    k4 = compute_spherical_einstein_maxwell_rhs(endpoint, params)
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
    "spherical_einstein_maxwell_rk4_step",
    "cartoon_prescribed_wave_rk4_step",
    "compact_cartoon_wave",
    "compute_cartoon_constraint_divergences",
    "compute_spherical_einstein_maxwell_rhs",
    "compute_cartoon_prescribed_rhs",
    "expand_cartoon_wave_axis",
    "fill_cartoon_wave_ghosts",
    "project_cartoon_wave_rhs",
    "reconstruct_cartoon_wave_support",
    "validate_cartoon_wave_grid",
]
