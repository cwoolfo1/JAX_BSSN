"""Fixed Kerr--Schild Schwarzschild data and an axial Maxwell mode."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

from JAX_BSSN.bssn import BSSNParameters, BSSNVariables

from JAX_BSSN.EM.derivatives import covariant_derivative_covector
from JAX_BSSN.EM.equations import compute_em_rhs
from JAX_BSSN.EM.geometry import LEVI_CIVITA_SYMBOL, compute_bssn_em_geometry
from JAX_BSSN.EM.variables import EMVariables


class SchwarzschildParameters(NamedTuple):
    """Parameters of a stationary Schwarzschild Kerr--Schild background."""

    mass: float = 1.0
    dx: float = 0.1
    x_min: float = 0.0
    y_min: float = 0.0
    z_min: float = 0.0


def grid_coordinates(shape, params: BSSNParameters, dtype=jnp.float64):
    """Return the cell coordinates encoded by ``BSSNParameters``."""

    x = params.x_min + params.dx * jnp.arange(shape[0], dtype=dtype)
    y = params.y_min + params.dx * jnp.arange(shape[1], dtype=dtype)
    z = params.z_min + params.dx * jnp.arange(shape[2], dtype=dtype)
    return jnp.meshgrid(x, y, z, indexing="ij")


def schwarzschild_background(
    time,
    wave: EMVariables,
    params: SchwarzschildParameters,
) -> tuple[BSSNVariables, BSSNVariables]:
    """Return stationary Schwarzschild data in Cartesian Kerr--Schild form.

    The BSSN sign convention is
    ``K_ij = -(partial_t gamma_ij - L_beta gamma_ij)/(2 alpha)``.
    Hence the stationary Kerr--Schild slice has ``K_ij = L_beta gamma_ij/(2
    alpha)``.
    """

    del time
    shape = wave.electric_field.shape[-3:]
    dtype = wave.electric_field.dtype
    x = params.x_min + params.dx * jnp.arange(shape[0], dtype=dtype)
    y = params.y_min + params.dx * jnp.arange(shape[1], dtype=dtype)
    z = params.z_min + params.dx * jnp.arange(shape[2], dtype=dtype)
    x, y, z = jnp.meshgrid(x, y, z, indexing="ij")

    position = jnp.stack((x, y, z), axis=0)
    radius = jnp.sqrt(jnp.einsum("i...,i...->...", position, position))
    radial_unit = position / radius

    mass = jnp.asarray(params.mass, dtype=dtype)
    q = 2.0 * mass / radius
    radial_outer = jnp.einsum("i...,j...->ij...", radial_unit, radial_unit)
    identity = jnp.eye(3, dtype=dtype)[:, :, None, None, None]

    physical_metric = identity + q * radial_outer
    W = (1.0 + q) ** (-1.0 / 6.0)
    conformal_metric = W**2 * physical_metric

    lapse = 1.0 / jnp.sqrt(1.0 + q)
    shift = q * radial_unit / (1.0 + q)

    K_prefactor = q / (radius * jnp.sqrt(1.0 + q))
    extrinsic_curvature = K_prefactor * (
        identity - (2.0 + 0.5 * q) * radial_outer
    )
    inverse_metric = identity - q * radial_outer / (1.0 + q)
    trace_K = jnp.einsum(
        "ij...,ij...->...", inverse_metric, extrinsic_curvature
    )
    traceless_K = W**2 * (
        extrinsic_curvature - physical_metric * trace_K / 3.0
    )

    # The wave-geometry reconstruction does not consume conformal_connection.
    vector_zero = jnp.zeros((3,) + shape, dtype=dtype)
    bssn = BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=W,
        traceless_K=traceless_K,
        trace_K=trace_K,
        conformal_connection=vector_zero,
        lapse=lapse,
        shift=shift,
    )
    bssn_rhs = BSSNVariables(*(jnp.zeros_like(field) for field in bssn))
    return bssn, bssn_rhs


def _regge_wheeler_rhs(radius, state, mass, omega, ell):
    """Radial equation for the Kerr--Schild amplitude chi(r)."""

    chi, chi_prime = state
    f = 1.0 - 2.0 * mass / radius
    f_prime = 2.0 * mass / radius**2
    angular_eigenvalue = ell * (ell + 1)

    first_derivative_coefficient = f_prime - 2.0j * omega * (1.0 - f)
    field_coefficient = (
        omega**2 * (2.0 - f)
        + 1.0j * omega * f_prime
        - angular_eigenvalue / radius**2
    )
    chi_second = -(
        first_derivative_coefficient * chi_prime + field_coefficient * chi
    ) / f
    return np.asarray((chi_prime, chi_second), dtype=np.complex128)


def solve_regge_wheeler_mode(
    radius,
    mass=1.0,
    omega=0.4,
    ell=1,
    rtol=2.0e-12,
    atol=2.0e-13,
):
    """Solve the ingoing electromagnetic Regge--Wheeler radial mode.

    Starting from ``psi = exp(-i omega t) psi(r)``, the Schwarzschild-time
    equation is transformed with

    ``chi = exp(i omega [r_star-r]) psi``.

    The resulting ODE is regular for the physical, purely ingoing horizon
    solution.  Its horizon regularity condition supplies ``d chi / dr``.
    The overall complex normalization is arbitrary and is set to one at the
    horizon.
    """

    radius = np.asarray(radius, dtype=float)
    horizon = 2.0 * mass
    angular_eigenvalue = ell * (ell + 1)
    f_prime_horizon = 1.0 / (2.0 * mass)
    chi_horizon = 1.0 + 0.0j
    chi_prime_horizon = (
        2.0 * omega**2
        + 1.0j * omega * f_prime_horizon
        - angular_eigenvalue / horizon**2
    ) * chi_horizon / (2.0j * omega)

    epsilon = 1.0e-6 * mass
    outer_start = horizon + epsilon
    outer_state = np.asarray(
        (chi_horizon + epsilon * chi_prime_horizon, chi_prime_horizon),
        dtype=np.complex128,
    )
    outer_radius = max(float(np.max(radius)), outer_start + epsilon)
    outer_solution = solve_ivp(
        _regge_wheeler_rhs,
        (outer_start, outer_radius),
        outer_state,
        args=(mass, omega, ell),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )

    inner_end = min(
        horizon - 2.0 * epsilon,
        max(float(np.min(radius)), 0.5 * mass),
    )
    inner_start = horizon - epsilon
    inner_state = np.asarray(
        (chi_horizon - epsilon * chi_prime_horizon, chi_prime_horizon),
        dtype=np.complex128,
    )
    inner_solution = solve_ivp(
        _regge_wheeler_rhs,
        (inner_start, inner_end),
        inner_state,
        args=(mass, omega, ell),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )
    if not outer_solution.success or not inner_solution.success:
        raise RuntimeError("Regge--Wheeler radial integration failed.")

    chi = np.empty(radius.shape, dtype=np.complex128)
    chi_prime = np.empty_like(chi)
    exterior = radius > horizon + epsilon
    interior = radius < horizon - epsilon
    horizon_band = ~(exterior | interior)

    if np.any(exterior):
        chi[exterior], chi_prime[exterior] = outer_solution.sol(radius[exterior])
    if np.any(interior):
        evaluation_radius = np.maximum(radius[interior], inner_end)
        chi[interior], chi_prime[interior] = inner_solution.sol(evaluation_radius)
        deep_interior = interior & (radius < inner_end)
        chi_prime[deep_interior] = 0.0
    chi[horizon_band] = chi_horizon + (
        radius[horizon_band] - horizon
    ) * chi_prime_horizon
    chi_prime[horizon_band] = chi_prime_horizon

    chi_second = np.zeros_like(chi)
    regular = (np.abs(radius - horizon) > epsilon) & (radius >= inner_end)
    if np.any(regular):
        regular_radius = radius[regular]
        f = 1.0 - 2.0 * mass / regular_radius
        f_prime = 2.0 * mass / regular_radius**2
        chi_second[regular] = -(
            (f_prime - 2.0j * omega * (1.0 - f)) * chi_prime[regular]
            + (
                omega**2 * (2.0 - f)
                + 1.0j * omega * f_prime
                - angular_eigenvalue / regular_radius**2
            )
            * chi[regular]
        ) / f

    return chi, chi_prime, chi_second


def _complex_fields_at_point(
    position,
    chi,
    chi_prime,
    chi_second,
    mass,
    omega,
):
    """Return the complex Eulerian E_i and B_i at one Cartesian point."""

    reference_radius = jnp.linalg.norm(position)

    def fields_at_position(local_position):
        radius = jnp.linalg.norm(local_position)
        radial_offset = radius - reference_radius
        local_chi = (
            chi + chi_prime * radial_offset + 0.5 * chi_second * radial_offset**2
        )
        local_chi_prime = chi_prime + chi_second * radial_offset

        x, y, z = local_position
        C = jnp.sqrt(3.0 / (4.0 * jnp.pi))
        radial_profile = C * local_chi / radius**2
        radial_profile_prime = C * (
            local_chi_prime / radius**2 - 2.0 * local_chi / radius**3
        )

        potential = jnp.asarray(
            (radial_profile * y, -radial_profile * x, 0.0j)
        )
        F_spatial = jnp.zeros((3, 3), dtype=potential.dtype)
        F_xy = -2.0 * radial_profile - (
            x**2 + y**2
        ) * radial_profile_prime / radius
        F_xz = -y * z * radial_profile_prime / radius
        F_yz = x * z * radial_profile_prime / radius
        F_spatial = F_spatial.at[0, 1].set(F_xy)
        F_spatial = F_spatial.at[1, 0].set(-F_xy)
        F_spatial = F_spatial.at[0, 2].set(F_xz)
        F_spatial = F_spatial.at[2, 0].set(-F_xz)
        F_spatial = F_spatial.at[1, 2].set(F_yz)
        F_spatial = F_spatial.at[2, 1].set(-F_yz)

        radial_unit = local_position / radius
        q = 2.0 * mass / radius
        lapse = 1.0 / jnp.sqrt(1.0 + q)
        shift = q * radial_unit / (1.0 + q)
        inverse_metric = jnp.eye(3) - q * jnp.outer(
            radial_unit, radial_unit
        ) / (1.0 + q)

        # E_i = F_{i mu} n^mu.  In particular F_ti is not E_i when beta != 0.
        electric = (
            1.0j * omega * potential
            - jnp.einsum("j,ij->i", shift, F_spatial)
        ) / lapse
        levi_civita = LEVI_CIVITA_SYMBOL * jnp.sqrt(1.0 + q)
        magnetic = 0.5 * jnp.einsum(
            "ijk,jm,kn,mn->i",
            levi_civita,
            inverse_metric,
            inverse_metric,
            F_spatial,
        )
        return jnp.stack((electric, magnetic), axis=0)

    fields = fields_at_position(position)
    field_derivatives = jax.jacfwd(fields_at_position)(position)

    radial_unit = position / reference_radius
    q = 2.0 * mass / reference_radius
    lapse = 1.0 / jnp.sqrt(1.0 + q)
    shift = q * radial_unit / (1.0 + q)

    def shift_at_position(local_position):
        radius = jnp.linalg.norm(local_position)
        local_q = 2.0 * mass / radius
        return local_q * local_position / radius / (1.0 + local_q)

    shift_derivative = jax.jacfwd(shift_at_position)(position)
    lie_derivatives = jnp.einsum(
        "j,aij->ai", shift, field_derivatives
    ) + jnp.einsum("aj,ji->ai", fields, shift_derivative)

    identity = jnp.eye(3)
    radial_outer = jnp.outer(radial_unit, radial_unit)
    inverse_metric = identity - q * radial_outer / (1.0 + q)
    K_covariant = q / (
        reference_radius * jnp.sqrt(1.0 + q)
    ) * (identity - (2.0 + 0.5 * q) * radial_outer)
    mixed_K = K_covariant @ inverse_metric
    field_dots = (
        -1.0j * omega * fields - lie_derivatives
    ) / lapse + jnp.einsum("ij,aj->ai", mixed_K, fields)
    return fields, field_dots


def schwarzschild_exact_template(
    shape,
    solver_params: BSSNParameters,
    params: SchwarzschildParameters,
    omega=0.4,
    ell=1,
    m=0,
) -> EMVariables:
    """Return complex amplitudes for the axial l=1, m=0 Maxwell mode."""

    if ell != 1 or m != 0:
        raise ValueError("The Cartesian axial harmonic currently implements l=1, m=0.")

    coordinates = grid_coordinates(shape, solver_params, dtype=jnp.float64)
    position = jnp.stack(coordinates, axis=-1).reshape((-1, 3))
    radius = np.linalg.norm(np.asarray(position), axis=1)
    chi, chi_prime, chi_second = solve_regge_wheeler_mode(
        radius,
        mass=float(params.mass),
        omega=omega,
        ell=ell,
    )

    fields, field_dots = jax.vmap(
        _complex_fields_at_point,
        in_axes=(0, 0, 0, 0, None, None),
    )(
        position,
        jnp.asarray(chi),
        jnp.asarray(chi_prime),
        jnp.asarray(chi_second),
        params.mass,
        omega,
    )
    fields = jnp.moveaxis(fields.reshape(shape + (2, 3)), (-2, -1), (0, 1))
    field_dots = jnp.moveaxis(
        field_dots.reshape(shape + (2, 3)), (-2, -1), (0, 1)
    )
    return EMVariables(
        electric_field=fields[0],
        electric_field_dot=field_dots[0],
        magnetic_field=fields[1],
        magnetic_field_dot=field_dots[1],
    )


def exact_wave_at_time(template: EMVariables, time, omega):
    """Take the real part of a complex monochromatic wave template."""

    phase = jnp.exp(-1.0j * omega * time)
    return jax.tree_util.tree_map(lambda field: jnp.real(phase * field), template)


def exact_wave_coordinate_rhs(template: EMVariables, time, omega):
    """Return the exact coordinate-time derivative of the wave state."""

    phase_derivative = -1.0j * omega * jnp.exp(-1.0j * omega * time)
    return jax.tree_util.tree_map(
        lambda field: jnp.real(phase_derivative * field), template
    )


def analytic_boundary_mask(
    shape,
    solver_params: BSSNParameters,
    excision_radius=1.5,
    boundary_points=2,
):
    """Mark the excision stencil buffer and analytic outer boundary shell."""

    x, y, z = grid_coordinates(shape, solver_params)
    radius = jnp.sqrt(x**2 + y**2 + z**2)
    inner = radius <= excision_radius + 2.0 * solver_params.dx

    i = jnp.arange(shape[0])[:, None, None]
    j = jnp.arange(shape[1])[None, :, None]
    k = jnp.arange(shape[2])[None, None, :]
    outer = (
        (i < boundary_points)
        | (i >= shape[0] - boundary_points)
        | (j < boundary_points)
        | (j >= shape[1] - boundary_points)
        | (k < boundary_points)
        | (k >= shape[2] - boundary_points)
    )
    return inner | outer


def _replace_masked_wave(wave, exact, mask):
    return jax.tree_util.tree_map(
        lambda numerical, boundary: jnp.where(
            mask[None, ...], boundary, numerical
        ),
        wave,
        exact,
    )


def _add_scaled(wave, rhs, scale):
    return jax.tree_util.tree_map(
        lambda field, derivative: field + scale * derivative,
        wave,
        rhs,
    )


def _schwarzschild_stage_rhs(
    wave,
    time,
    solver_params,
    background_params,
    exact_template,
    boundary_mask,
    omega,
):
    exact = exact_wave_at_time(exact_template, time, omega)
    wave = _replace_masked_wave(wave, exact, boundary_mask)
    bssn, bssn_rhs = schwarzschild_background(time, wave, background_params)
    rhs = compute_em_rhs(wave, bssn, bssn_rhs, solver_params)
    exact_rhs = exact_wave_coordinate_rhs(exact_template, time, omega)
    return wave, _replace_masked_wave(rhs, exact_rhs, boundary_mask)


@jax.jit
def schwarzschild_wave_rk4_step(
    wave,
    time,
    solver_params,
    background_params,
    exact_template,
    boundary_mask,
    omega,
):
    """Advance one RK4 step with analytic excision and outer boundary data."""

    dt = solver_params.dt
    wave, k1 = _schwarzschild_stage_rhs(
        wave,
        time,
        solver_params,
        background_params,
        exact_template,
        boundary_mask,
        omega,
    )
    midpoint, k2 = _schwarzschild_stage_rhs(
        _add_scaled(wave, k1, 0.5 * dt),
        time + 0.5 * dt,
        solver_params,
        background_params,
        exact_template,
        boundary_mask,
        omega,
    )
    del midpoint
    midpoint, k3 = _schwarzschild_stage_rhs(
        _add_scaled(wave, k2, 0.5 * dt),
        time + 0.5 * dt,
        solver_params,
        background_params,
        exact_template,
        boundary_mask,
        omega,
    )
    del midpoint
    endpoint, k4 = _schwarzschild_stage_rhs(
        _add_scaled(wave, k3, dt),
        time + dt,
        solver_params,
        background_params,
        exact_template,
        boundary_mask,
        omega,
    )
    del endpoint

    increment = jax.tree_util.tree_map(
        lambda rhs1, rhs2, rhs3, rhs4: (
            rhs1 + 2.0 * rhs2 + 2.0 * rhs3 + rhs4
        )
        / 6.0,
        k1,
        k2,
        k3,
        k4,
    )
    evolved = _add_scaled(wave, increment, dt)
    exact_endpoint = exact_wave_at_time(exact_template, time + dt, omega)
    return _replace_masked_wave(evolved, exact_endpoint, boundary_mask)


@jax.jit
def evolve_schwarzschild_wave_steps(
    wave,
    initial_time,
    solver_params,
    background_params,
    exact_template,
    boundary_mask,
    omega,
    num_steps,
):
    """Advance a fixed number of steps on the prescribed Schwarzschild metric."""

    def step(step_index, current_wave):
        time = initial_time + step_index * solver_params.dt
        return schwarzschild_wave_rk4_step(
            current_wave,
            time,
            solver_params,
            background_params,
            exact_template,
            boundary_mask,
            omega,
        )

    return jax.lax.fori_loop(0, num_steps, step, wave)


def maxwell_constraint_divergences(
    wave: EMVariables,
    time,
    solver_params: BSSNParameters,
    params: SchwarzschildParameters,
):
    """Return the solver's discrete curved divergences D_i E^i and D_i B^i."""

    bssn, bssn_rhs = schwarzschild_background(time, wave, params)
    geometry = compute_bssn_em_geometry(bssn, bssn_rhs, solver_params)

    def divergence(field):
        derivative = covariant_derivative_covector(
            field, geometry.christoffel, solver_params
        )
        return jnp.einsum(
            "ij...,ij...->...", geometry.inverse_metric, derivative
        )

    return divergence(wave.electric_field), divergence(wave.magnetic_field)
