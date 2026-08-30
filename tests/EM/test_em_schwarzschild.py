import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.evolution.boundaries import SOMMERFELD_BC

from JAX_BSSN.EM.schwarzschild import (
    SchwarzschildParameters,
    analytic_boundary_mask,
    evolve_schwarzschild_wave_steps,
    exact_wave_at_time,
    grid_coordinates,
    maxwell_constraint_divergences,
    schwarzschild_background,
    schwarzschild_exact_template,
    solve_regge_wheeler_mode,
)


def schwarzschild_test_parameters(grid_size=12):
    dx = 24.0 / grid_size
    first_coordinate = -12.0 + 0.5 * dx
    solver_params = BSSNParameters(
        dx=dx,
        dt=0.1 * dx,
        nu=0.0,
        xl_bc=SOMMERFELD_BC,
        xr_bc=SOMMERFELD_BC,
        yl_bc=SOMMERFELD_BC,
        yr_bc=SOMMERFELD_BC,
        zl_bc=SOMMERFELD_BC,
        zr_bc=SOMMERFELD_BC,
        x_min=first_coordinate,
        y_min=first_coordinate,
        z_min=first_coordinate,
    )
    background_params = SchwarzschildParameters(
        mass=1.0,
        dx=dx,
        x_min=first_coordinate,
        y_min=first_coordinate,
        z_min=first_coordinate,
    )
    return solver_params, background_params


def test_regge_wheeler_solution_obeys_ingoing_horizon_regularity():
    mass = 1.0
    omega = 0.4
    ell = 1
    horizon = 2.0 * mass
    radius = np.asarray((horizon, 2.5, 5.0, 10.0))
    chi, chi_prime, _ = solve_regge_wheeler_mode(
        radius, mass=mass, omega=omega, ell=ell
    )

    expected_horizon_derivative = (
        2.0 * omega**2
        + 1.0j * omega / (2.0 * mass)
        - ell * (ell + 1) / horizon**2
    ) * chi[0] / (2.0j * omega)

    np.testing.assert_allclose(chi[0], 1.0, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        chi_prime[0], expected_horizon_derivative, rtol=0.0, atol=1.0e-14
    )
    assert np.all(np.isfinite(chi))
    assert np.all(np.isfinite(chi_prime))


def test_kerr_schild_background_matches_metric_shift_and_extrinsic_curvature():
    solver_params, background_params = schwarzschild_test_parameters(8)
    shape = (8, 8, 8)
    template = schwarzschild_exact_template(
        shape, solver_params, background_params
    )
    em = exact_wave_at_time(template, 0.0, 0.4)
    bssn, bssn_rhs = schwarzschild_background(0.0, em, background_params)

    x, y, z = grid_coordinates(shape, solver_params)
    position = jnp.stack((x, y, z))
    radius = jnp.sqrt(jnp.einsum("i...,i...->...", position, position))
    radial_unit = position / radius
    q = 2.0 / radius
    radial_outer = jnp.einsum("i...,j...->ij...", radial_unit, radial_unit)
    identity = jnp.eye(3)[:, :, None, None, None]
    expected_metric = identity + q * radial_outer
    expected_shift = q * radial_unit / (1.0 + q)
    expected_K = q / (radius * jnp.sqrt(1.0 + q)) * (
        identity - (2.0 + 0.5 * q) * radial_outer
    )

    physical_metric = bssn.conformal_metric / bssn.conformal_factor**2
    inverse_metric = identity - q * radial_outer / (1.0 + q)
    reconstructed_K = bssn.traceless_K / bssn.conformal_factor**2
    reconstructed_K += physical_metric * bssn.trace_K / 3.0

    np.testing.assert_allclose(physical_metric, expected_metric, rtol=1.0e-14)
    np.testing.assert_allclose(bssn.shift, expected_shift, rtol=1.0e-14)
    np.testing.assert_allclose(reconstructed_K, expected_K, rtol=1.0e-14)
    np.testing.assert_allclose(
        bssn.trace_K,
        jnp.einsum("ij...,ij...->...", inverse_metric, expected_K),
        rtol=1.0e-14,
    )
    for field in bssn_rhs:
        np.testing.assert_allclose(field, 0.0, rtol=0.0, atol=0.0)

    point = jnp.asarray((4.0, 1.0, -2.0))

    def metric_at_position(local_position):
        local_radius = jnp.linalg.norm(local_position)
        local_radial_unit = local_position / local_radius
        local_q = 2.0 / local_radius
        return jnp.eye(3) + local_q * jnp.outer(
            local_radial_unit, local_radial_unit
        )

    def shift_at_position(local_position):
        local_radius = jnp.linalg.norm(local_position)
        local_q = 2.0 / local_radius
        return local_q * local_position / local_radius / (1.0 + local_q)

    point_metric = metric_at_position(point)
    point_shift = shift_at_position(point)
    metric_derivative = jax.jacfwd(metric_at_position)(point)
    shift_derivative = jax.jacfwd(shift_at_position)(point)
    metric_lie = jnp.einsum("k,ijk->ij", point_shift, metric_derivative)
    metric_lie += jnp.einsum("kj,ki->ij", point_metric, shift_derivative)
    metric_lie += jnp.einsum("ik,kj->ij", point_metric, shift_derivative)

    point_radius = jnp.linalg.norm(point)
    point_radial_unit = point / point_radius
    point_q = 2.0 / point_radius
    stationary_K = metric_lie * jnp.sqrt(1.0 + point_q) / 2.0
    formula_K = point_q / (point_radius * jnp.sqrt(1.0 + point_q)) * (
        jnp.eye(3)
        - (2.0 + 0.5 * point_q)
        * jnp.outer(point_radial_unit, point_radial_unit)
    )
    np.testing.assert_allclose(stationary_K, formula_K, rtol=1.0e-14)


def test_short_schwarzschild_mode_evolution_tracks_exact_fields():
    grid_size = 12
    shape = (grid_size, grid_size, grid_size)
    solver_params, background_params = schwarzschild_test_parameters(grid_size)
    template = schwarzschild_exact_template(
        shape, solver_params, background_params
    )
    initial = exact_wave_at_time(template, 0.0, 0.4)
    boundary_mask = analytic_boundary_mask(shape, solver_params)
    evolved = evolve_schwarzschild_wave_steps(
        initial,
        jnp.asarray(0.0),
        solver_params,
        background_params,
        template,
        boundary_mask,
        0.4,
        1,
    )
    exact = exact_wave_at_time(template, solver_params.dt, 0.4)
    jax.block_until_ready(evolved)

    x, y, z = grid_coordinates(shape, solver_params)
    radius = jnp.sqrt(x**2 + y**2 + z**2)
    validation_mask = (radius >= 5.0) & (radius <= 9.0) & ~boundary_mask

    def relative_l2(numerical, reference):
        numerator = jnp.sum(
            jnp.where(
                validation_mask[None, ...],
                (numerical - reference) ** 2,
                0.0,
            )
        )
        denominator = jnp.sum(
            jnp.where(validation_mask[None, ...], reference**2, 0.0)
        )
        return float(jnp.sqrt(numerator / denominator))

    assert relative_l2(evolved.electric_field, exact.electric_field) < 5.0e-3
    assert relative_l2(evolved.magnetic_field, exact.magnetic_field) < 7.0e-3

    div_E, div_B = maxwell_constraint_divergences(
        evolved, solver_params.dt, solver_params, background_params
    )
    assert float(jnp.max(jnp.where(validation_mask, jnp.abs(div_E), 0.0))) < 5.0e-4
    assert float(jnp.max(jnp.where(validation_mask, jnp.abs(div_B), 0.0))) < 3.0e-3
