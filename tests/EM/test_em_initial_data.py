import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.EM.second_order.initial_data import (
    conformal_vector_to_physical_covector,
    contract_conformal_electromagnetic_fields,
    electromagnetic_hamiltonian_residual,
    electromagnetic_hamiltonian_source,
    linearized_electromagnetic_hamiltonian_source,
    off_centered_toroidal_electric_seed,
    solve_electromagnetic_conformal_factor,
)


def test_literature_seed_uses_hl_normalization_and_cartesian_azimuthal_basis():
    amplitude = 0.08
    width = 1.2
    radial_center = 3.0
    grid = jnp.asarray([[[[2.0, 0.0, -0.5], [1.0, 2.0, 0.25]]]])

    conformal_electric = off_centered_toroidal_electric_seed(
        grid,
        amplitude=amplitude,
        width=width,
        radial_center=radial_center,
    )

    radius = jnp.sqrt(jnp.einsum("...i,...i->...", grid, grid))
    gaussian = jnp.exp(-((radius - radial_center) / width) ** 2)
    gaussian += jnp.exp(-((radius + radial_center) / width) ** 2)
    expected_phi = (
        -4.0
        * amplitude
        * gaussian
        / (jnp.sqrt(4.0 * jnp.pi) * width**2)
    )
    expected = expected_phi[None, ...] * jnp.stack(
        (-grid[..., 1], grid[..., 0], jnp.zeros_like(radius)), axis=0
    )

    np.testing.assert_allclose(conformal_electric, expected, rtol=2.0e-15)
    np.testing.assert_array_equal(conformal_electric[0, ..., 0], 0.0)
    np.testing.assert_array_equal(conformal_electric[2], 0.0)

    cylindrical_dot = (
        grid[..., 0] * conformal_electric[0]
        + grid[..., 1] * conformal_electric[1]
    )
    np.testing.assert_allclose(cylindrical_dot, 0.0, atol=2.0e-16)

    # E_HL=E_G/sqrt(4pi) must give the same physical energy density.
    psi = jnp.full(radius.shape, 1.35)
    electric_covector = conformal_vector_to_physical_covector(
        conformal_electric, psi
    )
    inverse_metric = (
        psi[None, None, ...] ** -4
        * jnp.eye(3)[:, :, None, None, None]
    )
    electric_up = jnp.einsum(
        "ij...,j...->i...", inverse_metric, electric_covector
    )
    hl_energy = 0.5 * jnp.einsum(
        "i...,i...->...", electric_covector, electric_up
    )

    gaussian_electric_up = (
        jnp.sqrt(4.0 * jnp.pi)
        * psi[None, ...] ** -6
        * conformal_electric
    )
    gaussian_metric = (
        psi[None, None, ...] ** 4
        * jnp.eye(3)[:, :, None, None, None]
    )
    gaussian_energy = jnp.einsum(
        "ij...,i...,j...->...",
        gaussian_metric,
        gaussian_electric_up,
        gaussian_electric_up,
    ) / (8.0 * jnp.pi)

    np.testing.assert_allclose(hl_energy, gaussian_energy, rtol=3.0e-15)


def test_electromagnetic_hamiltonian_source_linearization():
    psi = jnp.linspace(1.05, 1.35, 24).reshape((2, 3, 4))
    field_squared = jnp.linspace(0.01, 0.2, 24).reshape((2, 3, 4))
    delta_u = jnp.cos(jnp.arange(24)).reshape((2, 3, 4))
    epsilon = 1.0e-6

    finite_difference = (
        electromagnetic_hamiltonian_source(
            psi + epsilon * delta_u, field_squared
        )
        - electromagnetic_hamiltonian_source(
            psi - epsilon * delta_u, field_squared
        )
    ) / (2.0 * epsilon)
    expected = linearized_electromagnetic_hamiltonian_source(
        delta_u, psi, field_squared
    )

    np.testing.assert_allclose(
        finite_difference,
        expected,
        rtol=2.0e-9,
        atol=2.0e-10,
    )


def test_zero_field_returns_exact_flat_conformal_factor():
    field_squared = jnp.zeros((8, 9, 10), dtype=jnp.float64)
    psi, u, residual_history = solve_electromagnetic_conformal_factor(
        field_squared, dx=0.25
    )

    np.testing.assert_array_equal(psi, jnp.ones_like(psi))
    np.testing.assert_array_equal(u, jnp.zeros_like(u))
    assert residual_history == [0.0]


def _manufactured_hamiltonian_data(num_points):
    half_width = 1.0
    axis = jnp.linspace(-half_width, half_width, num_points)
    dx = float(axis[1] - axis[0])
    X, Y, Z = jnp.meshgrid(axis, axis, axis, indexing="ij")

    amplitude = 0.04
    wavenumber = jnp.pi / (2.0 * half_width)
    mode = (
        jnp.cos(wavenumber * X)
        * jnp.cos(wavenumber * Y)
        * jnp.cos(wavenumber * Z)
    )
    psi = 1.0 + amplitude * mode
    continuum_laplacian = -3.0 * wavenumber**2 * amplitude * mode
    field_squared = -continuum_laplacian * psi**3 / jnp.pi

    return psi, field_squared, dx


def test_hamiltonian_solve_is_second_order_for_manufactured_solution():
    errors = []
    residuals = []
    for num_points in (9, 17, 33):
        expected_psi, field_squared, dx = _manufactured_hamiltonian_data(
            num_points
        )
        psi, u, residual_history = solve_electromagnetic_conformal_factor(
            field_squared,
            dx,
            newton_tolerance=1.0e-11,
            cg_tolerance=1.0e-12,
        )
        residual = electromagnetic_hamiltonian_residual(
            u, field_squared, dx
        )

        interior_error = (
            psi[1:-1, 1:-1, 1:-1]
            - expected_psi[1:-1, 1:-1, 1:-1]
        )
        errors.append(
            float(jnp.sqrt(jnp.mean(interior_error**2)))
        )
        residuals.append(float(jnp.sqrt(jnp.mean(residual**2))))

        assert bool(jnp.all(jnp.isfinite(psi)))
        assert float(jnp.min(psi)) >= 1.0
        assert residual_history[-1] <= 1.0e-11
        for face in (
            u[0],
            u[-1],
            u[:, 0],
            u[:, -1],
            u[:, :, 0],
            u[:, :, -1],
        ):
            np.testing.assert_array_equal(face, jnp.zeros_like(face))

    orders = [
        math.log(errors[index] / errors[index + 1]) / math.log(2.0)
        for index in range(2)
    ]

    assert max(residuals) <= 1.0e-11
    assert min(orders) > 1.8


def test_toroidal_seed_produces_nonnegative_conformal_energy():
    axis = jnp.linspace(-4.0, 4.0, 9)
    X, Y, Z = jnp.meshgrid(axis, axis, axis, indexing="ij")
    grid = jnp.stack((X, Y, Z), axis=-1)
    conformal_electric = off_centered_toroidal_electric_seed(grid)
    conformal_magnetic = jnp.zeros_like(conformal_electric)

    field_squared = contract_conformal_electromagnetic_fields(
        conformal_electric, conformal_magnetic
    )

    assert bool(jnp.all(field_squared >= 0.0))
    assert float(jnp.max(field_squared)) > 0.0
    np.testing.assert_array_equal(field_squared[4, 4, :], 0.0)
