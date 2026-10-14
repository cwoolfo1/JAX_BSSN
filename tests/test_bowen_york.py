import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from JAX_BSSN.utilities.bowen_york_solver import (
    axisymmetric_cell_centered_grid,
    axisymmetric_laplacian,
    bowen_york_extrinsic_curvature,
    cell_centered_grid,
    solve_axisymmetric_conformal_factor,
    solve_conformal_factor,
)


def test_axisymmetric_grid_uses_positive_rho_and_symmetric_z_cell_centers():
    grid, dx = axisymmetric_cell_centered_grid(
        num_radial_points=8,
        radial_extent=4.0,
    )

    assert grid.shape == (8, 1, 16, 3)
    assert dx == 0.5
    np.testing.assert_allclose(grid[:, 0, 0, 0], (np.arange(8) + 0.5) * dx)
    np.testing.assert_allclose(grid[..., 1], 0.0)
    np.testing.assert_allclose(grid[0, 0, :, 2], -grid[0, 0, ::-1, 2])


def test_axisymmetric_laplacian_is_exact_for_quadratic_at_the_axis():
    dx = 0.2
    num_radial_points = 10
    num_z_points = 20
    rho = (jnp.arange(num_radial_points - 1) + 0.5) * dx
    z = -2.0 + (jnp.arange(1, num_z_points - 1) + 0.5) * dx
    R, Z = jnp.meshgrid(rho, z, indexing="ij")
    field = R**2 + Z**2

    result = axisymmetric_laplacian(field, dx)

    np.testing.assert_allclose(result[:-1, 1:-1], 6.0, atol=2.0e-13)


def test_bowen_york_curvature_is_symmetric_traceless_and_zero_at_zero_momentum():
    grid, _ = cell_centered_grid(num_points=8, half_width=4.0)
    positions = jnp.zeros((1, 3), dtype=grid.dtype)
    momentum = jnp.asarray([[0.2, -0.1, 0.5]], dtype=grid.dtype)

    curvature = bowen_york_extrinsic_curvature(momentum, positions, grid)

    np.testing.assert_allclose(curvature, jnp.swapaxes(curvature, -1, -2))
    np.testing.assert_allclose(
        jnp.trace(curvature, axis1=-2, axis2=-1),
        0.0,
        atol=2.0e-14,
    )
    zero_curvature = bowen_york_extrinsic_curvature(
        jnp.zeros_like(momentum),
        positions,
        grid,
    )
    np.testing.assert_array_equal(zero_curvature, jnp.zeros_like(zero_curvature))


def test_boosted_conformal_factor_converges_with_zero_boundary_correction():
    grid, dx = cell_centered_grid(num_points=10, half_width=5.0)
    psi, u, residual_history = solve_conformal_factor(
        masses=jnp.asarray([1.0]),
        positions=jnp.zeros((1, 3)),
        momenta=jnp.asarray([[0.0, 0.0, 0.2]]),
        grid=grid,
        dx=dx,
        newton_tolerance=1.0e-9,
        cg_tolerance=1.0e-11,
    )

    assert bool(jnp.all(jnp.isfinite(psi)))
    assert float(jnp.min(psi)) > 0.0
    assert residual_history[-1] <= 1.0e-9
    assert residual_history[-1] < residual_history[0]
    for face in (u[0], u[-1], u[:, 0], u[:, -1], u[:, :, 0], u[:, :, -1]):
        np.testing.assert_array_equal(face, jnp.zeros_like(face))


def test_axisymmetric_boosted_conformal_factor_converges_on_2d_grid():
    grid, dx = axisymmetric_cell_centered_grid(
        num_radial_points=12,
        radial_extent=5.0,
    )
    psi, u, residual_history = solve_axisymmetric_conformal_factor(
        masses=jnp.asarray([1.0]),
        positions=jnp.zeros((1, 3)),
        momenta=jnp.asarray([[0.0, 0.0, 0.2]]),
        grid=grid,
        dx=dx,
        newton_tolerance=1.0e-9,
        linear_tolerance=1.0e-11,
    )

    assert psi.shape == (12, 1, 24)
    assert u.shape == psi.shape
    assert bool(jnp.all(jnp.isfinite(psi)))
    assert float(jnp.min(psi)) > 0.0
    assert residual_history[-1] <= 1.0e-9
    assert residual_history[-1] < residual_history[0]
    np.testing.assert_array_equal(u[-1], jnp.zeros_like(u[-1]))
    np.testing.assert_array_equal(u[:, :, 0], jnp.zeros_like(u[:, :, 0]))
    np.testing.assert_array_equal(u[:, :, -1], jnp.zeros_like(u[:, :, -1]))


def test_axisymmetric_zero_momentum_returns_brill_lindquist_solution():
    grid, dx = axisymmetric_cell_centered_grid(8, 4.0)
    psi, u, residual_history = solve_axisymmetric_conformal_factor(
        masses=jnp.asarray([1.0]),
        positions=jnp.zeros((1, 3)),
        momenta=jnp.zeros((1, 3)),
        grid=grid,
        dx=dx,
    )

    radius = jnp.sqrt(jnp.sum(grid**2, axis=-1))
    np.testing.assert_array_equal(u, jnp.zeros_like(u))
    np.testing.assert_allclose(psi, 1.0 + 0.5 / radius)
    assert residual_history == [0.0]


@pytest.mark.parametrize(
    ("positions", "momenta", "message"),
    [
        ([[0.1, 0.0, 0.0]], [[0.0, 0.0, 0.2]], "z axis"),
        ([[0.0, 0.0, 0.0]], [[0.1, 0.0, 0.2]], "parallel to z"),
    ],
)
def test_axisymmetric_solver_rejects_nonaxisymmetric_punctures(
    positions,
    momenta,
    message,
):
    grid, dx = axisymmetric_cell_centered_grid(8, 4.0)

    with pytest.raises(ValueError, match=message):
        solve_axisymmetric_conformal_factor(
            masses=[1.0],
            positions=positions,
            momenta=momenta,
            grid=grid,
            dx=dx,
        )
