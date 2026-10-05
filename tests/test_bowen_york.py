import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.utilities.bowen_york_solver import (
    bowen_york_extrinsic_curvature,
    cell_centered_grid,
    solve_conformal_factor,
)


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
