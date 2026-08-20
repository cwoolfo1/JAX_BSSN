import jax
import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn import BSSNParameters, BSSNVariables
from JAX_BSSN.initialization import gauge_wave_analytic_state
from JAX_BSSN.refinement import (FMRPatchSpec, fill_fine_ghosts,
    fine_active_shape, fine_active_slice, fine_active_view, fine_coordinates,
    fmr_rk4_step, prolongate_to_fine, restrict_to_coarse)

jax.config.update("jax_enable_x64", True)


SPEC = FMRPatchSpec((4, 4, 4), (9, 9, 9))


def test_geometry_is_vertex_centered_with_four_guards():
    assert fine_active_shape(SPEC) == (11, 11, 11)
    X, Y, Z = fine_coordinates(SPEC, 0.1, (-0.5, -0.5, -0.5))
    active = fine_active_slice(SPEC)
    np.testing.assert_allclose(X[active][:, 0, 0], -0.1 + np.arange(11) * 0.05)
    np.testing.assert_allclose(Y[active][0, :, 0], -0.1 + np.arange(11) * 0.05)
    np.testing.assert_allclose(Z[active][0, 0, :], -0.1 + np.arange(11) * 0.05)


def test_quintic_prolongation_polynomials_and_leading_axes():
    n = 16
    x = jnp.arange(n, dtype=jnp.float64)
    # Patch and guards stay away from periodic wrapping for this exactness test.
    spec = FMRPatchSpec((5, 5, 5), (8, 8, 8))
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    scalar = X**5 + 2*Y**4 - Z**3 + X*Y
    got = prolongate_to_fine(scalar, spec)
    q = jnp.arange(3, 10.5, 0.5)
    qX, qY, qZ = jnp.meshgrid(q, q, q, indexing="ij")
    expected = qX**5 + 2*qY**4 - qZ**3 + qX*qY
    np.testing.assert_allclose(got, expected, rtol=2e-13, atol=2e-10)
    np.testing.assert_allclose(prolongate_to_fine(jnp.stack([scalar, 2*scalar]), spec)[1], 2*expected)
    tensor = jnp.stack([jnp.stack([scalar, 3*scalar])])
    assert prolongate_to_fine(tensor, spec).shape == (1, 2) + expected.shape


def _variables(field):
    return BSSNVariables(field, field + 1, field + 2, field + 3,
                         field + 4, field + 5, field + 6)


def test_ghost_fill_preserves_active_and_fills_faces_edges_corners():
    n = 16
    X, Y, Z = jnp.meshgrid(jnp.arange(n), jnp.arange(n), jnp.arange(n), indexing="ij")
    coarse = _variables((X + 2*Y + 3*Z).astype(jnp.float64))
    shape = tuple(x + 8 for x in fine_active_shape(SPEC))
    fine = _variables(jnp.full(shape, -999.0))
    filled = fill_fine_ghosts(coarse, fine, SPEC)
    expected = prolongate_to_fine(coarse[0], SPEC)
    mask = np.ones(shape, dtype=bool)
    mask[fine_active_slice(SPEC)] = False
    np.testing.assert_allclose(np.asarray(filled[0])[mask], np.asarray(expected)[mask])
    np.testing.assert_array_equal(np.asarray(filled[0][fine_active_slice(SPEC)]), -999.0)


def test_restriction_injects_all_variable_shapes_exactly():
    coarse_shape = (16, 16, 16)
    active_shape = fine_active_shape(SPEC)
    padded = tuple(n + 8 for n in active_shape)
    base = jnp.arange(np.prod(padded), dtype=jnp.float64).reshape(padded)
    coarse = _variables(jnp.zeros(coarse_shape))
    fine = _variables(base)
    result = restrict_to_coarse(coarse, fine, SPEC)
    expected = base[fine_active_slice(SPEC)][::2, ::2, ::2]
    covered = tuple(slice(l, h + 1) for l, h in zip(SPEC.coarse_lo, SPEC.coarse_hi))
    for offset, field in enumerate(result):
        np.testing.assert_array_equal(field[covered], expected + offset)


def test_stage_synchronous_gauge_wave_step_is_finite_and_accurate():
    n, dx = 10, 0.1
    x = -0.5 + jnp.arange(n) * dx
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    spec = FMRPatchSpec((3, 3, 3), (6, 6, 6))
    FX, FY, FZ = fine_coordinates(spec, dx, (-0.5, -0.5, -0.5))
    coarse = gauge_wave_analytic_state(X, Y, Z)
    fine = gauge_wave_analytic_state(FX, FY, FZ)
    dt = 0.005
    coarse_params = BSSNParameters(eta=0, kappa=0, g=0, nu=0, dx=dx,
        dt=dt, zero_shift=1, gauge=0, x_min=-0.5, y_min=-0.5, z_min=-0.5)
    fine_params = coarse_params._replace(dx=dx/2,
        x_min=float(FX[0, 0, 0]), y_min=float(FY[0, 0, 0]), z_min=float(FZ[0, 0, 0]))
    coarse, fine = fmr_rk4_step(coarse, fine, coarse_params, fine_params, spec)
    assert all(bool(jnp.all(jnp.isfinite(field))) for field in coarse + fine)
    exact = gauge_wave_analytic_state(FX, FY, FZ, dt)
    error = fine_active_view(fine.lapse - exact.lapse, spec)
    assert float(jnp.sqrt(jnp.mean(error**2))) < 2e-7
    determinant = jnp.linalg.det(jnp.moveaxis(fine.conformal_metric, (0, 1), (-2, -1)))
    np.testing.assert_allclose(fine_active_view(determinant, spec), 1.0, atol=2e-12)
