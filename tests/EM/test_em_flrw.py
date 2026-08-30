import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn import BSSNParameters

from JAX_BSSN.EM.equations import curved_source_terms
from JAX_BSSN.EM.evolve import evolve_prescribed_em_steps
from JAX_BSSN.EM.flrw import FLRWParameters, flrw_background, flrw_exact_fields
from JAX_BSSN.EM.geometry import compute_bssn_em_geometry


def test_flrw_metric_matches_3p1_geometry():
    shape = (2, 2, 8)
    time = 0.7
    params = FLRWParameters(H=0.1, length_z=1.0)
    solver_params = BSSNParameters(dx=0.125, dt=0.01, nu=0.0)
    em = flrw_exact_fields(shape, time, solver_params, params)
    bssn, bssn_rhs = flrw_background(time, em, params)
    geometry = compute_bssn_em_geometry(bssn, bssn_rhs, solver_params)

    a = 1.0 + params.H * time
    identity = np.eye(3)[:, :, None, None, None]
    expected_metric = np.broadcast_to(identity * a**2, (3, 3) + shape)

    np.testing.assert_allclose(bssn.lapse, a, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(bssn.shift, 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(geometry.metric, expected_metric, rtol=1.0e-14)
    np.testing.assert_allclose(
        geometry.inverse_metric, expected_metric / a**4, rtol=1.0e-14
    )
    metric_last = jnp.moveaxis(geometry.metric, (0, 1), (-2, -1))
    sqrt_gamma = jnp.sqrt(jnp.linalg.det(metric_last))
    np.testing.assert_allclose(sqrt_gamma, a**3, rtol=1.0e-14)
    np.testing.assert_allclose(
        geometry.extrinsic_curvature,
        -params.H * np.broadcast_to(identity, (3, 3) + shape),
        rtol=1.0e-14,
    )
    np.testing.assert_allclose(
        geometry.normal_normal_ricci,
        3.0 * params.H**2 / a**4,
        rtol=1.0e-14,
    )


def test_exact_fields_have_covector_and_projected_dot_scaling():
    shape = (1, 1, 16)
    time = 0.4
    H = 0.1
    params = FLRWParameters(H=H, length_z=1.0)
    solver_params = BSSNParameters(dx=1.0 / shape[2], dt=0.01, nu=0.0)
    em = flrw_exact_fields(shape, time, solver_params, params)

    a = 1.0 + H * time
    z = np.arange(shape[2]) / shape[2]
    phase = 2.0 * np.pi * (z - time)
    expected_field = np.cos(phase) / a
    expected_dot = (
        2.0 * np.pi * np.sin(phase) / a**2
        - 2.0 * H * np.cos(phase) / a**3
    )

    np.testing.assert_allclose(em.electric_field[0, 0, 0], expected_field)
    np.testing.assert_allclose(em.magnetic_field[1, 0, 0], expected_field)
    np.testing.assert_allclose(em.electric_field_dot[0, 0, 0], expected_dot)
    np.testing.assert_allclose(em.magnetic_field_dot[1, 0, 0], expected_dot)
    np.testing.assert_allclose(em.electric_field[1:], 0.0, atol=0.0)
    np.testing.assert_allclose(
        em.magnetic_field[jnp.asarray([0, 2])], 0.0, atol=0.0
    )


def test_H_zero_reduces_to_minkowski_plane_wave():
    shape = (1, 1, 16)
    time = 0.23
    params = FLRWParameters(H=0.0, length_z=1.0)
    solver_params = BSSNParameters(dx=1.0 / shape[2], dt=0.01, nu=0.0)
    em = flrw_exact_fields(shape, time, solver_params, params)
    bssn, _ = flrw_background(time, em, params)

    z = np.arange(shape[2]) / shape[2]
    phase = 2.0 * np.pi * (z - time)
    np.testing.assert_allclose(em.electric_field[0, 0, 0], np.cos(phase))
    np.testing.assert_allclose(
        em.electric_field_dot[0, 0, 0], 2.0 * np.pi * np.sin(phase)
    )
    np.testing.assert_allclose(bssn.lapse, 1.0, atol=0.0)
    np.testing.assert_allclose(bssn.conformal_factor, 1.0, atol=0.0)
    np.testing.assert_allclose(bssn.trace_K, 0.0, atol=0.0)


def test_flrw_exact_field_satisfies_continuum_projected_wave_source():
    shape = (1, 1, 1)
    H = 0.1
    params = FLRWParameters(H=H, length_z=1.0)
    solver_params = BSSNParameters(dx=1.0, dt=0.01, nu=0.0)
    em = flrw_exact_fields(shape, 0.0, solver_params, params)
    bssn, bssn_rhs = flrw_background(0.0, em, params)
    geometry = compute_bssn_em_geometry(bssn, bssn_rhs, solver_params)
    electric_source, magnetic_source = curved_source_terms(
        em, bssn, geometry, solver_params
    )

    k = 2.0 * np.pi
    exact_field_dot_rhs = -k**2 + 6.0 * H**2
    exact_laplacian = -k**2
    mixed_K_field_dot = (-H) * (-2.0 * H)
    required_source = exact_field_dot_rhs - exact_laplacian + mixed_K_field_dot

    np.testing.assert_allclose(
        electric_source[0, 0, 0, 0], required_source, atol=1.0e-13
    )
    np.testing.assert_allclose(
        magnetic_source[1, 0, 0, 0], required_source, atol=1.0e-13
    )


def test_short_flrw_evolution_tracks_exact_solution():
    grid_size = 24
    shape = (1, 1, grid_size)
    dx = 1.0 / grid_size
    dt = 0.1 * dx
    num_steps = 12
    final_time = num_steps * dt
    solver_params = BSSNParameters(dx=dx, dt=dt, nu=0.0)
    params = FLRWParameters(H=0.1, length_z=1.0)

    initial = flrw_exact_fields(shape, 0.0, solver_params, params)
    evolved = evolve_prescribed_em_steps(
        initial,
        jnp.asarray(0.0),
        solver_params,
        flrw_background,
        params,
        num_steps,
    )
    exact = flrw_exact_fields(shape, final_time, solver_params, params)
    jax.block_until_ready(evolved)

    E_error = jnp.sqrt(
        jnp.mean((evolved.electric_field - exact.electric_field) ** 2)
    )
    B_error = jnp.sqrt(
        jnp.mean((evolved.magnetic_field - exact.magnetic_field) ** 2)
    )
    assert float(E_error) < 5.0e-4
    assert float(B_error) < 5.0e-4
