import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn import BSSNParameters, BSSNVariables
from JAX_BSSN.cartoon.axisymmetry import axisymmetric_rk4_step
from JAX_BSSN.cartoon.spherical_symmetry import cartoon_rk4_step
from JAX_BSSN.evolution.boundaries import PERIODIC_BC, SOMMERFELD_BC

from JAX_BSSN.EM.cartoon.axisymmetry import (
    axisymmetric_einstein_maxwell_rk4_step,
    compact_axisymmetric_wave,
    compute_axisymmetric_constraint_divergences,
    compute_axisymmetric_einstein_maxwell_rhs,
    expand_axisymmetric_wave_plane,
    fill_axisymmetric_wave_ghosts,
    reconstruct_axisymmetric_wave_support,
    validate_axisymmetric_wave_grid,
)
from JAX_BSSN.EM.cartoon.spherical_symmetry import (
    compact_cartoon_wave,
    compute_cartoon_constraint_divergences,
    compute_spherical_einstein_maxwell_rhs,
    expand_cartoon_wave_axis,
    fill_cartoon_wave_ghosts,
    reconstruct_cartoon_wave_support,
    spherical_einstein_maxwell_rk4_step,
    validate_cartoon_wave_grid,
)
from JAX_BSSN.EM.evolve import compute_einstein_maxwell_rhs
from JAX_BSSN.EM.variables import EMVariables, EinsteinMaxwellVariables


def _flat_bssn(shape, dtype=jnp.float64):
    scalar = jnp.zeros(shape, dtype=dtype)
    vector = jnp.zeros((3,) + shape, dtype=dtype)
    identity = jnp.eye(3, dtype=dtype)[:, :, None, None, None]
    metric = jnp.broadcast_to(identity, (3, 3) + shape)
    return BSSNVariables(
        conformal_metric=metric,
        conformal_factor=jnp.ones(shape, dtype=dtype),
        traceless_K=jnp.zeros_like(metric),
        trace_K=scalar,
        conformal_connection=vector,
        lapse=jnp.ones(shape, dtype=dtype),
        shift=vector,
    )


def _zero_em(shape):
    vector = jnp.zeros((3,) + shape, dtype=jnp.float64)
    return EMVariables(vector, vector, vector, vector)


def _spherical_params(dx=0.2):
    return BSSNParameters(
        dx=dx,
        dt=0.02 * dx,
        nu=0.0,
        gauge=1,
        xl_bc=PERIODIC_BC,
        xr_bc=SOMMERFELD_BC,
        yl_bc=PERIODIC_BC,
        yr_bc=PERIODIC_BC,
        zl_bc=PERIODIC_BC,
        zr_bc=PERIODIC_BC,
        x_min=-3.5 * dx,
        y_min=-4.0 * dx,
        z_min=-4.0 * dx,
        mad_q=1.0,
    )


def _axisymmetric_params(num_z=17, dx=0.2, dt=None):
    return BSSNParameters(
        dx=dx,
        dt=0.02 * dx if dt is None else dt,
        nu=0.0,
        gauge=1,
        xl_bc=PERIODIC_BC,
        xr_bc=SOMMERFELD_BC,
        yl_bc=PERIODIC_BC,
        yr_bc=PERIODIC_BC,
        zl_bc=PERIODIC_BC,
        zr_bc=SOMMERFELD_BC,
        x_min=-3.5 * dx,
        y_min=-4.0 * dx,
        z_min=-(num_z - 1) * dx / 2.0,
        mad_q=1.0,
    )


def test_spherical_em_parity_round_trip_and_validation():
    num_radial_points = 8
    full_shape = (2 * num_radial_points, 1, 1)
    values = jnp.arange(3 * np.prod(full_shape), dtype=jnp.float64).reshape(
        (3,) + full_shape
    )
    full_em = EMVariables(values, 2.0 * values, -values, 0.5 * values)

    compact = compact_cartoon_wave(full_em)
    validate_cartoon_wave_grid(compact, _spherical_params())
    filled = fill_cartoon_wave_ghosts(compact)
    parity = np.asarray((-1.0, 1.0, 1.0))[:, None]
    positive = np.asarray(filled.electric_field[:, 4:8, 0, 0])
    np.testing.assert_array_equal(
        filled.electric_field[:, :4, 0, 0], positive[:, ::-1] * parity
    )

    expanded = expand_cartoon_wave_axis(compact)
    for original, result in zip(full_em, expanded):
        np.testing.assert_array_equal(
            result[:, num_radial_points:, 0, 0],
            original[:, num_radial_points:, 0, 0],
        )


def test_axisymmetric_em_parity_round_trip_preserves_azimuthal_field():
    num_rho, num_z = 8, 11
    full_shape = (2 * num_rho, 1, num_z)
    values = jnp.arange(3 * np.prod(full_shape), dtype=jnp.float64).reshape(
        (3,) + full_shape
    )
    full_em = EMVariables(values, 2.0 * values, -values, 0.5 * values)

    compact = compact_axisymmetric_wave(full_em)
    validate_axisymmetric_wave_grid(compact, _axisymmetric_params(num_z))
    filled = fill_axisymmetric_wave_ghosts(compact)
    parity = np.asarray((-1.0, -1.0, 1.0))[:, None, None]
    positive = np.asarray(filled.electric_field[:, 4:8, 0, :])
    np.testing.assert_array_equal(
        filled.electric_field[:, :4, 0, :],
        positive[:, ::-1, :] * parity,
    )

    expanded = expand_axisymmetric_wave_plane(compact)
    for original, result in zip(full_em, expanded):
        np.testing.assert_array_equal(
            result[:, num_rho:, 0, :], original[:, num_rho:, 0, :]
        )
    assert float(jnp.max(jnp.abs(expanded.electric_field[1]))) > 0.0


def test_spherical_reconstruction_recovers_smooth_radial_covector_and_jit():
    num_radial_points, dx = 16, 0.1
    params = _spherical_params(dx)
    radius = (jnp.arange(num_radial_points) + 0.5) * dx
    profile = radius * (1.0 + 0.05 * radius**2)
    compact_vector = jnp.zeros(
        (3, num_radial_points + 4, 1, 1), dtype=jnp.float64
    ).at[0, 4:, 0, 0].set(profile)
    em = EMVariables(
        compact_vector,
        2.0 * compact_vector,
        -compact_vector,
        0.5 * compact_vector,
    )

    support = jax.jit(
        lambda state: reconstruct_cartoon_wave_support(state, params)
    )(em)
    x = (jnp.arange(num_radial_points + 4) - 3.5) * dx
    y = (jnp.arange(9) - 4.0) * dx
    X, Y, Z = jnp.meshgrid(x, y, y, indexing="ij")
    radius_support = jnp.sqrt(X**2 + Y**2 + Z**2)
    expected = jnp.stack((X, Y, Z)) * (1.0 + 0.05 * radius_support**2)
    np.testing.assert_allclose(
        support.electric_field[:, :-4], expected[:, :-4], atol=3.0e-12
    )
    assert bool(jnp.all(jnp.isfinite(support.electric_field)))


def test_axisymmetric_reconstruction_rotates_azimuthal_and_axial_components():
    num_rho, num_z, dx = 16, 17, 0.1
    params = _axisymmetric_params(num_z, dx)
    rho = (jnp.arange(num_rho) + 0.5) * dx
    z = params.z_min + dx * jnp.arange(num_z)
    R, Z = jnp.meshgrid(rho, z, indexing="ij")
    reference = jnp.stack((R, 0.3 * R, Z * (1.0 + 0.05 * R**2)))
    compact_vector = jnp.zeros(
        (3, num_rho + 4, 1, num_z), dtype=jnp.float64
    ).at[:, 4:, 0, :].set(reference)
    em = EMVariables(
        compact_vector,
        2.0 * compact_vector,
        -compact_vector,
        0.5 * compact_vector,
    )

    support = jax.jit(
        lambda state: reconstruct_axisymmetric_wave_support(state, params)
    )(em)
    x = (jnp.arange(num_rho + 4) - 3.5) * dx
    y = (jnp.arange(9) - 4.0) * dx
    X, Y, Zs = jnp.meshgrid(x, y, z, indexing="ij")
    radius = jnp.sqrt(X**2 + Y**2)
    expected = jnp.stack(
        (
            X - 0.3 * Y,
            Y + 0.3 * X,
            Zs * (1.0 + 0.05 * radius**2),
        )
    )
    np.testing.assert_allclose(
        support.electric_field[:, :-4], expected[:, :-4], atol=3.0e-12
    )
    assert bool(jnp.all(jnp.isfinite(support.electric_field)))


def test_spherical_coupled_rhs_matches_cartesian_support_rhs():
    num_radial_points, dx = 16, 0.1
    params = _spherical_params(dx)
    radius = (jnp.arange(num_radial_points) + 0.5) * dx
    profiles = (radius, radius**3, -2.0 * radius, 0.5 * radius**3)
    fields = []
    for profile in profiles:
        field = jnp.zeros((3, num_radial_points + 4, 1, 1))
        fields.append(field.at[0, 4:, 0, 0].set(profile))
    compact = fill_cartoon_wave_ghosts(EMVariables(*fields))

    compact_state = EinsteinMaxwellVariables(
        bssn=_flat_bssn(compact.electric_field.shape[-3:]),
        em=compact,
    )
    compact_rhs = compute_spherical_einstein_maxwell_rhs(
        compact_state, params
    )
    x = (jnp.arange(num_radial_points + 4) - 3.5) * dx
    y = (jnp.arange(9) - 4.0) * dx
    X, Y, Z = jnp.meshgrid(x, y, y, indexing="ij")
    radius_squared = X**2 + Y**2 + Z**2
    position = jnp.stack((X, Y, Z))
    full = EMVariables(
        position,
        position * radius_squared,
        -2.0 * position,
        0.5 * position * radius_squared,
    )
    full_state = EinsteinMaxwellVariables(
        bssn=_flat_bssn(full.electric_field.shape[-3:]),
        em=full,
    )
    full_rhs = compute_einstein_maxwell_rhs(full_state, params)
    jax.block_until_ready((compact_rhs, full_rhs))

    for compact_field, full_field in zip(compact_rhs.em, full_rhs.em):
        np.testing.assert_allclose(
            compact_field[:, 4:-4, 0, 0],
            full_field[:, 4:-4, 4, 4],
            atol=2.0e-10,
            rtol=2.0e-10,
        )


def test_axisymmetric_coupled_rhs_matches_cartesian_support_rhs():
    num_rho, num_z, dx = 16, 17, 0.1
    params = _axisymmetric_params(num_z, dx)
    rho = (jnp.arange(num_rho) + 0.5) * dx
    z = params.z_min + dx * jnp.arange(num_z)
    R, Z = jnp.meshgrid(rho, z, indexing="ij")
    reference = jnp.stack((R, 0.2 * R, Z))
    fields = []
    for scale in (1.0, 0.5, -2.0, 0.25):
        field = jnp.zeros((3, num_rho + 4, 1, num_z))
        fields.append(field.at[:, 4:, 0, :].set(scale * reference))
    compact = fill_axisymmetric_wave_ghosts(EMVariables(*fields))

    compact_state = EinsteinMaxwellVariables(
        bssn=_flat_bssn(compact.electric_field.shape[-3:]),
        em=compact,
    )
    compact_rhs = compute_axisymmetric_einstein_maxwell_rhs(
        compact_state, params
    )
    x = (jnp.arange(num_rho + 4) - 3.5) * dx
    y = (jnp.arange(9) - 4.0) * dx
    X, Y, Zs = jnp.meshgrid(x, y, z, indexing="ij")
    full_vector = jnp.stack((X - 0.2 * Y, Y + 0.2 * X, Zs))
    full = EMVariables(
        full_vector,
        0.5 * full_vector,
        -2.0 * full_vector,
        0.25 * full_vector,
    )
    full_state = EinsteinMaxwellVariables(
        bssn=_flat_bssn(full.electric_field.shape[-3:]),
        em=full,
    )
    full_rhs = compute_einstein_maxwell_rhs(full_state, params)
    jax.block_until_ready((compact_rhs, full_rhs))

    for compact_field, full_field in zip(compact_rhs.em, full_rhs.em):
        np.testing.assert_allclose(
            compact_field[:, 4:-4, 0, 2:-2],
            full_field[:, 4:-4, 4, 2:-2],
            atol=2.0e-10,
            rtol=2.0e-10,
        )


def test_spherical_coupled_step_matches_bssn_cartoon_for_zero_maxwell():
    params = _spherical_params(0.2)
    bssn = _flat_bssn((12, 1, 1))
    state = EinsteinMaxwellVariables(bssn=bssn, em=_zero_em((12, 1, 1)))

    coupled = spherical_einstein_maxwell_rk4_step(state, params)
    reference = cartoon_rk4_step(bssn, params)
    jax.block_until_ready((coupled, reference))
    for result, expected in zip(coupled.bssn, reference):
        np.testing.assert_allclose(result, expected, atol=2.0e-13)
    for field in coupled.em:
        np.testing.assert_allclose(field, 0.0, atol=2.0e-13)
    div_E, div_B = compute_cartoon_constraint_divergences(
        coupled.em, coupled.bssn, params
    )
    np.testing.assert_allclose(div_E, 0.0, atol=2.0e-13)
    np.testing.assert_allclose(div_B, 0.0, atol=2.0e-13)


def test_axisymmetric_coupled_step_matches_bssn_cartoon_for_zero_maxwell():
    params = _axisymmetric_params(11, 0.2)
    bssn = _flat_bssn((12, 1, 11))
    state = EinsteinMaxwellVariables(bssn=bssn, em=_zero_em((12, 1, 11)))

    coupled = axisymmetric_einstein_maxwell_rk4_step(state, params)
    reference = axisymmetric_rk4_step(bssn, params)
    jax.block_until_ready((coupled, reference))
    for result, expected in zip(coupled.bssn, reference):
        np.testing.assert_allclose(result, expected, atol=2.0e-13)
    for field in coupled.em:
        np.testing.assert_allclose(field, 0.0, atol=2.0e-13)
    div_E, div_B = compute_axisymmetric_constraint_divergences(
        coupled.em, coupled.bssn, params
    )
    np.testing.assert_allclose(div_E, 0.0, atol=2.0e-13)
    np.testing.assert_allclose(div_B, 0.0, atol=2.0e-13)
