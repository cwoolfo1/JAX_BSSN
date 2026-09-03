import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn import BSSNParameters, BSSNVariables
from JAX_BSSN.evolution.time_evolve import rk4_step

from JAX_BSSN.EM.first_order import (
    DensitizedMaxwellState,
    FirstOrderEinsteinMaxwellState,
    bootstrap_densitized_maxwell_state,
    common_densitized_fields,
    compute_bssn_yee_geometry,
    compute_covariant_E,
    compute_covariant_H,
    compute_densitized_electromagnetic_energy_momentum,
    curl_E_to_densitized_B,
    curl_H_to_densitized_D,
    densitized_displacement_divergence,
    densitized_magnetic_divergence,
    densitized_maxwell_rhs,
    first_order_einstein_maxwell_step,
    initialize_first_order_einstein_maxwell_state,
    physical_fields_at_centers,
)
from JAX_BSSN.EM.first_order.cartoon.axisymmetry import (
    axisymmetric_densitized_constraint_divergences,
    axisymmetric_first_order_einstein_maxwell_step,
    expand_axisymmetric_densitized_state,
    fill_axisymmetric_densitized_ghosts,
    initialize_axisymmetric_first_order_state,
)
from JAX_BSSN.EM.first_order.staggering import (
    DISPLACEMENT_FIELD_LOCATIONS,
)
from JAX_BSSN.EM.second_order.energy_momentum import (
    compute_electromagnetic_energy_momentum,
)
from tests.EM.em_helpers import flat_bssn_variables


def _constant_bssn(shape, W, conformal_metric, lapse=1.0, shift=None):
    dtype = jnp.float64
    scalar = jnp.ones(shape, dtype=dtype)
    metric = jnp.asarray(conformal_metric, dtype=dtype)
    metric = jnp.broadcast_to(metric[:, :, None, None, None], (3, 3) + shape)
    if shift is None:
        shift = (0.0, 0.0, 0.0)
    shift_array = jnp.asarray(shift, dtype=dtype)[:, None, None, None]
    shift_array = jnp.broadcast_to(shift_array, (3,) + shape)
    return BSSNVariables(
        conformal_metric=metric,
        conformal_factor=W * scalar,
        traceless_K=jnp.zeros_like(metric),
        trace_K=jnp.zeros(shape, dtype=dtype),
        conformal_connection=jnp.zeros((3,) + shape, dtype=dtype),
        lapse=lapse * scalar,
        shift=shift_array,
    )


def test_density_factor_is_W_minus_three_not_metric_determinant():
    shape = (8, 6, 4)
    params = BSSNParameters(dx=0.2, dt=0.01, nu=0.0)
    # This deliberately has det(tilde gamma) != 1.  The volume factor must
    # nevertheless be the analytic BSSN factor W^-3.
    bssn = _constant_bssn(shape, 0.8, 2.0 * jnp.eye(3))
    physical_D = jnp.ones((3,) + shape)
    physical_B = 2.0 * jnp.ones((3,) + shape)
    geometry = compute_bssn_yee_geometry(bssn, params)

    for metric in geometry.displacement + geometry.magnetic:
        np.testing.assert_allclose(metric.W**-3, 0.8**-3)

    density_D = physical_D * 0.8**-3
    density_B = physical_B * 0.8**-3
    recovered_D, recovered_B = physical_fields_at_centers(
        density_D, density_B, bssn, params
    )
    np.testing.assert_allclose(recovered_D, physical_D)
    np.testing.assert_allclose(recovered_B, physical_B)


def test_manufactured_constitutive_relations_with_shift_and_offdiagonal_metric():
    shape = (7, 5, 3)
    params = BSSNParameters(dx=0.3, dt=0.02, nu=0.0)
    conformal_metric = jnp.asarray(
        [[1.2, 0.15, -0.08], [0.15, 0.9, 0.11], [-0.08, 0.11, 1.1]]
    )
    bssn = _constant_bssn(
        shape, 0.75, conformal_metric, lapse=0.83, shift=(0.12, -0.07, 0.04)
    )
    D_density_values = jnp.asarray((0.3, -0.2, 0.1))
    B_density_values = jnp.asarray((-0.15, 0.05, 0.25))
    D_density = jnp.broadcast_to(
        D_density_values[:, None, None, None], (3,) + shape
    )
    B_density = jnp.broadcast_to(
        B_density_values[:, None, None, None], (3,) + shape
    )
    geometry = compute_bssn_yee_geometry(bssn, params)
    E = compute_covariant_E(D_density, B_density, geometry, params)
    H = compute_covariant_H(D_density, B_density, geometry, params)

    W = 0.75
    D_up = W**3 * D_density_values
    B_up = W**3 * B_density_values
    metric = W**-2 * conformal_metric
    D_down = metric @ D_up
    B_down = metric @ B_up
    shift = jnp.asarray((0.12, -0.07, 0.04))
    expected_E = 0.83 * D_down + W**-3 * jnp.cross(shift, B_up)
    expected_H = 0.83 * B_down - W**-3 * jnp.cross(shift, D_up)
    expected_E = jnp.broadcast_to(expected_E[:, None, None, None], E.shape)
    expected_H = jnp.broadcast_to(expected_H[:, None, None, None], H.shape)
    np.testing.assert_allclose(E, expected_E)
    np.testing.assert_allclose(H, expected_H)


def test_yee_curls_are_second_order_and_divergence_of_curl_is_roundoff():
    errors = []
    for num_points in (32, 64, 128):
        dx = 2.0 * math.pi / num_points
        params = BSSNParameters(dx=dx, dt=0.1 * dx, nu=0.0)
        y_center = dx * (jnp.arange(num_points) + 0.5)
        electric = jnp.zeros((3, num_points, num_points, 1))
        electric = electric.at[2].set(
            jnp.broadcast_to(jnp.sin(y_center)[None, :, None], electric[2].shape)
        )
        curl = curl_E_to_densitized_B(electric, params)
        y_vertex = dx * jnp.arange(num_points)
        expected = jnp.cos(y_vertex)[None, :, None]
        errors.append(float(jnp.sqrt(jnp.mean((curl[0] - expected) ** 2))))
        np.testing.assert_allclose(
            densitized_magnetic_divergence(curl, params), 0.0, atol=2.0e-13
        )

        magnetic_covector = jnp.roll(electric, 1, axis=0)
        displacement_curl = curl_H_to_densitized_D(magnetic_covector, params)
        np.testing.assert_allclose(
            densitized_displacement_divergence(displacement_curl, params),
            0.0,
            atol=2.0e-13,
        )

    first_order = math.log2(errors[0] / errors[1])
    second_order = math.log2(errors[1] / errors[2])
    assert first_order > 1.99
    assert second_order > 1.99


def test_bootstrap_preserves_supplied_common_time_fields_exactly():
    shape = (3, 8, 4, 2)
    D = jnp.arange(np.prod(shape), dtype=jnp.float64).reshape(shape)
    B = -0.5 * D
    D_dot = jnp.sin(D)
    B_dot = jnp.cos(B)
    state = bootstrap_densitized_maxwell_state(D, B, D_dot, B_dot, 0.07)
    recovered_D, recovered_B = common_densitized_fields(state)
    np.testing.assert_allclose(recovered_D, D, rtol=0.0, atol=2.0e-14)
    np.testing.assert_array_equal(recovered_B, B)


def test_first_and_second_order_stress_energy_agree_for_physical_fields():
    shape = (8, 6, 4)
    params = BSSNParameters(dx=0.2, dt=0.01, nu=0.0)
    raw_metric = jnp.asarray(
        [[1.1, 0.12, 0.04], [0.12, 0.95, -0.07], [0.04, -0.07, 1.05]]
    )
    conformal_metric = raw_metric / jnp.linalg.det(raw_metric) ** (1.0 / 3.0)
    bssn = _constant_bssn(shape, 0.82, conformal_metric)
    D_up_values = jnp.asarray((0.2, -0.1, 0.07))
    B_up_values = jnp.asarray((-0.04, 0.16, 0.11))
    D_up = jnp.broadcast_to(D_up_values[:, None, None, None], (3,) + shape)
    B_up = jnp.broadcast_to(B_up_values[:, None, None, None], (3,) + shape)
    D_density = 0.82**-3 * D_up
    B_density = 0.82**-3 * B_up
    first = compute_densitized_electromagnetic_energy_momentum(
        D_density, B_density, bssn, params
    )

    physical_metric = conformal_metric / 0.82**2
    D_down = jnp.einsum("ij,j...->i...", physical_metric, D_up)
    B_down = jnp.einsum("ij,j...->i...", physical_metric, B_up)
    second = compute_electromagnetic_energy_momentum(D_down, B_down, bssn)
    for first_field, second_field in zip(first, second):
        np.testing.assert_allclose(first_field, second_field, rtol=3.0e-13)


def test_zero_fields_reduce_to_vacuum_bssn_step():
    shape = (8, 1, 1)
    params = BSSNParameters(dx=0.2, dt=0.002, nu=0.0, zero_shift=1)
    bssn = flat_bssn_variables(shape)
    zero = jnp.zeros((3,) + shape)
    em = DensitizedMaxwellState(zero, zero, zero, zero)
    coupled = first_order_einstein_maxwell_step(
        FirstOrderEinsteinMaxwellState(bssn, em), params
    )
    vacuum = rk4_step(bssn, params)
    jax.block_until_ready((coupled, vacuum))
    for coupled_field, vacuum_field in zip(coupled.bssn, vacuum):
        np.testing.assert_allclose(coupled_field, vacuum_field, atol=2.0e-14)
    for field in coupled.em:
        np.testing.assert_array_equal(field, zero)


def test_axisymmetric_density_parity_preserves_twist_components():
    num_rho, num_z = 8, 11
    full_shape = (3, 2 * num_rho, 1, num_z)
    values = jnp.arange(np.prod(full_shape), dtype=jnp.float64).reshape(full_shape)
    state = DensitizedMaxwellState(values, 2.0 * values, -values, 0.5 * values)

    # Compact through the public second-order-compatible state constructor,
    # then fill/expand with first-order density parity.
    from JAX_BSSN.EM.first_order.cartoon.axisymmetry import (
        compact_axisymmetric_densitized_state,
    )

    compact = compact_axisymmetric_densitized_state(state)
    filled = fill_axisymmetric_densitized_ghosts(compact)
    positive = np.asarray(filled.magnetic_current[:, 4:, 0, :])
    np.testing.assert_array_equal(
        filled.magnetic_current[0, :4, 0, :], -positive[0, :4][::-1]
    )
    np.testing.assert_array_equal(
        filled.magnetic_current[1, :4, 0, :], -positive[1, 1:5][::-1]
    )
    np.testing.assert_array_equal(
        filled.magnetic_current[2, :4, 0, :], positive[2, 1:5][::-1]
    )
    expanded = expand_axisymmetric_densitized_state(compact)
    assert float(jnp.max(jnp.abs(expanded.magnetic_current[1]))) > 0.0


def test_axisymmetric_reconstruction_uses_each_native_yee_location():
    from JAX_BSSN.EM.first_order.cartoon.axisymmetry import (
        reconstruct_axisymmetric_densitized_support,
    )

    num_rho, num_z, dx = 16, 17, 0.1
    shape = (3, num_rho + 4, 1, num_z)
    params = BSSNParameters(
        dx=dx,
        dt=0.01,
        nu=0.0,
        x_min=-3.5 * dx,
        y_min=-4.0 * dx,
        z_min=-(num_z - 1) * dx / 2.0,
        xr_bc=1,
        zl_bc=1,
        zr_bc=1,
    )
    field = jnp.zeros(shape, dtype=jnp.float64)
    for component, location in enumerate(DISPLACEMENT_FIELD_LOCATIONS):
        rho_offset = -0.5 if location[0] == "V" else 0.0
        z_offset = -0.5 if location[2] == "V" else 0.0
        rho = params.x_min + dx * (
            jnp.arange(num_rho + 4, dtype=jnp.float64) + rho_offset
        )
        z = params.z_min + dx * (
            jnp.arange(num_z, dtype=jnp.float64) + z_offset
        )
        R, Z = jnp.meshgrid(rho, z, indexing="ij")
        profiles = (R, 0.3 * R, Z * (1.0 + 0.05 * R**2))
        field = field.at[component, 4:, 0, :].set(profiles[component][4:])

    zero = jnp.zeros_like(field)
    state = DensitizedMaxwellState(zero, zero, field, field)
    support = reconstruct_axisymmetric_densitized_support(state, params)
    reconstructed = support.displacement_right_half

    for component, location in enumerate(DISPLACEMENT_FIELD_LOCATIONS):
        offsets = tuple(-0.5 if site == "V" else 0.0 for site in location)
        x = params.x_min + dx * (jnp.arange(num_rho + 4) + offsets[0])
        y = params.y_min + dx * (jnp.arange(9) + offsets[1])
        z = params.z_min + dx * (jnp.arange(num_z) + offsets[2])
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        radius = jnp.sqrt(X**2 + Y**2)
        expected = (
            X - 0.3 * Y,
            Y + 0.3 * X,
            Z * (1.0 + 0.05 * radius**2),
        )[component]
        np.testing.assert_allclose(
            reconstructed[component, :-4, :, 3:-3],
            expected[:-4, :, 3:-3],
            atol=2.0e-11,
            rtol=2.0e-11,
        )


def test_initializer_samples_all_six_native_W_factors():
    shape = (12, 10, 8)
    params = BSSNParameters(dx=0.1, dt=0.01, nu=0.0)
    bssn = flat_bssn_variables(shape)
    x = jnp.arange(shape[0])[:, None, None]
    bssn = bssn._replace(conformal_factor=0.8 + 0.005 * x)
    physical_D = jnp.ones((3,) + shape)
    physical_B = 2.0 * jnp.ones((3,) + shape)
    state = initialize_first_order_einstein_maxwell_state(
        bssn, physical_D, physical_B, params
    )
    geometry = compute_bssn_yee_geometry(state.bssn, params)
    expected_D = jnp.stack(
        tuple(geometry.displacement[i].W**-3 for i in range(3))
    )
    expected_B = jnp.stack(
        tuple(2.0 * geometry.magnetic[i].W**-3 for i in range(3))
    )
    expected_D = jnp.broadcast_to(expected_D, physical_D.shape)
    expected_B = jnp.broadcast_to(expected_B, physical_B.shape)
    np.testing.assert_allclose(state.em.magnetic_current, expected_B)
    recovered_D, _ = common_densitized_fields(state.em)
    np.testing.assert_allclose(recovered_D, expected_D)


def test_coupled_doubled_leapfrog_is_second_order_in_time():
    num_points = 32
    final_time = 0.1
    dx = 2.0 * math.pi / num_points
    shape = (num_points, 1, 1)
    bssn = flat_bssn_variables(shape)
    x_center = dx * (jnp.arange(num_points) + 0.5)
    x_vertex = dx * jnp.arange(num_points)
    D = jnp.zeros((3,) + shape).at[1, :, 0, 0].set(
        0.02 * jnp.sin(x_center)
    )
    B = jnp.zeros((3,) + shape).at[2, :, 0, 0].set(
        0.02 * jnp.sin(x_vertex)
    )

    solutions = []
    params_for_solution = []
    for num_steps in (4, 8, 16):
        params = BSSNParameters(
            dx=dx,
            dt=final_time / num_steps,
            nu=0.0,
            kappa=0.0,
            eta=0.0,
            g=0.0,
            zero_shift=1,
            x_min=0.5 * dx,
        )
        state = initialize_first_order_einstein_maxwell_state(
            bssn, D, B, params
        )
        for _ in range(num_steps):
            state = first_order_einstein_maxwell_step(state, params)
        solutions.append(state)
        params_for_solution.append(params)
    jax.block_until_ready(tuple(solutions))

    def difference(left_index, right_index):
        left = solutions[left_index]
        right = solutions[right_index]
        squared = sum(
            jnp.mean((a - b) ** 2) for a, b in zip(left.bssn, right.bssn)
        )
        left_fields = physical_fields_at_centers(
            *common_densitized_fields(left.em),
            left.bssn,
            params_for_solution[left_index],
        )
        right_fields = physical_fields_at_centers(
            *common_densitized_fields(right.em),
            right.bssn,
            params_for_solution[right_index],
        )
        squared += sum(
            jnp.mean((a - b) ** 2)
            for a, b in zip(left_fields, right_fields)
        )
        return float(jnp.sqrt(squared))

    coarse_medium = difference(0, 1)
    medium_fine = difference(1, 2)
    assert math.log2(coarse_medium / medium_fine) > 1.8


def test_spherical_reconstruction_uses_native_radial_yee_grids():
    from JAX_BSSN.EM.first_order.cartoon.spherical_symmetry import (
        reconstruct_spherical_densitized_support,
    )

    num_radial_points, dx = 16, 0.1
    shape = (3, num_radial_points + 4, 1, 1)
    params = BSSNParameters(
        dx=dx,
        dt=0.01,
        nu=0.0,
        x_min=-3.5 * dx,
        y_min=-4.0 * dx,
        z_min=-4.0 * dx,
        xr_bc=1,
    )
    radial_axis = params.x_min + dx * (
        jnp.arange(num_radial_points + 4) - 0.5
    )
    radial_profile = radial_axis * (1.0 + 0.05 * radial_axis**2)
    D = jnp.zeros(shape).at[0, 4:, 0, 0].set(radial_profile[4:])
    zero = jnp.zeros_like(D)
    state = DensitizedMaxwellState(zero, zero, D, D)
    support = reconstruct_spherical_densitized_support(state, params)

    for component, location in enumerate(DISPLACEMENT_FIELD_LOCATIONS):
        offsets = tuple(-0.5 if site == "V" else 0.0 for site in location)
        x = params.x_min + dx * (
            jnp.arange(num_radial_points + 4) + offsets[0]
        )
        y = params.y_min + dx * (jnp.arange(9) + offsets[1])
        z = params.z_min + dx * (jnp.arange(9) + offsets[2])
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        radius_squared = X**2 + Y**2 + Z**2
        expected = (X, Y, Z)[component] * (1.0 + 0.05 * radius_squared)
        np.testing.assert_allclose(
            support.displacement_right_half[component, :-4],
            expected[:-4],
            atol=2.0e-11,
            rtol=2.0e-11,
        )


def test_dynamic_metric_stage_updates_differ_from_frozen_metric_update():
    num_points = 24
    dx = 2.0 * math.pi / num_points
    params = BSSNParameters(
        dx=dx,
        dt=0.03,
        nu=0.0,
        eta=0.0,
        g=0.0,
        zero_shift=1,
        x_min=0.5 * dx,
    )
    shape = (num_points, 1, 1)
    x_center = dx * (jnp.arange(num_points) + 0.5)
    x_vertex = dx * jnp.arange(num_points)
    bssn = flat_bssn_variables(shape)._replace(
        lapse=(1.0 + 0.08 * jnp.sin(x_center))[:, None, None]
    )
    D = jnp.zeros((3,) + shape).at[1, :, 0, 0].set(
        0.1 * jnp.sin(x_center)
    )
    B = jnp.zeros((3,) + shape).at[2, :, 0, 0].set(
        0.1 * jnp.sin(x_vertex)
    )
    initial = initialize_first_order_einstein_maxwell_state(
        bssn, D, B, params
    )
    synchronized = first_order_einstein_maxwell_step(initial, params)

    # Reproduce only the electromagnetic doubled leapfrog while deliberately
    # freezing every constitutive evaluation to the t_n geometry.
    D_n, B_n = common_densitized_fields(initial.em)
    frozen_geometry = compute_bssn_yee_geometry(initial.bssn, params)
    _, B_rhs_n = densitized_maxwell_rhs(D_n, B_n, frozen_geometry, params)
    B_left_half = 0.5 * (
        initial.em.magnetic_previous + initial.em.magnetic_current
    )
    B_right_half = B_left_half + params.dt * B_rhs_n
    D_rhs_mid, B_rhs_mid = densitized_maxwell_rhs(
        initial.em.displacement_right_half,
        B_right_half,
        frozen_geometry,
        params,
    )
    D_next = D_n + params.dt * D_rhs_mid
    B_next = B_n + params.dt * B_rhs_mid
    D_rhs_end, _ = densitized_maxwell_rhs(
        D_next, B_next, frozen_geometry, params
    )
    frozen_right_half = (
        initial.em.displacement_right_half + params.dt * D_rhs_end
    )
    jax.block_until_ready((synchronized, frozen_right_half))

    difference = max(
        float(jnp.max(jnp.abs(synchronized.em.magnetic_current - B_next))),
        float(
            jnp.max(
                jnp.abs(
                    synchronized.em.displacement_right_half
                    - frozen_right_half
                )
            )
        ),
    )
    assert difference > 1.0e-9


def test_periodic_coupled_evolution_preserves_both_density_constraints():
    num_points = 32
    dx = 2.0 * math.pi / num_points
    params = BSSNParameters(
        dx=dx,
        dt=0.04,
        nu=0.0,
        eta=0.0,
        g=0.0,
        zero_shift=1,
        x_min=0.5 * dx,
    )
    shape = (num_points, 1, 1)
    x_center = dx * (jnp.arange(num_points) + 0.5)
    x_vertex = dx * jnp.arange(num_points)
    D = jnp.zeros((3,) + shape).at[1, :, 0, 0].set(
        0.03 * jnp.sin(2.0 * x_center)
    )
    B = jnp.zeros((3,) + shape).at[2, :, 0, 0].set(
        0.03 * jnp.sin(2.0 * x_vertex)
    )
    state = initialize_first_order_einstein_maxwell_state(
        flat_bssn_variables(shape), D, B, params
    )
    for _ in range(4):
        state = first_order_einstein_maxwell_step(state, params)
    D_density, B_density = common_densitized_fields(state.em)
    divergence_D = densitized_displacement_divergence(D_density, params)
    divergence_B = densitized_magnetic_divergence(B_density, params)
    jax.block_until_ready((divergence_D, divergence_B))
    np.testing.assert_allclose(divergence_D, 0.0, atol=2.0e-13)
    np.testing.assert_allclose(divergence_B, 0.0, atol=2.0e-13)


def test_axisymmetric_coupled_doubled_leapfrog_is_second_order_in_time():
    num_rho, num_z = 6, 9
    dx = 0.25
    final_time = 0.04
    shape = (num_rho + 4, 1, num_z)
    z_min = -(num_z - 1) * dx / 2.0

    rho = (jnp.arange(num_rho) + 0.5) * dx
    z = z_min + dx * jnp.arange(num_z)
    R, Z = jnp.meshgrid(rho, z, indexing="ij")
    D = jnp.zeros((3,) + shape).at[1, 4:, 0, :].set(
        0.1 * R * jnp.exp(-(R**2 + Z**2))
    )
    rho_vertex = jnp.arange(num_rho) * dx
    z_vertex = z_min + dx * (jnp.arange(num_z) - 0.5)
    R, Z = jnp.meshgrid(rho_vertex, z_vertex, indexing="ij")
    B = jnp.zeros_like(D).at[1, 4:, 0, :].set(
        -0.08 * R * jnp.exp(-(R**2 + Z**2))
    )

    solutions = []
    for num_steps in (2, 4, 8):
        params = BSSNParameters(
            dx=dx,
            dt=final_time / num_steps,
            nu=0.0,
            eta=0.0,
            g=0.0,
            zero_shift=1,
            x_min=-3.5 * dx,
            y_min=-4.0 * dx,
            z_min=z_min,
            xr_bc=1,
            zl_bc=1,
            zr_bc=1,
        )
        state = initialize_axisymmetric_first_order_state(
            flat_bssn_variables(shape), D, B, params
        )
        for _ in range(num_steps):
            state = axisymmetric_first_order_einstein_maxwell_step(
                state, params
            )
        solutions.append(state)
    jax.block_until_ready(tuple(solutions))

    def difference(left, right):
        D_left, B_left = common_densitized_fields(left.em)
        D_right, B_right = common_densitized_fields(right.em)
        squared = jnp.mean(
            (D_left[:, 4:-1] - D_right[:, 4:-1]) ** 2
        ) + jnp.mean((B_left[:, 4:-1] - B_right[:, 4:-1]) ** 2)
        squared += sum(
            jnp.mean(
                (
                    left_field[..., 4:-1, 0, 1:-1]
                    - right_field[..., 4:-1, 0, 1:-1]
                )
                ** 2
            )
            for left_field, right_field in zip(left.bssn, right.bssn)
        )
        return float(jnp.sqrt(squared))

    coarse_medium = difference(solutions[0], solutions[1])
    medium_fine = difference(solutions[1], solutions[2])
    assert math.log2(coarse_medium / medium_fine) > 1.8


def test_axisymmetric_nonroundoff_constraints_are_second_order_in_space():
    displacement_errors = []
    magnetic_errors = []
    for num_rho in (8, 16, 32):
        num_z = 2 * num_rho + 1
        dx = 2.0 / num_rho
        z_min = -(num_z - 1) * dx / 2.0
        shape = (3, num_rho + 4, 1, num_z)
        params = BSSNParameters(
            dx=dx,
            dt=0.01,
            nu=0.0,
            x_min=-3.5 * dx,
            y_min=-4.0 * dx,
            z_min=z_min,
            xr_bc=1,
            zl_bc=1,
            zr_bc=1,
        )
        rho_vertex = jnp.arange(num_rho) * dx
        rho_center = (jnp.arange(num_rho) + 0.5) * dx
        z_center = z_min + dx * jnp.arange(num_z)
        z_vertex = z_min + dx * (jnp.arange(num_z) - 0.5)

        D = jnp.zeros(shape)
        R, Z = jnp.meshgrid(rho_vertex, z_center, indexing="ij")
        D = D.at[0, 4:, 0, :].set(
            2.0 * Z * R * jnp.exp(-(R**2 + Z**2))
        )
        R, Z = jnp.meshgrid(rho_center, z_vertex, indexing="ij")
        D = D.at[2, 4:, 0, :].set(
            2.0 * (1.0 - R**2) * jnp.exp(-(R**2 + Z**2))
        )

        B = jnp.zeros(shape)
        R, Z = jnp.meshgrid(rho_center, z_vertex, indexing="ij")
        B = B.at[0, 4:, 0, :].set(
            1.4 * Z * R * jnp.exp(-(R**2 + Z**2))
        )
        R, Z = jnp.meshgrid(rho_vertex, z_center, indexing="ij")
        B = B.at[2, 4:, 0, :].set(
            1.4 * (1.0 - R**2) * jnp.exp(-(R**2 + Z**2))
        )

        em = DensitizedMaxwellState(B, B, D, D)
        divergence_D, divergence_B = (
            axisymmetric_densitized_constraint_divergences(em, params)
        )
        jax.block_until_ready((divergence_D, divergence_B))

        R, Z = jnp.meshgrid(rho_center, z_center, indexing="ij")
        D_mask = (R > 0.2) & (R < 1.0) & (jnp.abs(Z) < 1.0)
        R, Z = jnp.meshgrid(rho_vertex, z_vertex, indexing="ij")
        B_mask = (R > 0.2) & (R < 1.0) & (jnp.abs(Z) < 1.0)
        D_values = divergence_D[4:, 0, :]
        B_values = divergence_B[4:, 0, :]
        displacement_errors.append(
            float(
                jnp.sqrt(
                    jnp.sum(jnp.where(D_mask, D_values**2, 0.0))
                    / jnp.sum(D_mask)
                )
            )
        )
        magnetic_errors.append(
            float(
                jnp.sqrt(
                    jnp.sum(jnp.where(B_mask, B_values**2, 0.0))
                    / jnp.sum(B_mask)
                )
            )
        )

    for errors in (displacement_errors, magnetic_errors):
        assert math.log2(errors[0] / errors[1]) > 1.8
        assert math.log2(errors[1] / errors[2]) > 1.8
