import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from JAX_BSSN.bssn import BSSNParameters, BSSNVariables
from JAX_BSSN.evolution.time_evolve import rk4_step
from JAX_BSSN.evolution.boundaries import SOMMERFELD_BC
from JAX_BSSN.EM.first_order.geometry import _metric_fields_at_location
from JAX_BSSN.EM.first_order.boundaries import apply_densitized_sommerfeld_boundaries

from JAX_BSSN.EM.first_order import (
    CENTER_LOCATION,
    DensitizedMaxwellState,
    EinsteinMaxwellVariables,
    MAGNETIC_FIELD_LOCATIONS,
    bootstrap_densitized_maxwell_state,
    common_densitized_fields,
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
    interpolate_between_locations,
    physical_fields_at_centers,
)
from JAX_BSSN.EM.first_order.staggering import (
    DISPLACEMENT_FIELD_LOCATIONS,
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

    for location in DISPLACEMENT_FIELD_LOCATIONS + MAGNETIC_FIELD_LOCATIONS:
        W = interpolate_between_locations(
            bssn.conformal_factor, CENTER_LOCATION, location, params
        )
        np.testing.assert_allclose(W**-3, 0.8**-3)

    density_D = physical_D * 0.8**-3
    density_B = physical_B * 0.8**-3
    recovered_D, recovered_B = physical_fields_at_centers(
        density_D, density_B, bssn, params
    )
    np.testing.assert_allclose(recovered_D, physical_D)
    np.testing.assert_allclose(recovered_B, physical_B)


@pytest.mark.parametrize("metric_scale", [1.0, 2.0])
def test_manufactured_constitutive_relations_with_shift_and_offdiagonal_metric(metric_scale):
    shape = (7, 5, 3)
    params = BSSNParameters(dx=0.3, dt=0.02, nu=0.0)
    conformal_metric = jnp.asarray(
        [[1.2, 0.15, -0.08], [0.15, 0.9, 0.11], [-0.08, 0.11, 1.1]]
    )
    conformal_metric /= jnp.linalg.det(conformal_metric) ** (1.0 / 3.0)
    bssn = _constant_bssn(
        shape, 0.75, metric_scale * conformal_metric,
        lapse=0.83, shift=(0.12, -0.07, 0.04)
    )
    D_density_values = jnp.asarray((0.3, -0.2, 0.1))
    B_density_values = jnp.asarray((-0.15, 0.05, 0.25))
    D_density = jnp.broadcast_to(
        D_density_values[:, None, None, None], (3,) + shape
    )
    B_density = jnp.broadcast_to(
        B_density_values[:, None, None, None], (3,) + shape
    )
    E = compute_covariant_E(D_density, B_density, bssn, params)
    H = compute_covariant_H(D_density, B_density, bssn, params)

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


def _smooth_unit_metric(x, y, z):
    """Analytic SPD metric with unit determinant and off-diagonal entries."""
    phase = 0.3 * jnp.sin(x) + 0.2 * jnp.cos(y) + 0.1 * jnp.sin(z)
    eigenvalues = jnp.stack((jnp.exp(phase), jnp.exp(-0.4 * phase),
                            jnp.exp(-0.6 * phase)))
    rotation = jnp.asarray([[1., 2., 2.], [2., 1., -2.], [-2., 2., -1.]]) / 3.0
    return jnp.einsum("ia,a...,ja->ij...", rotation, eigenvalues, rotation)


def _matrix_last(metric):
    return np.moveaxis(np.asarray(metric), (0, 1), (-2, -1))


@pytest.mark.parametrize("shape", [(7, 6, 5), (7, 1, 5)])
@pytest.mark.parametrize("sommerfeld", [False, True])
def test_yee_metric_projection_preserves_W_and_volume(shape, sommerfeld):
    dx = 0.15
    params = BSSNParameters(dx=dx, dt=0.01, nu=0.0)
    if sommerfeld:
        params = params._replace(xl_bc=SOMMERFELD_BC, xr_bc=SOMMERFELD_BC,
                                 zl_bc=SOMMERFELD_BC, zr_bc=SOMMERFELD_BC)
    x, y, z = jnp.meshgrid(*(dx * (jnp.arange(n) + 0.5) for n in shape),
                           indexing="ij")
    bssn = _constant_bssn(shape, 1.0, jnp.eye(3))._replace(
        conformal_metric=_smooth_unit_metric(x, y, z),
        conformal_factor=0.8 + 0.03 * jnp.sin(x + y + z),
        lapse=0.9 + 0.02 * jnp.cos(x - z),
        shift=jnp.stack((0.02 * x, -0.03 * y, 0.01 * z)),
    )
    np.testing.assert_allclose(np.linalg.det(_matrix_last(bssn.conformal_metric)),
                               1.0, atol=1.e-12, rtol=0.0)
    for location in DISPLACEMENT_FIELD_LOCATIONS + MAGNETIC_FIELD_LOCATIONS:
        raw = interpolate_between_locations(bssn.conformal_metric,
                                             CENTER_LOCATION, location, params)
        raw_matrix = _matrix_last(raw)
        raw_det = np.linalg.det(raw_matrix)
        # An isolated shift along a singleton dimension performs no averaging.
        if any(site == "V" and n > 1 for site, n in zip(location, shape)):
            assert np.max(np.abs(raw_det - 1.0)) > 1.e-8
        W, lapse, shift, metric, inverse = _metric_fields_at_location(bssn, location, params)
        for actual, source in zip((W, lapse, shift),
                                  (bssn.conformal_factor, bssn.lapse, bssn.shift)):
            expected = interpolate_between_locations(source, CENTER_LOCATION, location, params)
            np.testing.assert_array_equal(actual, expected)
        matrix = _matrix_last(metric)
        np.testing.assert_allclose(matrix, raw_matrix / np.cbrt(raw_det)[..., None, None],
                                   atol=1.e-12, rtol=0.0)
        assert np.all(np.linalg.eigvalsh(matrix) > 0.0)
        np.testing.assert_allclose(np.linalg.det(matrix), 1.0, atol=1.e-12, rtol=0.0)
        np.testing.assert_allclose(matrix @ _matrix_last(inverse),
                                   np.broadcast_to(np.eye(3), matrix.shape),
                                   atol=1.e-12, rtol=0.0)
        volume = np.sqrt(np.linalg.det(_matrix_last(metric / W**2)))
        np.testing.assert_allclose(volume, W**-3, atol=1.e-12, rtol=0.0)


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_yee_constant_metric_projects_to_identity(scale):
    bssn = _constant_bssn((4, 3, 1), 0.8, scale * jnp.eye(3))
    for location in DISPLACEMENT_FIELD_LOCATIONS + MAGNETIC_FIELD_LOCATIONS:
        W, _, _, metric, inverse = _metric_fields_at_location(bssn, location, BSSNParameters())
        expected = np.broadcast_to(np.eye(3), _matrix_last(metric).shape)
        np.testing.assert_allclose(_matrix_last(metric), expected, atol=1.e-12, rtol=0.0)
        np.testing.assert_allclose(_matrix_last(inverse), expected, atol=1.e-12, rtol=0.0)
        np.testing.assert_array_equal(W, bssn.conformal_factor)


@pytest.mark.parametrize("diagonal", [(0., 1., 1.), (-1., 1., 1.), (-1., -1., 1.)])
def test_yee_projection_does_not_repair_invalid_geometry(diagonal):
    bssn = _constant_bssn((3, 2, 1), 0.8, jnp.diag(jnp.asarray(diagonal)))
    _, _, _, metric, inverse = _metric_fields_at_location(
        bssn, DISPLACEMENT_FIELD_LOCATIONS[0], BSSNParameters())
    # Also cover positive-determinant indefinite input: det=1 is not sufficient.
    assert not (np.all(np.isfinite(metric)) and np.all(np.isfinite(inverse)))


def test_yee_projected_metric_interpolation_is_second_order():
    errors = []
    for n in (16, 32, 64):
        dx = 2.0 * math.pi / n
        params = BSSNParameters(dx=dx)
        x, y, z = jnp.meshgrid(*(dx * (jnp.arange(n) + 0.5) for _ in range(3)),
                               indexing="ij")
        bssn = _constant_bssn((n, n, n), 0.8, jnp.eye(3))._replace(
            conformal_metric=_smooth_unit_metric(x, y, z))
        site_errors = []
        for location in DISPLACEMENT_FIELD_LOCATIONS + MAGNETIC_FIELD_LOCATIONS:
            coordinates = tuple(q - (0.5 * dx if site == "V" else 0.0)
                                for q, site in zip((x, y, z), location))
            expected = _smooth_unit_metric(*coordinates)
            metric = _metric_fields_at_location(bssn, location, params)[3]
            site_errors.append(float(jnp.sqrt(jnp.mean((metric - expected)**2))))
        errors.append(site_errors)
    orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert np.all(orders > 1.9), orders


def test_yee_sommerfeld_normals_use_projected_inverse():
    shape = (6, 1, 1)
    params = BSSNParameters(dx=0.2, xl_bc=SOMMERFELD_BC, xr_bc=SOMMERFELD_BC,
                            x_min=1.0)
    # Projection of 2I gives I, hence physical outward normal n^x = +/- W.
    bssn = _constant_bssn(shape, 0.8, 2.0 * jnp.eye(3), lapse=0.9,
                          shift=(0.1, 0.0, 0.0))
    x_center = 1.0 + params.dx * jnp.arange(shape[0])[:, None, None]
    densities = []
    for locations in (DISPLACEMENT_FIELD_LOCATIONS, MAGNETIC_FIELD_LOCATIONS):
        densities.append(jnp.stack(tuple(
            x_center - (0.5 * params.dx if location[0] == "V" else 0.0)
            for location in locations)))
    rhs = jnp.full((3,) + shape, 7.0)
    results = apply_densitized_sommerfeld_boundaries(*densities, rhs, rhs, bssn, params)
    for result, density, locations in zip(
        results, densities, (DISPLACEMENT_FIELD_LOCATIONS, MAGNETIC_FIELD_LOCATIONS)
    ):
        for component, location in enumerate(locations):
            x = np.asarray(density[component])
            transverse_squared = sum((0.5 * params.dx)**2
                                     for site in location[1:] if site == "V")
            falloff = 0.9 * x / np.sqrt(x**2 + transverse_squared)
            np.testing.assert_allclose(result[component, 0],
                                       0.1 + 0.9 * 0.8 - falloff[0], atol=1.e-12)
            np.testing.assert_allclose(result[component, -1],
                                       0.1 - 0.9 * 0.8 - falloff[-1], atol=1.e-12)
        np.testing.assert_array_equal(result[:, 1:-1], rhs[:, 1:-1])


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


def test_stress_energy_matches_analytic_physical_fields():
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
    rho = 0.5 * jnp.sum(D_up * D_down + B_up * B_down, axis=0)
    momentum = 0.82**-3 * jnp.cross(D_up, B_up, axisa=0, axisb=0, axisc=0)
    stress = (
        physical_metric[:, :, None, None, None] * rho
        - jnp.einsum("i...,j...->ij...", D_down, D_down)
        - jnp.einsum("i...,j...->ij...", B_down, B_down)
    )
    for actual, expected in zip(first, (rho, momentum, stress)):
        np.testing.assert_allclose(actual, expected, rtol=3.0e-13)


def test_zero_fields_reduce_to_vacuum_bssn_step():
    shape = (8, 1, 1)
    params = BSSNParameters(dx=0.2, dt=0.002, nu=0.0, zero_shift=1)
    bssn = flat_bssn_variables(shape)
    zero = jnp.zeros((3,) + shape)
    em = DensitizedMaxwellState(zero, zero, zero, zero)
    coupled = first_order_einstein_maxwell_step(
        EinsteinMaxwellVariables(bssn, em), params
    )
    assert isinstance(coupled, EinsteinMaxwellVariables)
    assert isinstance(coupled.bssn, BSSNVariables)
    assert isinstance(coupled.em, DensitizedMaxwellState)
    vacuum = rk4_step(bssn, params)
    jax.block_until_ready((coupled, vacuum))
    for coupled_field, vacuum_field in zip(coupled.bssn, vacuum):
        np.testing.assert_allclose(coupled_field, vacuum_field, atol=2.0e-14)
    for field in coupled.em:
        np.testing.assert_array_equal(field, zero)


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
    assert isinstance(state, EinsteinMaxwellVariables)
    assert isinstance(state.bssn, BSSNVariables)
    assert isinstance(state.em, DensitizedMaxwellState)
    expected_D = jnp.stack(
        tuple(
            interpolate_between_locations(
                state.bssn.conformal_factor,
                CENTER_LOCATION,
                location,
                params,
            )
            ** -3
            for location in DISPLACEMENT_FIELD_LOCATIONS
        )
    )
    expected_B = jnp.stack(
        tuple(
            2.0
            * interpolate_between_locations(
                state.bssn.conformal_factor,
                CENTER_LOCATION,
                location,
                params,
            )
            ** -3
            for location in MAGNETIC_FIELD_LOCATIONS
        )
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
    frozen_bssn = initial.bssn
    _, B_rhs_n = densitized_maxwell_rhs(D_n, B_n, frozen_bssn, params)
    B_left_half = 0.5 * (
        initial.em.magnetic_previous + initial.em.magnetic_current
    )
    B_right_half = B_left_half + params.dt * B_rhs_n
    D_rhs_mid, B_rhs_mid = densitized_maxwell_rhs(
        initial.em.displacement_right_half,
        B_right_half,
        frozen_bssn,
        params,
    )
    D_next = D_n + params.dt * D_rhs_mid
    B_next = B_n + params.dt * B_rhs_mid
    D_rhs_end, _ = densitized_maxwell_rhs(
        D_next, B_next, frozen_bssn, params
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
