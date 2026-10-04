import math
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest


from tests.EM.demo_helpers import load_em_demo_module


def _load_initial_data(formulation):
    pulse = load_em_demo_module(formulation, "initial_pulse")
    metric = load_em_demo_module(formulation, "initial_metric")
    return SimpleNamespace(**{
        name: value for module in (pulse, metric)
        for name, value in vars(module).items()
        if callable(value) and not name.startswith("_")
    })


@pytest.fixture(params=("first_order", "second_order"))
def initial_data(request):
    return _load_initial_data(request.param)


def test_literature_seed_uses_hl_normalization_and_cartesian_azimuthal_basis(
    initial_data,
):
    amplitude = 0.08
    width = 1.2
    radial_center = 3.0
    grid = jnp.asarray([[[[2.0, 0.0, -0.5], [1.0, 2.0, 0.25]]]])

    conformal_electric = initial_data.off_centered_toroidal_electric_seed(
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
    electric_covector = initial_data.conformal_vector_to_physical_covector(
        conformal_electric, psi
    )
    inverse_metric = (
        psi[None, None, ...] ** -4
        * jnp.eye(3)[:, :, None, None, None]
    )
    electric_up = jnp.einsum(
        "ij...,j...->i...", inverse_metric, electric_covector
    )
    physical_contravariant = (
        initial_data.conformal_vector_to_physical_contravariant(
            conformal_electric, psi
        )
    )
    np.testing.assert_allclose(physical_contravariant, electric_up)
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


def test_electromagnetic_hamiltonian_source_linearization(initial_data):
    psi = jnp.linspace(1.05, 1.35, 24).reshape((2, 3, 4))
    field_squared = jnp.linspace(0.01, 0.2, 24).reshape((2, 3, 4))
    delta_u = jnp.cos(jnp.arange(24)).reshape((2, 3, 4))
    epsilon = 1.0e-6

    finite_difference = (
        initial_data.electromagnetic_hamiltonian_source(
            psi + epsilon * delta_u, field_squared
        )
        - initial_data.electromagnetic_hamiltonian_source(
            psi - epsilon * delta_u, field_squared
        )
    ) / (2.0 * epsilon)
    expected = initial_data.linearized_electromagnetic_hamiltonian_source(
        delta_u, psi, field_squared
    )

    np.testing.assert_allclose(
        finite_difference,
        expected,
        rtol=2.0e-9,
        atol=2.0e-10,
    )


def test_zero_field_returns_exact_flat_conformal_factor(initial_data):
    field_squared = jnp.zeros((8, 16), dtype=jnp.float64)
    psi, u, residual_history = (
        initial_data.solve_electromagnetic_conformal_factor(
            field_squared, dx=0.25
        )
    )

    np.testing.assert_array_equal(psi, jnp.ones_like(psi))
    np.testing.assert_array_equal(u, jnp.zeros_like(u))
    assert residual_history == [0.0]


def _manufactured_hamiltonian_data(num_points):
    dx = 1.0 / (num_points - 0.5)
    rho = (jnp.arange(num_points) + 0.5) * dx
    z = (jnp.arange(2 * num_points) - (num_points - 0.5)) * dx
    RHO, Z = jnp.meshgrid(rho, z, indexing="ij")

    amplitude = 0.04
    wavenumber = jnp.pi / 2.0
    radial_mode = jnp.cos(wavenumber * RHO)
    z_mode = jnp.cos(wavenumber * Z)
    mode = radial_mode * z_mode
    psi = 1.0 + amplitude * mode
    radial_laplacian = amplitude * z_mode * (
        -wavenumber**2 * radial_mode
        - wavenumber * jnp.sin(wavenumber * RHO) / RHO
    )
    z_laplacian = -wavenumber**2 * amplitude * mode
    continuum_laplacian = radial_laplacian + z_laplacian
    field_squared = -continuum_laplacian * psi**3 / jnp.pi

    return psi, field_squared, dx


def test_hamiltonian_solve_is_second_order_for_manufactured_solution(
    initial_data,
):
    errors = []
    residuals = []
    for num_points in (9, 17, 33):
        expected_psi, field_squared, dx = _manufactured_hamiltonian_data(
            num_points
        )
        psi, u, residual_history = (
            initial_data.solve_electromagnetic_conformal_factor(
                field_squared,
                dx,
                newton_tolerance=1.0e-11,
                cg_tolerance=1.0e-12,
            )
        )
        residual = initial_data.electromagnetic_hamiltonian_residual(
            u, field_squared, dx
        )

        interior_error = psi[:-1, 1:-1] - expected_psi[:-1, 1:-1]
        errors.append(
            float(jnp.sqrt(jnp.mean(interior_error**2)))
        )
        residuals.append(float(jnp.sqrt(jnp.mean(residual**2))))

        assert bool(jnp.all(jnp.isfinite(psi)))
        assert float(jnp.min(psi)) >= 1.0
        assert residual_history[-1] <= 1.0e-11
        for face in (u[-1], u[:, 0], u[:, -1]):
            np.testing.assert_array_equal(face, jnp.zeros_like(face))

    orders = [
        math.log(errors[index] / errors[index + 1]) / math.log(2.0)
        for index in range(2)
    ]

    assert max(residuals) <= 1.0e-11
    assert min(orders) > 1.8


def test_cylindrical_linearized_operator_is_symmetric(initial_data):
    shape = (12, 22)
    psi = jnp.linspace(1.01, 1.2, np.prod(shape)).reshape(shape)
    field_squared = jnp.linspace(0.0, 0.1, np.prod(shape)).reshape(shape)
    left = jnp.sin(jnp.arange(np.prod(shape))).reshape(shape)
    right = jnp.cos(jnp.arange(np.prod(shape))).reshape(shape)

    operator_left = initial_data.linearized_electromagnetic_hamiltonian_operator(
        left, psi, field_squared, 0.1
    )
    operator_right = initial_data.linearized_electromagnetic_hamiltonian_operator(
        right, psi, field_squared, 0.1
    )

    np.testing.assert_allclose(
        jnp.vdot(left, operator_right),
        jnp.vdot(operator_left, right),
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_toroidal_seed_produces_nonnegative_conformal_energy(initial_data):
    axis = jnp.linspace(-4.0, 4.0, 9)
    X, Y, Z = jnp.meshgrid(axis, axis, axis, indexing="ij")
    grid = jnp.stack((X, Y, Z), axis=-1)
    conformal_electric = initial_data.off_centered_toroidal_electric_seed(grid)
    conformal_magnetic = jnp.zeros_like(conformal_electric)

    field_squared = initial_data.contract_conformal_electromagnetic_fields(
        conformal_electric, conformal_magnetic
    )

    assert bool(jnp.all(field_squared >= 0.0))
    assert float(jnp.max(field_squared)) > 0.0
    np.testing.assert_array_equal(field_squared[4, 4, :], 0.0)


@pytest.mark.parametrize("formulation", ["first_order", "second_order"])
def test_zero_pulse_builds_flat_compact_metric(formulation):
    pulse = load_em_demo_module(formulation, "initial_pulse")
    metric = load_em_demo_module(formulation, "initial_metric")
    grid = pulse.electromagnetic_cylindrical_grid(6, 12, 0.5)
    electric, magnetic = pulse.initial_pulse(grid, amplitude=0.0)
    squared = pulse.contract_conformal_electromagnetic_fields(electric, magnetic)
    bssn, psi, u, history = metric.initial_metric(squared[:, 0, :], 0.5)
    np.testing.assert_array_equal(electric, 0.0)
    np.testing.assert_array_equal(magnetic, 0.0)
    np.testing.assert_array_equal(psi, 1.0)
    np.testing.assert_array_equal(u, 0.0)
    assert history == [0.0]
    assert bssn.lapse.shape == (10, 1, 12)
    assert bssn.lapse.dtype == jnp.float64
    np.testing.assert_array_equal(bssn.lapse, 1.0)
    np.testing.assert_array_equal(bssn.conformal_factor, 1.0)
    for field in (bssn.traceless_K, bssn.trace_K, bssn.conformal_connection, bssn.shift):
        np.testing.assert_array_equal(field, 0.0)
    np.testing.assert_array_equal(
        bssn.conformal_metric,
        np.broadcast_to(np.eye(3)[:, :, None, None, None], (3, 3, 10, 1, 12)),
    )
