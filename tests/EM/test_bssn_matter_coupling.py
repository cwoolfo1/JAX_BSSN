"""Focused tests for physical matter sources in the Cartesian BSSN system."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn import BSSNParameters, BSSNVariables
from JAX_BSSN.bssn.constraints import (
    compute_all_constraints_with_matter,
    compute_hamiltonian_constraint,
    compute_hamiltonian_constraint_with_matter,
    compute_momentum_constraint,
    compute_momentum_constraint_and_derivative_with_matter,
    compute_momentum_constraint_with_matter,
)
from JAX_BSSN.bssn.geometry import W_FLOOR_VALUE
from JAX_BSSN.evolution.boundaries import SOMMERFELD_BC
from JAX_BSSN.evolution.derivatives import diff1_field
from JAX_BSSN.evolution.time_evolve import (
    compute_bssn_rhs,
    compute_bssn_rhs_with_matter,
)


def flat_bssn_variables(shape):
    conformal_metric = jnp.eye(3, dtype=jnp.float64)[
        :, :, None, None, None
    ]
    conformal_metric = jnp.broadcast_to(
        conformal_metric, (3, 3) + shape
    )
    scalar_zero = jnp.zeros(shape, dtype=jnp.float64)
    vector_zero = jnp.zeros((3,) + shape, dtype=jnp.float64)

    return BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=jnp.ones(shape, dtype=jnp.float64),
        traceless_K=jnp.zeros_like(conformal_metric),
        trace_K=scalar_zero,
        conformal_connection=vector_zero,
        lapse=jnp.ones(shape, dtype=jnp.float64),
        shift=vector_zero,
    )


def uniform_sources(shape):
    energy_density = 0.7 * jnp.ones(shape, dtype=jnp.float64)
    momentum_density = jnp.stack(
        [
            0.2 * jnp.ones(shape, dtype=jnp.float64),
            -0.3 * jnp.ones(shape, dtype=jnp.float64),
            0.5 * jnp.ones(shape, dtype=jnp.float64),
        ],
        axis=0,
    )
    spatial_stress = jnp.zeros((3, 3) + shape, dtype=jnp.float64)
    spatial_stress = spatial_stress.at[0, 0].set(0.6)
    spatial_stress = spatial_stress.at[1, 1].set(-0.1)
    spatial_stress = spatial_stress.at[2, 2].set(0.4)
    spatial_stress = spatial_stress.at[0, 1].set(0.15)
    spatial_stress = spatial_stress.at[1, 0].set(0.15)
    return energy_density, momentum_density, spatial_stress


def test_independent_matter_source_increments_and_zero_source_limit():
    shape = (7, 7, 7)
    vars = flat_bssn_variables(shape)
    params = BSSNParameters(
        dx=0.2, dt=0.01, nu=0.0, kappa=0.0, g=0.0, eta=0.0
    )
    energy_density, momentum_density, spatial_stress = uniform_sources(shape)

    vacuum_rhs = compute_bssn_rhs(vars, params)
    matter_rhs = compute_bssn_rhs_with_matter(
        vars,
        params,
        energy_density,
        momentum_density,
        spatial_stress,
    )

    stress_trace = jnp.einsum("ii...->...", spatial_stress)
    stress_tf = spatial_stress - (
        jnp.eye(3, dtype=jnp.float64)[:, :, None, None, None]
        * stress_trace[None, None]
        / 3.0
    )

    np.testing.assert_allclose(
        matter_rhs.trace_K - vacuum_rhs.trace_K,
        4.0 * jnp.pi * (energy_density + stress_trace),
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        matter_rhs.traceless_K - vacuum_rhs.traceless_K,
        -8.0 * jnp.pi * stress_tf,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        matter_rhs.conformal_connection - vacuum_rhs.conformal_connection,
        -16.0 * jnp.pi * momentum_density,
        rtol=2.0e-14,
        atol=2.0e-14,
    )

    for matter_field, vacuum_field in zip(matter_rhs, vacuum_rhs):
        assert bool(jnp.all(jnp.isfinite(matter_field)))
        assert matter_field.shape == vacuum_field.shape

    zero_matter_rhs = compute_bssn_rhs_with_matter(
        vars,
        params,
        jnp.zeros_like(energy_density),
        jnp.zeros_like(momentum_density),
        jnp.zeros_like(spatial_stress),
    )
    for zero_matter_field, vacuum_field in zip(zero_matter_rhs, vacuum_rhs):
        np.testing.assert_array_equal(zero_matter_field, vacuum_field)


def test_matter_constraints_and_coordinate_derivative():
    shape = (9, 7, 7)
    dx = 0.25
    vars = flat_bssn_variables(shape)
    params = BSSNParameters(dx=dx, dt=0.01, nu=0.0)

    x = dx * jnp.arange(shape[0], dtype=jnp.float64)
    phase = 2.0 * jnp.pi * x / (shape[0] * dx)
    wave = jnp.sin(phase)[:, None, None] * jnp.ones(shape)
    energy_density = 0.1 + 0.03 * wave
    momentum_density = jnp.stack((wave, -0.4 * wave, 0.2 * wave), axis=0)

    vacuum_hamiltonian = compute_hamiltonian_constraint(vars, params)
    vacuum_momentum = compute_momentum_constraint(vars, params)
    hamiltonian = compute_hamiltonian_constraint_with_matter(
        vars, params, energy_density
    )
    momentum = compute_momentum_constraint_with_matter(
        vars, params, momentum_density
    )
    momentum_and_derivative = (
        compute_momentum_constraint_and_derivative_with_matter(
            vars, params, momentum_density
        )
    )
    all_constraints = compute_all_constraints_with_matter(
        vars, params, energy_density, momentum_density
    )

    source_derivative = jnp.stack(
        [
            diff1_field(momentum_density, direction + 1, dx)
            for direction in range(3)
        ],
        axis=1,
    )
    np.testing.assert_allclose(
        hamiltonian - vacuum_hamiltonian,
        -16.0 * jnp.pi * energy_density,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        momentum - vacuum_momentum,
        -8.0 * jnp.pi * momentum_density,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        momentum_and_derivative[0], momentum, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        momentum_and_derivative[1],
        -8.0 * jnp.pi * source_derivative,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(all_constraints.hamiltonian, hamiltonian)
    np.testing.assert_allclose(all_constraints.momentum, momentum)


def test_kappa_damping_uses_full_matter_momentum_constraint():
    shape = (9, 7, 7)
    dx = 0.2
    vars = flat_bssn_variables(shape)
    params = BSSNParameters(
        dx=dx, dt=0.01, nu=0.0, kappa=0.8, g=0.0, eta=0.0
    )

    x = dx * jnp.arange(shape[0], dtype=jnp.float64)
    phase = 2.0 * jnp.pi * x / (shape[0] * dx)
    wave = jnp.sin(phase)[:, None, None] * jnp.ones(shape)
    momentum_density = jnp.stack((0.7 * wave, -0.2 * wave, 0.4 * wave), axis=0)
    energy_density = jnp.zeros(shape, dtype=jnp.float64)
    spatial_stress = jnp.zeros((3, 3) + shape, dtype=jnp.float64)

    vacuum_rhs = compute_bssn_rhs(vars, params)
    matter_rhs = compute_bssn_rhs_with_matter(
        vars,
        params,
        energy_density,
        momentum_density,
        spatial_stress,
    )

    source_derivative = jnp.stack(
        [
            diff1_field(momentum_density, direction + 1, dx)
            for direction in range(3)
        ],
        axis=1,
    )
    expected_damping = -4.0 * jnp.pi * params.kappa * (
        source_derivative + jnp.swapaxes(source_derivative, 0, 1)
    )
    np.testing.assert_allclose(
        matter_rhs.traceless_K - vacuum_rhs.traceless_K,
        expected_damping,
        rtol=3.0e-13,
        atol=3.0e-13,
    )


def test_matter_sources_use_active_conformal_factor_floor():
    shape = (7, 7, 7)
    vars = flat_bssn_variables(shape)._replace(
        conformal_factor=jnp.full(
            shape, 0.5 * W_FLOOR_VALUE, dtype=jnp.float64
        )
    )
    params = BSSNParameters(
        dx=0.2, dt=0.01, nu=0.0, kappa=0.0, g=0.0, eta=0.0
    )

    floor_scaled_stress = jnp.zeros(
        (3, 3) + shape, dtype=jnp.float64
    )
    floor_scaled_stress = floor_scaled_stress.at[0, 0].set(0.8)
    floor_scaled_stress = floor_scaled_stress.at[1, 1].set(-0.2)
    floor_scaled_stress = floor_scaled_stress.at[2, 2].set(0.5)
    spatial_stress = floor_scaled_stress / W_FLOOR_VALUE**2

    vacuum_rhs = compute_bssn_rhs(vars, params)
    matter_rhs = compute_bssn_rhs_with_matter(
        vars,
        params,
        jnp.zeros(shape, dtype=jnp.float64),
        jnp.zeros((3,) + shape, dtype=jnp.float64),
        spatial_stress,
    )

    floor_stress_trace = jnp.einsum("ii...->...", floor_scaled_stress)
    floor_stress_tf = floor_scaled_stress - (
        jnp.eye(3, dtype=jnp.float64)[:, :, None, None, None]
        * floor_stress_trace[None, None]
        / 3.0
    )
    np.testing.assert_allclose(
        matter_rhs.trace_K - vacuum_rhs.trace_K,
        4.0 * jnp.pi * floor_stress_trace,
        rtol=3.0e-14,
        atol=3.0e-14,
    )
    np.testing.assert_allclose(
        matter_rhs.traceless_K - vacuum_rhs.traceless_K,
        -8.0 * jnp.pi * floor_stress_tf,
        rtol=3.0e-14,
        atol=3.0e-14,
    )
    for field in matter_rhs:
        assert bool(jnp.all(jnp.isfinite(field)))


def test_sommerfeld_faces_replace_matter_source_rhs():
    shape = (7, 7, 7)
    vars = flat_bssn_variables(shape)
    params = BSSNParameters(
        dx=0.2,
        dt=0.01,
        nu=0.0,
        kappa=0.0,
        g=0.0,
        eta=0.0,
        xr_bc=SOMMERFELD_BC,
        x_min=-0.6,
        y_min=-0.6,
        z_min=-0.6,
    )
    energy_density, momentum_density, spatial_stress = uniform_sources(shape)

    rhs = compute_bssn_rhs_with_matter(
        vars,
        params,
        energy_density,
        momentum_density,
        spatial_stress,
    )

    np.testing.assert_allclose(rhs.trace_K[-1], 0.0, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        rhs.traceless_K[:, :, -1], 0.0, rtol=0.0, atol=1.0e-14
    )
    np.testing.assert_allclose(
        rhs.conformal_connection[:, -1], 0.0, rtol=0.0, atol=1.0e-14
    )
    assert float(jnp.max(jnp.abs(rhs.trace_K[-2]))) > 0.0
    assert float(jnp.max(jnp.abs(rhs.traceless_K[:, :, -2]))) > 0.0
    assert float(jnp.max(jnp.abs(rhs.conformal_connection[:, -2]))) > 0.0
