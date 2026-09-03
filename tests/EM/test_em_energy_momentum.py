import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn import BSSNVariables
from JAX_BSSN.bssn.tensor_algebra import invert_3x3_metric
from JAX_BSSN.EM.second_order.energy_momentum import (
    compute_electromagnetic_energy_momentum,
    compute_electromagnetic_stress_energy,
)


def bssn_geometry(metric, W=1.0):
    shape = metric.shape[2:]
    zeros = jnp.zeros(shape, dtype=metric.dtype)
    return BSSNVariables(
        conformal_metric=metric * W**2,
        conformal_factor=jnp.full(shape, W, dtype=metric.dtype),
        traceless_K=jnp.zeros_like(metric),
        trace_K=zeros,
        conformal_connection=jnp.zeros((3,) + shape, dtype=metric.dtype),
        lapse=jnp.ones(shape, dtype=metric.dtype),
        shift=jnp.zeros((3,) + shape, dtype=metric.dtype),
    )


def diagonal_metric(diagonal, shape):
    metric = jnp.zeros((3, 3) + shape, dtype=jnp.float64)
    for i, value in enumerate(diagonal):
        metric = metric.at[i, i].set(value)
    return metric


def test_minkowski_crossed_fields_match_energy_poynting_and_stress():
    shape = (2, 1, 1)
    bssn = bssn_geometry(diagonal_metric((1.0, 1.0, 1.0), shape))
    electric_field = jnp.zeros((3,) + shape, dtype=jnp.float64).at[0].set(2.0)
    magnetic_field = jnp.zeros_like(electric_field).at[1].set(3.0)

    energy_density, momentum_density, spatial_stress = (
        compute_electromagnetic_energy_momentum(
            electric_field, magnetic_field, bssn
        )
    )

    expected_momentum = jnp.zeros_like(electric_field).at[2].set(6.0)
    expected_stress = jnp.zeros_like(bssn.conformal_metric)
    expected_stress = expected_stress.at[0, 0].set(2.5)
    expected_stress = expected_stress.at[1, 1].set(-2.5)
    expected_stress = expected_stress.at[2, 2].set(6.5)

    np.testing.assert_allclose(energy_density, 6.5)
    np.testing.assert_allclose(momentum_density, expected_momentum)
    np.testing.assert_allclose(spatial_stress, expected_stress)


def test_curved_metric_projections_have_maxwell_trace_and_duality():
    shape = (3, 2, 1)
    metric = diagonal_metric((4.0, 9.0, 16.0), shape)
    bssn = bssn_geometry(metric)
    inverse_metric = invert_3x3_metric(metric)
    electric_field = jnp.broadcast_to(
        jnp.asarray([2.0, -3.0, 8.0])[:, None, None, None], (3,) + shape
    )
    magnetic_field = jnp.broadcast_to(
        jnp.asarray([-4.0, 6.0, 4.0])[:, None, None, None], (3,) + shape
    )

    energy_density, momentum_density, spatial_stress = (
        compute_electromagnetic_energy_momentum(
            electric_field, magnetic_field, bssn
        )
    )
    dual_energy, dual_momentum, dual_stress = (
        compute_electromagnetic_energy_momentum(
            magnetic_field, -electric_field, bssn
        )
    )

    stress_trace = jnp.einsum(
        "ij...,ij...->...", inverse_metric, spatial_stress
    )
    np.testing.assert_allclose(stress_trace, energy_density)
    np.testing.assert_allclose(dual_energy, energy_density)
    np.testing.assert_allclose(dual_momentum, momentum_density, atol=1.0e-14)
    np.testing.assert_allclose(dual_stress, spatial_stress)


def test_conformal_stress_path_remains_finite_at_the_W_floor():
    shape = (2, 2, 2)
    conformal_metric = diagonal_metric((1.0, 1.0, 1.0), shape)
    bssn = bssn_geometry(conformal_metric, W=1.0e-16)
    electric_field = jnp.ones((3,) + shape, dtype=jnp.float64)
    magnetic_field = 0.5 * electric_field

    stress_energy = compute_electromagnetic_stress_energy(
        electric_field, magnetic_field, bssn
    )

    assert bool(jnp.all(jnp.isfinite(stress_energy.energy_density)))
    assert bool(jnp.all(jnp.isfinite(stress_energy.momentum_density)))
    assert bool(jnp.all(jnp.isfinite(stress_energy.scaled_spatial_stress)))
