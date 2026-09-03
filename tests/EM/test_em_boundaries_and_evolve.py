import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.evolution.boundaries import PERIODIC_BC, SOMMERFELD_BC
from JAX_BSSN.evolution.time_evolve import rk4_step

from JAX_BSSN.EM.boundaries import apply_planar_sommerfeld_boundaries
from JAX_BSSN.EM.evolve import einstein_maxwell_rk4_step
from JAX_BSSN.EM.geometry import compute_bssn_em_geometry
from tests.EM.em_helpers import (
    flat_bssn_variables,
    zero_bssn_rhs,
    zero_em_variables,
)
from tests.initial_data import periodic_gauge_wave_state
from JAX_BSSN.EM.variables import EinsteinMaxwellVariables, EMVariables


def test_planar_sommerfeld_uses_face_sign_lapse_and_shift():
    shape = (8, 1, 1)
    dx = 0.25
    params = BSSNParameters(
        dx=dx,
        dt=0.01,
        nu=0.0,
        xl_bc=SOMMERFELD_BC,
        xr_bc=SOMMERFELD_BC,
        yl_bc=PERIODIC_BC,
        yr_bc=PERIODIC_BC,
        zl_bc=PERIODIC_BC,
        zr_bc=PERIODIC_BC,
    )
    bssn = flat_bssn_variables(shape)._replace(
        lapse=2.0 * jnp.ones(shape),
        shift=jnp.stack(
            (
                0.5 * jnp.ones(shape),
                jnp.zeros(shape),
                jnp.zeros(shape),
            )
        ),
    )
    geometry = compute_bssn_em_geometry(
        bssn, zero_bssn_rhs(bssn), params
    )

    x = dx * jnp.arange(shape[0], dtype=jnp.float64)[:, None, None]
    field = jnp.zeros((3,) + shape, dtype=jnp.float64).at[0].set(x)
    wave = EMVariables(field, field, field, field)
    bulk = EMVariables(*(7.0 * jnp.ones_like(field) for _ in range(4)))
    result = apply_planar_sommerfeld_boundaries(
        wave, bulk, bssn, geometry, params
    )

    for result_field in result:
        np.testing.assert_allclose(result_field[0, 0], 2.5, atol=1.0e-13)
        np.testing.assert_allclose(result_field[0, -1], -1.5, atol=1.0e-13)
        np.testing.assert_allclose(result_field[:, 1:-1], 7.0, atol=0.0)


def test_edge_uses_one_normalized_union_characteristic():
    shape = (6, 6, 1)
    dx = 0.2
    params = BSSNParameters(
        dx=dx,
        dt=0.01,
        nu=0.0,
        xl_bc=PERIODIC_BC,
        xr_bc=SOMMERFELD_BC,
        yl_bc=PERIODIC_BC,
        yr_bc=SOMMERFELD_BC,
        zl_bc=PERIODIC_BC,
        zr_bc=PERIODIC_BC,
    )
    bssn = flat_bssn_variables(shape)
    geometry = compute_bssn_em_geometry(
        bssn, zero_bssn_rhs(bssn), params
    )

    x = dx * jnp.arange(shape[0], dtype=jnp.float64)[:, None, None]
    y = dx * jnp.arange(shape[1], dtype=jnp.float64)[None, :, None]
    scalar = jnp.broadcast_to(x + y, shape)
    field = jnp.zeros((3,) + shape, dtype=jnp.float64).at[2].set(scalar)
    wave = EMVariables(field, field, field, field)
    zero_rhs = EMVariables(*(jnp.zeros_like(field) for _ in range(4)))
    result = apply_planar_sommerfeld_boundaries(
        wave, zero_rhs, bssn, geometry, params
    )

    expected = -np.sqrt(2.0)
    for result_field in result:
        np.testing.assert_allclose(
            result_field[2, -1, -1, 0], expected, rtol=0.0, atol=2.0e-13
        )


def test_periodic_boundaries_do_not_replace_rhs():
    shape = (8, 2, 2)
    params = BSSNParameters(dx=0.1, dt=0.01, nu=0.0)
    bssn = flat_bssn_variables(shape)
    geometry = compute_bssn_em_geometry(
        bssn, zero_bssn_rhs(bssn), params
    )
    field = jnp.ones((3,) + shape, dtype=jnp.float64)
    wave = EMVariables(field, field, field, field)
    bulk = EMVariables(*(3.0 * field for _ in range(4)))

    result = apply_planar_sommerfeld_boundaries(
        wave, bulk, bssn, geometry, params
    )
    for result_field, bulk_field in zip(result, bulk):
        np.testing.assert_array_equal(result_field, bulk_field)


def test_coupled_rk4_keeps_flat_zero_state_stationary():
    shape = (8, 1, 1)
    params = BSSNParameters(dx=0.1, dt=0.01, nu=0.0)
    bssn = flat_bssn_variables(shape)
    vector_zero = jnp.zeros((3,) + shape, dtype=jnp.float64)
    state = EinsteinMaxwellVariables(
        bssn=bssn,
        em=EMVariables(
            vector_zero, vector_zero, vector_zero, vector_zero
        ),
    )

    evolved = einstein_maxwell_rk4_step(state, params)
    jax.block_until_ready(evolved)

    for initial_field, evolved_field in zip(state.bssn, evolved.bssn):
        np.testing.assert_allclose(evolved_field, initial_field, atol=1.0e-13)
    for evolved_field in evolved.em:
        np.testing.assert_allclose(evolved_field, 0.0, atol=1.0e-13)


def test_cartesian_zero_em_step_matches_vacuum_bssn_step():
    grid_size = 8
    dx = 1.0 / grid_size
    x = jnp.linspace(-0.5, 0.5, grid_size, endpoint=False)[:, None, None]
    transverse_zero = jnp.zeros_like(x)
    bssn = periodic_gauge_wave_state(
        x,
        transverse_zero,
        transverse_zero,
        dx,
        amplitude=0.02,
    )
    params = BSSNParameters(
        dx=dx,
        dt=1.0e-4,
        nu=0.0,
        kappa=0.0,
        eta=0.0,
        g=0.0,
    )
    state = EinsteinMaxwellVariables(
        bssn=bssn,
        em=zero_em_variables((grid_size, 1, 1)),
    )

    coupled = einstein_maxwell_rk4_step(state, params)
    vacuum = rk4_step(bssn, params)
    jax.block_until_ready((coupled, vacuum))

    for actual, expected in zip(coupled.bssn, vacuum):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-14)
    for field in coupled.em:
        np.testing.assert_allclose(field, 0.0, rtol=0.0, atol=2.0e-14)
