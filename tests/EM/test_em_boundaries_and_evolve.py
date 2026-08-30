import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.evolution.boundaries import PERIODIC_BC, SOMMERFELD_BC

from JAX_BSSN.EM.boundaries import apply_planar_sommerfeld_boundaries
from JAX_BSSN.EM.equations import compute_em_rhs
from JAX_BSSN.EM.evolve import einstein_maxwell_rk4_step
from JAX_BSSN.EM.geometry import compute_bssn_em_geometry
from tests.EM.em_helpers import flat_bssn_variables, zero_bssn_rhs
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


def _wave_rk4_step(wave, bssn, bssn_rhs, params):
    dt = params.dt
    k1 = compute_em_rhs(wave, bssn, bssn_rhs, params)
    midpoint = jax.tree_util.tree_map(
        lambda value, rhs: value + 0.5 * dt * rhs, wave, k1
    )

    k2 = compute_em_rhs(midpoint, bssn, bssn_rhs, params)
    midpoint = jax.tree_util.tree_map(
        lambda value, rhs: value + 0.5 * dt * rhs, wave, k2
    )

    k3 = compute_em_rhs(midpoint, bssn, bssn_rhs, params)
    endpoint = jax.tree_util.tree_map(
        lambda value, rhs: value + dt * rhs, wave, k3
    )

    k4 = compute_em_rhs(endpoint, bssn, bssn_rhs, params)
    return jax.tree_util.tree_map(
        lambda value, rhs1, rhs2, rhs3, rhs4: value
        + dt * (rhs1 + 2.0 * rhs2 + 2.0 * rhs3 + rhs4) / 6.0,
        wave,
        k1,
        k2,
        k3,
        k4,
    )


@jax.jit
def _evolve_wave(wave, bssn, bssn_rhs, params, num_steps):
    return jax.lax.fori_loop(
        0,
        num_steps,
        lambda _, state: _wave_rk4_step(
            state, bssn, bssn_rhs, params
        ),
        wave,
    )


def _periodic_wave_error(grid_size):
    length = 2.0 * np.pi
    dx = length / grid_size
    dt = 0.1 * dx
    num_steps = round(0.25 / dt)
    final_time = num_steps * dt
    shape = (grid_size, 1, 1)
    x = dx * jnp.arange(grid_size, dtype=jnp.float64)

    profile = jnp.sin(x)[:, None, None]
    profile_dot = -jnp.cos(x)[:, None, None]
    E = jnp.zeros((3,) + shape, dtype=jnp.float64).at[1].set(profile)
    P = jnp.zeros_like(E).at[1].set(profile_dot)
    H = jnp.zeros_like(E).at[2].set(profile)
    Q = jnp.zeros_like(E).at[2].set(profile_dot)
    wave = EMVariables(E, P, H, Q)

    bssn = flat_bssn_variables(shape)
    bssn_rhs = zero_bssn_rhs(bssn)
    params = BSSNParameters(dx=dx, dt=dt, nu=0.0)
    evolved = _evolve_wave(wave, bssn, bssn_rhs, params, num_steps)
    jax.block_until_ready(evolved)

    expected = jnp.sin(x - final_time)
    error = jnp.sqrt(
        jnp.mean((evolved.electric_field[1, :, 0, 0] - expected) ** 2)
    )
    return float(error)


def test_periodic_plane_wave_converges_at_second_order():
    coarse = _periodic_wave_error(24)
    fine = _periodic_wave_error(48)

    assert coarse / fine > 3.7

