import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.evolution.time_evolve import rk4_step

from JAX_BSSN.EM.evolve import (
    compute_einstein_maxwell_rhs,
    einstein_maxwell_rk4_step,
    evolve_prescribed_em_steps,
)
from JAX_BSSN.EM.flrw import FLRWParameters, flrw_background, flrw_exact_fields
from JAX_BSSN.EM.variables import EMVariables, EinsteinMaxwellVariables
from tests.EM.em_helpers import flat_bssn_variables, zero_em_variables
from tests.initial_data import periodic_gauge_wave_state


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


def _duality_transform(em):
    return EMVariables(
        electric_field=em.magnetic_field,
        electric_field_dot=em.magnetic_field_dot,
        magnetic_field=-em.electric_field,
        magnetic_field_dot=-em.electric_field_dot,
    )


def test_synchronized_coupled_rhs_preserves_maxwell_duality_and_self_stress():
    grid_size = 8
    shape = (grid_size, 1, 1)
    dx = 2.0 * jnp.pi / grid_size
    x = dx * jnp.arange(grid_size, dtype=jnp.float64)
    sine = jnp.sin(x)[:, None, None]
    cosine = jnp.cos(x)[:, None, None]

    electric_field = jnp.zeros((3,) + shape, dtype=jnp.float64)
    electric_field = electric_field.at[1].set(0.02 * sine)
    electric_field_dot = jnp.zeros_like(electric_field)
    electric_field_dot = electric_field_dot.at[1].set(-0.02 * cosine)
    magnetic_field = jnp.zeros_like(electric_field)
    magnetic_field = magnetic_field.at[2].set(0.03 * cosine)
    magnetic_field_dot = jnp.zeros_like(electric_field)
    magnetic_field_dot = magnetic_field_dot.at[2].set(0.03 * sine)
    em = EMVariables(
        electric_field,
        electric_field_dot,
        magnetic_field,
        magnetic_field_dot,
    )
    bssn = flat_bssn_variables(shape)
    params = BSSNParameters(
        dx=float(dx),
        dt=1.0e-3,
        nu=0.0,
        kappa=0.0,
        eta=0.0,
        g=0.0,
        zero_shift=1,
    )

    rhs = compute_einstein_maxwell_rhs(
        EinsteinMaxwellVariables(bssn=bssn, em=em), params
    )
    dual_rhs = compute_einstein_maxwell_rhs(
        EinsteinMaxwellVariables(bssn=bssn, em=_duality_transform(em)),
        params,
    )
    expected_dual_em_rhs = _duality_transform(rhs.em)
    jax.block_until_ready((rhs, dual_rhs))

    for actual, expected in zip(dual_rhs.bssn, rhs.bssn):
        np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=2.0e-13)
    for actual, expected in zip(dual_rhs.em, expected_dual_em_rhs):
        np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=2.0e-13)

    # A nonzero trace-K source confirms that the invariant EM self-stress is
    # present in the synchronized BSSN stage, rather than only in diagnostics.
    assert float(jnp.max(jnp.abs(rhs.bssn.trace_K))) > 0.0


def _flrw_spatial_error(grid_size):
    shape = (1, 1, grid_size)
    dx = 1.0 / grid_size
    dt = 0.1 * dx
    num_steps = grid_size // 2
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

    squared_error = jnp.mean(
        (evolved.electric_field - exact.electric_field) ** 2
    )
    squared_error += jnp.mean(
        (evolved.magnetic_field - exact.magnetic_field) ** 2
    )
    return float(jnp.sqrt(squared_error))


def test_flrw_plane_wave_has_second_order_spatial_convergence():
    coarse_error = _flrw_spatial_error(24)
    fine_error = _flrw_spatial_error(48)
    order = math.log2(coarse_error / fine_error)

    print(
        f"FLRW spatial convergence: coarse={coarse_error:.8e}, "
        f"fine={fine_error:.8e}, order={order:.6f}"
    )
    assert order > 1.8


def _homogeneous_flrw_temporal_error(dt):
    shape = (1, 1, 1)
    final_time = 0.4
    num_steps = round(final_time / dt)
    solver_params = BSSNParameters(dx=1.0, dt=dt, nu=0.0)
    params = FLRWParameters(
        H=0.5,
        length_z=1.0,
        mode=0,
        electric_amplitude=1.0,
    )

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

    squared_error = sum(
        jnp.mean((numerical - reference) ** 2)
        for numerical, reference in zip(evolved, exact)
    )
    return float(jnp.sqrt(squared_error))


def test_prescribed_em_rk4_has_fourth_order_temporal_convergence():
    coarse_error = _homogeneous_flrw_temporal_error(0.05)
    medium_error = _homogeneous_flrw_temporal_error(0.025)
    fine_error = _homogeneous_flrw_temporal_error(0.0125)
    coarse_order = math.log2(coarse_error / medium_error)
    fine_order = math.log2(medium_error / fine_error)

    print(
        f"EM temporal convergence: errors=({coarse_error:.8e}, "
        f"{medium_error:.8e}, {fine_error:.8e}), "
        f"orders=({coarse_order:.6f}, {fine_order:.6f})"
    )
    assert coarse_order > 3.7
    assert fine_order > 3.7
