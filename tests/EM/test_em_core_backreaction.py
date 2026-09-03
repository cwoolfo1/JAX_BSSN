import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.evolution.time_evolve import compute_bssn_rhs_with_matter
from JAX_BSSN.EM.second_order.energy_momentum import (
    compute_electromagnetic_energy_momentum,
)
from JAX_BSSN.EM.second_order.equations import compute_em_rhs
from JAX_BSSN.EM.second_order.evolve import (
    compute_einstein_maxwell_rhs,
    einstein_maxwell_rk4_step,
)
from JAX_BSSN.EM.second_order.geometry import compute_bssn_em_geometry
from JAX_BSSN.EM.second_order.variables import EinsteinMaxwellVariables, EMVariables
from tests.EM.em_helpers import flat_bssn_variables, zero_bssn_rhs


def _duality_transform(em):
    return EMVariables(
        electric_field=em.magnetic_field,
        electric_field_dot=em.magnetic_field_dot,
        magnetic_field=-em.electric_field,
        magnetic_field_dot=-em.electric_field_dot,
    )


def test_em_ricci_is_removed_and_electric_weyl_includes_stress():
    shape = (6, 4, 4)
    bssn = flat_bssn_variables(shape)
    electric_field = jnp.zeros((3,) + shape, dtype=jnp.float64)
    electric_field = electric_field.at[0].set(2.0)
    magnetic_field = jnp.zeros_like(electric_field)

    rho, momentum_density, spatial_stress = (
        compute_electromagnetic_energy_momentum(
            electric_field, magnetic_field, bssn
        )
    )
    # For Maxwell stress S=rho, so Einstein's equation gives R_nn=8 pi rho.
    bssn_rhs = zero_bssn_rhs(bssn)._replace(trace_K=8.0 * jnp.pi * rho)
    params = BSSNParameters(dx=0.1, dt=0.01, nu=0.0)
    geometry = compute_bssn_em_geometry(
        bssn,
        bssn_rhs,
        params,
        backreaction_sources=(rho, momentum_density, spatial_stress),
    )

    metric = bssn.conformal_metric
    anisotropic_stress = spatial_stress - metric * rho / 3.0
    np.testing.assert_allclose(
        geometry.normal_normal_ricci, 0.0, rtol=0.0, atol=1.0e-13
    )
    np.testing.assert_allclose(
        geometry.electric_weyl,
        -4.0 * jnp.pi * anisotropic_stress,
        rtol=2.0e-15,
        atol=2.0e-14,
    )


def test_geometry_is_finite_at_W_and_lapse_floors():
    shape = (4, 4, 4)
    bssn = flat_bssn_variables(shape)._replace(
        conformal_factor=jnp.zeros(shape, dtype=jnp.float64),
        lapse=jnp.zeros(shape, dtype=jnp.float64),
    )
    params = BSSNParameters(dx=0.1, dt=0.01, nu=0.0)
    geometry = compute_bssn_em_geometry(
        bssn, zero_bssn_rhs(bssn), params
    )

    for field in geometry:
        assert bool(jnp.all(jnp.isfinite(field)))


def test_nonzero_stage_uses_synchronized_two_way_sources():
    shape = (8, 4, 4)
    bssn = flat_bssn_variables(shape)
    electric_field = jnp.zeros((3,) + shape, dtype=jnp.float64)
    electric_field = electric_field.at[0].set(0.2)
    magnetic_field = jnp.zeros_like(electric_field)
    magnetic_field = magnetic_field.at[1].set(0.1)
    vector_zero = jnp.zeros_like(electric_field)
    state = EinsteinMaxwellVariables(
        bssn=bssn,
        em=EMVariables(
            electric_field,
            vector_zero,
            magnetic_field,
            vector_zero,
        ),
    )
    params = BSSNParameters(
        dx=0.2,
        dt=1.0e-3,
        nu=0.0,
        kappa=0.0,
        zero_shift=1,
    )

    rho, momentum_density, spatial_stress = (
        compute_electromagnetic_energy_momentum(
            electric_field, magnetic_field, bssn
        )
    )
    expected_bssn_rhs = compute_bssn_rhs_with_matter(
        bssn,
        params,
        rho,
        momentum_density,
        spatial_stress,
    )
    expected_em_rhs = compute_em_rhs(
        state.em,
        bssn,
        expected_bssn_rhs,
        params,
        backreaction_sources=(rho, momentum_density, spatial_stress),
    )
    rhs = compute_einstein_maxwell_rhs(state, params)

    for actual, expected in zip(rhs.bssn, expected_bssn_rhs):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    for actual, expected in zip(rhs.em, expected_em_rhs):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    evolved = einstein_maxwell_rk4_step(state, params)
    assert float(jnp.max(jnp.abs(evolved.bssn.trace_K))) > 0.0
    assert float(jnp.max(jnp.abs(evolved.bssn.traceless_K))) > 0.0
    assert float(jnp.max(jnp.abs(evolved.bssn.conformal_connection))) > 0.0
    for field in (*evolved.bssn, *evolved.em):
        assert bool(jnp.all(jnp.isfinite(field)))


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
        np.testing.assert_allclose(
            actual, expected, rtol=2.0e-13, atol=2.0e-13
        )
    for actual, expected in zip(dual_rhs.em, expected_dual_em_rhs):
        np.testing.assert_allclose(
            actual, expected, rtol=2.0e-13, atol=2.0e-13
        )

    # A nonzero trace-K source confirms that the invariant EM self-stress is
    # present in the synchronized BSSN stage, rather than only in diagnostics.
    assert float(jnp.max(jnp.abs(rhs.bssn.trace_K))) > 0.0
