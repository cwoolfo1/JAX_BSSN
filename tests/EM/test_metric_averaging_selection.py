"""Original arithmetic transfer through initialization and changing geometry."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.EM.first_order.coupling import quadratic_moments, sources_from_moments
from JAX_BSSN.EM.first_order.evolve import first_order_einstein_maxwell_step
from JAX_BSSN.EM.first_order.geometry import build_yee_metric
from JAX_BSSN.EM.first_order.grid import make_em_grid
from JAX_BSSN.EM.first_order.maxwell import initialize_native_state
from demos.single_charged_blackhole.run_single_charged_blackhole import (
    SingleChargedBlackHoleConfig, build_single_charged_blackhole, state_is_valid,
)
from demos.single_charged_blackhole.resolution_restart import make_staged_step
from tests.EM.em_helpers import flat_bssn_variables


def assert_tree_close(actual, expected):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for a, b in zip(jax.tree_util.tree_leaves(actual),
                    jax.tree_util.tree_leaves(expected)):
        assert jnp.shape(a) == jnp.shape(b)
        assert bool(jnp.allclose(a, b, rtol=3e-12, atol=3e-13))


@pytest.mark.parametrize("boundary", ("periodic", "constant"))
def test_production_averages_bssn_variables_before_metric_projection(boundary):
    params = BSSNParameters(dx=.3, dt=.02)
    grid = make_em_grid((7, 8, 6), params, boundary_conditions=(boundary,)*3)
    x, y, z = grid.coordinates(params)
    bssn = flat_bssn_variables(grid.shape)
    metric = bssn.conformal_metric
    off_diagonal = jnp.broadcast_to(.025*jnp.sin(x+2*y-z), grid.shape)
    metric = metric.at[0, 1].set(off_diagonal).at[1, 0].set(off_diagonal)
    bssn = bssn._replace(
        conformal_factor=jnp.broadcast_to(.8+.04*jnp.cos(x+y+z), grid.shape),
        conformal_metric=metric,
        lapse=jnp.broadcast_to(.9+.06*jnp.cos(2*x-y+z), grid.shape),
        shift=jnp.stack(tuple(jnp.broadcast_to(value, grid.shape) for value in (
            .13*jnp.sin(2*x+y), -.08*jnp.cos(y-z), .05*jnp.sin(x+z),
        ))),
    )
    native = build_yee_metric(bssn, grid)
    compiled = jax.jit(lambda value: build_yee_metric(value, grid))(bssn)
    assert_tree_close(native, compiled)

    # A VVV site lies at the center of these eight BSSN samples. Average W
    # and the conformal metric first, then project and form physical geometry.
    block = (slice(2, 4), slice(3, 5), slice(1, 3))
    w = jnp.mean(bssn.conformal_factor[block])
    conformal = jnp.mean(bssn.conformal_metric[(Ellipsis,) + block], axis=(-3, -2, -1))
    conformal = conformal / jnp.linalg.det(conformal)**(1 / 3)
    index = (0, 0, 0, grid.guard_cells+2, grid.guard_cells+3, grid.guard_cells+1)
    expected = (
        jnp.mean(bssn.lapse[block]),
        jnp.mean(bssn.shift[(Ellipsis,) + block], axis=(-3, -2, -1)),
        conformal / w**2,
        jnp.linalg.inv(conformal) * w**2,
        w**-3,
    )
    actual = tuple(value[index] for value in native.vertex)
    assert_tree_close(actual, expected)
    # Volume averaging would give a different result for this varying W.
    assert float(jnp.abs(w**-3 - jnp.mean(bssn.conformal_factor[block]**-3))) > 1e-6
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in jax.tree_util.tree_leaves(native))


@pytest.fixture(scope="module")
def arithmetic_puncture():
    config = SingleChargedBlackHoleConfig(
        num_points=12, half_width=3., absorber_width=.75, final_time=.02,
        platform="cpu",
    )
    state, params, grid, _ = build_single_charged_blackhole(config)
    return config, state, params, grid


def test_demo_initializes_native_histories_with_arithmetic_metric(arithmetic_puncture):
    _, state, params, grid = arithmetic_puncture
    explicit = build_yee_metric(state.bssn, grid)
    expected = initialize_native_state(
        state.em.fields[0], state.em.synchronized_magnetic,
        explicit, params, grid, state.em.absorber,
    )
    assert_tree_close(state.em, expected)

    # The puncture has varying geometry and a nonzero native magnetic
    # half-step even though the common-time initial magnetic field vanishes.
    assert float(jnp.max(jnp.abs(explicit.D[0].sqrt_gamma-explicit.center.sqrt_gamma))) > 1e-4
    assert max(float(jnp.max(jnp.abs(a-b))) for a, b in zip(
        state.em.fields[1], state.em.synchronized_magnetic)) > 1e-8


def test_arithmetic_staged_step_matches_production_with_changing_shift(
        arithmetic_puncture, record_property):
    _, state, params, grid = arithmetic_puncture
    x, y, z = grid.coordinates(params)
    shift = jnp.stack(tuple(jnp.broadcast_to(value, grid.shape) for value in (
        .06*jnp.sin(x+.3*y), .04*jnp.sin(y-.2*z), -.05*jnp.sin(z+.4*x),
    )))
    bssn = state.bssn._replace(shift=shift)
    metric = build_yee_metric(bssn, grid)
    em = initialize_native_state(
        state.em.fields[0], state.em.synchronized_magnetic,
        metric, params, grid, state.em.absorber,
    )
    initial = state._replace(bssn=bssn, em=em)
    staged_step = make_staged_step(params, grid)
    staged, production = initial, initial

    for _ in range(2):
        staged = staged_step(staged)
        production = first_order_einstein_maxwell_step(production, params, grid)
        # Includes D, native B, both retained histories, synchronized B, all
        # native metric components, and the fixed reference sponge arrays.
        assert_tree_close(staged, production)
        for evolved in (staged, production):
            expected = build_yee_metric(evolved.bssn, grid)
            assert_tree_close(evolved.em.fields[6], expected)
        source_states = []
        for evolved in (staged, production):
            moments = quadratic_moments(
                evolved.em.fields[0], evolved.em.synchronized_magnetic, grid)
            source_states.append(sources_from_moments(moments, evolved.bssn))
        assert_tree_close(*source_states)

    assert bool(state_is_valid(staged))
    shift_change = float(jnp.max(jnp.abs(staged.bssn.shift-initial.bssn.shift)))
    metric_change = float(jnp.max(jnp.abs(
        staged.bssn.conformal_metric-initial.bssn.conformal_metric)))
    assert shift_change > 1e-8
    assert metric_change > 1e-8
    record_property("maximum_shift_change", shift_change)
    record_property("maximum_conformal_metric_change", metric_change)
    record_property("maximum_synchronized_B_difference", max(
        float(jnp.max(jnp.abs(a-b))) for a, b in zip(
            staged.em.synchronized_magnetic, production.em.synchronized_magnetic)))
