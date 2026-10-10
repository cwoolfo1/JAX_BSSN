"""PyPIC3D adapter, geometry, sources, and coupled stepping contracts."""

from unittest.mock import patch
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS
from PyPIC3D.solvers.GR_yee.time_loop import time_loop_static_metric
from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.evolution.time_evolve import rk4_step
from JAX_BSSN.EM.first_order import *
from JAX_BSSN.EM.first_order.geometry import build_yee_metric
from JAX_BSSN.EM.first_order.maxwell import empty_particles, refresh_state_metric
from tests.EM.em_helpers import flat_bssn_variables


def assert_tree_close(a, b, atol=2e-12):
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_allclose(x, y, atol=atol, rtol=2e-12)


def fixture_state(shape=(8, 1, 1), boundaries=("periodic",) * 3):
    p = BSSNParameters(dx=0.2, dt=0.004, nu=0, zero_shift=1)
    grid = make_em_grid(shape, p, boundary_conditions=boundaries)
    bssn = flat_bssn_variables(shape)
    return p, grid, bssn


@pytest.mark.parametrize("shape", [(8, 1, 1), (5, 6, 7)])
def test_native_coordinates_and_layout(shape):
    p, g, s = fixture_state(shape)
    a = jnp.arange(np.prod(shape)).reshape(shape)
    np.testing.assert_array_equal(g.from_tile(g.to_tile(a)), a)
    for loc in D_FIELD_LOCATIONS + B_FIELD_LOCATIONS:
        for axis, values in enumerate(g.coordinates(p, loc)):
            np.testing.assert_allclose(
                values.ravel(),
                p.dx * (np.arange(shape[axis]) + (0.5 if loc[axis] == "V" else 0)),
            )


def test_metric_native_projection_and_volumes():
    p, g, s = fixture_state((8, 5, 6))
    x, y, z = g.coordinates(p)
    W = jnp.broadcast_to(0.8 + 0.05 * jnp.sin(2 * jnp.pi * x / (8 * p.dx)), g.shape)
    raw = jnp.array([[1.4, 0.2, 0.1], [0.2, 1.1, -0.1], [0.1, -0.1, 0.9]])
    s = s._replace(
        conformal_factor=W,
        conformal_metric=jnp.broadcast_to(
            raw[:, :, None, None, None], (3, 3) + g.shape
        ),
        shift=jnp.broadcast_to(
            jnp.array([0.1, -0.2, 0.05])[:, None, None, None], (3,) + g.shape
        ),
    )
    m = build_yee_metric(s, g)
    for q in m.D + m.B + (m.center, m.vertex):
        np.testing.assert_allclose(jnp.linalg.det(q.gamma), q.sqrt_gamma**2, rtol=2e-13)
        identity = jnp.einsum("...ij,...jk->...ik", q.gamma, q.gamma_inv)
        np.testing.assert_allclose(
            identity, jnp.broadcast_to(jnp.eye(3), identity.shape), atol=1e-13
        )
    np.testing.assert_allclose(g.from_tile(m.center.sqrt_gamma), W**-3)
    assert not np.allclose(m.D[0].sqrt_gamma, m.D[1].sqrt_gamma)
    fields = jnp.ones((3,) + g.shape)
    initial = initialize_first_order_einstein_maxwell_state(s, fields, fields, p, g)
    for d, q in zip(initial.em.fields[0], initial.em.fields[6].D):
        np.testing.assert_allclose(d, q.sqrt_gamma, atol=2e-13)


@pytest.mark.parametrize(
    "boundaries", [("periodic",) * 3, ("conducting", "periodic", "periodic")]
)
def test_half_step_is_exact_imported_vacuum_loop(boundaries):
    p, g, s = fixture_state((9, 5, 1), boundaries)
    raw = jnp.array([[1.2, 0.1, 0], [0.1, 1.0, 0.05], [0, 0.05, 0.9]])
    s = s._replace(
        conformal_metric=jnp.broadcast_to(
            raw[:, :, None, None, None], (3, 3) + g.shape
        ),
        conformal_factor=jnp.full(g.shape, 0.9),
        shift=jnp.full((3,) + g.shape, 0.07),
    )
    rng = np.random.default_rng(42)
    d, b = (jnp.asarray(rng.normal(size=(3,) + g.shape)) for _ in range(2))
    initial = initialize_first_order_einstein_maxwell_state(s, d, b, p, g)
    particles, species = empty_particles(d.dtype)
    before = initial.em.fields
    _, expected = time_loop_static_metric(
        particles, species, before, g.static, g.dynamic(p, p.dt / 2)
    )
    with patch(
        "PyPIC3D.solvers.GR_yee.time_loop.hybrid_boris_geodesic_push",
        side_effect=AssertionError("particle push"),
    ):
        result = maxwell_half_step(initial.em, p, g)
    assert_tree_close(result.fields, expected)
    assert_tree_close(result.fields[7], before[:2])
    assert_tree_close(initial.em.fields, before)
    assert any(
        not np.allclose(a, b)
        for a, b in zip(result.synchronized_magnetic, result.fields[1])
    )


def test_metric_replacement_preserves_owned_densities_and_history():
    p, g, s = fixture_state()
    d = jnp.ones((3,) + g.shape)
    b = 2 * d
    initial = initialize_first_order_einstein_maxwell_state(s, d, b, p, g)
    new_metric = build_yee_metric(
        s._replace(conformal_factor=jnp.full(g.shape, 0.7)), g
    )
    changed = refresh_state_metric(initial.em, new_metric, g)
    for v, w in zip(
        changed.fields[:2] + changed.fields[7],
        initial.em.fields[:2] + initial.em.fields[7],
    ):
        for a, c in zip(v, w):
            np.testing.assert_array_equal(g.from_tile(a), g.from_tile(c))
    evolved = maxwell_half_step(changed, p, g)
    assert_tree_close(evolved.fields[0], changed.fields[0])
    assert_tree_close(evolved.fields[1], changed.fields[1])


def test_zero_fields_reduce_to_vacuum_rk4_and_scan():
    p, g, s = fixture_state((8, 1, 1))
    x = g.coordinates(p)[0]
    s = s._replace(lapse=jnp.broadcast_to(1 + 0.01 * jnp.sin(x), g.shape))
    zero = jnp.zeros((3,) + g.shape)
    initial = initialize_first_order_einstein_maxwell_state(s, zero, zero, p, g)

    @jax.jit
    def run(state):
        return jax.lax.scan(
            lambda s, i: (
                first_order_einstein_maxwell_step(s, p, g, time=i * p.dt),
                None,
            ),
            state,
            jnp.arange(2),
        )[0]

    final = run(initial)
    assert_tree_close(final.bssn, rk4_step(rk4_step(initial.bssn, p), p))
    assert_tree_close(common_physical_fields(final, g), (zero, zero))


def test_periodic_density_constraints_survive_coupled_steps():
    p, g, s = fixture_state((8, 1, 8))
    x, _, z = g.coordinates(p)
    d = (
        jnp.zeros((3,) + g.shape)
        .at[1]
        .set(
            0.02
            * jnp.sin(2 * jnp.pi * x / (8 * p.dx))
            * jnp.cos(2 * jnp.pi * z / (8 * p.dx))
        )
    )
    initial = initialize_first_order_einstein_maxwell_state(
        s, d, jnp.zeros_like(d), p, g
    )
    result = jax.lax.fori_loop(
        0, 3, lambda i, s: first_order_einstein_maxwell_step(s, p, g), initial
    )
    for value in first_order_constraint_divergences(result, p, g):
        np.testing.assert_allclose(value, 0, atol=2e-13)


@pytest.mark.parametrize(
    "boundaries", [("sommerfeld", "periodic", "periodic"), (1, 0, 0)]
)
def test_legacy_em_boundary_codes_are_rejected(boundaries):
    with pytest.raises(ValueError, match="Sommerfeld"):
        make_em_grid((8, 1, 1), BSSNParameters(), boundary_conditions=boundaries)


def test_metric_switch_rephases_history_without_changing_common_fields():
    from JAX_BSSN.EM.first_order.maxwell import rephase_state_metric

    p, grid, bssn = fixture_state()
    x = grid.coordinates(p)[0]
    zero = jnp.zeros((3,) + grid.shape)
    d = zero.at[1].set(jnp.sin(2 * jnp.pi * x / (grid.shape[0] * p.dx)))
    b = zero.at[2].set(jnp.cos(2 * jnp.pi * (x + p.dx / 2) / (grid.shape[0] * p.dx)))
    initial = initialize_first_order_einstein_maxwell_state(bssn, d, b, p, grid)
    identity = rephase_state_metric(initial.em, initial.em.fields[6], p, grid)
    assert_tree_close(identity, initial.em)
    changed = build_yee_metric(bssn._replace(lapse=bssn.lapse + 0.05), grid)
    shifted = rephase_state_metric(initial.em, changed, p, grid)
    assert_tree_close(
        common_densitized_fields(shifted), common_densitized_fields(initial.em)
    )
    assert not np.allclose(shifted.fields[1][2], initial.em.fields[1][2])
    assert not np.allclose(shifted.fields[7][0][1], initial.em.fields[7][0][1])
    assert_tree_close(shifted.fields[2:6], initial.em.fields[2:6])


def test_native_metric_interpolation_is_second_order():
    errors = []
    for n in (16, 32, 64):
        shape = (n, 1, 1)
        p = BSSNParameters(dx=2 * np.pi / n)
        grid = make_em_grid(shape, p)
        x = grid.coordinates(p)[0]
        bssn = flat_bssn_variables(shape)._replace(
            conformal_factor=0.8 + 0.05 * jnp.sin(x), lapse=1 + 0.1 * jnp.cos(x)
        )
        metric = build_yee_metric(bssn, grid)
        target = grid.coordinates(p, D_FIELD_LOCATIONS[0])[0]
        error = jnp.mean(
            (
                grid.from_tile(metric.D[0].sqrt_gamma)
                - (0.8 + 0.05 * jnp.sin(target)) ** -3
            )
            ** 2
        )
        error += jnp.mean(
            (grid.from_tile(metric.D[0].lapse) - (1 + 0.1 * jnp.cos(target))) ** 2
        )
        errors.append(float(jnp.sqrt(error)))
    assert np.all(np.asarray(errors[:-1]) / errors[1:] > 3.8), errors
