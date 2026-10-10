"""PyPIC3D conducting boundaries through the BSSN metric/layout adapter."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from PyPIC3D.boundary_conditions.staggered import refresh_fields
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS
from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.EM.first_order import *
from tests.EM.em_helpers import flat_bssn_variables


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_conducting_endpoints_and_imported_projection(axis):
    shape = tuple(9 if a == axis else 1 for a in range(3))
    boundaries = tuple("conducting" if a == axis else "periodic" for a in range(3))
    p = BSSNParameters(dx=0.1, dt=0.001)
    g = make_em_grid(shape, p, boundary_conditions=boundaries)
    values = jnp.broadcast_to(
        jnp.array([1.0, 2.0, 3.0])[:, None, None, None], (3,) + shape
    )
    state = initialize_first_order_einstein_maxwell_state(
        flat_bssn_variables(shape), values, values, p, g
    )
    d, b = common_densitized_fields(state.em)
    for vector, locations, kind in (
        (d, D_FIELD_LOCATIONS, "D"),
        (b, B_FIELD_LOCATIONS, "B"),
    ):
        repeated = refresh_fields(vector, g.static, locations, kind, state.em.fields[6])
        for v, r in zip(vector, repeated):
            np.testing.assert_allclose(v, r, atol=2e-13)
        for i, value in enumerate(vector):
            if (kind == "D" and i != axis) or (kind == "B" and i == axis):
                physical = g.from_tile(value)
                np.testing.assert_allclose(
                    jnp.take(physical, jnp.array([0, shape[axis] - 1]), axis=axis),
                    0,
                    atol=2e-13,
                )
    assert g.tile_shape[axis] == shape[axis] - 1
    assert state.em.fields[0][0].shape[axis + 3] == shape[axis] - 1 + 6


def test_coupled_cavity_resolves_standing_wave():
    cells = 16
    shape = (cells + 1, 1, 1)
    p = BSSNParameters(dx=1 / cells, dt=0.005, nu=0, zero_shift=1)
    g = make_em_grid(
        shape, p, boundary_conditions=("conducting", "periodic", "periodic")
    )
    x = g.coordinates(p)[0]
    amplitude = 1e-5
    zero = jnp.zeros((3,) + shape)
    d = zero.at[1].set(amplitude * jnp.sin(jnp.pi * x))
    state = initialize_first_order_einstein_maxwell_state(
        flat_bssn_variables(shape), d, zero, p, g
    )
    final = jax.lax.fori_loop(
        0,
        40,
        lambda i, s: first_order_einstein_maxwell_step(s, p, g, time=i * p.dt),
        state,
    )
    d, b = common_densitized_fields(final.em)
    expected_d = amplitude * jnp.sin(jnp.pi * x) * jnp.cos(0.2 * jnp.pi)
    expected_b = -amplitude * jnp.cos(jnp.pi * (x + 0.5 * p.dx)) * jnp.sin(0.2 * jnp.pi)
    np.testing.assert_allclose(g.from_tile(d[1]), expected_d, atol=0.003 * amplitude)
    np.testing.assert_allclose(
        g.from_tile(b[2])[:-1], expected_b[:-1], atol=0.003 * amplitude
    )
    np.testing.assert_allclose(g.from_tile(d[1])[jnp.array([0, -1])], 0, atol=1e-14)
    assert all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(final))
