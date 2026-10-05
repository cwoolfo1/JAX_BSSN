"""Native PEC wall traces, coupled edges, and evolution staging."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.EM.first_order import evolve
from JAX_BSSN.EM.first_order import pec
from JAX_BSSN.EM.first_order.staggering import (
    DISPLACEMENT_FIELD_LOCATIONS as DL,
    MAGNETIC_FIELD_LOCATIONS as BL,
)
from JAX_BSSN.EM.first_order.variables import DensitizedMaxwellState
from JAX_BSSN.EM.variables import EinsteinMaxwellVariables
from tests.EM.em_helpers import flat_bssn_variables


def _geometry(shape, oblique=True, varying=True):
    bssn = flat_bssn_variables(shape)
    x, y, z = jnp.meshgrid(*(jnp.arange(n) * 0.13 for n in shape), indexing="ij")
    raw = jnp.array([[1.5, .3, -.12], [.3, 1.2, .17], [-.12, .17, .9]]) if oblique else jnp.eye(3)
    metric = jnp.broadcast_to(raw[:, :, None, None, None], (3, 3) + shape)
    if varying:
        metric = metric.at[0, 0].add(.05 * jnp.sin(x))
        metric = metric.at[1, 1].add(.04 * jnp.cos(y))
        metric = metric.at[2, 2].add(.03 * jnp.sin(z))
    determinant = jnp.linalg.det(jnp.moveaxis(metric, (0, 1), (-2, -1)))
    return bssn._replace(conformal_metric=metric / determinant**(1/3),
                         conformal_factor=.8 + .04 * jnp.sin(x + y + z) if varying else jnp.full(shape, .8))


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_diagonal_wall_locations_and_reflections(axis):
    shape = tuple(10 if a == axis else 1 for a in range(3))
    boundary = pec.PECBoundary((axis,), 2)
    bssn = _geometry(shape, oblique=False, varying=False)
    values = jnp.broadcast_to(jnp.array([1., 2., 3.])[:, None, None, None], (3,) + shape)
    with jax.disable_jit():
        d, b = pec.apply_pec_boundaries(values, values, bssn, BSSNParameters(), boundary=boundary)
    low, high = boundary.walls(shape, axis)
    for fields, locations, kind in ((d, DL, "D"), (b, BL, "B")):
        for component, location in enumerate(locations):
            parity = (1 if component == axis else -1) * (1 if kind == "D" else -1)
            if location[axis] == "C":
                for wall in (low, high):
                    np.testing.assert_array_equal(fields[component][pec._plane(axis, wall)], 0.)
            for _, node, owner in pec._images(shape, location, axis, boundary):
                np.testing.assert_allclose(fields[component][pec._plane(axis, node)],
                                           parity * fields[component][pec._plane(axis, owner)], atol=1.e-14)


@pytest.mark.parametrize("axes", ((0, 1), (0, 1, 2)))
def test_oblique_varying_metric_edges_are_simultaneous_and_idempotent(axes):
    shape = (10, 10, 10 if 2 in axes else 1)
    boundary = pec.PECBoundary(axes, 2)
    params = BSSNParameters()
    bssn = _geometry(shape)
    rng = np.random.default_rng(41)
    d, b = (jnp.asarray(rng.normal(size=(3,) + shape)) for _ in range(2))
    with jax.disable_jit():
        once = pec.apply_pec_boundaries(d, b, bssn, params, boundary=boundary)
        twice = pec.apply_pec_boundaries(*once, bssn, params, boundary=boundary)
        metric = pec._metric_samples(bssn, params)
        for result, repeated, original, locations, kind in zip(once, twice, (d, b), (DL, BL), ("D", "B")):
            np.testing.assert_allclose(repeated, result, atol=2.e-12, rtol=2.e-13)
            physical = tuple(result[i] / metric[loc][1] for i, loc in enumerate(locations))
            snapshot = pec._prepare_normal_D(physical, boundary) if kind == "D" else physical
            # Check the defining projector equations, independent of the edge solver.
            for component, location in enumerate(locations):
                vector = pec._reconstruct(snapshot, locations, metric, location, params)
                inverse = metric[location][0]
                for axis in axes:
                    if location[axis] != "C":
                        continue
                    face = jnp.broadcast_to(pec._face_mask(shape, axis, boundary), shape)
                    for other in axes:
                        if other != axis:
                            nodes = jnp.arange(shape[other]).reshape((1,) * other + (shape[other],) + (1,) * (2-other))
                            lo, hi = boundary.walls(shape, other)
                            face &= (nodes >= lo + int(location[other] == "V")) & (nodes <= hi)
                    expected = inverse[component, axis] / inverse[axis, axis] * vector[axis] if kind == "D" else 0.
                    if kind == "D":
                        for other in axes:
                            if other != axis and location[other] == "C":
                                expected = jnp.where(pec._face_mask(shape, other, boundary), 0., expected)
                    np.testing.assert_allclose(np.asarray(physical[component])[face],
                                               np.broadcast_to(expected, shape)[face], atol=2.e-12)
            interior = (slice(None),) + tuple(slice(3, 7) if a in axes else slice(None) for a in range(3))
            np.testing.assert_array_equal(result[interior], original[interior])


def test_oblique_normal_displacement_and_shift_contract():
    shape = (10, 1, 1)
    boundary = pec.PECBoundary((0,), 2)
    params = BSSNParameters()
    bssn = _geometry(shape, varying=False)
    d = jnp.broadcast_to(jnp.array([2., 7., 8.])[:, None, None, None], (3,) + shape)
    b = jnp.broadcast_to(jnp.array([3., 0., 4.])[:, None, None, None], (3,) + shape)
    # Exercise the compiled public port, with both vector fields nonzero.
    result = pec.apply_pec_boundaries(d, b, bssn, params, boundary=boundary)
    shifted = pec.apply_pec_boundaries(d, b, bssn._replace(shift=bssn.shift.at[0].set(.3)), params, boundary=boundary)
    inverse = np.linalg.inv(np.asarray(bssn.conformal_metric[:, :, 0, 0, 0]))
    for wall in boundary.walls(shape, 0):
        np.testing.assert_allclose(result[0][:, wall, 0, 0], 2 * inverse[:, 0] / inverse[0, 0], atol=1.e-13)
        assert float(result[1][0, wall, 0, 0]) == 0.
    for actual, expected in zip(shifted, result):
        np.testing.assert_array_equal(actual, expected)


def test_pec_validation_and_disabled_identity():
    for args in (((0, 0), 2), ((3,), 2), ((0,), 1)):
        with pytest.raises(ValueError):
            pec.PECBoundary(*args)
    with pytest.raises(ValueError, match="more physical cells"):
        pec.PECBoundary((0,), 2).validate((7, 1, 1))
    shape = (4, 1, 1)
    d = jnp.ones((3,) + shape)
    bssn = flat_bssn_variables(shape)
    for result in pec.apply_pec_boundaries(d, d, bssn, BSSNParameters(), boundary=pec.PECBoundary(())):
        np.testing.assert_array_equal(result, d)


def test_coupler_uses_four_gravity_stages_and_stage_metric_for_pec(monkeypatch):
    shape = (10, 1, 1)
    boundary = pec.PECBoundary((0,), 2)
    params = BSSNParameters(dt=.02, nu=0.)
    bssn = _geometry(shape, varying=False)
    values = jnp.broadcast_to(jnp.array([2., 7., 8.])[:, None, None, None], (3,) + shape)
    calls = []
    stage_derivatives = []

    def gravity(s, d, b, p):
        calls.append(float(s.trace_K[0, 0, 0]))
        for actual, expected in zip((d, b), pec.apply_pec_boundaries(d, b, s, p, boundary=boundary)):
            np.testing.assert_allclose(actual, expected, atol=1.e-12)
        rhs = jax.tree_util.tree_map(jnp.zeros_like, s)
        metric_dot = .1 * (1. + s.trace_K)
        rhs = rhs._replace(trace_K=jnp.ones_like(s.trace_K),
                           conformal_metric=rhs.conformal_metric.at[0, 1].set(metric_dot).at[1, 0].set(metric_dot))
        stage_derivatives.append(rhs)
        return rhs

    monkeypatch.setattr(evolve, "_bssn_rhs_from_densities", gravity)
    monkeypatch.setattr(evolve, "densitized_maxwell_rhs", lambda d, b, s, p: (jnp.zeros_like(d), jnp.zeros_like(b)))
    with jax.disable_jit():
        state = EinsteinMaxwellVariables(bssn, DensitizedMaxwellState(values, values, values, values))
        result = evolve.first_order_einstein_maxwell_step(state, params, pec_boundary=boundary)
        np.testing.assert_allclose(result.em.magnetic_current,
                                   pec.enforce_pec_B(result.em.magnetic_current, result.bssn, params, boundary=boundary), atol=1.e-12)
        half_metric = evolve.enforce_algebraic_constraints(
            evolve._add_bssn_scaled(result.bssn, stage_derivatives[-1], .5 * params.dt)
        )
        np.testing.assert_allclose(
            result.em.displacement_right_half,
            pec.enforce_pec_D(result.em.displacement_right_half, half_metric, params, boundary=boundary),
            rtol=0., atol=1.e-12,
        )
    np.testing.assert_allclose(calls, [0., params.dt / 2, params.dt / 2, params.dt])
    np.testing.assert_allclose(result.bssn.trace_K, params.dt)


def test_coupled_cavity_retains_walls_and_resolves_standing_wave():
    cells, guard = 16, 3
    boundary = pec.PECBoundary((0,), guard)
    shape = (cells + 1 + 2 * guard, 1, 1)
    params = BSSNParameters(dx=1 / cells, dt=.005, x_min=-guard / cells,
                            nu=0., zero_shift=1)
    x = (jnp.arange(shape[0]) - guard) * params.dx
    amplitude = 1.e-5
    zero = jnp.zeros((3,) + shape)
    d = zero.at[1, :, 0, 0].set(amplitude * jnp.sin(jnp.pi * x))
    state = evolve.initialize_first_order_einstein_maxwell_state(
        flat_bssn_variables(shape), d, zero, params, pec_boundary=boundary
    )
    final = jax.lax.fori_loop(
        0, 40,
        lambda i, s: evolve.first_order_einstein_maxwell_step(
            s, params, time=i * params.dt, pec_boundary=boundary
        ),
        state,
    )
    dc, bc = evolve.common_densitized_fields(final.em)
    physical_c = slice(guard, -guard)
    physical_v = slice(guard + 1, -guard)
    expected_d = amplitude * jnp.sin(jnp.pi * x) * jnp.cos(.2 * jnp.pi)
    expected_b = -amplitude * jnp.cos(jnp.pi * (x - .5 * params.dx)) * jnp.sin(.2 * jnp.pi)
    np.testing.assert_allclose(dc[1, physical_c, 0, 0], expected_d[physical_c], atol=.003 * amplitude)
    np.testing.assert_allclose(bc[2, physical_v, 0, 0], expected_b[physical_v], atol=.003 * amplitude)
    for wall in boundary.walls(shape, 0):
        np.testing.assert_allclose(dc[1:, wall], 0., atol=1.e-14)
        np.testing.assert_allclose(bc[0, wall], 0., atol=1.e-14)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree_util.tree_leaves(final))
