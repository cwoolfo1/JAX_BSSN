"""Spatial contracts for the sole target-metric, unlimited shift reconstruction."""

from types import SimpleNamespace

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from PyPIC3D.boundary_conditions.ghost_cells import make_field_mesh
from PyPIC3D.boundary_conditions.staggered import refresh_fields
from PyPIC3D.diagnostics.static_metric import divergence
from PyPIC3D.relativity.core import Metric, YeeMetric, D_FIELD_LOCATIONS, B_FIELD_LOCATIONS
from PyPIC3D.relativity.field_interpolation import reconstruct_vector
from PyPIC3D.solvers.GR_yee.static_metric import (
    _upwind_shift_product, compute_covariant_E, compute_covariant_H, update_D, update_B,
)
from PyPIC3D.solvers.GR_yee.time_loop import time_loop_static_metric
from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.EM.first_order.grid import make_em_grid
from JAX_BSSN.EM.first_order.maxwell import initialize_native_state, maxwell_half_step


def geometry(q):
    x, y, z = q
    v = jnp.stack((.3+.1*jnp.sin(x), .2+.1*jnp.cos(y), .1+.1*jnp.sin(z)), -1)
    gamma = jnp.eye(3) + v[..., :, None]*v[..., None, :]
    beta = jnp.stack((.2*jnp.cos(y), -.15*jnp.sin(z), .1*jnp.sin(x)), -1)
    return Metric(.9+.04*jnp.sin(x+y), beta, gamma, jnp.linalg.inv(gamma),
                  jnp.sqrt(jnp.linalg.det(gamma)))


def metric_on_periodic_grid(n):
    def sample(location):
        q = jnp.meshgrid(*(2*jnp.pi*(jnp.arange(n)+(.5 if a == "V" else 0.))/n
                          for a in location), indexing="ij")
        return jax.tree.map(lambda a: a[None, None, None], geometry(q))
    return YeeMetric(tuple(sample(q) for q in D_FIELD_LOCATIONS),
                     tuple(sample(q) for q in B_FIELD_LOCATIONS),
                     sample(("C",)*3), sample(("V",)*3))


def take_offset(field, offset, axis):
    indices = (jnp.arange(field.shape[axis+3])+offset) % field.shape[axis+3]
    return jnp.take(field, indices, axis=axis+3)


def independent_shift(field, beta, source, axis):
    # Explicit three-point weights at a half node; no slope helper is reused.
    if source[axis] == "C":
        right = .25*field + take_offset(field, 1, axis) - .25*take_offset(field, 2, axis)
        left = -.25*take_offset(field, -1, axis) + field + .25*take_offset(field, 1, axis)
    else:
        right = .25*take_offset(field, -1, axis) + field - .25*take_offset(field, 1, axis)
        left = -.25*take_offset(field, -2, axis) + take_offset(field, -1, axis) + .25*field
    return beta*jnp.where(beta > 0, right, left)


@pytest.mark.parametrize("axis", (0, 1, 2))
@pytest.mark.parametrize("site", ("C", "V"))
@pytest.mark.parametrize("beta", (-.7, 0., .7))
def test_native_half_cell_signs_and_weights(axis, site, beta):
    field = jax.random.normal(jax.random.key(231), (1, 1, 1, 7, 8, 9))
    location = tuple(site if a == axis else "C" for a in range(3))
    actual = jax.jit(lambda f: _upwind_shift_product(f, beta, location, axis))(field)
    expected = independent_shift(field, beta, location, axis)
    assert jnp.allclose(actual, expected, rtol=3e-15, atol=3e-15)
    constant = _upwind_shift_product(jnp.ones_like(field), beta, location, axis)
    assert jnp.all(constant == beta)
    dimensions = [1]*6
    dimensions[axis+3] = field.shape[axis+3]
    coordinate = jnp.arange(field.shape[axis+3]).reshape(dimensions)
    affine = jnp.broadcast_to(.3*coordinate-.2, field.shape)
    target = coordinate + (.5 if site == "C" else -.5)
    expected_affine = jnp.broadcast_to(beta*(.3*target-.2), field.shape)
    reconstructed = _upwind_shift_product(affine, beta, location, axis)
    interior = [slice(None)]*6
    interior[axis+3] = slice(2, -2)
    assert jnp.allclose(reconstructed[tuple(interior)], expected_affine[tuple(interior)],
                        rtol=3e-15, atol=3e-15)


def test_E_H_full_tensor_reference_and_zero_shift():
    metric = metric_on_periodic_grid(6)
    fields = jax.random.normal(jax.random.key(239), (6, 1, 1, 1, 6, 6, 6))
    D, B = tuple(fields[:3]), tuple(fields[3:])
    for method, own, other, locations, other_locations, samples, sign in (
        (compute_covariant_E, D, B, D_FIELD_LOCATIONS, B_FIELD_LOCATIONS, metric.D, 1),
        (compute_covariant_H, B, D, B_FIELD_LOCATIONS, D_FIELD_LOCATIONS, metric.B, -1),
    ):
        expected = []
        for i, target in enumerate(locations):
            m = samples[i]
            vector = jnp.stack(reconstruct_vector(own, locations, target), -1)
            field = m.lapse*jnp.sum(m.gamma[..., i, :]*vector/m.sqrt_gamma[..., None], -1)
            j, k = (i+1) % 3, (i+2) % 3
            field += sign*independent_shift(other[k], m.shift[..., j], other_locations[k], j)
            field -= sign*independent_shift(other[j], m.shift[..., k], other_locations[j], k)
            expected.append(field)
        actual = jax.jit(method)(D, B, metric)
        assert jnp.allclose(jnp.stack(actual), jnp.stack(expected), rtol=3e-14, atol=3e-14)

    zero_metric = metric._replace(
        D=tuple(m._replace(shift=jnp.zeros_like(m.shift)) for m in metric.D),
        B=tuple(m._replace(shift=jnp.zeros_like(m.shift)) for m in metric.B))
    for method, own, locations, samples in (
        (compute_covariant_E, D, D_FIELD_LOCATIONS, zero_metric.D),
        (compute_covariant_H, B, B_FIELD_LOCATIONS, zero_metric.B),
    ):
        expected = []
        for i, target in enumerate(locations):
            m = samples[i]
            vector = reconstruct_vector(own, locations, target)
            lower = sum(m.gamma[..., i, j]*(vector[j]/m.sqrt_gamma) for j in range(3))
            expected.append(m.lapse*lower)
        assert jnp.array_equal(jnp.stack(method(D, B, zero_metric)), jnp.stack(expected))


def test_smooth_variable_geometry_second_order():
    def vector(q, magnetic):
        x, y, z = q
        return ((.4*jnp.cos(y+z), .7*jnp.sin(x+z), .2*jnp.cos(x+y)) if magnetic else
                (jnp.sin(x)+.2*jnp.cos(z), jnp.cos(y)+.1*jnp.sin(x), .3*jnp.cos(x+z)))

    errors = []
    for n in (8, 16, 32):
        metric = metric_on_periodic_grid(n)
        coordinates = {}
        for location in (*D_FIELD_LOCATIONS, *B_FIELD_LOCATIONS):
            coordinates[location] = jnp.meshgrid(
                *(2*jnp.pi*(jnp.arange(n)+(.5 if a == "V" else 0.))/n for a in location),
                indexing="ij")
        D = tuple(m.sqrt_gamma*vector(coordinates[q], False)[i][None, None, None]
                  for i, (q, m) in enumerate(zip(D_FIELD_LOCATIONS, metric.D)))
        B = tuple(m.sqrt_gamma*vector(coordinates[q], True)[i][None, None, None]
                  for i, (q, m) in enumerate(zip(B_FIELD_LOCATIONS, metric.B)))
        error = 0.
        for actual, locations, magnetic in (
            (compute_covariant_E(D, B, metric), D_FIELD_LOCATIONS, False),
            (compute_covariant_H(D, B, metric), B_FIELD_LOCATIONS, True),
        ):
            for i, q in enumerate(locations):
                m = geometry(coordinates[q])
                d, b = (jnp.stack(vector(coordinates[q], flag), -1) for flag in (False, True))
                own, other, sign = (b, d, -1) if magnetic else (d, b, 1)
                exact = (m.lapse*jnp.sum(m.gamma[..., i, :]*own, -1)
                         + sign*m.sqrt_gamma*jnp.cross(m.shift, other)[..., i])
                error += jnp.mean((actual[i][0, 0, 0]-exact)**2)
        errors.append(jnp.sqrt(error))
    ratios = jnp.asarray(errors[:-1])/jnp.asarray(errors[1:])
    assert jnp.all(ratios > 3.5), (errors, ratios)


def test_time_loop_rejects_one_guard_before_field_access():
    with pytest.raises(ValueError, match="guard_cells >= 2"):
        time_loop_static_metric(None, None, None, SimpleNamespace(guard_cells=1), None)


def test_nonorthogonal_PEC_preserves_target_metric_wall_row():
    params = BSSNParameters(dx=.2, dt=.001)
    grid = make_em_grid((6, 6, 6), params,
                        boundary_conditions=("conducting", "periodic", "periodic"))
    shape = (1, 1, 1)+tuple(n+2*grid.guard_cells for n in grid.tile_shape)
    gamma = jnp.array([[1.3, .12, -.08], [.12, 1.2, .09], [-.08, .09, 1.1]])
    m = Metric(jnp.ones(shape), jnp.broadcast_to(jnp.array([.25, -.1, .15]), shape+(3,)),
               jnp.broadcast_to(gamma, shape+(3, 3)),
               jnp.broadcast_to(jnp.linalg.inv(gamma), shape+(3, 3)),
               jnp.full(shape, jnp.sqrt(jnp.linalg.det(gamma))))
    metric = YeeMetric((m,)*3, (m,)*3, m, m)
    values = jax.random.normal(jax.random.key(241), (6,)+shape)
    D = refresh_fields(tuple(values[:3]), grid.static, D_FIELD_LOCATIONS, "D", metric)
    B = refresh_fields(tuple(values[3:]), grid.static, B_FIELD_LOCATIONS, "B", metric)
    E = compute_covariant_E(D, B, metric)
    for i in (1, 2):
        j, k = (i+1) % 3, (i+2) % 3
        shift = (independent_shift(B[k], m.shift[..., j], B_FIELD_LOCATIONS[k], j)
                 - independent_shift(B[j], m.shift[..., k], B_FIELD_LOCATIONS[j], k))
        target_vector = reconstruct_vector(D, D_FIELD_LOCATIONS, D_FIELD_LOCATIONS[i])
        baseline_fido = sum(gamma[i, a]*v/m.sqrt_gamma for a, v in enumerate(target_vector))
        faces = (0, 0, 0, jnp.array([3, 8]), slice(3, 9), slice(3, 9))
        # The pre-existing off-diagonal projector has an interpolation residual
        # for arbitrary native data. Preserve that row instead of asserting an
        # exact zero which the original constitutive method did not produce.
        assert jnp.allclose(((E[i]-shift)/m.lapse)[faces], baseline_fido[faces],
                            rtol=3e-13, atol=3e-13)
    repeated = refresh_fields(D, grid.static, D_FIELD_LOCATIONS, "D", metric)
    assert jnp.allclose(jnp.stack(repeated), jnp.stack(D), rtol=3e-13, atol=3e-13)


@pytest.mark.parametrize("axis", (0, 1, 2))
@pytest.mark.parametrize("boundary", ("conducting", "constant", "absorbing"))
def test_nonzero_shift_boundary_evolution(axis, boundary):
    shape = (6, 6, 6)
    params = BSSNParameters(dx=.2, dt=.001)
    boundaries = tuple(boundary if a == axis else "periodic" for a in range(3))
    grid = make_em_grid(shape, params, boundary_conditions=boundaries)
    sample_shape = (1, 1, 1)+tuple(n+2*grid.guard_cells for n in grid.tile_shape)
    # Diagonal geometry has an exact native zero-tangential FIDO wall row.
    # General off-diagonal projector convergence is covered by PyPIC3D's PEC tests.
    gamma = jnp.diag(jnp.array([1.3, 1.2, 1.1]))
    m = Metric(jnp.ones(sample_shape), jnp.broadcast_to(jnp.array([.25, -.1, .15]), sample_shape+(3,)),
               jnp.broadcast_to(gamma, sample_shape+(3, 3)),
               jnp.broadcast_to(jnp.linalg.inv(gamma), sample_shape+(3, 3)),
               jnp.full(sample_shape, jnp.sqrt(jnp.linalg.det(gamma))))
    metric = YeeMetric((m,)*3, (m,)*3, m, m)
    fields = jax.random.normal(jax.random.key(241), (6,)+sample_shape)
    em = initialize_native_state(tuple(fields[:3]), tuple(fields[3:]), metric, params, grid)
    result = jax.jit(lambda state: jax.lax.fori_loop(
        0, 4, lambda _, value: maxwell_half_step(value, params, grid), state))(em)
    assert all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree.leaves(result))
    if boundary == "conducting":
        D, B = result.fields[:2]
        E = compute_covariant_E(D, B, metric)
        physical = ((slice(None),)*3 + tuple(
            slice(None) if a == axis else slice(grid.guard_cells, grid.guard_cells+n)
            for a, n in enumerate(grid.tile_shape)))
        for i, location in enumerate(D_FIELD_LOCATIONS):
            if i == axis:
                continue
            lower = sum(gamma[i, j]*value for j, value in enumerate(
                reconstruct_vector(D, D_FIELD_LOCATIONS, location)))
            j, k = (i+1) % 3, (i+2) % 3
            shift = (independent_shift(B[k], m.shift[..., j], B_FIELD_LOCATIONS[k], j)
                     - independent_shift(B[j], m.shift[..., k], B_FIELD_LOCATIONS[j], k))
            # The conducting projector constrains the FIDO field. Coordinate
            # E includes a nonzero shift EMF at a wall with normal beta.
            fido = (E[i]-shift)/m.lapse
            faces = jnp.take(lower[physical], jnp.array([grid.guard_cells,
                            grid.guard_cells+grid.tile_shape[axis]]), axis=axis+3)
            assert jnp.max(jnp.abs(faces)) < 2e-12
            faces = jnp.take(fido[physical], jnp.array([grid.guard_cells,
                            grid.guard_cells+grid.tile_shape[axis]]), axis=axis+3)
            assert jnp.max(jnp.abs(faces)) < 2e-12
        faces = jnp.take(B[axis][physical], jnp.array([grid.guard_cells,
                        grid.guard_cells+grid.tile_shape[axis]]), axis=axis+3)
        assert jnp.max(jnp.abs(faces)) < 2e-12


@pytest.mark.skipif(jax.device_count() < 2, reason="needs two logical CPU devices")
@pytest.mark.parametrize("split_axis", (0, 1, 2))
def test_two_guard_tile_seams_and_divergence(split_axis):
    n, guards = 8, 2
    params = BSSNParameters(dx=2*jnp.pi/n, dt=.01)
    grid = make_em_grid((n,)*3, params)
    dynamic = grid.dynamic(params, params.dt)
    metric_global = metric_on_periodic_grid(n)
    values = jax.random.normal(jax.random.key(251), (6, n, n, n))

    def calculate(counts):
        widths = tuple(n//count for count in counts)
        static = grid.static._replace(guard_cells=guards, tile_shape=widths,
                                      field_mesh=make_field_mesh(counts))
        indices = []
        for axis, (count, width) in enumerate(zip(counts, widths)):
            td, sd = [1]*6, [1]*6
            td[axis], sd[axis+3] = count, width+2*guards
            indices.append((width*jnp.arange(count).reshape(td)
                            + jnp.arange(-guards, width+guards).reshape(sd)) % n)
        def tile(value):
            return value[indices[0], indices[1], indices[2]]
        metric = jax.tree.map(lambda v: tile(v.reshape((n,)*3+v.shape[6:])), metric_global)
        interior = (slice(None),)*3+tuple(slice(guards, guards+w) for w in widths)
        def owned(value):
            return value[interior].transpose(0, 3, 1, 4, 2, 5).reshape((n,)*3)
        fields = tuple(jnp.zeros_like(tile(v)).at[interior].set(tile(v)[interior]) for v in values)
        d = refresh_fields(fields[:3], static, D_FIELD_LOCATIONS, "D", metric)
        b = refresh_fields(fields[3:], static, B_FIELD_LOCATIONS, "B", metric)
        e, h = compute_covariant_E(d, b, metric), compute_covariant_H(d, b, metric)
        dn = update_D(d, h, tuple(jnp.zeros_like(v) for v in d), metric, static, dynamic, params.dt)
        bn = update_B(e, b, metric, static, dynamic, params.dt)
        for old, new, forward in ((d, dn, False), (b, bn, True)):
            delta = tuple(v-u for u, v in zip(old, new))
            assert jnp.max(jnp.abs(owned(divergence(delta, dynamic, forward=forward)))) < 4e-14
        return jnp.stack(tuple(owned(v) for group in (e, h, dn, bn) for v in group))
    counts = tuple(2 if axis == split_axis else 1 for axis in range(3))
    actual = jax.device_put(calculate(counts), jax.devices()[0])
    expected = jax.device_put(calculate((1,)*3), jax.devices()[0])
    assert jnp.allclose(actual, expected, rtol=3e-14, atol=3e-14)
