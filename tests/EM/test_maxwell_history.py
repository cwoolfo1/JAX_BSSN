"""Production native Maxwell histories, common-time sources and constraints."""

import importlib
import json

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS
from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.EM.first_order.grid import make_em_grid
from JAX_BSSN.EM.first_order.geometry import build_yee_metric
from tests.EM.em_helpers import flat_bssn_variables


def test_production_operator_reaches_initialization_rephase_and_synchronization(monkeypatch):
    maxwell = importlib.import_module("JAX_BSSN.EM.first_order.maxwell")
    params = BSSNParameters(dx=.2, dt=.004)
    grid = make_em_grid((8, 1, 1), params)
    bssn = flat_bssn_variables(grid.shape)
    x = grid.coordinates(params)[0]
    beta = bssn.shift.at[0].set(.15*jnp.sin(2*jnp.pi*x/(8*params.dx)))
    bssn = bssn._replace(shift=beta)
    metric = build_yee_metric(bssn, grid)
    d = tuple(grid.to_tile(jnp.broadcast_to(
        jnp.sin(grid.coordinates(params, loc)[0]+i), grid.shape
    )) for i, loc in enumerate(D_FIELD_LOCATIONS))
    b = tuple(grid.to_tile(jnp.broadcast_to(
        .2*jnp.cos(grid.coordinates(params, loc)[0]+i), grid.shape
    )) for i, loc in enumerate(B_FIELD_LOCATIONS))
    calls = []

    def record_electric(d, b, metric):
        calls.append("E")
        return electric(d, b, metric)

    def record_magnetic(d, b, metric):
        calls.append("H")
        return magnetic(d, b, metric)

    def record_loop(*args, **kwargs):
        calls.append("native_loop")
        return time_loop(*args, **kwargs)

    electric, magnetic = maxwell.compute_covariant_E, maxwell.compute_covariant_H
    time_loop = maxwell.time_loop_static_metric
    monkeypatch.setattr(maxwell, "compute_covariant_E", record_electric)
    monkeypatch.setattr(maxwell, "compute_covariant_H", record_magnetic)
    monkeypatch.setattr(maxwell, "time_loop_static_metric", record_loop)

    # Observe Python call paths rather than cached traces from another test.
    with jax.disable_jit():
        initial = maxwell.initialize_native_state(d, b, metric, params, grid)
        assert calls == ["E", "H"]
        calls.clear()
        advanced = maxwell.maxwell_half_step(initial, params, grid)
        assert calls == ["native_loop", "E", "E"]
        calls.clear()
        changed_metric = build_yee_metric(
            bssn._replace(lapse=1.02*bssn.lapse, shift=1.1*bssn.shift), grid
        )
        rephased = maxwell.rephase_state_metric(advanced, changed_metric, params, grid)
        assert calls == ["E", "H", "E", "H"]

    # A metric switch changes staggered histories, while the owned common-time
    # densities remain the same physical data, without redensitization.
    for before, after in zip(
            jax.tree_util.tree_leaves((advanced.fields[0], advanced.synchronized_magnetic)),
            jax.tree_util.tree_leaves((rephased.fields[0], rephased.synchronized_magnetic))):
        assert bool(jnp.array_equal(before, after))
    assert any(bool(jnp.any(a != b)) for a, b in zip(
        jax.tree_util.tree_leaves((advanced.fields[1], advanced.fields[7])),
        jax.tree_util.tree_leaves((rephased.fields[1], rephased.fields[7]))))


def test_variable_shift_sponge_histories_are_second_order(record_property):
    from jax.scipy.linalg import expm
    from PyPIC3D.solvers.GR_yee.static_metric import (
        compute_covariant_E, compute_covariant_H, update_D, update_B,
    )
    from JAX_BSSN.EM.first_order.maxwell import initialize_native_state, maxwell_half_step
    from JAX_BSSN.EM.first_order.coupling import quadratic_moments, sources_from_moments

    n = 24
    shape = (n, 1, 1)
    duration, rate = .3, .7
    params = BSSNParameters(dx=2*jnp.pi/n, dt=.01)
    grid = make_em_grid(shape, params)
    x = grid.coordinates(params)[0]
    bssn = flat_bssn_variables(shape)
    bssn = bssn._replace(shift=bssn.shift.at[0].set(.1+.15*jnp.sin(x)))
    metric = build_yee_metric(bssn, grid)
    zero = grid.to_tile(jnp.zeros(shape))

    def fields(vector):
        return ((zero, grid.to_tile(vector[:n].reshape(shape)), zero),
                (zero, zero, grid.to_tile(vector[n:].reshape(shape))))

    dynamic = grid.dynamic(params, 1.)

    def rhs(vector):
        # Exact integration of this small semidiscrete operator isolates the
        # history/time-centering error from spatial interpolation error.
        d, b = fields(vector)
        e = compute_covariant_E(d, b, metric)
        h = compute_covariant_H(d, b, metric)
        next_d = update_D(d, h, (zero,)*3, metric, grid.static, dynamic, 1.)
        next_b = update_B(e, b, metric, grid.static, dynamic, 1.)
        return jnp.concatenate((
            grid.from_tile(next_d[1]-d[1]).ravel(),
            grid.from_tile(next_b[2]-b[2]).ravel(),
        ))-rate*vector

    matrix = jax.jit(jax.jacfwd(rhs))(jnp.zeros(2*n))
    initial_vector = jnp.concatenate((
        jnp.sin(jnp.arange(n)*params.dx), jnp.sin((jnp.arange(n)+.5)*params.dx),
    ))
    d, b = fields(jnp.asarray(initial_vector))
    absorber = tuple((
        tuple(jnp.full_like(value, rate) for value in vector), (zero,)*3,
        tuple(jnp.zeros_like(value, dtype=bool) for value in vector),
    ) for vector in (d, b))

    @jax.jit
    def run(steps):
        p = params._replace(dt=duration/steps)
        initial = initialize_native_state(d, b, metric, p, grid, absorber)
        return jax.lax.fori_loop(
            0, 2*steps, lambda i, state: maxwell_half_step(state, p, grid), initial
        )

    def exact(time):
        return fields(jnp.asarray(expm(time*matrix)@initial_vector))

    exact_common = exact(duration)
    exact_sources = sources_from_moments(quadratic_moments(*exact_common, grid), bssn)
    errors = []
    for steps in (12, 24, 48):
        state = run(steps)
        h = duration/(2*steps)
        pairs = (
            (state.fields[0], exact_common[0]),
            (state.synchronized_magnetic, exact_common[1]),
            (state.fields[1], exact(duration-h/2)[1]),
            (state.fields[7][0], exact(duration-h)[0]),
            (state.fields[7][1], exact(duration-1.5*h)[1]),
        )
        error = [
            sum(float(jnp.mean((grid.from_tile(a)-grid.from_tile(b))**2))
                for a, b in zip(actual, expected))**.5
            for actual, expected in pairs
        ]
        sources = sources_from_moments(
            quadratic_moments(state.fields[0], state.synchronized_magnetic, grid), bssn
        )
        error.append(sum(float(jnp.mean((a-b)**2))
                         for a, b in zip(sources, exact_sources))**.5)
        errors.append(error)
    errors = jnp.asarray(errors)
    ratios = errors[:-1]/errors[1:]
    evidence = {
        "steps": [12, 24, 48],
        "columns": ["D_common", "B_common", "B_half", "D_previous", "B_previous", "sources"],
        "errors": errors.tolist(), "refinement_ratios": ratios.tolist(),
    }
    record_property("convergence_json", json.dumps(evidence))
    print(json.dumps(evidence, indent=2))
    assert bool(jnp.all((ratios > 3.4) & (ratios < 4.6))), (errors, ratios)


def test_variable_shift_preserves_both_native_gauss_constraints():
    from PyPIC3D.solvers.GR_yee.static_metric import update_D, update_B
    from PyPIC3D.diagnostics.static_metric import divergence
    from JAX_BSSN.EM.first_order.maxwell import initialize_native_state, maxwell_half_step

    params = BSSNParameters(dx=.4, dt=.01)
    grid = make_em_grid((8,)*3, params)
    x, y, z = grid.coordinates(params)
    bssn = flat_bssn_variables(grid.shape)
    wave_number = 2*jnp.pi/(8*params.dx)
    beta = jnp.stack(tuple(jnp.broadcast_to(.1*jnp.sin(wave_number*axis), grid.shape)
                           for axis in (x, y, z)))
    bssn = bssn._replace(shift=beta)
    metric = build_yee_metric(bssn, grid)
    dynamic = grid.dynamic(params, 1.)
    zero = tuple(grid.to_tile(jnp.zeros(grid.shape)) for _ in range(3))
    keys = jax.random.split(jax.random.PRNGKey(291), 6)
    potential_d, potential_b = (
        tuple(grid.to_tile(jax.random.normal(keys[3*i+j], grid.shape)) for j in range(3))
        for i in range(2)
    )
    # Native curls produce constraint-compatible random data without a
    # continuum-sampling divergence error. There is no sponge in this test:
    # spatially varying absorption is not Gauss preserving.
    d = update_D(zero, potential_b, zero, metric, grid.static, dynamic, 1.)
    b = update_B(potential_d, zero, metric, grid.static, dynamic, 1.)
    initial = initialize_native_state(d, b, metric, params, grid)
    final = jax.jit(lambda state: jax.lax.fori_loop(
        0, 30, lambda i, q: maxwell_half_step(q, params, grid), state
    ))(initial)
    for before, after, forward in (
        (d, final.fields[0], False),
        (b, final.synchronized_magnetic, True),
    ):
        start = grid.from_tile(divergence(before, dynamic, forward=forward))
        end = grid.from_tile(divergence(after, dynamic, forward=forward))
        assert bool(jnp.all(jnp.abs(start) < 2.e-12))
        assert bool(jnp.all(jnp.abs(end-start) < 2.e-12))


def test_reference_termination_covers_initial_histories_rephase_and_common_b():
    from JAX_BSSN.EM.first_order.absorber import make_reference_sponge
    from JAX_BSSN.EM.first_order.maxwell import (
        initialize_native_state, maxwell_half_step, rephase_state_metric,
    )

    params = BSSNParameters(dx=.5, dt=.01, x_min=-1.75, y_min=-1.75, z_min=-1.75)
    grid = make_em_grid((8,)*3, params, boundary_conditions=("constant",)*3)
    bssn = flat_bssn_variables(grid.shape)
    x, y, z = grid.coordinates(params)
    beta = jnp.stack(tuple(jnp.broadcast_to(.15*jnp.sin(axis), grid.shape)
                           for axis in (x, y, z)))
    bssn = bssn._replace(shift=beta)
    metric = build_yee_metric(bssn, grid)
    shape = grid.to_tile(jnp.zeros(grid.shape)).shape

    def native_fields(locations, scale):
        values = []
        for i, location in enumerate(locations):
            x, y, z = grid.coordinates(params, location, tiled=True)
            values.append(jnp.broadcast_to(
                scale*(jnp.sin(x+i)+.3*jnp.cos(y-i)+.2*jnp.sin(z)), shape))
        return tuple(values)

    reference_d = native_fields(D_FIELD_LOCATIONS, .1)
    reference_b = native_fields(B_FIELD_LOCATIONS, .03)
    absorber = make_reference_sponge(
        grid, params, reference_d, reference_b, width=.75)
    # The initializer must replace deliberately incorrect exterior data,
    # including physical-face V nodes, in every retained time level.
    d = tuple(value+.04 for value in reference_d)
    b = tuple(value-.02 for value in reference_b)
    initial = initialize_native_state(d, b, metric, params, grid, absorber)
    advanced = maxwell_half_step(initial, params, grid)
    changed_metric = build_yee_metric(
        bssn._replace(lapse=1.03*bssn.lapse, shift=-.8*bssn.shift), grid)
    rephased = rephase_state_metric(advanced, changed_metric, params, grid)
    final = maxwell_half_step(rephased, params, grid)

    for state in (initial, advanced, rephased, final):
        for vector, data in (
                (state.fields[0], absorber[0]), (state.fields[7][0], absorber[0]),
                (state.fields[1], absorber[1]), (state.fields[7][1], absorber[1]),
                (state.synchronized_magnetic, absorber[1])):
            _, reference, masks = data
            for value, fixed, mask in zip(vector, reference, masks):
                assert bool(jnp.all(jnp.isfinite(value)))
                assert bool(jnp.all(jnp.where(mask, value == fixed, True)))
        for actual, expected in zip(jax.tree_util.tree_leaves(state.absorber),
                                    jax.tree_util.tree_leaves(absorber)):
            assert bool(jnp.array_equal(actual, expected))
    assert any(bool(jnp.any(jnp.where(mask, False, a != b)))
               for a, b, mask in zip(
                   initial.fields[0], final.fields[0], absorber[0][2]))
