"""Production native histories under a prescribed nonuniform evolving gauge."""

import json
from functools import partial

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from PyPIC3D.solvers.GR_yee.static_metric import (
    compute_covariant_E, compute_covariant_H, update_D, update_B,
)
from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.EM.first_order.grid import make_em_grid
from JAX_BSSN.EM.first_order.geometry import build_yee_metric
from JAX_BSSN.EM.first_order.maxwell import (
    initialize_native_state, maxwell_half_step, rephase_state_metric,
)
from JAX_BSSN.EM.first_order.coupling import quadratic_moments, sources_from_moments
from tests.EM.em_helpers import flat_bssn_variables


def test_changing_nonuniform_shift_sponge_histories_and_sources(record_property):
    n, duration = 24, .4
    shape = (n, 1, 1)
    params = BSSNParameters(dx=2*jnp.pi/n, dt=.01)
    grid = make_em_grid(shape, params)
    base = flat_bssn_variables(shape)
    x = grid.coordinates(params)[0]
    zero = grid.to_tile(jnp.zeros(shape))

    def bssn_at(time, oscillation):
        lapse = 1 + (.08+.12*time)*jnp.cos(x)
        beta = .12*jnp.sin(x)+.3*(time-.2)+.08*oscillation*jnp.cos(x)
        return base._replace(lapse=lapse, shift=base.shift.at[0].set(beta))

    metric0 = build_yee_metric(bssn_at(0., 0.), grid)
    metric_t = jax.tree_util.tree_map(
        lambda b, a: b-a, build_yee_metric(bssn_at(1., 0.), grid), metric0)
    metric_s = jax.tree_util.tree_map(
        lambda b, a: b-a, build_yee_metric(bssn_at(0., 1.), grid), metric0)

    def metric_at(time, oscillation):
        # The geometry is affine in the prescribed gauge coefficients because
        # the spatial metric is fixed. The upstream Maxwell operator is not:
        # it must select its stencil from the current shift at each RK stage.
        return jax.tree_util.tree_map(
            lambda a, t, s: a+time*t+oscillation*s, metric0, metric_t, metric_s)

    t_check = .173
    for actual, expected in zip(
            jax.tree_util.tree_leaves(metric_at(t_check, jnp.sin(2*t_check))),
            jax.tree_util.tree_leaves(
                build_yee_metric(bssn_at(t_check, jnp.sin(2*t_check)), grid))):
        assert bool(jnp.allclose(actual, expected, rtol=3e-13, atol=3e-14))
    first_beta = bssn_at(0., 0.).shift[0]
    last_beta = bssn_at(duration, jnp.sin(2*duration)).shift[0]
    sign_changes = int(jnp.sum(first_beta*last_beta < 0))
    assert sign_changes > 0
    assert bool(jnp.any(first_beta < 0) & jnp.any(last_beta > 0))

    def fields(vector):
        return ((zero, grid.to_tile(vector[:n].reshape(shape)), zero),
                (zero, zero, grid.to_tile(vector[n:].reshape(shape))))

    center_x = jnp.arange(n)*params.dx
    vertex_x = center_x+.5*params.dx
    initial_vector = jnp.concatenate((
        jnp.sin(center_x)+.15*jnp.cos(2*center_x),
        .7*jnp.sin(vertex_x)-.1*jnp.cos(3*vertex_x),
    ))
    reference_vector = jnp.concatenate((
        .2*jnp.cos(2*center_x), .1*jnp.sin(vertex_x),
    ))
    rate = jnp.concatenate((.7+.2*jnp.cos(center_x), .7+.2*jnp.cos(vertex_x)))
    d, b = fields(initial_vector)
    reference_d, reference_b = fields(reference_vector)
    # Rates and nonzero references stay fixed while lapse/shift change.
    # Periodic masks are empty; no exterior boundary is part of this test.
    absorber = (
        ((zero, grid.to_tile(rate[:n].reshape(shape)), zero), reference_d,
         tuple(jnp.zeros_like(v, dtype=bool) for v in d)),
        ((zero, zero, grid.to_tile(rate[n:].reshape(shape))), reference_b,
         tuple(jnp.zeros_like(v, dtype=bool) for v in b)),
    )
    dynamic = grid.dynamic(params, 1.)

    def rhs(time, vector):
        metric = metric_at(time, jnp.sin(2*time))
        d, b = fields(vector)
        e = compute_covariant_E(d, b, metric)
        h = compute_covariant_H(d, b, metric)
        next_d = update_D(d, h, (zero,)*3, metric, grid.static, dynamic, 1.)
        next_b = update_B(e, b, metric, grid.static, dynamic, 1.)
        curl = jnp.concatenate((
            grid.from_tile(next_d[1]-d[1]).ravel(),
            grid.from_tile(next_b[2]-b[2]).ravel(),
        ))
        return curl-rate*(vector-reference_vector)

    @partial(jax.jit, static_argnames=("steps",))
    def reference(steps):
        dt = duration/steps

        def step(value, i):
            t = i*dt
            k1 = rhs(t, value)
            k2 = rhs(t+.5*dt, value+.5*dt*k1)
            k3 = rhs(t+.5*dt, value+.5*dt*k2)
            k4 = rhs(t+dt, value+dt*k3)
            following = value+dt*(k1+2*k2+2*k3+k4)/6
            return following, following

        _, values = jax.lax.scan(step, initial_vector, jnp.arange(steps))
        return jnp.concatenate((initial_vector[None], values))

    reference_coarse, reference_fine = reference(2048), reference(4096)
    reference_error = jnp.max(jnp.abs(reference_coarse-reference_fine[::2]))
    assert float(reference_error) < 1e-9

    @jax.jit
    def run(steps):
        p = params._replace(dt=duration/steps)
        initial = initialize_native_state(d, b, metric0, p, grid, absorber)

        def step(i, state):
            t = i*p.dt
            state = rephase_state_metric(
                state, metric_at(t, jnp.sin(2*t)), p, grid)
            state = maxwell_half_step(state, p, grid)
            t_next = (i+1)*p.dt
            state = rephase_state_metric(
                state, metric_at(t_next, jnp.sin(2*t_next)), p, grid)
            return maxwell_half_step(state, p, grid)

        return jax.lax.fori_loop(0, steps, step, initial)

    common = fields(reference_fine[-1])
    final_bssn = bssn_at(duration, jnp.sin(2*duration))
    exact_sources = sources_from_moments(quadratic_moments(*common, grid), final_bssn)
    errors = []
    for steps in (16, 32, 64):
        final = run(steps)
        quarter_stride = 4096//(4*steps)
        pairs = (
            (final.fields[0], common[0]),
            (final.synchronized_magnetic, common[1]),
            (final.fields[1], fields(reference_fine[-1-quarter_stride])[1]),
            (final.fields[7][0], fields(reference_fine[-1-2*quarter_stride])[0]),
            (final.fields[7][1], fields(reference_fine[-1-3*quarter_stride])[1]),
        )
        error = [
            sum(float(jnp.mean((grid.from_tile(a)-grid.from_tile(b))**2))
                for a, b in zip(actual, expected))**.5
            for actual, expected in pairs
        ]
        sources = sources_from_moments(
            quadratic_moments(final.fields[0], final.synchronized_magnetic, grid),
            final_bssn)
        error.extend(float(jnp.sqrt(jnp.mean((a-b)**2)))
                     for a, b in zip(sources, exact_sources))
        errors.append(error)
        for actual, original in zip(
                jax.tree_util.tree_leaves(final.absorber),
                jax.tree_util.tree_leaves(absorber)):
            assert bool(jnp.array_equal(actual, original))
    errors = jnp.asarray(errors)
    ratios = errors[:-1]/errors[1:]
    evidence = {
        "shape": list(shape), "duration": duration, "steps": [16, 32, 64],
        "columns": ["D_common", "B_common", "B_half", "D_previous", "B_previous",
                    "rho", "momentum", "stress"],
        "errors": errors.tolist(), "refinement_ratios": ratios.tolist(),
        "reference_2048_vs_4096_max_difference": float(reference_error),
        "shift_sign_changes_on_center_grid": sign_changes,
        "reference": "full production Maxwell operator at every RK4 stage",
    }
    record_property("convergence_json", json.dumps(evidence))
    print(json.dumps(evidence, indent=2))
    assert bool(jnp.all((ratios > 3.4) & (ratios < 4.6))), (errors, ratios)
