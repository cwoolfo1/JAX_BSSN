"""Analytic Maxwell and Einstein--Maxwell references, and time convergence."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.EM.first_order import *
from JAX_BSSN.EM.first_order import evolve
from JAX_BSSN.EM.first_order.maxwell import rephase_state_metric
from tests.EM.em_helpers import flat_bssn_variables
from tests.EM.em_wave_helpers import (
    GHOSTS,
    Packet,
    RosenSolution,
    prescribed_gauge,
    symbolic_audit,
)


def test_continuum_tensors_and_normalization():
    audit = symbolic_audit()
    assert audit["passed"], audit["nonzero_residuals"]


def test_rosen_native_sampling_and_short_coupled_packet():
    n = 64
    dx = 8 / n
    duration = 0.04
    z = -4 + (np.arange(n + 2 * GHOSTS) - GHOSTS) * dx
    p = BSSNParameters(dx=dx, dt=0.002, nu=0, zero_shift=1, z_min=float(z[0]))
    sol = RosenSolution(z, dx, p.dt, duration)
    grid = make_em_grid(
        (1, 1, z.size), p, boundary_conditions=("periodic", "periodic", "constant")
    )
    initial = sol.state(0, p, grid)
    final = jax.lax.fori_loop(
        0,
        20,
        lambda i, s: first_order_einstein_maxwell_step(
            s,
            p,
            grid,
            time=i * p.dt,
            prescribed_gauge=prescribed_gauge,
            gravity_boundary=sol.boundary,
        ),
        initial,
    )
    exact_bssn = sol.bssn(duration)
    for q, e in zip(final.bssn, exact_bssn):
        np.testing.assert_allclose(
            q[..., GHOSTS:-GHOSTS], e[..., GHOSTS:-GHOSTS], atol=3e-4, rtol=2e-3
        )
    for q, e in zip(common_densitized_fields(final.em), sol.fields(duration)):
        q = jnp.stack(tuple(grid.from_tile(v) for v in q))
        np.testing.assert_allclose(
            q[..., GHOSTS:-GHOSTS], e[..., GHOSTS:-GHOSTS], atol=2e-3
        )
    assert sol.metadata["interpolation_error_a"] < 1e-10
    assert sol.metadata["interpolation_error_ap"] < 1e-9


def test_coordinate_focusing_rejected():
    with pytest.raises(ValueError, match="focusing"):
        RosenSolution(np.linspace(-4, 4, 64), 0.125, 0.01, 4, Packet(amplitude=2.0))


def test_rk4_freezes_moments_and_uses_stage_metrics(monkeypatch):
    shape = (8, 1, 1)
    p = BSSNParameters(dt=0.01, nu=0)
    grid = make_em_grid(shape, p)
    s = flat_bssn_variables(shape)
    values = jnp.ones((3,) + shape) * 0.01
    initial = initialize_first_order_einstein_maxwell_state(s, values, values, p, grid)
    snapshots = []
    times = []
    real_sources = evolve.sources_from_moments

    def sources(moments, bssn):
        snapshots.append((moments, bssn.conformal_factor.copy()))
        return real_sources(moments, bssn)

    def rhs(q, p, *matter):
        result = jax.tree_util.tree_map(jnp.zeros_like, q)
        return result._replace(conformal_factor=jnp.ones_like(q.conformal_factor) * 0.1)

    def gauge(q, t):
        times.append(float(t))
        return (
            jnp.ones_like(q.lapse) * (1 + t),
            jnp.zeros_like(q.shift),
            jnp.ones_like(q.lapse),
            jnp.zeros_like(q.shift),
        )

    monkeypatch.setattr(evolve, "sources_from_moments", sources)
    monkeypatch.setattr(evolve, "compute_bssn_rhs_with_matter", rhs)
    # Isolate the RK stage contract. Actual Maxwell calls, synchronization,
    # and metric switches are covered by the numerical integration tests.
    monkeypatch.setattr(evolve, "maxwell_half_step", lambda em, p, g: em)
    monkeypatch.setattr(evolve, "rephase_state_metric", lambda em, m, p, g: em)
    with jax.disable_jit():
        final = first_order_einstein_maxwell_step(
            initial, p, grid, prescribed_gauge=gauge
        )
    assert len(snapshots) == 4
    for moments, _ in snapshots:
        for q, e in zip(moments, snapshots[0][0]):
            np.testing.assert_array_equal(q, e)
    np.testing.assert_allclose(
        [float(w[0, 0, 0]) for _, w in snapshots],
        [1, 1 + 0.05 * p.dt, 1 + 0.05 * p.dt, 1 + 0.1 * p.dt],
    )
    np.testing.assert_allclose(final.bssn.lapse, 1 + p.dt)
    assert set(times) == {0.0, p.dt / 2, p.dt}


def wave_convergence(varying_lapse):
    """Fixed spatial grid and exact semidiscrete Fourier mode remove dx error."""
    shape = (24, 1, 1)
    dx = 2 * np.pi / 24
    T = 0.5
    p = BSSNParameters(dx=dx, dt=0.01)
    grid = make_em_grid(shape, p)
    x = jnp.arange(shape[0])[:, None, None] * dx
    zero = jnp.zeros(shape)
    omega = 2 * np.sin(dx / 2) / dx

    def fields(t):
        return (
            jnp.stack((zero, jnp.sin(x - omega * t), zero)),
            jnp.stack((zero, zero, jnp.sin(x + dx / 2 - omega * t))),
        )

    @jax.jit
    def run(dt, nsteps):
        params = p._replace(dt=dt)
        initial = initialize_first_order_einstein_maxwell_state(
            flat_bssn_variables(shape), *fields(0), params, grid
        )
        em = initial.em
        h = dt / 2
        tile = lambda v: tuple(grid.to_tile(q) for q in v)
        d0, _ = fields(0)
        dm, _ = fields(-h)
        _, bh = fields(-h / 2)
        _, bp = fields(-1.5 * h)
        em = em._replace(
            fields=(
                tile(d0),
                tile(bh),
                *em.fields[2:7],
                (tile(dm), tile(bp)),
                em.fields[8],
            )
        )
        metric = em.fields[6]

        def metric_at(t):
            def f(q):
                return q._replace(
                    lapse=jnp.ones_like(q.lapse) * (1 + varying_lapse * t)
                )

            return metric._replace(
                D=tuple(f(q) for q in metric.D),
                B=tuple(f(q) for q in metric.B),
                center=f(metric.center),
                vertex=f(metric.vertex),
            )

        def body(i, em):
            em = rephase_state_metric(em, metric_at(i * dt), params, grid)
            em = maxwell_half_step(em, params, grid)
            em = rephase_state_metric(em, metric_at((i + 1) * dt), params, grid)
            return maxwell_half_step(em, params, grid)

        em = jax.lax.fori_loop(0, nsteps, body, em)
        return tuple(
            jnp.stack(tuple(grid.from_tile(q) for q in v))
            for v in common_densitized_fields(em)
        )

    exact = fields(T + varying_lapse * T * T / 2)
    errors = []
    for n in (20, 40, 80):
        actual = run(T / n, n)
        errors.append(
            float(jnp.sqrt(sum(jnp.mean((a - b) ** 2) for a, b in zip(actual, exact))))
        )
    return np.asarray(errors)


def test_static_metric_synchronization_is_second_order():
    errors = wave_convergence(0.0)
    assert np.all(errors[:-1] / errors[1:] > 3.8), errors


def test_changing_metric_synchronization_is_second_order():
    errors = wave_convergence(1.0)
    assert np.all(errors[:-1] / errors[1:] > 3.5), errors


def test_two_way_coupled_temporal_self_convergence():
    shape = (16, 1, 1)
    dx = 2 * np.pi / 16
    duration = 0.3
    p = BSSNParameters(dx=dx, dt=0.01, nu=0, zero_shift=1)
    grid = make_em_grid(shape, p)
    x = grid.coordinates(p)[0]
    zero = jnp.zeros((3,) + shape)
    d = zero.at[1].set(0.04 * jnp.sin(x))
    b = zero.at[2].set(0.04 * jnp.sin(x + 0.5 * dx))
    bssn = flat_bssn_variables(shape)._replace(trace_K=jnp.full(shape, 0.06))

    @jax.jit
    def run(steps):
        params = p._replace(dt=duration / steps)
        initial = initialize_first_order_einstein_maxwell_state(
            bssn, d, b, params, grid
        )
        result = jax.lax.fori_loop(
            0,
            steps,
            lambda i, s: first_order_einstein_maxwell_step(
                s, params, grid, time=i * params.dt
            ),
            initial,
        )
        native = tuple(
            jnp.stack(tuple(grid.from_tile(v) for v in q))
            for q in common_densitized_fields(result.em)
        )
        return result.bssn, native

    results = [run(n) for n in (12, 24, 48)]

    def difference(a, b):
        return float(
            jnp.sqrt(
                sum(
                    jnp.mean((x - y) ** 2)
                    for x, y in zip(
                        jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
                    )
                )
            )
        )

    for sector in (0, 1):
        errors = [
            difference(results[i][sector], results[i + 1][sector]) for i in range(2)
        ]
        assert errors[0] / errors[1] > 3.3, (sector, errors)
    assert not np.allclose(results[-1][0].conformal_factor, bssn.conformal_factor)
