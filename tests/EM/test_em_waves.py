"""In-memory Rosen wave accuracy, self-convergence, and stage regressions."""

import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import pytest
from JAX_BSSN.bssn.variables import BSSNParameters
from JAX_BSSN.bssn.constraints import (
    compute_det_gamma_violation,
    compute_trace_A_violation,
)
from JAX_BSSN.EM.first_order import evolve
from JAX_BSSN.EM.first_order.equations import (
    densitized_displacement_divergence,
    densitized_magnetic_divergence,
)
from tests.EM.em_wave_helpers import (
    GHOSTS, Packet, RosenSolution, prescribed_gauge, profile, symbolic_audit,
)


def solution(n=32,packet=Packet(),duration=.1):
    dx=8/n; dt=.2*dx
    z=-4+(np.arange(n+2*GHOSTS)-GHOSTS+.5)*dx
    return RosenSolution(z,dx,dt,duration,packet),BSSNParameters(dx=dx,dt=dt,nu=0,kappa=0,zero_shift=1)


def test_continuum_tensors_and_normalization():
    audit = symbolic_audit()
    assert audit['passed'], audit['nonzero_residuals']


def test_native_sampling_and_history():
    sol,p=solution()
    t=.03; d,b=sol.fields(t)
    for value,w in ((d[0],t-sol.z-2),(b[1],t-sol.z+p.dx/2-2)):
        expected=-sol.scale(w)[0]*profile(w,sol.packet)/jnp.sqrt(4*jnp.pi)
        np.testing.assert_allclose(value,expected,atol=1e-15)
    state=sol.state(t,p.dt)
    for actual,exact in zip(state.em,(sol.fields(t-p.dt)[1],b,sol.fields(t-p.dt/2)[0],sol.fields(t+p.dt/2)[0])):
        np.testing.assert_array_equal(actual,exact)
    assert sol.metadata['interpolation_error_a']<1e-10
    assert sol.metadata['interpolation_error_ap']<1e-9


def test_coordinate_focusing_rejected():
    with pytest.raises(ValueError,match='focusing'):
        solution(packet=Packet(amplitude=2.),duration=4.)


def test_stage_times_and_prescribed_gauge(monkeypatch):
    sol,p=solution(packet=Packet(amplitude=0.))
    state=sol.state(0.,p.dt)
    calls=[]
    def boundary(bssn,d,b,t):
        calls.append(float(t))
        d=d.at[...,0].set(t)
        b=b.at[...,0].set(2*t)
        return bssn,d,b
    def gauge(bssn,t):
        return (jnp.full_like(bssn.lapse,1+.1*t),jnp.full_like(bssn.shift,.2*t),
                jnp.full_like(bssn.lapse,.1),jnp.full_like(bssn.shift,.2))
    def gravity(bssn,d,b,params):
        # Every gravity stage must see the imposed gauge and common-time ghosts.
        expected=float(d[0,0,0,0])
        np.testing.assert_allclose(bssn.lapse,1+.1*expected)
        np.testing.assert_allclose(bssn.shift,.2*expected)
        return jax.tree_util.tree_map(jnp.zeros_like,bssn)
    monkeypatch.setattr(evolve,'_bssn_rhs_from_densities',gravity)
    monkeypatch.setattr(evolve,'densitized_maxwell_rhs',lambda d,b,s,p:(jnp.zeros_like(d),jnp.zeros_like(b)))
    with jax.disable_jit():
        result=evolve.first_order_einstein_maxwell_step(state,p,time=0.,prescribed_gauge=gauge,stage_boundary=boundary)
    np.testing.assert_allclose(calls,[0,-p.dt,p.dt/2,p.dt/2,p.dt,p.dt,1.5*p.dt])
    np.testing.assert_allclose(result.bssn.lapse,1+.1*p.dt)
    np.testing.assert_allclose(result.bssn.shift,.2*p.dt)
    np.testing.assert_allclose(result.em.magnetic_previous[...,0],0.)
    np.testing.assert_allclose(result.em.magnetic_current[...,0],2*p.dt)
    np.testing.assert_allclose(result.em.displacement_left_half[...,0],.5*p.dt)
    np.testing.assert_allclose(result.em.displacement_right_half[...,0],1.5*p.dt)


def test_identity_hooks_and_minkowski():
    sol,p=solution(packet=Packet(amplitude=0.))
    initial=sol.state(0.,p.dt)
    ordinary=evolve.first_order_einstein_maxwell_step(initial,p)
    identity=evolve.first_order_einstein_maxwell_step(initial,p,stage_boundary=lambda s,d,b,t:(s,d,b))
    wave=evolve.first_order_einstein_maxwell_step(initial,p,prescribed_gauge=prescribed_gauge,stage_boundary=sol.boundary)
    for actual in (ordinary,identity,wave):
        for q,r in zip(jax.tree_util.tree_leaves(actual),jax.tree_util.tree_leaves(initial)):
            np.testing.assert_allclose(q,r,rtol=0,atol=1e-13)
    for q,r in zip(jax.tree_util.tree_leaves(identity),jax.tree_util.tree_leaves(ordinary)):
        np.testing.assert_array_equal(q,r)


def test_short_coupled_packet_evolves_interior():
    sol,p=solution(n=64)
    initial=sol.state(0.,p.dt)
    out=evolve.first_order_einstein_maxwell_step(initial,p,prescribed_gauge=prescribed_gauge,stage_boundary=sol.boundary)
    exact=sol.bssn(p.dt)
    assert np.max(abs(np.asarray(out.bssn.conformal_factor-initial.bssn.conformal_factor)))>1e-7
    np.testing.assert_allclose(out.bssn.conformal_factor,exact.conformal_factor,rtol=0,atol=2e-6)
    np.testing.assert_array_equal(out.bssn.lapse,1.)
    for q,r in zip(out.bssn,exact):
        np.testing.assert_allclose(q[...,:GHOSTS],r[...,:GHOSTS],atol=1e-14)
        np.testing.assert_allclose(q[...,-GHOSTS:],r[...,-GHOSTS:],atol=1e-14)


# Keep every evolved field explicit so missing or vanishing signals cannot
# silently remove an equation from the accuracy or self-convergence checks.
_GRAVITY_TOLERANCES = {
    'conformal_metric': 5e-6,
    'conformal_factor': 2e-6,
    'traceless_K': 2e-5,
    'trace_K': 2e-5,
    'conformal_connection': 5e-5,
}
_WAVE_FIELDS = ('D', 'B', *_GRAVITY_TOLERANCES)


def _physical(value):
    return np.asarray(value)[..., 0, 0, GHOSTS:-GHOSTS]


def _wave_rms(value):
    return float(np.sqrt(np.mean(np.asarray(value)**2)))


def _wave_fields(bssn, d, b):
    return {
        'D': _physical(d),
        'B': _physical(b),
        **{name: _physical(getattr(bssn, name)) for name in _GRAVITY_TOLERANCES},
    }


def _assert_gauge_and_constraints(state, params, label):
    for name, expected in (('lapse', 1.), ('shift', 0.)):
        np.testing.assert_allclose(
            getattr(state.bssn, name), expected, rtol=0, atol=1e-14,
            err_msg=f'{label}: prescribed {name}',
        )
    d, b = evolve.common_densitized_fields(state.em)
    constraints = {
        'det_gamma': compute_det_gamma_violation(state.bssn),
        'trace_A': compute_trace_A_violation(state.bssn),
        'div_D': densitized_displacement_divergence(d, params),
        'div_B': densitized_magnetic_divergence(b, params),
    }
    for name, field in constraints.items():
        value = _wave_rms(_physical(field))
        assert np.isfinite(value) and 0 <= value < 1e-12, (
            f'{label}: {name} RMS={value}, expected <1e-12'
        )


@pytest.fixture(scope='module')
def em_wave_convergence_runs():
    """Three short CPU evolutions sharing one reference and compiled stepper."""
    n, duration = 128, .5
    dx = 8 / n
    step_counts = (40, 80, 160)
    results = []
    with jax.default_device(jax.devices('cpu')[0]):
        z = -4 + (np.arange(n + 2*GHOSTS) - GHOSTS + .5)*dx
        # The largest dt covers all three runs' staggered history and stages.
        sol = RosenSolution(z, dx, duration / step_counts[0], duration)
        reference = _wave_fields(sol.bssn(duration), *sol.fields(duration))

        @jax.jit
        def step(state, params, time):
            # Pass dt through params rather than closing over it: all refinements
            # reuse the same compiled function and static boundary callback.
            return evolve.first_order_einstein_maxwell_step(
                state, params, time=time,
                prescribed_gauge=prescribed_gauge, stage_boundary=sol.boundary,
            )

        for steps in step_counts:
            dt = duration / steps
            params = BSSNParameters(
                dx=dx, dt=dt, nu=0., kappa=0., zero_shift=1, z_min=float(z[0]),
            )
            state = sol.state(0., dt)
            label = f'N={n}, dt={dt:.8g}, steps={steps}'
            _assert_gauge_and_constraints(state, params, f'{label}, t=0')
            for i in range(steps):
                state = step(state, params, jnp.asarray(i*dt))
            _assert_gauge_and_constraints(state, params, f'{label}, t={duration}')

            # D is averaged to the integer time; B already lives there. Neither
            # field is spatially recentered before comparison to its native sites.
            numerical = _wave_fields(
                state.bssn, *evolve.common_densitized_fields(state.em),
            )
            for name in _WAVE_FIELDS:
                actual, exact = numerical[name], reference[name]
                assert actual.dtype == exact.dtype == np.dtype('float64'), f'{label}: {name}'
                assert actual.shape == exact.shape and actual.shape[-1] == n, f'{label}: {name}'
                assert np.isfinite(actual).all() and np.isfinite(exact).all(), f'{label}: {name}'
            results.append(dict(numerical=numerical, reference=reference, dt=dt))
    return results


def test_em_wave_analytical_comparison(em_wave_convergence_runs):
    finest = em_wave_convergence_runs[-1]
    for name in _WAVE_FIELDS:
        error = _wave_rms(finest['numerical'][name] - finest['reference'][name])
        label = f'{name}; N=128, dt={finest["dt"]}, t=0.5'
        assert np.isfinite(error), label
        if name in ('D', 'B'):
            scale = _wave_rms(finest['reference'][name])
            assert np.isfinite(scale) and scale > 1e-14, f'{label}: reference RMS={scale}'
            relative = error / scale
            assert relative < .02, f'{label}: relative RMS={relative}, expected <0.02'
        else:
            tolerance = _GRAVITY_TOLERANCES[name]
            assert error < tolerance, f'{label}: RMS={error}, expected <{tolerance}'


def test_em_wave_temporal_self_convergence(em_wave_convergence_runs):
    runs = em_wave_convergence_runs
    timesteps = np.asarray([case['dt'] for case in runs])
    np.testing.assert_allclose(timesteps[:-1] / timesteps[1:], 2., rtol=0, atol=1e-14)
    for name in _WAVE_FIELDS:
        differences = np.asarray([
            _wave_rms(runs[i]['numerical'][name] - runs[i+1]['numerical'][name])
            for i in (0, 1)
        ])
        with np.errstate(divide='ignore', invalid='ignore'):
            order = np.log2(differences[0] / differences[1])
        message = (
            f'{name}; N=128, dt={timesteps.tolist()}; '
            f'RMS differences={differences.tolist()}; observed order={order}; '
            'expected decreasing differences and order >1.7'
        )
        # These resolved signals must not disappear into roundoff.
        assert np.all(np.isfinite(differences)) and np.all(differences > 1e-14), message
        assert differences[1] < differences[0], message
        assert np.isfinite(order) and order > 1.7, message
