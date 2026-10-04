"""Exact packet, production stage hooks, and continuum reference regressions."""
import json

import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import pytest
from JAX_BSSN.bssn.variables import BSSNParameters
from JAX_BSSN.EM.first_order import evolve
from demos.EM_waves.solution import Packet, RosenSolution, GHOSTS, prescribed_gauge, profile


def solution(n=32,packet=Packet(),duration=.1):
    dx=8/n; dt=.2*dx
    z=-4+(np.arange(n+2*GHOSTS)-GHOSTS+.5)*dx
    return RosenSolution(z,dx,dt,duration,packet),BSSNParameters(dx=dx,dt=dt,nu=0,kappa=0,zero_shift=1)


def test_continuum_tensors_and_normalization():
    from demos.EM_waves.validate import symbolic_audit
    assert symbolic_audit()['passed']


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


# Explicit lists prevent a missing field or zero error from silently removing
# an equation from the convergence checks.
_WAVE_FIELDS = {
    'D': ('D_native', 'D_native_exact'),
    'B': ('B_native', 'B_native_exact'),
    'conformal_metric': ('bssn_conformal_metric', 'exact_conformal_metric'),
    'conformal_factor': ('bssn_conformal_factor', 'exact_conformal_factor'),
    'traceless_K': ('bssn_traceless_K', 'exact_traceless_K'),
    'trace_K': ('bssn_trace_K', 'exact_trace_K'),
    'conformal_connection': ('bssn_conformal_connection', 'exact_conformal_connection'),
}
_SPATIAL_FIELDS = {**_WAVE_FIELDS, 'rho': ('rho', 'rho_exact')}
_CONVERGING_CONSTRAINTS = ('hamiltonian', 'momentum', 'gamma_condition')
_ROUNDOFF_CONSTRAINTS = ('det_gamma', 'trace_A', 'div_D', 'div_B')


@pytest.fixture(scope='module')
def em_wave_convergence_runs(tmp_path_factory):
    """Five fresh production evolutions shared by space/time assertions."""
    from demos.EM_waves.run import run

    root = tmp_path_factory.mktemp('em_wave_convergence')
    cases = ((128, .2), (256, .2), (512, .2), (256, .1), (256, .05))
    results = {}
    for n, cfl in cases:
        output = root / f'n{n}_cfl{cfl:g}'
        rows = run(output, n=n, cfl=cfl, duration=4., frames=2, device='cpu')
        config = json.loads((output / 'configuration.json').read_text())
        label = f'N={n}, dt={config["dt"]:.8g}, output={output}'
        assert config['status'] == 'complete', label
        assert 'CPU' in config['device'].upper(), label
        assert len(rows) == 2, label
        np.testing.assert_array_equal([row['time'] for row in rows], [0., 4.], err_msg=label)

        # Snapshots contain only physical cells and common-time D/B (not
        # mismatched half-step histories from final_state.npz).
        with np.load(output / 'snapshots.npz') as data:
            np.testing.assert_array_equal(data['time'], [0., 4.], err_msg=label)
            assert data['z'].shape == (n,), label
            numerical, reference = {}, {}
            for name, (actual_key, exact_key) in _SPATIAL_FIELDS.items():
                actual, exact = data[actual_key], data[exact_key]
                assert actual.dtype == exact.dtype == np.dtype('float64'), f'{label}: {name}'
                assert actual.shape == exact.shape, f'{label}: {name}'
                assert actual.shape[0] == 2 and actual.shape[-1] == n, f'{label}: {name}'
                assert np.isfinite(actual).all() and np.isfinite(exact).all(), f'{label}: {name}'
                numerical[name], reference[name] = actual[-1].copy(), exact[-1].copy()
            for gauge, expected in (('lapse', 1.), ('shift', 0.)):
                np.testing.assert_allclose(
                    data[f'bssn_{gauge}'], expected, rtol=0, atol=1e-14,
                    err_msg=f'{label}: prescribed {gauge}',
                )

        for row in rows:
            for name in _ROUNDOFF_CONSTRAINTS:
                value = row['constraints'][name]
                assert np.isfinite(value) and 0 <= value < 1e-12, (
                    f'{label}, t={row["time"]}: {name} RMS={value}, expected <1e-12'
                )
        results[n, cfl] = dict(
            numerical=numerical, reference=reference,
            constraints=rows[-1]['constraints'], dt=config['dt'],
        )
    return results


def _wave_rms(value):
    return float(np.sqrt(np.mean(np.asarray(value)**2)))


def _assert_wave_convergence(field, levels, errors):
    errors = np.asarray(errors, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        orders = np.log2(errors[:-1] / errors[1:])
    message = (
        f'{field}; {levels}; RMS errors/differences={errors.tolist()}; '
        f'observed orders={orders.tolist()}; expected decreasing errors and order >1.7'
    )
    # These resolved signals are comfortably above roundoff. A zero/vanishing
    # error is a failure, not grounds for dropping the field from the study.
    assert np.all(np.isfinite(errors)) and np.all(errors > 1e-14), message
    assert np.all(errors[1:] < errors[:-1]), message
    assert np.all(np.isfinite(orders)) and np.all(orders > 1.7), message


def test_em_wave_spatial_convergence(em_wave_convergence_runs):
    resolutions = (128, 256, 512)
    runs = [em_wave_convergence_runs[n, .2] for n in resolutions]
    levels = f'N={resolutions}, dt={[case["dt"] for case in runs]}'
    for name in _SPATIAL_FIELDS:
        errors = [
            _wave_rms(case['numerical'][name] - case['reference'][name])
            for case in runs
        ]
        _assert_wave_convergence(name, levels, errors)
        if name in ('D', 'B'):
            scale = _wave_rms(runs[-1]['reference'][name])
            assert np.isfinite(scale) and scale > 1e-14, f'{name}: reference RMS={scale}'
            relative = errors[-1] / scale
            assert relative < .02, f'{name}; {levels}; finest relative RMS={relative}, expected <0.02'
    for name in _CONVERGING_CONSTRAINTS:
        _assert_wave_convergence(
            name, levels, [case['constraints'][name] for case in runs],
        )


def test_em_wave_temporal_convergence(em_wave_convergence_runs):
    runs = [em_wave_convergence_runs[256, cfl] for cfl in (.2, .1, .05)]
    timesteps = np.asarray([case['dt'] for case in runs])
    np.testing.assert_allclose(timesteps[:-1] / timesteps[1:], 2., rtol=0, atol=1e-14)
    for name in _WAVE_FIELDS:
        differences = [
            _wave_rms(runs[i]['numerical'][name] - runs[i+1]['numerical'][name])
            for i in (0, 1)
        ]
        _assert_wave_convergence(name, f'N=256, dt={timesteps.tolist()}', differences)
