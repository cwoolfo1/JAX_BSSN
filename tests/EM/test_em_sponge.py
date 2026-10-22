"""Native reference-sponge integration, placement, and time-centering tests."""

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from PyPIC3D.boundary_conditions.reference_sponge import sponge_advance
from PyPIC3D.boundary_conditions.staggered import refresh_fields
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS
from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.EM.first_order.grid import make_em_grid
from JAX_BSSN.EM.first_order.absorber import make_reference_sponge
from JAX_BSSN.EM.first_order.geometry import build_yee_metric
from JAX_BSSN.EM.first_order.maxwell import initialize_native_state, maxwell_half_step, rephase_state_metric
from JAX_BSSN.EM.first_order.evolve import common_densitized_fields, first_order_einstein_maxwell_step
from JAX_BSSN.EM.first_order.coupling import quadratic_moments, sources_from_moments
from JAX_BSSN.EM.variables import EinsteinMaxwellVariables
from tests.EM.em_helpers import flat_bssn_variables


def test_exact_relaxation_and_zero_rate():
    q = jnp.array([2., 3., 4.])
    ref = jnp.array([1., -1., 0.5])
    rate = jnp.array([0., 0.3, 40.])
    actual = jax.jit(sponge_advance)(q, jnp.zeros_like(q), .2, rate, ref)
    np.testing.assert_allclose(actual, ref + np.exp(-np.array(rate) * .2) * (q-ref), atol=1e-15)
    rhs = jnp.array([.3, -.5, .6])
    np.testing.assert_array_equal(sponge_advance(q, rhs, .2, jnp.zeros_like(q), ref), q + .2*rhs)
    np.testing.assert_array_equal(sponge_advance(ref, jnp.zeros_like(q), .2, rate, ref), ref)
    for dt in (.2, -.1):
        weight = np.array([dt, -np.expm1(-.3*dt)/.3, -np.expm1(-40*dt)/40])
        exact = ref + np.exp(-np.array(rate)*dt)*(q-ref) + weight*rhs
        np.testing.assert_allclose(sponge_advance(q,rhs,dt,rate,ref),exact,rtol=2e-15)


def test_native_profiles_and_reference_faces():
    p = BSSNParameters(dx=.25, dt=.05, x_min=-2.875, y_min=-2.875, z_min=-2.875)
    grid = make_em_grid((24,)*3, p, boundary_conditions=('constant',)*3)
    metric = build_yee_metric(flat_bssn_variables(grid.shape), grid)
    refs = tuple(jnp.ones((1,1,1,30,30,30)) * (i+1) for i in range(3))
    absorber = make_reference_sponge(grid, p, refs, refs, width=1.)
    for kind, locs, data in zip(('D','B'), (D_FIELD_LOCATIONS, B_FIELD_LOCATIONS), absorber):
        rates, reference, exterior = data
        refreshed = refresh_fields(tuple(q+1 for q in refs), grid.static, locs, kind, metric, data)
        for component, location in enumerate(locs):
            sigma = np.asarray(rates[component])[0,0,0]
            mask = np.asarray(exterior[component])
            np.testing.assert_array_equal(np.asarray(refreshed[component])[mask], np.asarray(reference[component])[mask])
            np.testing.assert_array_equal(np.asarray(refreshed[component])[~mask], np.asarray(refs[component]+1)[~mask])
            for axis, loc in enumerate(location):
                # C sites reflect around half indices; V includes lower halo face.
                index = [slice(3,27)]*3
                index[axis] = slice(2,27) if loc == 'V' else slice(3,27)
                slab = sigma[tuple(index)]
                np.testing.assert_allclose(slab, np.flip(slab, axis), atol=1e-13)
            assert np.all(sigma[8:21,8:21,8:21] == 0)
            coords = grid.coordinates(p,location,tiled=True)
            sigma_max = 5*np.log(1000)
            # Native offsets matter at each face; face contributions must add,
            # including intersections where two or three layers are active.
            for index in ((26,15,15),(26,26,15),(26,26,26)):
                xyz = [float(q.reshape(-1)[index[a]]) for a,q in enumerate(coords)]
                expected = sigma_max*sum(max(abs(q)-2,0)**4 for q in xyz)
                np.testing.assert_allclose(sigma[index],expected,rtol=2e-15)
    # A metric refresh must not turn fixed native references into fixed physical fields.
    q = flat_bssn_variables(grid.shape)._replace(conformal_factor=jnp.full(grid.shape,.8))
    changed = build_yee_metric(q, grid)
    for kind, locs, data in zip(('D','B'), (D_FIELD_LOCATIONS, B_FIELD_LOCATIONS), absorber):
        actual = refresh_fields(data[1], grid.static, locs, kind, changed, data)
        for a,b in zip(actual,data[1]):
            np.testing.assert_array_equal(a,b)


def damped_wave(nsteps, *, coupled=False, varying_lapse=False):
    shape = (24,1,1)
    T = .3
    p = BSSNParameters(dx=2*np.pi/24, dt=T/nsteps, nu=0)
    grid = make_em_grid(shape,p)
    bssn = flat_bssn_variables(shape)
    metric = build_yee_metric(bssn,grid)
    amp = .02 if coupled else 1.
    rate = .7
    omega = 2*np.sin(p.dx/2)/p.dx

    def fields(t):
        values = []
        for locs, active in ((D_FIELD_LOCATIONS,1),(B_FIELD_LOCATIONS,2)):
            vector = []
            for i, loc in enumerate(locs):
                x, _, _ = grid.coordinates(p,loc,tiled=True)
                f = amp*jnp.exp(-rate*t)*jnp.sin(x-omega*(t + int(varying_lapse)*t*t/2)) if i == active else jnp.zeros_like(x)
                vector.append(jnp.broadcast_to(f,(1,1,1,30,7,7)))
            values.append(tuple(vector))
        return tuple(values)

    d,b = fields(0.)
    absorber = tuple((tuple(jnp.full_like(q,rate) for q in v), tuple(jnp.zeros_like(q) for q in v),
                      tuple(jnp.zeros(q.shape,dtype=bool) for q in v)) for v in (d,b))
    em = initialize_native_state(d,b,metric,p,grid,absorber)
    if coupled:
        state = EinsteinMaxwellVariables(bssn,em)
        final = jax.jit(lambda s: jax.lax.fori_loop(0,nsteps,lambda i,q: first_order_einstein_maxwell_step(q,p,grid),s))(state)
        return np.concatenate([np.ravel(a) for a in jax.tree_util.tree_leaves((final.bssn, common_densitized_fields(final.em)))])

    def metric_at(t):
        return build_yee_metric(bssn._replace(lapse=bssn.lapse*(1+int(varying_lapse)*t)),grid)

    def step(i,q):
        q = rephase_state_metric(q,metric_at(i*p.dt),p,grid)
        q = maxwell_half_step(q,p,grid)
        q = rephase_state_metric(q,metric_at((i+1)*p.dt),p,grid)
        return maxwell_half_step(q,p,grid)

    em = jax.jit(lambda q: jax.lax.fori_loop(0,nsteps,step,q))(em)
    h = p.dt/2
    pairs = [(em.fields[0],fields(T)[0]), (em.synchronized_magnetic,fields(T)[1]),
             (em.fields[1],fields(T-h/2)[1]), (em.fields[7][0],fields(T-h)[0]),
             (em.fields[7][1],fields(T-1.5*h)[1])]
    errors = [sum(float(jnp.mean((grid.from_tile(a)-grid.from_tile(b))**2)) for a,b in zip(actual,exact))**.5
              for actual,exact in pairs]
    actual = sources_from_moments(quadratic_moments(*common_densitized_fields(em),grid),bssn)
    exact = sources_from_moments(quadratic_moments(*fields(T),grid),bssn)
    errors.append(sum(float(jnp.mean((a-b)**2)) for a,b in zip(actual,exact))**.5)
    return np.array(errors)


@pytest.mark.parametrize('varying_lapse',[False,True])
def test_damped_history_and_sources_second_order(varying_lapse):
    errors = np.array([damped_wave(n,varying_lapse=varying_lapse) for n in (12,24,48)])
    assert np.all(errors[:-1]/errors[1:] > 3.4), errors


def test_coupled_sponge_second_order():
    q = [damped_wave(n,coupled=True) for n in (12,24,48)]
    ratio = np.linalg.norm(q[0]-q[1])/np.linalg.norm(q[1]-q[2])
    assert 3.4 < ratio < 4.6, ratio
