"""Energy retention, adjointness, and stability of native quadratic coupling."""

import math

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn.variables import BSSNParameters
from JAX_BSSN.EM.first_order.coupling import (
    quadratic_moments, sources_from_moments, constitutive_fields,
)
from JAX_BSSN.EM.first_order.energy_momentum import compute_densitized_electromagnetic_energy_momentum
from JAX_BSSN.EM.first_order.equations import densitized_maxwell_rhs
from JAX_BSSN.EM.first_order.cartoon.cylindrical import (
    cylindrical_curls, cylindrical_divergences,
    axisymmetric_electromagnetic_sources,
)
from JAX_BSSN.EM.first_order.cartoon.axisymmetry import (
    _fill_vector_ghosts, _support_bssn,
)
from JAX_BSSN.EM.first_order.staggering import DISPLACEMENT_FIELD_LOCATIONS as DL, MAGNETIC_FIELD_LOCATIONS as BL
from tests.EM.em_helpers import flat_bssn_variables


def test_native_checkerboard_keeps_energy_and_stress():
    shape = (16, 1, 1)
    zero = jnp.zeros((3,) + shape)
    magnetic = zero.at[2, :, 0, 0].set((-1.)**jnp.arange(16))
    rho, momentum, stress = compute_densitized_electromagnetic_energy_momentum(
        zero, magnetic, flat_bssn_variables(shape), BSSNParameters(dx=.1))
    np.testing.assert_allclose(rho, .5, atol=1.e-14)
    np.testing.assert_allclose(momentum, 0., atol=1.e-14)
    expected = np.broadcast_to(np.diag([.5, .5, -.5])[:, :, None, None, None], stress.shape)
    np.testing.assert_allclose(stress, expected, atol=1.e-14)


def test_joint_moments_obey_stress_trace_and_energy_bound():
    shape = (8, 5, 7)
    rng = np.random.default_rng(19)
    d, b = (jnp.asarray(rng.normal(size=(3,) + shape)) for _ in range(2))
    raw = jnp.array([[1.5, .3, -.1], [.3, 1., .2], [-.1, .2, .8]])
    g = raw / jnp.linalg.det(raw)**(1/3)
    state = flat_bssn_variables(shape)._replace(
        conformal_metric=jnp.broadcast_to(g[:, :, None, None, None], (3, 3) + shape),
        conformal_factor=jnp.full(shape, .7))
    rho, momentum, stress = compute_densitized_electromagnetic_energy_momentum(d, b, state, BSSNParameters())
    inverse = .7**2 * jnp.linalg.inv(g)
    np.testing.assert_allclose(jnp.einsum("ij,ij...->...", inverse, stress), rho, atol=1.e-12)
    momentum_norm = jnp.sqrt(jnp.einsum("ij,i...,j...->...", inverse, momentum, momentum))
    assert np.all(rho >= 0)
    assert np.all(momentum_norm <= rho + 1.e-12)


def test_smooth_quadratic_sources_converge_at_second_order():
    errors = []
    for n in (16, 32, 64):
        dx = 2*math.pi/n
        p = BSSNParameters(dx=dx)
        x = dx*(jnp.arange(n)+.5)
        X, Z = jnp.meshgrid(x, x, indexing="ij")
        X, Z = X[:, None, :], Z[:, None, :]
        def fields(x, z):
            return (jnp.stack((.2*jnp.sin(x), .3*jnp.cos(z), .1*jnp.sin(x+z))),
                    jnp.stack((.1*jnp.cos(2*x), .2*jnp.sin(z), .15*jnp.cos(x-z))))
        native = []
        for which, locations in enumerate((DL, BL)):
            native.append(jnp.stack(tuple(fields(X-(dx/2 if loc[0] == "V" else 0),
                                                  Z-(dx/2 if loc[2] == "V" else 0))[which][i]
                                          for i, loc in enumerate(locations))))
        state = flat_bssn_variables((n, 1, n))
        shear = .4*jnp.sin(X)*jnp.cos(Z)
        state = state._replace(conformal_metric=state.conformal_metric.at[0, 2].set(shear).at[2, 0].set(shear).at[2, 2].set(1+shear**2))
        dc, bc = fields(X, Z)
        exact_moments = (jnp.einsum("i...,j...->ij...", dc, dc),
                         jnp.einsum("i...,j...->ij...", bc, bc),
                         jnp.einsum("i...,j...->ij...", dc, bc))
        exact = sources_from_moments(exact_moments, state)
        measured = compute_densitized_electromagnetic_energy_momentum(*native, state, p)
        errors.append([float(jnp.sqrt(jnp.mean((a-b)**2))) for a, b in zip(measured, exact)])
    assert np.all(np.log2(np.asarray(errors[:-1])/np.asarray(errors[1:])) > 1.8)


def test_one_sided_sampling_squares_extrapolated_fields():
    shape = (8, 1, 1)
    zero = jnp.zeros((3,) + shape)
    b = zero.at[2, :, 0, 0].set(jnp.arange(8, dtype=jnp.float64))
    p = BSSNParameters(dx=1., xl_bc=1, xr_bc=1)
    rho, _, _ = compute_densitized_electromagnetic_energy_momentum(zero, b, flat_bssn_variables(shape), p)
    # The missing upper vertex is linearly extrapolated to 8, then squared.
    np.testing.assert_allclose(rho[-1], .25*(7**2+8**2))
    assert np.all(rho >= 0)


def test_constitutive_is_energy_gradient_with_shift():
    shape = (6, 1, 5)
    p = BSSNParameters(dx=.2)
    rng = np.random.default_rng(22)
    d, b = (jnp.asarray(rng.normal(size=(3,) + shape)) for _ in range(2))
    x = jnp.arange(shape[0])[:, None, None]
    state = flat_bssn_variables(shape)._replace(
        lapse=jnp.broadcast_to(.9 + .03*jnp.sin(x), shape),
        conformal_factor=jnp.broadcast_to(.8 + .05*jnp.cos(x), shape),
        shift=jnp.broadcast_to(jnp.array([.1, -.04, .02])[:, None, None, None], (3,) + shape))

    def hamiltonian(d, b):
        rho, momentum, _ = sources_from_moments(quadratic_moments(d, b, p), state)
        return jnp.sum((state.lapse*rho - jnp.sum(state.shift*momentum, axis=0)) / state.conformal_factor**3)

    expected = jax.grad(hamiltonian, argnums=(0, 1))(d, b)
    actual = constitutive_fields(d, b, state, p)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, atol=2.e-13, rtol=2.e-13)


def test_periodic_sheared_metric_has_no_growing_maxwell_modes():
    # This smooth SPD unit-determinant metric had Re(lambda)=0.040228775
    # on the old target-site constitutive interpolation at this resolution.
    n = 12
    shape = (n, 1, n)
    size = n*n
    dx = 2*math.pi/n
    p = BSSNParameters(dx=dx, dt=.2*dx)
    x = dx*(jnp.arange(n)+.5)
    X, Z = jnp.meshgrid(x, x, indexing="ij")
    shear = 2*jnp.sin(X)[:, None, :]*jnp.sin(Z)[:, None, :]
    state = flat_bssn_variables(shape)
    metric = state.conformal_metric.at[0, 2].set(shear).at[2, 0].set(shear).at[2, 2].set(1+shear**2)
    state = state._replace(conformal_metric=metric)
    zero = jnp.zeros((3,) + shape)

    @jax.jit
    def action(q):
        d = zero.at[1].set(q[:size].reshape(shape))
        b = zero.at[0].set(q[size:2*size].reshape(shape)).at[2].set(q[2*size:].reshape(shape))
        dd, db = densitized_maxwell_rhs(d, b, state, p)
        return jnp.concatenate((dd[1].ravel(), db[0].ravel(), db[2].ravel()))

    matrix = np.stack([np.asarray(action(q)) for q in jnp.eye(3*size)], axis=1)
    eigenvalues = np.linalg.eigvals(matrix)
    assert np.max(eigenvalues.real) < 1.e-10


def _axis_setup(n=20, nz=17):
    dx = .1
    p = BSSNParameters(dx=dx, x_min=-3.5*dx, y_min=-4*dx,
                       z_min=-nz*dx/2, xr_bc=1, zl_bc=0, zr_bc=0)
    shape = (n+4, 1, nz)
    rng = np.random.default_rng(43)
    fields = []
    for locations in (DL, BL):
        field = jnp.asarray(rng.normal(size=(3,) + shape)).at[:, -5:].set(0.)
        fields.append(_fill_vector_ghosts(field, locations))
    return p, shape, fields


def test_cylindrical_curls_are_weighted_adjoint_and_divergence_free():
    p, shape, (e, h) = _axis_setup()
    dd, db = cylindrical_curls(e, h, p)
    divd, divb = cylindrical_divergences(dd, db, p)
    np.testing.assert_allclose(divd[4:-4], 0., atol=5.e-12)
    np.testing.assert_allclose(divb[4:-4], 0., atol=5.e-12)
    i = jnp.arange(shape[0]-4)
    mc = (i+.5)*p.dx**2
    mv = jnp.where(i == 0, p.dx**2/8, i*p.dx**2)
    md = jnp.stack((mv, mc, mc))[:, :, None, None]
    mb = jnp.stack((mc, mv, mv))[:, :, None, None]
    work = jnp.sum(md*e[:, 4:]*dd[:, 4:]) + jnp.sum(mb*h[:, 4:]*db[:, 4:])
    np.testing.assert_allclose(work, 0., atol=1.e-11)
    rc = (jnp.arange(shape[0])-3.5)*p.dx
    regular = jnp.zeros_like(e).at[1].set(jnp.broadcast_to(rc[:, None, None], shape))
    _, axis_rhs = cylindrical_curls(regular, jnp.zeros_like(h), p)
    np.testing.assert_allclose(axis_rhs[2, 4], -2., atol=1.e-13)


def test_cylindrical_constitutive_is_weighted_energy_gradient():
    p, shape, (d, b) = _axis_setup()
    d, b = d[:, 4:], b[:, 4:]
    state = flat_bssn_variables(d.shape[-3:])
    x = (jnp.arange(d.shape[1])+.5)[:, None, None]*p.dx
    z = jnp.arange(d.shape[-1])[None, None, :]*p.dx
    shear = .6*x*jnp.sin(z)
    state = state._replace(
        conformal_metric=state.conformal_metric.at[0, 2].set(shear).at[2, 0].set(shear).at[2, 2].set(1+shear**2),
        lapse=jnp.broadcast_to(.9+.03*jnp.sin(z), d.shape[-3:]),
        conformal_factor=jnp.broadcast_to(.8+.05*jnp.cos(x), d.shape[-3:]),
        shift=state.shift.at[0].set(.03*x).at[2].set(.02*jnp.cos(z)))
    rc = (jnp.arange(d.shape[1])+.5)*p.dx**2
    rv = jnp.where(jnp.arange(d.shape[1]) == 0, p.dx**2/8, jnp.arange(d.shape[1])*p.dx**2)
    md = jnp.stack((rv, rc, rc))[:, :, None, None]
    mb = jnp.stack((rc, rv, rv))[:, :, None, None]
    def energy(d, b):
        rho, momentum, _ = sources_from_moments(quadratic_moments(d, b, p, axisymmetric=True), state)
        return jnp.sum(rc[:, None, None]*(state.lapse*rho-jnp.sum(state.shift*momentum, axis=0))/state.conformal_factor**3)
    expected = jax.grad(energy, argnums=(0, 1))(d, b)
    e, h = constitutive_fields(d, b, state, p, axisymmetric=True)
    # Zero data in the outer buffer excludes the radiative halo closure.
    np.testing.assert_allclose((md*e)[:, :-4], expected[0][:, :-4], atol=2.e-13)
    np.testing.assert_allclose((mb*h)[:, :-4], expected[1][:, :-4], atol=2.e-13)


def test_cartoon_sources_reconstruct_moments_without_erasing_checkerboard():
    p, shape, _ = _axis_setup()
    state = flat_bssn_variables(shape)
    d = jnp.zeros((3,) + shape)
    b = d.at[2, 4:, 0, :].set((-1.)**jnp.arange(shape[0]-4)[:, None])
    b = _fill_vector_ghosts(b, BL)
    rho, momentum, stress = axisymmetric_electromagnetic_sources(d, b, _support_bssn(state, p), p)
    np.testing.assert_allclose(rho[4:-4, 4], .5, atol=2.e-13)
    np.testing.assert_allclose(momentum[:, 4:-4, 4], 0., atol=2.e-13)
    assert np.all(np.isfinite(stress))


def test_closed_cylindrical_sheared_metric_has_no_growing_modes():
    # Restrict the curl and constitutive maps separately: this closes the
    # boundary work, unlike restricting their product after evolution.
    n, nz = 6, 8
    p, shape, _ = _axis_setup(n+5, nz)
    state = flat_bssn_variables(shape)
    r = (jnp.arange(shape[0])-3.5)[:, None, None]*p.dx
    z = 2*math.pi*jnp.arange(nz)[None, None, :]/nz
    shear = 3*r*jnp.sin(z)
    state = state._replace(
        conformal_metric=state.conformal_metric.at[0, 2].set(shear).at[2, 0].set(shear).at[2, 2].set(1+shear**2))
    zero = jnp.zeros((3,) + shape)
    size = n*nz

    def unpack(q):
        d = zero.at[1, 4:4+n].set(q[:size].reshape(n, 1, nz))
        b = zero.at[0, 4:4+n].set(q[size:2*size].reshape(n, 1, nz)).at[2, 4:4+n].set(q[2*size:].reshape(n, 1, nz))
        return _fill_vector_ghosts(d, DL), _fill_vector_ghosts(b, BL)

    def pack(d, b, start):
        return jnp.concatenate((d[1, start:start+n].ravel(), b[0, start:start+n].ravel(), b[2, start:start+n].ravel()))

    @jax.jit
    def curl(q):
        return pack(*cylindrical_curls(*unpack(q), p), 4)

    @jax.jit
    def constitutive(q):
        d, b = unpack(q)
        positive = jax.tree_util.tree_map(lambda f: f[..., 4:, :, :], state)
        return pack(*constitutive_fields(d[:, 4:], b[:, 4:], positive, p, axisymmetric=True), 0)

    basis = jnp.eye(3*size)
    C = np.stack([np.asarray(curl(q)) for q in basis], axis=1)
    K = np.stack([np.asarray(constitutive(q)) for q in basis], axis=1)
    rc = (np.arange(n)+.5)*p.dx**2
    rv = np.where(np.arange(n) == 0, p.dx**2/8, np.arange(n)*p.dx**2)
    volumes = np.repeat(np.concatenate((rc, rc, rv)), nz)
    np.testing.assert_allclose(volumes[:, None]*C + C.T*volumes, 0., atol=1.e-13)
    np.testing.assert_allclose(volumes[:, None]*K, K.T*volumes, atol=1.e-13)
    assert np.linalg.eigvalsh(volumes[:, None]*K).min() > 0
    assert np.linalg.eigvals(C@K).real.max() < 1.e-10
