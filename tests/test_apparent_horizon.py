"""Exact-metric and convergence checks for the generalized horizon solver.

The same analytic Schwarzschild slice is handed to the solver as a radial line,
a signed y=0 plane and a full 3D volume, so the three symmetry paths are held to
one answer rather than to three separately tuned ones.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from JAX_BSSN.utilities.apparent_horizon import (
    find_apparent_horizon, angular_grid, angular_derivatives, surface, dense_jacobian,
)
from JAX_BSSN.utilities.apparent_horizon import build_geometry
import jax.numpy as jnp

MASS = 2.0


def schwarzschild_tensors(position, mass=MASS, slicing="isotropic",
                          scale=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0)):
    """Physical Schwarzschild tensors pulled back to stretched coordinates.

    ``scale`` maps the grid coordinates onto the coordinates the analytic
    solution is written in, so the horizon becomes a coordinate ellipsoid of
    semi-axes mass/(2 scale) about ``center``.
    """

    stretched = [factor*(q - shift) for q, factor, shift in zip(position, scale, center)]
    radius = np.sqrt(sum(q**2 for q in stretched))
    identity = np.eye(3).reshape((3, 3) + (1,)*radius.ndim)
    if slicing == "isotropic":
        gamma = identity*(1 + mass/(2*radius))**4
        Kij = np.zeros_like(gamma)
    else:
        normal = np.stack([q/radius for q in stretched])
        outer = normal[:, None]*normal[None, :]
        gamma = identity + (2*mass/radius)*outer
        Kij = (2*mass/(radius**2*np.sqrt(1 + 2*mass/radius)))*(
            identity - (2 + mass/radius)*outer)
    factors = np.asarray(scale)
    pullback = (factors[:, None]*factors[None, :]).reshape((3, 3) + (1,)*radius.ndim)
    return gamma*pullback, Kij*pullback


def schwarzschild_line(dx=0.02, extent=8.0, **keywords):
    radius = (np.arange(int(round(extent/dx))) + 0.5)*dx
    zero = np.zeros_like(radius)
    return (radius,) + schwarzschild_tensors((radius, zero, zero), **keywords)


def schwarzschild_plane(dx=0.05, extent=4.0, stretch=1.0, center_z=0.0, **keywords):
    count = 2*int(round(extent/dx))
    axis = (np.arange(count) - (count - 1)/2)*dx
    first, second = np.meshgrid(axis, axis, indexing="ij")
    tensors = schwarzschild_tensors(
        (first, np.zeros_like(first), second), scale=(1.0, 1.0, stretch),
        center=(0.0, 0.0, center_z), **keywords)
    return (axis, axis) + tensors


def schwarzschild_volume(dx=0.0625, extent=2.5, **keywords):
    count = 2*int(round(extent/dx))
    axis = (np.arange(count) - (count - 1)/2)*dx
    return (axis, axis, axis) + schwarzschild_tensors(
        np.meshgrid(axis, axis, axis, indexing="ij"), **keywords)


@pytest.fixture(autouse=True)
def release_compilations():
    yield
    jax.clear_caches()


@pytest.mark.parametrize('layout', ['line', 'plane', 'volume'])
@pytest.mark.parametrize('slicing', ['isotropic', 'kerr_schild'])
def test_exact_mass_all_layouts(layout, slicing):
    mass = 1.0 if slicing == 'kerr_schild' else MASS
    options = dict(mass=mass, slicing=slicing)
    if layout == 'line':
        r, gamma, Kij = schwarzschild_line(**options)
        axes = (r,)
    elif layout == 'plane':
        x, z, gamma, Kij = schwarzschild_plane(**options)
        axes = (x, z)
    else:
        x, y, z, gamma, Kij = schwarzschild_volume(dx=.05, extent=3., **options)
        axes = (x, y, z)
    exact_radius = 2*mass if slicing == 'kerr_schild' else mass/2
    horizon = find_apparent_horizon(axes, gamma, Kij, center=(0, 0, 0),
                                    initial_radius=1.15*exact_radius,
                                    n_theta=6, n_phi=12, max_angular=24)
    np.testing.assert_allclose(horizon['mass'], mass, rtol=1.e-3)
    np.testing.assert_allclose(horizon['radius'], exact_radius, atol=.002)
    assert horizon['radius'].shape[1] > 1  # No symmetry-reduced unknowns.
    assert horizon['residual'] < 1.e-8
    assert horizon['audit_residual'] < 1.e-4
    assert len(horizon['convergence']) >= 2


def test_nonspherical_guess_and_fixed_displaced_center():
    center = (0., 0., .15)
    x, z, gamma, Kij = schwarzschild_plane(stretch=1.3, center_z=center[2], dx=.03)
    theta, phi, _ = angular_grid(12, 24)
    guess = 1.1 + .05*np.sin(theta[:, None])**2*np.cos(2*phi[None, :])
    h = find_apparent_horizon((x, z), gamma, Kij, center=center, initial_radius=guess)
    np.testing.assert_array_equal(h['center'], center)
    np.testing.assert_allclose(h['mass'], MASS, rtol=.002)
    radius = h['radius']
    norm = radius*np.sqrt(np.sin(h['theta'][:, None])**2 + (1.3*np.cos(h['theta'][:, None]))**2)
    np.testing.assert_allclose(norm, 1., atol=.003)
    assert 'recenter_shift' not in h and 'seed_radius' not in h


def test_triaxial_volume():
    scale = (1., 1.25, 1.5)
    x, y, z, gamma, Kij = schwarzschild_volume(dx=.04, scale=scale)
    h = find_apparent_horizon((x,y,z), gamma, Kij, center=(0,0,0), initial_radius=.9)
    theta, phi = h['theta'][:, None], h['phi'][None, :]
    norm = h['radius']*np.sqrt((scale[0]*np.sin(theta)*np.cos(phi))**2
        + (scale[1]*np.sin(theta)*np.sin(phi))**2 + (scale[2]*np.cos(theta))**2)
    np.testing.assert_allclose(norm, 1., atol=.003)
    np.testing.assert_allclose(h['mass'], MASS, rtol=.002)
    assert np.ptp(h['radius'], axis=1).max() > .1


def test_spatial_convergence():
    errors = []
    for dx in (.12, .06):
        r, gamma, Kij = schwarzschild_line(dx=dx)
        h = find_apparent_horizon((r,), gamma, Kij, center=(0,0,0), initial_radius=1.1,
                                  n_theta=6, n_phi=12, max_angular=12)
        errors.append(abs(h['mass'] - MASS))
    assert errors[1] < errors[0]/4


def test_pole_stencil_and_angular_convergence():
    errors = []
    for n in (12, 24):
        theta, phi, _ = angular_grid(n, 2*n)
        field = jnp.asarray(np.sin(theta[:, None])*np.cos(phi[None, :]))
        dt, dtt, dp, dpp, dtp = angular_derivatives(field)
        expected = np.cos(theta[:, None])*np.cos(phi[None, :])
        errors.append(float(jnp.max(jnp.abs(dt-expected))))
        np.testing.assert_allclose(dpp, -field, atol=.001)
    assert errors[1] < errors[0]/12


def test_dense_jacobian_matches_full_ad_and_finite_difference():
    r, gamma, Kij = schwarzschild_line()
    table, sample = build_geometry((r,), gamma, Kij)
    theta, phi, _ = angular_grid(6, 12)
    h = jnp.asarray(1.1 + .03*np.sin(theta[:,None])**2*np.cos(2*phi[None,:]))
    residual = lambda shape: surface(shape, jnp.zeros(3), table, sample)[0].reshape(-1)
    jacobian = jax.jit(lambda q: dense_jacobian(residual,q))(h)
    reference = jax.jacfwd(residual)(h).reshape(h.size,h.size)
    np.testing.assert_allclose(jacobian, reference, atol=1.e-11)
    direction = jnp.asarray(np.random.default_rng(4).normal(size=h.shape))
    eps = 1.e-6
    fd = (residual(h+eps*direction)-residual(h-eps*direction))/(2*eps)
    np.testing.assert_allclose(jacobian @ direction.reshape(-1), fd, rtol=1.e-6, atol=1.e-8)


def test_failures_do_not_search_for_another_seed():
    r, gamma, Kij = schwarzschild_line()
    flat = np.broadcast_to(np.eye(3)[:,:,None], gamma.shape)
    with pytest.raises(RuntimeError, match='Newton'):
        find_apparent_horizon((r,), flat, Kij, center=(0,0,0), initial_radius=1.,
                              n_theta=6, n_phi=12, max_angular=12, max_iterations=2)
    with pytest.raises(RuntimeError, match='resolved domain'):
        find_apparent_horizon((r,), gamma, Kij, center=(0,0,0), initial_radius=20.)
    with pytest.raises(RuntimeError, match='Newton did not converge'):
        find_apparent_horizon((r,), gamma, Kij, center=(0,0,0), initial_radius=1.2,
                              max_iterations=0)
    bad = gamma.copy(); bad[0,0] *= -1
    with pytest.raises(RuntimeError, match='nonphysical'):
        find_apparent_horizon((r,), bad, Kij, center=(0,0,0), initial_radius=1.)


@pytest.mark.parametrize('guess', [np.ones(12), np.ones((12, 13)), -1., np.full((6,12),np.nan)])
def test_bad_initial_surface(guess):
    r, gamma, Kij = schwarzschild_line()
    with pytest.raises(ValueError):
        find_apparent_horizon((r,), gamma, Kij, center=(0,0,0), initial_radius=guess)


def test_center_and_guess_are_required():
    r, gamma, Kij = schwarzschild_line()
    with pytest.raises(TypeError):
        find_apparent_horizon((r,), gamma, Kij)


def test_residual_ad_on_both_sides_of_spatial_cell_boundary():
    r, gamma, Kij = schwarzschild_line()
    table, sample = build_geometry((r,), gamma, Kij)
    residual = jax.jit(lambda h: surface(h,jnp.zeros(3),table,sample)[0].reshape(-1))
    direction = jnp.ones((6,12))
    knot = r[55]
    for side in (-1., 1.):
        h = direction*(knot + side*1.e-5)
        _, derivative = jax.jvp(residual,(h,),(direction,))
        eps = 1.e-7
        finite_difference = (residual(h+eps*direction)-residual(h-eps*direction))/(2*eps)
        np.testing.assert_allclose(derivative,finite_difference,rtol=1.e-5,atol=1.e-7)
    # The expansion is continuous at a knot; its derivative need not be, since
    # cubic Hermite geometry is C1, not C2. Do not test a nonexistent derivative.
    np.testing.assert_allclose(residual(direction*(knot-1.e-7)),
                               residual(direction*(knot+1.e-7)),atol=1.e-6)


def test_float64_is_required():
    r, gamma, Kij = schwarzschild_line()
    previous = jax.config.x64_enabled
    jax.config.update('jax_enable_x64', False)
    try:
        with pytest.raises(ValueError,match='jax_enable_x64'):
            find_apparent_horizon((r,),gamma,Kij,center=(0,0,0),initial_radius=1.)
    finally:
        jax.config.update('jax_enable_x64', previous)
