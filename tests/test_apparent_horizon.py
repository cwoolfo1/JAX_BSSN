"""Exact-metric and convergence checks for the generalized horizon solver.

The same analytic Schwarzschild slice is handed to the solver as a radial line,
a signed y=0 plane and a full 3D volume, so the three symmetry paths are held to
one answer rather than to three separately tuned ones.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from JAX_BSSN.utilities.apparent_horizon import find_apparent_horizon

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


@pytest.fixture(scope="module")
def isotropic_volume_horizon():
    axis, _, _, gamma, Kij = schwarzschild_volume()
    return find_apparent_horizon((axis, axis, axis), gamma, Kij,
                                 n_theta=12, n_phi=24, max_angular=24)


def test_spherical_symmetry_resolves_both_slicings():
    radius, gamma, Kij = schwarzschild_line()
    horizon = find_apparent_horizon((radius,), gamma, Kij)
    assert horizon["symmetry"] == "spherical"
    np.testing.assert_allclose(horizon["radius"], MASS/2, rtol=1.e-6)
    np.testing.assert_allclose(horizon["area"], 16*np.pi*MASS**2, rtol=1.e-8)
    np.testing.assert_allclose(horizon["mass"], MASS, rtol=1.e-8)
    assert horizon["audit_residual"] < 1.e-4

    radius, gamma, Kij = schwarzschild_line(mass=1.0, slicing="kerr_schild")
    horizon = find_apparent_horizon(radius, gamma, Kij)
    np.testing.assert_allclose(horizon["radius"], 2.0, rtol=1.e-6)
    np.testing.assert_allclose(horizon["mass"], 1.0, rtol=1.e-6)


def test_axisymmetric_mass_and_spatial_convergence():
    errors = []
    for dx in (0.06, 0.03):
        x, z, gamma, Kij = schwarzschild_plane(dx=dx)
        horizon = find_apparent_horizon((x, z), gamma, Kij)
        assert horizon["symmetry"] == "axisymmetric"
        np.testing.assert_allclose(horizon["radius"], 1.0, atol=0.002)
        np.testing.assert_allclose(horizon["area"], 64*np.pi, rtol=0.001)
        np.testing.assert_allclose(horizon["mass"], MASS, rtol=0.0005)
        assert horizon["residual"] < 1.e-6
        assert horizon["audit_residual"] < 1.e-4
        assert horizon["mass_relative_change"] < 1.e-3
        assert len(horizon["convergence"]) >= 2
        errors.append(abs(horizon["mass"] - MASS))
    assert errors[1] < errors[0]/4


def test_axisymmetric_kerr_schild_extrinsic_curvature_and_seed_consistency():
    x, z, gamma, Kij = schwarzschild_plane(mass=1.0, slicing="kerr_schild")
    horizon = find_apparent_horizon((x, z), gamma, Kij, initial_radii=[1.6, 2.4])
    np.testing.assert_allclose(horizon["radius"], 2.0, atol=0.001)
    np.testing.assert_allclose(horizon["mass"], 1.0, rtol=0.0001)
    assert horizon["candidate_count"] >= 3
    assert len(horizon["search_failures"]) == 0


def test_axisymmetric_stretched_horizon_recenters_and_resolves_its_shape():
    stretch, offset = 1.4, 0.15
    x, z, gamma, Kij = schwarzschild_plane(stretch=stretch, center_z=offset)
    horizon = find_apparent_horizon((x, z), gamma, Kij)
    # The search starts at the grid origin; the puncture sits above it.
    np.testing.assert_allclose(horizon["center"], [0.0, 0.0, offset], atol=1.e-4)
    np.testing.assert_allclose(horizon["recenter_shift"][2], offset, atol=1.e-4)
    theta, shape = horizon["theta"], horizon["radius"][:, 0]
    axial = stretch*(shape*np.cos(theta) + horizon["center"][2] - offset)
    np.testing.assert_allclose(np.hypot(shape*np.sin(theta), axial), 1.0, atol=0.002)
    np.testing.assert_allclose(horizon["mass"], MASS, rtol=0.001)
    assert np.ptp(shape) > 0.2


def test_three_dimensional_isotropic_mass(isotropic_volume_horizon):
    horizon = isotropic_volume_horizon
    assert horizon["symmetry"] == "3d"
    assert horizon["radius"].shape == (24, 48)
    np.testing.assert_allclose(horizon["radius"], 1.0, atol=0.001)
    np.testing.assert_allclose(horizon["mass"], MASS, rtol=0.0005)
    assert horizon["residual"] < 1.e-6
    assert horizon["audit_residual"] < 1.e-4


def test_three_dimensional_triaxial_surface_varies_with_azimuth():
    scale = (1.0, 1.25, 1.5)
    axis, _, _, gamma, Kij = schwarzschild_volume(scale=scale)
    # The coarse test volume, stretched, caps the interpolated metric accuracy.
    horizon = find_apparent_horizon((axis, axis, axis), gamma, Kij, n_theta=12,
                                    n_phi=24, max_angular=24, audit_tolerance=2.e-3)
    theta, phi, shape = horizon["theta"][:, None], horizon["phi"][None, :], horizon["radius"]
    position = (shape*np.sin(theta)*np.cos(phi), shape*np.sin(theta)*np.sin(phi),
                shape*np.cos(theta))
    physical = np.sqrt(sum((factor*q)**2 for factor, q in zip(scale, position)))
    np.testing.assert_allclose(physical, 1.0, atol=0.003)
    np.testing.assert_allclose(horizon["mass"], MASS, rtol=0.002)
    assert np.ptp(shape, axis=1).max() > 0.1  # Genuine azimuthal structure.
    assert np.ptp(shape, axis=0).max() > 0.1
    assert horizon["audit_residual"] < 1.e-3


def test_every_symmetry_reports_the_same_mass(isotropic_volume_horizon):
    radial = find_apparent_horizon(*schwarzschild_line())
    x, z, gamma, Kij = schwarzschild_plane()
    axial = find_apparent_horizon((x, z), gamma, Kij)
    masses = [radial["mass"], axial["mass"], isotropic_volume_horizon["mass"]]
    np.testing.assert_allclose(masses, MASS, rtol=0.0005)
    assert max(masses) - min(masses) < 0.001


def test_angular_convergence_history_doubles_the_grid(isotropic_volume_horizon):
    history = isotropic_volume_horizon["convergence"]
    assert len(history) >= 2
    np.testing.assert_array_equal(history[:, 0], [12, 24])
    np.testing.assert_array_equal(history[:, 1], [24, 48])
    assert abs(history[-1, 2]/history[0, 2] - 1) < 1.e-3


def test_unpreconditioned_krylov_solve_agrees(isotropic_volume_horizon):
    axis, _, _, gamma, Kij = schwarzschild_volume()
    horizon = find_apparent_horizon((axis, axis, axis), gamma, Kij, n_theta=12,
                                    n_phi=24, max_angular=24, preconditioned=False)
    np.testing.assert_allclose(horizon["mass"], isotropic_volume_horizon["mass"],
                               rtol=1.e-8)


def test_flat_space_has_no_horizon():
    axis = (np.arange(80) - 39.5)*0.1
    gamma = np.broadcast_to(np.eye(3)[:, :, None, None], (3, 3, 80, 80))
    with pytest.raises(RuntimeError, match="No horizon seed"):
        find_apparent_horizon((axis, axis), gamma, np.zeros_like(gamma))


def test_insufficient_domain_fails():
    axis = (np.arange(12) - 5.5)*0.1
    gamma = np.broadcast_to(np.eye(3)[:, :, None, None], (3, 3, 12, 12))
    with pytest.raises(ValueError, match="Insufficient resolved domain"):
        find_apparent_horizon((axis, axis), gamma, np.zeros_like(gamma))


def test_horizon_outside_domain_is_not_extrapolated():
    x, z, gamma, Kij = schwarzschild_plane(extent=3.0, slicing="kerr_schild")
    with pytest.raises(RuntimeError, match="No horizon seed"):
        find_apparent_horizon((x, z), gamma, Kij)


def test_single_precision_is_refused(monkeypatch):
    monkeypatch.setattr(jax.config, "read", lambda name: False)
    with pytest.raises(ValueError, match="jax_enable_x64"):
        find_apparent_horizon(*schwarzschild_line())
