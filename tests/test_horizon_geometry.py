"""Cross-symmetry checks for the geometry providers the horizon solver samples."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from JAX_BSSN.cartoon.interpolation import lagrange6_stencil, lagrange6_weights
from JAX_BSSN.utilities.horizon_geometry import (
    AxisymmetricGeometry,
    CartesianGeometry,
    SphericalGeometry,
    build_geometry,
    interpolate_point,
    lagrange_weights,
)

MASS = 2.0
SAMPLES = np.array([[0.7, 0.3, 0.5], [-0.4, 0.9, -0.6], [1.2, -0.2, 0.8],
                    [0.0, 1.0, 0.3], [0.9, 0.0, -1.1]])


def conformal_factor(radius):
    return (1 + MASS/(2*radius))**4


def exact_isotropic(points):
    """Physical gamma_ij and d_k gamma_ij of isotropic Schwarzschild."""

    radius = np.linalg.norm(points, axis=1)
    gamma = np.eye(3)[None]*conformal_factor(radius)[:, None, None]
    slope = 4*(1 + MASS/(2*radius))**3*(-MASS/(2*radius**2))
    normal = points/radius[:, None]
    return gamma, np.eye(3)[None, None]*(slope[:, None]*normal)[:, :, None, None]


def isotropic_line(count=500, spacing=0.02):
    radius = (np.arange(count) + 0.5)*spacing
    gamma = np.zeros((3, 3, count))
    for index in range(3):
        gamma[index, index] = conformal_factor(radius)
    return radius, gamma, np.zeros_like(gamma)


def isotropic_plane(count=200, spacing=0.05):
    axis = (np.arange(count) - (count - 1)/2)*spacing
    first, second = np.meshgrid(axis, axis, indexing="ij")
    gamma = np.zeros((3, 3, count, count))
    for index in range(3):
        gamma[index, index] = conformal_factor(np.sqrt(first**2 + second**2))
    return axis, axis, gamma, np.zeros_like(gamma)


def isotropic_volume(count=80, spacing=0.1):
    axis = (np.arange(count) - (count - 1)/2)*spacing
    grids = np.meshgrid(axis, axis, axis, indexing="ij")
    gamma = np.zeros((3, 3, count, count, count))
    for index in range(3):
        gamma[index, index] = conformal_factor(np.sqrt(sum(q**2 for q in grids)))
    return axis, axis, axis, gamma, np.zeros_like(gamma)


def test_lagrange_weights_match_the_shared_cartoon_helper():
    for index in (3.0, 3.5, 7.2, 0.1, 12.999):
        nodes = lagrange6_stencil(jnp.asarray(index), 20)
        shared = np.asarray(lagrange6_weights(jnp.asarray(index), nodes))
        local = np.asarray(lagrange_weights(jnp.asarray(index), nodes))
        np.testing.assert_allclose(local, shared, atol=1.e-14)
        np.testing.assert_allclose(local.sum(), 1.0, atol=1.e-14)


def test_interpolation_is_exact_for_degree_five():
    grids = np.meshgrid(*(np.arange(size) for size in (9, 10, 11)), indexing="ij")
    table = jnp.asarray(np.stack([
        (1 + grids[0])**5 - 3*grids[1]**4*grids[2] + grids[0]*grids[1]*grids[2]**3,
        np.cos(1.0)*grids[2]**5 + grids[0]**2*grids[1]**2,
    ]))
    for index in ([3.4, 2.1, 7.8], [1.5, 6.25, 2.0], [7.0, 0.5, 9.9]):
        first, second, third = index
        expected = [(1 + first)**5 - 3*second**4*third + first*second*third**3,
                    np.cos(1.0)*third**5 + first**2*second**2]
        np.testing.assert_allclose(
            np.asarray(interpolate_point(table, jnp.asarray(index))), expected, rtol=1.e-11)


def test_every_symmetry_reproduces_the_same_cartesian_tensors():
    gamma, gradient = exact_isotropic(SAMPLES)
    providers = (SphericalGeometry(*isotropic_line()),
                 AxisymmetricGeometry(*isotropic_plane()),
                 CartesianGeometry(*isotropic_volume()))
    for provider in providers:
        sampled, derivative, curvature, resolved = provider.sample(
            provider.tables, jnp.asarray(SAMPLES))
        assert np.all(np.asarray(resolved)), provider.symmetry
        # Tolerances track the coarsest layout, the 3D volume; a wrong rotation
        # or a wrong azimuthal derivative would miss by order unity, not 1e-4.
        np.testing.assert_allclose(np.asarray(sampled), gamma, rtol=1.e-4, atol=1.e-12)
        np.testing.assert_allclose(np.asarray(derivative), gradient, rtol=1.e-3, atol=1.e-9)
        np.testing.assert_allclose(np.asarray(curvature), 0.0, atol=1.e-12)


def test_axisymmetric_azimuthal_derivative_matches_the_rotation_generator():
    # The y derivative on the stored plane is fixed by the rotation generator.
    generator = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    axis = (np.arange(200) - 99.5)*0.05
    first, second = np.meshgrid(axis, axis, indexing="ij")
    radius = np.sqrt(first**2 + second**2)
    gamma = np.zeros((3, 3, 200, 200))
    gamma[0, 0] = conformal_factor(radius)*(1 + 0.3*first**2)
    gamma[1, 1] = conformal_factor(radius)
    gamma[2, 2] = conformal_factor(radius)*(1 + 0.1*second)
    gamma[0, 2] = gamma[2, 0] = 0.05*first*second
    provider = AxisymmetricGeometry(axis, axis, gamma, np.zeros_like(gamma))
    points = np.array([[0.8, 0.0, 0.4], [1.3, 0.0, -0.7]])
    sampled, derivative, _, _ = provider.sample(provider.tables, jnp.asarray(points))
    for index, point in enumerate(points):
        metric = np.asarray(sampled)[index]
        expected = (generator @ metric - metric @ generator)/point[0]
        np.testing.assert_allclose(np.asarray(derivative)[index, 1], expected, atol=1.e-12)


def test_points_outside_the_resolved_margin_are_flagged():
    provider = CartesianGeometry(*isotropic_volume())
    low, high = provider.bounds[0]
    outside = jnp.asarray([[high - 0.05, 0.0, 0.0], [0.0, low + 0.05, 0.0],
                           [0.0, 0.0, 2*high]])
    assert not np.any(np.asarray(provider.sample(provider.tables, outside)[3]))
    inside = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert np.all(np.asarray(provider.sample(provider.tables, inside)[3]))


def test_non_uniform_axes_are_refused():
    axis, _, gamma, Kij = isotropic_plane()
    stretched = axis.copy()
    stretched[-1] += 0.01
    with pytest.raises(ValueError, match="uniformly spaced"):
        AxisymmetricGeometry(stretched, axis, gamma, Kij)


def test_unsigned_axisymmetric_plane_is_refused():
    axis, _, gamma, Kij = isotropic_plane()
    with pytest.raises(ValueError, match="signed x-z plane"):
        AxisymmetricGeometry(abs(axis) + 0.1, axis, gamma, Kij)


def test_spherically_symmetric_input_must_be_isotropic_about_the_axis():
    radius, gamma, Kij = isotropic_line()
    gamma[1, 1] *= 1.05
    with pytest.raises(ValueError, match="equal transverse components"):
        SphericalGeometry(radius, gamma, Kij)


def test_builder_selects_the_provider_from_the_coordinate_count():
    radius, gamma, Kij = isotropic_line()
    assert build_geometry(radius, gamma, Kij).symmetry == "spherical"
    assert build_geometry((radius,), gamma, Kij).symmetry == "spherical"
    axis, _, plane, plane_curvature = isotropic_plane()
    assert build_geometry((axis, axis), plane, plane_curvature).symmetry == "axisymmetric"
    with pytest.raises(ValueError, match="not '3d'"):
        build_geometry((axis, axis), plane, plane_curvature, symmetry="3d")
