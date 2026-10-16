"""The half-cell Yee convention shared with PyPIC3D, pinned without importing it.

``JAX_BSSN.evolution.coordinates`` and ``JAX_BSSN.EM.first_order.grid`` each
spell out where a staggered ``V`` sample sits, and they once disagreed by a full
cell.  PyPIC3D settles it: ``location_interpolate`` moves C to V by averaging
with the *upper* neighbour, so ``V[i] = (C[i] + C[i+1])/2 = C[i] + dx/2``.

These checks deliberately avoid importing PyPIC3D so they run wherever the pure
geometry does; ``tests/EM/test_uniform_grid.py`` cross-checks the hardcoded
location tuples below against the real ``D_FIELD_LOCATIONS``/``B_FIELD_LOCATIONS``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from JAX_BSSN.bssn.variables import BSSNParameters
from JAX_BSSN.evolution.coordinates import (
    axis_coordinates,
    cylindrical_volume_weights,
    grid_coordinates,
)

jax.config.update('jax_enable_x64', True)

# PyPIC3D.relativity.core D_FIELD_LOCATIONS + B_FIELD_LOCATIONS, hardcoded so
# this module stays importable without the EM stack.
YEE_LOCATIONS = (
    ("V", "C", "C"), ("C", "V", "C"), ("C", "C", "V"),
    ("C", "V", "V"), ("V", "C", "V"), ("V", "V", "C"),
)

SIZE = 12


def params(dx=0.25):
    return BSSNParameters(dx=dx, dt=0.1*dx, x_min=-3.5*dx, y_min=-4.0*dx, z_min=1.0)


@pytest.mark.parametrize('direction', range(3))
def test_vertex_sites_sit_half_a_cell_above_centers(direction):
    """V = C + dx/2: PyPIC3D's upper-half-cell rule, the convention that drifted."""
    p = params()
    centers = np.asarray(axis_coordinates(SIZE, direction, p, jnp.float64, "C"))
    vertices = np.asarray(axis_coordinates(SIZE, direction, p, jnp.float64, "V"))
    np.testing.assert_allclose(vertices - centers, 0.5*p.dx, atol=1e-15)


@pytest.mark.parametrize('direction', range(3))
def test_center_sites_are_the_unshifted_physical_samples(direction):
    """The C path carries no offset; boundaries.radial_derivative depends on it."""
    p = params()
    minimum = (p.x_min, p.y_min, p.z_min)[direction]
    expected = minimum + p.dx*np.arange(SIZE)
    np.testing.assert_allclose(
        np.asarray(axis_coordinates(SIZE, direction, p, jnp.float64, "C")),
        expected, atol=1e-15)
    # The default site must stay C: every production caller relies on it.
    np.testing.assert_array_equal(
        np.asarray(axis_coordinates(SIZE, direction, p, jnp.float64)), expected)


@pytest.mark.parametrize('location', YEE_LOCATIONS)
def test_grid_coordinates_staggers_each_axis_independently(location):
    p = params()
    shape = (SIZE, SIZE + 1, SIZE + 2)
    minima = (p.x_min, p.y_min, p.z_min)
    for direction, coordinate in enumerate(
        grid_coordinates(shape, p, jnp.float64, location)
    ):
        expected = minima[direction] + p.dx*(
            np.arange(shape[direction])
            + (0.5 if location[direction] == "V" else 0.0)
        )
        np.testing.assert_allclose(np.ravel(coordinate), expected, atol=1e-15)


@pytest.mark.parametrize('location', YEE_LOCATIONS)
def test_cylindrical_weights_use_the_radial_site_of_the_field(location):
    p = params()
    shape = (SIZE, 1, SIZE + 2)
    rho = p.x_min + p.dx*(
        np.arange(SIZE) + (0.5 if location[0] == "V" else 0.0)
    )
    np.testing.assert_allclose(
        np.asarray(cylindrical_volume_weights(shape, p, jnp.float64, location)),
        np.broadcast_to(np.abs(rho)[:, None], (SIZE, SIZE + 2)), atol=1e-15)


def test_default_grid_coordinates_are_unstaggered():
    """grid_coordinates() with no location is what the Sommerfeld radius uses."""
    p = params()
    shape = (SIZE, SIZE + 1, SIZE + 2)
    for explicit, default in zip(
        grid_coordinates(shape, p, jnp.float64, ("C", "C", "C")),
        grid_coordinates(shape, p, jnp.float64),
    ):
        np.testing.assert_array_equal(np.asarray(explicit), np.asarray(default))
