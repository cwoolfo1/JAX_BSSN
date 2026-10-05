"""Native Yee coordinates and parity for compact and signed Cartoon states."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from JAX_BSSN.EM.first_order.cartoon import (
    compact_axisymmetric_densitized_state,
    compact_spherical_densitized_state,
    expand_axisymmetric_densitized_state,
    expand_spherical_densitized_state,
)
from JAX_BSSN.EM.first_order.staggering import (
    DISPLACEMENT_FIELD_LOCATIONS,
    MAGNETIC_FIELD_LOCATIONS,
)
from JAX_BSSN.EM.first_order.variables import DensitizedMaxwellState


@pytest.fixture(params=[False, True], ids=["axisymmetric", "spherical"])
def symmetry(request):
    if request.param:
        return (compact_spherical_densitized_state,
                expand_spherical_densitized_state, (-1, 1, 1), 1)
    return (compact_axisymmetric_densitized_state,
            expand_axisymmetric_densitized_state, (-1, -1, 1), 3)


def test_native_coordinates_ghosts_and_round_trip(symmetry):
    compact, expand, parity, nz = symmetry
    n, dx = 7, 0.25
    locations_by_field = (MAGNETIC_FIELD_LOCATIONS,) * 2 + (DISPLACEMENT_FIELD_LOCATIONS,) * 2
    signed, expected_compact = [], []
    for history, locations in enumerate(locations_by_field):
        signed_components, compact_components = [], []
        for component, location in enumerate(locations):
            offset = 0.0 if location[0] == "V" else 0.5
            scale = (history + 1) * (component + 1) * np.arange(1, nz + 1)

            def profile(x):
                radial = x if parity[component] == -1 else 1.0 + x**2
                return radial[:, None, None] * scale[None, None, :]

            samples = profile((np.arange(2 * n) - n + offset) * dx)
            if location[0] == "V":
                # Only the far negative endpoint has no stored mirror sample.
                samples[0] = parity[component] * (2 * samples[-1] - samples[-2])
            signed_components.append(samples)
            compact_components.append(profile((np.arange(n + 4) - 4 + offset) * dx))
        signed.append(jnp.asarray(np.stack(signed_components)))
        expected_compact.append(np.stack(compact_components))

    full = DensitizedMaxwellState(*signed)
    reduced = jax.jit(compact)(full)
    for actual, expected in zip(reduced, expected_compact):
        assert actual.shape == (3, n + 4, 1, nz)
        np.testing.assert_array_equal(actual, expected)
    restored = jax.jit(expand)(reduced)
    assert isinstance(restored, DensitizedMaxwellState)
    for actual, expected in zip(restored, full):
        assert actual.shape == (3, 2 * n, 1, nz)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[:, n:], expected[:, n:])
    for actual, expected in zip(compact(restored), reduced):
        np.testing.assert_array_equal(actual, expected)


def test_radial_displacement_has_one_axis_vertex(symmetry):
    compact, expand, _, nz = symmetry
    n = 6
    zero = jnp.zeros((3, n + 4, 1, nz))
    displacement = zero.at[0, 4:].set(jnp.arange(n)[:, None, None])
    state = DensitizedMaxwellState(zero, zero, displacement, 2 * displacement)
    full = expand(state)
    for field, scale in ((full.displacement_left_half, 1),
                         (full.displacement_right_half, 2)):
        expected = np.broadcast_to(scale * np.arange(-n, n)[:, None, None], (2 * n, 1, nz))
        np.testing.assert_array_equal(field[0], expected)
        assert float(field[0, n - 1, 0, 0]) == -scale
        assert np.count_nonzero(np.asarray(field[0, :, 0, 0]) == 0) == 1
    np.testing.assert_array_equal(compact(full).displacement_left_half[0, :4, 0, 0], [-4, -3, -2, -1])


def test_odd_axis_values_are_projected_to_zero(symmetry):
    compact, expand, parity, nz = symmetry
    n = 6
    values = jnp.ones((3, 2 * n, 1, nz))
    reduced = compact(DensitizedMaxwellState(values, values, values, values))
    locations_by_field = (MAGNETIC_FIELD_LOCATIONS,) * 2 + (DISPLACEMENT_FIELD_LOCATIONS,) * 2
    for field, locations in zip(reduced, locations_by_field):
        for component, location in enumerate(locations):
            expected = 0 if location[0] == "V" and parity[component] == -1 else 1
            np.testing.assert_array_equal(field[component, 4], expected)
    # Expansion must also enforce regularity when given unconditioned input.
    corrupted = DensitizedMaxwellState(*(q.at[:, 4].set(7.0) for q in reduced))
    for field, locations in zip(expand(corrupted), locations_by_field):
        for component, location in enumerate(locations):
            if location[0] == "V" and parity[component] == -1:
                np.testing.assert_array_equal(field[component, n], 0)
