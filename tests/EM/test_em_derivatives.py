import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.evolution.boundaries import PERIODIC_BC, SOMMERFELD_BC

from JAX_BSSN.EM.derivatives import first_derivative, second_derivative


def periodic_errors(grid_size):
    length = 2.0 * np.pi
    dx = length / grid_size
    x = dx * jnp.arange(grid_size, dtype=jnp.float64)
    field = jnp.sin(2.0 * x)

    first = first_derivative(
        field, 0, dx, PERIODIC_BC, PERIODIC_BC
    )
    second = second_derivative(
        field, 0, dx, PERIODIC_BC, PERIODIC_BC
    )
    first_error = float(jnp.sqrt(jnp.mean((first - 2.0 * jnp.cos(2.0 * x)) ** 2)))
    second_error = float(jnp.sqrt(jnp.mean((second + 4.0 * field) ** 2)))
    return first_error, second_error


def test_periodic_derivatives_are_second_order():
    coarse = periodic_errors(32)
    fine = periodic_errors(64)

    assert coarse[0] / fine[0] > 3.8
    assert coarse[1] / fine[1] > 3.8


def test_sommerfeld_one_sided_closures_are_exact_for_quadratic():
    dx = 0.2
    x = dx * jnp.arange(8, dtype=jnp.float64)
    field = x**2

    first = first_derivative(
        field, 0, dx, SOMMERFELD_BC, SOMMERFELD_BC
    )
    second = second_derivative(
        field, 0, dx, SOMMERFELD_BC, SOMMERFELD_BC
    )

    np.testing.assert_allclose(first, 2.0 * x, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(second, 2.0, rtol=0.0, atol=4.0e-14)


def test_component_axes_are_not_differentiated():
    dx = 0.1
    x = dx * jnp.arange(12, dtype=jnp.float64)
    scalar = jnp.sin(x)[:, None, None]
    vector = jnp.stack((scalar, 2.0 * scalar, -scalar), axis=0)

    derivative = first_derivative(
        vector, 1, dx, PERIODIC_BC, PERIODIC_BC
    )
    scalar_derivative = first_derivative(
        scalar, 0, dx, PERIODIC_BC, PERIODIC_BC
    )

    np.testing.assert_allclose(
        derivative,
        jnp.stack(
            (scalar_derivative, 2.0 * scalar_derivative, -scalar_derivative),
            axis=0,
        ),
        rtol=0.0,
        atol=2.0e-14,
    )

