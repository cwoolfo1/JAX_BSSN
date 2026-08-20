"""Tests for periodic and Sommerfeld finite-difference closures."""

import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.boundaries import (
    PERIODIC_BC,
    SOMMERFELD_BC,
    SUPERGAUSSIAN_BC,
)
from JAX_BSSN.derivatives import diff1_field, diff6_field


class TestBoundaryAwareDerivatives(unittest.TestCase):
    def setUp(self):
        self.n = 12
        self.dx = 0.2
        self.x = -0.7 + self.dx * jnp.arange(self.n, dtype=jnp.float64)

    def test_periodic_first_and_sixth_derivatives_are_unchanged(self):
        field = jnp.sin(self.x) + 0.2 * jnp.cos(2.0 * self.x)

        expected_first = (
            2.0 / 3.0 * (jnp.roll(field, -1) - jnp.roll(field, 1)) / self.dx
            - 1.0 / 12.0 * (jnp.roll(field, -2) - jnp.roll(field, 2)) / self.dx
        )
        expected_sixth = (
            jnp.roll(field, -3)
            - 6.0 * jnp.roll(field, -2)
            + 15.0 * jnp.roll(field, -1)
            - 20.0 * field
            + 15.0 * jnp.roll(field, 1)
            - 6.0 * jnp.roll(field, 2)
            + jnp.roll(field, 3)
        ) / self.dx**6

        np.testing.assert_allclose(
            diff1_field(field, 0, self.dx, PERIODIC_BC, PERIODIC_BC),
            expected_first,
            rtol=0.0,
            atol=np.finfo(float).eps * 4,
        )
        np.testing.assert_allclose(
            diff6_field(field, 0, self.dx, PERIODIC_BC, PERIODIC_BC),
            expected_sixth,
            rtol=1.0e-13,
            atol=1.0e-10,
        )
        np.testing.assert_array_equal(
            diff1_field(
                field, 0, self.dx, SUPERGAUSSIAN_BC, SUPERGAUSSIAN_BC
            ),
            expected_first,
        )
        np.testing.assert_allclose(
            diff6_field(
                field, 0, self.dx, SUPERGAUSSIAN_BC, SUPERGAUSSIAN_BC
            ),
            expected_sixth,
            rtol=1.0e-13,
            atol=1.0e-10,
        )

    def test_first_derivative_lopsided_coefficients_and_array_axes(self):
        field_1d = self.x**4
        exact_1d = 4.0 * self.x**3

        left_0 = (
            -25.0 / 12.0 * field_1d[0]
            + 4.0 * field_1d[1]
            - 3.0 * field_1d[2]
            + 4.0 / 3.0 * field_1d[3]
            - 1.0 / 4.0 * field_1d[4]
        ) / self.dx
        left_1 = (
            -1.0 / 4.0 * field_1d[0]
            - 5.0 / 6.0 * field_1d[1]
            + 3.0 / 2.0 * field_1d[2]
            - 1.0 / 2.0 * field_1d[3]
            + 1.0 / 12.0 * field_1d[4]
        ) / self.dx
        right_1 = (
            -1.0 / 12.0 * field_1d[-5]
            + 1.0 / 2.0 * field_1d[-4]
            - 3.0 / 2.0 * field_1d[-3]
            + 5.0 / 6.0 * field_1d[-2]
            + 1.0 / 4.0 * field_1d[-1]
        ) / self.dx
        right_0 = (
            1.0 / 4.0 * field_1d[-5]
            - 4.0 / 3.0 * field_1d[-4]
            + 3.0 * field_1d[-3]
            - 4.0 * field_1d[-2]
            + 25.0 / 12.0 * field_1d[-1]
        ) / self.dx

        derivative = diff1_field(
            field_1d, 0, self.dx, SOMMERFELD_BC, SOMMERFELD_BC
        )
        np.testing.assert_allclose(
            derivative[jnp.asarray([0, 1, self.n - 2, self.n - 1])],
            jnp.asarray([left_0, left_1, right_1, right_0]),
            rtol=0.0,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(derivative, exact_1d, rtol=0.0, atol=2.0e-13)

        for direction in (0, 1, 2, 3, 4):
            reshape = (1,) * direction + (self.n,) + (1,) * (4 - direction)
            field = jnp.broadcast_to(field_1d.reshape(reshape), (2,) * direction + (self.n,) + (2,) * (4 - direction))
            exact = jnp.broadcast_to(exact_1d.reshape(reshape), field.shape)
            numerical = diff1_field(
                field, direction, self.dx, SOMMERFELD_BC, SOMMERFELD_BC
            )
            np.testing.assert_allclose(numerical, exact, rtol=0.0, atol=2.0e-13)

    def test_mixed_first_derivative_boundaries_only_replace_selected_side(self):
        field = jnp.exp(self.x)
        periodic = diff1_field(field, 0, self.dx)

        left_only = diff1_field(
            field, 0, self.dx, SOMMERFELD_BC, PERIODIC_BC
        )
        right_only = diff1_field(
            field, 0, self.dx, PERIODIC_BC, SOMMERFELD_BC
        )

        np.testing.assert_array_equal(left_only[-2:], periodic[-2:])
        np.testing.assert_array_equal(right_only[:2], periodic[:2])
        self.assertFalse(np.array_equal(np.asarray(left_only[:2]), np.asarray(periodic[:2])))
        self.assertFalse(np.array_equal(np.asarray(right_only[-2:]), np.asarray(periodic[-2:])))

    def test_first_derivative_boundary_closure_is_fourth_order(self):
        errors = []
        for n in (17, 33, 65):
            dx = 1.0 / (n - 1)
            x = dx * jnp.arange(n, dtype=jnp.float64)
            field = jnp.exp(x)
            derivative = diff1_field(
                field, 0, dx, SOMMERFELD_BC, SOMMERFELD_BC
            )
            indices = jnp.asarray([0, 1, n - 2, n - 1])
            errors.append(float(jnp.max(jnp.abs(derivative[indices] - field[indices]))))

        orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
        self.assertTrue(np.all(orders > 3.8), orders)

    def test_sixth_derivative_lopsided_closures(self):
        field = self.x**7
        exact = 5040.0 * self.x
        derivative = diff6_field(
            field, 0, self.dx, SOMMERFELD_BC, SOMMERFELD_BC
        )

        boundary_indices = jnp.asarray(
            [0, 1, 2, self.n - 3, self.n - 2, self.n - 1]
        )
        np.testing.assert_allclose(
            derivative[boundary_indices],
            exact[boundary_indices],
            rtol=0.0,
            atol=2.0e-9,
        )

    def test_sixth_derivative_boundary_closure_is_second_order(self):
        errors = []
        for n in (17, 33, 65):
            dx = 1.0 / (n - 1)
            x = dx * jnp.arange(n, dtype=jnp.float64)
            field = jnp.exp(x)
            derivative = diff6_field(
                field, 0, dx, SOMMERFELD_BC, SOMMERFELD_BC
            )
            indices = jnp.asarray([0, 1, 2, n - 3, n - 2, n - 1])
            errors.append(float(jnp.max(jnp.abs(derivative[indices] - field[indices]))))

        orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
        self.assertTrue(np.all(orders > 1.8), orders)


if __name__ == "__main__":
    unittest.main()
