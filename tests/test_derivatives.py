"""
Unit tests for finite difference derivatives.

This module tests all derivative operations against known analytical solutions
using various test functions with known derivatives.
"""

import unittest
import jax.numpy as jnp
import numpy as np
from jax import jit
import jax

from JAX_BSSN.derivatives import (
    diff1_field,
    diff6_field,
    compute_all_derivatives,
    gradient_3d,
    divergence_3d,
    laplacian_3d,
    periodic_indexing,
    get_stencil_indices
)


class TestDerivatives(unittest.TestCase):
    """Test suite for finite difference derivatives."""

    def test_first_derivative_1d_trigonometric(self):

        x = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        y = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        z = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        dx = x[1] - x[0]

        X, Y, Z = jnp.meshgrid(x,y,z, indexing='ij')
        # create the meshgrid

        f = jnp.sin(X)
        # create a sin vector field along the x direction
        dfdx_analytical = jnp.cos(X)
        # define the derivative of the vector field analytically
        dfdx_numerical = diff1_field(f, 0, dx)
        # calculate the derivative of the vector field numerically

        dfdx_error = dfdx_numerical - dfdx_analytical
        # calculate the error in the derivative wrt to x

        dfdy_error = diff1_field(f, 1, dx)
        # derivative should be zero

        dfdz_error = diff1_field(f, 2, dx)
        # derivative should be zero

        mean_error_x = jnp.mean(jnp.abs(dfdx_error))
        mean_error_y = jnp.mean(jnp.abs(dfdy_error))
        mean_error_z = jnp.mean(jnp.abs(dfdz_error))
        # calculate the mean error

        self.assertLess(mean_error_x, 1e-6, msg=f"Linear polynomial derivative failed in x direction with mean error {mean_error_x}")
        self.assertLess(mean_error_y, 1e-6, msg=f"Linear polynomial derivative failed in y direction with mean error {mean_error_y}")
        self.assertLess(mean_error_z, 1e-6, msg=f"Linear polynomial derivative failed in z direction with mean error {mean_error_z}")
        # ensure the mean errors in the derivatives are below a certain threshold

        max_error_x = jnp.max(jnp.abs(dfdx_error))
        max_error_y = jnp.max(jnp.abs(dfdy_error))
        max_error_z = jnp.max(jnp.abs(dfdz_error))
        # calculate the max error

        self.assertLess(max_error_x, 5e-6, msg=f'Linear polynomial derivative failed in x direction with max error {max_error_x}')
        self.assertLess(max_error_y, 5e-6, msg=f'Linear polynomial derivative failed in y direction with max error {max_error_y}')
        self.assertLess(max_error_z, 5e-6, msg=f'Linear polynomial derivative failed in z direction with max error {max_error_z}')
        # ensure the max errors in the derivatives are below a certain threshold

    def test_first_derivative_2d_trigonometric(self):

        x = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        y = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        z = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        dx = x[1] - x[0]

        X, Y, Z = jnp.meshgrid(x,y,z, indexing='ij')
        # create the meshgrid

        f = jnp.sin(X) * jnp.cos(Y)
        # create a sin vector field along the x direction
        dfdx_analytical = jnp.cos(X) * jnp.cos(Y)
        # define the derivative of the vector field analytically
        dfdx_numerical = diff1_field(f, 0, dx)
        # calculate the derivative of the vector field numerically
        dfdx_error = dfdx_numerical - dfdx_analytical
        # calculate the error in the derivative wrt x

        dfdy_analytical = -1 * jnp.sin(X) * jnp.sin(Y)
        # define the derivative of the vector field analytically
        dfdy_numerical = diff1_field(f, 1, dx)
        # calculate the derivative of the vector field numerically
        dfdy_error = dfdy_numerical - dfdy_analytical
        # calculate the error in the derivative wrt y

        dfdz_error = diff1_field(f, 2, dx)
        # derivative should be zero

        mean_error_x = jnp.mean(jnp.abs(dfdx_error))
        mean_error_y = jnp.mean(jnp.abs(dfdy_error))
        mean_error_z = jnp.mean(jnp.abs(dfdz_error))
        # calculate the mean error

        self.assertLess(mean_error_x, 1e-6, msg=f"Linear polynomial derivative failed in x direction with mean error {mean_error_x}")
        self.assertLess(mean_error_y, 1e-6, msg=f"Linear polynomial derivative failed in y direction with mean error {mean_error_y}")
        self.assertLess(mean_error_z, 1e-6, msg=f"Linear polynomial derivative failed in z direction with mean error {mean_error_z}")
        # ensure the mean errors in the derivatives are below a certain threshold

        max_error_x = jnp.max(jnp.abs(dfdx_error))
        max_error_y = jnp.max(jnp.abs(dfdy_error))
        max_error_z = jnp.max(jnp.abs(dfdz_error))
        # calculate the max error

        self.assertLess(max_error_x, 5e-6, msg=f'Linear polynomial derivative failed in x direction with max error {max_error_x}')
        self.assertLess(max_error_y, 5e-6, msg=f'Linear polynomial derivative failed in y direction with max error {max_error_y}')
        self.assertLess(max_error_z, 5e-6, msg=f'Linear polynomial derivative failed in z direction with max error {max_error_z}')
        # ensure the max errors in the derivatives are below a certain threshold

    def test_first_derivative_3d_trigonometric(self):

        x = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        y = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        z = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        dx = x[1] - x[0]

        X, Y, Z = jnp.meshgrid(x,y,z, indexing='ij')
        # create the meshgrid

        f = jnp.sin(X) * jnp.cos(Y) * jnp.sin(Z)
        # create a sin vector field along the x direction
        dfdx_analytical = jnp.cos(X) * jnp.cos(Y) * jnp.sin(Z)
        # define the derivative of the vector field analytically
        dfdx_numerical = diff1_field(f, 0, dx)
        # calculate the derivative of the vector field numerically
        dfdx_error = dfdx_numerical - dfdx_analytical
        # calculate the error in the derivative wrt x

        dfdy_analytical = -1 * jnp.sin(X) * jnp.sin(Y) * jnp.sin(Z)
        # define the derivative of the vector field analytically
        dfdy_numerical = diff1_field(f, 1, dx)
        # calculate the derivative of the vector field numerically
        dfdy_error = dfdy_numerical - dfdy_analytical
        # calculate the error in the derivative wrt y


        dfdz_analytical = jnp.sin(X) * jnp.cos(Y) * jnp.cos(Z)
        # define the derivative of the vector field analytically
        dfdz_numerical  = diff1_field(f, 2, dx)
        # calculate the derivative of the vector field numerically
        dfdz_error = dfdz_numerical - dfdz_analytical
        # derivative should be zero

        mean_error_x = jnp.mean(jnp.abs(dfdx_error))
        mean_error_y = jnp.mean(jnp.abs(dfdy_error))
        mean_error_z = jnp.mean(jnp.abs(dfdz_error))
        # calculate the mean error

        self.assertLess(mean_error_x, 1e-6, msg=f"Linear polynomial derivative failed in x direction with mean error {mean_error_x}")
        self.assertLess(mean_error_y, 1e-6, msg=f"Linear polynomial derivative failed in y direction with mean error {mean_error_y}")
        self.assertLess(mean_error_z, 1e-6, msg=f"Linear polynomial derivative failed in z direction with mean error {mean_error_z}")
        # ensure the mean errors in the derivatives are below a certain threshold

        max_error_x = jnp.max(jnp.abs(dfdx_error))
        max_error_y = jnp.max(jnp.abs(dfdy_error))
        max_error_z = jnp.max(jnp.abs(dfdz_error))
        # calculate the max error

        self.assertLess(max_error_x, 5e-6, msg=f'Linear polynomial derivative failed in x direction with max error {max_error_x}')
        self.assertLess(max_error_y, 5e-6, msg=f'Linear polynomial derivative failed in y direction with max error {max_error_y}')
        self.assertLess(max_error_z, 5e-6, msg=f'Linear polynomial derivative failed in z direction with max error {max_error_z}')
        # ensure the max errors in the derivatives are below a certain threshold


    def test_first_derivative_1d_quadratic(self):

        x = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        y = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        z = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        dx = x[1] - x[0]

        X, Y, Z = jnp.meshgrid(x,y,z, indexing='ij')
        # create the meshgrid

        f = X**2
        # create a sin vector field along the x direction
        dfdx_analytical = 2 * X
        # define the derivative of the vector field analytically
        dfdx_numerical = diff1_field(f, 0, dx)
        # calculate the derivative of the vector field numerically

        dfdx_error = dfdx_numerical - dfdx_analytical
        # calculate the error in the derivative wrt to x

        dfdy_error = diff1_field(f, 1, dx)
        # derivative should be zero

        dfdz_error = diff1_field(f, 2, dx)
        # derivative should be zero


        _slice = slice(3, -3) # ignore the first 3 points of the grid because quadratic is not periodic

        mean_error_x = jnp.mean(jnp.abs(dfdx_error[_slice, _slice, _slice]))
        mean_error_y = jnp.mean(jnp.abs(dfdy_error[_slice, _slice, _slice]))
        mean_error_z = jnp.mean(jnp.abs(dfdz_error[_slice, _slice, _slice]))
        # calculate the mean error


        self.assertLess(mean_error_x, 5e-6, msg=f"Linear polynomial derivative failed in x direction with mean error {mean_error_x}")
        self.assertLess(mean_error_y, 5e-6, msg=f"Linear polynomial derivative failed in y direction with mean error {mean_error_y}")
        self.assertLess(mean_error_z, 5e-6, msg=f"Linear polynomial derivative failed in z direction with mean error {mean_error_z}")
        # ensure the mean errors in the derivatives are below a certain threshold

        max_error_x = jnp.max(jnp.abs(dfdx_error[_slice, _slice, _slice]))
        max_error_y = jnp.max(jnp.abs(dfdy_error[_slice, _slice, _slice]))
        max_error_z = jnp.max(jnp.abs(dfdz_error[_slice, _slice, _slice]))
        # calculate the max error

        self.assertLess(max_error_x, 5e-5, msg=f'Linear polynomial derivative failed in x direction with max error {max_error_x}')
        self.assertLess(max_error_y, 5e-5, msg=f'Linear polynomial derivative failed in y direction with max error {max_error_y}')
        self.assertLess(max_error_z, 5e-5, msg=f'Linear polynomial derivative failed in z direction with max error {max_error_z}')
        # ensure the max errors in the derivatives are below a certain threshold


    def test_first_derivative_2d_quadratic(self):

        x = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        y = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        z = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        dx = x[1] - x[0]

        X, Y, Z = jnp.meshgrid(x,y,z, indexing='ij')
        # create the meshgrid

        f = X**2 + Y**2
        # create a polynomial

        dfdx_analytical = 2 * X
        # define the derivative of the vector field analytically
        dfdx_numerical = diff1_field(f, 0, dx)
        # calculate the derivative of the vector field numerically
        dfdx_error = dfdx_numerical - dfdx_analytical
        # calculate the error in the derivative wrt to x

        dfdy_analytical = 2 * Y
        # define the derivative of the vector field analytically
        dfdy_numerical  = diff1_field(f, 1, dx)
        # calculate the derivative of the vector field numerically
        dfdy_error = dfdy_numerical - dfdy_analytical
        # calculate the error in the derivative wrt to y

        dfdz_error = diff1_field(f, 2, dx)
        # derivative should be zero


        _slice = slice(3, -3) # ignore the first 3 points of the grid because quadratic is not periodic

        mean_error_x = jnp.mean(jnp.abs(dfdx_error[_slice, _slice, _slice]))
        mean_error_y = jnp.mean(jnp.abs(dfdy_error[_slice, _slice, _slice]))
        mean_error_z = jnp.mean(jnp.abs(dfdz_error[_slice, _slice, _slice]))
        # calculate the mean error


        self.assertLess(mean_error_x, 5e-6, msg=f"Linear polynomial derivative failed in x direction with mean error {mean_error_x}")
        self.assertLess(mean_error_y, 5e-6, msg=f"Linear polynomial derivative failed in y direction with mean error {mean_error_y}")
        self.assertLess(mean_error_z, 5e-6, msg=f"Linear polynomial derivative failed in z direction with mean error {mean_error_z}")
        # ensure the mean errors in the derivatives are below a certain threshold

        max_error_x = jnp.max(jnp.abs(dfdx_error[_slice, _slice, _slice]))
        max_error_y = jnp.max(jnp.abs(dfdy_error[_slice, _slice, _slice]))
        max_error_z = jnp.max(jnp.abs(dfdz_error[_slice, _slice, _slice]))
        # calculate the max error

        self.assertLess(max_error_x, 5e-5, msg=f'Linear polynomial derivative failed in x direction with max error {max_error_x}')
        self.assertLess(max_error_y, 5e-5, msg=f'Linear polynomial derivative failed in y direction with max error {max_error_y}')
        self.assertLess(max_error_z, 5e-5, msg=f'Linear polynomial derivative failed in z direction with max error {max_error_z}')
        # ensure the max errors in the derivatives are below a certain threshold


    def test_first_derivative_3d_quadratic(self):

        x = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        y = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        z = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        dx = x[1] - x[0]

        X, Y, Z = jnp.meshgrid(x,y,z, indexing='ij')
        # create the meshgrid

        f = X**2 + Y**2 + Z**2
        # create a polynomial

        dfdx_analytical = 2 * X
        # define the derivative of the vector field analytically
        dfdx_numerical = diff1_field(f, 0, dx)
        # calculate the derivative of the vector field numerically
        dfdx_error = dfdx_numerical - dfdx_analytical
        # calculate the error in the derivative wrt to x

        dfdy_analytical = 2 * Y
        # define the derivative of the vector field analytically
        dfdy_numerical = diff1_field(f, 1, dx)
        # calculate the derivative of the vector field numerically
        dfdy_error = dfdy_numerical - dfdy_analytical
        # calculate the error in the derivative wrt to y

        dfdz_analytical = 2 * Z
        # define the derivative of the vector field analytically
        dfdz_numerical = diff1_field(f, 2, dx)
        # calculate the derivative of the vector field numerically
        dfdz_error = dfdz_numerical - dfdz_analytical
        # calculate the error in the derivative wrt to z


        _slice = slice(3, -3) # ignore the first 3 points of the grid because quadratic is not periodic

        mean_error_x = jnp.mean(jnp.abs(dfdx_error[_slice, _slice, _slice]))
        mean_error_y = jnp.mean(jnp.abs(dfdy_error[_slice, _slice, _slice]))
        mean_error_z = jnp.mean(jnp.abs(dfdz_error[_slice, _slice, _slice]))
        # calculate the mean error


        self.assertLess(mean_error_x, 5e-6, msg=f"Linear polynomial derivative failed in x direction with mean error {mean_error_x}")
        self.assertLess(mean_error_y, 5e-6, msg=f"Linear polynomial derivative failed in y direction with mean error {mean_error_y}")
        self.assertLess(mean_error_z, 5e-6, msg=f"Linear polynomial derivative failed in z direction with mean error {mean_error_z}")
        # ensure the mean errors in the derivatives are below a certain threshold

        max_error_x = jnp.max(jnp.abs(dfdx_error[_slice, _slice, _slice]))
        max_error_y = jnp.max(jnp.abs(dfdy_error[_slice, _slice, _slice]))
        max_error_z = jnp.max(jnp.abs(dfdz_error[_slice, _slice, _slice]))
        # calculate the max error

        self.assertLess(max_error_x, 5e-5, msg=f'Linear polynomial derivative failed in x direction with max error {max_error_x}')
        self.assertLess(max_error_y, 5e-5, msg=f'Linear polynomial derivative failed in y direction with max error {max_error_y}')
        self.assertLess(max_error_z, 5e-5, msg=f'Linear polynomial derivative failed in z direction with max error {max_error_z}')
        # ensure the max errors in the derivatives are below a certain threshold


    def test_laplacian_3d(self):


        x = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        y = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        z = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
        dx = x[1] - x[0]

        X, Y, Z = jnp.meshgrid(x,y,z, indexing='ij')
        # create the meshgrid

        f = jnp.sin(X) + jnp.sin(Y) + jnp.sin(Z)
        # create a trignometric field

        lapl_analytical = -f
        lapl_numerical = laplacian_3d(f, dx)
        # compute the laplacian numerically
        lapl_error = lapl_numerical - lapl_analytical
        # calculate the error in the laplacian

        mean_error = jnp.mean(jnp.abs(lapl_error))

        self.assertLess(mean_error, 5e-5, msg=f"Linear polynomial derivative failed in x direction with mean error {mean_error}")
        # ensure the mean errors in the derivatives are below a certain threshold

        max_error = jnp.max(jnp.abs(lapl_error))
        # calculate the max error

        self.assertLess(max_error, 5e-4, msg=f'Linear polynomial derivative failed with max error {max_error}')
        # ensure the max errors in the derivatives are below a certain threshold



    # def test_sixth_derivative_1d_polynomial(self):

    #     x = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
    #     y = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
    #     z = jnp.linspace(-jnp.pi, jnp.pi, 100, endpoint=False)
    #     dx = x[1] - x[0]

    #     X, Y, Z = jnp.meshgrid(x,y,z, indexing='ij')
    #     # create the meshgrid

    #     f = jnp.sin(X)
    #     # create a sin vector field along the x direction
    #     dfdx6_analytical = -1 * jnp.sin(X)
    #     # define the derivative of the vector field analytically
    #     dfdx6_numerical = diff6_field(f, 0, dx)
    #     # calculate the derivative of the vector field numerically

    #     dfdx6_error = dfdx6_numerical - dfdx6_analytical
    #     # calculate the error in the derivative wrt to x

    #     dfdy6_error = diff6_field(f, 1, dx)
    #     # derivative should be zero

    #     dfdz6_error = diff6_field(f, 2, dx)
    #     # derivative should be zero

    #     mean_error_x = jnp.mean(jnp.abs(dfdx6_error))
    #     mean_error_y = jnp.mean(jnp.abs(dfdy6_error))
    #     mean_error_z = jnp.mean(jnp.abs(dfdz6_error))
    #     # calculate the mean error

    #     self.assertLess(mean_error_x, 1e-6, msg=f"Linear polynomial derivative failed in x direction with mean error {mean_error_x}")
    #     self.assertLess(mean_error_y, 1e-6, msg=f"Linear polynomial derivative failed in y direction with mean error {mean_error_y}")
    #     self.assertLess(mean_error_z, 1e-6, msg=f"Linear polynomial derivative failed in z direction with mean error {mean_error_z}")
    #     # ensure the mean errors in the derivatives are below a certain threshold

    #     max_error_x = jnp.max(jnp.abs(dfdx6_error))
    #     max_error_y = jnp.max(jnp.abs(dfdy6_error))
    #     max_error_z = jnp.max(jnp.abs(dfdz6_error))
    #     # calculate the max error

    #     self.assertLess(max_error_x, 5e-6, msg=f'Linear polynomial derivative failed in x direction with max error {max_error_x}')
    #     self.assertLess(max_error_y, 5e-6, msg=f'Linear polynomial derivative failed in y direction with max error {max_error_y}')
    #     self.assertLess(max_error_z, 5e-6, msg=f'Linear polynomial derivative failed in z direction with max error {max_error_z}')
    #     # ensure the max errors in the derivatives are below a certain threshold






    
    # def test_sixth_derivative_polynomial(self):
    #     """Test sixth derivative on polynomial (should be zero for degree < 6)."""
    #     # For polynomials of degree < 6, sixth derivative should be zero
    #     # Use a simpler quadratic for more stable results
    #     field = self.create_polynomial_field(degree=2)
        
    #     for direction in range(3):
    #         numerical = diff6_field(field, direction, self.dx)
    #         expected = jnp.zeros_like(field)
            
    #         # Test interior points and use looser tolerance due to boundary effects
    #         interior = slice(6, -6)  # Need larger margin for 6th derivatives
    #         error = numerical[interior, interior, interior] - expected[interior, interior, interior]
    #         max_error = jnp.max(jnp.abs(error))

    #         print(numerical)
    #         print(expected)
    #         self.assertLess(max_error, self.tol_6th,
    #             msg=f"Sixth derivative of degree-2 polynomial failed in direction {direction} with max error {max_error}"
    #         )
    
    # def test_sixth_derivative_trigonometric(self):
    #     """Test sixth derivative on trigonometric function."""
    #     field = self.create_trig_field()
        
    #     # For sin(kx), sixth derivative is -k^6 * sin(kx)
    #     # Test only interior points and use very loose tolerance
    #     interior = slice(6, -6)
        
    #     for direction in range(3):
    #         numerical = diff6_field(field, direction, self.dx)
            
    #         if direction == 0:
    #             k = self.kx
    #         elif direction == 1:
    #             k = self.ky
    #         else:
    #             k = self.kz
            
    #         analytical = -(k**6) * field

    #         # Plot numerical vs analytical side-by-side for a representative 2D slice
    #         import matplotlib.pyplot as plt

    #         mid = self.n // 2

    #         if direction == 0:
    #             num_slice = numerical[mid, :, :]
    #             anal_slice = analytical[mid, :, :]
    #             title_axes = ('y', 'z')
    #         elif direction == 1:
    #             num_slice = numerical[:, mid, :]
    #             anal_slice = analytical[:, mid, :]
    #             title_axes = ('x', 'z')
    #         else:
    #             num_slice = numerical[:, :, mid]
    #             anal_slice = analytical[:, :, mid]
    #             title_axes = ('x', 'y')

    #         num_np = np.asarray(num_slice)
    #         anal_np = np.asarray(anal_slice)
    #         diff_np = num_np - anal_np

    #         vmax = max(np.abs(num_np).max(), np.abs(anal_np).max(), 1e-12)

    #         fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #         im0 = axes[0].imshow(num_np.T, origin='lower', cmap='viridis', vmin=-vmax, vmax=vmax)
    #         axes[0].set_title(f'Numerical 6th derivative (dir={direction})')
    #         axes[0].set_xlabel(title_axes[0]); axes[0].set_ylabel(title_axes[1])
    #         plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    #         im1 = axes[1].imshow(anal_np.T, origin='lower', cmap='viridis', vmin=-vmax, vmax=vmax)
    #         axes[1].set_title('Analytical 6th derivative')
    #         axes[1].set_xlabel(title_axes[0]); axes[1].set_ylabel(title_axes[1])
    #         plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    #         im2 = axes[2].imshow(diff_np.T, origin='lower', cmap='bwr')
    #         axes[2].set_title('Difference (numerical - analytical)')
    #         axes[2].set_xlabel(title_axes[0]); axes[2].set_ylabel(title_axes[1])
    #         plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    #         plt.suptitle(f'6th derivative comparison (direction={direction})')
    #         plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    #         # Save a copy for later inspection and also try to show interactively
    #         plt.savefig(f'sixth_derivative_compare_dir{direction}.png', dpi=150)
    #         plt.show()
            
    #         # Use very loose tolerance for sixth derivatives due to numerical limitations
    #         error = numerical[interior, interior, interior] - analytical[interior, interior, interior]
    #         max_error = jnp.max(jnp.abs(error))
    #         self.assertLess(max_error, self.tol_6th,
    #             msg=f"Sixth derivative of trigonometric function failed in direction {direction} with max error {max_error}"
    #         )
    #         # np.testing.assert_allclose(
    #         #     numerical[interior, interior, interior], 
    #         #     analytical[interior, interior, interior], 
    #         #     atol=1e0, rtol=0.5,  # Very loose tolerance
    #         #     err_msg=f"Sixth derivative of trigonometric function failed in direction {direction}"
    #         # )
    
    # def test_compute_all_derivatives(self):
    #     """Test computing all first derivatives at once."""
    #     field = self.create_polynomial_field(degree=2)
        
    #     # Compute all derivatives
    #     all_derivs = compute_all_derivatives(field, self.dx)
        
    #     # Compare with individual computations
    #     for direction in range(3):
    #         individual = diff1_field(field, direction, self.dx)
    #         from_all = all_derivs[direction]

    #         error = individual - from_all
    #         max_error = jnp.max(jnp.abs(error))
    #         self.assertLess(max_error, self.tol_1st,
    #             msg=f"compute_all_derivatives doesn't match individual computation in direction {direction} with max error {max_error}"
    #         )
            
    
    # def test_gradient_3d(self):
    #     """Test 3D gradient computation."""
    #     field = self.create_exponential_field()
        
    #     # Compute gradient
    #     grad = gradient_3d(field, self.dx)
        
    #     # Compare with analytical gradient
    #     interior = slice(3, -3)
    #     for direction in range(3):
    #         analytical = self.analytical_derivative_exponential(direction)

    #         error = grad[direction][interior, interior, interior] - analytical[interior, interior, interior]
    #         max_error = jnp.max(jnp.abs(error))
    #         self.assertLess(max_error, self.tol_1st,
    #             msg=f"Gradient computation failed in direction {direction} with max error {max_error}"
    #         )
    
    # def test_divergence_3d_constant_field(self):
    #     """Test divergence of constant vector field."""
    #     # Constant vector field should have zero divergence
    #     vector_field = jnp.zeros((3, self.n, self.n, self.n))
    #     vector_field = vector_field.at[0].set(2.0)
    #     vector_field = vector_field.at[1].set(3.0)
    #     vector_field = vector_field.at[2].set(4.0)
        
    #     div = divergence_3d(vector_field, self.dx)
    #     expected = jnp.zeros_like(div)

    #     error = div - expected
    #     max_error = jnp.max(jnp.abs(error))
    #     self.assertLess(max_error, self.tol_1st,
    #         msg=f"Divergence of constant field failed with max error {max_error}"
    #     )
    
    # def test_divergence_3d_linear_field(self):
    #     """Test divergence of linear vector field."""
    #     # Vector field v = (x, y, z) should have divergence = 3
    #     vector_field = jnp.zeros((3, self.n, self.n, self.n))
    #     vector_field = vector_field.at[0].set(self.X)
    #     vector_field = vector_field.at[1].set(self.Y)
    #     vector_field = vector_field.at[2].set(self.Z)
        
    #     div = divergence_3d(vector_field, self.dx)
    #     expected = 3.0 * jnp.ones_like(div)
        
    #     # Test interior points to avoid boundary effects
    #     interior = slice(3, -3)
    #     error = div[interior, interior, interior] - expected[interior, interior, interior]
    #     max_error = jnp.max(jnp.abs(error))
    #     self.assertLess(max_error, self.tol_1st,
    #         msg=f"Divergence of linear field failed with max error {max_error}"
    #     )
    
    
    # def test_divergence_3d_quadratic_field(self):
    #     """Test divergence of quadratic vector field."""
    #     # Vector field v = (x^2, y^2, z^2) should have divergence = 2x + 2y + 2z
    #     vector_field = jnp.zeros((3, self.n, self.n, self.n))
    #     vector_field = vector_field.at[0].set(self.X**2)
    #     vector_field = vector_field.at[1].set(self.Y**2)
    #     vector_field = vector_field.at[2].set(self.Z**2)
        
    #     div = divergence_3d(vector_field, self.dx)
    #     expected = 2.0 * (self.X + self.Y + self.Z)
        
    #     # Test interior points to avoid boundary effects
    #     interior = slice(3, -3)
    #     error = div[interior, interior, interior] - expected[interior, interior, interior]
    #     max_error = jnp.max(jnp.abs(error))
    #     self.assertLess(max_error, self.tol_1st,
    #         msg=f"Divergence of quadratic field failed with max error {max_error}"
    #     )
    
    # def test_laplacian_3d_linear(self):
    #     """Test Laplacian of linear function."""
    #     # Linear function should have zero Laplacian
    #     field = self.create_polynomial_field(degree=1)
        
    #     lapl = laplacian_3d(field, self.dx)
    #     expected = jnp.zeros_like(field)
        
    #     # Test interior points to avoid boundary effects
    #     interior = slice(4, -4)  # Need larger margin for second derivatives
    #     error = lapl[interior, interior, interior] - expected[interior, interior, interior]
    #     average_error = jnp.mean(jnp.abs(error))
    #     self.assertLess(average_error, self.tol_1st,
    #         msg=f"Laplacian of linear function failed with average error {average_error}"
    #     )
    
    # def test_laplacian_3d_quadratic(self):
    #     """Test Laplacian of quadratic function."""
    #     # f = x^2 + 2y^2 + 3z^2 + xy
    #     # ∇²f = 2 + 4 + 6 = 12
    #     field = self.create_polynomial_field(degree=2)
        
    #     lapl = laplacian_3d(field, self.dx)
    #     expected = 12.0 * jnp.ones_like(field)
        
    #     # Test interior points to avoid boundary effects
    #     interior = slice(4, -4)  # Need larger margin for second derivatives
    #     error = lapl[interior, interior, interior] - expected[interior, interior, interior]
    #     average_error = jnp.mean(jnp.abs(error))
    #     self.assertLess(average_error, self.tol_1st,
    #         msg=f"Laplacian of quadratic function failed with average error {average_error}"
    #     )
    
    # def test_laplacian_3d_exponential(self):
    #     """Test Laplacian of exponential function."""
    #     # f = exp(-0.5(x² + y² + z²))
    #     # ∇²f = exp(-0.5(x² + y² + z²)) * ((x² + y² + z²) - 3)
    #     field = self.create_exponential_field()
    #     r_squared = self.X**2 + self.Y**2 + self.Z**2
        
    #     lapl = laplacian_3d(field, self.dx)
    #     expected = field * (r_squared - 3.0)
        
    #     # Test interior points to avoid boundary effects
    #     interior = slice(4, -4)  # Need larger margin for second derivatives
    #     error = lapl[interior, interior, interior] - expected[interior, interior, interior]
    #     average_error = jnp.mean(jnp.abs(error))
    #     self.assertLess(average_error, self.tol_1st,
    #         msg=f"Laplacian of exponential function failed with average error {average_error}"
    #     )
    
    # def test_consistency_gradient_divergence(self):
    #     """Test consistency between gradient and divergence operations."""
    #     # ∇ · ∇f = ∇²f (Laplacian)
    #     field = self.create_exponential_field()
        
    #     # Method 1: Direct Laplacian
    #     lapl_direct = laplacian_3d(field, self.dx)
        
    #     # Method 2: Divergence of gradient
    #     grad = gradient_3d(field, self.dx)
    #     lapl_indirect = divergence_3d(grad, self.dx)

    #     error = lapl_direct - lapl_indirect
    #     max_error = jnp.max(jnp.abs(error))
    #     self.assertLess(max_error, self.tol_1st,
    #         msg=f"Laplacian consistency failed with max error {max_error}"
    #     )
    
    # def test_derivative_chain_rule(self):
    #     """Test derivatives satisfy chain rule properties."""
    #     # For f(x,y,z) = sin(x) * cos(y) * sin(z)
    #     field = self.create_trig_field()
        
    #     # Test that mixed derivatives are symmetric (Schwarz theorem)
    #     # ∂²f/∂x∂y = ∂²f/∂y∂x
        
    #     # Compute ∂f/∂x, then ∂/∂y
    #     df_dx = diff1_field(field, 0, self.dx)
    #     d2f_dxdy = diff1_field(df_dx, 1, self.dx)
        
    #     # Compute ∂f/∂y, then ∂/∂x
    #     df_dy = diff1_field(field, 1, self.dx)
    #     d2f_dydx = diff1_field(df_dy, 0, self.dx)

    #     error = d2f_dxdy - d2f_dydx
    #     max_error = jnp.max(jnp.abs(error))
    #     self.assertLess(max_error, self.tol_1st,
    #         msg=f"Mixed derivatives symmetry failed with max error {max_error}"
    #     )
    
    # def test_derivative_accuracy_order(self):
    #     """Test that derivative accuracy improves with grid refinement."""
    #     # Use a simpler test that avoids boundary condition issues
    #     # Test with a smooth function in the interior
        
    #     # Create coarse and fine grids
    #     n_coarse = 16
    #     n_fine = 32
        
    #     x_coarse = jnp.linspace(-0.5, 0.5, n_coarse)
    #     x_fine = jnp.linspace(-0.5, 0.5, n_fine)
        
    #     dx_coarse = x_coarse[1] - x_coarse[0] 
    #     dx_fine = x_fine[1] - x_fine[0]
        
    #     # Use exponential function (smooth and well-behaved)
    #     field_coarse = jnp.exp(-x_coarse**2)
    #     field_fine = jnp.exp(-x_fine**2)
        
    #     analytical_coarse = -2 * x_coarse * jnp.exp(-x_coarse**2)
    #     analytical_fine = -2 * x_fine * jnp.exp(-x_fine**2)
        
    #     # Convert to 3D arrays
    #     field_3d_coarse = jnp.tile(field_coarse[None, None, :], (1, 1, 1))
    #     field_3d_fine = jnp.tile(field_fine[None, None, :], (1, 1, 1))
        
    #     numerical_coarse = diff1_field(field_3d_coarse, 2, dx_coarse)[0, 0, :]
    #     numerical_fine = diff1_field(field_3d_fine, 2, dx_fine)[0, 0, :]
        
    #     # Check error only in interior points
    #     interior_coarse = slice(3, -3)
    #     interior_fine = slice(6, -6)  # Scale the interior appropriately
        
    #     error_coarse = jnp.max(jnp.abs(numerical_coarse[interior_coarse] - analytical_coarse[interior_coarse]))
    #     error_fine = jnp.max(jnp.abs(numerical_fine[interior_fine] - analytical_fine[interior_fine]))
        
    #     # Just check that the fine grid is more accurate
    #     self.assertLess(error_fine, error_coarse, 
    #                    "Fine grid should be more accurate than coarse grid")
    
    # def test_vector_field_operations(self):
    #     """Test vector field operations on realistic field."""
    #     # Create a divergence-free vector field (should have zero divergence)
    #     # v = ∇ × A where A is a vector potential
    #     # For simplicity, use A = (0, 0, xy) so v = (y, -x, 0)
        
    #     vector_field = jnp.zeros((3, self.n, self.n, self.n))
    #     vector_field = vector_field.at[0].set(self.Y)   # v_x = y
    #     vector_field = vector_field.at[1].set(-self.X)  # v_y = -x
    #     vector_field = vector_field.at[2].set(0.0)      # v_z = 0
        
    #     # This field should have zero divergence
    #     div = divergence_3d(vector_field, self.dx)
    #     expected_div = jnp.zeros_like(div)
        
    #     np.testing.assert_allclose(
    #         div, expected_div, atol=self.tol_1st,
    #         err_msg="Divergence-free vector field should have zero divergence"
    #     )


if __name__ == '__main__':
    # Configure JAX for testing
    jax.config.update("jax_enable_x64", True)  # Use double precision
    
    unittest.main(verbosity=2)
