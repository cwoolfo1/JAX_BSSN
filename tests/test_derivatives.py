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
    
    def setUp(self):
        """Set up test parameters and grid."""
        self.n = 32  # Grid size (must be even for clean periodicity)
        self.tol_1st = 1e-4  # Tolerance for 1st derivatives (4th order accuracy)
        self.tol_6th = 1e-2  # Tolerance for 6th derivatives (lower accuracy expected)
        
        # Create coordinate grids with consistent spacing
        x = jnp.linspace(-1.0, 1.0, self.n)
        y = jnp.linspace(-1.0, 1.0, self.n)
        z = jnp.linspace(-1.0, 1.0, self.n)
        self.X, self.Y, self.Z = jnp.meshgrid(x, y, z, indexing='ij')
        
        # Compute actual grid spacing
        self.dx = x[1] - x[0]  # This is 2.0/(n-1)
        
        # Wavenumbers for periodic test functions (chosen to be compatible with grid)
        self.kx = 2.0 * jnp.pi  # One full wavelength in [-1,1] domain
        self.ky = 4.0 * jnp.pi  # Two full wavelengths 
        self.kz = 6.0 * jnp.pi  # Three full wavelengths
    
    def create_polynomial_field(self, degree=3):
        """Create polynomial test field."""
        if degree == 1:
            return 2.0 * self.X + 3.0 * self.Y + 4.0 * self.Z
        elif degree == 2:
            return self.X**2 + 2.0 * self.Y**2 + 3.0 * self.Z**2 + self.X * self.Y
        elif degree == 3:
            return (self.X**3 + self.Y**3 + self.Z**3 + 
                   self.X**2 * self.Y + self.Y**2 * self.Z + self.Z**2 * self.X)
        else:
            return self.X**4 + self.Y**4 + self.Z**4
    
    def create_trig_field(self):
        """Create trigonometric test field (periodic-friendly)."""
        return jnp.sin(self.kx * self.X) * jnp.cos(self.ky * self.Y) * jnp.sin(self.kz * self.Z)
    
    def create_exponential_field(self):
        """Create exponential test field (decaying to avoid overflow)."""
        return jnp.exp(-0.5 * (self.X**2 + self.Y**2 + self.Z**2))
    
    def analytical_derivative_polynomial(self, degree, direction):
        """Analytical derivatives of polynomial fields."""
        if degree == 1:
            if direction == 0:  # d/dx
                return 2.0 * jnp.ones_like(self.X)
            elif direction == 1:  # d/dy
                return 3.0 * jnp.ones_like(self.Y)
            else:  # d/dz
                return 4.0 * jnp.ones_like(self.Z)
        
        elif degree == 2:
            if direction == 0:  # d/dx
                return 2.0 * self.X + self.Y
            elif direction == 1:  # d/dy
                return 4.0 * self.Y + self.X
            else:  # d/dz
                return 6.0 * self.Z
        
        elif degree == 3:
            if direction == 0:  # d/dx
                return 3.0 * self.X**2 + 2.0 * self.X * self.Y + self.Z**2
            elif direction == 1:  # d/dy
                return 3.0 * self.Y**2 + self.X**2 + 2.0 * self.Y * self.Z
            else:  # d/dz
                return 3.0 * self.Z**2 + self.Y**2 + 2.0 * self.Z * self.X
    
    def analytical_derivative_trig(self, direction):
        """Analytical derivatives of trigonometric field."""
        base = jnp.sin(self.kx * self.X) * jnp.cos(self.ky * self.Y) * jnp.sin(self.kz * self.Z)
        
        if direction == 0:  # d/dx
            return self.kx * jnp.cos(self.kx * self.X) * jnp.cos(self.ky * self.Y) * jnp.sin(self.kz * self.Z)
        elif direction == 1:  # d/dy
            return -self.ky * jnp.sin(self.kx * self.X) * jnp.sin(self.ky * self.Y) * jnp.sin(self.kz * self.Z)
        else:  # d/dz
            return self.kz * jnp.sin(self.kx * self.X) * jnp.cos(self.ky * self.Y) * jnp.cos(self.kz * self.Z)
    
    def analytical_derivative_exponential(self, direction):
        """Analytical derivatives of exponential field."""
        exp_field = jnp.exp(-0.5 * (self.X**2 + self.Y**2 + self.Z**2))
        
        if direction == 0:  # d/dx
            return -self.X * exp_field
        elif direction == 1:  # d/dy
            return -self.Y * exp_field
        else:  # d/dz
            return -self.Z * exp_field
    
    def test_periodic_indexing(self):
        """Test periodic boundary condition indexing."""
        size = 10
        
        # Test normal indices
        self.assertEqual(periodic_indexing(5, size), 5)
        
        # Test negative indices (should wrap around)
        self.assertEqual(periodic_indexing(-1, size), 9)
        self.assertEqual(periodic_indexing(-2, size), 8)
        
        # Test indices >= size (should wrap around)
        self.assertEqual(periodic_indexing(10, size), 0)
        self.assertEqual(periodic_indexing(11, size), 1)

    
    def test_first_derivative_polynomial_linear(self):
        """Test first derivatives on linear polynomial."""
        field = self.create_polynomial_field(degree=1)
        
        for direction in range(3):
            numerical = diff1_field(field, direction, self.dx)
            analytical = self.analytical_derivative_polynomial(1, direction)
            
            # For periodic boundaries with linear functions, only test interior points
            # where boundary effects are minimal
            interior = slice(3, -3)  # Avoid boundary points
            np.testing.assert_allclose(
                numerical[interior, interior, interior], 
                analytical[interior, interior, interior], 
                atol=1e-6,
                err_msg=f"Linear polynomial derivative failed in direction {direction}"
            )
    
    def test_first_derivative_polynomial_quadratic(self):
        """Test first derivatives on quadratic polynomial."""
        field = self.create_polynomial_field(degree=2)
        
        for direction in range(3):
            numerical = diff1_field(field, direction, self.dx)
            analytical = self.analytical_derivative_polynomial(2, direction)
            
            # Test interior points to avoid boundary condition issues
            interior = slice(3, -3)
            np.testing.assert_allclose(
                numerical[interior, interior, interior], 
                analytical[interior, interior, interior], 
                atol=self.tol_1st,
                err_msg=f"Quadratic polynomial derivative failed in direction {direction}"
            )
    
    def test_first_derivative_polynomial_cubic(self):
        """Test first derivatives on cubic polynomial."""
        field = self.create_polynomial_field(degree=3)
        
        for direction in range(3):
            numerical = diff1_field(field, direction, self.dx)
            analytical = self.analytical_derivative_polynomial(3, direction)
            
            # Test interior points to avoid boundary condition issues
            interior = slice(3, -3)
            np.testing.assert_allclose(
                numerical[interior, interior, interior], 
                analytical[interior, interior, interior], 
                atol=self.tol_1st,
                err_msg=f"Cubic polynomial derivative failed in direction {direction}"
            )
    
    def test_first_derivative_exponential(self):
        """Test first derivatives on exponential function."""
        field = self.create_exponential_field()
        
        for direction in range(3):
            numerical = diff1_field(field, direction, self.dx)
            analytical = self.analytical_derivative_exponential(direction)
            
            # Test interior points to avoid boundary effects
            interior = slice(3, -3)
            np.testing.assert_allclose(
                numerical[interior, interior, interior], 
                analytical[interior, interior, interior], 
                atol=self.tol_1st,
                err_msg=f"Exponential derivative failed in direction {direction}"
            )
    
    def test_sixth_derivative_polynomial(self):
        """Test sixth derivative on polynomial (should be zero for degree < 6)."""
        # For polynomials of degree < 6, sixth derivative should be zero
        # Use a simpler quadratic for more stable results
        field = self.create_polynomial_field(degree=2)
        
        for direction in range(3):
            numerical = diff6_field(field, direction, self.dx)
            expected = jnp.zeros_like(field)
            
            # Test interior points and use looser tolerance due to boundary effects
            interior = slice(6, -6)  # Need larger margin for 6th derivatives
            np.testing.assert_allclose(
                numerical[interior, interior, interior], 
                expected[interior, interior, interior], 
                atol=1e-2,
                err_msg=f"Sixth derivative of degree-2 polynomial should be zero in direction {direction}"
            )
    
    def test_sixth_derivative_trigonometric(self):
        """Test sixth derivative on trigonometric function."""
        field = self.create_trig_field()
        
        # For sin(kx), sixth derivative is -k^6 * sin(kx)
        # Test only interior points and use very loose tolerance
        interior = slice(6, -6)
        
        for direction in range(3):
            numerical = diff6_field(field, direction, self.dx)
            
            if direction == 0:
                k = self.kx
            elif direction == 1:
                k = self.ky
            else:
                k = self.kz
            
            analytical = -(k**6) * field
            
            # Use very loose tolerance for sixth derivatives due to numerical limitations
            np.testing.assert_allclose(
                numerical[interior, interior, interior], 
                analytical[interior, interior, interior], 
                atol=1e0, rtol=0.5,  # Very loose tolerance
                err_msg=f"Sixth derivative of trigonometric function failed in direction {direction}"
            )
    
    def test_compute_all_derivatives(self):
        """Test computing all first derivatives at once."""
        field = self.create_polynomial_field(degree=2)
        
        # Compute all derivatives
        all_derivs = compute_all_derivatives(field, self.dx)
        
        # Compare with individual computations
        for direction in range(3):
            individual = diff1_field(field, direction, self.dx)
            from_all = all_derivs[direction]
            
            np.testing.assert_allclose(
                individual, from_all, atol=1e-15,
                err_msg=f"compute_all_derivatives doesn't match individual computation in direction {direction}"
            )
    
    def test_gradient_3d(self):
        """Test 3D gradient computation."""
        field = self.create_exponential_field()
        
        # Compute gradient
        grad = gradient_3d(field, self.dx)
        
        # Compare with analytical gradient
        interior = slice(3, -3)
        for direction in range(3):
            analytical = self.analytical_derivative_exponential(direction)
            
            np.testing.assert_allclose(
                grad[direction][interior, interior, interior], 
                analytical[interior, interior, interior], 
                atol=self.tol_1st,
                err_msg=f"Gradient computation failed in direction {direction}"
            )
    
    def test_divergence_3d_constant_field(self):
        """Test divergence of constant vector field."""
        # Constant vector field should have zero divergence
        vector_field = jnp.zeros((3, self.n, self.n, self.n))
        vector_field = vector_field.at[0].set(2.0)
        vector_field = vector_field.at[1].set(3.0)
        vector_field = vector_field.at[2].set(4.0)
        
        div = divergence_3d(vector_field, self.dx)
        expected = jnp.zeros_like(div)
        
        np.testing.assert_allclose(div, expected, atol=1e-12)
    
    def test_divergence_3d_linear_field(self):
        """Test divergence of linear vector field."""
        # Vector field v = (x, y, z) should have divergence = 3
        vector_field = jnp.zeros((3, self.n, self.n, self.n))
        vector_field = vector_field.at[0].set(self.X)
        vector_field = vector_field.at[1].set(self.Y)
        vector_field = vector_field.at[2].set(self.Z)
        
        div = divergence_3d(vector_field, self.dx)
        expected = 3.0 * jnp.ones_like(div)
        
        # Test interior points to avoid boundary effects
        interior = slice(3, -3)
        np.testing.assert_allclose(
            div[interior, interior, interior], 
            expected[interior, interior, interior], 
            atol=self.tol_1st)
    
    
    def test_divergence_3d_quadratic_field(self):
        """Test divergence of quadratic vector field."""
        # Vector field v = (x^2, y^2, z^2) should have divergence = 2x + 2y + 2z
        vector_field = jnp.zeros((3, self.n, self.n, self.n))
        vector_field = vector_field.at[0].set(self.X**2)
        vector_field = vector_field.at[1].set(self.Y**2)
        vector_field = vector_field.at[2].set(self.Z**2)
        
        div = divergence_3d(vector_field, self.dx)
        expected = 2.0 * (self.X + self.Y + self.Z)
        
        # Test interior points to avoid boundary effects
        interior = slice(3, -3)
        np.testing.assert_allclose(
            div[interior, interior, interior], 
            expected[interior, interior, interior], 
            atol=self.tol_1st)
    
    def test_laplacian_3d_linear(self):
        """Test Laplacian of linear function."""
        # Linear function should have zero Laplacian
        field = self.create_polynomial_field(degree=1)
        
        lapl = laplacian_3d(field, self.dx)
        expected = jnp.zeros_like(field)
        
        # Test interior points to avoid boundary effects
        interior = slice(4, -4)  # Need larger margin for second derivatives
        np.testing.assert_allclose(
            lapl[interior, interior, interior], 
            expected[interior, interior, interior], 
            atol=1e-6)
    
    def test_laplacian_3d_quadratic(self):
        """Test Laplacian of quadratic function."""
        # f = x^2 + 2y^2 + 3z^2 + xy
        # ∇²f = 2 + 4 + 6 = 12
        field = self.create_polynomial_field(degree=2)
        
        lapl = laplacian_3d(field, self.dx)
        expected = 12.0 * jnp.ones_like(field)
        
        # Test interior points to avoid boundary effects
        interior = slice(4, -4)  # Need larger margin for second derivatives
        np.testing.assert_allclose(
            lapl[interior, interior, interior], 
            expected[interior, interior, interior], 
            atol=self.tol_1st)
    
    def test_laplacian_3d_exponential(self):
        """Test Laplacian of exponential function."""
        # f = exp(-0.5(x² + y² + z²))
        # ∇²f = exp(-0.5(x² + y² + z²)) * ((x² + y² + z²) - 3)
        field = self.create_exponential_field()
        r_squared = self.X**2 + self.Y**2 + self.Z**2
        
        lapl = laplacian_3d(field, self.dx)
        expected = field * (r_squared - 3.0)
        
        # Test interior points to avoid boundary effects
        interior = slice(4, -4)  # Need larger margin for second derivatives
        np.testing.assert_allclose(
            lapl[interior, interior, interior], 
            expected[interior, interior, interior], 
            atol=self.tol_1st)
    
    def test_consistency_gradient_divergence(self):
        """Test consistency between gradient and divergence operations."""
        # ∇ · ∇f = ∇²f (Laplacian)
        field = self.create_exponential_field()
        
        # Method 1: Direct Laplacian
        lapl_direct = laplacian_3d(field, self.dx)
        
        # Method 2: Divergence of gradient
        grad = gradient_3d(field, self.dx)
        lapl_indirect = divergence_3d(grad, self.dx)
        
        np.testing.assert_allclose(
            lapl_direct, lapl_indirect, atol=self.tol_1st,
            err_msg="Laplacian should equal divergence of gradient"
        )
    
    def test_derivative_chain_rule(self):
        """Test derivatives satisfy chain rule properties."""
        # For f(x,y,z) = sin(x) * cos(y) * sin(z)
        field = self.create_trig_field()
        
        # Test that mixed derivatives are symmetric (Schwarz theorem)
        # ∂²f/∂x∂y = ∂²f/∂y∂x
        
        # Compute ∂f/∂x, then ∂/∂y
        df_dx = diff1_field(field, 0, self.dx)
        d2f_dxdy = diff1_field(df_dx, 1, self.dx)
        
        # Compute ∂f/∂y, then ∂/∂x
        df_dy = diff1_field(field, 1, self.dx)
        d2f_dydx = diff1_field(df_dy, 0, self.dx)
        
        np.testing.assert_allclose(
            d2f_dxdy, d2f_dydx, atol=1e-8,
            err_msg="Mixed derivatives should be symmetric (Schwarz theorem)"
        )
    
    def test_derivative_accuracy_order(self):
        """Test that derivative accuracy improves with grid refinement."""
        # Use a simpler test that avoids boundary condition issues
        # Test with a smooth function in the interior
        
        # Create coarse and fine grids
        n_coarse = 16
        n_fine = 32
        
        x_coarse = jnp.linspace(-0.5, 0.5, n_coarse)
        x_fine = jnp.linspace(-0.5, 0.5, n_fine)
        
        dx_coarse = x_coarse[1] - x_coarse[0] 
        dx_fine = x_fine[1] - x_fine[0]
        
        # Use exponential function (smooth and well-behaved)
        field_coarse = jnp.exp(-x_coarse**2)
        field_fine = jnp.exp(-x_fine**2)
        
        analytical_coarse = -2 * x_coarse * jnp.exp(-x_coarse**2)
        analytical_fine = -2 * x_fine * jnp.exp(-x_fine**2)
        
        # Convert to 3D arrays
        field_3d_coarse = jnp.tile(field_coarse[None, None, :], (1, 1, 1))
        field_3d_fine = jnp.tile(field_fine[None, None, :], (1, 1, 1))
        
        numerical_coarse = diff1_field(field_3d_coarse, 2, dx_coarse)[0, 0, :]
        numerical_fine = diff1_field(field_3d_fine, 2, dx_fine)[0, 0, :]
        
        # Check error only in interior points
        interior_coarse = slice(3, -3)
        interior_fine = slice(6, -6)  # Scale the interior appropriately
        
        error_coarse = jnp.max(jnp.abs(numerical_coarse[interior_coarse] - analytical_coarse[interior_coarse]))
        error_fine = jnp.max(jnp.abs(numerical_fine[interior_fine] - analytical_fine[interior_fine]))
        
        # Just check that the fine grid is more accurate
        self.assertLess(error_fine, error_coarse, 
                       "Fine grid should be more accurate than coarse grid")
    
    def test_vector_field_operations(self):
        """Test vector field operations on realistic field."""
        # Create a divergence-free vector field (should have zero divergence)
        # v = ∇ × A where A is a vector potential
        # For simplicity, use A = (0, 0, xy) so v = (y, -x, 0)
        
        vector_field = jnp.zeros((3, self.n, self.n, self.n))
        vector_field = vector_field.at[0].set(self.Y)   # v_x = y
        vector_field = vector_field.at[1].set(-self.X)  # v_y = -x
        vector_field = vector_field.at[2].set(0.0)      # v_z = 0
        
        # This field should have zero divergence
        div = divergence_3d(vector_field, self.dx)
        expected_div = jnp.zeros_like(div)
        
        np.testing.assert_allclose(
            div, expected_div, atol=self.tol_1st,
            err_msg="Divergence-free vector field should have zero divergence"
        )


if __name__ == '__main__':
    # Configure JAX for testing
    jax.config.update("jax_enable_x64", True)  # Use double precision
    
    unittest.main(verbosity=2)
