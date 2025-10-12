#!/usr/bin/env python3
"""
Test script for NR1_JAX numerical relativity code.

This script runs various tests to verify the correctness of the implementation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import time

# Enable 64-bit precision for testing
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_device', jax.devices('cpu')[0])

from bssn import BSSNVariables, BSSNParameters, bssn_evolution_step
from init import get_initial_data, flat_spacetime_data, gravitational_wave_data
from derivatives import diff1_field, compute_all_derivatives, laplacian_3d
from tensor_algebra import invert_3x3_metric, determinant_3x3_metric, christoffel_symbols_second_kind
from errors import compute_all_constraints, compute_constraint_norms
from kreiss_oliger import apply_ko_dissipation_bssn


def test_flat_spacetime():
    """Test that flat spacetime remains flat."""
    print("Testing flat spacetime evolution...")
    
    ni, nj, nk = 32, 32, 32
    dx = 0.1
    dt = 0.001
    
    # Initialize flat spacetime
    vars = flat_spacetime_data(ni, nj, nk, dx)
    params = BSSNParameters(dx=dx, dt=dt)
    print("Initialized flat spacetime data")

    violations = compute_all_constraints(vars, params)
    norms = compute_constraint_norms(violations)
    
    print(f"  Initial Hamiltonian constraint: {norms['hamiltonian_l2']:.2e}")
    print(f"  Initial momentum constraint: {norms['momentum_l2']:.2e}")
    
    # Evolve for a few steps
    for step in range(10):
        vars = bssn_evolution_step(vars, params)
    
    print("Evolved flat spacetime data for 10 steps")
    # Check that spacetime remains flat
    final_violations = compute_all_constraints(vars, params)
    final_norms = compute_constraint_norms(final_violations)
    
    print(f"  Final Hamiltonian constraint: {final_norms['hamiltonian_l2']:.2e}")
    print(f"  Final momentum constraint: {final_norms['momentum_l2']:.2e}")
    
    # Constraints should remain small
    assert final_norms['hamiltonian_l2'] < 1e-10, "Hamiltonian constraint violated"
    assert final_norms['momentum_l2'] < 1e-10, "Momentum constraint violated"
    
    print("  ✓ Flat spacetime test passed")


def test_derivatives():
    """Test finite difference derivatives."""
    print("Testing finite difference derivatives...")
    
    ni, nj, nk = 25, 25, 25
    dx = 0.05
    
    # Create coordinate arrays
    x = jnp.arange(ni) * dx
    y = jnp.arange(nj) * dx
    z = jnp.arange(nk) * dx
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    
    # Test function: f(x,y,z) = sin(x) * cos(y) * cos(z)
    f = jnp.sin(2*jnp.pi*X / (ni*dx)) * jnp.cos(2*jnp.pi*Y / (nj*dx)) * jnp.cos(2*jnp.pi*Z / (nk*dx))
    
    # Compute numerical derivatives
    df_dx = diff1_field(f, 0, dx)
    df_dy = diff1_field(f, 1, dx)
    df_dz = diff1_field(f, 2, dx)
    
    # Analytical derivatives
    df_dx_exact = 2*jnp.pi/(ni*dx) * jnp.cos(2*jnp.pi*X / (ni*dx)) * jnp.cos(2*jnp.pi*Y / (nj*dx)) * jnp.cos(2*jnp.pi*Z / (nk*dx))
    df_dy_exact = -2*jnp.pi/(nj*dx) * jnp.sin(2*jnp.pi*X / (ni*dx)) * jnp.sin(2*jnp.pi*Y / (nj*dx)) * jnp.cos(2*jnp.pi*Z / (nk*dx))
    df_dz_exact = -2*jnp.pi/(nk*dx) * jnp.sin(2*jnp.pi*X / (ni*dx)) * jnp.cos(2*jnp.pi*Y / (nj*dx)) * jnp.sin(2*jnp.pi*Z / (nk*dx))
    
    # Check errors (should be small for smooth functions)
    error_x = jnp.max(jnp.abs(df_dx - df_dx_exact))
    error_y = jnp.max(jnp.abs(df_dy - df_dy_exact))
    error_z = jnp.max(jnp.abs(df_dz - df_dz_exact))
    
    print(f"  Derivative error in x: {error_x:.2e}")
    print(f"  Derivative error in y: {error_y:.2e}")
    print(f"  Derivative error in z: {error_z:.2e}")
    
    # Errors should be small for 4th-order method
    assert error_x < 1e-3, "X derivative error too large"
    assert error_y < 1e-3, "Y derivative error too large"
    assert error_z < 1e-3, "Z derivative error too large"
    
    print("  ✓ Derivative test passed")


def test_tensor_operations():
    """Test tensor algebra operations."""
    print("Testing tensor operations...")
    
    ni, nj, nk = 8, 8, 8
    
    # Create a simple metric (slightly perturbed identity)
    metric = jnp.zeros((3, 3, ni, nj, nk))
    for i in range(3):
        metric = metric.at[i, i].set(1.0 + 0.1 * jnp.sin(jnp.arange(ni)[:, None, None]))
    
    # Test metric inversion
    inv_metric = invert_3x3_metric(metric)
    
    # Check that metric * inv_metric = identity
    identity_check = jnp.zeros((3, 3, ni, nj, nk))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                identity_check = identity_check.at[i, j].add(
                    metric[i, k] * inv_metric[k, j])
    
    # Should be close to identity matrix
    error = jnp.max(jnp.abs(identity_check - jnp.eye(3)[:, :, None, None, None]))
    print(f"  Metric inversion error: {error:.2e}")
    
    assert error < 1e-12, "Metric inversion failed"
    
    # Test determinant
    det = determinant_3x3_metric(metric)
    print(f"  Determinant range: [{jnp.min(det):.3f}, {jnp.max(det):.3f}]")
    
    assert jnp.all(det > 0), "Metric determinant should be positive"
    
    print("  ✓ Tensor operations test passed")


def test_gravitational_waves():
    """Test gravitational wave initial data."""
    print("Testing gravitational wave initial data...")
    
    ni, nj, nk = 32, 32, 32
    dx = 0.1
    
    # Create wave data
    vars = gravitational_wave_data(ni, nj, nk, dx, amplitude=0.01, wavelength=2.0)
    
    # Check that conformal metric determinant is 1
    det_gamma = determinant_3x3_metric(vars.conformal_metric)
    det_error = jnp.max(jnp.abs(det_gamma - 1.0))
    
    print(f"  det(γ) - 1 error: {det_error:.2e}")
    assert det_error < 1e-10, "Conformal metric determinant should be 1"
    
    # Check that wave has expected structure
    wave_amplitude = jnp.max(jnp.abs(vars.conformal_metric[0, 0] - 1.0))
    print(f"  Wave amplitude: {wave_amplitude:.3f}")
    
    assert wave_amplitude > 0.005, "Wave amplitude too small"
    assert wave_amplitude < 0.02, "Wave amplitude too large"
    
    print("  ✓ Gravitational wave test passed")


def test_constraint_preservation():
    """Test that constraints are preserved during evolution."""
    print("Testing constraint preservation...")
    
    ni, nj, nk = 24, 24, 24
    dx = 0.1
    dt = 0.001
    
    # Start with small wave
    vars = gravitational_wave_data(ni, nj, nk, dx, amplitude=0.001, wavelength=3.0)
    params = BSSNParameters(dx=dx, dt=dt)
    
    # Initial constraints
    initial_violations = compute_all_constraints(vars, params)
    initial_norms = compute_constraint_norms(initial_violations)
    
    print(f"  Initial Hamiltonian: {initial_norms['hamiltonian_l2']:.2e}")
    print(f"  Initial momentum: {initial_norms['momentum_l2']:.2e}")
    
    # Evolve for several steps
    for step in range(50):
        vars = bssn_evolution_step(vars, params)
    
    # Final constraints
    final_violations = compute_all_constraints(vars, params)
    final_norms = compute_constraint_norms(final_violations)
    
    print(f"  Final Hamiltonian: {final_norms['hamiltonian_l2']:.2e}")
    print(f"  Final momentum: {final_norms['momentum_l2']:.2e}")
    
    # Constraints should not grow too much
    ham_growth = final_norms['hamiltonian_l2'] / max(initial_norms['hamiltonian_l2'], 1e-15)
    mom_growth = final_norms['momentum_l2'] / max(initial_norms['momentum_l2'], 1e-15)
    
    print(f"  Constraint growth: Ham={ham_growth:.1f}x, Mom={mom_growth:.1f}x")
    
    assert ham_growth < 100, "Hamiltonian constraint grew too much"
    assert mom_growth < 100, "Momentum constraint grew too much"
    
    print("  ✓ Constraint preservation test passed")


def test_performance():
    """Test computational performance."""
    print("Testing performance...")
    
    ni, nj, nk = 64, 64, 64
    dx = 0.05
    dt = 0.0005
    
    # Initialize data
    vars = gravitational_wave_data(ni, nj, nk, dx, amplitude=0.01)
    params = BSSNParameters(dx=dx, dt=dt)
    
    # Compile the evolution function
    print("  Compiling JIT functions...")
    start_time = time.time()
    vars_compiled = bssn_evolution_step(vars, params)
    compile_time = time.time() - start_time
    
    print(f"  Compilation time: {compile_time:.2f} seconds")
    
    # Time evolution steps
    n_steps = 10
    start_time = time.time()
    
    for step in range(n_steps):
        vars = bssn_evolution_step(vars, params)
    
    evolution_time = time.time() - start_time
    time_per_step = evolution_time / n_steps
    
    print(f"  Evolution time: {evolution_time:.2f} seconds for {n_steps} steps")
    print(f"  Time per step: {time_per_step*1000:.1f} ms")
    print(f"  Grid points: {ni*nj*nk:,}")
    print(f"  Time per grid point per step: {time_per_step*1e6/(ni*nj*nk):.5f} μs")
    
    # Performance should be reasonable
    assert time_per_step < 5.0, "Evolution too slow"
    
    print("  ✓ Performance test passed")


def test_dissipation():
    """Test Kreiss-Oliger dissipation."""
    print("Testing Kreiss-Oliger dissipation...")
    
    ni, nj, nk = 32, 32, 32
    dx = 0.1
    
    # Create data with high-frequency noise
    vars = flat_spacetime_data(ni, nj, nk, dx)
    
    # Add high-frequency perturbation
    i_indices = jnp.arange(ni)
    noise = 0.001 * jnp.sin(10 * i_indices)[:, None, None]
    vars = vars._replace(
        conformal_factor=vars.conformal_factor + noise
    )
    
    # Apply dissipation
    dissipation = apply_ko_dissipation_bssn(vars, sigma=0.01, dx=dx)
    
    # Check that dissipation opposes the noise
    dissipation_magnitude = jnp.max(jnp.abs(dissipation.conformal_factor))
    print(f"  Dissipation magnitude: {dissipation_magnitude:.2e}")
    
    assert dissipation_magnitude > 1e-6, "Dissipation too weak"
    assert dissipation_magnitude < 1e-2, "Dissipation too strong"
    
    print("  ✓ Dissipation test passed")


def run_all_tests():
    """Run all tests."""
    print("Running NR1_JAX tests...\n")
    
    tests = [
        test_derivatives,
        test_tensor_operations, 
        test_flat_spacetime,
        test_gravitational_waves,
        test_constraint_preservation,
        test_dissipation,
        test_performance
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print()
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            failed += 1
            print()
    
    print(f"Test summary: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
