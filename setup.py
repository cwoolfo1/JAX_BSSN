#!/usr/bin/env python3
"""
Setup script for NR1_JAX.
"""

import subprocess
import sys
import os


def install_requirements():
    """Install required packages."""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False


def check_jax_installation():
    """Check that JAX is properly installed."""
    print("Checking JAX installation...")
    try:
        import jax
        import jax.numpy as jnp
        
        # Test basic functionality
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x)
        
        print(f"✓ JAX version: {jax.__version__}")
        print(f"✓ JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
        
        # Test JIT compilation
        @jax.jit
        def test_jit(x):
            return x**2 + 2*x + 1
        
        result = test_jit(5.0)
        print(f"✓ JIT compilation works: test_jit(5.0) = {result}")
        
        return True
    except Exception as e:
        print(f"✗ JAX installation issue: {e}")
        return False


def run_basic_tests():
    """Run basic functionality tests."""
    print("Running basic tests...")
    try:
        # Test imports
        from bssn import BSSNVariables, BSSNParameters
        from init import flat_spacetime_data
        from derivatives import diff1_field
        
        print("✓ All modules import successfully")
        
        # Test basic functionality
        vars = flat_spacetime_data(8, 8, 8, 0.1)
        print("✓ Can create initial data")
        
        # Test derivatives
        test_field = vars.conformal_factor
        deriv = diff1_field(test_field, 0, 0.1)
        print("✓ Can compute derivatives")
        
        return True
    except Exception as e:
        print(f"✗ Basic test failed: {e}")
        return False


def create_output_directory():
    """Create output directory for simulation results."""
    os.makedirs("output", exist_ok=True)
    print("✓ Output directory created")


def main():
    """Main setup function."""
    print("Setting up NR1_JAX...")
    print("=" * 50)
    
    success = True
    
    # Install requirements
    if not install_requirements():
        success = False
    
    # Check JAX
    if not check_jax_installation():
        success = False
    
    # Run basic tests
    if not run_basic_tests():
        success = False
    
    # Create directories
    create_output_directory()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Setup completed successfully!")
        print("\nYou can now run:")
        print("  python main.py --help                 # See command-line options")
        print("  python test_nr1_jax.py               # Run test suite")
        print("  python examples.py                   # Run example simulations")
        print("  python main.py --initial-data wave   # Run gravitational wave simulation")
    else:
        print("❌ Setup encountered issues. Please resolve them before proceeding.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
