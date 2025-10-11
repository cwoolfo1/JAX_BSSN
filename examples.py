#!/usr/bin/env python3
"""
Example scripts demonstrating different NR1_JAX simulations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

from main import run_simulation
from init import get_initial_data
from bssn import BSSNParameters


def example_gravitational_wave():
    """Example: Gravitational wave propagation."""
    print("Example: Gravitational Wave Propagation")
    print("=" * 50)
    
    # Run simulation with gravitational wave initial data
    final_vars, constraint_history = run_simulation(
        initial_data_type='wave',
        grid_size=48,
        final_time=0.5,
        plot_interval=0.1,
        save_interval=0.05,
        verbose=True
    )
    
    print("Gravitational wave simulation completed!")
    return final_vars, constraint_history


def example_schwarzschild_evolution():
    """Example: Schwarzschild black hole evolution."""
    print("Example: Schwarzschild Black Hole Evolution")
    print("=" * 50)
    
    # Run simulation with Schwarzschild initial data
    final_vars, constraint_history = run_simulation(
        initial_data_type='schwarzschild',
        grid_size=32,
        final_time=0.3,
        plot_interval=0.05,
        save_interval=0.02,
        verbose=True
    )
    
    print("Schwarzschild evolution completed!")
    return final_vars, constraint_history


def example_gaussian_pulse():
    """Example: Gaussian pulse evolution."""
    print("Example: Gaussian Pulse Evolution")
    print("=" * 50)
    
    # Run simulation with Gaussian pulse
    final_vars, constraint_history = run_simulation(
        initial_data_type='gaussian',
        grid_size=64,
        final_time=1.0,
        plot_interval=0.2,
        save_interval=0.1,
        verbose=True
    )
    
    print("Gaussian pulse simulation completed!")
    return final_vars, constraint_history


def convergence_test():
    """Example: Convergence test with different resolutions."""
    print("Example: Convergence Test")
    print("=" * 50)
    
    resolutions = [24, 32, 48]
    results = []
    
    for res in resolutions:
        print(f"\nRunning simulation with {res}³ grid...")
        
        final_vars, constraint_history = run_simulation(
            initial_data_type='wave',
            grid_size=res,
            final_time=0.2,
            plot_interval=1.0,  # No intermediate plots
            save_interval=1.0,  # No intermediate saves
            verbose=False
        )
        
        # Extract final constraint violations
        final_constraints = constraint_history[-1][1] if constraint_history else None
        results.append((res, final_vars, final_constraints))
    
    # Analyze convergence
    print("\nConvergence Analysis:")
    print("Resolution | Hamiltonian L2 | Momentum L2")
    print("-" * 45)
    
    for res, vars, constraints in results:
        if constraints:
            ham_l2 = constraints['hamiltonian_l2']
            mom_l2 = constraints['momentum_l2']
            print(f"{res:^10} | {ham_l2:^14.2e} | {mom_l2:^11.2e}")
    
    # Plot convergence
    if len(results) >= 2:
        plot_convergence_results(results)
    
    return results


def plot_convergence_results(results):
    """Plot convergence test results."""
    resolutions = [res for res, _, _ in results]
    ham_l2 = [constraints['hamiltonian_l2'] if constraints else 0 
              for _, _, constraints in results]
    mom_l2 = [constraints['momentum_l2'] if constraints else 0
              for _, _, constraints in results]
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.loglog(resolutions, ham_l2, 'o-', label='Hamiltonian')
    plt.loglog(resolutions, mom_l2, 's-', label='Momentum')
    
    # Expected 4th-order convergence line
    if len(resolutions) >= 2:
        slope_4th = ham_l2[0] * (resolutions[0] / np.array(resolutions))**4
        plt.loglog(resolutions, slope_4th, '--', alpha=0.7, label='4th order')
    
    plt.xlabel('Grid Resolution')
    plt.ylabel('L2 Constraint Violation')
    plt.title('Constraint Convergence')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    if len(resolutions) >= 2:
        # Compute convergence rates
        ham_rates = []
        mom_rates = []
        
        for i in range(1, len(resolutions)):
            dx_ratio = resolutions[i-1] / resolutions[i]
            
            if ham_l2[i] > 0 and ham_l2[i-1] > 0:
                ham_rate = np.log(ham_l2[i-1] / ham_l2[i]) / np.log(dx_ratio)
                ham_rates.append(ham_rate)
            
            if mom_l2[i] > 0 and mom_l2[i-1] > 0:
                mom_rate = np.log(mom_l2[i-1] / mom_l2[i]) / np.log(dx_ratio)
                mom_rates.append(mom_rate)
        
        x_pos = range(len(ham_rates))
        plt.bar([x - 0.2 for x in x_pos], ham_rates, 0.4, label='Hamiltonian', alpha=0.7)
        plt.bar([x + 0.2 for x in x_pos], mom_rates, 0.4, label='Momentum', alpha=0.7)
        plt.axhline(y=4, color='red', linestyle='--', alpha=0.7, label='Expected (4th order)')
        
        plt.xlabel('Resolution Pair')
        plt.ylabel('Convergence Rate')
        plt.title('Measured Convergence Rates')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('convergence_test.png', dpi=150)
    plt.show()


def performance_benchmark():
    """Example: Performance benchmark."""
    print("Example: Performance Benchmark")
    print("=" * 50)
    
    import time
    
    grid_sizes = [24, 32, 48, 64]
    times = []
    
    for grid_size in grid_sizes:
        print(f"\nBenchmarking {grid_size}³ grid...")
        
        # Initialize data
        ni, nj, nk, dx = grid_size, grid_size, grid_size, 0.1
        vars = get_initial_data('wave', ni, nj, nk, dx)
        params = BSSNParameters(dx=dx, dt=0.001)
        
        # Import the evolution function
        from bssn import bssn_evolution_step
        
        # Compile (first call)
        _ = bssn_evolution_step(vars, params)
        
        # Time multiple steps
        n_steps = 20
        start_time = time.time()
        
        for step in range(n_steps):
            vars = bssn_evolution_step(vars, params)
        
        total_time = time.time() - start_time
        time_per_step = total_time / n_steps
        grid_points = grid_size**3
        
        times.append(time_per_step)
        
        print(f"  Time per step: {time_per_step*1000:.1f} ms")
        print(f"  Time per grid point: {time_per_step*1e6/grid_points:.2f} μs")
        print(f"  Throughput: {grid_points/time_per_step/1e6:.2f} Mpts/s")
    
    # Plot performance scaling
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    grid_points = [gs**3 for gs in grid_sizes]
    plt.loglog(grid_points, times, 'o-')
    plt.xlabel('Grid Points')
    plt.ylabel('Time per Step (s)')
    plt.title('Performance Scaling')
    plt.grid(True)
    
    # Ideal linear scaling
    if len(times) >= 2:
        linear_scaling = times[0] * np.array(grid_points) / grid_points[0]
        plt.loglog(grid_points, linear_scaling, '--', alpha=0.7, label='Linear scaling')
        plt.legend()
    
    plt.subplot(1, 2, 2)
    throughput = [gp / t / 1e6 for gp, t in zip(grid_points, times)]
    plt.semilogx(grid_points, throughput, 'o-')
    plt.xlabel('Grid Points')
    plt.ylabel('Throughput (Mpts/s)')
    plt.title('Computational Throughput')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('performance_benchmark.png', dpi=150)
    plt.show()
    
    return dict(zip(grid_sizes, times))


def main():
    """Run example simulations."""
    print("NR1_JAX Example Simulations")
    print("=" * 60)
    
    examples = [
        ("Gravitational Wave", example_gravitational_wave),
        ("Gaussian Pulse", example_gaussian_pulse),
        ("Convergence Test", convergence_test),
        ("Performance Benchmark", performance_benchmark),
        # Schwarzschild takes longer, so commented out for quick testing
        # ("Schwarzschild Evolution", example_schwarzschild_evolution),
    ]
    
    for name, example_func in examples:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            result = example_func()
            print(f"✓ {name} completed successfully")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
        print()
    
    print("All examples completed!")


if __name__ == "__main__":
    main()
