"""
Main simulation driver for NR1 JAX numerical relativity code.

This script sets up and runs a numerical relativity simulation using the
BSSN formulation with JAX for high-performance computation.
"""

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple
import argparse
import json
import time

# Import our modules
from JAX_BSSN.bssn import BSSNVariables, BSSNParameters
from JAX_BSSN.initialization import get_initial_data, compute_adm_mass, compute_adm_momentum
from JAX_BSSN.kreiss_oliger import apply_ko_dissipation_bssn, get_optimal_dissipation_coefficient
from JAX_BSSN.errors import (compute_all_constraints, compute_constraint_norms,
                   compute_energy_density, print_constraint_summary,
                   monitor_simulation_health)
from JAX_BSSN.derivatives import compute_all_derivatives
from JAX_BSSN.plotting import plot_results, plot_constraint_evolution, save_data
from JAX_BSSN.initialization import setup_simulation_parameters
from JAX_BSSN.evolve import rk4_step



def run_simulation(initial_data_type: str = 'wave',
                  grid_size: int = 64,
                  final_time: float = 1.0,
                  plot_interval: float = 0.1,
                  save_interval: float = 0.05,
                  verbose: bool = True):
    """
    Run numerical relativity simulation.
    
    Args:
        initial_data_type: Type of initial data ('flat', 'wave', 'schwarzschild', etc.)
        grid_size: Grid size (cubic grid)
        final_time: Final simulation time
        plot_interval: Time interval for plotting
        save_interval: Time interval for saving data
        verbose: Whether to print progress
    """
    if verbose:
        print("Setting up NR1 JAX simulation...")
        print(f"Initial data: {initial_data_type}")
        print(f"Grid size: {grid_size}³")
        print(f"Final time: {final_time}")
    
    # Setup parameters
    params_dict = setup_simulation_parameters()
    ni, nj, nk, dx = grid_size, grid_size, grid_size, 0.1
    dt = 0.001
    t_final = final_time
    
    # Update parameters
    bssn_params = BSSNParameters(
        eta=2.0, f=2.0, g=0.75, dx=dx, dt=dt
    )
    ko_sigma = get_optimal_dissipation_coefficient(dx, dt)
    
    # Initialize data
    if verbose:
        print("Initializing data...")
    
    if initial_data_type == 'wave':
        vars = get_initial_data('wave', ni, nj, nk, dx, 
                               amplitude=0.1, wavelength=1.0)
    elif initial_data_type == 'schwarzschild':
        vars = get_initial_data('schwarzschild', ni, nj, nk, dx, mass=1.0)
    elif initial_data_type == 'gaussian':
        vars = get_initial_data('gaussian', ni, nj, nk, dx,
                               amplitude=0.1, width=1.0)
    else:
        vars = get_initial_data('flat', ni, nj, nk, dx)
    
    # Compute initial quantities
    adm_mass = compute_adm_mass(vars, dx)
    adm_momentum = compute_adm_momentum(vars, dx)
    
    if verbose:
        print(f"Initial ADM mass: {adm_mass:.6f}")
        print(f"Initial ADM momentum: {adm_momentum}")
        print()
    
    # Evolution loop
    t = 0.0
    step = 0
    next_plot_time = 0.0
    next_save_time = 0.0
    
    # Storage for constraint monitoring
    constraint_history = []
    
    if verbose:
        print("Starting evolution...")
    
    start_wall_time = time.time()
    
    while t < t_final:

        print(f"Step {step}, Time {t:.4f}")
        # Evolve one step
        vars = rk4_step(vars, bssn_params, ko_sigma)  # evolve using RK4
        t += dt
        step += 1
        
        # Monitor simulation health
        if not monitor_simulation_health(vars, bssn_params, t):
            print("Simulation stopped due to instability")
            break
        
        # Compute and store constraints
        if step % 10 == 0:  # Every 10 steps
            violations = compute_all_constraints(vars, bssn_params)
            norms = compute_constraint_norms(violations)
            constraint_history.append((t, norms))

            if verbose and step % 100 == 0:
                print_constraint_summary(violations, t)

        # Save data
        if t >= next_save_time:
            save_data(vars, t, step)
            next_save_time += save_interval
        
        # Plot results
        if t >= next_plot_time:
            plot_results(vars, t, bssn_params)
            next_plot_time += plot_interval
    
    wall_time = time.time() - start_wall_time
    
    if verbose:
        print(f"Evolution completed!")
        print(f"Final time: {t:.4f}")
        print(f"Total steps: {step}")
        print(f"Wall time: {wall_time:.2f} seconds")
        print(f"Time per step: {wall_time/step*1000:.2f} ms")
        print(f"Simulation time per wall time: {t/wall_time:.1f}x")

    # Final diagnostics
    violations = compute_all_constraints(vars, bssn_params)
    if verbose:
        print("\nFinal constraint violations:")
        print_constraint_summary(violations, t)
    
    # Plot constraint evolution
    if constraint_history:
        plot_constraint_evolution(constraint_history)
    
    return vars, constraint_history



def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='NR1 JAX Numerical Relativity Simulation')
    parser.add_argument('--initial-data', default='wave', 
                       choices=['flat', 'wave', 'schwarzschild', 'gaussian'],
                       help='Type of initial data')
    parser.add_argument('--grid-size', type=int, default=64,
                       help='Grid size (cubic)')
    parser.add_argument('--final-time', type=float, default=1.0,
                       help='Final simulation time')
    parser.add_argument('--plot-interval', type=float, default=0.1,
                       help='Time interval for plotting')
    parser.add_argument('--save-interval', type=float, default=0.05,
                       help='Time interval for saving data')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Run simulation
    final_vars, constraint_history = run_simulation(
        initial_data_type=args.initial_data,
        grid_size=args.grid_size,
        final_time=args.final_time,
        plot_interval=args.plot_interval,
        save_interval=args.save_interval,
        verbose=not args.quiet
    )
    
    return final_vars, constraint_history


if __name__ == "__main__":
    # Enable 64-bit precision
    jax.config.update("jax_enable_x64", True)
    
    # Run simulation with default parameters if called directly
    try:
        final_vars, constraint_history = main()
    except SystemExit:
        # Handle argparse exit gracefully
        pass
