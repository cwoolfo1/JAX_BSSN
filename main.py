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
from bssn import BSSNVariables, BSSNParameters, bssn_evolution_step
from init import get_initial_data, compute_adm_mass, compute_adm_momentum
from kreiss_oliger import (apply_ko_dissipation_bssn, apply_boundary_dissipation_bssn,
                          get_optimal_dissipation_coefficient)
from errors import (compute_all_constraints, compute_constraint_norms,
                   compute_energy_density, print_constraint_summary,
                   monitor_simulation_health)
from derivatives import compute_all_derivatives


def setup_simulation_parameters():
    """Set up default simulation parameters."""
    # Grid parameters
    ni, nj, nk = 64, 64, 64  # Grid size
    dx = 0.1                 # Grid spacing
    
    # Evolution parameters
    dt = 0.001              # Time step
    t_final = 1.0           # Final time
    
    # BSSN parameters
    bssn_params = BSSNParameters(
        eta=2.0,            # Gamma damping
        f=2.0,              # 1+log slicing parameter
        g=0.75,             # Gamma driver parameter
        dx=dx,
        dt=dt
    )
    
    # Dissipation parameters
    ko_sigma = get_optimal_dissipation_coefficient(dx, dt)
    
    return {
        'grid': (ni, nj, nk, dx),
        'evolution': (dt, t_final),
        'bssn_params': bssn_params,
        'ko_sigma': ko_sigma
    }


@jit 
def forward_euler_step(vars: BSSNVariables, params: BSSNParameters,
                        ko_sigma: float) -> BSSNVariables:
    """
    Perform one forward Euler timestep with dissipation.
    
    Args:
        vars: Current BSSN variables
        params: Evolution parameters
        ko_sigma: Kreiss-Oliger dissipation coefficient
        
    Returns:
        Updated BSSN variables
    """
    dt = params.dt
    
    # Compute BSSN time derivatives
    new_vars = bssn_evolution_step(vars, params)
    
    # Extract time derivatives
    dt_gamma = (new_vars.conformal_metric - vars.conformal_metric) / dt
    dt_W = (new_vars.conformal_factor - vars.conformal_factor) / dt
    dt_A = (new_vars.traceless_K - vars.traceless_K) / dt
    dt_K = (new_vars.trace_K - vars.trace_K) / dt
    dt_Gamma = (new_vars.conformal_connection - vars.conformal_connection) / dt
    dt_alpha = (new_vars.lapse - vars.lapse) / dt
    dt_beta = (new_vars.shift - vars.shift) / dt
    
    # Add Kreiss-Oliger dissipation
    dissipation = apply_ko_dissipation_bssn(vars, ko_sigma, params.dx)
    
    dt_gamma += dissipation.conformal_metric
    dt_W += dissipation.conformal_factor
    dt_A += dissipation.traceless_K
    dt_K += dissipation.trace_K
    dt_Gamma += dissipation.conformal_connection
    dt_alpha += dissipation.lapse
    dt_beta += dissipation.shift
    
    # Update variables
    new_vars = BSSNVariables(
        conformal_metric=vars.conformal_metric + dt * dt_gamma,
        conformal_factor=vars.conformal_factor + dt * dt_W,
        traceless_K=vars.traceless_K + dt * dt_A,
        trace_K=vars.trace_K + dt * dt_K,
        conformal_connection=vars.conformal_connection + dt * dt_Gamma,
        lapse=vars.lapse + dt * dt_alpha,
        shift=vars.shift + dt * dt_beta
    )
    
    return new_vars

@jit
def runge_kutta_4_step(vars: BSSNVariables, params: BSSNParameters,
                      ko_sigma: float) -> BSSNVariables:
    """
    Perform one 4th-order Runge-Kutta timestep with dissipation.
    
    Args:
        vars: Current BSSN variables
        params: Evolution parameters
        ko_sigma: Kreiss-Oliger dissipation coefficient
        
    Returns:
        Updated BSSN variables
    """
    dt = params.dt
    
    # RK4 intermediate steps
    def compute_rhs(v, add_dissipation=True):
        # Compute BSSN time derivatives
        new_v = bssn_evolution_step(v, params)
        
        # Extract time derivatives
        dt_gamma = (new_v.conformal_metric - v.conformal_metric) / dt
        dt_W = (new_v.conformal_factor - v.conformal_factor) / dt
        dt_A = (new_v.traceless_K - v.traceless_K) / dt
        dt_K = (new_v.trace_K - v.trace_K) / dt
        dt_Gamma = (new_v.conformal_connection - v.conformal_connection) / dt
        dt_alpha = (new_v.lapse - v.lapse) / dt
        dt_beta = (new_v.shift - v.shift) / dt
        
        if add_dissipation:
            # Add Kreiss-Oliger dissipation
            dissipation = apply_ko_dissipation_bssn(v, ko_sigma, params.dx)
            
            dt_gamma += dissipation.conformal_metric
            dt_W += dissipation.conformal_factor
            dt_A += dissipation.traceless_K
            dt_K += dissipation.trace_K
            dt_Gamma += dissipation.conformal_connection
            dt_alpha += dissipation.lapse
            dt_beta += dissipation.shift
        
        return BSSNVariables(
            conformal_metric=dt_gamma,
            conformal_factor=dt_W,
            traceless_K=dt_A,
            trace_K=dt_K,
            conformal_connection=dt_Gamma,
            lapse=dt_alpha,
            shift=dt_beta
        )
    
    # RK4 coefficients
    k1 = compute_rhs(vars)
    
    vars_temp = BSSNVariables(
        conformal_metric=vars.conformal_metric + 0.5 * dt * k1.conformal_metric,
        conformal_factor=vars.conformal_factor + 0.5 * dt * k1.conformal_factor,
        traceless_K=vars.traceless_K + 0.5 * dt * k1.traceless_K,
        trace_K=vars.trace_K + 0.5 * dt * k1.trace_K,
        conformal_connection=vars.conformal_connection + 0.5 * dt * k1.conformal_connection,
        lapse=vars.lapse + 0.5 * dt * k1.lapse,
        shift=vars.shift + 0.5 * dt * k1.shift
    )
    k2 = compute_rhs(vars_temp)
    
    vars_temp = BSSNVariables(
        conformal_metric=vars.conformal_metric + 0.5 * dt * k2.conformal_metric,
        conformal_factor=vars.conformal_factor + 0.5 * dt * k2.conformal_factor,
        traceless_K=vars.traceless_K + 0.5 * dt * k2.traceless_K,
        trace_K=vars.trace_K + 0.5 * dt * k2.trace_K,
        conformal_connection=vars.conformal_connection + 0.5 * dt * k2.conformal_connection,
        lapse=vars.lapse + 0.5 * dt * k2.lapse,
        shift=vars.shift + 0.5 * dt * k2.shift
    )
    k3 = compute_rhs(vars_temp)
    
    vars_temp = BSSNVariables(
        conformal_metric=vars.conformal_metric + dt * k3.conformal_metric,
        conformal_factor=vars.conformal_factor + dt * k3.conformal_factor,
        traceless_K=vars.traceless_K + dt * k3.traceless_K,
        trace_K=vars.trace_K + dt * k3.trace_K,
        conformal_connection=vars.conformal_connection + dt * k3.conformal_connection,
        lapse=vars.lapse + dt * k3.lapse,
        shift=vars.shift + dt * k3.shift
    )
    k4 = compute_rhs(vars_temp)
    
    # Final update
    new_vars = BSSNVariables(
        conformal_metric=vars.conformal_metric + (dt/6.0) * (k1.conformal_metric + 2*k2.conformal_metric + 2*k3.conformal_metric + k4.conformal_metric),
        conformal_factor=vars.conformal_factor + (dt/6.0) * (k1.conformal_factor + 2*k2.conformal_factor + 2*k3.conformal_factor + k4.conformal_factor),
        traceless_K=vars.traceless_K + (dt/6.0) * (k1.traceless_K + 2*k2.traceless_K + 2*k3.traceless_K + k4.traceless_K),
        trace_K=vars.trace_K + (dt/6.0) * (k1.trace_K + 2*k2.trace_K + 2*k3.trace_K + k4.trace_K),
        conformal_connection=vars.conformal_connection + (dt/6.0) * (k1.conformal_connection + 2*k2.conformal_connection + 2*k3.conformal_connection + k4.conformal_connection),
        lapse=vars.lapse + (dt/6.0) * (k1.lapse + 2*k2.lapse + 2*k3.lapse + k4.lapse),
        shift=vars.shift + (dt/6.0) * (k1.shift + 2*k2.shift + 2*k3.shift + k4.shift)
    )
    
    return new_vars


def save_data(vars: BSSNVariables, t: float, step: int, 
              output_dir: str = "output"):
    """Save simulation data to files."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save key quantities
    np.savez(f"{output_dir}/data_step_{step:06d}.npz",
             time=t,
             conformal_metric=np.array(vars.conformal_metric),
             conformal_factor=np.array(vars.conformal_factor),
             traceless_K=np.array(vars.traceless_K),
             trace_K=np.array(vars.trace_K),
             lapse=np.array(vars.lapse))


def plot_results(vars: BSSNVariables, t: float, params: BSSNParameters):
    """Create diagnostic plots."""
    ni, nj, nk = vars.conformal_factor.shape
    
    # Central slices
    i_center = ni // 2
    j_center = nj // 2
    k_center = nk // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'BSSN Evolution at t = {t:.3f}')
    
    # Conformal factor
    im1 = axes[0, 0].imshow(vars.conformal_factor[:, :, k_center], 
                           origin='lower', aspect='equal')
    axes[0, 0].set_title('Conformal Factor W (z=0)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Lapse function
    im2 = axes[0, 1].imshow(vars.lapse[:, :, k_center], 
                           origin='lower', aspect='equal')
    axes[0, 1].set_title('Lapse α (z=0)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Trace of K
    im3 = axes[0, 2].imshow(vars.trace_K[:, :, k_center], 
                           origin='lower', aspect='equal')
    axes[0, 2].set_title('Trace K (z=0)')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Conformal metric component
    im4 = axes[1, 0].imshow(vars.conformal_metric[0, 0, :, :, k_center], 
                           origin='lower', aspect='equal')
    axes[1, 0].set_title('γ_xx (z=0)')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Extrinsic curvature component
    im5 = axes[1, 1].imshow(vars.traceless_K[0, 0, :, :, k_center], 
                           origin='lower', aspect='equal')
    axes[1, 1].set_title('A_xx (z=0)')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Energy density
    energy = compute_energy_density(vars, params)
    im6 = axes[1, 2].imshow(energy[:, :, k_center], 
                           origin='lower', aspect='equal')
    axes[1, 2].set_title('Energy Density (z=0)')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(f'evolution_t_{t:.3f}.png', dpi=150)
    plt.show()


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
        # Evolve one step
        # vars = runge_kutta_4_step(vars, bssn_params, ko_sigma)
        vars = forward_euler_step(vars, bssn_params, ko_sigma)  # evolve using forward Euler
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


def plot_constraint_evolution(constraint_history: List[Tuple[float, dict]]):
    """Plot evolution of constraint violations."""
    times = [entry[0] for entry in constraint_history]
    
    hamiltonian_l2 = [entry[1]['hamiltonian_l2'] for entry in constraint_history]
    momentum_l2 = [entry[1]['momentum_l2'] for entry in constraint_history]
    det_gamma_l2 = [entry[1]['det_gamma_l2'] for entry in constraint_history]
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.semilogy(times, hamiltonian_l2, 'b-', label='Hamiltonian')
    plt.xlabel('Time')
    plt.ylabel('L2 Norm')
    plt.title('Hamiltonian Constraint')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.semilogy(times, momentum_l2, 'r-', label='Momentum')
    plt.xlabel('Time')
    plt.ylabel('L2 Norm')
    plt.title('Momentum Constraint')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.semilogy(times, det_gamma_l2, 'g-', label='det(γ)-1')
    plt.xlabel('Time')
    plt.ylabel('L2 Norm')
    plt.title('det(γ) = 1 Condition')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.loglog(times, hamiltonian_l2, 'b-', label='Hamiltonian')
    plt.loglog(times, momentum_l2, 'r-', label='Momentum')
    plt.loglog(times, det_gamma_l2, 'g-', label='det(γ)-1')
    plt.xlabel('Time')
    plt.ylabel('L2 Norm')
    plt.title('All Constraints')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('constraint_evolution.png', dpi=150)
    plt.show()


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
