"""
Main simulation driver for NR1 JAX numerical relativity code.

This script sets up and runs a numerical relativity simulation using the
BSSN formulation with JAX for high-performance computation.
"""

import argparse
import time

import jax
from tqdm import tqdm

from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.errors import (
    compute_all_constraints,
    compute_constraint_norms,
    print_constraint_summary,
    monitor_simulation_health,
)
from JAX_BSSN.initialization import (
    compute_adm_mass,
    compute_adm_momentum,
    get_initial_data,
    setup_simulation_parameters,
)
from JAX_BSSN.kreiss_oliger import get_optimal_dissipation_coefficient
from JAX_BSSN.plotting import plot_results, plot_constraint_evolution, save_data
from JAX_BSSN.evolve import rk4_step


def run_simulation(
    initial_data_type: str = "gauge_wave",
    grid_size: int = 64,
    final_time: float = 1.0,
    plot_interval: float = 0.1,
    save_interval: float = 0.05,
    verbose: bool = True,
):
    """
    Run numerical relativity simulation.

    Args:
        initial_data_type: Type of initial data ('gauge_wave', 'gowdy_wave')
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

    params_dict = setup_simulation_parameters()
    _, _, _, default_dx = params_dict["grid"]
    dx = default_dx
    dt = dx
    t_final = final_time

    bssn_params = BSSNParameters(
        eta=0.0,
        kappa=0.025,
        f=1.0,
        g=0.0,
        dx=dx,
        dt=dt,
    )
    ko_sigma = get_optimal_dissipation_coefficient(dx, dt)

    if verbose:
        print("Initializing data...")

    vars = get_initial_data(initial_data_type, grid_size, grid_size, grid_size, dx)

    adm_mass = compute_adm_mass(vars, dx)
    adm_momentum = compute_adm_momentum(vars, dx)

    if verbose:
        print(f"Initial ADM mass: {adm_mass:.6f}")
        print(f"Initial ADM momentum: {adm_momentum}")
        print()

    t = 0.0
    step = 0
    next_plot_time = 0.0
    next_save_time = 0.0

    Nt = int( (t_final - t) / dt )
    # compute the number of time steps

    constraint_history = []

    if verbose:
        print("Starting evolution...")

    start_wall_time = time.time()

    for t in tqdm(range(Nt)):
        vars = rk4_step(vars, bssn_params, ko_sigma)

        step += 1

        if step % 10 == 0:
            current_time = step * dt
            violations = compute_all_constraints(vars, bssn_params)
            norms = compute_constraint_norms(violations)
            constraint_history.append((current_time, norms))

    wall_time = time.time() - start_wall_time

    return vars, constraint_history


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="NR1 JAX Numerical Relativity Simulation")
    parser.add_argument(
        "--initial-data",
        default="gauge_wave",
        choices=["gauge_wave", "gowdy_wave"],
        help="Type of initial data",
    )
    parser.add_argument("--grid-size", type=int, default=64, help="Grid size (cubic)")
    parser.add_argument(
        "--final-time", type=float, default=1.0, help="Final simulation time"
    )
    parser.add_argument(
        "--plot-interval", type=float, default=0.1, help="Time interval for plotting"
    )
    parser.add_argument(
        "--save-interval", type=float, default=0.05, help="Time interval for saving data"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    final_vars, constraint_history = run_simulation(
        initial_data_type=args.initial_data,
        grid_size=args.grid_size,
        final_time=args.final_time,
        plot_interval=args.plot_interval,
        save_interval=args.save_interval,
        verbose=not args.quiet,
    )

    return final_vars, constraint_history


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    try:
        final_vars, constraint_history = main()
    except SystemExit:
        pass
