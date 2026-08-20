"""
Main simulation driver for NR1 JAX numerical relativity code.

This script sets up and runs a numerical relativity simulation using the
BSSN formulation with JAX for high-performance computation.
"""

import argparse
import time
from contextlib import nullcontext

import jax
from tqdm import tqdm

from JAX_BSSN.bssn import BSSNParameters, compute_momentum_constraint
from JAX_BSSN.diagnostics.openpmd import OpenPMDWriter
from JAX_BSSN.errors import (
    compute_all_constraints,
    compute_constraint_norms,
    compute_hamiltonian_constraint,
    print_constraint_summary,
    monitor_simulation_health,
)
from JAX_BSSN.initialization import (
    compute_adm_mass,
    compute_adm_momentum,
    get_initial_data,
    setup_simulation_parameters,
)
from JAX_BSSN.plotting import plot_results, plot_constraint_evolution, save_data
from JAX_BSSN.evolve import rk4_step


def run_simulation(
    initial_data_type: str = "gauge_wave",
    grid_size: int = 64,
    final_time: float = 1.0,
    plot_interval: float = 0.1,
    save_interval: float = 0.05,
    openpmd_output=None,
    verbose: bool = True,
):
    """
    Run numerical relativity simulation.

    Args:
        initial_data_type: Type of initial data ('gauge_wave', 'gowdy_wave', 'linear_wave')
        grid_size: Grid size (cubic grid)
        final_time: Final simulation time
        plot_interval: Time interval for plotting
        save_interval: Time interval between openPMD output iterations
        openpmd_output: Optional openPMD series filename
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

    if initial_data_type == "linear_wave":
        # Match the linear-wave notebook setup (Gauss coordinates).
        dt = dx / 4.0
        bssn_params = BSSNParameters(
            eta=0.0,
            kappa=0.0,
            nu=0.0,
            g=0.0,
            dx=dx,
            dt=dt,
        )
    else:
        bssn_params = BSSNParameters(
            eta=0.0,
            kappa=0.025,
            # momentum constraint damping
            nu=0.25,
            # Kreiss–Oliger dissipation coefficient
            g=0.0,
            dx=dx,
            dt=dt,
        )

    if verbose:
        print("Initializing data...")

    vars = get_initial_data(initial_data_type, grid_size, grid_size, grid_size, dx)

    initial_adm_mass = compute_adm_mass(vars, dx)
    initial_adm_momentum = compute_adm_momentum(vars, dx)

    if verbose:
        print(f"Initial ADM mass: {initial_adm_mass}")
        print(f"Initial ADM momentum: {initial_adm_momentum}")
        print()

    t = 0.0
    step = 0
    next_plot_time = 0.0
    next_save_time = 0.0

    Nt = int( (t_final - t) / dt )
    # compute the number of time steps

    constraint_history = []

    output_every = max(1, int(round(save_interval / dt)))
    first_grid_point = -(grid_size - 1) * dx / 2.0
    writer_context = nullcontext()
    if openpmd_output is not None:
        writer_context = OpenPMDWriter(
            openpmd_output,
            grid_spacing=(dx, dx, dx),
            grid_global_offset=(first_grid_point,) * 3,
            grid_position=(0.0, 0.0, 0.0),
            dt=dt,
            ghost_cells=0,
        )

    if verbose:
        print("Starting evolution...")
        if openpmd_output is not None:
            print(
                f"openPMD output: {writer_context.filename} "
                f"every {output_every} step(s)"
            )

    start_wall_time = time.time()

    with writer_context as writer:
        if writer is not None:
            hamiltonian = compute_hamiltonian_constraint(vars, bssn_params)
            momentum = compute_momentum_constraint(vars, bssn_params)
            writer.write(
                {
                    "lapse": vars.lapse,
                    "shift": tuple(vars.shift[i] for i in range(3)),
                    "K": vars.trace_K,
                    "W": vars.conformal_factor,
                    "hamiltonian_constraint": hamiltonian,
                    "momentum_constraint": tuple(momentum[i] for i in range(3)),
                },
                step=0,
                time=0.0,
            )

        for t in tqdm(range(Nt)):
            vars = rk4_step(vars, bssn_params)

            step += 1
            current_time = step * dt
            save_openpmd = writer is not None and (
                step % output_every == 0 or step == Nt
            )

            violations = None
            if step % 20 == 0:
                violations = compute_all_constraints(vars, bssn_params)
                norms = compute_constraint_norms(violations)
                constraint_history.append((current_time, norms))

            if save_openpmd:
                if violations is None:
                    hamiltonian = compute_hamiltonian_constraint(vars, bssn_params)
                    momentum = compute_momentum_constraint(vars, bssn_params)
                else:
                    hamiltonian = violations.hamiltonian
                    momentum = violations.momentum

                writer.write(
                    {
                        "lapse": vars.lapse,
                        "shift": tuple(vars.shift[i] for i in range(3)),
                        "K": vars.trace_K,
                        "W": vars.conformal_factor,
                        "hamiltonian_constraint": hamiltonian,
                        "momentum_constraint": tuple(
                            momentum[i] for i in range(3)
                        ),
                    },
                    step=step,
                    time=current_time,
                )

    wall_time = time.time() - start_wall_time

    final_adm_mass = compute_adm_mass(vars, dx)
    final_adm_momentum = compute_adm_momentum(vars, dx)

    if verbose:
        print(f"Final ADM mass: {final_adm_mass}")
        print(f"Final ADM momentum: {final_adm_momentum}")

    adm_mass_error = abs(final_adm_mass - initial_adm_mass) / ( abs(initial_adm_mass) + 1e-12)
    adm_momentum_error = abs(final_adm_momentum - initial_adm_momentum) / ( abs(initial_adm_momentum) + 1e-12)
    print(f"ADM mass relative error: {adm_mass_error}")
    print(f"ADM momentum relative error: {adm_momentum_error}")

    return vars, constraint_history


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="NR1 JAX Numerical Relativity Simulation")
    parser.add_argument(
        "--initial-data",
        default="gauge_wave",
        choices=["gauge_wave", "gowdy_wave", "linear_wave"],
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
    parser.add_argument(
        "--openpmd-output",
        default=None,
        help=(
            "Write a synchronous openPMD series to this filename using "
            "--save-interval as the cadence"
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    final_vars, constraint_history = run_simulation(
        initial_data_type=args.initial_data,
        grid_size=args.grid_size,
        final_time=args.final_time,
        plot_interval=args.plot_interval,
        save_interval=args.save_interval,
        openpmd_output=args.openpmd_output,
        verbose=not args.quiet,
    )

    return final_vars, constraint_history


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    final_vars, constraint_history = main()

    print("Simulation complete")
