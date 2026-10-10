"""Evolve Bowen–York head-on punctures with axisymmetric Cartoon methods."""

import json
import math
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from tqdm import tqdm

from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.cartoon.axisymmetry import (
    axisymmetric_plane_output_fields,
    axisymmetric_rk4_step,
    compact_axisymmetric_state,
    compute_axisymmetric_constraint_norms,
    compute_axisymmetric_constraints,
    validate_axisymmetric_grid,
)
from JAX_BSSN.cartoon.interpolation import lagrange6_nonperiodic
from JAX_BSSN.diagnostics.openpmd import OpenPMDWriter
from JAX_BSSN.evolution.boundaries import PERIODIC_BC, SOMMERFELD_BC
from JAX_BSSN.utilities.bowen_york_solver import (
    bowen_york_extrinsic_curvature,
    brill_lindquist_conformal_factor,
    cell_centered_grid,
    solve_conformal_factor,
)


# Bare puncture masses, coordinate separation, and inward momentum magnitude.
MASSES = (0.5, 0.5)
SEPARATION = 6.0
MOMENTUM = 0.05

# The full z interval has 2 * NUM_RADIAL_POINTS cells and the same spacing.
R_MAX = 12.0
NUM_RADIAL_POINTS = 48
CFL = 0.2
FINAL_TIME = 50.0
SNAPSHOT_COUNT = 250

NEWTON_TOLERANCE = 1.0e-10
MAX_NEWTON_ITERATIONS = 12
CG_TOLERANCE = 1.0e-10
MAX_CG_ITERATIONS = 1000

KAPPA = 0.002
ETA = 2.0
NU = 0.02
GAMMA_DRIVER = 0.75

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OVERWRITE = False
SHOW_PROGRESS = True


def bowen_york_plane_data(
    masses=MASSES,
    separation=SEPARATION,
    momentum=MOMENTUM,
    num_radial_points=NUM_RADIAL_POINTS,
    r_max=R_MAX,
    newton_tolerance=NEWTON_TOLERANCE,
    max_newton_iterations=MAX_NEWTON_ITERATIONS,
    cg_tolerance=CG_TOLERANCE,
    max_cg_iterations=MAX_CG_ITERATIONS,
    verbose=False,
):
    """Solve on a Cartesian cube and return compact BSSN data and residuals.

    Only the regular correction is interpolated to y=0. The singular factor
    and Bowen–York curvature are evaluated analytically at the plane samples.
    The solver's A_BY differs from the evolved BSSN curvature by psi**-6.
    """
    if len(masses) != 2 or any(not math.isfinite(m) or m <= 0 for m in masses):
        raise ValueError("masses must contain two finite positive bare masses")
    if not isinstance(num_radial_points, int) or num_radial_points < 6:
        raise ValueError("num_radial_points must be an integer of at least six")
    if not math.isfinite(r_max) or r_max <= 0:
        raise ValueError("r_max must be finite and positive")
    dx = r_max / num_radial_points
    if not math.isfinite(separation) or not 0 < separation < 2 * (r_max - 4 * dx):
        raise ValueError("separation must place both punctures inside the four-cell boundary margin")
    if not math.isfinite(momentum) or momentum < 0:
        raise ValueError("momentum must be a finite nonnegative inward magnitude")
    for name, value in (("newton_tolerance", newton_tolerance), ("cg_tolerance", cg_tolerance)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not isinstance(max_newton_iterations, int) or max_newton_iterations < 0:
        raise ValueError("max_newton_iterations must be a nonnegative integer")
    if not isinstance(max_cg_iterations, int) or max_cg_iterations < 1:
        raise ValueError("max_cg_iterations must be a positive integer")

    positions = jnp.asarray(((0.0, 0.0, -separation / 2), (0.0, 0.0, separation / 2)))
    momenta = jnp.asarray(((0.0, 0.0, momentum), (0.0, 0.0, -momentum)))
    grid, dx = cell_centered_grid(2 * num_radial_points, r_max)
    _, u, residual_history = solve_conformal_factor(
        masses, positions, momenta, grid, dx,
        newton_tolerance=newton_tolerance,
        max_newton_iterations=max_newton_iterations,
        cg_tolerance=cg_tolerance,
        max_cg_iterations=max_cg_iterations,
        verbose=verbose,
    )
    if not math.isfinite(residual_history[-1]) or residual_history[-1] > newton_tolerance:
        raise RuntimeError("Bowen–York solve returned an invalid final residual")
    u_plane = lagrange6_nonperiodic(u, jnp.asarray([num_radial_points - 0.5]), axis=1)
    plane = grid[:, :1, :, :].at[..., 1].set(0.0)
    psi = brill_lindquist_conformal_factor(masses, positions, plane) + u_plane
    if not bool(jnp.all(jnp.isfinite(psi) & (psi > 0))):
        raise RuntimeError("Interpolated conformal factor must be finite and positive")
    curvature = bowen_york_extrinsic_curvature(momenta, positions, plane)
    W = psi**-2
    metric = jnp.broadcast_to(jnp.eye(3)[:, :, None, None, None], (3, 3) + W.shape)
    state = compact_axisymmetric_state(BSSNVariables(
        conformal_metric=metric,
        conformal_factor=W,
        traceless_K=jnp.moveaxis(curvature, (-2, -1), (0, 1)) * psi**-6,
        trace_K=jnp.zeros_like(W),
        conformal_connection=jnp.zeros((3,) + W.shape),
        lapse=W,
        shift=jnp.zeros((3,) + W.shape),
    ))
    if not all(bool(jnp.all(jnp.isfinite(field))) for field in state):
        raise RuntimeError("Bowen–York initial state contains nonfinite fields")
    return state, residual_history


def run_axisymmetric_collision(
    masses=MASSES,
    separation=SEPARATION,
    momentum=MOMENTUM,
    num_radial_points=NUM_RADIAL_POINTS,
    r_max=R_MAX,
    cfl=CFL,
    final_time=FINAL_TIME,
    snapshot_count=SNAPSHOT_COUNT,
    newton_tolerance=NEWTON_TOLERANCE,
    max_newton_iterations=MAX_NEWTON_ITERATIONS,
    cg_tolerance=CG_TOLERANCE,
    max_cg_iterations=MAX_CG_ITERATIONS,
    kappa=KAPPA,
    eta=ETA,
    nu=NU,
    gamma_driver=GAMMA_DRIVER,
    output_dir=OUTPUT_DIR,
    overwrite=OVERWRITE,
    show_progress=SHOW_PROGRESS,
):
    """Run the collision and return the final compact state and run details."""
    if not math.isfinite(cfl) or cfl <= 0:
        raise ValueError("cfl must be finite and positive")
    if not math.isfinite(final_time) or final_time < 0:
        raise ValueError("final_time must be finite and nonnegative")
    if not isinstance(snapshot_count, int) or snapshot_count < 1:
        raise ValueError("snapshot_count must be a positive integer")
    if any(not math.isfinite(value) or value < 0 for value in (kappa, eta, nu, gamma_driver)):
        raise ValueError("gauge and damping coefficients must be finite and nonnegative")

    output_dir = Path(output_dir)
    openpmd_path = output_dir / "axisymmetric_blackhole_collision.h5"
    constraint_path = output_dir / "constraint_norms.txt"
    residual_path = output_dir / "newton_residual.txt"
    configuration_path = output_dir / "run_configuration.json"
    for path in (openpmd_path, constraint_path, residual_path, configuration_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {path}. Change OUTPUT_DIR or set OVERWRITE=True.")

    state, residual_history = bowen_york_plane_data(
        masses, separation, momentum, num_radial_points, r_max,
        newton_tolerance, max_newton_iterations, cg_tolerance, max_cg_iterations,
        verbose=show_progress,
    )
    dx = r_max / num_radial_points
    dt = cfl * dx
    num_steps = math.ceil(final_time / dt)
    output_interval = max(1, math.ceil(num_steps / snapshot_count))
    z_min = -r_max + 0.5 * dx
    params = BSSNParameters(
        eta=eta, kappa=kappa, nu=nu, g=gamma_driver,
        dx=dx, dt=dt, zero_shift=0, gauge=1,
        xl_bc=PERIODIC_BC, xr_bc=SOMMERFELD_BC,
        yl_bc=PERIODIC_BC, yr_bc=PERIODIC_BC,
        zl_bc=SOMMERFELD_BC, zr_bc=SOMMERFELD_BC,
        x_min=-3.5 * dx, y_min=-4 * dx, z_min=z_min, mad_q=1.0,
    )
    validate_axisymmetric_grid(state, params)
    advance = jax.jit(axisymmetric_rk4_step)
    fields_finite = jax.jit(lambda fields: jnp.all(jnp.stack([
        jnp.all(jnp.isfinite(field)) for field in fields
    ])))

    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        "masses": list(masses), "separation": separation, "momentum": momentum,
        "num_radial_points": num_radial_points, "num_z_points": 2 * num_radial_points,
        "r_max": r_max, "cfl": cfl, "final_time": final_time,
        "num_steps": num_steps, "snapshot_count": snapshot_count,
        "newton_tolerance": newton_tolerance, "max_newton_iterations": max_newton_iterations,
        "cg_tolerance": cg_tolerance, "max_cg_iterations": max_cg_iterations,
        "parameters": params._asdict(),
        "initial_solver": "Cartesian seven-point Laplacian; u=0 on all outer grid layers",
    }
    configuration_path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    with residual_path.open("w", encoding="utf-8") as stream:
        stream.write("# iteration hamiltonian_residual_rms\n")
        for iteration, residual in enumerate(residual_history):
            stream.write(f"{iteration} {residual:.16e}\n")

    with OpenPMDWriter(
        openpmd_path, grid_spacing=(dx, dx, dx),
        grid_global_offset=(z_min, 0.0, z_min),
        grid_position=(0.0, 0.0, 0.0), dt=dt, ghost_cells=0,
    ) as writer, constraint_path.open("w", encoding="utf-8") as constraint_file:
        progress = tqdm(range(num_steps + 1), desc="Evolving axisymmetric collision",
                        unit="state", disable=not show_progress)
        time = 0.0
        for step in progress:
            if step:
                next_time = min(step * dt, final_time)
                step_dt = next_time - time
                state = advance(state, params._replace(dt=step_dt))
                time = next_time
                writer.dt = step_dt
            if not bool(fields_finite(state)):
                raise RuntimeError(f"Nonfinite BSSN state at step {step}, t={time}")
            violations = compute_axisymmetric_constraints(state, params)
            norms = compute_axisymmetric_constraint_norms(violations, params=params)
            if not bool(fields_finite(violations)) or not all(math.isfinite(float(v)) for v in norms.values()):
                raise RuntimeError(f"Nonfinite constraints at step {step}, t={time}")
            if step == 0:
                constraint_file.write("# time " + " ".join(norms) + "\n")
            constraint_file.write(f"{time:.16e} " + " ".join(f"{float(v):.16e}" for v in norms.values()) + "\n")
            constraint_file.flush()
            if show_progress:
                progress.set_postfix(min_W=f"{float(jnp.min(state.conformal_factor)):.6e}",
                                     min_lapse=f"{float(jnp.min(state.lapse)):.6e}")
            if step % output_interval == 0 or step == num_steps:
                writer.write(axisymmetric_plane_output_fields(
                    state, violations.hamiltonian, violations.momentum,
                    det_gamma=violations.det_gamma, trace_A=violations.trace_A,
                    gamma_condition=violations.gamma_condition,
                ), step=step, time=time)

    print("Axisymmetric Cartoon Bowen–York collision demo")
    print(f"Nrho={num_radial_points}, Nz={2 * num_radial_points}, dx={dx:.8f}, "
          f"dt={dt:.8f}, steps={num_steps}, final time={time:.8f}")
    print(f"Final Newton residual: {residual_history[-1]:.6e}; final fields finite: True")
    return state, {
        "parameters": params, "num_steps": num_steps, "final_time": time,
        "openpmd_path": openpmd_path, "constraint_path": constraint_path,
        "residual_path": residual_path, "configuration_path": configuration_path,
        "residual_history": residual_history, "final_fields_finite": True,
    }


def main():
    run_axisymmetric_collision()


if __name__ == "__main__":
    main()
