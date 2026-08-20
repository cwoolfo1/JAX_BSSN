"""Evolve a plus-polarized linear wave through one fixed-refinement patch."""

import math
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from JAX_BSSN.boundaries import PERIODIC_BC
from JAX_BSSN.bssn import BSSNParameters, compute_momentum_constraint
from JAX_BSSN.diagnostics.openpmd import OpenPMDWriter
from JAX_BSSN.errors import compute_hamiltonian_constraint
from JAX_BSSN.initialization import create_coordinate_arrays, linear_wave_data
from JAX_BSSN.refinement import (
    FMRPatchSpec,
    fine_active_shape,
    fine_active_view,
    fine_coordinates,
    fmr_rk4_step,
)


def h_plus(variables):
    """Return the physical plus-polarized metric perturbation."""

    physical_metric_yy = (
        variables.conformal_metric[1, 1] / variables.conformal_factor**2
    )
    physical_metric_zz = (
        variables.conformal_metric[2, 2] / variables.conformal_factor**2
    )
    return 0.5 * (physical_metric_yy - physical_metric_zz)


def diagnostic_fields(variables, parameters):
    """Compute the wave and BSSN constraint fields for one grid level."""

    hamiltonian = compute_hamiltonian_constraint(variables, parameters)
    momentum = compute_momentum_constraint(variables, parameters)

    return {
        "h_plus": h_plus(variables),
        "lapse": variables.lapse,
        "shift": tuple(variables.shift[i] for i in range(3)),
        "K": variables.trace_K,
        "W": variables.conformal_factor,
        "hamiltonian_constraint": hamiltonian,
        "momentum_constraint": tuple(momentum[i] for i in range(3)),
    }


def _rms(field, mask=None):
    if mask is None:
        return jnp.sqrt(jnp.mean(field**2))

    return jnp.sqrt(
        jnp.sum(jnp.where(mask, field**2, 0.0)) / jnp.sum(mask)
    )


def _maximum_absolute(field, mask=None):
    absolute_field = jnp.abs(field)
    if mask is not None:
        absolute_field = jnp.where(mask, absolute_field, 0.0)
    return jnp.max(absolute_field)


def wave_error_diagnostics(
    coarse_fields,
    fine_fields,
    coarse_X,
    fine_X,
    coarse_uncovered,
    patch_spec,
    time,
    amplitude,
    wavelength,
):
    """Measure the numerical wave against the analytic traveling wave."""

    coarse_exact = amplitude * jnp.sin(
        2.0 * jnp.pi * (coarse_X - time) / wavelength
    )
    fine_exact = amplitude * jnp.sin(
        2.0 * jnp.pi * (fine_X - time) / wavelength
    )

    coarse_error = coarse_fields["h_plus"] - coarse_exact
    fine_error = fine_active_view(
        fine_fields["h_plus"] - fine_exact,
        patch_spec,
    )

    hamiltonian_coarse = coarse_fields["hamiltonian_constraint"]
    momentum_coarse = jnp.stack(coarse_fields["momentum_constraint"])
    hamiltonian_fine = fine_active_view(
        fine_fields["hamiltonian_constraint"],
        patch_spec,
    )
    momentum_fine = fine_active_view(
        jnp.stack(fine_fields["momentum_constraint"]),
        patch_spec,
    )

    return {
        "coarse_rms": float(_rms(coarse_error, coarse_uncovered)),
        "coarse_max": float(
            _maximum_absolute(coarse_error, coarse_uncovered)
        ),
        "fine_rms": float(_rms(fine_error)),
        "fine_max": float(_maximum_absolute(fine_error)),
        "coarse_hamiltonian_l2": float(
            _rms(hamiltonian_coarse, coarse_uncovered)
        ),
        "fine_hamiltonian_l2": float(_rms(hamiltonian_fine)),
        "coarse_momentum_l2": float(
            _rms(jnp.sqrt(jnp.sum(momentum_coarse**2, axis=0)), coarse_uncovered)
        ),
        "fine_momentum_l2": float(
            _rms(jnp.sqrt(jnp.sum(momentum_fine**2, axis=0)))
        ),
    }


def _print_diagnostics(step, time, diagnostics, amplitude):
    normalization = abs(amplitude)
    print(
        f"step={step:4d} time={time:.8f} "
        f"coarse h+ L2/A={diagnostics['coarse_rms'] / normalization:.6e} "
        f"max/A={diagnostics['coarse_max'] / normalization:.6e} "
        f"fine h+ L2/A={diagnostics['fine_rms'] / normalization:.6e} "
        f"max/A={diagnostics['fine_max'] / normalization:.6e}"
    )
    print(
        " " * 24
        + f"coarse ||H||2={diagnostics['coarse_hamiltonian_l2']:.6e} "
        f"||M||2={diagnostics['coarse_momentum_l2']:.6e} "
        f"fine ||H||2={diagnostics['fine_hamiltonian_l2']:.6e} "
        f"||M||2={diagnostics['fine_momentum_l2']:.6e}"
    )


def run_linear_wave(
    grid_size=32,
    amplitude=1.0e-8,
    wavelength=1.0,
    cfl=0.25,
    final_time=None,
    output_iterations=17,
    output_path=None,
    show_progress=True,
):
    """Run the stage-synchronous two-level linear-wave FMR demonstration."""

    if grid_size % 4 != 0:
        raise ValueError("the centered FMR patch requires grid_size divisible by 4")
    if amplitude == 0.0:
        raise ValueError("the normalized wave-error diagnostic requires nonzero amplitude")

    if final_time is None:
        final_time = wavelength
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "output" / "linear_wave_fmr.h5"
    else:
        output_path = Path(output_path)

    dx_coarse = wavelength / grid_size
    dx_fine = dx_coarse / 2.0
    num_steps = math.ceil(final_time / (cfl * dx_fine))
    dt = final_time / num_steps
    if output_iterations < 2:
        raise ValueError("output_iterations must include initial and final states")
    if output_iterations > num_steps + 1:
        raise ValueError("output_iterations cannot exceed num_steps + 1")

    first_coarse_point = -(grid_size - 1) * dx_coarse / 2.0
    coarse_origin = (first_coarse_point,) * 3
    coarse_X, _, _ = create_coordinate_arrays(
        grid_size, grid_size, grid_size, dx_coarse
    )

    patch_lo = (grid_size // 4,) * 3
    patch_hi = (3 * grid_size // 4 - 1,) * 3
    patch_spec = FMRPatchSpec(patch_lo, patch_hi)
    fine_X, fine_Y, fine_Z = fine_coordinates(
        patch_spec, dx_coarse, coarse_origin
    )
    fine_shape = tuple(
        size + 2 * patch_spec.ghost_width
        for size in fine_active_shape(patch_spec)
    )

    coarse_variables = linear_wave_data(
        grid_size,
        grid_size,
        grid_size,
        dx_coarse,
        amplitude=amplitude,
        wavelength=wavelength,
    )
    fine_variables = linear_wave_data(
        *fine_shape,
        dx_fine,
        amplitude=amplitude,
        wavelength=wavelength,
    )

    coarse_parameters = BSSNParameters(
        eta=0.0,
        kappa=0.0,
        nu=0.0,
        g=0.0,
        dx=dx_coarse,
        dt=dt,
        zero_shift=1,
        gauge=0,
        xl_bc=PERIODIC_BC,
        xr_bc=PERIODIC_BC,
        yl_bc=PERIODIC_BC,
        yr_bc=PERIODIC_BC,
        zl_bc=PERIODIC_BC,
        zr_bc=PERIODIC_BC,
        x_min=first_coarse_point,
        y_min=first_coarse_point,
        z_min=first_coarse_point,
    )
    fine_parameters = coarse_parameters._replace(
        dx=dx_fine,
        x_min=float(fine_X[0, 0, 0]),
        y_min=float(fine_Y[0, 0, 0]),
        z_min=float(fine_Z[0, 0, 0]),
    )

    # The complete production FMR step is compiled as one pure state transform.
    advance = jax.jit(
        lambda coarse, fine: fmr_rk4_step(
            coarse,
            fine,
            coarse_parameters,
            fine_parameters,
            patch_spec,
        )
    )

    coarse_uncovered = np.ones((grid_size,) * 3, dtype=bool)
    covered = tuple(
        slice(lower, upper + 1)
        for lower, upper in zip(patch_spec.coarse_lo, patch_spec.coarse_hi)
    )
    coarse_uncovered[covered] = False
    coarse_uncovered = jnp.asarray(coarse_uncovered)

    output_steps = tuple(
        int(step)
        for step in np.rint(
            np.linspace(0, num_steps, output_iterations)
        ).astype(int)
    )
    output_step_set = set(output_steps)

    active_start = tuple(
        first_coarse_point + index * dx_coarse for index in patch_spec.coarse_lo
    )
    writer = OpenPMDWriter(
        output_path,
        grid_spacing=(dx_coarse,) * 3,
        grid_global_offset=coarse_origin,
        dt=dt,
    )

    def write_iteration(step, time):
        jax.block_until_ready((coarse_variables, fine_variables))
        coarse_fields = diagnostic_fields(coarse_variables, coarse_parameters)
        fine_fields = diagnostic_fields(fine_variables, fine_parameters)
        diagnostics = wave_error_diagnostics(
            coarse_fields,
            fine_fields,
            coarse_X,
            fine_X,
            coarse_uncovered,
            patch_spec,
            time,
            amplitude,
            wavelength,
        )
        writer.write_levels(
            {
                "level_0": {
                    "fields": coarse_fields,
                    "grid_spacing": (dx_coarse,) * 3,
                    "grid_global_offset": coarse_origin,
                },
                "level_1": {
                    "fields": fine_fields,
                    "grid_spacing": (dx_fine,) * 3,
                    "grid_global_offset": active_start,
                    "ghost_cells": patch_spec.ghost_width,
                },
            },
            step=step,
            time=time,
        )
        _print_diagnostics(step, time, diagnostics, amplitude)
        return diagnostics

    print("Linear-wave fixed-mesh-refinement demo")
    print(
        f"coarse grid={grid_size}^3 dx={dx_coarse:.8f}, "
        f"fine active grid={fine_active_shape(patch_spec)} dx={dx_fine:.8f}"
    )
    print(
        f"patch coarse bounds={patch_spec.coarse_lo}:{patch_spec.coarse_hi}, "
        f"dt={dt:.8f}, steps={num_steps}, final time={final_time:.8f}"
    )
    print(f"openPMD output={output_path} ({output_iterations} iterations)")

    try:
        final_diagnostics = write_iteration(step=0, time=0.0)
        progress = tqdm(
            range(1, num_steps + 1),
            desc="Evolving linear wave",
            unit="step",
            disable=not show_progress,
        )
        for step in progress:
            coarse_variables, fine_variables = advance(
                coarse_variables, fine_variables
            )

            if step in output_step_set:
                time = step * dt
                final_diagnostics = write_iteration(step, time)
                progress.set_postfix(
                    coarse=f"{final_diagnostics['coarse_rms'] / abs(amplitude):.3e}",
                    fine=f"{final_diagnostics['fine_rms'] / abs(amplitude):.3e}",
                )
    finally:
        writer.close()

    normalized_coarse_error = final_diagnostics["coarse_rms"] / abs(amplitude)
    normalized_fine_error = final_diagnostics["fine_rms"] / abs(amplitude)
    if normalized_coarse_error >= 0.05 or normalized_fine_error >= 0.05:
        raise RuntimeError(
            "one-period h_plus RMS error exceeded 5% of the wave amplitude: "
            f"coarse={normalized_coarse_error:.6e}, "
            f"fine={normalized_fine_error:.6e}"
        )

    return coarse_variables, fine_variables, {
        "output_path": output_path,
        "num_steps": num_steps,
        "output_steps": output_steps,
        "final_time": num_steps * dt,
        "patch_spec": patch_spec,
        "diagnostics": final_diagnostics,
    }


def main():
    run_linear_wave()


if __name__ == "__main__":
    main()
