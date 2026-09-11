"""Local Schwarzschild initial data and fixed-duration reference evolution."""

import json
import os
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.cartoon.axisymmetry import (
    axisymmetric_plane_output_fields,
    axisymmetric_rk4_step,
    compact_axisymmetric_state,
    compute_axisymmetric_constraint_norms,
    compute_axisymmetric_constraints,
    validate_axisymmetric_grid,
)
from JAX_BSSN.diagnostics.openpmd import OpenPMDWriter
from collapse_io import atomic_write_json, parameters_to_dict, state_is_finite


def _write_schwarzschild_checkpoint(
    path,
    state,
    step,
    time,
    mass,
    params,
):
    """Atomically checkpoint a vacuum Schwarzschild control evolution."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"bssn_{name}": np.asarray(jax.device_get(field))
        for name, field in zip(BSSNVariables._fields, state)
    }
    metadata = {
        "format_version": 1,
        "kind": "schwarzschild_reference",
        "step": int(step),
        "time": float(time),
        "mass": float(mass),
        "parameters": parameters_to_dict(params),
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        np.savez(temporary_file, **arrays)
        temporary_path = Path(temporary_file.name)
    os.replace(temporary_path, path)


def _load_schwarzschild_checkpoint(path):
    with np.load(path, allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint["metadata_json"]))
        state = BSSNVariables(
            *(
                jnp.asarray(checkpoint[f"bssn_{name}"])
                for name in BSSNVariables._fields
            )
        )
    return state, metadata


def _schwarzschild_state(mass, params, num_radial_points, num_z_points):
    dx = float(params.dx)
    x = (
        np.arange(2 * num_radial_points) - (2 * num_radial_points - 1) / 2.0
    ) * dx
    z = float(params.z_min) + np.arange(num_z_points) * dx
    radius = jnp.sqrt(jnp.asarray(x[:, None] ** 2 + z[None, :] ** 2))[:, None, :]
    W = (1.0 + mass / (2.0 * radius)) ** -2
    metric = jnp.eye(3, dtype=W.dtype)[:, :, None, None, None] * jnp.ones(
        (3, 3) + W.shape, dtype=W.dtype
    )
    signed = BSSNVariables(
        conformal_metric=metric,
        conformal_factor=W,
        traceless_K=jnp.zeros_like(metric),
        trace_K=jnp.zeros_like(W),
        conformal_connection=jnp.zeros((3,) + W.shape, dtype=W.dtype),
        lapse=W,
        shift=jnp.zeros((3,) + W.shape, dtype=W.dtype),
    )
    return compact_axisymmetric_state(signed)


def run_schwarzschild_reference(
    mass,
    params,
    num_radial_points,
    num_z_points,
    final_time,
    output_dir,
    diagnostic_interval=0.5,
    snapshot_interval=1.0,
    show_progress=False,
):
    """Evolve a specified-mass puncture to a fixed final time."""

    from tqdm import tqdm

    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("the puncture mass must be finite and positive")

    output_dir = Path(output_dir)
    summary_path = output_dir / "run_summary.json"
    constraint_path = output_dir / "constraint_norms.txt"
    checkpoint_path = output_dir / "rolling_checkpoint.npz"
    output_dir.mkdir(parents=True, exist_ok=True)

    previous_summary = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as summary_file:
            previous_summary = json.load(summary_file)

    restarting = checkpoint_path.exists()
    if restarting:
        state, checkpoint_metadata = _load_schwarzschild_checkpoint(
            checkpoint_path
        )
        if not np.isclose(float(checkpoint_metadata["mass"]), float(mass)):
            raise ValueError("Schwarzschild checkpoint mass does not match the run")
        params = BSSNParameters(**checkpoint_metadata["parameters"])
        start_step = int(checkpoint_metadata["step"])
    else:
        protected_paths = (
            summary_path,
            constraint_path,
            output_dir / "schwarzschild_reference.h5",
        )
        if any(path.exists() for path in protected_paths):
            raise FileExistsError(
                f"cannot resume incomplete Schwarzschild data in {output_dir} "
                "without rolling_checkpoint.npz"
            )
        state = _schwarzschild_state(
            mass, params, num_radial_points, num_z_points
        )
        start_step = 0

    validate_axisymmetric_grid(state, params)
    advance = jax.jit(lambda current: axisymmetric_rk4_step(current, params))
    dt = float(params.dt)
    num_steps = int(np.floor(final_time / dt + 1.0e-12))
    if start_step > num_steps:
        raise ValueError(
            "Schwarzschild checkpoint is later than the final-time ceiling"
        )
    diagnostic_steps = max(1, int(round(diagnostic_interval / dt)))
    snapshot_steps = max(1, int(round(snapshot_interval / dt)))
    checkpoint_steps = diagnostic_steps

    if start_step == 0 and not (output_dir / "schwarzschild_reference.h5").exists():
        openpmd_path = output_dir / "schwarzschild_reference.h5"
    else:
        segment = 0
        openpmd_path = output_dir / (
            f"schwarzschild_reference_restart_{start_step:08d}_{segment:02d}.h5"
        )
        while openpmd_path.exists():
            segment += 1
            openpmd_path = output_dir / (
                f"schwarzschild_reference_restart_{start_step:08d}_{segment:02d}.h5"
            )

    status = "complete"
    final_step = start_step

    _write_schwarzschild_checkpoint(
        checkpoint_path,
        state,
        start_step,
        start_step * dt,
        mass,
        params,
    )

    writer = OpenPMDWriter(
        openpmd_path,
        grid_spacing=(params.dx, params.dx, params.dx),
        grid_global_offset=(
            -(num_radial_points - 0.5) * params.dx,
            0.0,
            params.z_min,
        ),
        grid_position=(0.0, 0.0, 0.0),
        dt=params.dt,
        ghost_cells=0,
    )

    def diagnose(step, constraint_file, write_fields):
        time = step * dt
        finite = state_is_finite(state)
        violations = compute_axisymmetric_constraints(state, params)
        norms = compute_axisymmetric_constraint_norms(violations)
        jax.block_until_ready((violations, norms))
        constraint_file.write(
            f"{step:d} {time:.16e} "
            f"{float(norms['hamiltonian_l2']):.16e} "
            f"{float(norms['momentum_l2']):.16e}\n"
        )
        constraint_file.flush()
        if write_fields or not finite:
            writer.write(
                axisymmetric_plane_output_fields(
                    state,
                    violations.hamiltonian,
                    violations.momentum,
                    violations.det_gamma,
                    violations.trace_A,
                    violations.gamma_condition,
                ),
                step,
                time,
            )
        return finite

    constraint_has_header = (
        constraint_path.exists() and constraint_path.stat().st_size > 0
    )
    mode = "a" if restarting else "w"
    with writer, constraint_path.open(mode, encoding="utf-8") as constraint_file:
        if not constraint_has_header:
            constraint_file.write("# step time hamiltonian_l2 momentum_l2\n")
        progress = tqdm(
            range(start_step, num_steps + 1),
            disable=not show_progress,
            desc="Schwarzschild reference",
        )
        for step in progress:
            diagnostic_due = (
                step in (start_step, num_steps) or step % diagnostic_steps == 0
            )
            snapshot_due = (
                step in (start_step, num_steps) or step % snapshot_steps == 0
            )
            if diagnostic_due or snapshot_due:
                finite = diagnose(step, constraint_file, snapshot_due)
                final_step = step
                if not finite:
                    status = "failed_nonfinite"
                    break
            if step < num_steps:
                state = advance(state)
                final_step = step + 1
                if final_step % checkpoint_steps == 0:
                    jax.block_until_ready(state)
                    _write_schwarzschild_checkpoint(
                        checkpoint_path,
                        state,
                        final_step,
                        final_step * dt,
                        mass,
                        params,
                    )

    jax.block_until_ready(state)
    _write_schwarzschild_checkpoint(
        checkpoint_path,
        state,
        final_step,
        final_step * dt,
        mass,
        params,
    )
    output_segments = list(previous_summary.get("output_segments", []))
    output_segments.append(str(openpmd_path))
    summary = {
        "kind": "schwarzschild_reference",
        "status": status,
        "mass_parameter": float(mass),
        "final_step": int(final_step),
        "final_time": float(final_step * dt),
        "final_time_ceiling": float(final_time),
        "parameters": parameters_to_dict(params),
        "openpmd_path": str(openpmd_path),
        "output_segments": output_segments,
        "checkpoint_path": str(checkpoint_path),
    }
    atomic_write_json(summary_path, summary)
    return summary


__all__ = ["run_schwarzschild_reference"]
