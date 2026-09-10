"""Shared restart, horizon, settling, and Schwarzschild-collapse utilities."""

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
from JAX_BSSN.diagnostics.apparent_horizon import (
    AxisymmetricHorizon,
    find_axisymmetric_apparent_horizon,
)
from JAX_BSSN.diagnostics.openpmd import OpenPMDWriter
from JAX_BSSN.EM.first_order.variables import DensitizedMaxwellState
from JAX_BSSN.EM.second_order.variables import EMVariables
from JAX_BSSN.EM.variables import EinsteinMaxwellVariables


def _json_value(value):
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def parameters_to_dict(params: BSSNParameters):
    return {name: _json_value(value) for name, value in params._asdict().items()}


def atomic_write_json(path, data):
    """Replace a JSON file only after its complete new contents are durable."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        json.dump(data, temporary_file, indent=2, sort_keys=True)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    os.replace(temporary_path, path)


def write_collapse_checkpoint(
    path,
    state,
    u,
    residual_history,
    formulation,
    step,
    time,
    params,
    configuration,
):
    """Atomically write the complete formulation-specific evolution state."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"bssn_{name}": np.asarray(jax.device_get(field))
        for name, field in zip(BSSNVariables._fields, state.bssn)
    }
    arrays.update(
        {
            f"em_{name}": np.asarray(jax.device_get(field))
            for name, field in zip(state.em._fields, state.em)
        }
    )
    arrays["u"] = np.asarray(jax.device_get(u))
    arrays["residual_history"] = np.asarray(residual_history, dtype=float)
    metadata = {
        "format_version": 1,
        "formulation": formulation,
        "step": int(step),
        "time": float(time),
        "parameters": parameters_to_dict(params),
        "configuration": configuration,
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


def load_collapse_checkpoint(path, expected_formulation=None):
    """Load a checkpoint written by :func:`write_collapse_checkpoint`."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint["metadata_json"]))
        formulation = metadata["formulation"]
        if expected_formulation is not None and formulation != expected_formulation:
            raise ValueError(
                f"checkpoint formulation is {formulation!r}, not "
                f"{expected_formulation!r}"
            )

        bssn = BSSNVariables(
            *(jnp.asarray(checkpoint[f"bssn_{name}"]) for name in BSSNVariables._fields)
        )
        if formulation == "first_order":
            em_type = DensitizedMaxwellState
        elif formulation == "second_order":
            em_type = EMVariables
        else:
            raise ValueError(f"unsupported checkpoint formulation {formulation!r}")
        em = em_type(
            *(jnp.asarray(checkpoint[f"em_{name}"]) for name in em_type._fields)
        )
        state = EinsteinMaxwellVariables(bssn=bssn, em=em)
        u = jnp.asarray(checkpoint["u"])
        residual_history = checkpoint["residual_history"].astype(float).tolist()

    return state, u, residual_history, metadata


def _write_schwarzschild_checkpoint(
    path,
    state,
    step,
    time,
    mass,
    params,
    horizon_coefficients,
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
        "horizon_coefficients": (
            None
            if horizon_coefficients is None
            else np.asarray(horizon_coefficients).tolist()
        ),
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


def horizon_to_dict(horizon: AxisymmetricHorizon | None):
    if horizon is None:
        return None
    return {
        "coefficients": horizon.coefficients.tolist(),
        "expansion_l2": horizon.expansion_l2,
        "expansion_linf": horizon.expansion_linf,
        "area": horizon.area,
        "irreducible_mass": horizon.irreducible_mass,
        "circumference_ratio": horizon.circumference_ratio,
        "polar_radius": horizon.polar_radius,
        "equatorial_radius": horizon.equatorial_radius,
    }


def horizon_from_dict(data):
    if data is None:
        return None
    return AxisymmetricHorizon(
        coefficients=np.asarray(data["coefficients"]),
        expansion_l2=float(data["expansion_l2"]),
        expansion_linf=float(data["expansion_linf"]),
        area=float(data["area"]),
        irreducible_mass=float(data["irreducible_mass"]),
        circumference_ratio=float(data["circumference_ratio"]),
        polar_radius=float(data["polar_radius"]),
        equatorial_radius=float(data["equatorial_radius"]),
    )


def _positive_scalar(field):
    return np.asarray(jax.device_get(field[4:, 0, :]))


def _cylindrical_coordinates(shape, params):
    num_radial_points, num_z_points = shape
    dx = float(params.dx)
    rho = (np.arange(num_radial_points) + 0.5) * dx
    z = float(params.z_min) + np.arange(num_z_points) * dx
    return np.meshgrid(rho, z, indexing="ij")


def _horizon_radius(coefficients, cylindrical_radius, z):
    radius = np.sqrt(cylindrical_radius**2 + z**2)
    mu = np.divide(z, radius, out=np.zeros_like(radius), where=radius > 0.0)
    full_coefficients = np.zeros(2 * len(coefficients) - 1)
    full_coefficients[::2] = coefficients
    return np.polynomial.legendre.legval(mu, full_coefficients)


def exterior_electromagnetic_energy(
    energy_density,
    bssn,
    horizon: AxisymmetricHorizon,
    params,
):
    """Integrate Eulerian EM energy outside two cells beyond the horizon."""

    rho_em = _positive_scalar(energy_density)
    W = np.maximum(_positive_scalar(bssn.conformal_factor), 1.0e-12)
    conformal_metric = np.asarray(
        jax.device_get(bssn.conformal_metric[:, :, 4:, 0, :])
    )
    conformal_metric = np.moveaxis(conformal_metric, (0, 1), (-2, -1))
    sqrt_conformal_determinant = np.sqrt(
        np.maximum(np.linalg.det(conformal_metric), 0.0)
    )
    RHO, Z = _cylindrical_coordinates(rho_em.shape, params)
    radius = np.sqrt(RHO**2 + Z**2)
    horizon_radius = _horizon_radius(horizon.coefficients, RHO, Z)
    exterior = radius > horizon_radius + 2.0 * float(params.dx)
    proper_volume = (
        2.0
        * np.pi
        * RHO
        * float(params.dx) ** 2
        * sqrt_conformal_determinant
        / W**3
    )
    return float(np.sum(np.where(exterior, rho_em * proper_volume, 0.0)))


def relative_exterior_field_changes(older, newer, horizon, params):
    """Return cylindrical relative L2 changes in lapse and W."""

    lapse_old, W_old = older
    lapse_new, W_new = newer
    RHO, Z = _cylindrical_coordinates(lapse_new.shape, params)
    radius = np.sqrt(RHO**2 + Z**2)
    horizon_radius = _horizon_radius(horizon.coefficients, RHO, Z)
    domain_radius = min(
        lapse_new.shape[0] * float(params.dx),
        max(
            abs(float(params.z_min)),
            abs(
                float(params.z_min)
                + (lapse_new.shape[1] - 1) * float(params.dx)
            ),
        ),
    )
    mask = (
        (radius > horizon_radius + 2.0 * float(params.dx))
        & (radius <= 0.65 * domain_radius)
    )
    weight = RHO * mask

    def relative(old, new):
        numerator = np.sqrt(np.sum(weight * (new - old) ** 2))
        denominator = np.sqrt(np.sum(weight * old**2))
        return float(numerator / max(denominator, np.finfo(float).tiny))

    return relative(lapse_old, lapse_new), relative(W_old, W_new)


def settling_criteria(horizon_samples, persistence_start_time, params):
    """Evaluate the agreed persistent-horizon and five-mass settling tests."""

    result = {
        "persistent_horizon": False,
        "mass_fractional_range": None,
        "maximum_circumference_distortion": None,
        "maximum_exterior_em_energy_fraction": None,
        "lapse_relative_l2_change": None,
        "W_relative_l2_change": None,
        "settled": False,
    }
    if not horizon_samples:
        return result

    latest = horizon_samples[-1]
    mass = latest["horizon"].irreducible_mass
    result["persistent_horizon"] = (
        latest["time"] - persistence_start_time >= 20.0 * mass
    )
    window_start = latest["time"] - 5.0 * mass
    window = [sample for sample in horizon_samples if sample["time"] >= window_start]
    if not window or window[0]["time"] > window_start + 1.1 * latest["diagnostic_interval"]:
        return result

    masses = np.asarray(
        [sample["horizon"].irreducible_mass for sample in window]
    )
    result["mass_fractional_range"] = float(
        (np.max(masses) - np.min(masses)) / np.mean(masses)
    )
    result["maximum_circumference_distortion"] = float(
        max(abs(sample["horizon"].circumference_ratio - 1.0) for sample in window)
    )
    result["maximum_exterior_em_energy_fraction"] = float(
        max(sample["exterior_em_energy"] for sample in window) / mass
    )
    lapse_change, W_change = relative_exterior_field_changes(
        window[0]["fields"],
        window[-1]["fields"],
        latest["horizon"],
        params,
    )
    result["lapse_relative_l2_change"] = lapse_change
    result["W_relative_l2_change"] = W_change
    result["settled"] = bool(
        result["persistent_horizon"]
        and result["mass_fractional_range"] < 1.0e-3
        and result["maximum_circumference_distortion"] < 1.0e-2
        and result["maximum_exterior_em_energy_fraction"] < 1.0e-4
        and lapse_change < 1.0e-3
        and W_change < 1.0e-3
    )
    return result


def compact_lapse_and_W(bssn):
    return (
        _positive_scalar(bssn.lapse).copy(),
        _positive_scalar(bssn.conformal_factor).copy(),
    )


def state_is_finite(state):
    return all(
        np.all(np.isfinite(np.asarray(jax.device_get(field))))
        for field in jax.tree_util.tree_leaves(state)
    )


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
    """Evolve a mass-matched puncture to a gauge-matched stationary reference."""

    from tqdm import tqdm

    output_dir = Path(output_dir)
    summary_path = output_dir / "run_summary.json"
    horizon_path = output_dir / "horizon_diagnostics.txt"
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
        previous_coefficients = checkpoint_metadata["horizon_coefficients"]
        if previous_coefficients is not None:
            previous_coefficients = np.asarray(previous_coefficients)
    else:
        protected_paths = (
            summary_path,
            horizon_path,
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
        previous_coefficients = np.zeros(4)
        previous_coefficients[0] = mass / 2.0

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

    samples = []
    written_steps = set()
    settled = False
    final_step = start_step

    _write_schwarzschild_checkpoint(
        checkpoint_path,
        state,
        start_step,
        start_step * dt,
        mass,
        params,
        previous_coefficients,
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

    def diagnose(step, horizon_file, constraint_file, write_fields):
        nonlocal previous_coefficients
        time = step * dt
        horizon = find_axisymmetric_apparent_horizon(
            state, params, previous_coefficients
        )
        if horizon is None:
            samples.clear()
            horizon_file.write(f"{step:d} {time:.16e} nan nan nan nan\n")
            horizon_file.flush()
        else:
            previous_coefficients = horizon.coefficients
            horizon_file.write(
                f"{step:d} {time:.16e} {horizon.irreducible_mass:.16e} "
                f"{horizon.area:.16e} {horizon.expansion_linf:.16e} "
                f"{horizon.circumference_ratio:.16e}\n"
            )
            horizon_file.flush()
            samples.append(
                {
                    "time": time,
                    "horizon": horizon,
                    "fields": compact_lapse_and_W(state),
                }
            )
        violations = compute_axisymmetric_constraints(state, params)
        norms = compute_axisymmetric_constraint_norms(violations)
        jax.block_until_ready((violations, norms))
        constraint_file.write(
            f"{step:d} {time:.16e} "
            f"{float(norms['hamiltonian_l2']):.16e} "
            f"{float(norms['momentum_l2']):.16e}\n"
        )
        constraint_file.flush()
        if write_fields:
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
            written_steps.add(step)

    horizon_has_header = horizon_path.exists() and horizon_path.stat().st_size > 0
    constraint_has_header = (
        constraint_path.exists() and constraint_path.stat().st_size > 0
    )
    mode = "a" if restarting else "w"
    with writer, horizon_path.open(
        mode, encoding="utf-8"
    ) as horizon_file, constraint_path.open(
        mode, encoding="utf-8"
    ) as constraint_file:
        if not horizon_has_header:
            horizon_file.write(
                "# step time irreducible_mass area expansion_linf "
                "circumference_ratio\n"
            )
        if not constraint_has_header:
            constraint_file.write("# step time hamiltonian_l2 momentum_l2\n")
        progress = tqdm(
            range(start_step, num_steps + 1),
            disable=not show_progress,
            desc="Schwarzschild reference",
        )
        for step in progress:
            if step % diagnostic_steps == 0:
                diagnose(
                    step,
                    horizon_file,
                    constraint_file,
                    step % snapshot_steps == 0,
                )
                if len(samples) >= 2:
                    current = samples[-1]
                    window_start = (
                        current["time"]
                        - 5.0 * current["horizon"].irreducible_mass
                    )
                    window = [
                        sample
                        for sample in samples
                        if sample["time"] >= window_start
                    ]
                    if window[0]["time"] <= window_start + 1.1 * diagnostic_interval:
                        masses = np.asarray(
                            [sample["horizon"].irreducible_mass for sample in window]
                        )
                        lapse_change, W_change = relative_exterior_field_changes(
                            window[0]["fields"],
                            window[-1]["fields"],
                            current["horizon"],
                            params,
                        )
                        settled = bool(
                            (np.max(masses) - np.min(masses)) / np.mean(masses)
                            < 1.0e-3
                            and max(
                                abs(sample["horizon"].circumference_ratio - 1.0)
                                for sample in window
                            )
                            < 1.0e-2
                            and lapse_change < 1.0e-3
                            and W_change < 1.0e-3
                        )
                if settled:
                    final_step = step
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
                        previous_coefficients,
                    )

        if final_step not in written_steps:
            diagnose(final_step, horizon_file, constraint_file, True)

    final_horizon = samples[-1]["horizon"] if samples else None
    jax.block_until_ready(state)
    _write_schwarzschild_checkpoint(
        checkpoint_path,
        state,
        final_step,
        final_step * dt,
        mass,
        params,
        previous_coefficients,
    )
    output_segments = list(previous_summary.get("output_segments", []))
    output_segments.append(str(openpmd_path))
    summary = {
        "kind": "schwarzschild_reference",
        "status": "settled" if settled else "incomplete",
        "mass_parameter": float(mass),
        "final_step": int(final_step),
        "final_time": float(final_step * dt),
        "parameters": parameters_to_dict(params),
        "horizon": horizon_to_dict(final_horizon),
        "openpmd_path": str(openpmd_path),
        "output_segments": output_segments,
        "checkpoint_path": str(checkpoint_path),
    }
    atomic_write_json(summary_path, summary)
    return summary


__all__ = [
    "atomic_write_json",
    "compact_lapse_and_W",
    "exterior_electromagnetic_energy",
    "horizon_from_dict",
    "horizon_to_dict",
    "load_collapse_checkpoint",
    "parameters_to_dict",
    "run_schwarzschild_reference",
    "settling_criteria",
    "state_is_finite",
    "write_collapse_checkpoint",
]
