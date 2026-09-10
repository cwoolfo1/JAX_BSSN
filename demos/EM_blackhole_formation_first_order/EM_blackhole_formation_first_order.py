"""Evolve electromagnetic dipole data with the first-order Yee solver.

The default is the off-centered time-symmetric family of Baumgarte,
Gundlach, and Hilditch.  Its amplitude is a literature-informed
supercritical candidate.  The evolution diagnoses apparent-horizon formation
and exterior settling directly.
"""

import argparse
import math
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from JAX_BSSN.bssn.constraints import (
    ConstraintViolations,
    compute_all_constraints_with_matter,
)
from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.cartoon.axisymmetry import (
    axisymmetric_plane_output_fields,
    compact_axisymmetric_state,
    compute_axisymmetric_constraint_norms,
    reconstruct_axisymmetric_support,
    validate_axisymmetric_grid,
)
from JAX_BSSN.cartoon.axisymmetry.reconstruction import (
    _expand_axisymmetric_vector,
    _project_scalar,
    _project_vector,
)
from JAX_BSSN.diagnostics.openpmd import OpenPMDWriter
from JAX_BSSN.diagnostics.apparent_horizon import (
    find_axisymmetric_apparent_horizon,
)
from JAX_BSSN.EM.collapse import (
    atomic_write_json,
    compact_lapse_and_W,
    exterior_electromagnetic_energy,
    horizon_from_dict,
    horizon_to_dict,
    load_collapse_checkpoint,
    parameters_to_dict,
    run_schwarzschild_reference,
    settling_criteria,
    state_is_finite,
    write_collapse_checkpoint,
)
from JAX_BSSN.EM.first_order.cartoon.axisymmetry import (
    axisymmetric_densitized_constraint_divergences,
    axisymmetric_first_order_einstein_maxwell_step,
    initialize_axisymmetric_first_order_state,
    reconstruct_axisymmetric_densitized_support,
)
from JAX_BSSN.EM.first_order.energy_momentum import (
    compute_densitized_electromagnetic_energy_momentum,
)
from JAX_BSSN.EM.first_order.evolve import (
    common_densitized_fields,
    common_physical_fields,
)
from JAX_BSSN.EM.variables import EinsteinMaxwellVariables
from initial_data import (
    conformal_vector_to_physical_contravariant,
    contract_conformal_electromagnetic_fields,
    off_centered_toroidal_electric_seed,
    solve_electromagnetic_conformal_factor,
)
from JAX_BSSN.evolution.boundaries import PERIODIC_BC, SOMMERFELD_BC


AMPLITUDE = 0.08
WIDTH = 1.0
RADIAL_CENTER = 3.0
DOMAIN_HALF_WIDTH = 24.0
NUM_RADIAL_POINTS = 192
NUM_Z_POINTS = 384
CFL = 0.2
FINAL_TIME = 40.0
SNAPSHOT_COUNT = 40
DIAGNOSTIC_INTERVAL = 0.5
CHECKPOINT_INTERVAL = 1.0

NEWTON_TOLERANCE = 1.0e-10
MAX_NEWTON_ITERATIONS = 12
CG_TOLERANCE = 1.0e-10
MAX_CG_ITERATIONS = 1000

KAPPA = 0.002
ETA = 2.0
NU = 0.02
GAMMA_DRIVER = 0.75


def axisymmetric_parameters(
    num_radial_points: int,
    num_z_points: int,
    domain_half_width: float,
    dt: float,
) -> BSSNParameters:
    """Return the compact Cartoon parameters used by the formation demo."""

    dx = domain_half_width / num_radial_points
    dz = 2.0 * domain_half_width / num_z_points
    if not jnp.isclose(dx, dz):
        raise ValueError("axisymmetric Cartoon requires equal rho and z spacing")
    if num_z_points % 2:
        raise ValueError("num_z_points must be even")

    z_min = -(num_z_points - 1) * dx / 2.0
    return BSSNParameters(
        eta=ETA,
        kappa=KAPPA,
        nu=NU,
        g=GAMMA_DRIVER,
        dx=dx,
        dt=dt,
        zero_shift=0,
        gauge=1,
        xl_bc=PERIODIC_BC,
        xr_bc=SOMMERFELD_BC,
        yl_bc=PERIODIC_BC,
        yr_bc=PERIODIC_BC,
        zl_bc=SOMMERFELD_BC,
        zr_bc=SOMMERFELD_BC,
        x_min=-3.5 * dx,
        y_min=-4.0 * dx,
        z_min=z_min,
        mad_q=1.0,
    )


def electromagnetic_cylindrical_grid(
    num_radial_points: int,
    num_z_points: int,
    dx: float,
):
    """Return the positive-rho, full-z cell-centred elliptic grid."""

    rho = (jnp.arange(num_radial_points, dtype=jnp.float64) + 0.5) * dx
    z = (
        jnp.arange(num_z_points, dtype=jnp.float64)
        - (num_z_points - 1) / 2.0
    ) * dx
    RHO, Z = jnp.meshgrid(rho, z, indexing="ij")
    grid = jnp.stack((RHO, jnp.zeros_like(RHO), Z), axis=-1)
    return grid[:, None, :, :]


def _flat_conformal_bssn_plane(psi_plane):
    """Map a solved conformal factor to time-symmetric BSSN variables."""

    W = psi_plane**-2
    shape = W.shape
    conformal_metric = (
        jnp.eye(3, dtype=W.dtype)[:, :, None, None, None]
        * jnp.ones((3, 3) + shape, dtype=W.dtype)
    )
    return BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=W,
        traceless_K=jnp.zeros_like(conformal_metric),
        trace_K=jnp.zeros(shape, dtype=W.dtype),
        conformal_connection=jnp.zeros((3,) + shape, dtype=W.dtype),
        lapse=W,
        shift=jnp.zeros((3,) + shape, dtype=W.dtype),
    )


def constrained_einstein_maxwell_data(
    amplitude: float,
    width: float,
    radial_center: float,
    num_radial_points: int,
    num_z_points: int,
    dx: float,
    params: BSSNParameters,
    newton_tolerance: float = NEWTON_TOLERANCE,
    max_newton_iterations: int = MAX_NEWTON_ITERATIONS,
    cg_tolerance: float = CG_TOLERANCE,
    max_cg_iterations: int = MAX_CG_ITERATIONS,
    verbose: bool = False,
):
    """Solve the constraints and bootstrap compact Yee-grid EM/BSSN data."""

    grid = electromagnetic_cylindrical_grid(
        num_radial_points, num_z_points, dx
    )
    conformal_electric = off_centered_toroidal_electric_seed(
        grid,
        amplitude=amplitude,
        width=width,
        radial_center=radial_center,
    )
    conformal_magnetic = jnp.zeros_like(conformal_electric)
    conformal_field_squared = contract_conformal_electromagnetic_fields(
        conformal_electric, conformal_magnetic
    )[:, 0, :]
    psi, u, residual_history = solve_electromagnetic_conformal_factor(
        conformal_field_squared,
        dx,
        newton_tolerance=newton_tolerance,
        max_newton_iterations=max_newton_iterations,
        cg_tolerance=cg_tolerance,
        max_cg_iterations=max_cg_iterations,
        verbose=verbose,
    )

    psi_plane = psi[:, None, :]
    signed_psi_plane = jnp.concatenate(
        (jnp.flip(psi_plane, axis=0), psi_plane), axis=0
    )
    bssn = compact_axisymmetric_state(
        _flat_conformal_bssn_plane(signed_psi_plane)
    )

    physical_displacement_plane = conformal_vector_to_physical_contravariant(
        conformal_electric, psi_plane
    )
    physical_displacement = jnp.zeros(
        (3, num_radial_points + 4, 1, num_z_points),
        dtype=physical_displacement_plane.dtype,
    ).at[:, 4:, 0, :].set(
        physical_displacement_plane[:, :, 0, :]
    )
    physical_magnetic = jnp.zeros_like(physical_displacement)

    # The toroidal seed has only D^y on the y=0 reference plane.  Its native
    # Yee location is centered in rho and z, so the elliptic cell-center sample
    # is already located correctly.  The initializer fills parity ghosts,
    # densitizes the physical fields, and constructs the leapfrog history.
    state = initialize_axisymmetric_first_order_state(
        bssn,
        physical_displacement,
        physical_magnetic,
        params,
    )

    return state, u, residual_history


@jax.jit
def compute_matter_aware_axisymmetric_constraints(
    state: EinsteinMaxwellVariables,
    params: BSSNParameters,
) -> ConstraintViolations:
    """Return compact Einstein constraints including electromagnetic matter."""

    support_bssn = reconstruct_axisymmetric_support(state.bssn, params)
    support_em = reconstruct_axisymmetric_densitized_support(state.em, params)
    displacement, magnetic = common_densitized_fields(support_em)
    energy_density, momentum_density, _ = (
        compute_densitized_electromagnetic_energy_momentum(
            displacement,
            magnetic,
            support_bssn,
            params,
        )
    )
    support = compute_all_constraints_with_matter(
        support_bssn,
        params,
        energy_density,
        momentum_density,
    )
    return ConstraintViolations(
        hamiltonian=_project_scalar(support.hamiltonian),
        momentum=_project_vector(support.momentum),
        det_gamma=_project_scalar(support.det_gamma),
        trace_A=_project_scalar(support.trace_A),
        gamma_condition=_project_vector(support.gamma_condition),
    )


@jax.jit
def compute_axisymmetric_em_energy_density(
    state: EinsteinMaxwellVariables,
    params: BSSNParameters,
) -> jnp.ndarray:
    """Return compact rho_EM evaluated on reconstructed Cartesian support."""

    support_bssn = reconstruct_axisymmetric_support(state.bssn, params)
    support_em = reconstruct_axisymmetric_densitized_support(state.em, params)
    displacement, magnetic = common_densitized_fields(support_em)
    energy_density, _, _ = compute_densitized_electromagnetic_energy_momentum(
        displacement,
        magnetic,
        support_bssn,
        params,
    )
    return _project_scalar(energy_density)


def _axisymmetric_scalar_norms(field):
    """Return the standard interior cylindrical L2 and Linf norms."""

    physical = field[4:-4, 0, 4:-4]
    rho = jnp.arange(physical.shape[0], dtype=field.dtype) + 0.5
    normalization = jnp.sum(rho) * physical.shape[1]
    l2_norm = jnp.sqrt(jnp.sum(rho[:, None] * physical**2) / normalization)
    linf_norm = jnp.max(jnp.abs(physical))
    return l2_norm, linf_norm


def _output_fields(state, violations, params):
    fields = axisymmetric_plane_output_fields(
        state.bssn,
        violations.hamiltonian,
        violations.momentum,
        violations.det_gamma,
        violations.trace_A,
        violations.gamma_condition,
    )
    displacement, magnetic = common_physical_fields(state, params)
    displacement = _expand_axisymmetric_vector(displacement)
    magnetic = _expand_axisymmetric_vector(magnetic)
    fields.update(
        {
            "D": tuple(displacement[i] for i in range(3)),
            "B": tuple(magnetic[i] for i in range(3)),
        }
    )
    return fields


def run_em_blackhole_formation_first_order(
    amplitude: float = AMPLITUDE,
    width: float = WIDTH,
    radial_center: float = RADIAL_CENTER,
    domain_half_width: float = DOMAIN_HALF_WIDTH,
    num_radial_points: int = NUM_RADIAL_POINTS,
    num_z_points: int = NUM_Z_POINTS,
    cfl: float = CFL,
    final_time: float = FINAL_TIME,
    snapshot_count: int = SNAPSHOT_COUNT,
    output_dir: str | Path | None = None,
    newton_tolerance: float = NEWTON_TOLERANCE,
    max_newton_iterations: int = MAX_NEWTON_ITERATIONS,
    cg_tolerance: float = CG_TOLERANCE,
    max_cg_iterations: int = MAX_CG_ITERATIONS,
    show_progress: bool = True,
    restart: str | Path | None = None,
    checkpoint_interval: float = CHECKPOINT_INTERVAL,
    diagnostic_interval: float = DIAGNOSTIC_INTERVAL,
    run_schwarzschild: bool = True,
):
    """Evolve until exterior settling or the hard final-time ceiling."""

    restart_path = Path(restart) if restart is not None else None
    if output_dir is None:
        output_dir = (
            restart_path.parent
            if restart_path is not None
            else Path(__file__).resolve().parent / "output"
        )
    output_dir = Path(output_dir)
    residual_path = output_dir / "hamiltonian_residual.txt"
    constraint_path = output_dir / "constraint_norms.txt"
    horizon_path = output_dir / "horizon_diagnostics.txt"
    checkpoint_path = output_dir / "rolling_checkpoint.npz"
    summary_path = output_dir / "run_summary.json"

    requested_configuration = {
        "amplitude": float(amplitude),
        "width": float(width),
        "radial_center": float(radial_center),
        "domain_half_width": float(domain_half_width),
        "num_radial_points": int(num_radial_points),
        "num_z_points": int(num_z_points),
        "cfl": float(cfl),
        "diagnostic_interval": float(diagnostic_interval),
        "checkpoint_interval": float(checkpoint_interval),
    }
    previous_summary = {}
    if restart_path is None:
        protected_paths = (
            residual_path,
            constraint_path,
            horizon_path,
            checkpoint_path,
            summary_path,
            output_dir / "EM_blackhole_formation.h5",
        )
        if any(path.exists() for path in protected_paths):
            raise FileExistsError(
                f"refusing to overwrite an existing collapse campaign in {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        dx = domain_half_width / num_radial_points
        dt = cfl * dx
        params = axisymmetric_parameters(
            num_radial_points, num_z_points, domain_half_width, dt
        )
        state, u, residual_history = constrained_einstein_maxwell_data(
            amplitude,
            width,
            radial_center,
            num_radial_points,
            num_z_points,
            dx,
            params,
            newton_tolerance=newton_tolerance,
            max_newton_iterations=max_newton_iterations,
            cg_tolerance=cg_tolerance,
            max_cg_iterations=max_cg_iterations,
            verbose=show_progress,
        )
        start_step = 0
        runtime_configuration = requested_configuration.copy()
        with residual_path.open("w", encoding="utf-8") as residual_file:
            residual_file.write("# iteration interior_residual_rms\n")
            for iteration, residual in enumerate(residual_history):
                residual_file.write(f"{iteration:d} {residual:.16e}\n")
    else:
        state, u, residual_history, checkpoint_metadata = load_collapse_checkpoint(
            restart_path, "first_order"
        )
        runtime_configuration = checkpoint_metadata["configuration"]
        for name in (
            "amplitude",
            "width",
            "radial_center",
            "domain_half_width",
            "num_radial_points",
            "num_z_points",
            "cfl",
        ):
            requested_configuration[name] = runtime_configuration[name]
        num_radial_points = int(runtime_configuration["num_radial_points"])
        num_z_points = int(runtime_configuration["num_z_points"])
        params = BSSNParameters(**checkpoint_metadata["parameters"])
        start_step = int(checkpoint_metadata["step"])
        dx = float(params.dx)
        if summary_path.exists():
            import json

            with summary_path.open("r", encoding="utf-8") as summary_file:
                previous_summary = json.load(summary_file)

    validate_axisymmetric_grid(state.bssn, params)
    jax.block_until_ready(state)
    dt = float(params.dt)
    num_steps = int(math.floor(final_time / float(params.dt) + 1.0e-12))
    if start_step > num_steps:
        raise ValueError("restart checkpoint is later than the final-time ceiling")

    if restart_path is None:
        openpmd_path = output_dir / "EM_blackhole_formation.h5"
    else:
        segment = 0
        openpmd_path = output_dir / (
            f"EM_blackhole_formation_restart_{start_step:08d}_{segment:02d}.h5"
        )
        while openpmd_path.exists():
            segment += 1
            openpmd_path = output_dir / (
                f"EM_blackhole_formation_restart_{start_step:08d}_{segment:02d}.h5"
            )
    if openpmd_path.exists():
        raise FileExistsError(f"refusing to overwrite {openpmd_path}")

    norm_names = (
        "hamiltonian_l2",
        "hamiltonian_linf",
        "momentum_l2",
        "momentum_linf",
        "det_gamma_l2",
        "det_gamma_linf",
        "trace_A_l2",
        "trace_A_linf",
        "gamma_l2",
        "gamma_linf",
        "displacement_divergence_l2",
        "displacement_divergence_linf",
        "magnetic_divergence_l2",
        "magnetic_divergence_linf",
        "max_rho_EM",
        "min_lapse",
        "min_W",
    )
    snapshot_steps = max(1, math.ceil(max(num_steps, 1) / max(snapshot_count, 1)))
    diagnostic_steps = max(1, int(round(diagnostic_interval / float(params.dt))))
    checkpoint_steps = max(1, int(round(checkpoint_interval / float(params.dt))))
    advance = jax.jit(
        lambda current: axisymmetric_first_order_einstein_maxwell_step(
            current, params
        )
    )
    progress = tqdm(
        range(start_step, num_steps + 1),
        disable=not show_progress,
        desc="First-order Einstein-Maxwell",
    )

    writer = OpenPMDWriter(
        openpmd_path,
        grid_spacing=(dx, dx, dx),
        grid_global_offset=(
            -(num_radial_points - 0.5) * dx,
            0.0,
            params.z_min,
        ),
        grid_position=(0.0, 0.0, 0.0),
        dt=dt,
        ghost_cells=0,
    )
    mode = "a" if restart_path is not None else "w"
    previous_coefficients = runtime_configuration.get("last_horizon_coefficients")
    if previous_coefficients is not None:
        previous_coefficients = np.asarray(previous_coefficients)
    persistence_start_time = runtime_configuration.get(
        "persistent_horizon_start_time"
    )
    horizon_samples = []
    diagnostic_records = list(previous_summary.get("diagnostics", []))
    written_steps = set()
    previously_settled = previous_summary.get("status") == "settled"
    previous_horizon = horizon_from_dict(previous_summary.get("horizon"))
    final_horizon = previous_horizon
    final_criteria = settling_criteria([], 0.0, params)
    status = "settled" if previously_settled else "incomplete"
    final_step = start_step

    def checkpoint_configuration():
        configuration = requested_configuration.copy()
        configuration["last_horizon_coefficients"] = (
            None
            if previous_coefficients is None
            else np.asarray(previous_coefficients).tolist()
        )
        configuration["persistent_horizon_start_time"] = persistence_start_time
        return configuration

    def diagnose(step, norm_file, horizon_file, write_fields):
        nonlocal previous_coefficients
        nonlocal persistence_start_time
        nonlocal horizon_samples
        nonlocal final_horizon
        nonlocal final_criteria

        time = step * float(params.dt)
        finite = state_is_finite(state)
        horizon = None
        if finite:
            horizon = find_axisymmetric_apparent_horizon(
                state.bssn, params, previous_coefficients
            )
        rho_em = compute_axisymmetric_em_energy_density(state, params)
        exterior_energy = np.nan
        final_horizon = horizon
        if horizon is None:
            persistence_start_time = None
            horizon_samples = []
            final_criteria = settling_criteria([], 0.0, params)
        else:
            previous_coefficients = horizon.coefficients
            final_horizon = horizon
            if persistence_start_time is None:
                persistence_start_time = time
            exterior_energy = exterior_electromagnetic_energy(
                rho_em, state.bssn, horizon, params
            )
            horizon_samples.append(
                {
                    "time": time,
                    "diagnostic_interval": diagnostic_steps * float(params.dt),
                    "horizon": horizon,
                    "exterior_em_energy": exterior_energy,
                    "fields": compact_lapse_and_W(state.bssn),
                }
            )
            final_criteria = settling_criteria(
                horizon_samples, persistence_start_time, params
            )
            keep_after = time - 6.0 * horizon.irreducible_mass
            horizon_samples = [
                sample for sample in horizon_samples if sample["time"] >= keep_after
            ]

        horizon_values = horizon_to_dict(horizon)
        diagnostic_records.append(
            {
                "step": int(step),
                "time": time,
                "finite": finite,
                "horizon": horizon_values,
                "exterior_em_energy": (
                    None if not np.isfinite(exterior_energy) else float(exterior_energy)
                ),
            }
        )
        if horizon is None:
            horizon_file.write(f"{step:d} {time:.16e} 0 nan nan nan nan\n")
        else:
            horizon_file.write(
                f"{step:d} {time:.16e} 1 {horizon.irreducible_mass:.16e} "
                f"{horizon.area:.16e} {horizon.expansion_linf:.16e} "
                f"{horizon.circumference_ratio:.16e}\n"
            )
        horizon_file.flush()

        violations = compute_matter_aware_axisymmetric_constraints(state, params)
        norms = compute_axisymmetric_constraint_norms(violations)
        displacement_divergence, magnetic_divergence = (
            axisymmetric_densitized_constraint_divergences(state.em, params)
        )
        displacement_l2, displacement_linf = _axisymmetric_scalar_norms(
            displacement_divergence
        )
        magnetic_l2, magnetic_linf = _axisymmetric_scalar_norms(
            magnetic_divergence
        )
        physical = (slice(4, None), 0, slice(None))
        norms.update(
            {
                "displacement_divergence_l2": displacement_l2,
                "displacement_divergence_linf": displacement_linf,
                "magnetic_divergence_l2": magnetic_l2,
                "magnetic_divergence_linf": magnetic_linf,
                "max_rho_EM": jnp.max(rho_em[physical]),
                "min_lapse": jnp.min(state.bssn.lapse[physical]),
                "min_W": jnp.min(state.bssn.conformal_factor[physical]),
            }
        )
        jax.block_until_ready((violations, norms))
        values = " ".join(f"{float(norms[name]):.16e}" for name in norm_names)
        norm_file.write(f"{step:d} {time:.16e} {values}\n")
        norm_file.flush()
        if write_fields:
            writer.write(_output_fields(state, violations, params), step, time)
            written_steps.add(step)
        return finite

    with writer, constraint_path.open(mode, encoding="utf-8") as norm_file, horizon_path.open(
        mode, encoding="utf-8"
    ) as horizon_file:
        if restart_path is None:
            norm_file.write("# step time " + " ".join(norm_names) + "\n")
            horizon_file.write(
                "# step time found irreducible_mass area expansion_linf circumference_ratio\n"
            )
        for step in progress:
            diagnostic_due = step == start_step or step % diagnostic_steps == 0
            snapshot_due = step == start_step or step % snapshot_steps == 0
            if diagnostic_due or snapshot_due:
                finite = diagnose(step, norm_file, horizon_file, snapshot_due)
                final_step = step
                if not finite:
                    status = "failed_nonfinite"
                    break
                if previously_settled:
                    status = "settled"
                    break
                if final_criteria["settled"]:
                    status = "settled"
                    break
            if step < num_steps:
                state = advance(state)
                final_step = step + 1
                if final_step % checkpoint_steps == 0:
                    jax.block_until_ready(state)
                    write_collapse_checkpoint(
                        checkpoint_path,
                        state,
                        u,
                        residual_history,
                        "first_order",
                        final_step,
                        final_step * float(params.dt),
                        params,
                        checkpoint_configuration(),
                    )

        if final_step not in written_steps:
            diagnose(final_step, norm_file, horizon_file, True)

    if previously_settled and status != "failed_nonfinite":
        status = "settled"
        final_criteria = previous_summary["settling_criteria"]
        if final_horizon is None:
            final_horizon = previous_horizon
    elif status == "incomplete" and final_criteria["settled"]:
        status = "settled"

    jax.block_until_ready(state)
    write_collapse_checkpoint(
        checkpoint_path,
        state,
        u,
        residual_history,
        "first_order",
        final_step,
        final_step * float(params.dt),
        params,
        checkpoint_configuration(),
    )
    output_segments = list(previous_summary.get("output_segments", []))
    output_segments.append(str(openpmd_path))
    summary = {
        "formulation": "first_order",
        "status": status,
        "final_step": int(final_step),
        "final_time": float(final_step * float(params.dt)),
        "final_time_ceiling": float(final_time),
        "configuration": requested_configuration,
        "parameters": parameters_to_dict(params),
        "initial_hamiltonian_residual_history": residual_history,
        "horizon": horizon_to_dict(final_horizon),
        "settling_criteria": final_criteria,
        "diagnostics": diagnostic_records,
        "checkpoint_path": str(checkpoint_path),
        "output_segments": output_segments,
        "schwarzschild_reference": None,
    }
    atomic_write_json(summary_path, summary)

    if status == "settled" and run_schwarzschild and final_horizon is not None:
        reference_dir = output_dir / "schwarzschild_reference"
        reference_summary_path = reference_dir / "run_summary.json"
        if reference_summary_path.exists():
            import json

            with reference_summary_path.open("r", encoding="utf-8") as reference_file:
                reference_summary = json.load(reference_file)
        if (
            not reference_summary_path.exists()
            or reference_summary.get("status") != "settled"
        ):
            reference_summary = run_schwarzschild_reference(
                final_horizon.irreducible_mass,
                params,
                num_radial_points,
                num_z_points,
                final_time,
                reference_dir,
                diagnostic_interval=diagnostic_interval,
                snapshot_interval=max(final_time / max(snapshot_count, 1), float(params.dt)),
                show_progress=show_progress,
            )
        summary["schwarzschild_reference"] = reference_summary
        atomic_write_json(summary_path, summary)

    return state, u, residual_history


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amplitude", type=float, default=AMPLITUDE)
    parser.add_argument("--width", type=float, default=WIDTH)
    parser.add_argument("--radial-center", type=float, default=RADIAL_CENTER)
    parser.add_argument(
        "--domain-half-width", type=float, default=DOMAIN_HALF_WIDTH
    )
    parser.add_argument("--num-rho", type=int, default=NUM_RADIAL_POINTS)
    parser.add_argument("--num-z", type=int, default=NUM_Z_POINTS)
    parser.add_argument("--cfl", type=float, default=CFL)
    parser.add_argument("--final-time", type=float, default=FINAL_TIME)
    parser.add_argument("--snapshots", type=int, default=SNAPSHOT_COUNT)
    parser.add_argument("--output-dir")
    parser.add_argument("--newton-tolerance", type=float, default=NEWTON_TOLERANCE)
    parser.add_argument("--cg-tolerance", type=float, default=CG_TOLERANCE)
    parser.add_argument("--restart", type=Path)
    parser.add_argument(
        "--checkpoint-interval", type=float, default=CHECKPOINT_INTERVAL
    )
    parser.add_argument(
        "--diagnostic-interval", type=float, default=DIAGNOSTIC_INTERVAL
    )
    parser.add_argument("--no-schwarzschild-reference", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_em_blackhole_formation_first_order(
        amplitude=args.amplitude,
        width=args.width,
        radial_center=args.radial_center,
        domain_half_width=args.domain_half_width,
        num_radial_points=args.num_rho,
        num_z_points=args.num_z,
        cfl=args.cfl,
        final_time=args.final_time,
        snapshot_count=args.snapshots,
        output_dir=args.output_dir,
        newton_tolerance=args.newton_tolerance,
        cg_tolerance=args.cg_tolerance,
        show_progress=not args.no_progress,
        restart=args.restart,
        checkpoint_interval=args.checkpoint_interval,
        diagnostic_interval=args.diagnostic_interval,
        run_schwarzschild=not args.no_schwarzschild_reference,
    )
