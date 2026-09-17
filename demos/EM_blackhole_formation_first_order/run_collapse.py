"""Evolve electromagnetic dipole data with the first-order Yee solver.

The default uses the time-symmetric family of Baumgarte, Gundlach, and
Hilditch with pulse-center parameter r0=0.  Black-hole formation at this
amplitude is unverified.  Edit simulation_parameters.py to configure a run.
Each fresh run records field snapshots and constraint diagnostics to a fixed
final time, then saves its final state.
Compiled JAX programs persist in .jax_cache beside this script; set
JAX_COMPILATION_CACHE_DIR to use another cache location across runs.
"""

import math
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

# Configure before importing solver modules that may trigger compilation.
# Keep the cache separate from simulation output so fresh runs can reuse it.
jax.config.update(
    "jax_compilation_cache_dir",
    jax.config.jax_compilation_cache_dir
    or str(Path(__file__).resolve().parent / ".jax_cache"),
)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

import jax.numpy as jnp
from tqdm import tqdm

from JAX_BSSN.bssn.constraints import (
    ConstraintViolations,
    compute_all_constraints_with_matter,
)
from JAX_BSSN.bssn.variables import BSSNParameters
from JAX_BSSN.cartoon.axisymmetry import (
    axisymmetric_plane_output_fields,
    compute_axisymmetric_constraint_norms,
    validate_axisymmetric_grid,
)
from JAX_BSSN.cartoon.axisymmetry.reconstruction import (
    _expand_axisymmetric_vector,
    _expand_axisymmetric_scalar,
    _project_scalar,
    _project_vector,
)
from JAX_BSSN.diagnostics.openpmd import OpenPMDWriter
from collapse_io import (
    atomic_write_json,
    parameters_to_dict,
    state_is_finite,
    write_collapse_checkpoint,
)
from JAX_BSSN.EM.first_order.cartoon.axisymmetry import (
    _support_bssn,
    axisymmetric_densitized_constraint_divergences,
    axisymmetric_first_order_einstein_maxwell_step,
    initialize_axisymmetric_first_order_state,
)
from JAX_BSSN.EM.first_order.cartoon.cylindrical import (
    axisymmetric_electromagnetic_sources, axisymmetric_physical_fields,
)
from JAX_BSSN.EM.first_order.evolve import (
    common_densitized_fields,
    common_physical_fields,
)
from JAX_BSSN.EM.variables import EinsteinMaxwellVariables
import simulation_parameters as settings
from JAX_BSSN.evolution.coordinates import cylindrical_volume_weights

from initial_metric import initial_metric
from initial_pulse import (
    conformal_vector_to_physical_contravariant,
    contract_conformal_electromagnetic_fields,
    electromagnetic_cylindrical_grid,
    initial_pulse,
)


def constrained_einstein_maxwell_data(
    amplitude: float,
    width: float,
    radial_center: float,
    num_radial_points: int,
    num_z_points: int,
    dx: float,
    params: BSSNParameters,
    newton_tolerance: float | None = None,
    max_newton_iterations: int | None = None,
    cg_tolerance: float | None = None,
    max_cg_iterations: int | None = None,
    verbose: bool = False,
):
    """Solve the constraints and bootstrap compact Yee-grid EM/BSSN data."""

    grid = electromagnetic_cylindrical_grid(
        num_radial_points, num_z_points, dx
    )
    conformal_electric, conformal_magnetic = initial_pulse(
        grid,
        amplitude=amplitude,
        width=width,
        radial_center=radial_center,
    )
    conformal_field_squared = contract_conformal_electromagnetic_fields(
        conformal_electric, conformal_magnetic
    )[:, 0, :]
    bssn, psi, u, residual_history = initial_metric(
        conformal_field_squared,
        dx,
        newton_tolerance=newton_tolerance,
        max_newton_iterations=max_newton_iterations,
        cg_tolerance=cg_tolerance,
        max_cg_iterations=max_cg_iterations,
        verbose=verbose,
    )

    psi_plane = psi[:, None, :]
    physical_displacement_plane = conformal_vector_to_physical_contravariant(
        conformal_electric, psi_plane
    )
    physical_displacement = jnp.zeros(
        (3, num_radial_points + 4, 1, num_z_points),
        dtype=physical_displacement_plane.dtype,
    ).at[:, 4:, 0, :].set(
        physical_displacement_plane[:, :, 0, :]
    )
    physical_magnetic_plane = conformal_vector_to_physical_contravariant(
        conformal_magnetic, psi_plane
    )
    physical_magnetic = jnp.zeros_like(physical_displacement).at[:, 4:, 0, :].set(
        physical_magnetic_plane[:, :, 0, :]
    )

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

    support_bssn = _support_bssn(state.bssn, params)
    displacement, magnetic = common_densitized_fields(state.em)
    energy_density, momentum_density, _ = (
        axisymmetric_electromagnetic_sources(
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

    support_bssn = _support_bssn(state.bssn, params)
    displacement, magnetic = common_densitized_fields(state.em)
    energy_density, _, _ = axisymmetric_electromagnetic_sources(
        displacement,
        magnetic,
        support_bssn,
        params,
    )
    return _project_scalar(energy_density)


def _axisymmetric_scalar_norms(field, params, location=("C", "C", "C")):
    """Return the standard interior cylindrical L2 and Linf norms."""

    physical = field[4:-4, 0, 4:-4]
    weight = cylindrical_volume_weights(field.shape, params, field.dtype, location)[4:-4, 4:-4]
    if location[0] == "V":
        weight = weight.at[0, :].set(params.dx / 8.0)
    normalization = jnp.sum(weight)
    l2_norm = jnp.sqrt(jnp.sum(weight * physical**2) / normalization)
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
    displacement, magnetic = axisymmetric_physical_fields(
        *common_densitized_fields(state.em), state.bssn, params
    )
    displacement = _expand_axisymmetric_vector(displacement)
    magnetic = _expand_axisymmetric_vector(magnetic)
    fields.update(
        {
            "rho_EM": _expand_axisymmetric_scalar(compute_axisymmetric_em_energy_density(state, params)),
            "D": tuple(displacement[i] for i in range(3)),
            "B": tuple(magnetic[i] for i in range(3)),
        }
    )
    return fields


def run_collapse():
    """Start a fresh run using the current simulation_parameters globals."""

    amplitude = settings.AMPLITUDE
    width = settings.WIDTH
    radial_center = settings.RADIAL_CENTER
    domain_half_width = settings.DOMAIN_HALF_WIDTH
    num_radial_points = settings.NUM_RADIAL_POINTS
    num_z_points = settings.NUM_Z_POINTS
    cfl = settings.CFL
    final_time = settings.FINAL_TIME
    snapshot_count = settings.SNAPSHOT_COUNT
    output_dir = settings.OUTPUT_DIR
    newton_tolerance = settings.NEWTON_TOLERANCE
    max_newton_iterations = settings.MAX_NEWTON_ITERATIONS
    cg_tolerance = settings.CG_TOLERANCE
    max_cg_iterations = settings.MAX_CG_ITERATIONS
    show_progress = settings.SHOW_PROGRESS
    diagnostic_interval = settings.DIAGNOSTIC_INTERVAL

    dx = settings.grid_spacing(num_radial_points, num_z_points, domain_half_width)
    if not math.isfinite(cfl) or cfl <= 0:
        raise ValueError("cfl must be finite and positive")
    if not math.isfinite(final_time) or final_time < 0:
        raise ValueError("final_time must be finite and nonnegative")
    if not math.isfinite(diagnostic_interval) or diagnostic_interval <= 0:
        raise ValueError("diagnostic_interval must be finite and positive")
    params = settings.axisymmetric_parameters(
        num_radial_points, num_z_points, domain_half_width, cfl * dx
    )

    dt = float(params.dt)
    num_steps = int(math.floor(final_time / dt + 1.0e-12))

    output_dir = Path(output_dir)
    residual_path = output_dir / "hamiltonian_residual.txt"
    constraint_path = output_dir / "constraint_norms.txt"
    checkpoint_path = output_dir / "final_checkpoint.npz"
    summary_path = output_dir / "run_summary.json"
    openpmd_path = output_dir / "EM_blackhole_formation.h5"
    protected_paths = (
        residual_path, constraint_path, checkpoint_path, summary_path,
        openpmd_path, output_dir / "rolling_checkpoint.npz",
        output_dir / "last_finite_checkpoint.npz",
    )
    if any(path.exists() for path in protected_paths):
        raise FileExistsError(
            f"refusing to overwrite an existing collapse run in {output_dir}"
        )

    configuration = {
        "amplitude": float(amplitude),
        "width": float(width),
        "radial_center": float(radial_center),
        "domain_half_width": float(domain_half_width),
        "num_radial_points": int(num_radial_points),
        "num_z_points": int(num_z_points),
        "cfl": float(cfl),
        "diagnostic_interval": float(diagnostic_interval),
    }
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

    validate_axisymmetric_grid(state.bssn, params)
    jax.block_until_ready(state)
    output_dir.mkdir(parents=True, exist_ok=True)
    with residual_path.open("w", encoding="utf-8") as residual_file:
        residual_file.write("# iteration interior_residual_rms\n")
        for iteration, residual in enumerate(residual_history):
            residual_file.write(f"{iteration:d} {residual:.16e}\n")

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
    diagnostic_steps = max(1, int(round(diagnostic_interval / dt)))
    advance = jax.jit(
        lambda current: axisymmetric_first_order_einstein_maxwell_step(
            current, params
        )
    )
    progress = tqdm(
        range(num_steps + 1),
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
    diagnostic_records = []
    status = "complete"
    final_step = 0

    def diagnose(step, norm_file, write_fields):
        time = step * dt
        finite = state_is_finite(state)
        rho_em = compute_axisymmetric_em_energy_density(state, params)
        diagnostic_records.append(
            {"step": int(step), "time": time, "finite": finite}
        )

        violations = compute_matter_aware_axisymmetric_constraints(state, params)
        norms = compute_axisymmetric_constraint_norms(violations, params=params)
        displacement_divergence, magnetic_divergence = (
            axisymmetric_densitized_constraint_divergences(state.em, params)
        )
        displacement_l2, displacement_linf = _axisymmetric_scalar_norms(
            displacement_divergence, params
        )
        magnetic_l2, magnetic_linf = _axisymmetric_scalar_norms(
            magnetic_divergence, params, ("V", "C", "V")
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
        if write_fields or not finite:
            writer.write(_output_fields(state, violations, params), step, time)
        return finite

    finite_flag = jax.jit(lambda current: jnp.all(jnp.stack([
        jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(current)
    ])))
    last_finite_state = state if bool(finite_flag(state)) else None
    last_finite_step = 0 if last_finite_state is not None else None
    with writer, constraint_path.open("w", encoding="utf-8") as norm_file:
        norm_file.write("# step time " + " ".join(norm_names) + "\n")
        for step in progress:
            finite = bool(finite_flag(state))
            if not finite:
                diagnose(step, norm_file, True)
                final_step = step
                status = "failed_nonfinite"
                break
            last_finite_state, last_finite_step = state, step
            diagnostic_due = (
                step in (0, num_steps) or step % diagnostic_steps == 0
            )
            snapshot_due = (
                step in (0, num_steps) or step % snapshot_steps == 0
            )
            if diagnostic_due or snapshot_due:
                finite = diagnose(step, norm_file, snapshot_due)
                final_step = step
                if not finite:
                    status = "failed_nonfinite"
                    break
            if step < num_steps:
                state = advance(state)
                final_step = step + 1

    jax.block_until_ready(state)
    write_collapse_checkpoint(
        checkpoint_path,
        state,
        u,
        residual_history,
        "first_order",
        final_step,
        final_step * dt,
        params,
        configuration,
    )
    if last_finite_state is not None:
        write_collapse_checkpoint(
            output_dir / "last_finite_checkpoint.npz", last_finite_state,
            u, residual_history, "first_order", last_finite_step,
            last_finite_step * dt, params, configuration,
        )
    summary = {
        "formulation": "first_order",
        "status": status,
        "last_finite_step": last_finite_step,
        "final_step": int(final_step),
        "final_time": float(final_step * dt),
        "final_time_ceiling": float(final_time),
        "configuration": configuration,
        "parameters": parameters_to_dict(params),
        "initial_hamiltonian_residual_history": residual_history,
        "diagnostics": diagnostic_records,
        "checkpoint_path": str(checkpoint_path),
        "output_segments": [str(openpmd_path)],
    }
    atomic_write_json(summary_path, summary)

    return state, u, residual_history


if __name__ == "__main__":
    run_collapse()
