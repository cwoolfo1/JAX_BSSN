"""Evolve electromagnetic dipole data with the second-order wave solver.

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
    reconstruct_axisymmetric_support,
    validate_axisymmetric_grid,
)
from JAX_BSSN.cartoon.axisymmetry.reconstruction import (
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
from JAX_BSSN.EM.second_order.cartoon.axisymmetry import (
    axisymmetric_einstein_maxwell_rk4_step,
    compact_axisymmetric_wave,
    compute_axisymmetric_constraint_divergences,
    expand_axisymmetric_wave_plane,
    project_axisymmetric_wave_rhs,
    reconstruct_axisymmetric_wave_support,
    validate_axisymmetric_wave_grid,
)
from JAX_BSSN.EM.second_order.diagnostics import electromagnetic_output_fields
from JAX_BSSN.EM.second_order.energy_momentum import (
    compute_electromagnetic_energy_momentum,
    compute_electromagnetic_stress_energy,
)
from JAX_BSSN.EM.second_order.equations import source_free_projected_field_dots
from JAX_BSSN.EM.second_order.geometry import compute_bssn_em_geometry
from initial_metric import initial_metric
from initial_pulse import (
    conformal_vector_to_physical_covector,
    contract_conformal_electromagnetic_fields,
    electromagnetic_cylindrical_grid,
    initial_pulse,
)
import simulation_parameters as settings
from JAX_BSSN.evolution.coordinates import cylindrical_volume_weights
from JAX_BSSN.EM.second_order.variables import EinsteinMaxwellVariables, EMVariables
from JAX_BSSN.evolution.time_evolve import compute_bssn_rhs_with_matter


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
    """Solve the constraints and return compact synchronized EM/BSSN data."""

    grid = electromagnetic_cylindrical_grid(
        num_radial_points, num_z_points, dx,
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
    electric_covector = conformal_vector_to_physical_covector(
        conformal_electric, psi_plane
    )
    parity = jnp.asarray((-1.0, -1.0, 1.0), dtype=electric_covector.dtype)
    parity = parity[:, None, None, None]
    electric_covector = jnp.concatenate(
        (jnp.flip(electric_covector, axis=1) * parity, electric_covector),
        axis=1,
    )
    magnetic_covector = conformal_vector_to_physical_covector(
        conformal_magnetic, psi_plane
    )
    magnetic_covector = jnp.concatenate(
        (jnp.flip(magnetic_covector, axis=1) * parity, magnetic_covector),
        axis=1,
    )
    zero = jnp.zeros_like(electric_covector)
    em = compact_axisymmetric_wave(
        EMVariables(
            electric_field=electric_covector,
            electric_field_dot=zero,
            magnetic_field=magnetic_covector,
            magnetic_field_dot=zero,
        )
    )

    # The wave variables store Eulerian projected derivatives.  Time symmetry
    # makes dot(E)=0, but dot(B) follows from the first-order Maxwell system.
    support_bssn = reconstruct_axisymmetric_support(bssn, params)
    support_em = reconstruct_axisymmetric_wave_support(em, params)
    stress_energy = compute_electromagnetic_energy_momentum(
        support_em.electric_field,
        support_em.magnetic_field,
        support_bssn,
    )
    support_bssn_rhs = compute_bssn_rhs_with_matter(
        support_bssn, params, *stress_energy
    )
    geometry = compute_bssn_em_geometry(
        support_bssn,
        support_bssn_rhs,
        params,
        backreaction_sources=stress_energy,
    )
    electric_dot, magnetic_dot = source_free_projected_field_dots(
        support_em.electric_field,
        support_em.magnetic_field,
        support_bssn,
        geometry,
        params,
    )
    support_em = EMVariables(
        electric_field=support_em.electric_field,
        electric_field_dot=electric_dot,
        magnetic_field=support_em.magnetic_field,
        magnetic_field_dot=magnetic_dot,
    )
    em = project_axisymmetric_wave_rhs(support_em)

    return EinsteinMaxwellVariables(bssn=bssn, em=em), u, residual_history


@jax.jit
def compute_matter_aware_axisymmetric_constraints(
    state: EinsteinMaxwellVariables,
    params: BSSNParameters,
) -> ConstraintViolations:
    """Return compact Einstein constraints including electromagnetic matter."""

    support_bssn = reconstruct_axisymmetric_support(state.bssn, params)
    support_em = reconstruct_axisymmetric_wave_support(state.em, params)
    stress_energy = compute_electromagnetic_stress_energy(
        support_em.electric_field,
        support_em.magnetic_field,
        support_bssn,
    )
    support = compute_all_constraints_with_matter(
        support_bssn,
        params,
        stress_energy.energy_density,
        stress_energy.momentum_density,
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
    support_em = reconstruct_axisymmetric_wave_support(state.em, params)
    stress_energy = compute_electromagnetic_stress_energy(
        support_em.electric_field,
        support_em.magnetic_field,
        support_bssn,
    )
    return _project_scalar(stress_energy.energy_density)


def _axisymmetric_scalar_norms(field, params):
    """Return the standard interior cylindrical L2 and Linf norms."""

    physical = field[4:-4, 0, 4:-4]
    weight = cylindrical_volume_weights(field.shape, params, field.dtype)[4:-4, 4:-4]
    normalization = jnp.sum(weight)
    l2_norm = jnp.sqrt(jnp.sum(weight * physical**2) / normalization)
    linf_norm = jnp.max(jnp.abs(physical))
    return l2_norm, linf_norm


def _output_fields(state, violations):
    fields = axisymmetric_plane_output_fields(
        state.bssn,
        violations.hamiltonian,
        violations.momentum,
        violations.det_gamma,
        violations.trace_A,
        violations.gamma_condition,
    )
    expanded_em = expand_axisymmetric_wave_plane(state.em)
    fields.update(electromagnetic_output_fields(expanded_em))
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
    validate_axisymmetric_wave_grid(state.em, params)
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
        "electric_divergence_l2",
        "electric_divergence_linf",
        "magnetic_divergence_l2",
        "magnetic_divergence_linf",
        "max_rho_EM",
        "min_lapse",
        "min_W",
    )
    snapshot_steps = max(1, math.ceil(max(num_steps, 1) / max(snapshot_count, 1)))
    diagnostic_steps = max(1, int(round(diagnostic_interval / dt)))
    advance = jax.jit(
        lambda current: axisymmetric_einstein_maxwell_rk4_step(
            current, params
        )
    )
    progress = tqdm(
        range(num_steps + 1),
        disable=not show_progress,
        desc="Second-order Einstein-Maxwell",
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
        electric_divergence, magnetic_divergence = (
            compute_axisymmetric_constraint_divergences(state.em, state.bssn, params)
        )
        electric_l2, electric_linf = _axisymmetric_scalar_norms(
            electric_divergence, params
        )
        magnetic_l2, magnetic_linf = _axisymmetric_scalar_norms(
            magnetic_divergence, params
        )
        physical = (slice(4, None), 0, slice(None))
        norms.update(
            {
                "electric_divergence_l2": electric_l2,
                "electric_divergence_linf": electric_linf,
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
            writer.write(_output_fields(state, violations), step, time)
        return finite

    with writer, constraint_path.open("w", encoding="utf-8") as norm_file:
        norm_file.write("# step time " + " ".join(norm_names) + "\n")
        for step in progress:
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
        "second_order",
        final_step,
        final_step * dt,
        params,
        configuration,
    )
    summary = {
        "formulation": "second_order",
        "status": status,
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
