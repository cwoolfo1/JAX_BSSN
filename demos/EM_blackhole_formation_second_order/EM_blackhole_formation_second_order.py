"""Evolve electromagnetic dipole data with the second-order wave solver.

The default is the off-centered time-symmetric family of Baumgarte,
Gundlach, and Hilditch.  Its amplitude is a literature-informed
supercritical candidate.  The evolution runs to a fixed final time and
records field snapshots and constraint diagnostics.
"""

import argparse
import math
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
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
    _project_scalar,
    _project_vector,
)
from JAX_BSSN.diagnostics.openpmd import OpenPMDWriter
from collapse_io import (
    atomic_write_json,
    load_collapse_checkpoint,
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
from initial_data import (
    conformal_vector_to_physical_covector,
    contract_conformal_electromagnetic_fields,
    off_centered_toroidal_electric_seed,
    solve_electromagnetic_conformal_factor,
)
from JAX_BSSN.EM.second_order.variables import EinsteinMaxwellVariables, EMVariables
from JAX_BSSN.evolution.boundaries import PERIODIC_BC, SOMMERFELD_BC
from JAX_BSSN.evolution.time_evolve import compute_bssn_rhs_with_matter


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
    """Solve the constraints and return compact synchronized EM/BSSN data."""

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

    electric_covector = conformal_vector_to_physical_covector(
        conformal_electric, psi_plane
    )
    parity = jnp.asarray((-1.0, -1.0, 1.0), dtype=electric_covector.dtype)
    parity = parity[:, None, None, None]
    electric_covector = jnp.concatenate(
        (jnp.flip(electric_covector, axis=1) * parity, electric_covector),
        axis=1,
    )
    magnetic_covector = jnp.zeros_like(electric_covector)
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


def _axisymmetric_scalar_norms(field):
    """Return the standard interior cylindrical L2 and Linf norms."""

    physical = field[4:-4, 0, 4:-4]
    rho = jnp.arange(physical.shape[0], dtype=field.dtype) + 0.5
    normalization = jnp.sum(rho) * physical.shape[1]
    l2_norm = jnp.sqrt(jnp.sum(rho[:, None] * physical**2) / normalization)
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


def run_em_blackhole_formation_second_order(
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
):
    """Evolve to the last whole timestep at or before final_time."""

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
            restart_path, "second_order"
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
    validate_axisymmetric_wave_grid(state.em, params)
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
        "electric_divergence_l2",
        "electric_divergence_linf",
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
        lambda current: axisymmetric_einstein_maxwell_rk4_step(
            current, params
        )
    )
    progress = tqdm(
        range(start_step, num_steps + 1),
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
    mode = "a" if restart_path is not None else "w"
    diagnostic_records = [
        {name: record[name] for name in ("step", "time", "finite")}
        for record in previous_summary.get("diagnostics", [])
        if record["step"] < start_step
    ]
    status = "complete"
    final_step = start_step

    def diagnose(step, norm_file, write_fields):
        time = step * float(params.dt)
        finite = state_is_finite(state)
        rho_em = compute_axisymmetric_em_energy_density(state, params)
        diagnostic_records.append(
            {"step": int(step), "time": time, "finite": finite}
        )

        violations = compute_matter_aware_axisymmetric_constraints(state, params)
        norms = compute_axisymmetric_constraint_norms(violations)
        electric_divergence, magnetic_divergence = (
            compute_axisymmetric_constraint_divergences(
                state.em, state.bssn, params
            )
        )
        electric_l2, electric_linf = _axisymmetric_scalar_norms(
            electric_divergence
        )
        magnetic_l2, magnetic_linf = _axisymmetric_scalar_norms(
            magnetic_divergence
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

    constraint_has_header = (
        constraint_path.exists() and constraint_path.stat().st_size > 0
    )
    with writer, constraint_path.open(mode, encoding="utf-8") as norm_file:
        if not constraint_has_header:
            norm_file.write("# step time " + " ".join(norm_names) + "\n")
        for step in progress:
            diagnostic_due = (
                step in (start_step, num_steps) or step % diagnostic_steps == 0
            )
            snapshot_due = (
                step in (start_step, num_steps) or step % snapshot_steps == 0
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
                if final_step % checkpoint_steps == 0:
                    jax.block_until_ready(state)
                    write_collapse_checkpoint(
                        checkpoint_path,
                        state,
                        u,
                        residual_history,
                        "second_order",
                        final_step,
                        final_step * float(params.dt),
                        params,
                        requested_configuration.copy(),
                    )

    jax.block_until_ready(state)
    write_collapse_checkpoint(
        checkpoint_path,
        state,
        u,
        residual_history,
        "second_order",
        final_step,
        final_step * float(params.dt),
        params,
        requested_configuration.copy(),
    )
    output_segments = list(previous_summary.get("output_segments", []))
    output_segments.append(str(openpmd_path))
    summary = {
        "formulation": "second_order",
        "status": status,
        "final_step": int(final_step),
        "final_time": float(final_step * float(params.dt)),
        "final_time_ceiling": float(final_time),
        "configuration": requested_configuration,
        "parameters": parameters_to_dict(params),
        "initial_hamiltonian_residual_history": residual_history,
        "diagnostics": diagnostic_records,
        "checkpoint_path": str(checkpoint_path),
        "output_segments": output_segments,
    }
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
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_em_blackhole_formation_second_order(
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
    )
