"""Evolve constraint-solved, self-gravitating electromagnetic dipole data.

The default is the off-centered time-symmetric family of Baumgarte,
Gundlach, and Hilditch.  Its amplitude is a literature-informed
supercritical candidate, but this demo has not been run or calibrated in this
code and does not contain an apparent-horizon finder.
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
    _project_scalar,
    _project_vector,
)
from JAX_BSSN.diagnostics.openpmd import OpenPMDWriter
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
from JAX_BSSN.EM.second_order.initial_data import (
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
DOMAIN_HALF_WIDTH = 16.0
NUM_RADIAL_POINTS = 96
NUM_Z_POINTS = 192
CFL = 0.2
FINAL_TIME = 12.0
SNAPSHOT_COUNT = 120

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
        zero_shift=1,
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


def electromagnetic_cartesian_grid(
    num_radial_points: int,
    num_z_points: int,
    dx: float,
):
    """Return the full elliptic grid and its exact central ``y=0`` index."""

    num_x = 2 * num_radial_points
    num_y = 2 * num_radial_points + 1
    x = (jnp.arange(num_x, dtype=jnp.float64) - (num_x - 1) / 2.0) * dx
    y = (jnp.arange(num_y, dtype=jnp.float64) - (num_y - 1) / 2.0) * dx
    z = (
        jnp.arange(num_z_points, dtype=jnp.float64)
        - (num_z_points - 1) / 2.0
    ) * dx
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    return jnp.stack((X, Y, Z), axis=-1), num_y // 2


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

    grid, center_y = electromagnetic_cartesian_grid(
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
    )
    psi, u, residual_history = solve_electromagnetic_conformal_factor(
        conformal_field_squared,
        dx,
        newton_tolerance=newton_tolerance,
        max_newton_iterations=max_newton_iterations,
        cg_tolerance=cg_tolerance,
        max_cg_iterations=max_cg_iterations,
        verbose=verbose,
    )

    psi_plane = psi[:, center_y : center_y + 1, :]
    conformal_electric_plane = conformal_electric[
        :, :, center_y : center_y + 1, :
    ]
    bssn = compact_axisymmetric_state(
        _flat_conformal_bssn_plane(psi_plane)
    )

    electric_covector = conformal_vector_to_physical_covector(
        conformal_electric_plane, psi_plane
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


def run_em_blackhole_formation(
    amplitude: float = AMPLITUDE,
    width: float = WIDTH,
    radial_center: float = RADIAL_CENTER,
    domain_half_width: float = DOMAIN_HALF_WIDTH,
    num_radial_points: int = NUM_RADIAL_POINTS,
    num_z_points: int = NUM_Z_POINTS,
    cfl: float = CFL,
    final_time: float = FINAL_TIME,
    snapshot_count: int = SNAPSHOT_COUNT,
    output_dir: str | Path = "output",
    newton_tolerance: float = NEWTON_TOLERANCE,
    max_newton_iterations: int = MAX_NEWTON_ITERATIONS,
    cg_tolerance: float = CG_TOLERANCE,
    max_cg_iterations: int = MAX_CG_ITERATIONS,
    show_progress: bool = True,
):
    """Build and evolve the literature-informed supercritical candidate."""

    dx = domain_half_width / num_radial_points
    num_steps = max(1, math.ceil(final_time / (cfl * dx)))
    dt = final_time / num_steps
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
    validate_axisymmetric_grid(state.bssn, params)
    validate_axisymmetric_wave_grid(state.em, params)
    jax.block_until_ready(state)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    residual_path = output_dir / "hamiltonian_residual.txt"
    constraint_path = output_dir / "constraint_norms.txt"
    openpmd_path = output_dir / "EM_blackhole_formation.h5"

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
    output_steps = set(
        np.linspace(
            0,
            num_steps,
            min(snapshot_count, num_steps) + 1,
            dtype=int,
        ).tolist()
    )
    advance = jax.jit(
        lambda current: axisymmetric_einstein_maxwell_rk4_step(
            current, params
        )
    )
    progress = tqdm(
        range(num_steps + 1),
        disable=not show_progress,
        desc="Einstein-Maxwell",
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
    with writer, constraint_path.open("w", encoding="utf-8") as norm_file:
        norm_file.write("# step time " + " ".join(norm_names) + "\n")
        for step in progress:
            if step in output_steps:
                violations = compute_matter_aware_axisymmetric_constraints(
                    state, params
                )
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
                rho_em = compute_axisymmetric_em_energy_density(state, params)
                physical = (slice(4, None), 0, slice(None))
                norms.update(
                    {
                        "electric_divergence_l2": electric_l2,
                        "electric_divergence_linf": electric_linf,
                        "magnetic_divergence_l2": magnetic_l2,
                        "magnetic_divergence_linf": magnetic_linf,
                        "max_rho_EM": jnp.max(rho_em[physical]),
                        "min_lapse": jnp.min(state.bssn.lapse[physical]),
                        "min_W": jnp.min(
                            state.bssn.conformal_factor[physical]
                        ),
                    }
                )
                jax.block_until_ready((violations, norms))
                writer.write(_output_fields(state, violations), step, step * dt)
                values = " ".join(
                    f"{float(norms[name]):.16e}" for name in norm_names
                )
                norm_file.write(f"{step:d} {step * dt:.16e} {values}\n")
                norm_file.flush()

            if step < num_steps:
                state = advance(state)

    jax.block_until_ready(state)
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
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--newton-tolerance", type=float, default=NEWTON_TOLERANCE)
    parser.add_argument("--cg-tolerance", type=float, default=CG_TOLERANCE)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_em_blackhole_formation(
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
    )
