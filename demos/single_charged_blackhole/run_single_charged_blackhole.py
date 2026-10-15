"""Run one charged puncture with the Cartesian Einstein--Maxwell system.

Edit the defaults below or pass :class:`SingleChargedBlackHoleConfig` to the
importable builder/runner.  The default uniform-grid run is a starting point,
not a validated long-time black-hole calculation.
"""

import os
from pathlib import Path


# This must be set before importing JAX and selecting a backend.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import csv
from dataclasses import asdict, dataclass
from functools import partial
import json
import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from tqdm import tqdm

from JAX_BSSN.bssn.constraints import compute_all_constraints_with_matter
from JAX_BSSN.bssn.tensor_algebra import determinant_3x3_metric
from JAX_BSSN.bssn.variables import BSSNParameters
from JAX_BSSN.diagnostics.openpmd import OpenPMDWriter
from JAX_BSSN.EM.first_order import (
    common_densitized_fields,
    common_physical_fields,
    compute_densitized_electromagnetic_energy_momentum,
    first_order_constraint_divergences,
    first_order_einstein_maxwell_step,
    make_em_grid,
)
from JAX_BSSN.evolution.boundaries import SOMMERFELD_BC
from JAX_BSSN.utilities import single_charged_puncture_state


MASS = 1.0
CHARGE = 0.5
POSITION = (0.0, 0.0, 0.0)

NUM_POINTS = 96
HALF_WIDTH = 12.0
CFL = 0.2
FINAL_TIME = 100.0
SNAPSHOT_INTERVALS = 50

ETA = 2.0
KAPPA = 0.002
NU = 0.02
GAMMA_DRIVER = 0.75

PLATFORM = "gpu"
DEVICE_INDEX = 0
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OVERWRITE = False
SHOW_PROGRESS = True


@dataclass(frozen=True)
class SingleChargedBlackHoleConfig:
    """Physical, numerical, device, and output settings for the demo."""

    mass: float = MASS
    charge: float = CHARGE
    position: tuple = POSITION
    num_points: int = NUM_POINTS
    half_width: float = HALF_WIDTH
    cfl: float = CFL
    final_time: float = FINAL_TIME
    snapshot_intervals: int = SNAPSHOT_INTERVALS
    eta: float = ETA
    kappa: float = KAPPA
    nu: float = NU
    gamma_driver: float = GAMMA_DRIVER
    platform: str = PLATFORM
    device_index: int = DEVICE_INDEX
    output_dir: Path = OUTPUT_DIR
    overwrite: bool = OVERWRITE
    show_progress: bool = SHOW_PROGRESS


def validate_config(config):
    """Validate settings before allocating the full Cartesian state."""
    if not math.isfinite(config.mass) or config.mass <= 0.0:
        raise ValueError("mass must be finite and positive")
    if not math.isfinite(config.charge) or abs(config.charge) >= config.mass:
        raise ValueError("charge must be finite and subextremal: |charge| < mass")
    if len(config.position) != 3 or any(
        not math.isfinite(value) for value in config.position
    ):
        raise ValueError("position must contain three finite coordinates")
    for name, minimum in (
        ("num_points", 12),
        ("snapshot_intervals", 1),
        ("device_index", 0),
    ):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if config.num_points % 2:
        raise ValueError("num_points must be even so a centered puncture avoids C nodes")
    for name in ("half_width", "cfl"):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("final_time", "eta", "kappa", "nu", "gamma_driver"):
        value = getattr(config, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if config.platform not in ("cpu", "gpu"):
        raise ValueError("platform must be 'cpu' or 'gpu'")

    dx = 2.0 * config.half_width / config.num_points
    margin = config.half_width - 4.5 * dx
    if any(abs(value) >= margin for value in config.position):
        raise ValueError("the puncture must lie inside the four-cell boundary margin")


def evolution_parameters(config):
    """Return fixed-step BSSN parameters and the number of evolution steps."""
    validate_config(config)
    dx = 2.0 * config.half_width / config.num_points
    proposed_dt = config.cfl * dx
    num_steps = math.ceil(config.final_time / proposed_dt)
    dt = config.final_time / num_steps if num_steps else proposed_dt
    origin = -config.half_width + 0.5 * dx
    params = BSSNParameters(
        eta=config.eta,
        kappa=config.kappa,
        nu=config.nu,
        g=config.gamma_driver,
        dx=dx,
        dt=dt,
        zero_shift=0,
        gauge=1,
        xl_bc=SOMMERFELD_BC,
        xr_bc=SOMMERFELD_BC,
        yl_bc=SOMMERFELD_BC,
        yr_bc=SOMMERFELD_BC,
        zl_bc=SOMMERFELD_BC,
        zr_bc=SOMMERFELD_BC,
        x_min=origin,
        y_min=origin,
        z_min=origin,
        mad_q=1.0,
    )
    return params, num_steps


def build_single_charged_blackhole(config=None):
    """Build the coupled state without selecting a device or writing output."""
    config = SingleChargedBlackHoleConfig() if config is None else config
    params, num_steps = evolution_parameters(config)
    shape = (config.num_points,) * 3
    em_grid = make_em_grid(
        shape, params, boundary_conditions=("constant",) * 3
    )
    state = single_charged_puncture_state(
        config.mass, config.charge, config.position, params, em_grid
    )
    return state, params, em_grid, num_steps


@jax.jit
def state_is_valid(state):
    """Check finiteness, positive W, and a positive conformal metric."""
    leaves = jax.tree_util.tree_leaves(state)
    finite = jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)))
    metric = state.bssn.conformal_metric
    positive_metric = jnp.all(
        (metric[0, 0] > 0.0)
        & (metric[0, 0] * metric[1, 1] - metric[0, 1] ** 2 > 0.0)
        & (determinant_3x3_metric(metric) > 0.0)
    )
    return finite & positive_metric & jnp.all(state.bssn.conformal_factor > 0.0)


@partial(jax.jit, static_argnames=("grid",))
def charged_blackhole_diagnostics(state, params, grid):
    """Return output fields and masked constraint norms at a common time."""
    displacement, magnetic = common_densitized_fields(state.em)
    rho, momentum, _stress = compute_densitized_electromagnetic_energy_momentum(
        displacement, magnetic, state.bssn, grid
    )
    constraints = compute_all_constraints_with_matter(
        state.bssn, params, rho, momentum
    )
    electric_divergence, magnetic_divergence = first_order_constraint_divergences(
        state, params, grid
    )
    physical_displacement, physical_magnetic = common_physical_fields(state, grid)

    diagnostic_fields = {
        "hamiltonian": constraints.hamiltonian,
        "momentum": constraints.momentum,
        "det_gamma": constraints.det_gamma,
        "trace_A": constraints.trace_A,
        "gamma": constraints.gamma_condition,
        "electric_divergence": electric_divergence,
        "magnetic_divergence": magnetic_divergence,
    }
    interior = jnp.zeros_like(state.bssn.conformal_factor, dtype=bool)
    interior = interior.at[4:-4, 4:-4, 4:-4].set(True)
    masks = {
        "interior": interior,
        "resolved": interior & (state.bssn.conformal_factor > 0.1),
    }
    norms = {
        "min_lapse": jnp.min(state.bssn.lapse),
        "min_W": jnp.min(state.bssn.conformal_factor),
    }
    for label, mask in masks.items():
        count = jnp.sum(mask)
        norms[f"{label}_points"] = count
        for name, field in diagnostic_fields.items():
            components = math.prod(field.shape[:-3])
            norms[f"{label}_{name}_rms"] = jnp.where(
                count > 0,
                jnp.sqrt(
                    jnp.sum(jnp.where(mask, field**2, 0.0))
                    / (count * components)
                ),
                jnp.nan,
            )
            norms[f"{label}_{name}_max"] = jnp.where(
                count > 0,
                jnp.max(jnp.where(mask, jnp.abs(field), 0.0)),
                jnp.nan,
            )

    output = {
        "lapse": state.bssn.lapse,
        "W": state.bssn.conformal_factor,
        "conformal_metric_xx": state.bssn.conformal_metric[0, 0],
        "conformal_metric_xy": state.bssn.conformal_metric[0, 1],
        "conformal_metric_xz": state.bssn.conformal_metric[0, 2],
        "conformal_metric_yy": state.bssn.conformal_metric[1, 1],
        "conformal_metric_yz": state.bssn.conformal_metric[1, 2],
        "conformal_metric_zz": state.bssn.conformal_metric[2, 2],
        "shift": state.bssn.shift,
        "K": state.bssn.trace_K,
        "rho_EM": rho,
        "D": physical_displacement,
        "B": physical_magnetic,
        "hamiltonian_constraint": constraints.hamiltonian,
        "momentum_constraint": constraints.momentum,
        "det_gamma_constraint": constraints.det_gamma,
        "trace_A_constraint": constraints.trace_A,
        "gamma_constraint": constraints.gamma_condition,
        "electric_divergence": electric_divergence,
        "magnetic_divergence": magnetic_divergence,
    }
    return norms, output


def select_device(platform, index):
    """Select an explicit JAX device, with no GPU-to-CPU fallback."""
    try:
        devices = jax.devices(platform)
    except RuntimeError as error:
        raise RuntimeError(f"JAX has no working {platform!r} backend") from error
    if not 0 <= index < len(devices):
        raise ValueError(
            f"device_index={index} is unavailable; {len(devices)} {platform} device(s) visible"
        )
    return devices[index]


def snapshot_steps(num_steps, intervals):
    """Return unique, approximately uniform output steps including endpoints."""
    if num_steps == 0:
        return (0,)
    count = min(num_steps, intervals)
    return tuple((index * num_steps + count - 1) // count for index in range(count + 1))


def _openpmd_fields(output):
    """Convert component-first JAX vectors to the writer's vector convention."""
    vector_names = ("D", "B", "momentum_constraint", "gamma_constraint", "shift")
    return {
        name: tuple(field[index] for index in range(3))
        if name in vector_names
        else field
        for name, field in output.items()
    }


def run_single_charged_blackhole(config=None):
    """Run the configured simulation and return its final state and metadata."""
    config = SingleChargedBlackHoleConfig() if config is None else config
    params, num_steps = evolution_parameters(config)
    output_dir = Path(config.output_dir)
    filenames = (
        "single_charged_blackhole.h5",
        "run_configuration.json",
        "diagnostics.csv",
    )
    for filename in filenames:
        path = output_dir / filename
        if path.exists() and not config.overwrite:
            raise FileExistsError(
                f"Output already exists: {path}; change output_dir or set overwrite=True"
            )

    device = select_device(config.platform, config.device_index)
    outputs = snapshot_steps(num_steps, config.snapshot_intervals)
    field_arrays = 30
    storage_bytes = len(outputs) * field_arrays * config.num_points**3 * 8
    print(
        f"{config.platform.upper()} {config.device_index}: {device.device_kind}; "
        f"float64; {config.num_points}^3 cells",
        flush=True,
    )
    print(
        f"{num_steps} steps, dt={params.dt:.8g}; {len(outputs)} snapshots; "
        f"approximately {storage_bytes / 2**30:.2f} GiB uncompressed",
        flush=True,
    )
    print(
        "Uniform-grid setup: long-time stability and physical accuracy are not validated.",
        flush=True,
    )

    with jax.default_device(device):
        state, params, grid, num_steps = build_single_charged_blackhole(config)
        jax.block_until_ready(state)
        if any(leaf.devices() != {device} for leaf in jax.tree_util.tree_leaves(state)):
            raise RuntimeError("initial fields were not all placed on the selected device")
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = asdict(config)
        metadata["output_dir"] = str(output_dir.resolve())
        metadata.update(
            parameters=params._asdict(),
            num_steps=num_steps,
            output_steps=outputs,
            estimated_snapshot_bytes=storage_bytes,
            jax_version=jax.__version__,
            device={
                "platform": device.platform,
                "id": device.id,
                "kind": device.device_kind,
            },
            initial_data=(
                "arXiv:1311.6483v2 Eqs. (2.8)-(2.13), single puncture u=1"
            ),
            field_convention=(
                "Native D^i=sqrt(gamma) E^i; physical D/B output is contravariant; "
                "paper charge converted from Gaussian units by 1/sqrt(4*pi); "
                "gamma_ij=conformal_metric_ij/W^2; shift is contravariant beta^i"
            ),
            maxwell_boundary="constant extrapolation on all faces",
            diagnostic_masks=(
                "interior excludes four outer cells; resolved also requires W>0.1"
            ),
        )
        (output_dir / filenames[1]).write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

        output_set = set(outputs)
        advance = jax.jit(
            lambda current: first_order_einstein_maxwell_step(
                current, params, grid
            )
        )
        with OpenPMDWriter(
            output_dir / filenames[0],
            grid_spacing=(params.dx,) * 3,
            grid_global_offset=(params.x_min, params.y_min, params.z_min),
            grid_position=(0.0, 0.0, 0.0),
            dt=params.dt,
            ghost_cells=0,
        ) as writer, (output_dir / filenames[2]).open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            diagnostic_writer = None
            progress = tqdm(
                range(num_steps + 1),
                desc="Single charged black hole",
                unit="state",
                disable=not config.show_progress,
            )
            for step in progress:
                time = config.final_time if step == num_steps else step * params.dt
                if not bool(state_is_valid(state)):
                    raise RuntimeError(
                        f"invalid evolved fields at step {step}, t={time:.8g}"
                    )
                if step in output_set:
                    norms, output = charged_blackhole_diagnostics(state, params, grid)
                    jax.block_until_ready((norms, output))
                    record = {
                        "step": step,
                        "time": time,
                        **{name: float(value) for name, value in norms.items()},
                    }
                    if diagnostic_writer is None:
                        diagnostic_writer = csv.DictWriter(
                            stream, fieldnames=tuple(record)
                        )
                        diagnostic_writer.writeheader()
                    diagnostic_writer.writerow(record)
                    stream.flush()
                    writer.write(_openpmd_fields(output), step=step, time=time)
                    if config.show_progress:
                        progress.set_postfix(
                            min_W=f"{record['min_W']:.3e}",
                            hamiltonian=f"{record['resolved_hamiltonian_rms']:.3e}",
                        )
                if step < num_steps:
                    state = advance(state)

    return state, {
        "parameters": params,
        "grid": grid,
        "num_steps": num_steps,
        "final_time": config.final_time,
        "output_dir": output_dir,
        "output_steps": outputs,
    }


def main():
    run_single_charged_blackhole()


if __name__ == "__main__":
    main()
