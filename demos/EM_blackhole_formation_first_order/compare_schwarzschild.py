"""Evolve a specified-mass vacuum puncture and compare its final BSSN fields.

Run from this directory with JAX_BSSN installed, or set PYTHONPATH=../..:

    python compare_schwarzschild.py --mass 1.0
    python compare_schwarzschild.py --plot-only --output-dir <previous comparison>

The puncture uses the supplied mass and the checkpoint's grid,
gauge, boundaries, and elapsed time. Component profiles are coordinate/gauge
dependent; matching these settings does not imply matching gauge histories.
Outputs are a final NPZ state, comparison metadata, progress diagnostics, and
16 PNG figures in a fresh directory. Existing run directories are never reused
for evolution; --plot-only regenerates their figures from the saved state.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.cartoon.axisymmetry import (
    axisymmetric_rk4_step,
    validate_axisymmetric_grid,
)
from collapse_io import atomic_write_json
from schwarzschild_reference import _schwarzschild_state


def load_bssn(path):
    """Read only the BSSN arrays, leaving the staggered EM state untouched."""
    with np.load(path, allow_pickle=False) as data:
        state = BSSNVariables(*(data[f"bssn_{name}"] for name in BSSNVariables._fields))
        metadata = json.loads(str(data["metadata_json"]))
    return state, metadata


def validate_mass(mass):
    if mass is None or not np.isfinite(mass) or mass <= 0:
        raise ValueError("supply --mass with a finite, positive puncture mass")
    return float(mass)


def evolve_reference(state, params, num_steps, output_dir, diagnostic_interval=0.5):
    """Advance the vacuum RK4 method for the requested number of steps."""
    advance = jax.jit(lambda current: axisymmetric_rk4_step(current, params))
    diagnostic_steps = max(1, int(round(diagnostic_interval / params.dt)))
    with (output_dir / "progress.txt").open("w", encoding="utf-8") as stream:
        stream.write("# step time finite min_lapse min_W max_abs_K\n")
        for step in tqdm(range(num_steps + 1), desc="Schwarzschild puncture"):
            if step % diagnostic_steps == 0 or step == num_steps:
                host = jax.device_get(state)
                finite = all(np.isfinite(field).all() for field in host)
                lapse = host.lapse[4:, 0, :]
                W = host.conformal_factor[4:, 0, :]
                K = host.trace_K[4:, 0, :]
                stream.write(
                    f"{step} {step * params.dt:.16e} {int(finite)} "
                    f"{lapse.min():.16e} {W.min():.16e} {np.abs(K).max():.16e}\n"
                )
                stream.flush()
                if not finite:
                    raise RuntimeError(f"non-finite vacuum state at step {step}; see {stream.name}")
            if step < num_steps:
                state = advance(state)
    return jax.device_get(state)


def save_reference(state, metadata, output_dir):
    arrays = {f"bssn_{name}": np.asarray(field) for name, field in zip(state._fields, state)}
    np.savez(output_dir / "schwarzschild_final.npz", **arrays,
             metadata_json=np.asarray(json.dumps(metadata)))
    atomic_write_json(output_dir / "comparison.json", metadata)


def component_groups(state):
    """Cartesian components on the positive-rho Cartoon plane, without guards."""
    groups = {"scalars": [
        (r"$W$", state.conformal_factor[4:, 0, :]),
        (r"$\alpha$", state.lapse[4:, 0, :]),
        (r"$K$", state.trace_K[4:, 0, :]),
    ]}
    groups["vectors"] = [
        (rf"${symbol}^{{{axis}}}$", field[i, 4:, 0, :])
        for symbol, field in [(r"\tilde\Gamma", state.conformal_connection),
                              (r"\beta", state.shift)]
        for i, axis in enumerate("xyz")
    ]
    for name, symbol, field in [
        ("metric", r"\tilde\gamma", state.conformal_metric),
        ("curvature", r"\tilde A", state.traceless_K),
    ]:
        groups[name] = [
            (rf"${symbol}_{{{'xyz'[i]}{'xyz'[j]}}}$", field[i, j, 4:, 0, :])
            for i in range(3) for j in range(i, 3)
        ]
    return groups


def profile(field, z, cut):
    if cut == "equatorial":
        # Linear interpolation at z=0 respects the odd/even component parity.
        upper = np.searchsorted(z, 0.0)
        weight = -z[upper - 1] / (z[upper] - z[upper - 1])
        return (1.0 - weight) * field[:, upper - 1] + weight * field[:, upper]
    return field[0, :]


def plot_comparison(em, reference, metadata, output_dir):
    mass = metadata["mass"]
    params = metadata["parameters"]
    nr, _, nz = em.conformal_factor[4:].shape
    rho = (np.arange(nr) + 0.5) * params["dx"]
    z = params["z_min"] + np.arange(nz) * params["dx"]
    em_groups = component_groups(em)
    reference_groups = component_groups(reference)
    title = (f"M={mass:.8f}; specified puncture mass\n"
             f"EM t={metadata['em_time']:g} ({metadata['em_status']}), "
             f"vacuum t={metadata['reference_time']:g}")
    if metadata.get("validation_only", False):
        title = "SMOKE TEST — short reference evolution\n" + title

    for cut in ("equatorial", "near_axis"):
        coordinate = rho / mass if cut == "equatorial" else z / mass
        cut_label = ("equator: interpolated z=0" if cut == "equatorial"
                     else f"near axis: rho={rho[0]:.6g} (rho/M={rho[0]/mass:.5g})")
        xlabel = r"$\rho/M$" if cut == "equatorial" else r"$z/M$"
        for region in ("full", "inner"):
            mask = np.ones(coordinate.shape, dtype=bool)
            if region == "inner":
                radius = rho if cut == "equatorial" else np.sqrt(rho[0]**2 + z**2)
                mask = radius / mass <= 10.0
            for group, components in em_groups.items():
                rows = len(components) // 3
                fig, axes = plt.subplots(
                    2 * rows, 3, figsize=(13, 5 * rows + 1.5), squeeze=False,
                    gridspec_kw={"height_ratios": [3, 1] * rows},
                    layout="constrained",
                )
                for index, ((label, field), (_, vacuum)) in enumerate(
                    zip(components, reference_groups[group])
                ):
                    row, column = divmod(index, 3)
                    ax, difference = axes[2 * row, column], axes[2 * row + 1, column]
                    y = profile(field, z, cut)[mask]
                    y_vacuum = profile(vacuum, z, cut)[mask]
                    ax.plot(coordinate[mask], y, label="EM collapse", linewidth=1.5)
                    ax.plot(coordinate[mask], y_vacuum, "--", label="Schwarzschild", linewidth=1.5)
                    ax.set_title(label)
                    difference.plot(coordinate[mask], y - y_vacuum, color="tab:purple")
                    difference.axhline(0.0, color="0.5", linewidth=0.5)
                    difference.set_ylabel("EM − vacuum")
                    difference.set_xlabel(xlabel)
                    for panel in (ax, difference):
                        panel.grid(alpha=0.25)
                        panel.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
                    if index == 0:
                        ax.legend(fontsize=9)
                fig.suptitle(f"{title}\n{cut_label}; {region} domain — gauge-dependent components")
                fig.savefig(output_dir / f"{group}_{cut}_{region}.png", dpi=150)
                plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mass", type=float)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if not args.plot_only:
        try:
            args.mass = validate_mass(args.mass)
        except ValueError as error:
            parser.error(str(error))

    input_dir = args.input_dir.resolve()
    with (input_dir / "run_summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    # Use the file beside the summary even when a copied run records an old
    # absolute path. Older summaries may omit the checkpoint path entirely.
    checkpoint_path = input_dir / Path(
        summary.get("checkpoint_path", "rolling_checkpoint.npz")
    ).name
    em, checkpoint = load_bssn(checkpoint_path)
    params = BSSNParameters(**checkpoint["parameters"])
    config = {
        name: checkpoint["configuration"][name]
        for name in (
            "amplitude", "width", "radial_center", "domain_half_width",
            "num_radial_points", "num_z_points", "cfl",
            "diagnostic_interval", "checkpoint_interval",
        )
        if name in checkpoint["configuration"]
    }
    nr, nz = config["num_radial_points"], config["num_z_points"]
    if checkpoint["formulation"] != "first_order":
        raise ValueError("expected a first-order EM checkpoint")
    if em.conformal_factor.shape != (nr + 4, 1, nz):
        raise ValueError("checkpoint grid dimensions disagree with the compact BSSN layout")
    validate_axisymmetric_grid(em, params)
    if not all(np.isfinite(field).all() for field in em):
        raise ValueError("the EM checkpoint contains non-finite BSSN fields")
    if (summary["final_step"] != checkpoint["step"]
            or not np.isclose(summary["final_time"], checkpoint["time"])
            or summary["parameters"] != checkpoint["parameters"]):
        raise ValueError("EM summary and checkpoint do not describe the same final state")
    if not np.isclose(checkpoint["time"], checkpoint["step"] * params.dt):
        raise ValueError("checkpoint time does not match its step and timestep")

    if args.plot_only:
        if args.output_dir is None:
            parser.error("--plot-only requires --output-dir")
        output_dir = args.output_dir.resolve()
        reference, metadata = load_bssn(output_dir / "schwarzschild_final.npz")
        if (metadata["parameters"] != checkpoint["parameters"]
                or metadata["em_time"] != checkpoint["time"]
                or metadata["em_step"] != checkpoint["step"]
                or any(a.shape != b.shape for a, b in zip(em, reference))):
            raise ValueError("saved reference is incompatible with the EM checkpoint")
        validate_mass(metadata["mass"])
        if args.mass is not None and args.mass != metadata["mass"]:
            parser.error("--mass differs from the saved reference mass")
        if not all(np.isfinite(field).all() for field in reference):
            raise ValueError("the saved reference contains non-finite BSSN fields")
    else:
        mass = args.mass
        output_dir = (args.output_dir or Path(__file__).resolve().parent /
                      f"schwarzschild_comparison_{datetime.now():%Y%m%d_%H%M%S_%f}").resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "mass": mass,
            "mass_source": "user_specified",
            "em_checkpoint": str(checkpoint_path),
            "em_time": checkpoint["time"], "em_step": checkpoint["step"],
            "reference_time": checkpoint["time"], "reference_step": checkpoint["step"],
            "parameters": checkpoint["parameters"],
            "configuration": config,
            "em_status": summary["status"],
            "notes": "Mass is the specified initial puncture parameter. "
                     "BSSN profiles depend on gauge history.",
        }
        atomic_write_json(output_dir / "comparison.json", {**metadata, "status": "running"})
        print(f"Puncture M={mass:.16g}; "
              f"evolving {checkpoint['step']} steps to t={checkpoint['time']:g}", flush=True)
        print(f"EM run status: {summary['status']}. Output: {output_dir}", flush=True)
        reference = _schwarzschild_state(mass, params, nr, nz)
        validate_axisymmetric_grid(reference, params)
        reference = evolve_reference(reference, params, checkpoint["step"], output_dir,
                                     config.get("diagnostic_interval", 0.5))
        metadata["status"] = "complete"
        save_reference(reference, metadata, output_dir)

    plot_comparison(em, reference, metadata, output_dir)
    print(f"Saved 16 comparison figures in {output_dir}")


if __name__ == "__main__":
    main()
