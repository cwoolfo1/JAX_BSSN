"""Compare final EM-collapse lapse and W with a gauge-matched Schwarzschild run."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import openpmd_api as io


def _scalar_component(mesh):
    return mesh[next(iter(mesh))]


def _series_from_path(path, reference=False):
    path = Path(path)
    if path.is_file():
        return path

    summary_path = path / "run_summary.json"
    with summary_path.open("r", encoding="utf-8") as summary_file:
        summary = json.load(summary_file)
    if reference:
        recorded_path = Path(summary["openpmd_path"])
    else:
        recorded_path = Path(summary["output_segments"][-1])

    # Campaigns are often copied from one machine to another.  Prefer the
    # recorded path when it still exists, then locate the same series beside
    # the copied summary.
    if recorded_path.exists():
        return recorded_path
    local_path = summary_path.parent / recorded_path.name
    if local_path.exists():
        return local_path
    return recorded_path


def load_final_lapse_and_W(path):
    """Return final scalar fields and their cell-centred x-z coordinates."""

    path = Path(path)
    series = io.Series(str(path), io.Access.read_only)
    steps = list(series.iterations)
    if not steps:
        series.close()
        raise ValueError(f"no iterations found in {path}")

    step = max(steps)
    iteration = series.iterations[step]
    lapse_mesh = iteration.meshes["lapse"]
    W_mesh = iteration.meshes["W"]
    lapse_component = _scalar_component(lapse_mesh)
    W_component = _scalar_component(W_mesh)
    lapse_pending = lapse_component.load_chunk()
    W_pending = W_component.load_chunk()
    time = float(iteration.time)
    series.flush()

    lapse = np.asarray(lapse_pending).copy()
    W = np.asarray(W_pending).copy()
    x = (
        float(lapse_mesh.grid_global_offset[0])
        + (np.arange(lapse.shape[0]) + float(lapse_component.position[0]))
        * float(lapse_mesh.grid_spacing[0])
    )
    z = (
        float(lapse_mesh.grid_global_offset[2])
        + (np.arange(lapse.shape[2]) + float(lapse_component.position[2]))
        * float(lapse_mesh.grid_spacing[2])
    )
    iteration.close()
    series.close()

    if not np.all(np.isfinite(lapse)) or not np.all(np.isfinite(W)):
        raise ValueError(f"final iteration {step} in {path} contains non-finite data")
    return step, time, x, z, lapse, W


def _axis_profiles(x, z, field):
    positive_x = np.flatnonzero(x > 0.0)
    equatorial_z = np.argsort(np.abs(z))[:2]
    equatorial = np.mean(field[positive_x, :, :][:, :, equatorial_z], axis=(1, 2))

    central_x = np.argsort(np.abs(x))[:2]
    positive_z = np.flatnonzero(z > 0.0)
    polar = np.mean(field[central_x, :, :][:, :, positive_z], axis=(0, 1))
    return (x[positive_x], equatorial), (z[positive_z], polar)


def _relative_errors(radius, formation, reference, inner_radius, outer_radius):
    mask = (radius >= inner_radius) & (radius <= outer_radius)
    if np.count_nonzero(mask) < 2:
        raise ValueError("the requested exterior comparison interval is empty")
    difference = formation[mask] - reference[mask]
    reference_view = reference[mask]
    l2 = np.linalg.norm(difference) / max(
        np.linalg.norm(reference_view), np.finfo(float).tiny
    )
    linf = np.max(np.abs(difference)) / max(
        np.max(np.abs(reference_view)), np.finfo(float).tiny
    )
    return {"relative_l2": float(l2), "relative_linf": float(linf)}


def _horizon_radius(metadata, direction):
    horizon = metadata.get("horizon")
    if horizon is None:
        return None
    return float(horizon[f"{direction}_radius"])


def plot_comparison(formation, reference, metadata, output):
    """Write PNG/PDF comparisons and exterior error norms."""

    formation_path = _series_from_path(formation)
    reference_path = _series_from_path(reference, reference=True)
    metadata_path = Path(metadata)
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        formation_metadata = json.load(metadata_file)
    reference_metadata_path = (
        Path(reference).parent / "run_summary.json"
        if Path(reference).is_file()
        else Path(reference) / "run_summary.json"
    )
    if reference_metadata_path.exists():
        with reference_metadata_path.open("r", encoding="utf-8") as reference_file:
            reference_metadata = json.load(reference_file)
    else:
        reference_metadata = formation_metadata.get("schwarzschild_reference", {})

    formation_data = load_final_lapse_and_W(formation_path)
    reference_data = load_final_lapse_and_W(reference_path)
    _, formation_time, x, z, lapse, W = formation_data
    _, reference_time, reference_x, reference_z, reference_lapse, reference_W = reference_data

    mass = float(formation_metadata["horizon"]["irreducible_mass"])
    fields = {"lapse": (lapse, reference_lapse), "W": (W, reference_W)}
    directions = ("equatorial", "polar")
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), sharex="col")
    errors = {
        "formation_time": formation_time,
        "reference_time": reference_time,
        "mass": mass,
        "comparison_interval": {},
        "fields": {},
    }

    formation_horizon = {
        direction: _horizon_radius(formation_metadata, direction)
        for direction in directions
    }
    reference_horizon = {
        direction: _horizon_radius(reference_metadata, direction)
        for direction in directions
    }

    for row, (field_name, (formation_field, reference_field)) in enumerate(fields.items()):
        formation_profiles = _axis_profiles(x, z, formation_field)
        reference_profiles = _axis_profiles(reference_x, reference_z, reference_field)
        errors["fields"][field_name] = {}
        for column, direction in enumerate(directions):
            formation_radius, formation_profile = formation_profiles[column]
            reference_radius, reference_profile = reference_profiles[column]
            reference_on_formation = np.interp(
                formation_radius, reference_radius, reference_profile
            )
            horizon_radii = [
                radius
                for radius in (
                    formation_horizon[direction],
                    reference_horizon[direction],
                )
                if radius is not None
            ]
            dx = float(np.median(np.diff(formation_radius)))
            inner_radius = max(horizon_radii) + 2.0 * dx
            outer_radius = 0.65 * min(formation_radius[-1], reference_radius[-1])
            errors["comparison_interval"][direction] = {
                "inner_radius": inner_radius,
                "outer_radius": outer_radius,
            }
            errors["fields"][field_name][direction] = _relative_errors(
                formation_radius,
                formation_profile,
                reference_on_formation,
                inner_radius,
                outer_radius,
            )

            axis = axes[row, column]
            axis.plot(
                formation_radius / mass,
                formation_profile,
                linewidth=2.0,
                label="EM collapse",
            )
            axis.plot(
                formation_radius / mass,
                reference_on_formation,
                "--",
                linewidth=2.0,
                label="Schwarzschild",
            )
            if formation_horizon[direction] is not None:
                axis.axvline(
                    formation_horizon[direction] / mass,
                    color="C0",
                    alpha=0.45,
                )
            if reference_horizon[direction] is not None:
                axis.axvline(
                    reference_horizon[direction] / mass,
                    color="C1",
                    linestyle="--",
                    alpha=0.45,
                )
            axis.set_title(direction.capitalize())
            axis.set_ylabel(r"$\alpha$" if field_name == "lapse" else r"$W$")
            axis.grid(alpha=0.25)

    for axis in axes[-1]:
        axis.set_xlabel(r"$r/M_{\rm irr}$")
    axes[0, 0].legend()
    figure.suptitle("Settled EM collapse and gauge-matched Schwarzschild")
    figure.tight_layout()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_stem = output.with_suffix("")
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    error_path = output_stem.with_name(output_stem.name + "_errors.json")
    figure.savefig(png_path, dpi=180)
    figure.savefig(pdf_path)
    plt.close(figure)
    with error_path.open("w", encoding="utf-8") as error_file:
        json.dump(errors, error_file, indent=2, sort_keys=True)
        error_file.write("\n")
    return png_path, pdf_path, error_path, errors


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formation", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    plot_comparison(
        args.formation,
        args.reference,
        args.metadata,
        args.output,
    )


if __name__ == "__main__":
    main()
