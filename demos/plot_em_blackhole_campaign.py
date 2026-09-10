"""Plot and summarize the two-formulation medium/high collapse campaign."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

if __package__:
    from .plot_em_blackhole_schwarzschild import plot_comparison
else:
    from plot_em_blackhole_schwarzschild import plot_comparison


FORMULATIONS = ("first_order", "second_order")
RESOLUTIONS = ("medium", "high")


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _read_constraint_history(path):
    """Return the named constraint columns and whether the history is finite."""

    path = Path(path)
    header = None
    rows = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.startswith("#"):
                header = line[1:].split()
            elif line.strip():
                rows.append([float(value) for value in line.split()])
    if header is None or not rows:
        return {}, False
    values = np.asarray(rows)
    return dict(zip(header, values[-1])), bool(np.all(np.isfinite(values)))


def _horizon_formation_time(diagnostics):
    for diagnostic in diagnostics:
        if diagnostic.get("horizon") is not None:
            return float(diagnostic["time"])
    return None


def _final_exterior_energy(diagnostics):
    for diagnostic in reversed(diagnostics):
        energy = diagnostic.get("exterior_em_energy")
        if energy is not None:
            return float(energy)
    return None


def _em_constraint_l2(constraints):
    names = (
        "displacement_divergence_l2",
        "electric_divergence_l2",
        "magnetic_divergence_l2",
    )
    values = [constraints[name] for name in names if name in constraints]
    return max(values) if values else None


def _error_value(errors, field, direction, norm):
    if errors is None:
        return None
    return errors["fields"][field][direction][norm]


def _campaign_record(formulation, resolution, directory, output_dir):
    directory = Path(directory)
    summary_path = directory / "run_summary.json"
    summary = _read_json(summary_path)
    diagnostics = summary.get("diagnostics", [])
    horizon = summary.get("horizon")
    constraints, constraint_history_finite = _read_constraint_history(
        directory / "constraint_norms.txt"
    )

    reference_summary = summary.get("schwarzschild_reference")
    reference_dir = directory / "schwarzschild_reference"
    errors = None
    plot_error = None
    if (
        summary.get("status") == "settled"
        and horizon is not None
        and reference_summary is not None
        and reference_summary.get("status") == "settled"
    ):
        output = output_dir / (
            f"{formulation}_{resolution}_schwarzschild_comparison.png"
        )
        _, _, _, errors = plot_comparison(
            directory,
            reference_dir,
            summary_path,
            output,
        )
    else:
        plot_error = "formation and Schwarzschild reference must both be settled"

    mass = None if horizon is None else float(horizon["irreducible_mass"])
    dx = float(summary["parameters"]["dx"])
    diameter_cells = None
    circumference_distortion = None
    if horizon is not None:
        minimum_radius = min(
            float(horizon["polar_radius"]),
            float(horizon["equatorial_radius"]),
        )
        diameter_cells = 2.0 * minimum_radius / dx
        circumference_distortion = abs(
            float(horizon["circumference_ratio"]) - 1.0
        )

    horizon_time = _horizon_formation_time(diagnostics)
    final_em_energy = _final_exterior_energy(diagnostics)
    return {
        "formulation": formulation,
        "resolution": resolution,
        "directory": str(directory),
        "amplitude": float(summary["configuration"]["amplitude"]),
        "num_rho": int(summary["configuration"]["num_radial_points"]),
        "num_z": int(summary["configuration"]["num_z_points"]),
        "dx": dx,
        "status": summary.get("status"),
        "reference_status": (
            None if reference_summary is None else reference_summary.get("status")
        ),
        "initial_horizon_absent": bool(
            diagnostics and diagnostics[0].get("horizon") is None
        ),
        "horizon_time": horizon_time,
        "formed_before_ceiling": bool(
            horizon_time is not None
            and horizon_time < float(summary["final_time_ceiling"])
        ),
        "remnant_mass": mass,
        "settling_time": (
            float(summary["final_time"])
            if summary.get("status") == "settled"
            else None
        ),
        "horizon_diameter_cells": diameter_cells,
        "circumference_distortion": circumference_distortion,
        "diagnostics_finite": bool(
            diagnostics and all(row.get("finite", False) for row in diagnostics)
        ),
        "constraint_history_finite": constraint_history_finite,
        "hamiltonian_l2": constraints.get("hamiltonian_l2"),
        "momentum_l2": constraints.get("momentum_l2"),
        "em_constraint_l2": _em_constraint_l2(constraints),
        "exterior_em_energy": final_em_energy,
        "exterior_em_energy_fraction": (
            None
            if mass is None or final_em_energy is None
            else final_em_energy / mass
        ),
        "lapse_equatorial_l2": _error_value(
            errors, "lapse", "equatorial", "relative_l2"
        ),
        "lapse_equatorial_linf": _error_value(
            errors, "lapse", "equatorial", "relative_linf"
        ),
        "lapse_polar_l2": _error_value(
            errors, "lapse", "polar", "relative_l2"
        ),
        "lapse_polar_linf": _error_value(
            errors, "lapse", "polar", "relative_linf"
        ),
        "W_equatorial_l2": _error_value(
            errors, "W", "equatorial", "relative_l2"
        ),
        "W_equatorial_linf": _error_value(
            errors, "W", "equatorial", "relative_linf"
        ),
        "W_polar_l2": _error_value(errors, "W", "polar", "relative_l2"),
        "W_polar_linf": _error_value(
            errors, "W", "polar", "relative_linf"
        ),
        "plot_error": plot_error,
    }


def _is_smaller(high, medium, name):
    return (
        high[name] is not None
        and medium[name] is not None
        and high[name] < medium[name]
    )


def evaluate_acceptance(records):
    """Evaluate the production acceptance tests from four campaign records."""

    by_run = {
        (record["formulation"], record["resolution"]): record
        for record in records
    }
    error_names = (
        "lapse_equatorial_l2",
        "lapse_equatorial_linf",
        "lapse_polar_l2",
        "lapse_polar_linf",
        "W_equatorial_l2",
        "W_equatorial_linf",
        "W_polar_l2",
        "W_polar_linf",
    )
    resolution_improvement = {}
    for formulation in FORMULATIONS:
        medium = by_run[(formulation, "medium")]
        high = by_run[(formulation, "high")]
        resolution_improvement[formulation] = {
            name: _is_smaller(high, medium, name) for name in error_names
        }
        mass_scale = max(
            abs(high["remnant_mass"] or 0.0), np.finfo(float).tiny
        )
        high["medium_high_mass_fractional_difference"] = (
            None
            if medium["remnant_mass"] is None or high["remnant_mass"] is None
            else abs(high["remnant_mass"] - medium["remnant_mass"])
            / mass_scale
        )

    high_masses = [
        by_run[(formulation, "high")]["remnant_mass"]
        for formulation in FORMULATIONS
    ]
    if any(mass is None for mass in high_masses):
        formulation_mass_difference = None
        mass_agreement = False
    else:
        formulation_mass_difference = abs(high_masses[0] - high_masses[1]) / np.mean(
            high_masses
        )
        mass_agreement = formulation_mass_difference < 0.05

    common_amplitude = len({record["amplitude"] for record in records}) == 1
    all_runs_settled = all(record["status"] == "settled" for record in records)
    all_references_settled = all(
        record["reference_status"] == "settled" for record in records
    )
    all_histories_finite = all(
        record["diagnostics_finite"] and record["constraint_history_finite"]
        for record in records
    )
    no_initial_horizons = all(
        record["initial_horizon_absent"] for record in records
    )
    all_horizons_formed = all(
        record["formed_before_ceiling"] for record in records
    )
    medium_horizons_resolved = all(
        by_run[(formulation, "medium")]["horizon_diameter_cells"] is not None
        and by_run[(formulation, "medium")]["horizon_diameter_cells"] >= 6.0
        for formulation in FORMULATIONS
    )
    all_errors_improve = all(
        passed
        for formulation in resolution_improvement.values()
        for passed in formulation.values()
    )
    accepted = bool(
        common_amplitude
        and all_runs_settled
        and all_references_settled
        and all_histories_finite
        and no_initial_horizons
        and all_horizons_formed
        and medium_horizons_resolved
        and all_errors_improve
        and mass_agreement
    )
    return {
        "accepted": accepted,
        "common_amplitude": common_amplitude,
        "all_runs_settled": all_runs_settled,
        "all_references_settled": all_references_settled,
        "all_histories_finite": all_histories_finite,
        "no_initial_horizons": no_initial_horizons,
        "all_horizons_formed_before_ceiling": all_horizons_formed,
        "medium_horizons_resolved_by_six_cells": medium_horizons_resolved,
        "resolution_improvement": resolution_improvement,
        "high_resolution_mass_fractional_difference": formulation_mass_difference,
        "high_resolution_masses_agree_within_five_percent": mass_agreement,
    }


def _format(value):
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6e}"
    return str(value)


def _write_markdown(path, records, acceptance):
    columns = (
        ("formulation", "formulation"),
        ("resolution", "resolution"),
        ("status", "status"),
        ("horizon_time", "horizon t"),
        ("remnant_mass", "M_irr"),
        ("settling_time", "settling t"),
        ("horizon_diameter_cells", "AH cells"),
        ("hamiltonian_l2", "H L2"),
        ("momentum_l2", "M L2"),
        ("em_constraint_l2", "EM div L2"),
        ("exterior_em_energy_fraction", "E_EM/M"),
        ("lapse_equatorial_l2", "lapse eq L2"),
        ("lapse_polar_l2", "lapse pol L2"),
        ("W_equatorial_l2", "W eq L2"),
        ("W_polar_l2", "W pol L2"),
        ("medium_high_mass_fractional_difference", "medium/high dM"),
    )
    lines = [
        "# EM black-hole collapse campaign",
        "",
        f"Overall acceptance: **{'PASS' if acceptance['accepted'] else 'FAIL'}**",
        "",
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for record in records:
        lines.append(
            "| "
            + " | ".join(_format(record.get(name)) for name, _ in columns)
            + " |"
        )
    lines.extend(("", "## Acceptance details", "", "```json"))
    lines.extend(json.dumps(acceptance, indent=2, sort_keys=True).splitlines())
    lines.extend(("```", ""))
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def build_campaign_report(campaigns, output_dir):
    """Generate all comparison plots, tables, and acceptance metadata."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for formulation in FORMULATIONS:
        for resolution in RESOLUTIONS:
            records.append(
                _campaign_record(
                    formulation,
                    resolution,
                    campaigns[(formulation, resolution)],
                    output_dir,
                )
            )

    acceptance = evaluate_acceptance(records)
    json_path = output_dir / "campaign_summary.json"
    csv_path = output_dir / "campaign_summary.csv"
    markdown_path = output_dir / "campaign_summary.md"
    acceptance_path = output_dir / "acceptance.json"
    json_path.write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    acceptance_path.write_text(
        json.dumps(acceptance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(markdown_path, records, acceptance)
    return records, acceptance


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-medium", type=Path, required=True)
    parser.add_argument("--first-high", type=Path, required=True)
    parser.add_argument("--second-medium", type=Path, required=True)
    parser.add_argument("--second-high", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    campaigns = {
        ("first_order", "medium"): args.first_medium,
        ("first_order", "high"): args.first_high,
        ("second_order", "medium"): args.second_medium,
        ("second_order", "high"): args.second_high,
    }
    _, acceptance = build_campaign_report(campaigns, args.output_dir)
    print(f"Campaign acceptance: {'PASS' if acceptance['accepted'] else 'FAIL'}")


if __name__ == "__main__":
    main()
