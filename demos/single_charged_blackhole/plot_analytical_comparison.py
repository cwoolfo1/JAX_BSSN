"""Compare a stationary spherical reconstruction of Cartesian data with RN.

Run without arguments; edit the paths and snapshot selection below. The axis
passes through the saved puncture position (y=z=0 for the centered demo).
The time shift assumes stationarity, including a time-independent areal map.
This diagnostic does not establish that the evolved spacetime is stationary.

The PDF's radial projection uses the physical metric, gamma_ij=tilde_gamma_ij/W^2.
Its diagonalization requires beta_r**2 in the numerator and division by
(dR/dr)**2 when converting to the areal component g_RR.
"""

import json
from pathlib import Path
import sys

import matplotlib
import numpy as np
import openpmd_api as io

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEMO_DIRECTORY = Path(__file__).resolve().parent
if str(DEMO_DIRECTORY.parents[1]) not in sys.path:
    sys.path.insert(0, str(DEMO_DIRECTORY.parents[1]))

from JAX_BSSN.diagnostics.mesh_coordinates import mesh_axis_coordinates

INPUT_DIRECTORY = DEMO_DIRECTORY / "output"
OUTPUT_DIRECTORY = INPUT_DIRECTORY / "analytical_comparison"
SNAPSHOT_STEP = None  # None selects the final saved iteration.
INNER_RADIUS_LIMIT = 10.0  # R/M
METRIC_COMPONENTS = (("xx", 0, 0), ("xy", 0, 1), ("xz", 0, 2),
                     ("yy", 1, 1), ("yz", 1, 2), ("zz", 2, 2))


def axis_stencil(coordinates, target):
    """Select a native sample or four bracketing samples with cubic weights."""
    dx = coordinates[1] - coordinates[0]
    nearest = int(np.argmin(abs(coordinates - target)))
    if abs(coordinates[nearest] - target) <= 1.e-10 * dx:
        return nearest, np.ones(1)
    start = int(np.searchsorted(coordinates, target)) - 2
    if start < 0 or start + 4 > coordinates.size:
        raise ValueError("The puncture axis needs four bracketing transverse samples")
    nodes = (coordinates[start:start + 4] - target) / dx
    weights = np.ones(4)
    for i in range(4):
        for j in range(4):
            if i != j:
                weights[i] *= -nodes[j] / (nodes[i] - nodes[j])
    return start, weights


def read_axis(input_directory=INPUT_DIRECTORY, step=SNAPSHOT_STEP):
    """Read only the slabs needed for the positive axis of one snapshot."""
    input_directory = Path(input_directory)
    configuration = json.loads((input_directory / "run_configuration.json").read_text())
    position = np.asarray(configuration["position"])
    series = io.Series(str(input_directory / "single_charged_blackhole.h5"),
                       io.Access.read_only, '{"defer_iteration_parsing": true}')
    try:
        step = max(series.iterations) if step is None else step
        iteration = series.iterations[step]
        iteration.open()
        meshes = iteration.meshes
        names = ["lapse", "W"] + ["conformal_metric_" + c for c, _, _ in METRIC_COMPONENTS]
        missing = set(names + ["shift"]) - set(meshes)
        if missing or ("shift" in meshes and not set("xyz").issubset(meshes["shift"])):
            raise ValueError(
                "Snapshot lacks the metric/shift fields needed for reconstruction "
                f"(missing meshes: {sorted(missing)}). Rerun the demo with the "
                "extended output; lapse and W alone cannot reconstruct evolved data."
            )
        scalar = io.Mesh_Record_Component.SCALAR
        lapse_mesh = meshes["lapse"]
        component = lapse_mesh[scalar]
        coordinates = [mesh_axis_coordinates(lapse_mesh, component, i) for i in range(3)]
        x, y, z = coordinates
        iy, wy = axis_stencil(y, position[1])
        iz, wz = axis_stencil(z, position[2])
        offset, extent = [0, iy, iz], [len(x), len(wy), len(wz)]
        records = {name: (meshes[name], meshes[name][scalar]) for name in names}
        records.update({"beta_" + c: (meshes["shift"], meshes["shift"][c]) for c in "xyz"})
        pending = {}
        for name, (mesh, record) in records.items():
            if any(not np.array_equal(mesh_axis_coordinates(mesh, record, i), coordinates[i])
                   for i in range(3)):
                raise ValueError(f"{name} is not on the same Cartesian grid as lapse")
            pending[name] = record.load_chunk(offset, extent)
        time = float(iteration.time)
        series.flush()
        fields = {name: np.array(array, copy=True) for name, array in pending.items()}
    finally:
        series.close()

    # Convert at the native samples before interpolating the nonlinear metric.
    gamma = np.empty((3, 3, len(x)))
    for name, i, j in METRIC_COMPONENTS:
        physical = fields["conformal_metric_" + name] / fields["W"]**2
        gamma[i, j] = np.einsum("ijk,j,k->i", physical, wy, wz)
        gamma[j, i] = gamma[i, j]
    alpha = np.einsum("ijk,j,k->i", fields["lapse"], wy, wz)
    beta = np.stack([np.einsum("ijk,j,k->i", fields["beta_" + c], wy, wz) for c in "xyz"])
    r = x - position[0]
    positive = r > 0.0
    # Match the demo's four-cell boundary margin, retaining raw samples for stencils.
    interior = (np.arange(len(x)) >= 4) & (np.arange(len(x)) < len(x) - 4)
    return {
        "r": r[positive], "alpha": alpha[positive],
        "gamma": gamma[:, :, positive], "beta": beta[:, positive],
        "interior": interior[positive],
    }, step, time, configuration


def reconstruct_metric(r, alpha, gamma, beta, interior):
    """Project onto +x, diagonalize, and retain the outer monotone branch."""
    # On the puncture-centered +x axis n=(1,0,0), so the PDF contraction
    # n^i n^j gamma_ij reduces exactly to gamma_xx (no angular averaging).
    gamma_rr = gamma[0, 0]
    beta_up_r = beta[0]
    beta_r = gamma_rr * beta_up_r
    gamma_T = 0.5 * (gamma[1, 1] + gamma[2, 2])
    D = alpha**2 - beta_r * beta_up_r
    with np.errstate(divide="ignore", invalid="ignore"):
        R = r * np.sqrt(gamma_T)
        transverse_difference = (gamma[1, 1] - gamma[2, 2]) / gamma_T
        dr = r[1] - r[0]
        dR_dr = np.full_like(R, np.nan)
        dR_dr[2:-2] = (R[:-4] - 8*R[1:-3] + 8*R[3:-1] - R[4:]) / (12*dr)
        g_rr = gamma_rr + beta_r**2 / D
        g_RR = g_rr / dR_dr**2

    # Walk inward from the last resolved sample; never sort across a throat.
    supported = interior.copy()
    supported[:2] = False
    supported[-2:] = False
    branch = np.zeros_like(supported)
    endpoints = np.flatnonzero(supported)
    if endpoints.size:
        end = endpoints[-1]
        outward = supported & np.isfinite(R) & (dR_dr > 0.0) & np.isfinite(dR_dr)
        breaks = np.flatnonzero(~outward[:end + 1])
        start = breaks[-1] + 1 if breaks.size else 0
        branch[start:end + 1] = True
    valid = (branch & (D > 0.0) & np.isfinite(D) & (gamma_rr > 0.0)
             & (gamma_T > 0.0) & np.isfinite(g_RR) & (g_RR > 0.0))
    return {
        "gamma_rr": gamma_rr, "gamma_T": gamma_T,
        "beta_up_r": beta_up_r, "beta_r": beta_r, "R": R, "dR_dr": dR_dr,
        "D": D, "g_rr_diagonal": g_rr, "g_RR": np.where(valid, g_RR, np.nan),
        "outer_branch": branch, "valid": valid,
        "transverse_relative_difference": transverse_difference,
    }


def main(input_directory=INPUT_DIRECTORY, output_directory=OUTPUT_DIRECTORY, step=SNAPSHOT_STEP):
    fields, step, time, configuration = read_axis(input_directory, step)
    profiles = reconstruct_metric(**fields)
    mass, charge = float(configuration["mass"]), float(configuration["charge"])
    R, numerical = profiles["R"], profiles["g_RR"]
    horizon = mass + np.sqrt(mass**2 - charge**2)
    # The saved charge is the geometric/Gaussian RN parameter, already Q.
    analytical = np.full_like(R, np.nan)
    exterior = profiles["outer_branch"] & (R > horizon)
    analytical[exterior] = 1.0 / (1.0 - 2*mass/R[exterior] + charge**2/R[exterior]**2)
    valid = profiles["valid"]
    if not np.any(valid & exterior):
        raise ValueError("No resolved exterior samples for the stationary RN comparison")
    radius = R / mass
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    # A fixed filename matches the editable single-snapshot workflow; the
    # actual step and time are recorded in both the figure and profile archive.
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    figure.suptitle("Charged puncture: stationary reconstruction and analytical RN\n"
                   f"M={mass:g}, Q/M={charge/mass:g}, step={step}, t/M={time/mass:g}; "
                   "no mass fit or rescaling")
    outer = np.max(radius[profiles["outer_branch"]])
    for axis, title, upper in zip(axes, ("Inner range", "Full resolved range"),
                                  (min(INNER_RADIUS_LIMIT, outer), outer)):
        axis.plot(radius, numerical, color="tab:blue", label="Stationary reconstruction")
        axis.plot(radius, analytical, "--", color="tab:orange", label="Analytical RN")
        if upper > horizon/mass:
            axis.set_xlim(horizon/mass, upper)
        else:
            axis.text(0.5, 0.5, "No samples in this range", transform=axis.transAxes, ha="center")
        axis.set_yscale("log")
        axis.set_xlabel(r"Areal radius $R/M$")
        axis.set_ylabel(r"$g_{RR}$")
        axis.set_title(title)
        axis.grid(alpha=0.25, which="both")
        axis.legend()
    path = output_directory / "final_grr.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    np.savez(output_directory / "final_grr_profiles.npz", **fields, **profiles,
             g_RR_RN=analytical, mass=mass, charge=charge, step=step, time=time,
             position=configuration["position"], stationary_assumption=True)

    print(f"Source: {Path(input_directory) / 'single_charged_blackhole.h5'}")
    print(f"step={step}, t/M={time/mass:g}; M={mass:g}, Q={charge:g}, R+/M={horizon/mass:.9g}")
    print(f"Positive x axis relative to puncture at {configuration['position']}")
    print("Assumes stationarity of the snapshot and areal-radius map.")
    print(f"Valid reconstruction: {valid.sum()}/{len(R)} samples, "
          f"R/M=[{radius[valid].min():.9g}, {radius[valid].max():.9g}]")
    print("Max |(gamma_yy-gamma_zz)/gamma_T| on valid samples: "
          f"{np.max(abs(profiles['transverse_relative_difference'][valid])):.6e}")
    comparison = valid & exterior
    for target in (3.0, 5.0, 10.0):
        # A target must be bracketed by adjacent valid samples, not a masked gap.
        brackets = np.flatnonzero(comparison[:-1] & comparison[1:]
                                  & (radius[:-1] <= target) & (radius[1:] >= target))
        if not brackets.size:
            print(f"R/M={target:g}: unavailable (outside valid coverage or in a masked gap)")
            continue
        left = brackets[0]
        index = left + int(abs(radius[left + 1] - target) < abs(radius[left] - target))
        relative = numerical[index] / analytical[index] - 1.0
        print(f"R/M={radius[index]:.6f}: g_RR={numerical[index]:.9g}, "
              f"RN={analytical[index]:.9g}, relative mismatch={relative:+.6e}")
    print(f"Saved {path} and {output_directory / 'final_grr_profiles.npz'}")
    return profiles


if __name__ == "__main__":
    main()
