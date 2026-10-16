"""Compare the final equatorial collision metric with Schwarzschild.

Run without arguments; edit the paths and plot limits below. On the positive
x axis at y=z=0, R=x*sqrt(gamma_yy) is the equatorial circumference radius.
It is the areal radius only under spherical symmetry. The height-function
transformation T=t-h(r), h'=b/D, also assumes stationarity of the metric and
radius map. This slice alone cannot establish either assumption.

gamma_ij = conformal_metric_ij / W**2
b = gamma_xj * beta^j, D = alpha**2 - gamma_ij * beta^i * beta^j
g_TT = -D, g_RR = (gamma_xx + b**2/D) / (dR/dr)**2
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
MASS = 1.0  # Fixed reference: saved bare puncture masses 0.5 + 0.5, not a fit.
INNER_RADIUS_LIMIT = 10.0  # R/M
METRIC_COMPONENTS = (("xx", 0, 0), ("xy", 0, 1), ("xz", 0, 2),
                     ("yy", 1, 1), ("yz", 1, 2), ("zz", 2, 2))


def equatorial_stencil(z):
    """Return four bracketing z samples and cubic weights evaluated at zero."""
    start = int(np.searchsorted(z, 0.0)) - 2
    if start < 0 or start + 4 > len(z):
        raise ValueError("The z=0 slice needs four bracketing grid samples")
    nodes = z[start:start + 4] / (z[1] - z[0])
    weights = np.ones(4)
    for i in range(4):
        for j in range(4):
            if i != j:
                weights[i] *= -nodes[j] / (nodes[i] - nodes[j])
    return start, weights


def read_final_slice(input_directory=INPUT_DIRECTORY):
    """Read only the four z planes needed for the final equatorial profile."""
    input_directory = Path(input_directory)
    configuration = json.loads((input_directory / "run_configuration.json").read_text())
    series = io.Series(str(input_directory / "axisymmetric_blackhole_collision.h5"),
                       io.Access.read_only, '{"defer_iteration_parsing": true}')
    try:
        step = max(series.iterations)
        iteration = series.iterations[step]
        iteration.open()
        meshes = iteration.meshes
        scalar = io.Mesh_Record_Component.SCALAR
        mesh, record = meshes["lapse"], meshes["lapse"][scalar]
        coordinates = [mesh_axis_coordinates(mesh, record, i) for i in range(3)]
        x, y, z = coordinates
        if len(y) != 1 or not np.isclose(y[0], 0.0):
            raise ValueError("Expected the saved axisymmetric y=0 plane")
        iz, weights = equatorial_stencil(z)
        names = ["lapse", "W"] + ["conformal_metric_" + c for c, _, _ in METRIC_COMPONENTS]
        records = {name: (meshes[name], meshes[name][scalar]) for name in names}
        records.update({"beta_" + c: (meshes["shift"], meshes["shift"][c]) for c in "xyz"})
        pending = {}
        for name, (mesh, record) in records.items():
            if any(not np.array_equal(mesh_axis_coordinates(mesh, record, i), coordinates[i])
                   for i in range(3)):
                raise ValueError(f"{name} is not on the same Cartesian grid as lapse")
            pending[name] = record.load_chunk([0, 0, iz], [len(x), 1, 4])
        time = float(iteration.time)
        series.flush()
        fields = {name: np.array(array[:, 0, :], copy=True) for name, array in pending.items()}
    finally:
        series.close()

    # Divide by W**2 at native samples before interpolating the physical metric.
    gamma = np.empty((3, 3, len(x)))
    for name, i, j in METRIC_COMPONENTS:
        physical = fields["conformal_metric_" + name] / fields["W"]**2
        gamma[i, j] = physical @ weights
        gamma[j, i] = gamma[i, j]
    alpha = fields["lapse"] @ weights
    beta = np.stack([fields["beta_" + c] @ weights for c in "xyz"])
    positive = x > 0.0
    # Match the evolution diagnostic's four-cell outer boundary margin.
    interior = np.arange(len(x)) < len(x) - 4
    return {
        "r": x[positive], "alpha": alpha[positive],
        "gamma": gamma[:, :, positive], "beta": beta[:, positive],
        "interior": interior[positive],
    }, step, time, configuration


def reconstruct_metric(r, alpha, gamma, beta, interior):
    """Diagonalize the stationary t-r block on the outer monotone branch."""
    beta_down = np.einsum("ijn,jn->in", gamma, beta)
    b = beta_down[0]
    D = alpha**2 - np.sum(beta_down * beta, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = r * np.sqrt(gamma[1, 1])
        dr = r[1] - r[0]
        dR_dr = np.full_like(R, np.nan)
        dR_dr[2:-2] = (R[:-4] - 8*R[1:-3] + 8*R[3:-1] - R[4:]) / (12*dr)
        g_RR = (gamma[0, 0] + b**2 / D) / dR_dr**2
        angular_difference = (gamma[1, 1] - gamma[2, 2]) / gamma[1, 1]

    supported = interior.copy()
    supported[:2] = False
    supported[-2:] = False
    branch = np.zeros_like(supported)
    endpoints = np.flatnonzero(supported)
    if endpoints.size:
        end = endpoints[-1]
        outward = supported & np.isfinite(R) & np.isfinite(dR_dr) & (dR_dr > 0.0)
        breaks = np.flatnonzero(~outward[:end + 1])
        start = breaks[-1] + 1 if breaks.size else 0
        branch[start:end + 1] = True
    valid = (branch & (D > 0.0) & np.isfinite(D) & (gamma[0, 0] > 0.0)
             & (gamma[1, 1] > 0.0) & (g_RR > 0.0) & np.isfinite(g_RR))
    return {
        "R": R, "dR_dr": dR_dr, "D": D,
        "g_RR": np.where(valid, g_RR, np.nan),
        "g_TT": np.where(valid, -D, np.nan),
        "outer_branch": branch, "valid": valid,
        "angular_relative_difference": angular_difference,
    }


def main(input_directory=INPUT_DIRECTORY, output_directory=OUTPUT_DIRECTORY):
    fields, step, time, configuration = read_final_slice(input_directory)
    profiles = reconstruct_metric(**fields)
    R, valid = profiles["R"], profiles["valid"]
    radius = R / MASS
    # The analytical exterior is independent of the numerical D=0 surface.
    exterior = profiles["outer_branch"] & (R > 2*MASS)
    analytical_grr = np.full_like(R, np.nan)
    analytical_gtt = np.full_like(R, np.nan)
    analytical_grr[exterior] = 1.0 / (1.0 - 2*MASS/R[exterior])
    analytical_gtt[exterior] = -(1.0 - 2*MASS/R[exterior])
    comparison = valid & exterior
    if not np.any(comparison):
        raise ValueError("No resolved exterior samples for the Schwarzschild comparison")

    bare_mass_sum = sum(configuration["masses"])
    outer = np.max(radius[profiles["outer_branch"]])
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    print(f"Source: {Path(input_directory) / 'axisymmetric_blackhole_collision.h5'}")
    print(f"step={step}, t={time:g}, t/M={time/MASS:g}; fixed reference M={MASS:g}")
    print(f"Saved bare puncture mass sum={bare_mass_sum:g}; no measured remnant mass or fit.")
    print("Cubic interpolation to z=0, y=0, x>0; assumes stationarity and spherical symmetry.")
    print("R=x*sqrt(gamma_yy) is the equatorial circumference radius, areal in spherical symmetry.")
    print(f"Valid numerical exterior: {valid.sum()}/{len(R)} samples, "
          f"R/M=[{radius[valid].min():.9g}, {radius[valid].max():.9g}]")
    print("Max |(gamma_yy-gamma_zz)/gamma_yy| on valid samples: "
          f"{np.max(abs(profiles['angular_relative_difference'][valid])):.6e}")

    for component, numerical, analytical, scale, label in (
        ("grr", profiles["g_RR"], analytical_grr, "log", r"$g_{RR}$"),
        ("gtt", profiles["g_TT"], analytical_gtt, "linear", r"$g_{TT}$"),
    ):
        figure, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
        figure.suptitle(
            "Final axisymmetric collision: stationary spherical reconstruction at z=0\n"
            f"step={step}, t/M={time/MASS:g}; fixed reference M={MASS:g} "
            f"(saved bare-mass sum={bare_mass_sum:g}); no fit or time rescaling",
            fontsize=12,
        )
        for axis, title, upper in zip(axes, ("Inner range", "Full resolved exterior"),
                                      (min(INNER_RADIUS_LIMIT, outer), outer)):
            axis.plot(radius, numerical, color="tab:blue", linewidth=1.8,
                      label="Stationary reconstruction")
            axis.plot(radius, analytical, "--", color="tab:orange", linewidth=1.8,
                      label="Analytical Schwarzschild")
            axis.set_xlim(2.0, upper)
            axis.set_yscale(scale)
            axis.set_xlabel(r"Areal radius $R/M$ (assuming spherical symmetry)")
            axis.set_ylabel(label)
            axis.set_title(title)
            axis.grid(alpha=0.25, which="both")
            axis.legend()
        path = output_directory / f"final_{component}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        print(f"Saved {path}")
        for target in (3.0, 5.0, 10.0):
            # Do not select a misleading nearest sample across a masked gap.
            brackets = np.flatnonzero(comparison[:-1] & comparison[1:]
                                      & (radius[:-1] <= target) & (radius[1:] >= target))
            if not brackets.size:
                print(f"{component} at R/M={target:g}: unavailable (outside coverage or in a gap)")
                continue
            left = brackets[0]
            index = left + int(abs(radius[left + 1] - target) < abs(radius[left] - target))
            relative = numerical[index] / analytical[index] - 1.0
            print(f"R/M={radius[index]:.6f}: {component}={numerical[index]:.9g}, "
                  f"Schwarzschild={analytical[index]:.9g}, relative mismatch={relative:+.6e}")
    return profiles


if __name__ == "__main__":
    main()
