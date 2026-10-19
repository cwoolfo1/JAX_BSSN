"""Compare the final spherical Cartoon metric with Schwarzschild.

Run without arguments; edit the paths and mass below. Assuming stationarity,
T = t - h(r), h' = beta_r / D removes the ADM time-radial cross term, where
D = alpha**2 - beta_r * beta**r. The areal map R(r) must also be stationary.
This diagnostic does not establish stationarity of an evolved snapshot.
"""

from pathlib import Path
import sys

import matplotlib
import numpy as np
import openpmd_api as io

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEMO_DIRECTORY = Path(__file__).resolve().parent
PACKAGE_ROOT = DEMO_DIRECTORY.parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from JAX_BSSN.diagnostics.mesh_coordinates import mesh_axis_coordinates

INPUT_PATH = DEMO_DIRECTORY / "output" / "cartoon_puncture.h5"
OUTPUT_DIRECTORY = INPUT_PATH.parent / "analytical_comparison"
MASS = 1.0  # Initial puncture mass; the archive does not store this parameter.
INNER_RADIUS_LIMIT = 10.0  # R/M


def read_final_state(path=INPUT_PATH):
    """Read only the final iteration, on its native positive Cartesian axis."""
    series = io.Series(str(path), io.Access.read_only,
                       '{"defer_iteration_parsing": true}')
    try:
        step = max(series.iterations)
        iteration = series.iterations[step]
        iteration.open()
        meshes = iteration.meshes
        names = ("lapse", "W", "conformal_metric_xx",
                 "conformal_metric_yy", "conformal_metric_zz")
        records = {name: (meshes[name], meshes[name][next(iter(meshes[name]))])
                   for name in names}
        records["beta"] = (meshes["shift"], meshes["shift"]["x"])
        mesh, component = records["lapse"]
        x = mesh_axis_coordinates(mesh, component, 0)
        pending = {}
        for name, (mesh, component) in records.items():
            if component.shape != [len(x), 1, 1] or not np.array_equal(
                mesh_axis_coordinates(mesh, component, 0), x
            ):
                raise ValueError(f"{name} is not on the same 1D Cartoon axis as lapse")
            pending[name] = component.load_chunk()
        time = float(iteration.time)
        series.flush()
        positive = x > 0.0
        fields = {name: np.array(array[:, 0, 0], dtype=float, copy=True)[positive]
                  for name, array in pending.items()}
        fields["r"] = x[positive]
    finally:
        series.close()
    return fields, step, time


def reconstruct_metric(fields):
    """Diagonalize the radial ADM block and convert to exact areal radius."""
    r, alpha, W = fields["r"], fields["lapse"], fields["W"]
    beta = fields["beta"]  # Saved shift is contravariant beta^r on the +x axis.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        gamma_rr = fields["conformal_metric_xx"] / W**2
        gamma_T = (fields["conformal_metric_yy"] + fields["conformal_metric_zz"]) / (2*W**2)
        beta_r = gamma_rr * beta
        D = alpha**2 - beta_r * beta
        R = r * np.sqrt(gamma_T)

        dr = r[1] - r[0]
        dR_dr = np.full_like(R, np.nan)
        dR_dr[2:-2] = (R[:-4] - 8*R[1:-3] + 8*R[3:-1] - R[4:]) / (12*dr)
        g_rr = gamma_rr + beta_r**2 / D
        g_RR = g_rr / dR_dr**2

    # Exclude unresolved origin stencils and the four-cell outer boundary margin.
    # Walk inward from the outer domain, never sorting across the puncture throat.
    supported = np.zeros(r.size, dtype=bool)
    supported[2:-4] = True
    outward = supported & np.isfinite(R) & np.isfinite(dR_dr) & (dR_dr > 0.0)
    branch = np.zeros_like(supported)
    endpoints = np.flatnonzero(supported)
    if endpoints.size:
        end = endpoints[-1]
        breaks = np.flatnonzero(~outward[:end + 1])
        start = breaks[-1] + 1 if breaks.size else 0
        branch[start:end + 1] = True

    valid = (branch & (W > 0.0) & (gamma_rr > 0.0) & (gamma_T > 0.0)
             & np.isfinite(gamma_rr) & np.isfinite(gamma_T)
             & np.isfinite(D) & (D > 0.0) & np.isfinite(g_RR) & (g_RR > 0.0))
    return {"R": R, "dR_dr": dR_dr, "D": D, "outer_branch": branch,
            "valid": valid, "g_RR": np.where(valid, g_RR, np.nan),
            "g_TT": np.where(valid, -D, np.nan)}


def main(input_path=INPUT_PATH, output_directory=OUTPUT_DIRECTORY, mass=MASS):
    fields, step, time = read_final_state(input_path)
    profiles = reconstruct_metric(fields)
    R = profiles["R"]
    radius = R / mass
    exterior = profiles["outer_branch"] & (R > 2*mass)
    analytical_rr = np.full_like(R, np.nan)
    analytical_tt = np.full_like(R, np.nan)
    analytical_rr[exterior] = 1 / (1 - 2*mass/R[exterior])
    analytical_tt[exterior] = -(1 - 2*mass/R[exterior])
    comparison = exterior & profiles["valid"]
    if not np.any(comparison):
        raise ValueError("No resolved exterior samples for the stationary comparison")

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    outer = np.max(radius[profiles["outer_branch"]])
    print(f"Source: {input_path}; step={step}, t/M={time/mass:g}")
    print(f"Initial puncture mass M={mass:g}; Schwarzschild horizon R/M=2")
    print("Reconstruction assumes a stationary snapshot and areal-radius map; no fit or rescaling.")
    print(f"Valid exterior samples: {comparison.sum()}/{R.size}")

    for name, numerical, analytical, scale, label in (
        ("grr", profiles["g_RR"], analytical_rr, "log", r"$g_{RR}$"),
        ("gtt", profiles["g_TT"], analytical_tt, "linear", r"$g_{TT}$"),
    ):
        figure, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
        figure.suptitle("Final spherical black hole: numerical metric vs. Schwarzschild\n"
                       f"Stationary reconstruction; M={mass:g}, step={step}, t/M={time/mass:g}")
        for axis, title, upper in zip(axes, ("Inner range", "Full resolved range"),
                                      (min(INNER_RADIUS_LIMIT, outer), outer)):
            axis.plot(radius, numerical, color="tab:blue", linewidth=1.8,
                      label="Numerical stationary reconstruction")
            axis.plot(radius, analytical, "--", color="tab:orange", linewidth=1.8,
                      label="Analytical Schwarzschild")
            axis.set_xlim(2.0, upper)
            axis.set_yscale(scale)
            axis.set_xlabel(r"Areal radius $R/M$")
            axis.set_ylabel(label)
            axis.set_title(title)
            axis.grid(alpha=0.25, which="both")
            axis.legend(fontsize=9)
        path = output_directory / f"final_{name}.png"
        figure.savefig(path, dpi=180, metadata={"Description":
            f"Source={input_path}; step={step}; time={time}; mass={mass}; stationary_assumption=True"})
        plt.close(figure)
        print(f"Saved {path}")

        for target in (3.0, 5.0, 10.0):
            # Report only targets bracketed by adjacent valid samples, not gaps.
            brackets = np.flatnonzero(comparison[:-1] & comparison[1:]
                                      & (radius[:-1] <= target) & (radius[1:] >= target))
            if not brackets.size:
                print(f"{name}, R/M={target:g}: unavailable (outside valid coverage or masked)")
                continue
            left = brackets[0]
            index = left + int(abs(radius[left + 1] - target) < abs(radius[left] - target))
            relative = numerical[index] / analytical[index] - 1.0
            print(f"{name}, R/M={radius[index]:.6f}: numerical={numerical[index]:.9g}, "
                  f"Schwarzschild={analytical[index]:.9g}, relative mismatch={relative:+.6e}")
    return profiles


if __name__ == "__main__":
    main()
