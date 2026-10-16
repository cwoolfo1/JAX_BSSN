"""Compare the final equatorial collision metric with Schwarzschild.

Run without arguments; edit the paths and plot limits below. On the positive
x axis at y=z=0, R=x*sqrt(gamma_yy) is the equatorial circumference radius.
It is the areal radius only under spherical symmetry. The height-function
transformation T=t-h(r), h'=b/D, also assumes stationarity of the metric and
radius map. This slice alone cannot establish either assumption.
The analytical mass comes from the common apparent horizon's physical area
on the full saved plane; that calculation does not assume stationarity.

gamma_ij = conformal_metric_ij / W**2
b = gamma_xj * beta^j, D = alpha**2 - gamma_ij * beta^i * beta^j
g_TT = -D, g_RR = (gamma_xx + b**2/D) / (dR/dr)**2
"""

import json
from pathlib import Path
import sys

import jax
import matplotlib
import numpy as np
import openpmd_api as io

jax.config.update("jax_enable_x64", True)  # The horizon solver refuses float32.

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEMO_DIRECTORY = Path(__file__).resolve().parent
if str(DEMO_DIRECTORY.parents[1]) not in sys.path:
    sys.path.insert(0, str(DEMO_DIRECTORY.parents[1]))

from JAX_BSSN.diagnostics.mesh_coordinates import mesh_axis_coordinates
from JAX_BSSN.utilities.apparent_horizon import find_apparent_horizon

INPUT_DIRECTORY = DEMO_DIRECTORY / "output"
OUTPUT_DIRECTORY = INPUT_DIRECTORY / "analytical_comparison_horizon_mass"
HORIZON_CENTER = (0.0, 0.0, 0.0)
HORIZON_THETA_NODES = 24
HORIZON_MAX_THETA_NODES = 96
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


def read_final_state(input_directory=INPUT_DIRECTORY):
    """Read the final y=0 plane for horizon finding and equatorial slicing."""
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
        names = ["lapse", "W", "K"] + [prefix + c
            for prefix in ("conformal_metric_", "traceless_K_")
            for c, _, _ in METRIC_COMPONENTS]
        records = {name: (meshes[name], meshes[name][scalar]) for name in names}
        records.update({"beta_" + c: (meshes["shift"], meshes["shift"][c]) for c in "xyz"})
        pending = {}
        for name, (mesh, record) in records.items():
            if any(not np.array_equal(mesh_axis_coordinates(mesh, record, i), coordinates[i])
                   for i in range(3)):
                raise ValueError(f"{name} is not on the same Cartesian grid as lapse")
            pending[name] = record.load_chunk()
        time = float(iteration.time)
        series.flush()
        fields = {name: np.array(array[:, 0, :], copy=True) for name, array in pending.items()}
    finally:
        series.close()

    # Divide by W**2 at native samples before interpolating the physical metric.
    gamma = np.empty((3, 3, len(x), len(z)))
    Kij = np.empty_like(gamma)
    for name, i, j in METRIC_COMPONENTS:
        conformal = fields["conformal_metric_" + name]
        gamma[i, j] = conformal / fields["W"]**2
        gamma[j, i] = gamma[i, j]
        Kij[i, j] = (fields["traceless_K_" + name] + conformal*fields["K"]/3) / fields["W"]**2
        Kij[j, i] = Kij[i, j]
    return {
        "x": x, "z": z, "gamma": gamma, "Kij": Kij, "alpha": fields["lapse"],
        "beta": np.stack([fields["beta_" + c] for c in "xyz"]),
    }, step, time, configuration


def equatorial_slice(state):
    """Interpolate the physical fields from the saved plane to z=0, x>0."""
    x = state["x"]
    iz, weights = equatorial_stencil(state["z"])
    gamma = state["gamma"][..., iz:iz + 4] @ weights
    alpha = state["alpha"][..., iz:iz + 4] @ weights
    beta = state["beta"][..., iz:iz + 4] @ weights
    positive = x > 0.0
    # Match the evolution diagnostic's four-cell outer boundary margin.
    interior = np.arange(len(x)) < len(x) - 4
    return {
        "r": x[positive], "alpha": alpha[positive],
        "gamma": gamma[:, :, positive], "beta": beta[:, positive],
        "interior": interior[positive],
    }


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
    state, step, time, configuration = read_final_state(input_directory)
    horizon = find_apparent_horizon(
        (state["x"], state["z"]), state["gamma"], state["Kij"],
        center=HORIZON_CENTER, n_theta=HORIZON_THETA_NODES,
        max_angular=HORIZON_MAX_THETA_NODES,
    )
    mass = horizon["mass"]
    fields = equatorial_slice(state)
    profiles = reconstruct_metric(**fields)
    R, valid = profiles["R"], profiles["valid"]
    radius = R / mass
    # The analytical exterior is independent of the numerical D=0 surface.
    exterior = profiles["outer_branch"] & (R > 2*mass)
    analytical_grr = np.full_like(R, np.nan)
    analytical_gtt = np.full_like(R, np.nan)
    analytical_grr[exterior] = 1.0 / (1.0 - 2*mass/R[exterior])
    analytical_gtt[exterior] = -(1.0 - 2*mass/R[exterior])
    comparison = valid & exterior
    if not np.any(comparison):
        raise ValueError("No resolved exterior samples for the Schwarzschild comparison")

    bare_mass_sum = sum(configuration["masses"])
    outer = np.max(radius[profiles["outer_branch"]])
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    print(f"Source: {Path(input_directory) / 'axisymmetric_blackhole_collision.h5'}")
    print(f"step={step}, t={time:g}, t/M={time/mass:g}; horizon-area mass M={mass:.12g}")
    print(f"Saved bare puncture mass sum={bare_mass_sum:g}; no mass fit or time rescaling.")
    print(f"Horizon area={horizon['area']:.12g}; coordinate radius range="
          f"[{horizon['radius'].min():.9g}, {horizon['radius'].max():.9g}]")
    print(f"Dimensionless expansion residual={horizon['residual']:.6e}; "
          f"independent audit={horizon['audit_residual']:.6e}")
    print(f"Relative angular mass change={horizon['mass_relative_change']:.6e}; "
          f"recentered by {horizon['recenter_shift'][2]:+.6e} in z")
    for polar, azimuthal, measured_mass, residual, audit, iterations in horizon["convergence"]:
        print(f"  {int(polar)}x{int(azimuthal)} angular nodes: M={measured_mass:.12g}, "
              f"residual={residual:.3e}, audit={audit:.3e}, "
              f"Newton iterations={int(iterations)}")
    for failure in horizon["search_failures"]:
        print(f"Additional seed failed: {failure}")
    print("Horizon mass is irreducible mass, identified with remnant mass for this nonspinning collision.")
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
            "Numerical vs Analytical Comparison at z=0\n",
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
            axis.set_xlabel(r"Areal radius $R/M$")
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
    archive = output_directory / "final_horizon.npz"
    np.savez(archive, **horizon, step=step, time=time,
             source=str(Path(input_directory) / "axisymmetric_blackhole_collision.h5"),
             convergence_columns=np.array(["n_theta", "n_phi", "mass", "residual",
                                           "audit_residual", "iterations"]),
             mass_definition="sqrt(area/(16*pi))")
    print(f"Saved {archive}")
    return profiles


if __name__ == "__main__":
    main()
