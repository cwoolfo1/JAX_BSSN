"""Build and briefly evolve a single Schwarzschild puncture."""

import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from tqdm import tqdm

from JAX_BSSN.boundaries import PERIODIC_BC
from JAX_BSSN.bssn import BSSNParameters, BSSNVariables
from JAX_BSSN.evolve import rk4_step


def single_puncture_data(
    mass: float = 1.0,
    x_wind: float = 32.0,
    y_wind: float = 32.0,
    z_wind: float = 32.0,
    Nx: int = 64,
    Ny: int = 64,
    Nz: int = 64,
) -> tuple[jnp.ndarray, BSSNVariables]:
    """Return the radius and time-symmetric BSSN data for one puncture."""

    if Nx % 2 != 0 or Ny % 2 != 0 or Nz % 2 != 0:
        raise ValueError("Nx, Ny, and Nz must be even so R=0 lies between grid cells")

    dx = x_wind / Nx
    dy = y_wind / Ny
    dz = z_wind / Nz

    if not (math.isclose(dx, dy) and math.isclose(dx, dz)):
        raise ValueError("x_wind/Nx, y_wind/Ny, and z_wind/Nz must be equal")

    # With an even cell count the origin is the shared vertex between eight
    # cell centers. The coordinate endpoints are half a grid cell inside the
    # specified cell-face domain.
    x = (jnp.arange(Nx, dtype=jnp.float64) - (Nx - 1) / 2.0) * dx
    y = (jnp.arange(Ny, dtype=jnp.float64) - (Ny - 1) / 2.0) * dy
    z = (jnp.arange(Nz, dtype=jnp.float64) - (Nz - 1) / 2.0) * dz
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    R = jnp.sqrt(X**2 + Y**2 + Z**2)

    # Time-symmetric Schwarzschild data with W = psi**(-2). The conformal
    # metric is Cartesian and the physical metric is gamma_ij = W**(-2) delta_ij.
    W = (1.0 + mass / (2.0 * R)) ** (-2)
    shape = R.shape

    conformal_metric = (
        jnp.eye(3, dtype=W.dtype)[:, :, None, None, None]
        * jnp.ones((3, 3) + shape, dtype=W.dtype)
    )

    vars = BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=W,
        traceless_K=jnp.zeros_like(conformal_metric),
        trace_K=jnp.zeros(shape, dtype=W.dtype),
        conformal_connection=jnp.zeros((3,) + shape, dtype=W.dtype),
        lapse=W,
        shift=jnp.zeros((3,) + shape, dtype=W.dtype),
    )

    return R, vars


def main():
    """Run a short moving-puncture-style evolution on a Cartesian grid."""

    mass = 1.0
    x_wind = 20.0 * mass
    y_wind = 20.0 * mass
    z_wind = 20.0 * mass
    Nx = 160
    Ny = 160
    Nz = 160
    cfl = 0.2
    num_steps = 10

    dx = x_wind / Nx
    dy = y_wind / Ny
    dz = z_wind / Nz
    dt = cfl * dx

    R, vars = single_puncture_data(
        mass=mass,
        x_wind=x_wind,
        y_wind=y_wind,
        z_wind=z_wind,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
    )
    # JAX arrays are immutable, so this retains the exact unfiltered data while
    # ``vars`` is rebound to the boundary-filtered RK4 states below.
    initial_vars = vars

    # The finite-difference stencils and boundary conditions remain periodic.
    params = BSSNParameters(
        eta=2.0,
        kappa=0.002,
        nu=0.002,
        g=0.75,
        dx=dx,
        dt=dt,
        zero_shift=0,
        gauge=1,
        xl_bc=PERIODIC_BC,
        xr_bc=PERIODIC_BC,
        yl_bc=PERIODIC_BC,
        yr_bc=PERIODIC_BC,
        zl_bc=PERIODIC_BC,
        zr_bc=PERIODIC_BC,
    )

    minimum_radius = float(jnp.min(R))
    expected_minimum_radius = math.sqrt(dx**2 + dy**2 + dz**2) / 2.0
    samples_origin = bool(jnp.any(R == 0.0))
    initial_fields_finite = all(
        bool(jnp.all(jnp.isfinite(field))) for field in initial_vars
    )

    print("Single-puncture black-hole demo")
    print(f"M = {mass:.6f}")
    print(
        "Cell-face domain = "
        f"x:[{-x_wind / 2.0:.6f}, {x_wind / 2.0:.6f}], "
        f"y:[{-y_wind / 2.0:.6f}, {y_wind / 2.0:.6f}], "
        f"z:[{-z_wind / 2.0:.6f}, {z_wind / 2.0:.6f}]"
    )
    print(
        f"Grid = ({Nx}, {Ny}, {Nz}), "
        f"spacing = ({dx:.6f}, {dy:.6f}, {dz:.6f})"
    )
    print(f"CFL = {cfl:.6f}, dt = {dt:.6f}")
    print(
        f"Minimum R = {minimum_radius:.12f} "
        f"(expected {expected_minimum_radius:.12f})"
    )
    print(f"R=0 sampled: {samples_origin}")
    print(
        "Initial W range = "
        f"[{float(jnp.min(initial_vars.conformal_factor)):.12f}, "
        f"{float(jnp.max(initial_vars.conformal_factor)):.12f}]"
    )
    print(
        "Initial lapse range = "
        f"[{float(jnp.min(initial_vars.lapse)):.12f}, "
        f"{float(jnp.max(initial_vars.lapse)):.12f}]"
    )
    print(f"Initial fields finite: {initial_fields_finite}")

    center_x = Nx // 2
    center_y = Ny // 2
    center_z = Nz // 2
    face_centers = {
        "x-": (0, center_y, center_z),
        "x+": (-1, center_y, center_z),
        "y-": (center_x, 0, center_z),
        "y+": (center_x, -1, center_z),
        "z-": (center_x, center_y, 0),
        "z+": (center_x, center_y, -1),
    }

    print("Initial outer-face values:")
    for face, index in face_centers.items():
        print(
            f"  {face}: W={float(initial_vars.conformal_factor[index]):.12f}, "
            f"lapse={float(initial_vars.lapse[index]):.12f}"
        )

    del R, initial_vars

    progress_bar = tqdm(
        range(1, num_steps + 1),
        desc="Evolving single puncture",
        unit="step",
    )
    for step in progress_bar:
        vars = rk4_step(vars, params)
        jax.block_until_ready(vars)

        minimum_W = float(jnp.min(vars.conformal_factor))
        minimum_lapse = float(jnp.min(vars.lapse))
        maximum_shift = float(jnp.max(jnp.abs(vars.shift)))
        progress_bar.set_postfix(
            min_W=f"{minimum_W:.6e}",
            min_lapse=f"{minimum_lapse:.6e}",
            max_shift=f"{maximum_shift:.6e}",
        )

    final_fields_finite = all(
        bool(jnp.all(jnp.isfinite(field))) for field in vars
    )

    print("Final outer-face values:")
    for face, index in face_centers.items():
        print(
            f"  {face}: W={float(vars.conformal_factor[index]):.12f}, "
            f"lapse={float(vars.lapse[index]):.12f}"
        )

    print(f"Final fields finite: {final_fields_finite}")


if __name__ == "__main__":
    main()
