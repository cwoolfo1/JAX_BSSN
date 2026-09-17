"""Edit these globals, then run ``python run_collapse.py``.

Uniform physical spacing is DOMAIN_HALF_WIDTH / NUM_RADIAL_POINTS.
The default box is rho in [0,12] and z in [-12,12].
"""

import math
from pathlib import Path

from JAX_BSSN.bssn.variables import BSSNParameters
from JAX_BSSN.evolution.boundaries import PERIODIC_BC, SOMMERFELD_BC

# Electromagnetic pulse.
AMPLITUDE = 0.913
WIDTH = 1.0
RADIAL_CENTER = 0.0

# Domain and evolution duration.
DOMAIN_HALF_WIDTH = 12.0
NUM_RADIAL_POINTS = 600
NUM_Z_POINTS = 2 * NUM_RADIAL_POINTS
CFL = 0.2
FINAL_TIME = 70.0

# Initial metric solver.
NEWTON_TOLERANCE = 5.0e-14
MAX_NEWTON_ITERATIONS = 500
CG_TOLERANCE = 1.0e-14
MAX_CG_ITERATIONS = 10000

# BSSN evolution.
KAPPA = 0.002
ETA = 2.0
NU = 0.02
GAMMA_DRIVER = 0.75
ZERO_SHIFT = 1
GAUGE = 1  # 1 = 1+log slicing; 0 = harmonic slicing.
MAD_Q = 1.0  # The axisymmetric Cartoon solver requires this to stay at 1.

XL_BC = PERIODIC_BC
# axisymmetry populates the left boundary of the rho domain so this
# is just a placeholder.

XR_BC = SOMMERFELD_BC
ZL_BC = SOMMERFELD_BC
ZR_BC = SOMMERFELD_BC

YL_BC = PERIODIC_BC
YR_BC = PERIODIC_BC
# The y boundaries are periodic because the Cartoon method uses a 3D grid.

# Output and diagnostics. Existing run files are never overwritten.
OUTPUT_DIR = Path(__file__).resolve().parent / "output_uniform"
SNAPSHOT_COUNT = 60
DIAGNOSTIC_INTERVAL = 0.5
SHOW_PROGRESS = True


def grid_spacing(num_radial_points, num_z_points, domain_half_width):
    """Validate the cell counts and return the common physical rho/z spacing."""

    if num_radial_points < 6:
        raise ValueError("num_radial_points must be at least six")
    if num_z_points < 10 or num_z_points % 2:
        raise ValueError("num_z_points must be even and at least ten")
    if not math.isfinite(domain_half_width) or domain_half_width <= 0:
        raise ValueError("domain_half_width must be finite and positive")
    return domain_half_width / num_radial_points


def axisymmetric_parameters(
    num_radial_points: int,
    num_z_points: int,
    domain_half_width: float,
    dt: float,
) -> BSSNParameters:
    """Use rho extent domain_half_width and z half-extent num_z_points*dx/2."""

    dx = grid_spacing(num_radial_points, num_z_points, domain_half_width)

    z_min = -(num_z_points - 1) * dx / 2.0
    return BSSNParameters(
        eta=ETA,
        kappa=KAPPA,
        nu=NU,
        g=GAMMA_DRIVER,
        dx=dx,
        dt=dt,
        zero_shift=ZERO_SHIFT,
        gauge=GAUGE,
        xl_bc=XL_BC,
        xr_bc=XR_BC,
        yl_bc=YL_BC,
        yr_bc=YR_BC,
        zl_bc=ZL_BC,
        zr_bc=ZR_BC,
        x_min=-3.5 * dx,
        y_min=-4.0 * dx,
        z_min=z_min,
        mad_q=MAD_Q,
    )
