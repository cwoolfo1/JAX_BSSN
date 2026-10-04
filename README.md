# JAX BSSN

<p align="center">
  <img src="docs/images/JAX_BSSN_logo.png" alt="JAX BSSN logo" width="400">
</p>

JAX-BSSN is an autodifferentiable implementation of the BSSN evolution system
for numerical relativity. JAX-BSSN features Kreiss–Oliger dissipation, momentum
constraint damping, and fixed mesh refinement.

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[demos,test]"
```

Install a JAX build appropriate for the intended CPU or accelerator when the
default dependency does not match the target hardware.

## Supported demos

The supported physical examples are:

1. Gauge wave
2. Linear wave
3. Spherical Cartoon puncture

4. Exact electromagnetic pp-wave packet in Rosen coordinates

Each demo owns its initial data and run configuration. There is no generic
simulation CLI or installed initial-data dispatcher.

Run the compact Einstein--Maxwell propagation test and make comparison movies:

```bash
python demos/EM_waves/run.py
python demos/EM_waves/make_movies.py --input demos/EM_waves/output
```

See [EM_waves](demos/EM_waves/README.md) for the exact solution, spatial/time
refinement validation, and numerical-versus-exact movie descriptions.

Run the periodic analytic gauge wave:

```bash
python demos/gauge_wave/gauge_wave.py
```

Run the linear wave through a nested fixed-refinement hierarchy and write composite
openPMD output:

```bash
python demos/linear_wave/linear_wave.py
```

Run the single Schwarzschild puncture from its demo directory so snapshots and
the optional movie helper share one output path:

```bash
cd demos/single_puncture
python single_puncture.py
python make_movies.py  # optional; requires ffmpeg
```

Run the compact spherical Cartoon puncture and write its complete reflected
axis to openPMD:

```bash
python demos/cartoon_puncture/cartoon_puncture.py
```

Run the z-axis axisymmetric Cartoon puncture and write a signed x-z plane:

```bash
python demos/axisymmetric_cartoon_puncture/axisymmetric_cartoon_puncture.py
```

Run a black hole with Bowen--York momentum along the positive z axis, then
render grouped BSSN movies with a tracked puncture position:

```bash
cd demos/axisymmetric_bowen_york
python axisymmetric_bowen_york.py
python make_movies.py  # optional; requires ffmpeg
```

Run constraint-solved electromagnetic dipole data with either the first-order
staggered Yee solver or the second-order cell-centered wave solver:

```bash
python demos/EM_blackhole_formation_first_order/run_collapse.py
python demos/EM_blackhole_formation_second_order/run_collapse.py
```

Configure either demo by editing `simulation_parameters.py` in its folder.
It contains pulse, domain, initial metric solver, BSSN evolution, and output
settings; there are no command-line configuration options. `initial_pulse.py`
creates the electromagnetic fields, `initial_metric.py` solves the Hamiltonian
constraint, and `run_collapse.py` assembles and evolves the coupled state.
Each folder also owns `collapse_io.py` and `make_movies.py`.

Both demos solve the rho-weighted cylindrical Hamiltonian constraint on the
positive-rho, full-z Cartoon plane, with regular zero flux at the axis and fixed
`u=0` outer boundaries. Defaults are amplitude 0.913, width 1, pulse-center
parameter 0, a 200-by-400 uniform grid, and final time 100. Cell edges span rho
from 0 to 12 and z from -12 to 12. The physical Cartesian spacing is 0.06 in
all directions, and `CFL=0.2` gives a timestep of 0.012.

Both demos start fresh, record field snapshots and constraint norms, and write
the final native state to `final_checkpoint.npz`. There is no restart or periodic
checkpoint option. Evolution stops at the last whole step at or before
`FINAL_TIME`, or reports `failed_nonfinite` when detected. Completion does not
establish black-hole formation or physical settling. Set `OUTPUT_DIR` to a
fresh directory to avoid overwriting existing run files; the default is
`output_uniform/` beside the script.

First-order openPMD fields contain physical contravariant `D` and `B`;
second-order fields contain physical covariant `E` and `B`, with projected time
derivatives retained in checkpoints. Each `make_movies.py` renders the saved
physical coordinates and computes electromagnetic energy density using the
appropriate spatial metric contraction. Checkpoints and summaries record the uniform grid parameters.
Readers reject legacy mapped checkpoints and meshes. Compiled JAX programs are cached in `.jax_cache/` beside the runner,
unless `JAX_COMPILATION_CACHE_DIR` is set.


The default production runs compile substantial JAX kernels. The
[demo guide](docs/demos.rst) describes the evolution and configuration options.

## Package structure

```text
JAX_BSSN/
├── bssn/          # state, tensor algebra, geometry, equations, raw constraints
├── cartoon/       # spherical and axisymmetric reconstruction/evolution
├── evolution/     # derivatives, boundaries, RHS assembly, projections, RK4
├── fmr/           # nested-patch geometry, transfers, stage-synchronous RK4
├── diagnostics/   # reductions, reporting, plotting, openPMD
└── utilities/     # reusable initial-data and numerical helper routines
```

Physical initial data lives under `demos/`, not in the installed source
package. Gauge- and linear-wave tests use `tests/initial_data.py` and do not
import executable demos.

## Tests

Install test dependencies and run the complete suite with:

```bash
python -m pip install -e ".[test]"
python -m pytest
```
