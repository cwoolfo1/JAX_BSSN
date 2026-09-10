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
3. Single puncture
4. Spherical Cartoon puncture
5. Axisymmetric Cartoon puncture
6. Axisymmetric boosted Bowen--York puncture
7. First-order Einstein--Maxwell black-hole formation candidate
8. Second-order Einstein--Maxwell black-hole formation candidate

Each demo owns its initial data and run configuration. There is no generic
simulation CLI or installed initial-data dispatcher.

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

Run the same constraint-solved electromagnetic dipole data with either the
first-order staggered Yee solver or the second-order cell-centered wave solver:

```bash
python demos/EM_blackhole_formation_first_order/EM_blackhole_formation_first_order.py
python demos/EM_blackhole_formation_second_order/EM_blackhole_formation_second_order.py
```

Each demo owns an identical local copy of the physical initial-data routines
and writes to an `output/` directory beside its script by default. The
first-order openPMD output contains physical `D` and `B`; the second-order
output contains `E` and `B`. The default amplitude remains a
literature-informed candidate until the medium/high campaign is completed.

The initial-data phase solves the rho-weighted Hamiltonian constraint directly
on the positive-rho, full-z Cartoon plane. It uses a conservative radial flux,
regular zero flux at the axis, and fixed `u=0` outer-rho and outer-z rows. Both
demos evolve the Gamma-driver shift, write rolling restart checkpoints, and
test a converged even-Legendre apparent horizon for persistence and exterior
settling. Existing campaign files are never overwritten; choose a fresh
`--output-dir` or continue `--restart path/to/rolling_checkpoint.npz`.

After a settled formation run, the demo evolves a mass-matched Schwarzschild
puncture with the same gauge and grid. Compare the final lapse and conformal
factor with:

```bash
python demos/plot_em_blackhole_schwarzschild.py \
  --formation path/to/collapse_high \
  --reference path/to/collapse_high/schwarzschild_reference \
  --metadata path/to/collapse_high/run_summary.json \
  --output path/to/collapse_high/schwarzschild_comparison.png
```

The default production runs compile substantial JAX kernels. The
[demo guide](docs/demos.rst) includes smaller smoke configurations.

Once all four medium/high runs and their automatic Schwarzschild controls are
settled, generate every overlay, the CSV/Markdown summary table, and the
machine-readable acceptance report together:

```bash
python demos/plot_em_blackhole_campaign.py \
  --first-medium path/to/first_order/collapse_medium \
  --first-high path/to/first_order/collapse_high \
  --second-medium path/to/second_order/collapse_medium \
  --second-high path/to/second_order/collapse_high \
  --output-dir path/to/campaign_plots
```

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
