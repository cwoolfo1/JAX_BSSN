# Einstein–Maxwell collapse: first order

Edit `simulation_parameters.py`, then run `python run_collapse.py` in this
folder. The demo uses the staggered Yee Maxwell formulation coupled to BSSN.
It starts fresh and writes to `output_uniform/`, refusing to overwrite existing
run files. Run `python make_movies.py` to render the saved fields.

The current editable settings use 600 radial and 1200 z cells, with cell edges
at rho = 0 and 12 and z = -12 and 12. Physical spacing is `dx=dy=dz=0.02`;
`CFL=0.2` gives `dt=0.004`. The z cell count can be chosen independently,
subject to the even-count and minimum-size checks; its extent is `Nz*dx`.
Cartoon radial ghosts and native Yee offsets use this same spacing.

The pulse defaults remain amplitude 0.913, width 1, and radial center 0.
`initial_metric.py` solves the rho-weighted cylindrical Hamiltonian constraint
with regular zero flux at the axis and fixed `u=0` outer boundaries. The Newton
solve retains diagonal CG preconditioning and explicit convergence checks.
Physical metric factors and cylindrical radial volume weights remain in the
coupled evolution and diagnostics. A completed run alone does not establish
black-hole formation or resolved critical collapse.

Snapshots save physical contravariant D and B. Checkpoints retain the complete
native BSSN and Maxwell state. Snapshots also save `rho_EM` from the same
quadratic moments used for the gravitational source. The energy movie reads
this record; for older snapshots it falls back to an approximate density from
the averaged D/B vectors, which can underestimate grid-scale energy.
Evolution ends at the last whole step at or
before `FINAL_TIME` (currently 70), or reports `failed_nonfinite` when detected.
The first-order runner checks finiteness each step and also saves the last
finite state to `last_finite_checkpoint.npz`.
There is no restart or periodic checkpoint option.

The separate [critical-collapse study runner](../em_critical_collapse/README.md)
adds periodic checkpoints, exact recovery, dense central proper-time and EM
invariant histories, self-refinement comparisons, and comparison with the paper.
It runs this same first-order evolution in isolated output directories and
records explicit experiment settings. The ongoing fixed-amplitude study uses
eta=.913, zero shift, and the multipole initial-metric boundary; its status is
documented in `../em_critical_collapse/results/first_order_progress.md`.

openPMD output uses uniform Cartesian geometry. Readers reject old mapped
meshes and checkpoints. Legacy uniform checkpoints with absent or null mapping
fields remain readable; any stored transverse spacing must equal `dx`.
Historical output directories are left untouched.

Compiled JAX programs are cached in `.jax_cache/` beside the runner unless
`JAX_COMPILATION_CACHE_DIR` is set. Regression coverage includes uniform
coordinates, derivative convergence, cylindrical Hamiltonian solves, checkpoint
compatibility, and short coupled runs in `tests/EM/`.

## Discrete electromagnetic coupling

The matter source forms native Yee products before spatial interpolation:
`<dd>`, `<bb>`, and `<db>`. A shared corner quadrature retains the squares of
alternating fields, preserves positive energy for positive definite metrics,
and supplies the full stress tensor and momentum density. Time centering of
the doubled leapfrog fields is unchanged. The conformal metric is normalized
at cell centers; covariant E/H use the transpose of those same corner gathers
with native dual volumes. This makes the interior constitutive operator the
derivative of the discrete electromagnetic Hamiltonian, including shift.
Radiative outer halos remain a boundary closure, not a closed energy system.

Maxwell evolution on the rho-z plane uses compatible cylindrical Yee curls
and flux divergences. Radial corner weights are annular half-cell volumes;
the axial magnetic degree of freedom at rho=0 has dual area `dx**2/8` and
the regular axis curl is `-4 E_phi(dx/2)/dx`. All native cylindrical fields
lie at y=0. BSSN still uses Cartoon reconstruction. Its matter support is
constructed by convex interpolation and rotation of the quadratic moments,
with squared radial falloff in the outer buffer.

These changes remove the reproduced periodic sheared-metric growing mode and
the loss of checkerboard-field energy. They do not establish stability of a
full nonlinear collapse run or of the radiative boundary treatment. Existing
checkpoint shapes remain readable, but restarting old data would use the new
spatial discretization and would not reproduce the historical trajectory.
