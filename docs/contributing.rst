Contributing
============

Change checklist
----------------

* Existing tests pass and new behavior has focused regression coverage.
* ``BSSNVariables`` field order and array layouts remain compatible.
* The denominator-free :math:`W^2R_{ij}` and
  :math:`W^2D_iD_j\alpha` sources remain intact unless a reformulation is the
  explicit purpose of the change.
* The conformal-factor floor protects inverse powers without clamping evolved
  ``W``.
* Finite-difference coefficients, leading-axis behavior, MAD weighting, and
  Kreiss--Oliger dissipation are unchanged unless documented and validated.
* Sommerfeld stays an RHS treatment; Super-Gaussian damping stays a state
  filter.
* Determinant-one and trace-free projections retain their RK stage placement.
* FMR guard filling, inclusive bounds, active slices, injection, and shared
  timestep retain their current semantics.
* openPMD names, component orientation, mesh geometry, ghost stripping, and
  composite overwrite behavior remain compatible.
* Physical initial data remains demo-local, and tests use test-local helpers.
* Exactly the gauge-wave, linear-wave, and single-puncture demos are presented
  as supported examples.
* HTML and link-check documentation builds pass with warnings treated as
  errors.

Scope communication
-------------------

Describe experimental features as experimental and unimplemented features as
limitations. In particular, do not present multiple refinement patches,
subcycling, dynamic regridding, matter sources, or a computed Gamma constraint
as current capabilities.

Review guidance
---------------

Keep pull requests narrow enough that numerical changes can be separated from
file moves and documentation changes. Include the commands used for focused
tests, full tests, convergence checks, and parity comparisons. If output
changes at roundoff rather than bitwise, report the maximum absolute and
relative differences rather than describing the result only as passing.
