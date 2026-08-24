"""Compatibility exports for the spherical Cartoon diagnostics API."""

from JAX_BSSN.cartoon.spherical_symmetry.diagnostics import (
    cartoon_axis_output_fields,
    compute_cartoon_constraint_norms,
    compute_cartoon_constraints,
    compute_spherical_symmetry_norms,
)

__all__ = [
    "cartoon_axis_output_fields",
    "compute_cartoon_constraint_norms",
    "compute_cartoon_constraints",
    "compute_spherical_symmetry_norms",
]
