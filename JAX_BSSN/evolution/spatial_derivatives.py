"""Uniform Cartesian derivatives with parameter-aware boundaries and MAD stencils."""

from JAX_BSSN.evolution.derivatives import (
    diff1_field,
    diff2_field,
    diff1_upwind_field,
    diff6_field,
)


def _direction(field, direction):
    return direction % field.ndim - (field.ndim - 3)


def _boundary_codes(params, direction):
    # Import lazily: bssn.__init__ itself imports these derivative wrappers.
    from JAX_BSSN.bssn.variables import get_boundary_codes as boundary_codes

    return boundary_codes(params, direction)


def diff1_physical(field, direction, params):
    d = _direction(field, direction)
    return diff1_field(
        field, direction, params.dx, *_boundary_codes(params, d), mad_q=params.mad_q
    )


def diff2_physical(field, direction, params):
    d = _direction(field, direction)
    return diff2_field(
        field, direction, params.dx, *_boundary_codes(params, d), mad_q=params.mad_q
    )


def diff1_upwind_physical(field, rhs_coefficient, direction, params):
    d = _direction(field, direction)
    return diff1_upwind_field(
        field,
        rhs_coefficient,
        direction,
        params.dx,
        *_boundary_codes(params, d),
        mad_q=params.mad_q
    )


def ko_dissipation(field, params):
    """Sixth-difference Kreiss–Oliger filter on the uniform grid."""
    terms = []
    for d in range(3):
        h = params.dx
        term = h**5 * diff6_field(
            field, field.ndim - 3 + d, h, *_boundary_codes(params, d)
        )

        terms.append(term)
    return params.nu / 64 * sum(terms)
