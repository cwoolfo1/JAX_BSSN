"""Spatial truncation convergence for axisymmetric Maxwell on Schwarzschild."""

import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.cartoon.axisymmetry.reconstruction import (
    AXISYMMETRIC_GHOST_CELLS,
)
from JAX_BSSN.evolution.boundaries import PERIODIC_BC, SOMMERFELD_BC
from JAX_BSSN.EM.cartoon.axisymmetry import (
    compact_axisymmetric_wave,
    compute_axisymmetric_prescribed_rhs,
)
from JAX_BSSN.EM.schwarzschild import (
    SchwarzschildParameters,
    exact_wave_at_time,
    exact_wave_coordinate_rhs,
    schwarzschild_background,
    schwarzschild_exact_template,
)


DOMAIN_HALF_WIDTH = 12.0
MASS = 1.0
OMEGA = 0.4
SAMPLE_TIME = 0.37


def _axisymmetric_parameters(num_rho):
    num_z = 2 * num_rho
    dx = DOMAIN_HALF_WIDTH / num_rho
    return BSSNParameters(
        dx=dx,
        dt=0.1 * dx,
        nu=0.0,
        xl_bc=PERIODIC_BC,
        xr_bc=SOMMERFELD_BC,
        yl_bc=PERIODIC_BC,
        yr_bc=PERIODIC_BC,
        zl_bc=SOMMERFELD_BC,
        zr_bc=SOMMERFELD_BC,
        x_min=-3.5 * dx,
        y_min=-4.0 * dx,
        z_min=-(num_z - 1) * dx / 2.0,
        mad_q=1.0,
    )


def _compact_schwarzschild_template(num_rho, params, background_params):
    signed_params = params._replace(
        x_min=-(num_rho - 0.5) * params.dx,
        y_min=0.0,
    )
    signed_background = background_params._replace(
        x_min=signed_params.x_min,
        y_min=0.0,
    )
    signed_template = schwarzschild_exact_template(
        (2 * num_rho, 1, 2 * num_rho),
        signed_params,
        signed_background,
        omega=OMEGA,
        ell=1,
        m=0,
    )
    return compact_axisymmetric_wave(signed_template)


def _relative_cylindrical_l2(numerical, exact, mask, rho):
    error_squared = jnp.where(
        mask[None, ...],
        rho[None, ...] * jnp.abs(numerical - exact) ** 2,
        0.0,
    )
    exact_squared = jnp.where(
        mask[None, ...],
        rho[None, ...] * jnp.abs(exact) ** 2,
        0.0,
    )
    return float(jnp.sqrt(jnp.sum(error_squared) / jnp.sum(exact_squared)))


def _axisymmetric_schwarzschild_rhs_errors(num_rho):
    num_z = 2 * num_rho
    params = _axisymmetric_parameters(num_rho)
    background_params = SchwarzschildParameters(
        mass=MASS,
        dx=params.dx,
        x_min=params.x_min,
        y_min=params.y_min,
        z_min=params.z_min,
    )
    template = _compact_schwarzschild_template(
        num_rho, params, background_params
    )
    exact = exact_wave_at_time(template, SAMPLE_TIME, OMEGA)
    numerical_rhs = compute_axisymmetric_prescribed_rhs(
        exact,
        SAMPLE_TIME,
        params,
        schwarzschild_background,
        background_params,
    )
    exact_rhs = exact_wave_coordinate_rhs(template, SAMPLE_TIME, OMEGA)
    jax.block_until_ready(numerical_rhs)

    # Measure the semi-discrete truncation error away from excision and the
    # outer faces.  The cylindrical rho weight is the axisymmetric volume
    # measure common to both resolutions.
    rho = (jnp.arange(num_rho, dtype=jnp.float64) + 0.5) * params.dx
    z = params.z_min + params.dx * jnp.arange(num_z, dtype=jnp.float64)
    R, Z = jnp.meshgrid(rho, z, indexing="ij")
    radius = jnp.sqrt(R**2 + Z**2)
    validation_mask = (radius >= 5.0) & (radius <= 9.0)

    physical = (
        slice(None),
        slice(AXISYMMETRIC_GHOST_CELLS, None),
        0,
        slice(None),
    )
    electric_error = _relative_cylindrical_l2(
        numerical_rhs.electric_field[physical],
        exact_rhs.electric_field[physical],
        validation_mask,
        R,
    )
    magnetic_error = _relative_cylindrical_l2(
        numerical_rhs.magnetic_field[physical],
        exact_rhs.magnetic_field[physical],
        validation_mask,
        R,
    )
    electric_dot_error = _relative_cylindrical_l2(
        numerical_rhs.electric_field_dot[physical],
        exact_rhs.electric_field_dot[physical],
        validation_mask,
        R,
    )
    magnetic_dot_error = _relative_cylindrical_l2(
        numerical_rhs.magnetic_field_dot[physical],
        exact_rhs.magnetic_field_dot[physical],
        validation_mask,
        R,
    )
    return (
        params.dx,
        electric_error,
        electric_dot_error,
        magnetic_error,
        magnetic_dot_error,
    )


def test_axisymmetric_schwarzschild_maxwell_rhs_is_second_order():
    (
        coarse_dx,
        coarse_electric,
        coarse_electric_dot,
        coarse_magnetic,
        coarse_magnetic_dot,
    ) = _axisymmetric_schwarzschild_rhs_errors(8)
    (
        fine_dx,
        fine_electric,
        fine_electric_dot,
        fine_magnetic,
        fine_magnetic_dot,
    ) = _axisymmetric_schwarzschild_rhs_errors(12)
    spacing_ratio = coarse_dx / fine_dx
    electric_order = math.log(coarse_electric / fine_electric) / math.log(
        spacing_ratio
    )
    magnetic_order = math.log(coarse_magnetic / fine_magnetic) / math.log(
        spacing_ratio
    )
    electric_dot_order = math.log(
        coarse_electric_dot / fine_electric_dot
    ) / math.log(spacing_ratio)
    magnetic_dot_order = math.log(
        coarse_magnetic_dot / fine_magnetic_dot
    ) / math.log(spacing_ratio)

    print(
        "Axisymmetric Schwarzschild Maxwell spatial convergence: "
        f"E=({coarse_electric:.12e}, {fine_electric:.12e}), "
        f"B=({coarse_magnetic:.12e}, {fine_magnetic:.12e}), "
        f"dotE=({coarse_electric_dot:.12e}, {fine_electric_dot:.12e}), "
        f"dotB=({coarse_magnetic_dot:.12e}, {fine_magnetic_dot:.12e}), "
        f"orders=({electric_order:.8f}, {magnetic_order:.8f}, "
        f"{electric_dot_order:.8f}, {magnetic_dot_order:.8f})"
    )
    assert electric_order >= 1.8
    assert magnetic_order >= 1.8
    assert electric_dot_order >= 1.8
    assert magnetic_dot_order >= 1.8
