import importlib.util
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from JAX_BSSN.bssn.variables import BSSNParameters
from JAX_BSSN.diagnostics.apparent_horizon import (
    AxisymmetricHorizon,
    find_axisymmetric_apparent_horizon,
)
from JAX_BSSN.diagnostics.openpmd import OpenPMDWriter
from JAX_BSSN.EM.collapse import (
    _schwarzschild_state,
    load_collapse_checkpoint,
    run_schwarzschild_reference,
    settling_criteria,
    write_collapse_checkpoint,
)
from JAX_BSSN.EM.first_order.variables import DensitizedMaxwellState
from JAX_BSSN.EM.second_order.variables import EMVariables
from JAX_BSSN.EM.variables import EinsteinMaxwellVariables
from JAX_BSSN.evolution.boundaries import PERIODIC_BC, SOMMERFELD_BC


ROOT = Path(__file__).resolve().parents[2]


def _parameters(num_radial_points, num_z_points, rho_max=4.0):
    dx = rho_max / num_radial_points
    return BSSNParameters(
        dx=dx,
        dt=0.1 * dx,
        zero_shift=0,
        gauge=1,
        xl_bc=PERIODIC_BC,
        xr_bc=SOMMERFELD_BC,
        yl_bc=PERIODIC_BC,
        yr_bc=PERIODIC_BC,
        zl_bc=SOMMERFELD_BC,
        zr_bc=SOMMERFELD_BC,
        x_min=-3.5 * dx,
        y_min=-4.0 * dx,
        z_min=-(num_z_points - 1) * dx / 2.0,
        mad_q=1.0,
    )


def test_apparent_horizon_recovers_schwarzschild_and_rejects_minkowski():
    errors = []
    for num_radial_points in (32, 64):
        num_z_points = 2 * num_radial_points
        params = _parameters(num_radial_points, num_z_points)
        schwarzschild = _schwarzschild_state(
            1.0, params, num_radial_points, num_z_points
        )
        horizon = find_axisymmetric_apparent_horizon(
            schwarzschild,
            params,
            initial_coefficients=np.asarray((0.5, 0.0, 0.0, 0.0)),
        )

        assert horizon is not None
        errors.append(abs(horizon.irreducible_mass - 1.0))
        assert abs(horizon.circumference_ratio - 1.0) < 2.0e-2
        assert horizon.expansion_linf * horizon.equatorial_radius < 5.0e-2

    assert errors[1] < errors[0]

    params = _parameters(32, 64)
    minkowski = _schwarzschild_state(0.0, params, 32, 64)
    assert find_axisymmetric_apparent_horizon(minkowski, params) is None


@pytest.mark.parametrize(
    ("formulation", "em_type"),
    (("first_order", DensitizedMaxwellState), ("second_order", EMVariables)),
)
def test_collapse_checkpoint_round_trip(tmp_path, formulation, em_type):
    params = _parameters(8, 16)
    bssn = _schwarzschild_state(1.0, params, 8, 16)
    em_field = jnp.arange(3 * 12 * 16, dtype=jnp.float64).reshape((3, 12, 1, 16))
    em = em_type(*(em_field + index for index in range(4)))
    state = EinsteinMaxwellVariables(bssn=bssn, em=em)
    u = jnp.arange(8 * 16, dtype=jnp.float64).reshape((8, 16))
    path = tmp_path / "rolling_checkpoint.npz"

    write_collapse_checkpoint(
        path,
        state,
        u,
        [1.0, 1.0e-12],
        formulation,
        17,
        0.34,
        params,
        {"amplitude": 0.08},
    )
    restored, restored_u, residual_history, metadata = load_collapse_checkpoint(
        path, formulation
    )

    for expected, actual in zip(state.bssn, restored.bssn):
        np.testing.assert_array_equal(actual, expected)
    for expected, actual in zip(state.em, restored.em):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(restored_u, u)
    assert residual_history == [1.0, 1.0e-12]
    assert metadata["step"] == 17
    assert metadata["time"] == 0.34


def test_settling_requires_persistence_and_all_exterior_thresholds():
    params = _parameters(8, 16)
    fields = (np.ones((8, 16)), np.ones((8, 16)))
    horizon = AxisymmetricHorizon(
        coefficients=np.asarray((0.5, 0.0, 0.0, 0.0)),
        expansion_l2=0.0,
        expansion_linf=0.0,
        area=16.0 * np.pi,
        irreducible_mass=1.0,
        circumference_ratio=1.0,
        polar_radius=0.5,
        equatorial_radius=0.5,
    )
    samples = [
        {
            "time": time,
            "diagnostic_interval": 0.5,
            "horizon": horizon,
            "exterior_em_energy": 1.0e-5,
            "fields": fields,
        }
        for time in np.arange(0.0, 25.5, 0.5)
    ]

    assert settling_criteria(samples, 0.0, params)["settled"]
    samples[-1] = dict(samples[-1], exterior_em_energy=2.0e-4)
    assert not settling_criteria(samples, 0.0, params)["settled"]


def test_schwarzschild_reference_checkpoint_is_restartable(tmp_path):
    params = _parameters(32, 64)
    output_dir = tmp_path / "reference"
    first_summary = run_schwarzschild_reference(
        1.0,
        params,
        32,
        64,
        0.0,
        output_dir,
        show_progress=False,
    )
    second_summary = run_schwarzschild_reference(
        1.0,
        params,
        32,
        64,
        0.0,
        output_dir,
        show_progress=False,
    )

    assert Path(first_summary["checkpoint_path"]).is_file()
    assert second_summary["final_step"] == 0
    assert len(second_summary["output_segments"]) == 2
    assert Path(second_summary["openpmd_path"]).is_file()


def test_plot_loader_and_final_schwarzschild_comparison(tmp_path):
    formation_path = tmp_path / "formation" / "profiles.h5"
    reference_path = tmp_path / "reference" / "profiles.h5"
    field = np.ones((16, 1, 32))
    for path, scale in ((formation_path, 0.8), (reference_path, 0.79)):
        with OpenPMDWriter(
            path,
            grid_spacing=(0.25, 0.25, 0.25),
            grid_global_offset=(-1.875, 0.0, -3.875),
            grid_position=(0.0, 0.0, 0.0),
            dt=0.1,
        ) as writer:
            writer.write({"lapse": field, "W": 0.5 * field}, 0, 0.0)
            writer.write(
                {"lapse": scale * field, "W": 0.5 * scale * field},
                3,
                0.3,
            )

    plot_path = ROOT / "demos" / "plot_em_blackhole_schwarzschild.py"
    spec = importlib.util.spec_from_file_location("em_bh_plot", plot_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    step, time, _, _, lapse, W = module.load_final_lapse_and_W(formation_path)

    assert step == 3
    assert time == pytest.approx(0.3)
    np.testing.assert_allclose(lapse, 0.8)
    np.testing.assert_allclose(W, 0.4)

    horizon = {
        "coefficients": [0.2, 0.0, 0.0, 0.0],
        "expansion_l2": 0.0,
        "expansion_linf": 0.0,
        "area": 16.0 * np.pi,
        "irreducible_mass": 1.0,
        "circumference_ratio": 1.0,
        "polar_radius": 0.2,
        "equatorial_radius": 0.2,
    }
    formation_metadata = tmp_path / "formation" / "run_summary.json"
    reference_metadata = tmp_path / "reference" / "run_summary.json"
    formation_metadata.write_text(
        json.dumps({"horizon": horizon}), encoding="utf-8"
    )
    reference_metadata.write_text(
        json.dumps({"horizon": horizon}), encoding="utf-8"
    )
    output = tmp_path / "comparison.png"
    png_path, pdf_path, error_path, errors = module.plot_comparison(
        formation_path,
        reference_path,
        formation_metadata,
        output,
    )

    assert png_path.is_file()
    assert pdf_path.is_file()
    assert error_path.is_file()
    assert errors["fields"]["lapse"]["equatorial"]["relative_l2"] > 0.0


def test_campaign_acceptance_requires_resolution_improvement_and_mass_agreement():
    from demos import plot_em_blackhole_campaign as module

    records = []
    error_names = (
        "lapse_equatorial_l2",
        "lapse_equatorial_linf",
        "lapse_polar_l2",
        "lapse_polar_linf",
        "W_equatorial_l2",
        "W_equatorial_linf",
        "W_polar_l2",
        "W_polar_linf",
    )
    for formulation, mass in (("first_order", 1.0), ("second_order", 1.02)):
        for resolution, error in (("medium", 2.0e-2), ("high", 1.0e-2)):
            record = {
                "formulation": formulation,
                "resolution": resolution,
                "amplitude": 0.08,
                "status": "settled",
                "reference_status": "settled",
                "initial_horizon_absent": True,
                "formed_before_ceiling": True,
                "diagnostics_finite": True,
                "constraint_history_finite": True,
                "horizon_diameter_cells": 8.0,
                "remnant_mass": mass,
            }
            record.update({name: error for name in error_names})
            records.append(record)

    acceptance = module.evaluate_acceptance(records)
    assert acceptance["accepted"]
    assert acceptance["high_resolution_masses_agree_within_five_percent"]

    records[-1]["W_polar_linf"] = 3.0e-2
    assert not module.evaluate_acceptance(records)["accepted"]
