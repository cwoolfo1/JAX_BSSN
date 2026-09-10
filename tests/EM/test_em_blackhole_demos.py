import importlib.util
import json
from pathlib import Path
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from JAX_BSSN.EM.first_order.variables import DensitizedMaxwellState
from JAX_BSSN.EM.second_order.variables import EMVariables


ROOT = Path(__file__).resolve().parents[2]


def _load_demo(formulation):
    demo_dir = ROOT / "demos" / f"EM_blackhole_formation_{formulation}"

    initial_data_spec = importlib.util.spec_from_file_location(
        "initial_data", demo_dir / "initial_data.py"
    )
    initial_data = importlib.util.module_from_spec(initial_data_spec)
    initial_data_spec.loader.exec_module(initial_data)

    module_name = f"em_blackhole_formation_{formulation}"
    demo_spec = importlib.util.spec_from_file_location(
        module_name, demo_dir / f"EM_blackhole_formation_{formulation}.py"
    )
    demo = importlib.util.module_from_spec(demo_spec)

    previous_initial_data = sys.modules.get("initial_data")
    sys.modules["initial_data"] = initial_data
    try:
        demo_spec.loader.exec_module(demo)
    finally:
        if previous_initial_data is None:
            del sys.modules["initial_data"]
        else:
            sys.modules["initial_data"] = previous_initial_data

    return demo


@pytest.fixture(scope="module", params=("first_order", "second_order"))
def initialized_demo(request):
    formulation = request.param
    demo = _load_demo(formulation)

    num_radial_points = 6
    num_z_points = 12
    domain_half_width = 3.0
    dx = domain_half_width / num_radial_points
    params = demo.axisymmetric_parameters(
        num_radial_points,
        num_z_points,
        domain_half_width,
        dt=0.02,
    )
    state, u, residual_history = demo.constrained_einstein_maxwell_data(
        amplitude=0.005,
        width=0.75,
        radial_center=1.0,
        num_radial_points=num_radial_points,
        num_z_points=num_z_points,
        dx=dx,
        params=params,
    )
    jax.block_until_ready(state)

    return formulation, demo, params, state, u, residual_history


def test_initialization_builds_finite_formulation_specific_state(
    initialized_demo,
):
    formulation, demo, params, state, u, residual_history = initialized_demo

    expected_shape = (3, 10, 1, 12)
    if formulation == "first_order":
        assert isinstance(state.em, DensitizedMaxwellState)
    else:
        assert isinstance(state.em, EMVariables)

    for field in state.em:
        assert field.shape == expected_shape
        assert bool(jnp.all(jnp.isfinite(field)))
    for field in state.bssn:
        assert bool(jnp.all(jnp.isfinite(field)))

    assert u.shape == (6, 12)
    assert residual_history[-1] <= 1.0e-10
    assert params.zero_shift == 0

    if formulation == "first_order":
        electric_field, _ = demo.common_physical_fields(state, params)
    else:
        electric_field = state.em.electric_field
    assert float(jnp.max(jnp.abs(electric_field[1]))) > 0.0
    np.testing.assert_allclose(electric_field[0], 0.0, atol=2.0e-14)
    np.testing.assert_allclose(electric_field[2], 0.0, atol=2.0e-14)


def test_diagnostics_return_expected_constraint_and_output_layout(
    initialized_demo,
):
    formulation, demo, params, state, _, _ = initialized_demo

    violations = demo.compute_matter_aware_axisymmetric_constraints(state, params)
    if formulation == "first_order":
        fields = demo._output_fields(state, violations, params)
        expected_em_records = {"D", "B"}
        absent_record = "E"
    else:
        fields = demo._output_fields(state, violations)
        expected_em_records = {"E", "B"}
        absent_record = "D"
    jax.block_until_ready((violations, fields))

    assert violations.hamiltonian.shape == state.bssn.conformal_factor.shape
    assert expected_em_records <= set(fields)
    assert absent_record not in fields
    for record in expected_em_records:
        assert len(fields[record]) == 3
        for component in fields[record]:
            assert component.shape == (12, 1, 12)
            assert np.isfinite(np.asarray(component)).all()


def test_formation_step_evolves_gamma_driver_shift(initialized_demo):
    formulation, demo, params, state, _, _ = initialized_demo
    if formulation == "first_order":
        advanced = demo.axisymmetric_first_order_einstein_maxwell_step(
            state, params
        )
    else:
        advanced = demo.axisymmetric_einstein_maxwell_rk4_step(state, params)
    jax.block_until_ready(advanced)

    shift_change = jnp.max(jnp.abs(advanced.bssn.shift - state.bssn.shift))
    assert float(shift_change) > 0.0


def test_restart_preserves_an_already_settled_formation(
    initialized_demo, tmp_path
):
    formulation, demo, params, state, u, residual_history = initialized_demo
    checkpoint_path = tmp_path / "rolling_checkpoint.npz"
    horizon = {
        "coefficients": [0.5, 0.0, 0.0, 0.0],
        "expansion_l2": 0.0,
        "expansion_linf": 0.0,
        "area": 16.0 * np.pi,
        "irreducible_mass": 1.0,
        "circumference_ratio": 1.0,
        "polar_radius": 0.5,
        "equatorial_radius": 0.5,
    }
    settling = {
        "persistent_horizon": True,
        "mass_fractional_range": 0.0,
        "maximum_circumference_distortion": 0.0,
        "maximum_exterior_em_energy_fraction": 0.0,
        "lapse_relative_l2_change": 0.0,
        "W_relative_l2_change": 0.0,
        "settled": True,
    }
    configuration = {
        "amplitude": 0.005,
        "width": 0.75,
        "radial_center": 1.0,
        "domain_half_width": 3.0,
        "num_radial_points": 6,
        "num_z_points": 12,
        "cfl": float(params.dt / params.dx),
        "diagnostic_interval": 0.5,
        "checkpoint_interval": 1.0,
        "last_horizon_coefficients": horizon["coefficients"],
        "persistent_horizon_start_time": 0.0,
    }
    demo.write_collapse_checkpoint(
        checkpoint_path,
        state,
        u,
        residual_history,
        formulation,
        0,
        0.0,
        params,
        configuration,
    )
    (tmp_path / "run_summary.json").write_text(
        json.dumps(
            {
                "status": "settled",
                "horizon": horizon,
                "settling_criteria": settling,
                "diagnostics": [],
                "output_segments": [],
            }
        ),
        encoding="utf-8",
    )

    run = getattr(demo, f"run_em_blackhole_formation_{formulation}")
    run(
        output_dir=tmp_path,
        restart=checkpoint_path,
        final_time=0.0,
        snapshot_count=1,
        show_progress=False,
        run_schwarzschild=False,
    )
    summary = json.loads((tmp_path / "run_summary.json").read_text())

    assert summary["status"] == "settled"
    assert summary["settling_criteria"]["settled"]
    assert summary["horizon"]["irreducible_mass"] == 1.0
