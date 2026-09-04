import importlib.util
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
    formulation, _, _, state, u, residual_history = initialized_demo

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

    assert u.shape == (12, 13, 12)
    assert residual_history[-1] <= 1.0e-10


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
