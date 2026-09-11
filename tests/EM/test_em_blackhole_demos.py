import json

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import openpmd_api as io

from JAX_BSSN.EM.first_order.variables import DensitizedMaxwellState
from JAX_BSSN.EM.second_order.variables import EMVariables
from tests.EM.demo_helpers import load_em_demo_module


def _load_demo(formulation):
    return load_em_demo_module(formulation, f"EM_blackhole_formation_{formulation}")


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


def _restart_fixture(initialized_demo, tmp_path, step=0):
    formulation, demo, params, state, u, residual_history = initialized_demo
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
        "last_horizon_coefficients": [0.5, 0.0, 0.0, 0.0],
        "persistent_horizon_start_time": 0.0,
    }
    checkpoint = tmp_path / "rolling_checkpoint.npz"
    demo.write_collapse_checkpoint(
        checkpoint, state, u, residual_history, formulation,
        step, step * float(params.dt), params, configuration,
    )
    return checkpoint


def _assert_final_output(demo, directory, expected_step, expected_time, status):
    summary = json.loads((directory / "run_summary.json").read_text())
    assert summary["status"] == status
    assert summary["final_step"] == expected_step
    assert summary["final_time"] == pytest.approx(expected_time)
    assert summary["diagnostics"][-1]["finite"] == (status == "complete")
    assert "horizon" not in json.dumps(summary)
    assert "settling" not in json.dumps(summary)
    assert "schwarzschild_reference" not in summary
    assert not (directory / "horizon_diagnostics.txt").exists()
    assert not (directory / "schwarzschild_reference").exists()
    _, _, _, metadata = demo.load_collapse_checkpoint(directory / "rolling_checkpoint.npz")
    assert metadata["step"] == expected_step
    assert metadata["time"] == pytest.approx(expected_time)
    assert "horizon" not in json.dumps(metadata)
    series = io.Series(summary["output_segments"][-1], io.Access.read_only)
    try:
        assert max(series.iterations) == expected_step
        assert series.iterations[expected_step].time == pytest.approx(expected_time)
    finally:
        series.close()
    return summary


def test_legacy_settled_restart_advances_to_fixed_time(initialized_demo, tmp_path):
    formulation, demo, params, state, _, _ = initialized_demo
    checkpoint = _restart_fixture(initialized_demo, tmp_path, step=1)
    (tmp_path / "run_summary.json").write_text(json.dumps({
        "status": "settled", "horizon": {"irreducible_mass": 1.0},
        "settling_criteria": {"settled": True},
        "diagnostics": [{"step": 0, "time": 0.0, "finite": True,
                         "horizon": {}, "exterior_em_energy": 0.1}],
        "output_segments": ["previous.h5"],
    }))
    dt = float(params.dt)
    run = getattr(demo, f"run_em_blackhole_formation_{formulation}")
    final, _, _ = run(
        restart=checkpoint, output_dir=tmp_path, final_time=2.5 * dt,
        snapshot_count=1, diagnostic_interval=10 * dt,
        checkpoint_interval=dt, show_progress=False,
    )
    summary = _assert_final_output(demo, tmp_path, 2, 2 * dt, "complete")
    assert summary["final_time_ceiling"] == pytest.approx(2.5 * dt)
    assert summary["output_segments"][0] == "previous.h5"
    assert summary["diagnostics"][0] == {"step": 0, "time": 0.0, "finite": True}
    assert not np.array_equal(np.asarray(final.bssn.lapse), np.asarray(state.bssn.lapse))


def test_new_run_completes_and_preserves_outputs(initialized_demo, tmp_path):
    formulation, demo, _, _, _, _ = initialized_demo
    assert demo.FINAL_TIME == (500.0 if formulation == "first_order" else 40.0)
    run = getattr(demo, f"run_em_blackhole_formation_{formulation}")
    options = dict(
        amplitude=0.005, width=0.75, radial_center=1.0,
        domain_half_width=3.0, num_radial_points=6, num_z_points=12,
        final_time=0.0, output_dir=tmp_path, snapshot_count=1, show_progress=False,
    )
    run(**options)
    _assert_final_output(demo, tmp_path, 0, 0.0, "complete")
    with pytest.raises(FileExistsError):
        run(**options)


@pytest.mark.parametrize("diagnostic_step", [1, 4])
def test_nonfinite_failure_including_final_step(
    initialized_demo, tmp_path, monkeypatch, diagnostic_step
):
    formulation, demo, params, _, _, _ = initialized_demo
    checkpoint = _restart_fixture(initialized_demo, tmp_path)

    def invalid_step(current, params):
        return current._replace(bssn=current.bssn._replace(
            lapse=jnp.full_like(current.bssn.lapse, jnp.nan)
        ))

    step_name = (
        "axisymmetric_first_order_einstein_maxwell_step"
        if formulation == "first_order"
        else "axisymmetric_einstein_maxwell_rk4_step"
    )
    monkeypatch.setattr(demo, step_name, invalid_step)
    dt = float(params.dt)
    run = getattr(demo, f"run_em_blackhole_formation_{formulation}")
    run(restart=checkpoint, output_dir=tmp_path, final_time=2 * dt,
        snapshot_count=1, diagnostic_interval=diagnostic_step * dt,
        show_progress=False)
    final_step = 1 if diagnostic_step == 1 else 2
    summary = _assert_final_output(
        demo, tmp_path, final_step, final_step * dt, "failed_nonfinite"
    )
    assert [row["step"] for row in summary["diagnostics"]] == [0, final_step]
