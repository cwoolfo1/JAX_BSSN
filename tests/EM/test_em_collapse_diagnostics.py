import importlib.util
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import openpmd_api as io

from JAX_BSSN.bssn.variables import BSSNParameters
from JAX_BSSN.diagnostics.openpmd import OpenPMDWriter
from tests.EM.demo_helpers import load_em_demo_module
from JAX_BSSN.EM.first_order.variables import DensitizedMaxwellState
from JAX_BSSN.EM.second_order.variables import EMVariables
from JAX_BSSN.EM.variables import EinsteinMaxwellVariables
from JAX_BSSN.evolution.boundaries import PERIODIC_BC, SOMMERFELD_BC


ROOT = Path(__file__).resolve().parents[2]
reference = load_em_demo_module("first_order", "schwarzschild_reference")
_schwarzschild_state = reference._schwarzschild_state
run_schwarzschild_reference = reference.run_schwarzschild_reference


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


@pytest.mark.parametrize(
    ("formulation", "em_type"),
    (("first_order", DensitizedMaxwellState), ("second_order", EMVariables)),
)
def test_collapse_checkpoint_round_trip(tmp_path, formulation, em_type):
    collapse_io = load_em_demo_module(formulation, "collapse_io")
    params = _parameters(8, 16)
    bssn = _schwarzschild_state(1.0, params, 8, 16)
    em_field = jnp.arange(3 * 12 * 16, dtype=jnp.float64).reshape((3, 12, 1, 16))
    em = em_type(*(em_field + index for index in range(4)))
    state = EinsteinMaxwellVariables(bssn=bssn, em=em)
    u = jnp.arange(8 * 16, dtype=jnp.float64).reshape((8, 16))
    path = tmp_path / "rolling_checkpoint.npz"

    collapse_io.write_collapse_checkpoint(
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
    restored, restored_u, residual_history, metadata = collapse_io.load_collapse_checkpoint(
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


def test_schwarzschild_reference_checkpoint_is_restartable(tmp_path):
    params = _parameters(8, 16)
    output_dir = tmp_path / "reference"
    first_summary = run_schwarzschild_reference(
        1.0,
        params,
        8,
        16,
        float(params.dt),
        output_dir,
        show_progress=False,
    )
    first_summary["status"] = "settled"
    (output_dir / "run_summary.json").write_text(json.dumps(first_summary))
    # Old reference checkpoints contained optional tracking metadata.
    checkpoint_path = Path(first_summary["checkpoint_path"])
    with np.load(checkpoint_path) as checkpoint:
        arrays = dict(checkpoint)
    metadata = json.loads(str(arrays["metadata_json"]))
    metadata["horizon_coefficients"] = [0.5, 0.0, 0.0, 0.0]
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez(checkpoint_path, **arrays)
    second_summary = run_schwarzschild_reference(
        1.0,
        params,
        8,
        16,
        2.5 * float(params.dt),
        output_dir,
        show_progress=False,
    )

    assert Path(first_summary["checkpoint_path"]).is_file()
    assert second_summary["final_step"] == 2
    assert second_summary["status"] == "complete"
    assert second_summary["final_time"] == pytest.approx(2 * float(params.dt))
    assert second_summary["final_time_ceiling"] == pytest.approx(2.5 * float(params.dt))
    assert len(second_summary["output_segments"]) == 2
    assert Path(second_summary["openpmd_path"]).is_file()
    assert "horizon" not in json.dumps(second_summary)
    assert not (output_dir / "horizon_diagnostics.txt").exists()
    with np.load(checkpoint_path) as checkpoint:
        metadata = json.loads(str(checkpoint["metadata_json"]))
    assert "horizon_coefficients" not in metadata
    assert metadata["step"] == 2
    series = io.Series(second_summary["openpmd_path"], io.Access.read_only)
    try:
        assert max(series.iterations) == 2
        assert series.iterations[2].time == pytest.approx(metadata["time"])
    finally:
        series.close()


@pytest.mark.parametrize("diagnostic_step", [1, 4])
def test_schwarzschild_nonfinite_failure(tmp_path, monkeypatch, diagnostic_step):
    params = _parameters(8, 16)

    def invalid_step(state, params):
        return state._replace(lapse=jnp.full_like(state.lapse, jnp.nan))

    monkeypatch.setattr(reference, "axisymmetric_rk4_step", invalid_step)
    summary = run_schwarzschild_reference(
        1.0, params, 8, 16, 2 * float(params.dt), tmp_path,
        diagnostic_interval=diagnostic_step * float(params.dt),
    )
    expected_step = 1 if diagnostic_step == 1 else 2
    assert summary["status"] == "failed_nonfinite"
    assert summary["final_step"] == expected_step
    with np.load(summary["checkpoint_path"]) as checkpoint:
        metadata = json.loads(str(checkpoint["metadata_json"]))
    assert metadata["step"] == expected_step
    series = io.Series(summary["openpmd_path"], io.Access.read_only)
    try:
        assert max(series.iterations) == expected_step
    finally:
        series.close()


@pytest.mark.parametrize("plotter", [
    "demos/plot_em_blackhole_schwarzschild.py",
    "demos/EM_blackhole_formation_first_order/plot_em_blackhole_schwarzschild.py",
])
def test_plot_loader_and_final_schwarzschild_comparison(tmp_path, plotter):
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

    plot_path = ROOT / plotter
    spec = importlib.util.spec_from_file_location("em_bh_plot", plot_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    step, time, _, _, lapse, W = module.load_final_lapse_and_W(formation_path)

    assert step == 3
    assert time == pytest.approx(0.3)
    np.testing.assert_allclose(lapse, 0.8)
    np.testing.assert_allclose(W, 0.4)

    output = tmp_path / "comparison.png"
    png_path, pdf_path, error_path, errors = module.plot_comparison(
        formation_path,
        reference_path,
        1.0,
        output,
    )

    assert png_path.is_file()
    assert pdf_path.is_file()
    assert error_path.is_file()
    assert errors["fields"]["lapse"]["equatorial"]["relative_l2"] > 0.0
    assert errors["comparison_interval"]["equatorial"]["inner_radius"] == 0.5
    for mass in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and positive"):
            module.plot_comparison(formation_path, reference_path, mass, output)
