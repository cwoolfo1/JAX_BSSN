import json
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from JAX_BSSN.bssn.variables import BSSNParameters
from tests.EM.demo_helpers import load_em_demo_module
from JAX_BSSN.EM.first_order.variables import DensitizedMaxwellState
from JAX_BSSN.EM.variables import EinsteinMaxwellVariables
from JAX_BSSN.evolution.boundaries import SOMMERFELD_BC


collapse_io = load_em_demo_module("first_order", "collapse_io")
reference = load_em_demo_module("first_order", "schwarzschild_reference")


@pytest.fixture(scope="module")
def comparison():
    return load_em_demo_module("first_order", "compare_schwarzschild")


@pytest.mark.parametrize("mass", [None, 0.0, -1.0, np.nan, np.inf])
def test_manual_comparison_requires_positive_mass(comparison, monkeypatch, mass):
    args = ["compare_schwarzschild.py"]
    if mass is not None:
        args += ["--mass", str(mass)]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as error:
        comparison.main()
    assert error.value.code == 2


def test_manual_comparison_and_plot_only_reuse(comparison, tmp_path, monkeypatch):
    nr, nz = 8, 16
    params = BSSNParameters(
        dx=0.5, dt=0.02, x_min=-1.75, y_min=-2.0, z_min=-3.75,
        xr_bc=SOMMERFELD_BC, zl_bc=SOMMERFELD_BC, zr_bc=SOMMERFELD_BC,
    )
    bssn = reference._schwarzschild_state(0.8, params, nr, nz)
    field = jnp.zeros((3,) + bssn.lapse.shape)
    state = EinsteinMaxwellVariables(bssn, DensitizedMaxwellState(*(field for _ in range(4))))
    configuration = {
        "num_radial_points": nr, "num_z_points": nz,
        "last_horizon_coefficients": [0.5, 0.0, 0.0, 0.0],
        "persistent_horizon_start_time": 0.0,
    }
    input_dir = tmp_path / "formation"
    output_dir = tmp_path / "comparison"
    collapse_io.write_collapse_checkpoint(
        input_dir / "rolling_checkpoint.npz", state, jnp.zeros((nr, nz)),
        [0.0], "first_order", 0, 0.0, params, configuration,
    )
    (input_dir / "run_summary.json").write_text(json.dumps({
        "final_step": 0, "final_time": 0.0, "status": "complete",
        "parameters": collapse_io.parameters_to_dict(params),
    }))
    args = ["compare_schwarzschild.py", "--input-dir", str(input_dir),
            "--output-dir", str(output_dir)]
    monkeypatch.setattr(sys, "argv", args + ["--mass", "1.0"])
    comparison.main()
    assert len(list(output_dir.glob("*.png"))) == 16
    metadata = json.loads((output_dir / "comparison.json").read_text())
    assert metadata["mass"] == 1.0
    assert metadata["mass_source"] == "user_specified"
    assert "mass_time" not in metadata
    assert "horizon" not in json.dumps(metadata)

    def unexpected_evolution(*args, **kwargs):
        pytest.fail("plot-only must use the saved state")

    monkeypatch.setattr(comparison, "evolve_reference", unexpected_evolution)
    monkeypatch.setattr(sys, "argv", args + ["--plot-only"])
    comparison.main()
    assert len(list(output_dir.glob("*.png"))) == 16
    monkeypatch.setattr(sys, "argv", args + ["--plot-only", "--mass", "2.0"])
    with pytest.raises(SystemExit) as error:
        comparison.main()
    assert error.value.code == 2
