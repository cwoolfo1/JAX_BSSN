"""Fixed-method fresh, restart, frozen, and failure-output contracts."""

from dataclasses import asdict, replace
import csv
import hashlib
import importlib
import inspect
import json

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.EM.variables import EinsteinMaxwellVariables
from JAX_BSSN.EM.first_order.grid import make_em_grid
from JAX_BSSN.EM.first_order.geometry import build_yee_metric
from JAX_BSSN.EM.first_order import maxwell
from tests.EM.em_helpers import flat_bssn_variables
from demos.single_charged_blackhole import run_resolution_restart as runner
from demos.single_charged_blackhole import run_frozen_background as frozen
from demos.single_charged_blackhole.run_single_charged_blackhole import MAXWELL_METHOD


def small_state():
    params = BSSNParameters(dx=.25, dt=.025)
    grid = make_em_grid((8, 1, 1), params)
    x = grid.coordinates(params)[0]
    bssn = flat_bssn_variables(grid.shape)
    bssn = bssn._replace(shift=bssn.shift.at[0].set(.1*jnp.sin(x)))
    metric = build_yee_metric(bssn, grid)
    zero = grid.to_tile(jnp.zeros(grid.shape))
    d = (zero, grid.to_tile(jnp.sin(x)), zero)
    b = (zero, zero, grid.to_tile(jnp.sin(x+.5*params.dx)))
    state = EinsteinMaxwellVariables(
        bssn, maxwell.initialize_native_state(d, b, metric, params, grid))
    return state, params, grid


def check_source_archive(output):
    manifest = json.loads((output/"source_manifest.json").read_text())
    for name in ("PyPIC3D.solvers.GR_yee.static_metric", "PyPIC3D.solvers.GR_yee.time_loop",
                 "JAX_BSSN.EM.first_order.grid", "JAX_BSSN.EM.first_order.maxwell",
                 "PyPIC3D.relativity.field_interpolation", "PyPIC3D.relativity.core",
                 "PyPIC3D.boundary_conditions.reference_sponge"):
        assert name in manifest
    for name, digest in manifest.items():
        archived = output/"sources"/(name.replace(".", "/")+".py")
        assert hashlib.sha256(archived.read_bytes()).hexdigest() == digest
        with open(importlib.import_module(name).__file__, "rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == digest
        assert "probe" not in name
        assert "hermite" not in name


def test_configuration_and_runner_signatures_have_no_method_selection():
    assert not {"reciprocal_constitutive", "shift_transport", "metric_interpolation"}.intersection(
        runner.SingleChargedBlackHoleConfig.__dataclass_fields__)
    assert tuple(inspect.signature(runner.main).parameters) == ("root", "mode")
    assert tuple(inspect.signature(frozen.main).parameters) == (
        "snapshot", "output", "duration", "dt", "shift_scale", "initial")


@pytest.mark.parametrize("mode,fail_after_initial", (
    ("initial", False), ("initial", True), ("seed", False), ("192", False),
))
def test_runner_metadata_checkpoints_and_source_archive(
    tmp_path, monkeypatch, mode, fail_after_initial,
):
    state, params, grid = small_state()
    config_class = runner.SingleChargedBlackHoleConfig
    start = 35. if mode == "192" else 0.
    built = []
    original_electric = maxwell.compute_covariant_E
    original_magnetic = maxwell.compute_covariant_H

    def configuration(**kwargs):
        assert "shift_transport" not in kwargs
        assert "reciprocal_constitutive" not in kwargs
        return replace(config_class(**kwargs), num_points=12, final_time=start+params.dt)

    def build(config):
        built.append("initial")
        assert maxwell.compute_covariant_E is original_electric
        assert maxwell.compute_covariant_H is original_magnetic
        return state, params, grid, 1

    def restart(path, config):
        assert path == tmp_path/"case"/"seed96"/"native_t35.npz"
        built.append("restart")
        return state, params, grid, start

    class Writer:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def write(self, *args, **kwargs):
            pass

    validity_calls = []

    def valid(current):
        validity_calls.append(current)
        return not fail_after_initial or len(validity_calls) == 1

    cpu = jax.devices("cpu")[0]
    monkeypatch.setattr(runner, "SingleChargedBlackHoleConfig", configuration)
    monkeypatch.setattr(runner, "build_single_charged_blackhole", build)
    monkeypatch.setattr(runner, "interpolated_restart", restart)
    monkeypatch.setattr(runner, "select_device", lambda *args: cpu)
    monkeypatch.setattr(runner.jax, "devices", lambda *args: [cpu])
    monkeypatch.setattr(runner, "make_staged_step", lambda *args: lambda value: value)
    monkeypatch.setattr(runner, "staged_state_is_valid", valid)
    monkeypatch.setattr(runner, "charged_blackhole_diagnostics", lambda *args: (
        {"charge_4M": .5, "undamped_resolved_hamiltonian_rms": 0.},
        {"rho_EM": jnp.ones(grid.shape)*.002}))
    monkeypatch.setattr(runner, "_openpmd_fields", lambda fields: fields)
    monkeypatch.setattr(runner, "OpenPMDWriter", Writer)

    runner.main(tmp_path/"case", mode)
    assert built == ["restart" if mode == "192" else "initial"]
    output = tmp_path/"case"/{"initial":"initial96", "seed":"seed96", "192":"restart192"}[mode]
    config = json.loads((output/"run_configuration.json").read_text())
    status = json.loads((output/"status.json").read_text())
    for record in (config, status):
        for key, value in MAXWELL_METHOD.items():
            assert record[key] == value
        assert "transport_probe" not in record
        assert "reciprocal_constitutive" not in record
    assert config["start_time"] == start
    assert config["restart_source"] == (str(tmp_path/"case"/"seed96"/"native_t35.npz")
                                           if mode == "192" else None)
    assert "Original tensor-product two-point arithmetic" in config["metric_interpolation"]
    assert status["status"] == ("numerical_failure" if fail_after_initial else "completed")
    assert status["time"] == start+params.dt
    if mode == "initial":
        checkpoints = ("native_t0.npz", "failed_native.npz") if fail_after_initial else ("native_t0.npz",)
    elif mode == "192":
        checkpoints = ("native_t35.npz",)
    else:
        checkpoints = ()
    for name in checkpoints:
        with jnp.load(output/name) as saved:
            assert json.loads(str(saved["configuration_json"])) == config
            assert json.loads(str(saved["parameters_json"])) == params._asdict()
            assert float(saved["time"]) == (params.dt if name == "failed_native.npz" else start)
            assert int(saved["guard_cells"]) == grid.guard_cells
            assert bool(jnp.array_equal(saved["D_y"], state.em.fields[0][1]))
            assert bool(jnp.array_equal(saved["synchronized_B_z"], state.em.synchronized_magnetic[2]))
    check_source_archive(output)


def test_frozen_saved_state_uses_native_initialization_and_fixed_diagnostic(tmp_path, monkeypatch):
    state, params, grid = small_state()
    config = frozen.SingleChargedBlackHoleConfig(num_points=12, half_width=3.)
    snapshot = tmp_path/"saved_t75.npz"
    d, b = state.em.fields[0], state.em.synchronized_magnetic
    # Historical metadata may still contain retired method selectors. Reading
    # saved arrays does not use those keys to change the current evolution.
    historical = dict(asdict(config), reciprocal_constitutive=True,
                      shift_transport="mc_upwind", metric_interpolation="hermite")
    jnp.savez(snapshot, configuration_json=json.dumps(historical, default=str),
              lapse=state.bssn.lapse, W=state.bssn.conformal_factor,
              conformal_metric=state.bssn.conformal_metric, shift=state.bssn.shift,
              D=jnp.stack([v[0,0,0] for v in d]),
              B=jnp.stack([v[0,0,0] for v in b]), time=75.)
    events = []
    initialize = frozen.initialize_native_state
    metric = build_yee_metric(state.bssn, grid)
    expected = frozen.native_killing_quadratic(d, b, metric, grid, params.dx)

    def build(actual_config):
        assert not hasattr(actual_config, "shift_transport")
        events.append("background")
        return state, params, grid, 1

    def initialize_native(*args):
        assert events == ["background"]
        events.append("native")
        return initialize(*args)

    monkeypatch.setattr(frozen, "build_single_charged_blackhole", build)
    monkeypatch.setattr(frozen, "initialize_native_state", initialize_native)
    output = tmp_path/"frozen"
    frozen.main(snapshot, output, params.dt, params.dt, 1., "saved")
    assert events == ["background", "native"]
    saved_config = json.loads((output/"configuration.json").read_text())
    for key, value in MAXWELL_METHOD.items():
        assert saved_config[key] == value
    assert saved_config["constitutive"] == "target_metric_upwind_unlimited"
    assert saved_config["metric_interpolation"] == "arithmetic"
    assert saved_config["integrator"] == "staggered"
    assert saved_config["history_reinitialized"] is True
    assert saved_config["snapshot_sha256"] == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert "not a conserved energy" in saved_config["native_killing_quadratic"]
    for name, vector in (("D", d), ("B", b)):
        assert saved_config["initial_native_sha256"][name] == hashlib.sha256(jnp.stack(vector).tobytes()).hexdigest()
    with (output/"diagnostics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["elapsed_time"]) for row in rows] == [0., params.dt]
    assert bool(jnp.allclose(float(rows[0]["native_killing_quadratic"]), expected,
                            rtol=1.e-13, atol=1.e-14))
    with jnp.load(output/"final_common_fields.npz") as saved:
        assert float(saved["elapsed_time"]) == params.dt
        assert bool(jnp.all(jnp.isfinite(saved["D"])))
        assert bool(jnp.all(jnp.isfinite(saved["B"])))
        assert not bool(jnp.array_equal(saved["D"], jnp.stack([v[0,0,0] for v in d])))
    check_source_archive(output)
