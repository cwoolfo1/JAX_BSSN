import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tests.EM.demo_helpers import ROOT, load_em_demo_module


@pytest.mark.parametrize("formulation,script", [
    ("first_order", "run_collapse.py"),
    ("second_order", "run_collapse.py"),
])
@pytest.mark.parametrize("working_directory", ["demo", "unrelated"])
def test_copied_demo_runs_standalone(tmp_path, formulation, script, working_directory):
    source = ROOT / "demos" / f"EM_blackhole_formation_{formulation}"
    copied = tmp_path / "standalone"
    copied.mkdir()
    for path in source.glob("*.py"):
        shutil.copy2(path, copied / path.name)
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    configuration = copied / "simulation_parameters.py"
    configuration.write_text(configuration.read_text() + "\n" + "\n".join([
        "AMPLITUDE = 0.0", "NUM_RADIAL_POINTS = 6", "NUM_Z_POINTS = 12",
        "DOMAIN_HALF_WIDTH = 3.0", "FINAL_TIME = 0.0", "SHOW_PROGRESS = False",
    ]))
    command = [sys.executable, str(copied / script)]
    result = subprocess.run(
        command,
        cwd=copied if working_directory == "demo" else tmp_path,
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    output_name = "output_uniform"
    assert (copied / output_name / "final_checkpoint.npz").is_file()
    assert not (tmp_path / output_name).exists()


def test_loading_both_demos_keeps_sibling_imports_local(monkeypatch):
    sentinels = {
        name: object()
        for name in ("collapse_io", "initial_data", "simulation_parameters",
                     "initial_pulse", "initial_metric")
    }
    for name, sentinel in sentinels.items():
        monkeypatch.setitem(sys.modules, name, sentinel)
    for formulation in ("first_order", "second_order", "first_order"):
        demo = load_em_demo_module(
            formulation, "run_collapse"
        )
        checkpoint_source = Path(demo.write_collapse_checkpoint.__code__.co_filename)
        assert checkpoint_source.parent == Path(demo.__file__).parent
        assert all(sys.modules[name] is sentinel for name, sentinel in sentinels.items())


@pytest.mark.parametrize("formulation", ["first_order", "second_order"])
def test_importing_modules_does_not_run_simulation(tmp_path, formulation):
    source = ROOT / "demos" / f"EM_blackhole_formation_{formulation}"
    for path in source.glob("*.py"):
        shutil.copy2(path, tmp_path / path.name)
    result = subprocess.run(
        [sys.executable, "-c",
         "import simulation_parameters, initial_pulse, initial_metric, run_collapse"],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "output_uniform").exists()
