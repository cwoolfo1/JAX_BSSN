import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tests.EM.demo_helpers import ROOT, load_em_demo_module


@pytest.mark.parametrize("formulation,script", [
    ("first_order", "EM_blackhole_formation_first_order.py"),
    ("second_order", "EM_blackhole_formation_second_order.py"),
    ("first_order", "compare_schwarzschild.py"),
])
@pytest.mark.parametrize("working_directory", ["demo", "unrelated"])
def test_copied_demo_runs_standalone(tmp_path, formulation, script, working_directory):
    source = ROOT / "demos" / f"EM_blackhole_formation_{formulation}"
    copied = tmp_path / "standalone"
    copied.mkdir()
    for path in source.glob("*.py"):
        shutil.copy2(path, copied / path.name)
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, str(copied / script), "--help"],
        cwd=copied if working_directory == "demo" else tmp_path,
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    expected_option = "--mass" if script == "compare_schwarzschild.py" else "--final-time"
    assert expected_option in result.stdout


def test_loading_both_demos_keeps_sibling_imports_local(monkeypatch):
    sentinels = {
        name: object()
        for name in ("collapse_io", "initial_data", "schwarzschild_reference")
    }
    for name, sentinel in sentinels.items():
        monkeypatch.setitem(sys.modules, name, sentinel)
    for formulation in ("first_order", "second_order", "first_order"):
        demo = load_em_demo_module(formulation, f"EM_blackhole_formation_{formulation}")
        checkpoint_source = Path(demo.load_collapse_checkpoint.__code__.co_filename)
        assert checkpoint_source.parent == Path(demo.__file__).parent
        assert all(sys.modules[name] is sentinel for name, sentinel in sentinels.items())
