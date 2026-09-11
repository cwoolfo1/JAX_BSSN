"""Load standalone EM demo modules without leaking sibling imports between tests."""

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]


def load_em_demo_module(formulation, name):
    directory = ROOT / "demos" / f"EM_blackhole_formation_{formulation}"
    siblings = ("collapse_io", "initial_data", "schwarzschild_reference")
    missing = object()
    previous = {sibling: sys.modules.get(sibling, missing) for sibling in siblings}

    def load(sibling):
        spec = importlib.util.spec_from_file_location(
            f"em_demo_{formulation}_{sibling}", directory / f"{sibling}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    try:
        sys.modules["collapse_io"] = load("collapse_io")
        if name == f"EM_blackhole_formation_{formulation}":
            sys.modules["initial_data"] = load("initial_data")
        if name in ("schwarzschild_reference", "compare_schwarzschild"):
            sys.modules["schwarzschild_reference"] = load("schwarzschild_reference")
        return sys.modules[name] if name in siblings else load(name)
    finally:
        for sibling, original in previous.items():
            if original is missing:
                sys.modules.pop(sibling, None)
            else:
                sys.modules[sibling] = original
