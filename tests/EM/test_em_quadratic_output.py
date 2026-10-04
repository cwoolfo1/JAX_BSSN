"""Short evolution, saved source density, and legacy movie compatibility."""

import importlib.util
import json
from pathlib import Path
import sys

import jax
import numpy as np
import openpmd_api as io
import pytest

from JAX_BSSN.diagnostics.openpmd import OpenPMDWriter
from JAX_BSSN.cartoon.axisymmetry.reconstruction import _expand_axisymmetric_scalar
from tests.EM.demo_helpers import load_em_demo_module, run_em_demo


def _movies():
    path = Path(__file__).resolve().parents[2] / "demos/EM_blackhole_formation_first_order/make_movies.py"
    name = "first_order_quadratic_movies"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_movie_restart_can_follow_a_snapshotless_initial_seed(tmp_path):
    movie = _movies()
    seed, child = tmp_path/'seed', tmp_path/'child'
    seed.mkdir(); child.mkdir()
    (seed/'run_summary.json').write_text(json.dumps(dict(output_segments=[],parent_run=None)))
    snapshot = child/'EM_blackhole_formation.h5'
    snapshot.touch()  # Only catalog discovery is under test here.
    (child/'run_summary.json').write_text(json.dumps(dict(
        output_segments=[snapshot.name],parent_run=str(seed))))
    assert movie.input_paths(child) == [snapshot]
    with pytest.raises(FileNotFoundError):
        movie.input_paths(seed)
    # A declared missing ancestor file must not be silently skipped.
    (seed/'run_summary.json').write_text(json.dumps(dict(
        output_segments=['missing.h5'],parent_run=None)))
    with pytest.raises(FileNotFoundError):
        movie.input_paths(child)


def test_movie_restart_includes_unfinalized_ancestor_snapshots(tmp_path):
    movie = _movies()
    parent, child = tmp_path/'parent', tmp_path/'child'
    parent.mkdir(); child.mkdir()
    old = parent/'EM_blackhole_formation.h5'
    new = child/'EM_blackhole_formation.h5'
    old.touch(); new.touch()
    metadata = dict(status='interrupted_unfinalized',
                    configuration=dict(no_snapshots=False))
    (parent/'run_summary.json').write_text(json.dumps(metadata))
    (child/'run_summary.json').write_text(json.dumps(dict(
        parent_run=str(parent), output_segments=[new.name])))
    assert movie.input_paths(child) == [old, new]
    old.unlink()
    with pytest.raises(FileNotFoundError):
        movie.input_paths(child)
    metadata['configuration']['no_snapshots'] = True
    (parent/'run_summary.json').write_text(json.dumps(metadata))
    assert movie.input_paths(child) == [new]


def test_cropped_snapshot_retains_physical_coordinates_and_values(tmp_path):
    from JAX_BSSN.diagnostics.mesh_coordinates import mesh_axis_coordinates
    runner_path = Path(__file__).resolve().parents[2]/'demos/em_critical_collapse/run_experiment.py'
    spec = importlib.util.spec_from_file_location('snapshot_test_runner', runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    selection, origin = runner.snapshot_layout(4, .5, -1.75, 1.)
    x = np.arange(-1.75, 2, .5)[:, None, None]
    z = np.arange(-1.75, 2, .5)[None, None, :]
    full = 10*x+z
    data = {'W': full[selection], 'shift': tuple(k*full[selection] for k in (1,2,3))}
    path = tmp_path/'cropped.h5'
    with OpenPMDWriter(path, (.5,)*3, origin, .01, grid_position=(0.,)*3) as writer:
        writer.write(data, 0, 0.)
    series = io.Series(str(path), io.Access.read_only)
    try:
        meshes = series.iterations[0].meshes
        for name in ('W', 'shift'):
            mesh = meshes[name]
            for key in mesh:
                component = mesh[key]
                xs = mesh_axis_coordinates(mesh, component, 0)
                zs = mesh_axis_coordinates(mesh, component, 2)
                np.testing.assert_array_equal(xs, [-.75,-.25,.25,.75])
                np.testing.assert_array_equal(zs, [-.75,-.25,.25,.75])
                actual = component.load_chunk()
                series.flush()
                factor = 'xyz'.index(key)+1 if name == 'shift' else 1
                np.testing.assert_array_equal(actual, factor*(10*xs[:,None,None]+zs[None,None,:]))
    finally:
        series.close()


@pytest.mark.parametrize("saved", [False, True])
def test_density_movie_uses_saved_moments_or_legacy_fallback(tmp_path, saved):
    movie = _movies()
    path = tmp_path / "density.h5"
    one = np.ones((8, 1, 8))
    fields = {"W": one, "D": (0*one, 0*one, 0*one), "B": (0*one, 0*one, one)}
    fields.update({f"conformal_metric_{c}": one if c[0] == c[1] else 0*one
                   for c in movie.TENSOR_COMPONENTS})
    if saved:
        fields["rho_EM"] = 2*one  # Deliberately different from mean-field energy.
    with OpenPMDWriter(path, (.1, .1, .1), (0., 0., 0.), .02) as writer:
        writer.write(fields, 0, 0.)
    values, _ = movie.load_movie(movie.MOVIES[-1], [(path, 0, 0.)])
    np.testing.assert_allclose(values, 2. if saved else .5)


def test_short_coupled_run_saves_the_source_density(tmp_path, monkeypatch):
    demo = load_em_demo_module("first_order", "run_collapse")
    run_em_demo(demo, "first_order", monkeypatch,
                amplitude=.005, width=.75, radial_center=1.,
                domain_half_width=3., num_radial_points=6, num_z_points=12,
                cfl=.2, final_time=.2, output_dir=tmp_path,
                snapshot_count=3, show_progress=False)
    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["final_step"] == 2
    checkpoint = load_em_demo_module("first_order", "collapse_io")
    state, _, _, metadata = checkpoint.load_collapse_checkpoint(summary["checkpoint_path"])
    params = checkpoint.parameters_from_dict(metadata["parameters"])
    expected = _expand_axisymmetric_scalar(demo.compute_axisymmetric_em_energy_density(state, params))
    jax.block_until_ready(expected)
    series = io.Series(summary["output_segments"][-1], io.Access.read_only)
    try:
        iteration = series.iterations[2]
        density = iteration.meshes["rho_EM"][io.Mesh_Record_Component.SCALAR].load_chunk()
        series.flush()
        np.testing.assert_allclose(density, expected, rtol=1.e-13, atol=1.e-15)
        assert np.isfinite(density).all()
        assert np.max(density) > 0
    finally:
        series.close()
