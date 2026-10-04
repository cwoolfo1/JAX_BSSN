"""Manufactured sensitivity checks; these fixtures are not collapse evidence."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def test_time_space_sensitivity_and_incompatible_inputs(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[2]/'demos/em_critical_collapse'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('time_resolution_check', scripts/'compare_time_resolution.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from convergence import compare
    from JAX_BSSN.bssn.variables import BSSNParameters

    def save(n, dt, *, time=1., amplitude=.913, path=None):
        h = 4./n
        shape = (n+4, 1, 2*n)
        scalar = np.ones(shape)
        vector = np.zeros((3,)+shape)
        tensor = np.eye(3)[:, :, None, None, None]*scalar
        params = BSSNParameters(dx=h, dt=dt, zero_shift=1, gauge=1)
        meta = dict(formulation='first_order', time=time, step=round(time/dt),
                    parameters=params._asdict(), configuration=dict(
                        nr=n, radius=4., amplitude=amplitude, center=0.,
                        initial_boundary='multipole', nu=params.nu, kappa=params.kappa))
        path = path or tmp_path/f'n{n}_dt{dt}.npz'
        np.savez(path, bssn_conformal_metric=tensor, bssn_conformal_factor=scalar,
                 bssn_lapse=scalar*(1+h*h+10*dt*dt), bssn_traceless_K=tensor*0,
                 bssn_trace_K=scalar*0, bssn_conformal_connection=vector, bssn_shift=vector,
                 em_displacement_left_half=vector, em_displacement_right_half=vector,
                 em_magnetic_current=vector, metadata_json=np.asarray(json.dumps(meta)))
        return path

    checkpoints = [save(n, .01) for n in (16, 32, 64)]
    compare(checkpoints, 'space', 1., tmp_path/'space', checkpoints=True)
    report = tmp_path/'space/space_convergence.json'
    other = save(32, .02)
    module.compare(report, other, 1, tmp_path/'valid')
    result = json.loads((tmp_path/'valid/time_resolution_comparison.json').read_text())
    # Exact manufactured changes: 10*(.02^2-.01^2) and .125^2-.0625^2.
    np.testing.assert_allclose(result['results']['lapse']['ratio'], .256, rtol=1e-10)
    assert result['results']['shift']['ratio'] is None
    assert len(result['checkpoint_sha256']) == 4

    save(32, .02, amplitude=.914, path=other)
    with pytest.raises(ValueError, match='amplitude'):
        module.compare(report, other, 1, tmp_path/'bad_amplitude')
    save(32, .02, time=1.02, path=other)
    with pytest.raises(ValueError, match='matching coordinate times'):
        module.compare(report, other, 1, tmp_path/'bad_time')
    save(32, .01, path=other)
    with pytest.raises(ValueError, match='twice'):
        module.compare(report, other, 1, tmp_path/'same_dt')
    save(32, .02, path=other)
    save(64, .01, time=1.02, path=checkpoints[-1])
    with pytest.raises(ValueError, match='Checkpoint changed'):
        module.compare(report, other, 1, tmp_path/'changed_archive')
