"""Browser-facing V2.2 job lifecycle without starting a GPU run."""
import json

from ui import server


def test_v22_job_validates_targets_and_prevents_duplicate(monkeypatch, tmp_path):
    class FakeThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(server.threading, 'Thread', FakeThread)
    monkeypatch.setitem(server.app.config, 'GEOMETRY19_JOB_ROOT', str(tmp_path))
    server.GEOMETRY19_JOBS.clear()
    client = server.app.test_client()
    endpoint = '/api/geometry19/v2-2/jobs'
    assert client.post(endpoint, json={'thrust_N': 1}).status_code == 400
    assert client.post(endpoint, json={
        'thrust_N': 1, 'tsfc_kg_N_s': 0, 'mass_kg': 1,
    }).status_code == 400
    response = client.post(endpoint, json={
        'thrust_N': 60, 'tsfc_kg_N_s': 0.000067, 'mass_kg': 1.77,
    })
    assert response.status_code == 202
    job_id = response.get_json()['job_id']
    assert client.get(f'{endpoint}/{job_id}').get_json()['state'] == 'queued'
    assert client.post(endpoint, json={
        'thrust_N': 70, 'tsfc_kg_N_s': 0.000067, 'mass_kg': 1.77,
    }).status_code == 409
    assert client.get(f'{endpoint}/{job_id}/result').status_code == 409
    server.GEOMETRY19_JOBS.clear()


def test_v22_job_status_survives_server_restart(monkeypatch, tmp_path):
    monkeypatch.setitem(server.app.config, 'GEOMETRY19_JOB_ROOT', str(tmp_path))
    job_id = 'a' * 32
    directory = tmp_path / job_id
    directory.mkdir()
    job = {
        'job_id': job_id, 'state': 'running', 'target': {'thrust_N': 60},
        'output_root': str(directory / 'run'), 'storage_root': str(directory / 'designs'),
    }
    (directory / 'job.json').write_text(json.dumps(job))
    server.GEOMETRY19_JOBS.clear()
    client = server.app.test_client()
    response = client.get(f'/api/geometry19/v2-2/jobs/{job_id}')
    assert response.status_code == 200
    assert response.get_json()['state'] == 'interrupted'
    assert json.loads((directory / 'job.json').read_text())['state'] == 'interrupted'
    assert client.get('/api/geometry19/v2-2/jobs/../result').status_code == 404


def test_completed_v22_job_is_recovered_after_restart(monkeypatch, tmp_path):
    monkeypatch.setitem(server.app.config, 'GEOMETRY19_JOB_ROOT', str(tmp_path))
    job_id = 'b' * 32
    directory = tmp_path / job_id
    runtime = directory / 'run'
    runtime.mkdir(parents=True)
    (runtime / 'Complete_Manifest.json').write_text('{}')
    job = {
        'job_id': job_id, 'state': 'running', 'target': {'thrust_N': 60},
        'output_root': str(runtime), 'storage_root': str(directory / 'designs'),
    }
    (directory / 'job.json').write_text(json.dumps(job))
    monkeypatch.setattr(server, '_completed_geometry19_pipeline_result',
                        lambda quality, output: ({'design_id': 'new'}, '', {}))
    server.GEOMETRY19_JOBS.clear()
    response = server.app.test_client().get(f'/api/geometry19/v2-2/jobs/{job_id}')
    assert response.get_json()['state'] == 'complete'
    assert json.loads((directory / 'job.json').read_text())['state'] == 'complete'


def test_saved_v22_result_is_rejected_if_integrity_check_fails(monkeypatch, tmp_path):
    monkeypatch.setitem(server.app.config, 'GEOMETRY19_JOB_ROOT', str(tmp_path))
    job_id = 'd' * 32
    runtime = tmp_path / job_id / 'run'
    runtime.mkdir(parents=True)
    (runtime / 'Complete_Manifest.json').write_text('{}')
    (runtime.parent / 'job.json').write_text(json.dumps({
        'job_id': job_id, 'state': 'complete', 'target': {},
        'output_root': str(runtime), 'storage_root': str(runtime.parent / 'designs'),
    }))
    monkeypatch.setattr(server, '_completed_geometry19_pipeline_result',
                        lambda quality, output: (_ for _ in ()).throw(ValueError('tampered')))
    server.GEOMETRY19_JOBS.clear()
    response = server.app.test_client().get(f'/api/geometry19/v2-2/jobs/{job_id}')
    assert response.get_json()['state'] == 'failed'


def test_v22_job_result_uses_its_own_runtime(monkeypatch):
    job_id = 'c' * 32
    server.GEOMETRY19_JOBS.clear()
    server.GEOMETRY19_JOBS[job_id] = {
        'job_id': job_id, 'state': 'complete', 'target': {},
        'output_root': '/isolated/run', 'storage_root': '/isolated/designs',
    }
    observed = []

    def fake_result(quality, runtime_root):
        observed.append((quality, runtime_root))
        return ({'component_urls': {
            'inlet': '/geometry19-pipeline-files/id/preview_mesh/inlet.stl',
        }}, '/isolated/designs', {'components': {'inlet': {'filename': 'inlet.stl'}}})

    monkeypatch.setattr(server, '_completed_geometry19_pipeline_result', fake_result)
    response = server.app.test_client().get(
        f'/api/geometry19/v2-2/jobs/{job_id}/result?quality=preview_mesh'
    )
    assert response.status_code == 200
    assert observed == [('preview_mesh', '/isolated/run')]
    assert response.get_json()['component_urls']['inlet'].startswith(
        f'/geometry19-v22-job-files/{job_id}/'
    )
    server.GEOMETRY19_JOBS.clear()
