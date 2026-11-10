"""V3 jobs: durable lifecycle, target identity, no delivery requirement, GPU isolation."""
import json
import subprocess
import pytest
from src.visualization import geometry22_jobs as jobs
from ui import server

TARGET = {"thrust_N": 60., "tsfc_kg_N_s": .000067, "mass_kg": 1.77}

@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setitem(server.app.config, "GEOMETRY22_JOB_ROOT", str(tmp_path / "jobs"))
    monkeypatch.setitem(server.app.config, "GEOMETRY22_PREVIEW_ROOT", str(tmp_path / "previews"))
    class IdleThread:
        def __init__(self, target, args, **kwargs):
            self.lock = args[1]
        def start(self):
            self.lock.close()
    monkeypatch.setattr(jobs.threading, "Thread", IdleThread)
    return server._geometry22_jobs()

@pytest.mark.parametrize('target', [None, {}, TARGET | {"device":"cpu"}, TARGET | {"mass_kg":True}, TARGET | {"mass_kg":0}, TARGET | {"mass_kg":"nan"}])
def test_invalid_targets_never_launch(service, target):
    assert server.app.test_client().post('/api/geometry22/jobs', json=target).status_code == 400
    assert not service.active


def test_single_job_status_and_gpu_conflict(service):
    client = server.app.test_client()
    response = client.post('/api/geometry22/jobs', json=TARGET)
    assert response.status_code == 202
    job = response.get_json()
    assert len(job['job_id']) == 64 and job['physical_gpu'] == 6
    assert client.post('/api/geometry22/jobs', json=TARGET).status_code == 409
    assert client.get('/api/geometry22/jobs/' + job['job_id']).get_json()['state'] == 'queued'
    assert client.get('/api/geometry22/jobs/' + job['job_id'] + '/result').status_code == 409
    assert client.post('/api/geometry19/v2-2/jobs', json=TARGET).status_code == 409
    assert client.get('/api/geometry22/jobs/bad').status_code == 404
    assert not (service.directory(job['job_id']) / 'run').exists()


def test_restart_interrupted_and_integrity_recovery(service, monkeypatch):
    job = service.submit(TARGET)
    service.active.clear()
    assert service.snapshot(job['job_id'])['state'] == 'interrupted'
    job['state'] = 'running'
    service.save(job)
    run = service.directory(job['job_id']) / 'run'
    run.mkdir()
    (run / 'Complete_Manifest.json').write_text('{}')
    monkeypatch.setattr(service.results, 'result', lambda _: {})
    assert service.snapshot(job['job_id'])['state'] == 'complete'
    monkeypatch.setattr(service.results, 'result', lambda _: (_ for _ in ()).throw(ValueError('tampered F')))
    assert service.snapshot(job['job_id'])['state'] == 'failed'


def test_no_hit_finishes_and_worker_binds_gpu6(service, monkeypatch):
    job = service.submit(TARGET)
    observed = []
    def run(command, **kwargs):
        observed.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, 'GPU6 identity', '')
    monkeypatch.setattr(jobs.subprocess, 'run', run)
    monkeypatch.setattr(service.results, 'result', lambda _: {'target_performance': TARGET, 'delivery_status':'NO_HIT', 'route_status':'supported'})
    lock = (service.directory(job['job_id']) / 'worker.lock').open('a+')
    service.run(job['job_id'], lock)
    data = service.snapshot(job['job_id'])
    assert data['state'] == 'complete' and data['delivery_status'] == 'NO_HIT'
    assert len(observed) == 3
    command, kwargs = observed[-1]
    assert command[command.index('--model')+1] == 'model V3'
    assert command[command.index('--pipeline-profile')+1] == 'v3_guided_cfm_development'
    assert command[command.index('--device')+1] == 'cuda:0' and '--export-mesh' not in command
    assert all(k['env']['CUDA_VISIBLE_DEVICES'] == '6' for _, k in observed)
    assert service.result(job['job_id'])['job_id'] == job['job_id']
    monkeypatch.setattr(service.results, 'result', lambda _: {'target_performance': TARGET | {'mass_kg':2}})
    with pytest.raises(ValueError, match='target differs'):
        service.result(job['job_id'])


def test_gpu_unavailable_fails_without_launching_cli(service, monkeypatch):
    job = service.submit(TARGET)
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(jobs.subprocess, 'run', fail)
    lock = (service.directory(job['job_id']) / 'worker.lock').open('a+')
    service.run(job['job_id'], lock)
    assert service.snapshot(job['job_id'])['state'] == 'failed'
    assert not (service.directory(job['job_id']) / 'run').exists()


def test_another_server_does_not_interrupt_locked_worker(service):
    import fcntl
    job = service.submit(TARGET)
    path = service.directory(job['job_id'])
    with (path / 'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        other = jobs.Geometry22Jobs(service.repository, service.root, service.results.previews)
        assert other.snapshot(job['job_id'])['state'] == 'queued'
        with pytest.raises(ValueError, match='already running'):
            other.submit(TARGET)
    assert other.snapshot(job['job_id'])['state'] == 'interrupted'
