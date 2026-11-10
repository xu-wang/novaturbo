"""The public selector resolves a model before loading model-specific options."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.ai import geometry19_pipeline_v3 as v3


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_wp19_23_pipeline.py'


def _cli():
    spec = importlib.util.spec_from_file_location('geometry19_pipeline_cli_s1', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _summary():
    return {
        'status': 'TEST_COMPLETE', 'terminal_source_stage': 'C',
        'delivery_status': 'PASS', 'delivered_design_ids': [],
        'paths': {'runtime_root': '/tmp/test', 'pipeline_manifest': '/tmp/test/manifest'},
    }


def test_default_v21_still_builds_its_profile_context(monkeypatch):
    cli = _cli()
    registry = SimpleNamespace(profile_names=('v2_1',), default_profile='v2_1')
    context = SimpleNamespace(protocol={
        'models': {
            'generator': {'variants': ['baseline']},
            'forward': {'variants': ['performance3_log']},
        },
        'generation': {
            'pipeline_sampling_modes': ['direct'], 'default_sampling_mode': 'direct',
        },
    })
    captured = {}
    monkeypatch.setattr(cli, 'load_pipeline_profile_registry', lambda root: registry)
    def build(root, **kwargs):
        captured.update(kwargs)
        return context
    monkeypatch.setattr(cli, 'build_pipeline_context', build)
    monkeypatch.setattr(cli, 'run_wp19_23_pipeline', lambda root, **kwargs: captured.update(kwargs) or _summary())

    assert cli.main(['--thrust-N', '1', '--tsfc-kg-N-s', '2', '--mass-kg', '3']) == 0
    assert captured['pipeline_profile'] == 'v2_1'
    assert captured['pipeline_context'] is context
    assert captured['model'] == 'model V2.1'


def test_v22_does_not_load_v21_profile_context(monkeypatch):
    cli = _cli()
    monkeypatch.setattr(cli, 'load_pipeline_profile_registry', lambda root: pytest.fail('V2.1 registry loaded'))
    monkeypatch.setattr(cli, 'build_pipeline_context', lambda *args, **kwargs: pytest.fail('V2.1 context built'))
    captured = {}
    monkeypatch.setattr(cli, 'run_wp19_23_pipeline', lambda root, **kwargs: captured.update(kwargs) or _summary())

    assert cli.main(['--model', 'model V2.2', '--thrust-N', '1',
                     '--tsfc-kg-N-s', '2', '--mass-kg', '3']) == 0
    assert captured['pipeline_context'] is None
    assert captured['pipeline_profile'] is None
    assert captured['model'] == 'model V2.2'


@pytest.mark.parametrize('option,value', [
    ('--pipeline-profile', 'v2_1'),
    ('--model-variant', 'baseline'),
    ('--forward-model-tag', 's0'),
    ('--sampling', 'direct'),
    ('--storage-root', '/tmp/store'),
    ('--restart-from-stage', 'F'),
    ('--export-mesh', None),
])
def test_v3_rejects_incompatible_options_before_dispatch(monkeypatch, capsys, option, value):
    cli = _cli()
    monkeypatch.setattr(cli, 'build_pipeline_context', lambda *args, **kwargs: pytest.fail('V2.1 context built'))
    monkeypatch.setattr(cli, 'run_wp19_23_pipeline', lambda *args, **kwargs: pytest.fail('dispatched'))
    argv = ['--model', 'model V3', '--thrust-N', '1',
            '--tsfc-kg-N-s', '2', '--mass-kg', '3', option]
    if value is not None:
        argv.append(value)
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
    assert option in capsys.readouterr().err


def test_valid_v3_selection_reaches_development_entry_without_v21_context(monkeypatch):
    cli = _cli()
    monkeypatch.setattr(cli, 'build_pipeline_context', lambda *args, **kwargs: pytest.fail('V2.1 context built'))
    captured = {}
    monkeypatch.setattr(cli, 'run_wp19_23_pipeline', lambda root, **kwargs: captured.update(kwargs) or _summary())
    assert cli.main(['--model', 'model V3', '--thrust-N', '1',
                     '--tsfc-kg-N-s', '2', '--mass-kg', '3']) == 0
    assert captured['pipeline_context'] is None
    assert captured['model'] == 'model V3'
    assert captured['device'] == 'cpu'


def test_direct_v3_dispatch_rejects_v21_profile_before_runtime(tmp_path):
    with pytest.raises(v3.Geometry19V3PendingError, match='--pipeline-profile'):
        v3.run_wp19_23_pipeline_v3(tmp_path, pipeline_profile='v2_1')


def test_v3_routed_profile_defaults_to_cuda_without_v21_context(monkeypatch):
    cli = _cli()
    monkeypatch.setattr(cli, 'build_pipeline_context', lambda *args, **kwargs: pytest.fail('V2.1 context built'))
    captured = {}
    monkeypatch.setattr(cli, 'run_wp19_23_pipeline', lambda root, **kwargs: captured.update(kwargs) or _summary())
    assert cli.main(['--model', 'model V3', '--pipeline-profile', 'v3_routed_development',
                     '--thrust-N', '1', '--tsfc-kg-N-s', '2', '--mass-kg', '3']) == 0
    assert captured['pipeline_profile'] == 'v3_routed_development'
    assert captured['device'] == 'cuda:0'
    assert captured['pipeline_context'] is None
