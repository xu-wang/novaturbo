import pytest
import json
from pathlib import Path
from types import MappingProxyType

from src.ai import geometry19_pipeline as pipeline
from src.ai import geometry19_pipeline_dispatch as dispatch
from src.ai import geometry19_pipeline_v2_2 as v22


def test_model_v22_dispatches_without_entering_v21(monkeypatch, tmp_path):
    expected = {'status': 'V2_2_PIPELINE_COMPLETED'}
    captured = {}
    def fake(root, **kwargs):
        captured.update(kwargs)
        return expected
    monkeypatch.setattr(v22, 'run_wp19_23_pipeline_v2_2', fake)
    result = dispatch.run_wp19_23_pipeline(
        tmp_path, target_values={'thrust_N': 1., 'tsfc_kg_N_s': 1., 'mass_kg': 1.},
        model='model V2.2')
    assert result is expected
    assert captured['target_values']['thrust_N'] == 1.


def test_model_v21_is_default_and_dispatches_to_preserved_core(monkeypatch, tmp_path):
    expected = {'status': 'PIPELINE_COMPLETED'}
    monkeypatch.setattr(pipeline, 'run_wp19_23_pipeline', lambda root, **kwargs: expected)
    assert dispatch.run_wp19_23_pipeline(tmp_path, target_values={}) is expected


def test_model_v3_dispatches_to_reserved_entry(monkeypatch, tmp_path):
    from src.ai import geometry19_pipeline_v3 as v3
    captured = {}
    def fake(root, **kwargs):
        captured.update(kwargs)
        return {'status': 'TEST_RESERVED'}
    monkeypatch.setattr(v3, 'run_wp19_23_pipeline_v3', fake)
    result = dispatch.run_wp19_23_pipeline(
        tmp_path, target_values={}, model='model V3')
    assert result['status'] == 'TEST_RESERVED'
    assert captured['target_values'] == {}


def test_unknown_model_is_rejected_before_runtime_resolution(tmp_path):
    with pytest.raises(pipeline.Geometry19PipelineError, match='model V2.1'):
        dispatch.run_wp19_23_pipeline(
            tmp_path, target_values={'thrust_N': 1., 'tsfc_kg_N_s': 1., 'mass_kg': 1.},
            model='V2.3')


def test_v22_target_adapter_preserves_requested_performance():
    target = {'target_id': 't', 'target_type': 'ID', 'thrust_N': 7.,
              'tsfc_kg_N_s': 2e-4, 'mass_kg': 3.}
    sentinel = MappingProxyType({'frozen': True})
    data = {'runtime_targets': [], 'targets': [], 'context': sentinel}
    result = v22._target_data(data, target)
    assert result['runtime_targets'][0]['performance'] == {
        'thrust_N': 7., 'tsfc_kg_N_s': 2e-4, 'mass_kg': 3.}
    assert result['targets'][0]['target_class'] == 'id'
    assert result['context'] is sentinel


def test_v22_rejects_mesh_storage_without_export_before_preflight(tmp_path):
    with pytest.raises(v22.Geometry19V22PipelineError, match='requires --export-mesh'):
        v22.run_wp19_23_pipeline_v2_2(
            tmp_path, target_values={}, storage_root=tmp_path / 'store'
        )


def test_v22_rejects_v21_profile_selection_before_preflight(tmp_path):
    with pytest.raises(v22.Geometry19V22PipelineError, match='model V2.1 protocols'):
        v22.run_wp19_23_pipeline_v2_2(
            tmp_path, target_values={}, pipeline_profile='v2_2_development'
        )


def test_v22_mesh_publication_uses_delivered_hash_without_physics(monkeypatch, tmp_path):
    source_design_hash = 'd' * 64
    design_id = 'a' * 64
    core = tmp_path / 'core'
    runtime = core / 'runtime'
    runtime.mkdir(parents=True)
    (runtime / 'Delivery_Geometry.json').write_text(json.dumps({'geometry': 19}))
    sentinel_design = type(
        'Design', (), {'as_mapping': lambda self: {'geometry': 19}}
    )()
    monkeypatch.setattr(v22.Geometry19, 'from_mapping', lambda payload: sentinel_design)

    calls = []
    class Store:
        def __init__(self, root, storage, physics_runtime=None):
            calls.append(('init', Path(storage), physics_runtime))
        def publish_mesh(self, design, quality):
            target = tmp_path / 'store' / design_id / quality
            target.mkdir(parents=True)
            component = target / 'compressor.stl'
            component.write_bytes(quality.encode())
            manifest = {
                'design_id': design_id,
                'components': {'compressor': {
                    'filename': component.name,
                    'sha256': v22.file_hash(component),
                    'size_bytes': component.stat().st_size,
                }},
            }
            manifest_path = target / 'manifest.json'
            manifest_path.write_text(json.dumps(manifest))
            calls.append(('publish', design, quality))
            return {
                'design_id': design_id,
                'geometry19': {'geometry': 19},
                'mesh_diagnostics': {'passed': True},
                'manifest_path': str(manifest_path),
                'manifest_sha256': v22.file_hash(manifest_path),
            }
        def get_components(self, requested_hash, quality):
            assert requested_hash == design_id
            return {'components': {'compressor': {}}}

    monkeypatch.setattr(v22, 'Geometry19DesignStore', Store)
    data = {'context': type('Context', (), {'physics_runtime': object()})()}
    assets = v22._publish_delivery_meshes(
        tmp_path, data, core, tmp_path / 'store', source_design_hash
    )
    assert set(assets) == {'preview_mesh', 'export_mesh'}
    assert assets['export_mesh']['source_design_hash'] == source_design_hash
    assert assets['export_mesh']['design_id'] == design_id
    assert [call[2] for call in calls if call[0] == 'publish'] == [
        'preview_mesh', 'export_mesh'
    ]


def test_v22_mesh_diagnostic_failure_stops_delivery(monkeypatch, tmp_path):
    core = tmp_path / 'core'
    runtime = core / 'runtime'
    runtime.mkdir(parents=True)
    (runtime / 'Delivery_Geometry.json').write_text(json.dumps({'geometry': 19}))
    design = type('Design', (), {'as_mapping': lambda self: {'geometry': 19}})()
    monkeypatch.setattr(v22.Geometry19, 'from_mapping', lambda payload: design)

    class Store:
        def __init__(self, *args, **kwargs):
            pass
        def publish_mesh(self, design, quality):
            return {
                'design_id': 'a' * 64,
                'geometry19': {'geometry': 19},
                'mesh_diagnostics': {'passed': False},
            }

    monkeypatch.setattr(v22, 'Geometry19DesignStore', Store)
    data = {'context': type('Context', (), {'physics_runtime': object()})()}
    with pytest.raises(v22.Geometry19V22PipelineError, match='diagnostics failed'):
        v22._publish_delivery_meshes(
            tmp_path, data, core, tmp_path / 'store', 'd' * 64
        )
