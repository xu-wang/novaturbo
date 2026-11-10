"""Production entry adapter for the validated V2.2 local B--F flow."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.ai import geometry19_performance_gated_g3 as g3
from src.ai.geometry19_local_full_pipeline_entry import run_target, verify_target
from src.ai.geometry19_performance_gated_protocol import file_hash, require, within
from src.physics import Geometry19
from src.visualization.geometry19_service import Geometry19DesignStore

CONFIG = 'config/experiments/geometry19_pipeline_v2_2.yaml'
MODEL_SELECTOR = 'model V2.2'


class Geometry19V22PipelineError(ValueError):
    """Raised when the production V2.2 adapter contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Geometry19V22PipelineError(message)


def _target_data(data: dict, target: Mapping[str, Any]) -> dict:
    # PhysicsRuntime contains immutable mappingproxy members; preserve those
    # verified objects and replace only the two target collections.
    result = dict(data)
    performance = {name: float(target[name]) for name in g3.adapter.OUTPUT_NAMES}
    result['runtime_targets'] = [{
        'performance': performance,
        'target_id': str(target['target_id']),
        'units': {'thrust_N': 'N', 'tsfc_kg_N_s': 'kg/(N*s)', 'mass_kg': 'kg'},
    }]
    result['targets'] = [{
        'target_index': 0, 'target_id': str(target['target_id']),
        'target_class': str(target['target_type']).lower(), 'performance': performance,
    }]
    return result


def _verify_mesh_asset(asset: Mapping[str, Any]) -> None:
    manifest_path = Path(str(asset['manifest_path'])).resolve()
    _require(manifest_path.is_file(), 'V2.2 mesh manifest is missing')
    _require(file_hash(manifest_path) == asset['manifest_sha256'], 'V2.2 mesh manifest changed')
    manifest = json.loads(manifest_path.read_text())
    _require(manifest['design_id'] == asset['design_id'], 'V2.2 mesh design identity changed')
    for component in manifest['components'].values():
        component_path = manifest_path.parent / component['filename']
        _require(component_path.is_file(), 'V2.2 mesh component is missing')
        _require(file_hash(component_path) == component['sha256'], 'V2.2 mesh component changed')


def _verify_complete(output: Path, lock_hash: str) -> dict:
    complete = json.loads((output / 'Complete_Manifest.json').read_text())
    _require(complete['lock_sha256'] == lock_hash, 'V2.2 pipeline lock changed')
    for name, digest in complete['files'].items():
        _require(file_hash(within(output, name)) == digest, 'V2.2 pipeline artifact changed')
    summary = json.loads((output / 'Pipeline_Summary.json').read_text())
    for asset in summary.get('mesh_assets', {}).values():
        _verify_mesh_asset(asset)
    return summary


def _publish_delivery_meshes(
    root: Path,
    data: Mapping[str, Any],
    core: Path,
    storage: Path,
    source_design_hash: str,
) -> dict[str, dict[str, Any]]:
    geometry = Geometry19.from_mapping(
        json.loads((core / 'runtime' / 'Delivery_Geometry.json').read_text())
    )
    store = Geometry19DesignStore(
        root, storage, physics_runtime=data['context'].physics_runtime
    )
    assets = {}
    for quality in ('preview_mesh', 'export_mesh'):
        response = store.publish_mesh(geometry, quality)
        _require(
            response['geometry19'] == geometry.as_mapping(),
            'V2.2 mesh changed delivered Geometry19 values',
        )
        _require(response['mesh_diagnostics'].get('passed'), 'V2.2 mesh diagnostics failed')
        design_id = str(response['design_id'])
        components = store.get_components(design_id, quality)['components']
        _require(bool(components), 'V2.2 mesh produced no STL components')
        asset = {
            'source_design_hash': source_design_hash,
            'design_id': design_id,
            'quality': quality,
            'directory': str(Path(response['manifest_path']).parent),
            'manifest_path': response['manifest_path'],
            'manifest_sha256': response['manifest_sha256'],
            'component_count': len(components),
        }
        _verify_mesh_asset(asset)
        assets[quality] = asset
    return assets


def run_wp19_23_pipeline_v2_2(
    repository_root: str | Path,
    *,
    target_path: str | Path | None = None,
    target_values: Mapping[str, float] | None = None,
    target_type: str = 'ID',
    expected_status: str = 'UNKNOWN',
    model_variant: str | None = None,
    forward_model_tag: str | None = None,
    sampling_mode: str | None = None,
    device: str = 'cuda:0',
    refinement_device: str | None = None,
    output_root: str | Path | None = None,
    storage_root: str | Path | None = None,
    export_mesh: bool = False,
    resume: bool = False,
    restart_from_stage: str | None = None,
    pipeline_profile: str | None = None,
    pipeline_context: Any | None = None,
) -> dict[str, Any]:
    """Run the V2.2 local flow through the official pipeline API."""
    from src.ai.geometry19_pipeline import _load_or_build_target

    _require(model_variant is None, '--model-variant belongs to model V2.1')
    _require(forward_model_tag is None, '--forward-model-tag belongs to model V2.1')
    _require(sampling_mode in (None, 'dflow'), 'model V2.2 uses its validated dflow path')
    _require(device == 'cuda:0', 'model V2.2 currently requires logical cuda:0')
    _require(refinement_device in (None, 'cuda:0'), 'model V2.2 refinement uses logical cuda:0')
    _require(export_mesh or storage_root is None, '--storage-root requires --export-mesh')
    _require(
        pipeline_profile in (None, 'v2_1'),
        '--pipeline-profile selects model V2.1 protocols; model V2.2 uses its frozen adapter config',
    )
    _require(restart_from_stage is None, 'model V2.2 supports resume, not stage restart')
    del pipeline_profile, pipeline_context

    root = Path(repository_root).resolve()
    config = yaml.safe_load((root / CONFIG).read_text())
    target = _load_or_build_target(
        target_path, target_values, target_type=target_type, expected_status=expected_status)
    data = _target_data(g3.preflight(root), target)
    resolved_storage = (
        Path(storage_root).resolve()
        if storage_root is not None
        else within(root, data['context'].protocol['artifact_protocol']['mesh_storage_root'])
    )
    target_id = str(target['target_id'])
    output = (Path(output_root).resolve() if output_root is not None else
              within(root, config['default_output_root']) /
              data['context'].physics_runtime.physics_bundle_hash[:12] / target_id)
    code_paths = [
        CONFIG, 'src/ai/geometry19_pipeline_v2_2.py',
        'src/ai/geometry19_local_full_pipeline.py',
        'src/ai/geometry19_local_full_pipeline_entry.py',
        'src/ai/geometry19_pipeline.py', 'src/ai/geometry19_pipeline_dispatch.py',
        'scripts/run_wp19_23_pipeline.py',
        'src/visualization/geometry19_service.py',
        'src/visualization/geometry19_adapter.py',
        'src/visualization/geometry19_mesh.py',
        'config/physics/geometry19_mesh_profiles_v1.yaml',
    ]
    identity = {
        'model': MODEL_SELECTOR, 'config': config,
        'code': {name: file_hash(root / name) for name in code_paths},
        'inputs': data['inputs'], 'runtime': data['context'].contract_mapping(),
        'target': target, 'device': device,
        'delivery_assets': {
            'export_mesh': bool(export_mesh),
            'storage_root': str(resolved_storage),
        },
    }
    if resume:
        _require(output.is_dir(), 'No model V2.2 pipeline run to resume')
        lock = json.loads((output / 'Protocol_Lock.json').read_text())
        _require(all(lock[key] == value for key, value in identity.items()), 'V2.2 pipeline identity changed')
    else:
        _require(not output.exists(), f'Refusing to overwrite pipeline run: {output}')
        output.mkdir(parents=True)
        lock = {**identity, 'created_at': datetime.now(timezone.utc).isoformat()}
        g3.seal_json(output / 'Protocol_Lock.json', lock)
        g3.seal_json(output / 'target.json', target)
    lock_hash = g3.digest(lock)
    with (output / '.runner.lock').open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'Complete_Manifest.json').is_file():
            return _verify_complete(output, lock_hash)
        core = output / 'core'
        _require(not core.exists(), 'Partial model V2.2 run exists; preserve and stop')
        g3.adapter._cuda_manifest()
        models, _ = g3.adapter.load_seed_models(
            root, data['inference'], config['training_seed'], data['decoders'], device)
        scorer_ids, scorer = g3._load_model(
            data['paths']['s2_oracle_local_ridge'], 'stable_38_oracle_root_local_ridge')
        core_summary = run_target(data, config, models, scorer_ids, scorer, 0, core, lock_hash)
        verify_target(core, lock_hash)
        f_summary = json.loads((core / 'F' / 'summary.json').read_text())
        delivered = [] if f_summary['status'] != 'PASS' else [str(f_summary['design_hash'])]
        mesh_assets = {}
        if export_mesh and delivered:
            mesh_assets = _publish_delivery_meshes(
                root, data, core, resolved_storage, delivered[0]
            )
        status = 'V2_2_PIPELINE_COMPLETED' if delivered else 'V2_2_PIPELINE_NO_DELIVERED_HIT'
        summary = {
            'status': status,
            'model': MODEL_SELECTOR,
            'target_id': target_id,
            'terminal_source_stage': core_summary['first_hit_stage'] or 'E',
            'delivery_status': f_summary['status'],
            'delivered_design_ids': delivered,
            'costs': core_summary['costs'],
            'production_entry_integrated': True,
            'formal_use_allowed': False,
            'core_runner_classification': 'validated_v2_2_local_BF_core',
            'mesh_export_integrated': True,
            'mesh_export_requested': bool(export_mesh),
            'mesh_assets': mesh_assets,
            'paths': {
                'runtime_root': str(output),
                'pipeline_manifest': str(output / 'pipeline_manifest.json'),
                'core_runtime': str(core),
                'design_storage': str(resolved_storage),
            },
        }
        g3.seal_json(output / 'Pipeline_Summary.json', summary)
        pipeline_manifest = {
            'manifest_version': 'geometry19_pipeline_model_selector_v1',
            'model': MODEL_SELECTOR, 'target_id': target_id,
            'physics_bundle_hash': data['context'].physics_runtime.physics_bundle_hash,
            'protocol_lock_sha256': file_hash(output / 'Protocol_Lock.json'),
            'core_target_manifest_sha256': file_hash(core / 'Target_Manifest.json'),
            'summary_sha256': file_hash(output / 'Pipeline_Summary.json'),
            'mesh_manifests': {
                quality: {
                    'path': asset['manifest_path'],
                    'sha256': asset['manifest_sha256'],
                }
                for quality, asset in mesh_assets.items()
            },
        }
        g3.seal_json(output / 'pipeline_manifest.json', pipeline_manifest)
        files = {str(path.relative_to(output)): file_hash(path) for path in output.rglob('*')
                 if path.is_file() and path.name not in {'.runner.lock', 'Complete_Manifest.json'}}
        g3.seal_json(output / 'Complete_Manifest.json', {'lock_sha256': lock_hash, 'files': files})
        return summary
