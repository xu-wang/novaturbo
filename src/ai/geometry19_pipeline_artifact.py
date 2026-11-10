"""Read-only integrity checks for completed historical pipeline artifacts."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any


class PipelineArtifactError(ValueError):
    """A recorded artifact is missing, changed, or escapes its directory."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PipelineArtifactError(message)


def _hash(path: Path) -> str:
    _require(path.is_file(), f'Missing artifact: {path}')
    digest = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise PipelineArtifactError(f'Cannot read artifact JSON: {path}') from exc
    _require(isinstance(value, dict), f'Artifact JSON must be an object: {path}')
    return value


def _within(root: Path, name: str) -> Path:
    _require(isinstance(name, str) and bool(name), 'Artifact path must be nonempty')
    _require(not Path(name).is_absolute(), f'Absolute artifact path is forbidden: {name}')
    path = (root / name).resolve()
    _require(path.is_relative_to(root), f'Artifact path escapes its directory: {name}')
    return path


def verify_completed_artifact(
    output_root: str | Path, repository_root: str | Path,
) -> dict[str, Any]:
    """Verify recorded bytes without requiring current code to match the old run."""
    output = Path(output_root).resolve()
    repository = Path(repository_root).resolve()
    complete = _json(output / 'Complete_Manifest.json')
    if complete.get('manifest_version') in {
        'geometry22_v3_development_pipeline_v1',
        'geometry22_pipeline_v3_routed_development_v1',
        'geometry22_pipeline_v3_cfm_development_v1',
        'geometry22_pipeline_v3_guided_development_v1',
    }:
        return _verify_v3_artifact(output, repository, complete)
    files = complete.get('files')
    _require(isinstance(files, dict) and bool(files), 'Complete manifest has no file list')
    for required in ('Protocol_Lock.json', 'Pipeline_Summary.json', 'pipeline_manifest.json'):
        _require(required in files, f'Complete manifest omits {required}')
    for name, expected in files.items():
        _require(isinstance(expected, str) and len(expected) == 64, f'Invalid hash for {name}')
        _require(_hash(_within(output, name)) == expected, f'Artifact changed: {name}')

    lock = _json(output / 'Protocol_Lock.json')
    canonical_lock = json.dumps(
        lock, sort_keys=True, separators=(',', ':'), allow_nan=False,
    ).encode('utf-8')
    _require(
        sha256(canonical_lock).hexdigest() == complete.get('lock_sha256'),
        'Protocol lock digest changed',
    )
    summary = _json(output / 'Pipeline_Summary.json')
    _require(summary.get('model') == lock.get('model'), 'Summary model differs from lock')
    assets = summary.get('mesh_assets', {})
    _require(isinstance(assets, dict), 'Mesh assets must be a mapping')
    verified_assets = {}
    for quality, asset in assets.items():
        _require(isinstance(asset, dict), f'Invalid mesh asset: {quality}')
        manifest_path = Path(asset['manifest_path']).resolve()
        _require(
            _hash(manifest_path) == asset['manifest_sha256'],
            f'Mesh manifest changed: {quality}',
        )
        mesh = _json(manifest_path)
        _require(mesh.get('design_id') == asset.get('design_id'), 'Mesh design ID changed')
        components = mesh.get('components')
        _require(isinstance(components, dict) and bool(components), 'Mesh components missing')
        for component in components.values():
            name = component['filename']
            _require(
                _hash(_within(manifest_path.parent, name)) == component['sha256'],
                f'Mesh component changed: {quality}/{name}',
            )
        verified_assets[quality] = len(components)

    mismatches = []
    for name, expected in lock.get('code', {}).items():
        path = _within(repository, name)
        if not path.is_file() or _hash(path) != expected:
            mismatches.append(name)
    return {
        'integrity_pass': True,
        'model': summary['model'],
        'status': summary['status'],
        'listed_file_count': len(files),
        'verified_mesh_assets': verified_assets,
        'current_code_identity_matches_lock': not mismatches,
        'current_code_mismatches': mismatches,
        'resume_compatibility_assessed': False,
    }


def _verify_v3_artifact(output: Path, repository: Path, complete: dict) -> dict:
    """Serve archived V3 results using their recorded identity, without CUDA inference."""
    files = complete.get('files')
    _require(isinstance(files, dict) and bool(files), 'V3 complete manifest has no files')
    for name in ('Protocol_Lock.json', 'Pipeline_Summary.json'):
        _require(name in files, f'V3 manifest omits {name}')
    for name, digest in files.items():
        _require(_hash(_within(output, name)) == digest, f'Artifact changed: {name}')
    lock = _json(output / 'Protocol_Lock.json')
    identity = lock['identity']
    try:
        if complete['manifest_version'] == 'geometry22_v3_development_pipeline_v1':
            from src.ai.geometry19_pipeline_v3 import _verify_complete
            summary = _verify_complete(output, identity)
        else:
            if complete['manifest_version'] == 'geometry22_pipeline_v3_guided_development_v1':
                from src.ai.geometry22_pipeline_v3_guided import VERSION, STAGES, verify_state, verify_relationships
            elif complete['manifest_version'] == 'geometry22_pipeline_v3_cfm_development_v1':
                from src.ai.geometry22_pipeline_v3_cfm import VERSION, STAGES, verify_state, verify_relationships
            else:
                from src.ai.geometry22_pipeline_v3_routed import VERSION, STAGES, verify_state, verify_relationships
            state = verify_state(output, identity)
            _require(state['status'] == 'COMPLETE', 'V3 routed run is not complete')
            _require(complete == {'manifest_version': VERSION, 'stages': STAGES, 'files': state['files'],
                                 'run_state_sha256': _hash(output / 'Run_State.json')}, 'V3 routed complete manifest changed')
            verify_relationships(output, identity, identity['protocol'], STAGES)
            summary = _json(output / 'Pipeline_Summary.json')
    except ValueError as exc:
        raise PipelineArtifactError(str(exc)) from exc
    mismatches = [name for name, digest in identity['source_code_sha256'].items()
                  if not _within(repository, name).is_file() or _hash(_within(repository, name)) != digest]
    return {'integrity_pass': True, 'model': summary['model'], 'status': summary['status'],
            'listed_file_count': len(files), 'verified_mesh_assets': {},
            'current_code_identity_matches_lock': not mismatches, 'current_code_mismatches': mismatches,
            'resume_compatibility_assessed': False, 'summary': summary}
