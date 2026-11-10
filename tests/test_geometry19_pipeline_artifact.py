"""Historical results remain verifiable after selector source changes."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from src.ai.geometry19_pipeline_artifact import (
    PipelineArtifactError, verify_completed_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / 'data/validation/v2_2_pipeline_integration_smoke_v1'


def test_completed_v22_result_is_readable_without_current_code_identity():
    result = verify_completed_artifact(HISTORICAL, ROOT)
    assert result['integrity_pass'] is True
    assert result['listed_file_count'] == 25
    assert result['model'] == 'model V2.2'
    assert result['resume_compatibility_assessed'] is False
    assert result['current_code_identity_matches_lock'] is False


def test_read_only_verifier_rejects_changed_listed_file(tmp_path):
    output = tmp_path / 'historical'
    shutil.copytree(HISTORICAL, output)
    (output / 'Pipeline_Summary.json').write_text('{}')
    with pytest.raises(PipelineArtifactError, match='Artifact changed: Pipeline_Summary.json'):
        verify_completed_artifact(output, ROOT)


def test_read_only_verifier_rejects_path_escape(tmp_path):
    output = tmp_path / 'historical'
    shutil.copytree(HISTORICAL, output)
    path = output / 'Complete_Manifest.json'
    complete = json.loads(path.read_text())
    complete['files']['../outside'] = '0' * 64
    path.write_text(json.dumps(complete))
    with pytest.raises(PipelineArtifactError, match='escapes its directory'):
        verify_completed_artifact(output, ROOT)
