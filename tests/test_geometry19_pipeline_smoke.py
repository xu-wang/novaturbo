"""Simplified-V2.2 full-pipeline smoke artifact contract tests."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from src.ai.geometry19_pipeline_smoke import (
    EXPECTED_STAGES,
    Geometry19PipelineSmokeError,
    run_v2_2_development_smoke,
    validate_v2_2_development_smoke,
)


MARKERS = {
    "simplified_model": True,
    "development_only": True,
    "formal_use_allowed": False,
    "target_hit": False,
}


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _smoke_fixture(root: Path) -> None:
    identity_hashes = {
        "pipeline_profile_hash": "1" * 64,
        "pipeline_context_hash": "2" * 64,
        "physics_bundle_hash": "3" * 64,
    }
    _write(
        root / "pipeline_summary.json",
        {
            "status": "PASS",
            "pipeline_profile": "v2_2_development",
            "pipeline_profile_mode": "production_development",
            "completed_stages": list(EXPECTED_STAGES),
            **identity_hashes,
            **MARKERS,
        },
    )
    _write(
        root / "wp19_23b/direct/summary.json",
        {"candidate_protocol": {"n_raw": 256}},
    )
    candidate = {"design_id": "a" * 64, **MARKERS}
    counts = {
        "delivered_count": 1,
        "selected_physics_recalculation_passed": 1,
        "preview_mesh_passed": 1,
        "json_round_trip_passed": 1,
        "ui_design_id_consistent": 1,
    }
    _write(root / "wp19_23f/summary.json", {"counts": counts, **MARKERS})
    _write(root / "wp19_23f/delivered_candidates.json", [candidate])
    _write(root / f"wp19_23f/ui_candidates/{'a' * 64}.json", candidate)
    stage_directories = (
        "wp19_23b/direct",
        "wp19_23c",
        "wp19_23d",
        "wp19_23e",
        "wp19_23f",
    )
    stage_records = []
    for stage_name, directory in zip(EXPECTED_STAGES, stage_directories, strict=True):
        stage_root = root / directory
        if not (stage_root / "summary.json").exists():
            _write(stage_root / "summary.json", {"work_package": stage_name})
        _write(stage_root / "manifest.json", {"stage": stage_name})
        stage_records.append(
            {
                "stage": stage_name,
                "directory": directory,
                "summary_sha256": _digest(stage_root / "summary.json"),
                "manifest_sha256": _digest(stage_root / "manifest.json"),
            }
        )
    _write(
        root / "pipeline_manifest.json",
        {
            "stage_order": list(EXPECTED_STAGES),
            "stages": stage_records,
            "hashes": {
                **identity_hashes,
                "pipeline_summary_hash": _digest(root / "pipeline_summary.json"),
            },
        },
    )


def test_v2_2_smoke_validator_accepts_complete_development_artifacts(
    tmp_path: Path,
) -> None:
    _smoke_fixture(tmp_path)

    result = validate_v2_2_development_smoke(
        tmp_path,
        elapsed_seconds=24.0,
        maximum_runtime_seconds=120.0,
    )

    assert result["status"] == "SMOKE_PASS"
    assert result["raw_candidate_count"] == 256
    assert result["delivered_count"] == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("marker", "development markers differ"),
        ("budget", "Raw candidate budget differs"),
        ("runtime", "exceeded CPU runtime budget"),
    ],
)
def test_v2_2_smoke_validator_rejects_contract_drift(
    tmp_path: Path, mutation: str, message: str
) -> None:
    _smoke_fixture(tmp_path)
    if mutation == "marker":
        summary_path = tmp_path / "pipeline_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["formal_use_allowed"] = True
        _write(summary_path, summary)
        manifest_path = tmp_path / "pipeline_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["hashes"]["pipeline_summary_hash"] = _digest(summary_path)
        _write(manifest_path, manifest)
    elif mutation == "budget":
        generation_path = tmp_path / "wp19_23b/direct/summary.json"
        _write(generation_path, {"candidate_protocol": {"n_raw": 128}})
        manifest_path = tmp_path / "pipeline_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["stages"][0]["summary_sha256"] = _digest(generation_path)
        _write(manifest_path, manifest)

    with pytest.raises(Geometry19PipelineSmokeError, match=message):
        validate_v2_2_development_smoke(
            tmp_path,
            elapsed_seconds=121.0 if mutation == "runtime" else 24.0,
            maximum_runtime_seconds=120.0,
        )


def test_v2_2_smoke_runner_rejects_invalid_time_budget_before_running(
    tmp_path: Path,
) -> None:
    with pytest.raises(Geometry19PipelineSmokeError, match="finite and positive"):
        run_v2_2_development_smoke(
            tmp_path,
            tmp_path / "target.yaml",
            tmp_path / "run",
            maximum_runtime_seconds=0.0,
        )

    assert not (tmp_path / "run").exists()
