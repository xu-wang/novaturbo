from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from src.ai.stage2_baseline import verify_manifest


def _write_manifest(root: Path, artifact: Path, sha256: str, **semantic):
    manifest = {
        "baseline_id": "test_baseline",
        "immutable": True,
        "code": {"source_commit": "unused"},
        "artifacts": [
            {
                "path": artifact.relative_to(root).as_posix(),
                "bytes": artifact.stat().st_size,
                "sha256": sha256,
            }
        ],
        "semantic_checks": [
            {
                "kind": "csv",
                "path": artifact.relative_to(root).as_posix(),
                **semantic,
            }
        ],
    }
    path = root / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return path


def test_verify_manifest_accepts_matching_artifact_and_csv_contract(tmp_path):
    artifact = tmp_path / "sample.csv"
    artifact.write_text("split,value\ntrain,1\nval,2\ntrain,3\n", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = _write_manifest(
        tmp_path,
        artifact,
        digest,
        row_count=3,
        value_counts={"split": {"train": 2, "val": 1}},
    )

    result = verify_manifest(manifest, repo_root=tmp_path, check_git=False)

    assert result["ok"] is True
    assert result["checked_artifacts"] == 1
    assert result["issues"] == []


def test_verify_manifest_reports_sha256_mismatch(tmp_path):
    artifact = tmp_path / "sample.csv"
    artifact.write_text("split\ntrain\n", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        artifact,
        "0" * 64,
        row_count=1,
        value_counts={"split": {"train": 1}},
    )

    result = verify_manifest(manifest, repo_root=tmp_path, check_git=False)

    assert result["ok"] is False
    assert result["issues"][0]["error"] == "sha256_mismatch"


def test_verify_manifest_reports_csv_semantic_mismatch(tmp_path):
    artifact = tmp_path / "sample.csv"
    artifact.write_text("split\ntrain\nval\n", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = _write_manifest(
        tmp_path,
        artifact,
        digest,
        row_count=3,
        value_counts={"split": {"train": 2}},
    )

    result = verify_manifest(manifest, repo_root=tmp_path, check_git=False)

    assert result["ok"] is False
    errors = {issue["error"] for issue in result["issues"]}
    assert errors == {"row_count_mismatch", "value_counts_mismatch"}


def test_s2_wp4_cfm_freeze_manifest_is_complete_and_valid():
    repo_root = Path(__file__).resolve().parents[1]
    manifest = repo_root / "config" / "stage2" / "s2_wp4_cfm_freeze_v1.yaml"

    result = verify_manifest(manifest, repo_root=repo_root, check_git=True)

    assert result["ok"] is True, result["issues"]
    assert result["baseline_id"] == "s2_wp4_cfm_freeze_v1"
    assert result["checked_artifacts"] == 26
    assert result["git"]["source_is_ancestor"] is True
    assert set(result["semantic_checks"]) == {
        "config/experiments/cfm_arch/cfm_0_generalization_targets.csv",
        "results/stage2/s2_wp4b/s2_wp4b_ledger.csv",
        "results/stage2/s2_wp4c/s2_wp4c_ledger.csv",
        "results/stage2/s2_wp4d/s2_wp4d_ledger.csv",
    }
