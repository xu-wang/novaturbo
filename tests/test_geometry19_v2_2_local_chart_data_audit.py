"""Tests for local-chart MoE P1 data/root/fold integrity evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.ai.geometry19_v2_2_local_chart_data_audit import (
    P1_PASS_STATUS,
    _effective_covariance_rank,
    run_data_root_fold_audit,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config/experiments/geometry19_v2_2_local_chart_moe_v1.yaml"
EVIDENCE = REPO_ROOT / "data/validation/v2_2_local_chart_moe_v1/audits"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_effective_covariance_rank_uses_centered_standard_numerical_rank() -> None:
    x = np.asarray(
        [
            [0.0, 0.0, 3.0],
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 3.0],
            [4.0, 8.0, 3.0],
        ],
        dtype=np.float64,
    )
    rank, tolerance, eigenvalues = _effective_covariance_rank(x)
    assert rank == 1
    assert tolerance > 0.0
    assert len(eigenvalues) == 3
    assert eigenvalues[0] > 0.0
    assert eigenvalues[1] == pytest.approx(0.0, abs=1.0e-12)


def test_frozen_p1_evidence_satisfies_row_lineage_and_fold_contracts() -> None:
    summary = json.loads((EVIDENCE / "input_audit.json").read_text(encoding="utf-8"))
    roots = _read_jsonl(EVIDENCE / "root_inventory.jsonl")
    folds = json.loads((EVIDENCE / "fold_inventory.json").read_text(encoding="utf-8"))

    assert summary["status"] == P1_PASS_STATUS
    assert summary["row_contract"]["rows"] == 4529
    assert summary["row_contract"]["unique_design_hashes"] == 4529
    assert summary["row_contract"]["all_training_candidate"] is True
    assert summary["row_contract"]["all_physics_domain_valid"] is True
    assert summary["row_contract"]["all_performance_finite_positive"] is True
    assert summary["root_contract"]["global_roots"] == 78
    assert summary["root_contract"]["existing_roots"] == 52
    assert summary["root_contract"]["new_roots"] == 26
    assert summary["root_contract"]["new_existing_root_overlap"] == []
    assert len(roots) == 78
    assert all(len(root["design_center"]) == 19 for root in roots)
    assert all(len(root["performance_center"]) == 3 for root in roots)
    assert all(len(root["normalized_covariance_eigenvalues"]) == 19 for root in roots)
    assert folds["cross_fold_root_leakage"] is False
    assert len(folds["folds"]) == 5
    assert all(
        list(fold["validation_roots_by_target_segment"].values()) == [8, 8]
        for fold in folds["folds"]
    )
    assert summary["confirmation_content_read"] is False
    assert summary["model_work_executed"] is False


def test_p1_artifact_hashes_and_frozen_protocol_are_consistent() -> None:
    summary = json.loads((EVIDENCE / "input_audit.json").read_text(encoding="utf-8"))
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    from src.ai.geometry19_surrogate import sha256_file

    assert summary["protocol_sha256"] == sha256_file(CONFIG)
    assert summary["artifacts"]["root_inventory_sha256"] == sha256_file(
        EVIDENCE / "root_inventory.jsonl"
    )
    assert summary["artifacts"]["fold_inventory_sha256"] == sha256_file(
        EVIDENCE / "fold_inventory.json"
    )
    assert summary["artifacts"]["report_sha256"] == sha256_file(
        EVIDENCE / "P1_Data_Root_Fold_Audit_Report.md"
    )
    assert config["inputs"]["sealed_confirmation"]["access"] == "sha256_only_no_parse"


def test_p1_never_text_parses_sealed_confirmation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    confirmation = (REPO_ROOT / config["inputs"]["sealed_confirmation"]["path"]).resolve()
    original = Path.read_text

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.resolve() == confirmation:
            raise AssertionError("sealed confirmation content was parsed")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    source_p0 = EVIDENCE / "protocol_input_lock.json"
    target_p0 = tmp_path / "audits/protocol_input_lock.json"
    target_p0.parent.mkdir(parents=True)
    target_p0.write_bytes(source_p0.read_bytes())
    result = run_data_root_fold_audit(REPO_ROOT, CONFIG, output_root=tmp_path)
    assert result["status"] == P1_PASS_STATUS
    assert result["confirmation_hash_checked"] is True
    assert result["confirmation_content_read"] is False
