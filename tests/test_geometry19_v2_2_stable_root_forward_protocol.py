"""Tests for the stable-root S2/S3 protocol lock."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.ai.geometry19_surrogate import sha256_file
from src.ai.geometry19_v2_2_stable_root_forward_protocol import (
    LOCK_PASS,
    Geometry19StableRootForwardProtocolError,
    lock_stable_root_forward_protocol,
    validate_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    REPO_ROOT
    / "config/experiments/geometry19_v2_2_stable_root_forward_fresh_eval_v1.yaml"
)
EVIDENCE = REPO_ROOT / "data/validation/v2_2_stable_root_forward_fresh_eval_v1"


def test_protocol_keeps_fresh_evaluation_and_inverse_design_locked() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_protocol(config)
    config["S2_model_development"]["fresh_evaluation_visible_during_selection"] = True
    with pytest.raises(Geometry19StableRootForwardProtocolError, match="leakage"):
        validate_protocol(config)


def test_frozen_protocol_lock_has_exact_population_models_and_ordering() -> None:
    lock = json.loads(
        (EVIDENCE / "audits/S2_S3_Protocol_Lock.json").read_text(encoding="utf-8")
    )
    assert lock["status"] == LOCK_PASS
    assert lock["stable_roots"] == 38
    assert lock["stable_development_rows"] == 2935
    assert lock["all_root_comparator_roots"] == 78
    assert lock["all_root_comparator_rows"] == 4529
    assert lock["model_arm_ids"] == [
        "all_78_global_root_balanced_ridge",
        "stable_38_pooled_root_balanced_ridge",
        "stable_38_nearest_root_local_ridge",
        "stable_38_oracle_root_local_ridge",
        "stable_38_global_plus_root_residual_ridge",
    ]
    assert lock["fresh_evaluation_attempts_per_root"] == 96
    assert lock["fresh_evaluation_maximum_attempts"] == 3648
    assert lock["s2_training_authorized"] is True
    assert lock["s3_generation_authorized_now"] is False
    assert lock["model_training_executed"] is False
    assert lock["fresh_coordinates_generated"] is False
    assert lock["fresh_physics_labels_generated"] is False
    assert lock["inverse_design_authorized"] is False
    assert lock["original_local_chart_p4_authorized"] is False
    assert lock["confirmation_content_read"] is False
    assert all(lock["future_artifacts_absent_at_lock"].values())
    assert lock["protocol_sha256"] == sha256_file(CONFIG)
    assert lock["report_sha256"] == sha256_file(REPO_ROOT / lock["report_path"])


def test_design_hash_inner_split_is_root_stratified_and_lineage_limited() -> None:
    lock = json.loads(
        (EVIDENCE / "audits/S2_S3_Protocol_Lock.json").read_text(encoding="utf-8")
    )
    audit = lock["inner_split_audit"]
    assert audit["all_38_roots_represented_in_every_validation_fold"] is True
    assert audit["fold_roots"] == {"0": 38, "1": 38, "2": 38, "3": 38}
    assert sum(audit["fold_rows"].values()) == 2935
    assert audit["roots_with_fewer_than_4_parent_groups"] == 28
    assert audit["parent_seed_id_overlap_is_known_inner_selection_limitation"] is True
    assert audit["final_claim_depends_on_fresh_S3"] is True
    assert all(
        sum(fold_counts.values()) >= 24
        and set(fold_counts) == {"0", "1", "2", "3"}
        for fold_counts in audit["root_fold_row_counts"].values()
    )


def test_protocol_lock_never_text_parses_confirmation(
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
    result = lock_stable_root_forward_protocol(REPO_ROOT, CONFIG, output_root=tmp_path)
    assert result["confirmation_hash_checked"] is True
    assert result["confirmation_content_read"] is False
    assert result["model_training_executed"] is False
    assert result["fresh_coordinates_generated"] is False
