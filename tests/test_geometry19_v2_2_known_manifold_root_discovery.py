"""Tests for the frozen V2.2 known-manifold root-discovery campaign."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import yaml

from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.physics.geometry19_v2_2_dataset import _read_jsonl
from src.physics.geometry19_v2_2_known_manifold_root_discovery import (
    KNOWN_MANIFOLD_ROOT_VERSION,
    _global_root_seed_plan,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPO_ROOT
    / "config/physics/geometry19_v2_2_known_manifold_root_discovery_v1.yaml"
)
EVIDENCE = REPO_ROOT / "data/validation/v2_2_known_manifold_root_discovery_v1"


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_global_seed_initialization_is_deterministic_and_has_no_parent() -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    base = _read_jsonl(REPO_ROOT / protocol["base_development"]["path"])
    context = build_pipeline_context(REPO_ROOT, pipeline_profile="v2_2_development")
    variables = list(context.physics_runtime.design_schema["design_variables"])
    kwargs = {
        "dataset_version": KNOWN_MANIFOLD_ROOT_VERSION,
        "segment_id": "low_mass_low_to_mid",
        "candidate_count": 32,
        "random_seed": 1234,
        "explained_variance_fraction": 0.995,
        "latent_scale": 0.72,
        "normal_score_clip": 2.5,
    }
    first, first_audit = _global_root_seed_plan(base, variables, **kwargs)
    second, second_audit = _global_root_seed_plan(base, variables, **kwargs)

    assert first == second
    assert first_audit == second_audit
    assert len(first) == 32
    assert len({row["root_seed_id"] for row in first}) == 32
    assert all(row["parent_seed_id"] is None for row in first)
    assert all(
        row["root_origin"] == "independent_global_root_center_pca_lhs"
        for row in first
    )


def test_frozen_known_manifold_root_discovery_passes_all_root_gates() -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    manifest = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    base = _read_jsonl(REPO_ROOT / protocol["base_development"]["path"])
    retained = _read_jsonl(EVIDENCE / "qualified_new_root_rows.jsonl")
    combined = _read_jsonl(EVIDENCE / "development_dataset.jsonl")
    folds = json.loads(
        (EVIDENCE / "development_root_folds.json").read_text(encoding="utf-8")
    )

    assert manifest["status"] == "KNOWN_MANIFOLD_ROOT_COVERAGE_PASS"
    assert manifest["root_count_gate_pass"] is True
    assert manifest["fold_gate_pass"] is True
    assert manifest["development_training_ready"] is True
    assert manifest["confirmation_content_read"] is False
    assert manifest["confirmation_reuse_allowed"] is False
    assert manifest["inter_branch_extrapolation_in_scope"] is False
    assert manifest["formal_use_allowed"] is False
    assert manifest["deployment_allowed"] is False

    assert manifest["counts"]["seed_evaluated"] == 8192
    assert manifest["counts"]["seed_training_candidates"] == 770
    assert manifest["counts"]["local_evaluated"] == 2048
    assert manifest["counts"]["local_training_candidates"] == 1919
    assert manifest["counts"]["development_rows_before"] == 3489
    assert manifest["counts"]["development_rows_after"] == 4529
    assert manifest["counts"]["global_roots_before"] == 52
    assert manifest["counts"]["global_roots_after"] == 78

    old_roots = {str(row["root_seed_id"]) for row in base}
    new_roots = {str(row["root_seed_id"]) for row in retained}
    assert old_roots.isdisjoint(new_roots)
    assert len(new_roots) == 26
    assert len(retained) == 1040
    assert len(combined) == 4529
    assert len({row["design_hash"] for row in combined}) == len(combined)
    assert all(row["training_candidate"] for row in retained)
    assert all(
        row["root_origin"] == "independent_global_root_center_pca_lhs"
        for row in retained
    )

    for segment in protocol["sampling"]["segments"]:
        values = manifest["counts"]["by_segment"][segment]
        assert values["roots_before"] == 27
        assert values["selected_seed_pool"] == 16
        assert values["qualified_seed_roots"] == 16
        assert values["retained_new_roots"] == 13
        assert values["roots_after"] == 40
        assert values["effective_roots_after"] >= 35.0
        assert values["validation_roots_by_fold"] == [8, 8, 8, 8, 8]
        segment_new_roots = {
            str(row["root_seed_id"])
            for row in retained
            if row["assigned_segment"] == segment
        }
        assert len(segment_new_roots) == 13
        assert all(
            sum(
                row["assigned_segment"] == segment
                and str(row["root_seed_id"]) == root_id
                for row in retained
            )
            == 40
            for root_id in segment_new_roots
        )

    assignments = folds["root_assignments"]
    assert set(assignments) == {str(row["root_seed_id"]) for row in combined}
    assert set(assignments.values()) == {0, 1, 2, 3, 4}
    assert all(
        summary["validation_roots_by_target_segment"] == {
            "low_mass_low_to_mid": 8,
            "low_mass_mid_to_high": 8,
        }
        for summary in folds["fold_summaries"]
    )

    paths = manifest["paths"]
    assert manifest["hashes"]["accepted_new_root_rows_hash"] == _hash(
        REPO_ROOT / paths["accepted_new_root_rows"]
    )
    assert manifest["hashes"]["qualified_new_root_rows_hash"] == _hash(
        REPO_ROOT / paths["qualified_new_root_rows"]
    )
    assert manifest["hashes"]["development_dataset_hash"] == _hash(
        REPO_ROOT / paths["development_dataset"]
    )
    assert manifest["hashes"]["development_root_folds_hash"] == _hash(
        REPO_ROOT / paths["development_root_folds"]
    )


def test_retained_roots_are_bound_to_a_global_seed_shard() -> None:
    manifest = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    retained_ids = {
        root_id
        for values in manifest["retained_root_ids"].values()
        for root_id in values
    }
    global_seeds = []
    for segment in manifest["sampling"]["segments"]:
        global_seeds.extend(_read_jsonl(EVIDENCE / "shards" / f"global-{segment}.jsonl"))
    seed_by_root = {str(row["root_seed_id"]): row for row in global_seeds}

    assert retained_ids <= set(seed_by_root)
    assert all(seed_by_root[root_id]["training_candidate"] for root_id in retained_ids)
    assert all(seed_by_root[root_id]["parent_seed_id"] is None for root_id in retained_ids)
    assert all(
        0.055
        <= float(
            next(
                selected["minimum_l2_to_existing"]
                for audit in manifest["seed_model_audit"].values()
                for selected in audit["selected"]
                if selected["root_seed_id"] == root_id
            )
        )
        <= 0.65
        for root_id in retained_ids
    )
