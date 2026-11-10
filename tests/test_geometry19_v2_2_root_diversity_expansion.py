"""Tests for V2.2 independent-root data expansion."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import yaml

from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.physics.geometry19_v2_2_root_diversity_expansion import (
    _select_novel_root_models,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPO_ROOT / "config/physics/geometry19_v2_2_independent_root_expansion_v1.yaml"
EVIDENCE = REPO_ROOT / "data/validation/v2_2_independent_root_expansion_v1"


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_selected_roots_are_novel_and_design_separated() -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    source = [
        json.loads(line)
        for line in (REPO_ROOT / protocol["source_observations"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    existing = [
        json.loads(line)
        for line in (REPO_ROOT / protocol["base_dataset"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    context = build_pipeline_context(REPO_ROOT, pipeline_profile="v2_2_development")
    variables = list(context.physics_runtime.design_schema["design_variables"])
    sampling = protocol["sampling"]

    for segment in sampling["segments"]:
        models = _select_novel_root_models(
            context.target_domain,
            variables,
            source,
            existing,
            segment,
            sampling,
            sampling["rounds"][0],
        )
        existing_ids = {
            row["root_seed_id"]
            for row in existing
            if row["assigned_segment"] == segment
        }
        assert len(models) == sampling["root_pool_size_per_segment"]
        assert not ({model["root_seed_id"] for model in models} & existing_ids)
        assert all(
            model["minimum_l2_to_existing_segment_rows"]
            >= sampling["minimum_normalized_l2_separation"]
            for model in models
        )
        assert all(
            model["minimum_l2_to_selected_novel_centers"] is None
            or model["minimum_l2_to_selected_novel_centers"]
            >= sampling["minimum_normalized_l2_separation"]
            for model in models
        )


def test_frozen_independent_root_expansion_passes_diversity_gate() -> None:
    manifest = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    accepted_path = EVIDENCE / "accepted_new_roots.jsonl"
    combined_path = EVIDENCE / "combined_accepted_dataset.jsonl"
    accepted = [
        json.loads(line)
        for line in accepted_path.read_text(encoding="utf-8").splitlines()
    ]
    base = [
        json.loads(line)
        for line in (
            REPO_ROOT
            / "data/validation/v2_2_directed_segment_recovery_v1/combined_accepted_dataset.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert manifest["status"] == "ROOT_DIVERSITY_GATE_PASS"
    assert manifest["data_expansion_gate_pass"] is True
    assert manifest["formal_use_allowed"] is False
    assert manifest["surrogate_retraining_ready"] is False
    assert manifest["counts"]["evaluated"] == 2816
    assert manifest["counts"]["physics_domain_valid"] == 1696
    assert manifest["counts"]["accepted_unique"] == 747
    assert manifest["counts"]["combined_accepted_unique"] == 2045
    assert manifest["hashes"]["accepted_new_roots_hash"] == _hash(accepted_path)
    assert manifest["hashes"]["combined_accepted_dataset_hash"] == _hash(combined_path)
    assert len({row["design_hash"] for row in accepted}) == len(accepted)

    expected = {
        "low_mass_low_to_mid": (5, 13, 305),
        "low_mass_mid_to_high": (7, 13, 442),
    }
    for segment, (qualified, combined_roots, row_count) in expected.items():
        values = manifest["counts"]["by_segment"][segment]
        assert values["qualified_novel_roots"] == qualified
        assert values["combined_roots"] == combined_roots
        assert values["accepted_rows"] == row_count
        old_roots = {
            row["root_seed_id"] for row in base if row["assigned_segment"] == segment
        }
        new_roots = {
            row["root_seed_id"]
            for row in accepted
            if row["assigned_segment"] == segment
        }
        assert old_roots.isdisjoint(new_roots)
