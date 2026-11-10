"""Tests for local-sensitivity recovery of V2.2 target-domain segments."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import yaml

from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.physics.geometry19_v2_2_directed_recovery import (
    DIRECTED_RECOVERY_VERSION,
    _select_models,
    run_directed_recovery,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPO_ROOT / "config/physics/geometry19_v2_2_directed_segment_recovery_v1.yaml"
)
EVIDENCE = REPO_ROOT / "data/validation/v2_2_directed_segment_recovery_v1"


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_local_sensitivity_centers_predict_lower_segment_distance() -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    context = build_pipeline_context(REPO_ROOT, pipeline_profile="v2_2_development")
    base_raw = REPO_ROOT / protocol["base_expansion"]["raw_dataset"]
    observations = [
        json.loads(line) for line in base_raw.read_text(encoding="utf-8").splitlines()
    ]
    variables = list(context.physics_runtime.design_schema["design_variables"])
    round_spec = protocol["sampling"]["rounds"][0]

    for segment_id in protocol["sampling"]["segments"]:
        models = _select_models(
            context.target_domain,
            variables,
            observations,
            segment_id,
            protocol["sampling"],
            round_spec,
        )
        assert len(models) == protocol["sampling"]["root_model_count"]
        assert all(model["jacobian_rank"] == 3 for model in models)
        assert all(
            model["predicted_distance"] < model["anchor_distance"]
            for model in models
        )


def test_small_directed_recovery_is_hash_resumable(tmp_path: Path) -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    protocol["sampling"]["segments"] = ["low_mass_low_to_mid"]
    protocol["sampling"]["root_model_count"] = 3
    protocol["sampling"]["rounds"] = [
        {
            "samples_per_segment": 3,
            "trust_radius_fraction": 0.02,
            "perturbation_radius_fraction": 0.001,
            "step_gain": 0.8,
        }
    ]
    protocol["coverage_gate"] = {
        "minimum_rows_per_segment": 0,
        "minimum_root_seeds_per_segment": 0,
    }
    local_protocol = tmp_path / "protocol.yaml"
    local_protocol.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    output = tmp_path / "directed"

    first = run_directed_recovery(REPO_ROOT, local_protocol, output)
    second = run_directed_recovery(
        REPO_ROOT, local_protocol, output, resume=True
    )

    assert first == second
    assert first["dataset_version"] == DIRECTED_RECOVERY_VERSION
    assert first["counts"]["recovery_evaluated"] == 3
    assert len(first["shards"]) == 1


def test_committed_directed_recovery_passes_coverage_and_has_no_split_leakage() -> None:
    manifest = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    combined_path = EVIDENCE / "combined_accepted_dataset.jsonl"
    combined = [
        json.loads(line)
        for line in combined_path.read_text(encoding="utf-8").splitlines()
    ]

    assert manifest["status"] == "COVERAGE_GATE_PASS"
    assert manifest["coverage_gate_pass"] is True
    assert manifest["training_ready"] is True
    assert manifest["formal_use_allowed"] is False
    assert manifest["counts"] == {
        "recovery_evaluated": 1920,
        "recovery_physics_domain_valid": 1367,
        "recovery_training_candidates": 1045,
        "combined_accepted_unique": 1298,
        "by_segment": {
            "cross_branch_mid": 325,
            "high_mass_low_to_mid": 261,
            "high_mass_mid_to_high": 206,
            "low_mass_low_to_mid": 356,
            "low_mass_mid_to_high": 150,
        },
        "root_seeds_by_segment": {
            "cross_branch_mid": 7,
            "high_mass_low_to_mid": 5,
            "high_mass_mid_to_high": 6,
            "low_mass_low_to_mid": 8,
            "low_mass_mid_to_high": 6,
        },
        "splits": {"test": 237, "train": 797, "validation": 264},
    }
    assert manifest["hashes"]["combined_accepted_dataset_hash"] == _hash(
        combined_path
    )
    assert len({row["design_hash"] for row in combined}) == len(combined)

    observed_designs: set[str] = set()
    observed_groups: dict[tuple[str, str], str] = {}
    for split_name in ("train", "validation", "test"):
        split_path = EVIDENCE / "splits" / f"{split_name}.jsonl"
        assert manifest["hashes"][f"{split_name}_split_hash"] == _hash(split_path)
        for line in split_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            assert row["design_hash"] not in observed_designs
            observed_designs.add(row["design_hash"])
            key = (row["assigned_segment"], row["root_seed_id"])
            assert observed_groups.setdefault(key, split_name) == split_name
    assert observed_designs == {row["design_hash"] for row in combined}
