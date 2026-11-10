"""Tests for assigned-segment adaptive V2.2 data expansion."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import yaml

from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.physics.geometry19_v2_2_balanced_dataset import (
    BALANCED_DATASET_VERSION,
    _grouped_split,
    _segment_assessment,
    run_segment_balanced_expansion,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPO_ROOT
    / "config/physics/geometry19_v2_2_segment_balanced_expansion_v1.yaml"
)
EVIDENCE = REPO_ROOT / "data/validation/v2_2_segment_balanced_expansion_v1"


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_each_domain_segment_accepts_its_own_midpoint() -> None:
    context = build_pipeline_context(REPO_ROOT, pipeline_profile="v2_2_development")
    domain = context.target_domain
    assert domain is not None

    for segment_id, start_name, end_name in domain.segments:
        midpoint = {
            axis: (left * right) ** 0.5
            for axis, left, right in zip(
                ("thrust_N", "tsfc_kg_N_s", "mass_kg"),
                domain.nodes[start_name],
                domain.nodes[end_name],
            )
        }
        assessment = _segment_assessment(domain, midpoint, segment_id)
        assert assessment["inside"] is True
        assert assessment["normalized_distance"] < 1.0e-10


def test_grouped_split_never_leaks_root_seed_between_splits() -> None:
    rows = [
        {
            "assigned_segment": "segment-a",
            "root_seed_id": f"root-{root}",
            "design_hash": f"{root}-{index}",
        }
        for root in range(4)
        for index in range(root + 1)
    ]
    splits, assignments = _grouped_split(
        rows, {"train": 0.7, "validation": 0.15, "test": 0.15}
    )

    assert set(assignments.values()) == {"train", "validation", "test"}
    observed: dict[str, str] = {}
    for split_name, split_rows in splits.items():
        for row in split_rows:
            previous = observed.setdefault(row["root_seed_id"], split_name)
            assert previous == split_name


def test_small_balanced_campaign_is_resumable_and_segment_explicit(
    tmp_path: Path,
) -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    protocol["sampling"]["parent_pool_size"] = 3
    protocol["sampling"]["rounds"] = [
        {"samples_per_segment": 2, "local_radius_fraction": 0.002}
    ]
    protocol["coverage_gate"] = {
        "minimum_rows_per_segment": 0,
        "minimum_root_seeds_per_segment": 0,
    }
    local_protocol = tmp_path / "protocol.yaml"
    local_protocol.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    output = tmp_path / "balanced"

    first = run_segment_balanced_expansion(
        REPO_ROOT, local_protocol, output
    )
    second = run_segment_balanced_expansion(
        REPO_ROOT, local_protocol, output, resume=True
    )

    assert first == second
    assert first["dataset_version"] == BALANCED_DATASET_VERSION
    assert first["counts"]["evaluated"] == 10
    assert len(first["shards"]) == 5
    raw = [
        json.loads(line)
        for line in (output / "raw_dataset.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert set(row["assigned_segment"] for row in raw) == set(
        protocol["sampling"]["segments"]
    )


def test_committed_balanced_expansion_records_coverage_failure_without_leakage() -> None:
    manifest = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    accepted = [
        json.loads(line)
        for line in (EVIDENCE / "accepted_dataset.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert manifest["status"] == "COVERAGE_GATE_FAIL"
    assert manifest["coverage_gate_pass"] is False
    assert manifest["training_ready"] is False
    assert manifest["counts"]["evaluated"] == 3000
    assert manifest["counts"]["physics_domain_valid"] == 2764
    assert manifest["counts"]["accepted_unique"] == 253
    assert manifest["counts"]["by_segment"] == {
        "cross_branch_mid": 67,
        "high_mass_low_to_mid": 11,
        "high_mass_mid_to_high": 21,
        "low_mass_low_to_mid": 4,
        "low_mass_mid_to_high": 150,
    }
    assert len({row["design_hash"] for row in accepted}) == len(accepted)
    assert manifest["hashes"]["raw_dataset_hash"] == _hash(
        EVIDENCE / "raw_dataset.jsonl"
    )
    assert manifest["hashes"]["accepted_dataset_hash"] == _hash(
        EVIDENCE / "accepted_dataset.jsonl"
    )

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
    assert observed_designs == {row["design_hash"] for row in accepted}
