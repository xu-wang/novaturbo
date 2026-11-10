"""Tests for the restricted-domain simplified-V2.2 dataset protocol."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import yaml

from src.physics.geometry19_v2_2_dataset import (
    DATASET_VERSION,
    _sample_plan,
    run_restricted_pilot,
)
from src.ai.geometry19_pipeline_context import build_pipeline_context


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPO_ROOT / "config/physics/geometry19_v2_2_restricted_pilot_v1.yaml"
SEEDS = REPO_ROOT / "data/validation/geometry19_v2_2_audited_seed_pack_v1.jsonl"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_pilot_protocol_is_restricted_nonformal_and_reproducible() -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    context = build_pipeline_context(REPO_ROOT, pipeline_profile="v2_2_development")
    seeds = _rows(SEEDS)

    first = _sample_plan(context.physics_runtime.design_schema, seeds, protocol["sampling"])
    second = _sample_plan(context.physics_runtime.design_schema, seeds, protocol["sampling"])

    assert protocol["dataset_version"] == DATASET_VERSION
    assert protocol["formal_use_allowed"] is False
    assert protocol["training_ready"] is False
    assert len(first) == 1024
    assert first == second
    assert len({row["design_hash"] for row in first}) > 900
    assert all(isinstance(row["geometry19"]["compressor_blade_count"], int) for row in first)


def test_small_pilot_writes_hash_checked_resumable_artifacts(tmp_path: Path) -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    protocol["sampling"]["sample_count"] = 8
    protocol["sampling"]["shard_size"] = 4
    protocol["pilot_gate"] = {
        "minimum_physics_domain_valid": 0,
        "minimum_training_candidates": 0,
    }
    local_protocol = tmp_path / "protocol.yaml"
    local_protocol.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    output = tmp_path / "pilot"

    first = run_restricted_pilot(REPO_ROOT, local_protocol, output)
    second = run_restricted_pilot(REPO_ROOT, local_protocol, output, resume=True)

    assert first == second
    assert first["status"] == "PILOT_GATE_PASS"
    assert first["counts"]["evaluated"] == 8
    assert len(first["shards"]) == 2
    assert (output / "raw_dataset.jsonl").is_file()
    assert (output / "training_candidates.jsonl").is_file()


def test_committed_pilot_evidence_matches_its_hash_chain() -> None:
    root = REPO_ROOT / "data/validation/v2_2_restricted_pilot_v1"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "PILOT_GATE_PASS"
    assert manifest["formal_use_allowed"] is False
    assert manifest["training_ready"] is False
    assert manifest["counts"] == {
        "engineering_feasible": 0,
        "evaluated": 1024,
        "inside_target_domain": 169,
        "numerically_valid": 995,
        "physics_domain_valid": 967,
        "solver_converged": 995,
        "training_candidates": 164,
    }
    assert manifest["hashes"]["seed_pack_hash"] == _hash(SEEDS)
    assert manifest["hashes"]["sample_plan_hash"] == _hash(root / "sample_plan.jsonl")
    assert manifest["hashes"]["raw_dataset_hash"] == _hash(root / "raw_dataset.jsonl")
    assert manifest["hashes"]["training_candidates_hash"] == _hash(
        root / "training_candidates.jsonl"
    )
    assert len(manifest["shards"]) == 8
    for record in manifest["shards"]:
        shard = root / "shards" / f"part-{record['shard_index']:05d}.jsonl"
        assert record["output_hash"] == _hash(shard)
