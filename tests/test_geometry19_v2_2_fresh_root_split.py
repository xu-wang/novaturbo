"""Tests for the fresh globally root-disjoint V2.2 split."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from src.physics.geometry19_v2_2_dataset import Geometry19V22DatasetError
from src.physics.geometry19_v2_2_fresh_root_split import freeze_fresh_root_split


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPO_ROOT / "config/physics/geometry19_v2_2_fresh_low_mass_confirmation_split_v1.yaml"
EVIDENCE = REPO_ROOT / "data/validation/v2_2_fresh_low_mass_confirmation_split_v1"


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_fresh_split_is_globally_root_disjoint_and_resumable(tmp_path: Path) -> None:
    output = tmp_path / "split"
    first = freeze_fresh_root_split(REPO_ROOT, PROTOCOL, output)
    second = freeze_fresh_root_split(REPO_ROOT, PROTOCOL, output, resume=True)
    assert first == second
    assert first["status"] == "FRESH_ROOT_SPLIT_PASS"
    assert first["counts"]["train"]["rows"] == 1645
    assert first["counts"]["validation"]["rows"] == 244
    assert first["counts"]["confirmation"]["rows"] == 156

    roots: dict[str, set[str]] = {}
    for role in ("train", "validation", "confirmation"):
        rows = [
            json.loads(line)
            for line in (output / "splits" / f"{role}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        roots[role] = {row["root_seed_id"] for row in rows}
    assert roots["train"].isdisjoint(roots["validation"] | roots["confirmation"])
    assert roots["validation"].isdisjoint(roots["confirmation"])


def test_confirmation_root_freshness_is_enforced(tmp_path: Path) -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    protocol["roles"]["confirmation_roots_by_segment"]["low_mass_low_to_mid"] = (
        "51eb4807184fe000b651106385ea7bfe3c44dcf0a2c01447cbd3d3646690bead"
    )
    local = tmp_path / "protocol.yaml"
    local.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    with pytest.raises(Geometry19V22DatasetError, match="not globally fresh"):
        freeze_fresh_root_split(REPO_ROOT, local, tmp_path / "out")


def test_frozen_fresh_split_matches_manifest() -> None:
    manifest = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FRESH_ROOT_SPLIT_PASS"
    assert manifest["training_ready"] is True
    assert manifest["full_domain_confirmation_ready"] is False
    assert {role: values["rows"] for role, values in manifest["counts"].items()} == {
        "train": 1645,
        "validation": 244,
        "confirmation": 156,
    }
    assert {role: values["global_roots"] for role, values in manifest["counts"].items()} == {
        "train": 20,
        "validation": 2,
        "confirmation": 2,
    }
    for role in ("train", "validation", "confirmation"):
        assert manifest["hashes"][f"{role}_split_hash"] == _hash(
            EVIDENCE / "splits" / f"{role}.jsonl"
        )
