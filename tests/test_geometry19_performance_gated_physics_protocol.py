"""Protocol-only locking, fixed sharding and split-safe eligibility."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.ai import geometry19_performance_gated_physics_protocol as protocol
from src.ai.geometry19_performance_gated_protocol import ProtocolError, read_rows, file_hash
from src.physics.runtime import PhysicsRuntime

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/experiments/geometry19_performance_gated_g0c_physics_v1.yaml"


@pytest.fixture
def config():
    return yaml.safe_load(CONFIG.read_text())


def test_lock_reproducible_without_evaluation_models_or_confirmation(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Physics/model execution forbidden in protocol lock")
    monkeypatch.setattr(PhysicsRuntime, "evaluate", forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    original = Path.read_text
    def guarded(path, *args, **kwargs):
        assert path.name not in {"confirmation.jsonl", "S2_Model_Manifest.json", "S3_v4_1_Sealed_V2_2_Evaluations.jsonl"}
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded)
    one = protocol.lock_protocol(ROOT, CONFIG, output_root=tmp_path / "one")
    two = protocol.lock_protocol(ROOT, CONFIG, output_root=tmp_path / "two")
    assert one == two and one["status"] == protocol.PASS
    assert one["shards"] == 132 and one["rows_per_shard"] == 128
    assert not one["physics_evaluations_executed"] and not one["confirmation_content_read"]
    assert not one["authorization"]["physics_acquisition_authorized_now"]
    for artifact in one["artifacts"].values():
        assert file_hash(tmp_path / "one" / artifact["path"]) == artifact["sha256"]
    shards = read_rows(tmp_path / "one/Shard_Registry.jsonl")
    assert sum(s["row_count"] for s in shards) == 16896
    assert all(s["split_counts"] == {"train": 102, "validation": 26} for s in shards)
    with pytest.raises(ProtocolError, match="overwrite"):
        protocol.lock_protocol(ROOT, CONFIG, output_root=tmp_path / "one")


@pytest.mark.parametrize("section,key,value", [
    ("authorization", "physics_acquisition_authorized_now", True),
    ("population", "shards", 33),
    ("gates", "each_root_minimum_valid_validation", 63),
    ("eligibility", "override_legacy_training_candidate", "keep_legacy"),
    ("execution", "adaptive_top_up", True),
    ("execution", "orphan_partial_shard", "overwrite"),
])
def test_contract_changes_rejected(config, section, key, value):
    config[section][key] = value
    with pytest.raises(ProtocolError):
        protocol.validate_protocol(config)


def test_context_mismatch_does_not_write(tmp_path, config):
    config["physics"]["context_sha256"] = "0" * 64
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ProtocolError, match="Physics context mismatch"):
        protocol.lock_protocol(ROOT, path, output_root=tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def valid_row(split):
    return {"split": split, "solver": {"converged": True},
        "constraint_assessment": {"numerically_valid": True, "physics_domain_valid": True},
        "target_domain_assessment": {"inside": True},
        "performance": {"thrust_N": 65., "tsfc_kg_N_s": 6e-5, "mass_kg": 1.8},
        "training_candidate": True}


def test_valid_validation_never_enters_training():
    flags = protocol.eligibility_flags(valid_row("validation"))
    assert flags == {"physics_eligible": True, "training_eligible": False,
        "validation_eligible": True, "training_candidate": False}
    assert protocol.eligibility_flags(valid_row("train"))["training_candidate"] is True


@pytest.mark.parametrize("failure", ["solver", "domain", "nonfinite", "missing"])
def test_ineligible_rows_cannot_be_training_candidates(failure):
    row = valid_row("train")
    if failure == "solver":
        row["solver"]["converged"] = False
    elif failure == "domain":
        row["target_domain_assessment"]["inside"] = False
    elif failure == "nonfinite":
        row["performance"]["mass_kg"] = float("nan")
    else:
        row["performance"] = None
    assert not any(protocol.eligibility_flags(row).values())


def test_changed_row_order_cannot_be_resharded():
    rows = read_rows(ROOT / "data/validation/v2_2_performance_gated_local_cfm_v1/training_coordinates/g0b_v1/Prelabel_Coordinates.jsonl")
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ProtocolError, match="order changed"):
        protocol.make_shards(rows)
