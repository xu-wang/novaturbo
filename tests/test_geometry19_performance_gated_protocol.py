"""G0a evidence integrity, source isolation and deterministic target selection."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from src.ai import geometry19_performance_gated_protocol as protocol

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/experiments/geometry19_v2_2_performance_gated_local_cfm_v1.yaml"


@pytest.fixture
def config():
    return yaml.safe_load(CONFIG.read_text())


@pytest.fixture
def evidence(config):
    rows = {k: protocol.read_rows(ROOT / config["inputs"][k]["path"]) for k in
            ("anchor_registry", "stable_root_registry", "development_dataset", "physics_ledger")}
    variables = yaml.safe_load((ROOT / config["inputs"]["schema"]["path"]).read_text())["design_variables"]
    return [rows[k] for k in ("anchor_registry", "stable_root_registry", "development_dataset", "physics_ledger")] + [variables]


def test_deterministic_lock_runtime_minimization_and_no_overwrite(tmp_path, config, monkeypatch):
    confirmation = (ROOT / config["inputs"]["sealed_confirmation"]["path"]).resolve()
    original = Path.read_text

    def guarded(path, *args, **kwargs):
        assert path.resolve() != confirmation, "Confirmation parsed"
        assert path.suffix != ".npz", "Model parsed"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    out = tmp_path / "one"
    first = protocol.lock_protocol(ROOT, CONFIG, output_root=out)
    second = protocol.lock_protocol(ROOT, CONFIG, output_root=tmp_path / "two")
    assert first == second
    assert first["status"] == protocol.PASS
    assert first["targets"] == 99 and first["roots"] == 33
    assert first["target_class_counts"] == {"center": 33, "typical": 33, "outer": 33}
    runtime = protocol.read_rows(out / "targets/Runtime_Targets.jsonl")
    audit = protocol.read_rows(out / "targets/Target_Registry.jsonl")
    assert [r["target_id"] for r in runtime] == [r["target_id"] for r in audit]
    assert all(set(r) == {"target_id", "performance", "units"} for r in runtime)
    assert all(set(r["performance"]) == set(protocol.OUTPUTS) for r in runtime)
    assert all(len(r["target_id"]) == 64 for r in runtime)
    assert all("geometry19" not in r for r in audit)
    assert all(r["actual_local_distance"] <= .2 + 1e-12 for r in audit)
    assert all(first["future_artifacts_absent"].values())
    assert not first["model_arrays_loaded"] and not first["confirmation_content_read"]
    before = protocol.file_hash(out / "audits/G0a_Protocol_Target_Lock.json")
    with pytest.raises(protocol.ProtocolError, match="overwrite"):
        protocol.lock_protocol(ROOT, CONFIG, output_root=out)
    assert before == protocol.file_hash(out / "audits/G0a_Protocol_Target_Lock.json")


@pytest.mark.parametrize("section,key,value", [
    ("authorization", "training_authorized", True),
    ("models", "condition_dim", 4),
    ("models", "root_embedding", True),
    ("training_data_plan", "v4_1_training_allowed", True),
    ("targets", "source_root_in_runtime_view", True),
    ("evaluation", "target_root_for_scoring", True),
])
def test_rejects_authority_and_leakage_changes(config, section, key, value):
    config[section][key] = value
    with pytest.raises(protocol.ProtocolError):
        protocol.validate_protocol(config)


def test_input_hash_mismatch_leaves_no_outputs(tmp_path, config):
    config["inputs"]["anchor_registry"]["sha256"] = "0" * 64
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / "absent"
    with pytest.raises(protocol.ProtocolError, match="Input hash mismatch"):
        protocol.lock_protocol(ROOT, path, output_root=output)
    assert not output.exists()


def test_selection_does_not_rank_by_performance(config, evidence):
    first, _ = protocol.select_targets(config, *evidence)
    changed = deepcopy(evidence)
    for row in changed[3]:
        row["performance"] = {k: 1.0 for k in protocol.OUTPUTS}
    second, _ = protocol.select_targets(config, *changed)
    assert [r["source_design_hash"] for r in first] == [r["source_design_hash"] for r in second]


def test_missing_shell_cannot_be_replaced_by_outside_support(config, evidence, monkeypatch):
    original = protocol.geometry_distance

    def outside(row, *args):
        return 0.3 if row.get("radius_fraction") == .2 else original(row, *args)

    monkeypatch.setattr(protocol, "geometry_distance", outside)
    with pytest.raises(protocol.ProtocolError, match="No legal outer"):
        protocol.select_targets(config, *evidence)


def test_center_cannot_be_replaced(config, evidence):
    evidence[0][0]["medoid_design_hash"] = "0" * 64
    with pytest.raises(protocol.ProtocolError, match="center source"):
        protocol.select_targets(config, *evidence)


def test_physics_context_tamper_rejected(config, evidence):
    row = next(r for r in evidence[3] if r["evaluation_eligible"])
    row["pipeline_context_hash"] = "0" * 64
    with pytest.raises(protocol.ProtocolError, match="context changed"):
        protocol.select_targets(config, *evidence)


def test_geometry_tamper_rejected(config, evidence):
    center = next(r for r in evidence[2] if r["design_hash"] == evidence[0][0]["medoid_design_hash"])
    center["geometry19"]["shaft_speed_rpm"] += 1
    with pytest.raises(protocol.ProtocolError, match="Center identity changed"):
        protocol.select_targets(config, *evidence)


def test_nonfinite_performance_rejected(config, evidence):
    row = next(r for r in evidence[3] if r["evaluation_eligible"])
    row["performance"]["mass_kg"] = float("nan")
    with pytest.raises(protocol.ProtocolError, match="Invalid target performance"):
        protocol.select_targets(config, *evidence)
