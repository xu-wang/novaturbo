from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.ai.geometry19_performance_gated_physics import acquire_shard, data_gates, verify_inputs
from src.ai.geometry19_performance_gated_protocol import canonical, ProtocolError
from src.ai import geometry19_performance_gated_physics as acquisition

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/experiments/geometry19_performance_gated_g0c_physics_v1.yaml"
CONTEXT = SimpleNamespace(context_hash="ctx", physics_runtime=SimpleNamespace(physics_bundle_hash="bundle", model_version="v"),
                          target_domain=SimpleNamespace(source_hash="domain"))


def result(row, context):
    return {**deepcopy(row), "solver": {"converged": True},
            "performance": {"thrust_N": 1.0, "tsfc_kg_N_s": 1.0, "mass_kg": 1.0},
            "constraint_assessment": {"numerically_valid": True, "physics_domain_valid": True},
            "target_domain_assessment": {"inside": True}, "training_candidate": True,
            "pipeline_context_hash": "ctx", "physics_bundle_hash": "bundle", "physics_model_version": "v", "target_domain_hash": "domain"}


@pytest.fixture
def batch(tmp_path):
    (tmp_path / "shards").mkdir()
    rows = [{"design_hash": str(i), "split": split, "label_status": "PRELABEL_GEOMETRY_ONLY", "geometry19": {"x": i}}
            for i, split in enumerate(("train", "validation"))]
    shard = {"future_output": "shards/part-00000.jsonl", "shard_input_sha256": sha256(canonical(rows)).hexdigest()}
    return tmp_path, shard, rows


def test_resume_without_calls_and_validation_exclusion(batch):
    out, shard, rows = batch
    actual, seal = acquire_shard(out, shard, rows, CONTEXT, {}, result)
    assert actual[0]["training_candidate"] and actual[1]["validation_eligible"]
    assert not actual[1]["training_candidate"]
    def forbidden(*args):
        pytest.fail("Resume called physics")
    assert acquire_shard(out, shard, rows, CONTEXT, {}, forbidden) == (actual, seal)


def test_orphan_preserved_no_replay(batch):
    out, shard, rows = batch
    calls = []
    def crash(row, ctx):
        calls.append(row)
        raise RuntimeError("synthetic infrastructure failure")
    with pytest.raises(RuntimeError):
        acquire_shard(out, shard, rows, CONTEXT, {}, crash)
    before = {p: p.read_bytes() for p in (out / "shards").iterdir()}
    with pytest.raises(ProtocolError, match="Orphan"):
        acquire_shard(out, shard, rows, CONTEXT, {}, crash)
    assert len(calls) == 1 and before == {p: p.read_bytes() for p in (out / "shards").iterdir()}


@pytest.mark.parametrize("changed", ["identity", "output", "journal"])
def test_tamper_rejected(batch, changed):
    out, shard, rows = batch
    acquire_shard(out, shard, rows, CONTEXT, {}, result)
    identity = {"different": True} if changed == "identity" else {}
    if changed != "identity":
        path = out / ("shards/part-00000.jsonl" if changed == "output" else "shards/part-00000.calls.jsonl")
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ProtocolError, match="seal mismatch"):
        acquire_shard(out, shard, rows, CONTEXT, identity, result)


def test_coordinate_mutation_rejected(batch):
    out, shard, rows = batch
    def mutate(row, ctx):
        row["geometry19"]["x"] = -999
        return result(row, ctx)
    with pytest.raises(ProtocolError, match="Immutable"):
        acquire_shard(out, shard, rows, CONTEXT, {}, mutate)
    assert rows[0]["geometry19"]["x"] == 0


def test_physical_failure_retained_without_retry(batch):
    out, shard, rows = batch
    def fail(row, ctx):
        r = result(row, ctx)
        r["solver"] = {"converged": False, "failure_code": "nonconvergence"}
        return r
    actual, seal = acquire_shard(out, shard, rows, CONTEXT, {}, fail)
    assert len(actual) == seal["actual_calls"] == 2
    assert all(not r["physics_eligible"] for r in actual)


def test_full_population_gates_and_missing_population():
    gates = yaml.safe_load(CONFIG.read_text())["gates"]
    rows = [{"root_seed_id": str(root), "band_index": band, "training_eligible": i < 102,
             "validation_eligible": i >= 102, "solver": {"converged": True}}
            for root in range(33) for band in range(4) for i in range(128)]
    assert data_gates(rows, gates)["status"] == "G0C_DATA_GATE_PASS"
    for r in rows[:128]:
        r["training_eligible"] = r["validation_eligible"] = False
    summary = data_gates(rows, gates)
    assert summary["status"] == "G0C_DATA_GATE_FAIL"
    assert len(summary["failures"]) == 2
    with pytest.raises(ProtocolError, match="Incomplete"):
        data_gates(rows[:-1], gates)


def test_real_frozen_inputs_no_solver():
    c, rows, shards, context, identity = verify_inputs(ROOT, CONFIG)
    assert len(rows) == 16896 and len(shards) == 132
    assert context.context_hash == identity["pipeline_context_sha256"]


def test_orchestrator_seals_before_gates_and_completed_resume(tmp_path, monkeypatch):
    """Small synthetic run tests lifecycle; production population validation is not bypassed in the CLI."""
    rows = [{"design_hash": str(i), "split": split, "label_status": "PRELABEL_GEOMETRY_ONLY"}
            for i, split in enumerate(("train", "validation"))]
    shard = {"future_output": "shards/part-00000.jsonl", "shard_input_sha256": sha256(canonical(rows)).hexdigest(),
             "start_row_inclusive": 0, "stop_row_exclusive": 2}
    config = {"artifacts": {"future_physics_root": "synthetic"}, "gates": {}}
    monkeypatch.setattr(acquisition, "verify_inputs", lambda *a: (config, rows, [shard], CONTEXT, {}))
    calls = []
    def evaluate(row, ctx):
        calls.append(row)
        return result(row, ctx)
    original = acquisition.acquire_shard
    monkeypatch.setattr(acquisition, "acquire_shard", lambda *args: original(*args, evaluator=evaluate))
    def gate(evaluated, _):
        assert (tmp_path / "synthetic/Acquisition_Manifest.json").is_file()
        assert (tmp_path / "synthetic/Sealed_Evaluations.jsonl").is_file()
        return {"status": "SYNTHETIC_TEST_ONLY", "valid_train": 1, "valid_validation": 1,
                "failures": [], "next_stage": "test_only"}
    monkeypatch.setattr(acquisition, "data_gates", gate)
    first = acquisition.run(tmp_path, CONFIG, authorize=True)
    assert len(calls) == 2
    assert acquisition.run(tmp_path, CONFIG, resume=True) == first
    assert len(calls) == 2
    (tmp_path / "synthetic/Acquisition_Report.md").write_text("tampered")
    with pytest.raises(ProtocolError, match="Final artifact changed"):
        acquisition.run(tmp_path, CONFIG, resume=True)
    assert len(calls) == 2
    (tmp_path / "synthetic/shards/part-00000.seal.json").unlink()
    with pytest.raises(ProtocolError, match="Orphan"):
        acquisition.run(tmp_path, CONFIG, resume=True)
    assert len(calls) == 2


def test_requires_explicit_execution_mode():
    with pytest.raises(ProtocolError, match="Choose"):
        acquisition.run(ROOT, CONFIG)
    with pytest.raises(ProtocolError, match="Choose"):
        acquisition.run(ROOT, CONFIG, authorize=True, resume=True)
