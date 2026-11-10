import json
from pathlib import Path

import pytest

from scripts import audit_geometry19_performance_gated_training as audit
from src.ai.geometry19_performance_gated_protocol import ProtocolError, file_hash


@pytest.fixture
def completed(tmp_path, monkeypatch):
    output = tmp_path/"training"
    output.mkdir()
    def save(path, value): path.write_text(json.dumps(value))
    settings = {"maximum_epochs": 1, "minimum_epochs": 1, "patience": 2, "min_delta": 1e-5}
    config = {"artifacts": {"training_output": "training"}, "preparation_lock": {"sha256": "prep"}}
    protocol = {"parent_models": {"gate": settings, "local_cfm": settings}}
    monkeypatch.setattr(audit, "verified_data", lambda root: (config, None, protocol, {"root_order": list(range(33))}, None, None))
    monkeypatch.setattr(audit, "verify_implementation", lambda root, config: "implementation")
    runs = []
    for seed in [42, 43, 44]:
        for arm in ["gate", "local_cfm", "pooled_cfm"]:
            folder = output/f"seed{seed}"/arm
            folder.mkdir(parents=True)
            checkpoint = folder/"best_epoch_0001.pt"
            checkpoint.write_bytes(b"synthetic-not-a-real-model")
            identity = {"arm": arm, "seed": seed, "implementation_lock_sha256": "implementation",
                        "preparation_lock_sha256": "prep", "settings": settings}
            save(folder/"Identity.json", identity)
            save(folder/"epoch_0001.json", {"epoch": 1, "steps": 48 if arm == "gate" else 32,
                 "training_loss": 1., "validation_objective": 1., "best_epoch": 1, "stale_epochs": 0})
            summary = {"seed": seed, "arm": arm, "epochs": 1, "best_epoch": 1, "best_validation_objective": 1.,
                       "reload_validation_objective": 1., "checkpoint": checkpoint.name, "checkpoint_sha256": file_hash(checkpoint)}
            if arm == "gate":
                summary["gate_diagnostics"] = {"macro_top1": .9, "macro_top2": .96,
                                               "per_root_top2": [.96]*33, "diagnostic_pass": True}
            save(folder/"Summary.json", summary)
            runs.append(summary)
    save(output/"Runtime_Manifest.json", {"implementation_lock_sha256": "implementation", "preparation_lock": config["preparation_lock"],
         "environment": {"visible_devices": "6", "logical_device": "cuda:0"}})
    save(output/"Training_Summary.json", {"status": "G1_G2_TRAINING_COMPLETE", "run_count": 9, "runs": runs,
         "generation_executed": False, "physics_evaluations": 0})
    return tmp_path, output


def test_complete_ledger_passes_without_loading_models(completed, monkeypatch):
    monkeypatch.setattr("torch.load", lambda *a, **k: pytest.fail("Read-only audit must not load models"))
    result = audit.audit(completed[0])
    assert result["status"] == "G1_G2_TRAINING_AUDIT_PASS"
    assert result["all_nine_checkpoints_verified"] and len(result["runs"]) == 9


@pytest.mark.parametrize("tamper", ["checkpoint", "epoch_steps", "best_epoch", "missing_seed", "failure"])
def test_corrupt_results_rejected(completed, tamper):
    root, output = completed
    gate = output/"seed42"/"gate"
    if tamper == "checkpoint":
        (gate/"best_epoch_0001.pt").write_bytes(b"changed")
    elif tamper == "failure":
        (output/"Failure.json").write_text("{}")
    else:
        path = output/"Training_Summary.json" if tamper == "missing_seed" else gate/"epoch_0001.json"
        value = json.loads(path.read_text())
        if tamper == "missing_seed": value["runs"].pop()
        elif tamper == "epoch_steps": value["steps"] = 1
        else: value["best_epoch"] = 2
        path.write_text(json.dumps(value))
    with pytest.raises(ProtocolError): audit.audit(root)
    assert not (output/"Training_Audit.json").exists()
