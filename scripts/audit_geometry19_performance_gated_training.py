#!/usr/bin/env python3
"""Audit completed frozen G1/G2 training; never load models or evaluate physics."""
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ai.geometry19_performance_gated_training import verified_data, verify_implementation, Selection
from src.ai.geometry19_performance_gated_protocol import file_hash, require, within
from src.ai.geometry19_performance_gated_physics import write_json


def audit(root):
    config, _, protocol, decoders, _, _ = verified_data(root)
    implementation_hash = verify_implementation(root, config)
    out = within(root, config["artifacts"]["training_output"])
    require(not (out/"Failure.json").exists(), "Failed run exists")
    summary = json.loads((out/"Training_Summary.json").read_text())
    manifest = json.loads((out/"Runtime_Manifest.json").read_text())
    require(manifest["implementation_lock_sha256"] == implementation_hash and
            manifest["preparation_lock"] == config["preparation_lock"], "Runtime provenance mismatch")
    require(manifest["environment"]["visible_devices"] == "6" and
            manifest["environment"]["logical_device"] == "cuda:0", "Wrong runtime GPU")
    require(summary["status"] == "G1_G2_TRAINING_COMPLETE" and summary["run_count"] == 9 and
            len(summary["runs"]) == 9 and not summary["generation_executed"] and
            summary["physics_evaluations"] == 0, "Incomplete training or wrong scope")
    expected = {(seed, arm) for seed in [42, 43, 44] for arm in ["gate", "local_cfm", "pooled_cfm"]}
    require({(r["seed"], r["arm"]) for r in summary["runs"]} == expected, "Missing/duplicate training arm")
    models, gates = [], []
    for result in summary["runs"]:
        seed, arm = result["seed"], result["arm"]
        folder = out/f"seed{seed}"/arm
        require(json.loads((folder/"Summary.json").read_text()) == result, "Run summary mismatch")
        settings = protocol["parent_models"]["gate" if arm == "gate" else "local_cfm"]
        identity = json.loads((folder/"Identity.json").read_text())
        require(identity == {"arm": arm, "seed": seed, "implementation_lock_sha256": implementation_hash,
                             "preparation_lock_sha256": config["preparation_lock"]["sha256"], "settings": settings}, "Run identity mismatch")
        files = sorted(folder.glob("epoch_*.json"))
        require(len(files) == result["epochs"] and len(files) <= settings["maximum_epochs"], "Epoch count mismatch")
        selection = Selection()
        for epoch, path in enumerate(files, 1):
            record = json.loads(path.read_text())
            require(record["epoch"] == epoch and record["steps"] == (48 if arm == "gate" else 32) and
                    math.isfinite(record["training_loss"]), "Bad epoch record")
            improved, stop = selection.update(epoch, record["validation_objective"], settings)
            require(record["best_epoch"] == selection.best_epoch and record["stale_epochs"] == selection.stale,
                    "Checkpoint/early-stop selection mismatch")
            require(not stop or epoch == len(files), "Training continued past frozen early-stop rule")
            require((folder/f"best_epoch_{epoch:04d}.pt").exists() == improved, "Best-checkpoint ledger mismatch")
        require(stop or len(files) == settings["maximum_epochs"], "Premature termination")
        require(result["best_epoch"] == selection.best_epoch and result["best_validation_objective"] == selection.best,
                "Wrong selected checkpoint")
        require(abs(result["reload_validation_objective"]-selection.best) <= 1e-7*max(1., abs(selection.best)), "Reload discrepancy")
        checkpoint = within(folder, result["checkpoint"])
        require(checkpoint.name == f"best_epoch_{selection.best_epoch:04d}.pt" and
                file_hash(checkpoint) == result["checkpoint_sha256"], "Checkpoint hash mismatch")
        models.append({**result, "checkpoint_path": str(checkpoint.relative_to(root))})
        if arm == "gate":
            d = result["gate_diagnostics"]
            require(len(d["per_root_top2"]) == 33, "Missing root diagnostics")
            expected_pass = d["macro_top1"] >= .8 and d["macro_top2"] >= .95 and min(d["per_root_top2"]) >= .8
            require(d["diagnostic_pass"] == expected_pass, "Diagnostic status mismatch")
            gates.append({"seed": seed, **d, "below_top2_0_8_roots": [decoders["root_order"][i]
                          for i, value in enumerate(d["per_root_top2"]) if value < .8]})
    aggregates = {}
    for arm in ["gate", "local_cfm", "pooled_cfm"]:
        values = [r["best_validation_objective"] for r in models if r["arm"] == arm]
        aggregates[arm] = {"mean": statistics.mean(values), "sample_std_ddof1": statistics.stdev(values)}
    result = {"status": "G1_G2_TRAINING_AUDIT_PASS", "auditor_sha256": file_hash(Path(__file__)),
              "implementation_lock_sha256": implementation_hash, "training_summary_sha256": file_hash(out/"Training_Summary.json"),
              "runtime_manifest_sha256": file_hash(out/"Runtime_Manifest.json"), "runs": models, "gate_diagnostics": gates,
              "validation_objectives_by_arm": aggregates, "objectives_not_comparable_between_arms": True,
              "all_nine_checkpoints_verified": True, "all_epoch_selection_rules_verified": True,
              "model_arrays_loaded_by_audit": False, "generation_executed": False, "physics_evaluations": 0,
              "generation_quality_or_closure_pass_claimed": False}
    write_json(out/"Training_Audit.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(audit(ROOT), indent=2))
