"""Freeze, open and audit seven registered holdout center targets via the public CLI."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from src.ai.geometry19_pipeline_artifact import verify_completed_artifact
from src.ai.geometry22_pipeline_v3_routed import CONFIG as PIPELINE_CONFIG, code_hashes, read, sha, source
from src.physics.constraint_profiles import ConstraintProfile, assess_constraint_profile
from src.physics.geometry22_design import Geometry22Design
from src.physics.v3_development_runtime import build_v3_development_physics_runtime

CONFIG = "config/experiments/geometry22_v3_final_holdout_v1.yaml"
CODE = "src/ai/v3_final_holdout.py"
NAMES = ("thrust_N", "tsfc_kg_N_s", "mass_kg")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def rows(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def write_new(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def coordinates(registry: list[dict], training_centers: list[dict]) -> list[dict]:
    require(Counter(t["split"] for t in registry) == {"development": 32, "validation": 6, "holdout": 7}
            and len({t["root_id"] for t in registry}) == 45
            and len({t["design_id"] for t in registry}) == 45, "Holdout registry split leakage or population changed")
    held = [t for t in registry if t["split"] == "holdout"]
    require(not ({t["design_id"] for t in held} & {t["center_id"] for t in training_centers}), "Holdout center occurs in training support")
    for t in held:
        require(Geometry22Design.from_mapping(t["design"]).design_id() == t["design_id"], "Holdout registered design identity changed")
    return [{"target_source_design_id": t["design_id"], "source_root_id": t["root_id"],
             "design": t["design"], "split": "holdout"} for t in held]


def config_at(root: Path) -> dict:
    cfg = yaml.safe_load((root / CONFIG).read_text())
    require(cfg["version"] == "geometry22_v3_final_holdout_v1"
            and cfg["target_rule"] == "all_seven_registered_holdout_center_points_no_filter"
            and cfg["total_target_denominator"] == 7 and cfg["pipeline_profile"] == "v3_routed_development"
            and cfg["physical_gpu_index"] == 6 and cfg["model_refitting_allowed"] is False
            and cfg["formal_use_allowed"] is False and cfg["deployment_allowed"] is False
            and cfg["pipeline_config_path"] == PIPELINE_CONFIG, "Final holdout protocol changed scope")
    require(sha(root / cfg["registry_path"]) == cfg["registry_sha256"], "Holdout registry changed")
    require(sha(root / PIPELINE_CONFIG) == cfg["pipeline_config_sha256"], "Final pipeline protocol changed")
    source(root)
    return cfg


def identity(root: Path, cfg: dict) -> dict:
    pipeline, fit, local, support, _, checkpoint = source(root)
    return {"config_sha256": sha(root / CONFIG), "registry_sha256": cfg["registry_sha256"],
            "code_sha256": {**code_hashes(root), CODE: sha(root / CODE),
                            "src/ai/geometry19_pipeline_artifact.py": sha(root / "src/ai/geometry19_pipeline_artifact.py")},
            "pipeline_config_sha256": cfg["pipeline_config_sha256"], "pipeline_protocol": pipeline,
            "gate_checkpoint_sha256": checkpoint["sha256"], "selected_gate_model_id": fit["selected_neural_model_id"],
            "selected_local_model_id": local["selected_model_id"],
            "target_coordinates": coordinates(rows(root / cfg["registry_path"]), support),
            "runtime_contract": build_v3_development_physics_runtime(root).contract_mapping()}


def freeze(root: Path, cfg: dict) -> dict:
    output = root / cfg["output_root"]
    require(not output.exists(), "Final holdout output already exists; preserve and stop")
    frozen = identity(root, cfg)
    output.mkdir(parents=True)
    lock = {"status": "final_holdout_protocol_frozen_before_labels", "identity": frozen,
            "target_labels_opened": False, "model_refitted": False, "formal_use_allowed": False, "deployment_allowed": False}
    write_new(output / "protocol_lock.json", lock)
    return lock


def verify_lock(root: Path, cfg: dict) -> dict:
    lock = read(root / cfg["output_root"] / "protocol_lock.json")
    require(lock["status"] == "final_holdout_protocol_frozen_before_labels" and lock["identity"] == identity(root, cfg)
            and lock["target_labels_opened"] is False and lock["model_refitted"] is False,
            "Final holdout frozen model, code, coordinates or runtime changed")
    return lock


def label(root: Path, cfg: dict) -> dict:
    lock = verify_lock(root, cfg)
    output = root / cfg["output_root"]
    require(not (output / "target_labels.json").exists() and not (output / "target_label_manifest.json").exists(), "Holdout labels already opened")
    runtime = build_v3_development_physics_runtime(root)
    profile = ConstraintProfile.from_yaml(root / lock["identity"]["pipeline_protocol"]["sources"]["diagnostic_profile_path"])
    labeled = []
    for t in lock["identity"]["target_coordinates"]:
        entry = {**t, "runtime_contract": runtime.contract_mapping(), "training_allowed": False,
                 "formal_use_allowed": False, "deployment_allowed": False}
        try:
            result = runtime.evaluate(Geometry22Design.from_mapping(t["design"]))
            require(result.design_id == t["target_source_design_id"], "Holdout label backend design identity changed")
            engine = result.engine_result
            perf = {n: float(engine.performance[n]) for n in NAMES} if engine.solver.converged else None
            usable = bool(perf is not None and all(math.isfinite(v) and v > 0 for v in perf.values()))
            diagnostic = assess_constraint_profile(engine, profile, conservation_tolerances=runtime.base_runtime.conservation_tolerances).as_mapping()
            entry.update(solver_converged=bool(engine.solver.converged),
                         failure_code=engine.solver.failure_code.value if engine.solver.failure_code else None,
                         target_performance=perf if usable else None, usable_target=usable,
                         diagnostic_domain_valid=bool(diagnostic["numerically_valid"] and diagnostic["physics_domain_valid"]),
                         engine_feasible=bool(engine.feasible), label_execution_error=None)
        except Exception as exc:
            entry.update(solver_converged=False, failure_code="label_execution_error", target_performance=None,
                         usable_target=False, diagnostic_domain_valid=False, engine_feasible=False,
                         label_execution_error={"error_type": type(exc).__name__, "error": str(exc)})
        labeled.append(entry)
    write_new(output / "target_labels.json", labeled)
    manifest = {"status": "seven_holdout_targets_opened_once", "protocol_lock_sha256": sha(output / "protocol_lock.json"),
                "target_labels_sha256": sha(output / "target_labels.json"), "total_targets": 7,
                "usable_targets": sum(t["usable_target"] for t in labeled),
                "model_refitted": False, "training_allowed": False, "formal_use_allowed": False, "deployment_allowed": False}
    write_new(output / "target_label_manifest.json", manifest)
    return manifest


def verify_labels(root: Path, cfg: dict) -> list[dict]:
    lock = verify_lock(root, cfg)
    output = root / cfg["output_root"]
    manifest = read(output / "target_label_manifest.json")
    require(manifest["protocol_lock_sha256"] == sha(output / "protocol_lock.json")
            and manifest["target_labels_sha256"] == sha(output / "target_labels.json"), "Holdout labels changed")
    labeled = read(output / "target_labels.json")
    require(len(labeled) == 7, "Holdout label denominator changed")
    for t, c in zip(labeled, lock["identity"]["target_coordinates"]):
        require(all(t[k] == c[k] for k in c) and t["training_allowed"] is False
                and t["runtime_contract"] == lock["identity"]["runtime_contract"], "Holdout label identity or split changed")
        perf = t["target_performance"]
        require(t["usable_target"] == bool(t["solver_converged"] and perf is not None
                and set(perf) == set(NAMES) and all(math.isfinite(v) and v > 0 for v in perf.values())), "Holdout target usability changed")
    require(manifest["total_targets"] == 7 and manifest["usable_targets"] == sum(t["usable_target"] for t in labeled), "Holdout label counts changed")
    return labeled


def case_report(root: Path, cfg: dict, t: dict) -> dict:
    base = {"source_design_id": t["target_source_design_id"], "label_usable": t["usable_target"]}
    if not t["usable_target"]:
        return {**base, "status": "TARGET_LABEL_FAILED", "software_hit_5pct": False,
                "domain_valid_software_hit_5pct": False, "delivered_hit": False, "physics_evaluations": 0}
    case = root / cfg["output_root"] / "cases" / t["target_source_design_id"]
    outcome = read(case / "execution_record.json")
    require(outcome["target_yaml_sha256"] == sha(case / "target.yaml"), "Holdout CLI target document changed")
    require(yaml.safe_load((case / "target.yaml").read_text()) == t["target_performance"], "Holdout performance passed to CLI changed")
    for name, digest in outcome["log_sha256"].items():
        require(sha(case / name) == digest, "Holdout execution log changed")
    if outcome["returncode"]:
        run = case / "run"
        require(outcome["failure_files"] == {str(p.relative_to(run)): sha(p) for p in sorted(run.rglob("*"))
                if p.is_file() and p.name != ".runner.lock"}, "Holdout failure artifacts changed")
        return {**base, "status": "CLI_EXECUTION_FAILED", "software_hit_5pct": False,
                "domain_valid_software_hit_5pct": False, "delivered_hit": False, "physics_evaluations": None,
                "failure_returncode": outcome["returncode"]}
    run = case / "run"
    checked = verify_completed_artifact(run, root)
    require(checked["integrity_pass"] and checked["current_code_identity_matches_lock"], "Holdout CLI artifact identity mismatch")
    require(outcome["complete_manifest_sha256"] == sha(run / "Complete_Manifest.json"), "Holdout complete manifest changed")
    summary = checked["summary"]
    evaluations = sum([read(run / stage / "evaluations.json") for stage in "CDE"], [])
    f = read(run / "F/summary.json")
    require(outcome["completed_resume_pass"] is True, "Holdout completed resume was not verified")
    return {**base, "status": "ABSTAINED" if summary["route_status"] == "abstained_no_target_support" else "EVALUATED",
            "software_hit_5pct": summary["software_hit_5pct"],
            "domain_valid_software_hit_5pct": any(r["software_hit_5pct"] and r["diagnostic_domain_valid"] for r in evaluations),
            "delivered_hit": summary["delivery_status"] == "PASS", "physics_evaluations": summary["physics_evaluation_count"],
            "solver_converged": sum(r["solver_converged"] for r in evaluations), "candidate_evaluations": len(evaluations),
            "diagnostic_domain_valid": sum(r["diagnostic_domain_valid"] for r in evaluations),
            "engine_feasible": sum(r["feasible"] for r in evaluations),
            "f_recompute_equal": f["independent_recompute"] in evaluations if f["independent_recompute"] is not None else None,
            "eligible_center_count": summary["eligible_center_count"], "f_rho": f["independent_recompute"]["rho"] if f["independent_recompute"] is not None else None,
            "complete_manifest_sha256": sha(run / "Complete_Manifest.json"), "pipeline_summary": summary}


def aggregate(cases: list[dict]) -> dict:
    require(len(cases) == 7 and len({c["source_design_id"] for c in cases}) == 7, "Final holdout denominator must remain seven")
    return {"total_target_denominator": 7, "status_counts": dict(sorted(Counter(c["status"] for c in cases).items())),
            "software_hit_targets_5pct": sum(c["software_hit_5pct"] for c in cases),
            "domain_valid_software_hit_targets_5pct": sum(c["domain_valid_software_hit_5pct"] for c in cases),
            "delivered_hit_targets": sum(c["delivered_hit"] for c in cases),
            "candidate_evaluations": sum(c.get("candidate_evaluations", 0) for c in cases),
            "solver_converged": sum(c.get("solver_converged", 0) for c in cases),
            "diagnostic_domain_valid": sum(c.get("diagnostic_domain_valid", 0) for c in cases),
            "engine_feasible": sum(c.get("engine_feasible", 0) for c in cases),
            "f_recomputations": sum(c.get("f_recompute_equal") is not None for c in cases),
            "f_recompute_equal": sum(c.get("f_recompute_equal") is True for c in cases),
            "sealed_cli_physics_calls": sum(c["physics_evaluations"] or 0 for c in cases),
            "target_label_calls": 7, "failed_cli_call_counts_known": not any(c["status"] == "CLI_EXECUTION_FAILED" for c in cases)}


def evaluate(root: Path, cfg: dict) -> dict:
    targets = verify_labels(root, cfg)
    output = root / cfg["output_root"]
    require(not (output / "complete_manifest.json").exists(), "Final holdout check already complete; use verify")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "6" and torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Final holdout CLI inference requires physical GPU 6")
    print({"physical_gpu_index": 6, "visible_gpu_count": 1, "device_name": torch.cuda.get_device_name(0)}, flush=True)
    for t in targets:
        if not t["usable_target"]:
            continue
        case = output / "cases" / t["target_source_design_id"]
        if (case / "execution_record.json").exists():
            case_report(root, cfg, t)
            continue
        case.mkdir(parents=True, exist_ok=True)
        path = case / "target.yaml"
        if path.exists():
            require(yaml.safe_load(path.read_text()) == t["target_performance"], "Existing holdout target changed")
        else:
            path.write_text(yaml.safe_dump(t["target_performance"]))
        cmd = [sys.executable, str(root / "scripts/run_wp19_23_pipeline.py"), "--model", "model V3",
               "--pipeline-profile", cfg["pipeline_profile"], "--target", str(path), "--output-root", str(case / "run")]
        # Recover an interrupted invocation under the pipeline's frozen stage policy.
        run = case / "run"
        result = subprocess.run(cmd + (["--resume"] if run.exists() else []), cwd=root, capture_output=True, text=True)
        (case / "stdout.txt").write_text(result.stdout)
        (case / "stderr.txt").write_text(result.stderr)
        record = {"returncode": result.returncode, "target_yaml_sha256": sha(path),
                  "log_sha256": {n: sha(case / n) for n in ("stdout.txt", "stderr.txt")}}
        if result.returncode == 0:
            resumed = subprocess.run(cmd + ["--resume"], cwd=root, capture_output=True, text=True)
            (case / "resume_stdout.txt").write_text(resumed.stdout)
            (case / "resume_stderr.txt").write_text(resumed.stderr)
            record["log_sha256"].update({n: sha(case / n) for n in ("resume_stdout.txt", "resume_stderr.txt")})
            record.update(completed_resume_pass=resumed.returncode == 0, complete_manifest_sha256=sha(run / "Complete_Manifest.json"))
        else:
            record["failure_files"] = {str(p.relative_to(run)): sha(p) for p in sorted(run.rglob("*")) if p.is_file() and p.name != ".runner.lock"}
        write_new(case / "execution_record.json", record)
        print(f"Final holdout CLI {t['target_source_design_id'][:12]}: returncode={result.returncode}", flush=True)
    cases = [case_report(root, cfg, t) for t in targets]
    report = {"status": "seven_center_final_holdout_software_check_complete", "metrics": aggregate(cases), "cases": cases,
              "target_holdout_labels_opened": True, "training_holdout_labels_parsed": False, "model_refitted": False,
              "independent_physics_validation": False, "physical_gpu_index": 6,
              "formal_use_allowed": False, "deployment_allowed": False}
    write_new(output / "report.json", report)
    manifest = {"status": report["status"], "protocol_lock_sha256": sha(output / "protocol_lock.json"),
                "target_label_manifest_sha256": sha(output / "target_label_manifest.json"), "report_sha256": sha(output / "report.json"),
                "execution_record_sha256": {t["target_source_design_id"]: sha(output / "cases" / t["target_source_design_id"] / "execution_record.json")
                                            for t in targets if t["usable_target"]}}
    write_new(output / "complete_manifest.json", manifest)
    return verify_complete(root, cfg)


def verify_complete(root: Path, cfg: dict) -> dict:
    targets = verify_labels(root, cfg)
    output = root / cfg["output_root"]
    manifest = read(output / "complete_manifest.json")
    require(manifest["protocol_lock_sha256"] == sha(output / "protocol_lock.json")
            and manifest["target_label_manifest_sha256"] == sha(output / "target_label_manifest.json")
            and manifest["report_sha256"] == sha(output / "report.json"), "Final holdout complete artifacts changed")
    require(manifest["execution_record_sha256"] == {t["target_source_design_id"]: sha(output / "cases" / t["target_source_design_id"] / "execution_record.json")
            for t in targets if t["usable_target"]}, "Final holdout execution records changed")
    report = read(output / "report.json")
    cases = [case_report(root, cfg, t) for t in targets]
    require(report["cases"] == cases and report["metrics"] == aggregate(cases) and report["model_refitted"] is False
            and report["training_holdout_labels_parsed"] is False and report["target_holdout_labels_opened"] is True
            and report["formal_use_allowed"] is False and report["deployment_allowed"] is False, "Final holdout result summary changed")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "label", "evaluate", "verify"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    cfg = config_at(root)
    result = {"freeze": freeze, "label": label, "evaluate": evaluate, "verify": verify_complete}[args.command](root, cfg)
    print(json.dumps(result.get("metrics", result), sort_keys=True))


if __name__ == "__main__":
    main()
