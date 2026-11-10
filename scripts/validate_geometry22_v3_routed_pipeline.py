#!/usr/bin/env python3
"""Exercise the public routed V3 CLI on opened L6h targets and an abstention probe."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from src.ai.geometry19_pipeline_artifact import verify_completed_artifact
from src.ai.geometry22_pipeline_v3_routed import sha, write


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "6" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("CLI acceptance requires physical GPU 6")
    print({"cuda_available": True, "device_count": 1, "device_name": torch.cuda.get_device_name(0), "physical_gpu_index": 6}, flush=True)
    output = ROOT / "data/validation/v3_routed_pipeline_acceptance_v1"
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "data/validation/v3_22d_fresh_validation_l6h_v1/target_labels.jsonl"
    labels = [json.loads(s) for s in source.read_text().splitlines()]
    if len(labels) != 6 or any(t["split"] != "validation" for t in labels):
        raise ValueError("Acceptance target population changed")
    targets = [(t["target_id"], t["target_performance"]) for t in labels]
    targets.append(("out_of_support_probe", {"thrust_N": 1e9, "tsfc_kg_N_s": 1., "mass_kg": 1e8}))
    reports = []
    for target_id, performance in targets:
        case = output / target_id
        case.mkdir()
        path = case / "target.yaml"
        path.write_text(yaml.safe_dump(performance))
        cmd = [sys.executable, str(ROOT / "scripts/run_wp19_23_pipeline.py"), "--model", "model V3",
               "--pipeline-profile", "v3_routed_development", "--target", str(path), "--output-root", str(case / "run")]
        for args in (cmd, cmd + ["--resume"]):
            result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
            print(result.stdout, end="", flush=True)
            if result.returncode:
                raise RuntimeError(result.stderr)
        report = verify_completed_artifact(case / "run", ROOT)
        summary = report["summary"]
        if target_id == "out_of_support_probe":
            if summary["route_status"] != "abstained_no_target_support" or summary["physics_evaluation_count"] != 0:
                raise ValueError("Out-of-support target was evaluated")
        elif summary["candidate_count"] != 24 or summary["physics_evaluation_count"] != 25:
            raise ValueError("Routed CLI budget changed")
        reports.append({"source_target_id": target_id, "integrity_pass": report["integrity_pass"],
                        "complete_resume_pass": True, "summary": summary})
    write(output / "acceptance_report.json", {
        "status": "v3_routed_public_cli_software_acceptance_complete", "source_target_labels_sha256": sha(source),
        "acceptance_script_sha256": sha(Path(__file__)), "diagnostic_targets": 6, "abstention_probes": 1,
        "new_independent_validation": False, "holdout_labels_parsed": False,
        "physical_gpu_index": 6, "cases": reports, "formal_use_allowed": False, "deployment_allowed": False})
    print("V3 routed CLI acceptance complete", flush=True)


if __name__ == "__main__":
    main()
