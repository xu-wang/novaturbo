"""Unified experimental B--F entry for the V2.2 local inverse-design flow."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import time

import yaml

from src.ai import geometry19_performance_gated_g3 as g3
from src.ai.geometry19_local_full_pipeline import LocalTask
from src.ai.geometry19_delivery import performance_matches
from src.ai.geometry19_performance_gated_protocol import canonical, file_hash, require, within
from src.physics import Geometry19, derive_geometry_state

CONFIG = "config/experiments/geometry19_local_full_pipeline_difficult_v1.yaml"
STAGES = ("B", "C", "D", "E", "F")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    payload = b"".join(canonical(row) + b"\n" for row in rows)
    g3.adapter._write_once(path, payload)


def seal_stage(directory: Path, summary: dict, ledger: list[dict] | None = None) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    g3.seal_json(directory / "summary.json", summary)
    names = ["summary.json"]
    if ledger is not None:
        write_jsonl(directory / "ledger.jsonl", ledger)
        names.append("ledger.jsonl")
    seal = {"stage": summary["stage"], "files": {name: file_hash(directory / name) for name in names}}
    g3.seal_json(directory / "Seal.json", seal)
    return seal


def verify_stage(directory: Path, expected_stage: str) -> dict:
    require(directory.is_dir(), f"Missing stage {expected_stage}")
    seal = json.loads((directory / "Seal.json").read_text())
    require(seal["stage"] == expected_stage, "Stage identity changed")
    for name, expected in seal["files"].items():
        require(file_hash(directory / name) == expected, f"Stage {expected_stage} artifact changed")
    return seal


def verify_target(directory: Path, lock_hash: str) -> dict:
    manifest = json.loads((directory / "Target_Manifest.json").read_text())
    require(manifest["lock_sha256"] == lock_hash, "Target lock changed")
    for stage in STAGES:
        verify_stage(directory / stage, stage)
    for name, expected in manifest["files"].items():
        require(file_hash(within(directory, name)) == expected, "Target artifact changed")
    return json.loads((directory / "Pipeline_Summary.json").read_text())


def stage_rows(task: LocalTask, start: int) -> list[dict]:
    return deepcopy(task.ledger[start:])


def deliver_exact_geometry(task: LocalTask, result: dict) -> dict:
    """Validate and recompute the serialized float64 geometry without a float32 round trip."""
    geometry = json.loads((task.output / "Delivery_Geometry.json").read_text()) if (task.output / "Delivery_Geometry.json").exists() else deepcopy(result["geometry19"])
    if not (task.output / "Delivery_Geometry.json").exists():
        g3.seal_json(task.output / "Delivery_Geometry.json", geometry)
    expert, distance = g3.adapter._assign_pooled_support(geometry, task.dec, .20)
    require(expert is not None, "Serialized delivery geometry left local support")
    design = Geometry19.from_mapping(geometry)
    derive_geometry_state(design, task.context.physics_runtime.model_config)
    row = {
        "valid": True, "geometry_hard_valid": True, "support_valid": True,
        "geometry19": design.as_mapping(), "design_hash": result["design_hash"],
        "support_root_id": expert["root_seed_id"], "support_root_index": expert["root_index"],
        "local_distance": distance,
    }
    recalculated = task.evaluate(row, "F", force=True)
    passed = bool(
        recalculated["development_diagnostic_hit"]
        and performance_matches(result["performance"], recalculated["performance"])
    )
    g3.seal_json(task.output / "F.json", {
        "passed": passed, "reference": result, "recalculated": recalculated,
        "cache_bypassed": True, "serialized_float64_geometry_reused_exactly": True,
    })
    return {"passed": passed, "design_hash": result["design_hash"]}


def run_target(data, config, models, scorer_ids, scorer, index: int, directory: Path, lock_hash: str) -> dict:
    require(not directory.exists(), "Partial target exists; preserve and stop")
    directory.mkdir(parents=True)
    runtime = directory / "runtime"
    started = time.monotonic()
    task = LocalTask(data, config, models, scorer_ids, scorer, index, runtime)
    gradient = task.derivative_audit()

    ranked = task.generate()
    b_payload = json.loads((runtime / "B.json").read_text())
    b_summary = {
        "stage": "B", "status": "CANDIDATES_READY", "target_index": index,
        "selected_expert": task.route, "raw_candidates": config["raw_candidates"],
        "valid_unique_ranked": len(ranked), "dflow_enabled": True,
        "gradient_audit": gradient,
    }
    seal_stage(directory / "B", b_summary, b_payload["candidates"])

    before = len(task.ledger)
    hit = task.stage_c()
    c_rows = stage_rows(task, before)
    c_summary = {
        "stage": "C", "status": "HIT" if hit else "NO_HIT", "target_index": index,
        "budget": config["budgets"]["C"], "physics_calls": task.costs["C_calls"],
        "cache_hits": task.costs["C_cache"], "hit_design_hash": None if hit is None else hit["design_hash"],
    }
    seal_stage(directory / "C", c_summary, c_rows)

    if hit is not None:
        d_summary = {"stage": "D", "status": "SKIPPED_C_HIT", "target_index": index, "physics_calls": 0}
        seal_stage(directory / "D", d_summary, [])
    else:
        before = len(task.ledger)
        hit = task.refine(ranked[:8])
        d_rows = stage_rows(task, before)
        d_summary = {
            "stage": "D", "status": "HIT" if hit else "NO_HIT", "target_index": index,
            "budget": config["budgets"]["D"], "physics_calls": task.costs["D_calls"],
            "cache_hits": task.costs["D_cache"], "hit_design_hash": None if hit is None else hit["design_hash"],
        }
        require(d_summary["physics_calls"] <= config["budgets"]["D"], "D exceeded budget")
        seal_stage(directory / "D", d_summary, d_rows)

    if hit is not None:
        trigger = "SKIPPED_C_HIT" if c_summary["status"] == "HIT" else "SKIPPED_D_HIT"
        e_summary = {"stage": "E", "status": trigger, "target_index": index, "physics_calls": 0}
        seal_stage(directory / "E", e_summary, [])
    else:
        before = len(task.ledger)
        hit = task.search(ranked)
        e_rows = stage_rows(task, before)
        e_summary = {
            "stage": "E", "status": "HIT" if hit else "NO_HIT", "target_index": index,
            "budget": config["budgets"]["E"], "physics_calls": task.costs["E_calls"],
            "cache_hits": task.costs["E_cache"], "hit_design_hash": None if hit is None else hit["design_hash"],
        }
        require(e_summary["physics_calls"] <= config["budgets"]["E"], "E exceeded budget")
        seal_stage(directory / "E", e_summary, e_rows)

    if hit is None:
        f_summary = {"stage": "F", "status": "NO_DELIVERABLE_HIT", "target_index": index, "physics_calls": 0}
        delivery = None
        seal_stage(directory / "F", f_summary, [])
        first_hit_stage = None
    else:
        first_hit_stage = next(name for name, summary in (("C", c_summary), ("D", d_summary), ("E", e_summary)) if summary["status"] == "HIT")
        before = len(task.ledger)
        g3.seal_json(task.output / "Delivery_Geometry.json", hit["geometry19"])
        delivery = deliver_exact_geometry(task, hit)
        f_rows = stage_rows(task, before)
        f_summary = {
            "stage": "F", "status": "PASS" if delivery["passed"] else "FAIL",
            "target_index": index, "physics_calls": task.costs["F_calls"],
            "source_stage": first_hit_stage, "design_hash": delivery["design_hash"],
            "cache_bypassed": True,
        }
        require(f_summary["physics_calls"] == 1, "F must independently recalculate once")
        seal_stage(directory / "F", f_summary, f_rows)

    summary = {
        "work_package": "V2.2-LOCAL-BF-EXPERIMENTAL", "target_index": index,
        "target_class": data["targets"][index]["target_class"],
        "status": "PASS" if delivery and delivery["passed"] else "NO_DELIVERED_HIT",
        "first_hit_stage": first_hit_stage,
        "completed_stages": list(STAGES), "costs": dict(task.costs),
        "elapsed_seconds": time.monotonic() - started,
        "production_entry": False, "formal_use_allowed": False,
    }
    g3.seal_json(directory / "Pipeline_Summary.json", summary)
    files = {
        str(path.relative_to(directory)): file_hash(path)
        for path in directory.rglob("*")
        if path.is_file() and path.name != "Target_Manifest.json"
    }
    g3.seal_json(directory / "Target_Manifest.json", {"lock_sha256": lock_hash, "files": files})
    return summary


def run(root: str | Path, *, resume: bool = False, verify_only: bool = False) -> dict:
    root = Path(root).resolve()
    config = yaml.safe_load((root / CONFIG).read_text())
    require(config["stages"] == list(STAGES), "B--F stage order changed")
    data = g3.preflight(root)
    output = within(root, config["output"])
    code_paths = [
        CONFIG, "src/ai/geometry19_local_full_pipeline_entry.py",
        "scripts/run_geometry19_local_full_pipeline_entry.py",
        "tests/test_geometry19_local_full_pipeline_entry.py",
        "src/ai/geometry19_local_full_pipeline.py",
    ]
    identity = {
        "protocol_sha256": file_hash(root / CONFIG),
        "code": {path: file_hash(root / path) for path in code_paths},
        "inputs": data["inputs"], "runtime": data["context"].contract_mapping(),
        "targets": config["targets"],
    }
    if resume or verify_only:
        require(output.is_dir(), "No campaign to resume or verify")
        lock = json.loads((output / "Protocol_Lock.json").read_text())
        require(all(lock[key] == value for key, value in identity.items()), "Campaign identity changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        lock = {**identity, "created_at": datetime.now(timezone.utc).isoformat()}
        g3.seal_json(output / "Protocol_Lock.json", lock)
    lock_hash = g3.digest(lock)
    with (output / ".runner.lock").open("a") as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        summaries = []
        for index in config["targets"]:
            directory = output / f"target{index:03d}"
            if directory.exists():
                require((directory / "Target_Manifest.json").is_file(), "Partial target exists; preserve and stop")
                summaries.append(verify_target(directory, lock_hash))
                continue
            require(not verify_only, "Verified campaign is incomplete")
            if "models" not in locals():
                g3.adapter._cuda_manifest()
                models, _ = g3.adapter.load_seed_models(root, data["inference"], config["training_seed"], data["decoders"], "cuda:0")
                scorer_ids, scorer = g3._load_model(data["paths"]["s2_oracle_local_ridge"], "stable_38_oracle_root_local_ridge")
            summary = run_target(data, config, models, scorer_ids, scorer, index, directory, lock_hash)
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
        campaign = {
            "status": "N3_UNIFIED_BF_DIFFICULT_COMPLETED",
            "targets": summaries,
            "first_hit_stage_counts": dict(__import__("collections").Counter(str(row["first_hit_stage"]) for row in summaries)),
            "all_stage_contracts_verified": True,
            "all_delivery_pass": all(row["status"] == "PASS" for row in summaries),
            "production_entry_integrated": False,
            "resume_unit": "completed_target",
        }
        if (output / "Summary.json").exists():
            require(json.loads((output / "Summary.json").read_text()) == campaign, "Campaign summary changed")
        elif not verify_only:
            g3.seal_json(output / "Summary.json", campaign)
        require((output / "Summary.json").is_file(), "Campaign summary missing")
        files = {
            str(path.relative_to(output)): file_hash(path)
            for path in output.rglob("*")
            if path.is_file() and path.name not in {".runner.lock", "Complete_Manifest.json"}
        }
        complete = {"lock_sha256": lock_hash, "files": files}
        if (output / "Complete_Manifest.json").exists():
            require(json.loads((output / "Complete_Manifest.json").read_text()) == complete, "Complete manifest changed")
        elif not verify_only:
            g3.seal_json(output / "Complete_Manifest.json", complete)
        require((output / "Complete_Manifest.json").is_file(), "Complete manifest missing")
        return campaign
