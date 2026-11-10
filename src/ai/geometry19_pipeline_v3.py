"""Isolated V3 22D B--F development flow using explicit software fixtures."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import yaml

from src.ai.v3_development_assets import OUTPUT_NAMES, V3FixtureAssetError, load_v3_fixture_assets
from src.physics.v3_development_runtime import build_v3_development_physics_runtime


CONFIG = "config/experiments/geometry22_pipeline_v3_development_v1.yaml"
MODEL_SELECTOR = "model V3"
STAGES = ("B", "C", "D", "E", "F")
IDENTITY_CODE = (
    "src/ai/geometry19_pipeline_v3.py",
    "src/ai/v3_development_assets.py",
    "src/physics/v3_development_runtime.py",
    "src/ai/geometry19_pipeline_dispatch.py",
    "scripts/run_wp19_23_pipeline.py",
)


class Geometry19V3PendingError(ValueError):
    """A V3 development option, asset, or run identity is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Geometry19V3PendingError(message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rho(performance: Mapping[str, float] | None, target: Mapping[str, Any], tolerances: Mapping[str, float]) -> float | None:
    if performance is None:
        return None
    return max(
        abs(float(performance[name]) - float(target[name]))
        / (float(target[name]) * float(tolerances[name]))
        for name in OUTPUT_NAMES
    )


def _load_target_v3(
    target_path: str | Path | None,
    target_values: Mapping[str, float] | None,
    target_type: str,
    expected_status: str,
) -> dict[str, Any]:
    _require((target_path is None) != (target_values is None), "Specify exactly one V3 target source")
    if target_path is not None:
        path = Path(target_path)
        try:
            source = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise Geometry19V3PendingError("Cannot read V3 target document") from exc
        _require(isinstance(source, Mapping), "V3 target document must be a mapping")
        _require(all(name in source for name in OUTPUT_NAMES), "V3 target requires three performance values")
        values = {name: source[name] for name in OUTPUT_NAMES}
        target_type = str(source.get("target_type", target_type))
        expected_status = str(source.get("expected_status", expected_status))
    else:
        _require(set(target_values) == set(OUTPUT_NAMES), "V3 target requires three performance values")
        values = target_values
    _require(target_type in {"ID", "INTERPOLATION", "BOUNDARY_SPARSE", "OUT_OF_EMPIRICAL_SUPPORT"}, "Invalid V3 target type")
    _require(expected_status in {"KNOWN_FEASIBLE", "UNKNOWN", "EXPECTED_INFEASIBLE"}, "Invalid V3 expected status")
    try:
        performance = {name: float(values[name]) for name in OUTPUT_NAMES}
    except (TypeError, ValueError) as exc:
        raise Geometry19V3PendingError("V3 target values must be numeric") from exc
    _require(all(math.isfinite(value) and value > 0 for value in performance.values()), "V3 target values must be positive and finite")
    target = {
        "target_contract_version": "geometry22_v3_development_target_v1",
        "target_type": target_type,
        "expected_status": expected_status,
        **performance,
    }
    encoded = json.dumps(target, sort_keys=True, separators=(",", ":")).encode("utf-8")
    target["target_id"] = hashlib.sha256(encoded).hexdigest()
    return target


def _verify_complete(output: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    try:
        lock = _read(output / "Protocol_Lock.json")
        manifest = _read(output / "Complete_Manifest.json")
        _require(lock["identity"] == identity, "V3 pipeline identity changed")
        _require(manifest["manifest_version"] == "geometry22_v3_development_pipeline_v1", "V3 manifest version changed")
        _require(manifest["lock_sha256"] == _sha(output / "Protocol_Lock.json"), "V3 lock changed")
        _require(manifest["stages"] == list(STAGES), "V3 stage inventory changed")
        for name, digest in manifest["files"].items():
            relative = Path(name)
            _require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe V3 manifest path")
            item = (output / relative).resolve()
            _require(item.is_relative_to(output.resolve()), "V3 manifest path escapes run")
            _require(_sha(item) == digest, f"V3 artifact changed: {name}")
        actual = {
            str(path.relative_to(output))
            for path in output.rglob("*")
            if path.is_file() and path.name != "Complete_Manifest.json"
        }
        _require(actual == set(manifest["files"]), "V3 artifact inventory changed")
        summary = _read(output / "Pipeline_Summary.json")
        _require(summary["model"] == MODEL_SELECTOR, "V3 summary model changed")
        _verify_stage_identity(output, lock, summary)
        return summary
    except Geometry19V3PendingError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise Geometry19V3PendingError("Cannot verify complete V3 pipeline run") from exc


def _verify_stage_identity(output: Path, lock: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    """Check the B--F relationship as well as recorded file bytes."""
    from src.physics.geometry22_design import Geometry22Design

    contract = lock["identity"]["runtime_contract"]
    schema_hash = contract["design_schema_hash"]
    physics_hash = contract["physics_bundle_hash"]
    candidates = _read(output / "B" / "candidates.json")
    _require(isinstance(candidates, list), "V3 B candidates changed shape")
    candidate_ids = []
    for index, row in enumerate(candidates):
        design = Geometry22Design.from_mapping(row["design"])
        _require(row["rank"] == index and row["design_id"] == design.design_id(), "V3 B design identity changed")
        candidate_ids.append(row["design_id"])
    _require(len(set(candidate_ids)) == len(candidate_ids), "V3 B has duplicate designs")
    _require(len(candidate_ids) in (0, 45), "V3 fixture candidate count changed")
    _require(_read(output / "B" / "summary.json")["count"] == len(candidate_ids), "V3 B count changed")

    evaluated = []
    stage_counts = []
    for stage in ("C", "D", "E"):
        rows = _read(output / stage / "evaluations.json")
        _require(isinstance(rows, list), f"V3 {stage} evaluations changed shape")
        stage_counts.append(len(rows))
        for row in rows:
            design = Geometry22Design.from_mapping(row["design"])
            _require(row["design_id"] == design.design_id(), f"V3 {stage} design identity changed")
            _require(row["design_schema_hash"] == schema_hash, f"V3 {stage} schema changed")
            _require(row["physics_bundle_hash"] == physics_hash, f"V3 {stage} physics changed")
            _require(row["formal_use_allowed"] is False and row["deployment_allowed"] is False, f"V3 {stage} use status changed")
            _require((row["performance"] is None) == (not row["solver_converged"]), f"V3 {stage} solver status changed")
            _require(row["rho"] is None or math.isfinite(row["rho"]), f"V3 {stage} target distance changed")
        stage_summary = _read(output / stage / "summary.json")
        _require(stage_summary["evaluated_count"] == len(rows), f"V3 {stage} count changed")
        _require(stage_summary["converged_count"] == sum(row["solver_converged"] for row in rows), f"V3 {stage} convergence count changed")
        _require(stage_summary["feasible_count"] == sum(row["feasible"] for row in rows), f"V3 {stage} feasibility count changed")
        evaluated.extend(rows)
    _require(stage_counts == ([8, 8, 29] if candidate_ids else [0, 0, 0]), "V3 stage budgets changed")
    _require([row["design_id"] for row in evaluated] == candidate_ids, "V3 stage candidate order changed")
    f = _read(output / "F" / "summary.json")
    selected = next((row for row in evaluated if row["design_id"] == f["selected_design_id"]), None)
    converged = [row for row in evaluated if row["rho"] is not None]
    eligible = [row for row in converged if row["feasible"]]
    expected_selected = min(eligible or converged, key=lambda row: (row["rho"], row["design_id"])) if converged else None
    _require(selected == expected_selected, "V3 F selection changed")
    if selected is None:
        _require(f["status"] == "NO_CANDIDATES" and f["independent_recompute"] is None, "V3 F empty status changed")
    else:
        _require(f["independent_recompute"] == selected, "V3 F recompute identity changed")
        delivery = bool(selected["feasible"] and selected["rho"] is not None and selected["rho"] <= 1.0)
        _require(f["status"] == ("PASS" if delivery else "NO_HIT"), "V3 F delivery status changed")
    delivered_ids = [selected["design_id"]] if selected is not None and f["status"] == "PASS" else []
    _require(summary["delivery_status"] == f["status"], "V3 delivery summary changed")
    _require(summary["status"] == ("V3_DEVELOPMENT_COMPLETED" if delivered_ids else "V3_DEVELOPMENT_NO_DELIVERED_HIT"), "V3 terminal status changed")
    _require(summary["delivered_design_ids"] == delivered_ids, "V3 delivered designs changed")
    _require(summary["candidate_count"] == len(candidate_ids), "V3 candidate count changed")
    _require(summary["physics_evaluation_count"] == len(evaluated) + (selected is not None), "V3 evaluation count changed")
    _require(summary["design_dimension"] == 22 and summary["design_schema_hash"] == schema_hash, "V3 summary schema changed")
    _require(summary["physics_bundle_hash"] == physics_hash, "V3 summary physics changed")
    _require(summary["formal_use_allowed"] is False and summary["deployment_allowed"] is False, "V3 summary use status changed")


def _evaluate(design, runtime, target, tolerances) -> dict[str, Any]:
    result = runtime.evaluate(design)
    engine = result.engine_result
    performance = (
        {name: float(engine.performance[name]) for name in OUTPUT_NAMES}
        if engine.solver.converged else None
    )
    return {
        "design_id": result.design_id,
        "design": design.as_mapping(),
        "design_schema_hash": result.design_schema_hash,
        "physics_bundle_hash": result.physics_bundle_hash,
        "physics_model_version": result.physics_model_version,
        "solver_converged": engine.solver.converged,
        "solver_failure_code": engine.solver.failure_code.value if engine.solver.failure_code else None,
        "feasible": bool(engine.feasible),
        "performance": performance,
        "rho": _rho(performance, target, tolerances),
        "physics_validation_status": "not_completed",
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }


def _execute(output: Path, identity: Mapping[str, Any], runtime, assets, target, config) -> dict[str, Any]:
    tolerances = config["relative_tolerances"]
    _json(output / "Protocol_Lock.json", {
        "identity": identity,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    _json(output / "target.json", target)

    # B ranks stored labels only. Failed rows remain candidates with no score.
    by_id = {row["design_id"]: row for row in assets.rows}
    ranked = sorted(
        assets.candidate_designs(),
        key=lambda design: (
            math.inf if by_id[design.design_id()]["performance"] is None else
            _rho(by_id[design.design_id()]["performance"], target, tolerances),
            design.design_id(),
        ),
    )
    b_rows = [
        {
            "rank": index,
            "design_id": design.design_id(),
            "design": design.as_mapping(),
            "forward_fixture_rho": _rho(by_id[design.design_id()]["performance"], target, tolerances),
            "forward_fixture_label_available": by_id[design.design_id()]["performance"] is not None,
        }
        for index, design in enumerate(ranked)
    ]
    _json(output / "B" / "candidates.json", b_rows)
    _json(output / "B" / "summary.json", {"status": "COMPLETE" if b_rows else "NO_CANDIDATES", "count": len(b_rows)})

    cursor = 0
    evaluated: list[dict[str, Any]] = []
    for stage in ("C", "D", "E"):
        budget = int(config["stage_budgets"][stage])
        current = ranked[cursor : cursor + budget]
        cursor += budget
        rows = [_evaluate(design, runtime, target, tolerances) for design in current]
        evaluated.extend(rows)
        _json(output / stage / "evaluations.json", rows)
        _json(output / stage / "summary.json", {
            "status": "COMPLETE" if rows else "NO_CANDIDATES",
            "evaluated_count": len(rows),
            "converged_count": sum(row["solver_converged"] for row in rows),
            "feasible_count": sum(row["feasible"] for row in rows),
        })
    _require(cursor >= len(ranked), "V3 stage budgets did not cover all fixture candidates")

    converged = [row for row in evaluated if row["rho"] is not None]
    eligible = [row for row in converged if row["feasible"]]
    selected = min(eligible or converged, key=lambda row: (row["rho"], row["design_id"])) if converged else None
    if selected is None:
        f_summary = {"status": "NO_CANDIDATES", "selected_design_id": None, "independent_recompute": None}
    else:
        from src.physics.geometry22_design import Geometry22Design

        # F constructs a fresh runtime and does not reuse a C/D/E result.
        f_runtime = build_v3_development_physics_runtime(runtime.repository_root)
        _require(f_runtime.physics_bundle_hash == runtime.physics_bundle_hash, "V3 F runtime changed")
        f_result = _evaluate(Geometry22Design.from_mapping(selected["design"]), f_runtime, target, tolerances)
        _require(f_result == selected, "V3 F independent recompute differs")
        delivered = bool(f_result["feasible"] and f_result["rho"] is not None and f_result["rho"] <= 1.0)
        f_summary = {
            "status": "PASS" if delivered else "NO_HIT",
            "selected_design_id": selected["design_id"],
            "independent_recompute": f_result,
        }
    _json(output / "F" / "summary.json", f_summary)
    delivered_ids = [f_summary["selected_design_id"]] if f_summary["status"] == "PASS" else []
    summary = {
        "status": "V3_DEVELOPMENT_COMPLETED" if delivered_ids else "V3_DEVELOPMENT_NO_DELIVERED_HIT",
        "model": MODEL_SELECTOR,
        "target_id": target["target_id"],
        "terminal_source_stage": "F",
        "delivery_status": f_summary["status"],
        "delivered_design_ids": delivered_ids,
        "candidate_count": len(b_rows),
        "physics_evaluation_count": len(evaluated) + (selected is not None),
        "design_dimension": 22,
        "design_schema_hash": runtime.design_schema_hash,
        "physics_bundle_hash": runtime.physics_bundle_hash,
        "asset_version": assets.manifest["asset_version"],
        "asset_status": "software_fixture_only_unvalidated",
        "physics_validation_status": "not_completed",
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "paths": {
            "runtime_root": str(output),
            "pipeline_manifest": str(output / "Complete_Manifest.json"),
        },
    }
    _json(output / "Pipeline_Summary.json", summary)
    files = {
        str(path.relative_to(output)): _sha(path)
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    _json(output / "Complete_Manifest.json", {
        "manifest_version": "geometry22_v3_development_pipeline_v1",
        "lock_sha256": _sha(output / "Protocol_Lock.json"),
        "stages": list(STAGES),
        "files": files,
    })
    return summary


def run_wp19_23_pipeline_v3(
    repository_root: str | Path,
    *,
    target_path: str | Path | None = None,
    target_values: Mapping[str, float] | None = None,
    target_type: str = "ID",
    expected_status: str = "UNKNOWN",
    output_root: str | Path | None = None,
    resume: bool = False,
    device: str = "cpu",
    refinement_device: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the CPU-only 22D fixture path with immutable output and strict resume."""
    forbidden = {
        "pipeline_profile": "--pipeline-profile",
        "model_variant": "--model-variant",
        "forward_model_tag": "--forward-model-tag",
        "sampling_mode": "--sampling",
        "storage_root": "--storage-root",
        "restart_from_stage": "--restart-from-stage",
        "pipeline_context": "V2.1 pipeline context",
    }
    for key, option in forbidden.items():
        if kwargs.get(key) is not None:
            raise Geometry19V3PendingError(f"{option} is not available for model V3 development")
    _require(not kwargs.get("export_mesh"), "--export-mesh is not available for model V3 development")
    _require(device == "cpu" and refinement_device is None, "model V3 development uses CPU only")
    _require(set(kwargs) <= set(forbidden) | {"export_mesh"}, "Unknown model V3 pipeline option")

    root = Path(repository_root).resolve()
    config_path = root / CONFIG
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require(config["model_selector"] == MODEL_SELECTOR, "V3 config model mismatch")
    _require(config["stages"] == list(STAGES), "V3 config stage mismatch")
    _require(sum(config["stage_budgets"].values()) == 45, "V3 fixture budget mismatch")
    _require(all(float(config["relative_tolerances"][name]) > 0 for name in OUTPUT_NAMES), "V3 tolerances invalid")
    target = _load_target_v3(
        target_path, target_values, target_type=target_type, expected_status=expected_status
    )
    runtime = build_v3_development_physics_runtime(root)
    try:
        assets = load_v3_fixture_assets(root / config["asset_root"], runtime)
    except V3FixtureAssetError as exc:
        raise Geometry19V3PendingError("V3 fixture asset preflight failed") from exc
    identity = {
        "model": MODEL_SELECTOR,
        "config_sha256": _sha(config_path),
        "source_code_sha256": {name: _sha(root / name) for name in IDENTITY_CODE},
        "runtime_contract": runtime.contract_mapping(),
        "asset_manifest_sha256": _sha(root / config["asset_root"] / "complete_manifest.json"),
        "asset_file_sha256": assets.manifest["files"],
        "target": target,
        "device": "cpu",
    }
    output = (
        Path(output_root).resolve() if output_root is not None else
        root / config["default_output_root"] / runtime.physics_bundle_hash[:12] / target["target_id"]
    )
    if resume:
        _require(output.is_dir(), "No model V3 development run to resume")
        _require(not (output / "Failure_Manifest.json").is_file(), "Failed model V3 run exists; preserve and stop")
        _require((output / "Complete_Manifest.json").is_file(), "Partial model V3 run exists; preserve and stop")
        return _verify_complete(output, identity)
    _require(not output.exists(), f"Refusing to overwrite V3 pipeline run: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".v3_pipeline_", dir=output.parent))
    try:
        summary = _execute(temporary, identity, runtime, assets, target, config)
        summary["paths"] = {
            "runtime_root": str(output),
            "pipeline_manifest": str(output / "Complete_Manifest.json"),
        }
        _json(temporary / "Pipeline_Summary.json", summary)
        complete = _read(temporary / "Complete_Manifest.json")
        complete["files"]["Pipeline_Summary.json"] = _sha(temporary / "Pipeline_Summary.json")
        _json(temporary / "Complete_Manifest.json", complete)
        os.rename(temporary, output)
    except Exception as exc:
        _json(temporary / "Failure_Summary.json", {
            "status": "V3_DEVELOPMENT_FAILED",
            "model": MODEL_SELECTOR,
            "target_id": target["target_id"],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "formal_use_allowed": False,
            "deployment_allowed": False,
        })
        _json(temporary / "Failure_Manifest.json", {
            "status": "V3_DEVELOPMENT_FAILED",
            "files": {
                str(path.relative_to(temporary)): _sha(path)
                for path in sorted(temporary.rglob("*"))
                if path.is_file() and path.name != "Failure_Manifest.json"
            },
        })
        os.rename(temporary, output)
        raise Geometry19V3PendingError(f"V3 development run failed; see {output / 'Failure_Summary.json'}") from exc
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return _verify_complete(output, identity)
