"""Preview/UI/JSON delivery closure for WP19-23F."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from src.ai.geometry19_inverse_contract import load_target
from src.ai.geometry19_hit_alignment import metrics_from_relative_errors
from src.ai.geometry19_selection import (
    rank_delivery_candidates,
)
from src.ai.geometry19_pipeline_context import PipelineContext, resolve_pipeline_context
from src.ai.geometry19_surrogate import sha256_file
from src.physics import Geometry19
from src.visualization import Geometry19DesignStore


class Geometry19DeliveryError(ValueError):
    """Raised when a WP19-23F delivery artifact fails closure checks."""


def validate_delivery_source_summary(
    summary: Mapping[str, Any], target_id: str
) -> str:
    """Validate a terminal inverse stage and return its WP identity."""

    source_stage = str(summary.get("work_package"))
    if source_stage not in {"WP19-23C", "WP19-23D", "WP19-23E"}:
        raise Geometry19DeliveryError("Delivery source must be WP19-23C, D, or E")
    if summary.get("target_id") != target_id:
        raise Geometry19DeliveryError(f"{source_stage} target mismatch")
    if not summary.get("downstream_stop"):
        raise Geometry19DeliveryError(f"{source_stage} is not ready for delivery")
    return source_stage


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def performance_matches(
    reference: Mapping[str, Any],
    recalculated: Mapping[str, Any],
    *,
    relative_tolerance: float = 1.0e-12,
) -> bool:
    for name in ("thrust_N", "tsfc_kg_N_s", "mass_kg"):
        left = float(reference[name])
        right = float(recalculated[name])
        if not math.isfinite(left) or not math.isfinite(right):
            return False
        if abs(left - right) > relative_tolerance * max(abs(left), abs(right), 1.0e-15):
            return False
    return True


def run_delivery_stage(
    repository_root: str | Path,
    target_path: str | Path,
    *,
    nsga2_root: str | Path | None = None,
    source_root: str | Path | None = None,
    output_root: str | Path | None = None,
    storage_root: str | Path | None = None,
    network_architecture_modified: bool = False,
    export_mesh: bool = False,
    pipeline_context: PipelineContext | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    target_source = Path(target_path).resolve()
    target = load_target(target_source)
    context = resolve_pipeline_context(root, pipeline_context)
    contract_path = context.contract_path
    contract = context.contract
    protocol = context.protocol
    physics_runtime = context.physics_runtime
    simplified_model = bool(
        protocol.get("delivery", {}).get("simplified_model", False)
    )
    formal_use_allowed = bool(protocol["formal_use_allowed"])
    development_only = bool(
        protocol.get("delivery", {}).get(
            "development_only", not formal_use_allowed
        )
    )
    if (
        context.profile.mode != "production_development"
        or protocol.get("delivery", {}).get("enabled") is False
    ):
        raise Geometry19DeliveryError(
            "Delivery is disabled for the selected pipeline profile"
        )
    base = root / protocol["artifact_protocol"]["runtime_root"] / target.target_id
    if source_root is not None and nsga2_root is not None:
        raise Geometry19DeliveryError("Specify source_root or nsga2_root, not both")
    if source_root is not None:
        source = Path(source_root).resolve()
    elif nsga2_root is not None:
        source = Path(nsga2_root).resolve()
    else:
        source = base / "wp19_23e"
    runtime = Path(output_root).resolve() if output_root else base / "wp19_23f"
    mesh_storage = Path(storage_root).resolve() if storage_root else (
        root / protocol["artifact_protocol"]["mesh_storage_root"]
    )
    if runtime.exists():
        raise Geometry19DeliveryError(f"Refusing to overwrite delivery run: {runtime}")
    runtime.mkdir(parents=True)
    try:
        source_summary_path = source / "summary.json"
        source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
        candidates = json.loads((source / "delivered_candidates.json").read_text(encoding="utf-8"))
        source_stage = validate_delivery_source_summary(source_summary, target.target_id)
        if not candidates:
            raise Geometry19DeliveryError(f"{source_stage} produced no deliverable candidate")
        maximum = int(protocol["delivery"]["maximum_candidates"])
        if len(candidates) > maximum:
            raise Geometry19DeliveryError("Delivery candidate count exceeds protocol maximum")
        schema_path = physics_runtime.source_paths["design_schema"]
        schema = physics_runtime.design_schema
        fields = list(schema["design_variables"])
        dominance_contract = context.constraint_dominance
        ranked_candidates = rank_delivery_candidates(candidates, maximum, fields)
        if len(ranked_candidates) != len(candidates):
            raise Geometry19DeliveryError(
                "Delivery source contains duplicate or non-physical candidates"
            )
        candidates = ranked_candidates
        model_config = physics_runtime.model_config
        if model_config.calibration_pack_hash != source_summary["hashes"]["calibration_hash"]:
            raise Geometry19DeliveryError("Delivery calibration hash mismatch")
        source_physics_bundle_hash = source_summary["hashes"].get(
            "physics_bundle_hash"
        )
        if (
            source_physics_bundle_hash is not None
            and source_physics_bundle_hash != physics_runtime.physics_bundle_hash
        ):
            raise Geometry19DeliveryError("Delivery physics bundle mismatch")
        store = Geometry19DesignStore(
            root,
            mesh_storage,
            physics_runtime=physics_runtime,
        )
        delivered = []
        design_ids = set()
        ui_root = runtime / "ui_candidates"
        ui_root.mkdir()
        for rank, candidate in enumerate(candidates, start=1):
            design = Geometry19.from_mapping(candidate["geometry19"])
            response = store.regenerate(
                design,
                quality="preview_mesh",
            )
            design_id = str(response["design_id"])
            if design_id in design_ids:
                raise Geometry19DeliveryError("Delivery contains duplicate design_id")
            design_ids.add(design_id)
            if response["geometry19"] != design.as_mapping():
                raise Geometry19DeliveryError("UI and delivery Geometry19 mappings differ")
            if not response["mesh_diagnostics"].get("passed"):
                raise Geometry19DeliveryError("Preview mesh diagnostics did not pass")
            if not response["solver"].get("converged"):
                raise Geometry19DeliveryError(
                    "Delivery selected-profile recalculation did not converge"
                )
            if not response["constraints"].get("numerically_valid"):
                raise Geometry19DeliveryError("Delivery numerical hard constraints failed")
            physics_calibration_hash = response["physics"]["calibration_pack_hash"]
            if physics_calibration_hash != model_config.calibration_pack_hash:
                raise Geometry19DeliveryError("Recalculated physics calibration hash mismatch")
            if response.get("physics_bundle_hash") != physics_runtime.physics_bundle_hash:
                raise Geometry19DeliveryError("Recalculated physics bundle mismatch")
            visualization_manifest = json.loads(
                (
                    mesh_storage
                    / design_id
                    / "preview_mesh"
                    / "manifest.json"
                ).read_text(encoding="utf-8")
            )
            visualization_calibration_hash = visualization_manifest["calibration_hash"]
            if visualization_calibration_hash != response["calibration_hash"]:
                raise Geometry19DeliveryError("Visualization calibration provenance mismatch")
            recalculated = response["physics"]["performance"]
            if not performance_matches(candidate["performance"], recalculated):
                raise Geometry19DeliveryError(
                    "Delivery selected-profile recalculation changed performance"
                )
            errors = contract.diagnostic_acceptance.normalized_errors(target, recalculated)
            hit_alignment = metrics_from_relative_errors(
                errors, contract.diagnostic_acceptance.relative_tolerances
            )
            diagnostic_hit = contract.diagnostic_acceptance.is_diagnostic_hit(
                target,
                recalculated,
                solver_converged=True,
                numerical_hard_constraints_passed=True,
            )
            if bool(candidate["development_diagnostic_hit"]) != diagnostic_hit:
                raise Geometry19DeliveryError("Delivery recalculation changed diagnostic hit status")
            export_payload: dict[str, Any] | None = None
            if export_mesh:
                export_response = store.regenerate(
                    design,
                    quality="export_mesh",
                )
                if export_response["design_id"] != design_id:
                    raise Geometry19DeliveryError("Export mesh changed design_id")
                if export_response["geometry19"] != design.as_mapping():
                    raise Geometry19DeliveryError("Export mesh changed Geometry19 mapping")
                if not export_response["mesh_diagnostics"].get("passed"):
                    raise Geometry19DeliveryError("Export mesh diagnostics did not pass")
                if export_response["calibration_hash"] != visualization_calibration_hash:
                    raise Geometry19DeliveryError("Export mesh calibration provenance mismatch")
                export_manifest_path = (
                    mesh_storage / design_id / "export_mesh" / "manifest.json"
                )
                export_manifest = json.loads(
                    export_manifest_path.read_text(encoding="utf-8")
                )
                export_components = store.get_components(design_id, "export_mesh")
                if not export_components["components"]:
                    raise Geometry19DeliveryError("Export mesh produced no STL components")
                export_payload = {
                    "status": "GENERATED",
                    "quality": "export_mesh",
                    "directory": str(export_manifest_path.parent),
                    "manifest": export_manifest,
                    "components": export_components["components"],
                }
            ui_payload = {
                "delivery_rank": rank,
                "delivery_status": (
                    "DEVELOPMENT_DIAGNOSTIC_HIT"
                    if diagnostic_hit
                    else "BEST_APPROXIMATION_NOT_HIT"
                ),
                "formal_hit": None,
                "formal_use_allowed": formal_use_allowed,
                "simplified_model": simplified_model,
                "development_only": development_only,
                "target_hit": False,
                "target": target.as_mapping(),
                "source_evaluation_id": candidate["evaluation_id"],
                "source_design_hash": candidate["design_hash"],
                "source_candidate_disposition": candidate.get("candidate_disposition"),
                "constraint_dominance": candidate.get("constraint_dominance"),
                "design_id": design_id,
                "geometry19": design.as_mapping(),
                "performance": recalculated,
                "relative_errors": errors,
                "total_true_error": sum(errors.values()) / 3.0,
                "hit_alignment": hit_alignment,
                "root_classification": candidate["root_classification"],
                "flow_root_count": candidate["flow_root_count"],
                "selected_flow_root_index": candidate["selected_flow_root_index"],
                "constraint_assessment": response["constraints"],
                "mesh_quality": response["mesh_quality"],
                "mesh_diagnostics": response["mesh_diagnostics"],
                "component_urls": response["component_urls"],
                "geometry_hash": response["geometry_hash"],
                "lattice_hash": response["lattice_hash"],
                "overlay_parameters": response["overlay_parameters"],
                "simulation": response["simulation"],
                "schema_hash": response["schema_hash"],
                "calibration_hash": physics_calibration_hash,
                "physics_calibration_pack_hash": physics_calibration_hash,
                "visualization_calibration_hash": visualization_calibration_hash,
                "operating_point_policy": response["operating_point_policy"],
                "operating_point_policy_hash": response["operating_point_policy_hash"],
                "export_mesh_status": (
                    "GENERATED" if export_payload is not None else "ON_DEMAND_NOT_GENERATED"
                ),
                "export_mesh": export_payload,
            }
            ui_path = ui_root / f"{design_id}.json"
            _write_json(ui_path, ui_payload)
            reloaded = json.loads(ui_path.read_text(encoding="utf-8"))
            if (
                reloaded["design_id"] != design_id
                or reloaded["geometry19"] != design.as_mapping()
                or not performance_matches(reloaded["performance"], recalculated, relative_tolerance=0.0)
            ):
                raise Geometry19DeliveryError("UI-ready JSON round trip is not lossless")
            delivered.append(ui_payload)
        all_hits = all(row["delivery_status"] == "DEVELOPMENT_DIAGNOSTIC_HIT" for row in delivered)
        summary = {
            "work_package": "WP19-23F",
            "status": (
                "CLOSED_LOOP_DEVELOPMENT_DIAGNOSTIC_HIT_DELIVERED"
                if all_hits
                else "CLOSED_LOOP_BEST_APPROXIMATION_DELIVERED"
            ),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "target_id": target.target_id,
            "source_stage": source_stage,
            "formal_use_allowed": formal_use_allowed,
            "simplified_model": simplified_model,
            "development_only": development_only,
            "target_hit": False,
            "formal_hit": None,
            "network_architecture_modified": bool(network_architecture_modified),
            "export_mesh_requested": bool(export_mesh),
            "counts": {
                "delivered_count": len(delivered),
                "v2_1_recalculation_passed": len(delivered),
                "selected_physics_recalculation_passed": len(delivered),
                "preview_mesh_passed": len(delivered),
                "json_round_trip_passed": len(delivered),
                "ui_design_id_consistent": len(delivered),
                "export_mesh_generated": sum(
                    row["export_mesh_status"] == "GENERATED" for row in delivered
                ),
                "development_diagnostic_hits": sum(
                    row["delivery_status"] == "DEVELOPMENT_DIAGNOSTIC_HIT"
                    for row in delivered
                ),
            },
            "delivered_design_ids": [row["design_id"] for row in delivered],
            "hashes": {
                "protocol_hash": sha256_file(contract_path),
                "target_hash": sha256_file(target_source),
                "schema_hash": sha256_file(root / protocol["contracts"]["design_schema"]),
                "calibration_hash": model_config.calibration_pack_hash,
                "physics_backend_id": physics_runtime.backend_id,
                "physics_bundle_hash": physics_runtime.physics_bundle_hash,
                "constraint_dominance_hash": dominance_contract.source_hash,
                "source_summary_hash": sha256_file(source_summary_path),
                **(
                    {"wp19_23e_summary_hash": sha256_file(source_summary_path)}
                    if source_stage == "WP19-23E"
                    else {}
                ),
                "code_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
            },
        }
        _write_json(runtime / "target.json", target.as_mapping())
        _write_json(runtime / "delivered_candidates.json", delivered)
        _write_json(runtime / "summary.json", summary)
        _write_json(
            runtime / "manifest.json",
            {
                "protocol_version": protocol["protocol_version"],
                "target_id": target.target_id,
                "stage": "WP19-23F",
                "required_files": [
                    "manifest.json",
                    "target.json",
                    "delivered_candidates.json",
                    "summary.json",
                    "ui_candidates",
                ],
                "summary_sha256": sha256_file(runtime / "summary.json"),
            },
        )
        return summary
    except Exception as exc:
        _write_json(
            runtime / "failure.json",
            {"status": "FAILED", "error_type": type(exc).__name__, "message": str(exc)},
        )
        raise
