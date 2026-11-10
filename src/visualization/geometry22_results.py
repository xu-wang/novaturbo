"""Read verified V3 results and publish separately identified 22D meshes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from src.ai.geometry19_pipeline_artifact import verify_completed_artifact
from src.physics.geometry22_design import Geometry22Design, geometry22_schema_metadata
from src.visualization.geometry22_mesh import Geometry22MeshStore


class Geometry22ResultError(ValueError):
    """A V3 result or mesh preview fails integrity or identity checks."""


def require(condition, message):
    if not condition:
        raise Geometry22ResultError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


class Geometry22Results:
    def __init__(self, repository_root, results_root, preview_root):
        self.repository = Path(repository_root).resolve()
        self.results = Path(results_root).resolve()
        self.previews = Path(preview_root).resolve()

    def runtime(self, result_id):
        require(isinstance(result_id, str) and re.fullmatch(r"[0-9a-f]{64}", result_id), "Invalid V3 result ID")
        path = (self.results / result_id / "run").resolve()
        require(path.is_relative_to(self.results), "V3 result path escapes configured root")
        if not (path / "Complete_Manifest.json").is_file():
            raise FileNotFoundError("Completed V3 result not found")
        return path

    def list_results(self):
        items = []
        if not self.results.is_dir():
            return items
        for path in sorted(self.results.iterdir()):
            if not re.fullmatch(r"[0-9a-f]{64}", path.name) or not path.is_dir():
                continue
            try:
                data = self.result(path.name)
                items.append({key: data[key] for key in ("result_id", "target_id", "delivery_status", "software_hit_5pct", "design_id", "route_status")}
                             | {"integrity_pass": True})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                items.append({"result_id": path.name, "integrity_pass": False, "error": str(exc)})
        return items

    def result(self, result_id):
        runtime = self.runtime(result_id)
        verified = verify_completed_artifact(runtime, self.repository)
        summary = verified["summary"]
        require(summary.get("model") == "model V3" and summary.get("pipeline_profile") in {"v3_routed_development", "v3_local_cfm_development", "v3_guided_cfm_development"},
                "Result is not the trained V3 routed pipeline")
        lock = read(runtime / "Protocol_Lock.json")["identity"]
        target = {n: lock["target"][n] for n in ("thrust_N", "tsfc_kg_N_s", "mass_kg")}
        f = read(runtime / "F/summary.json")
        selected = f["independent_recompute"]
        payload = {"result_id": result_id, "model": "model V3", "pipeline_profile": summary["pipeline_profile"], "target_id": summary["target_id"], "target_performance": target,
                   "status": summary["status"], "delivery_status": summary["delivery_status"],
                   "route_status": summary["route_status"], "eligible_center_count": summary["eligible_center_count"],
                   "software_hit_5pct": summary["software_hit_5pct"], "candidate_count": summary["candidate_count"],
                   "physics_evaluation_count": summary["physics_evaluation_count"],
                   "design_id": None, "geometry22": None, "selected": selected, "relative_errors": None,
                   "f_recompute_equal": selected is not None, "integrity_pass": True,
                   "current_code_identity_matches_lock": verified["current_code_identity_matches_lock"],
                   "source_complete_sha256": sha(runtime / "Complete_Manifest.json"),
                   "design_schema_hash": summary["design_schema_hash"], "physics_bundle_hash": summary["physics_bundle_hash"],
                   "physics_validation_status": summary["physics_validation_status"],
                   "formal_use_allowed": summary["formal_use_allowed"], "deployment_allowed": summary["deployment_allowed"],
                   "preview_available": selected is not None, "mesh_scope": "22d_design_driven_visual_mesh",
                   "schema": geometry22_schema_metadata()}
        if selected is not None:
            design = Geometry22Design.from_mapping(selected["design"])
            require(design.design_id() == selected["design_id"], "V3 selected 22D identity changed")
            payload.update(design_id=design.design_id(), geometry22=design.as_mapping(),
                           component_parameters=design.component_parameters(), geometry19_projection_hash=design.geometry19_projection_hash(),
                           relative_errors={n: selected["performance"][n] / target[n] - 1 for n in target})
        return payload

    def preview(self, result_id, *, generate=False, expected_preview_id=None):
        payload = self.result(result_id)
        require(payload["preview_available"], "V3 result has no selected diagnostic design to preview")
        design = Geometry22Design.from_mapping(payload["geometry22"])
        store = Geometry22MeshStore(self.repository, self.previews / "mesh22")
        if generate:
            manifest, directory = store.publish(design)
        else:
            require(expected_preview_id is not None, "V3 preview ID is required")
            manifest, directory = store.get(expected_preview_id, design=design, quality="preview_mesh")
        preview_id = manifest["mesh_id"]
        require(expected_preview_id is None or expected_preview_id == preview_id, "V3 preview does not match current result identity")
        payload.update(store.response(manifest, f"/geometry22-preview-files/{result_id}"))
        payload["preview_id"] = preview_id
        return payload, directory, manifest
