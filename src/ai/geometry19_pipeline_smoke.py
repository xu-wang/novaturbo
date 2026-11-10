"""Executable acceptance checks for the simplified V2.2 B--F pipeline smoke."""

from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

from src.ai.geometry19_pipeline import run_wp19_23_pipeline


class Geometry19PipelineSmokeError(ValueError):
    """Raised when a V2.2 software smoke artifact violates its contract."""


EXPECTED_STAGES = (
    "WP19-23B-direct",
    "WP19-23C",
    "WP19-23D",
    "WP19-23E",
    "WP19-23F",
)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise Geometry19PipelineSmokeError(f"Required smoke artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_development_markers(payload: dict[str, Any], label: str) -> None:
    expected = {
        "simplified_model": True,
        "development_only": True,
        "formal_use_allowed": False,
        "target_hit": False,
    }
    actual = {name: payload.get(name) for name in expected}
    if actual != expected:
        raise Geometry19PipelineSmokeError(
            f"{label} development markers differ: {actual}"
        )


def validate_v2_2_development_smoke(
    runtime_root: str | Path,
    *,
    expected_raw_candidates: int = 256,
    elapsed_seconds: float | None = None,
    maximum_runtime_seconds: float | None = None,
) -> dict[str, Any]:
    """Validate B--F identity, recalculation, JSON, mesh, and time-budget evidence."""

    runtime = Path(runtime_root).resolve()
    summary = _read_json(runtime / "pipeline_summary.json")
    manifest = _read_json(runtime / "pipeline_manifest.json")
    generation = _read_json(runtime / "wp19_23b/direct/summary.json")
    delivery = _read_json(runtime / "wp19_23f/summary.json")
    candidates = _read_json(runtime / "wp19_23f/delivered_candidates.json")

    if summary.get("status") != "PASS":
        raise Geometry19PipelineSmokeError("Pipeline summary status is not PASS")
    if summary.get("pipeline_profile") != "v2_2_development":
        raise Geometry19PipelineSmokeError("Smoke did not use v2_2_development")
    if summary.get("pipeline_profile_mode") != "production_development":
        raise Geometry19PipelineSmokeError("Smoke profile mode is not development")
    if tuple(summary.get("completed_stages", ())) != EXPECTED_STAGES:
        raise Geometry19PipelineSmokeError("Smoke did not complete the fixed B--F chain")
    if tuple(manifest.get("stage_order", ())) != EXPECTED_STAGES:
        raise Geometry19PipelineSmokeError("Aggregate manifest stage order differs")
    summary_hash = _sha256_file(runtime / "pipeline_summary.json")
    if manifest.get("hashes", {}).get("pipeline_summary_hash") != summary_hash:
        raise Geometry19PipelineSmokeError("Aggregate manifest summary hash differs")
    for name in (
        "pipeline_profile_hash",
        "pipeline_context_hash",
        "physics_bundle_hash",
    ):
        value = summary.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise Geometry19PipelineSmokeError(f"Smoke {name} is invalid")
        if manifest.get("hashes", {}).get(name) != value:
            raise Geometry19PipelineSmokeError(f"Aggregate manifest {name} differs")
    stage_records = manifest.get("stages")
    if not isinstance(stage_records, list) or len(stage_records) != len(EXPECTED_STAGES):
        raise Geometry19PipelineSmokeError("Aggregate manifest stage records differ")
    for expected_name, record in zip(EXPECTED_STAGES, stage_records, strict=True):
        if record.get("stage") != expected_name:
            raise Geometry19PipelineSmokeError("Aggregate manifest stage identity differs")
        stage_root = runtime / str(record.get("directory", ""))
        if record.get("summary_sha256") != _sha256_file(stage_root / "summary.json"):
            raise Geometry19PipelineSmokeError(
                f"Aggregate manifest stage summary hash differs: {expected_name}"
            )
        if record.get("manifest_sha256") != _sha256_file(stage_root / "manifest.json"):
            raise Geometry19PipelineSmokeError(
                f"Aggregate manifest stage manifest hash differs: {expected_name}"
            )
    _require_development_markers(summary, "pipeline summary")
    _require_development_markers(delivery, "delivery summary")

    observed_raw = int(generation["candidate_protocol"]["n_raw"])
    if observed_raw != int(expected_raw_candidates):
        raise Geometry19PipelineSmokeError(
            f"Raw candidate budget differs: {observed_raw} != {expected_raw_candidates}"
        )
    delivered_count = int(delivery["counts"]["delivered_count"])
    if delivered_count < 1 or len(candidates) != delivered_count:
        raise Geometry19PipelineSmokeError("Smoke delivered-candidate count is invalid")
    for count_name in (
        "selected_physics_recalculation_passed",
        "preview_mesh_passed",
        "json_round_trip_passed",
        "ui_design_id_consistent",
    ):
        if int(delivery["counts"].get(count_name, -1)) != delivered_count:
            raise Geometry19PipelineSmokeError(
                f"Delivery closure count failed: {count_name}"
            )
    for index, candidate in enumerate(candidates):
        _require_development_markers(candidate, f"UI candidate {index}")
        ui_path = runtime / "wp19_23f/ui_candidates" / f"{candidate['design_id']}.json"
        _require_development_markers(_read_json(ui_path), f"UI JSON {index}")

    if elapsed_seconds is not None and maximum_runtime_seconds is not None:
        if elapsed_seconds > maximum_runtime_seconds:
            raise Geometry19PipelineSmokeError(
                "Smoke exceeded CPU runtime budget: "
                f"{elapsed_seconds:.3f}s > {maximum_runtime_seconds:.3f}s"
            )
    return {
        "status": "SMOKE_PASS",
        "pipeline_profile": "v2_2_development",
        "runtime_root": str(runtime),
        "raw_candidate_count": observed_raw,
        "delivered_count": delivered_count,
        "completed_stages": list(EXPECTED_STAGES),
        "elapsed_seconds": elapsed_seconds,
        "maximum_runtime_seconds": maximum_runtime_seconds,
        "simplified_model": True,
        "development_only": True,
        "formal_use_allowed": False,
        "target_hit": False,
    }


def run_v2_2_development_smoke(
    repository_root: str | Path,
    target_path: str | Path,
    output_root: str | Path,
    *,
    maximum_runtime_seconds: float = 120.0,
) -> dict[str, Any]:
    """Run the fixed CPU/direct simplified-V2.2 smoke and validate its artifacts."""

    if not math.isfinite(maximum_runtime_seconds) or maximum_runtime_seconds <= 0.0:
        raise Geometry19PipelineSmokeError(
            "maximum_runtime_seconds must be finite and positive"
        )
    started = perf_counter()
    run_wp19_23_pipeline(
        repository_root,
        target_path=target_path,
        pipeline_profile="v2_2_development",
        model_variant="baseline",
        sampling_mode="direct",
        device="cpu",
        refinement_device="cpu",
        output_root=output_root,
    )
    elapsed = perf_counter() - started
    return validate_v2_2_development_smoke(
        output_root,
        elapsed_seconds=elapsed,
        maximum_runtime_seconds=maximum_runtime_seconds,
    )
