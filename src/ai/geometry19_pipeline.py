"""Conditional target-to-visualization orchestration for WP19-23.

WP19_23_PIPELINE_SYNC_REQUIRED: every new WP19-23 runtime capability must be
exposed through this orchestrator in the same change that introduces it.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import yaml

from src.ai.geometry19_candidates import run_candidate_protocol
from src.ai.geometry19_delivery import run_delivery_stage
from src.ai.geometry19_forward_model import (
    Geometry19ForwardModelError,
    resolve_forward_model_spec,
)
from src.ai.geometry19_inverse_contract import (
    DIAGNOSTIC_PROFILE,
    TARGET_CONTRACT_VERSION,
    Geometry19InverseContractError,
    Geometry19InverseTarget,
    WP1923Contract,
    derive_target_id,
    load_target,
)
from src.ai.geometry19_nsga2 import run_nsga2_stage
from src.ai.geometry19_pipeline_context import (
    PipelineContext,
    build_pipeline_context,
    resolve_pipeline_context,
)
from src.ai.geometry19_physics import run_true_physics_stage
from src.ai.geometry19_refinement import run_refinement_stage
from src.ai.geometry19_surrogate import sha256_file
from src.physics import Environment, Geometry19


PIPELINE_SYNC_MARKER = "WP19_23_PIPELINE_SYNC_REQUIRED"
RAW_PROVENANCE_VERSION = "wp19_23_raw_provenance_v2"
PIPELINE_MANIFEST_VERSION = "wp19_23_pipeline_manifest_v1"
PIPELINE_STAGE_ORDER = (
    "WP19-23B-single-generation-direct-or-dflow",
    "WP19-23C",
    "WP19-23D-conditional",
    "WP19-23E-conditional",
    "WP19-23F",
)
class Geometry19PipelineError(ValueError):
    """Raised when the target-to-visualization orchestrator is inconsistent."""


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _finite_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _shadow_physics_row(
    row: Mapping[str, Any],
    context: PipelineContext,
) -> dict[str, Any]:
    design = Geometry19.from_mapping(row["geometry19"])
    result = context.physics_runtime.evaluate(design, Environment())
    failure = result.solver.failure_code
    return {
        "candidate_id": int(row["candidate_id"]),
        "design_hash": str(row["design_hash"]),
        "geometry19": design.as_mapping(),
        "physics_backend_id": context.physics_runtime.backend_id,
        "physics_model_version": result.physics_model_version,
        "physics_bundle_hash": context.physics_runtime.physics_bundle_hash,
        "solver_converged": result.solver.converged,
        "solver_failure_code": failure.value if failure is not None else None,
        "solver_evaluations": result.solver.evaluations,
        "solver_iterations": result.solver.iterations,
        "flow_root_count": int(result.operating_point.get("flow_root_count", 0)),
        "selected_flow_root_mass_flow_air_kg_s": result.operating_point.get(
            "selected_flow_root_mass_flow_air_kg_s"
        ),
        "performance": _finite_json(dict(result.performance)),
        "inlet": _finite_json(dict(result.components.get("inlet", {}))),
        "compressor": _finite_json(dict(result.components.get("compressor", {}))),
        "combustor": _finite_json(dict(result.components.get("combustor", {}))),
        "turbine": _finite_json(dict(result.components.get("turbine", {}))),
        "nozzle": _finite_json(dict(result.components.get("nozzle", {}))),
        "conservation": _finite_json(asdict(result.conservation)),
    }


def _run_paired_physics_shadow(
    repository_root: Path,
    generation_root: Path,
    generation_summary: Mapping[str, Any],
    shadow_context: PipelineContext,
    output_root: Path,
) -> dict[str, Any]:
    """Evaluate one frozen candidate set with V2.1 and V2.2; never deliver."""

    shadow_contract = shadow_context.protocol.get("shadow")
    if (
        shadow_context.profile.mode != "paired_physics_shadow"
        or not isinstance(shadow_contract, Mapping)
        or shadow_contract.get("delivery_enabled") is not False
        or shadow_contract.get("formal_use_allowed") is not False
        or shadow_contract.get("enabled") is not True
        or shadow_contract.get("execute_stages")
        != ["B", "C_V2_1", "C_V2_2", "SHADOW_REPORT"]
        or shadow_contract.get("forbidden_stages") != ["D", "E", "F"]
    ):
        raise Geometry19PipelineError("Shadow profile safety contract is incomplete")
    baseline_context = build_pipeline_context(
        repository_root,
        pipeline_profile=str(shadow_contract["baseline_profile"]),
    )
    candidate_rows = _read_jsonl(generation_root / "candidate_filter_ledger.jsonl")
    by_id = {int(row["candidate_id"]): row for row in candidate_rows}
    kept_ids = [
        int(value)
        for value in generation_summary["candidate_protocol"]["kept_candidate_ids"]
    ]
    selected = [by_id[candidate_id] for candidate_id in kept_ids]
    if any(not row.get("valid") or not row.get("geometry19") for row in selected):
        raise Geometry19PipelineError("Shadow kept-candidate set contains invalid designs")

    baseline_rows = [_shadow_physics_row(row, baseline_context) for row in selected]
    enhanced_rows = [_shadow_physics_row(row, shadow_context) for row in selected]
    paired_rows = []
    for baseline, enhanced in zip(baseline_rows, enhanced_rows, strict=True):
        if (
            baseline["candidate_id"] != enhanced["candidate_id"]
            or baseline["design_hash"] != enhanced["design_hash"]
        ):
            raise Geometry19PipelineError("Shadow pair identity mismatch")
        performance_delta = {
            name: (
                float(enhanced["performance"][name])
                - float(baseline["performance"][name])
            )
            for name in sorted(set(baseline["performance"]) & set(enhanced["performance"]))
        }
        paired_rows.append(
            {
                "candidate_id": baseline["candidate_id"],
                "design_hash": baseline["design_hash"],
                "baseline": baseline,
                "enhanced": enhanced,
                "delta_enhanced_minus_baseline": {
                    "performance": performance_delta,
                    "flow_root_count": (
                        enhanced["flow_root_count"] - baseline["flow_root_count"]
                    ),
                    "selected_flow_root_mass_flow_air_kg_s": (
                        float(enhanced["selected_flow_root_mass_flow_air_kg_s"])
                        - float(baseline["selected_flow_root_mass_flow_air_kg_s"])
                        if enhanced["selected_flow_root_mass_flow_air_kg_s"] is not None
                        and baseline["selected_flow_root_mass_flow_air_kg_s"] is not None
                        else None
                    ),
                },
            }
        )

    _write_jsonl(output_root / "v2_1/physics_forward_ledger.jsonl", baseline_rows)
    _write_jsonl(output_root / "v2_2/physics_forward_ledger.jsonl", enhanced_rows)
    _write_jsonl(output_root / "paired_physics_ledger.jsonl", paired_rows)
    map_evaluated_rows = [
        row for row in enhanced_rows if "map_in_domain" in row["compressor"]
    ]
    map_in_domain = sum(
        bool(row["compressor"]["map_in_domain"]) for row in map_evaluated_rows
    )
    turbine_map_evaluated_rows = [
        row for row in enhanced_rows if "map_in_domain" in row["turbine"]
    ]
    turbine_map_in_domain = sum(
        bool(row["turbine"]["map_in_domain"])
        for row in turbine_map_evaluated_rows
    )
    turbine_thermo_rows = [
        row
        for row in turbine_map_evaluated_rows
        if "outlet_enthalpy_residual_J_kg" in row["turbine"]
    ]
    inlet_map_evaluated_rows = [
        row for row in enhanced_rows if "map_in_domain" in row["inlet"]
    ]
    inlet_map_in_domain = sum(
        bool(row["inlet"]["map_in_domain"]) for row in inlet_map_evaluated_rows
    )
    combustor_map_evaluated_rows = [
        row for row in enhanced_rows if "map_in_domain" in row["combustor"]
    ]
    combustor_map_in_domain = sum(
        bool(row["combustor"]["map_in_domain"])
        for row in combustor_map_evaluated_rows
    )
    nozzle_map_evaluated_rows = [
        row for row in enhanced_rows if "map_in_domain" in row["nozzle"]
    ]
    nozzle_map_in_domain = sum(
        bool(row["nozzle"]["map_in_domain"]) for row in nozzle_map_evaluated_rows
    )
    summary = {
        "work_package": "WP19-23-SHADOW",
        "status": "PAIRED_PHYSICS_SHADOW_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_use_allowed": False,
        "delivery_enabled": False,
        "paired_candidate_count": len(paired_rows),
        "baseline_converged_count": sum(row["solver_converged"] for row in baseline_rows),
        "enhanced_converged_count": sum(row["solver_converged"] for row in enhanced_rows),
        "enhanced_map_evaluated_count": len(map_evaluated_rows),
        "enhanced_map_in_domain_count": map_in_domain,
        "enhanced_map_out_of_domain_count": len(map_evaluated_rows) - map_in_domain,
        "enhanced_map_status_unavailable_count": (
            len(enhanced_rows) - len(map_evaluated_rows)
        ),
        "enhanced_turbine_map_evaluated_count": len(
            turbine_map_evaluated_rows
        ),
        "enhanced_turbine_map_in_domain_count": turbine_map_in_domain,
        "enhanced_turbine_map_out_of_domain_count": (
            len(turbine_map_evaluated_rows) - turbine_map_in_domain
        ),
        "enhanced_turbine_map_status_unavailable_count": (
            len(enhanced_rows) - len(turbine_map_evaluated_rows)
        ),
        "enhanced_turbine_thermo_closure_evaluated_count": len(
            turbine_thermo_rows
        ),
        "enhanced_turbine_max_abs_enthalpy_residual_J_kg": max(
            (
                abs(float(row["turbine"]["outlet_enthalpy_residual_J_kg"]))
                for row in turbine_thermo_rows
            ),
            default=None,
        ),
        "enhanced_turbine_max_abs_isentropic_efficiency_residual_J_kg": max(
            (
                abs(
                    float(
                        row["turbine"][
                            "isentropic_efficiency_residual_J_kg"
                        ]
                    )
                )
                for row in turbine_thermo_rows
            ),
            default=None,
        ),
        "enhanced_inlet_map_evaluated_count": len(inlet_map_evaluated_rows),
        "enhanced_inlet_map_in_domain_count": inlet_map_in_domain,
        "enhanced_inlet_map_out_of_domain_count": (
            len(inlet_map_evaluated_rows) - inlet_map_in_domain
        ),
        "enhanced_inlet_map_status_unavailable_count": (
            len(enhanced_rows) - len(inlet_map_evaluated_rows)
        ),
        "enhanced_combustor_map_evaluated_count": len(
            combustor_map_evaluated_rows
        ),
        "enhanced_combustor_map_in_domain_count": combustor_map_in_domain,
        "enhanced_combustor_map_out_of_domain_count": (
            len(combustor_map_evaluated_rows) - combustor_map_in_domain
        ),
        "enhanced_combustor_map_status_unavailable_count": (
            len(enhanced_rows) - len(combustor_map_evaluated_rows)
        ),
        "enhanced_nozzle_map_evaluated_count": len(nozzle_map_evaluated_rows),
        "enhanced_nozzle_map_in_domain_count": nozzle_map_in_domain,
        "enhanced_nozzle_map_out_of_domain_count": (
            len(nozzle_map_evaluated_rows) - nozzle_map_in_domain
        ),
        "enhanced_nozzle_map_status_unavailable_count": (
            len(enhanced_rows) - len(nozzle_map_evaluated_rows)
        ),
        "baseline_physics_bundle_hash": baseline_context.physics_runtime.physics_bundle_hash,
        "enhanced_physics_bundle_hash": shadow_context.physics_runtime.physics_bundle_hash,
        "thermo_model_hash": shadow_context.physics_runtime.provenance_hashes.get(
            "thermo_model_hash"
        ),
        "component_map_registry_hash": (
            shadow_context.physics_runtime.provenance_hashes.get(
                "component_map_registry_hash"
            )
        ),
        "compressor_map_hash": shadow_context.physics_runtime.provenance_hashes.get(
            "compressor_map_hash"
        ),
        "turbine_map_hash": shadow_context.physics_runtime.provenance_hashes.get(
            "turbine_map_hash"
        ),
        "inlet_map_hash": shadow_context.physics_runtime.provenance_hashes.get(
            "inlet_map_hash"
        ),
        "combustor_map_hash": shadow_context.physics_runtime.provenance_hashes.get(
            "combustor_map_hash"
        ),
        "nozzle_map_hash": shadow_context.physics_runtime.provenance_hashes.get(
            "nozzle_map_hash"
        ),
    }
    _write_json(output_root / "summary.json", summary)
    _write_json(
        output_root / "manifest.json",
        {
            "stage": "WP19-23-SHADOW",
            "target_id": generation_summary["target_id"],
            "required_files": [
                "manifest.json",
                "summary.json",
                "v2_1/physics_forward_ledger.jsonl",
                "v2_2/physics_forward_ledger.jsonl",
                "paired_physics_ledger.jsonl",
            ],
            "summary_sha256": sha256_file(output_root / "summary.json"),
        },
    )
    return summary


def _stage_artifact_entries(
    runtime: Path, stage_directory: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = stage_directory / "manifest.json"
    if not manifest_path.is_file():
        raise Geometry19PipelineError(f"Completed stage has no manifest: {stage_directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = manifest.get("required_files")
    if not isinstance(required, list) or not required:
        raise Geometry19PipelineError(f"Stage manifest has no required_files: {manifest_path}")
    artifacts: list[dict[str, Any]] = []
    for relative_value in required:
        relative = Path(str(relative_value))
        if relative.is_absolute() or ".." in relative.parts:
            raise Geometry19PipelineError(
                f"Unsafe required stage artifact path: {relative}"
            )
        source = stage_directory / relative
        if source.is_dir():
            files = sorted(path for path in source.rglob("*") if path.is_file())
            if not files:
                raise Geometry19PipelineError(f"Required stage directory is empty: {source}")
        elif source.is_file():
            files = [source]
        else:
            raise Geometry19PipelineError(f"Required stage artifact is missing: {source}")
        for path in files:
            artifacts.append(
                {
                    "path": str(path.relative_to(runtime)),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    artifacts.sort(key=lambda row: row["path"])
    summary_path = stage_directory / "summary.json"
    expected_summary_hash = manifest.get("summary_sha256")
    if expected_summary_hash != sha256_file(summary_path):
        raise Geometry19PipelineError(f"Stage summary hash mismatch: {stage_directory}")
    return manifest, artifacts


def _write_pipeline_manifest(
    runtime: Path,
    target_id: str,
    protocol: Mapping[str, Any],
    pipeline_context: PipelineContext,
    contract_path: Path,
    target_path: Path,
    pipeline_contract_path: Path,
    pipeline_summary_path: Path,
    stages: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    stage_records = []
    all_artifacts: list[dict[str, Any]] = []
    for stage_name, directory in stages:
        stage_manifest, artifacts = _stage_artifact_entries(runtime, directory)
        stage_summary = json.loads(
            (directory / "summary.json").read_text(encoding="utf-8")
        )
        if stage_manifest.get("target_id") != target_id:
            raise Geometry19PipelineError(f"Stage manifest target mismatch: {directory}")
        stage_records.append(
            {
                "stage": stage_name,
                "work_package": stage_summary.get("work_package"),
                "status": stage_summary.get("status"),
                "directory": str(directory.relative_to(runtime)),
                "manifest_sha256": sha256_file(directory / "manifest.json"),
                "summary_sha256": sha256_file(directory / "summary.json"),
                "provenance_hashes": dict(stage_summary.get("hashes") or {}),
                "artifact_count": len(artifacts),
                "artifacts": artifacts,
            }
        )
        all_artifacts.extend(artifacts)
    payload = {
        "manifest_version": PIPELINE_MANIFEST_VERSION,
        "protocol_version": protocol["protocol_version"],
        "target_id": target_id,
        "stage_order": [name for name, _ in stages],
        "stages": stage_records,
        "artifact_count": len(all_artifacts),
        "hashes": {
            "pipeline_profile_hash": pipeline_context.profile.profile_hash,
            "protocol_hash": sha256_file(contract_path),
            "pipeline_context_hash": pipeline_context.context_hash,
            "physics_bundle_hash": (
                pipeline_context.physics_runtime.physics_bundle_hash
            ),
            "target_hash": sha256_file(target_path),
            "pipeline_contract_hash": sha256_file(pipeline_contract_path),
            "pipeline_summary_hash": sha256_file(pipeline_summary_path),
            "constraint_dominance_hash": sha256_file(
                contract_path.parent.parent.parent
                / protocol["contracts"]["constraint_dominance"]
            ),
            "code_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=contract_path.parent.parent.parent,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        },
    }
    if pipeline_context.target_domain is not None:
        payload["hashes"]["target_domain_hash"] = (
            pipeline_context.target_domain.source_hash
        )
    _write_json(runtime / "pipeline_manifest.json", payload)
    return payload


def build_target_mapping(
    *,
    thrust_N: float,
    tsfc_kg_N_s: float,
    mass_kg: float,
    target_type: str = "ID",
    expected_status: str = "UNKNOWN",
) -> dict[str, Any]:
    """Build a canonical target document and derive its content identifier."""

    value = {
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "target_type": str(target_type),
        "expected_status": str(expected_status),
        "thrust_N": float(thrust_N),
        "tsfc_kg_N_s": float(tsfc_kg_N_s),
        "mass_kg": float(mass_kg),
        "acceptance_profile": DIAGNOSTIC_PROFILE,
    }
    value["target_id"] = derive_target_id(value)
    # Reuse the frozen validator rather than duplicating positive/finite checks.
    return Geometry19InverseTarget.from_mapping(value).as_mapping()


def _load_or_build_target(
    target_path: str | Path | None,
    target_values: Mapping[str, float] | None,
    *,
    target_type: str,
    expected_status: str,
) -> dict[str, Any]:
    if (target_path is None) == (target_values is None):
        raise Geometry19PipelineError(
            "Specify exactly one of target_path or target performance values"
        )
    if target_path is not None:
        return load_target(target_path).as_mapping()
    assert target_values is not None
    required = {"thrust_N", "tsfc_kg_N_s", "mass_kg"}
    if set(target_values) != required:
        raise Geometry19PipelineError(
            f"Target performance keys must be {sorted(required)}"
        )
    return build_target_mapping(
        thrust_N=float(target_values["thrust_N"]),
        tsfc_kg_N_s=float(target_values["tsfc_kg_N_s"]),
        mass_kg=float(target_values["mass_kg"]),
        target_type=target_type,
        expected_status=expected_status,
    )


def _read_completed_stage(directory: Path, work_package: str) -> dict[str, Any] | None:
    if not directory.exists():
        return None
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        raise Geometry19PipelineError(
            f"Stage directory exists without a completed summary: {directory}"
        )
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise Geometry19PipelineError(
            f"Stage directory exists without a manifest: {directory}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_files = manifest.get("required_files")
    if not isinstance(required_files, list) or not required_files:
        raise Geometry19PipelineError(f"Stage manifest is incomplete: {manifest_path}")
    for relative_value in required_files:
        relative = Path(str(relative_value))
        if relative.is_absolute() or ".." in relative.parts:
            raise Geometry19PipelineError(
                f"Unsafe required stage artifact path: {relative}"
            )
        if not (directory / relative).exists():
            raise Geometry19PipelineError(
                f"Required resumed-stage artifact is missing: {directory / relative}"
            )
    if manifest.get("summary_sha256") != sha256_file(summary_path):
        raise Geometry19PipelineError(f"Stage summary hash mismatch: {directory}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("work_package") != work_package:
        raise Geometry19PipelineError(f"Stage identity mismatch at {directory}")
    return summary


def _candidate_summary(directory: Path, mode: str) -> dict[str, Any] | None:
    summary = _read_completed_stage(directory, "WP19-23B")
    if summary is not None and summary.get("sampling_mode") != mode:
        raise Geometry19PipelineError(f"Candidate sampling mode mismatch at {directory}")
    return summary


def _prepare_delivery_restart(
    runtime: Path,
    generation_root: Path,
    physics_root: Path,
    refinement_root: Path,
    nsga2_root: Path,
    delivery_root: Path,
    sampling_mode: str,
) -> None:
    """Validate the completed upstream chain before explicitly restarting F."""

    generation = _candidate_summary(generation_root, sampling_mode)
    if generation is None:
        raise Geometry19PipelineError(
            "Cannot restart F before the candidate stage is complete"
        )
    if int(generation["candidate_protocol"]["n_valid"]) == 0:
        terminal = _read_completed_stage(nsga2_root, "WP19-23E")
        if terminal is None:
            raise Geometry19PipelineError(
                "Cannot restart F before cold-start WP19-23E is complete"
            )
    else:
        terminal = _read_completed_stage(physics_root, "WP19-23C")
        if terminal is None:
            raise Geometry19PipelineError(
                "Cannot restart F before WP19-23C is complete"
            )
        if not terminal.get("downstream_stop"):
            terminal = _read_completed_stage(refinement_root, "WP19-23D")
            if terminal is None:
                raise Geometry19PipelineError(
                    "Cannot restart F before WP19-23D is complete"
                )
            if not terminal.get("downstream_stop"):
                terminal = _read_completed_stage(nsga2_root, "WP19-23E")
                if terminal is None:
                    raise Geometry19PipelineError(
                        "Cannot restart F before WP19-23E is complete"
                    )
    if not terminal.get("downstream_stop"):
        raise Geometry19PipelineError(
            "Cannot restart F because the upstream terminal stage is not deliverable"
        )
    root_artifacts = tuple(
        runtime / name
        for name in (
            "pipeline_summary.json",
            "pipeline_manifest.json",
            "pipeline_failure.json",
        )
    )
    if delivery_root.is_symlink():
        raise Geometry19PipelineError("Refusing to restart a symlinked F stage")
    for path in root_artifacts:
        if path.is_symlink():
            raise Geometry19PipelineError(f"Refusing to replace symlinked artifact: {path}")
    if delivery_root.exists():
        shutil.rmtree(delivery_root)
    for path in root_artifacts:
        if path.exists():
            path.unlink()


def _resolve_generator_variant(
    root: Path,
    contract: WP1923Contract,
    model_variant: str | None,
) -> tuple[str, dict[str, Any], Path, Path]:
    resolved = contract.default_generator_variant if model_variant is None else str(model_variant)
    try:
        spec = contract.generator_variant(resolved)
    except Geometry19InverseContractError as exc:
        raise Geometry19PipelineError(f"Unsupported model variant: {resolved}") from exc
    configured = Path(spec["checkpoint"])
    checkpoint = configured if configured.is_absolute() else root / configured
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise Geometry19PipelineError(f"Generator checkpoint is missing: {checkpoint}")
    configured_normalizer = Path(spec["normalizer"])
    normalizer = (
        configured_normalizer
        if configured_normalizer.is_absolute()
        else root / configured_normalizer
    ).resolve()
    if not normalizer.is_file():
        raise Geometry19PipelineError(f"Generator normalizer is missing: {normalizer}")
    return resolved, spec, checkpoint, normalizer


def resolve_generator_checkpoint(
    repository_root: str | Path,
    model_variant: str | None = None,
    *,
    pipeline_profile: str | None = None,
) -> Path:
    """Resolve a generator checkpoint exclusively from the frozen protocol."""

    root = Path(repository_root).resolve()
    context = build_pipeline_context(root, pipeline_profile=pipeline_profile)
    return _resolve_generator_variant(root, context.contract, model_variant)[2]


def run_wp19_23_pipeline(
    repository_root: str | Path,
    *,
    target_path: str | Path | None = None,
    target_values: Mapping[str, float] | None = None,
    target_type: str = "ID",
    expected_status: str = "UNKNOWN",
    model_variant: str | None = None,
    forward_model_tag: str | None = None,
    sampling_mode: str | None = None,
    device: str = "cuda:0",
    refinement_device: str | None = None,
    output_root: str | Path | None = None,
    storage_root: str | Path | None = None,
    export_mesh: bool = False,
    resume: bool = False,
    restart_from_stage: str | None = None,
    pipeline_profile: str | None = None,
    pipeline_context: PipelineContext | None = None,
) -> dict[str, Any]:
    """Run WP19-23B→F with hit-triggered stopping and visualization delivery."""

    if restart_from_stage is not None:
        restart_from_stage = str(restart_from_stage).upper()
        if restart_from_stage != "F":
            raise Geometry19PipelineError("Only restart_from_stage='F' is supported")
        if resume:
            raise Geometry19PipelineError(
                "Use resume or restart_from_stage, not both"
            )
        resume = True
    root = Path(repository_root).resolve()
    pipeline_context = resolve_pipeline_context(
        root,
        pipeline_context,
        pipeline_profile=pipeline_profile,
    )
    contract_path = pipeline_context.contract_path
    contract = pipeline_context.contract
    protocol = pipeline_context.protocol
    model_variant, generator_spec, generator_checkpoint, generator_normalizer = (
        _resolve_generator_variant(root, contract, model_variant)
    )
    try:
        forward_spec = resolve_forward_model_spec(
            root, protocol, logical_tag=forward_model_tag
        )
    except Geometry19ForwardModelError as exc:
        raise Geometry19PipelineError(str(exc)) from exc
    forward_model_tag = forward_spec.logical_tag
    sampling_mode = (
        str(protocol["generation"]["default_sampling_mode"])
        if sampling_mode is None
        else str(sampling_mode)
    )
    target_mapping = _load_or_build_target(
        target_path,
        target_values,
        target_type=target_type,
        expected_status=expected_status,
    )
    target_domain_assessment = (
        pipeline_context.target_domain.assess(target_mapping).as_mapping()
        if pipeline_context.target_domain is not None
        else None
    )
    target_id = str(target_mapping["target_id"])
    refinement_runtime_device = refinement_device or device
    runtime = (
        Path(output_root).resolve()
        if output_root is not None
        else root
        / "data/inverse_design/geometry_lowdim_pipeline"
        / pipeline_context.profile.name
        / pipeline_context.physics_runtime.physics_bundle_hash[:12]
        / target_id
        / model_variant
        / forward_model_tag
    )
    if restart_from_stage is not None and not runtime.exists():
        raise Geometry19PipelineError(
            f"Cannot restart a pipeline run that does not exist: {runtime}"
        )
    if runtime.exists() and not resume:
        raise Geometry19PipelineError(f"Refusing to overwrite pipeline run: {runtime}")
    runtime.mkdir(parents=True, exist_ok=True)
    canonical_target_path = runtime / "target.yaml"
    if canonical_target_path.exists():
        existing = load_target(canonical_target_path).as_mapping()
        if existing != target_mapping:
            raise Geometry19PipelineError("Resume target differs from the stored target")
    else:
        _write_yaml(canonical_target_path, target_mapping)

    if sampling_mode not in protocol["generation"]["pipeline_sampling_modes"]:
        raise Geometry19PipelineError(
            "sampling_mode is not enabled by the pipeline contract"
        )
    is_paired_shadow = pipeline_context.profile.mode == "paired_physics_shadow"
    if is_paired_shadow and (sampling_mode != "direct" or export_mesh):
        raise Geometry19PipelineError(
            "Paired physics shadow requires direct sampling and forbids mesh export"
        )
    run_contract = {
        "pipeline_sync_marker": PIPELINE_SYNC_MARKER,
        "pipeline_stage_order": list(PIPELINE_STAGE_ORDER),
        "pipeline_profile": pipeline_context.profile.contract_mapping(),
        "profile_registry_version": pipeline_context.profile_registry_version,
        "target_id": target_id,
        "model_variant": model_variant,
        "sampling_mode": sampling_mode,
        "device": device,
        "refinement_device": refinement_runtime_device,
        "export_mesh": bool(export_mesh),
        "generator_checkpoint": str(generator_checkpoint),
        "generator_checkpoint_hash": sha256_file(generator_checkpoint),
        "generator_normalizer": str(generator_normalizer),
        "generator_normalizer_hash": sha256_file(generator_normalizer),
        "generator_model_tag": generator_spec["model_tag"],
        "forward_model": forward_spec.as_mapping(),
        "protocol_hash": sha256_file(contract_path),
        "candidate_filter_contract": {
            "discrete_policy": protocol["candidate_filter"]["discrete_policy"],
            "discrete_absolute_tolerance_count": protocol["candidate_filter"][
                "discrete_absolute_tolerance_count"
            ],
            "raw_provenance_version": RAW_PROVENANCE_VERSION,
            "dflow_per_candidate_metrics": True,
            "model_bundle_hashes": True,
            "selection": protocol["candidate_filter"]["select_before_physics"],
            "hit_alignment": dict(protocol["hit_alignment"]),
        },
        "constraint_dominance": {
            "contract_version": protocol["optimizer"]["constraint_dominance"],
            "sha256": sha256_file(root / protocol["contracts"]["constraint_dominance"]),
        },
        "pipeline_manifest_version": PIPELINE_MANIFEST_VERSION,
        "pipeline_context": pipeline_context.contract_mapping(),
        "target_domain_assessment": target_domain_assessment,
        "code_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }
    contract_output = runtime / "pipeline_contract.json"
    if contract_output.exists():
        existing_contract = json.loads(contract_output.read_text(encoding="utf-8"))
        if existing_contract != run_contract:
            raise Geometry19PipelineError("Resume arguments differ from pipeline_contract.json")
    else:
        _write_json(contract_output, run_contract)

    if (
        target_domain_assessment is not None
        and not target_domain_assessment["inside"]
    ):
        summary = {
            "work_package": "WP19-23-PIPELINE",
            "status": target_domain_assessment["status"],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "target_id": target_id,
            "model_variant": model_variant,
            "forward_model_tag": forward_model_tag,
            "sampling_mode": sampling_mode,
            "pipeline_profile": pipeline_context.profile.name,
            "pipeline_profile_mode": pipeline_context.profile.mode,
            "pipeline_profile_hash": pipeline_context.profile.profile_hash,
            "profile_registry_version": pipeline_context.profile_registry_version,
            "pipeline_context_hash": pipeline_context.context_hash,
            "physics_backend_id": pipeline_context.physics_runtime.backend_id,
            "physics_model_version": pipeline_context.physics_runtime.model_version,
            "physics_bundle_hash": pipeline_context.physics_runtime.physics_bundle_hash,
            "simplified_model": True,
            "development_only": True,
            "formal_use_allowed": False,
            "target_hit": False,
            "target_domain_assessment": target_domain_assessment,
            "completed_stages": [],
            "terminal_source_stage": "TARGET_DOMAIN_PREFLIGHT",
            "terminal_source_status": target_domain_assessment["status"],
            "delivery_status": "NOT_RUN_OUTSIDE_TARGET_DOMAIN",
            "delivered_design_ids": [],
            "paths": {
                "runtime_root": str(runtime),
                "target": str(canonical_target_path),
                "pipeline_manifest": str(runtime / "pipeline_manifest.json"),
            },
            "pipeline_contract_hash": sha256_file(contract_output),
        }
        pipeline_summary_path = runtime / "pipeline_summary.json"
        _write_json(pipeline_summary_path, summary)
        _write_pipeline_manifest(
            runtime,
            target_id,
            protocol,
            pipeline_context,
            contract_path,
            canonical_target_path,
            contract_output,
            pipeline_summary_path,
            (),
        )
        return summary

    candidate_root = runtime / "wp19_23b"
    generation_root = candidate_root / sampling_mode
    physics_root = runtime / "wp19_23c"
    refinement_root = runtime / "wp19_23d"
    nsga2_root = runtime / "wp19_23e"
    delivery_root = runtime / "wp19_23f"
    if restart_from_stage == "F":
        _prepare_delivery_restart(
            runtime,
            generation_root,
            physics_root,
            refinement_root,
            nsga2_root,
            delivery_root,
            sampling_mode,
        )
    completed: list[str] = []
    try:
        generation = _candidate_summary(generation_root, sampling_mode) if resume else None
        if generation is None:
            generation = run_candidate_protocol(
                root,
                canonical_target_path,
                mode=sampling_mode,
                device=device,
                output_root=generation_root,
                generator_checkpoint=generator_checkpoint,
                forward_model_tag=forward_model_tag,
                pipeline_context=pipeline_context,
            )
        actual_generator_tag = (
            generation.get("model_hashes", {}).get("generator", {}).get("model_tag")
        )
        if (
            actual_generator_tag is not None
            and actual_generator_tag != generator_spec["model_tag"]
        ):
            raise Geometry19PipelineError(
                "Generator checkpoint model_tag differs from the selected contract variant"
            )
        actual_forward_tag = (
            generation.get("model_hashes", {}).get("forward", {}).get("model_tag")
            or generation.get("model_hashes", {}).get("ranking_forward", {}).get("model_tag")
        )
        if actual_forward_tag is not None and actual_forward_tag != forward_spec.model_tag:
            raise Geometry19PipelineError(
                "Forward checkpoint model_tag differs from the selected contract tag"
            )
        completed.append(f"WP19-23B-{sampling_mode}")

        if is_paired_shadow:
            shadow_root = runtime / "shadow"
            shadow = (
                _read_completed_stage(shadow_root, "WP19-23-SHADOW")
                if resume
                else None
            )
            if shadow is None:
                shadow = _run_paired_physics_shadow(
                    root,
                    generation_root,
                    generation,
                    pipeline_context,
                    shadow_root,
                )
            completed.append("WP19-23-SHADOW")
            summary = {
                "work_package": "WP19-23-PIPELINE",
                "status": "PAIRED_PHYSICS_SHADOW_COMPLETE",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "target_id": target_id,
                "model_variant": model_variant,
                "forward_model_tag": forward_model_tag,
                "sampling_mode": sampling_mode,
                "pipeline_profile": pipeline_context.profile.name,
                "pipeline_profile_mode": pipeline_context.profile.mode,
                "pipeline_profile_hash": pipeline_context.profile.profile_hash,
                "profile_registry_version": pipeline_context.profile_registry_version,
                "pipeline_context_hash": pipeline_context.context_hash,
                "physics_backend_id": pipeline_context.physics_runtime.backend_id,
                "physics_model_version": pipeline_context.physics_runtime.model_version,
                "physics_bundle_hash": pipeline_context.physics_runtime.physics_bundle_hash,
                "formal_use_allowed": False,
                "target_domain_assessment": target_domain_assessment,
                "delivery_enabled": False,
                "completed_stages": completed,
                "terminal_source_stage": shadow["work_package"],
                "terminal_source_status": shadow["status"],
                "delivery_status": "DISABLED_FOR_SHADOW",
                "delivered_design_ids": [],
                "shadow": shadow,
                "paths": {
                    "runtime_root": str(runtime),
                    "target": str(canonical_target_path),
                    "shadow": str(shadow_root),
                    "pipeline_manifest": str(runtime / "pipeline_manifest.json"),
                },
                "pipeline_contract_hash": sha256_file(contract_output),
            }
            pipeline_summary_path = runtime / "pipeline_summary.json"
            _write_json(pipeline_summary_path, summary)
            _write_pipeline_manifest(
                runtime,
                target_id,
                protocol,
                pipeline_context,
                contract_path,
                canonical_target_path,
                contract_output,
                pipeline_summary_path,
                (
                    (f"WP19-23B-{sampling_mode}", generation_root),
                    ("WP19-23-SHADOW", shadow_root),
                ),
            )
            return summary

        no_valid_candidates = int(generation["candidate_protocol"]["n_valid"]) == 0
        if no_valid_candidates:
            nsga2 = _read_completed_stage(nsga2_root, "WP19-23E") if resume else None
            if nsga2 is None:
                nsga2 = run_nsga2_stage(
                    root,
                    canonical_target_path,
                    candidate_root=generation_root,
                    sampling_mode=sampling_mode,
                    cold_start=True,
                    output_root=nsga2_root,
                    pipeline_context=pipeline_context,
                )
            completed.append("WP19-23E-cold-start")
            terminal_source = nsga2
            delivery_source = nsga2_root
        else:
            physics = _read_completed_stage(physics_root, "WP19-23C") if resume else None
            if physics is None:
                physics = run_true_physics_stage(
                    root,
                    canonical_target_path,
                    candidate_root=candidate_root,
                    output_root=physics_root,
                    sampling_mode=sampling_mode,
                    pipeline_context=pipeline_context,
                )
            completed.append("WP19-23C")

            delivery_source = physics_root
            terminal_source = physics
            if not physics["downstream_stop"]:
                refinement = (
                    _read_completed_stage(refinement_root, "WP19-23D") if resume else None
                )
                if refinement is None:
                    refinement = run_refinement_stage(
                        root,
                        canonical_target_path,
                        physics_root=physics_root,
                        output_root=refinement_root,
                        device=refinement_runtime_device,
                        forward_model_tag=forward_model_tag,
                        pipeline_context=pipeline_context,
                    )
                completed.append("WP19-23D")
                terminal_source = refinement
                delivery_source = refinement_root
                if not refinement["downstream_stop"]:
                    nsga2 = _read_completed_stage(nsga2_root, "WP19-23E") if resume else None
                    if nsga2 is None:
                        nsga2 = run_nsga2_stage(
                            root,
                            canonical_target_path,
                            physics_root=physics_root,
                            refinement_root=refinement_root,
                            output_root=nsga2_root,
                            pipeline_context=pipeline_context,
                        )
                    completed.append("WP19-23E")
                    terminal_source = nsga2
                    delivery_source = nsga2_root

        delivery = _read_completed_stage(delivery_root, "WP19-23F") if resume else None
        if delivery is None:
            delivery = run_delivery_stage(
                root,
                canonical_target_path,
                source_root=delivery_source,
                output_root=delivery_root,
                storage_root=storage_root,
                network_architecture_modified=bool(
                    generator_spec["network_architecture_modified"]
                ),
                export_mesh=export_mesh,
                pipeline_context=pipeline_context,
            )
        completed.append("WP19-23F")
        summary = {
            "work_package": "WP19-23-PIPELINE",
            "status": "PASS",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "target_id": target_id,
            "model_variant": model_variant,
            "forward_model_tag": forward_model_tag,
            "forward_model_model_tag": forward_spec.model_tag,
            "sampling_mode": sampling_mode,
            "pipeline_profile": pipeline_context.profile.name,
            "pipeline_profile_mode": pipeline_context.profile.mode,
            "pipeline_profile_hash": pipeline_context.profile.profile_hash,
            "profile_registry_version": pipeline_context.profile_registry_version,
            "pipeline_context_hash": pipeline_context.context_hash,
            "physics_backend_id": pipeline_context.physics_runtime.backend_id,
            "physics_model_version": pipeline_context.physics_runtime.model_version,
            "physics_bundle_hash": (
                pipeline_context.physics_runtime.physics_bundle_hash
            ),
            "simplified_model": bool(
                protocol.get("delivery", {}).get("simplified_model", False)
            ),
            "development_only": bool(
                protocol.get("delivery", {}).get(
                    "development_only", not protocol["formal_use_allowed"]
                )
            ),
            "formal_use_allowed": bool(protocol["formal_use_allowed"]),
            "target_hit": False,
            "target_domain_assessment": target_domain_assessment,
            "candidate_filter_contract": run_contract["candidate_filter_contract"],
            "export_mesh": bool(export_mesh),
            "completed_stages": completed,
            "terminal_source_stage": terminal_source["work_package"],
            "terminal_source_status": terminal_source["status"],
            "delivery_status": delivery["status"],
            "delivered_design_ids": delivery["delivered_design_ids"],
            "paths": {
                "runtime_root": str(runtime),
                "target": str(canonical_target_path),
                "delivery": str(delivery_root),
                "pipeline_manifest": str(runtime / "pipeline_manifest.json"),
                "mesh_storage": str(
                    Path(storage_root).resolve()
                    if storage_root is not None
                    else (root / protocol["artifact_protocol"]["mesh_storage_root"]).resolve()
                ),
            },
            "pipeline_contract_hash": sha256_file(contract_output),
        }
        pipeline_summary_path = runtime / "pipeline_summary.json"
        _write_json(pipeline_summary_path, summary)
        stage_directory_by_name = {
            f"WP19-23B-{sampling_mode}": generation_root,
            "WP19-23C": physics_root,
            "WP19-23D": refinement_root,
            "WP19-23E": nsga2_root,
            "WP19-23E-cold-start": nsga2_root,
            "WP19-23F": delivery_root,
        }
        aggregate_stages = [
            (name, stage_directory_by_name[name]) for name in completed
        ]
        _write_pipeline_manifest(
            runtime,
            target_id,
            protocol,
            pipeline_context,
            contract_path,
            canonical_target_path,
            contract_output,
            pipeline_summary_path,
            aggregate_stages,
        )
        return summary
    except Exception as exc:
        _write_json(
            runtime / "pipeline_failure.json",
            {
                "status": "FAILED",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "target_id": target_id,
                "pipeline_profile": pipeline_context.profile.name,
                "pipeline_context_hash": pipeline_context.context_hash,
                "completed_stages": completed,
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise
