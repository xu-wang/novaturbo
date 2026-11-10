"""Reusable, resumable WP19-21 development-dataset generator."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import yaml

from src.physics import ConstraintProfile, Environment, Geometry19, ModelConfig
from src.physics.sanity_doe_v2 import sample_geometry19_designs
from src.physics.wp19_19_doe import (
    _mesh_worker,
    _solve_record,
    _status_stratified_ids,
)


class WP1921Error(ValueError):
    """Raised when the dataset protocol or resume contract is invalid."""


def _load_yaml(path: Path) -> tuple[dict[str, Any], str]:
    source = path.read_bytes()
    payload = yaml.safe_load(source)
    if not isinstance(payload, dict):
        raise WP1921Error(f"YAML must contain a mapping: {path}")
    return payload, sha256(source).hexdigest()


def _canonical_hash(payload: Any) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, records: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _solve_shard_worker(
    shard: list[tuple[int, Mapping[str, Any]]],
    calibration_path: str,
    profile_path: str,
    tolerances_path: str,
    environment_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    config = ModelConfig.from_yaml(calibration_path)
    profile = ConstraintProfile.from_yaml(profile_path)
    tolerances = yaml.safe_load(Path(tolerances_path).read_text(encoding="utf-8"))
    environment = Environment(**environment_payload)
    return [
        _solve_record(
            sample_id,
            Geometry19.from_mapping(design),
            environment,
            config,
            profile,
            tolerances,
        )
        for sample_id, design in shard
    ]


def _valid_shard(path: Path, expected_ids: list[int]) -> bool:
    try:
        records = _read_jsonl(path)
    except (OSError, json.JSONDecodeError):
        return False
    return [int(record.get("sample_id", -1)) for record in records] == expected_ids


def _normalize_record(
    record: Mapping[str, Any],
    environment: Mapping[str, Any],
    provenance: Mapping[str, Any],
    preview: Mapping[str, Any] | None,
    preview_requested: bool,
) -> dict[str, Any]:
    physics = record["physics"]
    return {
        "sample_id": int(record["sample_id"]),
        "x_19": record["geometry19"],
        "control_3": {
            "ambient_temperature_K": environment["temperature_K"],
            "ambient_pressure_Pa": environment["pressure_Pa"],
            "flight_mach": environment["mach"],
        },
        "derived_geometry": record["derived_geometry"],
        "geometry_mapping": {
            "valid": record["geometry_mapping_valid"],
            "error": record["geometry_mapping_error"],
            "design_id": record["design_id"],
            "solid_geometry_hash": record["geometry_hash"],
            "lattice_hash": record["lattice_hash"],
        },
        "physics_outputs": {
            "performance": physics.get("performance", {}),
            "operating_point": physics.get("operating_point", {}),
            "stations": physics.get("stations", {}),
            "components": physics.get("components", {}),
            "mass_breakdown_kg": physics.get("mass_breakdown_kg", {}),
            "formal_diagnostics": physics.get("formal_diagnostics", {}),
            "conservation": physics.get("conservation", {}),
        },
        "constraint_values": {
            "all": physics.get("constraints", {}),
            "thermodynamic": physics.get("thermodynamic_constraints", {}),
            "assembly": physics.get("assembly_constraints", {}),
            "structural": physics.get("structural_constraints", {}),
            "thermal": physics.get("thermal_constraints", {}),
            "manufacturing": physics.get("manufacturing_constraints", {}),
            "assessment": record["constraint_assessment"],
        },
        "solver_status": {
            **physics["solver"],
            "classification": record["classification"],
            "conservation_passed": record["conservation"]["passed"],
        },
        "mesh_preview_reference": {
            "requested": preview_requested,
            "available": bool(preview and preview.get("passed")),
            "quality": "preview_mesh" if preview_requested else None,
            "design_id": preview.get("design_id") if preview else record["design_id"],
            "storage_policy": "hash_addressed_on_demand",
        },
        "provenance": dict(provenance),
    }


def _render_report(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    return "\n".join(
        [
            "# WP19-21 开发数据集生成",
            "",
            f"> 状态：**{report['status']}**",
            f"> 数据级别：`{report['dataset_tier']}`；正式用途允许：`{str(report['formal_use_allowed']).lower()}`",
            "",
            "## 结果",
            "",
            f"- 样本：`{summary['sample_count']}`；分片：`{summary['shard_count']}`。",
            f"- 收敛：`{summary['converged_count']}`；几何映射有效：`{summary['geometry_mapping_valid_count']}`。",
            f"- preview：`{summary['preview_passed_count']}/{summary['preview_requested_count']}`。",
            f"- 合并数据：`{report['artifacts']['dataset_jsonl']}`。",
            "",
            "## 使用限制",
            "",
            "当前 schema/calibration 仍为 `formal_dataset_generation_ready=false`。本产物仅用于开发、管线验证和后续代理模型冒烟，不得标记为正式标定数据集。",
            "",
        ]
    )


def run_wp19_21(
    protocol_path: str | Path,
    repository_root: str | Path,
    *,
    output_root_override: str | Path | None = None,
    sample_count_override: int | None = None,
    preview_budget_override: int | None = None,
    workers_override: int | None = None,
    shard_size_override: int | None = None,
    seed_override: int | None = None,
    campaign_id: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    protocol_source = Path(protocol_path).resolve()
    protocol, protocol_hash = _load_yaml(protocol_source)
    if protocol.get("dataset_tier") != "development_bootstrap":
        raise WP1921Error("Only development_bootstrap generation is currently supported")
    if protocol.get("formal_use_allowed") is not False:
        raise WP1921Error("Bootstrap calibration cannot authorize formal dataset use")
    paths = {
        name: root / protocol[name]
        for name in (
            "design_schema", "calibration_pack", "constraint_profile",
            "operating_point_policy", "visualization_calibration",
            "conservation_tolerances",
        )
    }
    loaded = {name: _load_yaml(path) for name, path in paths.items()}
    config = ModelConfig.from_yaml(paths["calibration_pack"])
    schema = loaded["design_schema"][0]
    if config.formal_dataset_generation_ready or schema.get("formal_dataset_generation_ready"):
        raise WP1921Error("Development-bootstrap protocol conflicts with a formal-ready config")
    sample_count = int(
        protocol["sample_count"]
        if sample_count_override is None else sample_count_override
    )
    preview_budget = int(
        protocol["mesh_preview"]["budget"]
        if preview_budget_override is None else preview_budget_override
    )
    workers = int(protocol["workers"] if workers_override is None else workers_override)
    shard_size = int(
        protocol["shard_size"] if shard_size_override is None else shard_size_override
    )
    seed = int(protocol["seed"] if seed_override is None else seed_override)
    if not (sample_count > 0 and 0 <= preview_budget <= sample_count and workers > 0 and shard_size > 0):
        raise WP1921Error("Invalid sample, preview, worker, or shard configuration")
    runtime_root = (
        Path(output_root_override).resolve()
        if output_root_override is not None
        else root / protocol["artifacts"]["runtime_root"]
    )
    contract = {
        "protocol_hash": protocol_hash,
        "sample_count": sample_count,
        "preview_budget": preview_budget,
        "workers": workers,
        "shard_size": shard_size,
        "seed": seed,
        "source_hashes": {name: value[1] for name, value in loaded.items()},
    }
    if campaign_id is not None:
        contract["campaign_id"] = campaign_id
    contract["contract_hash"] = _canonical_hash(contract)
    contract_path = runtime_root / "resume_contract.json"
    if runtime_root.exists() and not resume:
        raise WP1921Error(f"Output exists; pass resume=True: {runtime_root}")
    if contract_path.is_file():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise WP1921Error("Existing output has a different resume contract")
    runtime_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(contract_path, contract)
    snapshot = runtime_root / "config_snapshot"
    snapshot.mkdir(exist_ok=True)
    shutil.copy2(protocol_source, snapshot / protocol_source.name)
    for path in paths.values():
        shutil.copy2(path, snapshot / path.name)

    designs = sample_geometry19_designs(paths["design_schema"], sample_count, seed)
    indexed = [(index, design.as_mapping()) for index, design in enumerate(designs)]
    shards = [indexed[start : start + shard_size] for start in range(0, sample_count, shard_size)]
    shard_root = runtime_root / "shards"
    pending = []
    for shard_id, shard in enumerate(shards):
        path = shard_root / f"part-{shard_id:05d}.jsonl"
        expected = [sample_id for sample_id, _ in shard]
        if not (resume and _valid_shard(path, expected)):
            pending.append((shard_id, shard, path))
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _solve_shard_worker,
                    shard,
                    str(paths["calibration_pack"]),
                    str(paths["constraint_profile"]),
                    str(paths["conservation_tolerances"]),
                    protocol["environment"],
                ): (shard_id, path)
                for shard_id, shard, path in pending
            }
            for future in as_completed(futures):
                shard_id, path = futures[future]
                _write_jsonl_atomic(path, future.result())
                print(f"dataset shard {shard_id + 1}/{len(shards)} complete", flush=True)
    raw_records = []
    for shard_id in range(len(shards)):
        raw_records.extend(_read_jsonl(shard_root / f"part-{shard_id:05d}.jsonl"))
    raw_records.sort(key=lambda record: int(record["sample_id"]))
    if len(raw_records) != sample_count:
        raise WP1921Error("Merged shard count does not match sample_count")

    selected_ids = _status_stratified_ids(raw_records, preview_budget)
    mesh_path = runtime_root / "mesh_preview_audit.jsonl"
    mesh_records = _read_jsonl(mesh_path) if resume else []
    mesh_by_id = {int(record["sample_id"]): record for record in mesh_records}
    record_by_id = {int(record["sample_id"]): record for record in raw_records}
    pending_ids = [sample_id for sample_id in selected_ids if sample_id not in mesh_by_id]
    if pending_ids:
        storage_root = root / protocol["artifacts"]["mesh_storage_root"]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _mesh_worker,
                    str(root), str(storage_root), sample_id,
                    record_by_id[sample_id]["geometry19"], "preview_mesh",
                ): sample_id
                for sample_id in pending_ids
            }
            for future in as_completed(futures):
                record = future.result()
                mesh_by_id[int(record["sample_id"])] = record
                print(f"preview sample {record['sample_id']} complete", flush=True)
        mesh_records = [mesh_by_id[sample_id] for sample_id in sorted(mesh_by_id)]
        _write_jsonl_atomic(mesh_path, mesh_records)

    provenance = {
        "physics_spec_version": config.physics_spec_version,
        "schema_hash": loaded["design_schema"][1],
        "calibration_hash": config.calibration_pack_hash,
        "constraint_profile": loaded["constraint_profile"][0]["profile_version"],
        "constraint_profile_hash": loaded["constraint_profile"][1],
        "operating_point_policy_hash": loaded["operating_point_policy"][1],
        "visualization_calibration_hash": loaded["visualization_calibration"][1],
        "dataset_protocol_hash": protocol_hash,
        "resume_contract_hash": contract["contract_hash"],
        "formal_dataset_generation_ready": False,
        "campaign_id": campaign_id,
        "campaign_seed": seed,
    }
    selected = set(selected_ids)
    normalized = [
        _normalize_record(
            record, protocol["environment"], provenance,
            mesh_by_id.get(int(record["sample_id"])),
            int(record["sample_id"]) in selected,
        )
        for record in raw_records
    ]
    dataset_path = runtime_root / "dataset.jsonl"
    _write_jsonl_atomic(dataset_path, normalized)
    converged = sum(record["physics"]["solver"]["converged"] for record in raw_records)
    mapping_valid = sum(record["geometry_mapping_valid"] for record in raw_records)
    conservation_passed = sum(record["conservation"]["passed"] for record in raw_records)
    classification_distribution: dict[str, int] = {}
    for record in raw_records:
        classification = str(record["classification"])
        classification_distribution[classification] = (
            classification_distribution.get(classification, 0) + 1
        )
    preview_passed = sum(bool(mesh_by_id[sample_id].get("passed")) for sample_id in selected_ids)
    status = "COMPLETE" if preview_passed == len(selected_ids) else "PREVIEW_FAILED"
    report = {
        "work_package": "WP19-21",
        "protocol_version": protocol["protocol_version"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "dataset_tier": protocol["dataset_tier"],
        "formal_use_allowed": False,
        "campaign_id": campaign_id,
        "seed": seed,
        "summary": {
            "sample_count": sample_count,
            "shard_count": len(shards),
            "converged_count": converged,
            "conservation_passed_count": conservation_passed,
            "geometry_mapping_valid_count": mapping_valid,
            "classification_distribution": classification_distribution,
            "preview_requested_count": len(selected_ids),
            "preview_passed_count": preview_passed,
            "dataset_record_count": len(normalized),
        },
        "provenance": provenance,
        "artifacts": {
            "runtime_root": str(runtime_root.relative_to(root)) if runtime_root.is_relative_to(root) else str(runtime_root),
            "dataset_jsonl": str(dataset_path.relative_to(root)) if dataset_path.is_relative_to(root) else str(dataset_path),
            "resume_contract": (
                str(contract_path.relative_to(root))
                if contract_path.is_relative_to(root) else str(contract_path)
            ),
            "mesh_preview_audit": (
                str(mesh_path.relative_to(root))
                if mesh_path.is_relative_to(root) else str(mesh_path)
            ),
        },
    }
    _write_json_atomic(runtime_root / "manifest.json", report)
    (runtime_root / "REPORT.md").write_text(_render_report(report), encoding="utf-8")
    if output_root_override is None:
        _write_json_atomic(root / protocol["artifacts"]["summary_json"], report)
        report_path = root / protocol["artifacts"]["report_markdown"]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_render_report(report), encoding="utf-8")
    return report
