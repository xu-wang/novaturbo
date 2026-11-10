"""Resumable WP19-19 permissive DOE and budgeted mesh audit."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from src.physics.constraint_profiles import ConstraintProfile, assess_constraint_profile
from src.physics.engine_match import (
    Environment,
    OperatingPointSelectionPolicy,
    solve_design_point,
)
from src.physics.geometry_state import Geometry19
from src.physics.model_config import ModelConfig
from src.physics.sanity_doe_v2 import sample_geometry19_designs
from src.visualization.geometry19_adapter import geometry19_to_assembly_params
from src.visualization.geometry19_service import Geometry19DesignStore


class WP1919Error(ValueError):
    """Raised when the WP19-19 protocol or artifact state is invalid."""


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load_yaml(path: Path) -> tuple[dict[str, Any], str]:
    source = path.read_bytes()
    payload = yaml.safe_load(source)
    if not isinstance(payload, dict):
        raise WP1919Error(f"YAML must contain a mapping: {path}")
    return payload, sha256(source).hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True))
        stream.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WP1919Error(
                    f"Invalid JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise WP1919Error(f"JSONL record must be an object: {path}")
            records.append(record)
    return records


def _git_state(repo_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "dirty_paths": [line[3:] for line in status.splitlines() if line],
    }


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _conservation_audit(
    physics: Mapping[str, Any], tolerances: Mapping[str, Any]
) -> dict[str, Any]:
    if not physics.get("solver", {}).get("converged"):
        return {"passed": False, "checks": {}}
    conservation = physics["conservation"]
    mass_flow = abs(float(physics["operating_point"]["mass_flow_gas_kg_s"]))
    normalized_limit = float(tolerances["normalized_flow_absolute"])
    limits = {
        "combustor_energy_W": float(tolerances["combustor_energy_absolute_W"]),
        "shaft_power_W": float(tolerances["shaft_power_absolute_W"]),
        "mass_flow_kg_s": normalized_limit * max(mass_flow, 1.0e-12),
        "normalized_flow": normalized_limit,
        "nozzle_velocity_m_s": float(
            tolerances["nozzle_velocity_absolute_m_s"]
        ),
    }
    checks = {
        name: {
            "value": conservation.get(name),
            "absolute_tolerance": limit,
            "passed": conservation.get(name) is not None
            and abs(float(conservation[name])) <= limit,
        }
        for name, limit in limits.items()
    }
    return {"passed": all(check["passed"] for check in checks.values()), "checks": checks}


def _classification(physics: Mapping[str, Any]) -> str:
    solver = physics["solver"]
    operating = physics.get("operating_point", {})
    if solver["converged"]:
        return (
            "CONVERGED_SELECTED_MULTIPLE"
            if int(operating.get("flow_root_count", 1)) > 1
            else "CONVERGED_UNIQUE"
        )
    return str(solver.get("failure_code") or "UNCLASSIFIED_FAILURE")


def _solve_record(
    sample_id: int,
    design: Geometry19,
    environment: Environment,
    config: ModelConfig,
    constraint_profile: ConstraintProfile,
    tolerances: Mapping[str, Any],
) -> dict[str, Any]:
    geometry19 = design.as_mapping()
    visualization = None
    geometry_error = None
    try:
        visualization = geometry19_to_assembly_params(design, config)
    except (ArithmeticError, TypeError, ValueError) as exc:
        geometry_error = str(exc)
    result = solve_design_point(
        design,
        environment,
        config,
        operating_point_policy=(
            OperatingPointSelectionPolicy.REFERENCE_CONTINUATION_HIGH_FLOW_V1
        ),
    )
    physics = _jsonable(result)
    assessment = assess_constraint_profile(
        result,
        constraint_profile,
        conservation_tolerances=tolerances,
    ).as_mapping()
    record = {
        "sample_id": sample_id,
        "geometry19": geometry19,
        "geometry_mapping_valid": visualization is not None,
        "geometry_mapping_error": geometry_error,
        "design_id": visualization.design_hash if visualization else None,
        "geometry_hash": visualization.solid_geometry_hash if visualization else None,
        "lattice_hash": visualization.lattice_hash if visualization else None,
        "derived_geometry": (
            _jsonable(visualization.geometry_state) if visualization else None
        ),
        "classification": _classification(physics),
        "physics": physics,
        "constraint_assessment": assessment,
        "conservation": _conservation_audit(physics, tolerances),
    }
    return record


def _status_stratified_ids(
    records: Iterable[Mapping[str, Any]], budget: int
) -> list[int]:
    if budget <= 0:
        return []
    groups: dict[str, list[int]] = defaultdict(list)
    for record in records:
        if record.get("geometry_mapping_valid"):
            groups[str(record["classification"])].append(int(record["sample_id"]))
    selected: list[int] = []
    group_names = sorted(groups)
    cursor = 0
    while len(selected) < budget and group_names:
        name = group_names[cursor % len(group_names)]
        values = groups[name]
        position = (cursor // len(group_names))
        if position < len(values):
            selected.append(values[position])
        elif all(position >= len(groups[item]) for item in group_names):
            break
        cursor += 1
    if len(selected) < budget:
        remaining = sorted(
            int(record["sample_id"])
            for record in records
            if record.get("geometry_mapping_valid")
            and int(record["sample_id"]) not in set(selected)
        )
        if remaining:
            positions = np.linspace(
                0, len(remaining) - 1, min(budget - len(selected), len(remaining))
            )
            selected.extend(remaining[int(round(value))] for value in positions)
    return sorted(set(selected))[:budget]


def _mesh_worker(
    repo_root: str,
    storage_root: str,
    sample_id: int,
    geometry19: Mapping[str, Any],
    quality: str,
) -> dict[str, Any]:
    try:
        store = Geometry19DesignStore(repo_root, storage_root)
        response = store.regenerate(
            Geometry19.from_mapping(geometry19),
            quality=quality,
            constraint_profile="pipeline_debug_v1",
        )
        diagnostics = response["mesh_diagnostics"]
        return {
            "sample_id": sample_id,
            "quality": quality,
            "design_id": response["design_id"],
            "passed": bool(diagnostics["passed"]),
            "component_count": len(diagnostics["component_diagnostics"]),
            "error": None,
        }
    except Exception as exc:
        return {
            "sample_id": sample_id,
            "quality": quality,
            "design_id": None,
            "passed": False,
            "component_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _run_mesh_audit(
    repo_root: Path,
    storage_root: Path,
    records: list[dict[str, Any]],
    preview_budget: int,
    export_budget: int,
    workers: int,
    audit_path: Path,
) -> dict[str, Any]:
    preview_ids = _status_stratified_ids(records, preview_budget)
    preview_records = [
        record for record in records if int(record["sample_id"]) in preview_ids
    ]
    export_ids = _status_stratified_ids(preview_records, export_budget)
    by_id = {int(record["sample_id"]): record for record in records}
    expected = [(sample_id, "preview_mesh") for sample_id in preview_ids]
    expected.extend((sample_id, "export_mesh") for sample_id in export_ids)
    completed = _read_jsonl(audit_path)
    completed_keys = {
        (int(item["sample_id"]), str(item["quality"])) for item in completed
    }
    pending = [item for item in expected if item not in completed_keys]
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _mesh_worker,
                    str(repo_root),
                    str(storage_root),
                    sample_id,
                    by_id[sample_id]["geometry19"],
                    quality,
                ): (sample_id, quality)
                for sample_id, quality in pending
            }
            for future in as_completed(futures):
                result = future.result()
                _append_jsonl(audit_path, result)
                completed.append(result)
    mapping_failures = [
        {
            "sample_id": record["sample_id"],
            "quality": "mapping_audit",
            "passed": False,
            "error": record["geometry_mapping_error"],
        }
        for record in records
        if not record["geometry_mapping_valid"]
    ]
    counts = Counter(str(item["quality"]) for item in completed)
    passes = Counter(
        str(item["quality"]) for item in completed if item.get("passed")
    )
    return {
        "selection": "deterministic_status_stratified",
        "preview_sample_ids": preview_ids,
        "export_sample_ids": export_ids,
        "generated_count": dict(counts),
        "passed_count": dict(passes),
        "failed_meshes": [item for item in completed if not item.get("passed")],
        "forced_geometry_mapping_failure_audit": mapping_failures,
        "passed": len(completed) == len(expected)
        and all(item.get("passed") for item in completed),
    }


def _summary(records: list[dict[str, Any]], mesh: Mapping[str, Any]) -> dict[str, Any]:
    classifications = Counter(str(record["classification"]) for record in records)
    converged = [record for record in records if record["physics"]["solver"]["converged"]]
    conservation_passed = sum(record["conservation"]["passed"] for record in converged)
    accepted = sum(record["constraint_assessment"]["accepted"] for record in records)
    violation_counts: Counter[str] = Counter()
    mapping_failures: Counter[str] = Counter()
    for record in records:
        violation_counts.update(record["constraint_assessment"]["violated_constraints"])
        if not record["geometry_mapping_valid"]:
            mapping_failures.update([str(record["geometry_mapping_error"])])

    def distribution(name: str) -> dict[str, float | int] | None:
        values = [
            float(record["physics"]["performance"][name])
            for record in converged
            if record["physics"]["performance"].get(name) is not None
        ]
        if not values:
            return None
        array = np.asarray(values, dtype=float)
        quantiles = np.quantile(array, [0.05, 0.5, 0.95])
        return {
            "count": len(values),
            "min": float(array.min()),
            "p05": float(quantiles[0]),
            "median": float(quantiles[1]),
            "p95": float(quantiles[2]),
            "max": float(array.max()),
            "mean": float(array.mean()),
        }

    compact_mesh = {
        "selection": mesh["selection"],
        "preview_requested_count": len(mesh["preview_sample_ids"]),
        "export_requested_count": len(mesh["export_sample_ids"]),
        "generated_count": mesh["generated_count"],
        "passed_count": mesh["passed_count"],
        "failed_meshes": mesh["failed_meshes"],
        "forced_geometry_mapping_failure_count": len(
            mesh["forced_geometry_mapping_failure_audit"]
        ),
        "passed": mesh["passed"],
    }
    return {
        "sample_count": len(records),
        "geometry_mapping_valid_count": sum(
            record["geometry_mapping_valid"] for record in records
        ),
        "geometry_mapping_valid_rate": sum(
            record["geometry_mapping_valid"] for record in records
        )
        / len(records),
        "converged_count": len(converged),
        "convergence_rate": len(converged) / len(records),
        "conservation_passed_count": conservation_passed,
        "conservation_passed_given_converged_rate": (
            conservation_passed / len(converged) if converged else 0.0
        ),
        "pipeline_debug_accepted_count": accepted,
        "pipeline_debug_accepted_rate": accepted / len(records),
        "classification_distribution": dict(sorted(classifications.items())),
        "geometry_mapping_failure_distribution": dict(
            sorted(mapping_failures.items(), key=lambda item: (-item[1], item[0]))
        ),
        "performance_distribution": {
            name: distribution(name)
            for name in ("thrust_N", "tsfc_kg_N_s", "mass_kg")
        },
        "constraint_violation_distribution": dict(
            sorted(violation_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "mesh_audit": compact_mesh,
    }


def render_report(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# WP19-19 4096 点宽松 DOE 报告",
        "",
        f"- 状态：`{report['status']}`",
        f"- 样本数：{summary['sample_count']}",
        f"- 几何映射有效率：{summary['geometry_mapping_valid_rate']:.6f}",
        f"- 求解收敛率：{summary['convergence_rate']:.6f}",
        "- 收敛案例守恒通过率："
        f"{summary['conservation_passed_given_converged_rate']:.6f}",
        f"- pipeline debug 接受率：{summary['pipeline_debug_accepted_rate']:.6f}",
        "",
        "## 求解分类",
        "",
        "| classification | count |",
        "|---|---:|",
    ]
    for name, count in summary["classification_distribution"].items():
        lines.append(f"| `{name}` | {count} |")
    lines.extend(
        [
            "",
            "## 几何映射失败",
            "",
            "| failure | count |",
            "|---|---:|",
        ]
    )
    for name, count in summary["geometry_mapping_failure_distribution"].items():
        lines.append(f"| `{name}` | {count} |")
    lines.extend(
        [
            "",
            "## 性能分布（收敛样本）",
            "",
            "| output | min | median | p95 | max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, distribution in summary["performance_distribution"].items():
        if distribution is None:
            continue
        lines.append(
            f"| `{name}` | {distribution['min']:.6g} | "
            f"{distribution['median']:.6g} | {distribution['p95']:.6g} | "
            f"{distribution['max']:.6g} |"
        )
    mesh = summary["mesh_audit"]
    lines.extend(
        [
            "",
            "## 网格预算",
            "",
            f"- preview：{mesh['passed_count'].get('preview_mesh', 0)}/"
            f"{mesh['generated_count'].get('preview_mesh', 0)}",
            f"- export：{mesh['passed_count'].get('export_mesh', 0)}/"
            f"{mesh['generated_count'].get('export_mesh', 0)}",
            "- 几何映射失败审计："
            f"{mesh['forced_geometry_mapping_failure_count']}",
            "",
            "完整逐样本账本、输入、配置快照和网格审计位于运行时产物目录。",
            "WP19-20 将在此数据上执行 19D 敏感性与可视化方向审计。",
            "",
        ]
    )
    return "\n".join(lines)


def run_wp19_19(
    protocol_path: str | Path,
    repo_root: str | Path,
    *,
    sample_count_override: int | None = None,
    preview_budget_override: int | None = None,
    export_budget_override: int | None = None,
    workers_override: int | None = None,
    physics_only: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    protocol_source = Path(protocol_path).resolve()
    protocol, protocol_hash = _load_yaml(protocol_source)
    if protocol.get("work_package") != "WP19-19":
        raise WP1919Error("Protocol must declare work_package=WP19-19")
    paths = {
        name: root / str(protocol[name])
        for name in (
            "design_schema",
            "calibration_pack",
            "constraint_profile",
            "operating_point_policy",
            "visualization_calibration",
            "mesh_profiles",
            "conservation_tolerances",
        )
    }
    loaded = {name: _load_yaml(path) for name, path in paths.items()}
    policy = loaded["operating_point_policy"][0]
    if policy.get("solver_policy") != (
        OperatingPointSelectionPolicy.REFERENCE_CONTINUATION_HIGH_FLOW_V1.value
    ):
        raise WP1919Error("Unsupported operating-point policy")
    config = ModelConfig.from_yaml(paths["calibration_pack"])
    profile = ConstraintProfile.from_yaml(paths["constraint_profile"])
    environment = Environment(**protocol["environment"])
    sample_count = int(
        sample_count_override
        if sample_count_override is not None
        else protocol["sample_count"]
    )
    mesh_config = protocol["mesh_budget"]
    preview_budget = int(
        preview_budget_override
        if preview_budget_override is not None
        else mesh_config["preview_mesh"]
    )
    export_budget = int(
        export_budget_override
        if export_budget_override is not None
        else mesh_config["export_mesh"]
    )
    workers = int(
        workers_override
        if workers_override is not None
        else mesh_config["workers"]
    )
    if not (sample_count > 0 and 0 <= export_budget <= preview_budget <= sample_count):
        raise WP1919Error("Invalid sample or mesh budgets")
    if workers <= 0:
        raise WP1919Error("workers must be positive")
    artifacts = protocol["artifacts"]
    runtime_root = root / artifacts["runtime_root"]
    runtime_root.mkdir(parents=True, exist_ok=True)
    snapshot_root = runtime_root / "config_snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(protocol_source, snapshot_root / protocol_source.name)
    for path in paths.values():
        shutil.copy2(path, snapshot_root / path.name)
    designs = sample_geometry19_designs(
        paths["design_schema"], sample_count, int(protocol["seed"])
    )
    inputs_path = runtime_root / "inputs" / "geometry19.jsonl"
    records_path = runtime_root / "results" / "samples.jsonl"
    records = _read_jsonl(records_path)
    completed_ids = {int(record["sample_id"]) for record in records}
    if not inputs_path.is_file():
        for sample_id, design in enumerate(designs):
            _append_jsonl(
                inputs_path,
                {"sample_id": sample_id, "geometry19": design.as_mapping()},
            )
    for sample_id, design in enumerate(designs):
        if sample_id in completed_ids:
            continue
        record = _solve_record(
            sample_id,
            design,
            environment,
            config,
            profile,
            loaded["conservation_tolerances"][0],
        )
        _append_jsonl(records_path, record)
        records.append(record)
    records.sort(key=lambda item: int(item["sample_id"]))
    mesh_audit_path = runtime_root / "results" / "mesh_audit.jsonl"
    if physics_only:
        mesh = {
            "selection": "not_run_physics_only",
            "preview_sample_ids": [],
            "export_sample_ids": [],
            "generated_count": {},
            "passed_count": {},
            "failed_meshes": [],
            "forced_geometry_mapping_failure_audit": [
                record for record in records if not record["geometry_mapping_valid"]
            ],
            "passed": False,
        }
        status = "PHYSICS_COMPLETE_MESH_PENDING"
    else:
        mesh = _run_mesh_audit(
            root,
            root / artifacts["mesh_storage_root"],
            records,
            preview_budget,
            export_budget,
            workers,
            mesh_audit_path,
        )
        status = "COMPLETE" if mesh["passed"] else "MESH_AUDIT_FAILED"
    summary = _summary(records, mesh)
    if summary["conservation_passed_count"] != summary["converged_count"]:
        status = "CONSERVATION_AUDIT_FAILED"
    provenance = {
        "protocol_hash": protocol_hash,
        "git": _git_state(root),
        "physics_spec_version": config.physics_spec_version,
        "schema_hash": loaded["design_schema"][1],
        "calibration_hash": config.calibration_pack_hash,
        "constraint_profile_hash": profile.source_hash,
        "operating_point_policy_hash": loaded["operating_point_policy"][1],
        "visualization_calibration_hash": loaded["visualization_calibration"][1],
        "mesh_profiles_hash": loaded["mesh_profiles"][1],
        "conservation_tolerances_hash": loaded["conservation_tolerances"][1],
        "implementation_file_hashes": {
            str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
            for path in (
                root / "src/physics/wp19_19_doe.py",
                root / "src/physics/engine_match.py",
                root / "src/visualization/geometry19_adapter.py",
                root / "src/visualization/geometry19_mesh.py",
                root / "src/geometry/lattice.py",
                root / "scripts/run_wp19_19_permissive_doe.py",
            )
        },
    }
    report = {
        "work_package": "WP19-19",
        "protocol_version": protocol["protocol_version"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "seed": int(protocol["seed"]),
        "environment": asdict(environment),
        "mesh_budget": {
            "preview_mesh": preview_budget,
            "export_mesh": export_budget,
            "workers": workers,
        },
        "summary": summary,
        "classification_definitions": {
            "CONVERGED_UNIQUE": "One valid whole-engine flow root converged.",
            "CONVERGED_SELECTED_MULTIPLE": (
                "Multiple valid roots were retained and the frozen "
                "reference-continuation high-flow branch was selected."
            ),
            "NO_FLOW_MATCH": "No valid root exists in the frozen inlet-capacity domain.",
            "FLOW_RESIDUAL_DISCONTINUITY": (
                "A sign-changing bracket failed the final finite-residual gate."
            ),
            "GEOMETRY_INVALID": "The Geometry19 hard geometry contract was violated.",
        },
        "provenance": provenance,
        "runtime_artifacts": {
            "root": _display_path(runtime_root, root),
            "inputs": _display_path(inputs_path, root),
            "sample_ledger": _display_path(records_path, root),
            "mesh_audit": _display_path(mesh_audit_path, root),
        },
    }
    _write_json_atomic(runtime_root / "manifest.json", report)
    _write_json_atomic(root / artifacts["summary_json"], report)
    report_path = root / artifacts["report_markdown"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(report), encoding="utf-8")
    failures_path = runtime_root / "failures" / "failures.jsonl"
    if not failures_path.exists():
        for record in records:
            if not record["physics"]["solver"]["converged"]:
                _append_jsonl(failures_path, record)
    return report
