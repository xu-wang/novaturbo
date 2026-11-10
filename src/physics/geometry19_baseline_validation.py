"""Versioned multi-case golden baseline for the Geometry19 V2.1 runtime."""

from __future__ import annotations

from dataclasses import asdict, fields, replace
from enum import Enum
from hashlib import sha256
import json
import math
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy
import scipy
import yaml

from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.ai.geometry19_pipeline_profiles import load_pipeline_profile_registry
from src.physics.engine_match import Environment, Geometry19, solve_design_point


BASELINE_CONTRACT_VERSION = "geometry19_v2_1_baseline_manifest_v1"
DEFAULT_SUITE_PATH = Path("config/physics/geometry19_v2_1_golden_cases_v1.yaml")
DEFAULT_MANIFEST_PATH = Path(
    "config/physics/baselines/geometry19_v2_1_baseline_manifest_v1.json"
)
RESULT_GROUPS = (
    "solver",
    "operating_point",
    "stations",
    "components",
    "performance",
    "constraints",
    "mass_breakdown_kg",
    "mass_volume_breakdown_m3",
    "formal_diagnostics",
    "thermodynamic_constraints",
    "assembly_constraints",
    "structural_constraints",
    "thermal_constraints",
    "manufacturing_constraints",
    "assembly_diagnostics",
    "structural_diagnostics",
    "conservation",
)


class Geometry19BaselineError(ValueError):
    """Raised when the baseline contract is invalid or no longer reproduced."""


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_hash(payload: Any) -> str:
    return sha256(_canonical_bytes(payload)).hexdigest()


def canonical_baseline_payload_hash(payload: Any) -> str:
    """Return the public canonical hash used by the baseline manifest contract."""

    return _canonical_hash(payload)


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _normalize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise Geometry19BaselineError(
        f"Unsupported value in golden snapshot: {type(value).__name__}"
    )


def _read_yaml(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        source = path.read_bytes()
        payload = yaml.safe_load(source)
    except (OSError, yaml.YAMLError) as exc:
        raise Geometry19BaselineError(f"Cannot load golden suite: {path}") from exc
    if not isinstance(payload, dict):
        raise Geometry19BaselineError("Golden suite must be a YAML mapping")
    return source, payload


def load_golden_suite(path: str | Path) -> dict[str, Any]:
    """Load the frozen case definitions under an exact schema."""

    _, suite = _read_yaml(Path(path).resolve())
    required = {
        "suite_version",
        "pipeline_profile",
        "expected_physics_model_version",
        "expected_operating_point_policy",
        "tolerances",
        "cases",
    }
    if set(suite) != required:
        raise Geometry19BaselineError("Golden suite top-level keys mismatch")
    if suite["suite_version"] != Path(path).stem:
        raise Geometry19BaselineError("Golden suite filename/version mismatch")
    if suite["pipeline_profile"] != "v2_1":
        raise Geometry19BaselineError("Golden suite must freeze the v2_1 profile")
    expected_tolerances = {
        "solver",
        "operating_point",
        "stations",
        "components",
        "performance",
        "constraints",
        "mass",
        "diagnostics",
        "conservation",
    }
    tolerances = suite["tolerances"]
    if not isinstance(tolerances, dict) or set(tolerances) != expected_tolerances:
        raise Geometry19BaselineError("Golden tolerance groups mismatch")
    for name, tolerance in tolerances.items():
        if not isinstance(tolerance, dict) or set(tolerance) != {
            "relative",
            "absolute",
        }:
            raise Geometry19BaselineError(f"Tolerance {name} contract mismatch")
        values = (float(tolerance["relative"]), float(tolerance["absolute"]))
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise Geometry19BaselineError(f"Tolerance {name} must be nonnegative")
    cases = suite["cases"]
    if not isinstance(cases, list) or len(cases) < 8:
        raise Geometry19BaselineError("Golden suite requires at least eight cases")
    expected_case_keys = {
        "case_id",
        "description",
        "design_overrides",
        "environment",
        "expected_solver",
    }
    expected_solver_keys = {
        "converged",
        "failure_code",
        "root_count",
        "selected_root_index",
    }
    geometry_names = {field.name for field in fields(Geometry19)}
    case_ids = []
    for case in cases:
        if not isinstance(case, dict) or set(case) != expected_case_keys:
            raise Geometry19BaselineError("Golden case keys mismatch")
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not case_id:
            raise Geometry19BaselineError("Golden case_id must be non-empty")
        case_ids.append(case_id)
        if not isinstance(case["description"], str) or not case["description"]:
            raise Geometry19BaselineError(f"Golden case {case_id} needs a description")
        overrides = case["design_overrides"]
        if not isinstance(overrides, dict) or not set(overrides).issubset(
            geometry_names
        ):
            raise Geometry19BaselineError(
                f"Golden case {case_id} has invalid overrides"
            )
        environment = case["environment"]
        if not isinstance(environment, dict) or set(environment) != {
            "temperature_K",
            "pressure_Pa",
            "mach",
        }:
            raise Geometry19BaselineError(
                f"Golden case {case_id} environment contract mismatch"
            )
        expected = case["expected_solver"]
        if not isinstance(expected, dict) or set(expected) != expected_solver_keys:
            raise Geometry19BaselineError(
                f"Golden case {case_id} solver expectation mismatch"
            )
    if len(set(case_ids)) != len(case_ids):
        raise Geometry19BaselineError("Golden case IDs must be unique")
    return suite


def _reference_design(context: Any) -> Geometry19:
    return Geometry19.from_mapping(
        {
            item["name"]: item["reference_value"]
            for item in context.physics_runtime.design_schema["design_variables"]
        }
    )


def _snapshot(result: Any) -> dict[str, Any]:
    raw = asdict(result)
    snapshot = {
        name: raw[name]
        for name in RESULT_GROUPS
    }
    snapshot.update(
        {
            "feasible": raw["feasible"],
            "core_feasible": raw["core_feasible"],
            "thermodynamic_feasible": raw["thermodynamic_feasible"],
            "assembly_diagnostics_pending": raw["assembly_diagnostics_pending"],
            "structural_diagnostics_pending": raw["structural_diagnostics_pending"],
            "feasibility_complete": raw["feasibility_complete"],
            "feasibility_scope": raw["feasibility_scope"],
            "physics_model_version": raw["physics_model_version"],
            "calibration_pack_hash": raw["calibration_pack_hash"],
            "calibration_status": raw["calibration_status"],
            "formal_dataset_generation_ready": raw[
                "formal_dataset_generation_ready"
            ],
        }
    )
    return _normalize(snapshot)


def _solver_contract(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    solver = snapshot["solver"]
    operating = snapshot["operating_point"]
    return {
        "converged": solver["converged"],
        "failure_code": solver["failure_code"],
        "root_count": operating.get("flow_root_count", 0),
        "selected_root_index": operating.get("selected_flow_root_index"),
    }


def build_v2_1_baseline_manifest(
    repository_root: str | Path,
    *,
    suite_path: str | Path = DEFAULT_SUITE_PATH,
) -> dict[str, Any]:
    """Evaluate all golden cases and return a deterministic candidate manifest."""

    root = Path(repository_root).resolve()
    suite_source = (root / suite_path).resolve()
    suite = load_golden_suite(suite_source)
    profile_registry = load_pipeline_profile_registry(root)
    if profile_registry.default_profile != "v2_1":
        raise Geometry19BaselineError(
            "Geometry19 default pipeline profile drifted from v2_1"
        )
    context = build_pipeline_context(
        root,
        pipeline_profile=str(suite["pipeline_profile"]),
        profile_registry=profile_registry,
    )
    runtime = context.physics_runtime
    if runtime.model_version != suite["expected_physics_model_version"]:
        raise Geometry19BaselineError("Golden suite selected the wrong physics model")
    if runtime.operating_point_policy.value != suite["expected_operating_point_policy"]:
        raise Geometry19BaselineError(
            "Golden suite selected the wrong operating-point policy"
        )
    reference = _reference_design(context)
    records = []
    for case in suite["cases"]:
        design = replace(reference, **case["design_overrides"])
        environment = Environment(**case["environment"])
        through_runtime = runtime.evaluate(design, environment)
        direct = solve_design_point(
            design,
            environment,
            runtime.model_config,
            operating_point_policy=runtime.operating_point_policy,
        )
        runtime_snapshot = _snapshot(through_runtime)
        direct_snapshot = _snapshot(direct)
        if runtime_snapshot != direct_snapshot:
            raise Geometry19BaselineError(
                f"V2.1 runtime/direct parity failed for {case['case_id']}"
            )
        actual_solver = _solver_contract(runtime_snapshot)
        if actual_solver != case["expected_solver"]:
            raise Geometry19BaselineError(
                f"Golden solver expectation failed for {case['case_id']}: "
                f"expected={case['expected_solver']}, actual={actual_solver}"
            )
        inputs = {
            "design": _normalize(asdict(design)),
            "environment": _normalize(asdict(environment)),
        }
        records.append(
            {
                "case_id": case["case_id"],
                "description": case["description"],
                "input_sha256": _canonical_hash(inputs),
                "inputs": inputs,
                "expected_result": runtime_snapshot,
            }
        )
    profile_identity = context.profile.contract_mapping()
    profile_identity["protocol_path"] = str(
        context.profile.protocol_path.relative_to(root)
    )
    context_identity = {
        "pipeline_profile": profile_identity,
        "profile_registry_version": context.profile_registry_version,
        "context_hash": context.context_hash,
        "provenance_hashes": dict(context.provenance_hashes),
        "physics_runtime": runtime.contract_mapping(),
    }
    payload = {
        "manifest_version": BASELINE_CONTRACT_VERSION,
        "suite_version": suite["suite_version"],
        "suite_sha256": _file_hash(suite_source),
        "generator": {
            "tool": "scripts/verify_geometry19_v2_1_baseline.py",
            "contract_version": BASELINE_CONTRACT_VERSION,
        },
        "execution_environment": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "pyyaml": yaml.__version__,
        },
        "context": context_identity,
        "tolerances": suite["tolerances"],
        "case_count": len(records),
        "cases": records,
    }
    return {**payload, "manifest_payload_sha256": _canonical_hash(payload)}


def _tolerance_group(path: Sequence[str]) -> str:
    top = path[0]
    if top in {"mass_breakdown_kg", "mass_volume_breakdown_m3"}:
        return "mass"
    if top in {
        "formal_diagnostics",
        "thermodynamic_constraints",
        "assembly_constraints",
        "structural_constraints",
        "thermal_constraints",
        "manufacturing_constraints",
        "assembly_diagnostics",
        "structural_diagnostics",
    }:
        return "diagnostics"
    if top in {
        "solver",
        "operating_point",
        "stations",
        "components",
        "performance",
        "constraints",
        "conservation",
    }:
        return top
    return "solver"


def _compare(
    expected: Any,
    actual: Any,
    tolerances: Mapping[str, Any],
    path: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    location = ".".join(path) or "<root>"
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return [] if expected == actual else [
            {"path": location, "expected": expected, "actual": actual}
        ]
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        rule = tolerances[_tolerance_group(path)]
        matches = math.isclose(
            float(actual),
            float(expected),
            rel_tol=float(rule["relative"]),
            abs_tol=float(rule["absolute"]),
        )
        return [] if matches else [
            {
                "path": location,
                "expected": expected,
                "actual": actual,
                "absolute_error": abs(float(actual) - float(expected)),
                "relative_tolerance": float(rule["relative"]),
                "absolute_tolerance": float(rule["absolute"]),
            }
        ]
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return [
                {
                    "path": location,
                    "expected_keys": sorted(expected),
                    "actual_keys": sorted(actual),
                }
            ]
        differences = []
        for key in sorted(expected):
            differences.extend(
                _compare(expected[key], actual[key], tolerances, (*path, key))
            )
        return differences
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [
                {
                    "path": location,
                    "expected_length": len(expected),
                    "actual_length": len(actual),
                }
            ]
        differences = []
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual, strict=True)
        ):
            differences.extend(
                _compare(
                    expected_item,
                    actual_item,
                    tolerances,
                    (*path, str(index)),
                )
            )
        return differences
    return [{"path": location, "expected": expected, "actual": actual}]


def verify_v2_1_baseline(
    repository_root: str | Path,
    *,
    suite_path: str | Path = DEFAULT_SUITE_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Re-evaluate and compare every field with the committed golden manifest."""

    root = Path(repository_root).resolve()
    expected_path = (root / manifest_path).resolve()
    try:
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Geometry19BaselineError(
            f"Cannot load V2.1 baseline manifest: {expected_path}"
        ) from exc
    required = {
        "manifest_version",
        "suite_version",
        "suite_sha256",
        "generator",
        "execution_environment",
        "context",
        "tolerances",
        "case_count",
        "cases",
        "manifest_payload_sha256",
    }
    if not isinstance(expected, dict) or set(expected) != required:
        raise Geometry19BaselineError("Baseline manifest keys mismatch")
    payload = {
        key: value
        for key, value in expected.items()
        if key != "manifest_payload_sha256"
    }
    if _canonical_hash(payload) != expected["manifest_payload_sha256"]:
        raise Geometry19BaselineError("Baseline manifest payload hash mismatch")
    actual = build_v2_1_baseline_manifest(root, suite_path=suite_path)
    identity_keys = (
        "manifest_version",
        "suite_version",
        "suite_sha256",
        "generator",
        "execution_environment",
        "context",
        "case_count",
    )
    differences = []
    for key in identity_keys:
        if expected[key] != actual[key]:
            differences.append(
                {"path": key, "expected": expected[key], "actual": actual[key]}
            )
    expected_cases = {case["case_id"]: case for case in expected["cases"]}
    actual_cases = {case["case_id"]: case for case in actual["cases"]}
    if set(expected_cases) != set(actual_cases):
        differences.append(
            {
                "path": "cases",
                "expected_case_ids": sorted(expected_cases),
                "actual_case_ids": sorted(actual_cases),
            }
        )
    else:
        for case_id in sorted(expected_cases):
            expected_case = expected_cases[case_id]
            actual_case = actual_cases[case_id]
            for key in ("description", "input_sha256", "inputs"):
                if expected_case[key] != actual_case[key]:
                    differences.append(
                        {
                            "path": f"cases.{case_id}.{key}",
                            "expected": expected_case[key],
                            "actual": actual_case[key],
                        }
                    )
            differences.extend(
                {
                    **difference,
                    "path": f"cases.{case_id}.expected_result.{difference['path']}",
                }
                for difference in _compare(
                    expected_case["expected_result"],
                    actual_case["expected_result"],
                    expected["tolerances"],
                )
            )
    return {
        "validation_id": "geometry19_v2_1_baseline_verification_v1",
        "manifest_path": str(expected_path),
        "manifest_payload_sha256": expected["manifest_payload_sha256"],
        "case_count": len(actual_cases),
        "difference_count": len(differences),
        "differences": differences,
        "passed": not differences,
    }
